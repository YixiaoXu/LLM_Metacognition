#!/usr/bin/env python
"""Train a two-stage decoupler against sparse multi-neuron module targets.

This is an experimental replacement for the target-by-target objective.  It
keeps the mature Decoupler and E2RecursivePurifier architectures so existing
runtime intervention code can load the checkpoints, but changes the supervised
signal in both stages:

1. A broad, label-free neuron pool is selected from activation variability and
   cross-split stability only.
2. Sparse module directions are discovered with cross-fitted partial CCA
   between semantic-residual target activations and the current meta code.
3. E1 predicts each module's semantic component; E2 predicts the residual that
   E1 cannot explain.
4. The recursive purifier predicts the same module residuals, with an optional
   early refresh of the module directions before they are frozen.

The script exports module_refiner/aligned-cache/cluster artifacts compatible
with the existing refined-residual bridge and distribution-transport pipeline.
The legacy gamma and frozen-sidecar trainers are intentionally untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import _bootstrap  # noqa: F401  # Legacy direct-script compatibility.

import fit_soft_residual_clusters as residual_lib
import train_decoupler as base
from metacog.clustering import centroid_silhouette_score, run_kmeans
from refined_residual_runtime import ModuleRefiner, TaskAlignedModuleRefiner
from semantic_residual_controls import (
    crossfit_strongest_semantic_prediction,
)


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class ExternalSemanticCandidateDecoder(nn.Module):
    """Predict an external semantic activation and expose its fixed target map.

    The trainable part predicts the external model representation F.  The
    external-F -> candidate-neuron map is fitted once on the training split
    and then frozen, so the semantic branch is not optimized directly against
    the current layer-i/module target.
    """

    def __init__(
        self,
        input_dim: int,
        external_dim: int,
        candidate_dim: int,
        hidden_dim: int,
        dropout: float,
        external_to_candidate: torch.Tensor,
        head_type: str = "mlp",
        head_depth: int = 4,
    ) -> None:
        super().__init__()
        self.external_head = build_external_semantic_head(
            head_type,
            input_dim,
            external_dim,
            hidden_dim,
            dropout,
            head_depth,
        )
        self.external_to_candidate = nn.Linear(external_dim, candidate_dim)
        with torch.no_grad():
            self.external_to_candidate.weight.copy_(
                external_to_candidate[:-1].t()
            )
            self.external_to_candidate.bias.copy_(external_to_candidate[-1])
        set_trainable(self.external_to_candidate, False)

    def predict_external(self, value: torch.Tensor) -> torch.Tensor:
        return self.external_head(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.external_to_candidate(self.predict_external(value))


class ResidualSemanticBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class DeepResidualSemanticHead(nn.Module):
    """Higher-capacity Z1 -> external-activation semantic readout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        depth: int,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            ResidualSemanticBlock(hidden_dim, dropout)
            for _ in range(max(1, int(depth)))
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.input(value)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class EnsembleExternalSemanticHead(nn.Module):
    """Convex ensemble of semantic readouts trained only against external F."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        depth: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.mlp = base.MLP(input_dim, output_dim, hidden_dim, dropout)
        self.deep = DeepResidualSemanticHead(
            input_dim, output_dim, hidden_dim, dropout, depth
        )
        self.mixture_logits = nn.Parameter(torch.zeros(3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weights = self.mixture_logits.softmax(dim=0)
        predictions = torch.stack(
            [self.linear(value), self.mlp(value), self.deep(value)], dim=0
        )
        return (weights.view(-1, 1, 1) * predictions).sum(dim=0)

    def mixture_weights(self) -> List[float]:
        return self.mixture_logits.detach().softmax(dim=0).cpu().tolist()


def build_external_semantic_head(
    head_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    depth: int,
) -> nn.Module:
    head_type = str(head_type).strip().lower()
    if head_type == "linear":
        return nn.Linear(input_dim, output_dim)
    if head_type == "mlp":
        return base.MLP(input_dim, output_dim, hidden_dim, dropout)
    if head_type == "deep_residual":
        return DeepResidualSemanticHead(
            input_dim, output_dim, hidden_dim, dropout, depth
        )
    if head_type == "ensemble":
        return EnsembleExternalSemanticHead(
            input_dim, output_dim, hidden_dim, dropout, depth
        )
    raise ValueError(f"Unsupported external semantic head: {head_type}")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage sparse combination-aware residual decoupler."
    )
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument(
        "--external-semantic-activation-dir",
        default="",
        help=(
            "Optional activation cache from an independent model. When set, "
            "the semantic head is trained to predict the selected external "
            "layer rather than candidate layer-i activations."
        ),
    )
    parser.add_argument(
        "--external-semantic-layer",
        type=int,
        default=None,
        help="Layer id in --external-semantic-activation-dir used as F.",
    )
    parser.add_argument(
        "--external-semantic-max-dim",
        type=int,
        default=0,
        help="Optional leading dimension cap for external F; 0 keeps all dimensions.",
    )
    parser.add_argument(
        "--external-semantic-ridge",
        type=float,
        default=1.0,
        help="Ridge coefficient for the frozen F -> candidate-target map.",
    )
    parser.add_argument(
        "--external-semantic-head-type",
        choices=["linear", "mlp", "deep_residual", "ensemble"],
        default="mlp",
        help=(
            "Architecture used for frozen Z1 -> external activation F. "
            "mlp is the historical default; ensemble combines linear, MLP, "
            "and deep residual branches without using module labels."
        ),
    )
    parser.add_argument(
        "--external-semantic-head-depth",
        type=int,
        default=4,
        help="Residual-block count for deep_residual and ensemble heads.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--layer-fracs", nargs=3, type=float, default=None)
    parser.add_argument("--prev-layer", type=int, default=None)
    parser.add_argument("--layer-i", type=int, default=None)
    parser.add_argument("--next-layer", type=int, default=None)
    parser.add_argument("--prev-offset", type=int, default=1)
    parser.add_argument("--next-offset", type=int, default=1)
    parser.add_argument("--semantic-anchor-offsets", nargs="*", type=int, default=[1])
    parser.add_argument("--semantic-anchor-weights", nargs="*", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-by-base-id",
        action="store_true",
        help="Split generation-step rows by base prompt id, keeping all steps together.",
    )
    parser.add_argument(
        "--base-id-step-pattern",
        default=r"::step\d+$",
        help="Regex removed from a row id to recover its base prompt id.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-top-k", type=int, default=768)
    parser.add_argument("--candidate-min-rate", type=float, default=0.05)
    parser.add_argument("--candidate-max-rate", type=float, default=0.95)
    parser.add_argument("--candidate-max-rate-drift", type=float, default=0.12)
    parser.add_argument("--binary-quantile", type=float, default=0.75)
    parser.add_argument("--num-modules", type=int, default=8)
    parser.add_argument("--module-support", type=int, default=24)
    parser.add_argument("--module-max-memberships", type=int, default=2)
    parser.add_argument(
        "--module-direction-mode",
        choices=["learned", "random_support"],
        default="learned",
        help=(
            "How module supports are constructed. learned discovers sparse "
            "supports from the residual cross-covariance; random_support "
            "draws the same-size supports uniformly from the candidate pool "
            "and keeps them fixed for the full training run."
        ),
    )
    parser.add_argument(
        "--random-support-seed",
        type=int,
        default=-1,
        help="Seed for the random-support null; negative reuses --seed.",
    )
    parser.add_argument("--direction-ridge", type=float, default=1e-2)
    parser.add_argument("--direction-semantic-penalty", type=float, default=5.0)
    parser.add_argument("--direction-ema", type=float, default=0.50)
    parser.add_argument("--direction-refresh-start", type=int, default=15)
    parser.add_argument("--direction-refresh-every", type=int, default=5)
    parser.add_argument("--direction-freeze-epoch", type=int, default=55)
    parser.add_argument("--direction-max-samples", type=int, default=8192)
    parser.add_argument("--direction-min-cosine", type=float, default=0.50)
    parser.add_argument("--lambda-next", type=float, default=1.0)
    parser.add_argument("--lambda-prev", type=float, default=1.0)
    parser.add_argument("--lambda-orth", type=float, default=0.10)
    parser.add_argument("--lambda-e2-prev-cov", type=float, default=0.30)
    parser.add_argument("--lambda-semantic-module", type=float, default=0.20)
    parser.add_argument("--lambda-semantic-candidate", type=float, default=0.20)
    parser.add_argument("--lambda-residual-module", type=float, default=0.20)
    parser.add_argument("--lambda-joint-module", type=float, default=0.10)
    parser.add_argument(
        "--semantic-candidate-source",
        choices=["z1", "z1_prev", "prev"],
        default="z1",
        help=(
            "Semantic teacher input. z1 is the primary decomposition-aligned "
            "choice: Z1 is inferred from the later layer and trained to recover "
            "the earlier-layer semantic anchor. prev is retained only as a "
            "direct layer-transition diagnostic/control."
        ),
    )
    parser.add_argument(
        "--semantic-probe-mode",
        choices=["joint", "frozen_aligned", "staged_z1_aligned"],
        default="joint",
        help=(
            "joint preserves the legacy jointly optimized semantic head. "
            "frozen_aligned pretrains one semantic probe, freezes it, and reuses "
            "the exact same probe for training residual targets and held-out RF. "
            "staged_z1_aligned first stabilizes the Hk->Z1/Z2 reconstruction, "
            "then freezes E1 and fits the reusable semantic probe from Z1 only."
        ),
    )
    parser.add_argument(
        "--semantic-encoder-warmup-epochs",
        type=int,
        default=30,
        help=(
            "Reconstruction-only warmup epochs before fitting the frozen Z1 "
            "semantic probe in staged_z1_aligned mode."
        ),
    )
    parser.add_argument("--semantic-probe-hidden-dim", type=int, default=512)
    parser.add_argument("--semantic-probe-epochs", type=int, default=30)
    parser.add_argument("--semantic-probe-lr", type=float, default=1e-3)
    parser.add_argument(
        "--semantic-probe-weight-decay", type=float, default=1e-4
    )
    parser.add_argument(
        "--run-matched-latent-probe-audit",
        action="store_true",
        help=(
            "After training, fit capacity-matched frozen Z1->Tj and Z2->Tj "
            "probes on the same splits. This audit never selects modules."
        ),
    )
    parser.add_argument(
        "--lambda-main-semantic-cluster-mmd",
        type=float,
        default=0.0,
        help=(
            "Match nonlinear configured semantic-source distributions across "
            "soft residual-score clusters during first-stage training."
        ),
    )
    parser.add_argument(
        "--semantic-cluster-mmd-projection-dim", type=int, default=128
    )
    parser.add_argument(
        "--semantic-cluster-mmd-temperature", type=float, default=0.35
    )
    parser.add_argument(
        "--semantic-cluster-mmd-thresholds",
        nargs="*",
        type=float,
        default=[-0.75, 0.0, 0.75],
    )
    parser.add_argument("--lambda-var", type=float, default=0.01)
    parser.add_argument("--var-floor", type=float, default=0.25)
    parser.add_argument("--meta-start-epoch", type=int, default=20)
    parser.add_argument(
        "--semantic-freeze-epoch",
        type=int,
        default=-1,
        help=(
            "Freeze E1, its previous-layer decoder and the semantic target teacher "
            "before residual optimization. Negative preserves the legacy direction-freeze timing."
        ),
    )
    parser.add_argument("--semantic-invariance-start-epoch", type=int, default=0)
    parser.add_argument("--semantic-invariance-rf-floor", type=float, default=0.0)
    parser.add_argument("--semantic-invariance-gain-floor", type=float, default=0.0)
    parser.add_argument("--semantic-invariance-gate-margin", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--checkpoint-start-epoch", type=int, default=55)
    parser.add_argument("--checkpoint-residual-gain-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-total-gain-weight", type=float, default=0.02)
    parser.add_argument("--checkpoint-residual-fraction-weight", type=float, default=0.20)
    parser.add_argument("--checkpoint-positive-module-weight", type=float, default=0.002)
    parser.add_argument("--checkpoint-min-residual-gain", type=float, default=0.0)
    parser.add_argument("--checkpoint-min-residual-fraction", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-semantic-cluster-mmd-weight", type=float, default=0.0
    )
    parser.add_argument("--semantic-bacc-proxy-dim", type=int, default=64)
    parser.add_argument("--semantic-bacc-proxy-ridge", type=float, default=1.0)
    parser.add_argument(
        "--meta-upper-bound-mode",
        choices=["none", "ridge"],
        default="none",
        help=(
            "Report an optimistic meta-only predictive upper proxy. It is "
            "not used for training or checkpoint selection."
        ),
    )
    parser.add_argument("--meta-upper-bound-ridge", type=float, default=1.0)
    parser.add_argument(
        "--meta-upper-bound-dim",
        type=int,
        default=0,
        help="Number of Z2 dimensions used by the upper-bound ridge probe; 0 uses all.",
    )
    parser.add_argument(
        "--lambda-semantic-bacc-absorption",
        type=float,
        default=0.0,
        help=(
            "Natural semantic-head absorption loss. The semantic branch is "
            "trained to preserve soft module-state boundaries, so residual "
            "code is not forced to carry them. This is not adversarial noise."
        ),
    )
    parser.add_argument(
        "--semantic-bacc-absorption-temperature",
        type=float,
        default=0.35,
        help="Temperature for the soft module-state targets used by absorption.",
    )
    parser.add_argument(
        "--semantic-bacc-absorption-start-epoch",
        type=int,
        default=0,
        help="Epoch at which the semantic soft-boundary absorption loss starts.",
    )
    parser.add_argument(
        "--semantic-bacc-absorption-unfreeze-semantic",
        action="store_true",
        help=(
            "Allow the semantic decoder to update under the absorption loss "
            "after staged/frozen probe initialization. Z1 remains frozen in "
            "staged mode, preserving the decomposition input."
        ),
    )
    parser.add_argument("--checkpoint-semantic-bacc-proxy-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-max-semantic-bacc-proxy", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-continuous-semantic-r2-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty on validation R2 when nonlinear semantic signatures "
            "predict continuous module residual scores."
        ),
    )
    parser.add_argument(
        "--checkpoint-max-continuous-semantic-r2",
        type=float,
        default=1.0,
        help="Validation ceiling for the continuous semantic R2 proxy.",
    )
    parser.add_argument(
        "--second-stage",
        choices=["purifier", "main_direct"],
        default="purifier",
        help=(
            "Run the recursive purifier or export/refine the first-stage E2 residual "
            "representation directly. main_direct avoids purifier compression loss."
        ),
    )
    parser.add_argument(
        "--reuse-main-dir",
        default="",
        help=(
            "Optional joint-v2 output directory containing best_model.pt and "
            "joint_module_heads.pt. Reuses its first-stage optimum and only runs the "
            "requested second-stage/module export."
        ),
    )
    parser.add_argument("--purifier-epochs", type=int, default=70)
    parser.add_argument("--purifier-semantic-dim", type=int, default=256)
    parser.add_argument("--purifier-meta-dim", type=int, default=32)
    parser.add_argument("--purifier-hidden-dim", type=int, default=1024)
    parser.add_argument("--purifier-lr", type=float, default=1e-4)
    parser.add_argument("--purifier-semantic-warmup", type=int, default=14)
    parser.add_argument("--purifier-z2-warmup", type=int, default=10)
    parser.add_argument("--purifier-direction-refresh-start", type=int, default=25)
    parser.add_argument("--purifier-direction-refresh-every", type=int, default=5)
    parser.add_argument("--purifier-direction-freeze-epoch", type=int, default=40)
    parser.add_argument(
        "--purifier-direction-mode",
        choices=["fixed_main", "adaptive"],
        default="adaptive",
        help="Keep first-stage module identities fixed or rediscover them in purified-meta space.",
    )
    parser.add_argument("--lambda-purifier-semantic-z2", type=float, default=0.30)
    parser.add_argument("--lambda-purifier-semantic", type=float, default=1.20)
    parser.add_argument("--lambda-purifier-meta-z2", type=float, default=0.40)
    parser.add_argument("--lambda-purifier-recon-z2", type=float, default=0.08)
    parser.add_argument("--lambda-purifier-semantic-module", type=float, default=0.20)
    parser.add_argument("--lambda-purifier-residual-module", type=float, default=0.20)
    parser.add_argument("--lambda-purifier-joint-module", type=float, default=0.10)
    parser.add_argument("--lambda-purifier-main-residual-distill", type=float, default=0.0)
    parser.add_argument("--lambda-purifier-orth", type=float, default=0.10)
    parser.add_argument("--lambda-purifier-meta-sem-cov", type=float, default=1.0)
    parser.add_argument("--lambda-purifier-var", type=float, default=0.01)
    parser.add_argument("--purifier-checkpoint-start-epoch", type=int, default=40)
    parser.add_argument("--module-code-dim", type=int, default=8)
    parser.add_argument(
        "--module-refiner-code-mode",
        choices=["latent", "task_aligned"],
        default="latent",
        help=(
            "Cluster/intervene on the refiner bottleneck or on its one-dimensional "
            "module-loading-aligned residual prediction."
        ),
    )
    parser.add_argument("--module-refiner-hidden-dim", type=int, default=192)
    parser.add_argument("--module-refiner-epochs", type=int, default=80)
    parser.add_argument("--module-refiner-lr", type=float, default=3e-4)
    parser.add_argument(
        "--module-refiner-workers",
        type=int,
        default=1,
        help=(
            "Number of single-GPU subprocesses used after module directions are "
            "frozen. Each worker refines and audits a disjoint module shard; "
            "joint direction discovery remains in the parent process."
        ),
    )
    parser.add_argument("--lambda-module-refiner-semantic-cov", type=float, default=0.0)
    parser.add_argument(
        "--lambda-module-refiner-semantic-cluster-mmd",
        type=float,
        default=0.0,
        help=(
            "Directly match semantic-control distributions across soft refiner-code "
            "clusters; unlike covariance loss this targets the final BAcc partition."
        ),
    )
    parser.add_argument(
        "--module-refiner-checkpoint-mmd-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--module-refiner-min-relative-gain", type=float, default=0.02
    )
    parser.add_argument("--module-refiner-invariance-start-epoch", type=int, default=10)
    parser.add_argument("--module-refiner-invariance-gain-margin", type=float, default=0.02)
    parser.add_argument("--module-refiner-checkpoint-bacc-weight", type=float, default=0.0)
    parser.add_argument("--module-refiner-checkpoint-bacc-target", type=float, default=0.50)
    parser.add_argument("--module-refiner-max-semantic-bacc-proxy", type=float, default=1.0)
    parser.add_argument(
        "--module-refiner-semantic-projection-dim",
        type=int,
        default=64,
        help=(
            "Random-projection width used by the non-adversarial nonlinear semantic "
            "covariance control. Set 0 to use raw semantic features."
        ),
    )
    parser.add_argument(
        "--module-code-semantic-residualization",
        choices=[
            "none",
            "crossfit_ridge",
            "crossfit_strongest",
            "crossfit_location_scale",
            "crossfit_knn_rank",
        ],
        default="none",
        help=(
            "Optionally remove the component of each continuous module target that "
            "is predictable from held-out semantic controls before fitting the module "
            "refiner. ridge removes conditional location, location_scale also normalizes "
            "conditional dispersion, and knn_rank converts the target to a train-neighbor "
            "conditional quantile."
        ),
    )
    parser.add_argument(
        "--module-contribution-mode",
        choices=["residual_target", "conditional_incremental"],
        default="residual_target",
        help=(
            "residual_target preserves the legacy path that asks a meta-only "
            "refiner to reconstruct a sample-wise semantic-residualized target. "
            "conditional_incremental instead trains the refiner on the learnable "
            "pre-subtraction module residual, then calibrates its held-out "
            "incremental contribution over the same strong semantic baseline."
        ),
    )
    parser.add_argument(
        "--module-semantic-probe-profile",
        choices=["legacy_posthoc", "aligned_training_head"],
        default="legacy_posthoc",
        help=(
            "aligned_training_head reuses the frozen previous-layer semantic "
            "teacher as the RF baseline and disables an additional posthoc "
            "redefinition of semantic information."
        ),
    )
    parser.add_argument(
        "--module-conditional-calibration-ridge",
        type=float,
        default=1e-3,
        help=(
            "Train-only ridge used to calibrate a scalar task-aligned meta code "
            "against the residual left by the strongest semantic baseline."
        ),
    )
    parser.add_argument(
        "--module-code-semantic-ridge",
        type=float,
        default=1e-2,
        help="Ridge coefficient for cross-fitted semantic module-target prediction.",
    )
    parser.add_argument(
        "--module-code-semantic-folds",
        type=int,
        default=5,
        help="Training-fold count used by semantic module-target residualization.",
    )
    parser.add_argument(
        "--module-code-semantic-mlp-hidden-dim",
        type=int,
        default=128,
        help="Hidden width for the nonlinear cross-fitted semantic teacher.",
    )
    parser.add_argument(
        "--module-code-semantic-mlp-epochs",
        type=int,
        default=12,
        help="Epochs per fold for the nonlinear semantic teacher.",
    )
    parser.add_argument(
        "--module-code-semantic-mlp-repeats",
        type=int,
        default=1,
        help="Independent MLP teachers averaged within each cross-fit fold.",
    )
    parser.add_argument(
        "--module-code-semantic-mlp-dropout",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--module-code-semantic-mlp-lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--module-code-semantic-mlp-weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--module-code-semantic-removal-strength",
        type=float,
        default=1.0,
        help=(
            "Fraction of the cross-fitted semantic prediction removed from the module "
            "target; values above one provide an explicit over-removal stress test."
        ),
    )
    parser.add_argument(
        "--module-code-semantic-knn-k",
        type=int,
        default=128,
        help="Semantic-neighbor count for cross-fitted conditional-rank codes.",
    )
    parser.add_argument(
        "--module-code-semantic-knn-dim",
        type=int,
        default=32,
        help="Leading compact semantic-control dimensions used for KNN matching.",
    )
    parser.add_argument(
        "--module-code-semantic-scale-min",
        type=float,
        default=0.10,
        help="Minimum relative conditional scale in location-scale residualization.",
    )
    parser.add_argument(
        "--module-code-semantic-scale-max",
        type=float,
        default=10.0,
        help="Maximum relative conditional scale in location-scale residualization.",
    )
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument(
        "--module-cluster-policy",
        choices=["forced", "diagnostic", "require_multimodal"],
        default="forced",
        help=(
            "forced preserves legacy KMeans evidence; diagnostic exports clusters "
            "but treats them as descriptive unless a train-fitted mixture generalizes; "
            "require_multimodal also requires that evidence for strict module gates."
        ),
    )
    parser.add_argument(
        "--module-multimodality-min-bic-gain",
        type=float,
        default=10.0,
        help="Minimum train BIC(1 Gaussian)-BIC(2 Gaussian) supporting two states.",
    )
    parser.add_argument(
        "--module-multimodality-min-heldout-ll-gain",
        type=float,
        default=0.01,
        help=(
            "Minimum per-sample held-out log-likelihood gain of the train-fitted "
            "two-component mixture over a single Gaussian."
        ),
    )
    parser.add_argument(
        "--module-multimodality-min-separation",
        type=float,
        default=1.50,
        help="Minimum standardized separation between the two fitted component means.",
    )
    parser.add_argument(
        "--module-multimodality-min-component-fraction",
        type=float,
        default=0.10,
        help="Minimum fitted mixture weight for either component.",
    )
    parser.add_argument(
        "--module-multimodality-max-valley-ratio",
        type=float,
        default=0.80,
        help=(
            "Maximum mixture density between component means relative to the weaker "
            "component peak. Lower values provide clearer evidence of two modes."
        ),
    )
    parser.add_argument(
        "--module-continuous-semantic-probe-folds",
        type=int,
        default=5,
        help="Cross-fitting folds for semantic prediction of the continuous module code.",
    )
    parser.add_argument(
        "--module-continuous-semantic-probe-ridge",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--module-selection-continuous-semantic-r2-penalty",
        type=float,
        default=0.25,
        help=(
            "Selection penalty on positive semantic R2 of a continuous module when "
            "held-out multimodality is not supported."
        ),
    )
    parser.add_argument(
        "--module-selection-max-continuous-semantic-r2",
        type=float,
        default=1.0,
        help=(
            "Selection-split ceiling for the strongest continuous semantic "
            "probe. Values >=1 disable this gate."
        ),
    )
    parser.add_argument("--clip-quantile", type=float, default=0.99)
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--kmeans-restarts", type=int, default=20)
    parser.add_argument("--semantic-cluster-epochs", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutation-tests", type=int, default=2000)
    parser.add_argument("--export-min-residual-fraction", type=float, default=0.20)
    parser.add_argument("--module-selection-residual-gain-weight", type=float, default=1.0)
    parser.add_argument("--module-selection-total-gain-weight", type=float, default=0.02)
    parser.add_argument("--module-selection-residual-fraction-weight", type=float, default=0.40)
    # The active pipeline uses continuous semantic predictability instead of a
    # discretized BAcc gate. These defaults satisfy legacy export internals.
    parser.set_defaults(
        export_max_semantic_bacc=1.0,
        bacc_as_diagnostic=True,
        module_selection_bacc_penalty=0.0,
        module_selection_bacc_target=1.0,
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def split_base_id(sample_id: str, pattern: str) -> str:
    value = str(sample_id)
    return re.sub(pattern, "", value)


def deterministic_split(
    ids: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_by_base_id: bool = False,
    base_id_step_pattern: str = r"::step\d+$",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    n = len(ids)
    if n < 10:
        raise ValueError("At least 10 activation rows are required.")
    if not split_by_base_id:
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        test_n = max(1, int(n * test_ratio)) if test_ratio > 0.0 else 0
        val_n = max(1, int(n * val_ratio))
        if test_n + val_n >= n:
            raise ValueError("Validation/test ratios leave no training rows.")
        test = order[:test_n]
        val = order[test_n : test_n + val_n]
        train = order[test_n + val_n :]
        return train, val, test, {
            "mode": "row_random",
            "seed": seed,
            "total_rows": n,
            "train_rows": int(train.numel()),
            "val_rows": int(val.numel()),
            "test_rows": int(test.numel()),
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }

    groups: Dict[str, List[int]] = {}
    for row_index, sample_id in enumerate(ids):
        groups.setdefault(split_base_id(str(sample_id), base_id_step_pattern), []).append(row_index)
    group_ids = list(groups)
    if len(group_ids) < 10:
        raise ValueError("At least 10 base prompt groups are required for grouped splitting.")
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    test_groups = max(1, int(len(group_ids) * test_ratio)) if test_ratio > 0.0 else 0
    val_groups = max(1, int(len(group_ids) * val_ratio))
    if test_groups + val_groups >= len(group_ids):
        raise ValueError("Validation/test ratios leave no training groups.")
    test_group_ids = set(group_ids[:test_groups])
    val_group_ids = set(group_ids[test_groups : test_groups + val_groups])
    train_indices, val_indices, test_indices = [], [], []
    for base_id, row_indices in groups.items():
        if base_id in test_group_ids:
            test_indices.extend(row_indices)
        elif base_id in val_group_ids:
            val_indices.extend(row_indices)
        else:
            train_indices.extend(row_indices)
    train = torch.tensor(sorted(train_indices), dtype=torch.long)
    val = torch.tensor(sorted(val_indices), dtype=torch.long)
    test = torch.tensor(sorted(test_indices), dtype=torch.long)
    return train, val, test, {
        "mode": "group_by_base_id",
        "seed": seed,
        "base_id_step_pattern": base_id_step_pattern,
        "total_rows": n,
        "total_groups": len(group_ids),
        "train_groups": len(group_ids) - len(test_group_ids) - len(val_group_ids),
        "val_groups": len(val_group_ids),
        "test_groups": len(test_group_ids),
        "train_rows": int(train.numel()),
        "val_rows": int(val.numel()),
        "test_rows": int(test.numel()),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }


def standardize_from_train(
    value: torch.Tensor, train_idx: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = value.index_select(0, train_idx)
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return ((value - mean) / std).contiguous(), mean, std


def load_external_semantic_features(
    activation_dir: str,
    layer: int,
    target_ids: Sequence[str],
    train_idx: torch.Tensor,
    max_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load and train-standardize an independent model's aligned activation layer."""
    path = os.path.join(activation_dir, f"layer_{int(layer):03d}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"External semantic layer cache not found: {path}")
    external_ids, raw = base.load_layer(path)
    index_by_id = {str(sample_id): index for index, sample_id in enumerate(external_ids)}
    missing = [str(sample_id) for sample_id in target_ids if str(sample_id) not in index_by_id]
    if missing:
        raise ValueError(
            "External semantic activation ids do not cover the target cache; "
            f"missing={len(missing)} examples={missing[:5]}"
        )
    order = torch.tensor(
        [index_by_id[str(sample_id)] for sample_id in target_ids], dtype=torch.long
    )
    aligned = raw.index_select(0, order).float()
    if max_dim > 0:
        if max_dim > aligned.size(1):
            raise ValueError(
                f"--external-semantic-max-dim={max_dim} exceeds external width {aligned.size(1)}"
            )
        aligned = aligned[:, :max_dim].contiguous()
    standardized, mean, std = standardize_from_train(aligned, train_idx)
    info = {
        "activation_dir": activation_dir,
        "layer": int(layer),
        "raw_dim": int(raw.size(1)),
        "dim_used": int(standardized.size(1)),
        "row_count": int(standardized.size(0)),
        "source": "external_model_activation",
        "standardization": "target-cache train split mean/std",
    }
    return standardized, mean, std, info


def fit_external_to_candidate_map(
    external_features: torch.Tensor,
    candidate_values: torch.Tensor,
    train_idx: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fit the frozen external-semantic proxy used to form residual targets."""
    coefficients = ridge_coefficients(
        external_features.index_select(0, train_idx),
        candidate_values.index_select(0, train_idx),
        ridge,
    )
    with torch.inference_mode():
        design = torch.cat(
            [external_features.float(), torch.ones(external_features.size(0), 1)],
            dim=1,
        )
        prediction = design @ coefficients
    rows = {}
    for name, index in (("train", train_idx),):
        target = candidate_values.index_select(0, index)
        pred = prediction.index_select(0, index)
        mse = F.mse_loss(pred, target)
        prior = F.mse_loss(
            target,
            target.mean(dim=0, keepdim=True).expand_as(target),
        )
        rows[name] = {
            "n": int(index.numel()),
            "mse": float(mse.item()),
            "prior_mse": float(prior.item()),
            "r2_vs_train_mean": float(1.0 - mse.item() / max(prior.item(), 1e-12)),
        }
    return coefficients, {
        "ridge": float(ridge),
        "input_dim": int(external_features.size(1)),
        "output_dim": int(candidate_values.size(1)),
        "fit_split": "train_only",
        "train_fit": rows["train"],
    }


def binary_entropy(rate: torch.Tensor) -> torch.Tensor:
    p = rate.clamp(1e-6, 1.0 - 1e-6)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def select_candidate_pool(
    raw: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], torch.Tensor]:
    train_raw = raw.index_select(0, train_idx)
    val_raw = raw.index_select(0, val_idx)
    threshold = torch.quantile(train_raw, args.binary_quantile, dim=0)
    train_rate = (train_raw >= threshold).float().mean(dim=0)
    val_rate = (val_raw >= threshold).float().mean(dim=0)
    drift = (train_rate - val_rate).abs()
    train_std = train_raw.std(dim=0, unbiased=False)
    val_std = val_raw.std(dim=0, unbiased=False)
    std_ratio = torch.minimum(train_std, val_std) / torch.maximum(train_std, val_std).clamp_min(1e-6)
    entropy = torch.minimum(binary_entropy(train_rate), binary_entropy(val_rate))
    stable = (
        (train_rate >= args.candidate_min_rate)
        & (train_rate <= args.candidate_max_rate)
        & (val_rate >= args.candidate_min_rate)
        & (val_rate <= args.candidate_max_rate)
        & (drift <= args.candidate_max_rate_drift)
        & (train_std > 1e-6)
        & (val_std > 1e-6)
    )
    score = entropy * std_ratio * torch.exp(-drift / 0.05)
    score = torch.where(stable, score, torch.full_like(score, -math.inf))
    eligible = torch.isfinite(score).nonzero(as_tuple=False).flatten()
    if eligible.numel() < args.num_modules * max(2, args.module_support // 2):
        relaxed = (
            (train_rate > 0.01)
            & (train_rate < 0.99)
            & (val_rate > 0.01)
            & (val_rate < 0.99)
            & (train_std > 1e-6)
            & (val_std > 1e-6)
        )
        score = torch.where(relaxed, entropy * std_ratio * torch.exp(-drift / 0.10), torch.full_like(score, -math.inf))
        eligible = torch.isfinite(score).nonzero(as_tuple=False).flatten()
        fallback = True
    else:
        fallback = False
    if eligible.numel() < args.num_modules:
        raise ValueError(f"Only {eligible.numel()} stable candidate neurons remain.")
    take = min(args.candidate_top_k, int(eligible.numel()))
    selected = torch.topk(score, k=take).indices.sort().values
    rows = []
    selected_set = set(selected.tolist())
    for neuron in range(raw.size(1)):
        if neuron not in selected_set:
            continue
        rows.append(
            {
                "neuron": neuron,
                "candidate_score": float(score[neuron].item()),
                "train_std": float(train_std[neuron].item()),
                "val_std": float(val_std[neuron].item()),
                "std_stability_ratio": float(std_ratio[neuron].item()),
                "activation_rate_train": float(train_rate[neuron].item()),
                "activation_rate_val": float(val_rate[neuron].item()),
                "activation_rate_drift_abs": float(drift[neuron].item()),
                "prior_entropy_min_nats": float(entropy[neuron].item()),
                "selection_uses_target_predictability": False,
                "relaxed_stability_fallback": fallback,
            }
        )
    rows.sort(key=lambda row: row["candidate_score"], reverse=True)
    return selected, threshold, rows, stable


def centered_covariance_penalty(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = (left - left.mean(dim=0, keepdim=True)) / left.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-4)
    right = (right - right.mean(dim=0, keepdim=True)) / right.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-4)
    return (left.t() @ right / max(left.size(0), 1)).square().mean()


def fixed_random_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a fixed projection used only to expose broad nonlinear semantics."""
    width = min(max(1, int(output_dim)), max(1, int(input_dim)))
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, width, generator=generator, dtype=torch.float32)
    matrix /= math.sqrt(max(input_dim, 1))
    return matrix.to(device)


def nonlinear_semantic_signature(
    semantic: torch.Tensor,
    projection: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compact semantic features with location, saturation and scale components."""
    value = semantic.float().detach()
    if projection is not None:
        value = value @ projection.to(value)
    value = (value - value.mean(dim=0, keepdim=True)) / value.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-4)
    squared = value.square()
    squared = squared - squared.mean(dim=0, keepdim=True)
    return torch.cat([value, torch.tanh(value), squared], dim=1)


def soft_cluster_semantic_mmd(
    score: torch.Tensor,
    semantic_signature: torch.Tensor,
    temperature: float,
    thresholds: Sequence[float],
) -> torch.Tensor:
    """Differentiable semantic separation of score-induced soft clusters.

    The semantic signature is fixed. Gradients pass through the soft cluster
    memberships into residual/module scores, aligning the training penalty with
    the downstream cluster BAcc question without an adversarial classifier.
    """
    if score.ndim == 1:
        score = score.view(-1, 1)
    standardized = (score - score.mean(dim=0, keepdim=True)) / score.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-4)
    signature = semantic_signature.float()
    signature = (signature - signature.mean(dim=0, keepdim=True)) / signature.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-4)
    values = []
    threshold_values = list(thresholds) or [0.0]
    tau = max(float(temperature), 1e-3)
    for threshold in threshold_values:
        membership = torch.sigmoid((standardized - float(threshold)) / tau)
        positive_mass = membership.sum(dim=0).clamp_min(1.0)
        negative = 1.0 - membership
        negative_mass = negative.sum(dim=0).clamp_min(1.0)
        positive_mean = membership.t().matmul(signature) / positive_mass.view(-1, 1)
        negative_mean = negative.t().matmul(signature) / negative_mass.view(-1, 1)
        values.append((positive_mean - negative_mean).square().mean())
    return torch.stack(values).mean()


def semantic_bacc_absorption_loss(
    semantic_prediction: torch.Tensor,
    module_target: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Make the semantic branch absorb score-boundary information naturally.

    Downstream BAcc is induced by thresholding continuous module scores.  This
    loss trains the semantic prediction to preserve the same soft threshold
    probabilities.  It is a teacher-side supervised loss, not a gradient
    reversal/adversarial perturbation, and therefore moves the predictable
    boundary information into the semantic component before residualization.
    """
    tau = max(float(temperature), 1e-3)
    center = center.to(semantic_prediction).view(1, -1)
    scale = scale.to(semantic_prediction).view(1, -1).clamp_min(1e-4)
    target_probability = torch.sigmoid(
        (module_target.detach() - center) / (scale * tau)
    )
    semantic_logit = (semantic_prediction - center) / (scale * tau)
    return F.binary_cross_entropy_with_logits(
        semantic_logit, target_probability
    )


def variance_floor(value: torch.Tensor, floor: float) -> torch.Tensor:
    return F.relu(floor - value.std(dim=0, unbiased=False)).square().mean()


def ridge_coefficients(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    x = x.float()
    y = y.float()
    x = torch.cat([x, torch.ones(x.size(0), 1, dtype=x.dtype)], dim=1)
    gram = x.t() @ x
    eye = torch.eye(gram.size(0), dtype=gram.dtype)
    eye[-1, -1] = 0.0
    system = gram + float(ridge) * eye
    rhs = x.t() @ y
    try:
        return torch.linalg.solve(system, rhs)
    except RuntimeError:
        return torch.linalg.pinv(system) @ rhs


def binary_balanced_accuracy(
    target: torch.Tensor, prediction: torch.Tensor
) -> float:
    target = target.bool()
    prediction = prediction.bool()
    recalls = []
    for value in (False, True):
        mask = target == value
        if mask.any():
            recalls.append(float((prediction[mask] == value).float().mean().item()))
    return float(sum(recalls) / len(recalls)) if recalls else math.nan


def semantic_score_bacc_proxy(
    train_semantic: torch.Tensor,
    train_score: torch.Tensor,
    eval_semantic: torch.Tensor,
    eval_score: torch.Tensor,
    ridge: float,
    max_dim: int,
) -> Dict[str, Any]:
    """Cheap cross-split proxy for the final semantic-cluster BAcc probe.

    Labels are induced only from train-score medians. A nonlinear semantic
    signature predicts those labels on train and a disjoint evaluation split.
    This is used for checkpoint selection, never as final evidence.
    """
    width = min(max(1, int(max_dim)), train_semantic.size(1))
    train_x = train_semantic[:, :width].float()
    eval_x = eval_semantic[:, :width].float()
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    eval_x = (eval_x - mean) / std
    train_score = train_score.float()
    eval_score = eval_score.float()
    if train_score.ndim == 1:
        train_score = train_score.view(-1, 1)
    if eval_score.ndim == 1:
        eval_score = eval_score.view(-1, 1)
    threshold = train_score.median(dim=0).values
    train_labels = (train_score > threshold).float()
    eval_labels = eval_score > threshold
    coefficients = ridge_coefficients(train_x, train_labels, ridge)
    train_design = torch.cat(
        [train_x, torch.ones(train_x.size(0), 1, dtype=train_x.dtype)], dim=1
    )
    eval_design = torch.cat(
        [eval_x, torch.ones(eval_x.size(0), 1, dtype=eval_x.dtype)], dim=1
    )
    train_prediction = train_design @ coefficients >= 0.5
    eval_prediction = eval_design @ coefficients >= 0.5
    train_values, eval_values = [], []
    for column in range(train_labels.size(1)):
        train_values.append(
            binary_balanced_accuracy(
                train_labels[:, column] > 0.5, train_prediction[:, column]
            )
        )
        eval_values.append(
            binary_balanced_accuracy(eval_labels[:, column], eval_prediction[:, column])
        )
    return {
        "train_mean": float(sum(train_values) / len(train_values)),
        "train_max": float(max(train_values)),
        "eval_mean": float(sum(eval_values) / len(eval_values)),
        "eval_max": float(max(eval_values)),
        "per_score_train": train_values,
        "per_score_eval": eval_values,
        "semantic_dim": width,
    }


def semantic_score_r2_proxy(
    train_semantic: torch.Tensor,
    train_score: torch.Tensor,
    eval_semantic: torch.Tensor,
    eval_score: torch.Tensor,
    ridge: float,
    max_dim: int,
) -> Dict[str, Any]:
    """Cross-split continuous leakage proxy used for checkpoint selection."""
    width = min(max(1, int(max_dim)), train_semantic.size(1))
    train_x = train_semantic[:, :width].float()
    eval_x = eval_semantic[:, :width].float()
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    eval_x = (eval_x - mean) / std
    train_score = train_score.float()
    eval_score = eval_score.float()
    if train_score.ndim == 1:
        train_score = train_score.view(-1, 1)
    if eval_score.ndim == 1:
        eval_score = eval_score.view(-1, 1)
    coefficients = ridge_coefficients(train_x, train_score, ridge)
    train_design = torch.cat(
        [train_x, torch.ones(train_x.size(0), 1)], dim=1
    )
    eval_design = torch.cat(
        [eval_x, torch.ones(eval_x.size(0), 1)], dim=1
    )
    train_prediction = train_design @ coefficients
    eval_prediction = eval_design @ coefficients

    def r2_columns(
        target: torch.Tensor, prediction: torch.Tensor
    ) -> List[float]:
        mse = (target - prediction).square().mean(dim=0)
        baseline = (
            target - target.mean(dim=0, keepdim=True)
        ).square().mean(dim=0).clamp_min(1e-12)
        return [float(value) for value in (1.0 - mse / baseline).tolist()]

    train_values = r2_columns(train_score, train_prediction)
    eval_values = r2_columns(eval_score, eval_prediction)
    return {
        "train_mean": float(sum(train_values) / len(train_values)),
        "train_max": float(max(train_values)),
        "eval_mean": float(sum(eval_values) / len(eval_values)),
        "eval_max": float(max(eval_values)),
        "per_score_train": train_values,
        "per_score_eval": eval_values,
        "semantic_dim": width,
    }


def meta_only_upper_bound_proxy(
    train_meta: torch.Tensor,
    train_target: torch.Tensor,
    eval_meta: torch.Tensor,
    eval_target: torch.Tensor,
    eval_semantic_prediction: torch.Tensor,
    ridge: float,
    max_dim: int,
) -> Dict[str, Any]:
    """Estimate an optimistic upper proxy for information available in Z2.

    This intentionally fits ``Z2 -> target`` without conditioning on the
    semantic branch.  Therefore it can include semantic leakage and must not
    be interpreted as a clean meta contribution.  It is useful as an upper
    comparison against the residualized RF: if this value is much larger than
    RF, the residualized estimate may be conservative or the semantic control
    may be absorbing overlapping information.

    The second bound is model-free: after the fixed semantic predictor, the
    remaining semantic MSE is the maximum additional reduction any perfect
    residual predictor could obtain under this decomposition.
    """
    train_meta = train_meta.float()
    eval_meta = eval_meta.float()
    train_target = train_target.float()
    eval_target = eval_target.float()
    eval_semantic_prediction = eval_semantic_prediction.float()
    if train_target.ndim == 1:
        train_target = train_target.view(-1, 1)
    if eval_target.ndim == 1:
        eval_target = eval_target.view(-1, 1)
    if eval_semantic_prediction.ndim == 1:
        eval_semantic_prediction = eval_semantic_prediction.view(-1, 1)

    width = train_meta.size(1)
    if max_dim > 0:
        width = min(width, int(max_dim))
    variance = train_meta.var(dim=0, unbiased=False)
    selected = torch.argsort(variance, descending=True)[:width]
    train_x = train_meta.index_select(1, selected)
    eval_x = eval_meta.index_select(1, selected)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    eval_x = (eval_x - mean) / std
    coefficients = ridge_coefficients(train_x, train_target, ridge)
    train_design = torch.cat(
        [train_x, torch.ones(train_x.size(0), 1, dtype=train_x.dtype)], dim=1
    )
    eval_design = torch.cat(
        [eval_x, torch.ones(eval_x.size(0), 1, dtype=eval_x.dtype)], dim=1
    )
    train_prediction = train_design @ coefficients
    eval_prediction = eval_design @ coefficients

    prior_error = (
        eval_target - train_target.mean(dim=0, keepdim=True)
    ).square()
    semantic_error = (eval_target - eval_semantic_prediction).square()
    upper_error = (eval_target - eval_prediction).square()
    meta_only_gain = prior_error.mean(dim=0) - upper_error.mean(dim=0)
    positive_meta_gain = meta_only_gain.clamp_min(0.0).sum()
    prior_mass = prior_error.mean(dim=0).clamp_min(1e-8).sum()
    semantic_residual_mass = semantic_error.mean(dim=0).clamp_min(1e-8).sum()
    residual_capacity = semantic_residual_mass / prior_mass
    return {
        "probe": "ridge_z2_to_target_without_semantic_conditioning",
        "warning": (
            "optimistic upper proxy; it may include semantic leakage and is "
            "not a clean meta-information estimate"
        ),
        "feature_dim": int(width),
        "train_mse": float(
            (train_target - train_prediction).square().mean().item()
        ),
        "eval_mse": float(upper_error.mean().item()),
        "eval_gain_mse": float(meta_only_gain.mean().item()),
        "eval_gain_mse_positive_mean": float(
            meta_only_gain.clamp_min(0.0).mean().item()
        ),
        "meta_only_fraction_of_total": float(
            (positive_meta_gain / prior_mass).item()
        ),
        "meta_only_fraction_of_semantic_residual": float(
            (positive_meta_gain / semantic_residual_mass).item()
        ),
        "semantic_residual_capacity_fraction_of_total": float(
            residual_capacity.item()
        ),
        "semantic_residual_capacity_fraction_of_residual": 1.0,
        "per_module_meta_only_gain_mse": [
            float(value) for value in meta_only_gain.tolist()
        ],
        "per_module_semantic_residual_mse": [
            float(value) for value in semantic_error.mean(dim=0).tolist()
        ],
    }


def split_semantic_bacc_proxy(
    train_z1: torch.Tensor,
    eval_z1: torch.Tensor,
    train_prev: torch.Tensor,
    eval_prev: torch.Tensor,
    train_score: torch.Tensor,
    eval_score: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    semantic_override_train: Optional[torch.Tensor] = None,
    semantic_override_eval: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if semantic_override_train is not None or semantic_override_eval is not None:
        if semantic_override_train is None or semantic_override_eval is None:
            raise ValueError("Both semantic override splits are required.")
        return semantic_score_bacc_proxy(
            semantic_override_train,
            train_score,
            semantic_override_eval,
            eval_score,
            args.semantic_bacc_proxy_ridge,
            args.semantic_bacc_proxy_dim,
        )
    train_n = train_z1.size(0)
    combined_z1 = torch.cat([train_z1, eval_z1], dim=0)
    combined_prev = torch.cat([train_prev, eval_prev], dim=0)
    semantic_parts = (
        [combined_prev]
        if args.semantic_candidate_source == "prev"
        else (
            [combined_z1]
            if args.semantic_candidate_source == "z1"
            else [combined_z1, combined_prev]
        )
    )
    semantic = build_module_semantic_controls(
        semantic_parts,
        torch.arange(train_n),
        max(8, int(math.ceil(args.semantic_bacc_proxy_dim / 3))),
        seed,
    )
    return semantic_score_bacc_proxy(
        semantic[:train_n],
        train_score,
        semantic[train_n:],
        eval_score,
        args.semantic_bacc_proxy_ridge,
        args.semantic_bacc_proxy_dim,
    )


def split_semantic_continuous_proxy(
    train_z1: torch.Tensor,
    eval_z1: torch.Tensor,
    train_prev: torch.Tensor,
    eval_prev: torch.Tensor,
    train_score: torch.Tensor,
    eval_score: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    semantic_override_train: Optional[torch.Tensor] = None,
    semantic_override_eval: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if semantic_override_train is not None or semantic_override_eval is not None:
        if semantic_override_train is None or semantic_override_eval is None:
            raise ValueError("Both semantic override splits are required.")
        return semantic_score_r2_proxy(
            semantic_override_train,
            train_score,
            semantic_override_eval,
            eval_score,
            args.semantic_bacc_proxy_ridge,
            args.semantic_bacc_proxy_dim,
        )
    train_n = train_z1.size(0)
    combined_z1 = torch.cat([train_z1, eval_z1], dim=0)
    combined_prev = torch.cat([train_prev, eval_prev], dim=0)
    semantic_parts = (
        [combined_prev]
        if args.semantic_candidate_source == "prev"
        else (
            [combined_z1]
            if args.semantic_candidate_source == "z1"
            else [combined_z1, combined_prev]
        )
    )
    semantic = build_module_semantic_controls(
        semantic_parts,
        torch.arange(train_n),
        max(8, int(math.ceil(args.semantic_bacc_proxy_dim / 3))),
        seed,
    )
    return semantic_score_r2_proxy(
        semantic[:train_n],
        train_score,
        semantic[train_n:],
        eval_score,
        args.semantic_bacc_proxy_ridge,
        args.semantic_bacc_proxy_dim,
    )


def ridge_residual(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor, ridge: float) -> torch.Tensor:
    coef = ridge_coefficients(train_x, train_y, ridge)
    design = torch.cat([eval_x.float(), torch.ones(eval_x.size(0), 1)], dim=1)
    return eval_y.float() - design @ coef


def crossfit_residuals(
    semantic: torch.Tensor,
    target: torch.Tensor,
    meta: torch.Tensor,
    ridge: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = semantic.size(0)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = max(2, n // 2)
    folds = (order[:cut], order[cut:])
    target_result = torch.empty_like(target, dtype=torch.float32)
    meta_result = torch.empty_like(meta, dtype=torch.float32)
    assigned = torch.zeros(n, dtype=torch.bool)
    for fit, evaluate in (folds, folds[::-1]):
        if fit.numel() < 2 or evaluate.numel() < 2:
            continue
        target_result[evaluate] = ridge_residual(
            semantic.index_select(0, fit),
            target.index_select(0, fit),
            semantic.index_select(0, evaluate),
            target.index_select(0, evaluate),
            ridge,
        )
        meta_result[evaluate] = ridge_residual(
            semantic.index_select(0, fit),
            meta.index_select(0, fit),
            semantic.index_select(0, evaluate),
            meta.index_select(0, evaluate),
            ridge,
        )
        assigned[evaluate] = True
    if not assigned.all():
        raise ValueError("Cross-fit direction refresh needs at least four rows.")
    return target_result, meta_result


def sparse_directions(
    cross_covariance: torch.Tensor,
    num_modules: int,
    support: int,
    max_memberships: int,
    semantic_covariance: Optional[torch.Tensor] = None,
    semantic_penalty: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    transformed_cross = cross_covariance.float()
    cholesky = None
    if semantic_covariance is not None and semantic_penalty > 0.0:
        identity = torch.eye(semantic_covariance.size(0), dtype=torch.float32)
        denominator = identity + float(semantic_penalty) * semantic_covariance.float()
        cholesky = torch.linalg.cholesky(denominator + 1e-4 * identity)
        transformed_cross = torch.linalg.solve_triangular(
            cholesky, transformed_cross, upper=False
        )
    u, singular, _ = torch.linalg.svd(transformed_cross, full_matrices=False)
    if cholesky is not None:
        u = torch.linalg.solve_triangular(cholesky.t(), u, upper=True)
    available = min(num_modules, u.size(1))
    directions = torch.zeros(num_modules, u.size(0))
    memberships = torch.zeros(u.size(0), dtype=torch.long)
    for module in range(available):
        raw = u[:, module]
        ranking = torch.argsort(raw.abs(), descending=True)
        chosen: List[int] = []
        for index in ranking.tolist():
            if memberships[index] >= max_memberships:
                continue
            chosen.append(index)
            if len(chosen) >= support:
                break
        if not chosen:
            chosen = ranking[: min(support, ranking.numel())].tolist()
        directions[module, chosen] = raw[chosen]
        memberships[chosen] += 1
    directions = F.normalize(directions, dim=1)
    singular_out = torch.zeros(num_modules)
    singular_out[:available] = singular[:available]
    return directions, singular_out


def align_directions(
    previous: Optional[torch.Tensor], current: torch.Tensor
) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
    if previous is None:
        return current, list(range(current.size(0))), torch.ones(current.size(0))
    cosine = F.normalize(previous.float(), dim=1) @ F.normalize(current.float(), dim=1).t()
    remaining = set(range(current.size(0)))
    aligned = torch.zeros_like(current)
    permutation: List[int] = []
    similarities = []
    for row in range(previous.size(0)):
        choice = max(remaining, key=lambda col: abs(float(cosine[row, col].item())))
        sign = 1.0 if cosine[row, choice] >= 0 else -1.0
        aligned[row] = current[choice] * sign
        permutation.append(choice)
        similarities.append(abs(cosine[row, choice]))
        remaining.remove(choice)
    return aligned, permutation, torch.stack(similarities)


def refresh_module_directions(
    target_candidates: torch.Tensor,
    semantic: torch.Tensor,
    meta: torch.Tensor,
    previous: Optional[torch.Tensor],
    args: argparse.Namespace,
    seed: int,
    semantic_target_prediction: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if args.module_direction_mode == "random_support":
        # This is a support-selection null, not a random hidden-space dose.
        # The same random supports and loadings are kept fixed throughout the
        # run so training cannot repeatedly redraw a favorable null module.
        if previous is not None:
            return previous.contiguous(), {
                "mode": "random_support",
                "fixed": True,
                "direction_cosine_mean": 1.0,
                "direction_cosine_min": 1.0,
                "support_jaccard_mean": 1.0,
                "crossfit_rows": int(target_candidates.size(0)),
                "support": int(args.module_support),
                "random_support_seed": int(
                    args.seed if args.random_support_seed < 0 else args.random_support_seed
                ),
            }
        random_seed = args.seed if args.random_support_seed < 0 else args.random_support_seed
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        directions = torch.zeros(
            args.num_modules, target_candidates.size(1), dtype=torch.float32
        )
        support = min(args.module_support, target_candidates.size(1))
        support_rows = []
        for module in range(args.num_modules):
            chosen = torch.randperm(
                target_candidates.size(1), generator=generator
            )[:support]
            weights = torch.randn(support, generator=generator)
            directions[module, chosen] = F.normalize(weights.view(1, -1), dim=1).view(-1)
            support_rows.append(chosen.tolist())
        return directions.contiguous(), {
            "mode": "random_support",
            "fixed": True,
            "support": int(support),
            "random_support_seed": int(random_seed),
            "candidate_pool_dim": int(target_candidates.size(1)),
            "support_rows": support_rows,
            "direction_cosine_mean": 1.0,
            "direction_cosine_min": 1.0,
            "support_jaccard_mean": 1.0,
            "crossfit_rows": int(target_candidates.size(0)),
        }
    if args.direction_max_samples > 0 and target_candidates.size(0) > args.direction_max_samples:
        index = torch.randperm(
            target_candidates.size(0), generator=torch.Generator().manual_seed(seed + 1)
        )[: args.direction_max_samples]
        target_candidates = target_candidates.index_select(0, index)
        semantic = semantic.index_select(0, index)
        meta = meta.index_select(0, index)
        if semantic_target_prediction is not None:
            semantic_target_prediction = semantic_target_prediction.index_select(0, index)
    crossfit_target, residual_meta = crossfit_residuals(
        semantic.cpu(), target_candidates.cpu(), meta.cpu(), args.direction_ridge, seed
    )
    if semantic_target_prediction is None:
        residual_target = crossfit_target
        semantic_target = target_candidates.cpu() - residual_target
    else:
        semantic_target = semantic_target_prediction.float().cpu()
        residual_target = target_candidates.float().cpu() - semantic_target
        residual_target = residual_target - residual_target.mean(dim=0, keepdim=True)
    residual_target = residual_target / residual_target.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-5)
    residual_meta = residual_meta / residual_meta.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-5)
    cross = residual_target.t() @ residual_meta / max(residual_target.size(0), 1)
    centered_semantic = semantic_target - semantic_target.mean(dim=0, keepdim=True)
    semantic_covariance = centered_semantic.t() @ centered_semantic / max(
        centered_semantic.size(0), 1
    )
    proposed, singular = sparse_directions(
        cross,
        args.num_modules,
        min(args.module_support, cross.size(0)),
        args.module_max_memberships,
        semantic_covariance=semantic_covariance,
        semantic_penalty=args.direction_semantic_penalty,
    )
    direction_norms = proposed.norm(dim=1)
    if (direction_norms < 1e-6).any():
        bad = (direction_norms < 1e-6).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "Partial-CCA returned empty module directions "
            f"{bad}; reduce --num-modules or enlarge the candidate/meta dimensions."
        )
    proposed, permutation, cosine = align_directions(previous, proposed)
    if previous is not None:
        mixed = float(args.direction_ema) * previous.float() + (1.0 - float(args.direction_ema)) * proposed
        # Re-sparsify after EMA so the exported support remains explicit.
        sparse = torch.zeros_like(mixed)
        for row in range(mixed.size(0)):
            keep = torch.topk(mixed[row].abs(), k=min(args.module_support, mixed.size(1))).indices
            sparse[row, keep] = mixed[row, keep]
        proposed = F.normalize(sparse, dim=1)
    support_jaccard = []
    post_update_cosines = []
    if previous is not None:
        for old, new in zip(previous, proposed):
            a = old.abs() > 0
            b = new.abs() > 0
            support_jaccard.append(float((a & b).sum().item() / max((a | b).sum().item(), 1)))
            post_update_cosines.append(
                float(F.cosine_similarity(old.view(1, -1), new.view(1, -1)).abs().item())
            )
    return proposed.contiguous(), {
        "singular_values": singular.tolist(),
        "alignment_permutation": permutation,
        "proposed_direction_cosine_mean": float(cosine.mean().item()),
        "proposed_direction_cosine_min": float(cosine.min().item()),
        "direction_cosine_mean": float(sum(post_update_cosines) / len(post_update_cosines)) if post_update_cosines else 1.0,
        "direction_cosine_min": float(min(post_update_cosines)) if post_update_cosines else 1.0,
        "support_jaccard_mean": float(sum(support_jaccard) / len(support_jaccard)) if support_jaccard else 1.0,
        "crossfit_rows": int(residual_target.size(0)),
        "support": int(args.module_support),
        "semantic_penalty": float(args.direction_semantic_penalty),
        "fixed_semantic_target_prediction": semantic_target_prediction is not None,
    }


@dataclass
class ModuleMetrics:
    score: float
    total_gain: float
    semantic_gain: float
    residual_gain: float
    residual_fraction: float
    positive_modules: int
    rows: List[Dict[str, Any]]


def module_information_metrics(
    target: torch.Tensor,
    semantic_prediction: torch.Tensor,
    residual_prediction: torch.Tensor,
    train_mean: torch.Tensor,
    score_weights: Tuple[float, float, float, float] = (1.0, 0.02, 0.20, 0.002),
) -> ModuleMetrics:
    prior_error = (target - train_mean.view(1, -1)).square()
    semantic_error = (target - semantic_prediction).square()
    combined_error = (target - semantic_prediction - residual_prediction).square()
    semantic_gain_by = (prior_error - semantic_error).mean(dim=0)
    residual_gain_by = (semantic_error - combined_error).mean(dim=0)
    total_gain_by = (prior_error - combined_error).mean(dim=0)
    positive_semantic = semantic_gain_by.clamp_min(0.0).sum()
    positive_residual = residual_gain_by.clamp_min(0.0).sum()
    positive_mass = positive_semantic + positive_residual
    residual_fraction = float((positive_residual / positive_mass.clamp_min(1e-8)).item())
    rows = []
    for module in range(target.size(1)):
        rows.append(
            {
                "module": module,
                "prior_mse": float(prior_error[:, module].mean().item()),
                "semantic_mse": float(semantic_error[:, module].mean().item()),
                "combined_mse": float(combined_error[:, module].mean().item()),
                "semantic_gain_mse": float(semantic_gain_by[module].item()),
                "residual_gain_mse": float(residual_gain_by[module].item()),
                "total_gain_mse": float(total_gain_by[module].item()),
                "residual_fraction_positive": float(
                    residual_gain_by[module].clamp_min(0.0)
                    / (
                        residual_gain_by[module].clamp_min(0.0)
                        + semantic_gain_by[module].clamp_min(0.0)
                    ).clamp_min(1e-8)
                ),
            }
        )
    total_gain = float(total_gain_by.mean().item())
    semantic_gain = float(semantic_gain_by.mean().item())
    residual_gain = float(residual_gain_by.mean().item())
    positive_modules = int((residual_gain_by > 0.0).sum().item())
    # Checkpoint selection should reward information that is unavailable to
    # the frozen semantic control.  A large total gain alone is deliberately
    # weak evidence because it can be dominated by the semantic branch.
    residual_weight, total_weight, fraction_weight, positive_weight = score_weights
    score = (
        residual_weight * residual_gain
        + total_weight * total_gain
        + fraction_weight * residual_fraction
        + positive_weight * positive_modules
    )
    return ModuleMetrics(
        score=score,
        total_gain=total_gain,
        semantic_gain=semantic_gain,
        residual_gain=residual_gain,
        residual_fraction=residual_fraction,
        positive_modules=positive_modules,
        rows=rows,
    )


def semantic_source_features(
    z1: torch.Tensor,
    prev: Optional[torch.Tensor],
    source: str,
) -> torch.Tensor:
    if source == "z1":
        return z1
    if prev is None:
        raise ValueError(f"{source} semantic teacher requires previous-layer features")
    if source == "prev":
        return prev
    if source == "z1_prev":
        return torch.cat([z1, prev], dim=1)
    raise ValueError(f"Unsupported semantic candidate source: {source}")


def train_frozen_semantic_candidate_teacher(
    teacher: nn.Module,
    semantic_features: torch.Tensor,
    candidate_values: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    source_label: str,
) -> Dict[str, Any]:
    """Fit one semantic teacher and reuse its exact predictions at every RF stage."""
    loader = DataLoader(
        TensorDataset(
            semantic_features.index_select(0, train_idx),
            candidate_values.index_select(0, train_idx),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 151),
    )
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=args.semantic_probe_lr,
        weight_decay=args.semantic_probe_weight_decay,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_mse = math.inf
    best_epoch = 0
    history = []
    progress = tqdm(
        range(1, args.semantic_probe_epochs + 1),
        desc=f"Aligned {source_label} semantic probe",
        unit="epoch",
    )
    for epoch in progress:
        teacher.train()
        total = 0.0
        seen = 0
        for prev_cpu, target_cpu in loader:
            prev = prev_cpu.to(device, non_blocking=device.type == "cuda")
            target = target_cpu.to(device, non_blocking=device.type == "cuda")
            prediction = teacher(prev)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().item()) * prev.size(0)
            seen += prev.size(0)
        teacher.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, val_idx.numel(), args.encode_batch_size):
                index = val_idx[start : start + args.encode_batch_size]
                predictions.append(
                    teacher(
                        semantic_features.index_select(0, index).to(device)
                    ).float().cpu()
                )
        val_prediction = torch.cat(predictions)
        val_target = candidate_values.index_select(0, val_idx)
        val_mse = float(F.mse_loss(val_prediction, val_target).item())
        row = {
            "epoch": epoch,
            "train_mse": total / max(seen, 1),
            "val_mse": val_mse,
        }
        history.append(row)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = cpu_state_dict(teacher)
        progress.set_postfix(
            train=f"{row['train_mse']:.4f}",
            val=f"{val_mse:.4f}",
            best=f"{best_val_mse:.4f}",
        )
    if best_state is None:
        raise RuntimeError("Aligned semantic probe did not produce a checkpoint.")
    teacher.load_state_dict(best_state, strict=True)
    teacher.eval()
    set_trainable(teacher, False)
    write_csv(
        os.path.join(args.output_dir, "aligned_semantic_probe_history.csv"),
        history,
    )
    return {
        "mode": args.semantic_probe_mode,
        "source": source_label,
        "target": "candidate_neuron_continuous_activation",
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_mse,
        "hidden_dim": args.semantic_probe_hidden_dim,
        "epochs": args.semantic_probe_epochs,
        "lr": args.semantic_probe_lr,
        "weight_decay": args.semantic_probe_weight_decay,
        "dropout": args.dropout,
        "reused_for_training_and_heldout_rf": True,
    }


def train_frozen_external_semantic_teacher(
    teacher: nn.Module,
    semantic_features: torch.Tensor,
    external_target: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Fit Z1 -> external F and freeze the exact teacher used downstream."""
    loader = DataLoader(
        TensorDataset(
            semantic_features.index_select(0, train_idx),
            external_target.index_select(0, train_idx),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 153),
    )
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=args.semantic_probe_lr,
        weight_decay=args.semantic_probe_weight_decay,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_mse = math.inf
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    progress = tqdm(
        range(1, args.semantic_probe_epochs + 1),
        desc="External semantic F probe",
        unit="epoch",
    )
    for epoch in progress:
        teacher.train()
        total = 0.0
        seen = 0
        for feature_cpu, target_cpu in loader:
            feature = feature_cpu.to(device, non_blocking=device.type == "cuda")
            target = target_cpu.to(device, non_blocking=device.type == "cuda")
            prediction = teacher(feature)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().item()) * feature.size(0)
            seen += feature.size(0)
        teacher.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, val_idx.numel(), args.encode_batch_size):
                index = val_idx[start : start + args.encode_batch_size]
                predictions.append(
                    teacher(semantic_features.index_select(0, index).to(device))
                    .float()
                    .cpu()
                )
        val_prediction = torch.cat(predictions)
        val_target = external_target.index_select(0, val_idx)
        val_mse = float(F.mse_loss(val_prediction, val_target).item())
        row = {
            "epoch": epoch,
            "train_mse": total / max(seen, 1),
            "val_mse": val_mse,
        }
        history.append(row)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = cpu_state_dict(teacher)
        progress.set_postfix(
            train=f"{row['train_mse']:.4f}",
            val=f"{val_mse:.4f}",
            best=f"{best_val_mse:.4f}",
        )
    if best_state is None:
        raise RuntimeError("External semantic probe produced no checkpoint.")
    teacher.load_state_dict(best_state, strict=True)
    teacher.eval()
    set_trainable(teacher, False)
    per_sample_loss = torch.empty(semantic_features.size(0), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, semantic_features.size(0), args.encode_batch_size):
            stop = min(start + args.encode_batch_size, semantic_features.size(0))
            prediction = teacher(semantic_features[start:stop].to(device)).float().cpu()
            per_sample_loss[start:stop] = (
                prediction - external_target[start:stop]
            ).square().mean(dim=1)
    split_metrics: Dict[str, Dict[str, Any]] = {}
    loss_rows = []
    for split_name, split_index in (
        ("train", train_idx),
        ("selection", val_idx),
        ("heldout", test_idx if test_idx.numel() else val_idx),
    ):
        losses = per_sample_loss.index_select(0, split_index)
        split_metrics[split_name] = {
            "n": int(losses.numel()),
            "mse": float(losses.mean().item()),
        }
        loss_rows.extend(
            {
                "row_index": int(row_index),
                "split": split_name,
                "external_semantic_mse": float(loss),
            }
            for row_index, loss in zip(split_index.tolist(), losses.tolist())
        )
    write_csv(
        os.path.join(args.output_dir, "external_semantic_probe_losses.csv"),
        loss_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "external_semantic_probe_history.csv"),
        history,
    )
    return {
        "mode": "external_activation",
        "source": "frozen_z1_from_hk",
        "target": "external_model_activation",
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_mse,
        "split_metrics": split_metrics,
        "external_dim": int(external_target.size(1)),
        "hidden_dim": args.semantic_probe_hidden_dim,
        "epochs": args.semantic_probe_epochs,
        "lr": args.semantic_probe_lr,
        "weight_decay": args.semantic_probe_weight_decay,
        "head_type": args.external_semantic_head_type,
        "head_depth": args.external_semantic_head_depth,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in teacher.parameters())
        ),
        "ensemble_weights": (
            teacher.mixture_weights()
            if isinstance(teacher, EnsembleExternalSemanticHead)
            else None
        ),
        "reused_for_training_and_heldout_rf": True,
    }


def encode_main_latents(
    model: base.Decoupler,
    next_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    values: Dict[str, List[torch.Tensor]] = {"z1": [], "z2": []}
    model.eval()
    with torch.inference_mode():
        for start in range(0, next_features.size(0), batch_size):
            out = model(next_features[start : start + batch_size].to(device))
            values["z1"].append(out["z1"].float().cpu())
            values["z2"].append(out["z2"].float().cpu())
    return {key: torch.cat(parts) for key, parts in values.items()}


def pretrain_staged_z1_encoder(
    model: base.Decoupler,
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Stabilize Hk->Z1/Z2 before defining either target-prediction probe."""
    dataset = TensorDataset(
        x_prev.index_select(0, train_idx),
        x_next.index_select(0, train_idx),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 131),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = math.inf
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    progress = tqdm(
        range(1, args.semantic_encoder_warmup_epochs + 1),
        desc="Staged Z1/Z2 reconstruction warmup",
        unit="epoch",
    )
    for epoch in progress:
        model.train()
        totals = {"loss": 0.0, "next": 0.0, "prev": 0.0, "orth": 0.0, "cov": 0.0}
        seen = 0
        for prev_cpu, next_cpu in loader:
            prev = prev_cpu.to(device, non_blocking=device.type == "cuda")
            next_value = next_cpu.to(
                device, non_blocking=device.type == "cuda"
            )
            out = model(next_value)
            losses = {
                "next": F.mse_loss(out["x_next_hat"], next_value),
                "prev": F.mse_loss(out["x_prev_hat"], prev),
                "orth": base.orthogonality_loss(out["z1"], out["z2"]),
                "cov": centered_covariance_penalty(out["z2"], prev),
            }
            loss = (
                args.lambda_next * losses["next"]
                + args.lambda_prev * losses["prev"]
                + args.lambda_orth * losses["orth"]
                + args.lambda_e2_prev_cov * losses["cov"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_n = prev.size(0)
            totals["loss"] += float(loss.detach().item()) * batch_n
            for key, value in losses.items():
                totals[key] += float(value.detach().item()) * batch_n
            seen += batch_n

        model.eval()
        validation = {"next": 0.0, "prev": 0.0, "orth": 0.0, "cov": 0.0}
        val_seen = 0
        with torch.inference_mode():
            for start in range(0, val_idx.numel(), args.encode_batch_size):
                index = val_idx[start : start + args.encode_batch_size]
                prev = x_prev.index_select(0, index).to(device)
                next_value = x_next.index_select(0, index).to(device)
                out = model(next_value)
                values = {
                    "next": F.mse_loss(out["x_next_hat"], next_value),
                    "prev": F.mse_loss(out["x_prev_hat"], prev),
                    "orth": base.orthogonality_loss(out["z1"], out["z2"]),
                    "cov": centered_covariance_penalty(out["z2"], prev),
                }
                batch_n = index.numel()
                for key, value in values.items():
                    validation[key] += float(value.item()) * batch_n
                val_seen += batch_n
        validation = {
            key: value / max(val_seen, 1)
            for key, value in validation.items()
        }
        val_score = (
            args.lambda_next * validation["next"]
            + args.lambda_prev * validation["prev"]
            + args.lambda_orth * validation["orth"]
            + args.lambda_e2_prev_cov * validation["cov"]
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{key}": value / max(seen, 1)
                for key, value in totals.items()
            },
            **{f"val_{key}": value for key, value in validation.items()},
            "val_score": val_score,
        }
        history.append(row)
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = cpu_state_dict(model)
        progress.set_postfix(
            best=f"{best_score:.4f}",
            next=f"{validation['next']:.4f}",
            prev=f"{validation['prev']:.4f}",
        )
    if best_state is None:
        raise RuntimeError("Staged Z1 encoder warmup produced no checkpoint.")
    model.load_state_dict(best_state, strict=True)
    write_csv(
        os.path.join(args.output_dir, "semantic_encoder_warmup_history.csv"),
        history,
    )
    torch.save(
        model.state_dict(),
        os.path.join(args.output_dir, "semantic_encoder_warmup_model.pt"),
    )
    return {
        "epochs": args.semantic_encoder_warmup_epochs,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "objective": "Hk reconstruction + Z1-to-Hi reconstruction",
    }


def fit_matched_latent_target_probe(
    name: str,
    features: torch.Tensor,
    targets: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fit one capacity-matched latent->target probe on frozen representations."""
    torch.manual_seed(seed)
    probe = base.MLP(
        features.size(1),
        targets.size(1),
        args.semantic_probe_hidden_dim,
        args.dropout,
    ).to(device)
    loader = DataLoader(
        TensorDataset(
            features.index_select(0, train_idx),
            targets.index_select(0, train_idx),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed + 1),
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.semantic_probe_lr,
        weight_decay=args.semantic_probe_weight_decay,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_mse = math.inf
    best_epoch = 0
    progress = tqdm(
        range(1, args.semantic_probe_epochs + 1),
        desc=f"Matched {name}->Tj probe",
        unit="epoch",
    )
    for epoch in progress:
        probe.train()
        for feature_cpu, target_cpu in loader:
            feature = feature_cpu.to(
                device, non_blocking=device.type == "cuda"
            )
            target = target_cpu.to(
                device, non_blocking=device.type == "cuda"
            )
            loss = F.mse_loss(probe(feature), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
            optimizer.step()
        probe.eval()
        val_parts = []
        with torch.inference_mode():
            for start in range(0, val_idx.numel(), args.encode_batch_size):
                index = val_idx[start : start + args.encode_batch_size]
                val_parts.append(
                    probe(features.index_select(0, index).to(device))
                    .float()
                    .cpu()
                )
        val_prediction = torch.cat(val_parts)
        val_mse = float(
            F.mse_loss(
                val_prediction, targets.index_select(0, val_idx)
            ).item()
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = cpu_state_dict(probe)
        progress.set_postfix(best=f"{best_val_mse:.4f}", val=f"{val_mse:.4f}")
    if best_state is None:
        raise RuntimeError(f"Matched {name} probe produced no checkpoint.")
    probe.load_state_dict(best_state, strict=True)
    probe.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, features.size(0), args.encode_batch_size):
            predictions.append(
                probe(features[start : start + args.encode_batch_size].to(device))
                .float()
                .cpu()
            )
    prediction = torch.cat(predictions)
    train_mean = targets.index_select(0, train_idx).mean(
        dim=0, keepdim=True
    )
    split_metrics = {}
    for split_name, split_index in (
        ("train", train_idx),
        ("selection", val_idx),
        ("heldout", test_idx if test_idx.numel() else val_idx),
    ):
        target = targets.index_select(0, split_index)
        predicted = prediction.index_select(0, split_index)
        mse = float(F.mse_loss(predicted, target).item())
        prior_mse = float(
            F.mse_loss(train_mean.expand_as(target), target).item()
        )
        split_metrics[split_name] = {
            "n": int(split_index.numel()),
            "mse": mse,
            "prior_mse": prior_mse,
            "gain_mse": prior_mse - mse,
            "r2_vs_train_mean": 1.0 - mse / max(prior_mse, 1e-12),
        }
    probe.cpu()
    return prediction, {
        "name": name,
        "input_dim": int(features.size(1)),
        "output_targets": int(targets.size(1)),
        "hidden_dim": args.semantic_probe_hidden_dim,
        "epochs": args.semantic_probe_epochs,
        "lr": args.semantic_probe_lr,
        "weight_decay": args.semantic_probe_weight_decay,
        "dropout": args.dropout,
        "initialization_seed": seed,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_mse,
        "splits": split_metrics,
    }


def matched_z1_z2_target_probe_audit(
    encoded: Dict[str, torch.Tensor],
    candidate_values: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Compare candidate-level Tj decodability from fixed Z1/Z2 at matched capacity."""
    # Do not use discovered module targets here: those directions were selected
    # partly for Z2 predictability and would bias this direct comparison toward
    # Z2. Candidate neurons were selected only by activation statistics and
    # cross-split stability.
    targets = candidate_values
    shared_seed = args.seed + 9101
    _, z1_summary = fit_matched_latent_target_probe(
        "Z1",
        encoded["z1"],
        targets,
        train_idx,
        val_idx,
        test_idx,
        args,
        device,
        shared_seed,
    )
    _, z2_summary = fit_matched_latent_target_probe(
        "Z2",
        encoded["z2"],
        targets,
        train_idx,
        val_idx,
        test_idx,
        args,
        device,
        shared_seed,
    )
    heldout_z1 = z1_summary["splits"]["heldout"]
    heldout_z2 = z2_summary["splits"]["heldout"]
    summary = {
        "purpose": (
            "Capacity-matched direct prediction of the full statistically "
            "selected layer-j candidate-neuron pool from the two layer-k "
            "latent branches."
        ),
        "target_level": "candidate_neurons_before_module_discovery",
        "candidate_target_count": int(candidate_values.size(1)),
        "participates_in_module_selection": False,
        "same_architecture_hyperparameters_and_initialization": True,
        "z1": z1_summary,
        "z2": z2_summary,
        "heldout_comparison": {
            "z2_minus_z1_gain_mse": (
                heldout_z2["gain_mse"] - heldout_z1["gain_mse"]
            ),
            "z2_minus_z1_r2": (
                heldout_z2["r2_vs_train_mean"]
                - heldout_z1["r2_vs_train_mean"]
            ),
        },
        "conditional_incremental_note": (
            "Direct Z2 prediction can be redundant with Z1. The primary module "
            "RF remains the incremental reduction after the frozen Z1 probe."
        ),
    }
    write_json(
        os.path.join(args.output_dir, "matched_z1_z2_target_probe_audit.json"),
        summary,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def predict_main(
    model: base.Decoupler,
    semantic_candidate_decoder: nn.Module,
    residual_head: nn.Module,
    next_features: torch.Tensor,
    prev_features: Optional[torch.Tensor],
    semantic_candidate_source: str,
    device: torch.device,
    batch_size: int,
    directions: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    values: Dict[str, List[torch.Tensor]] = {
        "z1": [],
        "z2": [],
        "semantic_candidates": [],
        "semantic_external": [],
        "semantic": [],
        "residual": [],
    }
    has_external_semantic = hasattr(semantic_candidate_decoder, "predict_external")
    model.eval()
    semantic_candidate_decoder.eval()
    residual_head.eval()
    direction_device = directions.to(device) if directions is not None else None
    with torch.inference_mode():
        for start in range(0, next_features.size(0), batch_size):
            batch = next_features[start : start + batch_size].to(device)
            out = model(batch)
            previous = (
                None
                if prev_features is None
                else prev_features[start : start + batch_size].to(device)
            )
            semantic_input = semantic_source_features(
                out["z1"], previous, semantic_candidate_source
            )
            semantic_candidates = semantic_candidate_decoder(semantic_input)
            if has_external_semantic:
                semantic_external = semantic_candidate_decoder.predict_external(
                    semantic_input
                )
            semantic_modules = (
                semantic_candidates @ direction_device.t()
                if direction_device is not None
                else semantic_candidates
            )
            values["z1"].append(out["z1"].float().cpu())
            values["z2"].append(out["z2"].float().cpu())
            values["semantic_candidates"].append(semantic_candidates.float().cpu())
            if has_external_semantic:
                values["semantic_external"].append(semantic_external.float().cpu())
            values["semantic"].append(semantic_modules.float().cpu())
            values["residual"].append(residual_head(out["z2"]).float().cpu())
    result = {
        key: torch.cat(parts)
        for key, parts in values.items()
        if parts
    }
    return result


def train_main_stage(
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    candidate_values: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    external_semantic_values: Optional[torch.Tensor] = None,
) -> Tuple[base.Decoupler, nn.Module, nn.Module, torch.Tensor, Dict[str, Any]]:
    dim = x_next.size(1)
    model = base.Decoupler(dim, args.latent_dim, args.hidden_dim, args.dropout, "continuous_binary").to(device)
    semantic_input_dim = (
        dim
        if args.semantic_candidate_source == "prev"
        else args.latent_dim
        + (dim if args.semantic_candidate_source == "z1_prev" else 0)
    )
    semantic_hidden_dim = (
        args.semantic_probe_hidden_dim
        if args.semantic_probe_mode
        in {"frozen_aligned", "staged_z1_aligned"}
        else max(256, args.hidden_dim // 2)
    )
    external_mode = bool(args.external_semantic_activation_dir)
    external_map_summary: Optional[Dict[str, Any]] = None
    if external_mode:
        if external_semantic_values is None:
            raise ValueError("External semantic mode requires loaded external features.")
        if args.semantic_candidate_source != "z1":
            raise ValueError(
                "External semantic mode currently requires --semantic-candidate-source z1."
            )
        if args.semantic_probe_mode != "staged_z1_aligned":
            raise ValueError(
                "External semantic mode currently requires --semantic-probe-mode staged_z1_aligned."
            )
        external_map, external_map_summary = fit_external_to_candidate_map(
            external_semantic_values,
            candidate_values,
            train_idx,
            args.external_semantic_ridge,
        )
        semantic_candidate_decoder = ExternalSemanticCandidateDecoder(
            semantic_input_dim,
            external_semantic_values.size(1),
            candidate_values.size(1),
            semantic_hidden_dim,
            args.dropout,
            external_map,
            args.external_semantic_head_type,
            args.external_semantic_head_depth,
        ).to(device)
    else:
        semantic_candidate_decoder = base.MLP(
            semantic_input_dim,
            candidate_values.size(1),
            semantic_hidden_dim,
            args.dropout,
        ).to(device)
    residual_head = base.MLP(
        args.latent_dim, args.num_modules, max(128, args.hidden_dim // 2), args.dropout
    ).to(device)
    aligned_semantic_summary: Dict[str, Any] = {
        "mode": "joint",
        "source": args.semantic_candidate_source,
    }
    encoder_warmup_summary: Optional[Dict[str, Any]] = None
    if args.semantic_probe_mode == "frozen_aligned":
        if args.semantic_candidate_source != "prev":
            raise ValueError(
                "--semantic-probe-mode frozen_aligned requires "
                "--semantic-candidate-source prev."
            )
        aligned_semantic_summary = train_frozen_semantic_candidate_teacher(
            semantic_candidate_decoder,
            x_prev,
            candidate_values,
            train_idx,
            val_idx,
            args,
            device,
            "previous_layer_only",
        )
    elif args.semantic_probe_mode == "staged_z1_aligned":
        if args.semantic_candidate_source != "z1":
            raise ValueError(
                "--semantic-probe-mode staged_z1_aligned requires "
                "--semantic-candidate-source z1."
            )
        encoder_warmup_summary = pretrain_staged_z1_encoder(
            model,
            x_prev,
            x_next,
            train_idx,
            val_idx,
            args,
            device,
        )
        fixed_latents = encode_main_latents(
            model, x_next, device, args.encode_batch_size
        )
        if external_mode:
            aligned_semantic_summary = train_frozen_external_semantic_teacher(
                semantic_candidate_decoder.external_head,
                fixed_latents["z1"],
                external_semantic_values,
                train_idx,
                val_idx,
                test_idx,
                args,
                device,
            )
            aligned_semantic_summary["external_to_candidate_map"] = external_map_summary
        else:
            aligned_semantic_summary = train_frozen_semantic_candidate_teacher(
                semantic_candidate_decoder,
                fixed_latents["z1"],
                candidate_values,
                train_idx,
                val_idx,
                args,
                device,
                "frozen_z1_from_hk",
            )
        aligned_semantic_summary["encoder_warmup"] = (
            encoder_warmup_summary
        )
        set_trainable(model.e1, False)
        set_trainable(model.d_prev, False)
        if args.semantic_bacc_absorption_unfreeze_semantic and not external_mode:
            set_trainable(semantic_candidate_decoder, True)
    optimizer_parameters = list(model.parameters()) + list(
        residual_head.parameters()
    )
    if (
        args.semantic_probe_mode not in {"frozen_aligned", "staged_z1_aligned"}
        or (args.semantic_bacc_absorption_unfreeze_semantic and not external_mode)
    ):
        optimizer_parameters += list(semantic_candidate_decoder.parameters())
    optimizer_parameters = [
        parameter
        for parameter in optimizer_parameters
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial = predict_main(
        model,
        semantic_candidate_decoder,
        residual_head,
        x_next.index_select(0, train_idx),
        x_prev.index_select(0, train_idx),
        args.semantic_candidate_source,
        device,
        args.encode_batch_size,
    )
    directions, initial_refresh = refresh_module_directions(
        candidate_values.index_select(0, train_idx),
        semantic_source_features(
            initial["z1"],
            x_prev.index_select(0, train_idx),
            args.semantic_candidate_source,
        ),
        initial["z2"],
        None,
        args,
        args.seed + 1001,
        semantic_target_prediction=(
            initial["semantic_candidates"]
            if args.semantic_probe_mode
            in {"frozen_aligned", "staged_z1_aligned"}
            else None
        ),
    )
    direction_history: List[Dict[str, Any]] = [{"epoch": 0, "stage": "initial", **initial_refresh}]
    fixed_semantic_refresh_started = False
    history: List[Dict[str, Any]] = []
    dataset = TensorDataset(
        x_prev.index_select(0, train_idx),
        x_next.index_select(0, train_idx),
        candidate_values.index_select(0, train_idx),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 201),
    )
    semantic_signature_dim = (
        external_semantic_values.size(1)
        if external_mode and external_semantic_values is not None
        else (
            dim
            if args.semantic_candidate_source == "prev"
            else args.latent_dim
            + (dim if args.semantic_candidate_source == "z1_prev" else 0)
        )
    )
    main_semantic_projection = fixed_random_projection(
        semantic_signature_dim,
        args.semantic_cluster_mmd_projection_dim,
        args.seed + 1701,
        device,
    )
    best_score = -math.inf
    best_bundle: Optional[Dict[str, Any]] = None
    semantic_freeze_epoch = (
        args.direction_freeze_epoch
        if args.semantic_freeze_epoch < 0
        else args.semantic_freeze_epoch
    )
    invariance_scale = (
        1.0
        if args.semantic_invariance_rf_floor <= 0.0
        and args.semantic_invariance_gain_floor <= 0.0
        else 0.0
    )
    progress = tqdm(range(1, args.epochs + 1), desc="Joint-v2 main", unit="epoch")
    direction_dir = os.path.join(args.output_dir, "module_directions", "main")
    os.makedirs(direction_dir, exist_ok=True)
    torch.save(directions, os.path.join(direction_dir, "directions_epoch_000.pt"))

    for epoch in progress:
        should_refresh = (
            args.direction_refresh_start <= epoch < args.direction_freeze_epoch
            and (epoch - args.direction_refresh_start) % args.direction_refresh_every == 0
        )
        if should_refresh:
            encoded = predict_main(
                model,
                semantic_candidate_decoder,
                residual_head,
                x_next.index_select(0, train_idx),
                x_prev.index_select(0, train_idx),
                args.semantic_candidate_source,
                device,
                args.encode_batch_size,
                directions,
            )
            # The epoch-0 directions were only a bootstrap while the semantic
            # decoder was untrained.  Replace them completely at the first
            # fixed-semantic refresh, then use EMA for later refreshes.
            previous_directions = directions if fixed_semantic_refresh_started else None
            directions, refresh = refresh_module_directions(
                candidate_values.index_select(0, train_idx),
                semantic_source_features(
                    encoded["z1"],
                    x_prev.index_select(0, train_idx),
                    args.semantic_candidate_source,
                ),
                encoded["z2"],
                previous_directions,
                args,
                args.seed + 1001 + epoch,
                semantic_target_prediction=encoded["semantic_candidates"],
            )
            fixed_semantic_refresh_started = True
            direction_history.append({"epoch": epoch, "stage": "main", **refresh})
            torch.save(directions, os.path.join(direction_dir, f"directions_epoch_{epoch:03d}.pt"))

        model.train()
        semantic_absorption_active = (
            args.lambda_semantic_bacc_absorption > 0.0
            and epoch >= args.semantic_bacc_absorption_start_epoch
        )
        semantic_frozen = (
            args.semantic_probe_mode
            in {"frozen_aligned", "staged_z1_aligned"}
            or epoch >= semantic_freeze_epoch
        ) and not (
            semantic_absorption_active
            and args.semantic_bacc_absorption_unfreeze_semantic
        )
        if semantic_frozen:
            if args.semantic_probe_mode != "frozen_aligned":
                set_trainable(model.e1, False)
                set_trainable(model.d_prev, False)
            set_trainable(semantic_candidate_decoder, False)
            if args.semantic_probe_mode != "frozen_aligned":
                model.e1.eval()
                model.d_prev.eval()
            semantic_candidate_decoder.eval()
        elif args.semantic_probe_mode == "staged_z1_aligned":
            # Keep the decomposition input fixed while allowing the semantic
            # head itself to absorb soft module-state boundaries.
            set_trainable(model.e1, False)
            set_trainable(model.d_prev, False)
            model.e1.eval()
            model.d_prev.eval()
            set_trainable(semantic_candidate_decoder, True)
            semantic_candidate_decoder.train()
        else:
            set_trainable(model.e1, True)
            set_trainable(model.d_prev, True)
            set_trainable(semantic_candidate_decoder, True)
            semantic_candidate_decoder.train()
        residual_head.train()
        totals = {
            key: 0.0
            for key in (
                "loss",
                "next",
                "prev",
                "orth",
                "cov",
                "semantic_candidate",
                "semantic",
                "residual",
                "joint",
                "semantic_cluster_mmd",
                "semantic_bacc_absorption",
                "var",
            )
        }
        seen = 0
        direction_device = directions.to(device)
        reference_module_target = (
            candidate_values.index_select(0, train_idx) @ directions.t()
        ).float()
        reference_module_center = reference_module_target.median(dim=0).values
        reference_module_scale = reference_module_target.std(
            dim=0, unbiased=False
        ).clamp_min(1e-4)
        for prev_cpu, next_cpu, candidate_cpu in loader:
            prev = prev_cpu.to(device, non_blocking=True)
            next_value = next_cpu.to(device, non_blocking=True)
            candidates = candidate_cpu.to(device, non_blocking=True)
            module_target = candidates @ direction_device.t()
            out = model(next_value)
            # stop-gradient prevents the target decoder from teaching E1 to
            # absorb newly discovered residual directions.
            semantic_input = semantic_source_features(
                out["z1"].detach(),
                prev.detach(),
                args.semantic_candidate_source,
            )
            semantic_candidates = semantic_candidate_decoder(semantic_input)
            semantic_prediction = semantic_candidates @ direction_device.t()
            residual_target = module_target - semantic_prediction.detach()
            residual_prediction = residual_head(out["z2"])
            effective_invariance_scale = (
                invariance_scale
                if epoch >= args.semantic_invariance_start_epoch
                else 0.0
            )
            if (
                args.lambda_main_semantic_cluster_mmd > 0.0
                and effective_invariance_scale > 0.0
            ):
                semantic_signature = nonlinear_semantic_signature(
                    semantic_source_features(
                        out["z1"].detach(),
                        prev.detach(),
                        args.semantic_candidate_source,
                    ),
                    main_semantic_projection,
                )
                semantic_cluster_mmd = soft_cluster_semantic_mmd(
                    residual_prediction,
                    semantic_signature,
                    args.semantic_cluster_mmd_temperature,
                    args.semantic_cluster_mmd_thresholds,
                )
            else:
                semantic_cluster_mmd = residual_prediction.new_zeros(())
            if semantic_absorption_active:
                semantic_bacc_absorption = semantic_bacc_absorption_loss(
                    semantic_prediction,
                    module_target,
                    reference_module_center,
                    reference_module_scale,
                    args.semantic_bacc_absorption_temperature,
                )
            else:
                semantic_bacc_absorption = residual_prediction.new_zeros(())
            losses = {
                "next": F.mse_loss(out["x_next_hat"], next_value),
                "prev": F.mse_loss(out["x_prev_hat"], prev),
                "orth": base.orthogonality_loss(out["z1"], out["z2"]),
                "cov": centered_covariance_penalty(
                    out["z2"],
                    torch.cat([prev, semantic_candidates.detach()], dim=1),
                ),
                "semantic_candidate": (
                    residual_prediction.new_zeros(())
                    if external_mode
                    else F.mse_loss(semantic_candidates, candidates)
                ),
                "semantic": (
                    residual_prediction.new_zeros(())
                    if external_mode
                    else F.mse_loss(semantic_prediction, module_target)
                ),
                "residual": F.mse_loss(residual_prediction, residual_target),
                "joint": F.mse_loss(semantic_prediction + residual_prediction, module_target),
                "semantic_cluster_mmd": semantic_cluster_mmd,
                "semantic_bacc_absorption": semantic_bacc_absorption,
                "var": variance_floor(out["z1"], args.var_floor) + variance_floor(out["z2"], args.var_floor),
            }
            residual_scale = 0.0 if epoch < args.meta_start_epoch else min(
                1.0, (epoch - args.meta_start_epoch + 1) / 10.0
            )
            loss = (
                args.lambda_next * losses["next"]
                + args.lambda_prev * losses["prev"]
                + args.lambda_orth * losses["orth"]
                + args.lambda_e2_prev_cov * losses["cov"]
                + args.lambda_semantic_candidate * losses["semantic_candidate"]
                + args.lambda_semantic_module * losses["semantic"]
                + residual_scale
                * (
                    args.lambda_residual_module * losses["residual"]
                    + args.lambda_joint_module * losses["joint"]
                    + args.lambda_semantic_bacc_absorption
                    * losses["semantic_bacc_absorption"]
                    + args.lambda_main_semantic_cluster_mmd
                    * effective_invariance_scale
                    * losses["semantic_cluster_mmd"]
                )
                + args.lambda_var * losses["var"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(semantic_candidate_decoder.parameters())
                + list(residual_head.parameters()),
                5.0,
            )
            optimizer.step()
            batch_n = prev.size(0)
            totals["loss"] += float(loss.detach().item()) * batch_n
            for key, value in losses.items():
                totals[key] += float(value.detach().item()) * batch_n
            seen += batch_n

        row: Dict[str, Any] = {
            "epoch": epoch,
            "semantic_teacher_frozen": semantic_frozen,
            "semantic_invariance_scale": effective_invariance_scale,
            **{
                f"train_{key}": value / max(seen, 1)
                for key, value in totals.items()
            },
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            prediction = predict_main(
                model,
                semantic_candidate_decoder,
                residual_head,
                x_next.index_select(0, val_idx),
                x_prev.index_select(0, val_idx),
                args.semantic_candidate_source,
                device,
                args.encode_batch_size,
                directions,
            )
            target = candidate_values.index_select(0, val_idx) @ directions.t()
            train_mean = (
                candidate_values.index_select(0, train_idx) @ directions.t()
            ).mean(dim=0)
            metrics = module_information_metrics(
                target,
                prediction["semantic"],
                prediction["residual"],
                train_mean,
                (
                    args.checkpoint_residual_gain_weight,
                    args.checkpoint_total_gain_weight,
                    args.checkpoint_residual_fraction_weight,
                    args.checkpoint_positive_module_weight,
                ),
            )
            if (
                args.lambda_main_semantic_cluster_mmd > 0.0
                or args.checkpoint_semantic_cluster_mmd_weight > 0.0
            ):
                val_semantic_source = (
                    prediction["semantic_external"]
                    if args.external_semantic_activation_dir
                    and "semantic_external" in prediction
                    else semantic_source_features(
                        prediction["z1"],
                        x_prev.index_select(0, val_idx),
                        args.semantic_candidate_source,
                    )
                )
                val_semantic_signature = nonlinear_semantic_signature(
                    val_semantic_source,
                    main_semantic_projection.cpu(),
                )
                val_semantic_cluster_mmd = float(
                    soft_cluster_semantic_mmd(
                        prediction["residual"],
                        val_semantic_signature,
                        args.semantic_cluster_mmd_temperature,
                        args.semantic_cluster_mmd_thresholds,
                    ).item()
                )
            else:
                val_semantic_cluster_mmd = 0.0
            bacc_proxy_active = (
                args.checkpoint_semantic_bacc_proxy_weight > 0.0
                or args.checkpoint_max_semantic_bacc_proxy < 1.0
            )
            continuous_proxy_active = (
                args.checkpoint_continuous_semantic_r2_weight > 0.0
                or args.checkpoint_max_continuous_semantic_r2 < 1.0
            )
            upper_bound_active = args.meta_upper_bound_mode != "none"
            if bacc_proxy_active or continuous_proxy_active or upper_bound_active:
                train_prediction = predict_main(
                    model,
                    semantic_candidate_decoder,
                    residual_head,
                    x_next.index_select(0, train_idx),
                    x_prev.index_select(0, train_idx),
                    args.semantic_candidate_source,
                    device,
                    args.encode_batch_size,
                    directions,
                )
            else:
                train_prediction = None
            if bacc_proxy_active:
                assert train_prediction is not None
                bacc_proxy = split_semantic_bacc_proxy(
                    train_prediction["z1"],
                    prediction["z1"],
                    x_prev.index_select(0, train_idx),
                    x_prev.index_select(0, val_idx),
                    train_prediction["residual"],
                    prediction["residual"],
                    args,
                    args.seed + 1751,
                    (
                        train_prediction.get("semantic_external")
                        if args.external_semantic_activation_dir
                        else None
                    ),
                    (
                        prediction.get("semantic_external")
                        if args.external_semantic_activation_dir
                        else None
                    ),
                )
            else:
                bacc_proxy = {
                    "train_mean": 0.5,
                    "train_max": 0.5,
                    "eval_mean": 0.5,
                    "eval_max": 0.5,
                }
            if continuous_proxy_active:
                assert train_prediction is not None
                continuous_proxy = split_semantic_continuous_proxy(
                    train_prediction["z1"],
                    prediction["z1"],
                    x_prev.index_select(0, train_idx),
                    x_prev.index_select(0, val_idx),
                    train_prediction["residual"],
                    prediction["residual"],
                    args,
                    args.seed + 1771,
                    (
                        train_prediction.get("semantic_external")
                        if args.external_semantic_activation_dir
                        else None
                    ),
                    (
                        prediction.get("semantic_external")
                        if args.external_semantic_activation_dir
                        else None
                    ),
                )
            else:
                continuous_proxy = {
                    "train_mean": 0.0,
                    "train_max": 0.0,
                    "eval_mean": 0.0,
                    "eval_max": 0.0,
                }
            if upper_bound_active:
                assert train_prediction is not None
                upper_bound = meta_only_upper_bound_proxy(
                    train_prediction["z2"],
                    candidate_values.index_select(0, train_idx)
                    @ directions.t(),
                    prediction["z2"],
                    target,
                    prediction["semantic"],
                    args.meta_upper_bound_ridge,
                    args.meta_upper_bound_dim,
                )
            else:
                upper_bound = {
                    "mode": "none",
                    "meta_only_fraction_of_total": 0.0,
                    "meta_only_fraction_of_semantic_residual": 0.0,
                    "semantic_residual_capacity_fraction_of_total": 0.0,
                }
            aligned_checkpoint_score = (
                metrics.score
                - args.checkpoint_semantic_cluster_mmd_weight
                * val_semantic_cluster_mmd
                - args.checkpoint_semantic_bacc_proxy_weight
                * max(0.0, bacc_proxy["eval_mean"] - 0.5)
                - args.checkpoint_continuous_semantic_r2_weight
                * max(0.0, continuous_proxy["eval_mean"])
            )
            margin = max(args.semantic_invariance_gate_margin, 1e-6)
            rf_gate = min(
                1.0,
                max(
                    0.0,
                    (metrics.residual_fraction - args.semantic_invariance_rf_floor)
                    / margin,
                ),
            )
            gain_gate = min(
                1.0,
                max(
                    0.0,
                    (metrics.residual_gain - args.semantic_invariance_gain_floor)
                    / margin,
                ),
            )
            invariance_scale = rf_gate * gain_gate
            latest_stability = direction_history[-1]
            stability_ok = float(latest_stability.get("direction_cosine_mean", 1.0)) >= args.direction_min_cosine
            row.update(
                {
                    "val_checkpoint_score": aligned_checkpoint_score,
                    "val_information_score": metrics.score,
                    "val_semantic_cluster_mmd": val_semantic_cluster_mmd,
                    "val_semantic_bacc_proxy_train_mean": bacc_proxy["train_mean"],
                    "val_semantic_bacc_proxy_train_max": bacc_proxy["train_max"],
                    "val_semantic_bacc_proxy_mean": bacc_proxy["eval_mean"],
                    "val_semantic_bacc_proxy_max": bacc_proxy["eval_max"],
                    "val_continuous_semantic_r2_proxy_train_mean": (
                        continuous_proxy["train_mean"]
                    ),
                    "val_continuous_semantic_r2_proxy_train_max": (
                        continuous_proxy["train_max"]
                    ),
                    "val_continuous_semantic_r2_proxy_mean": (
                        continuous_proxy["eval_mean"]
                    ),
                    "val_continuous_semantic_r2_proxy_max": (
                        continuous_proxy["eval_max"]
                    ),
                    "val_meta_upper_bound_mode": args.meta_upper_bound_mode,
                    "val_meta_only_fraction_of_total": upper_bound.get(
                        "meta_only_fraction_of_total", 0.0
                    ),
                    "val_meta_only_fraction_of_semantic_residual": upper_bound.get(
                        "meta_only_fraction_of_semantic_residual", 0.0
                    ),
                    "val_semantic_residual_capacity_fraction_of_total": upper_bound.get(
                        "semantic_residual_capacity_fraction_of_total", 0.0
                    ),
                    "val_total_gain_mse": metrics.total_gain,
                    "val_semantic_gain_mse": metrics.semantic_gain,
                    "val_residual_gain_mse": metrics.residual_gain,
                    "val_residual_fraction": metrics.residual_fraction,
                    "val_positive_modules": metrics.positive_modules,
                    "direction_cosine_mean": latest_stability.get("direction_cosine_mean", 1.0),
                    "direction_support_jaccard_mean": latest_stability.get("support_jaccard_mean", 1.0),
                }
            )
            eligible = (
                epoch >= args.checkpoint_start_epoch
                and epoch >= args.direction_freeze_epoch
                and stability_ok
                and metrics.total_gain > 0.0
                and metrics.residual_gain >= args.checkpoint_min_residual_gain
                and metrics.residual_fraction >= args.checkpoint_min_residual_fraction
                and bacc_proxy["eval_mean"]
                <= args.checkpoint_max_semantic_bacc_proxy
                and continuous_proxy["eval_mean"]
                <= args.checkpoint_max_continuous_semantic_r2
            )
            if eligible and aligned_checkpoint_score > best_score:
                best_score = aligned_checkpoint_score
                metric_payload = {
                    **metrics.__dict__,
                    "semantic_cluster_mmd": val_semantic_cluster_mmd,
                    "semantic_bacc_proxy": bacc_proxy,
                    "continuous_semantic_r2_proxy": continuous_proxy,
                    "meta_only_upper_bound": upper_bound,
                    "aligned_checkpoint_score": aligned_checkpoint_score,
                }
                best_bundle = {
                    "epoch": epoch,
                    "score": aligned_checkpoint_score,
                    "metrics": metric_payload,
                    "model": cpu_state_dict(model),
                    "semantic_candidate_decoder": cpu_state_dict(semantic_candidate_decoder),
                    "residual_head": cpu_state_dict(residual_head),
                    "directions": directions.clone(),
                }
        history.append(row)
        progress.set_postfix(
            loss=f"{row['train_loss']:.4f}",
            gap=f"{row.get('val_residual_gain_mse', float('nan')):.4f}",
            mmd=f"{row.get('val_semantic_cluster_mmd', float('nan')):.4f}",
            pbacc=f"{row.get('val_semantic_bacc_proxy_mean', float('nan')):.3f}",
            pr2=f"{row.get('val_continuous_semantic_r2_proxy_mean', float('nan')):.3f}",
            rf=f"{row.get('val_residual_fraction', float('nan')):.3f}",
            ub=f"{row.get('val_meta_only_fraction_of_total', float('nan')):.3f}",
        )

    if best_bundle is None:
        prediction = predict_main(
            model,
            semantic_candidate_decoder,
            residual_head,
            x_next.index_select(0, val_idx),
            x_prev.index_select(0, val_idx),
            args.semantic_candidate_source,
            device,
            args.encode_batch_size,
            directions,
        )
        target = candidate_values.index_select(0, val_idx) @ directions.t()
        train_mean = (candidate_values.index_select(0, train_idx) @ directions.t()).mean(dim=0)
        metrics = module_information_metrics(
            target,
            prediction["semantic"],
            prediction["residual"],
            train_mean,
            (
                args.checkpoint_residual_gain_weight,
                args.checkpoint_total_gain_weight,
                args.checkpoint_residual_fraction_weight,
                args.checkpoint_positive_module_weight,
            ),
        )
        if (
            args.lambda_main_semantic_cluster_mmd > 0.0
            or args.checkpoint_semantic_cluster_mmd_weight > 0.0
        ):
            val_semantic_signature = nonlinear_semantic_signature(
                semantic_source_features(
                    prediction["z1"],
                    x_prev.index_select(0, val_idx),
                    args.semantic_candidate_source,
                ),
                main_semantic_projection.cpu(),
            )
            val_semantic_cluster_mmd = float(
                soft_cluster_semantic_mmd(
                    prediction["residual"],
                    val_semantic_signature,
                    args.semantic_cluster_mmd_temperature,
                    args.semantic_cluster_mmd_thresholds,
                ).item()
            )
        else:
            val_semantic_cluster_mmd = 0.0
        bacc_proxy_active = (
            args.checkpoint_semantic_bacc_proxy_weight > 0.0
            or args.checkpoint_max_semantic_bacc_proxy < 1.0
        )
        continuous_proxy_active = (
            args.checkpoint_continuous_semantic_r2_weight > 0.0
            or args.checkpoint_max_continuous_semantic_r2 < 1.0
        )
        upper_bound_active = args.meta_upper_bound_mode != "none"
        if bacc_proxy_active or continuous_proxy_active or upper_bound_active:
            train_prediction = predict_main(
                model,
                semantic_candidate_decoder,
                residual_head,
                x_next.index_select(0, train_idx),
                x_prev.index_select(0, train_idx),
                args.semantic_candidate_source,
                device,
                args.encode_batch_size,
                directions,
            )
        else:
            train_prediction = None
        if bacc_proxy_active:
            assert train_prediction is not None
            bacc_proxy = split_semantic_bacc_proxy(
                train_prediction["z1"],
                prediction["z1"],
                x_prev.index_select(0, train_idx),
                x_prev.index_select(0, val_idx),
                train_prediction["residual"],
                prediction["residual"],
                args,
                args.seed + 1751,
            )
        else:
            bacc_proxy = {
                "train_mean": 0.5,
                "train_max": 0.5,
                "eval_mean": 0.5,
                "eval_max": 0.5,
            }
        if continuous_proxy_active:
            assert train_prediction is not None
            continuous_proxy = split_semantic_continuous_proxy(
                train_prediction["z1"],
                prediction["z1"],
                x_prev.index_select(0, train_idx),
                x_prev.index_select(0, val_idx),
                train_prediction["residual"],
                prediction["residual"],
                args,
                args.seed + 1771,
            )
        else:
            continuous_proxy = {
                "train_mean": 0.0,
                "train_max": 0.0,
                "eval_mean": 0.0,
                "eval_max": 0.0,
            }
        if upper_bound_active:
            assert train_prediction is not None
            upper_bound = meta_only_upper_bound_proxy(
                train_prediction["z2"],
                candidate_values.index_select(0, train_idx) @ directions.t(),
                prediction["z2"],
                target,
                prediction["semantic"],
                args.meta_upper_bound_ridge,
                args.meta_upper_bound_dim,
            )
        else:
            upper_bound = {
                "mode": "none",
                "meta_only_fraction_of_total": 0.0,
                "meta_only_fraction_of_semantic_residual": 0.0,
                "semantic_residual_capacity_fraction_of_total": 0.0,
            }
        aligned_checkpoint_score = (
            metrics.score
            - args.checkpoint_semantic_cluster_mmd_weight
            * val_semantic_cluster_mmd
            - args.checkpoint_semantic_bacc_proxy_weight
            * max(0.0, bacc_proxy["eval_mean"] - 0.5)
            - args.checkpoint_continuous_semantic_r2_weight
            * max(0.0, continuous_proxy["eval_mean"])
        )
        best_bundle = {
            "epoch": args.epochs,
            "score": aligned_checkpoint_score,
            "metrics": {
                **metrics.__dict__,
                "semantic_cluster_mmd": val_semantic_cluster_mmd,
                "semantic_bacc_proxy": bacc_proxy,
                "continuous_semantic_r2_proxy": continuous_proxy,
                "meta_only_upper_bound": upper_bound,
                "aligned_checkpoint_score": aligned_checkpoint_score,
            },
            "model": cpu_state_dict(model),
            "semantic_candidate_decoder": cpu_state_dict(semantic_candidate_decoder),
            "residual_head": cpu_state_dict(residual_head),
            "directions": directions.clone(),
            "fallback": True,
        }
    model.load_state_dict(best_bundle["model"], strict=True)
    semantic_candidate_decoder.load_state_dict(
        best_bundle["semantic_candidate_decoder"], strict=True
    )
    residual_head.load_state_dict(best_bundle["residual_head"], strict=True)
    directions = best_bundle["directions"].float()
    write_csv(os.path.join(args.output_dir, "main_history.csv"), history)
    write_csv(os.path.join(args.output_dir, "module_direction_history_main.csv"), direction_history)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
    torch.save(
        {
            "semantic_candidate_decoder": semantic_candidate_decoder.state_dict(),
            "residual_head": residual_head.state_dict(),
            "module_directions": directions,
            "best_epoch": best_bundle["epoch"],
            "best_metrics": best_bundle["metrics"],
            "fallback": bool(best_bundle.get("fallback", False)),
            "semantic_probe": aligned_semantic_summary,
        },
        os.path.join(args.output_dir, "joint_module_heads.pt"),
    )
    return model, semantic_candidate_decoder, residual_head, directions, {
        "best_epoch": int(best_bundle["epoch"]),
        "best_score": float(best_bundle["score"]),
        "best_metrics": best_bundle["metrics"],
        "fallback": bool(best_bundle.get("fallback", False)),
        "direction_refreshes": len(direction_history),
        "semantic_probe": aligned_semantic_summary,
    }


def resolve_reuse_main_dir(path: str) -> str:
    source = os.path.normpath(path)
    if os.path.isfile(os.path.join(source, "best_model.pt")):
        return source
    nested = os.path.join(source, "decoupler_joint_v2")
    if os.path.isfile(os.path.join(nested, "best_model.pt")):
        return nested
    raise FileNotFoundError(
        f"{path} has neither best_model.pt nor decoupler_joint_v2/best_model.pt"
    )


def load_reused_main_stage(
    source_dir: str,
    x_next: torch.Tensor,
    candidate_neurons: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[base.Decoupler, nn.Module, nn.Module, torch.Tensor, Dict[str, Any]]:
    source_dir = resolve_reuse_main_dir(source_dir)
    config_path = os.path.join(source_dir, "config.json")
    source_config = json.load(open(config_path, encoding="utf-8"))
    required_matches = {
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "prev_layer": args.prev_layer,
        "layer_i": args.layer_i,
        "next_layer": args.next_layer,
        "candidate_top_k": args.candidate_top_k,
        "num_modules": args.num_modules,
        "module_support": args.module_support,
        "semantic_candidate_source": args.semantic_candidate_source,
        "semantic_probe_mode": args.semantic_probe_mode,
        "semantic_probe_hidden_dim": args.semantic_probe_hidden_dim,
        "semantic_encoder_warmup_epochs": (
            args.semantic_encoder_warmup_epochs
        ),
        "seed": args.seed,
        "split_seed": args.split_seed,
    }
    if source_config.get("external_semantic_activation_dir") or args.external_semantic_activation_dir:
        required_matches.update(
            {
                "external_semantic_head_type": args.external_semantic_head_type,
                "external_semantic_head_depth": args.external_semantic_head_depth,
            }
        )
    source_defaults = {
        "semantic_candidate_source": "z1",
        "semantic_probe_mode": "joint",
        "semantic_probe_hidden_dim": 512,
        "semantic_encoder_warmup_epochs": 30,
    }
    mismatches = {
        key: {
            "source": source_config.get(key, source_defaults.get(key)),
            "requested": value,
        }
        for key, value in required_matches.items()
        if source_config.get(key, source_defaults.get(key)) != value
    }
    if mismatches:
        raise ValueError(f"Reused first-stage config mismatch: {mismatches}")

    candidate_path = os.path.join(source_dir, "joint_candidate_pool.csv")
    with open(candidate_path, encoding="utf-8", newline="") as handle:
        source_candidates = sorted(
            int(row["neuron"]) for row in csv.DictReader(handle)
        )
    if source_candidates != candidate_neurons.tolist():
        requested = set(candidate_neurons.tolist())
        source = set(source_candidates)
        raise ValueError(
            "Reused first-stage candidate pool differs from the current data/config: "
            f"source_only={sorted(source - requested)[:8]} "
            f"current_only={sorted(requested - source)[:8]}"
        )

    model = base.Decoupler(
        x_next.size(1),
        args.latent_dim,
        args.hidden_dim,
        args.dropout,
        "continuous_binary",
    ).to(device)
    model_state = torch.load(
        os.path.join(source_dir, "best_model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    heads = torch.load(
        os.path.join(source_dir, "joint_module_heads.pt"),
        map_location="cpu",
        weights_only=False,
    )
    semantic_state = heads["semantic_candidate_decoder"]
    semantic_input_dim = (
        x_next.size(1)
        if args.semantic_candidate_source == "prev"
        else args.latent_dim
        + (
            x_next.size(1)
            if args.semantic_candidate_source == "z1_prev"
            else 0
        )
    )
    semantic_hidden_dim = (
        args.semantic_probe_hidden_dim
        if args.semantic_probe_mode in {"frozen_aligned", "staged_z1_aligned"}
        else max(256, args.hidden_dim // 2)
    )
    external_checkpoint = "external_to_candidate.weight" in semantic_state
    if external_checkpoint:
        if not args.external_semantic_activation_dir:
            raise ValueError(
                "Reused checkpoint contains an external semantic decoder, but the "
                "current run has no --external-semantic-activation-dir."
            )
        external_weight = semantic_state["external_to_candidate.weight"]
        external_dim = int(external_weight.size(1))
        checkpoint_candidate_dim = int(external_weight.size(0))
        if checkpoint_candidate_dim != candidate_neurons.numel():
            raise ValueError(
                "Reused external semantic decoder predicts "
                f"{checkpoint_candidate_dim} candidates, expected "
                f"{candidate_neurons.numel()}."
            )
        external_map_placeholder = torch.zeros(
            external_dim + 1,
            checkpoint_candidate_dim,
            dtype=external_weight.dtype,
        )
        semantic_candidate_decoder = ExternalSemanticCandidateDecoder(
            semantic_input_dim,
            external_dim,
            checkpoint_candidate_dim,
            semantic_hidden_dim,
            args.dropout,
            external_map_placeholder,
            source_config.get(
                "external_semantic_head_type", args.external_semantic_head_type
            ),
            int(
                source_config.get(
                    "external_semantic_head_depth", args.external_semantic_head_depth
                )
            ),
        ).to(device)
    else:
        if args.external_semantic_activation_dir:
            raise ValueError(
                "Current run requests an external semantic decoder, but the reused "
                "checkpoint contains a legacy candidate-only MLP."
            )
        semantic_candidate_decoder = base.MLP(
            semantic_input_dim,
            candidate_neurons.numel(),
            semantic_hidden_dim,
            args.dropout,
        ).to(device)
    residual_head = base.MLP(
        args.latent_dim,
        args.num_modules,
        max(128, args.hidden_dim // 2),
        args.dropout,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    semantic_candidate_decoder.load_state_dict(
        semantic_state, strict=True
    )
    residual_head.load_state_dict(heads["residual_head"], strict=True)
    directions = torch.as_tensor(heads["module_directions"]).float()
    if directions.shape != (args.num_modules, candidate_neurons.numel()):
        raise ValueError(
            "Reused module directions have shape "
            f"{tuple(directions.shape)}, expected "
            f"({args.num_modules}, {candidate_neurons.numel()})."
        )

    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
    torch.save(heads, os.path.join(args.output_dir, "joint_module_heads.pt"))
    for filename in (
        "main_history.csv",
        "module_direction_history_main.csv",
    ):
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(args.output_dir, filename)
        if (
            os.path.isfile(source_path)
            and os.path.abspath(source_path) != os.path.abspath(destination_path)
        ):
            shutil.copy2(source_path, destination_path)
    write_json(
        os.path.join(args.output_dir, "reused_main_source.json"),
        {
            "source_dir": source_dir,
            "best_epoch": int(heads.get("best_epoch", -1)),
            "best_metrics": heads.get("best_metrics", {}),
            "candidate_count": int(candidate_neurons.numel()),
            "module_count": int(directions.size(0)),
        },
    )
    metrics = heads.get("best_metrics", {})
    return model, semantic_candidate_decoder, residual_head, directions, {
        "best_epoch": int(heads.get("best_epoch", -1)),
        "best_score": float(metrics.get("score", float("nan"))),
        "best_metrics": metrics,
        "fallback": bool(heads.get("fallback", False)),
        "direction_refreshes": None,
        "reused": True,
        "source_dir": source_dir,
        "semantic_probe": heads.get(
            "semantic_probe",
            {
                "mode": args.semantic_probe_mode,
                "source": args.semantic_candidate_source,
            },
        ),
    }


def build_main_direct_encoding(
    model: base.Decoupler,
    semantic_candidate_decoder: nn.Module,
    residual_head: nn.Module,
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    candidate_values: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    directions: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    main = predict_main(
        model,
        semantic_candidate_decoder,
        residual_head,
        x_next,
        x_prev,
        args.semantic_candidate_source,
        device,
        args.encode_batch_size,
        directions,
    )
    val_target = candidate_values.index_select(0, val_idx) @ directions.t()
    train_mean = (
        candidate_values.index_select(0, train_idx) @ directions.t()
    ).mean(dim=0)
    metrics = module_information_metrics(
        val_target,
        main["semantic"].index_select(0, val_idx),
        main["residual"].index_select(0, val_idx),
        train_mean,
    )
    output_dir = os.path.join(args.output_dir, "main_direct")
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "source": "first_stage_best_checkpoint",
        "latent_source": "main_z2",
        "best_metrics": metrics.__dict__,
        "residual_gain_retention_vs_main": 1.0,
        "compression_stage_skipped": True,
        "module_target_mode": "sparse_continuous_combination",
        "prediction_target": "semantic_residual_module_state",
        "semantic_control": "frozen_main_candidate_decoder",
    }
    write_json(os.path.join(output_dir, "summary.json"), summary)
    result = {
        "z1": main["z1"],
        "z2": main["z2"],
        "semantic_code": main["z1"],
        "meta": main["z2"],
        "semantic": main["semantic"],
        "residual": main["residual"],
        "semantic_candidates": main["semantic_candidates"],
    }
    if "semantic_external" in main:
        result["semantic_external"] = main["semantic_external"]
    return result, summary


def predict_purifier(
    purifier: base.E2RecursivePurifier,
    residual_head: nn.Module,
    z2: torch.Tensor,
    semantic_candidate_prediction: torch.Tensor,
    directions: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    keys = ("semantic_code", "meta", "semantic", "residual", "semantic_z2", "z2_hat")
    values: Dict[str, List[torch.Tensor]] = {key: [] for key in keys}
    purifier.eval()
    residual_head.eval()
    direction_device = directions.to(device)
    with torch.inference_mode():
        for start in range(0, z2.size(0), batch_size):
            batch = z2[start : start + batch_size].to(device)
            out = purifier(batch)
            semantic_modules = (
                semantic_candidate_prediction[start : start + batch_size].to(device)
                @ direction_device.t()
            )
            values["semantic_code"].append(out["semantic"].float().cpu())
            values["meta"].append(out["meta"].float().cpu())
            values["semantic"].append(semantic_modules.float().cpu())
            values["residual"].append(residual_head(out["meta"]).float().cpu())
            values["semantic_z2"].append(out["semantic_z2_hat"].float().cpu())
            values["z2_hat"].append(out["z2_hat"].float().cpu())
    return {key: torch.cat(parts) for key, parts in values.items()}


def set_trainable(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train_purifier_stage(
    model: base.Decoupler,
    main_semantic_candidate_decoder: nn.Module,
    main_residual_head: nn.Module,
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    candidate_values: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    main_directions: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[base.E2RecursivePurifier, nn.Module, torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    main_encoded = predict_main(
        model,
        main_semantic_candidate_decoder,
        main_residual_head,
        x_next,
        x_prev,
        args.semantic_candidate_source,
        device,
        args.encode_batch_size,
        main_directions,
    )
    z1_all = main_encoded["z1"]
    z2_all = main_encoded["z2"]
    # This is the single semantic control used throughout direction discovery,
    # training, validation and export.  It is frozen before purifier training,
    # so the semantic branch cannot chase a direction after it is selected.
    semantic_candidates_all = main_encoded["semantic_candidates"].detach()
    main_teacher_residual_all = main_encoded["residual"].detach()
    main_val_target = candidate_values.index_select(0, val_idx) @ main_directions.t()
    main_train_mean = (
        candidate_values.index_select(0, train_idx) @ main_directions.t()
    ).mean(dim=0)
    main_reference_metrics = module_information_metrics(
        main_val_target,
        main_encoded["semantic"].index_select(0, val_idx),
        main_teacher_residual_all.index_select(0, val_idx),
        main_train_mean,
    )
    if (
        args.purifier_direction_mode != "fixed_main"
        and args.lambda_purifier_main_residual_distill > 0.0
    ):
        raise ValueError(
            "Main-residual distillation requires --purifier-direction-mode fixed_main "
            "because adaptive directions change the meaning of each output channel."
        )
    purifier = base.E2RecursivePurifier(
        args.latent_dim,
        args.purifier_semantic_dim,
        args.purifier_meta_dim,
        args.purifier_hidden_dim,
        args.dropout,
        args.latent_dim,
        x_prev.size(1),
        x_next.size(1),
        meta_input_mode="semantic_residual",
    ).to(device)
    residual_head = base.MLP(
        args.purifier_meta_dim,
        args.num_modules,
        max(128, args.purifier_hidden_dim // 2),
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(purifier.parameters()) + list(residual_head.parameters()),
        lr=args.purifier_lr,
        weight_decay=args.weight_decay,
    )
    dataset = TensorDataset(
        z1_all.index_select(0, train_idx),
        z2_all.index_select(0, train_idx),
        x_prev.index_select(0, train_idx),
        candidate_values.index_select(0, train_idx),
        semantic_candidates_all.index_select(0, train_idx),
        main_teacher_residual_all.index_select(0, train_idx),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 301),
    )
    directions = main_directions.clone()
    direction_history: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    best_score = -math.inf
    best_bundle: Optional[Dict[str, Any]] = None
    warmup_total = args.purifier_semantic_warmup + args.purifier_z2_warmup
    progress = tqdm(range(1, args.purifier_epochs + 1), desc="Joint-v2 purifier", unit="epoch")
    direction_dir = os.path.join(args.output_dir, "module_directions", "purifier")
    os.makedirs(direction_dir, exist_ok=True)
    torch.save(directions, os.path.join(direction_dir, "directions_epoch_000.pt"))
    if args.purifier_direction_mode == "fixed_main":
        direction_history.append(
            {
                "epoch": 0,
                "stage": "fixed_main",
                "direction_cosine_mean": 1.0,
                "direction_cosine_min": 1.0,
                "support_jaccard_mean": 1.0,
                "fixed_semantic_target_prediction": True,
                "direction_mode": "fixed_main",
            }
        )

    for epoch in progress:
        residual_stage = epoch > warmup_total
        if residual_stage:
            set_trainable(purifier, True)
            set_trainable(purifier.semantic_encoder, False)
            set_trainable(purifier.semantic_to_z2, False)
            set_trainable(purifier.semantic_to_z1, False)
            set_trainable(purifier.semantic_to_prev, False)
            set_trainable(purifier.semantic_to_i, False)
            set_trainable(residual_head, True)
        else:
            set_trainable(purifier, True)
            set_trainable(residual_head, False)

        should_refresh = (
            args.purifier_direction_mode == "adaptive"
            and residual_stage
            and args.purifier_direction_refresh_start <= epoch < args.purifier_direction_freeze_epoch
            and (epoch - args.purifier_direction_refresh_start) % args.purifier_direction_refresh_every == 0
        )
        if should_refresh:
            encoded = predict_purifier(
                purifier,
                residual_head,
                z2_all.index_select(0, train_idx),
                semantic_candidates_all.index_select(0, train_idx),
                directions,
                device,
                args.encode_batch_size,
            )
            directions, refresh = refresh_module_directions(
                candidate_values.index_select(0, train_idx),
                semantic_source_features(
                    encoded["semantic_code"],
                    x_prev.index_select(0, train_idx),
                    args.semantic_candidate_source,
                ),
                encoded["meta"],
                directions,
                args,
                args.seed + 3001 + epoch,
                semantic_target_prediction=semantic_candidates_all.index_select(0, train_idx),
            )
            direction_history.append({"epoch": epoch, "stage": "purifier", **refresh})
            torch.save(directions, os.path.join(direction_dir, f"directions_epoch_{epoch:03d}.pt"))

        purifier.train()
        residual_head.train(residual_stage)
        totals = {key: 0.0 for key in ("loss", "semantic_z2", "semantic", "meta_z2", "recon_z2", "semantic_module", "residual_module", "joint_module", "main_residual_distill", "orth", "cov", "var")}
        seen = 0
        direction_device = directions.to(device)
        for (
            z1_cpu,
            z2_cpu,
            prev_cpu,
            candidate_cpu,
            semantic_candidate_cpu,
            main_teacher_cpu,
        ) in loader:
            z1 = z1_cpu.to(device, non_blocking=True)
            z2 = z2_cpu.to(device, non_blocking=True)
            prev = prev_cpu.to(device, non_blocking=True)
            candidates = candidate_cpu.to(device, non_blocking=True)
            semantic_candidates = semantic_candidate_cpu.to(device, non_blocking=True)
            main_teacher_residual = main_teacher_cpu.to(device, non_blocking=True)
            module_target = candidates @ direction_device.t()
            out = purifier(z2)
            semantic_prediction = semantic_candidates @ direction_device.t()
            residual_prediction = residual_head(out["meta"])
            residual_target = module_target - semantic_prediction.detach()
            losses = {
                "semantic_z2": F.mse_loss(out["semantic_z2_hat"], z2),
                "semantic": F.mse_loss(out["z1_hat"], z1) + F.mse_loss(out["prev_hat"], prev),
                "meta_z2": F.mse_loss(out["meta_residual_hat"], z2 - out["semantic_z2_hat"].detach()),
                "recon_z2": F.mse_loss(out["z2_hat"], z2),
                "semantic_module": F.mse_loss(semantic_prediction, module_target),
                "residual_module": F.mse_loss(residual_prediction, residual_target),
                "joint_module": F.mse_loss(semantic_prediction + residual_prediction, module_target),
                "main_residual_distill": F.mse_loss(
                    residual_prediction, main_teacher_residual
                ),
                "orth": base.orthogonality_loss(out["semantic"], out["meta"]),
                "cov": centered_covariance_penalty(
                    out["meta"],
                    torch.cat(
                        [
                            semantic_source_features(
                                out["semantic"],
                                prev,
                                args.semantic_candidate_source,
                            ),
                            semantic_prediction.detach(),
                        ],
                        dim=1,
                    ),
                ),
                "var": variance_floor(out["semantic"], args.var_floor) + variance_floor(out["meta"], args.var_floor),
            }
            if residual_stage:
                ramp = min(1.0, (epoch - warmup_total) / 10.0)
                loss = (
                    args.lambda_purifier_meta_z2 * losses["meta_z2"]
                    + args.lambda_purifier_recon_z2 * losses["recon_z2"]
                    + ramp
                    * (
                        args.lambda_purifier_residual_module * losses["residual_module"]
                        + args.lambda_purifier_joint_module * losses["joint_module"]
                        + args.lambda_purifier_main_residual_distill
                        * losses["main_residual_distill"]
                    )
                    + args.lambda_purifier_orth * losses["orth"]
                    + args.lambda_purifier_meta_sem_cov * losses["cov"]
                    + args.lambda_purifier_var * losses["var"]
                )
            else:
                loss = (
                    args.lambda_purifier_semantic_z2 * losses["semantic_z2"]
                    + args.lambda_purifier_semantic * losses["semantic"]
                    + args.lambda_purifier_var * losses["var"]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = [parameter for group in optimizer.param_groups for parameter in group["params"] if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            batch_n = z1.size(0)
            totals["loss"] += float(loss.detach().item()) * batch_n
            for key, value in losses.items():
                totals[key] += float(value.detach().item()) * batch_n
            seen += batch_n

        row: Dict[str, Any] = {"epoch": epoch, "stage": "residual_meta" if residual_stage else "semantic", **{f"train_{key}": value / max(seen, 1) for key, value in totals.items()}}
        if epoch % args.eval_every == 0 or epoch == args.purifier_epochs:
            prediction = predict_purifier(
                purifier,
                residual_head,
                z2_all.index_select(0, val_idx),
                semantic_candidates_all.index_select(0, val_idx),
                directions,
                device,
                args.encode_batch_size,
            )
            target = candidate_values.index_select(0, val_idx) @ directions.t()
            train_mean = (candidate_values.index_select(0, train_idx) @ directions.t()).mean(dim=0)
            metrics = module_information_metrics(target, prediction["semantic"], prediction["residual"], train_mean)
            fixed_prediction = predict_purifier(
                purifier,
                residual_head,
                z2_all.index_select(0, val_idx),
                semantic_candidates_all.index_select(0, val_idx),
                main_directions,
                device,
                args.encode_batch_size,
            )
            fixed_metrics = module_information_metrics(
                main_val_target,
                fixed_prediction["semantic"],
                fixed_prediction["residual"],
                main_train_mean,
            )
            retention = (
                fixed_metrics.residual_gain / main_reference_metrics.residual_gain
                if main_reference_metrics.residual_gain > 1e-8
                else float("nan")
            )
            row.update(
                {
                    "val_checkpoint_score": metrics.score,
                    "val_total_gain_mse": metrics.total_gain,
                    "val_semantic_gain_mse": metrics.semantic_gain,
                    "val_residual_gain_mse": metrics.residual_gain,
                    "val_residual_fraction": metrics.residual_fraction,
                    "val_positive_modules": metrics.positive_modules,
                    "val_fixed_main_residual_gain_mse": fixed_metrics.residual_gain,
                    "val_fixed_main_residual_fraction": fixed_metrics.residual_fraction,
                    "val_fixed_main_positive_modules": fixed_metrics.positive_modules,
                    "val_main_residual_gain_reference_mse": main_reference_metrics.residual_gain,
                    "val_main_residual_fraction_reference": main_reference_metrics.residual_fraction,
                    "val_residual_gain_retention_vs_main": retention,
                }
            )
            stable = not direction_history or float(direction_history[-1].get("direction_cosine_mean", 1.0)) >= args.direction_min_cosine
            directions_ready = (
                args.purifier_direction_mode == "fixed_main"
                or epoch >= args.purifier_direction_freeze_epoch
            )
            eligible = (
                epoch >= args.purifier_checkpoint_start_epoch
                and directions_ready
                and stable
                and metrics.total_gain > 0.0
                and metrics.residual_gain > 0.0
            )
            if eligible and metrics.score > best_score:
                best_score = metrics.score
                best_bundle = {
                    "epoch": epoch,
                    "score": metrics.score,
                    "metrics": metrics.__dict__,
                    "fixed_main_metrics": fixed_metrics.__dict__,
                    "residual_gain_retention_vs_main": retention,
                    "purifier": cpu_state_dict(purifier),
                    "residual_head": cpu_state_dict(residual_head),
                    "directions": directions.clone(),
                }
        history.append(row)
        progress.set_postfix(
            stage=row["stage"],
            gap=f"{row.get('val_residual_gain_mse', float('nan')):.4f}",
            rf=f"{row.get('val_residual_fraction', float('nan')):.3f}",
            keep=f"{row.get('val_residual_gain_retention_vs_main', float('nan')):.2f}",
            loss=f"{row['train_loss']:.4f}",
        )

    if best_bundle is None:
        prediction = predict_purifier(
            purifier,
            residual_head,
            z2_all.index_select(0, val_idx),
            semantic_candidates_all.index_select(0, val_idx),
            directions,
            device,
            args.encode_batch_size,
        )
        target = candidate_values.index_select(0, val_idx) @ directions.t()
        train_mean = (candidate_values.index_select(0, train_idx) @ directions.t()).mean(dim=0)
        metrics = module_information_metrics(target, prediction["semantic"], prediction["residual"], train_mean)
        fixed_prediction = predict_purifier(
            purifier,
            residual_head,
            z2_all.index_select(0, val_idx),
            semantic_candidates_all.index_select(0, val_idx),
            main_directions,
            device,
            args.encode_batch_size,
        )
        fixed_metrics = module_information_metrics(
            main_val_target,
            fixed_prediction["semantic"],
            fixed_prediction["residual"],
            main_train_mean,
        )
        retention = (
            fixed_metrics.residual_gain / main_reference_metrics.residual_gain
            if main_reference_metrics.residual_gain > 1e-8
            else float("nan")
        )
        best_bundle = {
            "epoch": args.purifier_epochs,
            "score": metrics.score,
            "metrics": metrics.__dict__,
            "fixed_main_metrics": fixed_metrics.__dict__,
            "residual_gain_retention_vs_main": retention,
            "purifier": cpu_state_dict(purifier),
            "residual_head": cpu_state_dict(residual_head),
            "directions": directions.clone(),
            "fallback": True,
        }
    purifier.load_state_dict(best_bundle["purifier"], strict=True)
    residual_head.load_state_dict(best_bundle["residual_head"], strict=True)
    directions = best_bundle["directions"].float()
    purifier_dir = os.path.join(args.output_dir, "e2_recursive_purifier")
    os.makedirs(purifier_dir, exist_ok=True)
    torch.save(purifier.state_dict(), os.path.join(purifier_dir, "best_purifier.pt"))
    torch.save(
        {
            "residual_head": residual_head.state_dict(),
            "module_directions": directions,
            "semantic_source": "frozen_main_candidate_decoder",
            "best_epoch": best_bundle["epoch"],
            "best_metrics": best_bundle["metrics"],
            "best_fixed_main_metrics": best_bundle["fixed_main_metrics"],
            "main_reference_metrics": main_reference_metrics.__dict__,
            "residual_gain_retention_vs_main": best_bundle[
                "residual_gain_retention_vs_main"
            ],
            "direction_mode": args.purifier_direction_mode,
            "fallback": bool(best_bundle.get("fallback", False)),
        },
        os.path.join(purifier_dir, "joint_module_heads.pt"),
    )
    write_csv(os.path.join(purifier_dir, "history.csv"), history)
    write_csv(os.path.join(purifier_dir, "module_direction_history.csv"), direction_history)
    final_encoding = predict_purifier(
        purifier,
        residual_head,
        z2_all,
        semantic_candidates_all,
        directions,
        device,
        args.encode_batch_size,
    )
    summary = {
        "best_epoch": int(best_bundle["epoch"]),
        "best_score": float(best_bundle["score"]),
        "best_metrics": best_bundle["metrics"],
        "best_fixed_main_metrics": best_bundle["fixed_main_metrics"],
        "main_reference_metrics": main_reference_metrics.__dict__,
        "residual_gain_retention_vs_main": best_bundle[
            "residual_gain_retention_vs_main"
        ],
        "fallback": bool(best_bundle.get("fallback", False)),
        "direction_refreshes": sum(
            1 for row in direction_history if row.get("stage") == "purifier"
        ),
        "direction_history_rows": len(direction_history),
        "direction_mode": args.purifier_direction_mode,
        "main_residual_distill_weight": args.lambda_purifier_main_residual_distill,
        "module_target_mode": "sparse_continuous_combination",
        "prediction_target": "semantic_residual_module_state",
        "semantic_control": "frozen_main_candidate_decoder",
    }
    write_json(os.path.join(purifier_dir, "summary.json"), summary)
    final_result = {
        "z1": z1_all,
        "z2": z2_all,
        "semantic_candidates": semantic_candidates_all,
        **final_encoding,
    }
    if "semantic_external" in main_encoded:
        final_result["semantic_external"] = main_encoded["semantic_external"]
    return purifier, residual_head, directions, summary, final_result


def build_module_semantic_controls(
    semantic_parts: Sequence[torch.Tensor],
    train_idx: torch.Tensor,
    projection_dim: int,
    seed: int,
) -> torch.Tensor:
    """Build compact linear and nonlinear semantic controls without a large concat."""
    if not semantic_parts:
        raise ValueError("At least one semantic feature block is required.")
    total_dim = sum(int(part.size(1)) for part in semantic_parts)
    width = int(projection_dim)
    if width <= 0:
        return torch.cat(semantic_parts, dim=1).float()
    generator = torch.Generator().manual_seed(seed)
    projected = torch.zeros(semantic_parts[0].size(0), width, dtype=torch.float32)
    for part in semantic_parts:
        train = part.index_select(0, train_idx).float()
        mean = train.mean(dim=0, keepdim=True)
        std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
        matrix = torch.randn(
            part.size(1), width, generator=generator, dtype=torch.float32
        ) / math.sqrt(max(total_dim, 1))
        for start in range(0, part.size(0), 1024):
            standardized = (part[start : start + 1024].float() - mean) / std
            projected[start : start + standardized.size(0)] += standardized @ matrix
    train_projected = projected.index_select(0, train_idx)
    scale = train_projected.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    projected = projected / scale
    squared = projected.square()
    squared = squared - squared.index_select(0, train_idx).mean(dim=0, keepdim=True)
    return torch.cat([projected, torch.tanh(projected), squared], dim=1).contiguous()


def fit_semantic_ridge(
    features: torch.Tensor,
    target: torch.Tensor,
    ridge: float,
) -> Dict[str, torch.Tensor]:
    """Fit a centered ridge model used only as an explicit semantic control."""
    features = features.float()
    target = target.float().view(-1, 1)
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    target_mean = target.mean(dim=0, keepdim=True)
    x = (features - feature_mean) / feature_std
    y = target - target_mean
    scale = float(max(x.size(0), 1))
    gram = x.t().matmul(x) / scale
    gram = gram + max(float(ridge), 0.0) * torch.eye(
        gram.size(0), dtype=gram.dtype
    )
    rhs = x.t().matmul(y) / scale
    try:
        weight = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        weight = torch.linalg.pinv(gram).matmul(rhs)
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "weight": weight,
        "ridge": torch.tensor(float(ridge)),
    }


def predict_semantic_ridge(
    state: Dict[str, torch.Tensor], features: torch.Tensor
) -> torch.Tensor:
    standardized = (
        features.float() - state["feature_mean"]
    ) / state["feature_std"]
    return (standardized.matmul(state["weight"]) + state["target_mean"]).flatten()


def regression_diagnostics(target: torch.Tensor, prediction: torch.Tensor) -> Dict[str, float]:
    target = target.float().flatten()
    prediction = prediction.float().flatten()
    mse = float((target - prediction).square().mean().item())
    mean_mse = float((target - target.mean()).square().mean().item())
    return {
        "n": int(target.numel()),
        "mse": mse,
        "mean_baseline_mse": mean_mse,
        "r2": 1.0 - mse / max(mean_mse, 1e-12),
    }


def crossfit_semantic_module_prediction(
    semantic_controls: torch.Tensor,
    target: torch.Tensor,
    train_idx: torch.Tensor,
    selection_idx: torch.Tensor,
    evaluation_idx: torch.Tensor,
    folds: int,
    ridge: float,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    """Predict a module target from semantics without in-fold train leakage."""
    train_idx = train_idx.long()
    folds = max(2, min(int(folds), int(train_idx.numel())))
    target = target.float().flatten()
    predictions = torch.empty_like(target)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(train_idx.numel(), generator=generator)
    fold_assignment = torch.empty(train_idx.numel(), dtype=torch.long)
    fold_assignment[permutation] = torch.arange(train_idx.numel()) % folds
    for fold in range(folds):
        held_local = (fold_assignment == fold).nonzero(as_tuple=False).flatten()
        fit_local = (fold_assignment != fold).nonzero(as_tuple=False).flatten()
        held_index = train_idx.index_select(0, held_local)
        fit_index = train_idx.index_select(0, fit_local)
        state = fit_semantic_ridge(
            semantic_controls.index_select(0, fit_index),
            target.index_select(0, fit_index),
            ridge,
        )
        predictions.index_copy_(
            0,
            held_index,
            predict_semantic_ridge(
                state, semantic_controls.index_select(0, held_index)
            ),
        )
    full_state = fit_semantic_ridge(
        semantic_controls.index_select(0, train_idx),
        target.index_select(0, train_idx),
        ridge,
    )
    nontrain_mask = torch.ones(target.numel(), dtype=torch.bool)
    nontrain_mask[train_idx] = False
    nontrain_idx = nontrain_mask.nonzero(as_tuple=False).flatten()
    if nontrain_idx.numel():
        predictions.index_copy_(
            0,
            nontrain_idx,
            predict_semantic_ridge(
                full_state, semantic_controls.index_select(0, nontrain_idx)
            ),
        )
    diagnostics = {
        "mode": "crossfit_ridge",
        "folds": folds,
        "ridge": float(ridge),
        "train_oof": regression_diagnostics(
            target.index_select(0, train_idx), predictions.index_select(0, train_idx)
        ),
        "selection": regression_diagnostics(
            target.index_select(0, selection_idx),
            predictions.index_select(0, selection_idx),
        ),
        "evaluation": regression_diagnostics(
            target.index_select(0, evaluation_idx),
            predictions.index_select(0, evaluation_idx),
        ),
    }
    return predictions, diagnostics, full_state


def project_module_code_1d(
    train_code: torch.Tensor,
    selection_code: torch.Tensor,
    evaluation_code: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Project a module code onto one train-fitted axis for modality diagnosis."""
    train_code = train_code.float()
    selection_code = selection_code.float()
    evaluation_code = evaluation_code.float()
    mean = train_code.mean(dim=0, keepdim=True)
    centered = train_code - mean
    if train_code.size(1) == 1:
        axis = torch.ones(1, 1, dtype=train_code.dtype)
        projection = "task_aligned_scalar"
    else:
        covariance = centered.t().matmul(centered) / max(train_code.size(0) - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        axis = eigenvectors[:, -1:].contiguous()
        projection = "train_pc1"
    train_value = centered.matmul(axis).flatten()
    scale = train_value.std(unbiased=False).clamp_min(1e-6)

    def apply(value: torch.Tensor) -> torch.Tensor:
        return ((value.float() - mean).matmul(axis).flatten() / scale).contiguous()

    return (
        train_value / scale,
        apply(selection_code),
        apply(evaluation_code),
        {
            "projection": projection,
            "source_code_dim": int(train_code.size(1)),
            "axis": axis.flatten().tolist(),
            "train_scale": float(scale.item()),
        },
    )


def gaussian_mixture_log_likelihood(
    value: torch.Tensor, state: Dict[str, torch.Tensor]
) -> torch.Tensor:
    value = value.float().flatten().view(-1, 1)
    mean = state["mean"].float().view(1, -1)
    variance = state["variance"].float().view(1, -1).clamp_min(1e-6)
    weight = state["weight"].float().view(1, -1).clamp_min(1e-8)
    component = (
        weight.log()
        - 0.5 * (math.log(2.0 * math.pi) + variance.log())
        - 0.5 * (value - mean).square() / variance
    )
    return torch.logsumexp(component, dim=1)


def fit_univariate_gaussian_mixture(
    value: torch.Tensor,
    components: int,
    seed: int,
    restarts: int = 8,
    iterations: int = 100,
) -> Dict[str, torch.Tensor]:
    """Small deterministic EM fit used only for continuous-vs-discrete auditing."""
    value = value.float().flatten()
    value = value[torch.isfinite(value)]
    if value.numel() < max(20, 4 * components):
        raise ValueError("Too few finite module-code values for mixture diagnosis.")
    global_variance = value.var(unbiased=False).clamp_min(1e-4)
    if components == 1:
        state = {
            "mean": value.mean().view(1),
            "variance": global_variance.view(1),
            "weight": torch.ones(1),
        }
        state["log_likelihood"] = gaussian_mixture_log_likelihood(
            value, state
        ).sum()
        return state
    if components != 2:
        raise ValueError("The modality diagnostic currently supports one or two components.")

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_log_likelihood = -math.inf
    generator = torch.Generator().manual_seed(seed)
    quantiles = torch.quantile(value, torch.tensor([0.25, 0.75]))
    for restart in range(max(1, restarts)):
        if restart == 0:
            mean = quantiles.clone()
        else:
            chosen = torch.randint(
                value.numel(), (2,), generator=generator
            )
            mean = value.index_select(0, chosen).sort().values
        variance = global_variance.repeat(2)
        weight = torch.full((2,), 0.5)
        state = {"mean": mean, "variance": variance, "weight": weight}
        for _ in range(max(1, iterations)):
            x = value.view(-1, 1)
            component = (
                state["weight"].clamp_min(1e-8).log().view(1, -1)
                - 0.5
                * (
                    math.log(2.0 * math.pi)
                    + state["variance"].clamp_min(1e-6).log().view(1, -1)
                )
                - 0.5
                * (x - state["mean"].view(1, -1)).square()
                / state["variance"].clamp_min(1e-6).view(1, -1)
            )
            responsibility = component.softmax(dim=1)
            mass = responsibility.sum(dim=0).clamp_min(1e-4)
            new_weight = (mass / value.numel()).clamp_min(1e-4)
            new_weight = new_weight / new_weight.sum()
            new_mean = responsibility.t().matmul(value) / mass
            new_variance = (
                responsibility
                * (x - new_mean.view(1, -1)).square()
            ).sum(dim=0) / mass
            new_variance = new_variance.clamp_min(
                max(1e-5, float(global_variance.item()) * 1e-4)
            )
            delta = max(
                float((new_mean - state["mean"]).abs().max().item()),
                float((new_variance - state["variance"]).abs().max().item()),
            )
            state = {
                "mean": new_mean,
                "variance": new_variance,
                "weight": new_weight,
            }
            if delta < 1e-6:
                break
        order = state["mean"].argsort()
        state = {
            "mean": state["mean"].index_select(0, order),
            "variance": state["variance"].index_select(0, order),
            "weight": state["weight"].index_select(0, order),
        }
        log_likelihood = float(
            gaussian_mixture_log_likelihood(value, state).sum().item()
        )
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_state = {key: item.clone() for key, item in state.items()}
    if best_state is None:
        raise RuntimeError("Gaussian-mixture fitting did not produce a finite state.")
    best_state["log_likelihood"] = torch.tensor(best_log_likelihood)
    return best_state


def module_state_structure_diagnostics(
    train_code: torch.Tensor,
    selection_code: torch.Tensor,
    evaluation_code: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, Any]:
    """Test whether a train-fitted two-state model generalizes beyond KMeans."""
    train_value, selection_value, evaluation_value, projection = (
        project_module_code_1d(train_code, selection_code, evaluation_code)
    )
    single = fit_univariate_gaussian_mixture(train_value, 1, seed)
    mixture = fit_univariate_gaussian_mixture(train_value, 2, seed + 1)
    train_n = max(int(train_value.numel()), 1)
    bic_single = (
        -2.0 * float(single["log_likelihood"].item())
        + 2.0 * math.log(train_n)
    )
    bic_mixture = (
        -2.0 * float(mixture["log_likelihood"].item())
        + 5.0 * math.log(train_n)
    )
    bic_gain = bic_single - bic_mixture

    pooled_scale = (
        0.5 * (mixture["variance"][0] + mixture["variance"][1])
    ).sqrt().clamp_min(1e-6)
    separation = float(
        ((mixture["mean"][1] - mixture["mean"][0]).abs() / pooled_scale).item()
    )
    grid = torch.linspace(
        float(mixture["mean"][0].item()),
        float(mixture["mean"][1].item()),
        257,
    )
    grid_density = gaussian_mixture_log_likelihood(grid, mixture).exp()
    peak_density = gaussian_mixture_log_likelihood(mixture["mean"], mixture).exp()
    valley_ratio = float(
        (
            grid_density.min()
            / peak_density.min().clamp_min(1e-12)
        ).item()
    )

    def heldout_gain(value: torch.Tensor) -> float:
        if value.numel() == 0:
            return math.nan
        return float(
            (
                gaussian_mixture_log_likelihood(value, mixture)
                - gaussian_mixture_log_likelihood(value, single)
            ).mean().item()
        )

    selection_ll_gain = heldout_gain(selection_value)
    evaluation_ll_gain = heldout_gain(evaluation_value)
    structural_gate = (
        bic_gain >= args.module_multimodality_min_bic_gain
        and separation >= args.module_multimodality_min_separation
        and float(mixture["weight"].min().item())
        >= args.module_multimodality_min_component_fraction
        and valley_ratio <= args.module_multimodality_max_valley_ratio
    )
    selection_supported = (
        structural_gate
        and selection_ll_gain >= args.module_multimodality_min_heldout_ll_gain
    )
    evaluation_supported = (
        structural_gate
        and evaluation_ll_gain >= args.module_multimodality_min_heldout_ll_gain
    )
    return {
        **projection,
        "train_bic_single": bic_single,
        "train_bic_two_component": bic_mixture,
        "train_bic_gain_two_vs_one": bic_gain,
        "component_means": mixture["mean"].tolist(),
        "component_variances": mixture["variance"].tolist(),
        "component_weights": mixture["weight"].tolist(),
        "component_separation": separation,
        "valley_ratio": valley_ratio,
        "selection_log_likelihood_gain_per_sample": selection_ll_gain,
        "evaluation_log_likelihood_gain_per_sample": evaluation_ll_gain,
        "train_structure_supported": bool(structural_gate),
        "selection_discrete_state_supported": bool(selection_supported),
        "evaluation_discrete_state_supported": bool(evaluation_supported),
        "recommended_interpretation": (
            "discrete_state"
            if selection_supported and evaluation_supported
            else "continuous_axis"
        ),
        "gate": {
            "min_bic_gain": args.module_multimodality_min_bic_gain,
            "min_heldout_log_likelihood_gain": (
                args.module_multimodality_min_heldout_ll_gain
            ),
            "min_component_separation": (
                args.module_multimodality_min_separation
            ),
            "min_component_fraction": (
                args.module_multimodality_min_component_fraction
            ),
            "max_valley_ratio": args.module_multimodality_max_valley_ratio,
        },
    }


def semantic_knn_indices(
    semantic_controls: torch.Tensor,
    train_idx: torch.Tensor,
    neighbors: int,
    dimensions: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Find train-only semantic neighbors for every row, excluding train self-matches."""
    train_count = int(train_idx.numel())
    if train_count < 2:
        raise ValueError("Conditional-rank residualization requires at least two train rows.")
    neighbors = min(max(1, int(neighbors)), train_count - 1)
    dimensions = min(max(1, int(dimensions)), int(semantic_controls.size(1)))
    compact = F.normalize(semantic_controls[:, :dimensions].float(), dim=1)
    reference = compact.index_select(0, train_idx).to(device)
    train_position = torch.full((compact.size(0),), -1, dtype=torch.long)
    train_position[train_idx] = torch.arange(train_idx.numel())
    outputs = []
    progress = tqdm(
        range(0, compact.size(0), batch_size),
        desc="Semantic KNN graph",
        leave=False,
    )
    for start in progress:
        end = min(start + batch_size, compact.size(0))
        similarity = compact[start:end].to(device).matmul(reference.t())
        positions = train_position[start:end]
        local_rows = (positions >= 0).nonzero(as_tuple=False).flatten()
        if local_rows.numel():
            similarity[
                local_rows.to(device), positions.index_select(0, local_rows).to(device)
            ] = -float("inf")
        local_neighbors = similarity.topk(neighbors, dim=1).indices.cpu()
        outputs.append(train_idx.index_select(0, local_neighbors.flatten()).view_as(local_neighbors))
    return torch.cat(outputs, dim=0).contiguous()


def conditional_rank_code(
    score: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    neighbor_scores = score.index_select(0, neighbor_indices.flatten()).view_as(
        neighbor_indices
    )
    value = score.view(-1, 1)
    rank = (neighbor_scores < value).float().sum(dim=1)
    rank += 0.5 * (neighbor_scores == value).float().sum(dim=1)
    probability = (rank + 0.5) / float(neighbor_indices.size(1) + 1)
    probability = probability.clamp(1e-4, 1.0 - 1e-4)
    code = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    conditional_mean = neighbor_scores.mean(dim=1)
    residual = score - conditional_mean
    diagnostics = {
        "conditional_probability_mean": float(probability.mean().item()),
        "conditional_probability_std": float(probability.std(unbiased=False).item()),
        "code_mean": float(code.mean().item()),
        "code_std": float(code.std(unbiased=False).item()),
        "remaining_variance_ratio": float(
            residual.var(unbiased=False).item()
            / max(float(score.var(unbiased=False).item()), 1e-12)
        ),
    }
    return code, conditional_mean, diagnostics


def inverse_conditional_rank_code(
    code: torch.Tensor,
    reference_score: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    neighbor_scores = reference_score.index_select(
        0, neighbor_indices.flatten()
    ).view_as(neighbor_indices)
    ordered = neighbor_scores.sort(dim=1).values
    probability = 0.5 * (1.0 + torch.erf(code.float() / math.sqrt(2.0)))
    position = probability.clamp(0.0, 1.0) * float(ordered.size(1) - 1)
    lower = position.floor().long()
    upper = position.ceil().long()
    fraction = position - lower.float()
    lower_value = ordered.gather(1, lower.view(-1, 1)).flatten()
    upper_value = ordered.gather(1, upper.view(-1, 1)).flatten()
    return lower_value + fraction * (upper_value - lower_value)


def train_module_refiner(
    meta: torch.Tensor,
    base_delta: torch.Tensor,
    semantic_controls: torch.Tensor,
    module_loading: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Tuple[ModuleRefiner, Dict[str, Any]]:
    if args.module_refiner_code_mode == "task_aligned":
        model = TaskAlignedModuleRefiner(
            meta.size(1),
            args.module_code_dim,
            args.module_refiner_hidden_dim,
            base_delta.size(1),
            args.dropout,
            module_loading,
        ).to(device)
    else:
        model = ModuleRefiner(
            meta.size(1),
            args.module_code_dim,
            args.module_refiner_hidden_dim,
            base_delta.size(1),
            args.dropout,
        ).to(device)
    loader = DataLoader(
        TensorDataset(
            meta.index_select(0, train_idx),
            base_delta.index_select(0, train_idx),
            semantic_controls.index_select(0, train_idx),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.module_refiner_lr, weight_decay=args.weight_decay
    )
    train_target = base_delta.index_select(0, train_idx)
    validation_target = base_delta.index_select(0, val_idx)
    train_target_mean = train_target.mean(dim=0, keepdim=True)
    validation_baseline_mse = float(
        F.mse_loss(
            train_target_mean.expand_as(validation_target), validation_target
        ).item()
    )
    best_state = None
    best_objective = math.inf
    best_val_mse = math.inf
    best_val_semantic = math.inf
    best_val_semantic_cluster_mmd = math.inf
    best_val_relative_mse = math.inf
    best_val_relative_gain = -math.inf
    best_val_bacc_proxy_train = math.inf
    best_val_bacc_proxy_eval = math.inf
    best_epoch = 0
    fallback_state = None
    fallback_metrics: Dict[str, float] = {}
    fallback_objective = math.inf
    refiner_invariance_scale = (
        1.0 if args.module_refiner_min_relative_gain <= 0.0 else 0.0
    )
    progress = tqdm(
        range(1, args.module_refiner_epochs + 1),
        desc="Module refiner",
        leave=False,
        unit="epoch",
    )
    for epoch in progress:
        model.train()
        total = 0.0
        seen = 0
        for meta_cpu, target_cpu, semantic_cpu in loader:
            meta_batch = meta_cpu.to(device, non_blocking=True)
            target = target_cpu.to(device, non_blocking=True)
            semantic_batch = semantic_cpu.to(device, non_blocking=True)
            code, prediction = model(meta_batch)
            prediction_loss = F.mse_loss(prediction, target)
            semantic_loss = centered_covariance_penalty(code, semantic_batch)
            effective_refiner_invariance = (
                refiner_invariance_scale
                if epoch >= args.module_refiner_invariance_start_epoch
                else 0.0
            )
            if (
                args.lambda_module_refiner_semantic_cluster_mmd > 0.0
                and effective_refiner_invariance > 0.0
            ):
                semantic_cluster_mmd = soft_cluster_semantic_mmd(
                    code,
                    semantic_batch,
                    args.semantic_cluster_mmd_temperature,
                    args.semantic_cluster_mmd_thresholds,
                )
            else:
                semantic_cluster_mmd = code.new_zeros(())
            loss = (
                prediction_loss
                + args.lambda_module_refiner_semantic_cov * semantic_loss
                + args.lambda_module_refiner_semantic_cluster_mmd
                * effective_refiner_invariance
                * semantic_cluster_mmd
                + 0.01 * variance_floor(code, 0.25)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().item()) * meta_batch.size(0)
            seen += meta_batch.size(0)
        model.eval()
        with torch.inference_mode():
            prediction_parts = []
            code_parts = []
            for start in range(0, val_idx.numel(), args.encode_batch_size):
                index = val_idx[start : start + args.encode_batch_size]
                code, prediction = model(meta.index_select(0, index).to(device))
                prediction_parts.append(prediction.float().cpu())
                code_parts.append(code.float().cpu())
            val_prediction = torch.cat(prediction_parts)
            val_code = torch.cat(code_parts)
        val_mse = float(
            F.mse_loss(val_prediction, base_delta.index_select(0, val_idx)).item()
        )
        val_semantic = float(
            centered_covariance_penalty(
                val_code, semantic_controls.index_select(0, val_idx)
            ).item()
        )
        if (
            args.lambda_module_refiner_semantic_cluster_mmd > 0.0
            or args.module_refiner_checkpoint_mmd_weight > 0.0
        ):
            val_semantic_cluster_mmd = float(
                soft_cluster_semantic_mmd(
                    val_code,
                    semantic_controls.index_select(0, val_idx),
                    args.semantic_cluster_mmd_temperature,
                    args.semantic_cluster_mmd_thresholds,
                ).item()
            )
        else:
            val_semantic_cluster_mmd = 0.0
        refiner_bacc_proxy_active = (
            args.module_refiner_checkpoint_bacc_weight > 0.0
            or args.module_refiner_max_semantic_bacc_proxy < 1.0
        )
        if refiner_bacc_proxy_active:
            train_code, _ = refiner_outputs(
                model,
                meta.index_select(0, train_idx),
                device,
                args.encode_batch_size,
            )
            refiner_bacc_proxy = semantic_score_bacc_proxy(
                semantic_controls.index_select(0, train_idx),
                train_code,
                semantic_controls.index_select(0, val_idx),
                val_code,
                args.semantic_bacc_proxy_ridge,
                args.semantic_bacc_proxy_dim,
            )
        else:
            refiner_bacc_proxy = {
                "train_mean": 0.5,
                "train_max": 0.5,
                "eval_mean": 0.5,
                "eval_max": 0.5,
            }
        val_relative_mse = val_mse / max(validation_baseline_mse, 1e-8)
        val_relative_gain = 1.0 - val_relative_mse
        gain_margin = max(args.module_refiner_invariance_gain_margin, 1e-6)
        refiner_invariance_scale = min(
            1.0,
            max(
                0.0,
                (
                    val_relative_gain
                    - args.module_refiner_min_relative_gain
                )
                / gain_margin,
            ),
        )
        gain_shortfall = max(
            0.0, args.module_refiner_min_relative_gain - val_relative_gain
        )
        val_objective = (
            val_relative_mse
            + 2.0 * gain_shortfall
            + args.lambda_module_refiner_semantic_cov * val_semantic
            + args.module_refiner_checkpoint_mmd_weight
            * val_semantic_cluster_mmd
            + args.module_refiner_checkpoint_bacc_weight
            * max(
                0.0,
                refiner_bacc_proxy["eval_mean"]
                - args.module_refiner_checkpoint_bacc_target,
            )
        )
        checkpoint_metrics = {
            "epoch": float(epoch),
            "mse": val_mse,
            "relative_mse": val_relative_mse,
            "relative_gain": val_relative_gain,
            "semantic_cov": val_semantic,
            "semantic_cluster_mmd": val_semantic_cluster_mmd,
            "semantic_bacc_proxy_train": refiner_bacc_proxy["train_mean"],
            "semantic_bacc_proxy_eval": refiner_bacc_proxy["eval_mean"],
        }
        if val_objective < fallback_objective:
            fallback_objective = val_objective
            fallback_state = cpu_state_dict(model)
            fallback_metrics = checkpoint_metrics
        information_gate_pass = (
            val_relative_gain >= args.module_refiner_min_relative_gain
            and refiner_bacc_proxy["eval_mean"]
            <= args.module_refiner_max_semantic_bacc_proxy
        )
        if information_gate_pass and val_objective < best_objective:
            best_objective = val_objective
            best_val_mse = val_mse
            best_val_semantic = val_semantic
            best_val_semantic_cluster_mmd = val_semantic_cluster_mmd
            best_val_relative_mse = val_relative_mse
            best_val_relative_gain = val_relative_gain
            best_val_bacc_proxy_train = refiner_bacc_proxy["train_mean"]
            best_val_bacc_proxy_eval = refiner_bacc_proxy["eval_mean"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)
        progress.set_postfix(
            train=f"{total / max(seen, 1):.4f}",
            gain=f"{val_relative_gain:.3f}",
            mmd=f"{val_semantic_cluster_mmd:.4f}",
            pbacc=f"{refiner_bacc_proxy['eval_mean']:.3f}",
            val=f"{val_relative_mse:.3f}",
        )
    refiner_information_gate_pass = best_state is not None
    if best_state is None:
        if fallback_state is None:
            raise RuntimeError("Module refiner did not produce a finite checkpoint.")
        best_state = fallback_state
        best_objective = fallback_objective
        best_epoch = int(fallback_metrics["epoch"])
        best_val_mse = fallback_metrics["mse"]
        best_val_relative_mse = fallback_metrics["relative_mse"]
        best_val_relative_gain = fallback_metrics["relative_gain"]
        best_val_semantic = fallback_metrics["semantic_cov"]
        best_val_semantic_cluster_mmd = fallback_metrics[
            "semantic_cluster_mmd"
        ]
        best_val_bacc_proxy_train = fallback_metrics["semantic_bacc_proxy_train"]
        best_val_bacc_proxy_eval = fallback_metrics["semantic_bacc_proxy_eval"]
    model.load_state_dict(best_state, strict=True)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_mse,
        "best_validation_relative_mse": best_val_relative_mse,
        "best_validation_relative_gain": best_val_relative_gain,
        "best_validation_semantic_cov": best_val_semantic,
        "best_validation_semantic_cluster_mmd": best_val_semantic_cluster_mmd,
        "best_validation_objective": best_objective,
        "information_gate_pass": refiner_information_gate_pass,
        "minimum_relative_gain": args.module_refiner_min_relative_gain,
        "best_semantic_bacc_proxy_train": best_val_bacc_proxy_train,
        "best_semantic_bacc_proxy_eval": best_val_bacc_proxy_eval,
        "code_mode": args.module_refiner_code_mode,
        "semantic_cov_weight": args.lambda_module_refiner_semantic_cov,
        "semantic_cluster_mmd_weight": args.lambda_module_refiner_semantic_cluster_mmd,
        "checkpoint_semantic_cluster_mmd_weight": args.module_refiner_checkpoint_mmd_weight,
    }


def refiner_outputs(
    model: ModuleRefiner,
    meta: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    codes, outputs = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, meta.size(0), batch_size):
            code, output = model(meta[start : start + batch_size].to(device))
            codes.append(code.float().cpu())
            outputs.append(output.float().cpu())
    return torch.cat(codes), torch.cat(outputs)


def fit_conditional_incremental_calibration(
    raw_code: torch.Tensor,
    semantic_residual_target: torch.Tensor,
    train_idx: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Calibrate a meta-only code as an additive gain over a fixed semantic model."""
    x = raw_code.float().view(-1, 1)
    y = semantic_residual_target.float().view(-1, 1)
    train_x = x.index_select(0, train_idx)
    train_y = y.index_select(0, train_idx)
    coefficient = ridge_coefficients(train_x, train_y, ridge)
    scale = coefficient[0, 0].detach().cpu()
    bias = coefficient[1, 0].detach().cpu()
    calibrated = x * scale + bias
    train_baseline = train_y.mean().expand_as(train_y)
    train_prediction = calibrated.index_select(0, train_idx)
    baseline_mse = float(F.mse_loss(train_baseline, train_y).item())
    calibrated_mse = float(F.mse_loss(train_prediction, train_y).item())
    return calibrated.flatten(), scale, {
        "mode": "train_only_affine_incremental",
        "scale": float(scale.item()),
        "bias": float(bias.item()),
        "ridge": float(ridge),
        "train_target_variance": float(train_y.var(unbiased=False).item()),
        "train_baseline_mse": baseline_mse,
        "train_calibrated_mse": calibrated_mse,
        "train_relative_gain": (
            1.0 - calibrated_mse / max(baseline_mse, 1e-12)
        ),
    }


def save_module_plot(module_dir: str, rows: Sequence[Dict[str, Any]], bacc: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [f"M{int(row['module'])}" for row in rows]
    semantic = [float(row["semantic_gain_mse"]) for row in rows]
    residual = [float(row["residual_gain_mse"]) for row in rows]
    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    x = list(range(len(labels)))
    axis.bar(x, semantic, label="Semantic gain", color="#4C78A8")
    axis.bar(x, residual, bottom=semantic, label="Residual gain", color="#E45756")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Held-out MSE reduction")
    axis.set_title(f"Combination-level information (semantic BAcc={bacc:.3f})")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(os.path.join(module_dir, "module_information.png"), dpi=200)
    plt.close(figure)


def save_module_state_plot(
    module_dir: str,
    train_code: torch.Tensor,
    selection_code: torch.Tensor,
    evaluation_code: torch.Tensor,
    diagnostics: Dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    train, selection, evaluation, _ = project_module_code_1d(
        train_code, selection_code, evaluation_code
    )
    values = torch.cat([train, selection, evaluation]).float()
    low = float(torch.quantile(values, 0.005).item())
    high = float(torch.quantile(values, 0.995).item())
    grid = torch.linspace(low, high, 400)
    mean = torch.tensor(diagnostics["component_means"]).float()
    variance = torch.tensor(diagnostics["component_variances"]).float()
    weight = torch.tensor(diagnostics["component_weights"]).float()
    component_density = (
        weight.view(1, -1)
        / torch.sqrt(2.0 * math.pi * variance).view(1, -1)
        * torch.exp(
            -0.5
            * (grid.view(-1, 1) - mean.view(1, -1)).square()
            / variance.view(1, -1)
        )
    )

    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.hist(
        train.numpy(),
        bins=60,
        density=True,
        alpha=0.30,
        color="#0072B2",
        label="Train code",
    )
    axis.hist(
        evaluation.numpy(),
        bins=60,
        density=True,
        histtype="step",
        linewidth=1.4,
        color="#D55E00",
        label="Held-out code",
    )
    axis.plot(
        grid.numpy(),
        component_density.sum(dim=1).numpy(),
        color="#111111",
        linewidth=1.8,
        label="Train-fitted 2-Gaussian density",
    )
    for column, color in enumerate(("#009E73", "#CC79A7")):
        axis.plot(
            grid.numpy(),
            component_density[:, column].numpy(),
            color=color,
            linewidth=1.0,
            linestyle="--",
        )
    axis.set_xlabel("Train-standardized module coordinate")
    axis.set_ylabel("Density")
    axis.set_title(
        f"{diagnostics['recommended_interpretation'].replace('_', ' ')} | "
        f"BIC gain={diagnostics['train_bic_gain_two_vs_one']:.1f}, "
        f"held-out LL gain="
        f"{diagnostics['evaluation_log_likelihood_gain_per_sample']:.3f}"
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        os.path.join(module_dir, "module_state_distribution.png"), dpi=220
    )
    plt.close(figure)


def export_modules(
    ids: Sequence[str],
    raw_i: torch.Tensor,
    candidate_values: torch.Tensor,
    candidate_neurons: torch.Tensor,
    threshold: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    prev_features: torch.Tensor,
    encoded: Dict[str, torch.Tensor],
    semantic_prediction: torch.Tensor,
    directions: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    config: Dict[str, Any],
    module_indices: Optional[Sequence[int]] = None,
    finalize: bool = True,
    write_shared_artifacts: bool = True,
) -> Dict[str, Any]:
    module_root = os.path.join(args.output_dir, "joint_residual_modules")
    os.makedirs(module_root, exist_ok=True)
    # Keep the legacy aggregate split artifact for downstream tools that were
    # originally written for discover_soft_residual_meta_modules.py.  Each
    # module also stores the same ids in its intervention-target artifact, but
    # exporting this file preserves the common module contract.
    if write_shared_artifacts:
        torch.save(
            {
                "probe_train_ids": [ids[index] for index in train_idx.tolist()],
                "module_selection_ids": [ids[index] for index in val_idx.tolist()],
                "module_evaluation_ids": [
                    ids[index]
                    for index in (test_idx if test_idx.numel() else val_idx).tolist()
                ],
            },
            os.path.join(module_root, "meta_module_signatures.pt"),
        )
    latent_source = (
        "main_z2" if args.second_stage == "main_direct" else "purified_meta"
    )
    if args.external_semantic_activation_dir:
        if "semantic_external" not in encoded:
            raise ValueError(
                "External semantic export requires cached semantic_external predictions."
            )
        # Posthoc semantic leakage probes use the same external representation
        # that supervised the semantic head; do not silently fall back to Z1.
        semantic_control_parts = [encoded["semantic_external"]]
    elif args.module_semantic_probe_profile == "aligned_training_head":
        aligned_pair = (
            args.semantic_candidate_source == "prev"
            and args.semantic_probe_mode == "frozen_aligned"
        ) or (
            args.semantic_candidate_source == "z1"
            and args.semantic_probe_mode == "staged_z1_aligned"
        )
        if not aligned_pair:
            raise ValueError(
                "aligned_training_head requires either prev/frozen_aligned or "
                "z1/staged_z1_aligned."
            )
        semantic_control_parts = [
            prev_features
            if args.semantic_candidate_source == "prev"
            else encoded["z1"]
        ]
    elif args.semantic_candidate_source == "prev":
        semantic_control_parts = [prev_features]
    elif args.semantic_candidate_source == "z1":
        semantic_control_parts = [encoded["z1"]]
    else:
        semantic_control_parts = [encoded["z1"], prev_features]
    refiner_semantic_controls = build_module_semantic_controls(
        semantic_control_parts,
        train_idx,
        args.module_refiner_semantic_projection_dim,
        args.seed + 4900,
    )
    selection_idx = val_idx
    evaluation_idx = test_idx if test_idx.numel() else val_idx
    aligned_semantic_probe_split_mse: Dict[str, float] = {}
    if args.module_semantic_probe_profile == "aligned_training_head":
        for split_name, split_index in (
            ("train", train_idx),
            ("selection", selection_idx),
            ("heldout", evaluation_idx),
        ):
            aligned_semantic_probe_split_mse[split_name] = float(
                F.mse_loss(
                    encoded["semantic_candidates"].index_select(
                        0, split_index
                    ),
                    candidate_values.index_select(0, split_index),
                ).item()
            )
    selected_module_indices = list(
        range(args.num_modules) if module_indices is None else module_indices
    )
    if not selected_module_indices:
        raise ValueError("Module export received an empty module shard.")
    if min(selected_module_indices) < 0 or max(selected_module_indices) >= args.num_modules:
        raise ValueError(
            f"Module shard {selected_module_indices} is outside 0..{args.num_modules - 1}."
        )
    shard_index = torch.tensor(selected_module_indices, dtype=torch.long)
    shard_directions = directions.index_select(0, shard_index)
    shard_module_scores = candidate_values @ shard_directions.t()
    score_column = {
        module: column for column, module in enumerate(selected_module_indices)
    }
    semantic_neighbor_graph = None
    if args.module_code_semantic_residualization == "crossfit_knn_rank":
        semantic_neighbor_graph = semantic_knn_indices(
            refiner_semantic_controls,
            train_idx,
            args.module_code_semantic_knn_k,
            args.module_code_semantic_knn_dim,
            device,
            args.encode_batch_size,
        )
    module_rows = []
    summaries = []
    for module in selected_module_indices:
        module_dir = os.path.join(module_root, f"module_{module:02d}")
        os.makedirs(module_dir, exist_ok=True)
        loading = directions[module]
        support_columns = (loading.abs() > 0).nonzero(as_tuple=False).flatten()
        support_neurons = candidate_neurons.index_select(0, support_columns)
        support_weights = loading.index_select(0, support_columns)
        module_target_scores = shard_module_scores[:, score_column[module]]
        residual_scalar_before_code_control = (
            module_target_scores - semantic_prediction[:, module]
        )
        semantic_code_prediction = torch.zeros_like(residual_scalar_before_code_control)
        semantic_baseline_addition = torch.zeros_like(
            residual_scalar_before_code_control
        )
        conditional_scale = torch.ones_like(residual_scalar_before_code_control)
        semantic_code_diagnostics: Dict[str, Any] = {
            "mode": "none",
            "removal_strength": 0.0,
        }
        semantic_code_teacher: Optional[Dict[str, Any]] = None
        residualization_mode = (
            "aligned_training_head"
            if args.module_semantic_probe_profile == "aligned_training_head"
            else args.module_code_semantic_residualization
        )
        if residualization_mode in {"crossfit_ridge", "crossfit_location_scale"}:
            (
                semantic_code_prediction,
                semantic_code_diagnostics,
                semantic_mean_teacher,
            ) = crossfit_semantic_module_prediction(
                refiner_semantic_controls,
                residual_scalar_before_code_control,
                train_idx,
                selection_idx,
                evaluation_idx,
                args.module_code_semantic_folds,
                args.module_code_semantic_ridge,
                args.seed + 4950 + module,
            )
            semantic_code_diagnostics["removal_strength"] = float(
                args.module_code_semantic_removal_strength
            )
            semantic_code_teacher = {"mean_state": semantic_mean_teacher}
        elif residualization_mode == "crossfit_strongest":
            (
                semantic_code_prediction,
                semantic_code_diagnostics,
                semantic_code_teacher,
            ) = crossfit_strongest_semantic_prediction(
                refiner_semantic_controls,
                residual_scalar_before_code_control,
                train_idx,
                selection_idx,
                evaluation_idx,
                families=("ridge", "mlp"),
                folds=args.module_code_semantic_folds,
                ridge=args.module_code_semantic_ridge,
                mlp_hidden_dim=args.module_code_semantic_mlp_hidden_dim,
                mlp_dropout=args.module_code_semantic_mlp_dropout,
                mlp_epochs=args.module_code_semantic_mlp_epochs,
                mlp_batch_size=args.batch_size,
                mlp_lr=args.module_code_semantic_mlp_lr,
                mlp_weight_decay=args.module_code_semantic_mlp_weight_decay,
                mlp_repeats=args.module_code_semantic_mlp_repeats,
                device=device,
                seed=args.seed + 4950 + module,
                progress_desc=f"Module {module} semantic residual teacher",
            )
            semantic_code_diagnostics["removal_strength"] = float(
                args.module_code_semantic_removal_strength
            )
        effective_semantic_removal_strength = (
            float(args.module_code_semantic_removal_strength)
            if residualization_mode
            in {
                "crossfit_ridge",
                "crossfit_strongest",
                "crossfit_location_scale",
            }
            else (1.0 if residualization_mode == "crossfit_knn_rank" else 0.0)
        )
        semantic_code_component = (
            effective_semantic_removal_strength * semantic_code_prediction
        )
        centered_residual_scalar = (
            residual_scalar_before_code_control - semantic_code_component
        )
        semantic_baseline_addition = semantic_code_component
        residual_scalar = centered_residual_scalar
        if residualization_mode == "crossfit_location_scale":
            log_variance_target = torch.log(centered_residual_scalar.square() + 1e-4)
            (
                log_variance_prediction,
                scale_diagnostics,
                semantic_scale_teacher,
            ) = crossfit_semantic_module_prediction(
                refiner_semantic_controls,
                log_variance_target,
                train_idx,
                selection_idx,
                evaluation_idx,
                args.module_code_semantic_folds,
                args.module_code_semantic_ridge,
                args.seed + 4975 + module,
            )
            unconditional_scale = centered_residual_scalar.index_select(
                0, train_idx
            ).std(unbiased=False).clamp_min(1e-5)
            conditional_scale = torch.exp(0.5 * log_variance_prediction).clamp(
                min=float(args.module_code_semantic_scale_min * unconditional_scale),
                max=float(args.module_code_semantic_scale_max * unconditional_scale),
            )
            train_standardized = centered_residual_scalar.index_select(
                0, train_idx
            ) / conditional_scale.index_select(0, train_idx)
            calibration = train_standardized.square().mean().sqrt().clamp_min(1e-5)
            conditional_scale = conditional_scale * calibration
            residual_scalar = centered_residual_scalar / conditional_scale.clamp_min(1e-6)
            semantic_code_diagnostics.update(
                {
                    "mode": "crossfit_location_scale",
                    "scale_train_oof_r2": scale_diagnostics["train_oof"]["r2"],
                    "scale_selection_r2": scale_diagnostics["selection"]["r2"],
                    "scale_evaluation_r2": scale_diagnostics["evaluation"]["r2"],
                    "conditional_scale_mean": float(conditional_scale.mean().item()),
                    "conditional_scale_std": float(
                        conditional_scale.std(unbiased=False).item()
                    ),
                }
            )
            semantic_code_teacher["log_variance_state"] = semantic_scale_teacher
        elif residualization_mode == "crossfit_knn_rank":
            if semantic_neighbor_graph is None:
                raise RuntimeError("Conditional-rank residualization lacks semantic neighbors")
            residual_scalar, semantic_baseline_addition, rank_diagnostics = (
                conditional_rank_code(
                    residual_scalar_before_code_control,
                    semantic_neighbor_graph,
                )
            )
            semantic_code_prediction = semantic_baseline_addition
            semantic_code_diagnostics = {
                "mode": "crossfit_knn_rank",
                "removal_strength": 1.0,
                "neighbors": int(semantic_neighbor_graph.size(1)),
                "semantic_dimensions": int(args.module_code_semantic_knn_dim),
                **rank_diagnostics,
                "train_neighbor_mean": regression_diagnostics(
                    residual_scalar_before_code_control.index_select(0, train_idx),
                    semantic_baseline_addition.index_select(0, train_idx),
                ),
                "selection_neighbor_mean": regression_diagnostics(
                    residual_scalar_before_code_control.index_select(0, selection_idx),
                    semantic_baseline_addition.index_select(0, selection_idx),
                ),
                "evaluation_neighbor_mean": regression_diagnostics(
                    residual_scalar_before_code_control.index_select(0, evaluation_idx),
                    semantic_baseline_addition.index_select(0, evaluation_idx),
                ),
            }
        train_before = residual_scalar_before_code_control.index_select(0, train_idx)
        train_after = residual_scalar.index_select(0, train_idx)
        semantic_code_diagnostics["train_remaining_variance_ratio"] = float(
            train_after.var(unbiased=False).item()
            / max(float(train_before.var(unbiased=False).item()), 1e-12)
        )
        if residualization_mode == "aligned_training_head":
            semantic_code_diagnostics.update(
                {
                    "mode": "aligned_training_head",
                    "semantic_source": args.semantic_candidate_source,
                    "additional_posthoc_subtraction": False,
                    "note": (
                        "The frozen training semantic teacher is the held-out RF "
                        "baseline; no second probe redefines its residual."
                    ),
                }
            )
        if residualization_mode != "none":
            diagnostic_r2 = semantic_code_diagnostics.get(
                "train_oof",
                semantic_code_diagnostics.get("train_neighbor_mean", {}),
            ).get("r2", math.nan)
            selection_r2 = semantic_code_diagnostics.get(
                "selection",
                semantic_code_diagnostics.get("selection_neighbor_mean", {}),
            ).get("r2", math.nan)
            evaluation_r2 = semantic_code_diagnostics.get(
                "evaluation",
                semantic_code_diagnostics.get("evaluation_neighbor_mean", {}),
            ).get("r2", math.nan)
            print(
                f"[joint-v2][module {module}] semantic code subtraction "
                f"mode={residualization_mode} train_r2={diagnostic_r2:.4f} "
                f"selection_r2={selection_r2:.4f} heldout_r2={evaluation_r2:.4f} "
                f"remaining_var={semantic_code_diagnostics['train_remaining_variance_ratio']:.4f}",
                flush=True,
            )
        strengthened_semantic_prediction = (
            semantic_prediction[:, module] + semantic_baseline_addition
        )
        if args.module_contribution_mode == "conditional_incremental":
            if args.module_refiner_code_mode != "task_aligned":
                raise ValueError(
                    "--module-contribution-mode conditional_incremental requires "
                    "--module-refiner-code-mode task_aligned."
                )
            refiner_target_scalar = residual_scalar_before_code_control
        else:
            refiner_target_scalar = residual_scalar
        base_delta = (
            refiner_target_scalar.view(-1, 1)
            * support_weights.view(1, -1)
        )
        refiner, refiner_training = train_module_refiner(
            encoded["meta"],
            base_delta,
            refiner_semantic_controls,
            support_weights,
            train_idx,
            val_idx,
            args,
            device,
            args.seed + 5000 + module,
        )
        raw_codes, predicted_delta = refiner_outputs(
            refiner, encoded["meta"], device, args.encode_batch_size
        )
        denominator = support_weights.square().sum().clamp_min(1e-8)
        raw_predicted_code_scalar = (
            predicted_delta * support_weights.view(1, -1)
        ).sum(dim=1) / denominator
        code_affine_scale = torch.tensor(1.0)
        code_affine_bias = torch.tensor(0.0)
        contribution_calibration: Dict[str, Any] = {
            "mode": "identity",
            "scale": 1.0,
            "bias": 0.0,
        }
        if args.module_contribution_mode == "conditional_incremental":
            (
                predicted_residual_scalar,
                code_affine_scale,
                contribution_calibration,
            ) = fit_conditional_incremental_calibration(
                raw_predicted_code_scalar,
                residual_scalar,
                train_idx,
                args.module_conditional_calibration_ridge,
            )
            code_affine_bias = torch.tensor(
                float(contribution_calibration["bias"])
            )
            codes = (
                raw_codes * code_affine_scale
                + code_affine_bias
            )
        elif residualization_mode == "crossfit_location_scale":
            codes = raw_codes
            predicted_residual_scalar = (
                raw_predicted_code_scalar * conditional_scale
            )
        elif residualization_mode == "crossfit_knn_rank":
            codes = raw_codes
            if semantic_neighbor_graph is None:
                raise RuntimeError("Conditional-rank decoding lacks semantic neighbors")
            predicted_conditional_score = inverse_conditional_rank_code(
                raw_predicted_code_scalar,
                residual_scalar_before_code_control,
                semantic_neighbor_graph,
            )
            predicted_residual_scalar = (
                predicted_conditional_score - semantic_baseline_addition
            )
        else:
            codes = raw_codes
            predicted_residual_scalar = raw_predicted_code_scalar
        if args.module_semantic_probe_profile == "aligned_training_head":
            (
                _,
                continuous_semantic_diagnostics,
                _,
            ) = crossfit_strongest_semantic_prediction(
                refiner_semantic_controls,
                predicted_residual_scalar,
                train_idx,
                selection_idx,
                evaluation_idx,
                families=("mlp_deep",),
                folds=args.module_continuous_semantic_probe_folds,
                ridge=args.module_continuous_semantic_probe_ridge,
                mlp_hidden_dim=args.semantic_probe_hidden_dim,
                mlp_dropout=args.dropout,
                mlp_epochs=args.semantic_probe_epochs,
                mlp_batch_size=args.batch_size,
                mlp_lr=args.semantic_probe_lr,
                mlp_weight_decay=args.semantic_probe_weight_decay,
                mlp_repeats=1,
                device=device,
                seed=args.seed + 5750 + module,
                progress_desc=(
                    f"Module {module} aligned "
                    f"{args.semantic_candidate_source} leakage probe"
                ),
            )
        elif residualization_mode == "crossfit_strongest":
            (
                _,
                continuous_semantic_diagnostics,
                _,
            ) = crossfit_strongest_semantic_prediction(
                refiner_semantic_controls,
                predicted_residual_scalar,
                train_idx,
                selection_idx,
                evaluation_idx,
                families=("ridge", "mlp"),
                folds=args.module_continuous_semantic_probe_folds,
                ridge=args.module_continuous_semantic_probe_ridge,
                mlp_hidden_dim=args.module_code_semantic_mlp_hidden_dim,
                mlp_dropout=args.module_code_semantic_mlp_dropout,
                mlp_epochs=args.module_code_semantic_mlp_epochs,
                mlp_batch_size=args.batch_size,
                mlp_lr=args.module_code_semantic_mlp_lr,
                mlp_weight_decay=args.module_code_semantic_mlp_weight_decay,
                mlp_repeats=args.module_code_semantic_mlp_repeats,
                device=device,
                seed=args.seed + 5750 + module,
                progress_desc=f"Module {module} continuous leakage probe",
            )
        else:
            (
                _,
                continuous_semantic_diagnostics,
                _,
            ) = crossfit_semantic_module_prediction(
                refiner_semantic_controls,
                predicted_residual_scalar,
                train_idx,
                selection_idx,
                evaluation_idx,
                args.module_continuous_semantic_probe_folds,
                args.module_continuous_semantic_probe_ridge,
                args.seed + 5750 + module,
            )
        train_code, transform = residual_lib.fit_transform_state(
            codes.index_select(0, train_idx),
            min(args.module_code_dim, max(1, codes.size(1))),
            args.clip_quantile,
            False,
        )
        selection_code = residual_lib.apply_transform(codes.index_select(0, selection_idx), transform)
        eval_code = residual_lib.apply_transform(codes.index_select(0, evaluation_idx), transform)
        raw_train_code = codes.index_select(0, train_idx)
        raw_selection_code = codes.index_select(0, selection_idx)
        raw_eval_code = codes.index_select(0, evaluation_idx)
        state_structure = module_state_structure_diagnostics(
            raw_train_code,
            raw_selection_code,
            raw_eval_code,
            args,
            args.seed + 5900 + module,
        )
        state_structure["diagnostic_feature_source"] = (
            "raw_refiner_code_before_cluster_transform"
        )
        train_labels, centroids, inertia = run_kmeans(
            train_code,
            args.num_clusters,
            args.kmeans_iters,
            args.kmeans_restarts,
            args.seed + 6000 + module,
        )
        selection_labels = residual_lib.nearest_labels(selection_code, centroids)
        eval_labels = residual_lib.nearest_labels(eval_code, centroids)
        counts = torch.bincount(eval_labels, minlength=args.num_clusters)
        if args.semantic_candidate_source == "prev":
            semantic_train_features = prev_features.index_select(0, train_idx)
            semantic_selection_features = prev_features.index_select(
                0, selection_idx
            )
            semantic_eval_features = prev_features.index_select(
                0, evaluation_idx
            )
        elif args.semantic_candidate_source == "z1":
            semantic_train_features = encoded["z1"].index_select(0, train_idx)
            semantic_selection_features = encoded["z1"].index_select(
                0, selection_idx
            )
            semantic_eval_features = encoded["z1"].index_select(
                0, evaluation_idx
            )
        else:
            semantic_train_features = torch.cat(
                [
                    encoded["z1"].index_select(0, train_idx),
                    prev_features.index_select(0, train_idx),
                ],
                dim=1,
            )
            semantic_selection_features = torch.cat(
                [
                    encoded["z1"].index_select(0, selection_idx),
                    prev_features.index_select(0, selection_idx),
                ],
                dim=1,
            )
            semantic_eval_features = torch.cat(
                [
                    encoded["z1"].index_select(0, evaluation_idx),
                    prev_features.index_select(0, evaluation_idx),
                ],
                dim=1,
            )
        semantic_selection_probe = residual_lib.semantic_cluster_probe(
            semantic_train_features,
            train_labels,
            semantic_selection_features,
            selection_labels,
            args.num_clusters,
            device,
            args.seed + 7000 + module,
            epochs=args.semantic_cluster_epochs,
            batch_size=args.batch_size,
        )
        # The same deterministic seed/hyperparameters retrain an identical
        # semantic probe; test remains untouched by module selection.
        semantic_probe = residual_lib.semantic_cluster_probe(
            semantic_train_features,
            train_labels,
            semantic_eval_features,
            eval_labels,
            args.num_clusters,
            device,
            args.seed + 7000 + module,
            epochs=args.semantic_cluster_epochs,
            batch_size=args.batch_size,
        )
        semantic_test = residual_lib.semantic_predictability_test(
            eval_labels,
            semantic_probe["predictions"],
            args.num_clusters,
            args.bootstrap_samples,
            args.permutation_tests,
            args.seed + 8000 + module,
            progress_desc=f"Module {module} semantic BAcc",
        )
        selection_target = module_target_scores.index_select(
            0, selection_idx
        ).view(-1, 1)
        selection_semantic = strengthened_semantic_prediction.index_select(
            0, selection_idx
        ).view(-1, 1)
        selection_residual = predicted_residual_scalar.index_select(
            0, selection_idx
        ).view(-1, 1)
        eval_target = module_target_scores.index_select(
            0, evaluation_idx
        ).view(-1, 1)
        eval_semantic = strengthened_semantic_prediction.index_select(
            0, evaluation_idx
        ).view(-1, 1)
        eval_residual = predicted_residual_scalar.index_select(
            0, evaluation_idx
        ).view(-1, 1)
        train_target = module_target_scores.index_select(0, train_idx).view(-1, 1)
        train_mean = train_target.mean(dim=0)
        selection_information = module_information_metrics(
            selection_target, selection_semantic, selection_residual, train_mean
        )
        information = module_information_metrics(
            eval_target, eval_semantic, eval_residual, train_mean
        )
        if args.meta_upper_bound_mode != "none":
            selection_upper_bound = meta_only_upper_bound_proxy(
                encoded["meta"].index_select(0, train_idx),
                train_target,
                encoded["meta"].index_select(0, selection_idx),
                selection_target,
                selection_semantic,
                args.meta_upper_bound_ridge,
                args.meta_upper_bound_dim,
            )
            heldout_upper_bound = meta_only_upper_bound_proxy(
                encoded["meta"].index_select(0, train_idx),
                train_target,
                encoded["meta"].index_select(0, evaluation_idx),
                eval_target,
                eval_semantic,
                args.meta_upper_bound_ridge,
                args.meta_upper_bound_dim,
            )
        else:
            selection_upper_bound = {"mode": "none"}
            heldout_upper_bound = {"mode": "none"}
        information_row = information.rows[0]
        information_row.update(
            {
                "meta_only_upper_fraction_total": heldout_upper_bound.get(
                    "meta_only_fraction_of_total", float("nan")
                ),
                "meta_only_upper_fraction_semantic_residual": heldout_upper_bound.get(
                    "meta_only_fraction_of_semantic_residual", float("nan")
                ),
                "semantic_residual_capacity_fraction_total": heldout_upper_bound.get(
                    "semantic_residual_capacity_fraction_of_total", float("nan")
                ),
            }
        )
        residual_gain_samples = (
            (eval_target - eval_semantic).square()
            - (eval_target - eval_semantic - eval_residual).square()
        ).flatten()
        generator = torch.Generator().manual_seed(args.seed + 8500 + module)
        gain_ci = residual_lib.bootstrap_ci(
            residual_gain_samples,
            args.bootstrap_samples,
            generator,
            progress_desc=f"Module {module} residual gain bootstrap",
        )
        gain_p = residual_lib.signflip_p(
            residual_gain_samples,
            args.permutation_tests,
            generator,
            progress_desc=f"Module {module} residual gain sign-flip",
        )
        information_row.update(
            {
                "residual_gain_bootstrap_ci95_low": gain_ci[0],
                "residual_gain_bootstrap_ci95_high": gain_ci[1],
                "residual_gain_signflip_one_sided_p": gain_p,
                "residual_gain_significant_positive_0p05": bool(
                    gain_ci[0] > 0.0 and gain_p < 0.05
                ),
            }
        )
        cluster_min_fraction = float(counts.min().item() / max(counts.sum().item(), 1))
        bacc = float(semantic_test["balanced_accuracy"])
        selection_bacc = float(semantic_selection_probe["balanced_accuracy"])
        selection_cluster_counts = torch.bincount(selection_labels, minlength=args.num_clusters)
        selection_cluster_min_fraction = float(
            selection_cluster_counts.min().item() / max(selection_cluster_counts.sum().item(), 1)
        )
        selection_information_gate_pass = (
            selection_information.residual_fraction >= args.export_min_residual_fraction
            and selection_information.residual_gain > 0.0
            and selection_information.total_gain > 0.0
        )
        selection_continuous_semantic_r2 = float(
            continuous_semantic_diagnostics.get(
                "max_probe_r2", {}
            ).get(
                "selection",
                continuous_semantic_diagnostics["selection"]["r2"],
            )
        )
        heldout_continuous_semantic_r2 = float(
            continuous_semantic_diagnostics.get(
                "max_probe_r2", {}
            ).get(
                "evaluation",
                continuous_semantic_diagnostics["evaluation"]["r2"],
            )
        )
        selection_continuous_leakage_gate_pass = (
            selection_continuous_semantic_r2
            <= args.module_selection_max_continuous_semantic_r2
        )
        selection_cluster_gate_pass = (
            (args.bacc_as_diagnostic or selection_bacc <= args.export_max_semantic_bacc)
            and selection_cluster_min_fraction >= 0.10
        )
        heldout_information_gate_pass = (
            information.residual_fraction >= args.export_min_residual_fraction
            and information.residual_gain > 0.0
            and information.total_gain > 0.0
            and bool(information_row["residual_gain_significant_positive_0p05"])
        )
        heldout_cluster_gate_pass = (
            (args.bacc_as_diagnostic or bacc <= args.export_max_semantic_bacc)
            and cluster_min_fraction >= 0.10
        )
        selection_discrete_supported = bool(
            state_structure["selection_discrete_state_supported"]
        )
        heldout_discrete_supported = bool(
            state_structure["evaluation_discrete_state_supported"]
        )
        if args.module_cluster_policy == "forced":
            selection_gate_pass = (
                selection_information_gate_pass and selection_cluster_gate_pass
            )
            heldout_gate_pass = (
                heldout_information_gate_pass and heldout_cluster_gate_pass
            )
        elif args.module_cluster_policy == "require_multimodal":
            selection_gate_pass = (
                selection_information_gate_pass
                and selection_cluster_gate_pass
                and selection_discrete_supported
            )
            heldout_gate_pass = (
                heldout_information_gate_pass
                and heldout_cluster_gate_pass
                and heldout_discrete_supported
            )
        else:
            # Diagnostic mode does not discard an informative continuous axis
            # merely because KMeans was not supported by the underlying density.
            selection_gate_pass = (
                selection_information_gate_pass
                and selection_continuous_leakage_gate_pass
            )
            heldout_gate_pass = heldout_information_gate_pass
        print(
            f"[joint-v2][module {module}] support={support_neurons.numel()} "
            f"heldout_RF={information.residual_fraction:.3f} "
            f"gain={information.residual_gain:.3f} BAcc={bacc:.3f} "
            f"continuous_sem_R2="
            f"{heldout_continuous_semantic_r2:.3f} "
            f"BIC_gain={state_structure['train_bic_gain_two_vs_one']:.1f} "
            f"separation={state_structure['component_separation']:.2f} "
            f"heldout_LL_gain="
            f"{state_structure['evaluation_log_likelihood_gain_per_sample']:.3f} "
            f"state={state_structure['recommended_interpretation']}",
            flush=True,
        )
        eval_predicted_code = predicted_residual_scalar.index_select(
            0, evaluation_idx
        )
        eval_true_code = residual_scalar.index_select(0, evaluation_idx)
        assignment_rows = []
        for row_index, source_index in enumerate(evaluation_idx.tolist()):
            assignment_rows.append(
                {
                    "id": ids[source_index],
                    "activation_cluster": int(eval_labels[row_index].item()),
                    "cluster": int(eval_labels[row_index].item()),
                    "module": module,
                    "module_score": float(eval_target[row_index].item()),
                    "predicted_residual_score": float(eval_residual[row_index].item()),
                    "predicted_code_score": float(eval_predicted_code[row_index].item()),
                    "true_code_target": float(eval_true_code[row_index].item()),
                    "model_feature_norm": float(eval_code[row_index].norm().item()),
                }
            )
        assignment_path = os.path.join(module_dir, "soft_residual_cluster_assignments.csv")
        write_csv(assignment_path, assignment_rows)
        feature_path = os.path.join(module_dir, "soft_residual_cluster_features.pt")
        torch.save(
            {
                "ids": [ids[index] for index in evaluation_idx.tolist()],
                "model_features": eval_code.float(),
                "raw_residual_features": base_delta.index_select(0, evaluation_idx).float(),
                "labels": eval_labels.long(),
                "selected_neurons": support_neurons.long(),
                "module_loading": support_weights.float(),
                "transform": transform,
                "summary": {
                    "source": "joint_v2_sparse_module_code",
                    "latent_source": latent_source,
                    "module": module,
                    "combination_level_training": True,
                },
            },
            feature_path,
        )
        target_path = os.path.join(module_dir, "soft_residual_intervention_targets.pt")
        selected_raw = raw_i.index_select(0, evaluation_idx).index_select(1, support_neurons)
        selected_threshold = threshold.index_select(0, support_neurons)
        torch.save(
            {
                "schema_version": 3,
                "selected_neurons": support_neurons.long(),
                "module_loading": support_weights.float(),
                "val_target_binary": (selected_raw >= selected_threshold.view(1, -1)).to(torch.uint8),
                "binary_activation_threshold": threshold.float(),
                "continuous_mean": target_mean.float(),
                "continuous_std": target_std.float(),
                "train_ids": [ids[index] for index in train_idx.tolist()],
                "selection_ids": [ids[index] for index in val_idx.tolist()],
                "val_ids": [ids[index] for index in evaluation_idx.tolist()],
                "test_ids": [ids[index] for index in evaluation_idx.tolist()],
                "evaluation_ids": [ids[index] for index in evaluation_idx.tolist()],
                "metadata": {
                    "activation_dir": args.activation_dir,
                    "model_label": args.model_label,
                    "prev_layer": config["prev_layer"],
                    "layer_i": config["layer_i"],
                    "next_layer": config["next_layer"],
                    "target_mode": "continuous_binary",
                    "selection_source": "joint_v2_sparse_module",
                    "latent_source": latent_source,
                    "module": module,
                    "combination_level_training": True,
                },
            },
            target_path,
        )
        refiner_path = os.path.join(module_dir, "module_refiner.pt")
        torch.save(
            {
                "refinement_mode": (
                    "task_aligned_module"
                    if args.module_refiner_code_mode == "task_aligned"
                    else "module"
                ),
                "input_dim": int(encoded["meta"].size(1)),
                "code_dim": int(codes.size(1)),
                "internal_code_dim": int(args.module_code_dim),
                "hidden_dim": int(args.module_refiner_hidden_dim),
                "output_dim": int(support_neurons.numel()),
                "dropout": float(args.dropout),
                "state_dict": cpu_state_dict(refiner),
                "latent_source": latent_source,
                "module_loading": support_weights.float(),
                "selected_neurons": support_neurons.long(),
                "training": refiner_training,
                "semantic_code_residualization": semantic_code_diagnostics,
                "module_contribution_mode": args.module_contribution_mode,
                "module_semantic_probe_profile": (
                    args.module_semantic_probe_profile
                ),
                "code_affine_scale": code_affine_scale.float(),
                "code_affine_bias": code_affine_bias.float(),
                "conditional_incremental_calibration": contribution_calibration,
            },
            refiner_path,
        )
        semantic_code_teacher_path = os.path.join(
            module_dir, "semantic_code_teacher.pt"
        )
        if semantic_code_teacher is not None:
            torch.save(
                {
                    "state_dict": semantic_code_teacher,
                    "diagnostics": semantic_code_diagnostics,
                    "semantic_projection_dim": int(
                        args.module_refiner_semantic_projection_dim
                    ),
                    "removal_strength": float(
                        args.module_code_semantic_removal_strength
                    ),
                    "note": (
                        "Training rows used out-of-fold predictions; this full-train "
                        "semantic teacher state was used only for "
                        "selection/evaluation rows."
                    ),
                },
                semantic_code_teacher_path,
            )
        cache_path = os.path.join(module_dir, "aligned_probe_cache.pt")
        torch.save(
            {
                "train": {
                    "ids": [ids[index] for index in train_idx.tolist()],
                    "meta": encoded["meta"].index_select(0, train_idx).float(),
                    "base_delta": base_delta.index_select(0, train_idx).float(),
                    "semantic_controls": refiner_semantic_controls.index_select(
                        0, train_idx
                    ).float(),
                },
                "selection": {
                    "ids": [ids[index] for index in val_idx.tolist()],
                    "meta": encoded["meta"].index_select(0, val_idx).float(),
                    "base_delta": base_delta.index_select(0, val_idx).float(),
                    "semantic_controls": refiner_semantic_controls.index_select(
                        0, val_idx
                    ).float(),
                },
                "evaluation": {
                    "ids": [ids[index] for index in evaluation_idx.tolist()],
                    "meta": encoded["meta"].index_select(0, evaluation_idx).float(),
                    "base_delta": base_delta.index_select(0, evaluation_idx).float(),
                    "semantic_controls": refiner_semantic_controls.index_select(
                        0, evaluation_idx
                    ).float(),
                },
                "module_loading": support_weights.float(),
                "selected_neurons": support_neurons.long(),
                "latent_source": latent_source,
                "semantic_controls_note": (
                    f"Compact {args.semantic_candidate_source} semantic controls "
                    "fitted on training rows and saved for conditional audits."
                ),
            },
            cache_path,
        )
        summary = {
            "source": "joint_v2_sparse_module",
            "latent_source": latent_source,
            "module_refiner_code_mode": args.module_refiner_code_mode,
            "module_code_semantic_residualization": semantic_code_diagnostics,
            "module_contribution_mode": args.module_contribution_mode,
            "module_semantic_probe_profile": (
                args.module_semantic_probe_profile
            ),
            "aligned_semantic_probe_split_mse": (
                aligned_semantic_probe_split_mse
            ),
            "conditional_incremental_calibration": contribution_calibration,
            "module": module,
            "combination_level_training": True,
            "module_member_neurons": support_neurons.tolist(),
            "module_loading": support_weights.tolist(),
            "selected_target_count": int(support_neurons.numel()),
            "cluster_counts": counts.tolist(),
            "cluster_min_fraction": cluster_min_fraction,
            "cluster_silhouette": float(centroid_silhouette_score(eval_code, eval_labels, centroids)),
            "cluster_inertia": float(inertia),
            "cluster_policy": args.module_cluster_policy,
            "cluster_interpretation": state_structure[
                "recommended_interpretation"
            ],
            "discrete_state_evidence": state_structure,
            "continuous_semantic_predictability": (
                continuous_semantic_diagnostics
            ),
            "heldout_residual_fraction_mean": information.residual_fraction,
            "heldout_residual_fraction_weighted_positive": information.residual_fraction,
            "semantic_cluster_mlp_bacc": bacc,
            "semantic_bacc_as_diagnostic": bool(args.bacc_as_diagnostic),
            "semantic_cluster_mlp_test": semantic_test,
            "selection_metrics": {
                "residual_fraction": selection_information.residual_fraction,
                "residual_gain_mse": selection_information.residual_gain,
                "total_gain_mse": selection_information.total_gain,
                "semantic_cluster_mlp_bacc": selection_bacc,
                "semantic_bacc_as_diagnostic": bool(args.bacc_as_diagnostic),
                "cluster_min_fraction": selection_cluster_min_fraction,
                "information_gate_pass": selection_information_gate_pass,
                "cluster_gate_pass": selection_cluster_gate_pass,
                "discrete_state_supported": selection_discrete_supported,
                "selection_hard_gate_pass": selection_gate_pass,
            },
            "information_metric": "continuous_module_mse_reduction",
            "information": {
                "total_gain_sum_nats": information.total_gain,
                "residual_gain_sum_nats": information.residual_gain,
                "semantic_gain_sum_nats": information.semantic_gain,
                "positive_joint_target_count": information.positive_modules,
                "legacy_nats_field_note": "Values are held-out MSE reductions for a continuous module state, not CE nats.",
                **information_row,
            },
            "meta_information_upper_bound": {
                "selection": selection_upper_bound,
                "heldout": heldout_upper_bound,
                "interpretation": (
                    "Z2-only target prediction is an optimistic upper proxy and "
                    "may contain semantic leakage; semantic residual capacity is "
                    "the model-free maximum remaining under the fixed semantic probe."
                ),
            },
            "training": {
                "any_selection_hard_gate_pass": selection_gate_pass,
                "best_selection": {"selection_hard_gate_pass": selection_gate_pass},
                "module_refiner": refiner_training,
            },
            "config": {
                "aligned_probe_cache": cache_path,
                "decoupler_dir": args.output_dir,
                "module_target": "semantic_residual_sparse_combination",
                "module_target_after_code_control": (
                    "conditional_incremental_over_aligned_frozen_"
                    f"{args.semantic_candidate_source}_baseline"
                    if (
                        args.module_contribution_mode
                        == "conditional_incremental"
                        and args.module_semantic_probe_profile
                        == "aligned_training_head"
                    )
                    else (
                    "conditional_incremental_over_strong_semantic_baseline"
                    if args.module_contribution_mode == "conditional_incremental"
                    else (
                        "semantic_control_residualized_sparse_combination"
                        if args.module_code_semantic_residualization != "none"
                        else "semantic_residual_sparse_combination"
                    )
                    )
                ),
                "latent_source": latent_source,
                "cluster_policy": args.module_cluster_policy,
            },
            "artifacts": {
                "assignments": assignment_path,
                "features": feature_path,
                "targets": target_path,
                "refiner": refiner_path,
                "aligned_probe_cache": cache_path,
                "semantic_code_teacher": (
                    semantic_code_teacher_path
                    if semantic_code_teacher is not None
                    else None
                ),
                "state_distribution_plot": (
                    os.path.join(module_dir, "module_state_distribution.png")
                    if not args.no_plots
                    else None
                ),
            },
        }
        write_json(os.path.join(module_dir, "module_summary.json"), summary)
        if not args.no_plots:
            save_module_plot(module_dir, [information_row], bacc)
            save_module_state_plot(
                module_dir,
                raw_train_code,
                raw_selection_code,
                raw_eval_code,
                state_structure,
            )
        cluster_metric_supported = (
            args.module_cluster_policy == "forced"
            or selection_discrete_supported
        )
        cluster_selection_term = (
            -args.module_selection_bacc_penalty
            * max(0.0, selection_bacc - args.module_selection_bacc_target)
            + 0.05
            * float(
                centroid_silhouette_score(
                    selection_code, selection_labels, centroids
                )
            )
            if cluster_metric_supported
            else -args.module_selection_continuous_semantic_r2_penalty
            * max(0.0, selection_continuous_semantic_r2)
        )
        selection_score = (
            args.module_selection_residual_gain_weight
            * selection_information.residual_gain
            + args.module_selection_total_gain_weight
            * selection_information.total_gain
            + args.module_selection_residual_fraction_weight
            * selection_information.residual_fraction
            + cluster_selection_term
        )
        module_rows.append(
            {
                "module": module,
                "selection_gate_pass": selection_gate_pass,
                "heldout_gate_pass": heldout_gate_pass,
                "selection_information_gate_pass": selection_information_gate_pass,
                "heldout_information_gate_pass": heldout_information_gate_pass,
                "selection_cluster_gate_pass": selection_cluster_gate_pass,
                "selection_continuous_leakage_gate_pass": (
                    selection_continuous_leakage_gate_pass
                ),
                "heldout_cluster_gate_pass": heldout_cluster_gate_pass,
                "selection_discrete_state_supported": selection_discrete_supported,
                "heldout_discrete_state_supported": heldout_discrete_supported,
                "cluster_interpretation": state_structure[
                    "recommended_interpretation"
                ],
                "selection_score": selection_score,
                "selection_residual_fraction": selection_information.residual_fraction,
                "selection_residual_gain_mse": selection_information.residual_gain,
                "selection_total_gain_mse": selection_information.total_gain,
                "selection_semantic_bacc": selection_bacc,
                "selection_cluster_min_fraction": selection_cluster_min_fraction,
                "heldout_residual_fraction": information.residual_fraction,
                "heldout_residual_gain_mse": information.residual_gain,
                "heldout_total_gain_mse": information.total_gain,
                "selection_meta_only_upper_fraction_total": selection_upper_bound.get(
                    "meta_only_fraction_of_total", float("nan")
                ),
                "heldout_meta_only_upper_fraction_total": heldout_upper_bound.get(
                    "meta_only_fraction_of_total", float("nan")
                ),
                "heldout_meta_only_upper_fraction_semantic_residual": heldout_upper_bound.get(
                    "meta_only_fraction_of_semantic_residual", float("nan")
                ),
                "heldout_semantic_residual_capacity_fraction_total": heldout_upper_bound.get(
                    "semantic_residual_capacity_fraction_of_total", float("nan")
                ),
                "heldout_semantic_bacc": bacc,
                "heldout_cluster_min_fraction": cluster_min_fraction,
                "continuous_semantic_r2_train_oof": float(
                    continuous_semantic_diagnostics["train_oof"]["r2"]
                ),
                "continuous_semantic_r2_selection": (
                    selection_continuous_semantic_r2
                ),
                "continuous_semantic_r2_heldout": float(
                    heldout_continuous_semantic_r2
                ),
                "continuous_semantic_probe_family": (
                    continuous_semantic_diagnostics.get(
                        "selected_family", "ridge"
                    )
                ),
                "multimodality_bic_gain": float(
                    state_structure["train_bic_gain_two_vs_one"]
                ),
                "multimodality_component_separation": float(
                    state_structure["component_separation"]
                ),
                "multimodality_valley_ratio": float(
                    state_structure["valley_ratio"]
                ),
                "multimodality_selection_ll_gain": float(
                    state_structure[
                        "selection_log_likelihood_gain_per_sample"
                    ]
                ),
                "multimodality_heldout_ll_gain": float(
                    state_structure[
                        "evaluation_log_likelihood_gain_per_sample"
                    ]
                ),
                "support_count": int(support_neurons.numel()),
                "module_contribution_mode": args.module_contribution_mode,
                "module_semantic_probe_profile": (
                    args.module_semantic_probe_profile
                ),
                "conditional_calibration_scale": float(
                    contribution_calibration["scale"]
                ),
                "conditional_calibration_bias": float(
                    contribution_calibration["bias"]
                ),
                "conditional_calibration_train_relative_gain": float(
                    contribution_calibration.get("train_relative_gain", math.nan)
                ),
                "module_code_semantic_residualization": args.module_code_semantic_residualization,
                "module_code_semantic_removal_strength": effective_semantic_removal_strength,
                "semantic_code_train_oof_r2": float(
                    semantic_code_diagnostics.get(
                        "train_oof",
                        semantic_code_diagnostics.get("train_neighbor_mean", {}),
                    ).get("r2", math.nan)
                ),
                "semantic_code_selection_r2": float(
                    semantic_code_diagnostics.get(
                        "selection",
                        semantic_code_diagnostics.get("selection_neighbor_mean", {}),
                    ).get("r2", math.nan)
                ),
                "semantic_code_evaluation_r2": float(
                    semantic_code_diagnostics.get(
                        "evaluation",
                        semantic_code_diagnostics.get("evaluation_neighbor_mean", {}),
                    ).get("r2", math.nan)
                ),
                "module_dir": module_dir,
            }
        )
        summaries.append(summary)

    module_rows.sort(key=lambda row: int(row["module"]))
    if not finalize:
        return {
            "module_root": module_root,
            "module_rows": module_rows,
            "aligned_semantic_probe_split_mse": aligned_semantic_probe_split_mse,
        }
    return finalize_module_export(
        module_root,
        module_rows,
        aligned_semantic_probe_split_mse,
        args,
    )


def finalize_module_export(
    module_root: str,
    module_rows: Sequence[Dict[str, Any]],
    aligned_semantic_probe_split_mse: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    module_rows = sorted(module_rows, key=lambda row: int(row["module"]))
    expected = list(range(args.num_modules))
    actual = [int(row["module"]) for row in module_rows]
    if actual != expected:
        raise RuntimeError(
            f"Incomplete module export: expected {expected}, received {actual}."
        )
    feasible = [row for row in module_rows if row["selection_gate_pass"]]
    selected = max(feasible or module_rows, key=lambda row: row["selection_score"])
    selected_payload = {
        "selected_module": int(selected["module"]),
        "selected_module_dir": selected["module_dir"],
        "selection_gate_pass": bool(selected["selection_gate_pass"]),
        "fallback_used": not bool(selected["selection_gate_pass"]),
        "selection_score": float(selected["selection_score"]),
        "selection_residual_fraction": float(selected["selection_residual_fraction"]),
        "selection_semantic_bacc": float(selected["selection_semantic_bacc"]),
        "heldout_residual_fraction": float(selected["heldout_residual_fraction"]),
        "heldout_semantic_bacc": float(selected["heldout_semantic_bacc"]),
        "residual_fraction": float(selected["heldout_residual_fraction"]),
        "semantic_bacc": float(selected["heldout_semantic_bacc"]),
        "heldout_gate_pass": bool(selected["heldout_gate_pass"]),
        "cluster_policy": args.module_cluster_policy,
        "cluster_interpretation": selected["cluster_interpretation"],
        "selection_discrete_state_supported": bool(
            selected["selection_discrete_state_supported"]
        ),
        "heldout_discrete_state_supported": bool(
            selected["heldout_discrete_state_supported"]
        ),
        "continuous_semantic_r2_heldout": float(
            selected["continuous_semantic_r2_heldout"]
        ),
        "aligned_semantic_probe_split_mse": (
            aligned_semantic_probe_split_mse
        ),
        "evidence_tier": (
            (
                "strict_discrete_joint_v2"
                if selected["heldout_discrete_state_supported"]
                else "continuous_module_joint_v2"
            )
            if selected["selection_gate_pass"] and selected["heldout_gate_pass"]
            else "diagnostic_joint_v2_fallback"
        ),
    }
    write_csv(os.path.join(module_root, "module_comparison.csv"), module_rows)
    write_json(os.path.join(module_root, "selected_joint_module.json"), selected_payload)
    return {
        "module_root": module_root,
        "selected": selected_payload,
        "module_rows": module_rows,
        "aligned_semantic_probe_split_mse": (
            aligned_semantic_probe_split_mse
        ),
    }


def export_modules_parallel(
    ids: Sequence[str],
    raw_i: torch.Tensor,
    candidate_values: torch.Tensor,
    candidate_neurons: torch.Tensor,
    threshold: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    prev_features: torch.Tensor,
    encoded: Dict[str, torch.Tensor],
    semantic_prediction: torch.Tensor,
    directions: torch.Tensor,
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Refine independent frozen modules in same-GPU subprocess shards."""

    workers = min(max(int(args.module_refiner_workers), 1), int(args.num_modules))
    module_root = os.path.join(args.output_dir, "joint_residual_modules")
    worker_root = os.path.join(module_root, "worker_state")
    log_root = os.path.join(module_root, "worker_logs")
    os.makedirs(worker_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    torch.save(
        {
            "probe_train_ids": [ids[index] for index in train_idx.tolist()],
            "module_selection_ids": [ids[index] for index in val_idx.tolist()],
            "module_evaluation_ids": [
                ids[index]
                for index in (test_idx if test_idx.numel() else val_idx).tolist()
            ],
        },
        os.path.join(module_root, "meta_module_signatures.pt"),
    )
    cache_path = os.path.join(worker_root, "module_export_cache.pt")
    payload = {
        "ids": list(ids),
        "raw_i": raw_i.detach().cpu(),
        "candidate_values": candidate_values.detach().cpu(),
        "candidate_neurons": candidate_neurons.detach().cpu(),
        "threshold": threshold.detach().cpu(),
        "target_mean": target_mean.detach().cpu(),
        "target_std": target_std.detach().cpu(),
        "train_idx": train_idx.detach().cpu(),
        "val_idx": val_idx.detach().cpu(),
        "test_idx": test_idx.detach().cpu(),
        "prev_features": prev_features.detach().cpu(),
        "encoded": {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in encoded.items()
        },
        "semantic_prediction": semantic_prediction.detach().cpu(),
        "directions": directions.detach().cpu(),
        "args": vars(args),
        "config": config,
    }
    print(
        f"[module-workers] write mmap cache for {args.num_modules} modules: "
        f"{cache_path}",
        flush=True,
    )
    torch.save(payload, cache_path)

    shards = [list(range(worker, args.num_modules, workers)) for worker in range(workers)]
    processes: List[Tuple[int, subprocess.Popen, Any, str]] = []
    worker_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "export_joint_v2_module_worker.py",
    )
    for worker, shard in enumerate(shards):
        result_path = os.path.join(worker_root, f"worker_{worker:02d}.pt")
        log_path = os.path.join(log_root, f"worker_{worker:02d}.log")
        if os.path.exists(result_path):
            os.remove(result_path)
        command = [
            sys.executable,
            worker_script,
            "--cache",
            cache_path,
            "--result",
            result_path,
            "--worker-id",
            str(worker),
            "--module-indices",
            *[str(module) for module in shard],
        ]
        environment = os.environ.copy()
        # Sixteen GPU workers should not each spawn a full CPU BLAS pool.
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["NUMEXPR_NUM_THREADS"] = "1"
        environment["MPLCONFIGDIR"] = os.path.join(worker_root, f"mpl_{worker:02d}")
        os.makedirs(environment["MPLCONFIGDIR"], exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((worker, process, log_handle, result_path))
        print(
            f"[module-workers] start worker={worker:02d} pid={process.pid} "
            f"modules={','.join(map(str, shard))} log={log_path}",
            flush=True,
        )

    failures = []
    try:
        for worker, process, log_handle, result_path in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code != 0 or not os.path.isfile(result_path):
                failures.append((worker, return_code))
                print(
                    f"[module-workers] failed worker={worker:02d} exit={return_code}",
                    flush=True,
                )
            else:
                print(f"[module-workers] complete worker={worker:02d}", flush=True)
    except BaseException:
        for _, process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, log_handle, _ in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not log_handle.closed:
                log_handle.close()
        raise
    if failures:
        raise RuntimeError(
            "Module-refiner workers failed: "
            + ", ".join(f"worker={worker} exit={code}" for worker, code in failures)
            + f". Inspect {log_root}."
        )

    module_rows: List[Dict[str, Any]] = []
    aligned_semantic_probe_split_mse: Optional[Dict[str, float]] = None
    for worker, _, _, result_path in processes:
        result = torch.load(result_path, map_location="cpu", weights_only=False)
        module_rows.extend(result["module_rows"])
        current = result["aligned_semantic_probe_split_mse"]
        if aligned_semantic_probe_split_mse is None:
            aligned_semantic_probe_split_mse = current
        elif current != aligned_semantic_probe_split_mse:
            raise RuntimeError(
                f"Worker {worker} returned inconsistent aligned semantic diagnostics."
            )
    try:
        os.remove(cache_path)
    except OSError:
        pass
    return finalize_module_export(
        module_root,
        module_rows,
        aligned_semantic_probe_split_mse or {},
        args,
    )


def save_direction_membership(
    path: str,
    directions: torch.Tensor,
    candidate_neurons: torch.Tensor,
) -> None:
    rows = []
    for module, direction in enumerate(directions):
        for column in (direction.abs() > 0).nonzero(as_tuple=False).flatten().tolist():
            rows.append(
                {
                    "module": module,
                    "candidate_column": column,
                    "neuron": int(candidate_neurons[column].item()),
                    "loading": float(direction[column].item()),
                    "loading_abs": float(direction[column].abs().item()),
                }
            )
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.direction_refresh_every <= 0:
        raise ValueError("Main direction refresh interval must be positive.")
    if args.second_stage == "purifier" and args.purifier_direction_refresh_every <= 0:
        raise ValueError("Direction refresh intervals must be positive.")
    if args.num_modules <= 1 or args.module_support <= 1:
        raise ValueError("Use at least two modules with at least two target neurons each.")
    if args.module_refiner_workers <= 0:
        raise ValueError("--module-refiner-workers must be positive.")
    if len(args.semantic_anchor_offsets or []) != 1:
        raise ValueError(
            "joint_v2 currently requires one semantic anchor offset so runtime normalization remains exact."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    device = choose_device(args.device)
    manifest = base.load_activation_manifest(args.activation_dir)
    prev_layer, layer_i, next_layer, layer_fracs, num_layers = base.resolve_layer_triplet(args, manifest)
    args.prev_layer = prev_layer
    args.layer_i = layer_i
    args.next_layer = next_layer
    ids_i, raw_i = base.load_layer(os.path.join(args.activation_dir, f"layer_{layer_i:03d}.pt"))
    ids_prev, raw_prev = base.load_layer(os.path.join(args.activation_dir, f"layer_{prev_layer:03d}.pt"))
    ids_next, raw_next = base.load_layer(os.path.join(args.activation_dir, f"layer_{next_layer:03d}.pt"))
    if ids_i != ids_prev or ids_i != ids_next:
        raise ValueError("Activation layer ids/order do not match.")
    train_idx, val_idx, test_idx, split_summary = deterministic_split(
        ids_i,
        args.val_ratio,
        args.test_ratio,
        args.split_seed,
        args.split_by_base_id,
        args.base_id_step_pattern,
    )
    x_prev, prev_mean, prev_std = standardize_from_train(raw_prev, train_idx)
    x_next, next_mean, next_std = standardize_from_train(raw_next, train_idx)
    x_i, target_mean, target_std = standardize_from_train(raw_i, train_idx)
    external_semantic_values: Optional[torch.Tensor] = None
    external_semantic_mean: Optional[torch.Tensor] = None
    external_semantic_std: Optional[torch.Tensor] = None
    external_semantic_info: Optional[Dict[str, Any]] = None
    if args.external_semantic_activation_dir:
        if args.external_semantic_layer is None:
            raise ValueError(
                "--external-semantic-layer is required with "
                "--external-semantic-activation-dir."
            )
        external_semantic_values, external_semantic_mean, external_semantic_std, external_semantic_info = load_external_semantic_features(
            args.external_semantic_activation_dir,
            args.external_semantic_layer,
            ids_i,
            train_idx,
            args.external_semantic_max_dim,
        )
        print(
            "[joint-v2] external semantic target "
            f"layer={args.external_semantic_layer} "
            f"dim={external_semantic_values.size(1)} "
            f"head={args.external_semantic_head_type} "
            f"depth={args.external_semantic_head_depth} "
            f"source={args.external_semantic_activation_dir}",
            flush=True,
        )
    candidate_neurons, threshold, candidate_rows, _ = select_candidate_pool(
        raw_i, train_idx, val_idx, args
    )
    candidate_values = x_i.index_select(1, candidate_neurons)
    if candidate_neurons.numel() < args.num_modules:
        raise ValueError("Candidate pool is smaller than the requested module count.")
    write_csv(os.path.join(args.output_dir, "joint_candidate_pool.csv"), candidate_rows)
    split_rows = []
    role_by_index = {}
    for role, index in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for value in index.tolist():
            role_by_index[value] = role
    for row_index, sample_id in enumerate(ids_i):
        split_rows.append(
            {
                "row_index": row_index,
                "id": sample_id,
                "base_id": split_base_id(sample_id, args.base_id_step_pattern),
                "split": role_by_index[row_index],
            }
        )
    write_csv(os.path.join(args.output_dir, "split_assignments.csv"), split_rows)
    write_json(os.path.join(args.output_dir, "split_manifest.json"), split_summary)
    config = {
        **vars(args),
        "prev_layer": prev_layer,
        "layer_i": layer_i,
        "next_layer": next_layer,
        "resolved_prev_layer": prev_layer,
        "resolved_next_layer": next_layer,
        "resolved_layer_fracs": layer_fracs,
        "num_model_layers": num_layers,
        "target_mode": "continuous_binary",
        "main_prediction_target": "semantic_residual_sparse_module",
        "joint_v2": True,
        "module_target_mode": "sparse_continuous_combination",
        "semantic_target_mode": (
            "external_model_activation"
            if args.external_semantic_activation_dir
            else "candidate_neuron_activation"
        ),
    }
    reuse_source_dir = (
        resolve_reuse_main_dir(args.reuse_main_dir)
        if args.reuse_main_dir
        else ""
    )
    defer_reuse_config_write = bool(
        reuse_source_dir
        and os.path.abspath(reuse_source_dir) == os.path.abspath(args.output_dir)
    )
    # When resuming export in place, keep the original first-stage contract on
    # disk until load_reused_main_stage has validated it.  Writing the new
    # config first would make the compatibility check compare the request with
    # itself instead of the frozen checkpoint that is being reused.
    if not defer_reuse_config_write:
        write_json(os.path.join(args.output_dir, "config.json"), config)
    normalization = {
        "model_label": args.model_label,
        "activation_manifest": manifest,
        "num_model_layers": num_layers,
        "layer_fractions": layer_fracs,
        "prev_layer": prev_layer,
        "layer_i": layer_i,
        "next_layer": next_layer,
        "prev_gap": layer_i - prev_layer,
        "next_gap": next_layer - layer_i,
        "prev_mean": prev_mean,
        "prev_std": prev_std,
        "semantic_anchors": {
            "mode": "single_layer_train_standardized",
            "offsets": [layer_i - prev_layer],
            "layers": [prev_layer],
            "weights": [1.0],
        },
        "next_mean": next_mean,
        "next_std": next_std,
        "target": {
            "mode": "continuous_binary",
            "train_target": "standardized_continuous",
            "eval_target": "hard_binary",
            "binary_quantile": args.binary_quantile,
            "mean": target_mean,
            "std": target_std,
            "threshold": threshold,
        },
        "external_semantic": (
            {
                **(external_semantic_info or {}),
                "mean": external_semantic_mean,
                "std": external_semantic_std,
            }
            if external_semantic_values is not None
            else None
        ),
        "split": split_summary,
        "main_prediction_target": "semantic_residual_sparse_module",
    }
    torch.save(normalization, os.path.join(args.output_dir, "normalization.pt"))

    print(
        f"[joint-v2] rows train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()} "
        f"layers={prev_layer},{layer_i},{next_layer} candidates={candidate_neurons.numel()} "
        f"modules={args.num_modules} support={args.module_support}",
        flush=True,
    )
    if args.reuse_main_dir:
        print(f"[joint-v2] reuse first-stage optimum: {args.reuse_main_dir}", flush=True)
        (
            model,
            main_semantic_decoder,
            main_residual_head,
            main_directions,
            main_summary,
        ) = load_reused_main_stage(
            args.reuse_main_dir,
            x_next,
            candidate_neurons,
            args,
            device,
        )
    else:
        (
            model,
            main_semantic_decoder,
            main_residual_head,
            main_directions,
            main_summary,
        ) = train_main_stage(
            x_prev,
            x_next,
            candidate_values,
            train_idx,
            val_idx,
            test_idx,
            args,
            device,
            external_semantic_values,
        )
    if defer_reuse_config_write:
        write_json(os.path.join(args.output_dir, "config.json"), config)
    save_direction_membership(
        os.path.join(args.output_dir, "main_module_membership.csv"),
        main_directions,
        candidate_neurons,
    )
    purifier: Optional[nn.Module] = None
    if args.second_stage == "purifier":
        (
            purifier,
            purifier_residual_head,
            directions,
            stage_summary,
            encoded,
        ) = train_purifier_stage(
            model,
            main_semantic_decoder,
            main_residual_head,
            x_prev,
            x_next,
            candidate_values,
            train_idx,
            val_idx,
            main_directions,
            args,
            device,
        )
        del purifier_residual_head
        save_direction_membership(
            os.path.join(args.output_dir, "purifier_module_membership.csv"),
            directions,
            candidate_neurons,
        )
    else:
        print(
            "[joint-v2] skip recursive purifier; optimize modules directly from main E2",
            flush=True,
        )
        directions = main_directions
        encoded, stage_summary = build_main_direct_encoding(
            model,
            main_semantic_decoder,
            main_residual_head,
            x_prev,
            x_next,
            candidate_values,
            train_idx,
            val_idx,
            directions,
            args,
            device,
        )
        save_direction_membership(
            os.path.join(args.output_dir, "main_direct_module_membership.csv"),
            directions,
            candidate_neurons,
        )
    matched_latent_probe_audit = None
    if args.run_matched_latent_probe_audit:
        matched_latent_probe_audit = matched_z1_z2_target_probe_audit(
            encoded,
            candidate_values,
            train_idx,
            val_idx,
            test_idx,
            args,
            device,
        )
    del main_semantic_decoder, main_residual_head
    # Module export only needs cached CPU encodings; release the two training
    # networks before fitting per-module refiners and semantic probes.
    model.cpu()
    if purifier is not None:
        purifier.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    export_arguments = (
        ids_i,
        raw_i,
        candidate_values,
        candidate_neurons,
        threshold,
        target_mean,
        target_std,
        train_idx,
        val_idx,
        test_idx,
        x_prev,
        encoded,
        encoded["semantic"],
        directions,
        args,
    )
    if args.module_refiner_workers > 1:
        print(
            f"[joint-v2] parallel module refinement workers="
            f"{min(args.module_refiner_workers, args.num_modules)} device={device}",
            flush=True,
        )
        module_export = export_modules_parallel(
            *export_arguments,
            config,
        )
    else:
        module_export = export_modules(
            *export_arguments,
            device,
            config,
        )
    analysis = {
        "method": (
            "joint_v2_main_direct_sparse_module_decoupling"
            if args.second_stage == "main_direct"
            else "joint_v2_two_stage_sparse_module_decoupling"
        ),
        "scientific_contract": {
            "candidate_selection": "variance, activation entropy, and cross-split stability only",
            "module_discovery": (
                "semantic-variance-penalized cross-fitted partial CCA with sparse support"
            ),
            "semantic_control": (
                "one frozen candidate decoder from "
                f"{args.semantic_candidate_source}; reused for training and "
                "held-out RF"
            ),
            "semantic_teacher_freeze_epoch": (
                0
                if args.semantic_probe_mode
                in {"frozen_aligned", "staged_z1_aligned"}
                else (
                    args.direction_freeze_epoch
                    if args.semantic_freeze_epoch < 0
                    else args.semantic_freeze_epoch
                )
            ),
            "semantic_probe_mode": args.semantic_probe_mode,
            "semantic_target_mode": (
                "external_model_activation"
                if args.external_semantic_activation_dir
                else "candidate_neuron_activation"
            ),
            "external_semantic_source": (
                {
                    "activation_dir": args.external_semantic_activation_dir,
                    "layer": args.external_semantic_layer,
                    "max_dim": args.external_semantic_max_dim,
                    "ridge_to_candidate_target": args.external_semantic_ridge,
                    "head_type": args.external_semantic_head_type,
                    "head_depth": args.external_semantic_head_depth,
                    "fit_split": "train_only",
                    "posthoc_control": "same external activation prediction",
                }
                if args.external_semantic_activation_dir
                else None
            ),
            "meta_information_evaluation": {
                "residual_fraction": (
                    "conservative incremental gain after the fixed semantic "
                    "control; it is a lower-bound-style estimate"
                ),
                "meta_only_upper_bound": (
                    "optimistic Z2-only ridge prediction of the target; it may "
                    "include semantic leakage and is not used as clean evidence"
                ),
                "semantic_residual_capacity": (
                    "remaining MSE after the semantic control divided by prior "
                    "MSE; a model-free maximum under the fixed semantic probe"
                ),
                "enabled_upper_bound_mode": args.meta_upper_bound_mode,
            },
            "semantic_bacc_absorption": {
                "enabled": args.lambda_semantic_bacc_absorption > 0.0,
                "lambda": args.lambda_semantic_bacc_absorption,
                "temperature": args.semantic_bacc_absorption_temperature,
                "start_epoch": args.semantic_bacc_absorption_start_epoch,
                "semantic_decoder_unfrozen": args.semantic_bacc_absorption_unfreeze_semantic,
                "mechanism": (
                    "soft continuous module-state boundary prediction by the "
                    "semantic branch; no gradient reversal or adversarial noise"
                ),
            },
            "first_stage_target": (
                "E2 predicts candidate-module state unexplained by the frozen semantic control"
            ),
            "second_stage": args.second_stage,
            "downstream_latent_source": (
                "main_z2" if args.second_stage == "main_direct" else "purified_meta"
            ),
            "purifier_target": (
                "skipped; module refinement consumes first-stage E2 and predicts the "
                "fixed-control residual module state"
                if args.second_stage == "main_direct"
                else "purified meta predicts the same fixed-control residual module state"
            ),
            "purifier_direction_mode": args.purifier_direction_mode,
            "purifier_main_residual_distillation": (
                args.lambda_purifier_main_residual_distill
            ),
            "semantic_cluster_alignment": {
                "training_penalty": "soft score-cluster nonlinear semantic MMD",
                "constraint_policy": (
                    "activate invariance loss only while validation residual gain/RF "
                    "remain above configured floors"
                ),
                "checkpoint_proxy": (
                    "train-induced score labels predicted from semantics and evaluated "
                    "on a disjoint validation split"
                ),
                "main_weight": args.lambda_main_semantic_cluster_mmd,
                "main_checkpoint_weight": args.checkpoint_semantic_cluster_mmd_weight,
                "module_refiner_weight": args.lambda_module_refiner_semantic_cluster_mmd,
                "module_refiner_checkpoint_weight": args.module_refiner_checkpoint_mmd_weight,
                "information_gates": {
                    "main_min_residual_gain": args.checkpoint_min_residual_gain,
                    "main_min_residual_fraction": args.checkpoint_min_residual_fraction,
                    "refiner_min_relative_gain": args.module_refiner_min_relative_gain,
                },
            },
            "module_code_semantic_residualization": (
                args.module_code_semantic_residualization
            ),
            "module_semantic_probe_profile": (
                args.module_semantic_probe_profile
            ),
            "module_code_semantic_control_protocol": (
                "the exact frozen training semantic probe defines both "
                "the training residual target and held-out RF baseline; no "
                "posthoc probe subtraction is applied"
                if args.module_semantic_probe_profile
                == "aligned_training_head"
                else (
                "training OOF predictions plus a train-only full ridge model for "
                "selection/evaluation"
                if args.module_code_semantic_residualization == "crossfit_ridge"
                else (
                    "cross-fitted ridge and MLP teachers; the strongest "
                    "selection-split semantic predictor defines the residual"
                    if args.module_code_semantic_residualization
                    == "crossfit_strongest"
                    else (
                    "cross-fitted conditional location and scale normalization"
                    if args.module_code_semantic_residualization
                    == "crossfit_location_scale"
                    else (
                        "train-only semantic-neighbor conditional-rank normalization"
                        if args.module_code_semantic_residualization
                        == "crossfit_knn_rank"
                        else "disabled"
                    )
                    )
                )
                )
            ),
            "module_state_interpretation": {
                "policy": args.module_cluster_policy,
                "continuous_primary_metric": (
                    "cross-fitted semantic R2 for the task-aligned module code"
                ),
                "discrete_state_gate": (
                    "train-fitted one-vs-two Gaussian BIC, component separation, "
                    "density-valley depth, and held-out likelihood gain"
                ),
                "kmeans_role": (
                    "legacy causal artifact plus descriptive partition; interpreted "
                    "as a discrete state only when the mixture gate generalizes"
                ),
            },
            "evaluation_split_used_for_selection": False,
        },
        "layers": {"prev_layer": prev_layer, "layer_i": layer_i, "next_layer": next_layer},
        "split": split_summary,
        "candidate_count": int(candidate_neurons.numel()),
        "num_modules": args.num_modules,
        "module_support": args.module_support,
        "main": main_summary,
        "second_stage": stage_summary,
        "purifier": stage_summary if args.second_stage == "purifier" else None,
        "main_direct": stage_summary if args.second_stage == "main_direct" else None,
        "matched_z1_z2_target_probe_audit": matched_latent_probe_audit,
        "selected_module": module_export["selected"],
        "aligned_semantic_probe_split_mse": module_export[
            "aligned_semantic_probe_split_mse"
        ],
        "artifacts": {
            "best_model": "best_model.pt",
            "normalization": "normalization.pt",
            "purifier": (
                "e2_recursive_purifier/best_purifier.pt"
                if args.second_stage == "purifier"
                else None
            ),
            "candidate_pool": "joint_candidate_pool.csv",
            "external_semantic_probe_losses": (
                "external_semantic_probe_losses.csv"
                if args.external_semantic_activation_dir
                else None
            ),
            "main_membership": "main_module_membership.csv",
            "stage_membership": (
                "purifier_module_membership.csv"
                if args.second_stage == "purifier"
                else "main_direct_module_membership.csv"
            ),
            "module_root": "joint_residual_modules",
            "matched_z1_z2_target_probe_audit": (
                "matched_z1_z2_target_probe_audit.json"
                if args.run_matched_latent_probe_audit
                else None
            ),
        },
    }
    write_json(os.path.join(args.output_dir, "analysis_summary.json"), analysis)
    print(json.dumps(module_export["selected"], ensure_ascii=False, indent=2), flush=True)
    print(f"[joint-v2] complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
