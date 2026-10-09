#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

import _bootstrap  # noqa: F401  # Legacy direct-script compatibility.

from model_utils import (
    generation_eos_token_id,
    last_token_logits_kwargs,
    load_model_and_tokenizer,
)

from intervene_activations import (
    ActivationRecorder,
    GenerationContext,
    activation_delta_columns,
    batch_iter,
    build_generation_prompt,
    evaluate_generated_answer,
    get_decoder_layers,
    load_intervention_targets,
    model_input_device,
    normalize_number,
    parse_record_layers,
    read_jsonl,
    safe_torch_load,
    split_layer_output,
    summarize_accuracy_rows,
    summarize_mode,
    write_csv,
    write_jsonl,
)
from train_decoupler import Decoupler, E2RecursivePurifier
from refined_residual_runtime import (
    load_module_refiner,
    runtime_code,
)
from publication_plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    apply_nmi_style,
    cluster_color,
    metric_label,
    panel_label,
    save_figure,
    style_axis,
    wilson_interval,
)
from metric_profiles import family_for_metric, metrics_for_profile
from metacog.evaluation import compute_style_metrics, profile_names
from metacog.clustering import centroid_silhouette_score, run_kmeans
from metacog.statistics import bh_adjust, paired_bootstrap_and_signflip


CLUSTER_NAMES = "abcdefghijklmnopqrstuvwxyz"
LATENT_CLUSTER_SOURCES = {"e2_latent", "purified_meta", "purified_semantic"}
FIXED_CLUSTER_SOURCE = "soft_residual"


class GenerationProgressHeartbeat(LogitsProcessor):
    """Report progress from inside a long ``model.generate`` call.

    The outer batch progress bar cannot advance until generation returns.  This
    processor observes sequence length only and returns logits unchanged, so it
    provides a heartbeat without altering decoding.
    """

    def __init__(
        self,
        mode: str,
        batch_index: int,
        total_batches: int,
        batch_size: int,
        prompt_width: int,
        max_new_tokens: int,
        interval_seconds: float,
    ) -> None:
        self.mode = str(mode)
        self.batch_index = int(batch_index)
        self.total_batches = int(total_batches)
        self.batch_size = int(batch_size)
        self.prompt_width = int(prompt_width)
        self.max_new_tokens = int(max_new_tokens)
        self.interval_seconds = max(float(interval_seconds), 0.0)
        self.started_at = time.monotonic()
        self.last_reported_at = self.started_at
        self.last_reported_step = 0

    def _batch_checkpoint(self) -> bool:
        interval = max(1, int(math.ceil(self.total_batches / 20.0)))
        return (
            self.batch_index == 1
            or self.batch_index == self.total_batches
            or self.batch_index % interval == 0
        )

    def _write(self, step: int, status: str) -> None:
        now = time.monotonic()
        elapsed = max(now - self.started_at, 1e-9)
        sequence_tokens_per_second = (
            float(step * self.batch_size) / elapsed
        )
        tqdm.write(
            "[generate-progress] "
            f"mode={self.mode} batch={self.batch_index}/{self.total_batches} "
            f"batch_pct={100.0 * self.batch_index / max(self.total_batches, 1):.1f} "
            f"token_step={step}/{self.max_new_tokens} rows={self.batch_size} "
            f"elapsed={elapsed:.1f}s sequence_tok_s={sequence_tokens_per_second:.2f} "
            f"status={status}"
        )
        self.last_reported_at = now
        self.last_reported_step = int(step)

    def start(self) -> None:
        if not self._batch_checkpoint():
            return
        tqdm.write(
            "[generate-progress] "
            f"mode={self.mode} batch={self.batch_index}/{self.total_batches} "
            f"batch_pct={100.0 * self.batch_index / max(self.total_batches, 1):.1f} "
            f"token_step=0/{self.max_new_tokens} rows={self.batch_size} "
            "status=started"
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        step = max(1, int(input_ids.size(1)) - self.prompt_width + 1)
        now = time.monotonic()
        if now - self.last_reported_at >= self.interval_seconds:
            self._write(step, "running")
        return scores

    def finalize(self, generated_steps: int) -> None:
        generated_steps = max(int(generated_steps), 0)
        elapsed = time.monotonic() - self.started_at
        if generated_steps != self.last_reported_step and (
            self._batch_checkpoint() or elapsed >= self.interval_seconds
        ):
            self._write(generated_steps, "complete")


class GeneratedTokenLogprobRecorder(LogitsProcessor):
    """Accumulate selected-token log probabilities without retaining all logits."""

    def __init__(self, batch_size: int, eos_token_ids: Sequence[int]):
        self.batch_size = int(batch_size)
        self.eos_token_ids = {int(value) for value in eos_token_ids}
        self.active = torch.ones(self.batch_size, dtype=torch.bool)
        self.values: List[List[float]] = [[] for _ in range(self.batch_size)]
        self.prefix_entropy_values: List[List[float]] = [
            [] for _ in range(self.batch_size)
        ]
        self.prefix_max_probability_values: List[List[float]] = [
            [] for _ in range(self.batch_size)
        ]
        self.prefix_logit_margin_values: List[List[float]] = [
            [] for _ in range(self.batch_size)
        ]
        self.previous_scores: Optional[torch.Tensor] = None

    def _record(self, token_ids: torch.Tensor) -> None:
        if self.previous_scores is None:
            return
        token_ids = token_ids.to(self.previous_scores.device).long()
        scores = self.previous_scores.float()
        log_normalizer = torch.logsumexp(scores, dim=1)
        selected = scores.gather(1, token_ids.view(-1, 1)).squeeze(1)
        log_probs = selected - log_normalizer
        active = self.active.to(log_probs.device)
        record_prefix_distribution = any(
            bool(self.active[index])
            and len(self.prefix_entropy_values[index]) < 16
            for index in range(self.batch_size)
        )
        if record_prefix_distribution:
            all_log_probs = scores - log_normalizer.view(-1, 1)
            probabilities = all_log_probs.exp()
            entropy = -(probabilities * all_log_probs).sum(dim=1)
            top_probabilities, _ = torch.topk(probabilities, k=2, dim=1)
            top_logits, _ = torch.topk(scores, k=2, dim=1)
        for index in torch.nonzero(active, as_tuple=False).flatten().tolist():
            self.values[index].append(float(log_probs[index].item()))
            if (
                record_prefix_distribution
                and len(self.prefix_entropy_values[index]) < 16
            ):
                self.prefix_entropy_values[index].append(
                    float(entropy[index].item())
                )
                self.prefix_max_probability_values[index].append(
                    float(top_probabilities[index, 0].item())
                )
                self.prefix_logit_margin_values[index].append(
                    float((top_logits[index, 0] - top_logits[index, 1]).item())
                )
        if self.eos_token_ids:
            ended = torch.zeros_like(active)
            for eos_id in self.eos_token_ids:
                ended |= token_ids == int(eos_id)
            self.active &= (~ended.detach().cpu())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.previous_scores is not None:
            self._record(input_ids[:, -1])
        self.previous_scores = scores.detach()
        return scores

    def finalize(self, sequences: torch.Tensor) -> None:
        if self.previous_scores is not None and sequences.size(0) == self.batch_size:
            self._record(sequences[:, -1])
        self.previous_scores = None

    def summaries(self, checkpoints: Sequence[int]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for index, values in enumerate(self.values):
            total = float(sum(values))
            prefix_entropy = self.prefix_entropy_values[index]
            prefix_max_probability = self.prefix_max_probability_values[index]
            prefix_logit_margin = self.prefix_logit_margin_values[index]
            row: Dict[str, Any] = {
                "generated_logprob_token_count": len(values),
                "generated_cumulative_logprob": total,
                "generated_mean_logprob": total / len(values) if values else math.nan,
                "generated_min_logprob": min(values) if values else math.nan,
                "generated_token_logprobs_json": json.dumps(values),
                "generated_prefix16_entropy_mean": (
                    float(sum(prefix_entropy) / len(prefix_entropy))
                    if prefix_entropy
                    else math.nan
                ),
                "generated_prefix16_negative_entropy_mean": (
                    float(-sum(prefix_entropy) / len(prefix_entropy))
                    if prefix_entropy
                    else math.nan
                ),
                "generated_prefix16_max_probability_mean": (
                    float(sum(prefix_max_probability) / len(prefix_max_probability))
                    if prefix_max_probability
                    else math.nan
                ),
                "generated_prefix16_logit_margin_mean": (
                    float(sum(prefix_logit_margin) / len(prefix_logit_margin))
                    if prefix_logit_margin
                    else math.nan
                ),
                "generated_prefix16_distribution_token_count": len(prefix_entropy),
            }
            prefix = {
                str(int(step)): float(sum(values[: int(step)]))
                for step in checkpoints
                if int(step) > 0 and len(values) >= int(step)
            }
            row["generated_cumulative_logprob_prefix_json"] = json.dumps(prefix)
            output.append(row)
        return output


def describe_prompt_confidence(
    logits: torch.Tensor,
    tokenizer,
) -> List[Dict[str, Any]]:
    """Summarize prompt-end predictive concentration before generation.

    These are confidence proxies, not calibrated correctness probabilities.
    Calibration and correctness prediction are handled on held-out rows by the
    downstream proxy-confidence analysis.
    """
    logits = logits.detach().float().cpu()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1)
    normalized_entropy = entropy / max(math.log(max(probs.size(1), 2)), 1e-12)
    top_probs, top_ids = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
    top_logits = torch.gather(logits, 1, top_ids)
    output = []
    for index in range(logits.size(0)):
        top1_probability = float(top_probs[index, 0].item())
        top2_probability = (
            float(top_probs[index, 1].item()) if top_probs.size(1) > 1 else 0.0
        )
        top1_logit = float(top_logits[index, 0].item())
        top2_logit = (
            float(top_logits[index, 1].item())
            if top_logits.size(1) > 1
            else top1_logit
        )
        token_id = int(top_ids[index, 0].item())
        output.append(
            {
                "prompt_end_entropy": float(entropy[index].item()),
                "prompt_end_negative_entropy": float(-entropy[index].item()),
                "prompt_end_normalized_entropy": float(
                    normalized_entropy[index].item()
                ),
                "prompt_end_max_probability": top1_probability,
                "prompt_end_top1_top2_probability_margin": (
                    top1_probability - top2_probability
                ),
                "prompt_end_top1_top2_logit_margin": top1_logit - top2_logit,
                "prompt_end_argmax_id": token_id,
                "prompt_end_argmax_text": tokenizer.decode(
                    [token_id], skip_special_tokens=False
                ),
            }
        )
    return output


class MainZ2RuntimeAdapter(torch.nn.Module):
    """Expose first-stage Z2 through the purifier runtime interface.

    Main-direct modules are refined from the first-stage E2 code.  This adapter
    makes that code path explicit and reconstructs Z2 exactly, avoiding the
    lossy recursive-purifier round trip while preserving the intervention API.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.meta_dim = int(latent_dim)
        self.semantic_dim = int(latent_dim)
        self.register_parameter(
            "device_anchor",
            torch.nn.Parameter(torch.empty(0), requires_grad=False),
        )

    def forward(self, z2: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"semantic": torch.zeros_like(z2), "meta": z2}

    def reconstruct_z2(self, joint: torch.Tensor) -> torch.Tensor:
        return joint[..., -self.meta_dim :]


@dataclass
class ClusterResult:
    name: str
    feature_indices: torch.Tensor
    raw_features: torch.Tensor
    model_features: torch.Tensor
    labels: torch.Tensor
    raw_centroids: torch.Tensor
    model_centroids: torch.Tensor
    pca2: torch.Tensor
    inertia: float


@dataclass
class InterventionSpec:
    name: str
    cluster_result: ClusterResult
    source_cluster: int
    target_cluster: int
    alpha: float
    patch_mode: str
    control: str
    extreme_fraction: float = 0.1
    extreme_min_count: int = 8


class NextTokenAuditState:
    """Cache prompt-end baseline logits and audit intervention-induced movement."""

    def __init__(
        self,
        tokenizer,
        num_clusters: int,
        top_k: int,
        score_mode: str = "js_contribution",
        min_mean_probability: float = 0.0,
        min_selected_tokens: int = 0,
    ):
        self.tokenizer = tokenizer
        self.num_clusters = int(num_clusters)
        self.top_k = max(int(top_k), 0)
        self.score_mode = str(score_mode)
        self.min_mean_probability = max(float(min_mean_probability), 0.0)
        self.min_selected_tokens = max(int(min_selected_tokens), 0)
        self.baseline_logits: Dict[str, torch.Tensor] = {}
        self.cluster_counts = torch.zeros(self.num_clusters, dtype=torch.long)
        self.logit_sums: Optional[torch.Tensor] = None
        self.log_prob_sums: Optional[torch.Tensor] = None
        self.prob_sums: Optional[torch.Tensor] = None
        self.token_ids = torch.empty(0, dtype=torch.long)
        self.prototypes = torch.empty(self.num_clusters, 0, dtype=torch.float32)
        self.token_rows: List[Dict[str, Any]] = []
        self.external_prototypes = False
        self.external_prototype_file: Optional[str] = None
        self.discovery_cluster_counts = torch.zeros(
            self.num_clusters, dtype=torch.long
        )

    def load_prototypes(self, path: str) -> None:
        payload = safe_torch_load(path)
        token_ids = torch.as_tensor(payload.get("token_ids", [])).long().flatten()
        prototypes = torch.as_tensor(payload.get("prototypes", [])).float()
        if token_ids.numel() == 0:
            raise ValueError(f"External next-token prototype file has no token ids: {path}")
        if prototypes.ndim != 2 or prototypes.size(0) != self.num_clusters:
            raise ValueError(
                "External next-token prototypes have incompatible cluster shape: "
                f"{tuple(prototypes.shape)} vs {self.num_clusters} clusters."
            )
        if prototypes.size(1) != token_ids.numel():
            raise ValueError(
                "External next-token token_ids/prototypes are not aligned: "
                f"{token_ids.numel()} vs {prototypes.size(1)}."
            )
        prototype_space = str(payload.get("prototype_space", "log_probability"))
        if prototype_space != "log_probability":
            raise ValueError(
                f"Unsupported external prototype space: {prototype_space!r}."
            )
        counts = torch.as_tensor(
            payload.get("cluster_counts", torch.zeros(self.num_clusters))
        ).long().flatten()
        if counts.numel() != self.num_clusters:
            raise ValueError(
                f"External prototype cluster counts have size {counts.numel()}, "
                f"expected {self.num_clusters}."
            )
        self.token_ids = token_ids.cpu()
        self.prototypes = prototypes.cpu()
        self.discovery_cluster_counts = counts.cpu()
        self.token_rows = [dict(row) for row in payload.get("token_rows", [])]
        self.external_prototypes = True
        self.external_prototype_file = os.path.abspath(path)

    def record_baseline(
        self, ids: Sequence[str], labels: Sequence[int], logits: torch.Tensor
    ) -> None:
        logits_cpu = logits.detach().float().cpu()
        if self.external_prototypes:
            for row_index, (sample_id, label) in enumerate(zip(ids, labels)):
                cluster = int(label)
                if cluster < 0 or cluster >= self.num_clusters:
                    continue
                self.baseline_logits[str(sample_id)] = logits_cpu[row_index].to(
                    torch.float16
                )
                self.cluster_counts[cluster] += 1
            return
        log_probs_cpu = F.log_softmax(logits_cpu, dim=-1)
        probs_cpu = log_probs_cpu.exp()
        if self.logit_sums is None:
            self.logit_sums = torch.zeros(
                self.num_clusters, logits_cpu.size(1), dtype=torch.float32
            )
            self.log_prob_sums = torch.zeros_like(self.logit_sums)
            self.prob_sums = torch.zeros_like(self.logit_sums)
        assert self.log_prob_sums is not None and self.prob_sums is not None
        for row_index, (sample_id, label) in enumerate(zip(ids, labels)):
            cluster = int(label)
            if cluster < 0 or cluster >= self.num_clusters:
                continue
            self.baseline_logits[str(sample_id)] = logits_cpu[row_index].to(torch.float16)
            self.logit_sums[cluster].add_(logits_cpu[row_index])
            self.log_prob_sums[cluster].add_(log_probs_cpu[row_index])
            self.prob_sums[cluster].add_(probs_cpu[row_index])
            self.cluster_counts[cluster] += 1

    def finalize(self) -> Dict[str, Any]:
        if self.external_prototypes:
            return {
                "enabled": bool(self.token_ids.numel()),
                "prototype_source": "external_discovery_split",
                "external_prototype_file": self.external_prototype_file,
                "baseline_sample_count": len(self.baseline_logits),
                "evaluation_cluster_counts": self.cluster_counts.tolist(),
                "discovery_cluster_counts": self.discovery_cluster_counts.tolist(),
                "selected_token_count": int(self.token_ids.numel()),
                "prototype_space": "log_probability",
            }
        if (
            self.logit_sums is None
            or self.log_prob_sums is None
            or self.prob_sums is None
        ):
            return {"enabled": False, "reason": "no_baseline_logits"}
        denominators = self.cluster_counts.clamp_min(1).float().view(-1, 1)
        mean_logits = self.logit_sums / denominators
        mean_log_probs = self.log_prob_sums / denominators
        mean_probs = self.prob_sums / denominators
        if self.top_k == 0:
            return {
                "enabled": False,
                "reason": "top_k_is_zero",
                "baseline_sample_count": len(self.baseline_logits),
                "cluster_counts": self.cluster_counts.tolist(),
                "selected_token_count": 0,
            }
        logit_range = mean_logits.max(dim=0).values - mean_logits.min(dim=0).values
        mean_probability = mean_probs.mean(dim=0)
        probability_mixture = mean_probability.clamp_min(1e-12)
        js_contribution = (
            mean_probs
            * (
                mean_probs.clamp_min(1e-12).log()
                - probability_mixture.view(1, -1).log()
            )
        ).mean(dim=0)
        if self.score_mode == "js_contribution":
            score = js_contribution
        elif self.score_mode == "probability_weighted_logit":
            score = logit_range * mean_probability.sqrt()
        elif self.score_mode == "logit_range":
            score = logit_range
        else:
            raise ValueError(
                f"Unknown next-token discriminative score mode: {self.score_mode}"
            )
        raw_score = score
        finite_score = torch.nan_to_num(
            raw_score,
            nan=float("-inf"),
            posinf=1e30,
            neginf=float("-inf"),
        )
        eligible = mean_probability >= self.min_mean_probability
        eligible_count = int(eligible.sum().item())

        def ranked_candidates(probability_floor: float) -> List[int]:
            mask = mean_probability >= probability_floor
            ranked_score = finite_score.masked_fill(~mask, float("-inf"))
            count = int(mask.sum().item())
            candidate_count = min(
                count,
                max(self.top_k * 80, self.top_k + 1024, 4096),
            )
            if candidate_count <= 0:
                return []
            if not torch.isfinite(ranked_score).any():
                ranked_score = mean_probability.masked_fill(~mask, float("-inf"))
            return torch.topk(ranked_score, k=candidate_count).indices.tolist()

        special_ids = {int(value) for value in getattr(self.tokenizer, "all_special_ids", [])}
        selected: List[int] = []
        rows: List[Dict[str, Any]] = []
        selected_set = set()

        def append_visible(candidate_ids: Sequence[int], relaxed: bool) -> None:
            for token_id in candidate_ids:
                token_id = int(token_id)
                if token_id in selected_set or token_id in special_ids:
                    continue
                token_text = self.tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
                if not token_text.strip():
                    continue
                row: Dict[str, Any] = {
                    "rank": len(selected) + 1,
                    "token_id": token_id,
                    "token_text": token_text,
                    "discriminative_score": float(raw_score[token_id].item()),
                    "discriminative_score_mode": self.score_mode,
                    "discriminative_logit_score": float(logit_range[token_id].item()),
                    "js_contribution": float(js_contribution[token_id].item()),
                    "mean_probability": float(mean_probability[token_id].item()),
                    "passed_requested_probability_floor": bool(
                        mean_probability[token_id].item() >= self.min_mean_probability
                    ),
                    "probability_floor_fallback": bool(relaxed),
                }
                for cluster in range(self.num_clusters):
                    row[f"mean_logit_cluster_{cluster}"] = float(
                        mean_logits[cluster, token_id].item()
                    )
                    row[f"mean_log_prob_cluster_{cluster}"] = float(
                        mean_log_probs[cluster, token_id].item()
                    )
                    row[f"mean_prob_cluster_{cluster}"] = float(
                        mean_probs[cluster, token_id].item()
                    )
                rows.append(row)
                selected.append(token_id)
                selected_set.add(token_id)
                if len(selected) >= self.top_k:
                    return

        append_visible(ranked_candidates(self.min_mean_probability), relaxed=False)
        required = min(self.top_k, self.min_selected_tokens)
        probability_fallback_used = len(selected) < required
        if probability_fallback_used:
            # The requested floor can leave only whitespace or special tokens.
            # Backfill visible tokens using the same independent discovery split.
            append_visible(ranked_candidates(0.0), relaxed=True)
        self.token_ids = torch.tensor(selected, dtype=torch.long)
        self.prototypes = (
            mean_log_probs.index_select(1, self.token_ids)
            if self.token_ids.numel()
            else torch.empty(self.num_clusters, 0)
        )
        self.token_rows = rows
        return {
            "enabled": bool(self.token_ids.numel()),
            "baseline_sample_count": len(self.baseline_logits),
            "cluster_counts": (
                self.discovery_cluster_counts.tolist()
                if self.external_prototypes
                else self.cluster_counts.tolist()
            ),
            "evaluation_cluster_counts": self.cluster_counts.tolist(),
            "selected_token_count": int(self.token_ids.numel()),
            "blank_and_special_tokens_excluded": True,
            "score_mode": self.score_mode,
            "min_mean_probability": self.min_mean_probability,
            "min_selected_tokens": self.min_selected_tokens,
            "eligible_token_count": eligible_count,
            "probability_floor_fallback_used": probability_fallback_used,
            "selected_below_probability_floor": sum(
                not bool(row["passed_requested_probability_floor"]) for row in rows
            ),
            "prototype_space": "log_probability",
        }

    def describe_logits(self, logits: torch.Tensor, top_k: int) -> List[Dict[str, Any]]:
        logits_float = logits.detach().float()
        log_probs = F.log_softmax(logits_float, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)
        top_count = min(max(int(top_k), 1), logits_float.size(1))
        top_probs, top_ids = torch.topk(probs, k=top_count, dim=1)
        argmax = top_ids[:, 0]
        rows: List[Dict[str, Any]] = []
        for row_index in range(logits_float.size(0)):
            payload = []
            for rank in range(top_count):
                token_id = int(top_ids[row_index, rank].item())
                payload.append(
                    {
                        "token_id": token_id,
                        "token_text": self.tokenizer.decode(
                            [token_id], clean_up_tokenization_spaces=False
                        ),
                        "prob": float(top_probs[row_index, rank].item()),
                    }
                )
            rows.append(
                {
                    "next_token_argmax_id": int(argmax[row_index].item()),
                    "next_token_argmax_text": self.tokenizer.decode(
                        [int(argmax[row_index].item())],
                        clean_up_tokenization_spaces=False,
                    ),
                    "next_token_entropy": float(entropy[row_index].item()),
                    "next_token_max_probability": float(top_probs[row_index, 0].item()),
                    "next_token_top_probability_tokens": json.dumps(
                        payload, ensure_ascii=False
                    ),
                }
            )
        return rows

    def compare(
        self,
        ids: Sequence[str],
        mode_logits: torch.Tensor,
        source_cluster: int,
        target_cluster: int,
        allow_prototype_movement: bool,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        missing = [sample_id for sample_id in ids if str(sample_id) not in self.baseline_logits]
        if missing:
            raise ValueError(
                f"Missing cached baseline next-token logits for {len(missing)} ids. "
                f"Examples: {', '.join(map(str, missing[:5]))}"
            )
        device = mode_logits.device
        base_logits = torch.stack(
            [self.baseline_logits[str(sample_id)] for sample_id in ids]
        ).to(device=device, dtype=torch.float32)
        mode_logits_float = mode_logits.detach().float()
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        mode_log_probs = F.log_softmax(mode_logits_float, dim=-1)
        base_probs = base_log_probs.exp()
        mode_probs = mode_log_probs.exp()
        midpoint = 0.5 * (base_probs + mode_probs)
        midpoint_log = midpoint.clamp_min(1e-12).log()
        kl_base_mode = (base_probs * (base_log_probs - mode_log_probs)).sum(dim=1)
        kl_mode_base = (mode_probs * (mode_log_probs - base_log_probs)).sum(dim=1)
        js = 0.5 * (
            (base_probs * (base_log_probs - midpoint_log)).sum(dim=1)
            + (mode_probs * (mode_log_probs - midpoint_log)).sum(dim=1)
        )
        total_variation = 0.5 * (base_probs - mode_probs).abs().sum(dim=1)
        base_entropy = -(base_probs * base_log_probs).sum(dim=1)
        mode_entropy = -(mode_probs * mode_log_probs).sum(dim=1)
        base_top_prob, base_argmax = base_probs.max(dim=1)
        mode_top_prob, mode_argmax = mode_probs.max(dim=1)
        top_count = min(10, base_logits.size(1))
        base_top_ids = torch.topk(base_logits, k=top_count, dim=1).indices
        mode_top_ids = torch.topk(mode_logits_float, k=top_count, dim=1).indices
        top_overlap = (
            (base_top_ids.unsqueeze(2) == mode_top_ids.unsqueeze(1))
            .any(dim=2)
            .float()
            .mean(dim=1)
        )
        mode_descriptions = self.describe_logits(mode_logits_float, top_k)
        movement: Dict[str, torch.Tensor] = {}
        if allow_prototype_movement and self.token_ids.numel():
            token_ids = self.token_ids.to(device)
            prototypes = self.prototypes.to(device=device, dtype=torch.float32)
            base_selected = base_log_probs.index_select(1, token_ids)
            mode_selected = mode_log_probs.index_select(1, token_ids)
            source = prototypes[int(source_cluster)].view(1, -1)
            target = prototypes[int(target_cluster)].view(1, -1)
            if int(source_cluster) == int(target_cluster):
                before_margin = -(base_selected - target).norm(dim=1)
                after_margin = -(mode_selected - target).norm(dim=1)
                projection = torch.zeros_like(before_margin)
            else:
                before_margin = (base_selected - source).pow(2).sum(dim=1) - (
                    base_selected - target
                ).pow(2).sum(dim=1)
                after_margin = (mode_selected - source).pow(2).sum(dim=1) - (
                    mode_selected - target
                ).pow(2).sum(dim=1)
                direction = target - source
                projection = ((mode_selected - base_selected) * direction).sum(dim=1) / direction.norm(
                    dim=1
                ).clamp_min(1e-8)
            movement = {
                "next_token_auto_logit_margin_before": before_margin,
                "next_token_auto_logit_margin_after": after_margin,
                "next_token_auto_logit_margin_delta_toward_target": after_margin
                - before_margin,
                "next_token_auto_logit_moved_toward_target": after_margin
                > before_margin,
                "next_token_auto_logit_delta_projection_to_target": projection,
                "next_token_auto_logit_delta_l2": (
                    mode_selected - base_selected
                ).norm(dim=1),
            }
        output: List[Dict[str, Any]] = []
        for row_index, description in enumerate(mode_descriptions):
            entry = dict(description)
            entry.update(
                {
                    "next_token_base_argmax_id": int(base_argmax[row_index].item()),
                    "next_token_base_argmax_text": self.tokenizer.decode(
                        [int(base_argmax[row_index].item())],
                        clean_up_tokenization_spaces=False,
                    ),
                    "next_token_argmax_changed": bool(
                        base_argmax[row_index].item() != mode_argmax[row_index].item()
                    ),
                    "next_token_kl_base_to_mode": float(kl_base_mode[row_index].item()),
                    "next_token_kl_mode_to_base": float(kl_mode_base[row_index].item()),
                    "next_token_js_divergence": float(js[row_index].item()),
                    "next_token_total_variation": float(total_variation[row_index].item()),
                    "next_token_base_entropy": float(base_entropy[row_index].item()),
                    "next_token_delta_entropy": float(
                        (mode_entropy[row_index] - base_entropy[row_index]).item()
                    ),
                    "next_token_base_max_probability": float(
                        base_top_prob[row_index].item()
                    ),
                    "next_token_delta_max_probability": float(
                        (mode_top_prob[row_index] - base_top_prob[row_index]).item()
                    ),
                    "next_token_full_logit_delta_l2": float(
                        (mode_logits_float[row_index] - base_logits[row_index]).norm().item()
                    ),
                    "next_token_top10_overlap": float(top_overlap[row_index].item()),
                }
            )
            for name, values in movement.items():
                value = values[row_index]
                entry[name] = (
                    bool(value.item())
                    if value.dtype == torch.bool
                    else float(value.item())
                )
            output.append(entry)
        return output

    def save(self, output_dir: str) -> Dict[str, Any]:
        write_csv(
            os.path.join(output_dir, "next_token_discriminative_tokens.csv"),
            self.token_rows,
        )
        torch.save(
            {
                "token_ids": self.token_ids,
                "prototypes": self.prototypes,
                "cluster_counts": (
                    self.discovery_cluster_counts
                    if self.external_prototypes
                    else self.cluster_counts
                ),
                "prototype_space": "log_probability",
                "source_prototype_file": self.external_prototype_file,
                "token_rows": self.token_rows,
            },
            os.path.join(output_dir, "next_token_logit_prototypes.pt"),
        )
        return {
            "enabled": bool(self.token_ids.numel()),
            "baseline_sample_count": len(self.baseline_logits),
            "cluster_counts": self.cluster_counts.tolist(),
            "selected_token_count": int(self.token_ids.numel()),
            "score_mode": self.score_mode,
            "min_mean_probability": self.min_mean_probability,
            "prototype_space": "log_probability",
            "prototype_source": (
                "external_discovery_split"
                if self.external_prototypes
                else "current_baseline"
            ),
            "external_prototype_file": self.external_prototype_file,
            "tokens_file": "next_token_discriminative_tokens.csv",
            "prototypes_file": "next_token_logit_prototypes.pt",
        }


class ClusterPatchIntervention:
    def __init__(
        self,
        context: GenerationContext,
        layer_module: torch.nn.Module,
        feature_indices: torch.Tensor,
        alpha: float,
        apply_to_generated: bool,
        generated_patch_steps: int,
    ):
        self.context = context
        self.feature_indices = feature_indices.long()
        self.alpha = float(alpha)
        self.apply_to_generated = apply_to_generated
        self.generated_patch_steps = int(generated_patch_steps)
        self.handle = layer_module.register_forward_hook(self.hook)

    def close(self) -> None:
        self.handle.remove()

    def hook(self, _module, _inputs, output):
        hidden, rebuild = split_layer_output(output)
        if hidden.ndim != 3 or self.feature_indices.numel() == 0:
            return output
        seq_len = hidden.size(1)
        is_prefill = seq_len > 1
        if is_prefill:
            self.context.current_step = -1
            self.context.generated_step = 0
        else:
            self.context.current_step = int(self.context.generated_step)
            self.context.generated_step += 1
        should_patch_generated = (
            self.apply_to_generated
            and not is_prefill
            and (self.generated_patch_steps < 0 or self.context.current_step < self.generated_patch_steps)
        )
        if not is_prefill and not should_patch_generated:
            return output

        batch = hidden.size(0)
        apply_mask = self.context.apply_mask
        patch_values = self.context.patch_values
        if apply_mask is None or patch_values is None or apply_mask.numel() != batch:
            return output
        rows = torch.nonzero(apply_mask.bool(), as_tuple=False).flatten()
        if rows.numel() == 0:
            return output

        rows = rows.to(hidden.device)
        features = self.feature_indices.to(hidden.device)
        patch = patch_values.to(device=hidden.device, dtype=hidden.dtype)
        pos = hidden.size(1) - 1 if is_prefill else 0
        edited = hidden.clone()
        current = edited[rows, pos][:, features]
        target = patch[rows]
        replacement = current.mul(1.0 - self.alpha).add(target, alpha=self.alpha)
        edited[rows[:, None], pos, features[None, :]] = replacement
        return rebuild(edited)


class RefinedResidualCodeIntervention:
    """Edit a refined residual code with constrained hidden-space optimization."""

    def __init__(
        self,
        context: GenerationContext,
        layer_module: torch.nn.Module,
        decoupler: Decoupler,
        purifier: E2RecursivePurifier,
        normalization: Dict[str, Any],
        refiner: torch.nn.Module,
        code_centroids: torch.Tensor,
        source_cluster: int,
        target_cluster: int,
        alpha: float,
        code_intervention: str,
        max_delta_rel_norm: float,
        apply_to_generated: bool,
        generated_patch_steps: int,
        control: str,
        random_seed: int = 0,
        semantic_penalty: float = 0.0,
        z1_penalty: float = 0.0,
        max_semantic_relative_delta: float = 0.0,
        dose_reference: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
        code_reference_pool: Optional[torch.Tensor] = None,
        random_direction_mode: str = "empirical_pca",
        dose_match_tolerance: float = 0.25,
        dose_hidden_tolerance: float = 0.5,
        dose_min_direction_cosine: float = 0.5,
        direct_steps: int = 12,
        direct_step_scale: float = 1.5,
        direct_hidden_penalty: float = 0.05,
        dose_semantic_tolerance: float = 0.0,
        random_dose_mode: str = "strict_code",
        random_axis_penalty: float = 10.0,
        random_max_real_axis_cosine: float = 0.25,
        random_hidden_candidates: int = 8,
    ):
        self.context = context
        self.decoupler = decoupler
        self.purifier = purifier
        self.normalization = normalization
        self.refiner = refiner
        self.code_centroids = code_centroids.float()
        self.source_cluster = int(source_cluster)
        self.target_cluster = int(target_cluster)
        self.alpha = float(alpha)
        self.code_intervention = str(code_intervention)
        self.max_delta_rel_norm = float(max_delta_rel_norm)
        self.apply_to_generated = bool(apply_to_generated)
        self.generated_patch_steps = int(generated_patch_steps)
        self.control = str(control)
        self.random_seed = int(random_seed)
        self.semantic_penalty = max(float(semantic_penalty), 0.0)
        self.z1_penalty = max(float(z1_penalty), 0.0)
        self.max_semantic_relative_delta = max(
            float(max_semantic_relative_delta), 0.0
        )
        self.dose_reference = dose_reference
        self.random_direction_mode = str(random_direction_mode)
        self.dose_match_tolerance = max(float(dose_match_tolerance), 0.0)
        self.dose_hidden_tolerance = max(float(dose_hidden_tolerance), 0.0)
        self.dose_min_direction_cosine = float(dose_min_direction_cosine)
        self.direct_steps = max(int(direct_steps), 1)
        self.direct_step_scale = max(float(direct_step_scale), 1e-4)
        self.direct_hidden_penalty = max(float(direct_hidden_penalty), 0.0)
        self.dose_semantic_tolerance = max(float(dose_semantic_tolerance), 0.0)
        self.random_dose_mode = str(random_dose_mode)
        if self.random_dose_mode not in {"strict_code", "hidden_semantic"}:
            raise ValueError(
                "random_dose_mode must be 'strict_code' or 'hidden_semantic'"
            )
        self.random_axis_penalty = max(float(random_axis_penalty), 0.0)
        self.random_max_real_axis_cosine = float(random_max_real_axis_cosine)
        self.random_hidden_candidates = max(int(random_hidden_candidates), 1)
        self.random_code_direction: Optional[torch.Tensor] = None
        self.real_code_direction: Optional[torch.Tensor] = None
        self.hidden_delta_reference: Optional[Dict[Tuple[str, int], torch.Tensor]] = None
        if self.control in {"random_direction", "random_hidden_direction"}:
            centers = self.code_centroids.float().cpu()
            real_direction = centers[self.target_cluster] - centers[self.source_cluster]
            self.real_code_direction = real_direction.clone()
        if self.control == "random_direction":
            centers = self.code_centroids.float().cpu()
            real_direction = centers[self.target_cluster] - centers[self.source_cluster]
            generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
            random_direction = None
            if self.random_direction_mode == "empirical_pca" and code_reference_pool is not None:
                pool = torch.as_tensor(code_reference_pool).float().cpu()
                if pool.ndim == 2 and pool.size(0) >= 2 and pool.size(1) == real_direction.numel():
                    centered = pool - pool.mean(dim=0, keepdim=True)
                    _, singular_values, components = torch.linalg.svd(
                        centered, full_matrices=False
                    )
                    rank = int((singular_values > 1e-6).sum().item())
                    if rank > 0:
                        coefficients = torch.randn(
                            rank, generator=generator, dtype=torch.float32
                        ) * singular_values[:rank].clamp_min(1e-6)
                        random_direction = coefficients @ components[:rank]
            if random_direction is None:
                random_direction = torch.randn(
                    real_direction.shape, generator=generator, dtype=torch.float32
                )
            real_norm_sq = real_direction.pow(2).sum()
            if float(real_norm_sq.item()) > 1e-12:
                random_direction = random_direction - (
                    random_direction.dot(real_direction) / real_norm_sq
                ) * real_direction
            if float(random_direction.norm().item()) <= 1e-8:
                random_direction = torch.randn(
                    real_direction.shape, generator=generator, dtype=torch.float32
                )
                if float(real_norm_sq.item()) > 1e-12:
                    random_direction = random_direction - (
                        random_direction.dot(real_direction) / real_norm_sq
                    ) * real_direction
            random_norm = random_direction.norm().clamp_min(1e-8)
            target_norm = real_direction.norm().clamp_min(1e-8)
            self.random_code_direction = random_direction * (target_norm / random_norm)
        self.rows: List[Dict[str, Any]] = []
        # The real main intervention is cached in-memory as vectors and then
        # passed to the hidden-space random control. Vectors are never written
        # to CSV/JSONL; only scalar audit fields are serialized.
        self.hidden_delta_vectors: Dict[Tuple[str, int], torch.Tensor] = {}
        self.handle = layer_module.register_forward_hook(self.hook)

    def close(self) -> None:
        self.handle.remove()

    def _ensure_device(self, device: torch.device) -> None:
        for module in (self.decoupler, self.purifier, self.refiner):
            if next(module.parameters()).device != device:
                module.to(device)

    def _target_code(
        self,
        current_code: torch.Tensor,
        dose_magnitudes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        centers = self.code_centroids.to(current_code)
        source = centers[self.source_cluster].view(1, -1)
        target = centers[self.target_cluster].view(1, -1)
        if self.control == "random_direction":
            if self.random_code_direction is None:
                raise RuntimeError("random_direction control was not initialized")
            direction = self.random_code_direction.to(current_code).view(1, -1)
            if dose_magnitudes is not None:
                direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
                direction = direction * dose_magnitudes.view(-1, 1)
                return current_code + direction
            return current_code + self.alpha * direction
        if self.control == "random_hidden_direction":
            # The hidden-space random control is deliberately not assigned a
            # code target. Its code movement is an emergent diagnostic only.
            return current_code
        if self.source_cluster == self.target_cluster:
            # The within-cluster control receives a real perturbation while
            # remaining inside its original refined-code basin.
            direction = self.alpha * (target - current_code)
            if dose_magnitudes is not None:
                direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
                direction = direction * dose_magnitudes.view(-1, 1)
            return current_code + direction
        if self.code_intervention == "direction":
            return current_code + self.alpha * (target - source)
        if self.code_intervention == "centroid":
            return current_code.mul(1.0 - self.alpha).add(target, alpha=self.alpha)
        raise ValueError(f"Unknown refined-code intervention: {self.code_intervention}")

    def _sample_random_hidden_direction(
        self,
        x_raw: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        z1: torch.Tensor,
        semantic: torch.Tensor,
        reference_hidden_vectors: torch.Tensor,
        reference_hidden_relative: torch.Tensor,
        reference_semantic_relative: torch.Tensor,
        reference_accepted: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Construct a valid random control directly in hidden space.

        Each candidate is Gaussian, exactly orthogonal to the observed main
        intervention delta, and matched to its hidden norm. Among a small
        fixed candidate set, the candidate with the closest semantic drift is
        selected. No code-space inverse, logits, or downstream behavior enters
        this selection.
        """
        original = x_raw.detach().float()
        x_norm = original.norm(dim=1, keepdim=True).clamp_min(1e-6)
        batch_size, hidden_dim = original.shape
        generator = torch.Generator(device=original.device)
        generator.manual_seed(
            int(self.random_seed)
            + 1009 * max(int(self.context.current_step), 0)
        )
        main_delta = reference_hidden_vectors.to(original)
        main_norm_sq = main_delta.square().sum(dim=1, keepdim=True)
        target_abs_norm = reference_hidden_relative.view(-1, 1).to(original) * x_norm
        target_abs_norm = target_abs_norm.clamp_min(1e-8)
        target_semantic = reference_semantic_relative.view(-1, 1).to(original)

        best_score = torch.full(
            (batch_size,), float("inf"), device=original.device, dtype=torch.float32
        )
        best_replacement = original.clone()
        best_z1 = z1.detach().clone()
        best_semantic = semantic.detach().clone()
        best_delta = torch.zeros_like(original)
        best_iteration = torch.zeros(
            batch_size, dtype=torch.long, device=original.device
        )
        best_valid = torch.zeros(batch_size, dtype=torch.bool, device=original.device)

        for candidate_index in range(self.random_hidden_candidates):
            random_delta = torch.randn(
                (batch_size, hidden_dim),
                generator=generator,
                device=original.device,
                dtype=original.dtype,
            )
            valid_reference = reference_accepted & (main_norm_sq.squeeze(1) > 1e-12)
            projection = (
                (random_delta * main_delta).sum(dim=1, keepdim=True)
                / main_norm_sq.clamp_min(1e-12)
            )
            random_delta = random_delta - projection * main_delta
            random_delta = random_delta / random_delta.norm(dim=1, keepdim=True).clamp_min(1e-8)
            random_delta = random_delta * target_abs_norm
            candidate_raw = original + random_delta
            candidate_std = (candidate_raw - mean) / std
            candidate_z1 = self.decoupler.e1(candidate_std)
            candidate_z2 = self.decoupler.e2(candidate_std)
            candidate_semantic = self.purifier(candidate_z2)["semantic"]
            semantic_relative = (
                candidate_semantic - semantic
            ).norm(dim=1, keepdim=True) / semantic.norm(dim=1, keepdim=True).clamp_min(1e-6)
            score = (semantic_relative - target_semantic).abs().squeeze(1)
            score = torch.where(valid_reference, score, torch.full_like(score, float("inf")))
            improved = score < best_score
            if bool(improved.any().item()):
                mask = improved.view(-1, 1)
                best_score = torch.where(improved, score, best_score)
                best_replacement = torch.where(mask, candidate_raw, best_replacement)
                best_z1 = torch.where(mask, candidate_z1, best_z1)
                best_semantic = torch.where(mask, candidate_semantic, best_semantic)
                best_delta = torch.where(mask, random_delta, best_delta)
                best_iteration = torch.where(
                    improved,
                    torch.full_like(best_iteration, candidate_index + 1),
                    best_iteration,
                )
                best_valid |= improved

        hidden_relative = best_delta.norm(dim=1, keepdim=True) / x_norm
        return {
            "best_objective": best_score,
            "best_replacement": best_replacement,
            "best_code": self._runtime_code_from_hidden(best_replacement, mean, std),
            "best_z1_after": best_z1,
            "best_semantic_after": best_semantic,
            "best_hidden_relative_delta_raw": hidden_relative,
            "best_clip_scale": torch.ones_like(hidden_relative),
            "best_iteration": best_iteration,
            "best_scale": hidden_relative.squeeze(1),
            "best_requested_code": torch.zeros(
                (batch_size, self.code_centroids.size(1)),
                device=original.device,
                dtype=original.dtype,
            ),
            "best_objective_target_code": torch.zeros(
                (batch_size, self.code_centroids.size(1)),
                device=original.device,
                dtype=original.dtype,
            ),
            "best_accepted": best_valid,
            "best_main_hidden_cosine": F.cosine_similarity(
                best_delta, main_delta, dim=1, eps=1e-8
            ),
            "best_main_hidden_projection": (
                (best_delta * main_delta).sum(dim=1)
                / main_norm_sq.squeeze(1).clamp_min(1e-8)
            ),
        }

    def _runtime_code_from_hidden(
        self, hidden: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        normalized = (hidden - mean) / std
        return runtime_code(self.refiner, self.purifier(self.decoupler.e2(normalized))["meta"])

    def _objective_target_code(
        self,
        current_code: torch.Tensor,
        requested_code: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scientific target used to score an online code edit.

        Most interventions request the target directly. Calibrated mechanisms may
        request an amplified proxy while still being judged against the original
        target by overriding this method.
        """
        return requested_code

    def _candidate_code_eligibility(
        self,
        current_code: torch.Tensor,
        requested_code: torch.Tensor,
        objective_target_code: torch.Tensor,
        candidate_code: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones(
            candidate_code.size(0), dtype=torch.bool, device=candidate_code.device
        )

    def _direct_optimize_hidden(
        self,
        x_raw: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        current_code: torch.Tensor,
        z1: torch.Tensor,
        semantic: torch.Tensor,
        dose_matching: bool,
        dose_reference_accepted: torch.Tensor,
        dose_reference_code_delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Solve the code edit directly in hidden space.

        This bypasses the learned code decoder.  The per-sample update follows
        the gradient of code error while explicitly penalizing E1, purified
        semantic, and hidden-state drift.  Projection onto a relative-norm ball
        makes the delivered perturbation comparable across samples.
        """
        with torch.inference_mode(False), torch.enable_grad():
            # The surrounding generation path runs under inference_mode. Its
            # normalization tensors therefore cannot be saved by autograd even
            # after inference_mode is locally disabled. Materialize ordinary
            # tensors before constructing the differentiable objective.
            original = x_raw.detach().clone().float()
            mean_reference = mean.detach().clone().float()
            std_reference = std.detach().clone().float()
            current_reference = current_code.detach().clone().float()
            z1_reference = z1.detach().clone().float()
            semantic_reference = semantic.detach().clone().float()
            x_norm = original.norm(dim=1, keepdim=True).clamp_min(1e-6)
            requested_code = self._target_code(
                current_reference,
                dose_reference_code_delta if dose_matching else None,
            ).detach()
            objective_target = self._objective_target_code(
                current_reference, requested_code
            ).detach()
            objective_norm = (
                objective_target - current_reference
            ).norm(dim=1).clamp_min(1e-8)
            max_relative = (
                self.max_delta_rel_norm
                if self.max_delta_rel_norm > 0.0
                else 0.03
            )
            step_relative = (
                max_relative
                * self.direct_step_scale
                / max(self.direct_steps, 1)
            )
            delta = torch.zeros_like(original, requires_grad=True)
            batch_size = original.size(0)
            best_objective = torch.ones(
                batch_size, device=original.device, dtype=torch.float32
            )
            best_replacement = original.detach().clone()
            best_code = current_reference.detach().clone()
            best_z1 = z1_reference.detach().clone()
            best_semantic = semantic_reference.detach().clone()
            best_iteration = torch.zeros(
                batch_size, dtype=torch.long, device=original.device
            )
            best_accepted = torch.zeros(
                batch_size, dtype=torch.bool, device=original.device
            )

            for iteration in range(1, self.direct_steps + 2):
                candidate_raw = original + delta
                candidate_std = (
                    candidate_raw - mean_reference
                ) / std_reference
                candidate_z1 = self.decoupler.e1(candidate_std)
                candidate_z2 = self.decoupler.e2(candidate_std)
                candidate_purifier = self.purifier(candidate_z2)
                candidate_semantic = candidate_purifier["semantic"]
                candidate_code = runtime_code(
                    self.refiner,
                    candidate_purifier["meta"],
                )
                code_error = (
                    candidate_code - objective_target
                ).norm(dim=1) / objective_norm
                random_real_axis_projection = torch.zeros(
                    batch_size, device=candidate_code.device, dtype=candidate_code.dtype
                )
                if (
                    self.control == "random_direction"
                    and self.real_code_direction is not None
                ):
                    real_direction = self.real_code_direction.to(candidate_code)
                    real_norm = real_direction.norm().clamp_min(1e-8)
                    actual_delta = candidate_code - current_reference
                    random_real_axis_projection = (
                        actual_delta.matmul(real_direction)
                        / (real_norm * objective_norm).clamp_min(1e-8)
                    )
                z1_relative = (
                    candidate_z1 - z1_reference
                ).norm(dim=1) / z1_reference.norm(dim=1).clamp_min(1e-6)
                semantic_relative = (
                    candidate_semantic - semantic_reference
                ).norm(dim=1) / semantic_reference.norm(dim=1).clamp_min(1e-6)
                hidden_relative = delta.norm(dim=1) / x_norm.flatten()
                objective = (
                    code_error.square()
                    + self.semantic_penalty * semantic_relative.square()
                    + self.z1_penalty * z1_relative.square()
                    + self.direct_hidden_penalty * hidden_relative.square()
                    + (
                        self.random_axis_penalty
                        * random_real_axis_projection.square()
                        if self.control == "random_direction"
                        else 0.0
                    )
                )
                eligible = torch.isfinite(objective)
                eligible &= self._candidate_code_eligibility(
                    current_reference,
                    requested_code,
                    objective_target,
                    candidate_code,
                )
                if dose_matching:
                    eligible &= dose_reference_accepted
                if self.max_semantic_relative_delta > 0.0:
                    eligible &= (
                        semantic_relative
                        <= self.max_semantic_relative_delta
                    )
                improved = eligible & (objective < best_objective)
                if bool(improved.any().item()):
                    mask = improved.view(-1, 1)
                    best_objective = torch.where(
                        improved, objective.detach(), best_objective
                    )
                    best_replacement = torch.where(
                        mask, candidate_raw.detach(), best_replacement
                    )
                    best_code = torch.where(
                        mask, candidate_code.detach(), best_code
                    )
                    best_z1 = torch.where(
                        mask, candidate_z1.detach(), best_z1
                    )
                    best_semantic = torch.where(
                        mask, candidate_semantic.detach(), best_semantic
                    )
                    best_iteration = torch.where(
                        improved,
                        torch.full_like(best_iteration, iteration),
                        best_iteration,
                    )
                    best_accepted |= improved
                if iteration > self.direct_steps:
                    break
                gradient = torch.autograd.grad(
                    objective.sum(), delta, retain_graph=False
                )[0]
                normalized_gradient = gradient / gradient.norm(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
                delta = (
                    delta - step_relative * x_norm * normalized_gradient
                ).detach()
                radius = max_relative * x_norm
                delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-8)
                delta = delta * torch.minimum(
                    torch.ones_like(delta_norm), radius / delta_norm
                )
                delta.requires_grad_(True)

        hidden_relative = (
            best_replacement - original
        ).norm(dim=1, keepdim=True) / x_norm
        return {
            "best_objective": best_objective,
            "best_replacement": best_replacement,
            "best_code": best_code,
            "best_z1_after": best_z1,
            "best_semantic_after": best_semantic,
            "best_hidden_relative_delta_raw": hidden_relative,
            "best_clip_scale": torch.ones_like(hidden_relative),
            "best_iteration": best_iteration,
            "best_scale": torch.full(
                (batch_size,),
                float(step_relative),
                device=original.device,
            ),
            "best_requested_code": requested_code,
            "best_objective_target_code": objective_target,
            "best_accepted": best_accepted,
        }

    def hook(self, _module, _inputs, output):
        hidden, rebuild = split_layer_output(output)
        if hidden.ndim != 3:
            return output
        seq_len = hidden.size(1)
        is_prefill = seq_len > 1
        if is_prefill:
            self.context.current_step = -1
            self.context.generated_step = 0
        else:
            self.context.current_step = int(self.context.generated_step)
            self.context.generated_step += 1
        should_patch_generated = (
            self.apply_to_generated
            and not is_prefill
            and (
                self.generated_patch_steps < 0
                or self.context.current_step < self.generated_patch_steps
            )
        )
        if not is_prefill and not should_patch_generated:
            return output

        apply_mask = self.context.apply_mask
        if apply_mask is None or apply_mask.numel() != hidden.size(0):
            return output
        rows_cpu = torch.nonzero(apply_mask.bool(), as_tuple=False).flatten()
        if rows_cpu.numel() == 0:
            return output

        self._ensure_device(hidden.device)
        rows = rows_cpu.to(hidden.device)
        position = hidden.size(1) - 1 if is_prefill else 0
        edited = hidden.clone()
        x_raw = edited[rows, position].float()
        ids = self.context.current_ids or []
        row_ids = [ids[row_index] if row_index < len(ids) else "" for row_index in rows_cpu.tolist()]
        dose_matching = self.dose_reference is not None and self.control != "main"
        dose_reference_accepted = torch.zeros(
            len(row_ids), dtype=torch.bool, device=x_raw.device
        )
        dose_reference_code_delta = torch.zeros(
            len(row_ids), dtype=torch.float32, device=x_raw.device
        )
        dose_reference_hidden_delta = torch.zeros_like(dose_reference_code_delta)
        dose_reference_semantic_delta = torch.zeros_like(dose_reference_code_delta)
        dose_reference_hidden_vectors = torch.zeros_like(x_raw)
        if dose_matching:
            for local_index, sample_id in enumerate(row_ids):
                reference = self.dose_reference.get(
                    (str(sample_id), int(self.context.current_step))
                )
                if reference is None or not bool(reference.get("code_correction_accepted")):
                    continue
                dose_reference_accepted[local_index] = True
                dose_reference_code_delta[local_index] = float(
                    reference.get("actual_code_delta_l2") or 0.0
                )
                dose_reference_hidden_delta[local_index] = float(
                    reference.get("hidden_relative_delta") or 0.0
                )
                dose_reference_semantic_delta[local_index] = float(
                    reference.get("semantic_relative_delta") or 0.0
                )
                vector = reference.get("_hidden_delta_vector")
                if vector is not None:
                    vector = torch.as_tensor(vector, dtype=x_raw.dtype, device=x_raw.device).flatten()
                    if vector.numel() == x_raw.size(1):
                        dose_reference_hidden_vectors[local_index] = vector
        mean = self.normalization["next_mean"].to(x_raw).view(1, -1)
        std = self.normalization["next_std"].to(x_raw).view(1, -1).clamp_min(1e-6)
        x_std = (x_raw - mean) / std

        with torch.inference_mode():
            z1 = self.decoupler.e1(x_std)
            z2 = self.decoupler.e2(x_std)
            purifier_out = self.purifier(z2)
            semantic = purifier_out["semantic"]
            meta = purifier_out["meta"]
            current_code = runtime_code(self.refiner, meta)
            base_requested_code = (
                current_code
                if self.control == "random_hidden_direction"
                else self._target_code(
                    current_code,
                    dose_reference_code_delta if dose_matching else None,
                )
            )
            batch_size = x_raw.size(0)
            if self.control == "random_hidden_direction":
                direct_result = self._sample_random_hidden_direction(
                    x_raw,
                    mean,
                    std,
                    z1,
                    semantic,
                    dose_reference_hidden_vectors,
                    dose_reference_hidden_delta,
                    dose_reference_semantic_delta,
                    dose_reference_accepted,
                )
                direct_result["best_requested_code"] = current_code.detach().clone()
                direct_result["best_objective_target_code"] = current_code.detach().clone()
            else:
                direct_result = self._direct_optimize_hidden(
                    x_raw,
                    mean,
                    std,
                    current_code,
                    z1,
                    semantic,
                    dose_matching,
                    dose_reference_accepted,
                    dose_reference_code_delta,
                )
            best_objective = direct_result["best_objective"]
            best_replacement = direct_result["best_replacement"]
            best_code = direct_result["best_code"]
            best_z1_after = direct_result["best_z1_after"]
            best_semantic_after = direct_result["best_semantic_after"]
            best_hidden_relative_delta_raw = direct_result[
                "best_hidden_relative_delta_raw"
            ]
            best_clip_scale = direct_result["best_clip_scale"]
            best_iteration = direct_result["best_iteration"]
            best_scale = direct_result["best_scale"]
            best_requested_code = direct_result["best_requested_code"]
            best_objective_target_code = direct_result[
                "best_objective_target_code"
            ]
            best_accepted = direct_result["best_accepted"]
            best_main_hidden_cosine = direct_result.get(
                "best_main_hidden_cosine", torch.full_like(best_hidden_relative_delta_raw.flatten(), float("nan"))
            )
            best_main_hidden_projection = direct_result.get(
                "best_main_hidden_projection", torch.full_like(best_hidden_relative_delta_raw.flatten(), float("nan"))
            )

            replacement = best_replacement
            code_after = best_code
            z1_after = best_z1_after
            semantic_after = best_semantic_after
            hidden_relative_delta_raw = best_hidden_relative_delta_raw
            clip_scale = best_clip_scale
            delta_raw = replacement - x_raw

            centers = self.code_centroids.to(current_code)
            before_margin = code_cluster_margin(
                current_code, centers, self.source_cluster, self.target_cluster
            )
            after_margin = code_cluster_margin(
                code_after, centers, self.source_cluster, self.target_cluster
            )
            hidden_rel_delta = delta_raw.norm(dim=1) / x_raw.norm(dim=1).clamp_min(1e-6)
            z1_cosine = F.cosine_similarity(z1_after, z1, dim=1)
            semantic_cosine = F.cosine_similarity(
                semantic_after, semantic, dim=1
            )
            z1_relative_delta = (z1_after - z1).norm(dim=1) / z1.norm(dim=1).clamp_min(1e-6)
            semantic_relative_delta = (
                semantic_after - semantic
            ).norm(dim=1) / semantic.norm(dim=1).clamp_min(1e-6)
            requested_code = best_requested_code
            objective_target_code = best_objective_target_code
            requested_code_delta = (requested_code - current_code).norm(dim=1)
            requested_code_direction = requested_code - current_code
            actual_code_direction = code_after - current_code
            actual_code_delta = actual_code_direction.norm(dim=1)
            code_requested_error = (code_after - requested_code).norm(dim=1)
            code_requested_error_before = requested_code_delta.clamp_min(1e-8)
            code_requested_error_ratio = code_requested_error / code_requested_error_before
            code_edit_fraction_achieved = 1.0 - code_requested_error_ratio
            direction_denominator = requested_code_direction.pow(2).sum(
                dim=1
            ).clamp_min(1e-8)
            code_direction_fraction_achieved = (
                actual_code_direction * requested_code_direction
            ).sum(dim=1) / direction_denominator
            code_orthogonal_delta = actual_code_direction - (
                code_direction_fraction_achieved.view(-1, 1)
                * requested_code_direction
            )
            code_orthogonal_delta_ratio = (
                code_orthogonal_delta.norm(dim=1)
                / requested_code_delta.clamp_min(1e-8)
            )
            objective_code_direction = objective_target_code - current_code
            objective_code_delta = objective_code_direction.norm(dim=1).clamp_min(1e-8)
            objective_code_error = (code_after - objective_target_code).norm(dim=1)
            objective_code_error_ratio = objective_code_error / objective_code_delta
            objective_direction_fraction = (
                actual_code_direction * objective_code_direction
            ).sum(dim=1) / objective_code_direction.pow(2).sum(dim=1).clamp_min(1e-8)
            objective_direction_cosine = F.cosine_similarity(
                actual_code_direction,
                objective_code_direction,
                dim=1,
                eps=1e-8,
            )
            dose_code_ratio = actual_code_delta / dose_reference_code_delta.clamp_min(1e-8)
            dose_hidden_ratio = hidden_rel_delta / dose_reference_hidden_delta.clamp_min(1e-8)
            dose_direction_cosine = F.cosine_similarity(
                actual_code_direction, requested_code_direction, dim=1, eps=1e-8
            )
            if self.real_code_direction is not None:
                real_direction = self.real_code_direction.to(actual_code_direction)
                random_real_axis_cosine = F.cosine_similarity(
                    actual_code_direction,
                    real_direction.view(1, -1).expand_as(actual_code_direction),
                    dim=1,
                    eps=1e-8,
                ).abs()
                random_real_axis_projection = (
                    actual_code_direction.matmul(real_direction)
                    / real_direction.pow(2).sum().clamp_min(1e-8)
                )
            else:
                random_real_axis_cosine = torch.full_like(actual_code_delta, float("nan"))
                random_real_axis_projection = torch.full_like(actual_code_delta, float("nan"))
            if dose_matching:
                dose_match_accepted = dose_reference_accepted & best_accepted
                # A strict random control matches the delivered hidden and
                # semantic perturbation, but must not be selected by matching
                # the real module's code displacement.  The old code-dose
                # criterion made the random control a second optimized target.
                if self.control not in {"random_hidden_direction"} and not (
                    self.control == "random_direction"
                    and self.random_dose_mode == "hidden_semantic"
                ):
                    dose_match_accepted &= (
                        ((dose_code_ratio - 1.0).abs() <= self.dose_match_tolerance)
                        & (dose_direction_cosine >= self.dose_min_direction_cosine)
                    )
                if (
                    self.control == "random_direction"
                    and self.random_max_real_axis_cosine >= 0.0
                ):
                    dose_match_accepted &= (
                        random_real_axis_cosine
                        <= self.random_max_real_axis_cosine
                    )
                if self.dose_hidden_tolerance > 0.0:
                    dose_match_accepted &= (
                        (dose_hidden_ratio - 1.0).abs() <= self.dose_hidden_tolerance
                    )
                if self.dose_semantic_tolerance > 0.0:
                    if self.control == "random_hidden_direction":
                        # The hidden-space random control is selected by an
                        # absolute semantic-drift match. A ratio is unstable
                        # when the real intervention has essentially zero
                        # semantic drift.
                        dose_match_accepted &= (
                            (semantic_relative_delta - dose_reference_semantic_delta).abs()
                            <= self.dose_semantic_tolerance
                        )
                    else:
                        semantic_ratio = (
                            semantic_relative_delta
                            / dose_reference_semantic_delta.clamp_min(1e-8)
                        )
                        dose_match_accepted &= (
                            (semantic_ratio - 1.0).abs()
                            <= self.dose_semantic_tolerance
                        )
            else:
                dose_match_accepted = best_accepted.clone()

        edited[rows, position] = replacement.to(hidden.dtype)
        for local_index, row_index in enumerate(rows_cpu.tolist()):
            self.rows.append(
                {
                    "id": ids[row_index] if row_index < len(ids) else "",
                    "mode": self.context.mode,
                    "step": int(self.context.current_step),
                    "is_prefill": bool(is_prefill),
                    "source_cluster": self.source_cluster,
                    "target_cluster": self.target_cluster,
                    "control": self.control,
                    "random_direction_seed": (
                        self.random_seed
                        if self.control in {"random_direction", "random_hidden_direction"}
                        else None
                    ),
                    "alpha": self.alpha,
                    "runtime_solver": "direct_trust_region",
                    "code_intervention": self.code_intervention,
                    "direct_iteration_selected": int(
                        best_iteration[local_index].item()
                    ),
                    "direct_step_relative": float(
                        best_scale[local_index].item()
                    ),
                    "code_correction_accepted": bool(
                        best_accepted[local_index].item()
                    ),
                    "code_correction_objective": float(
                        best_objective[local_index].item()
                    ),
                    "dose_matching_enabled": bool(dose_matching),
                    "dose_reference_accepted": bool(
                        dose_reference_accepted[local_index].item()
                    ),
                    "dose_reference_code_delta_l2": float(
                        dose_reference_code_delta[local_index].item()
                    ),
                    "dose_reference_hidden_relative_delta": float(
                        dose_reference_hidden_delta[local_index].item()
                    ),
                    "dose_reference_semantic_relative_delta": float(
                        dose_reference_semantic_delta[local_index].item()
                    ),
                    "dose_match_accepted": bool(
                        dose_match_accepted[local_index].item()
                    ),
                    "dose_direction_cosine": float(
                        dose_direction_cosine[local_index].item()
                    ),
                    "random_real_axis_cosine": float(
                        random_real_axis_cosine[local_index].item()
                    ),
                    "random_real_axis_projection": float(
                        random_real_axis_projection[local_index].item()
                    ),
                    "random_dose_mode": (
                        self.random_dose_mode
                        if self.control in {"random_direction", "random_hidden_direction"}
                        else None
                    ),
                    "random_hidden_candidates": (
                        self.random_hidden_candidates
                        if self.control == "random_hidden_direction"
                        else None
                    ),
                    "random_hidden_main_cosine": float(
                        best_main_hidden_cosine[local_index].item()
                    ),
                    "random_hidden_main_projection": float(
                        best_main_hidden_projection[local_index].item()
                    ),
                    "hidden_relative_delta_raw": float(
                        hidden_relative_delta_raw[local_index].item()
                    ),
                    "hidden_relative_delta": float(hidden_rel_delta[local_index].item()),
                    "hidden_norm_clip_scale": float(clip_scale[local_index].item()),
                    "hidden_norm_clipped": bool(clip_scale[local_index].item() < 0.999999),
                    "z1_cosine": float(z1_cosine[local_index].item()),
                    "z1_relative_delta": float(z1_relative_delta[local_index].item()),
                    "semantic_cosine": float(semantic_cosine[local_index].item()),
                    "semantic_relative_delta": float(
                        semantic_relative_delta[local_index].item()
                    ),
                    "code_margin_before": float(before_margin[local_index].item()),
                    "code_margin_after": float(after_margin[local_index].item()),
                    "code_margin_delta_toward_target": float(
                        (after_margin[local_index] - before_margin[local_index]).item()
                    ),
                    "code_moved_toward_target": bool(
                        (after_margin[local_index] - before_margin[local_index]).item() > 0.0
                    ),
                    "requested_code_delta_l2": float(
                        requested_code_delta[local_index].item()
                    ),
                    "actual_code_delta_l2": float(actual_code_delta[local_index].item()),
                    "actual_to_reference_code_delta_ratio": (
                        float(
                            actual_code_delta[local_index].item()
                            / max(dose_reference_code_delta[local_index].item(), 1e-8)
                        )
                        if dose_matching
                        and bool(dose_reference_accepted[local_index].item())
                        else None
                    ),
                    "actual_to_reference_hidden_delta_ratio": (
                        float(
                            hidden_rel_delta[local_index].item()
                            / max(dose_reference_hidden_delta[local_index].item(), 1e-8)
                        )
                        if dose_matching
                        and bool(dose_reference_accepted[local_index].item())
                        else None
                    ),
                    "code_requested_error_l2": float(code_requested_error[local_index].item()),
                    "code_requested_error_ratio": float(
                        code_requested_error_ratio[local_index].item()
                    ),
                    "code_edit_fraction_achieved": float(
                        code_edit_fraction_achieved[local_index].item()
                    ),
                    "code_direction_fraction_achieved": float(
                        code_direction_fraction_achieved[local_index].item()
                    ),
                    "code_orthogonal_delta_ratio": float(
                        code_orthogonal_delta_ratio[local_index].item()
                    ),
                    "objective_target_code_delta_l2": float(
                        objective_code_delta[local_index].item()
                    ),
                    "objective_target_error_l2": float(
                        objective_code_error[local_index].item()
                    ),
                    "objective_target_error_ratio": float(
                        objective_code_error_ratio[local_index].item()
                    ),
                    "objective_direction_fraction_achieved": float(
                        objective_direction_fraction[local_index].item()
                    ),
                    "objective_direction_cosine": float(
                        objective_direction_cosine[local_index].item()
                    ),
                }
            )
            if self.control == "main":
                self.hidden_delta_vectors[
                    (str(ids[row_index] if row_index < len(ids) else ""), int(self.context.current_step))
                ] = delta_raw[local_index].detach().float().cpu().half()
        return rebuild(edited)


def write_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_layer_matrix(activation_dir: str, layer_id: int, ids: Sequence[str], feature_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
    path = os.path.join(activation_dir, f"layer_{layer_id:03d}.pt")
    obj = safe_torch_load(path)
    all_ids = list(obj["ids"])
    features = obj["features"].float()
    id_to_index = {sample_id: idx for idx, sample_id in enumerate(all_ids)}
    missing = [sample_id for sample_id in ids if sample_id not in id_to_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} ids are missing from {path}. Examples: {preview}")
    row_indices = torch.tensor([id_to_index[sample_id] for sample_id in ids], dtype=torch.long)
    matrix = features[row_indices]
    if feature_indices is not None:
        matrix = matrix[:, feature_indices.long()]
    return matrix.contiguous()


def load_activation_ids(activation_dir: str, layer_id: int) -> List[str]:
    path = os.path.join(activation_dir, f"layer_{layer_id:03d}.pt")
    obj = safe_torch_load(path)
    return list(obj["ids"])


def hidden_size_from_activation(activation_dir: str, layer_id: int) -> int:
    path = os.path.join(activation_dir, f"layer_{layer_id:03d}.pt")
    obj = safe_torch_load(path)
    return int(obj["features"].shape[1])


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fixed_cluster_data(
    assignment_path: str,
    feature_path: str,
    ids: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load externally fitted labels/features, optionally restricted to ids.

    Fixed-cluster evaluation is often run on a post-treatment subset (for
    example, only critical samples).  That subset may contain one cluster even
    though the fitted artifact contains all clusters.  Callers can pass
    ``ids=None`` to obtain the complete artifact and compute global centroids.
    """
    if not assignment_path or not feature_path:
        raise ValueError(
            "--cluster-source soft_residual requires both --fixed-cluster-assignments "
            "and --fixed-cluster-features."
        )
    label_by_id: Dict[str, int] = {}
    with open(assignment_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = str(row.get("id", ""))
            label_text = row.get("activation_cluster", row.get("cluster", ""))
            if sample_id and label_text not in {None, ""}:
                label_by_id[sample_id] = int(label_text)

    feature_obj = safe_torch_load(feature_path)
    feature_ids = [str(sample_id) for sample_id in feature_obj.get("ids", [])]
    features = feature_obj.get("model_features")
    if not isinstance(features, torch.Tensor) or len(feature_ids) != features.shape[0]:
        raise ValueError(f"Invalid fixed cluster feature bundle: {feature_path}")
    feature_index = {sample_id: idx for idx, sample_id in enumerate(feature_ids)}
    if ids is None:
        selected_ids = [
            sample_id for sample_id in feature_ids if sample_id in label_by_id
        ]
    else:
        selected_ids = [str(sample_id) for sample_id in ids]
    missing = [
        sample_id
        for sample_id in selected_ids
        if sample_id not in label_by_id or sample_id not in feature_index
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} requested ids are absent from fixed residual clustering artifacts. "
            f"Examples: {', '.join(missing[:5])}"
        )
    labels = torch.tensor(
        [label_by_id[sample_id] for sample_id in selected_ids], dtype=torch.long
    )
    row_indices = torch.tensor(
        [feature_index[sample_id] for sample_id in selected_ids], dtype=torch.long
    )
    model_features = features[row_indices].float().contiguous()
    info = {
        "source": FIXED_CLUSTER_SOURCE,
        "assignment_path": assignment_path,
        "feature_path": feature_path,
        "input_dim": int(model_features.shape[1]),
        "already_transformed": True,
        "artifact_summary": feature_obj.get("summary", {}),
        "ids": selected_ids,
    }
    return model_features, labels, info


def fixed_cluster_available_ids(assignment_path: str, feature_path: str) -> set[str]:
    """Return ids present in both fixed-cluster artifacts."""
    if not assignment_path or not feature_path:
        raise ValueError(
            "Fixed-cluster id filtering requires both assignment and feature artifacts."
        )
    assignment_ids = set()
    with open(assignment_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("id", ""))
            if sample_id:
                assignment_ids.add(sample_id)

    feature_obj = safe_torch_load(feature_path)
    feature_ids = {str(sample_id) for sample_id in feature_obj.get("ids", [])}
    return assignment_ids & feature_ids


def restrict_rows_to_fixed_clusters(
    rows: Sequence[Dict],
    assignment_path: str,
    feature_path: str,
    max_samples: int = 0,
    seed: int = 0,
) -> List[Dict]:
    """Filter before sampling so a global cap does not discard fixed ids first."""
    available_ids = fixed_cluster_available_ids(assignment_path, feature_path)
    matched = [row for row in rows if str(row.get("id", "")) in available_ids]
    if not matched:
        raise ValueError("--fixed-cluster-ids-only removed every evaluation row.")
    if max_samples > 0 and len(matched) > max_samples:
        generator = random.Random(int(seed))
        chosen = sorted(generator.sample(range(len(matched)), int(max_samples)))
        matched = [matched[index] for index in chosen]
    return matched


def load_evaluation_ids(path: str) -> set[str]:
    """Load an explicit evaluation-id allowlist from txt, csv, json, or jsonl."""
    if not path:
        return set()
    suffix = os.path.splitext(path)[1].lower()
    ids: set[str] = set()
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = str(row.get("id", "")).strip()
                if sample_id:
                    ids.add(sample_id)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        values = value.get("ids", []) if isinstance(value, dict) else value
        if not isinstance(values, list):
            raise ValueError(f"Evaluation-id JSON must be a list or contain an 'ids' list: {path}")
        ids.update(str(item).strip() for item in values if str(item).strip())
    elif suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                sample_id = str(value.get("id", "") if isinstance(value, dict) else value).strip()
                if sample_id:
                    ids.add(sample_id)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            ids.update(line.strip() for line in handle if line.strip())
    if not ids:
        raise ValueError(f"Evaluation-id allowlist is empty: {path}")
    return ids


def transform_cluster_features(
    features: torch.Tensor,
    mode: str,
    standardize: bool,
    pca_dim: int,
    source: str,
    l2_normalize: bool = False,
    robust_clip_quantile: float = 0.0,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict]:
    x = features.float()
    info = {
        "source": source,
        "mode": mode,
        "standardize": standardize,
        "pca_dim": pca_dim,
        "l2_normalize": l2_normalize,
        "robust_clip_quantile": robust_clip_quantile,
        "input_dim": int(x.shape[1]),
    }
    if extra_info:
        info.update(extra_info)
    if standardize:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        info["standardized_mean_abs"] = float(mean.abs().mean().item())
        info["standardized_std_mean"] = float(std.mean().item())
    if robust_clip_quantile > 0.0:
        if not 0.0 < robust_clip_quantile <= 1.0:
            raise ValueError("--cluster-robust-clip-quantile must be in (0, 1].")
        clip_value = torch.quantile(x.abs().flatten(), robust_clip_quantile).clamp_min(1e-6)
        x = x.clamp(min=-float(clip_value.item()), max=float(clip_value.item()))
        info["robust_clip_value"] = float(clip_value.item())
    if pca_dim > 0 and x.shape[1] > pca_dim:
        x_centered = x - x.mean(dim=0, keepdim=True)
        _, _, v = torch.linalg.svd(x_centered, full_matrices=False)
        components = v[:pca_dim].t().contiguous()
        x = x_centered @ components
        info["pca_dim_used"] = int(pca_dim)
        info["pca_component_shape"] = list(components.shape)
    else:
        info["pca_dim_used"] = int(x.shape[1])
    if l2_normalize:
        norms = x.norm(dim=1, keepdim=True).clamp_min(1e-6)
        info["pre_l2_norm_mean"] = float(norms.mean().item())
        info["pre_l2_norm_p95"] = float(torch.quantile(norms.flatten(), 0.95).item())
        x = x / norms
    return x.contiguous(), info


def resolve_decoupler_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.decoupler_dir:
        config_path = os.path.join(args.decoupler_dir, "config.json")
        checkpoint_path = args.decoupler_checkpoint or os.path.join(args.decoupler_dir, "best_model.pt")
        normalization_path = args.decoupler_normalization or os.path.join(args.decoupler_dir, "normalization.pt")
    else:
        config_path = ""
        checkpoint_path = args.decoupler_checkpoint or ""
        normalization_path = args.decoupler_normalization or ""
    if not checkpoint_path or not normalization_path:
        raise ValueError("--cluster-source e2_latent/purified_meta/purified_semantic requires --decoupler-dir, or both --decoupler-checkpoint and --decoupler-normalization.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    if not os.path.exists(normalization_path):
        raise FileNotFoundError(normalization_path)
    return config_path, checkpoint_path, normalization_path


def infer_decoupler_arch(state_dict: Dict[str, torch.Tensor], dim: int, config: Dict[str, Any]) -> Tuple[int, int, float, str]:
    latent_dim = int(config.get("latent_dim") or state_dict["e2.net.8.weight"].shape[0])
    hidden_dim = int(config.get("hidden_dim") or state_dict["e2.net.0.weight"].shape[0])
    dropout = float(config.get("dropout", 0.0))
    target_mode = str(config.get("target_mode", "continuous_binary"))
    state_dim = int(state_dict["e2.net.0.weight"].shape[1])
    if state_dim != dim:
        raise ValueError(f"Decoupler input dim ({state_dim}) does not match next-layer activation dim ({dim}).")
    return latent_dim, hidden_dim, dropout, target_mode


def infer_purifier_arch(state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, int, float]:
    required = [
        "semantic_encoder.net.0.weight",
        "semantic_encoder.net.8.weight",
        "meta_encoder.net.8.weight",
        "semantic_to_z1.net.8.weight",
        "semantic_to_prev.net.8.weight",
        "meta_to_i.net.8.weight",
    ]
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise ValueError(f"Purifier checkpoint is missing keys: {missing[:5]}")
    z2_dim = int(state_dict["semantic_encoder.net.0.weight"].shape[1])
    semantic_dim = int(state_dict["semantic_encoder.net.8.weight"].shape[0])
    meta_dim = int(state_dict["meta_encoder.net.8.weight"].shape[0])
    hidden_dim = int(state_dict["semantic_encoder.net.0.weight"].shape[0])
    z1_dim = int(state_dict["semantic_to_z1.net.8.weight"].shape[0])
    prev_dim = int(state_dict["semantic_to_prev.net.8.weight"].shape[0])
    target_dim = int(state_dict["meta_to_i.net.8.weight"].shape[0])
    dropout = float(config.get("dropout", 0.0))
    return z2_dim, semantic_dim, meta_dim, hidden_dim, z1_dim, prev_dim, target_dim, dropout


def resolve_purifier_checkpoint(args: argparse.Namespace) -> str:
    checkpoint = getattr(args, "purifier_checkpoint", None)
    if checkpoint:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        return checkpoint
    decoupler_dir = getattr(args, "decoupler_dir", None)
    if not decoupler_dir:
        raise ValueError("--cluster-source purified_meta/purified_semantic requires --decoupler-dir or --purifier-checkpoint.")
    checkpoint = os.path.join(decoupler_dir, "e2_recursive_purifier", "best_purifier.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"{checkpoint} not found. Run train_decoupler.py with --run-e2-recursive-purifier, "
            "or pass --purifier-checkpoint explicitly."
        )
    return checkpoint


def standardize_next_features(x_next: torch.Tensor, normalization: Dict[str, Any]) -> torch.Tensor:
    if "next_mean" not in normalization or "next_std" not in normalization:
        raise ValueError("normalization.pt must contain next_mean and next_std for --cluster-source e2_latent.")
    mean = normalization["next_mean"].float().view(1, -1)
    std = normalization["next_std"].float().view(1, -1).clamp_min(1e-6)
    if x_next.shape[1] != mean.shape[1]:
        raise ValueError(f"Next activation dim ({x_next.shape[1]}) does not match normalization dim ({mean.shape[1]}).")
    return (x_next.float() - mean) / std


def code_cluster_margin(
    code: torch.Tensor,
    centroids: torch.Tensor,
    source_cluster: int,
    target_cluster: int,
) -> torch.Tensor:
    if source_cluster == target_cluster:
        return -torch.norm(code - centroids[target_cluster], dim=1)
    source_distance = torch.norm(code - centroids[source_cluster], dim=1)
    target_distance = torch.norm(code - centroids[target_cluster], dim=1)
    return source_distance - target_distance


def checkpoint_state_dict(value: Any, path: str) -> Dict[str, torch.Tensor]:
    if isinstance(value, dict):
        state_dict = value.get("model_state_dict") or value.get("state_dict") or value
    else:
        state_dict = value
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} does not contain a state_dict")
    return state_dict


def load_refined_residual_components(
    args: argparse.Namespace,
    next_dim: int,
    device: torch.device,
) -> Tuple[
    Decoupler,
    torch.nn.Module,
    Dict[str, Any],
    torch.nn.Module,
    Dict[str, Any],
]:
    config_path, checkpoint_path, normalization_path = resolve_decoupler_paths(args)
    config = read_json(config_path) if config_path and os.path.exists(config_path) else {}
    decoupler_state = checkpoint_state_dict(safe_torch_load(checkpoint_path), checkpoint_path)
    latent_dim, hidden_dim, dropout, target_mode = infer_decoupler_arch(
        decoupler_state, next_dim, config
    )
    decoupler = Decoupler(next_dim, latent_dim, hidden_dim, dropout, target_mode).to(device)
    decoupler.load_state_dict(decoupler_state, strict=False)
    decoupler.eval()

    normalization = safe_torch_load(normalization_path)
    if "next_mean" not in normalization or "next_std" not in normalization:
        raise ValueError(f"{normalization_path} lacks next_mean/next_std")
    if int(torch.as_tensor(normalization["next_mean"]).numel()) != next_dim:
        raise ValueError("Refined residual normalization does not match next-layer hidden size.")

    module_dir = os.path.normpath(args.refined_module_dir)
    refiner_path = os.path.join(module_dir, "module_refiner.pt")
    refiner, refiner_info = load_module_refiner(refiner_path, device)
    latent_source = str(refiner_info.get("latent_source") or "purified_meta")
    purifier_path: Optional[str]
    if latent_source == "main_z2":
        purifier_path = None
        purifier = MainZ2RuntimeAdapter(latent_dim).to(device)
        purifier_meta_dim = latent_dim
        runtime_semantic_control = "decoupler_z1_only"
    elif latent_source == "purified_meta":
        purifier_path = resolve_purifier_checkpoint(args)
        purifier_state = checkpoint_state_dict(
            safe_torch_load(purifier_path), purifier_path
        )
        purifier_arch = infer_purifier_arch(purifier_state, config)
        if purifier_arch[0] != latent_dim:
            raise ValueError(
                f"Purifier z2 dim ({purifier_arch[0]}) does not match "
                f"decoupler latent dim ({latent_dim})."
            )
        (
            purifier_z2_dim,
            semantic_dim,
            purifier_meta_dim,
            purifier_hidden_dim,
            z1_dim,
            prev_dim,
            target_dim,
            purifier_dropout,
        ) = purifier_arch
        purifier = E2RecursivePurifier(
            purifier_z2_dim,
            semantic_dim,
            purifier_meta_dim,
            purifier_hidden_dim,
            purifier_dropout,
            z1_dim,
            prev_dim,
            target_dim,
        ).to(device)
        purifier.load_state_dict(purifier_state, strict=False)
        purifier.eval()
        runtime_semantic_control = "purifier_semantic_and_decoupler_z1"
    else:
        raise ValueError(
            f"Unsupported refined module latent_source={latent_source!r}; "
            "expected purified_meta or main_z2."
        )
    if int(refiner_info["input_dim"]) != int(purifier_meta_dim):
        raise ValueError(
            f"Refiner input dim ({refiner_info['input_dim']}) does not match "
            f"runtime meta dim ({purifier_meta_dim})."
        )
    for module in (decoupler, purifier, refiner):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    info = {
        "decoupler_checkpoint": checkpoint_path,
        "normalization": normalization_path,
        "purifier_checkpoint": purifier_path,
        "latent_source": latent_source,
        "runtime_semantic_control": runtime_semantic_control,
        "lossless_main_z2_adapter": latent_source == "main_z2",
        "refiner_checkpoint": refiner_path,
        "code_dim": int(refiner_info["code_dim"]),
        "runtime_solver": "direct_trust_region",
    }
    return decoupler, purifier, normalization, refiner, info


def code_centroids_from_labels(
    code: torch.Tensor, labels: torch.Tensor, num_clusters: int
) -> torch.Tensor:
    values = []
    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        if not mask.any():
            raise ValueError(f"Refined-code cluster {cluster_id} is empty.")
        values.append(code[mask].mean(dim=0))
    return torch.stack(values)


def compute_latent_features(
    activation_dir: str,
    next_layer: int,
    ids: Sequence[str],
    args: argparse.Namespace,
    source: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict]:
    if args.e2_batch_size <= 0:
        raise ValueError("--e2-batch-size must be positive.")
    source = source or getattr(args, "cluster_source", "e2_latent")
    if source not in LATENT_CLUSTER_SOURCES:
        raise ValueError(f"Unsupported latent source: {source}")
    config_path, checkpoint_path, normalization_path = resolve_decoupler_paths(args)
    config = read_json(config_path) if config_path and os.path.exists(config_path) else {}
    state_obj = safe_torch_load(checkpoint_path)
    if isinstance(state_obj, dict):
        state_dict = state_obj.get("model_state_dict") or state_obj.get("state_dict") or state_obj
    else:
        state_dict = state_obj
    if not isinstance(state_dict, dict) or "e2.net.0.weight" not in state_dict:
        raise ValueError(f"{checkpoint_path} does not look like a train_decoupler state_dict.")

    normalization = safe_torch_load(normalization_path)
    norm_next_layer = normalization.get("next_layer")
    if norm_next_layer is not None and int(norm_next_layer) != int(next_layer):
        raise ValueError(f"Decoupler normalization expects next_layer={norm_next_layer}, but target metadata uses next_layer={next_layer}.")

    x_next = load_layer_matrix(activation_dir, next_layer, ids, None)
    x_next = standardize_next_features(x_next, normalization)
    latent_dim, hidden_dim, dropout, target_mode = infer_decoupler_arch(state_dict, x_next.shape[1], config)
    if args.e2_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.e2_device)
    model = Decoupler(x_next.shape[1], latent_dim, hidden_dim, dropout, target_mode).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    purifier = None
    purifier_info: Dict[str, Any] = {}
    purifier_missing: List[str] = []
    purifier_unexpected: List[str] = []
    purifier_checkpoint = None
    if source in {"purified_meta", "purified_semantic"}:
        purifier_checkpoint = resolve_purifier_checkpoint(args)
        purifier_obj = safe_torch_load(purifier_checkpoint)
        if isinstance(purifier_obj, dict):
            purifier_state = purifier_obj.get("model_state_dict") or purifier_obj.get("state_dict") or purifier_obj
        else:
            purifier_state = purifier_obj
        if not isinstance(purifier_state, dict) or "semantic_encoder.net.0.weight" not in purifier_state:
            raise ValueError(f"{purifier_checkpoint} does not look like an E2RecursivePurifier state_dict.")
        p_z2_dim, semantic_dim, meta_dim, p_hidden_dim, z1_dim, prev_dim, target_dim, p_dropout = infer_purifier_arch(purifier_state, config)
        if int(p_z2_dim) != int(latent_dim):
            raise ValueError(f"Purifier z2_dim ({p_z2_dim}) does not match decoupler latent_dim ({latent_dim}).")
        purifier = E2RecursivePurifier(
            p_z2_dim,
            semantic_dim,
            meta_dim,
            p_hidden_dim,
            p_dropout,
            z1_dim,
            prev_dim,
            target_dim,
        ).to(device)
        purifier_missing, purifier_unexpected = purifier.load_state_dict(purifier_state, strict=False)
        purifier.eval()
        purifier_info = {
            "purifier_checkpoint": purifier_checkpoint,
            "purifier_semantic_dim": semantic_dim,
            "purifier_meta_dim": meta_dim,
            "purifier_hidden_dim": p_hidden_dim,
            "purifier_z1_dim": z1_dim,
            "purifier_prev_dim": prev_dim,
            "purifier_target_dim": target_dim,
            "purifier_missing_keys": list(purifier_missing),
            "purifier_unexpected_keys": list(purifier_unexpected),
        }
    latents = []
    desc = {
        "e2_latent": "Encode E2 latents",
        "purified_meta": "Encode purified meta latents",
        "purified_semantic": "Encode purified semantic latents",
    }[source]
    for start in tqdm(range(0, x_next.shape[0], args.e2_batch_size), desc=desc, leave=True):
        batch = x_next[start : start + args.e2_batch_size].to(device)
        with torch.inference_mode():
            z2 = model.e2(batch)
            if source == "e2_latent":
                encoded = z2
            else:
                assert purifier is not None
                out = purifier(z2)
                encoded = out["meta"] if source == "purified_meta" else out["semantic"]
            latents.append(encoded.detach().cpu())
    if device.type == "cuda":
        if purifier is not None:
            del purifier
        del model
        torch.cuda.empty_cache()
    info = {
        "latent_source": source,
        "decoupler_dir": args.decoupler_dir,
        "decoupler_checkpoint": checkpoint_path,
        "decoupler_normalization": normalization_path,
        "decoupler_latent_dim": latent_dim,
        "decoupler_hidden_dim": hidden_dim,
        "decoupler_target_mode": target_mode,
        "decoupler_missing_keys": list(missing),
        "decoupler_unexpected_keys": list(unexpected),
        "e2_batch_size": int(args.e2_batch_size),
        "e2_device": str(device),
    }
    info.update(purifier_info)
    return torch.cat(latents, dim=0).contiguous(), info


def compute_e2_latents(
    activation_dir: str,
    next_layer: int,
    ids: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict]:
    return compute_latent_features(activation_dir, next_layer, ids, args, source="e2_latent")


def val_group_lookup(targets: Dict, group_basis: str) -> Dict[str, str]:
    val_ids = list(targets.get("val_ids", []))
    group_key = "val_true_score_group" if group_basis == "true" else "val_pred_score_group"
    group_tensor = targets.get(group_key)
    if group_tensor is None:
        return {sample_id: "unknown" for sample_id in val_ids}
    return {
        sample_id: ["low", "mid", "high"][int(group_id)] if int(group_id) in {0, 1, 2} else "unknown"
        for sample_id, group_id in zip(val_ids, group_tensor.tolist())
    }


def build_scope_rows(
    data_path: str,
    activation_dir: str,
    layer_id: int,
    targets: Dict,
    group_basis: str,
    eval_group: str,
    max_samples: int,
    eval_scope: str,
    id_regex: str = "",
) -> List[Dict]:
    data_rows = {row["id"]: row for row in read_jsonl(data_path)}
    cached_ids = load_activation_ids(activation_dir, layer_id)
    id_pattern = re.compile(id_regex) if id_regex else None
    if eval_scope == "val":
        scope_ids = list(targets.get("val_ids", []))
    elif eval_scope == "train":
        scope_ids = list(targets.get("train_ids", []))
    elif eval_scope == "test":
        scope_ids = list(targets.get("test_ids", []))
        if not scope_ids and targets.get("metadata", {}).get("cluster_eval_scope") == "test":
            scope_ids = list(targets.get("val_ids", []))
    elif eval_scope == "all_cached":
        scope_ids = cached_ids
    else:
        raise ValueError(f"Unknown eval scope: {eval_scope}")
    if not scope_ids:
        raise ValueError(f"No ids found for --eval-scope {eval_scope}.")

    cached_set = set(cached_ids)
    val_groups = val_group_lookup(targets, group_basis)
    train_set = set(targets.get("train_ids", []))
    val_ids = list(targets.get("val_ids", []))
    val_set = set(val_ids)
    test_set = set(targets.get("test_ids", []))
    val_index_by_id = {sample_id: idx for idx, sample_id in enumerate(val_ids)}
    rows = []
    for idx, sample_id in enumerate(scope_ids):
        if id_pattern is not None and not id_pattern.search(str(sample_id)):
            continue
        if sample_id not in data_rows or sample_id not in cached_set:
            continue
        group = val_groups.get(sample_id, "unknown")
        if eval_group != "all" and group != eval_group:
            continue
        item = dict(data_rows[sample_id])
        item["eval_index"] = idx
        item["eval_scope"] = eval_scope
        if sample_id in test_set:
            item["source_split"] = "test"
        elif sample_id in val_set:
            item["source_split"] = "val"
            item["val_index"] = val_index_by_id[sample_id]
        elif sample_id in train_set:
            item["source_split"] = "train"
        else:
            item["source_split"] = "cached_only"
        item["meta_group"] = group
        rows.append(item)
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return rows


def pick_feature_indices(
    selected: torch.Tensor,
    hidden_size: int,
    feature_set: str,
    random_count: int,
    seed: int,
) -> torch.Tensor:
    selected = selected.long().unique(sorted=True)
    if feature_set == "selected":
        return selected
    generator = torch.Generator().manual_seed(int(seed))
    selected_set = set(int(x) for x in selected.tolist())
    if feature_set == "random":
        candidates = torch.arange(hidden_size, dtype=torch.long)
    elif feature_set == "random_nonselected":
        candidates = torch.tensor([idx for idx in range(hidden_size) if idx not in selected_set], dtype=torch.long)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    count = random_count if random_count > 0 else selected.numel()
    if candidates.numel() < count:
        raise ValueError(f"Need {count} features, but only {candidates.numel()} candidates are available.")
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[:count]].sort().values


def prepare_cluster_features(
    raw_features: torch.Tensor,
    feature_indices: torch.Tensor,
    targets: Dict,
    mode: str,
    standardize: bool,
    pca_dim: int,
    l2_normalize: bool = False,
    robust_clip_quantile: float = 0.0,
) -> Tuple[torch.Tensor, Dict]:
    if mode == "continuous":
        x = raw_features.float()
    elif mode == "binary":
        threshold = targets.get("binary_activation_threshold")
        if not isinstance(threshold, torch.Tensor):
            raise ValueError("--cluster-feature-mode binary requires binary_activation_threshold in intervention_targets.pt.")
        threshold = threshold[0, feature_indices.long()].float()
        x = (raw_features.float() >= threshold.view(1, -1)).float()
    else:
        raise ValueError(f"Unknown cluster feature mode: {mode}")

    return transform_cluster_features(
        x,
        mode,
        standardize,
        pca_dim,
        source="i_layer_selected",
        l2_normalize=l2_normalize,
        robust_clip_quantile=robust_clip_quantile,
    )


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.empty(x.shape[0], 2)
    x_centered = x.float() - x.float().mean(dim=0, keepdim=True)
    if x_centered.shape[1] == 1:
        return torch.cat([x_centered, torch.zeros(x_centered.shape[0], 1)], dim=1)
    _, _, v = torch.linalg.svd(x_centered, full_matrices=False)
    coords = x_centered @ v[:2].t()
    if coords.shape[1] == 1:
        coords = torch.cat([coords, torch.zeros(coords.shape[0], 1)], dim=1)
    return coords[:, :2].contiguous()


def auto_select_k(model_features: torch.Tensor, args: argparse.Namespace) -> Tuple[int, List[Dict]]:
    k_min = max(2, int(args.k_min))
    k_max = min(int(args.k_max), max(2, model_features.shape[0] - 1))
    if k_min > k_max:
        raise ValueError(f"Invalid auto-k range: k_min={k_min}, k_max={k_max}.")
    candidates = []
    best = None
    for k in range(k_min, k_max + 1):
        labels, centroids, inertia = run_kmeans(
            model_features,
            k,
            args.kmeans_iters,
            args.kmeans_restarts,
            args.seed + 17 * k,
        )
        counts = torch.bincount(labels, minlength=k).float()
        fractions = counts / max(labels.numel(), 1)
        min_fraction = float(fractions.min().item())
        max_fraction = float(fractions.max().item())
        silhouette = centroid_silhouette_score(model_features, labels, centroids)
        small_cluster_penalty = max(0.0, float(args.min_cluster_fraction) - min_fraction) / max(float(args.min_cluster_fraction), 1e-6)
        imbalance_penalty = float(fractions.std(unbiased=False).item())
        if args.auto_k_metric == "silhouette":
            score = silhouette
        else:
            score = silhouette - args.auto_k_small_cluster_weight * small_cluster_penalty - args.auto_k_imbalance_weight * imbalance_penalty
        row = {
            "k": k,
            "score": float(score),
            "centroid_silhouette": float(silhouette),
            "inertia": float(inertia),
            "min_cluster_fraction": min_fraction,
            "max_cluster_fraction": max_fraction,
            "small_cluster_penalty": float(small_cluster_penalty),
            "imbalance_penalty": float(imbalance_penalty),
            "cluster_counts": [int(x) for x in counts.tolist()],
        }
        candidates.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
    assert best is not None
    return int(best["k"]), candidates


def raw_centroids_from_labels(raw_features: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    centroids = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        if mask.any():
            centroids.append(raw_features[mask].mean(dim=0))
        else:
            centroids.append(torch.zeros(raw_features.shape[1], dtype=raw_features.dtype))
    return torch.stack(centroids, dim=0).contiguous()


def build_cluster_result_from_prepared(
    name: str,
    prepared_cluster_features: torch.Tensor,
    patch_features: torch.Tensor,
    feature_indices: torch.Tensor,
    args: argparse.Namespace,
    seed_offset: int = 0,
    fixed_labels: Optional[torch.Tensor] = None,
    model_centroids_override: Optional[torch.Tensor] = None,
) -> ClusterResult:
    model_features = prepared_cluster_features.float()
    if fixed_labels is None:
        labels, model_centroids, inertia = run_kmeans(
            model_features,
            args.num_clusters,
            args.kmeans_iters,
            args.kmeans_restarts,
            args.seed + seed_offset,
        )
    else:
        labels = fixed_labels.long()
        if model_centroids_override is None:
            model_centroids = raw_centroids_from_labels(
                model_features, labels, args.num_clusters
            )
        else:
            model_centroids = model_centroids_override.float().contiguous()
            if model_centroids.ndim != 2 or model_centroids.size(0) != args.num_clusters:
                raise ValueError(
                    "Global fixed-cluster model centroids have an incompatible shape: "
                    f"{tuple(model_centroids.shape)}"
                )
        inertia = (model_features - model_centroids[labels]).pow(2).sum(dim=1).mean().item()
    raw_centroids = raw_centroids_from_labels(patch_features.float(), labels, args.num_clusters)
    return ClusterResult(
        name=name,
        feature_indices=feature_indices.long(),
        raw_features=patch_features.float(),
        model_features=model_features.float(),
        labels=labels.long(),
        raw_centroids=raw_centroids.float(),
        model_centroids=model_centroids.float(),
        pca2=pca_2d(model_features),
        inertia=float(inertia),
    )


def build_cluster_result(
    name: str,
    raw_features: torch.Tensor,
    feature_indices: torch.Tensor,
    targets: Dict,
    args: argparse.Namespace,
    seed_offset: int = 0,
    fixed_labels: Optional[torch.Tensor] = None,
) -> Tuple[ClusterResult, Dict]:
    model_features, feature_info = prepare_cluster_features(
        raw_features,
        feature_indices,
        targets,
        args.cluster_feature_mode,
        not args.no_standardize,
        args.pca_dim,
        args.cluster_l2_normalize,
        args.cluster_robust_clip_quantile,
    )
    return (
        build_cluster_result_from_prepared(
            name,
            model_features,
            raw_features,
            feature_indices,
            args,
            seed_offset=seed_offset,
            fixed_labels=fixed_labels,
        ),
        feature_info,
    )


def cluster_label_name(cluster_id: int) -> str:
    if 0 <= cluster_id < len(CLUSTER_NAMES):
        return CLUSTER_NAMES[cluster_id]
    return f"c{cluster_id}"


def add_cluster_fields(rows: List[Dict], cluster_result: ClusterResult) -> None:
    distances = torch.cdist(
        cluster_result.model_features.float(),
        cluster_result.model_centroids.float(),
    )
    for idx, row in enumerate(rows):
        cluster_id = int(cluster_result.labels[idx].item())
        own_distance = float(distances[idx, cluster_id].item())
        other_distances = distances[idx].clone()
        other_distances[cluster_id] = float("inf")
        nearest_other_distance = float(other_distances.min().item())
        margin = nearest_other_distance - own_distance
        row["activation_cluster"] = cluster_id
        row["activation_cluster_name"] = cluster_label_name(cluster_id)
        row["cluster_pca_x"] = float(cluster_result.pca2[idx, 0].item())
        row["cluster_pca_y"] = float(cluster_result.pca2[idx, 1].item())
        row["cluster_feature_norm"] = float(cluster_result.raw_features[idx].norm().item())
        row["cluster_patch_feature_norm"] = float(cluster_result.raw_features[idx].norm().item())
        row["cluster_model_feature_norm"] = float(cluster_result.model_features[idx].norm().item())
        row["cluster_active_rate"] = float((cluster_result.raw_features[idx] > 0).float().mean().item())
        row["cluster_own_distance"] = own_distance
        row["cluster_nearest_other_distance"] = nearest_other_distance
        row["cluster_margin"] = margin
        row["cluster_margin_normalized"] = float(
            margin / max(nearest_other_distance + own_distance, 1e-8)
        )


def sample_indices_by_cluster(
    labels: torch.Tensor,
    max_samples_per_cluster: int,
    strategy: str,
    seed: int,
) -> torch.Tensor:
    if max_samples_per_cluster <= 0:
        return torch.arange(labels.numel(), dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for cluster_id in range(int(labels.max().item()) + 1):
        indices = torch.nonzero(labels == cluster_id, as_tuple=False).flatten()
        if indices.numel() > max_samples_per_cluster:
            if strategy == "random":
                order = torch.randperm(indices.numel(), generator=generator)
                indices = indices[order[:max_samples_per_cluster]]
            elif strategy == "first":
                indices = indices[:max_samples_per_cluster]
            else:
                raise ValueError(f"Unknown --cluster-sample-strategy: {strategy}")
        selected.append(indices)
    if not selected:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(selected, dim=0).sort().values


def subset_cluster_result(cluster_result: ClusterResult, indices: torch.Tensor) -> ClusterResult:
    indices = indices.long()
    return ClusterResult(
        name=cluster_result.name,
        feature_indices=cluster_result.feature_indices,
        raw_features=cluster_result.raw_features[indices].contiguous(),
        model_features=cluster_result.model_features[indices].contiguous(),
        labels=cluster_result.labels[indices].contiguous(),
        raw_centroids=cluster_result.raw_centroids,
        model_centroids=cluster_result.model_centroids,
        pca2=cluster_result.pca2[indices].contiguous(),
        inertia=cluster_result.inertia,
    )


def assignment_rows(rows: List[Dict], cluster_result: ClusterResult, baseline_by_id: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    out = []
    baseline_by_id = baseline_by_id or {}
    distances = torch.cdist(
        cluster_result.model_features.float(),
        cluster_result.model_centroids.float(),
    )
    for idx, row in enumerate(rows):
        base = baseline_by_id.get(row["id"], {})
        cluster_id = int(cluster_result.labels[idx].item())
        own_distance = float(distances[idx, cluster_id].item())
        other_distances = distances[idx].clone()
        other_distances[cluster_id] = float("inf")
        nearest_other_distance = float(other_distances.min().item())
        margin = nearest_other_distance - own_distance
        out.append(
            {
                "id": row["id"],
                "eval_scope": row.get("eval_scope"),
                "source_split": row.get("source_split"),
                "eval_index": row.get("eval_index"),
                "val_index": row.get("val_index"),
                "meta_group": row.get("meta_group"),
                "activation_cluster": cluster_id,
                "activation_cluster_name": cluster_label_name(cluster_id),
                "cluster_pca_x": float(cluster_result.pca2[idx, 0].item()),
                "cluster_pca_y": float(cluster_result.pca2[idx, 1].item()),
                "cluster_feature_norm": float(cluster_result.raw_features[idx].norm().item()),
                "cluster_patch_feature_norm": float(cluster_result.raw_features[idx].norm().item()),
                "cluster_model_feature_norm": float(cluster_result.model_features[idx].norm().item()),
                "cluster_own_distance": own_distance,
                "cluster_nearest_other_distance": nearest_other_distance,
                "cluster_margin": margin,
                "cluster_margin_normalized": float(
                    margin / max(nearest_other_distance + own_distance, 1e-8)
                ),
                "baseline_correct": base.get("correct"),
                "baseline_pred_answer": base.get("pred_answer"),
                "gold_answer": normalize_number(row.get("numeric_answer")),
                "question": row.get("question"),
            }
        )
    return out


def cluster_static_summary(rows: List[Dict], cluster_result: ClusterResult, baseline_rows: Optional[List[Dict]] = None) -> Dict:
    baseline_by_id = {row["id"]: row for row in baseline_rows or []}
    summary = {
        "name": cluster_result.name,
        "feature_count": int(cluster_result.feature_indices.numel()),
        "features": cluster_result.feature_indices.tolist(),
        "patch_feature_dim": int(cluster_result.raw_features.shape[1]),
        "model_feature_dim": int(cluster_result.model_features.shape[1]),
        "num_clusters": int(cluster_result.raw_centroids.shape[0]),
        "inertia": cluster_result.inertia,
        "clusters": {},
        "centroid_l2": {},
    }
    for cluster_id in range(cluster_result.raw_centroids.shape[0]):
        indices = torch.nonzero(cluster_result.labels == cluster_id, as_tuple=False).flatten()
        cluster_rows = [rows[int(i)] for i in indices.tolist()]
        values = cluster_result.raw_features[indices] if indices.numel() else torch.empty(0, cluster_result.raw_features.shape[1])
        model_values = cluster_result.model_features[indices] if indices.numel() else torch.empty(0, cluster_result.model_features.shape[1])
        entry = {
            "n": len(cluster_rows),
            "fraction": len(cluster_rows) / max(len(rows), 1),
            "patch_feature_norm_mean": float(values.norm(dim=1).mean().item()) if indices.numel() else None,
            "patch_feature_mean_abs": float(values.abs().mean().item()) if indices.numel() else None,
            "model_feature_norm_mean": float(model_values.norm(dim=1).mean().item()) if indices.numel() else None,
            "model_feature_mean_abs": float(model_values.abs().mean().item()) if indices.numel() else None,
            "feature_norm_mean": float(values.norm(dim=1).mean().item()) if indices.numel() else None,
            "feature_mean_abs": float(values.abs().mean().item()) if indices.numel() else None,
            "meta_group_counts": {
                group: sum(1 for row in cluster_rows if row.get("meta_group") == group)
                for group in ["low", "mid", "high", "unknown"]
            },
        }
        if baseline_rows:
            baseline_cluster_rows = [baseline_by_id[row["id"]] for row in cluster_rows if row["id"] in baseline_by_id]
            entry["baseline"] = summarize_accuracy_rows(baseline_cluster_rows)
        summary["clusters"][str(cluster_id)] = entry

    for src in range(cluster_result.raw_centroids.shape[0]):
        for dst in range(cluster_result.raw_centroids.shape[0]):
            if src == dst:
                continue
            key = f"{src}->{dst}"
            summary["centroid_l2"][key] = float((cluster_result.raw_centroids[src] - cluster_result.raw_centroids[dst]).norm().item())
    return summary


def centroid_activation_state_rows(cluster_result: ClusterResult) -> List[Dict]:
    rows: List[Dict] = []
    feature_ids = cluster_result.feature_indices.long().cpu().tolist()
    for cluster_id in range(cluster_result.raw_centroids.shape[0]):
        centroid = cluster_result.raw_centroids[cluster_id].float().cpu()
        for feature_position, neuron_index in enumerate(feature_ids):
            rows.append(
                {
                    "cluster": cluster_id,
                    "cluster_name": cluster_label_name(cluster_id),
                    "feature_position": feature_position,
                    "neuron_index": int(neuron_index),
                    "centroid_activation": float(centroid[feature_position].item()),
                }
            )
    return rows


def parse_transitions(specs: Sequence[str], k: int) -> List[Tuple[int, int]]:
    if not specs or list(specs) == ["all_pairs"]:
        return [(src, dst) for src in range(k) for dst in range(k) if src != dst]
    transitions = []
    for spec in specs:
        if spec == "all_pairs":
            transitions.extend((src, dst) for src in range(k) for dst in range(k) if src != dst)
            continue
        if ":" not in spec:
            raise ValueError(f"Invalid transition '{spec}'. Use source:target, e.g. 0:1, or all_pairs.")
        src_text, dst_text = spec.split(":", 1)
        src, dst = int(src_text), int(dst_text)
        if src < 0 or src >= k or dst < 0 or dst >= k:
            raise ValueError(f"Transition {spec} is out of range for {k} clusters.")
        if src != dst:
            transitions.append((src, dst))
    seen = set()
    unique = []
    for transition in transitions:
        if transition not in seen:
            seen.add(transition)
            unique.append(transition)
    return unique


def build_patch_values_for_batch(
    batch_rows: List[Dict],
    row_index_by_id: Dict[str, int],
    spec: InterventionSpec,
    donor_state: Dict[str, int],
    meta_group: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_count = int(spec.cluster_result.feature_indices.numel())
    patch_values = torch.zeros(len(batch_rows), feature_count, dtype=torch.float32)
    apply_mask = torch.zeros(len(batch_rows), dtype=torch.bool)
    labels = spec.cluster_result.labels
    raw_features = spec.cluster_result.raw_features
    target_indices = torch.nonzero(labels == spec.target_cluster, as_tuple=False).flatten()
    if target_indices.numel() == 0:
        return patch_values, apply_mask

    extreme_target_indices: Optional[torch.Tensor] = None
    extreme_target_value: Optional[torch.Tensor] = None
    if spec.patch_mode == "extreme_prototype":
        extreme_target_indices = select_extreme_target_indices(
            spec.cluster_result,
            spec.source_cluster,
            spec.target_cluster,
            spec.extreme_fraction,
            spec.extreme_min_count,
        )
        if extreme_target_indices.numel() == 0:
            return patch_values, apply_mask
        extreme_target_value = raw_features[extreme_target_indices].mean(dim=0)

    for row_idx, row in enumerate(batch_rows):
        global_idx = row_index_by_id[row["id"]]
        source_match = int(labels[global_idx].item()) == spec.source_cluster
        group_match = meta_group == "all" or row.get("meta_group") == meta_group
        if not source_match or not group_match:
            continue
        apply_mask[row_idx] = True
        if spec.patch_mode == "prototype":
            patch_values[row_idx] = spec.cluster_result.raw_centroids[spec.target_cluster]
        elif spec.patch_mode == "extreme_prototype":
            assert extreme_target_value is not None
            patch_values[row_idx] = extreme_target_value
        elif spec.patch_mode == "donor":
            offset = donor_state.get(spec.name, 0)
            donor_idx = int(target_indices[offset % target_indices.numel()].item())
            donor_state[spec.name] = offset + 1
            patch_values[row_idx] = raw_features[donor_idx]
        elif spec.patch_mode == "farthest_donor":
            donor_idx = farthest_donor_index(spec.cluster_result, global_idx, target_indices)
            patch_values[row_idx] = raw_features[donor_idx]
        else:
            raise ValueError(f"Unknown patch mode: {spec.patch_mode}")
    return patch_values, apply_mask


def build_apply_mask_for_batch(
    batch_rows: List[Dict],
    row_index_by_id: Dict[str, int],
    spec: InterventionSpec,
    meta_group: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generation callback for latent/code interventions that need no raw patch values."""
    apply_mask = torch.zeros(len(batch_rows), dtype=torch.bool)
    for row_index, row in enumerate(batch_rows):
        global_index = row_index_by_id[row["id"]]
        source_match = (
            int(spec.cluster_result.labels[global_index].item()) == spec.source_cluster
        )
        group_match = meta_group == "all" or row.get("meta_group") == meta_group
        apply_mask[row_index] = bool(source_match and group_match)
    return torch.empty(len(batch_rows), 0), apply_mask


def summarize_refined_intervention_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(name: str, subset: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[float]:
        source = rows if subset is None else subset
        values = [float(row[name]) for row in source if row.get(name) is not None]
        return float(sum(values) / len(values)) if values else None

    accepted_rows = [row for row in rows if bool(row.get("code_correction_accepted"))]
    dose_matched_rows = [row for row in rows if bool(row.get("dose_match_accepted"))]
    relative = torch.tensor(
        [float(row["hidden_relative_delta"]) for row in rows], dtype=torch.float32
    ) if rows else torch.empty(0)
    return {
        "n": len(rows),
        "prefill_n": sum(bool(row.get("is_prefill")) for row in rows),
        "generated_step_n": sum(not bool(row.get("is_prefill")) for row in rows),
        "hidden_relative_delta_raw_mean": mean("hidden_relative_delta_raw"),
        "hidden_relative_delta_mean": mean("hidden_relative_delta"),
        "hidden_relative_delta_p95": (
            float(torch.quantile(relative, 0.95).item()) if relative.numel() else None
        ),
        "z1_cosine_mean": mean("z1_cosine"),
        "z1_relative_delta_mean": mean("z1_relative_delta"),
        "semantic_cosine_mean": mean("semantic_cosine"),
        "semantic_relative_delta_mean": mean("semantic_relative_delta"),
        "hidden_norm_clip_scale_mean": mean("hidden_norm_clip_scale"),
        "hidden_norm_clipped_rate": (
            float(sum(bool(row.get("hidden_norm_clipped")) for row in rows) / len(rows))
            if rows
            else None
        ),
        "code_correction_accepted_rate": (
            float(
                sum(bool(row.get("code_correction_accepted")) for row in rows)
                / len(rows)
            )
            if rows
            else None
        ),
        "code_correction_accepted_n": len(accepted_rows),
        "code_correction_iteration_selected_mean": mean(
            "code_correction_iteration_selected"
        ),
        "code_correction_scale_selected_mean": mean(
            "code_correction_scale_selected"
        ),
        "dose_request_scale_selected_mean": mean("dose_request_scale_selected"),
        "code_correction_objective_mean": mean("code_correction_objective"),
        "dose_matching_enabled": bool(
            any(bool(row.get("dose_matching_enabled")) for row in rows)
        ),
        "dose_reference_accepted_rate": (
            float(
                sum(bool(row.get("dose_reference_accepted")) for row in rows)
                / len(rows)
            )
            if rows
            else None
        ),
        "dose_reference_code_delta_l2_mean": mean(
            "dose_reference_code_delta_l2"
        ),
        "dose_reference_hidden_relative_delta_mean": mean(
            "dose_reference_hidden_relative_delta"
        ),
        "dose_match_accepted_rate": (
            float(sum(bool(row.get("dose_match_accepted")) for row in rows) / len(rows))
            if rows
            else None
        ),
        "dose_match_accepted_n": len(dose_matched_rows),
        "dose_direction_cosine_mean": mean("dose_direction_cosine"),
        "code_margin_delta_toward_target_mean": mean(
            "code_margin_delta_toward_target"
        ),
        "code_moved_toward_target_rate": (
            float(
                sum(bool(row.get("code_moved_toward_target")) for row in rows)
                / len(rows)
            )
            if rows
            else None
        ),
        "requested_code_delta_l2_mean": mean("requested_code_delta_l2"),
        "actual_code_delta_l2_mean": mean("actual_code_delta_l2"),
        "actual_to_reference_code_delta_ratio_mean": mean(
            "actual_to_reference_code_delta_ratio"
        ),
        "actual_to_reference_hidden_delta_ratio_mean": mean(
            "actual_to_reference_hidden_delta_ratio"
        ),
        "code_requested_error_l2_mean": mean("code_requested_error_l2"),
        "code_requested_error_ratio_mean": mean("code_requested_error_ratio"),
        "code_edit_fraction_achieved_mean": mean("code_edit_fraction_achieved"),
        "code_direction_fraction_achieved_mean": mean(
            "code_direction_fraction_achieved"
        ),
        "code_orthogonal_delta_ratio_mean": mean("code_orthogonal_delta_ratio"),
        "objective_target_code_delta_l2_mean": mean(
            "objective_target_code_delta_l2"
        ),
        "objective_target_error_ratio_mean": mean(
            "objective_target_error_ratio"
        ),
        "objective_direction_fraction_achieved_mean": mean(
            "objective_direction_fraction_achieved"
        ),
        "objective_direction_cosine_mean": mean("objective_direction_cosine"),
        # Overall means include intentional no-op fallbacks. These conditional
        # summaries expose the fidelity of edits that actually reached the
        # online acceptance and strict achieved-dose gates.
        "accepted_hidden_relative_delta_mean": mean(
            "hidden_relative_delta", accepted_rows
        ),
        "accepted_semantic_relative_delta_mean": mean(
            "semantic_relative_delta", accepted_rows
        ),
        "accepted_requested_code_delta_l2_mean": mean(
            "requested_code_delta_l2", accepted_rows
        ),
        "accepted_actual_code_delta_l2_mean": mean(
            "actual_code_delta_l2", accepted_rows
        ),
        "accepted_code_edit_fraction_achieved_mean": mean(
            "code_edit_fraction_achieved", accepted_rows
        ),
        "accepted_code_direction_fraction_achieved_mean": mean(
            "code_direction_fraction_achieved", accepted_rows
        ),
        "accepted_code_orthogonal_delta_ratio_mean": mean(
            "code_orthogonal_delta_ratio", accepted_rows
        ),
        "accepted_objective_target_error_ratio_mean": mean(
            "objective_target_error_ratio", accepted_rows
        ),
        "accepted_objective_direction_fraction_achieved_mean": mean(
            "objective_direction_fraction_achieved", accepted_rows
        ),
        "accepted_objective_direction_cosine_mean": mean(
            "objective_direction_cosine", accepted_rows
        ),
        "accepted_code_margin_delta_toward_target_mean": mean(
            "code_margin_delta_toward_target", accepted_rows
        ),
        "accepted_code_moved_toward_target_rate": (
            float(
                sum(bool(row.get("code_moved_toward_target")) for row in accepted_rows)
                / len(accepted_rows)
            )
            if accepted_rows
            else None
        ),
        "dose_matched_actual_to_reference_code_delta_ratio_mean": mean(
            "actual_to_reference_code_delta_ratio", dose_matched_rows
        ),
        "dose_matched_actual_to_reference_hidden_delta_ratio_mean": mean(
            "actual_to_reference_hidden_delta_ratio", dose_matched_rows
        ),
        "dose_matched_direction_cosine_mean": mean(
            "dose_direction_cosine", dose_matched_rows
        ),
        "dose_matched_semantic_relative_delta_mean": mean(
            "semantic_relative_delta", dose_matched_rows
        ),
    }


NEXT_TOKEN_AUDIT_METRICS = [
    "next_token_js_divergence",
    "next_token_total_variation",
    "next_token_delta_entropy",
    "next_token_delta_max_probability",
    "next_token_full_logit_delta_l2",
    "next_token_top10_overlap",
    "next_token_auto_logit_margin_delta_toward_target",
    "next_token_auto_logit_delta_projection_to_target",
]


def summarize_next_token_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    applied = [row for row in rows if bool(row.get("intervention_applied"))]

    def mean(name: str) -> Optional[float]:
        values = [
            float(row[name])
            for row in applied
            if row.get(name) is not None and math.isfinite(float(row[name]))
        ]
        return float(sum(values) / len(values)) if values else None

    summary: Dict[str, Any] = {
        "n": len(rows),
        "applied_n": len(applied),
        "argmax_changed_rate": (
            float(
                sum(bool(row.get("next_token_argmax_changed")) for row in applied)
                / len(applied)
            )
            if applied
            else None
        ),
        "auto_logit_moved_toward_target_rate": (
            float(
                sum(
                    bool(row.get("next_token_auto_logit_moved_toward_target"))
                    for row in applied
                )
                / len(applied)
            )
            if applied
            and any(
                row.get("next_token_auto_logit_moved_toward_target") is not None
                for row in applied
            )
            else None
        ),
    }
    for metric in NEXT_TOKEN_AUDIT_METRICS:
        summary[f"{metric}_mean"] = mean(metric)
    return summary


def paired_intervention_dose_summary(
    main_by_id: Dict[str, Dict[str, Any]],
    control_by_id: Dict[str, Dict[str, Any]],
    all_matched_ids: Sequence[str],
    analysis_ids: Sequence[str],
    control_label: str,
) -> Dict[str, Any]:
    """Describe acceptance and achieved dose for a paired intervention contrast."""

    def rate(rows: Dict[str, Dict[str, Any]], ids: Sequence[str]) -> Optional[float]:
        if not ids:
            return None
        return float(
            sum(bool(rows[sample_id].get("code_correction_accepted")) for sample_id in ids)
            / len(ids)
        )

    def mean(
        rows: Dict[str, Dict[str, Any]], ids: Sequence[str], field: str
    ) -> Optional[float]:
        values = [
            float(rows[sample_id][field])
            for sample_id in ids
            if rows[sample_id].get(field) is not None
            and math.isfinite(float(rows[sample_id][field]))
        ]
        return float(sum(values) / len(values)) if values else None

    return {
        "matched_source_n": int(len(all_matched_ids)),
        "both_accepted_n": int(
            sum(
                bool(main_by_id[sample_id].get("code_correction_accepted"))
                and bool(control_by_id[sample_id].get("code_correction_accepted"))
                for sample_id in all_matched_ids
            )
        ),
        "both_dose_matched_n": int(
            sum(
                bool(main_by_id[sample_id].get("code_correction_accepted"))
                and bool(control_by_id[sample_id].get("dose_match_accepted"))
                for sample_id in all_matched_ids
            )
        ),
        "main_accept_rate": rate(main_by_id, all_matched_ids),
        f"{control_label}_accept_rate": rate(control_by_id, all_matched_ids),
        f"{control_label}_dose_match_rate": (
            float(
                sum(
                    bool(control_by_id[sample_id].get("dose_match_accepted"))
                    for sample_id in all_matched_ids
                )
                / len(all_matched_ids)
            )
            if all_matched_ids
            else None
        ),
        "main_actual_code_delta_l2_mean": mean(
            main_by_id, analysis_ids, "actual_code_delta_l2"
        ),
        f"{control_label}_actual_code_delta_l2_mean": mean(
            control_by_id, analysis_ids, "actual_code_delta_l2"
        ),
        "main_hidden_relative_delta_mean": mean(
            main_by_id, analysis_ids, "hidden_relative_delta"
        ),
        f"{control_label}_hidden_relative_delta_mean": mean(
            control_by_id, analysis_ids, "hidden_relative_delta"
        ),
        "main_semantic_relative_delta_mean": mean(
            main_by_id, analysis_ids, "semantic_relative_delta"
        ),
        f"{control_label}_semantic_relative_delta_mean": mean(
            control_by_id, analysis_ids, "semantic_relative_delta"
        ),
        f"{control_label}_actual_to_reference_code_delta_ratio_mean": mean(
            control_by_id, analysis_ids, "actual_to_reference_code_delta_ratio"
        ),
        f"{control_label}_actual_to_reference_hidden_delta_ratio_mean": mean(
            control_by_id, analysis_ids, "actual_to_reference_hidden_delta_ratio"
        ),
        f"{control_label}_dose_direction_cosine_mean": mean(
            control_by_id, analysis_ids, "dose_direction_cosine"
        ),
    }


def plot_next_token_audit(
    output_dir: str,
    summary: Dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    modes = list(summary.get("modes", {}))
    if not modes:
        return
    apply_nmi_style()
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    labels = []
    for mode in modes:
        item = summary["modes"][mode]
        labels.append(
            f"{item.get('control', 'main')} {item.get('source_cluster')}\u2192{item.get('target_cluster')}"
        )
    panel_metrics = [
        ("next_token_js_divergence_mean", "JS divergence"),
        ("next_token_delta_entropy_mean", "\u0394 entropy"),
        ("next_token_delta_max_probability_mean", "\u0394 max probability"),
        ("auto_logit_moved_toward_target_rate", "Logit movement rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 6.4))
    colors = [
        COLORS["blue"]
        if summary["modes"][mode].get("control") == "main"
        else COLORS["gray"]
        for mode in modes
    ]
    positions = list(range(len(modes)))
    for panel_index, (axis, (metric, label)) in enumerate(
        zip(axes.flatten(), panel_metrics)
    ):
        values = [float(summary["modes"][mode].get(metric) or 0.0) for mode in modes]
        axis.barh(positions, values, color=colors, alpha=0.9)
        axis.axvline(0.0, color=COLORS["black"], linewidth=0.7)
        axis.set_yticks(positions, labels if panel_index % 2 == 0 else [""] * len(labels))
        axis.set_xlabel(label)
        style_axis(axis, grid="x")
        panel_label(axis, chr(ord("a") + panel_index))
    fig.tight_layout()
    save_figure(fig, os.path.join(plot_dir, "next_token_intervention_audit"))
    plt.close(fig)


def plot_refined_runtime_audit(
    output_dir: str,
    audit_by_mode: Dict[str, Dict[str, Any]],
    specs: Dict[str, InterventionSpec],
) -> None:
    import matplotlib.pyplot as plt

    modes = [mode for mode in specs if mode in audit_by_mode]
    if not modes:
        return
    apply_nmi_style()
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    labels = [
        f"{specs[mode].control} {specs[mode].source_cluster}\u2192{specs[mode].target_cluster}"
        for mode in modes
    ]
    colors = [
        COLORS["blue"] if specs[mode].control == "main" else COLORS["gray"]
        for mode in modes
    ]
    panels = [
        ("code_direction_fraction_achieved_mean", "Requested-direction fraction"),
        ("code_orthogonal_delta_ratio_mean", "Orthogonal code / requested norm"),
        ("semantic_relative_delta_mean", "Purified-semantic relative drift"),
        ("z1_relative_delta_mean", "E1 relative drift"),
    ]
    positions = list(range(len(modes)))
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_IN, 6.2))
    for panel_index, (axis, (metric, label)) in enumerate(
        zip(axes.flatten(), panels)
    ):
        values = [float(audit_by_mode[mode].get(metric) or 0.0) for mode in modes]
        axis.barh(positions, values, color=colors, alpha=0.9)
        axis.axvline(0.0, color=COLORS["black"], linewidth=0.7)
        axis.set_yticks(
            positions, labels if panel_index % 2 == 0 else [""] * len(labels)
        )
        axis.set_xlabel(label)
        style_axis(axis, grid="x")
        panel_label(axis, chr(ord("a") + panel_index))
    fig.tight_layout()
    save_figure(fig, os.path.join(plot_dir, "refined_runtime_fidelity"))
    plt.close(fig)


def select_extreme_target_indices(
    cluster_result: ClusterResult,
    source_cluster: int,
    target_cluster: int,
    fraction: float,
    min_count: int,
) -> torch.Tensor:
    labels = cluster_result.labels
    target_indices = torch.nonzero(labels == target_cluster, as_tuple=False).flatten()
    if target_indices.numel() == 0:
        return target_indices
    keep_count = int(math.ceil(float(target_indices.numel()) * max(float(fraction), 0.0)))
    keep_count = max(int(min_count), keep_count, 1)
    keep_count = min(keep_count, int(target_indices.numel()))

    features = cluster_result.model_features.float()
    centroids = cluster_result.model_centroids.float()
    target_features = features[target_indices]
    dist_to_target = torch.norm(target_features - centroids[target_cluster].view(1, -1), dim=1)
    if centroids.shape[0] <= 1:
        score = -dist_to_target
    elif source_cluster != target_cluster:
        dist_to_source = torch.norm(target_features - centroids[source_cluster].view(1, -1), dim=1)
        score = dist_to_source - dist_to_target
    else:
        other_ids = [idx for idx in range(centroids.shape[0]) if idx != target_cluster]
        other_centroids = centroids[other_ids]
        dist_to_other = torch.cdist(target_features, other_centroids).min(dim=1).values
        score = dist_to_other - dist_to_target
    selected = torch.topk(score, k=keep_count, largest=True).indices
    return target_indices[selected]


def farthest_donor_index(cluster_result: ClusterResult, source_index: int, target_indices: torch.Tensor) -> int:
    if target_indices.numel() == 0:
        raise ValueError("Cannot choose farthest donor from an empty target cluster.")
    features = cluster_result.model_features.float()
    source = features[int(source_index)].view(1, -1)
    target_features = features[target_indices]
    distances = torch.cdist(source, target_features).flatten()
    if target_indices.numel() > 1:
        same = target_indices == int(source_index)
        distances = distances.masked_fill(same, float("-inf"))
    return int(target_indices[int(torch.argmax(distances).item())].item())


STYLE_METRICS = metrics_for_profile("safety", include_quality=True)


def ensure_style_metrics(row: Dict, max_new_tokens: int) -> Dict:
    if "style_refusal_score" not in row:
        row.update(compute_style_metrics(row.get("generated_text", ""), int(row.get("generated_tokens") or 0), max_new_tokens))
    return row


def safe_mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def safe_median(values: Sequence[float]) -> Optional[float]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def safe_std(values: Sequence[float]) -> Optional[float]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def summarize_style_rows(rows: List[Dict], metrics: Sequence[str] = STYLE_METRICS) -> Dict:
    summary = {"n": len(rows)}
    for metric in metrics:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) is not None
            and math.isfinite(float(row[metric]))
        ]
        summary[metric] = {
            "mean": safe_mean(values),
            "median": safe_median(values),
            "std": safe_std(values),
        }
    return summary


def permutation_p_value(a: List[float], b: List[float], samples: int, seed: int) -> Optional[float]:
    a = [float(value) for value in a if math.isfinite(float(value))]
    b = [float(value) for value in b if math.isfinite(float(value))]
    if samples <= 0 or not a or not b:
        return None
    observed = abs((sum(a) / len(a)) - (sum(b) / len(b)))
    pooled = [float(x) for x in a + b]
    n_a = len(a)
    rng = random.Random(seed)
    count = 0
    for _ in range(samples):
        rng.shuffle(pooled)
        diff = abs((sum(pooled[:n_a]) / n_a) - (sum(pooled[n_a:]) / max(len(pooled) - n_a, 1)))
        if diff >= observed:
            count += 1
    return float((count + 1) / (samples + 1))


def style_pairwise_tests(
    rows: List[Dict],
    group_key: str,
    metrics: Sequence[str],
    samples: int,
    seed: int,
    metric_profile: str,
) -> List[Dict]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        group = str(row.get(group_key, "unknown"))
        groups.setdefault(group, []).append(row)
    out = []
    names = sorted(groups.keys(), key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            for metric in metrics:
                a = [
                    float(row[metric]) for row in groups[left]
                    if row.get(metric) is not None and math.isfinite(float(row[metric]))
                ]
                b = [
                    float(row[metric]) for row in groups[right]
                    if row.get(metric) is not None and math.isfinite(float(row[metric]))
                ]
                if not a or not b:
                    continue
                p_value = permutation_p_value(a, b, samples, seed + len(out) * 13)
                out.append(
                    {
                        "group_a": left,
                        "group_b": right,
                        "metric": metric,
                        "n_a": len(a),
                        "n_b": len(b),
                        "mean_a": safe_mean(a),
                        "mean_b": safe_mean(b),
                        "mean_diff_a_minus_b": safe_mean(a) - safe_mean(b),
                        "permutation_p": p_value,
                        "fdr_family": family_for_metric(metric, metric_profile),
                    }
                )
    for family in sorted({str(row["fdr_family"]) for row in out}):
        indices = [index for index, row in enumerate(out) if row["fdr_family"] == family]
        q_values = bh_adjust([out[index].get("permutation_p") for index in indices])
        for index, q_value in zip(indices, q_values):
            out[index]["fdr_family_size"] = len(indices)
            out[index]["bh_fdr_q"] = q_value
            p_value = out[index].get("permutation_p")
            out[index]["significant_p_0p05"] = bool(
                p_value is not None and math.isfinite(float(p_value)) and float(p_value) < 0.05
            )
    return out


def style_normalizer(rows: List[Dict], metrics: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    normalizer = {}
    for metric in metrics:
        values = [
            float(row[metric]) for row in rows
            if row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        mean = safe_mean(values) or 0.0
        std = safe_std(values) or 1.0
        normalizer[metric] = (float(mean), max(float(std), 1e-6))
    return normalizer


def style_vector(row: Dict, normalizer: Dict[str, Tuple[float, float]], metrics: Sequence[str]) -> torch.Tensor:
    values = []
    for metric in metrics:
        mean, std = normalizer[metric]
        raw = row.get(metric, mean)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = mean
        if not math.isfinite(value):
            value = mean
        values.append((value - mean) / std)
    return torch.tensor(values, dtype=torch.float32)


def style_centroids_by_cluster(rows: List[Dict], normalizer: Dict[str, Tuple[float, float]], metrics: Sequence[str]) -> Dict[int, torch.Tensor]:
    grouped: Dict[int, List[torch.Tensor]] = {}
    for row in rows:
        cluster = int(row.get("activation_cluster", -1))
        if cluster < 0:
            continue
        grouped.setdefault(cluster, []).append(style_vector(row, normalizer, metrics))
    return {
        cluster: torch.stack(vectors, dim=0).mean(dim=0)
        for cluster, vectors in grouped.items()
        if vectors
    }


def summarize_style_shift_to_target(
    rows: List[Dict],
    baseline_by_id: Dict[str, Dict],
    centroids: Dict[int, torch.Tensor],
    normalizer: Dict[str, Tuple[float, float]],
    metrics: Sequence[str],
    source_cluster: int,
    target_cluster: int,
) -> Dict:
    if source_cluster not in centroids or target_cluster not in centroids:
        return {"n": 0}
    source_centroid = centroids[source_cluster]
    target_centroid = centroids[target_cluster]
    deltas = []
    before_margins = []
    after_margins = []
    for row in rows:
        base = baseline_by_id.get(row["id"])
        if not base:
            continue
        before = style_vector(base, normalizer, metrics)
        after = style_vector(row, normalizer, metrics)
        before_margin = float(torch.norm(before - source_centroid).item() - torch.norm(before - target_centroid).item())
        after_margin = float(torch.norm(after - source_centroid).item() - torch.norm(after - target_centroid).item())
        before_margins.append(before_margin)
        after_margins.append(after_margin)
        deltas.append(after_margin - before_margin)
    return {
        "n": len(deltas),
        "mean_before_target_margin": safe_mean(before_margins),
        "mean_after_target_margin": safe_mean(after_margins),
        "mean_delta_toward_target": safe_mean(deltas),
        "median_delta_toward_target": safe_median(deltas),
        "moved_toward_target_rate": safe_mean([1.0 if value > 0 else 0.0 for value in deltas]),
    }


def write_style_outputs(
    output_dir: str,
    prefix: str,
    rows: List[Dict],
    group_key: str,
    permutation_samples: int,
    seed: int,
    max_new_tokens: int,
    metric_profile: str,
) -> Dict:
    metrics = metrics_for_profile(metric_profile, include_quality=True)
    for row in rows:
        ensure_style_metrics(row, max_new_tokens)
    metric_rows = [
        {
            "id": row.get("id"),
            "mode": row.get("mode"),
            "activation_cluster": row.get("activation_cluster"),
            "activation_cluster_name": row.get("activation_cluster_name"),
            **({"correct": row.get("correct")} if metric_profile == "math" else {}),
            **{metric: row.get(metric) for metric in metrics},
        }
        for row in rows
    ]
    write_csv(os.path.join(output_dir, f"{prefix}_style_metrics.csv"), metric_rows)
    tested_metrics = [metric for metric in metrics if metric != "style_truncated"]
    pairwise = style_pairwise_tests(
        rows,
        group_key,
        tested_metrics,
        permutation_samples,
        seed,
        metric_profile,
    )
    write_csv(os.path.join(output_dir, f"{prefix}_style_pairwise_tests.csv"), pairwise)
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key, "unknown")), []).append(row)
    summary = {
        "metric_profile": metric_profile,
        "reported_metrics": metrics,
        "fdr_policy": "Benjamini-Hochberg within prespecified metric families; quality diagnostics are not tested.",
        "global": summarize_style_rows(rows, metrics),
        "groups": {
            group: summarize_style_rows(group_rows, metrics)
            for group, group_rows in sorted(groups.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]))
        },
        "pairwise_tests_file": f"{prefix}_style_pairwise_tests.csv",
        "metrics_file": f"{prefix}_style_metrics.csv",
    }
    write_json(os.path.join(output_dir, f"{prefix}_style_summary.json"), summary)
    return summary


def plot_style_metrics(
    output_dir: str,
    rows: List[Dict],
    prefix: str,
    metric_profile: str,
) -> None:
    import matplotlib.pyplot as plt

    apply_nmi_style()
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if metric_profile == "math":
        metrics = [
            "generated_tokens",
            "style_line_count",
            "style_equation_line_count",
            "style_step_marker_count",
            "style_self_correction_marker_count",
            "correct",
        ]
    elif metric_profile == "safety":
        metrics = [
            "generated_tokens",
            "style_refusal_score",
            "style_redirect_score",
            "style_policy_language_score",
            "style_hedging_score",
            "style_harmful_detail_score",
        ]
    else:
        metrics = [
            "generated_tokens",
            "style_paragraph_count",
            "style_bullet_line_count",
            "style_explanation_marker_count",
            "style_hedging_score",
            "style_second_person_count",
        ]
    clusters = sorted({int(row.get("activation_cluster", -1)) for row in rows if row.get("activation_cluster") is not None})
    if not clusters:
        return
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_IN, 4.4))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        values = [
            [
                float(row[metric]) for row in rows
                if int(row.get("activation_cluster", -1)) == cluster
                and row.get(metric) is not None
                and math.isfinite(float(row[metric]))
            ]
            for cluster in clusters
        ]
        boxes = ax.boxplot(
            values,
            labels=[str(cluster) for cluster in clusters],
            showfliers=False,
            widths=0.58,
            patch_artist=True,
            medianprops={"color": COLORS["black"], "linewidth": 1.0},
            whiskerprops={"color": COLORS["gray"], "linewidth": 0.8},
            capprops={"color": COLORS["gray"], "linewidth": 0.8},
        )
        for patch, cluster in zip(boxes["boxes"], clusters):
            patch.set_facecolor(cluster_color(cluster))
            patch.set_alpha(0.72)
            patch.set_edgecolor("none")
        ax.set_ylabel(metric_label(metric))
        ax.set_xlabel("Cluster")
        style_axis(ax)
    for axis in axes[len(metrics) :]:
        axis.axis("off")
    fig.tight_layout()
    save_figure(fig, os.path.join(plot_dir, f"{prefix}_style_by_cluster"))
    plt.close(fig)

    if prefix == "baseline" and len(clusters) >= 2:
        heatmap_metrics = metrics_for_profile(metric_profile)
        matrix = []
        for cluster in clusters:
            row_values = []
            for metric in heatmap_metrics:
                all_values = [
                    float(row[metric]) for row in rows
                    if row.get(metric) is not None and math.isfinite(float(row[metric]))
                ]
                cluster_values = [
                    float(row[metric])
                    for row in rows
                    if int(row.get("activation_cluster", -1)) == cluster
                    and row.get(metric) is not None
                    and math.isfinite(float(row[metric]))
                ]
                global_mean = safe_mean(all_values) or 0.0
                global_std = safe_std(all_values) or 1.0
                cluster_mean = safe_mean(cluster_values)
                row_values.append(((cluster_mean if cluster_mean is not None else global_mean) - global_mean) / max(global_std, 1e-6))
            matrix.append(row_values)
        limit = max(1.0, min(2.5, max(abs(value) for row in matrix for value in row)))
        figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, max(2.0, 0.42 * len(clusters) + 1.2)))
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label("Standardized mean difference")
        axis.set_xticks(
            range(len(heatmap_metrics)),
            [metric_label(metric) for metric in heatmap_metrics],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(len(clusters)), [f"Cluster {cluster}" for cluster in clusters])
        figure.tight_layout()
        save_figure(figure, os.path.join(plot_dir, "baseline_cluster_style_heatmap"))
        plt.close(figure)


INTERVENTION_STYLE_METRICS = metrics_for_profile("safety")


def margin_strata_ids(
    ids: Sequence[str], baseline_by_id: Dict[str, Dict]
) -> Dict[str, List[str]]:
    """Split paired samples into equal-size strata by initial cluster confidence."""
    valid = [
        sample_id
        for sample_id in ids
        if baseline_by_id.get(sample_id, {}).get("cluster_margin_normalized")
        is not None
        and math.isfinite(
            float(baseline_by_id[sample_id]["cluster_margin_normalized"])
        )
    ]
    ordered = sorted(
        valid,
        key=lambda sample_id: float(
            baseline_by_id[sample_id]["cluster_margin_normalized"]
        ),
    )
    if len(ordered) < 6:
        return {"all": list(ids)}
    first = len(ordered) // 3
    second = 2 * len(ordered) // 3
    return {
        "all": list(ids),
        "low": ordered[:first],
        "mid": ordered[first:second],
        "high": ordered[second:],
    }


def margin_stratum_summary(
    ids: Sequence[str], baseline_by_id: Dict[str, Dict]
) -> Dict[str, Any]:
    values = [
        float(baseline_by_id[sample_id]["cluster_margin_normalized"])
        for sample_id in ids
        if baseline_by_id.get(sample_id, {}).get("cluster_margin_normalized")
        is not None
        and math.isfinite(
            float(baseline_by_id[sample_id]["cluster_margin_normalized"])
        )
    ]
    return {
        "margin_n": len(values),
        "margin_mean": safe_mean(values),
        "margin_min": min(values) if values else None,
        "margin_max": max(values) if values else None,
    }


def paired_style_populations(
    left_by_id: Dict[str, Dict], right_by_id: Dict[str, Dict]
) -> Dict[str, List[str]]:
    matched = sorted(set(left_by_id) & set(right_by_id))
    untruncated = [
        sample_id
        for sample_id in matched
        if float(left_by_id[sample_id].get("style_truncated") or 0.0) == 0.0
        and float(right_by_id[sample_id].get("style_truncated") or 0.0) == 0.0
    ]
    return {"all_paired": matched, "both_untruncated": untruncated}


def adjust_style_fdr(rows: List[Dict], metric_profile: str) -> None:
    families = sorted(
        {
            (
                str(row.get("analysis_population")),
                str(row.get("margin_stratum")),
                family_for_metric(str(row.get("metric")), metric_profile),
            )
            for row in rows
        }
    )
    for population, stratum, metric_family in families:
        indices = [
            index
            for index, row in enumerate(rows)
            if row.get("analysis_population") == population
            and row.get("margin_stratum") == stratum
            and family_for_metric(str(row.get("metric")), metric_profile)
            == metric_family
        ]
        q_values = bh_adjust(
            [rows[index].get("signflip_two_sided_p") for index in indices]
        )
        for index, q_value in zip(indices, q_values):
            rows[index]["fdr_family"] = metric_family
            rows[index]["fdr_family_size"] = len(indices)
            rows[index]["bh_fdr_q"] = q_value
            p_value = rows[index].get("signflip_two_sided_p")
            rows[index]["significant_p_0p05"] = bool(
                p_value is not None and math.isfinite(float(p_value)) and float(p_value) < 0.05
            )


def intervention_style_effect_rows(
    all_rows: Dict[str, List[Dict]],
    specs_by_name: Dict[str, InterventionSpec],
    bootstrap_samples: int = 0,
    permutation_tests: int = 0,
    seed: int = 42,
    metric_profile: str = "all",
) -> List[Dict]:
    metrics = metrics_for_profile(metric_profile)
    baseline = {row["id"]: row for row in all_rows.get("baseline", [])}
    baseline_std = {
        metric: max(
            safe_std([float(row[metric]) for row in baseline.values() if row.get(metric) is not None]) or 1.0,
            1e-6,
        )
        for metric in metrics
    }
    output: List[Dict] = []
    for mode, rows in all_rows.items():
        if mode == "baseline" or mode not in specs_by_name:
            continue
        spec = specs_by_name[mode]
        mode_by_id = {
            str(row["id"]): row
            for row in rows
            if int(row.get("intervention_feature_cluster", -1))
            == int(spec.source_cluster)
        }
        for population, population_ids in paired_style_populations(
            mode_by_id, baseline
        ).items():
            for stratum, analysis_ids in margin_strata_ids(
                population_ids, baseline
            ).items():
                for metric_index, metric in enumerate(metrics):
                    deltas = [
                        float(mode_by_id[sample_id][metric])
                        - float(baseline[sample_id][metric])
                        for sample_id in analysis_ids
                        if mode_by_id[sample_id].get(metric) is not None
                        and baseline[sample_id].get(metric) is not None
                        and math.isfinite(float(mode_by_id[sample_id][metric]))
                        and math.isfinite(float(baseline[sample_id][metric]))
                    ]
                    result = paired_bootstrap_and_signflip(
                        deltas,
                        bootstrap_samples,
                        permutation_tests,
                        seed + 1009 * len(output) + metric_index,
                    )
                    mean_delta = result.get("mean")
                    output.append(
                        {
                            "mode": mode,
                            "control": spec.control,
                            "source_cluster": spec.source_cluster,
                            "target_cluster": spec.target_cluster,
                            "alpha": spec.alpha,
                            "analysis_population": population,
                            "margin_stratum": stratum,
                            **margin_stratum_summary(analysis_ids, baseline),
                            "metric": metric,
                            "paired_mean_delta": mean_delta,
                            "paired_median_delta": safe_median(deltas),
                            "standardized_mean_delta": (
                                float(mean_delta) / baseline_std[metric]
                                if mean_delta is not None
                                else None
                            ),
                            **result,
                        }
                    )
    adjust_style_fdr(output, metric_profile)
    return output


def intervention_style_contrast_rows(
    all_rows: Dict[str, List[Dict]],
    specs_by_name: Dict[str, InterventionSpec],
    primary_mode_filter: str,
    bootstrap_samples: int,
    permutation_tests: int,
    seed: int,
    metric_profile: str,
) -> List[Dict]:
    if not primary_mode_filter:
        return []
    baseline = {row["id"]: row for row in all_rows.get("baseline", [])}
    output: List[Dict] = []
    primary_modes = [
        mode
        for mode in all_rows
        if mode in specs_by_name and primary_mode_filter in mode
    ]
    for primary_mode in primary_modes:
        primary_spec = specs_by_name[primary_mode]
        primary_by_id = {
            str(row["id"]): row
            for row in all_rows[primary_mode]
            if int(row.get("intervention_feature_cluster", -1))
            == int(primary_spec.source_cluster)
        }
        for control_mode, control_rows in all_rows.items():
            if control_mode in {"baseline", primary_mode} or control_mode not in specs_by_name:
                continue
            control_spec = specs_by_name[control_mode]
            if (
                int(control_spec.source_cluster),
                int(control_spec.target_cluster),
                float(control_spec.alpha),
            ) != (
                int(primary_spec.source_cluster),
                int(primary_spec.target_cluster),
                float(primary_spec.alpha),
            ):
                continue
            control_by_id = {
                str(row["id"]): row
                for row in control_rows
                if int(row.get("intervention_feature_cluster", -1))
                == int(control_spec.source_cluster)
            }
            for population, population_ids in paired_style_populations(
                primary_by_id, control_by_id
            ).items():
                for stratum, analysis_ids in margin_strata_ids(
                    population_ids, baseline
                ).items():
                    for metric_index, metric in enumerate(
                        metrics_for_profile(metric_profile)
                    ):
                        deltas = [
                            float(primary_by_id[sample_id][metric])
                            - float(control_by_id[sample_id][metric])
                            for sample_id in analysis_ids
                            if primary_by_id[sample_id].get(metric) is not None
                            and control_by_id[sample_id].get(metric) is not None
                            and math.isfinite(float(primary_by_id[sample_id][metric]))
                            and math.isfinite(float(control_by_id[sample_id][metric]))
                        ]
                        result = paired_bootstrap_and_signflip(
                            deltas,
                            bootstrap_samples,
                            permutation_tests,
                            seed + 20011 + 1009 * len(output) + metric_index,
                        )
                        output.append(
                            {
                                "primary_mode": primary_mode,
                                "control_mode": control_mode,
                                "comparison": f"{primary_mode}_minus_{control_mode}",
                                "source_cluster": primary_spec.source_cluster,
                                "target_cluster": primary_spec.target_cluster,
                                "alpha": primary_spec.alpha,
                                "analysis_population": population,
                                "margin_stratum": stratum,
                                **margin_stratum_summary(analysis_ids, baseline),
                                "metric": metric,
                                "paired_mean_delta": result.get("mean"),
                                "paired_median_delta": safe_median(deltas),
                                **result,
                            }
                        )
    adjust_style_fdr(output, metric_profile)
    return output


def plot_intervention_effects(
    output_dir: str,
    summary: Dict[str, Any],
    effect_rows: List[Dict],
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    modes = list(summary.get("modes", {}).keys())
    if not modes:
        return
    plot_effect_rows = [
        row
        for row in effect_rows
        if row.get("analysis_population", "all_paired") == "all_paired"
        and row.get("margin_stratum", "all") == "all"
    ]

    def short_label(mode: str) -> str:
        item = summary["modes"][mode]
        return (
            f"{item.get('control', 'main')} a={item.get('alpha')} "
            f"{item.get('source_cluster')}->{item.get('target_cluster')}"
        )

    labels = [short_label(mode) for mode in modes]
    y = list(range(len(modes)))
    changed_text = [float(summary["modes"][mode].get("changed_text_rate") or 0.0) for mode in modes]
    changed_answer = [float(summary["modes"][mode].get("changed_answer_rate") or 0.0) for mode in modes]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(modes))))
    ax.barh([value - 0.18 for value in y], changed_text, height=0.34, label="changed text")
    ax.barh([value + 0.18 for value in y], changed_answer, height=0.34, label="changed parsed answer")
    ax.set_yticks(y, labels)
    ax.set_xlabel("rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "intervention_response_change_rates.png"), dpi=180)
    plt.close(fig)

    movements = []
    for mode in modes:
        shift = summary["modes"][mode].get("source_cluster_style_shift_to_target", {})
        movements.append(float(shift.get("mean_delta_toward_target") or 0.0))
    colors = ["#2f855a" if value > 0 else "#b44d4d" for value in movements]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(modes))))
    ax.barh(y, movements, color=colors)
    ax.axvline(0.0, color="#222222", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("standardized style movement toward target cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "intervention_style_movement_to_target.png"), dpi=180)
    plt.close(fig)

    metrics = []
    for row in plot_effect_rows:
        if row["metric"] not in metrics:
            metrics.append(row["metric"])
    by_key = {(row["mode"], row["metric"]): row for row in plot_effect_rows}
    matrix = [
        [float(by_key.get((mode, metric), {}).get("standardized_mean_delta") or 0.0) for metric in metrics]
        for mode in modes
    ]
    plt.figure(figsize=(max(11, 0.9 * len(metrics)), max(5, 0.42 * len(modes))))
    plt.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="paired standardized mean change")
    plt.xticks(range(len(metrics)), metrics, rotation=40, ha="right")
    plt.yticks(range(len(modes)), labels)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "intervention_style_effect_heatmap.png"), dpi=180)
    plt.close()


def cuda_memory_snapshot(tag: str) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"tag": tag, "cuda_available": False}
    device = torch.cuda.current_device()
    return {
        "tag": tag,
        "cuda_available": True,
        "device": int(device),
        "allocated_gb": float(torch.cuda.memory_allocated(device) / (1024 ** 3)),
        "reserved_gb": float(torch.cuda.memory_reserved(device) / (1024 ** 3)),
        "max_allocated_gb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
    }


def maybe_log_cuda(args: argparse.Namespace, tag: str) -> None:
    if args.log_cuda_memory:
        print(json.dumps(cuda_memory_snapshot(tag), ensure_ascii=False), flush=True)


def progress_accuracy_snapshot(rows: List[Dict], mode: str, batch_idx: int, total_batches: int) -> Dict[str, Any]:
    def summarize(current_rows: List[Dict]) -> Dict[str, Any]:
        summary = summarize_accuracy_rows(current_rows)
        generated_tokens = [int(row.get("generated_tokens") or 0) for row in current_rows]
        truncated_n = sum(1 for row in current_rows if float(row.get("style_truncated") or 0.0) > 0.0)
        valid_choice_n = sum(1 for row in current_rows if row.get("valid_choice"))
        summary.update(
            {
                "generated_tokens_mean": safe_mean(generated_tokens),
                "truncated_n": truncated_n,
                "truncated_rate": truncated_n / max(len(current_rows), 1),
                "valid_choice_n": valid_choice_n,
                "valid_choice_rate": valid_choice_n / max(len(current_rows), 1),
            }
        )
        return summary

    clusters = sorted({int(row.get("activation_cluster", -1)) for row in rows})
    cluster_summaries = {
        str(cluster_id): summarize(
            [row for row in rows if int(row.get("activation_cluster", -1)) == cluster_id]
        )
        for cluster_id in clusters
    }
    valid_cluster_acc = [
        (cluster_id, summary.get("accuracy"))
        for cluster_id, summary in cluster_summaries.items()
        if summary.get("accuracy") is not None
    ]
    if valid_cluster_acc:
        low_cluster, low_acc = min(valid_cluster_acc, key=lambda item: item[1])
        high_cluster, high_acc = max(valid_cluster_acc, key=lambda item: item[1])
        accuracy_range = float(high_acc - low_acc)
    else:
        low_cluster = high_cluster = None
        accuracy_range = None
    return {
        "mode": mode,
        "batch_idx": int(batch_idx),
        "total_batches": int(total_batches),
        "generated_rows": int(len(rows)),
        "overall": summarize(rows),
        "clusters": cluster_summaries,
        "low_cluster": low_cluster,
        "high_cluster": high_cluster,
        "accuracy_range_high_minus_low": accuracy_range,
    }


def progress_accuracy_csv_rows(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    overall = snapshot.get("overall", {})
    rows.append(
        {
            "mode": snapshot.get("mode"),
            "batch_idx": snapshot.get("batch_idx"),
            "total_batches": snapshot.get("total_batches"),
            "generated_rows": snapshot.get("generated_rows"),
            "cluster": "all",
            "n": overall.get("n"),
            "correct_n": overall.get("correct_n"),
            "accuracy": overall.get("accuracy"),
            "parseable_n": overall.get("parseable_n"),
            "parse_rate": overall.get("parse_rate"),
            "accuracy_given_parse": overall.get("accuracy_given_parse"),
            "strict_hash_accuracy": overall.get("strict_hash_accuracy"),
            "strict_hash_parse_rate": overall.get("strict_hash_parse_rate"),
            "strict_hash_valid_choice_rate": overall.get("strict_hash_valid_choice_rate"),
            "valid_choice_rate": overall.get("valid_choice_rate"),
            "generated_tokens_mean": overall.get("generated_tokens_mean"),
            "truncated_rate": overall.get("truncated_rate"),
            "accuracy_range_high_minus_low": snapshot.get("accuracy_range_high_minus_low"),
            "low_cluster": snapshot.get("low_cluster"),
            "high_cluster": snapshot.get("high_cluster"),
        }
    )
    for cluster_id, summary in sorted(snapshot.get("clusters", {}).items(), key=lambda item: int(item[0])):
        rows.append(
            {
                "mode": snapshot.get("mode"),
                "batch_idx": snapshot.get("batch_idx"),
                "total_batches": snapshot.get("total_batches"),
                "generated_rows": snapshot.get("generated_rows"),
                "cluster": cluster_id,
                "n": summary.get("n"),
                "correct_n": summary.get("correct_n"),
                "accuracy": summary.get("accuracy"),
                "parseable_n": summary.get("parseable_n"),
                "parse_rate": summary.get("parse_rate"),
                "accuracy_given_parse": summary.get("accuracy_given_parse"),
                "strict_hash_accuracy": summary.get("strict_hash_accuracy"),
                "strict_hash_parse_rate": summary.get("strict_hash_parse_rate"),
                "strict_hash_valid_choice_rate": summary.get("strict_hash_valid_choice_rate"),
                "valid_choice_rate": summary.get("valid_choice_rate"),
                "generated_tokens_mean": summary.get("generated_tokens_mean"),
                "truncated_rate": summary.get("truncated_rate"),
                "accuracy_range_high_minus_low": snapshot.get("accuracy_range_high_minus_low"),
                "low_cluster": snapshot.get("low_cluster"),
                "high_cluster": snapshot.get("high_cluster"),
            }
        )
    return rows


def write_progress_accuracy(args: argparse.Namespace, snapshot: Dict[str, Any], append_csv: bool) -> None:
    write_json(os.path.join(args.output_dir, f"{snapshot['mode']}_progress_accuracy.json"), snapshot)
    csv_path = os.path.join(args.output_dir, f"{snapshot['mode']}_progress_accuracy.csv")
    rows = progress_accuracy_csv_rows(snapshot)
    if not append_csv or not os.path.exists(csv_path):
        write_csv(csv_path, rows)
        return
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerows(rows)


def write_progress_examples(args: argparse.Namespace, mode: str, rows: List[Dict]) -> None:
    limit = int(getattr(args, "progress_example_count", 0) or 0)
    if limit <= 0 or not rows:
        return
    buckets = [
        [row for row in rows if float(row.get("style_truncated") or 0.0) > 0.0],
        [row for row in rows if not row.get("parse_success") and float(row.get("style_truncated") or 0.0) == 0.0],
        [row for row in rows if row.get("parse_success") and not row.get("correct")],
        [row for row in rows if row.get("correct")],
    ]
    selected = []
    seen = set()
    per_bucket = max(1, math.ceil(limit / len(buckets)))
    for bucket in buckets:
        for row in bucket[-per_bucket:]:
            sample_id = row.get("id")
            if sample_id in seen:
                continue
            seen.add(sample_id)
            selected.append(row)
    if len(selected) < limit:
        for row in reversed(rows):
            sample_id = row.get("id")
            if sample_id in seen:
                continue
            seen.add(sample_id)
            selected.append(row)
            if len(selected) >= limit:
                break
    write_jsonl(os.path.join(args.output_dir, f"{mode}_progress_examples.jsonl"), selected[:limit])


def generate_rows(
    model,
    tokenizer,
    rows: List[Dict],
    context: GenerationContext,
    mode: str,
    args: argparse.Namespace,
    patch_builder: Optional[Callable[[List[Dict]], Tuple[torch.Tensor, torch.Tensor]]] = None,
    next_token_state: Optional[NextTokenAuditState] = None,
    intervention_spec: Optional[InterventionSpec] = None,
    checkpoint_file: Optional[str] = None,
) -> List[Dict]:
    context.mode = mode
    output_rows = []
    input_device = model_input_device(model)
    do_sample = args.temperature > 0.0
    total_batches = math.ceil(len(rows) / max(args.batch_size, 1))
    progress_every = int(getattr(args, "progress_accuracy_every", 0) or 0)
    prompt_confidence_logit_kwargs = (
        last_token_logits_kwargs(model)
        if args.record_prompt_confidence and not args.next_token_audit
        else {}
    )
    if args.record_prompt_confidence and not args.next_token_audit:
        print(
            "[generation-stage] prompt_confidence=enabled "
            f"last_token_logits={bool(prompt_confidence_logit_kwargs)}",
            flush=True,
        )
    progress_bar = tqdm(
        batch_iter(rows, args.batch_size),
        total=total_batches,
        desc=f"Generate {mode}",
        leave=True,
        disable=not sys.stderr.isatty(),
    )
    progress_csv_written = False
    for batch_idx, batch in enumerate(progress_bar, start=1):
        prompts = [build_generation_prompt(row, tokenizer, args) for row in batch]
        ids = [row["id"] for row in batch]
        context.current_ids = ids
        context.generated_step = 0
        context.current_step = -1
        if patch_builder is None:
            context.apply_mask = torch.zeros(len(batch), dtype=torch.bool)
            context.patch_values = None
        else:
            patch_values, apply_mask = patch_builder(batch)
            context.apply_mask = apply_mask
            context.patch_values = patch_values

        exact_prompt_ids = [row.get("prompt_token_ids") for row in batch]
        if all(
            isinstance(values, list) and values
            for values in exact_prompt_ids
        ):
            sequences = [
                [int(value) for value in values][-int(args.max_length) :]
                for values in exact_prompt_ids
            ]
            width = max(len(values) for values in sequences)
            pad_id = int(tokenizer.pad_token_id)
            encoded = {
                "input_ids": torch.tensor(
                    [[pad_id] * (width - len(values)) + values for values in sequences],
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    [[0] * (width - len(values)) + [1] * len(values) for values in sequences],
                    dtype=torch.long,
                ),
            }
        elif any(values is not None for values in exact_prompt_ids):
            raise ValueError(
                "A generation batch mixes exact prompt-token rows with text-only rows."
            )
        else:
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
        prompt_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        mask = context.apply_mask if context.apply_mask is not None else torch.zeros(len(batch), dtype=torch.bool)
        prompt_confidence_metrics: List[Dict[str, Any]] = [
            {} for _ in batch
        ]
        if args.record_prompt_confidence and not args.next_token_audit:
            confidence_started_at = time.monotonic()
            if batch_idx == 1:
                tqdm.write(
                    "[generate-progress] "
                    f"mode={mode} batch={batch_idx}/{total_batches} "
                    f"phase=prompt_confidence prompt_width={encoded['input_ids'].size(1)} "
                    f"last_token_logits={bool(prompt_confidence_logit_kwargs)} "
                    "status=started"
                )
            with torch.inference_mode():
                confidence_outputs = model(
                    **encoded,
                    use_cache=False,
                    **prompt_confidence_logit_kwargs,
                )
                confidence_logits = confidence_outputs.logits[:, -1, :]
            prompt_confidence_metrics = describe_prompt_confidence(
                confidence_logits, tokenizer
            )
            confidence_elapsed = time.monotonic() - confidence_started_at
            confidence_checkpoint = (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % max(1, int(math.ceil(total_batches / 20.0))) == 0
            )
            if confidence_checkpoint or confidence_elapsed >= max(
                float(args.generation_progress_seconds), 10.0
            ):
                tqdm.write(
                    "[generate-progress] "
                    f"mode={mode} batch={batch_idx}/{total_batches} "
                    f"batch_pct={100.0 * batch_idx / max(total_batches, 1):.1f} "
                    "phase=prompt_confidence "
                    f"elapsed={confidence_elapsed:.1f}s status=complete"
                )
            del confidence_outputs, confidence_logits
        next_token_metrics: List[Dict[str, Any]] = [{} for _ in batch]
        if args.next_token_audit:
            if next_token_state is None:
                raise RuntimeError("--next-token-audit requires an initialized audit state")
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False)
                mode_logits = outputs.logits[:, -1, :]
            if mode == "baseline":
                next_token_state.record_baseline(
                    ids,
                    [int(row.get("activation_cluster", -1)) for row in batch],
                    mode_logits,
                )
                next_token_metrics = next_token_state.describe_logits(
                    mode_logits, args.next_token_top_k
                )
            else:
                if intervention_spec is None:
                    raise RuntimeError("Next-token intervention mode is missing its spec")
                next_token_metrics = next_token_state.compare(
                    ids,
                    mode_logits,
                    intervention_spec.source_cluster,
                    intervention_spec.target_cluster,
                    intervention_spec.control != "shuffled_labels",
                    args.next_token_top_k,
                )
            generated_token_ids = mode_logits.argmax(dim=-1, keepdim=True)
            decoded = tokenizer.batch_decode(
                generated_token_ids, skip_special_tokens=False
            )
            generated_token_counts = [1 for _ in batch]
            del outputs, mode_logits
        else:
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "use_cache": True,
            }
            logits_processors: List[LogitsProcessor] = []
            generation_heartbeat = None
            if args.generation_progress_seconds > 0.0:
                generation_heartbeat = GenerationProgressHeartbeat(
                    mode=mode,
                    batch_index=batch_idx,
                    total_batches=total_batches,
                    batch_size=len(batch),
                    prompt_width=int(encoded["input_ids"].size(1)),
                    max_new_tokens=int(args.max_new_tokens),
                    interval_seconds=float(args.generation_progress_seconds),
                )
                generation_heartbeat.start()
                logits_processors.append(generation_heartbeat)
            logprob_recorder = None
            if args.save_generated_token_logprobs:
                eos_ids = generation_eos_token_id(model, tokenizer)
                if eos_ids is None:
                    eos_ids = []
                elif isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                logprob_recorder = GeneratedTokenLogprobRecorder(len(batch), eos_ids)
                logits_processors.append(logprob_recorder)
            if logits_processors:
                generation_kwargs["logits_processor"] = LogitsProcessorList(
                    logits_processors
                )
            eos_token_id = generation_eos_token_id(model, tokenizer)
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_kwargs)
            if generation_heartbeat is not None:
                generation_heartbeat.finalize(
                    int(generated.size(1) - encoded["input_ids"].size(1))
                )
            if logprob_recorder is not None:
                logprob_recorder.finalize(generated)
                generated_logprob_metrics = logprob_recorder.summaries(
                    args.token_logprob_checkpoints
                )
            else:
                generated_logprob_metrics = [{} for _ in batch]
            prompt_width = encoded["input_ids"].size(1)
            generated_token_ids = generated[:, prompt_width:]
            decoded = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            generated_token_counts = (
                (generated_token_ids != tokenizer.pad_token_id)
                .sum(dim=1)
                .detach()
                .cpu()
                .tolist()
            )
        if args.next_token_audit:
            generated_logprob_metrics = [{} for _ in batch]
        generated_token_lists = generated_token_ids.detach().cpu().tolist()
        batch_output_start = len(output_rows)
        for row, prompt, prompt_tokens, generated_tokens, generated_ids, text, applied, confidence_metrics, token_metrics, logprob_metrics in zip(
            batch,
            prompts,
            prompt_token_counts,
            generated_token_counts,
            generated_token_lists,
            decoded,
            mask.tolist(),
            prompt_confidence_metrics,
            next_token_metrics,
            generated_logprob_metrics,
        ):
            answer_eval = evaluate_generated_answer(text, row, args.answer_extraction, args.answer_tolerance)
            out_row = {
                "id": row["id"],
                "eval_scope": row.get("eval_scope"),
                "source_split": row.get("source_split"),
                "eval_index": row.get("eval_index"),
                "val_index": row.get("val_index"),
                "meta_group": row.get("meta_group"),
                "activation_cluster": row.get("activation_cluster"),
                "activation_cluster_name": row.get("activation_cluster_name"),
                "cluster_own_distance": row.get("cluster_own_distance"),
                "cluster_nearest_other_distance": row.get(
                    "cluster_nearest_other_distance"
                ),
                "cluster_margin": row.get("cluster_margin"),
                "cluster_margin_normalized": row.get(
                    "cluster_margin_normalized"
                ),
                "mode": mode,
                "intervention_applied": bool(applied and mode != "baseline"),
                "question": row.get("question"),
                "source": row.get("source"),
                "options": row.get("options"),
                "correct_option_number": row.get("correct_option_number"),
                "correct_letter": row.get("correct_letter"),
                "prompt": prompt,
                "source_prompt": row.get("prompt"),
                "prompt_style": args.prompt_style,
                "use_chat_template": args.use_chat_template,
                "prompt_tokens": int(prompt_tokens),
                "generated_tokens": int(generated_tokens),
                "generated_text": text,
                **answer_eval,
                **confidence_metrics,
                **token_metrics,
                **logprob_metrics,
            }
            out_row.update(compute_style_metrics(text, int(generated_tokens), args.max_new_tokens))
            if args.save_generated_token_ids:
                out_row["generated_token_ids_json"] = json.dumps(
                    [int(value) for value in generated_ids[: int(generated_tokens)]],
                    ensure_ascii=False,
                )
            if args.next_token_audit:
                out_row["style_truncated"] = 0.0
            output_rows.append(out_row)
        if checkpoint_file and len(output_rows) > batch_output_start:
            os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)
            with open(checkpoint_file, "a", encoding="utf-8") as handle:
                for checkpoint_row in output_rows[batch_output_start:]:
                    handle.write(
                        json.dumps(checkpoint_row, ensure_ascii=False, allow_nan=True)
                        + "\n"
                    )
                handle.flush()
        context.current_ids = []
        context.apply_mask = None
        context.patch_values = None
        if args.next_token_audit:
            del encoded, generated_token_ids
        else:
            del encoded, generated, generated_token_ids
        if args.clear_cuda_cache_every > 0 and batch_idx % args.clear_cuda_cache_every == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        maybe_log_cuda(args, f"{mode}:batch:{batch_idx}")
        progress_bar.set_postfix(
            rows=len(output_rows),
            last_batch_tokens=max(generated_token_counts, default=0),
            refresh=True,
        )
        if progress_every > 0 and output_rows and (batch_idx % progress_every == 0 or batch_idx == total_batches):
            snapshot = progress_accuracy_snapshot(output_rows, mode, batch_idx, total_batches)
            write_progress_accuracy(args, snapshot, append_csv=progress_csv_written)
            write_progress_examples(args, mode, output_rows)
            progress_csv_written = True
            overall_acc = snapshot.get("overall", {}).get("accuracy")
            progress_bar.set_postfix(
                n=int(snapshot.get("generated_rows", 0)),
                acc="nan" if overall_acc is None else f"{float(overall_acc):.4f}",
                gap="nan"
                if snapshot.get("accuracy_range_high_minus_low") is None
                else f"{float(snapshot['accuracy_range_high_minus_low']):.4f}",
            )
    context.current_ids = []
    context.apply_mask = None
    context.patch_values = None
    context.generated_step = 0
    context.current_step = -1
    if args.clear_cuda_cache_between_modes and torch.cuda.is_available():
        torch.cuda.empty_cache()
    maybe_log_cuda(args, f"{mode}:done")
    return output_rows


def mode_name(spec: InterventionSpec) -> str:
    src = cluster_label_name(spec.source_cluster)
    dst = cluster_label_name(spec.target_cluster)
    alpha = str(spec.alpha).replace(".", "p")
    parts = [spec.cluster_result.name, f"{src}_to_{dst}", spec.patch_mode, f"a{alpha}"]
    if spec.control != "main":
        parts.append(spec.control)
    return "_".join(parts)


def build_intervention_specs(
    main_cluster: ClusterResult,
    controls: Dict[str, ClusterResult],
    args: argparse.Namespace,
) -> List[InterventionSpec]:
    transitions = parse_transitions(args.transitions, args.num_clusters)
    extreme_fraction = getattr(args, "extreme_prototype_fraction", 0.1)
    extreme_min_count = getattr(args, "extreme_prototype_min_count", 8)
    specs = []
    for alpha in args.alphas:
        for src, dst in transitions:
            specs.append(
                InterventionSpec(
                    name="",
                    cluster_result=main_cluster,
                    source_cluster=src,
                    target_cluster=dst,
                    alpha=alpha,
                    patch_mode=args.patch_mode,
                    control="main",
                    extreme_fraction=extreme_fraction,
                    extreme_min_count=extreme_min_count,
                )
            )
        for control_name, cluster_result in controls.items():
            for src, dst in transitions:
                specs.append(
                    InterventionSpec(
                        name="",
                        cluster_result=cluster_result,
                        source_cluster=src,
                        target_cluster=dst,
                        alpha=alpha,
                        patch_mode=args.patch_mode,
                        control=control_name,
                        extreme_fraction=extreme_fraction,
                        extreme_min_count=extreme_min_count,
                    )
                )
        for random_control in ("random_direction", "random_hidden_direction"):
            if random_control not in args.controls:
                continue
            for src, dst in transitions:
                specs.append(
                    InterventionSpec(
                        name="",
                        cluster_result=main_cluster,
                        source_cluster=src,
                        target_cluster=dst,
                        alpha=alpha,
                        patch_mode=args.patch_mode,
                        control=random_control,
                        extreme_fraction=extreme_fraction,
                        extreme_min_count=extreme_min_count,
                    )
                )
    for spec in specs:
        spec.name = mode_name(spec)
    if args.max_intervention_modes > 0:
        specs = specs[: args.max_intervention_modes]
    return specs


def write_intervention_summary(
    path: str,
    all_rows: Dict[str, List[Dict]],
    record_layers: Sequence[int],
    bootstrap_samples: int,
    style_permutation_samples: int,
    seed: int,
    specs_by_name: Dict[str, InterventionSpec],
    metric_profile: str,
) -> Dict:
    def task_summary(rows: List[Dict]) -> Dict:
        return (
            summarize_accuracy_rows(rows)
            if metric_profile == "math"
            else {"n": len(rows), "task_performance_not_reported": True}
        )

    def mode_summary(rows: List[Dict], mode_seed: int) -> Dict:
        value = summarize_mode(
            rows, baseline_by_id, record_layers, bootstrap_samples, mode_seed
        )
        if metric_profile == "math":
            return value
        excluded = {
            "correct_n", "accuracy", "parseable_n", "parse_rate",
            "unparseable_n", "baseline_correct_n", "baseline_accuracy",
            "accuracy_delta_vs_baseline", "changed_answer_rate",
            "correct_to_wrong", "wrong_to_correct", "net_correct_change",
            "mcnemar_exact_p", "accuracy_delta_ci95",
        }
        return {key: item for key, item in value.items() if key not in excluded}

    baseline_rows = all_rows["baseline"]
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    reported_metrics = metrics_for_profile(metric_profile, include_quality=True)
    style_metrics_for_shift = [
        metric for metric in metrics_for_profile(metric_profile)
        if metric not in {"style_hash_answer", "style_answer_marker"}
    ]
    normalizer = style_normalizer(baseline_rows, style_metrics_for_shift)
    style_centroids = style_centroids_by_cluster(baseline_rows, normalizer, style_metrics_for_shift)
    summary = {
        "baseline": task_summary(baseline_rows),
        "metric_profile": metric_profile,
        "baseline_style": summarize_style_rows(baseline_rows, reported_metrics),
        "baseline_clusters": {},
        "baseline_cluster_style": {},
        "baseline_style_pairwise_tests_file": "baseline_style_pairwise_tests.csv",
        "modes": {},
        "mode_clusters": {},
    }
    for cluster_id in sorted({int(row["activation_cluster"]) for row in baseline_rows}):
        cluster_rows = [row for row in baseline_rows if int(row["activation_cluster"]) == cluster_id]
        summary["baseline_clusters"][str(cluster_id)] = task_summary(cluster_rows)
        summary["baseline_cluster_style"][str(cluster_id)] = summarize_style_rows(
            cluster_rows, reported_metrics
        )

    for mode, rows in all_rows.items():
        if mode == "baseline":
            continue
        current_mode_summary = mode_summary(rows, seed)
        current_mode_summary["style"] = summarize_style_rows(rows, reported_metrics)
        spec = specs_by_name.get(mode)
        if spec is not None:
            current_mode_summary["source_cluster"] = spec.source_cluster
            current_mode_summary["target_cluster"] = spec.target_cluster
            current_mode_summary["alpha"] = spec.alpha
            current_mode_summary["patch_mode"] = spec.patch_mode
            current_mode_summary["control"] = spec.control
            source_rows = [row for row in rows if int(row.get("intervention_feature_cluster", -1)) == spec.source_cluster]
            current_mode_summary["source_cluster_summary"] = mode_summary(source_rows, seed)
            current_mode_summary["source_cluster_style"] = summarize_style_rows(
                source_rows, reported_metrics
            )
            current_mode_summary["source_cluster_style_shift_to_target"] = summarize_style_shift_to_target(
                source_rows,
                baseline_by_id,
                style_centroids,
                normalizer,
                style_metrics_for_shift,
                spec.source_cluster,
                spec.target_cluster,
            )
        summary["modes"][mode] = current_mode_summary
        for cluster_id in sorted({int(row["activation_cluster"]) for row in rows}):
            cluster_rows = [row for row in rows if int(row["activation_cluster"]) == cluster_id]
            summary["mode_clusters"].setdefault(mode, {})[str(cluster_id)] = mode_summary(
                cluster_rows, seed
            )
            summary["mode_clusters"].setdefault(mode, {})[str(cluster_id)]["style"] = summarize_style_rows(
                cluster_rows, reported_metrics
            )
    write_json(path, summary)
    return summary


def plot_clusters(output_dir: str, cluster_result: ClusterResult, rows: List[Dict], baseline_rows: Optional[List[Dict]]) -> None:
    import matplotlib.pyplot as plt

    apply_nmi_style()
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    coords = cluster_result.pca2
    labels = cluster_result.labels
    figure, axis = plt.subplots(figsize=(3.5, 3.05))
    for cluster_id in range(cluster_result.raw_centroids.shape[0]):
        mask = labels == cluster_id
        axis.scatter(
            coords[mask, 0].numpy(),
            coords[mask, 1].numpy(),
            s=8,
            alpha=0.45,
            color=cluster_color(cluster_id),
            edgecolors="none",
            rasterized=True,
            label=f"Cluster {cluster_id} (n={int(mask.sum())})",
        )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(markerscale=1.5)
    style_axis(axis, grid=None)
    figure.tight_layout()
    save_figure(figure, os.path.join(plot_dir, f"{cluster_result.name}_cluster_pca"))
    plt.close(figure)

    if baseline_rows:
        baseline_by_id = {row["id"]: row for row in baseline_rows}
        accuracies = []
        correct_counts = []
        names = []
        counts = []
        for cluster_id in range(cluster_result.raw_centroids.shape[0]):
            cluster_ids = [rows[idx]["id"] for idx in torch.nonzero(labels == cluster_id, as_tuple=False).flatten().tolist()]
            cluster_baseline = [baseline_by_id[sample_id] for sample_id in cluster_ids if sample_id in baseline_by_id]
            correct_n = sum(1 for row in cluster_baseline if row.get("correct"))
            correct_counts.append(correct_n)
            accuracies.append(correct_n / max(len(cluster_baseline), 1))
            counts.append(len(cluster_baseline))
            names.append(str(cluster_id))
        intervals = [wilson_interval(correct, count) for correct, count in zip(correct_counts, counts)]
        lower = [max(0.0, acc - interval[0]) for acc, interval in zip(accuracies, intervals)]
        upper = [max(0.0, interval[1] - acc) for acc, interval in zip(accuracies, intervals)]
        figure, axis = plt.subplots(figsize=(3.5, 2.8))
        axis.bar(
            names,
            accuracies,
            color=[cluster_color(index) for index in range(len(names))],
            width=0.62,
            yerr=[lower, upper],
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "ecolor": COLORS["black"]},
            zorder=3,
        )
        for idx, (acc, count) in enumerate(zip(accuracies, counts)):
            axis.text(idx, min(0.98, acc + upper[idx] + 0.025), f"n={count}", ha="center", va="bottom", fontsize=7)
        axis.set_ylim(0, min(1.0, max([acc + err for acc, err in zip(accuracies, upper)] + [0.05]) + 0.14))
        axis.set_xlabel("Activation cluster")
        axis.set_ylabel("Baseline accuracy")
        style_axis(axis)
        figure.tight_layout()
        save_figure(figure, os.path.join(plot_dir, f"{cluster_result.name}_cluster_accuracy"))
        plt.close(figure)

    centroid = cluster_result.raw_centroids
    limit = max(float(centroid.abs().max().item()), 1e-6)
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.4))
    image = axis.imshow(centroid.numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
    colorbar.set_label("Mean raw activation")
    axis.set_xlabel("Selected feature rank")
    axis.set_ylabel("Cluster")
    figure.tight_layout()
    save_figure(figure, os.path.join(plot_dir, f"{cluster_result.name}_centroid_heatmap"))
    plt.close(figure)


def load_baseline_file(path: str, eval_rows: List[Dict], args: argparse.Namespace) -> List[Dict]:
    return align_baseline_rows(read_jsonl(path), eval_rows, args, require_all=True, source=path)


def align_baseline_rows(
    source_rows: List[Dict],
    eval_rows: List[Dict],
    args: argparse.Namespace,
    require_all: bool,
    source: str,
) -> List[Dict]:
    by_id = {str(row["id"]): row for row in source_rows if row.get("id") is not None}
    expected_ids = [str(row["id"]) for row in eval_rows]
    missing = [sample_id for sample_id in expected_ids if sample_id not in by_id]
    if require_all and missing:
        raise ValueError(
            f"Baseline reuse file {source} misses {len(missing)}/{len(expected_ids)} "
            f"requested ids; examples: {missing[:5]}"
        )
    rows = []
    for eval_row in eval_rows:
        if eval_row["id"] not in by_id:
            continue
        row = dict(by_id[eval_row["id"]])
        row["activation_cluster"] = eval_row.get("activation_cluster")
        row["activation_cluster_name"] = eval_row.get("activation_cluster_name")
        for field in (
            "cluster_own_distance",
            "cluster_nearest_other_distance",
            "cluster_margin",
            "cluster_margin_normalized",
        ):
            row[field] = eval_row.get(field)
        eval_context = dict(eval_row)
        eval_context["source_prompt"] = eval_row.get("prompt") or row.get("source_prompt")
        eval_context["gold_answer"] = row.get("gold_answer") or normalize_number(eval_row.get("numeric_answer"))
        if row.get("answer_extraction") != args.answer_extraction or "parse_success" not in row:
            row.update(
                evaluate_generated_answer(
                    row.get("generated_text", ""),
                    eval_context,
                    args.answer_extraction,
                    args.answer_tolerance,
                )
            )
        ensure_style_metrics(row, args.max_new_tokens)
        rows.append(row)
    return rows


def read_generation_checkpoint(path: str) -> List[Dict]:
    rows: List[Dict] = []
    needs_repair = False
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return rows
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if line_index == len(lines) - 1:
                print(
                    f"[baseline-checkpoint] ignore incomplete trailing row: {path}",
                    flush=True,
                )
                needs_repair = True
                break
            raise
        if isinstance(value, dict) and value.get("id") is not None:
            rows.append(value)
    # A terminated process may have appended the same completed batch before
    # the shell retried it.  Last-write-wins deduplication keeps resume stable.
    deduplicated = {str(row["id"]): row for row in rows}
    deduplicated_rows = list(deduplicated.values())
    if len(deduplicated_rows) != len(rows):
        needs_repair = True
    if needs_repair:
        write_jsonl(path, deduplicated_rows)
    return deduplicated_rows


def effective_generated_patch_steps(args: argparse.Namespace) -> int:
    if args.generated_patch_steps is not None:
        return int(args.generated_patch_steps)
    return -1 if args.apply_to_generated else 0


def effective_apply_to_generated(args: argparse.Namespace) -> bool:
    return effective_generated_patch_steps(args) != 0


def truncated_rate(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if float(row.get("style_truncated") or 0.0) > 0.0) / len(rows))


def check_truncation(rows: List[Dict], max_rate: float, label: str) -> None:
    if max_rate >= 1.0:
        return
    rate = truncated_rate(rows)
    if rate > max_rate:
        raise RuntimeError(
            f"{label} truncated rate {rate:.4f} is above --max-truncated-rate {max_rate:.4f}. "
            "Increase --max-new-tokens or disable the threshold."
        )


def filter_untruncated_rows(rows: List[Dict], baseline_by_id: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    filtered = []
    for row in rows:
        if float(row.get("style_truncated") or 0.0) > 0.0:
            continue
        if baseline_by_id is not None:
            base = baseline_by_id.get(row.get("id"))
            if base is not None and float(base.get("style_truncated") or 0.0) > 0.0:
                continue
        filtered.append(row)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="Optional. If omitted, only clustering/static summaries are written.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--activation-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-set", choices=["selected", "random", "random_nonselected"], default="selected")
    parser.add_argument("--random-feature-count", type=int, default=0)
    parser.add_argument(
        "--cluster-source",
        choices=["true_i_selected", "e2_latent", "purified_meta", "purified_semantic", FIXED_CLUSTER_SOURCE],
        default="true_i_selected",
        help=(
            "Feature space used for clustering. With the default intervention space, prototypes are "
            "computed from selected i-layer neurons; --intervention-space refined_residual instead "
            "edits the post-selection residual code."
        ),
    )
    parser.add_argument("--decoupler-dir", default=None, help="Directory produced by train_decoupler.py; used for e2_latent/purified_meta/purified_semantic cluster sources.")
    parser.add_argument("--decoupler-checkpoint", default=None, help="Optional decoupler checkpoint path. Defaults to DECOUPLER_DIR/best_model.pt.")
    parser.add_argument("--decoupler-normalization", default=None, help="Optional normalization.pt path. Defaults to DECOUPLER_DIR/normalization.pt.")
    parser.add_argument("--purifier-checkpoint", default=None, help="Optional E2 recursive purifier checkpoint. Defaults to DECOUPLER_DIR/e2_recursive_purifier/best_purifier.pt.")
    parser.add_argument(
        "--intervention-space",
        choices=["neurons", "refined_residual"],
        default="neurons",
        help="Edit selected layer-i neurons or a post-selection refined residual code.",
    )
    parser.add_argument(
        "--refined-module-dir",
        default=None,
        help="Directory containing module_refiner.pt and refined soft-residual cluster artifacts.",
    )
    parser.add_argument(
        "--refined-code-intervention",
        choices=["direction", "centroid"],
        default="direction",
        help="Move along the source-to-target refined-code direction or toward the target centroid.",
    )
    parser.add_argument(
        "--refined-direct-steps",
        type=int,
        default=12,
        help="Projected hidden-space optimization steps for direct_opt.",
    )
    parser.add_argument(
        "--refined-direct-step-scale",
        type=float,
        default=1.5,
        help="Relative step multiplier inside the hidden-state norm ball.",
    )
    parser.add_argument(
        "--refined-direct-hidden-penalty",
        type=float,
        default=0.05,
        help="Penalty on relative hidden-state movement for direct_opt.",
    )
    parser.add_argument(
        "--refined-code-semantic-penalty",
        type=float,
        default=0.0,
        help="Penalty on relative purified-semantic drift during runtime correction.",
    )
    parser.add_argument(
        "--refined-code-z1-penalty",
        type=float,
        default=0.0,
        help="Penalty on relative E1 drift during runtime correction.",
    )
    parser.add_argument(
        "--refined-code-max-semantic-rel-delta",
        type=float,
        default=0.0,
        help="Reject correction candidates above this semantic drift; <=0 disables.",
    )
    parser.add_argument(
        "--refined-code-dose-match-controls",
        action="store_true",
        help=(
            "For refined-code controls, reuse the corresponding main intervention's "
            "accepted mask and per-sample achieved code/hidden dose."
        ),
    )
    parser.add_argument(
        "--refined-code-dose-semantic-tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum relative semantic-drift mismatch for strict dose matching; "
            "<=0 disables this check."
        ),
    )
    parser.add_argument(
        "--refined-code-dose-match-tolerance",
        type=float,
        default=0.25,
        help="Maximum relative code-L2 mismatch for strict dose-matched analysis.",
    )
    parser.add_argument(
        "--refined-code-dose-hidden-tolerance",
        type=float,
        default=0.5,
        help="Maximum relative hidden-delta mismatch; <=0 disables this strict check.",
    )
    parser.add_argument(
        "--refined-code-dose-min-direction-cosine",
        type=float,
        default=0.5,
        help="Minimum achieved/requested code-direction cosine for strict dose matching.",
    )
    parser.add_argument(
        "--refined-code-random-direction-mode",
        choices=["empirical_pca", "isotropic"],
        default="empirical_pca",
        help=(
            "Draw the orthogonal random control in the empirical runtime-code PCA "
            "subspace or from an isotropic Gaussian."
        ),
    )
    parser.add_argument(
        "--refined-code-random-seed",
        type=int,
        default=None,
        help=(
            "Seed used only to draw refined-code random directions. By default "
            "--seed is used. Set this independently to keep evaluation samples "
            "and the fitted refiner fixed while constructing a random-direction ensemble."
        ),
    )
    parser.add_argument(
        "--refined-code-random-dose-mode",
        choices=["strict_code", "hidden_semantic"],
        default="strict_code",
        help=(
            "Dose contract for random_direction. hidden_semantic matches only "
            "the main intervention's hidden norm and semantic drift, while "
            "strict_code retains the legacy code-L2 matching rule."
        ),
    )
    parser.add_argument(
        "--refined-code-random-axis-penalty",
        type=float,
        default=10.0,
        help=(
            "Penalty on runtime code displacement projected onto the real "
            "module axis for random_direction."
        ),
    )
    parser.add_argument(
        "--refined-code-random-max-real-axis-cosine",
        type=float,
        default=0.25,
        help=(
            "Maximum absolute runtime code cosine with the real module axis "
            "for accepted random controls; <0 disables this check."
        ),
    )
    parser.add_argument(
        "--refined-code-random-hidden-candidates",
        type=int,
        default=8,
        help=(
            "Number of fixed Gaussian hidden-space candidates for the true "
            "random_hidden_direction control; candidates are orthogonalized "
            "to the observed main hidden delta and semantic-drift matched."
        ),
    )
    parser.add_argument(
        "--max-delta-rel-norm",
        type=float,
        default=0.05,
        help="Maximum hidden-state delta norm divided by the original norm. <=0 disables clipping.",
    )
    parser.add_argument("--fixed-cluster-assignments", default=None, help="CSV produced by fit_soft_residual_clusters.py.")
    parser.add_argument("--fixed-cluster-features", default=None, help="Feature bundle produced by fit_soft_residual_clusters.py.")
    parser.add_argument(
        "--fixed-cluster-ids-only",
        action="store_true",
        help="Restrict evaluation to ids present in fixed cluster assignments; useful when posthoc probes reserve an isolated test subset.",
    )
    parser.add_argument("--e2-batch-size", type=int, default=1024)
    parser.add_argument("--e2-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--auto-k", action="store_true", help="Choose k automatically before generation using clustering quality and size penalties.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--auto-k-metric", choices=["composite", "silhouette"], default="composite")
    parser.add_argument("--min-cluster-fraction", type=float, default=0.05)
    parser.add_argument("--auto-k-small-cluster-weight", type=float, default=1.0)
    parser.add_argument("--auto-k-imbalance-weight", type=float, default=0.25)
    parser.add_argument("--cluster-feature-mode", choices=["continuous", "binary"], default="continuous")
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--cluster-l2-normalize", action="store_true", help="L2-normalize final clustering features, useful when E2 norm outliers dominate k-means.")
    parser.add_argument("--cluster-robust-clip-quantile", type=float, default=0.0, help="Clip standardized clustering features by absolute quantile before PCA. 0 disables.")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--kmeans-restarts", type=int, default=20)
    parser.add_argument("--transitions", nargs="+", default=["all_pairs"], help="all_pairs or source:target items such as 0:1 2:0.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--patch-mode",
        choices=["prototype", "donor", "extreme_prototype", "farthest_donor"],
        default="prototype",
    )
    parser.add_argument(
        "--extreme-prototype-fraction",
        type=float,
        default=0.1,
        help="For --patch-mode extreme_prototype, average the most target-like fraction of target-cluster samples.",
    )
    parser.add_argument(
        "--extreme-prototype-min-count",
        type=int,
        default=8,
        help="Minimum number of target-cluster samples averaged by --patch-mode extreme_prototype.",
    )
    parser.set_defaults(controls=[])
    parser.add_argument("--max-intervention-modes", type=int, default=0, help="Limit generated intervention modes for quick tests.")
    parser.add_argument(
        "--intervention-source-only",
        action="store_true",
        help=(
            "Generate each intervention mode only for rows in that mode's source "
            "cluster. The baseline still covers every selected cluster."
        ),
    )
    parser.add_argument("--baseline-file", default=None, help="Optional baseline_generations.jsonl to reuse.")
    parser.add_argument("--baseline-only", action="store_true", help="Generate/evaluate baseline by cluster and skip all intervention modes.")
    parser.add_argument(
        "--generation-checkpoint-file",
        default="",
        help=(
            "Append completed generation batches to this JSONL file and resume "
            "missing ids after interruption. Intended for deterministic baseline runs."
        ),
    )
    parser.add_argument("--eval-scope", choices=["val", "train", "test", "all_cached"], default="val", help="Sample pool: validation, training, untouched test ids from the target file, or all ids in the activation cache.")
    parser.add_argument(
        "--id-regex",
        default="",
        help=(
            "Optional regular expression used to keep only matching evaluation ids. "
            "For generated-step caches, examples include '::step16$' or '::step(16|32)$'."
        ),
    )
    parser.add_argument(
        "--evaluation-id-file",
        default="",
        help=(
            "Optional txt/csv/json/jsonl allowlist intersected with the selected evaluation role. "
            "Use this to run a frozen sample rule without resampling or changing held-out roles."
        ),
    )
    parser.add_argument("--eval-group", choices=["all", "low", "mid", "high"], default="all")
    parser.add_argument("--apply-meta-group", choices=["all", "low", "mid", "high"], default="all")
    parser.add_argument("--group-basis", choices=["true", "pred"], default="true")
    parser.add_argument("--max-samples-per-cluster", type=int, default=0, help="After clustering the selected eval scope, keep at most this many samples per cluster for generation/intervention. 0 keeps all.")
    parser.add_argument("--cluster-sample-strategy", choices=["first", "random"], default="random")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--generation-progress-seconds",
        type=float,
        default=30.0,
        help=(
            "Emit a line-oriented heartbeat every N seconds from inside long "
            "autoregressive generation calls. 0 disables."
        ),
    )
    parser.add_argument("--prompt-style", choices=["data", "direct", "cot"], default="data")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--chat-template-enable-thinking", choices=["auto", "true", "false"], default="auto")
    parser.add_argument(
        "--answer-extraction",
        choices=["none", "mathqa_choice"],
        default="none",
    )
    parser.add_argument("--answer-tolerance", type=float, default=0.0)
    parser.add_argument("--accuracy-bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--progress-accuracy-every",
        type=int,
        default=0,
        help="During generation, write current cumulative overall/per-cluster accuracy every N batches. 0 disables.",
    )
    parser.add_argument(
        "--progress-example-count",
        type=int,
        default=12,
        help="Keep this many representative generated texts in <mode>_progress_examples.jsonl at each progress refresh.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--next-token-audit",
        action="store_true",
        help=(
            "Run one prompt-end forward pass per mode instead of autoregressive generation. "
            "Records full-distribution KL/JS/entropy metrics and cluster-logit prototype movement."
        ),
    )
    parser.add_argument(
        "--record-prompt-confidence",
        action="store_true",
        help=(
            "Before normal generation, record prompt-end entropy, maximum "
            "probability, and top-1/top-2 margins. Unlike --next-token-audit, "
            "this keeps the subsequent full generation."
        ),
    )
    parser.add_argument(
        "--next-token-top-k",
        type=int,
        default=8,
        help="Store this many highest-probability tokens per sample in next-token audit mode.",
    )
    parser.add_argument(
        "--next-token-auto-logit-top-k",
        type=int,
        default=64,
        help="Automatically discover this many nonblank cluster-discriminative tokens.",
    )
    parser.add_argument(
        "--next-token-auto-logit-score",
        choices=["js_contribution", "probability_weighted_logit", "logit_range"],
        default="js_contribution",
        help=(
            "Token ranking score. js_contribution emphasizes tokens that carry "
            "both cluster discrimination and non-negligible probability mass."
        ),
    )
    parser.add_argument(
        "--next-token-auto-min-mean-prob",
        type=float,
        default=0.0,
        help="Exclude automatic prototype tokens below this mean baseline probability.",
    )
    parser.add_argument(
        "--next-token-auto-min-selected",
        type=int,
        default=0,
        help=(
            "Require at least this many visible automatic prototype tokens. If the "
            "probability floor leaves fewer, backfill from the same discovery split "
            "with the floor relaxed; 0 disables backfill."
        ),
    )
    parser.add_argument(
        "--next-token-prototype-file",
        default="",
        help=(
            "Load frozen discriminative token ids and cluster log-probability prototypes "
            "from an independent baseline-only discovery run."
        ),
    )
    parser.add_argument("--next-token-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--next-token-permutation-tests", type=int, default=2000)
    parser.add_argument("--apply-to-generated", action="store_true")
    parser.add_argument(
        "--generated-patch-steps",
        type=int,
        default=None,
        help="Override --apply-to-generated. 0=prefill only, N=patch first N generated-token steps, -1=patch all generated-token steps.",
    )
    parser.add_argument(
        "--max-truncated-rate",
        type=float,
        default=1.0,
        help="Raise if baseline style_truncated rate is above this value. Use e.g. 0.05 for style experiments.",
    )
    parser.add_argument(
        "--style-untruncated-only",
        action="store_true",
        help="Also write style summaries restricted to rows where baseline and current generation are not truncated.",
    )
    parser.add_argument("--record-layers", default="none")
    parser.add_argument("--save-recorded-activations", action="store_true")
    parser.add_argument(
        "--save-generated-token-ids",
        action="store_true",
        help="Store exact generated token ids in JSONL for trajectory comparisons.",
    )
    parser.add_argument(
        "--save-generated-token-logprobs",
        action="store_true",
        help=(
            "Record each selected token's model log probability during generation. "
            "Uses an online recorder and retains only one vocabulary-sized score tensor."
        ),
    )
    parser.add_argument(
        "--token-logprob-checkpoints",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32, 64, 128, 256, 512],
        help="Prefix lengths reported in generated_cumulative_logprob_prefix_json.",
    )
    parser.add_argument("--style-permutation-tests", type=int, default=500)
    parser.add_argument(
        "--metric-profile",
        choices=profile_names(),
        default="safety",
        help="Prespecified behavior family used for summaries and FDR correction.",
    )
    parser.add_argument(
        "--style-bootstrap-samples",
        type=int,
        default=2000,
        help="Paired bootstrap replicates for intervention style effects.",
    )
    parser.add_argument(
        "--style-primary-mode-filter",
        default="",
        help=(
            "Substring identifying the primary intervention mode for direct paired "
            "style contrasts against all matched controls. Empty disables contrasts."
        ),
    )
    parser.add_argument("--clear-cuda-cache-every", type=int, default=0, help="Call torch.cuda.empty_cache every N generation batches. 0 disables.")
    parser.add_argument("--clear-cuda-cache-between-modes", action="store_true")
    parser.add_argument("--log-cuda-memory", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto", choices=["auto", "balanced", "balanced_low_0", "sequential", "single", "none"])
    parser.add_argument("--max-memory", nargs="*", default=None)
    parser.add_argument("--attn-implementation", default=None, help="Optional transformers attention implementation, e.g. eager, sdpa, flash_attention_2.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.generation_progress_seconds < 0.0:
        raise ValueError("--generation-progress-seconds must be non-negative.")
    if args.clear_cuda_cache_every < 0:
        raise ValueError("--clear-cuda-cache-every must be non-negative.")
    if args.cluster_robust_clip_quantile < 0.0 or args.cluster_robust_clip_quantile > 1.0:
        raise ValueError("--cluster-robust-clip-quantile must be in [0, 1].")
    if args.min_cluster_fraction <= 0.0 or args.min_cluster_fraction >= 1.0:
        raise ValueError("--min-cluster-fraction must be in (0, 1).")
    if args.generated_patch_steps is not None and args.generated_patch_steps < -1:
        raise ValueError("--generated-patch-steps must be -1, 0, or a positive integer.")
    if args.max_truncated_rate < 0.0 or args.max_truncated_rate > 1.0:
        raise ValueError("--max-truncated-rate must be in [0, 1].")
    if args.progress_accuracy_every < 0:
        raise ValueError("--progress-accuracy-every must be non-negative.")
    if args.refined_direct_steps <= 0:
        raise ValueError("--refined-direct-steps must be positive.")
    if args.refined_direct_step_scale <= 0.0:
        raise ValueError("--refined-direct-step-scale must be positive.")
    if args.refined_direct_hidden_penalty < 0.0:
        raise ValueError("--refined-direct-hidden-penalty must be non-negative.")
    if args.refined_code_semantic_penalty < 0.0 or args.refined_code_z1_penalty < 0.0:
        raise ValueError("Refined-code preservation penalties must be non-negative.")
    if args.refined_code_dose_match_tolerance < 0.0:
        raise ValueError("--refined-code-dose-match-tolerance must be non-negative.")
    if args.refined_code_dose_hidden_tolerance < 0.0:
        raise ValueError("--refined-code-dose-hidden-tolerance must be non-negative.")
    if not -1.0 <= args.refined_code_dose_min_direction_cosine <= 1.0:
        raise ValueError("--refined-code-dose-min-direction-cosine must be in [-1, 1].")
    if args.refined_code_max_semantic_rel_delta < 0.0:
        raise ValueError("--refined-code-max-semantic-rel-delta must be non-negative.")
    if args.next_token_top_k <= 0:
        raise ValueError("--next-token-top-k must be positive.")
    if args.next_token_auto_logit_top_k < 0:
        raise ValueError("--next-token-auto-logit-top-k must be non-negative.")
    if args.next_token_auto_min_selected < 0:
        raise ValueError("--next-token-auto-min-selected must be non-negative.")
    if args.next_token_auto_min_mean_prob < 0.0 or args.next_token_auto_min_mean_prob > 1.0:
        raise ValueError("--next-token-auto-min-mean-prob must be in [0, 1].")
    if args.next_token_bootstrap_samples < 0 or args.next_token_permutation_tests < 0:
        raise ValueError("Next-token bootstrap/permutation counts must be non-negative.")
    if args.next_token_prototype_file and not args.next_token_audit:
        raise ValueError("--next-token-prototype-file requires --next-token-audit.")
    if args.next_token_prototype_file and not os.path.exists(
        args.next_token_prototype_file
    ):
        raise FileNotFoundError(args.next_token_prototype_file)
    if args.next_token_audit and args.baseline_file:
        raise ValueError(
            "--next-token-audit must compute prompt-end baseline logits in this run; "
            "do not pass --baseline-file."
        )
    if args.intervention_space == "refined_residual":
        if args.cluster_source != FIXED_CLUSTER_SOURCE:
            raise ValueError(
                "--intervention-space refined_residual requires --cluster-source soft_residual."
            )
        if not args.refined_module_dir:
            raise ValueError(
                "--intervention-space refined_residual requires --refined-module-dir."
            )
        if "random_neurons" in args.controls:
            raise ValueError(
                "random_neurons is not a valid control for refined residual-code intervention."
            )
        continuous_controls = globals().get("CONTINUOUS_CONTROL_MODES", ())
        if (
            args.refined_code_dose_match_controls
            and not args.controls
            and not continuous_controls
        ):
            raise ValueError(
                "--refined-code-dose-match-controls requires at least one control."
            )
    elif any(
        control in args.controls
        for control in ("random_direction", "random_hidden_direction")
    ):
        raise ValueError(
            "random direction controls are only valid with --intervention-space refined_residual."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    targets = load_intervention_targets(args.target_file)
    metadata = targets.get("metadata", {})
    layer_i = int(metadata.get("layer_i"))
    next_layer = int(metadata.get("next_layer", layer_i + 1))
    activation_dir = args.activation_dir or metadata.get("activation_dir")
    if not activation_dir:
        raise ValueError("--activation-dir is required when intervention target metadata has no activation_dir.")

    selected = targets["selected_neurons"].long()
    hidden_size = hidden_size_from_activation(activation_dir, layer_i)
    feature_indices = pick_feature_indices(selected, hidden_size, args.feature_set, args.random_feature_count, args.seed)

    fixed_ids_only = args.cluster_source == FIXED_CLUSTER_SOURCE and args.fixed_cluster_ids_only
    # An explicit evaluation-id file is an allowlist. Load the full scope
    # before applying max-samples so random capping cannot discard all of the
    # requested critical/held-out ids.
    has_evaluation_id_file = bool(args.evaluation_id_file)
    scope_rows = build_scope_rows(
        args.data,
        activation_dir,
        layer_i,
        targets,
        args.group_basis,
        args.eval_group,
        0 if fixed_ids_only or has_evaluation_id_file else args.max_samples,
        args.eval_scope,
        args.id_regex,
    )
    if not scope_rows:
        raise ValueError("No evaluation rows selected. Check --data, --eval-scope, --eval-group, and activation cache ids.")
    if has_evaluation_id_file:
        allowed_ids = load_evaluation_ids(args.evaluation_id_file)
        before = len(scope_rows)
        scope_rows = [row for row in scope_rows if str(row.get("id", "")) in allowed_ids]
        if not scope_rows:
            raise ValueError(
                "--evaluation-id-file has no overlap with the selected evaluation role and fixed clusters."
            )
        print(
            f"[evaluation-ids] restricted evaluation ids from {before} to {len(scope_rows)} "
            f"using {args.evaluation_id_file}",
            flush=True,
        )
    # Intersect with fixed-cluster artifacts after the explicit allowlist, so
    # max_samples is sampled from the requested IDs rather than the full pool.
    if fixed_ids_only:
        before = len(scope_rows)
        scope_rows = restrict_rows_to_fixed_clusters(
            scope_rows,
            args.fixed_cluster_assignments,
            args.fixed_cluster_features,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(
            f"[fixed-clusters] restricted evaluation ids from {before} to {len(scope_rows)} "
            "using assignment/feature intersection",
            flush=True,
        )
    ids = [row["id"] for row in scope_rows]
    patch_features = load_layer_matrix(activation_dir, layer_i, ids, feature_indices)
    fixed_labels: Optional[torch.Tensor] = None
    fixed_global_features: Optional[torch.Tensor] = None
    fixed_global_labels: Optional[torch.Tensor] = None
    fixed_global_ids: Optional[List[str]] = None
    fixed_global_model_centroids: Optional[torch.Tensor] = None
    if args.cluster_source == FIXED_CLUSTER_SOURCE:
        if args.auto_k:
            raise ValueError("--auto-k cannot be combined with fixed soft-residual assignments.")
        fixed_global_features, fixed_global_labels, fixed_feature_info = load_fixed_cluster_data(
            args.fixed_cluster_assignments,
            args.fixed_cluster_features,
            None,
        )
        fixed_global_ids = list(fixed_feature_info["ids"])
        inferred_k = int(fixed_global_labels.max().item()) + 1
        if int(args.num_clusters) != inferred_k:
            raise ValueError(
                f"--num-clusters={args.num_clusters}, but fixed assignments contain "
                f"{inferred_k} clusters in the complete artifact."
            )
        fixed_global_model_centroids = raw_centroids_from_labels(
            fixed_global_features.float(), fixed_global_labels, args.num_clusters
        )
        cluster_model_features, fixed_labels, main_feature_info = load_fixed_cluster_data(
            args.fixed_cluster_assignments,
            args.fixed_cluster_features,
            ids,
        )
        main_feature_info["global_cluster_sample_count"] = int(fixed_global_labels.numel())
        main_feature_info["global_cluster_counts"] = torch.bincount(
            fixed_global_labels, minlength=args.num_clusters
        ).tolist()
        cluster_name = "soft_residual_clusters"
    elif args.cluster_source in LATENT_CLUSTER_SOURCES:
        if args.cluster_feature_mode != "continuous":
            raise ValueError("--cluster-source e2_latent/purified_meta/purified_semantic currently requires --cluster-feature-mode continuous.")
        latent_features, latent_info = compute_latent_features(activation_dir, next_layer, ids, args, source=args.cluster_source)
        cluster_model_features, main_feature_info = transform_cluster_features(
            latent_features,
            mode="continuous",
            standardize=not args.no_standardize,
            pca_dim=args.pca_dim,
            source=args.cluster_source,
            l2_normalize=args.cluster_l2_normalize,
            robust_clip_quantile=args.cluster_robust_clip_quantile,
            extra_info=latent_info,
        )
        cluster_name = f"{args.cluster_source}_clusters" if args.feature_set == "selected" else f"{args.cluster_source}_{args.feature_set}_patch_clusters"
    else:
        cluster_name = "selected_clusters" if args.feature_set == "selected" else f"{args.feature_set}_clusters"
        cluster_model_features, main_feature_info = prepare_cluster_features(
            patch_features,
            feature_indices,
            targets,
            args.cluster_feature_mode,
            not args.no_standardize,
            args.pca_dim,
            args.cluster_l2_normalize,
            args.cluster_robust_clip_quantile,
        )
    auto_k_candidates: List[Dict] = []
    if args.auto_k:
        selected_k, auto_k_candidates = auto_select_k(cluster_model_features, args)
        args.num_clusters = selected_k
        main_feature_info["auto_k_selected"] = selected_k
        main_feature_info["auto_k_candidates"] = auto_k_candidates
        write_csv(os.path.join(args.output_dir, "auto_k_candidates.csv"), auto_k_candidates)
    full_main_cluster = build_cluster_result_from_prepared(
        cluster_name,
        cluster_model_features,
        patch_features,
        feature_indices,
        args,
        fixed_labels=fixed_labels,
        model_centroids_override=fixed_global_model_centroids,
    )
    add_cluster_fields(scope_rows, full_main_cluster)

    controls: Dict[str, ClusterResult] = {}
    control_info: Dict[str, Dict] = {}
    if "random_neurons" in args.controls:
        random_indices = pick_feature_indices(selected, hidden_size, "random_nonselected", feature_indices.numel(), args.seed + 1009)
        random_features = load_layer_matrix(activation_dir, layer_i, ids, random_indices)
        controls["random_neurons"] = build_cluster_result("random_neurons", random_features, random_indices, targets, args, seed_offset=1009)[0]
        control_info["random_neurons"] = {"features": random_indices.tolist()}
    if "shuffled_labels" in args.controls:
        generator = torch.Generator().manual_seed(args.seed + 2027)
        shuffled_labels = full_main_cluster.labels[torch.randperm(full_main_cluster.labels.numel(), generator=generator)]
        controls["shuffled_labels"] = build_cluster_result_from_prepared(
            "shuffled_labels",
            full_main_cluster.model_features,
            patch_features,
            feature_indices,
            args,
            seed_offset=2027,
            fixed_labels=shuffled_labels,
        )
        control_info["shuffled_labels"] = {"features": feature_indices.tolist()}

    full_controls = dict(controls)
    sample_indices = sample_indices_by_cluster(
        full_main_cluster.labels,
        args.max_samples_per_cluster,
        args.cluster_sample_strategy,
        args.seed,
    )
    eval_rows = [scope_rows[int(idx)] for idx in sample_indices.tolist()]
    main_cluster = subset_cluster_result(full_main_cluster, sample_indices)
    controls = {
        name: subset_cluster_result(cluster_result, sample_indices)
        for name, cluster_result in full_controls.items()
    }

    baseline_rows: Optional[List[Dict]] = None
    baseline_checkpoint_rows: List[Dict] = []
    baseline_generation_rows = eval_rows
    if args.baseline_file:
        baseline_rows = load_baseline_file(args.baseline_file, eval_rows, args)
        check_truncation(baseline_rows, args.max_truncated_rate, "baseline")
    elif args.baseline_only and args.generation_checkpoint_file:
        checkpoint_rows = read_generation_checkpoint(args.generation_checkpoint_file)
        if checkpoint_rows:
            baseline_checkpoint_rows = align_baseline_rows(
                checkpoint_rows,
                eval_rows,
                args,
                require_all=False,
                source=args.generation_checkpoint_file,
            )
            checkpoint_ids = {str(row["id"]) for row in baseline_checkpoint_rows}
            baseline_generation_rows = [
                row for row in eval_rows if str(row["id"]) not in checkpoint_ids
            ]
            print(
                "[baseline-checkpoint] "
                f"recovered={len(baseline_checkpoint_rows)} "
                f"missing={len(baseline_generation_rows)} "
                f"file={args.generation_checkpoint_file}",
                flush=True,
            )
            if not baseline_generation_rows:
                baseline_rows = align_baseline_rows(
                    baseline_checkpoint_rows,
                    eval_rows,
                    args,
                    require_all=True,
                    source=args.generation_checkpoint_file,
                )
    baseline_by_id = {row["id"]: row for row in baseline_rows or []}

    write_csv(os.path.join(args.output_dir, "full_cluster_assignments.csv"), assignment_rows(scope_rows, full_main_cluster, {}))
    write_csv(os.path.join(args.output_dir, "cluster_assignments.csv"), assignment_rows(eval_rows, main_cluster, baseline_by_id))
    centroid_state_file = os.path.join(args.output_dir, "cluster_centroid_activation_states.csv")
    write_csv(centroid_state_file, centroid_activation_state_rows(main_cluster))
    cluster_summary = {
        "layer_i": layer_i,
        "next_layer": next_layer,
        "activation_dir": activation_dir,
        "target_file": args.target_file,
        "eval_scope": args.eval_scope,
        "scope_sample_count": len(scope_rows),
        "eval_sample_count": len(eval_rows),
        "max_samples": args.max_samples,
        "max_samples_per_cluster": args.max_samples_per_cluster,
        "cluster_sample_strategy": args.cluster_sample_strategy,
        "cluster_source": args.cluster_source,
        "auto_k": args.auto_k,
        "auto_k_candidates": auto_k_candidates,
        "cluster_l2_normalize": args.cluster_l2_normalize,
        "cluster_robust_clip_quantile": args.cluster_robust_clip_quantile,
        "decoupler_dir": args.decoupler_dir,
        "decoupler_checkpoint": args.decoupler_checkpoint,
        "decoupler_normalization": args.decoupler_normalization,
        "purifier_checkpoint": args.purifier_checkpoint,
        "feature_set": args.feature_set,
        "feature_indices": feature_indices.tolist(),
        "centroid_activation_states_file": "cluster_centroid_activation_states.csv",
        "centroid_activation_states": {
            str(cluster_id): {
                str(int(feature)): float(main_cluster.raw_centroids[cluster_id, feature_pos].item())
                for feature_pos, feature in enumerate(feature_indices.tolist())
            }
            for cluster_id in range(main_cluster.raw_centroids.shape[0])
        },
        "feature_info": main_feature_info,
        "full": {
            "main": cluster_static_summary(scope_rows, full_main_cluster, None),
            "controls": {
                name: cluster_static_summary(scope_rows, cluster_result, None)
                for name, cluster_result in full_controls.items()
            },
        },
        "main": cluster_static_summary(
            eval_rows,
            main_cluster,
            baseline_rows if args.metric_profile == "math" else None,
        ),
        "controls": {
            name: cluster_static_summary(
                eval_rows,
                cluster_result,
                baseline_rows if args.metric_profile == "math" else None,
            )
            for name, cluster_result in controls.items()
        },
        "control_info": control_info,
        "visualizations": [
            f"plots/{cluster_name}_cluster_pca.png",
            f"plots/{cluster_name}_centroid_heatmap.png",
            "plots/baseline_style_by_cluster.png",
            "plots/baseline_cluster_style_heatmap.png",
        ] if not args.no_plots else [],
    }
    write_json(os.path.join(args.output_dir, "cluster_summary.json"), cluster_summary)
    if not args.no_plots:
        plot_clusters(
            args.output_dir,
            main_cluster,
            eval_rows,
            baseline_rows if args.metric_profile == "math" else None,
        )

    if args.model_path is None and baseline_rows is None:
        print(
            json.dumps(
                {
                    "cluster_source": args.cluster_source,
                    "scope_samples": len(scope_rows),
                    "evaluation_samples": len(eval_rows),
                    "cluster_counts": {
                        cluster: values.get("n")
                        for cluster, values in cluster_summary["main"]["clusters"].items()
                    },
                    "output_dir": args.output_dir,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.model_path is None or (args.baseline_only and baseline_rows is not None):
        baseline_style_summary = write_style_outputs(
            args.output_dir,
            "baseline",
            baseline_rows,
            "activation_cluster",
            args.style_permutation_tests,
            args.seed,
            args.max_new_tokens,
            args.metric_profile,
        )
        cluster_summary["baseline_style"] = baseline_style_summary
        write_json(os.path.join(args.output_dir, "cluster_summary.json"), cluster_summary)
        if args.baseline_only:
            write_jsonl(
                os.path.join(args.output_dir, "baseline_generations.jsonl"),
                baseline_rows,
            )
            write_csv(
                os.path.join(args.output_dir, "cluster_baseline_results.csv"),
                baseline_rows,
            )
            run_config = vars(args).copy()
            run_config.update(
                {
                    "layer_i": layer_i,
                    "next_layer": next_layer,
                    "feature_indices": feature_indices.tolist(),
                    "baseline_reused": True,
                    "baseline_reuse_source": args.baseline_file,
                    "summary": {
                        "baseline": (
                            summarize_accuracy_rows(baseline_rows)
                            if args.metric_profile == "math"
                            else {"n": len(baseline_rows), "task_performance_not_reported": True}
                        ),
                        "baseline_style": baseline_style_summary,
                    },
                }
            )
            write_json(os.path.join(args.output_dir, "run_config.json"), run_config)
        if not args.no_plots:
            plot_style_metrics(
                args.output_dir, baseline_rows, "baseline", args.metric_profile
            )
        print(
            json.dumps(
                {
                    "cluster_source": args.cluster_source,
                    "evaluation_samples": len(eval_rows),
                    "baseline_reused": baseline_rows is not None,
                    "output_dir": args.output_dir,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(
        f"[generation-stage] loading_model path={args.model_path}",
        flush=True,
    )
    model, tokenizer = load_model_and_tokenizer(args, padding_side="left")
    print(
        f"[generation-stage] model_ready device={model_input_device(model)}",
        flush=True,
    )
    layers = get_decoder_layers(model)
    if layer_i < 0 or layer_i >= len(layers):
        raise ValueError(f"layer_i={layer_i} is out of range for model with {len(layers)} layers.")
    record_layers = parse_record_layers(args.record_layers, layer_i, next_layer)
    bad_record_layers = [layer for layer in record_layers if layer < 0 or layer >= len(layers)]
    if bad_record_layers:
        raise ValueError(f"record layers out of range: {bad_record_layers}")

    context = GenerationContext()
    next_token_state = (
        NextTokenAuditState(
            tokenizer,
            int(args.num_clusters),
            int(args.next_token_auto_logit_top_k),
            str(args.next_token_auto_logit_score),
            float(args.next_token_auto_min_mean_prob),
            int(args.next_token_auto_min_selected),
        )
        if args.next_token_audit
        else None
    )
    if next_token_state is not None and args.next_token_prototype_file:
        next_token_state.load_prototypes(args.next_token_prototype_file)
        print(
            "[next-token] loaded frozen prototypes from "
            f"{args.next_token_prototype_file}",
            flush=True,
        )
    next_token_baseline_info: Dict[str, Any] = {}
    recorder = ActivationRecorder(context, layers, record_layers, feature_indices) if record_layers else None
    all_generations: Dict[str, List[Dict]] = {}
    refined_components = None
    refined_runtime_codes = None
    refined_runtime_centroid_codes = None
    refined_runtime_centroid_labels = None
    refined_component_info: Dict[str, Any] = {}
    refined_audit_rows: List[Dict[str, Any]] = []
    if args.intervention_space == "refined_residual":
        if next_layer < 0 or next_layer >= len(layers):
            raise ValueError(
                f"next_layer={next_layer} is out of range for model with {len(layers)} layers."
            )
        component_device = model_input_device(model)
        refined_components = load_refined_residual_components(
            args,
            hidden_size_from_activation(activation_dir, next_layer),
            component_device,
        )
        refined_component_info = refined_components[4]
        refined_runtime_codes = cluster_model_features.float()
        if refined_runtime_codes.size(1) != int(refined_component_info["code_dim"]):
            raise ValueError(
                "Fixed continuous-axis features do not match the module refiner code dimension."
            )
        if fixed_global_ids is not None and fixed_global_labels is not None:
            assert fixed_global_features is not None
            refined_runtime_centroid_codes = fixed_global_features.float()
            refined_runtime_centroid_labels = fixed_global_labels.long()
            cluster_counts = torch.bincount(
                refined_runtime_centroid_labels, minlength=args.num_clusters
            )
            if (cluster_counts == 0).any():
                raise ValueError(
                    "Continuous-axis artifact lacks at least one fixed cluster: "
                    f"counts={cluster_counts.tolist()}"
                )
        cluster_summary["refined_residual_intervention"] = refined_component_info
        write_json(os.path.join(args.output_dir, "cluster_summary.json"), cluster_summary)
    try:
        if baseline_rows is None:
            print(
                "[generation-stage] baseline_generation_start "
                f"rows={len(eval_rows)} batch_size={args.batch_size} "
                f"max_new_tokens={args.max_new_tokens} "
                f"record_prompt_confidence={args.record_prompt_confidence}",
                flush=True,
            )
            generated_baseline_rows = generate_rows(
                model,
                tokenizer,
                baseline_generation_rows,
                context,
                "baseline",
                args,
                patch_builder=None,
                next_token_state=next_token_state,
                checkpoint_file=(
                    args.generation_checkpoint_file
                    if args.baseline_only and args.generation_checkpoint_file
                    else None
                ),
            )
            baseline_rows = align_baseline_rows(
                baseline_checkpoint_rows + generated_baseline_rows,
                eval_rows,
                args,
                require_all=True,
                source=args.generation_checkpoint_file or "fresh generation",
            )
        if args.next_token_audit:
            assert next_token_state is not None
            next_token_baseline_info = next_token_state.finalize()
            next_token_baseline_info.update(next_token_state.save(args.output_dir))
            write_json(
                os.path.join(args.output_dir, "next_token_baseline_audit.json"),
                next_token_baseline_info,
            )
        else:
            check_truncation(baseline_rows, args.max_truncated_rate, "baseline")
        all_generations["baseline"] = baseline_rows
        cluster_summary["main"] = cluster_static_summary(
            eval_rows,
            main_cluster,
            baseline_rows if args.metric_profile == "math" else None,
        )
        cluster_summary["controls"] = {
            name: cluster_static_summary(
                eval_rows,
                cluster_result,
                baseline_rows if args.metric_profile == "math" else None,
            )
            for name, cluster_result in controls.items()
        }
        write_json(os.path.join(args.output_dir, "cluster_summary.json"), cluster_summary)
        if not args.no_plots:
            plot_clusters(
                args.output_dir,
                main_cluster,
                eval_rows,
            baseline_rows if args.metric_profile == "math" else None,
            )
        baseline_records = recorder.records.get("baseline", {}) if recorder else {}
        baseline_style_summary = (
            {"skipped": True, "reason": "next_token_audit"}
            if args.next_token_audit
            else write_style_outputs(
                args.output_dir,
                "baseline",
                baseline_rows,
                "activation_cluster",
                args.style_permutation_tests,
                args.seed,
                args.max_new_tokens,
                args.metric_profile,
            )
        )
        cluster_summary["baseline_style"] = baseline_style_summary
        if not args.no_plots and not args.next_token_audit:
            plot_style_metrics(
                args.output_dir, baseline_rows, "baseline", args.metric_profile
            )
        write_json(os.path.join(args.output_dir, "cluster_summary.json"), cluster_summary)

        if args.baseline_only:
            all_generations["baseline"] = baseline_rows
            write_jsonl(os.path.join(args.output_dir, "baseline_generations.jsonl"), baseline_rows)
            write_csv(os.path.join(args.output_dir, "cluster_baseline_results.csv"), baseline_rows)
            run_config = vars(args).copy()
            run_config.update(
                {
                    "layer_i": layer_i,
                    "next_layer": next_layer,
                    "feature_indices": feature_indices.tolist(),
                    "record_layers": record_layers,
                    "summary": {
                        "baseline": (
                            summarize_accuracy_rows(baseline_rows)
                            if args.metric_profile == "math"
                            else {"n": len(baseline_rows), "task_performance_not_reported": True}
                        ),
                        "baseline_style": baseline_style_summary,
                        "baseline_clusters": {
                            str(cluster_id): (
                                {
                                    "n": sum(
                                        int(row.get("activation_cluster", -1)) == cluster_id
                                        for row in baseline_rows
                                    ),
                                    "task_performance_not_reported": True,
                                }
                                if args.metric_profile != "math"
                                else summarize_accuracy_rows(
                                    [row for row in baseline_rows if int(row.get("activation_cluster", -1)) == cluster_id]
                                )
                            )
                            for cluster_id in range(args.num_clusters)
                        },
                    },
                }
            )
            write_json(os.path.join(args.output_dir, "run_config.json"), run_config)
            if args.save_recorded_activations and recorder is not None:
                torch.save(recorder.records, os.path.join(args.output_dir, "recorded_selected_activations.pt"))
            cluster_console = {
                cluster: {
                    "n": cluster_summary["main"]["clusters"][cluster]["n"],
                    "generated_tokens_mean": (
                        baseline_style_summary.get("groups", {})
                        .get(cluster, {})
                        .get("generated_tokens", {})
                        .get("mean")
                    ),
                }
                for cluster in cluster_summary["main"]["clusters"]
            }
            print("[cluster] baseline complete")
            print(
                json.dumps(
                    {
                        "cluster_source": args.cluster_source,
                        "evaluation_samples": len(eval_rows),
                        "clusters": cluster_console,
                        "truncated_rate": truncated_rate(baseline_rows),
                        "output_dir": args.output_dir,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        specs = build_intervention_specs(main_cluster, controls, args)
        row_index_by_id = {row["id"]: idx for idx, row in enumerate(eval_rows)}
        donor_state: Dict[str, int] = {}
        main_dose_reference_by_transition: Dict[
            Tuple[int, int, float], Dict[Tuple[str, int], Dict[str, Any]]
        ] = {}
        main_dose_reference_by_source: Dict[
            Tuple[int, float], Dict[Tuple[str, int], Dict[str, Any]]
        ] = {}
        for spec in specs:
            mode_eval_rows = (
                [
                    row
                    for row in eval_rows
                    if int(
                        spec.cluster_result.labels[row_index_by_id[row["id"]]].item()
                    )
                    == int(spec.source_cluster)
                ]
                if args.intervention_source_only
                else eval_rows
            )
            if not mode_eval_rows:
                raise RuntimeError(
                    f"No source-cluster rows remain for intervention mode {spec.name}."
                )
            if args.intervention_space == "refined_residual":
                assert refined_components is not None
                assert refined_runtime_codes is not None
                patch_builder = lambda batch, current_spec=spec: build_apply_mask_for_batch(
                    batch,
                    row_index_by_id,
                    current_spec,
                    args.apply_meta_group,
                )
                centroid_codes = (
                    refined_runtime_centroid_codes
                    if refined_runtime_centroid_codes is not None
                    else refined_runtime_codes
                )
                centroid_labels = (
                    refined_runtime_centroid_labels
                    if refined_runtime_centroid_labels is not None
                    else spec.cluster_result.labels
                )
                code_centroids = code_centroids_from_labels(
                    centroid_codes,
                    centroid_labels,
                    args.num_clusters,
                )
                decoupler, purifier, normalization, refiner = refined_components[:4]
                dose_reference = None
                if args.refined_code_dose_match_controls and spec.control != "main":
                    if spec.control == "random_direction":
                        dose_reference = main_dose_reference_by_transition.get(
                            (
                                int(spec.source_cluster),
                                int(spec.target_cluster),
                                float(spec.alpha),
                            )
                        )
                    else:
                        # Mechanism controls such as mean-only, covariance-only,
                        # centroid, and uncalibrated OT must be compared at the
                        # main intervention's achieved code/hidden dose.
                        dose_reference = main_dose_reference_by_transition.get(
                            (
                                int(spec.source_cluster),
                                int(spec.target_cluster),
                                float(spec.alpha),
                            )
                        )
                    if dose_reference is None:
                        raise RuntimeError(
                            "Dose-matched control has no preceding main intervention "
                            f"reference for {spec.name}."
                        )
                intervention = RefinedResidualCodeIntervention(
                    context,
                    layers[next_layer],
                    decoupler,
                    purifier,
                    normalization,
                    refiner,
                    code_centroids,
                    spec.source_cluster,
                    spec.target_cluster,
                    spec.alpha,
                    args.refined_code_intervention,
                    args.max_delta_rel_norm,
                    effective_apply_to_generated(args),
                    effective_generated_patch_steps(args),
                    spec.control,
                    (
                        args.refined_code_random_seed
                        if spec.control == "random_direction"
                        and args.refined_code_random_seed is not None
                        else args.seed
                    )
                    + 10007 * int(spec.source_cluster)
                    + 1009 * int(spec.target_cluster),
                    args.refined_code_semantic_penalty,
                    args.refined_code_z1_penalty,
                    args.refined_code_max_semantic_rel_delta,
                    dose_reference,
                    (
                        refined_runtime_centroid_codes
                        if refined_runtime_centroid_codes is not None
                        else refined_runtime_codes
                    ),
                    args.refined_code_random_direction_mode,
                    args.refined_code_dose_match_tolerance,
                    args.refined_code_dose_hidden_tolerance,
                    args.refined_code_dose_min_direction_cosine,
                    args.refined_direct_steps,
                    args.refined_direct_step_scale,
                    args.refined_direct_hidden_penalty,
                    args.refined_code_dose_semantic_tolerance,
                    args.refined_code_random_dose_mode,
                    args.refined_code_random_axis_penalty,
                    args.refined_code_random_max_real_axis_cosine,
                    args.refined_code_random_hidden_candidates,
                )
            else:
                patch_builder = lambda batch, current_spec=spec: build_patch_values_for_batch(
                    batch,
                    row_index_by_id,
                    current_spec,
                    donor_state,
                    args.apply_meta_group,
                )
                intervention = ClusterPatchIntervention(
                    context,
                    layers[layer_i],
                    spec.cluster_result.feature_indices,
                    spec.alpha,
                    effective_apply_to_generated(args),
                    effective_generated_patch_steps(args),
                )
            try:
                rows = generate_rows(
                    model,
                    tokenizer,
                    mode_eval_rows,
                    context,
                    spec.name,
                    args,
                    patch_builder=patch_builder,
                    next_token_state=next_token_state,
                    intervention_spec=spec,
                )
            finally:
                intervention.close()
                if isinstance(intervention, RefinedResidualCodeIntervention):
                    refined_audit_rows.extend(intervention.rows)
                    audit_reference = {
                        (str(item.get("id", "")), int(item.get("step", -1))): dict(item)
                        for item in intervention.rows
                    }
                    if spec.control == "main":
                        for key, vector in intervention.hidden_delta_vectors.items():
                            if key in audit_reference:
                                audit_reference[key]["_hidden_delta_vector"] = vector
                    if spec.control == "main":
                        main_dose_reference_by_transition[
                            (
                                int(spec.source_cluster),
                                int(spec.target_cluster),
                                float(spec.alpha),
                            )
                        ] = audit_reference
                        main_dose_reference_by_source[
                            (int(spec.source_cluster), float(spec.alpha))
                        ] = audit_reference
                    mode_audit_stem = f"refined_residual_intervention_audit_{spec.name}"
                    write_csv(
                        os.path.join(args.output_dir, f"{mode_audit_stem}.csv"),
                        intervention.rows,
                    )
                    write_json(
                        os.path.join(args.output_dir, f"{mode_audit_stem}_summary.json"),
                        summarize_refined_intervention_audit(intervention.rows),
                    )
                if args.clear_cuda_cache_between_modes and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                maybe_log_cuda(args, f"{spec.name}:after_hook_close")
            mode_records = recorder.records.get(spec.name, {}) if recorder else {}
            baseline_by_id = {row["id"]: row for row in baseline_rows}
            prefill_audit_by_id = {}
            if isinstance(intervention, RefinedResidualCodeIntervention):
                prefill_audit_by_id = {
                    str(item.get("id", "")): item
                    for item in intervention.rows
                    if bool(item.get("is_prefill"))
                }
            for row in rows:
                base = baseline_by_id.get(row["id"], {})
                audit_item = prefill_audit_by_id.get(str(row["id"]))
                if audit_item is not None:
                    for audit_field in (
                        "code_correction_accepted",
                        "dose_matching_enabled",
                        "dose_reference_accepted",
                        "dose_reference_code_delta_l2",
                        "dose_reference_hidden_relative_delta",
                        "dose_match_accepted",
                        "dose_direction_cosine",
                        "dose_request_scale_selected",
                        "requested_code_delta_l2",
                        "actual_code_delta_l2",
                        "actual_to_reference_code_delta_ratio",
                        "hidden_relative_delta",
                        "actual_to_reference_hidden_delta_ratio",
                        "semantic_relative_delta",
                        "z1_relative_delta",
                        "objective_target_code_delta_l2",
                        "objective_target_error_ratio",
                        "objective_direction_fraction_achieved",
                        "objective_direction_cosine",
                    ):
                        row[audit_field] = audit_item.get(audit_field)
                global_idx = row_index_by_id[row["id"]]
                row["baseline_pred_answer"] = base.get("pred_answer")
                row["baseline_correct"] = base.get("correct")
                row["changed_text"] = row.get("generated_text") != base.get("generated_text")
                row["changed_answer"] = row.get("pred_answer") != base.get("pred_answer")
                row["main_activation_cluster"] = row.get("activation_cluster")
                row["main_activation_cluster_name"] = row.get("activation_cluster_name")
                row["intervention_feature_cluster"] = int(spec.cluster_result.labels[global_idx].item())
                row["intervention_feature_cluster_name"] = cluster_label_name(row["intervention_feature_cluster"])
                row["intervention_source_cluster"] = spec.source_cluster
                row["intervention_target_cluster"] = spec.target_cluster
                row["intervention_alpha"] = spec.alpha
                row["intervention_patch_mode"] = spec.patch_mode
                row["intervention_control"] = spec.control
                row.update(activation_delta_columns(row, baseline_records, mode_records, record_layers))
            if recorder is not None and not args.save_recorded_activations:
                recorder.records.pop(spec.name, None)
            all_generations[spec.name] = rows
            write_jsonl(os.path.join(args.output_dir, f"{spec.name}_generations.jsonl"), rows)
    finally:
        if recorder is not None:
            recorder.close()

    write_jsonl(os.path.join(args.output_dir, "baseline_generations.jsonl"), all_generations["baseline"])
    flat_rows = []
    for mode, rows in all_generations.items():
        flat_rows.extend(rows)
    write_csv(os.path.join(args.output_dir, "cluster_intervention_results.csv"), flat_rows)
    final_specs = {
        spec.name: spec for spec in build_intervention_specs(main_cluster, controls, args)
    }
    untruncated_style_summary = None
    if args.next_token_audit:
        baseline_cluster_summary: Dict[str, Any] = {}
        for cluster_id in range(int(args.num_clusters)):
            cluster_rows = [
                row
                for row in all_generations["baseline"]
                if int(row.get("activation_cluster", -1)) == cluster_id
            ]
            baseline_cluster_summary[str(cluster_id)] = {
                "n": len(cluster_rows),
                "entropy_mean": (
                    float(
                        sum(float(row["next_token_entropy"]) for row in cluster_rows)
                        / len(cluster_rows)
                    )
                    if cluster_rows
                    else None
                ),
                "max_probability_mean": (
                    float(
                        sum(
                            float(row["next_token_max_probability"])
                            for row in cluster_rows
                        )
                        / len(cluster_rows)
                    )
                    if cluster_rows
                    else None
                ),
            }
        summary = {
            "next_token_audit": True,
            "baseline": {
                "n": len(all_generations["baseline"]),
                "clusters": baseline_cluster_summary,
                "automatic_logit_prototypes": next_token_baseline_info,
            },
            "modes": {},
        }
        for mode, rows in all_generations.items():
            if mode == "baseline":
                continue
            mode_summary = summarize_next_token_rows(rows)
            spec = final_specs.get(mode)
            if spec is not None:
                mode_summary.update(
                    {
                        "source_cluster": int(spec.source_cluster),
                        "target_cluster": int(spec.target_cluster),
                        "alpha": float(spec.alpha),
                        "control": spec.control,
                    }
                )
            summary["modes"][mode] = mode_summary
        if not args.no_plots:
            plot_next_token_audit(
                args.output_dir,
                summary,
            )
        summary["next_token_visualizations"] = (
            [
                "plots/next_token_intervention_audit.png",
            ]
            if not args.no_plots
            else []
        )
        summary["all_generations_style"] = {
            "skipped": True,
            "reason": "next_token_audit",
        }
        summary["truncation"] = {"not_applicable": True}
    else:
        all_style_summary = write_style_outputs(
            args.output_dir,
            "all_generations",
            flat_rows,
            "activation_cluster",
            args.style_permutation_tests,
            args.seed,
            args.max_new_tokens,
            args.metric_profile,
        )
        if args.style_untruncated_only:
            baseline_by_id_for_filter = {
                row["id"]: row for row in all_generations["baseline"]
            }
            untruncated_flat_rows = filter_untruncated_rows(
                flat_rows, baseline_by_id_for_filter
            )
            untruncated_style_summary = write_style_outputs(
                args.output_dir,
                "all_generations_untruncated",
                untruncated_flat_rows,
                "activation_cluster",
                args.style_permutation_tests,
                args.seed + 303,
                args.max_new_tokens,
                args.metric_profile,
            )
        if not args.no_plots:
            plot_style_metrics(
                args.output_dir,
                flat_rows,
                "all_generations",
                args.metric_profile,
            )
        summary = write_intervention_summary(
            os.path.join(args.output_dir, "cluster_intervention_summary.json"),
            all_generations,
            record_layers,
            args.accuracy_bootstrap_samples,
            args.style_permutation_tests,
            args.seed,
            final_specs,
            args.metric_profile,
        )
        effect_rows = intervention_style_effect_rows(
            all_generations,
            final_specs,
            args.style_bootstrap_samples,
            args.style_permutation_tests,
            args.seed,
            args.metric_profile,
        )
        write_csv(
            os.path.join(args.output_dir, "intervention_style_effects.csv"), effect_rows
        )
        print(
            "[style-paired] baseline contrasts "
            f"rows={len(effect_rows)} "
            f"p_significant={sum(bool(row.get('significant_p_0p05')) for row in effect_rows)}",
            flush=True,
        )
        summary["intervention_style_effects_file"] = "intervention_style_effects.csv"
        style_contrasts = intervention_style_contrast_rows(
            all_generations,
            final_specs,
            args.style_primary_mode_filter,
            args.style_bootstrap_samples,
            args.style_permutation_tests,
            args.seed + 65537,
            args.metric_profile,
        )
        if style_contrasts:
            write_csv(
                os.path.join(
                    args.output_dir,
                    "intervention_style_prespecified_contrasts.csv",
                ),
                style_contrasts,
            )
            summary["intervention_style_prespecified_contrasts_file"] = (
                "intervention_style_prespecified_contrasts.csv"
            )
            print(
                "[style-paired] mode contrasts "
                f"rows={len(style_contrasts)} "
            f"p_significant={sum(bool(row.get('significant_p_0p05')) for row in style_contrasts)}",
                flush=True,
            )
        summary["intervention_visualizations"] = (
            [
                "plots/intervention_response_change_rates.png",
                "plots/intervention_style_movement_to_target.png",
                "plots/intervention_style_effect_heatmap.png",
            ]
            if not args.no_plots
            else []
        )
        if not args.no_plots:
            plot_intervention_effects(args.output_dir, summary, effect_rows)
        summary["all_generations_style"] = all_style_summary
        summary["truncation"] = {
            "baseline_truncated_rate": truncated_rate(all_generations["baseline"]),
            "max_truncated_rate": args.max_truncated_rate,
            "style_untruncated_only": bool(args.style_untruncated_only),
            "untruncated_row_count": len(
                filter_untruncated_rows(
                    flat_rows,
                    {row["id"]: row for row in all_generations["baseline"]},
                )
            )
            if args.style_untruncated_only
            else None,
        }
    if args.intervention_space == "refined_residual":
        audit_file = "refined_residual_intervention_audit.csv"
        write_csv(os.path.join(args.output_dir, audit_file), refined_audit_rows)
        audit_by_mode = {
            mode: summarize_refined_intervention_audit(
                [row for row in refined_audit_rows if row.get("mode") == mode]
            )
            for mode in all_generations
            if mode != "baseline"
        }
        summary["refined_residual_intervention"] = {
            "audit_file": audit_file,
            "component_info": refined_component_info,
            "overall": summarize_refined_intervention_audit(refined_audit_rows),
            "modes": audit_by_mode,
            "visualizations": (
                ["plots/refined_runtime_fidelity.png"]
                if not args.no_plots
                else []
            ),
        }
        if not args.no_plots:
            plot_refined_runtime_audit(args.output_dir, audit_by_mode, final_specs)
    if untruncated_style_summary is not None:
        summary["all_generations_untruncated_style"] = untruncated_style_summary
    write_json(os.path.join(args.output_dir, "cluster_intervention_summary.json"), summary)

    run_config = vars(args).copy()
    run_config.update(
        {
            "layer_i": layer_i,
            "next_layer": next_layer,
            "feature_indices": feature_indices.tolist(),
            "record_layers": record_layers,
            "effective_apply_to_generated": effective_apply_to_generated(args),
            "effective_generated_patch_steps": effective_generated_patch_steps(args),
            "summary": summary,
        }
    )
    write_json(os.path.join(args.output_dir, "run_config.json"), run_config)
    if args.save_recorded_activations and recorder is not None:
        torch.save(recorder.records, os.path.join(args.output_dir, "recorded_selected_activations.pt"))
    mode_console = []
    for mode, values in summary.get("modes", {}).items():
        movement = values.get("source_cluster_style_shift_to_target", {})
        mode_console.append(
            {
                "mode": mode,
                "control": values.get("control"),
                "alpha": values.get("alpha"),
                "transition": f"{values.get('source_cluster')}->{values.get('target_cluster')}",
                "changed_text_rate": values.get("changed_text_rate"),
                "changed_answer_rate": values.get("changed_answer_rate"),
                "style_movement_to_target": movement.get("mean_delta_toward_target"),
            }
        )
    print("[cluster] intervention analysis complete")
    print(json.dumps({"modes": mode_console, "output_dir": args.output_dir}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(
        "cluster_intervention.py is an internal continuous-runner core. "
        "Use scripts/run_experiment.sh."
    )
