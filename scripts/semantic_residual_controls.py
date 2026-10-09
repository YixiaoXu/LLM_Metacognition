#!/usr/bin/env python
"""Cross-fitted semantic controls for continuous residual targets.

The helpers in this module deliberately fit several fixed probe families and
select the family with the highest selection-split R2.  This is conservative:
the strongest measured semantic explanation is removed, rather than the most
convenient one.  Evaluation rows are never used to choose the probe family.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


_REPORTED_MLP_BACKENDS: set[tuple[str, str]] = set()


class SemanticRegressionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        depth: int = 1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def regression_metrics(
    target: torch.Tensor, prediction: torch.Tensor
) -> Dict[str, float]:
    target = target.float().flatten()
    prediction = prediction.float().flatten()
    mse = float((target - prediction).square().mean().item())
    baseline = float(
        (target - target.mean()).square().mean().clamp_min(1e-12).item()
    )
    return {
        "n": int(target.numel()),
        "mse": mse,
        "mean_baseline_mse": baseline,
        "r2": float(1.0 - mse / baseline),
    }


def _ridge_fit(
    features: torch.Tensor, target: torch.Tensor, ridge: float
) -> Dict[str, torch.Tensor]:
    features = features.float()
    target = target.float().view(-1, 1)
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-5)
    target_mean = target.mean(dim=0, keepdim=True)
    design = (features - feature_mean) / feature_std
    scale = float(max(design.size(0), 1))
    gram = design.t().matmul(design) / scale
    gram = gram + max(float(ridge), 0.0) * torch.eye(
        gram.size(0), dtype=gram.dtype
    )
    rhs = design.t().matmul(target - target_mean) / scale
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


def _ridge_predict(
    state: Dict[str, torch.Tensor], features: torch.Tensor
) -> torch.Tensor:
    standardized = (
        features.float() - state["feature_mean"]
    ) / state["feature_std"]
    return (
        standardized.matmul(state["weight"]) + state["target_mean"]
    ).flatten()


def _mlp_fit(
    features: torch.Tensor,
    target: torch.Tensor,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    depth: int = 1,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    features = features.float().cpu()
    target = target.float().flatten().cpu()
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-5)
    target_mean = target.mean()
    target_std = target.std(unbiased=False).clamp_min(1e-5)
    standardized_x = (features - feature_mean) / feature_std
    standardized_y = ((target - target_mean) / target_std).view(-1, 1)

    # DataLoader with pin_memory still copies every sample from host memory on
    # every epoch.  These semantic probes revisit the same matrix many times,
    # so keep one standardized copy on the accelerator and shuffle only row
    # indices.  This preserves the minibatch objective, epoch count and seeded
    # sampling while removing the dominant CPU collation and PCIe traffic.
    training_backend = "cpu_indexed"
    training_x = standardized_x
    training_y = standardized_y
    if device.type == "cuda":
        try:
            training_x = standardized_x.to(device)
            training_y = standardized_y.to(device)
            training_backend = "cuda_resident"
        except torch.OutOfMemoryError:
            training_x = standardized_x
            training_y = standardized_y
            torch.cuda.empty_cache()
            training_backend = "cpu_indexed_oom_fallback"

    backend_key = (str(device), training_backend)
    if backend_key not in _REPORTED_MLP_BACKENDS:
        resident_mib = (
            training_x.numel() * training_x.element_size()
            + training_y.numel() * training_y.element_size()
        ) / (1024**2)
        print(
            f"[semantic-control] mlp_backend={training_backend} "
            f"device={device} rows={training_x.size(0)} "
            f"dim={training_x.size(1)} resident_mib={resident_mib:.1f}",
            flush=True,
        )
        _REPORTED_MLP_BACKENDS.add(backend_key)

    model = SemanticRegressionMLP(
        standardized_x.size(1),
        int(hidden_dim),
        float(dropout),
        int(depth),
    ).to(device)
    effective_batch_size = min(
        max(1, int(batch_size)), standardized_x.size(0)
    )
    shuffle_generator = torch.Generator().manual_seed(int(seed) + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    model.train()
    final_loss = math.nan
    for _ in range(max(1, int(epochs))):
        total = 0.0
        seen = 0
        permutation = torch.randperm(
            training_x.size(0), generator=shuffle_generator
        )
        for start in range(0, training_x.size(0), effective_batch_size):
            batch_indices = permutation[
                start : start + effective_batch_size
            ]
            if training_x.device.type == "cuda":
                batch_indices = batch_indices.to(device)
                batch_x = training_x.index_select(0, batch_indices)
                batch_y = training_y.index_select(0, batch_indices)
            else:
                batch_x = training_x.index_select(0, batch_indices).to(
                    device, non_blocking=device.type == "cuda"
                )
                batch_y = training_y.index_select(0, batch_indices).to(
                    device, non_blocking=device.type == "cuda"
                )
            prediction = model(batch_x)
            loss = F.mse_loss(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().item()) * batch_x.size(0)
            seen += batch_x.size(0)
        final_loss = total / max(seen, 1)
    return {
        "family": "mlp",
        "input_dim": int(standardized_x.size(1)),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "depth": int(depth),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean.view(1),
        "target_std": target_std.view(1),
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "epochs": int(epochs),
        "final_train_loss": float(final_loss),
        "training_backend": training_backend,
    }


def _mlp_predict(
    state: Dict[str, Any],
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model = SemanticRegressionMLP(
        int(state["input_dim"]),
        int(state["hidden_dim"]),
        float(state["dropout"]),
        int(state.get("depth", 1)),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    features = features.float().cpu()
    standardized = (
        features - state["feature_mean"]
    ) / state["feature_std"]
    chunks = []
    with torch.inference_mode():
        for start in range(0, standardized.size(0), max(1, int(batch_size))):
            value = standardized[start : start + int(batch_size)].to(
                device, non_blocking=device.type == "cuda"
            )
            chunks.append(model(value).flatten().float().cpu())
    prediction = torch.cat(chunks)
    return (
        prediction * torch.as_tensor(state["target_std"]).flatten()[0]
        + torch.as_tensor(state["target_mean"]).flatten()[0]
    )


def _fold_assignment(n: int, folds: int, seed: int) -> torch.Tensor:
    folds = max(2, min(int(folds), int(n)))
    permutation = torch.randperm(
        n, generator=torch.Generator().manual_seed(int(seed))
    )
    assignment = torch.empty(n, dtype=torch.long)
    assignment[permutation] = torch.arange(n) % folds
    return assignment


def crossfit_strongest_semantic_prediction(
    semantic_controls: torch.Tensor,
    target: torch.Tensor,
    train_idx: torch.Tensor,
    selection_idx: torch.Tensor,
    evaluation_idx: torch.Tensor,
    *,
    families: Sequence[str] = ("ridge", "mlp"),
    folds: int = 5,
    ridge: float = 1e-2,
    mlp_hidden_dim: int = 128,
    mlp_dropout: float = 0.10,
    mlp_epochs: int = 12,
    mlp_batch_size: int = 512,
    mlp_lr: float = 1e-3,
    mlp_weight_decay: float = 1e-4,
    mlp_repeats: int = 1,
    device: torch.device | str = "cpu",
    seed: int = 42,
    progress_desc: str = "Strong semantic control",
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    """Return predictions from the strongest selection-split probe family."""
    device = torch.device(device)
    controls = semantic_controls.float().cpu()
    target = target.float().flatten().cpu()
    train_idx = train_idx.long().cpu()
    selection_idx = selection_idx.long().cpu()
    evaluation_idx = evaluation_idx.long().cpu()
    normalized_families = []
    for family in families:
        family = str(family).strip().lower()
        if family not in {"ridge", "mlp", "mlp_deep"}:
            raise ValueError(f"Unsupported semantic probe family: {family}")
        if family not in normalized_families:
            normalized_families.append(family)
    if not normalized_families:
        raise ValueError("At least one semantic probe family is required.")

    assignment = _fold_assignment(train_idx.numel(), folds, seed)
    candidate_predictions: Dict[str, torch.Tensor] = {}
    candidate_states: Dict[str, Any] = {}
    candidate_diagnostics: Dict[str, Any] = {}
    total_fits = len(normalized_families) * (
        int(assignment.max().item()) + 2
    )
    mlp_family_count = sum(
        family in {"mlp", "mlp_deep"} for family in normalized_families
    )
    if mlp_family_count:
        total_fits += mlp_family_count * (
            max(1, int(mlp_repeats)) - 1
        ) * (
            int(assignment.max().item()) + 2
        )
    progress = tqdm(total=total_fits, desc=progress_desc, leave=False)

    for family_index, family in enumerate(normalized_families):
        repeat_predictions = []
        repeat_states = []
        repeat_diagnostics = []
        repeats = (
            max(1, int(mlp_repeats))
            if family in {"mlp", "mlp_deep"}
            else 1
        )
        for repeat in range(repeats):
            prediction = torch.empty_like(target)
            for fold in range(int(assignment.max().item()) + 1):
                held_local = (assignment == fold).nonzero(
                    as_tuple=False
                ).flatten()
                fit_local = (assignment != fold).nonzero(
                    as_tuple=False
                ).flatten()
                held_idx = train_idx.index_select(0, held_local)
                fit_idx = train_idx.index_select(0, fit_local)
                fit_x = controls.index_select(0, fit_idx)
                fit_y = target.index_select(0, fit_idx)
                held_x = controls.index_select(0, held_idx)
                fit_seed = (
                    int(seed)
                    + 100003 * family_index
                    + 1009 * repeat
                    + 17 * fold
                )
                if family == "ridge":
                    fold_state = _ridge_fit(fit_x, fit_y, ridge)
                    fold_prediction = _ridge_predict(fold_state, held_x)
                else:
                    fold_state = _mlp_fit(
                        fit_x,
                        fit_y,
                        mlp_hidden_dim,
                        mlp_dropout,
                        mlp_epochs,
                        mlp_batch_size,
                        mlp_lr,
                        mlp_weight_decay,
                        device,
                        fit_seed,
                        2 if family == "mlp_deep" else 1,
                    )
                    fold_prediction = _mlp_predict(
                        fold_state, held_x, device, mlp_batch_size
                    )
                prediction.index_copy_(0, held_idx, fold_prediction)
                progress.update(1)

            full_x = controls.index_select(0, train_idx)
            full_y = target.index_select(0, train_idx)
            full_seed = int(seed) + 100003 * family_index + 1009 * repeat + 997
            if family == "ridge":
                full_state = _ridge_fit(full_x, full_y, ridge)
                nontrain_prediction = _ridge_predict(full_state, controls)
            else:
                full_state = _mlp_fit(
                    full_x,
                    full_y,
                    mlp_hidden_dim,
                    mlp_dropout,
                    mlp_epochs,
                    mlp_batch_size,
                    mlp_lr,
                    mlp_weight_decay,
                    device,
                    full_seed,
                    2 if family == "mlp_deep" else 1,
                )
                nontrain_prediction = _mlp_predict(
                    full_state, controls, device, mlp_batch_size
                )
            nontrain = torch.ones(target.numel(), dtype=torch.bool)
            nontrain[train_idx] = False
            prediction[nontrain] = nontrain_prediction[nontrain]
            progress.update(1)
            repeat_predictions.append(prediction)
            repeat_states.append(full_state)
            repeat_diagnostics.append(
                {
                    "repeat": repeat,
                    "train_oof": regression_metrics(
                        target.index_select(0, train_idx),
                        prediction.index_select(0, train_idx),
                    ),
                    "selection": regression_metrics(
                        target.index_select(0, selection_idx),
                        prediction.index_select(0, selection_idx),
                    ),
                    "evaluation": regression_metrics(
                        target.index_select(0, evaluation_idx),
                        prediction.index_select(0, evaluation_idx),
                    ),
                }
            )
        averaged = torch.stack(repeat_predictions).mean(dim=0)
        candidate_predictions[family] = averaged
        candidate_states[family] = repeat_states
        candidate_diagnostics[family] = {
            "family": family,
            "repeats": repeats,
            "repeat_diagnostics": repeat_diagnostics,
            "train_oof": regression_metrics(
                target.index_select(0, train_idx),
                averaged.index_select(0, train_idx),
            ),
            "selection": regression_metrics(
                target.index_select(0, selection_idx),
                averaged.index_select(0, selection_idx),
            ),
            "evaluation": regression_metrics(
                target.index_select(0, evaluation_idx),
                averaged.index_select(0, evaluation_idx),
            ),
        }
    progress.close()

    selected_family = max(
        normalized_families,
        key=lambda family: candidate_diagnostics[family]["selection"]["r2"],
    )
    selected_prediction = candidate_predictions[selected_family]
    selected_diagnostics = candidate_diagnostics[selected_family]
    max_r2 = {
        split: max(
            candidate_diagnostics[family][split]["r2"]
            for family in normalized_families
        )
        for split in ("train_oof", "selection", "evaluation")
    }
    diagnostics = {
        **selected_diagnostics,
        "mode": "crossfit_strongest",
        "selected_family": selected_family,
        "selection_rule": "maximum_selection_split_r2",
        "candidate_diagnostics": candidate_diagnostics,
        "max_probe_r2": max_r2,
        "folds": int(assignment.max().item()) + 1,
        "ridge": float(ridge),
        "mlp_hidden_dim": int(mlp_hidden_dim),
        "mlp_epochs": int(mlp_epochs),
        "mlp_repeats": int(mlp_repeats),
    }
    teacher_state = {
        "mode": "crossfit_strongest",
        "selected_family": selected_family,
        "candidate_states": candidate_states,
        "selection_rule": "maximum_selection_split_r2",
    }
    print(
        f"[semantic-control] selected={selected_family} "
        f"selection_R2={selected_diagnostics['selection']['r2']:.4f} "
        f"heldout_R2={selected_diagnostics['evaluation']['r2']:.4f} "
        f"max_heldout_R2={max_r2['evaluation']:.4f}",
        flush=True,
    )
    return selected_prediction, diagnostics, teacher_state
