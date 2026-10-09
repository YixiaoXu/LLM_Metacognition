#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import re
import shutil
import weakref
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from tqdm import tqdm as _tqdm

from publication_plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    apply_nmi_style,
    panel_label,
    save_figure,
    style_axis,
)


_COMPACT_PROGRESS = False
_DEVICE_TENSOR_CACHE_MAX_ITEMS = 64
_DEVICE_TENSOR_CACHE: "OrderedDict[Tuple[Any, ...], Tuple[Any, torch.Tensor]]" = OrderedDict()


def set_compact_progress(enabled: bool) -> None:
    global _COMPACT_PROGRESS
    _COMPACT_PROGRESS = bool(enabled)


def compact_progress_enabled() -> bool:
    return _COMPACT_PROGRESS


def tqdm(*args, **kwargs):
    """Hide nested progress bars while a top-level compact bar is active."""
    if _COMPACT_PROGRESS:
        kwargs["disable"] = True
    return _tqdm(*args, **kwargs)


def cached_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Move an immutable tensor once and reuse the converted device copy.

    Dynamic residual and erasure states are replaced, not mutated, between
    refreshes. A weak source reference prevents an object-id reuse from
    returning a stale tensor, while the bounded LRU avoids retaining old GPU
    states indefinitely.
    """

    target_dtype = tensor.dtype if dtype is None else dtype
    if tensor.device == device and tensor.dtype == target_dtype:
        return tensor
    key = (id(tensor), str(device), target_dtype)
    cached = _DEVICE_TENSOR_CACHE.get(key)
    if cached is not None and cached[0]() is tensor:
        _DEVICE_TENSOR_CACHE.move_to_end(key)
        return cached[1]
    converted = tensor.to(
        device=device,
        dtype=target_dtype,
        non_blocking=bool(device.type == "cuda" and tensor.device.type == "cpu" and tensor.is_pinned()),
    )
    _DEVICE_TENSOR_CACHE[key] = (weakref.ref(tensor), converted)
    _DEVICE_TENSOR_CACHE.move_to_end(key)
    while len(_DEVICE_TENSOR_CACHE) > _DEVICE_TENSOR_CACHE_MAX_ITEMS:
        _DEVICE_TENSOR_CACHE.popitem(last=False)
    return converted


def tensor_pair_loader(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create the probe loader with asynchronous host transfer support."""

    return DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=device.type == "cuda")


FirstStageRegularizer = Callable[
    [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, int, bool],
    Dict[str, torch.Tensor],
]
_FIRST_STAGE_REGULARIZER: Optional[FirstStageRegularizer] = None


def set_first_stage_regularizer(regularizer: Optional[FirstStageRegularizer]) -> None:
    """Install an optional experimental regularizer without changing default training."""
    global _FIRST_STAGE_REGULARIZER
    _FIRST_STAGE_REGULARIZER = regularizer


def first_stage_regularizer_metrics(
    out: Dict[str, torch.Tensor],
    z2_signal: torch.Tensor,
    x_prev: torch.Tensor,
    epoch: int,
    training: bool,
) -> Dict[str, torch.Tensor]:
    if _FIRST_STAGE_REGULARIZER is None:
        return {}
    metrics = _FIRST_STAGE_REGULARIZER(out, z2_signal, x_prev, epoch, training)
    if "penalty" not in metrics:
        raise ValueError("First-stage regularizer must return a 'penalty' tensor.")
    return metrics


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoupler(nn.Module):
    def __init__(self, dim: int, latent_dim: int, hidden_dim: int, dropout: float, target_mode: str):
        super().__init__()
        self.e1 = MLP(dim, latent_dim, hidden_dim, dropout)
        self.e2 = MLP(dim, latent_dim, hidden_dim, dropout)
        self.d_next = MLP(latent_dim * 2, dim, hidden_dim, dropout)
        self.d_prev = MLP(latent_dim, dim, hidden_dim, dropout)
        self.predict_i = MLP(latent_dim, dim, hidden_dim, dropout)
        self.predict_i_semantic = MLP(latent_dim, dim, hidden_dim, dropout)
        self.e2_prev_adversary = MLP(latent_dim, dim, hidden_dim, dropout)
        self.target_mode = target_mode

    def forward(self, x_next: torch.Tensor) -> Dict[str, torch.Tensor]:
        z1 = self.e1(x_next)
        z2 = self.e2(x_next)
        x_next_hat = self.d_next(torch.cat([z1, z2], dim=-1))
        x_prev_hat = self.d_prev(z1)
        x_i_hat = self.predict_i(z2)
        x_i_semantic_hat = self.predict_i_semantic(z1)
        return {
            "z1": z1,
            "z2": z2,
            "x_next_hat": x_next_hat,
            "x_prev_hat": x_prev_hat,
            "x_i_hat": x_i_hat,
            "x_i_semantic_hat": x_i_semantic_hat,
        }


class E2RecursivePurifier(nn.Module):
    def __init__(
        self,
        z2_dim: int,
        semantic_dim: int,
        meta_dim: int,
        hidden_dim: int,
        dropout: float,
        z1_dim: int,
        prev_dim: int,
        target_dim: int,
        meta_input_mode: str = "z2",
    ):
        super().__init__()
        if meta_input_mode not in {"z2", "semantic_residual"}:
            raise ValueError(f"Unsupported purifier meta_input_mode: {meta_input_mode}")
        self.semantic_encoder = MLP(z2_dim, semantic_dim, hidden_dim, dropout)
        self.semantic_to_z2 = MLP(semantic_dim, z2_dim, hidden_dim, dropout)
        self.meta_encoder = MLP(z2_dim, meta_dim, hidden_dim, dropout)
        self.meta_to_residual = MLP(meta_dim, z2_dim, hidden_dim, dropout)
        self.reconstruct_z2 = MLP(semantic_dim + meta_dim, z2_dim, hidden_dim, dropout)
        self.semantic_to_z1 = MLP(semantic_dim, z1_dim, hidden_dim, dropout)
        self.semantic_to_prev = MLP(semantic_dim, prev_dim, hidden_dim, dropout)
        self.semantic_to_i = MLP(semantic_dim, target_dim, hidden_dim, dropout)
        self.meta_to_i = MLP(meta_dim, target_dim, hidden_dim, dropout)
        self.z2_dim = z2_dim
        self.semantic_dim = semantic_dim
        self.meta_dim = meta_dim
        self._meta_input_mode = meta_input_mode
        self.register_buffer(
            "_meta_input_mode_code",
            torch.tensor(1 if meta_input_mode == "semantic_residual" else 0, dtype=torch.long),
        )

    @property
    def meta_input_mode(self) -> str:
        return self._meta_input_mode

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        mode_key = prefix + "_meta_input_mode_code"
        if mode_key in state_dict:
            self._meta_input_mode = "semantic_residual" if int(state_dict[mode_key].item()) == 1 else "z2"
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Backward compatibility for purifier checkpoints saved before the
        # semantic baseline prediction head was added.
        semantic_i_prefix = prefix + "semantic_to_i."
        missing_keys[:] = [key for key in missing_keys if not key.startswith(semantic_i_prefix)]

    def forward(self, z2: torch.Tensor) -> Dict[str, torch.Tensor]:
        semantic = self.semantic_encoder(z2)
        semantic_z2_hat = self.semantic_to_z2(semantic)
        if self.meta_input_mode == "semantic_residual":
            meta_input = z2 - semantic_z2_hat.detach()
        else:
            meta_input = z2
        meta = self.meta_encoder(meta_input)
        meta_residual_hat = self.meta_to_residual(meta)
        return {
            "semantic": semantic,
            "meta": meta,
            "semantic_z2_hat": semantic_z2_hat,
            "meta_input": meta_input,
            "meta_residual_hat": meta_residual_hat,
            "z2_hat": self.reconstruct_z2(torch.cat([semantic, meta], dim=-1)),
            "z1_hat": self.semantic_to_z1(semantic),
            "prev_hat": self.semantic_to_prev(semantic),
            "semantic_i_hat": self.semantic_to_i(semantic),
            "i_hat": self.meta_to_i(meta),
        }


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def configure_purifier_training_stage(
    purifier: E2RecursivePurifier,
    stage: str,
    freeze_semantic_after_warmup: bool,
) -> None:
    if stage == "semantic_encoder":
        set_requires_grad(purifier, False)
        set_requires_grad(purifier.semantic_encoder, True)
        set_requires_grad(purifier.semantic_to_z1, True)
        set_requires_grad(purifier.semantic_to_prev, True)
        set_requires_grad(purifier.semantic_to_i, True)
        return
    if stage == "semantic_z2_decoder":
        set_requires_grad(purifier, False)
        set_requires_grad(purifier.semantic_to_z2, True)
        return
    if stage != "residual_meta":
        raise ValueError(f"Unknown purifier training stage: {stage}")
    set_requires_grad(purifier, True)
    if freeze_semantic_after_warmup:
        set_requires_grad(purifier.semantic_encoder, False)
        set_requires_grad(purifier.semantic_to_z2, False)
        set_requires_grad(purifier.semantic_to_z1, False)
        set_requires_grad(purifier.semantic_to_prev, False)
        set_requires_grad(purifier.semantic_to_i, False)


def load_layer(path: str) -> Tuple[list, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    return obj["ids"], obj["features"].float()


def load_activation_manifest(activation_dir: str) -> Dict[str, Any]:
    path = os.path.join(activation_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_num_layers_from_cache(activation_dir: str) -> Optional[int]:
    layer_ids = []
    for filename in os.listdir(activation_dir):
        if filename.startswith("layer_") and filename.endswith(".pt"):
            try:
                layer_ids.append(int(filename[len("layer_") : -len(".pt")]))
            except ValueError:
                pass
    if not layer_ids:
        return None
    return max(layer_ids) + 1


def fraction_to_layer_id(fraction: float, num_layers: int) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Layer fraction must be in [0, 1], got {fraction}.")
    return max(0, min(num_layers - 1, int(math.floor(fraction * num_layers + 0.5))))


def resolve_layer_triplet(args: argparse.Namespace, manifest: Dict[str, Any]) -> Tuple[int, int, int, Optional[List[float]], Optional[int]]:
    num_layers = manifest.get("num_model_layers")
    if num_layers is None:
        num_layers = infer_num_layers_from_cache(args.activation_dir)
    num_layers = int(num_layers) if num_layers is not None else None

    layer_fracs = args.layer_fracs
    if layer_fracs is not None:
        if num_layers is None:
            raise ValueError("--layer-fracs requires activation manifest.json or a cache with layer_*.pt files.")
        prev_layer, layer_i, next_layer = [fraction_to_layer_id(value, num_layers) for value in layer_fracs]
        args.prev_layer = prev_layer
        args.layer_i = layer_i
        args.next_layer = next_layer
        return prev_layer, layer_i, next_layer, list(layer_fracs), num_layers

    if args.layer_i is None:
        raise ValueError("Set --layer-i, or use --layer-fracs PREV_FRAC I_FRAC NEXT_FRAC for model-size-normalized layer selection.")
    prev_layer = args.prev_layer if args.prev_layer is not None else args.layer_i - args.prev_offset
    next_layer = args.next_layer if args.next_layer is not None else args.layer_i + args.next_offset
    if num_layers is not None:
        layer_fracs = [prev_layer / num_layers, args.layer_i / num_layers, next_layer / num_layers]
    return prev_layer, args.layer_i, next_layer, layer_fracs, num_layers


def load_semantic_anchor_consensus(
    activation_dir: str,
    layer_i: int,
    fallback_prev_layer: int,
    offsets: Optional[List[int]],
    weights: Optional[List[float]],
) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build a shared semantic target from independently standardized earlier layers."""
    if not offsets:
        ids, raw = load_layer(os.path.join(activation_dir, f"layer_{fallback_prev_layer:03d}.pt"))
        standardized, mean, std = standardize(raw)
        return ids, standardized, mean, std, {
            "mode": "single_layer",
            "offsets": [layer_i - fallback_prev_layer],
            "layers": [fallback_prev_layer],
            "weights": [1.0],
            "anchor_means": [mean],
            "anchor_stds": [std],
            "consensus_mean": torch.zeros_like(mean),
            "consensus_std": torch.ones_like(std),
        }

    normalized_offsets = [int(value) for value in offsets]
    if any(value <= 0 for value in normalized_offsets):
        raise ValueError("--semantic-anchor-offsets must contain positive integers.")
    if len(set(normalized_offsets)) != len(normalized_offsets):
        raise ValueError("--semantic-anchor-offsets must not contain duplicates.")
    anchor_layers = [layer_i - value for value in normalized_offsets]
    if any(value < 0 for value in anchor_layers):
        raise ValueError(
            f"Semantic anchor layers {anchor_layers} are invalid for layer-i={layer_i}."
        )

    if weights is None:
        normalized_weights = torch.full((len(anchor_layers),), 1.0 / len(anchor_layers))
    else:
        if len(weights) != len(anchor_layers):
            raise ValueError("--semantic-anchor-weights must match --semantic-anchor-offsets in length.")
        normalized_weights = torch.tensor(weights, dtype=torch.float32)
        if not torch.isfinite(normalized_weights).all() or (normalized_weights < 0).any():
            raise ValueError("--semantic-anchor-weights must be finite and non-negative.")
        if normalized_weights.sum().item() <= 0.0:
            raise ValueError("--semantic-anchor-weights must have a positive sum.")
        normalized_weights /= normalized_weights.sum()

    anchor_ids: Optional[list] = None
    anchor_features = []
    anchor_means = []
    anchor_stds = []
    for layer in anchor_layers:
        ids, raw = load_layer(os.path.join(activation_dir, f"layer_{layer:03d}.pt"))
        if anchor_ids is None:
            anchor_ids = ids
        elif ids != anchor_ids:
            raise ValueError("Semantic anchor layer files have different example ids/order.")
        standardized, mean, std = standardize(raw)
        anchor_features.append(standardized)
        anchor_means.append(mean)
        anchor_stds.append(std)

    consensus_raw = sum(
        float(weight) * feature
        for weight, feature in zip(normalized_weights.tolist(), anchor_features)
    )
    consensus, consensus_mean, consensus_std = standardize(consensus_raw)
    assert anchor_ids is not None
    return anchor_ids, consensus, consensus_mean, consensus_std, {
        "mode": "standardized_weighted_consensus",
        "offsets": normalized_offsets,
        "layers": anchor_layers,
        "weights": normalized_weights.tolist(),
        "anchor_means": anchor_means,
        "anchor_stds": anchor_stds,
        "consensus_mean": consensus_mean,
        "consensus_std": consensus_std,
    }


def unpack_batch(batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        x_prev, x_i_train, x_next = batch
        return x_prev, x_i_train, x_i_train, x_next
    if len(batch) == 4:
        x_prev, x_i_train, x_i_eval, x_next = batch
        return x_prev, x_i_train, x_i_eval, x_next
    raise ValueError(f"Expected 3 or 4 tensors per batch, got {len(batch)}.")


def standardize(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std, mean, std


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def orthogonality_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1 - z1.mean(dim=0, keepdim=True), dim=0)
    z2 = F.normalize(z2 - z2.mean(dim=0, keepdim=True), dim=0)
    cross = z1.T @ z2 / z1.size(0)
    return cross.pow(2).mean()


def cross_cov_loss(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    cov = z.T @ x / z.size(0)
    return cov.pow(2).mean()


def variance_floor_loss(z: torch.Tensor, floor: float = 0.5) -> torch.Tensor:
    std = z.std(dim=0, unbiased=False)
    return F.relu(floor - std).pow(2).mean()


def margin_adversary_penalty(adversary_mse: torch.Tensor, margin: float) -> torch.Tensor:
    return F.relu(margin - adversary_mse)


def ramp_weight(max_weight: float, epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if max_weight <= 0.0 or epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return max_weight
    return max_weight * min(1.0, (epoch - start_epoch + 1) / ramp_epochs)


def dynamic_selection_limit(
    ranked_count: int,
    top_k: int,
    min_selected: int,
    fallback_used: bool,
) -> int:
    """Use only the warm-start floor when strict target selection falls back."""
    if ranked_count <= 0:
        return 0
    if fallback_used and min_selected > 0:
        return min(ranked_count, min_selected)
    limit = ranked_count if top_k <= 0 else min(ranked_count, top_k)
    return max(limit, min(ranked_count, min_selected))


def make_binary_targets(x_i: torch.Tensor, quantile: float) -> Tuple[torch.Tensor, torch.Tensor]:
    threshold = torch.quantile(x_i.abs(), quantile, dim=0, keepdim=True)
    return (x_i.abs() >= threshold).float(), threshold


def training_target_mode(mode: str) -> str:
    return "continuous" if mode == "continuous_binary" else mode


def is_binary_eval_mode(mode: str) -> bool:
    return mode in {"binary", "continuous_binary"}


def continuous_binary_logits(pred: torch.Tensor, binary_eval_meta: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = cached_to_device(binary_eval_meta["mean"], pred.device, pred.dtype)
    std = cached_to_device(binary_eval_meta["std"], pred.device, pred.dtype)
    threshold = cached_to_device(binary_eval_meta["threshold"], pred.device, pred.dtype)
    raw_pred = pred * std + mean
    return (raw_pred.abs() - threshold) / threshold.clamp_min(1e-6)


def binary_eval_logits(pred: torch.Tensor, mode: str, binary_eval_meta: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    if mode == "binary":
        return pred
    if mode == "continuous_binary":
        if binary_eval_meta is None:
            raise ValueError("continuous_binary evaluation requires binary_eval_meta.")
        return continuous_binary_logits(pred, binary_eval_meta)
    raise ValueError(f"Mode {mode} does not use binary evaluation.")


def prediction_loss(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    mode = training_target_mode(mode)
    if mode == "continuous":
        return normalized_mse(pred, target)
    if mode == "binary":
        return F.binary_cross_entropy_with_logits(pred, target)
    raise ValueError(mode)


def weighted_feature_mean(loss: torch.Tensor, feature_weights: Optional[torch.Tensor]) -> torch.Tensor:
    if feature_weights is None:
        return loss.mean()
    weights = cached_to_device(feature_weights, loss.device, loss.dtype).view(1, -1)
    denom = weights.sum().clamp_min(1e-8) * loss.size(0)
    return (loss * weights).sum() / denom


def weighted_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    feature_weights: Optional[torch.Tensor] = None,
    projection_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if projection_matrix is not None:
        projection = cached_to_device(projection_matrix, pred.device, pred.dtype)
        pred = pred.float() @ projection
        target = target.float() @ projection
        if feature_weights is None:
            return F.mse_loss(pred, target)
        return weighted_feature_mean((pred - target).pow(2), feature_weights)
    mode = training_target_mode(mode)
    if feature_weights is None:
        return prediction_loss(pred, target, mode)
    if mode == "continuous":
        return weighted_feature_mean((pred - target).pow(2), feature_weights)
    if mode == "binary":
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return weighted_feature_mean(loss, feature_weights)
    raise ValueError(mode)


def apply_semantic_residual_state(
    z2: torch.Tensor,
    state: Optional[Dict[str, List[torch.Tensor]]],
) -> torch.Tensor:
    if not state or not state.get("bases"):
        return z2
    out = z2
    means = state.get("means", [])
    bases = state.get("bases", [])
    for mean, basis in zip(means, bases):
        if basis.numel() == 0:
            continue
        mean = cached_to_device(mean, out.device, out.dtype)
        basis = cached_to_device(basis, out.device, out.dtype)
        centered = out - mean
        out = centered - (centered @ basis) @ basis.T + mean
    return out


def has_semantic_residual_state(state: Optional[Dict[str, List[torch.Tensor]]]) -> bool:
    return bool(state and state.get("bases"))


def predict_i_with_optional_residual(
    model: Decoupler,
    z2: torch.Tensor,
    residual_state: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> torch.Tensor:
    return model.predict_i(apply_semantic_residual_state(z2, residual_state))


def main_prediction_components(
    model: Decoupler,
    z1: torch.Tensor,
    z2_signal: torch.Tensor,
    prediction_target: str = "raw",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined prediction, semantic baseline, and meta delta/raw head.

    raw:      E2 predicts the original target directly.
    residual: Z1 predicts a detached semantic baseline and E2 predicts the
              remaining delta, so the prediction objective no longer rewards
              E2 for relearning what the semantic channel can already explain.
    """
    if prediction_target not in {"raw", "residual"}:
        raise ValueError(f"Unsupported main prediction target: {prediction_target}")
    semantic_hat = model.predict_i_semantic(z1)
    meta_hat = model.predict_i(z2_signal)
    if prediction_target == "residual":
        return semantic_hat.detach() + meta_hat, semantic_hat, meta_hat
    return meta_hat, semantic_hat, meta_hat


def binary_prior_logits(train_target: torch.Tensor, val_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prior = train_target.float().mean(dim=0, keepdim=True).clamp(eps, 1.0 - eps)
    return torch.logit(prior).expand_as(val_target).contiguous()


def binary_ce_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    feature_weights: Optional[torch.Tensor] = None,
) -> float:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    if feature_weights is None:
        return loss.mean().item()
    weights = feature_weights.to(device=loss.device, dtype=loss.dtype).view(1, -1)
    return (loss * weights).sum().div(loss.size(0) * weights.sum().clamp_min(1e-8)).item()


def binary_information_metrics(
    name: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    prior_ce: float,
    threshold: float,
    feature_scope: str = "all",
    feature_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    ce = binary_ce_from_logits(logits, target, feature_weights)
    info_nats = prior_ce - ce
    probs = torch.sigmoid(logits.float())
    target = target.float()
    if feature_weights is None:
        scoped_probs = probs
        scoped_target = target
        feature_weight_sum = float(target.size(1))
    else:
        selected = feature_weights.float() > 1e-8
        if selected.sum().item() == 0:
            scoped_probs = probs
            scoped_target = target
        else:
            scoped_probs = probs[:, selected]
            scoped_target = target[:, selected]
        feature_weight_sum = float(feature_weights.float().sum().item())
    row: Dict[str, float] = {
        "name": name,
        "feature_scope": feature_scope,
        "feature_weight_sum": feature_weight_sum,
        "ce_nats_per_label": ce,
        "prior_ce_nats_per_label": prior_ce,
        "info_gain_nats_per_label_vs_prior": info_nats,
        "info_gain_bits_per_label_vs_prior": info_nats / math.log(2.0),
        "relative_ce_reduction_vs_prior": info_nats / max(prior_ce, 1e-8),
        "positive_rate": scoped_target.mean().item(),
        "mean_predicted_probability": scoped_probs.mean().item(),
    }
    return row


def information_comparison_row(name: str, stronger: Dict, weaker: Dict) -> Dict[str, float]:
    delta_nats = weaker["ce_nats_per_label"] - stronger["ce_nats_per_label"]
    return {
        "comparison": name,
        "feature_scope": stronger.get("feature_scope", "all"),
        "stronger_model": stronger["name"],
        "weaker_model": weaker["name"],
        "delta_ce_nats_per_label": delta_nats,
        "delta_bits_per_label": delta_nats / math.log(2.0),
        "relative_ce_reduction": delta_nats / max(weaker["ce_nats_per_label"], 1e-8),
    }


def gaussian_mi_proxy_from_r2(r2: float) -> float:
    r2_clamped = min(max(float(r2), 0.0), 0.999999)
    return -0.5 * math.log(1.0 - r2_clamped)


def per_neuron_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    min_activation_rate: float,
    max_activation_rate: float,
    min_pos: int,
    min_neg: int,
) -> List[Dict]:
    probs = torch.sigmoid(logits.float())
    target = target.float()
    ce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    rows = []
    for neuron in range(target.size(1)):
        y = target[:, neuron]
        prob = probs[:, neuron]
        pos = int(y.sum().item())
        neg = int(y.numel() - pos)
        activation_rate = y.mean().item()
        valid = (
            activation_rate >= min_activation_rate
            and activation_rate <= max_activation_rate
            and pos >= min_pos
            and neg >= min_neg
        )
        if pos > 0:
            mean_prob_active = prob[y == 1].mean().item()
        else:
            mean_prob_active = float("nan")
        if neg > 0:
            mean_prob_inactive = prob[y == 0].mean().item()
        else:
            mean_prob_inactive = float("nan")
        rows.append(
            {
                "neuron": neuron,
                "num_positive": pos,
                "num_negative": neg,
                "activation_rate": activation_rate,
                "mean_predicted_probability": prob.mean().item(),
                "ce_nats": ce[:, neuron].mean().item(),
                "mean_prob_active": mean_prob_active,
                "mean_prob_inactive": mean_prob_inactive,
                "probability_margin": mean_prob_active - mean_prob_inactive,
                "is_variable": valid,
            }
        )
    return rows


def write_csv(path: str, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows and fieldnames is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _first_existing_key(row: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return key
    return None


def load_fixed_target_weights(args: argparse.Namespace, target_dim: int) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, torch.Tensor]]]:
    path = getattr(args, "fixed_target_file", "")
    if not path:
        return None, None, None
    if not os.path.exists(path):
        raise FileNotFoundError(f"--fixed-target-file does not exist: {path}")
    weights = torch.zeros(target_dim, dtype=torch.float32)
    rows: List[Dict[str, Any]] = []
    source_meta: Dict[str, Any] = {"path": path}
    if path.endswith(".pt") or path.endswith(".pth"):
        bundle = torch.load(path, map_location="cpu")
        if isinstance(bundle, dict) and "weights" in bundle:
            raw = bundle["weights"].float().flatten()
            source_meta.update({k: v for k, v in bundle.items() if k != "weights" and not torch.is_tensor(v)})
        elif torch.is_tensor(bundle):
            raw = bundle.float().flatten()
        else:
            raise ValueError(f"Unsupported fixed target pt format: {path}")
        if raw.numel() != target_dim:
            raise ValueError(f"Fixed target weight dim mismatch: got {raw.numel()}, expected {target_dim}.")
        weights = raw.clone()
        selected = torch.nonzero(weights > 0, as_tuple=False).flatten()
        rows = [{"feature_index": int(idx), "weight": float(weights[idx].item())} for idx in selected.tolist()]
    else:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            raw_rows = list(reader)
        index_column = args.fixed_target_index_column
        weight_column = args.fixed_target_weight_column
        if raw_rows and not index_column:
            index_column = _first_existing_key(raw_rows[0], ["feature_index", "neuron", "target_index", "index"])
        if raw_rows and not weight_column:
            weight_column = _first_existing_key(raw_rows[0], ["weight", "target_weight", "training_weight", "residual_fraction", "residual_gap_bits"])
        if not index_column or not weight_column:
            raise ValueError("Could not infer fixed target index/weight columns; pass --fixed-target-index-column and --fixed-target-weight-column.")
        parsed = []
        for row in raw_rows:
            try:
                idx = int(float(row[index_column]))
                weight = float(row[weight_column])
            except (KeyError, TypeError, ValueError):
                continue
            if idx < 0 or idx >= target_dim or not math.isfinite(weight):
                continue
            if weight < args.fixed_target_min_weight:
                continue
            parsed.append((idx, max(0.0, weight), row))
        parsed.sort(key=lambda item: item[1], reverse=True)
        if args.fixed_target_top_k > 0:
            parsed = parsed[: args.fixed_target_top_k]
        for idx, weight, row in parsed:
            weights[idx] = weight
            rows.append({**row, "feature_index": idx, "weight": weight})
        source_meta.update({"index_column": index_column, "weight_column": weight_column})
    if args.fixed_target_binarize:
        weights = (weights > args.fixed_target_min_weight).float()
    if args.fixed_target_normalize and weights.sum() > 0:
        weights = weights * (float((weights > 0).sum().item()) / weights.sum().clamp_min(1e-8))
    selected = weights > args.dynamic_target_export_threshold
    summary = {
        "epoch": 0,
        "target_type": "neuron",
        "source": "fixed_target_file",
        "fixed_target_file": path,
        "selected_count": int(selected.sum().item()),
        "eligible_count": int(selected.sum().item()),
        "strict_eligible_count": int(selected.sum().item()),
        "variable_count": target_dim,
        "weight_sum": float(weights.sum().item()),
        "weight_mean": float(weights.mean().item()) if weights.numel() else float("nan"),
        "weight_nonzero": int((weights > 0).sum().item()),
        "prediction_weight_scale": 1.0 if selected.any() else 0.0,
        "score_mean_selected": float("nan"),
        "e2_gain_nats_mean_selected": float("nan"),
        "e1_gain_nats_mean_selected": float("nan"),
        "semantic_gap_nats_mean_selected": float("nan"),
        "semantic_gain_nats_mean_selected": float("nan"),
        "semantic_gap_ci_low_mean_selected": float("nan"),
        "dynamic_residual_fraction_mean_selected": float("nan"),
        "dynamic_leakage_explained_fraction_mean_selected": float("nan"),
        "selection_rule": "fixed",
        "selection_objective": "targeted_refit_from_previous_purifier",
        **source_meta,
    }
    tensors = {
        "weights": weights,
        "score": torch.full_like(weights, float("nan")),
        "semantic_gap_nats": torch.full_like(weights, float("nan")),
        "e2_gain_nats": torch.full_like(weights, float("nan")),
        "main_e2_gain_nats": torch.full_like(weights, float("nan")),
        "semantic_gain_nats": torch.full_like(weights, float("nan")),
        "prior_ce": torch.full_like(weights, float("nan")),
        "e2_ce": torch.full_like(weights, float("nan")),
        "main_e2_ce": torch.full_like(weights, float("nan")),
        "best_semantic_ce": torch.full_like(weights, float("nan")),
        "recovered_z1_ce": torch.full_like(weights, float("nan")),
        "recovered_prev_ce": torch.full_like(weights, float("nan")),
        "recovered_both_ce": torch.full_like(weights, float("nan")),
        "best_semantic_control_idx": torch.full_like(weights, -1, dtype=torch.long),
        "semantic_gap_ci_low": torch.full_like(weights, float("nan")),
        "activation_rate": torch.full_like(weights, float("nan")),
        "variable": torch.ones_like(weights, dtype=torch.bool),
        "eligible": selected.clone(),
        "strict_eligible": selected.clone(),
        "selected": selected.clone(),
        "candidate": selected.clone(),
        "confirmed": selected.clone(),
        "claim_selected": selected.clone(),
    }
    return weights, summary, tensors


@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    train_next_mse: float
    train_prev_mse: float
    train_orth: float
    train_e2_prev_cov: float
    train_e2_prev_adv_mse: float
    train_e2_prev_adv_penalty: float
    train_var_floor: float
    train_pred_loss: float
    val_next_mse: float
    val_prev_mse: float
    val_pred_loss: float
    val_pred_aux: float
    val_orth: float
    val_e2_prev_cov: float
    val_e2_prev_adv_mse: float
    val_e2_prev_adv_penalty: float
    val_z1_std_mean: float
    val_z2_std_mean: float
    val_z1_z2_cos_abs_mean: float
    gate_open: bool
    pred_weight: float
    e2_prev_adv_weight: float
    dynamic_target_count: int
    dynamic_target_weight_sum: float
    dynamic_target_score_mean_selected: float
    dynamic_target_e2_gain_mean_selected: float
    dynamic_target_e1_gain_mean_selected: float
    dynamic_target_semantic_gap_mean_selected: float
    dynamic_target_semantic_gain_mean_selected: float
    dynamic_target_semantic_gap_ci_low_mean_selected: float
    dynamic_target_residual_fraction_mean_selected: float
    dynamic_target_leakage_explained_fraction_mean_selected: float
    constrained_score: float


def evaluate(
    model: Decoupler,
    loader: DataLoader,
    device: torch.device,
    target_mode: str,
    binary_threshold: float,
    use_e2_prev_adversary: bool = False,
    e2_prev_adv_margin: float = 0.9,
    binary_eval_meta: Optional[Dict[str, torch.Tensor]] = None,
    semantic_residual_state: Optional[Dict[str, List[torch.Tensor]]] = None,
    target_feature_weights: Optional[torch.Tensor] = None,
    target_projection: Optional[torch.Tensor] = None,
    regularizer_epoch: int = 0,
    main_prediction_target: str = "raw",
) -> Dict[str, float]:
    model.eval()
    totals = {
        "next": 0.0,
        "prev": 0.0,
        "pred": 0.0,
        "semantic_pred": 0.0,
        "meta_pred": 0.0,
        "orth": 0.0,
        "e2_prev_cov": 0.0,
        "e2_prev_adv_mse": 0.0,
        "e2_prev_adv_penalty": 0.0,
    }
    z1_chunks = []
    z2_chunks = []
    pred_chunks = []
    target_chunks = []
    regularizer_totals: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            x_prev, x_i, x_i_eval, x_next = unpack_batch(batch)
            x_prev = move_to_device(x_prev, device)
            x_i = move_to_device(x_i, device)
            x_i_eval = move_to_device(x_i_eval, device)
            x_next = move_to_device(x_next, device)
            out = model(x_next)
            if has_semantic_residual_state(semantic_residual_state):
                z2_signal = apply_semantic_residual_state(out["z2"], semantic_residual_state)
            else:
                z2_signal = out["z2"]
            x_i_hat, x_i_semantic_hat, x_i_meta_hat = main_prediction_components(
                model,
                out["z1"],
                z2_signal,
                main_prediction_target,
            )
            bs = x_next.size(0)
            totals["next"] += normalized_mse(out["x_next_hat"], x_next).item() * bs
            totals["prev"] += normalized_mse(out["x_prev_hat"], x_prev).item() * bs
            totals["pred"] += weighted_prediction_loss(x_i_hat, x_i, target_mode, target_feature_weights, target_projection).item() * bs
            totals["semantic_pred"] += weighted_prediction_loss(x_i_semantic_hat, x_i, target_mode, target_feature_weights, target_projection).item() * bs
            totals["meta_pred"] += weighted_prediction_loss(x_i_meta_hat, x_i, target_mode, target_feature_weights, target_projection).item() * bs
            totals["orth"] += orthogonality_loss(out["z1"], z2_signal).item() * bs
            totals["e2_prev_cov"] += cross_cov_loss(z2_signal, x_prev).item() * bs
            extra_regularizer = first_stage_regularizer_metrics(
                out,
                z2_signal,
                x_prev,
                regularizer_epoch,
                False,
            )
            for name, value in extra_regularizer.items():
                if name == "penalty":
                    continue
                regularizer_totals[name] = regularizer_totals.get(name, 0.0) + float(value.detach().item()) * bs
            if use_e2_prev_adversary:
                adv_pred = model.e2_prev_adversary(z2_signal)
                adv_mse = F.mse_loss(adv_pred, x_prev)
                adv_penalty = margin_adversary_penalty(adv_mse, e2_prev_adv_margin)
                totals["e2_prev_adv_mse"] += adv_mse.item() * bs
                totals["e2_prev_adv_penalty"] += adv_penalty.item() * bs
            z1_chunks.append(out["z1"].detach().cpu())
            z2_chunks.append(z2_signal.detach().cpu())
            pred_chunks.append(x_i_hat.detach().cpu())
            target_chunks.append(x_i_eval.detach().cpu())
            n += bs
    stats = {k: v / max(n, 1) for k, v in totals.items()}
    stats.update({name: value / max(n, 1) for name, value in regularizer_totals.items()})
    if not use_e2_prev_adversary:
        stats["e2_prev_adv_mse"] = float("nan")
        stats["e2_prev_adv_penalty"] = float("nan")
    z1 = torch.cat(z1_chunks, dim=0)
    z2 = torch.cat(z2_chunks, dim=0)
    stats["z1_std_mean"] = z1.std(dim=0, unbiased=False).mean().item()
    stats["z2_std_mean"] = z2.std(dim=0, unbiased=False).mean().item()
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    stats["z1_z2_cos_abs_mean"] = (z1n * z2n).sum(dim=-1).abs().mean().item()
    pred_tensor = torch.cat(pred_chunks, dim=0)
    target_tensor = torch.cat(target_chunks, dim=0)
    pred = pred_tensor.flatten()
    target = target_tensor.flatten()
    if not is_binary_eval_mode(target_mode):
        pred_z = (pred - pred.mean()) / pred.std(unbiased=False).clamp_min(1e-6)
        target_z = (target - target.mean()) / target.std(unbiased=False).clamp_min(1e-6)
        stats["pred_aux"] = (pred_z * target_z).mean().item()
    else:
        eval_logits = binary_eval_logits(pred_tensor, target_mode, binary_eval_meta)
        stats["pred_aux"] = -binary_ce_from_logits(eval_logits, target_tensor)
    return stats


def write_history_files(history: List[Dict], output_dir: str) -> None:
    jsonl_path = os.path.join(output_dir, "history.jsonl")
    csv_path = os.path.join(output_dir, "history.csv")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    if history:
        fieldnames: List[str] = []
        for row in history:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)


def append_batch_metrics(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def collect_analysis_tensors(
    model: Decoupler,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    target_mode: str = "continuous",
    binary_eval_meta: Optional[Dict[str, torch.Tensor]] = None,
    semantic_residual_state: Optional[Dict[str, List[torch.Tensor]]] = None,
    main_prediction_target: str = "raw",
) -> Dict[str, torch.Tensor]:
    model.eval()
    chunks = {
        "x_prev": [],
        "x_i": [],
        "x_i_train": [],
        "x_next": [],
        "z1": [],
        "z2": [],
        "z2_raw": [],
        "z2_clean": [],
        "x_prev_hat": [],
        "x_i_hat": [],
        "x_i_hat_train": [],
        "x_i_semantic_hat_train": [],
        "x_i_meta_hat_train": [],
        "x_next_hat": [],
    }
    seen = 0
    with torch.no_grad():
        for batch in loader:
            x_prev, x_i_train, x_i_eval, x_next = unpack_batch(batch)
            remaining = max_samples - seen
            if remaining <= 0:
                break
            x_prev = move_to_device(x_prev[:remaining], device)
            x_i_train = move_to_device(x_i_train[:remaining], device)
            x_i_eval = move_to_device(x_i_eval[:remaining], device)
            x_next = move_to_device(x_next[:remaining], device)
            out = model(x_next)
            if has_semantic_residual_state(semantic_residual_state):
                z2_clean = apply_semantic_residual_state(out["z2"], semantic_residual_state)
            else:
                z2_clean = out["z2"]
            x_i_hat_train, x_i_semantic_hat, x_i_meta_hat = main_prediction_components(
                model,
                out["z1"],
                z2_clean,
                main_prediction_target,
            )
            if is_binary_eval_mode(target_mode):
                x_i_hat_eval = binary_eval_logits(x_i_hat_train, target_mode, binary_eval_meta)
            else:
                x_i_hat_eval = x_i_hat_train
            batch = {
                "x_prev": x_prev,
                "x_i": x_i_eval,
                "x_i_train": x_i_train,
                "x_next": x_next,
                "z1": out["z1"],
                "z2": z2_clean,
                "z2_raw": out["z2"],
                "z2_clean": z2_clean,
                "x_prev_hat": out["x_prev_hat"],
                "x_i_hat": x_i_hat_eval,
                "x_i_hat_train": x_i_hat_train,
                "x_i_semantic_hat_train": x_i_semantic_hat,
                "x_i_meta_hat_train": x_i_meta_hat,
                "x_next_hat": out["x_next_hat"],
            }
            for key, value in batch.items():
                chunks[key].append(value.detach().cpu())
            seen += x_next.size(0)
    return {key: torch.cat(value, dim=0) for key, value in chunks.items() if value}


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=2, center=False)
    return x @ v[:, :2]


def save_binary_activation_comparison(
    tensors: Dict[str, torch.Tensor],
    plot_dir: str,
    pred_threshold: float,
    max_samples: int,
    max_features: int,
    prefix: str = "binary_activation",
) -> None:
    import matplotlib.pyplot as plt

    target = tensors["x_i"].float()
    pred_prob = torch.sigmoid(tensors["x_i_hat"].float())
    pred_binary = (pred_prob >= pred_threshold).float()

    sample_count = min(max_samples, target.size(0))
    feature_count = min(max_features, target.size(1))
    activity = target[:sample_count].mean(dim=0)
    feature_idx = torch.argsort(activity, descending=True)[:feature_count]

    true_view = target[:sample_count, feature_idx]
    pred_view = pred_binary[:sample_count, feature_idx]
    diff_view = pred_view - true_view

    stacked = torch.cat([true_view, pred_view], dim=0).numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(stacked, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    plt.axhline(sample_count - 0.5, color="tab:red", linewidth=1.2)
    plt.xlabel("selected neuron/features, sorted by true activation rate")
    plt.ylabel("samples: true on top, predicted on bottom")
    plt.title(f"Binary activation comparison, prediction threshold={pred_threshold:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_true_vs_pred.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(12, 3.5))
    plt.imshow(diff_view.numpy(), aspect="auto", interpolation="nearest", cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar(label="-1 false negative, 0 match, +1 false positive")
    plt.xlabel("selected neuron/features, sorted by true activation rate")
    plt.ylabel("sample")
    plt.title("Binary activation prediction errors")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_error_map.png"), dpi=180)
    plt.close()

    true_rate = true_view.mean(dim=0)
    pred_rate = pred_view.mean(dim=0)
    plt.figure(figsize=(10, 4))
    x = torch.arange(feature_count).numpy()
    plt.plot(x, true_rate.numpy(), label="true activation rate")
    plt.plot(x, pred_rate.numpy(), label="predicted activation rate")
    plt.xlabel("selected neuron/features, sorted by true activation rate")
    plt.ylabel("activation rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_rate_compare.png"), dpi=180)
    plt.close()


def save_plots(
    history: List[Dict],
    tensors: Dict[str, torch.Tensor],
    output_dir: str,
    target_mode: str,
    binary_pred_threshold: float,
    binary_plot_samples: int,
    binary_plot_features: int,
    extended_diagnostic_plots: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_nmi_style()
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    def line_plot(filename: str, series: List[Tuple[str, str]], ylabel: str, logy: bool = False) -> None:
        available = []
        for key, label in series:
            values = [row.get(key, float("nan")) for row in history]
            if any(math.isfinite(float(value)) for value in values):
                available.append((label, values))
        if not available:
            return
        figure, axis = plt.subplots(figsize=(3.5, 2.8))
        for label, values in available:
            axis.plot(epochs, values, label=label)
        if logy:
            axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.legend(loc="best")
        style_axis(axis)
        figure.tight_layout()
        save_figure(figure, os.path.join(plot_dir, os.path.splitext(filename)[0]))
        plt.close(figure)

    line_plot(
        "reconstruction_losses.png",
        [("train_next_mse", "train next"), ("train_prev_mse", "train prev"), ("val_next_mse", "val next"), ("val_prev_mse", "val prev")],
        "MSE",
        logy=True,
    )
    line_plot("prediction_loss.png", [("train_pred_loss", "train pred"), ("val_pred_loss", "val pred")], "prediction loss", logy=True)
    line_plot("prediction_aux_metric.png", [("val_pred_aux", "corr" if not is_binary_eval_mode(target_mode) else "negative CE")], "validation auxiliary metric")
    line_plot("decoupling_losses.png", [("train_orth", "train orth"), ("val_orth", "val orth"), ("train_e2_prev_cov", "train e2-prev"), ("val_e2_prev_cov", "val e2-prev")], "regularization value", logy=True)
    line_plot(
        "e2_prev_adversary.png",
        [
            ("train_e2_prev_adv_mse", "train adversary MSE"),
            ("val_e2_prev_adv_mse", "val adversary MSE"),
            ("train_e2_prev_adv_penalty", "train margin penalty"),
            ("val_e2_prev_adv_penalty", "val margin penalty"),
            ("e2_prev_adv_weight", "adversary weight"),
        ],
        "value",
    )
    line_plot("latent_statistics.png", [("val_z1_std_mean", "z1 std mean"), ("val_z2_std_mean", "z2 std mean"), ("val_z1_z2_cos_abs_mean", "|cos(z1,z2)|")], "value")
    line_plot("prediction_weight_and_score.png", [("pred_weight", "pred weight"), ("constrained_score", "constrained score")], "value")
    if history and "dynamic_target_count" in history[0]:
        line_plot(
            "dynamic_target_selection.png",
            [
                ("dynamic_target_count", "selected targets"),
                ("dynamic_target_weight_sum", "target weight sum"),
                ("dynamic_target_e2_gain_mean_selected", "selected E2 gain"),
                ("dynamic_target_semantic_gap_mean_selected", "E2 vs semantic-control gap"),
                ("dynamic_target_semantic_gap_ci_low_mean_selected", "semantic gap CI low"),
                ("dynamic_target_residual_fraction_mean_selected", "residual fraction"),
                ("dynamic_target_leakage_explained_fraction_mean_selected", "leakage explained fraction"),
            ],
            "value",
        )

    if not extended_diagnostic_plots or "z1" not in tensors or tensors["z1"].size(0) < 3:
        return

    z1_2d = pca_2d(tensors["z1"])
    z2_2d = pca_2d(tensors["z2"])
    plt.figure(figsize=(7, 6))
    plt.scatter(z1_2d[:, 0], z1_2d[:, 1], s=10, alpha=0.55, label="z1")
    plt.scatter(z2_2d[:, 0], z2_2d[:, 1], s=10, alpha=0.55, label="z2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latent_pca.png"), dpi=160)
    plt.close()

    z1 = (tensors["z1"] - tensors["z1"].mean(dim=0, keepdim=True)) / tensors["z1"].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    z2 = (tensors["z2"] - tensors["z2"].mean(dim=0, keepdim=True)) / tensors["z2"].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    corr = (z1.T @ z2 / z1.size(0)).abs()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr.numpy(), aspect="auto", cmap="magma")
    plt.colorbar(label="abs correlation")
    plt.xlabel("z2 dimension")
    plt.ylabel("z1 dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "z1_z2_abs_corr.png"), dpi=160)
    plt.close()

    next_err = (tensors["x_next_hat"] - tensors["x_next"]).pow(2).mean(dim=1)
    prev_err = (tensors["x_prev_hat"] - tensors["x_prev"]).pow(2).mean(dim=1)
    pred_err = prediction_error_per_sample(tensors["x_i_hat"], tensors["x_i"], target_mode)
    plt.figure(figsize=(7, 6))
    plt.scatter(next_err.numpy(), prev_err.numpy(), c=pred_err.numpy(), s=14, alpha=0.7, cmap="viridis")
    plt.colorbar(label="prediction error")
    plt.xlabel("x_{i+1} reconstruction MSE")
    plt.ylabel("x_{i-1} reconstruction MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sample_error_tradeoff.png"), dpi=160)
    plt.close()

    flat_target = tensors["x_i"].flatten()
    flat_pred = tensors["x_i_hat"].flatten()
    max_points = min(5000, flat_target.numel())
    idx = torch.randperm(flat_target.numel())[:max_points]
    plt.figure(figsize=(6, 6))
    if not is_binary_eval_mode(target_mode):
        plt.scatter(flat_target[idx].numpy(), flat_pred[idx].numpy(), s=4, alpha=0.25)
        plt.xlabel("target x_i")
        plt.ylabel("predicted x_i")
    else:
        probs = torch.sigmoid(flat_pred[idx])
        plt.scatter(flat_target[idx].numpy(), probs.numpy(), s=4, alpha=0.25)
        plt.xlabel("target active status")
        plt.ylabel("predicted probability")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "prediction_scatter.png"), dpi=160)
    plt.close()

    if is_binary_eval_mode(target_mode) and extended_diagnostic_plots:
        save_binary_activation_comparison(
            tensors=tensors,
            plot_dir=plot_dir,
            pred_threshold=binary_pred_threshold,
            max_samples=binary_plot_samples,
            max_features=binary_plot_features,
        )


def prediction_error_per_sample(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if not is_binary_eval_mode(mode):
        return (pred - target).pow(2).mean(dim=1)
    return F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean(dim=1)


def evaluate_reconstruction_only(model: Decoupler, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals = {"next": 0.0, "prev": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            x_prev, _, _, x_next = unpack_batch(batch)
            x_prev = x_prev.to(device)
            x_next = x_next.to(device)
            out = model(x_next)
            bs = x_next.size(0)
            totals["next"] += normalized_mse(out["x_next_hat"], x_next).item() * bs
            totals["prev"] += normalized_mse(out["x_prev_hat"], x_prev).item() * bs
            n += bs
    return {key: value / max(n, 1) for key, value in totals.items()}


def run_recon_gate_calibration(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    dim: int,
    args: argparse.Namespace,
    device: torch.device,
    prev_layer: int,
    next_layer: int,
) -> Dict:
    calibration_dir = os.path.join(args.output_dir, "gate_calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    batch_size = args.calibration_batch_size if args.calibration_batch_size > 0 else args.batch_size
    latent_dim = args.calibration_latent_dim if args.calibration_latent_dim > 0 else args.latent_dim
    hidden_dim = args.calibration_hidden_dim if args.calibration_hidden_dim > 0 else args.hidden_dim
    dropout = args.dropout if args.calibration_dropout is None else args.calibration_dropout
    lr = args.calibration_lr if args.calibration_lr > 0 else args.lr

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    calibrator = Decoupler(dim, latent_dim, hidden_dim, dropout, args.target_mode).to(device)
    recon_params = [
        param
        for name, param in calibrator.named_parameters()
        if name.startswith("e1.") or name.startswith("e2.") or name.startswith("d_next.") or name.startswith("d_prev.")
    ]
    optimizer = torch.optim.AdamW(recon_params, lr=lr, weight_decay=args.weight_decay)

    history = []
    best_next = math.inf
    best_prev = math.inf
    best_joint = math.inf
    best_joint_epoch = 0
    calibration_progress = (
        _tqdm(
            range(1, args.calibration_epochs + 1),
            desc="Gate calibration",
            unit="epoch",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        if compact_progress_enabled()
        else tqdm(range(1, args.calibration_epochs + 1), desc="Gate calibration", leave=True)
    )
    for epoch in calibration_progress:
        calibrator.train()
        totals = {"loss": 0.0, "next": 0.0, "prev": 0.0}
        seen = 0
        for batch in tqdm(train_loader, desc=f"Calibration epoch {epoch}", leave=False):
            x_prev, _, _, x_next = unpack_batch(batch)
            x_prev = x_prev.to(device)
            x_next = x_next.to(device)
            out = calibrator(x_next)
            loss_next = normalized_mse(out["x_next_hat"], x_next)
            loss_prev = normalized_mse(out["x_prev_hat"], x_prev)
            loss = args.lambda_next * loss_next + args.lambda_prev * loss_prev
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recon_params, 1.0)
            optimizer.step()
            bs = x_next.size(0)
            totals["loss"] += loss.item() * bs
            totals["next"] += loss_next.item() * bs
            totals["prev"] += loss_prev.item() * bs
            seen += bs

        val = evaluate_reconstruction_only(calibrator, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(seen, 1),
            "train_next_mse": totals["next"] / max(seen, 1),
            "train_prev_mse": totals["prev"] / max(seen, 1),
            "val_next_mse": val["next"],
            "val_prev_mse": val["prev"],
            "val_joint_mse": val["next"] + val["prev"],
        }
        history.append(row)
        best_next = min(best_next, val["next"])
        best_prev = min(best_prev, val["prev"])
        if row["val_joint_mse"] < best_joint:
            best_joint = row["val_joint_mse"]
            best_joint_epoch = epoch
            torch.save(calibrator.state_dict(), os.path.join(calibration_dir, "best_reconstruction_only_model.pt"))
        if compact_progress_enabled():
            calibration_progress.set_postfix(
                next=f"{row['val_next_mse']:.4f}",
                prev=f"{row['val_prev_mse']:.4f}",
                best=f"{best_joint:.4f}",
                refresh=False,
            )
        else:
            print(json.dumps({"gate_calibration": row}, ensure_ascii=False))

    write_csv(os.path.join(calibration_dir, "gate_calibration_history.csv"), history)
    calibrated_next = max(args.calibration_min_gate, best_next * args.calibration_gate_factor)
    calibrated_prev = max(args.calibration_min_gate, best_prev * args.calibration_gate_factor)
    summary = {
        "prev_layer": prev_layer,
        "layer_i": args.layer_i,
        "next_layer": next_layer,
        "prev_gap": args.layer_i - prev_layer,
        "next_gap": next_layer - args.layer_i,
        "best_val_next_mse": best_next,
        "best_val_prev_mse": best_prev,
        "best_joint_val_mse": best_joint,
        "best_joint_epoch": best_joint_epoch,
        "calibration_gate_factor": args.calibration_gate_factor,
        "calibrated_recon_gate_next": calibrated_next,
        "calibrated_recon_gate_prev": calibrated_prev,
        "settings": {
            "epochs": args.calibration_epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "lr": lr,
        },
    }
    with open(os.path.join(calibration_dir, "gate_calibration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in history]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row["val_next_mse"] for row in history], label="val next MSE")
        plt.plot(epochs, [row["val_prev_mse"] for row in history], label="val prev MSE")
        plt.axhline(calibrated_next, color="tab:blue", linestyle="--", linewidth=1, label="next gate")
        plt.axhline(calibrated_prev, color="tab:orange", linestyle="--", linewidth=1, label="prev gate")
        plt.xlabel("calibration epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(calibration_dir, "gate_calibration.png"), dpi=180)
        plt.close()
    return summary


def collect_probe_tensors(
    model: Decoupler,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    desc: str = "Collecting probe tensors",
    target_mode: str = "binary",
    binary_eval_meta: Optional[Dict[str, torch.Tensor]] = None,
    semantic_residual_state: Optional[Dict[str, List[torch.Tensor]]] = None,
    main_prediction_target: str = "raw",
) -> Dict[str, torch.Tensor]:
    model.eval()
    chunks = {
        "x_prev": [],
        "x_i": [],
        "x_i_train": [],
        "z1": [],
        "z2": [],
        "z2_raw": [],
        "z2_clean": [],
        "main_logits": [],
        "main_pred_train": [],
        "main_semantic_pred_train": [],
        "main_meta_pred_train": [],
    }
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            x_prev, x_i_train, x_i_eval, x_next = unpack_batch(batch)
            remaining = max_samples - seen if max_samples > 0 else x_next.size(0)
            if remaining <= 0:
                break
            x_prev = x_prev[:remaining].to(device)
            x_i_train = x_i_train[:remaining].to(device)
            x_i_eval = x_i_eval[:remaining].to(device)
            x_next = x_next[:remaining].to(device)
            out = model(x_next)
            if has_semantic_residual_state(semantic_residual_state):
                z2_for_probe = apply_semantic_residual_state(out["z2"], semantic_residual_state)
            else:
                z2_for_probe = out["z2"]
            main_pred_train, semantic_pred_train, meta_pred_train = main_prediction_components(
                model,
                out["z1"],
                z2_for_probe,
                main_prediction_target,
            )
            if is_binary_eval_mode(target_mode):
                main_logits = binary_eval_logits(main_pred_train, target_mode, binary_eval_meta)
            else:
                main_logits = main_pred_train
            batch = {
                "x_prev": x_prev,
                "x_i": x_i_eval,
                "x_i_train": x_i_train,
                "z1": out["z1"],
                "z2": z2_for_probe,
                "z2_raw": out["z2"],
                "z2_clean": z2_for_probe,
                "main_logits": main_logits,
                "main_pred_train": main_pred_train,
                "main_semantic_pred_train": semantic_pred_train,
                "main_meta_pred_train": meta_pred_train,
            }
            for key, value in batch.items():
                chunks[key].append(value.detach().cpu())
            seen += x_next.size(0)
    return {key: torch.cat(value, dim=0) for key, value in chunks.items() if value}


def train_binary_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    threshold: float,
    name: str,
    output_dir: str,
    feature_weights: Optional[torch.Tensor] = None,
) -> Tuple[Dict, torch.Tensor]:
    probe = LinearProbe(train_x.size(1), train_y.size(1)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    history = []
    feature_weights_device = feature_weights.to(device) if feature_weights is not None else None
    for epoch in tqdm(range(1, epochs + 1), desc=f"Probe {name}", leave=False):
        probe.train()
        total = 0.0
        seen = 0
        for x_b, y_b in tqdm(loader, desc=f"{name} epoch {epoch}", leave=False):
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            logits = probe(x_b)
            loss = weighted_prediction_loss(logits, y_b, "binary", feature_weights_device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x_b.size(0)
            seen += x_b.size(0)
        history.append({"epoch": epoch, "loss": total / max(seen, 1)})

    probe.eval()
    val_logits = []
    with torch.no_grad():
        for start in range(0, val_x.size(0), batch_size):
            logits = probe(val_x[start : start + batch_size].to(device))
            val_logits.append(logits.cpu())
    val_logits = torch.cat(val_logits, dim=0)
    prior_logits = binary_prior_logits(train_y, val_y)
    prior_ce = binary_ce_from_logits(prior_logits, val_y, feature_weights)
    ce = binary_ce_from_logits(val_logits, val_y, feature_weights)
    metrics = {
        "name": name,
        "ce_nats_per_label": ce,
        "prior_ce_nats_per_label": prior_ce,
        "info_gain_nats_per_label_vs_prior": prior_ce - ce,
        "info_gain_bits_per_label_vs_prior": (prior_ce - ce) / math.log(2.0),
        "relative_ce_reduction_vs_prior": (prior_ce - ce) / max(prior_ce, 1e-8),
        "final_train_loss": history[-1]["loss"] if history else None,
    }
    os.makedirs(os.path.join(output_dir, "probes"), exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(output_dir, "probes", f"{name}.pt"))
    write_csv(os.path.join(output_dir, "probes", f"{name}_history.csv"), history)
    return metrics, val_logits


def train_binary_mlp_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    name: str,
    output_dir: str,
    feature_weights: Optional[torch.Tensor] = None,
) -> Tuple[Dict, torch.Tensor]:
    probe = MLP(train_x.size(1), train_y.size(1), hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    history = []
    feature_weights_device = feature_weights.to(device) if feature_weights is not None else None
    for epoch in tqdm(range(1, epochs + 1), desc=f"Probe {name}", leave=False):
        probe.train()
        total = 0.0
        seen = 0
        for x_b, y_b in tqdm(loader, desc=f"{name} epoch {epoch}", leave=False):
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            logits = probe(x_b)
            loss = weighted_prediction_loss(logits, y_b, "binary", feature_weights_device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x_b.size(0)
            seen += x_b.size(0)
        history.append({"epoch": epoch, "loss": total / max(seen, 1)})
    probe.eval()
    val_logits = []
    with torch.no_grad():
        for start in range(0, val_x.size(0), batch_size):
            val_logits.append(probe(val_x[start : start + batch_size].to(device)).cpu())
    val_logits = torch.cat(val_logits, dim=0)
    prior_logits = binary_prior_logits(train_y, val_y)
    prior_ce = binary_ce_from_logits(prior_logits, val_y, feature_weights)
    ce = binary_ce_from_logits(val_logits, val_y, feature_weights)
    metrics = {
        "name": name,
        "probe_type": "mlp",
        "ce_nats_per_label": ce,
        "prior_ce_nats_per_label": prior_ce,
        "info_gain_nats_per_label_vs_prior": prior_ce - ce,
        "info_gain_bits_per_label_vs_prior": (prior_ce - ce) / math.log(2.0),
        "relative_ce_reduction_vs_prior": (prior_ce - ce) / max(prior_ce, 1e-8),
        "final_train_loss": history[-1]["loss"] if history else None,
    }
    os.makedirs(os.path.join(output_dir, "probes"), exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(output_dir, "probes", f"{name}.pt"))
    write_csv(os.path.join(output_dir, "probes", f"{name}_history.csv"), history)
    return metrics, val_logits


def center_features(x: torch.Tensor) -> torch.Tensor:
    return x.float() - x.float().mean(dim=0, keepdim=True)


def mean_align_prediction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return pred.float() - pred.float().mean(dim=0, keepdim=True) + target.float().mean(dim=0, keepdim=True)


def affine_calibrate_prediction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    pred_std = pred_centered.std(dim=0, keepdim=True, unbiased=False)
    target_std = target_centered.std(dim=0, keepdim=True, unbiased=False)
    scaled = pred_centered / pred_std.clamp_min(1e-6) * target_std
    scaled = torch.where(pred_std > 1e-6, scaled, torch.zeros_like(scaled))
    return scaled + target.mean(dim=0, keepdim=True)


def flattened_centered_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_c = center_features(pred).flatten()
    target_c = center_features(target).flatten()
    denom = pred_c.std(unbiased=False).clamp_min(1e-8) * target_c.std(unbiased=False).clamp_min(1e-8)
    return ((pred_c - pred_c.mean()) * (target_c - target_c.mean())).mean().div(denom).item()


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float()).pow(2).mean(dim=1)


def per_sample_debiased_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (center_features(pred) - center_features(target)).pow(2).mean(dim=1)


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target).item()
    pred_centered = center_features(pred)
    target_centered = center_features(target)
    mse_debiased = F.mse_loss(pred_centered, target_centered).item()
    pred_mean_aligned = mean_align_prediction(pred, target)
    mse_mean_aligned = F.mse_loss(pred_mean_aligned, target).item()
    pred_affine = affine_calibrate_prediction(pred, target)
    mse_affine = F.mse_loss(pred_affine, target).item()
    sse = (pred - target).pow(2).sum()
    sst = target_centered.pow(2).sum().clamp_min(1e-8)
    sse_debiased = (pred_centered - target_centered).pow(2).sum()
    r2 = (1 - sse / sst).item()
    r2_debiased = (1 - sse_debiased / sst).item()
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_std = pred.std(dim=0, unbiased=False)
    target_std = target.std(dim=0, unbiased=False)
    return {
        "mse": mse,
        "mse_debiased": mse_debiased,
        "mse_mean_aligned": mse_mean_aligned,
        "mse_affine_calibrated": mse_affine,
        "r2": r2,
        "r2_debiased": r2_debiased,
        "centered_corr": flattened_centered_corr(pred, target),
        "mean_abs_feature_bias": (pred_mean - target_mean).abs().mean().item(),
        "mean_feature_bias": (pred_mean - target_mean).mean().item(),
        "pred_mean_abs": pred_mean.abs().mean().item(),
        "target_mean_abs": target_mean.abs().mean().item(),
        "pred_std_mean": pred_std.mean().item(),
        "target_std_mean": target_std.mean().item(),
        "pred_std_to_target_std_ratio": (pred_std.mean() / target_std.mean().clamp_min(1e-8)).item(),
    }


def train_regression_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    name: str,
    output_dir: str,
) -> Tuple[Dict, torch.Tensor]:
    probe = LinearProbe(train_x.size(1), train_y.size(1)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in tqdm(range(1, epochs + 1), desc=f"Probe {name}", leave=False):
        probe.train()
        total = 0.0
        seen = 0
        for x_b, y_b in tqdm(loader, desc=f"{name} epoch {epoch}", leave=False):
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            pred = probe(x_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x_b.size(0)
            seen += x_b.size(0)
        history.append({"epoch": epoch, "loss": total / max(seen, 1)})

    probe.eval()
    val_pred = []
    with torch.no_grad():
        for start in range(0, val_x.size(0), batch_size):
            pred = probe(val_x[start : start + batch_size].to(device))
            val_pred.append(pred.cpu())
    val_pred = torch.cat(val_pred, dim=0)
    metrics = regression_metrics(val_pred, val_y)
    metrics["name"] = name
    metrics["final_train_loss"] = history[-1]["loss"] if history else None
    os.makedirs(os.path.join(output_dir, "probes"), exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(output_dir, "probes", f"{name}.pt"))
    write_csv(os.path.join(output_dir, "probes", f"{name}_history.csv"), history)
    return metrics, val_pred


def train_regression_mlp_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    name: str,
    output_dir: str,
) -> Tuple[Dict, torch.Tensor]:
    probe = MLP(train_x.size(1), train_y.size(1), hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in tqdm(range(1, epochs + 1), desc=f"MLP probe {name}", leave=False):
        probe.train()
        total = 0.0
        seen = 0
        for x_b, y_b in tqdm(loader, desc=f"{name} epoch {epoch}", leave=False):
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            pred = probe(x_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x_b.size(0)
            seen += x_b.size(0)
        history.append({"epoch": epoch, "loss": total / max(seen, 1)})
    val_pred = predict_linear_probe(probe, val_x, device, batch_size)
    metrics = regression_metrics(val_pred, val_y)
    metrics["name"] = name
    metrics["final_train_loss"] = history[-1]["loss"] if history else None
    os.makedirs(os.path.join(output_dir, "probes"), exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(output_dir, "probes", f"{name}.pt"))
    write_csv(os.path.join(output_dir, "probes", f"{name}_history.csv"), history)
    return metrics, val_pred


def train_regression_mlp_probe_with_train_pred(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    name: str,
    output_dir: str,
) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    probe = MLP(train_x.size(1), train_y.size(1), hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in tqdm(range(1, epochs + 1), desc=f"MLP probe {name}", leave=False):
        probe.train()
        total = 0.0
        seen = 0
        for x_b, y_b in tqdm(loader, desc=f"{name} epoch {epoch}", leave=False):
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            pred = probe(x_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x_b.size(0)
            seen += x_b.size(0)
        history.append({"epoch": epoch, "loss": total / max(seen, 1)})
    train_pred = predict_linear_probe(probe, train_x, device, batch_size)
    val_pred = predict_linear_probe(probe, val_x, device, batch_size)
    metrics = regression_metrics(val_pred, val_y)
    metrics["name"] = name
    metrics["final_train_loss"] = history[-1]["loss"] if history else None
    os.makedirs(os.path.join(output_dir, "probes"), exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(output_dir, "probes", f"{name}.pt"))
    write_csv(os.path.join(output_dir, "probes", f"{name}_history.csv"), history)
    return metrics, train_pred, val_pred


def train_regression_mlp_probe_predict_only(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probe = MLP(train_x.size(1), train_y.size(1), hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = tensor_pair_loader(train_x, train_y, batch_size, True, device)
    for epoch in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        probe.train()
        for x_b, y_b in tqdm(loader, desc=f"{desc} epoch {epoch}", leave=False):
            x_b = move_to_device(x_b, device)
            y_b = move_to_device(y_b, device)
            pred = probe(x_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
    train_pred = predict_linear_probe(probe, train_x, device, batch_size)
    val_pred = predict_linear_probe(probe, val_x, device, batch_size)
    return train_pred, val_pred


def predict_linear_probe(probe: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    probe.eval()
    preds = []
    loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=bool(device.type == "cuda" and x.device.type == "cpu"),
    )
    with torch.no_grad():
        for (x_batch,) in loader:
            preds.append(probe(move_to_device(x_batch, device)).cpu())
    return torch.cat(preds, dim=0)


def train_regression_linear_probe_with_train_pred(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    name: str,
    output_dir: str,
) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    probe = LinearProbe(train_x.size(1), train_y.size(1)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in tqdm(range(1, epochs + 1), desc=f"Probe {name}", leave=False):
        probe.train()
        total = 0.0
        seen = 0
        for x_b, y_b in tqdm(loader, desc=f"{name} epoch {epoch}", leave=False):
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            pred = probe(x_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x_b.size(0)
            seen += x_b.size(0)
        history.append({"epoch": epoch, "loss": total / max(seen, 1)})

    train_pred = predict_linear_probe(probe, train_x, device, batch_size)
    val_pred = predict_linear_probe(probe, val_x, device, batch_size)
    metrics = regression_metrics(val_pred, val_y)
    metrics["name"] = name
    metrics["final_train_loss"] = history[-1]["loss"] if history else None
    os.makedirs(os.path.join(output_dir, "probes"), exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(output_dir, "probes", f"{name}.pt"))
    write_csv(os.path.join(output_dir, "probes", f"{name}_history.csv"), history)
    return metrics, train_pred, val_pred


def train_regression_linear_probe_predict_only(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probe = LinearProbe(train_x.size(1), train_y.size(1)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = tensor_pair_loader(train_x, train_y, batch_size, True, device)
    for epoch in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        probe.train()
        for x_b, y_b in tqdm(loader, desc=f"{desc} epoch {epoch}", leave=False):
            x_b = move_to_device(x_b, device)
            y_b = move_to_device(y_b, device)
            pred = probe(x_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
    train_pred = predict_linear_probe(probe, train_x, device, batch_size)
    val_pred = predict_linear_probe(probe, val_x, device, batch_size)
    return train_pred, val_pred


def train_regression_linear_probe_predict_only_with_weight(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probe = LinearProbe(train_x.size(1), train_y.size(1)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = tensor_pair_loader(train_x, train_y, batch_size, True, device)
    for epoch in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        probe.train()
        for x_b, y_b in tqdm(loader, desc=f"{desc} epoch {epoch}", leave=False):
            x_b = move_to_device(x_b, device)
            y_b = move_to_device(y_b, device)
            pred = probe(x_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
    train_pred = predict_linear_probe(probe, train_x, device, batch_size)
    val_pred = predict_linear_probe(probe, val_x, device, batch_size)
    weight = probe.linear.weight.detach().cpu().float()
    return train_pred, val_pred, weight


def row_space_basis(weight: torch.Tensor, max_rank: int, min_sv_ratio: float) -> torch.Tensor:
    if weight.numel() == 0:
        return torch.empty(weight.size(1), 0)
    try:
        _, singular_values, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    except RuntimeError:
        return torch.empty(weight.size(1), 0)
    if singular_values.numel() == 0 or singular_values[0].item() <= 0.0:
        return torch.empty(weight.size(1), 0)
    keep = singular_values >= (singular_values[0] * min_sv_ratio)
    rank = int(keep.sum().item())
    if max_rank > 0:
        rank = min(rank, max_rank)
    if rank <= 0:
        return torch.empty(weight.size(1), 0)
    return vh[:rank].T.contiguous()


def remove_feature_basis(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    basis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if basis.numel() == 0:
        return train_x, val_x
    basis = basis.to(train_x.device, dtype=train_x.dtype)
    mean = train_x.mean(dim=0, keepdim=True)
    train_centered = train_x - mean
    val_centered = val_x - mean.to(val_x.device)
    train_residual = train_centered - (train_centered @ basis) @ basis.T
    val_basis = basis.to(val_x.device, dtype=val_x.dtype)
    val_residual = val_centered - (val_centered @ val_basis) @ val_basis.T
    return train_residual + mean, val_residual + mean.to(val_x.device)


def semantic_residualize_z2_for_dynamic_targets(
    train_tensors: Dict[str, torch.Tensor],
    val_tensors: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict[str, List[torch.Tensor]]]:
    train_clean = train_tensors["z2"].float()
    val_clean = val_tensors["z2"].float()
    original_train_l2 = train_clean.norm(dim=1).mean().item()
    original_val_l2 = val_clean.norm(dim=1).mean().item()
    removed_dims = []
    means: List[torch.Tensor] = []
    bases: List[torch.Tensor] = []

    for round_idx in range(args.dynamic_semantic_residual_rounds):
        _, _, z1_weight = train_regression_linear_probe_predict_only_with_weight(
            train_clean,
            train_tensors["z1"],
            val_clean,
            device,
            args.dynamic_target_probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            f"Dynamic semantic residual E2->Z1 r{round_idx + 1} epoch {epoch}",
        )
        _, _, prev_weight = train_regression_linear_probe_predict_only_with_weight(
            train_clean,
            train_tensors["x_prev"],
            val_clean,
            device,
            args.dynamic_target_probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            f"Dynamic semantic residual E2->prev r{round_idx + 1} epoch {epoch}",
        )
        basis = row_space_basis(
            torch.cat([z1_weight, prev_weight], dim=0),
            args.dynamic_semantic_residual_rank,
            args.dynamic_semantic_residual_min_sv_ratio,
        )
        if basis.numel() == 0:
            break
        means.append(train_clean.mean(dim=0, keepdim=True).cpu())
        bases.append(basis.cpu())
        train_clean, val_clean = remove_feature_basis(train_clean, val_clean, basis)
        removed_dims.append(int(basis.size(1)))

    summary = {
        "semantic_residual_rounds_requested": int(args.dynamic_semantic_residual_rounds),
        "semantic_residual_rounds_applied": int(len(removed_dims)),
        "semantic_residual_removed_dims": int(sum(removed_dims)),
        "semantic_residual_removed_dims_by_round": " ".join(str(x) for x in removed_dims),
        "semantic_residual_train_l2_ratio": float(train_clean.norm(dim=1).mean().item() / max(original_train_l2, 1e-8)),
        "semantic_residual_val_l2_ratio": float(val_clean.norm(dim=1).mean().item() / max(original_val_l2, 1e-8)),
    }
    residual_state = {"means": means, "bases": bases}
    return train_clean, val_clean, summary, residual_state


def bootstrap_mean_ci(values: torch.Tensor, n_bootstrap: int, seed: int, desc: str = "Bootstrap mean CI") -> Dict[str, Optional[float]]:
    values = values.flatten().float()
    if n_bootstrap <= 0 or values.numel() == 0:
        mean = values.mean().item() if values.numel() else float("nan")
        return {"mean": mean, "ci_low": None, "ci_high": None}
    generator = torch.Generator().manual_seed(seed)
    means = []
    n = values.numel()
    for _ in tqdm(range(n_bootstrap), desc=desc, leave=False):
        idx = torch.randint(0, n, (n,), generator=generator)
        means.append(values[idx].mean().item())
    tensor = torch.tensor(means)
    return {
        "mean": values.mean().item(),
        "ci_low": torch.quantile(tensor, 0.025).item(),
        "ci_high": torch.quantile(tensor, 0.975).item(),
    }


def add_relative_improvement(metrics: Dict[str, Dict]) -> None:
    mean_mse = metrics["mean_baseline"]["mse"]
    mean_mse_debiased = metrics["mean_baseline"].get("mse_debiased", mean_mse)
    mean_mse_affine = metrics["mean_baseline"].get("mse_affine_calibrated", mean_mse)
    for name, row in metrics.items():
        row["relative_mse_improvement_vs_mean"] = (mean_mse - row["mse"]) / max(mean_mse, 1e-8)
        row["relative_mse_debiased_improvement_vs_mean"] = (mean_mse_debiased - row.get("mse_debiased", row["mse"])) / max(mean_mse_debiased, 1e-8)
        row["relative_mse_affine_calibrated_improvement_vs_mean"] = (mean_mse_affine - row.get("mse_affine_calibrated", row["mse"])) / max(mean_mse_affine, 1e-8)


def add_gaussian_mi_proxy(metrics: Dict[str, Dict]) -> None:
    for row in metrics.values():
        row["gaussian_mi_proxy_nats_per_dim_from_r2"] = gaussian_mi_proxy_from_r2(row.get("r2", 0.0))
        row["gaussian_mi_proxy_nats_per_dim_from_r2_debiased"] = gaussian_mi_proxy_from_r2(row.get("r2_debiased", 0.0))


def per_sample_binary_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    feature_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    if feature_weights is None:
        return loss.mean(dim=1)
    weights = feature_weights.to(device=loss.device, dtype=loss.dtype).view(1, -1)
    return (loss * weights).sum(dim=1) / weights.sum().clamp_min(1e-8)


def per_feature_binary_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none").mean(dim=0)


def train_binary_linear_probe_logits_only(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    desc: str,
) -> torch.Tensor:
    probe = LinearProbe(train_x.size(1), train_y.size(1)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = tensor_pair_loader(train_x, train_y, batch_size, True, device)
    for epoch in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        probe.train()
        for x_b, y_b in tqdm(loader, desc=f"{desc} epoch {epoch}", leave=False):
            x_b = move_to_device(x_b, device)
            y_b = move_to_device(y_b, device)
            logits = probe(x_b)
            loss = F.binary_cross_entropy_with_logits(logits, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
    return predict_linear_probe(probe, val_x, device, batch_size)


def train_binary_mlp_probe_logits_only(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    desc: str,
) -> torch.Tensor:
    probe = MLP(train_x.size(1), train_y.size(1), hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = tensor_pair_loader(train_x, train_y, batch_size, True, device)
    for epoch in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        probe.train()
        for x_b, y_b in tqdm(loader, desc=f"{desc} epoch {epoch}", leave=False):
            x_b = move_to_device(x_b, device)
            y_b = move_to_device(y_b, device)
            logits = probe(x_b)
            loss = F.binary_cross_entropy_with_logits(logits, y_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
    return predict_linear_probe(probe, val_x, device, batch_size)


def dynamic_target_start_epoch(args: argparse.Namespace) -> int:
    return args.pred_ramp_start_epoch if args.dynamic_target_start_epoch <= 0 else args.dynamic_target_start_epoch


def orthonormalize_columns(x: torch.Tensor, max_cols: int) -> torch.Tensor:
    if x.numel() == 0 or max_cols <= 0:
        return torch.empty(x.size(0), 0, dtype=x.dtype)
    q, _ = torch.linalg.qr(x.float(), mode="reduced")
    return q[:, : min(max_cols, q.size(1))].contiguous()


def top_pca_directions(x: torch.Tensor, k: int) -> torch.Tensor:
    x = center_features(x.float())
    rank = min(int(k), x.size(0) - 1, x.size(1))
    if rank <= 0:
        return torch.empty(x.size(1), 0)
    try:
        _, _, v = torch.pca_lowrank(x, q=rank, center=False)
        return v[:, :rank].contiguous()
    except RuntimeError:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        return vh[:rank].T.contiguous()


def random_directions(dim: int, k: int, seed: int) -> torch.Tensor:
    if k <= 0:
        return torch.empty(dim, 0)
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(dim, k, generator=generator)
    return orthonormalize_columns(raw, k)


def make_direction_candidates(
    train_target: torch.Tensor,
    e2_val_pred: torch.Tensor,
    semantic_val_pred: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
) -> torch.Tensor:
    dim = train_target.size(1)
    k = min(max(1, args.dynamic_direction_candidates), dim)
    source = args.dynamic_direction_source
    if source == "pca":
        return top_pca_directions(train_target, k)
    if source == "e2_pls":
        return top_pca_directions(e2_val_pred, k)
    if source == "residual_pls":
        return top_pca_directions(e2_val_pred - semantic_val_pred, k)
    if source == "random":
        return random_directions(dim, k, args.seed + 9100 + epoch)
    if source == "mixed":
        residual_k = max(1, k // 2)
        pca_k = max(1, (k - residual_k) // 2)
        random_k = max(0, k - residual_k - pca_k)
        raw = torch.cat(
            [
                top_pca_directions(e2_val_pred - semantic_val_pred, residual_k),
                top_pca_directions(train_target, pca_k),
                random_directions(dim, random_k, args.seed + 9200 + epoch),
            ],
            dim=1,
        )
        return orthonormalize_columns(raw, k)
    raise ValueError(f"Unknown dynamic direction source: {source}")


def project_and_standardize(
    train_target: torch.Tensor,
    val_target: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    directions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    train_y = train_target.float() @ directions.float()
    val_y = val_target.float() @ directions.float()
    mean = train_y.mean(dim=0, keepdim=True)
    std = train_y.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_y = (train_y - mean) / std
    val_y = (val_y - mean) / std
    projected = {}
    for name, pred in predictions.items():
        projected[name] = (pred.float() @ directions.float() - mean) / std
    return train_y, val_y, projected, mean.squeeze(0), std.squeeze(0)


def per_direction_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float()).pow(2).mean(dim=0)


def gaussian_info_from_mse(prior_mse: torch.Tensor, pred_mse: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log(prior_mse.float().clamp_min(1e-8) / pred_mse.float().clamp_min(1e-8))


def compute_dynamic_direction_weights(
    model: Decoupler,
    train_tensors: Dict[str, torch.Tensor],
    val_tensors: Dict[str, torch.Tensor],
    train_z2_targets: torch.Tensor,
    val_z2_targets: torch.Tensor,
    residual_summary: Dict,
    residual_state: Dict[str, List[torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
    train_target = train_tensors["x_i_train"].float()
    val_target = val_tensors["x_i_train"].float()
    train_e2_pred, val_e2_pred = train_regression_linear_probe_predict_only(
        train_z2_targets,
        train_target,
        val_z2_targets,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic direction residual-E2 target probe epoch {epoch}",
    )
    train_z1_recovered, val_z1_recovered = train_regression_linear_probe_predict_only(
        train_z2_targets,
        train_tensors["z1"],
        val_z2_targets,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic direction residual-E2->Z1 leakage probe epoch {epoch}",
    )
    train_prev_recovered, val_prev_recovered = train_regression_linear_probe_predict_only(
        train_z2_targets,
        train_tensors["x_prev"],
        val_z2_targets,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic direction residual-E2->prev leakage probe epoch {epoch}",
    )
    _, val_z1_control = train_regression_linear_probe_predict_only(
        train_z1_recovered,
        train_target,
        val_z1_recovered,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic direction recovered-Z1 target control epoch {epoch}",
    )
    _, val_prev_control = train_regression_linear_probe_predict_only(
        train_prev_recovered,
        train_target,
        val_prev_recovered,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic direction recovered-prev target control epoch {epoch}",
    )
    _, val_both_control = train_regression_linear_probe_predict_only(
        torch.cat([train_z1_recovered, train_prev_recovered], dim=1),
        train_target,
        torch.cat([val_z1_recovered, val_prev_recovered], dim=1),
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic direction recovered-Z1-prev target control epoch {epoch}",
    )

    directions = make_direction_candidates(train_target, val_e2_pred, val_both_control, args, epoch)
    if directions.numel() == 0:
        weights = torch.empty(0)
        tensors = {
            "weights": weights,
            "score": weights,
            "semantic_gap_nats": weights,
            "e2_gain_nats": weights,
            "main_e2_gain_nats": weights,
            "semantic_gain_nats": weights,
            "prior_ce": weights,
            "e2_ce": weights,
            "main_e2_ce": weights,
            "best_semantic_ce": weights,
            "recovered_z1_ce": weights,
            "recovered_prev_ce": weights,
            "recovered_both_ce": weights,
            "best_semantic_control_idx": torch.empty(0, dtype=torch.long),
            "semantic_gap_ci_low": weights,
            "activation_rate": weights,
            "variable": torch.empty(0, dtype=torch.bool),
            "eligible": torch.empty(0, dtype=torch.bool),
            "selected": torch.empty(0, dtype=torch.bool),
            "projection_matrix": directions,
            "semantic_residual_state": residual_state,
        }
        summary = {
            "epoch": epoch,
            **residual_summary,
            "target_type": "direction",
            "direction_source": args.dynamic_direction_source,
            "selected_count": 0,
            "eligible_count": 0,
            "strict_eligible_count": 0,
            "relaxed_min_selected_fallback_used": False,
            "variable_count": 0,
            "weight_sum": 0.0,
            "weight_mean": float("nan"),
            "weight_nonzero": 0,
        }
        return weights, tensors, summary

    predictions = {
        "e2": val_e2_pred,
        "main_e2": val_tensors["main_pred_train"],
        "recovered_z1": val_z1_control,
        "recovered_prev": val_prev_control,
        "recovered_both": val_both_control,
    }
    _, val_y, projected, direction_mean, direction_std = project_and_standardize(train_target, val_target, predictions, directions)
    prior_pred = torch.zeros_like(val_y)
    prior_mse = per_direction_mse(prior_pred, val_y)
    e2_mse = per_direction_mse(projected["e2"], val_y)
    main_e2_mse = per_direction_mse(projected["main_e2"], val_y)
    z1_mse = per_direction_mse(projected["recovered_z1"], val_y)
    prev_mse = per_direction_mse(projected["recovered_prev"], val_y)
    both_mse = per_direction_mse(projected["recovered_both"], val_y)
    semantic_stack = torch.stack([z1_mse, prev_mse, both_mse], dim=0)
    best_semantic_mse, best_semantic_control_idx = semantic_stack.min(dim=0)

    e2_gain = gaussian_info_from_mse(prior_mse, e2_mse)
    main_e2_gain = gaussian_info_from_mse(prior_mse, main_e2_mse)
    semantic_gain = gaussian_info_from_mse(prior_mse, best_semantic_mse)
    semantic_gap = gaussian_info_from_mse(best_semantic_mse, e2_mse)
    score = semantic_gap
    direction_variance = val_y.var(dim=0, unbiased=False)
    variable = direction_variance >= args.dynamic_direction_min_variance

    if args.dynamic_target_bootstrap_samples > 0:
        e2_loss = (projected["e2"] - val_y).pow(2)
        control_preds = torch.stack([projected["recovered_z1"], projected["recovered_prev"], projected["recovered_both"]], dim=0)
        gather_idx = best_semantic_control_idx.long().view(1, 1, -1).expand(1, control_preds.size(1), -1)
        best_control_pred = control_preds.gather(0, gather_idx).squeeze(0)
        semantic_loss = (best_control_pred - val_y).pow(2)
        gap_samples = 0.5 * torch.log(semantic_loss.clamp_min(1e-8) / e2_loss.clamp_min(1e-8))
        generator = torch.Generator().manual_seed(args.seed + 3300 + epoch)
        boot_means = []
        for _ in tqdm(range(args.dynamic_target_bootstrap_samples), desc=f"Dynamic direction bootstrap epoch {epoch}", leave=False):
            idx = torch.randint(0, gap_samples.size(0), (gap_samples.size(0),), generator=generator)
            boot_means.append(gap_samples[idx].mean(dim=0))
        gap_ci_low = torch.quantile(torch.stack(boot_means, dim=0), 0.025, dim=0)
    else:
        gap_ci_low = torch.full_like(score, float("nan"))

    eligible = variable & (e2_gain >= args.dynamic_target_e2_gain_min) & (score >= args.dynamic_target_score_min)
    if args.dynamic_target_require_positive_ci:
        eligible = eligible & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
    claim_score_min = (
        args.dynamic_target_score_min
        if args.dynamic_target_claim_score_min is None
        else args.dynamic_target_claim_score_min
    )
    claim_e2_gain_min = (
        args.dynamic_target_e2_gain_min
        if args.dynamic_target_claim_e2_gain_min is None
        else args.dynamic_target_claim_e2_gain_min
    )
    strict_eligible = variable & (e2_gain >= claim_e2_gain_min) & (score >= claim_score_min)
    if args.dynamic_target_require_positive_ci:
        strict_eligible = strict_eligible & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
    strict_eligible_count = int(strict_eligible.sum().item())
    fallback_used = False
    if eligible.sum().item() < args.dynamic_target_min_selected:
        fallback_mask = variable & (e2_gain >= args.dynamic_target_e2_gain_min)
        if args.dynamic_target_require_positive_ci:
            fallback_mask = fallback_mask & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
        if fallback_mask.sum().item() > 0:
            eligible = fallback_mask
            fallback_used = True

    weights = torch.zeros_like(score)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_indices.numel() > 0:
        ranked = eligible_indices[torch.argsort(score[eligible_indices], descending=True)]
        limit = dynamic_selection_limit(
            ranked.numel(),
            args.dynamic_target_top_k,
            args.dynamic_target_min_selected,
            fallback_used,
        )
        selected = ranked[:limit]
        if args.dynamic_target_weight_mode == "topk":
            weights[selected] = 1.0
        elif args.dynamic_target_weight_mode == "soft":
            temp = max(args.dynamic_target_temperature, 1e-6)
            soft = torch.sigmoid((score[selected] - args.dynamic_target_score_min) / temp)
            weights[selected] = soft.clamp_min(args.dynamic_target_min_soft_weight)
        else:
            raise ValueError(args.dynamic_target_weight_mode)

    selected_mask = weights > args.dynamic_target_export_threshold
    selected_count = int(selected_mask.sum().item())
    if selected_count > 0:
        selected_score = score[selected_mask]
        selected_e2_gain = e2_gain[selected_mask]
        selected_semantic_gain = semantic_gain[selected_mask]
        selected_gap_ci_low = gap_ci_low[selected_mask]
        selected_variance = direction_variance[selected_mask]
        residual_fraction = (selected_score / selected_e2_gain.clamp_min(1e-8)).mean().item()
        leakage_fraction = (selected_semantic_gain / selected_e2_gain.clamp_min(1e-8)).mean().item()
    else:
        selected_score = torch.empty(0)
        selected_e2_gain = torch.empty(0)
        selected_semantic_gain = torch.empty(0)
        selected_gap_ci_low = torch.empty(0)
        selected_variance = torch.empty(0)
        residual_fraction = float("nan")
        leakage_fraction = float("nan")

    tensors = {
        "weights": weights.cpu(),
        "score": score.cpu(),
        "semantic_gap_nats": semantic_gap.cpu(),
        "e2_gain_nats": e2_gain.cpu(),
        "main_e2_gain_nats": main_e2_gain.cpu(),
        "semantic_gain_nats": semantic_gain.cpu(),
        "prior_ce": prior_mse.cpu(),
        "e2_ce": e2_mse.cpu(),
        "main_e2_ce": main_e2_mse.cpu(),
        "best_semantic_ce": best_semantic_mse.cpu(),
        "recovered_z1_ce": z1_mse.cpu(),
        "recovered_prev_ce": prev_mse.cpu(),
        "recovered_both_ce": both_mse.cpu(),
        "best_semantic_control_idx": best_semantic_control_idx.cpu(),
        "semantic_gap_ci_low": gap_ci_low.cpu(),
        "activation_rate": direction_variance.cpu(),
        "direction_variance": direction_variance.cpu(),
        "direction_mean": direction_mean.cpu(),
        "direction_std": direction_std.cpu(),
        "variable": variable.cpu(),
        "eligible": eligible.cpu(),
        "strict_eligible": strict_eligible.cpu(),
        "selected": selected_mask.cpu(),
        "projection_matrix": directions.cpu(),
        "semantic_residual_state": residual_state,
    }
    summary = {
        "epoch": epoch,
        **residual_summary,
        "target_type": "direction",
        "direction_source": args.dynamic_direction_source,
        "direction_candidates": int(directions.size(1)),
        "selected_count": selected_count,
        "eligible_count": int(eligible.sum().item()),
        "strict_eligible_count": strict_eligible_count,
        "strict_positive_gap_sum": score[strict_eligible].clamp_min(0.0).sum().item() if strict_eligible.any() else 0.0,
        "relaxed_min_selected_fallback_used": fallback_used,
        "variable_count": int(variable.sum().item()),
        "weight_sum": float(weights.sum().item()),
        "weight_mean": float(weights.mean().item()) if weights.numel() else float("nan"),
        "weight_nonzero": int((weights > 0).sum().item()),
        "score_mean_selected": selected_score.mean().item() if selected_score.numel() else float("nan"),
        "score_median_selected": selected_score.median().item() if selected_score.numel() else float("nan"),
        "semantic_gap_nats_mean_selected": selected_score.mean().item() if selected_score.numel() else float("nan"),
        "semantic_gap_ci_low_mean_selected": selected_gap_ci_low[torch.isfinite(selected_gap_ci_low)].mean().item()
        if torch.isfinite(selected_gap_ci_low).any()
        else float("nan"),
        "e2_gain_nats_mean_selected": selected_e2_gain.mean().item() if selected_e2_gain.numel() else float("nan"),
        "semantic_gain_nats_mean_selected": selected_semantic_gain.mean().item() if selected_semantic_gain.numel() else float("nan"),
        "e1_gain_nats_mean_selected": selected_semantic_gain.mean().item() if selected_semantic_gain.numel() else float("nan"),
        "activation_rate_mean_selected": selected_variance.mean().item() if selected_variance.numel() else float("nan"),
        "dynamic_residual_fraction_mean_selected": residual_fraction,
        "dynamic_leakage_explained_fraction_mean_selected": leakage_fraction,
        "selection_rule": args.dynamic_target_weight_mode,
        "selection_objective": "direction_e2_vs_recovered_semantic_control_gaussian_info_gap",
        "top_k": args.dynamic_target_top_k,
        "score_min": args.dynamic_target_score_min,
        "e2_gain_min": args.dynamic_target_e2_gain_min,
        "claim_score_min": claim_score_min,
        "claim_e2_gain_min": claim_e2_gain_min,
        "bootstrap_samples": args.dynamic_target_bootstrap_samples,
        "require_positive_ci": args.dynamic_target_require_positive_ci,
    }
    return weights.cpu(), tensors, summary


def compute_dynamic_target_weights(
    model: Decoupler,
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
    binary_eval_meta: Dict[str, torch.Tensor],
    old_weights: Optional[torch.Tensor],
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    train_tensors = collect_probe_tensors(
        model,
        train_loader,
        device,
        args.dynamic_target_max_train_samples,
        f"Dynamic targets train epoch {epoch}",
        args.target_mode,
        binary_eval_meta,
        None,
        args.main_prediction_target,
    )
    val_tensors = collect_probe_tensors(
        model,
        val_loader,
        device,
        args.dynamic_target_max_val_samples,
        f"Dynamic targets val epoch {epoch}",
        args.target_mode,
        binary_eval_meta,
        None,
        args.main_prediction_target,
    )

    train_z2_targets, val_z2_targets, residual_summary, residual_state = semantic_residualize_z2_for_dynamic_targets(
        train_tensors,
        val_tensors,
        args,
        device,
        epoch,
    )
    if args.dynamic_target_mode == "direction":
        return compute_dynamic_direction_weights(
            model,
            train_tensors,
            val_tensors,
            train_z2_targets,
            val_z2_targets,
            residual_summary,
            residual_state,
            args,
            device,
            epoch,
        )

    e2_logits = train_binary_linear_probe_logits_only(
        train_z2_targets,
        train_tensors["x_i"],
        val_z2_targets,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic residual-E2 target probe epoch {epoch}",
    )
    train_z1_recovered, val_z1_recovered = train_regression_linear_probe_predict_only(
        train_z2_targets,
        train_tensors["z1"],
        val_z2_targets,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic residual-E2->Z1 leakage probe epoch {epoch}",
    )
    train_prev_recovered, val_prev_recovered = train_regression_linear_probe_predict_only(
        train_z2_targets,
        train_tensors["x_prev"],
        val_z2_targets,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic residual-E2->prev leakage probe epoch {epoch}",
    )
    recovered_z1_logits = train_binary_linear_probe_logits_only(
        train_z1_recovered,
        train_tensors["x_i"],
        val_z1_recovered,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic recovered-Z1 target control epoch {epoch}",
    )
    recovered_prev_logits = train_binary_linear_probe_logits_only(
        train_prev_recovered,
        train_tensors["x_i"],
        val_prev_recovered,
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic recovered-prev target control epoch {epoch}",
    )
    recovered_both_logits = train_binary_linear_probe_logits_only(
        torch.cat([train_z1_recovered, train_prev_recovered], dim=1),
        train_tensors["x_i"],
        torch.cat([val_z1_recovered, val_prev_recovered], dim=1),
        device,
        args.dynamic_target_probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Dynamic recovered-Z1-prev target control epoch {epoch}",
    )

    prior_logits = binary_prior_logits(train_tensors["x_i"], val_tensors["x_i"])
    prior_ce = per_feature_binary_ce(prior_logits, val_tensors["x_i"])
    e2_ce = per_feature_binary_ce(e2_logits, val_tensors["x_i"])
    main_e2_ce = per_feature_binary_ce(val_tensors["main_logits"], val_tensors["x_i"])
    recovered_z1_ce = per_feature_binary_ce(recovered_z1_logits, val_tensors["x_i"])
    recovered_prev_ce = per_feature_binary_ce(recovered_prev_logits, val_tensors["x_i"])
    recovered_both_ce = per_feature_binary_ce(recovered_both_logits, val_tensors["x_i"])
    semantic_stack = torch.stack([recovered_z1_ce, recovered_prev_ce, recovered_both_ce], dim=0)
    best_semantic_ce, best_semantic_control_idx = semantic_stack.min(dim=0)
    e2_gain = prior_ce - e2_ce
    main_e2_gain = prior_ce - main_e2_ce
    semantic_gain = prior_ce - best_semantic_ce
    semantic_gap = best_semantic_ce - e2_ce
    score = semantic_gap

    if args.dynamic_target_bootstrap_samples > 0:
        e2_loss = F.binary_cross_entropy_with_logits(e2_logits.float(), val_tensors["x_i"].float(), reduction="none")
        best_semantic_logits = torch.stack([recovered_z1_logits, recovered_prev_logits, recovered_both_logits], dim=0)
        gather_idx = best_semantic_control_idx.long().view(1, 1, -1).expand(1, best_semantic_logits.size(1), -1)
        gathered_semantic_logits = best_semantic_logits.gather(0, gather_idx).squeeze(0)
        best_semantic_loss = F.binary_cross_entropy_with_logits(gathered_semantic_logits.float(), val_tensors["x_i"].float(), reduction="none")
        gap_samples = best_semantic_loss - e2_loss
        generator = torch.Generator().manual_seed(args.seed + 3100 + epoch)
        boot_means = []
        for _ in tqdm(range(args.dynamic_target_bootstrap_samples), desc=f"Dynamic target bootstrap epoch {epoch}", leave=False):
            idx = torch.randint(0, gap_samples.size(0), (gap_samples.size(0),), generator=generator)
            boot_means.append(gap_samples[idx].mean(dim=0))
        gap_ci_low = torch.quantile(torch.stack(boot_means, dim=0), 0.025, dim=0)
    else:
        gap_ci_low = torch.full_like(score, float("nan"))

    train_activation_rate = train_tensors["x_i"].float().mean(dim=0)
    activation_rate = val_tensors["x_i"].float().mean(dim=0)
    activation_rate_drift = (train_activation_rate - activation_rate).abs()
    pos = val_tensors["x_i"].float().sum(dim=0)
    neg = val_tensors["x_i"].size(0) - pos
    variable = (
        (train_activation_rate >= args.min_neuron_activation_rate)
        & (train_activation_rate <= args.max_neuron_activation_rate)
        & (activation_rate >= args.min_neuron_activation_rate)
        & (activation_rate <= args.max_neuron_activation_rate)
        & (activation_rate_drift <= args.max_neuron_activation_rate_drift)
        & (pos >= args.min_neuron_positive)
        & (neg >= args.min_neuron_negative)
    )
    eligible = variable & (e2_gain >= args.dynamic_target_e2_gain_min) & (score >= args.dynamic_target_score_min)
    if args.dynamic_target_require_positive_ci:
        eligible = eligible & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
    claim_score_min = (
        args.dynamic_target_score_min
        if args.dynamic_target_claim_score_min is None
        else args.dynamic_target_claim_score_min
    )
    claim_e2_gain_min = (
        args.dynamic_target_e2_gain_min
        if args.dynamic_target_claim_e2_gain_min is None
        else args.dynamic_target_claim_e2_gain_min
    )
    strict_eligible = variable & (e2_gain >= claim_e2_gain_min) & (score >= claim_score_min)
    if args.dynamic_target_require_positive_ci:
        strict_eligible = strict_eligible & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
    strict_eligible_count = int(strict_eligible.sum().item())
    fallback_used = False
    if eligible.sum().item() < args.dynamic_target_min_selected:
        fallback_mask = variable & (e2_gain >= args.dynamic_target_e2_gain_min)
        if args.dynamic_target_require_positive_ci:
            fallback_mask = fallback_mask & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
        if fallback_mask.sum().item() > 0:
            eligible = fallback_mask
            fallback_used = True

    weights = torch.zeros_like(score)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_indices.numel() > 0:
        ranked = eligible_indices[torch.argsort(score[eligible_indices], descending=True)]
        limit = dynamic_selection_limit(
            ranked.numel(),
            args.dynamic_target_top_k,
            args.dynamic_target_min_selected,
            fallback_used,
        )
        selected = ranked[:limit]
        if args.dynamic_target_weight_mode == "topk":
            weights[selected] = 1.0
        elif args.dynamic_target_weight_mode == "soft":
            temp = max(args.dynamic_target_temperature, 1e-6)
            soft = torch.sigmoid((score[selected] - args.dynamic_target_score_min) / temp)
            weights[selected] = soft.clamp_min(args.dynamic_target_min_soft_weight)
        else:
            raise ValueError(args.dynamic_target_weight_mode)

    if old_weights is not None and args.dynamic_target_ema > 0.0:
        weights = args.dynamic_target_ema * old_weights.float() + (1.0 - args.dynamic_target_ema) * weights
        weights = torch.where(variable, weights, torch.zeros_like(weights))

    selected_mask = weights > args.dynamic_target_export_threshold
    selected_count = int(selected_mask.sum().item())
    if selected_count > 0:
        selected_score = score[selected_mask]
        selected_e2_gain = e2_gain[selected_mask]
        selected_semantic_gain = semantic_gain[selected_mask]
        selected_gap_ci_low = gap_ci_low[selected_mask]
        selected_activation = activation_rate[selected_mask]
    else:
        selected_score = torch.empty(0)
        selected_e2_gain = torch.empty(0)
        selected_semantic_gain = torch.empty(0)
        selected_gap_ci_low = torch.empty(0)
        selected_activation = torch.empty(0)

    tensors = {
        "weights": weights.cpu(),
        "score": score.cpu(),
        "semantic_gap_nats": semantic_gap.cpu(),
        "e2_gain_nats": e2_gain.cpu(),
        "main_e2_gain_nats": main_e2_gain.cpu(),
        "semantic_gain_nats": semantic_gain.cpu(),
        "prior_ce": prior_ce.cpu(),
        "e2_ce": e2_ce.cpu(),
        "main_e2_ce": main_e2_ce.cpu(),
        "best_semantic_ce": best_semantic_ce.cpu(),
        "recovered_z1_ce": recovered_z1_ce.cpu(),
        "recovered_prev_ce": recovered_prev_ce.cpu(),
        "recovered_both_ce": recovered_both_ce.cpu(),
        "best_semantic_control_idx": best_semantic_control_idx.cpu(),
        "semantic_gap_ci_low": gap_ci_low.cpu(),
        "activation_rate": activation_rate.cpu(),
        "train_activation_rate": train_activation_rate.cpu(),
        "activation_rate_drift": activation_rate_drift.cpu(),
        "variable": variable.cpu(),
        "eligible": eligible.cpu(),
        "strict_eligible": strict_eligible.cpu(),
        "selected": selected_mask.cpu(),
        "semantic_residual_state": residual_state,
    }
    summary = {
        "epoch": epoch,
        **residual_summary,
        "selected_count": selected_count,
        "eligible_count": int(eligible.sum().item()),
        "strict_eligible_count": strict_eligible_count,
        "strict_positive_gap_sum": score[strict_eligible].clamp_min(0.0).sum().item() if strict_eligible.any() else 0.0,
        "relaxed_min_selected_fallback_used": fallback_used,
        "variable_count": int(variable.sum().item()),
        "weight_sum": float(weights.sum().item()),
        "weight_mean": float(weights.mean().item()),
        "weight_nonzero": int((weights > 0).sum().item()),
        "score_mean_selected": selected_score.mean().item() if selected_score.numel() else float("nan"),
        "score_median_selected": selected_score.median().item() if selected_score.numel() else float("nan"),
        "semantic_gap_nats_mean_selected": selected_score.mean().item() if selected_score.numel() else float("nan"),
        "semantic_gap_ci_low_mean_selected": selected_gap_ci_low[torch.isfinite(selected_gap_ci_low)].mean().item()
        if torch.isfinite(selected_gap_ci_low).any()
        else float("nan"),
        "e2_gain_nats_mean_selected": selected_e2_gain.mean().item() if selected_e2_gain.numel() else float("nan"),
        "semantic_gain_nats_mean_selected": selected_semantic_gain.mean().item() if selected_semantic_gain.numel() else float("nan"),
        "e1_gain_nats_mean_selected": selected_semantic_gain.mean().item() if selected_semantic_gain.numel() else float("nan"),
        "activation_rate_mean_selected": selected_activation.mean().item() if selected_activation.numel() else float("nan"),
        "selection_rule": args.dynamic_target_weight_mode,
        "selection_objective": "e2_vs_recovered_semantic_control_gap",
        "top_k": args.dynamic_target_top_k,
        "score_min": args.dynamic_target_score_min,
        "e2_gain_min": args.dynamic_target_e2_gain_min,
        "claim_score_min": claim_score_min,
        "claim_e2_gain_min": claim_e2_gain_min,
        "bootstrap_samples": args.dynamic_target_bootstrap_samples,
        "require_positive_ci": args.dynamic_target_require_positive_ci,
    }
    return weights.cpu(), tensors, summary


def persistent_target_policy_enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.dynamic_target_candidate_top_k > 0
        or args.dynamic_target_exploration_weight > 0.0
        or args.dynamic_target_score_ema > 0.0
        or args.dynamic_target_confirm_evals > 1
        or args.dynamic_target_drop_patience_evals > 1
    )


def apply_persistent_target_policy(
    raw_weights: torch.Tensor,
    tensors: Dict[str, torch.Tensor],
    raw_summary: Dict[str, Any],
    state: Optional[Dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any], Dict[str, torch.Tensor]]:
    """Keep promising targets trainable while reserving claims for persistent strict targets."""
    score = tensors["score"].float()
    gain = tensors["e2_gain_nats"].float()
    variable = tensors["variable"].bool()
    strict = tensors.get("strict_eligible", tensors["eligible"]).bool()
    potential = score.nan_to_num(nan=-1e9, neginf=-1e9, posinf=1e9)
    potential = potential + args.dynamic_target_candidate_gain_weight * gain.clamp_min(0.0)
    raw_selected = raw_weights.float() > args.dynamic_target_export_threshold

    if state is None or state["confirmed"].numel() != score.numel():
        state = {
            "potential_ema": potential.clone(),
            "positive_streak": torch.zeros_like(strict, dtype=torch.long),
            "negative_streak": torch.zeros_like(strict, dtype=torch.long),
            "confirmed": torch.zeros_like(strict, dtype=torch.bool),
            "eval_count": torch.zeros_like(strict, dtype=torch.long),
            "positive_count": torch.zeros_like(strict, dtype=torch.long),
            "selection_count": torch.zeros_like(strict, dtype=torch.long),
            "gap_mean": torch.zeros_like(score, dtype=torch.float32),
            "gap_m2": torch.zeros_like(score, dtype=torch.float32),
        }
    else:
        alpha = args.dynamic_target_score_ema
        state["potential_ema"] = alpha * state["potential_ema"].float() + (1.0 - alpha) * potential
        for key, value in {
            "eval_count": torch.zeros_like(strict, dtype=torch.long),
            "positive_count": torch.zeros_like(strict, dtype=torch.long),
            "selection_count": torch.zeros_like(strict, dtype=torch.long),
            "gap_mean": torch.zeros_like(score, dtype=torch.float32),
            "gap_m2": torch.zeros_like(score, dtype=torch.float32),
        }.items():
            if key not in state:
                state[key] = value

    eval_count_prev = state["eval_count"].long()
    positive_count_prev = state["positive_count"].long()
    selection_count_prev = state["selection_count"].long()
    gap_mean_prev = state["gap_mean"].float()
    gap_m2_prev = state["gap_m2"].float()
    eval_mask = variable
    eval_count = eval_count_prev + eval_mask.long()
    score_for_stats = score.nan_to_num(nan=0.0, neginf=0.0, posinf=0.0)
    delta = score_for_stats - gap_mean_prev
    gap_mean = torch.where(eval_mask, gap_mean_prev + delta / eval_count.clamp_min(1).float(), gap_mean_prev)
    delta2 = score_for_stats - gap_mean
    gap_m2 = torch.where(eval_mask, gap_m2_prev + delta * delta2, gap_m2_prev)
    positive_count = positive_count_prev + (strict & eval_mask).long()
    selection_count = selection_count_prev + (raw_selected & eval_mask).long()
    eval_denom = eval_count.clamp_min(1).float()
    target_positive_rate = positive_count.float() / eval_denom
    target_selection_rate = selection_count.float() / eval_denom
    gap_var = torch.where(
        eval_count > 1,
        gap_m2 / (eval_count - 1).clamp_min(1).float(),
        torch.zeros_like(gap_m2),
    ).clamp_min(0.0)
    gap_std = gap_var.sqrt()
    target_stability_score = (
        gap_mean.clamp_min(0.0)
        * target_positive_rate
        * (0.5 + 0.5 * target_selection_rate)
        / (1.0 + gap_std)
    )
    stability_weight = float(getattr(args, "dynamic_target_stability_weight", 0.0))
    stability_rank_score = score.nan_to_num(nan=-1e9, neginf=-1e9, posinf=1e9) + stability_weight * target_stability_score

    positive_streak = torch.where(strict, state["positive_streak"] + 1, torch.zeros_like(state["positive_streak"]))
    negative_streak = torch.where(strict, torch.zeros_like(state["negative_streak"]), state["negative_streak"] + 1)
    confirmed = state["confirmed"].clone()
    confirmed |= positive_streak >= args.dynamic_target_confirm_evals
    confirmed &= negative_streak < args.dynamic_target_drop_patience_evals
    confirmed &= variable

    candidate_limit = args.dynamic_target_candidate_top_k
    candidate = torch.zeros_like(variable)
    variable_indices = torch.nonzero(variable, as_tuple=False).flatten()
    if variable_indices.numel() > 0:
        candidate_rank_score = state["potential_ema"] + stability_weight * target_stability_score
        ranked = variable_indices[
            torch.argsort(candidate_rank_score[variable_indices], descending=True)
        ]
        limit = ranked.numel() if candidate_limit <= 0 else min(ranked.numel(), candidate_limit)
        candidate[ranked[:limit]] = True
    candidate |= confirmed

    weights = torch.zeros_like(score)
    exploration = float(args.dynamic_target_exploration_weight)
    if exploration > 0.0:
        weights[candidate] = exploration
    confirmed_indices = torch.nonzero(confirmed, as_tuple=False).flatten()
    full_weight = torch.zeros_like(confirmed)
    if confirmed_indices.numel() > 0:
        ranked = confirmed_indices[torch.argsort(stability_rank_score[confirmed_indices], descending=True)]
        limit = ranked.numel() if args.dynamic_target_top_k <= 0 else min(ranked.numel(), args.dynamic_target_top_k)
        selected = ranked[:limit]
        full_weight[selected] = True
        if args.dynamic_target_weight_mode == "topk":
            weights[selected] = 1.0
        else:
            temperature = max(args.dynamic_target_temperature, 1e-6)
            soft = torch.sigmoid((score[selected] - args.dynamic_target_score_min) / temperature)
            weights[selected] = soft.clamp_min(args.dynamic_target_min_soft_weight)

    selected_mask = weights > args.dynamic_target_export_threshold
    confirmed_strict = confirmed & strict
    claim_selected = full_weight & strict
    prediction_weight_scale = float(weights.max().item()) if weights.numel() and selected_mask.any() else 0.0

    tensors = dict(tensors)
    tensors.update(
        {
            "instantaneous_weights": raw_weights.cpu(),
            "instantaneous_selected": tensors["selected"].cpu(),
            "candidate_potential": potential.cpu(),
            "candidate_potential_ema": state["potential_ema"].cpu(),
            "stability_eval_count": eval_count.cpu(),
            "stability_positive_count": positive_count.cpu(),
            "stability_selection_count": selection_count.cpu(),
            "stability_positive_rate": target_positive_rate.cpu(),
            "stability_selection_rate": target_selection_rate.cpu(),
            "stability_gap_mean": gap_mean.cpu(),
            "stability_gap_std": gap_std.cpu(),
            "target_stability_score": target_stability_score.cpu(),
            "stability_rank_score": stability_rank_score.cpu(),
            "positive_streak": positive_streak.cpu(),
            "negative_streak": negative_streak.cpu(),
            "candidate": candidate.cpu(),
            "confirmed": confirmed.cpu(),
            "confirmed_strict": confirmed_strict.cpu(),
            "full_weight": full_weight.cpu(),
            "claim_selected": claim_selected.cpu(),
            "weights": weights.cpu(),
            "selected": selected_mask.cpu(),
        }
    )

    selected_score = score[selected_mask]
    selected_gain = gain[selected_mask]
    selected_semantic_gain = tensors["semantic_gain_nats"].float()[selected_mask]
    selected_ci = tensors["semantic_gap_ci_low"].float()[selected_mask]
    selected_activation = tensors["activation_rate"].float()[selected_mask]
    selected_stability = target_stability_score[selected_mask]
    selected_positive_rate = target_positive_rate[selected_mask]
    selected_selection_rate = target_selection_rate[selected_mask]
    selected_gap_std = gap_std[selected_mask]
    summary = dict(raw_summary)
    summary.update(
        {
            "instantaneous_selected_count": int((raw_weights > args.dynamic_target_export_threshold).sum().item()),
            "candidate_count": int(candidate.sum().item()),
            "confirmed_count": int(confirmed.sum().item()),
            "confirmed_strict_count": int(confirmed_strict.sum().item()),
            "full_weight_count": int(full_weight.sum().item()),
            "claim_target_count": int(claim_selected.sum().item()),
            "confirmed_strict_positive_gap_sum": score[confirmed_strict].clamp_min(0.0).sum().item() if confirmed_strict.any() else 0.0,
            "selected_count": int(selected_mask.sum().item()),
            "weight_sum": float(weights.sum().item()),
            "weight_mean": float(weights.mean().item()) if weights.numel() else float("nan"),
            "weight_nonzero": int(selected_mask.sum().item()),
            "prediction_weight_scale": prediction_weight_scale,
            "score_mean_selected": selected_score.mean().item() if selected_score.numel() else float("nan"),
            "semantic_gap_nats_mean_selected": selected_score.mean().item() if selected_score.numel() else float("nan"),
            "semantic_gap_ci_low_mean_selected": selected_ci[torch.isfinite(selected_ci)].mean().item()
            if torch.isfinite(selected_ci).any()
            else float("nan"),
            "e2_gain_nats_mean_selected": selected_gain.mean().item() if selected_gain.numel() else float("nan"),
            "semantic_gain_nats_mean_selected": selected_semantic_gain.mean().item() if selected_semantic_gain.numel() else float("nan"),
            "e1_gain_nats_mean_selected": selected_semantic_gain.mean().item() if selected_semantic_gain.numel() else float("nan"),
            "activation_rate_mean_selected": selected_activation.mean().item() if selected_activation.numel() else float("nan"),
            "target_stability_score_mean_selected": selected_stability.mean().item() if selected_stability.numel() else float("nan"),
            "target_positive_rate_mean_selected": selected_positive_rate.mean().item() if selected_positive_rate.numel() else float("nan"),
            "target_selection_rate_mean_selected": selected_selection_rate.mean().item() if selected_selection_rate.numel() else float("nan"),
            "target_gap_std_mean_selected": selected_gap_std.mean().item() if selected_gap_std.numel() else float("nan"),
            "persistent_candidate_policy": True,
            "candidate_top_k": int(args.dynamic_target_candidate_top_k),
            "exploration_weight": exploration,
            "score_ema": float(args.dynamic_target_score_ema),
            "stability_weight": stability_weight,
            "confirm_evals": int(args.dynamic_target_confirm_evals),
            "drop_patience_evals": int(args.dynamic_target_drop_patience_evals),
        }
    )
    next_state = {
        "potential_ema": state["potential_ema"].cpu(),
        "positive_streak": positive_streak.cpu(),
        "negative_streak": negative_streak.cpu(),
        "confirmed": confirmed.cpu(),
        "eval_count": eval_count.cpu(),
        "positive_count": positive_count.cpu(),
        "selection_count": selection_count.cpu(),
        "gap_mean": gap_mean.cpu(),
        "gap_m2": gap_m2.cpu(),
    }
    return weights.cpu(), tensors, summary, next_state


def save_dynamic_target_update(output_dir: str, tensors: Dict[str, torch.Tensor], summary: Dict) -> None:
    dynamic_dir = os.path.join(output_dir, "dynamic_targets")
    os.makedirs(dynamic_dir, exist_ok=True)
    epoch = int(summary["epoch"])
    torch.save({"summary": summary, **tensors}, os.path.join(dynamic_dir, f"target_weights_epoch_{epoch:03d}.pt"))
    torch.save({"summary": summary, **tensors}, os.path.join(dynamic_dir, "target_weights_latest.pt"))
    rows = []
    weights = tensors["weights"]
    score = tensors["score"]
    semantic_gap = tensors.get("semantic_gap_nats", score)
    e2_gain = tensors["e2_gain_nats"]
    main_e2_gain = tensors.get("main_e2_gain_nats", torch.full_like(e2_gain, float("nan")))
    semantic_gain = tensors.get("semantic_gain_nats", tensors.get("e1_gain_nats", torch.full_like(e2_gain, float("nan"))))
    prior_ce = tensors.get("prior_ce", torch.full_like(e2_gain, float("nan")))
    e2_ce = tensors.get("e2_ce", torch.full_like(e2_gain, float("nan")))
    main_e2_ce = tensors.get("main_e2_ce", torch.full_like(e2_gain, float("nan")))
    best_semantic_ce = tensors.get("best_semantic_ce", torch.full_like(e2_gain, float("nan")))
    recovered_z1_ce = tensors.get("recovered_z1_ce", torch.full_like(e2_gain, float("nan")))
    recovered_prev_ce = tensors.get("recovered_prev_ce", torch.full_like(e2_gain, float("nan")))
    recovered_both_ce = tensors.get("recovered_both_ce", torch.full_like(e2_gain, float("nan")))
    best_semantic_control_idx = tensors.get("best_semantic_control_idx", torch.full_like(e2_gain, -1, dtype=torch.long)).long()
    semantic_gap_ci_low = tensors.get("semantic_gap_ci_low", torch.full_like(e2_gain, float("nan")))
    activation_rate = tensors["activation_rate"]
    stability_eval_count = tensors.get("stability_eval_count", torch.zeros_like(e2_gain, dtype=torch.long)).long()
    stability_positive_rate = tensors.get("stability_positive_rate", torch.full_like(e2_gain, float("nan"))).float()
    stability_selection_rate = tensors.get("stability_selection_rate", torch.full_like(e2_gain, float("nan"))).float()
    stability_gap_mean = tensors.get("stability_gap_mean", torch.full_like(e2_gain, float("nan"))).float()
    stability_gap_std = tensors.get("stability_gap_std", torch.full_like(e2_gain, float("nan"))).float()
    target_stability_score = tensors.get("target_stability_score", torch.full_like(e2_gain, float("nan"))).float()
    stability_rank_score = tensors.get("stability_rank_score", torch.full_like(e2_gain, float("nan"))).float()
    selected = tensors["selected"]
    eligible = tensors["eligible"]
    strict_eligible = tensors.get("strict_eligible", eligible)
    candidate = tensors.get("candidate", torch.zeros_like(selected))
    confirmed = tensors.get("confirmed", torch.zeros_like(selected))
    claim_selected = tensors.get("claim_selected", torch.zeros_like(selected))
    variable = tensors["variable"]
    control_names = ["recovered_z1", "recovered_prev", "recovered_z1_prev"]
    target_type = summary.get("target_type", "neuron")
    for idx in range(weights.numel()):
        control_idx = int(best_semantic_control_idx[idx].item())
        rows.append(
            {
                "epoch": epoch,
                "target_type": target_type,
                "target_index": idx,
                "neuron": idx,
                "weight": weights[idx].item(),
                "selected": bool(selected[idx].item()),
                "eligible": bool(eligible[idx].item()),
                "strict_eligible": bool(strict_eligible[idx].item()),
                "candidate": bool(candidate[idx].item()),
                "confirmed": bool(confirmed[idx].item()),
                "claim_selected": bool(claim_selected[idx].item()),
                "variable": bool(variable[idx].item()),
                "score_nats": score[idx].item(),
                "score_bits": score[idx].item() / math.log(2.0),
                "semantic_gap_nats": semantic_gap[idx].item(),
                "semantic_gap_bits": semantic_gap[idx].item() / math.log(2.0),
                "semantic_gap_ci_low_nats": semantic_gap_ci_low[idx].item(),
                "semantic_gap_ci_low_bits": semantic_gap_ci_low[idx].item() / math.log(2.0),
                "e2_gain_nats": e2_gain[idx].item(),
                "e2_gain_bits": e2_gain[idx].item() / math.log(2.0),
                "main_e2_gain_nats": main_e2_gain[idx].item(),
                "main_e2_gain_bits": main_e2_gain[idx].item() / math.log(2.0),
                "semantic_gain_nats": semantic_gain[idx].item(),
                "semantic_gain_bits": semantic_gain[idx].item() / math.log(2.0),
                "e1_gain_nats_legacy_alias": semantic_gain[idx].item(),
                "e1_gain_bits_legacy_alias": semantic_gain[idx].item() / math.log(2.0),
                "prior_ce": prior_ce[idx].item(),
                "e2_ce": e2_ce[idx].item(),
                "main_e2_ce": main_e2_ce[idx].item(),
                "best_semantic_ce": best_semantic_ce[idx].item(),
                "recovered_z1_ce": recovered_z1_ce[idx].item(),
                "recovered_prev_ce": recovered_prev_ce[idx].item(),
                "recovered_both_ce": recovered_both_ce[idx].item(),
                "best_semantic_control_idx": control_idx,
                "best_semantic_control": control_names[control_idx] if 0 <= control_idx < len(control_names) else "unknown",
                "activation_rate": activation_rate[idx].item(),
                "stability_eval_count": int(stability_eval_count[idx].item()),
                "stability_positive_rate": stability_positive_rate[idx].item(),
                "stability_selection_rate": stability_selection_rate[idx].item(),
                "stability_gap_mean": stability_gap_mean[idx].item(),
                "stability_gap_std": stability_gap_std[idx].item(),
                "target_stability_score": target_stability_score[idx].item(),
                "stability_rank_score": stability_rank_score[idx].item(),
                "direction_source": summary.get("direction_source"),
            }
        )
    write_csv(os.path.join(dynamic_dir, f"target_weights_epoch_{epoch:03d}.csv"), rows)
    write_csv(os.path.join(dynamic_dir, "target_weights_latest.csv"), rows)
    summary_path = os.path.join(dynamic_dir, "target_weight_history.jsonl")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


def paired_ce_gap_test(
    stronger_name: str,
    weaker_name: str,
    stronger_logits: torch.Tensor,
    weaker_logits: torch.Tensor,
    target: torch.Tensor,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
    alpha: float,
    desc: str,
    feature_scope: str = "all",
    feature_weights: Optional[torch.Tensor] = None,
) -> Dict:
    stronger_ce = per_sample_binary_ce(stronger_logits, target, feature_weights)
    weaker_ce = per_sample_binary_ce(weaker_logits, target, feature_weights)
    gap_nats = weaker_ce - stronger_ce
    observed = gap_nats.mean().item()
    ci = bootstrap_mean_ci(gap_nats, n_bootstrap, seed, f"Bootstrap CE gap {desc}")
    p_value = None
    if n_permutations > 0 and gap_nats.numel() > 0:
        generator = torch.Generator().manual_seed(seed + 17)
        exceed = 0
        for _ in tqdm(range(n_permutations), desc=f"Sign-flip CE gap {desc}", leave=False):
            signs = torch.where(
                torch.rand(gap_nats.numel(), generator=generator) < 0.5,
                torch.tensor(-1.0),
                torch.tensor(1.0),
            )
            null_value = (gap_nats * signs).mean().item()
            if null_value >= observed:
                exceed += 1
        p_value = (exceed + 1) / (n_permutations + 1)
    ci_low = ci.get("ci_low")
    significant = bool(p_value is not None and p_value <= alpha and ci_low is not None and ci_low > 0.0)
    return {
        "comparison": desc,
        "feature_scope": feature_scope,
        "feature_weight_sum": None if feature_weights is None else float(feature_weights.float().sum().item()),
        "stronger_model": stronger_name,
        "weaker_model": weaker_name,
        "hypothesis": f"{stronger_name} has lower validation CE than {weaker_name}",
        "mean_ce_gap_nats_per_label": observed,
        "mean_ce_gap_bits_per_label": observed / math.log(2.0),
        "bootstrap_ci_low_nats": ci_low,
        "bootstrap_ci_high_nats": ci.get("ci_high"),
        "bootstrap_ci_low_bits": None if ci_low is None else ci_low / math.log(2.0),
        "bootstrap_ci_high_bits": None if ci.get("ci_high") is None else ci.get("ci_high") / math.log(2.0),
        "signflip_one_sided_p": p_value,
        "alpha": alpha,
        "significant_stronger_better": significant,
        "n": int(gap_nats.numel()),
    }


def load_dynamic_feature_weights(output_dir: str, dim: int) -> Optional[torch.Tensor]:
    path = os.path.join(output_dir, "dynamic_targets", "target_weights_latest.pt")
    if not os.path.exists(path):
        return None
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return None
    weights = obj.get("weights") if isinstance(obj, dict) else None
    if weights is None or weights.numel() != dim:
        return None
    weights = weights.float().flatten()
    if weights.sum().item() <= 0.0:
        return None
    return weights


def feature_scope_rows(dynamic_weights: Optional[torch.Tensor]) -> List[Tuple[str, Optional[torch.Tensor]]]:
    scopes: List[Tuple[str, Optional[torch.Tensor]]] = [("all", None)]
    if dynamic_weights is not None and dynamic_weights.sum().item() > 0.0:
        scopes.append(("dynamic_selected", dynamic_weights.float()))
    return scopes


def weighted_per_sample_mean(values: torch.Tensor, feature_weights: Optional[torch.Tensor]) -> torch.Tensor:
    values = values.float()
    if feature_weights is None:
        return values.mean(dim=1)
    weights = feature_weights.to(device=values.device, dtype=values.dtype).view(1, -1)
    return (values * weights).sum(dim=1) / weights.sum().clamp_min(1e-8)


def select_feature_scope(x: torch.Tensor, feature_weights: Optional[torch.Tensor]) -> torch.Tensor:
    if feature_weights is None:
        return x.float()
    selected = feature_weights.float() > 1e-8
    if selected.sum().item() == 0:
        return x.float()
    return x.float()[:, selected]


def per_sample_continuous_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return weighted_per_sample_mean((pred.float() - target.float()).pow(2), feature_weights)


def per_sample_gaussian_nll_from_mse(sample_mse: torch.Tensor, global_mse: float) -> torch.Tensor:
    variance = max(float(global_mse), 1e-8)
    return 0.5 * (math.log(2.0 * math.pi * variance) + sample_mse.float() / variance)


def continuous_information_metrics(
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    prior_pred: torch.Tensor,
    feature_scope: str,
    feature_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    pred = pred.float()
    target = target.float()
    prior_pred = prior_pred.float()
    pred_mse_samples = per_sample_continuous_mse(pred, target, feature_weights)
    prior_mse_samples = per_sample_continuous_mse(prior_pred, target, feature_weights)
    mse = pred_mse_samples.mean().item()
    prior_mse = prior_mse_samples.mean().item()
    mse_gain = prior_mse - mse
    gaussian_nll = per_sample_gaussian_nll_from_mse(pred_mse_samples, mse).mean().item()
    prior_gaussian_nll = per_sample_gaussian_nll_from_mse(prior_mse_samples, prior_mse).mean().item()
    gaussian_gain = prior_gaussian_nll - gaussian_nll
    scoped_pred = select_feature_scope(pred, feature_weights)
    scoped_target = select_feature_scope(target, feature_weights)
    return {
        "name": name,
        "feature_scope": feature_scope,
        "feature_weight_sum": float(feature_weights.sum().item()) if feature_weights is not None else float(target.size(1)),
        "mse": mse,
        "prior_mse": prior_mse,
        "mse_gain_vs_prior": mse_gain,
        "relative_mse_reduction_vs_prior": mse_gain / max(prior_mse, 1e-8),
        "r2_vs_prior": 1.0 - mse / max(prior_mse, 1e-8),
        "gaussian_nll_nats_per_value": gaussian_nll,
        "prior_gaussian_nll_nats_per_value": prior_gaussian_nll,
        "gaussian_info_gain_nats_per_value_vs_prior": gaussian_gain,
        "gaussian_info_gain_bits_per_value_vs_prior": gaussian_gain / math.log(2.0),
        "gaussian_log_mse_ratio_nats_per_value": 0.5 * math.log(max(prior_mse, 1e-8) / max(mse, 1e-8)),
        "centered_corr": flattened_centered_corr(scoped_pred, scoped_target),
    }


def continuous_comparison_row(name: str, stronger: Dict, weaker: Dict) -> Dict[str, float]:
    mse_delta = weaker["mse"] - stronger["mse"]
    nll_delta = weaker["gaussian_nll_nats_per_value"] - stronger["gaussian_nll_nats_per_value"]
    return {
        "comparison": name,
        "feature_scope": stronger.get("feature_scope"),
        "stronger_model": stronger["name"],
        "weaker_model": weaker["name"],
        "delta_mse": mse_delta,
        "relative_mse_reduction": mse_delta / max(weaker["mse"], 1e-8),
        "delta_gaussian_nll_nats_per_value": nll_delta,
        "delta_gaussian_nll_bits_per_value": nll_delta / math.log(2.0),
    }


def paired_continuous_gap_test(
    stronger_name: str,
    weaker_name: str,
    stronger_pred: torch.Tensor,
    weaker_pred: torch.Tensor,
    target: torch.Tensor,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
    alpha: float,
    desc: str,
    feature_scope: str,
    feature_weights: Optional[torch.Tensor] = None,
    loss_kind: str = "mse",
) -> Dict:
    stronger_mse = per_sample_continuous_mse(stronger_pred, target, feature_weights)
    weaker_mse = per_sample_continuous_mse(weaker_pred, target, feature_weights)
    if loss_kind == "mse":
        stronger_loss = stronger_mse
        weaker_loss = weaker_mse
        unit = "mse"
    elif loss_kind == "gaussian_nll":
        stronger_loss = per_sample_gaussian_nll_from_mse(stronger_mse, stronger_mse.mean().item())
        weaker_loss = per_sample_gaussian_nll_from_mse(weaker_mse, weaker_mse.mean().item())
        unit = "nll_nats_per_value"
    else:
        raise ValueError(f"Unknown continuous loss kind: {loss_kind}")
    gap = weaker_loss - stronger_loss
    observed = gap.mean().item()
    ci = bootstrap_mean_ci(gap, n_bootstrap, seed, f"Bootstrap continuous {loss_kind} gap {desc}")
    p_value = None
    if n_permutations > 0 and gap.numel() > 0:
        generator = torch.Generator().manual_seed(seed + 17)
        exceed = 0
        for _ in tqdm(range(n_permutations), desc=f"Sign-flip continuous {loss_kind} gap {desc}", leave=False):
            signs = torch.where(
                torch.rand(gap.numel(), generator=generator) < 0.5,
                torch.tensor(-1.0),
                torch.tensor(1.0),
            )
            null_value = (gap * signs).mean().item()
            if null_value >= observed:
                exceed += 1
        p_value = (exceed + 1) / (n_permutations + 1)
    ci_low = ci.get("ci_low")
    significant = bool(p_value is not None and p_value <= alpha and ci_low is not None and ci_low > 0.0)
    return {
        "comparison": desc,
        "feature_scope": feature_scope,
        "loss_kind": loss_kind,
        "stronger_model": stronger_name,
        "weaker_model": weaker_name,
        "hypothesis": f"{stronger_name} has lower validation {loss_kind} than {weaker_name}",
        f"mean_gap_{unit}": observed,
        f"bootstrap_ci_low_{unit}": ci_low,
        f"bootstrap_ci_high_{unit}": ci.get("ci_high"),
        "mean_gap_bits_per_value": observed / math.log(2.0) if loss_kind == "gaussian_nll" else None,
        "bootstrap_ci_low_bits_per_value": None if loss_kind != "gaussian_nll" or ci_low is None else ci_low / math.log(2.0),
        "bootstrap_ci_high_bits_per_value": None
        if loss_kind != "gaussian_nll" or ci.get("ci_high") is None
        else ci.get("ci_high") / math.log(2.0),
        "signflip_one_sided_p": p_value,
        "alpha": alpha,
        "significant_stronger_better": significant,
        "n": int(gap.numel()),
    }


def _plot_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def save_information_analysis_plots(
    info_dir: str,
    summary: Dict,
    semantic_control_test_rows: List[Dict],
    continuous_control_test_rows: List[Dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(info_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    binary_conclusions = summary.get("binary_leakage_insufficiency_conclusion", {}) or {}
    scope_order = [scope for scope in ["all", "dynamic_selected"] if scope in binary_conclusions]
    if scope_order:
        x = torch.arange(len(scope_order), dtype=torch.float32).numpy()
        width = 0.24
        e2_info = [_plot_float(binary_conclusions[s].get("e2_predictive_info_bits_per_label")) for s in scope_order]
        leak_info = [_plot_float(binary_conclusions[s].get("leakage_predictive_info_bits_per_label")) for s in scope_order]
        residual_info = [_plot_float(binary_conclusions[s].get("residual_meta_info_bits_per_label")) for s in scope_order]

        plt.figure(figsize=(8, 4.8))
        plt.bar(x - width, e2_info, width=width, label="E2 predictive info")
        plt.bar(x, leak_info, width=width, label="best leakage-control info")
        plt.bar(x + width, residual_info, width=width, label="residual meta info")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(x, scope_order)
        plt.ylabel("bits / binary label")
        plt.title("Binary Predictive Information")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "binary_predictive_information_by_scope.png"), dpi=180)
        plt.close()

        leak_frac = [_plot_float(binary_conclusions[s].get("leakage_explained_fraction")) for s in scope_order]
        residual_frac = [_plot_float(binary_conclusions[s].get("residual_fraction")) for s in scope_order]
        plt.figure(figsize=(8, 4.8))
        plt.bar(x - width / 2, leak_frac, width=width, label="leakage explained fraction")
        plt.bar(x + width / 2, residual_frac, width=width, label="residual fraction")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        plt.xticks(x, scope_order)
        plt.ylabel("fraction of E2 predictive info")
        plt.title("Leakage-Explained vs Residual Fractions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "binary_leakage_residual_fractions.png"), dpi=180)
        plt.close()

        gaps, low_err, high_err, pvals, labels = [], [], [], [], []
        for scope in scope_order:
            test = binary_conclusions[scope].get("best_control_test") or {}
            mean_bits = _plot_float(test.get("mean_ce_gap_bits_per_label"))
            low_bits = _plot_float(test.get("bootstrap_ci_low_bits"))
            high_bits = _plot_float(test.get("bootstrap_ci_high_bits"))
            gaps.append(mean_bits)
            low_err.append(max(0.0, mean_bits - low_bits) if math.isfinite(mean_bits) and math.isfinite(low_bits) else 0.0)
            high_err.append(max(0.0, high_bits - mean_bits) if math.isfinite(mean_bits) and math.isfinite(high_bits) else 0.0)
            pvals.append(_plot_float(test.get("signflip_one_sided_p")))
            labels.append(scope)
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(labels, gaps, color="tab:blue", alpha=0.8)
        plt.errorbar(labels, gaps, yerr=[low_err, high_err], fmt="none", color="black", capsize=4)
        plt.axhline(0, color="black", linewidth=0.8)
        for idx, pval in enumerate(pvals):
            if math.isfinite(pval) and math.isfinite(gaps[idx]):
                plt.text(idx, gaps[idx], f"p={pval:.3g}", ha="center", va="bottom" if gaps[idx] >= 0 else "top", fontsize=9)
        plt.ylabel("CE gap vs best leakage control (bits / label)")
        plt.title("Binary Residual Meta-Information Test")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "binary_best_control_gap_ci.png"), dpi=180)
        plt.close()

    key = summary.get("key_quantities", {}) or {}
    leak_items = [
        ("E2->prev", _plot_float(key.get("e2_to_prev_gaussian_mi_proxy_bits_per_dim_debiased"))),
        ("E2->Z1", _plot_float(key.get("e2_to_z1_gaussian_mi_proxy_bits_per_dim_debiased"))),
        ("avg", _plot_float(key.get("semantic_leakage_mi_proxy_bits_per_dim"))),
    ]
    if any(math.isfinite(v) for _, v in leak_items):
        plt.figure(figsize=(6.5, 4.5))
        plt.bar([k for k, _ in leak_items], [v for _, v in leak_items], color=["tab:orange", "tab:green", "tab:red"])
        plt.ylabel("Gaussian linear MI proxy (bits / dim)")
        plt.title("Semantic Leakage Information Proxy")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "semantic_leakage_mi_proxy_bits.png"), dpi=180)
        plt.close()

    best_rows = [row for row in semantic_control_test_rows if row.get("comparison") == "e2_to_i_linear_better_than_best_semantic_control"]
    if best_rows:
        labels = [str(row.get("feature_scope", "all")) for row in best_rows]
        gaps = [_plot_float(row.get("mean_ce_gap_bits_per_label")) for row in best_rows]
        ci_low = [_plot_float(row.get("bootstrap_ci_low_bits")) for row in best_rows]
        ci_high = [_plot_float(row.get("bootstrap_ci_high_bits")) for row in best_rows]
        pvals = [_plot_float(row.get("signflip_one_sided_p")) for row in best_rows]
        low_err = [max(0.0, m - l) if math.isfinite(m) and math.isfinite(l) else 0.0 for m, l in zip(gaps, ci_low)]
        high_err = [max(0.0, h - m) if math.isfinite(m) and math.isfinite(h) else 0.0 for m, h in zip(gaps, ci_high)]
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(labels, gaps, color="tab:purple", alpha=0.8)
        plt.errorbar(labels, gaps, yerr=[low_err, high_err], fmt="none", color="black", capsize=4)
        plt.axhline(0, color="black", linewidth=0.8)
        for idx, pval in enumerate(pvals):
            if math.isfinite(pval) and math.isfinite(gaps[idx]):
                plt.text(idx, gaps[idx], f"p={pval:.3g}", ha="center", va="bottom" if gaps[idx] >= 0 else "top", fontsize=9)
        plt.ylabel("best-control CE gap (bits / label)")
        plt.title("Best Semantic-Control Tests by Scope")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "binary_best_control_tests_by_scope.png"), dpi=180)
        plt.close()

    continuous_best_rows = [
        row
        for row in continuous_control_test_rows
        if row.get("comparison") == "continuous_e2_to_i_better_than_best_semantic_control"
        and row.get("loss_kind") == "gaussian_nll"
    ]
    if continuous_best_rows:
        labels = [str(row.get("feature_scope", "all")) for row in continuous_best_rows]
        gaps = [_plot_float(row.get("mean_gap_bits_per_value")) for row in continuous_best_rows]
        ci_low = [_plot_float(row.get("bootstrap_ci_low_bits_per_value")) for row in continuous_best_rows]
        ci_high = [_plot_float(row.get("bootstrap_ci_high_bits_per_value")) for row in continuous_best_rows]
        low_err = [max(0.0, m - l) if math.isfinite(m) and math.isfinite(l) else 0.0 for m, l in zip(gaps, ci_low)]
        high_err = [max(0.0, h - m) if math.isfinite(m) and math.isfinite(h) else 0.0 for m, h in zip(gaps, ci_high)]
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(labels, gaps, color="tab:cyan", alpha=0.85)
        plt.errorbar(labels, gaps, yerr=[low_err, high_err], fmt="none", color="black", capsize=4)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Gaussian-NLL gap (bits / value)")
        plt.title("Continuous Residual Information Tests by Scope")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "continuous_best_control_nll_tests_by_scope.png"), dpi=180)
        plt.close()


def standardize_with_train(train_x: torch.Tensor, val_x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.float().mean(dim=0, keepdim=True)
    std = train_x.float().std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return (train_x.float() - mean) / std, (val_x.float() - mean) / std


def leakage_matched_pca_oracle_features(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_y: torch.Tensor,
    budget_r2: float,
    max_pca_dim: int,
    name: str,
    selection: str = "supervised",
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    budget = max(0.0, float(budget_r2))
    train_std, val_std = standardize_with_train(train_x, val_x)
    max_rank = max(0, min(int(max_pca_dim), train_std.size(0) - 1, train_std.size(1)))
    if budget <= 0.0 or max_rank <= 0:
        zeros_train = torch.zeros(train_std.size(0), 1)
        zeros_val = torch.zeros(val_std.size(0), 1)
        return zeros_train, zeros_val, {
            "name": name,
            "budget_r2": budget,
            "selected_pcs": 0,
            "feature_dim_used": 1,
            "actual_train_explained_r2": 0.0,
            "max_pca_dim": int(max_pca_dim),
            "note": "zero leakage budget; using a constant feature so the probe can learn only per-neuron base rates",
        }
    q = max_rank
    _, s, v = torch.pca_lowrank(train_std, q=q, center=False)
    total = train_std.pow(2).sum().clamp_min(1e-8)
    explained = s.pow(2) / total
    all_train_proj = train_std @ v[:, :q]
    all_val_proj = val_std @ v[:, :q]
    if selection == "supervised":
        pc = (all_train_proj - all_train_proj.mean(dim=0, keepdim=True)) / all_train_proj.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        y = (train_y.float() - train_y.float().mean(dim=0, keepdim=True)) / train_y.float().std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        predictive_score = (pc.T @ y / pc.size(0)).pow(2).mean(dim=1)
        order = torch.argsort(predictive_score, descending=True)
    elif selection == "variance":
        predictive_score = torch.full((q,), float("nan"))
        order = torch.arange(q)
    else:
        raise ValueError(f"Unknown semantic oracle selection: {selection}")
    selected = []
    actual = 0.0
    for idx in order.tolist():
        selected.append(idx)
        actual += float(explained[idx].item())
        if actual >= budget:
            break
    if not selected:
        selected = [int(order[0].item())]
        actual = float(explained[selected[0]].item())
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    train_proj = all_train_proj[:, selected_tensor]
    val_proj = all_val_proj[:, selected_tensor]
    return train_proj.contiguous(), val_proj.contiguous(), {
        "name": name,
        "budget_r2": budget,
        "selected_pcs": len(selected),
        "selected_pc_indices": " ".join(str(idx) for idx in selected),
        "feature_dim_used": len(selected),
        "actual_train_explained_r2": actual,
        "max_pca_dim": int(max_pca_dim),
        "over_budget": actual > budget,
        "first_pc_explained_r2": explained[0].item() if explained.numel() else 0.0,
        "selection": selection,
        "mean_selected_predictive_score": None
        if selection != "supervised"
        else predictive_score[selected_tensor].mean().item(),
        "note": "PCA oracle uses true semantic features and enough PCs to meet or slightly exceed the measured E2 semantic leakage budget. Supervised selection chooses Yi-predictive PCs first, making this a favorable semantic-only control.",
    }


def run_information_analysis(
    analysis_dir: str,
    train_tensors: Dict[str, torch.Tensor],
    val_tensors: Dict[str, torch.Tensor],
    main_logits: torch.Tensor,
    e1_logits: torch.Tensor,
    e2_logits: torch.Tensor,
    leakage_metrics: Dict[str, Dict],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict:
    info_dir = os.path.join(analysis_dir, "information_analysis")
    os.makedirs(info_dir, exist_ok=True)
    prior_logits = binary_prior_logits(train_tensors["x_i"], val_tensors["x_i"])
    dynamic_feature_weights = load_dynamic_feature_weights(args.output_dir, val_tensors["x_i"].size(1))
    binary_scopes = feature_scope_rows(dynamic_feature_weights)
    prior_ce_by_scope = {
        scope_name: binary_ce_from_logits(prior_logits, val_tensors["x_i"], weights)
        for scope_name, weights in binary_scopes
    }
    prior_ce = prior_ce_by_scope["all"]

    e1e2_metrics, e1e2_logits = train_binary_linear_probe(
        torch.cat([train_tensors["z1"], train_tensors["z2"]], dim=1),
        train_tensors["x_i"],
        torch.cat([val_tensors["z1"], val_tensors["z2"]], dim=1),
        val_tensors["x_i"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        args.binary_pred_threshold,
        "e1e2_to_i_linear",
        analysis_dir,
    )

    predictor_rows: List[Dict] = []
    predictors_by_scope: Dict[str, Dict[str, Dict]] = {}
    comparison_rows: List[Dict] = []
    for scope_name, weights in binary_scopes:
        scoped_rows = [
            binary_information_metrics("prior_activation_rate", prior_logits, val_tensors["x_i"], prior_ce_by_scope[scope_name], args.binary_pred_threshold, scope_name, weights),
            binary_information_metrics("main_e2_mlp", main_logits, val_tensors["x_i"], prior_ce_by_scope[scope_name], args.binary_pred_threshold, scope_name, weights),
            binary_information_metrics("e1_to_i_linear", e1_logits, val_tensors["x_i"], prior_ce_by_scope[scope_name], args.binary_pred_threshold, scope_name, weights),
            binary_information_metrics("e2_to_i_linear", e2_logits, val_tensors["x_i"], prior_ce_by_scope[scope_name], args.binary_pred_threshold, scope_name, weights),
            binary_information_metrics("e1e2_to_i_linear", e1e2_logits, val_tensors["x_i"], prior_ce_by_scope[scope_name], args.binary_pred_threshold, scope_name, weights),
        ]
        predictor_rows.extend(scoped_rows)
        by_name_scope = {row["name"]: row for row in scoped_rows}
        predictors_by_scope[scope_name] = by_name_scope
        comparison_rows.extend(
            [
                information_comparison_row("incremental_e2_given_e1", by_name_scope["e1e2_to_i_linear"], by_name_scope["e1_to_i_linear"]),
                information_comparison_row("incremental_e1_given_e2", by_name_scope["e1e2_to_i_linear"], by_name_scope["e2_to_i_linear"]),
                information_comparison_row("main_e2_mlp_vs_e1_linear", by_name_scope["main_e2_mlp"], by_name_scope["e1_to_i_linear"]),
                information_comparison_row("e2_linear_vs_e1_linear", by_name_scope["e2_to_i_linear"], by_name_scope["e1_to_i_linear"]),
            ]
        )
    by_name = predictors_by_scope["all"]

    generator = torch.Generator().manual_seed(args.seed + 701)
    random_train_z2 = torch.randn(train_tensors["z2"].shape, generator=generator)
    random_val_z2 = torch.randn(val_tensors["z2"].shape, generator=generator)
    semantic_leakage_metrics = {}
    e2_z1_metrics, train_z1_recovered, val_z1_recovered = train_regression_linear_probe_with_train_pred(
        train_tensors["z2"],
        train_tensors["z1"],
        val_tensors["z2"],
        val_tensors["z1"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        "e2_to_z1",
        analysis_dir,
    )
    semantic_leakage_metrics["e2_to_z1"] = e2_z1_metrics
    random_z1_metrics, _, random_z1_pred = train_regression_linear_probe_with_train_pred(
        random_train_z2,
        train_tensors["z1"],
        random_val_z2,
        val_tensors["z1"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        "random_z_to_z1",
        analysis_dir,
    )
    semantic_leakage_metrics["random_z_to_z1"] = random_z1_metrics
    mean_z1_pred = train_tensors["z1"].mean(dim=0, keepdim=True).expand_as(val_tensors["z1"])
    semantic_leakage_metrics["mean_baseline"] = regression_metrics(mean_z1_pred, val_tensors["z1"])
    semantic_leakage_metrics["mean_baseline"]["name"] = "mean_z1_baseline"
    add_relative_improvement(semantic_leakage_metrics)
    add_gaussian_mi_proxy(semantic_leakage_metrics)
    add_gaussian_mi_proxy(leakage_metrics)

    e2_prev_recovered_metrics, train_prev_recovered, val_prev_recovered = train_regression_linear_probe_with_train_pred(
        train_tensors["z2"],
        train_tensors["x_prev"],
        val_tensors["z2"],
        val_tensors["x_prev"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        "info_e2_to_prev_recovered",
        analysis_dir,
    )
    e2_prev_recovered_metrics["gaussian_mi_proxy_nats_per_dim_from_r2"] = gaussian_mi_proxy_from_r2(e2_prev_recovered_metrics.get("r2", 0.0))
    e2_prev_recovered_metrics["gaussian_mi_proxy_nats_per_dim_from_r2_debiased"] = gaussian_mi_proxy_from_r2(e2_prev_recovered_metrics.get("r2_debiased", 0.0))

    semantic_control_rows: List[Dict] = []
    semantic_control_logits: Dict[str, torch.Tensor] = {}

    def add_semantic_control(name: str, control_type: str, train_x: torch.Tensor, val_x: torch.Tensor) -> Dict:
        metrics, logits = train_binary_linear_probe(
            train_x,
            train_tensors["x_i"],
            val_x,
            val_tensors["x_i"],
            device,
            args.probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            args.binary_pred_threshold,
            name,
            analysis_dir,
        )
        first_row: Optional[Dict] = None
        for scope_name, weights in binary_scopes:
            row = binary_information_metrics(
                name,
                logits,
                val_tensors["x_i"],
                prior_ce_by_scope[scope_name],
                args.binary_pred_threshold,
                scope_name,
                weights,
            )
            row["control_type"] = control_type
            row["final_train_loss"] = metrics.get("final_train_loss")
            semantic_control_rows.append(row)
            if scope_name == "all":
                first_row = row
        semantic_control_logits[name] = logits
        return first_row or semantic_control_rows[-1]

    add_semantic_control("recovered_z1_from_e2_to_i", "actual_recovered_semantics", train_z1_recovered, val_z1_recovered)
    add_semantic_control("recovered_prev_from_e2_to_i", "actual_recovered_semantics", train_prev_recovered, val_prev_recovered)
    add_semantic_control(
        "recovered_z1_prev_from_e2_to_i",
        "actual_recovered_semantics",
        torch.cat([train_z1_recovered, train_prev_recovered], dim=1),
        torch.cat([val_z1_recovered, val_prev_recovered], dim=1),
    )

    z1_budget = max(0.0, float(semantic_leakage_metrics["e2_to_z1"].get("r2_debiased", 0.0))) * args.semantic_oracle_budget_multiplier
    prev_budget = max(0.0, float(leakage_metrics["e2_to_prev"].get("r2_debiased", 0.0))) * args.semantic_oracle_budget_multiplier
    oracle_budget_rows = []
    train_z1_oracle, val_z1_oracle, z1_oracle_info = leakage_matched_pca_oracle_features(
        train_tensors["z1"],
        val_tensors["z1"],
        train_tensors["x_i"],
        z1_budget,
        args.semantic_oracle_max_pca_dim,
        "oracle_z1_matched_to_e2_to_z1",
        args.semantic_oracle_selection,
    )
    oracle_budget_rows.append(z1_oracle_info)
    train_prev_oracle, val_prev_oracle, prev_oracle_info = leakage_matched_pca_oracle_features(
        train_tensors["x_prev"],
        val_tensors["x_prev"],
        train_tensors["x_i"],
        prev_budget,
        args.semantic_oracle_max_pca_dim,
        "oracle_prev_matched_to_e2_to_prev",
        args.semantic_oracle_selection,
    )
    oracle_budget_rows.append(prev_oracle_info)
    oracle_budget_rows.append(
        {
            "name": "oracle_z1_prev_combined",
            "budget_r2": None,
            "selected_pcs": int(z1_oracle_info["selected_pcs"]) + int(prev_oracle_info["selected_pcs"]),
            "feature_dim_used": int(train_z1_oracle.size(1) + train_prev_oracle.size(1)),
            "actual_train_explained_r2": None,
            "z1_actual_train_explained_r2": z1_oracle_info["actual_train_explained_r2"],
            "prev_actual_train_explained_r2": prev_oracle_info["actual_train_explained_r2"],
            "note": "Combined oracle concatenates leakage-budgeted Z1 and Xprev PCA oracle features.",
        }
    )
    add_semantic_control("oracle_z1_budget_to_i", "leakage_matched_semantic_oracle", train_z1_oracle, val_z1_oracle)
    add_semantic_control("oracle_prev_budget_to_i", "leakage_matched_semantic_oracle", train_prev_oracle, val_prev_oracle)
    add_semantic_control(
        "oracle_z1_prev_budget_to_i",
        "leakage_matched_semantic_oracle",
        torch.cat([train_z1_oracle, train_prev_oracle], dim=1),
        torch.cat([val_z1_oracle, val_prev_oracle], dim=1),
    )

    continuous_predictor_rows: List[Dict] = []
    continuous_comparison_rows: List[Dict] = []
    continuous_control_rows: List[Dict] = []
    continuous_control_test_rows: List[Dict] = []
    continuous_predictions: Dict[str, torch.Tensor] = {}
    continuous_conclusions: Dict[str, Dict] = {}
    continuous_available = args.target_mode == "continuous_binary" and "x_i_train" in train_tensors and "x_i_train" in val_tensors
    dynamic_feature_weights = load_dynamic_feature_weights(args.output_dir, val_tensors["x_i_train"].size(1)) if continuous_available else None

    if continuous_available:
        train_y_cont = train_tensors["x_i_train"]
        val_y_cont = val_tensors["x_i_train"]
        prior_cont_pred = train_y_cont.float().mean(dim=0, keepdim=True).expand_as(val_y_cont).contiguous()
        main_cont_pred = val_tensors["main_pred_train"].float()
        e1_cont_metrics, e1_cont_pred = train_regression_linear_probe(
            train_tensors["z1"],
            train_y_cont,
            val_tensors["z1"],
            val_y_cont,
            device,
            args.probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            "e1_to_i_continuous",
            analysis_dir,
        )
        e2_cont_metrics, e2_cont_pred = train_regression_linear_probe(
            train_tensors["z2"],
            train_y_cont,
            val_tensors["z2"],
            val_y_cont,
            device,
            args.probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            "e2_to_i_continuous",
            analysis_dir,
        )
        e1e2_cont_metrics, e1e2_cont_pred = train_regression_linear_probe(
            torch.cat([train_tensors["z1"], train_tensors["z2"]], dim=1),
            train_y_cont,
            torch.cat([val_tensors["z1"], val_tensors["z2"]], dim=1),
            val_y_cont,
            device,
            args.probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            "e1e2_to_i_continuous",
            analysis_dir,
        )
        continuous_predictions.update(
            {
                "prior_mean_continuous": prior_cont_pred,
                "main_e2_mlp_continuous": main_cont_pred,
                "e1_to_i_continuous": e1_cont_pred,
                "e2_to_i_continuous": e2_cont_pred,
                "e1e2_to_i_continuous": e1e2_cont_pred,
            }
        )

        continuous_control_predictions: Dict[str, torch.Tensor] = {}

        def add_continuous_control(name: str, control_type: str, train_x: torch.Tensor, val_x: torch.Tensor) -> None:
            metrics, pred = train_regression_linear_probe(
                train_x,
                train_y_cont,
                val_x,
                val_y_cont,
                device,
                args.probe_epochs,
                args.probe_batch_size,
                args.probe_lr,
                args.probe_weight_decay,
                f"{name}_continuous",
                analysis_dir,
            )
            continuous_control_predictions[name] = pred
            continuous_predictions[f"{name}_continuous"] = pred
            for scope_name, weights in feature_scope_rows(dynamic_feature_weights):
                row = continuous_information_metrics(name, pred, val_y_cont, prior_cont_pred, scope_name, weights)
                row["control_type"] = control_type
                row["final_train_loss"] = metrics.get("final_train_loss")
                continuous_control_rows.append(row)

        add_continuous_control("recovered_z1_from_e2_to_i", "actual_recovered_semantics", train_z1_recovered, val_z1_recovered)
        add_continuous_control("recovered_prev_from_e2_to_i", "actual_recovered_semantics", train_prev_recovered, val_prev_recovered)
        add_continuous_control(
            "recovered_z1_prev_from_e2_to_i",
            "actual_recovered_semantics",
            torch.cat([train_z1_recovered, train_prev_recovered], dim=1),
            torch.cat([val_z1_recovered, val_prev_recovered], dim=1),
        )
        train_z1_oracle_cont, val_z1_oracle_cont, z1_oracle_cont_info = leakage_matched_pca_oracle_features(
            train_tensors["z1"],
            val_tensors["z1"],
            train_y_cont,
            z1_budget,
            args.semantic_oracle_max_pca_dim,
            "continuous_oracle_z1_matched_to_e2_to_z1",
            args.semantic_oracle_selection,
        )
        train_prev_oracle_cont, val_prev_oracle_cont, prev_oracle_cont_info = leakage_matched_pca_oracle_features(
            train_tensors["x_prev"],
            val_tensors["x_prev"],
            train_y_cont,
            prev_budget,
            args.semantic_oracle_max_pca_dim,
            "continuous_oracle_prev_matched_to_e2_to_prev",
            args.semantic_oracle_selection,
        )
        oracle_budget_rows.extend([z1_oracle_cont_info, prev_oracle_cont_info])
        oracle_budget_rows.append(
            {
                "name": "continuous_oracle_z1_prev_combined",
                "budget_r2": None,
                "selected_pcs": int(z1_oracle_cont_info["selected_pcs"]) + int(prev_oracle_cont_info["selected_pcs"]),
                "feature_dim_used": int(train_z1_oracle_cont.size(1) + train_prev_oracle_cont.size(1)),
                "actual_train_explained_r2": None,
                "z1_actual_train_explained_r2": z1_oracle_cont_info["actual_train_explained_r2"],
                "prev_actual_train_explained_r2": prev_oracle_cont_info["actual_train_explained_r2"],
                "note": "Continuous combined oracle concatenates leakage-budgeted Z1 and Xprev PCA oracle features selected against continuous x_i.",
            }
        )
        add_continuous_control("oracle_z1_budget_to_i", "leakage_matched_semantic_oracle", train_z1_oracle_cont, val_z1_oracle_cont)
        add_continuous_control("oracle_prev_budget_to_i", "leakage_matched_semantic_oracle", train_prev_oracle_cont, val_prev_oracle_cont)
        add_continuous_control(
            "oracle_z1_prev_budget_to_i",
            "leakage_matched_semantic_oracle",
            torch.cat([train_z1_oracle_cont, train_prev_oracle_cont], dim=1),
            torch.cat([val_z1_oracle_cont, val_prev_oracle_cont], dim=1),
        )

        for scope_name, weights in feature_scope_rows(dynamic_feature_weights):
            rows = [
                continuous_information_metrics("prior_mean_continuous", prior_cont_pred, val_y_cont, prior_cont_pred, scope_name, weights),
                continuous_information_metrics("main_e2_mlp_continuous", main_cont_pred, val_y_cont, prior_cont_pred, scope_name, weights),
                continuous_information_metrics("e1_to_i_continuous", e1_cont_pred, val_y_cont, prior_cont_pred, scope_name, weights),
                continuous_information_metrics("e2_to_i_continuous", e2_cont_pred, val_y_cont, prior_cont_pred, scope_name, weights),
                continuous_information_metrics("e1e2_to_i_continuous", e1e2_cont_pred, val_y_cont, prior_cont_pred, scope_name, weights),
            ]
            continuous_predictor_rows.extend(rows)
            by_cont_name = {row["name"]: row for row in rows}
            continuous_comparison_rows.extend(
                [
                    continuous_comparison_row(
                        "continuous_incremental_e2_given_e1",
                        by_cont_name["e1e2_to_i_continuous"],
                        by_cont_name["e1_to_i_continuous"],
                    ),
                    continuous_comparison_row(
                        "continuous_incremental_e1_given_e2",
                        by_cont_name["e1e2_to_i_continuous"],
                        by_cont_name["e2_to_i_continuous"],
                    ),
                    continuous_comparison_row(
                        "continuous_main_e2_mlp_vs_e1_linear",
                        by_cont_name["main_e2_mlp_continuous"],
                        by_cont_name["e1_to_i_continuous"],
                    ),
                    continuous_comparison_row(
                        "continuous_e2_linear_vs_e1_linear",
                        by_cont_name["e2_to_i_continuous"],
                        by_cont_name["e1_to_i_continuous"],
                    ),
                ]
            )

            scoped_controls = [row for row in continuous_control_rows if row["feature_scope"] == scope_name]
            for control in scoped_controls:
                name = str(control["name"])
                pred = continuous_control_predictions[name]
                for loss_kind in ["mse", "gaussian_nll"]:
                    continuous_control_test_rows.append(
                        paired_continuous_gap_test(
                            "e2_to_i_continuous",
                            name,
                            e2_cont_pred,
                            pred,
                            val_y_cont,
                            args.bootstrap_samples,
                            args.permutation_tests,
                            args.seed + 1200 + len(continuous_control_test_rows),
                            args.information_alpha,
                            f"continuous_e2_to_i_better_than_{name}",
                            scope_name,
                            weights,
                            loss_kind,
                        )
                    )
                    continuous_control_test_rows.append(
                        paired_continuous_gap_test(
                            "main_e2_mlp_continuous",
                            name,
                            main_cont_pred,
                            pred,
                            val_y_cont,
                            args.bootstrap_samples,
                            args.permutation_tests,
                            args.seed + 1300 + len(continuous_control_test_rows),
                            args.information_alpha,
                            f"continuous_main_e2_mlp_better_than_{name}",
                            scope_name,
                            weights,
                            loss_kind,
                        )
                    )

            best_cont_control = min(scoped_controls, key=lambda row: row["mse"]) if scoped_controls else None
            if best_cont_control is not None:
                best_name = str(best_cont_control["name"])
                best_pred = continuous_control_predictions[best_name]
                best_mse_test = paired_continuous_gap_test(
                    "e2_to_i_continuous",
                    f"best_semantic_control:{best_name}",
                    e2_cont_pred,
                    best_pred,
                    val_y_cont,
                    args.bootstrap_samples,
                    args.permutation_tests,
                    args.seed + 1400 + len(continuous_control_test_rows),
                    args.information_alpha,
                    "continuous_e2_to_i_better_than_best_semantic_control",
                    scope_name,
                    weights,
                    "mse",
                )
                best_nll_test = paired_continuous_gap_test(
                    "e2_to_i_continuous",
                    f"best_semantic_control:{best_name}",
                    e2_cont_pred,
                    best_pred,
                    val_y_cont,
                    args.bootstrap_samples,
                    args.permutation_tests,
                    args.seed + 1500 + len(continuous_control_test_rows),
                    args.information_alpha,
                    "continuous_e2_to_i_better_than_best_semantic_control",
                    scope_name,
                    weights,
                    "gaussian_nll",
                )
                continuous_control_test_rows.extend([best_mse_test, best_nll_test])
                e2_row = by_cont_name["e2_to_i_continuous"]
                continuous_conclusions[scope_name] = {
                    "primary_e2_predictor": "e2_to_i_continuous",
                    "best_semantic_control": best_name,
                    "best_semantic_control_type": best_cont_control.get("control_type"),
                    "best_semantic_control_mse": best_cont_control["mse"],
                    "primary_e2_mse": e2_row["mse"],
                    "primary_e2_r2_vs_prior": e2_row["r2_vs_prior"],
                    "primary_e2_gaussian_info_bits_vs_prior": e2_row["gaussian_info_gain_bits_per_value_vs_prior"],
                    "supported_by_mse_at_alpha": bool(best_mse_test["significant_stronger_better"]),
                    "supported_by_gaussian_nll_at_alpha": bool(best_nll_test["significant_stronger_better"]),
                    "alpha": args.information_alpha,
                    "best_control_test_mse": best_mse_test,
                    "best_control_test_gaussian_nll": best_nll_test,
                }

    semantic_control_test_rows = []
    binary_conclusions: Dict[str, Dict] = {}
    primary_e2_name = "e2_to_i_linear"
    primary_e2_logits = e2_logits
    main_e2_name = "main_e2_mlp"
    for scope_name, weights in binary_scopes:
        scoped_controls = [row for row in semantic_control_rows if row.get("feature_scope", "all") == scope_name]
        for row in scoped_controls:
            name = str(row["name"])
            semantic_control_test_rows.append(
                paired_ce_gap_test(
                    primary_e2_name,
                    name,
                    primary_e2_logits,
                    semantic_control_logits[name],
                    val_tensors["x_i"],
                    args.bootstrap_samples,
                    args.permutation_tests,
                    args.seed + 800 + len(semantic_control_test_rows),
                    args.information_alpha,
                    f"{primary_e2_name}_better_than_{name}",
                    scope_name,
                    weights,
                )
            )
            semantic_control_test_rows.append(
                paired_ce_gap_test(
                    main_e2_name,
                    name,
                    main_logits,
                    semantic_control_logits[name],
                    val_tensors["x_i"],
                    args.bootstrap_samples,
                    args.permutation_tests,
                    args.seed + 900 + len(semantic_control_test_rows),
                    args.information_alpha,
                    f"{main_e2_name}_better_than_{name}",
                    scope_name,
                    weights,
                )
            )

        best_semantic_control_scope = min(scoped_controls, key=lambda row: row["ce_nats_per_label"]) if scoped_controls else None
        best_control_test_scope = None
        if best_semantic_control_scope is not None:
            best_name = str(best_semantic_control_scope["name"])
            best_control_test_scope = paired_ce_gap_test(
                primary_e2_name,
                f"best_semantic_control:{best_name}",
                primary_e2_logits,
                semantic_control_logits[best_name],
                val_tensors["x_i"],
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed + 1000 + len(semantic_control_test_rows),
                args.information_alpha,
                f"{primary_e2_name}_better_than_best_semantic_control",
                scope_name,
                weights,
            )
            semantic_control_test_rows.append(best_control_test_scope)

        e2_row_scope = predictors_by_scope[scope_name][primary_e2_name]
        leakage_info_bits = None if best_semantic_control_scope is None else best_semantic_control_scope["info_gain_bits_per_label_vs_prior"]
        e2_info_bits = e2_row_scope["info_gain_bits_per_label_vs_prior"]
        residual_info_bits = None if best_control_test_scope is None else best_control_test_scope["mean_ce_gap_bits_per_label"]
        binary_conclusions[scope_name] = {
            "primary_e2_predictor": primary_e2_name,
            "feature_scope": scope_name,
            "feature_weight_sum": None if weights is None else float(weights.float().sum().item()),
            "best_semantic_control": None if best_semantic_control_scope is None else best_semantic_control_scope["name"],
            "best_semantic_control_type": None if best_semantic_control_scope is None else best_semantic_control_scope["control_type"],
            "best_semantic_control_ce_nats_per_label": None if best_semantic_control_scope is None else best_semantic_control_scope["ce_nats_per_label"],
            "primary_e2_ce_nats_per_label": e2_row_scope["ce_nats_per_label"],
            "e2_predictive_info_bits_per_label": e2_info_bits,
            "leakage_predictive_info_bits_per_label": leakage_info_bits,
            "residual_meta_info_bits_per_label": residual_info_bits,
            "leakage_explained_fraction": None if leakage_info_bits is None or abs(e2_info_bits) < 1e-8 else leakage_info_bits / e2_info_bits,
            "residual_fraction": None if residual_info_bits is None or abs(e2_info_bits) < 1e-8 else residual_info_bits / e2_info_bits,
            "supported_at_alpha": bool(best_control_test_scope is not None and best_control_test_scope["significant_stronger_better"]),
            "alpha": args.information_alpha,
            "best_control_test": best_control_test_scope,
        }

    best_semantic_control = next(
        (row for row in semantic_control_rows if row.get("feature_scope", "all") == "all" and row["name"] == binary_conclusions.get("all", {}).get("best_semantic_control")),
        None,
    )
    best_control_test = binary_conclusions.get("all", {}).get("best_control_test")

    leakage_proxy_rows = []
    for source, metric_dict in [
        ("prev_leakage", leakage_metrics),
        ("z1_semantic_leakage", semantic_leakage_metrics),
        ("actual_recovered_prev_leakage", {"info_e2_to_prev_recovered": e2_prev_recovered_metrics}),
    ]:
        for name, row in metric_dict.items():
            leakage_proxy_rows.append(
                {
                    "source": source,
                    "name": name,
                    "mse": row.get("mse"),
                    "mse_debiased": row.get("mse_debiased"),
                    "r2": row.get("r2"),
                    "r2_debiased": row.get("r2_debiased"),
                    "gaussian_mi_proxy_nats_per_dim_from_r2": row.get("gaussian_mi_proxy_nats_per_dim_from_r2"),
                    "gaussian_mi_proxy_nats_per_dim_from_r2_debiased": row.get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased"),
                    "relative_mse_debiased_improvement_vs_mean": row.get("relative_mse_debiased_improvement_vs_mean"),
                }
            )

    summary = {
        "interpretation": {
            "prediction_info": "CE(prior activation rate) - CE(probe) is an operational lower-bound style estimate of probe-accessible information about binary i-layer activations.",
            "binary_scopes": "Binary information is reported for all target neurons and, when dynamic target weights are available, for dynamic_selected target neurons.",
            "residual_meta_info": "residual_meta_info = E2 predictive info - best leakage/semantic-control predictive info = CE(best semantic control) - CE(E2).",
            "residual_fraction": "residual_fraction divides residual_meta_info by E2 predictive info; use it only when E2 predictive info is positive and nontrivial.",
            "continuous_prediction_info": "For continuous i-layer activations, MSE/R2 and Gaussian NLL are computed against standardized x_i_train. Gaussian info gain is an operational continuous analogue of CE improvement.",
            "conditional_info": "incremental_e2_given_e1 measures how much CE improves when Z2 is added to Z1.",
            "leakage_proxy": "Gaussian MI proxy from R2 is not a strict upper bound; use it as a comparable leakage estimate against mean/random/shuffled baselines.",
            "semantic_leakage_mi_proxy": "semantic_leakage_mi_proxy is the average of E2->Xprev and E2->Z1 debiased Gaussian linear MI proxies, reported per target dimension.",
            "actual_recovered_semantics_control": "Train E2->Z1 and E2->Xprev, then test whether those recovered semantic variables can explain E2->Yi prediction.",
            "leakage_matched_oracle_control": "Use true Z1/Xprev PCA features with an explained-variance budget matched to measured E2 semantic leakage; this is a favorable semantic-only control.",
            "statistical_test": "Binary paired CE gap and continuous paired MSE/Gaussian-NLL gaps use bootstrap CI over samples and one-sided sign-flip permutation tests; positive gap means the E2 predictor has lower loss than the semantic control.",
        },
        "predictor_information": predictor_rows,
        "conditional_comparisons": comparison_rows,
        "semantic_controls": semantic_control_rows,
        "semantic_control_tests": semantic_control_test_rows,
        "binary_leakage_insufficiency_conclusion": binary_conclusions,
        "continuous_available": continuous_available,
        "continuous_predictor_information": continuous_predictor_rows,
        "continuous_conditional_comparisons": continuous_comparison_rows,
        "continuous_semantic_controls": continuous_control_rows,
        "continuous_semantic_control_tests": continuous_control_test_rows,
        "continuous_leakage_insufficiency_conclusion": continuous_conclusions,
        "semantic_oracle_budgets": oracle_budget_rows,
        "prev_leakage_proxy": leakage_metrics,
        "z1_semantic_leakage_proxy": semantic_leakage_metrics,
        "actual_recovered_prev_leakage_proxy": e2_prev_recovered_metrics,
        "leakage_insufficiency_conclusion": {
            "primary_e2_predictor": primary_e2_name,
            "best_semantic_control": None if best_semantic_control is None else best_semantic_control["name"],
            "best_semantic_control_type": None if best_semantic_control is None else best_semantic_control["control_type"],
            "best_semantic_control_ce_nats_per_label": None if best_semantic_control is None else best_semantic_control["ce_nats_per_label"],
            "primary_e2_ce_nats_per_label": by_name[primary_e2_name]["ce_nats_per_label"],
            "supported_at_alpha": bool(best_control_test is not None and best_control_test["significant_stronger_better"]),
            "alpha": args.information_alpha,
            "best_control_test": best_control_test,
        },
        "key_quantities": {
            "prior_ce_nats_per_label": prior_ce,
            "e2_bits_per_label_vs_prior": by_name["e2_to_i_linear"]["info_gain_bits_per_label_vs_prior"],
            "main_e2_mlp_bits_per_label_vs_prior": by_name["main_e2_mlp"]["info_gain_bits_per_label_vs_prior"],
            "e1e2_bits_per_label_vs_prior": by_name["e1e2_to_i_linear"]["info_gain_bits_per_label_vs_prior"],
            "incremental_e2_given_e1_bits_per_label": comparison_rows[0]["delta_bits_per_label"],
            "e2_to_prev_gaussian_mi_proxy_nats_per_dim_debiased": leakage_metrics["e2_to_prev"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased"),
            "e2_to_z1_gaussian_mi_proxy_nats_per_dim_debiased": semantic_leakage_metrics["e2_to_z1"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased"),
            "e2_to_prev_gaussian_mi_proxy_bits_per_dim_debiased": None
            if leakage_metrics["e2_to_prev"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased") is None
            else leakage_metrics["e2_to_prev"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased") / math.log(2.0),
            "e2_to_z1_gaussian_mi_proxy_bits_per_dim_debiased": None
            if semantic_leakage_metrics["e2_to_z1"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased") is None
            else semantic_leakage_metrics["e2_to_z1"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased") / math.log(2.0),
            "semantic_leakage_mi_proxy_nats_per_dim": (
                float(leakage_metrics["e2_to_prev"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased", 0.0))
                + float(semantic_leakage_metrics["e2_to_z1"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased", 0.0))
            )
            / 2.0,
            "semantic_leakage_mi_proxy_bits_per_dim": (
                float(leakage_metrics["e2_to_prev"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased", 0.0))
                + float(semantic_leakage_metrics["e2_to_z1"].get("gaussian_mi_proxy_nats_per_dim_from_r2_debiased", 0.0))
            )
            / (2.0 * math.log(2.0)),
            "best_semantic_control_bits_per_label_vs_prior": None if best_semantic_control is None else best_semantic_control["info_gain_bits_per_label_vs_prior"],
            "e2_minus_best_semantic_control_bits_per_label": None if best_control_test is None else best_control_test["mean_ce_gap_bits_per_label"],
            "binary_scopes": list(binary_conclusions.keys()),
            "binary_dynamic_selected_available": "dynamic_selected" in binary_conclusions,
            "binary_all_e2_predictive_info_bits_per_label": binary_conclusions.get("all", {}).get("e2_predictive_info_bits_per_label"),
            "binary_all_leakage_predictive_info_bits_per_label": binary_conclusions.get("all", {}).get("leakage_predictive_info_bits_per_label"),
            "binary_all_residual_meta_info_bits_per_label": binary_conclusions.get("all", {}).get("residual_meta_info_bits_per_label"),
            "binary_all_leakage_explained_fraction": binary_conclusions.get("all", {}).get("leakage_explained_fraction"),
            "binary_all_residual_fraction": binary_conclusions.get("all", {}).get("residual_fraction"),
            "binary_all_supported": binary_conclusions.get("all", {}).get("supported_at_alpha"),
            "binary_dynamic_selected_e2_predictive_info_bits_per_label": binary_conclusions.get("dynamic_selected", {}).get("e2_predictive_info_bits_per_label"),
            "binary_dynamic_selected_leakage_predictive_info_bits_per_label": binary_conclusions.get("dynamic_selected", {}).get("leakage_predictive_info_bits_per_label"),
            "binary_dynamic_selected_residual_meta_info_bits_per_label": binary_conclusions.get("dynamic_selected", {}).get("residual_meta_info_bits_per_label"),
            "binary_dynamic_selected_leakage_explained_fraction": binary_conclusions.get("dynamic_selected", {}).get("leakage_explained_fraction"),
            "binary_dynamic_selected_residual_fraction": binary_conclusions.get("dynamic_selected", {}).get("residual_fraction"),
            "binary_dynamic_selected_gap_ci_low_bits": None
            if binary_conclusions.get("dynamic_selected", {}).get("best_control_test") is None
            else binary_conclusions.get("dynamic_selected", {}).get("best_control_test", {}).get("bootstrap_ci_low_bits"),
            "binary_dynamic_selected_gap_p": None
            if binary_conclusions.get("dynamic_selected", {}).get("best_control_test") is None
            else binary_conclusions.get("dynamic_selected", {}).get("best_control_test", {}).get("signflip_one_sided_p"),
            "binary_dynamic_selected_supported": binary_conclusions.get("dynamic_selected", {}).get("supported_at_alpha"),
            "continuous_scopes": list(continuous_conclusions.keys()),
            "continuous_dynamic_selected_available": bool(dynamic_feature_weights is not None),
            "continuous_all_e2_r2_vs_prior": next(
                (row["r2_vs_prior"] for row in continuous_predictor_rows if row["name"] == "e2_to_i_continuous" and row["feature_scope"] == "all"),
                None,
            ),
            "continuous_all_e2_gaussian_bits_vs_prior": next(
                (
                    row["gaussian_info_gain_bits_per_value_vs_prior"]
                    for row in continuous_predictor_rows
                    if row["name"] == "e2_to_i_continuous" and row["feature_scope"] == "all"
                ),
                None,
            ),
            "continuous_dynamic_selected_e2_r2_vs_prior": next(
                (
                    row["r2_vs_prior"]
                    for row in continuous_predictor_rows
                    if row["name"] == "e2_to_i_continuous" and row["feature_scope"] == "dynamic_selected"
                ),
                None,
            ),
            "continuous_dynamic_selected_e2_gaussian_bits_vs_prior": next(
                (
                    row["gaussian_info_gain_bits_per_value_vs_prior"]
                    for row in continuous_predictor_rows
                    if row["name"] == "e2_to_i_continuous" and row["feature_scope"] == "dynamic_selected"
                ),
                None,
            ),
        },
        "artifacts": {
            "predictor_information_csv": "information_analysis/predictor_information.csv",
            "conditional_comparisons_csv": "information_analysis/conditional_comparisons.csv",
            "leakage_proxy_csv": "information_analysis/leakage_proxy.csv",
            "semantic_controls_csv": "information_analysis/semantic_controls_information.csv",
            "semantic_control_tests_csv": "information_analysis/semantic_control_tests.csv",
            "continuous_predictor_information_csv": "information_analysis/continuous_predictor_information.csv" if continuous_available else None,
            "continuous_conditional_comparisons_csv": "information_analysis/continuous_conditional_comparisons.csv" if continuous_available else None,
            "continuous_semantic_controls_csv": "information_analysis/continuous_semantic_controls_information.csv" if continuous_available else None,
            "continuous_semantic_control_tests_csv": "information_analysis/continuous_semantic_control_tests.csv" if continuous_available else None,
            "semantic_oracle_budgets_csv": "information_analysis/semantic_oracle_budgets.csv",
            "predictions_pt": "information_analysis/information_predictions.pt",
            "information_plots_dir": None if args.no_plots else "information_analysis/plots",
        },
    }
    if not args.no_plots:
        try:
            save_information_analysis_plots(info_dir, summary, semantic_control_test_rows, continuous_control_test_rows)
        except Exception as exc:
            summary["artifacts"]["information_plots_error"] = str(exc)
    write_csv(os.path.join(info_dir, "predictor_information.csv"), predictor_rows)
    write_csv(os.path.join(info_dir, "conditional_comparisons.csv"), comparison_rows)
    write_csv(os.path.join(info_dir, "leakage_proxy.csv"), leakage_proxy_rows)
    write_csv(os.path.join(info_dir, "semantic_controls_information.csv"), semantic_control_rows)
    write_csv(os.path.join(info_dir, "semantic_control_tests.csv"), semantic_control_test_rows)
    if continuous_available:
        write_csv(os.path.join(info_dir, "continuous_predictor_information.csv"), continuous_predictor_rows)
        write_csv(os.path.join(info_dir, "continuous_conditional_comparisons.csv"), continuous_comparison_rows)
        write_csv(os.path.join(info_dir, "continuous_semantic_controls_information.csv"), continuous_control_rows)
        write_csv(os.path.join(info_dir, "continuous_semantic_control_tests.csv"), continuous_control_test_rows)
    write_csv(os.path.join(info_dir, "semantic_oracle_budgets.csv"), oracle_budget_rows)
    torch.save(
        {
            "prior_logits": prior_logits,
            "e1e2_to_i_logits": e1e2_logits,
            "train_z1_recovered_from_e2": train_z1_recovered,
            "val_z1_recovered_from_e2": val_z1_recovered,
            "train_prev_recovered_from_e2": train_prev_recovered,
            "val_prev_recovered_from_e2": val_prev_recovered,
            "random_z_to_z1_pred": random_z1_pred,
            "mean_z1_pred": mean_z1_pred,
            "semantic_control_logits": semantic_control_logits,
            "train_z1_oracle": train_z1_oracle,
            "val_z1_oracle": val_z1_oracle,
            "train_prev_oracle": train_prev_oracle,
            "val_prev_oracle": val_prev_oracle,
            "continuous_predictions": continuous_predictions,
            "dynamic_feature_weights": dynamic_feature_weights,
        },
        os.path.join(info_dir, "information_predictions.pt"),
    )
    with open(os.path.join(info_dir, "information_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def base_example_id(example_id: str, step_pattern: str = r"::step\d+$") -> str:
    try:
        return re.sub(step_pattern, "", str(example_id))
    except re.error:
        return re.sub(r"::step\d+$", "", str(example_id))


def dataset_indices(subset: torch.utils.data.Dataset) -> List[int]:
    if isinstance(subset, Subset):
        parent = subset.dataset
        parent_indices = dataset_indices(parent)
        return [parent_indices[int(idx)] for idx in subset.indices]
    indices = getattr(subset, "indices", None)
    if indices is not None:
        return [int(idx) for idx in indices]
    return list(range(len(subset)))


def subset_ids(all_ids: List[str], subset: torch.utils.data.Dataset, max_samples: int = 0) -> List[str]:
    indices = dataset_indices(subset)
    if max_samples > 0:
        indices = indices[:max_samples]
    return [all_ids[int(idx)] for idx in indices]


def split_dataset_with_manifest(
    dataset: torch.utils.data.Dataset,
    all_ids: List[str],
    args: argparse.Namespace,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Optional[torch.utils.data.Dataset], Dict[str, Any]]:
    n = len(dataset)
    if n < 3:
        raise ValueError("Need at least 3 examples to create train/val splits.")
    generator = torch.Generator().manual_seed(args.split_seed)
    if args.split_by_base_id:
        groups: Dict[str, List[int]] = {}
        for idx, example_id in enumerate(all_ids):
            groups.setdefault(base_example_id(example_id, args.base_id_step_pattern), []).append(idx)
        group_ids = list(groups)
        perm = torch.randperm(len(group_ids), generator=generator).tolist()
        shuffled_groups = [group_ids[i] for i in perm]
        test_groups = int(len(shuffled_groups) * args.test_ratio)
        val_groups = int(len(shuffled_groups) * args.val_ratio)
        if args.test_ratio > 0.0:
            test_groups = max(1, test_groups)
        val_groups = max(1, val_groups)
        if test_groups + val_groups >= len(shuffled_groups):
            raise ValueError("Grouped split leaves no training groups; reduce --val-ratio/--test-ratio.")
        test_group_ids = shuffled_groups[:test_groups]
        val_group_ids = shuffled_groups[test_groups : test_groups + val_groups]
        train_group_ids = shuffled_groups[test_groups + val_groups :]
        test_indices = [idx for gid in test_group_ids for idx in groups[gid]]
        val_indices = [idx for gid in val_group_ids for idx in groups[gid]]
        train_indices = [idx for gid in train_group_ids for idx in groups[gid]]
        split_summary = {
            "mode": "group_by_base_id",
            "base_id_step_pattern": args.base_id_step_pattern,
            "seed": args.split_seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "total_rows": n,
            "total_groups": len(groups),
            "train_rows": len(train_indices),
            "val_rows": len(val_indices),
            "test_rows": len(test_indices),
            "train_groups": len(train_group_ids),
            "val_groups": len(val_group_ids),
            "test_groups": len(test_group_ids),
        }
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices) if test_indices else None
        return train_ds, val_ds, test_ds, split_summary

    val_size = max(1, int(n * args.val_ratio))
    test_size = max(1, int(n * args.test_ratio)) if args.test_ratio > 0.0 else 0
    train_size = n - val_size - test_size
    if train_size <= 0:
        raise ValueError("Split leaves no training rows; reduce --val-ratio/--test-ratio.")
    if test_size > 0:
        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    else:
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
        test_ds = None
    split_summary = {
        "mode": "row_random",
        "seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "total_rows": n,
        "train_rows": train_size,
        "val_rows": val_size,
        "test_rows": test_size,
    }
    return train_ds, val_ds, test_ds, split_summary


def write_split_assignments(
    output_dir: str,
    all_ids: List[str],
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    test_ds: Optional[torch.utils.data.Dataset],
    args: argparse.Namespace,
) -> None:
    rows = []
    split_to_indices = {
        "train": dataset_indices(train_ds),
        "val": dataset_indices(val_ds),
        "test": dataset_indices(test_ds) if test_ds is not None else [],
    }
    for split, indices in split_to_indices.items():
        for idx in indices:
            example_id = all_ids[int(idx)]
            rows.append(
                {
                    "row_index": int(idx),
                    "id": example_id,
                    "base_id": base_example_id(example_id, args.base_id_step_pattern),
                    "split": split,
                }
            )
    rows.sort(key=lambda row: row["row_index"])
    write_csv(os.path.join(output_dir, "split_assignments.csv"), rows)


def tertile_groups(scores: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Optional[float]]]:
    scores = scores.float().cpu()
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long), {"low_mid_cut": None, "mid_high_cut": None}
    if scores.numel() < 3 or scores.std(unbiased=False).item() <= 1e-8:
        return torch.ones(scores.numel(), dtype=torch.long), {"low_mid_cut": None, "mid_high_cut": None}
    cuts = torch.quantile(scores, torch.tensor([1.0 / 3.0, 2.0 / 3.0]))
    groups = torch.zeros(scores.numel(), dtype=torch.long)
    groups[scores > cuts[0]] = 1
    groups[scores > cuts[1]] = 2
    return groups, {"low_mid_cut": cuts[0].item(), "mid_high_cut": cuts[1].item()}


def group_name(group_id: int) -> str:
    return ["low", "mid", "high"][int(group_id)] if int(group_id) in {0, 1, 2} else "unknown"


def dynamic_intervention_candidate_rows(output_dir: str, count: int, export_threshold: float) -> Tuple[str, List[Dict]]:
    path = os.path.join(output_dir, "dynamic_targets", "target_weights_latest.pt")
    if not os.path.exists(path):
        return "dynamic_targets_missing", []
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return "dynamic_targets_unreadable", []
    if not isinstance(obj, dict) or "weights" not in obj:
        return "dynamic_targets_invalid", []
    summary = obj.get("summary", {}) if isinstance(obj.get("summary", {}), dict) else {}
    if summary.get("target_type", "neuron") != "neuron":
        return "dynamic_direction_targets_not_neuron_intervention_targets", []

    weights = obj["weights"].float().flatten()
    selected = obj.get("selected", weights > export_threshold).bool().flatten()
    score = obj.get("score", torch.zeros_like(weights)).float().flatten()
    semantic_gap = obj.get("semantic_gap_nats", score).float().flatten()
    e2_gain = obj.get("e2_gain_nats", torch.full_like(weights, float("nan"))).float().flatten()
    main_e2_gain = obj.get("main_e2_gain_nats", torch.full_like(weights, float("nan"))).float().flatten()
    semantic_gain = obj.get("semantic_gain_nats", torch.full_like(weights, float("nan"))).float().flatten()
    gap_ci_low = obj.get("semantic_gap_ci_low", torch.full_like(weights, float("nan"))).float().flatten()
    best_semantic_ce = obj.get("best_semantic_ce", torch.full_like(weights, float("nan"))).float().flatten()
    e2_ce = obj.get("e2_ce", torch.full_like(weights, float("nan"))).float().flatten()
    prior_ce = obj.get("prior_ce", torch.full_like(weights, float("nan"))).float().flatten()
    activation_rate = obj.get("activation_rate", torch.full_like(weights, float("nan"))).float().flatten()
    best_control_idx = obj.get("best_semantic_control_idx", torch.full_like(weights, -1, dtype=torch.long)).long().flatten()
    control_names = ["recovered_z1", "recovered_prev", "recovered_z1_prev"]

    candidate_idx = torch.nonzero((weights > export_threshold) | selected, as_tuple=False).flatten()
    if candidate_idx.numel() == 0:
        return "dynamic_semantic_gap_empty", []
    ranking_score = semantic_gap[candidate_idx].nan_to_num(-1e9) + 1e-3 * weights[candidate_idx] + 1e-6 * e2_gain[candidate_idx].nan_to_num(0.0)
    order = torch.argsort(ranking_score, descending=True)
    ranked = candidate_idx[order]
    limit = min(max(1, count), ranked.numel())
    rows = []
    for rank, neuron_tensor in enumerate(ranked[:limit], start=1):
        neuron = int(neuron_tensor.item())
        control_idx = int(best_control_idx[neuron].item())
        rows.append(
            {
                "dynamic_rank": rank,
                "neuron": neuron,
                "weight": weights[neuron].item(),
                "score_nats": score[neuron].item(),
                "semantic_gap_nats": semantic_gap[neuron].item(),
                "semantic_gap_bits": semantic_gap[neuron].item() / math.log(2.0),
                "semantic_gap_ci_low_nats": gap_ci_low[neuron].item(),
                "semantic_gap_ci_low_bits": gap_ci_low[neuron].item() / math.log(2.0),
                "e2_gain_nats": e2_gain[neuron].item(),
                "e2_gain_bits": e2_gain[neuron].item() / math.log(2.0),
                "main_e2_gain_nats": main_e2_gain[neuron].item(),
                "main_e2_gain_bits": main_e2_gain[neuron].item() / math.log(2.0),
                "semantic_gain_nats": semantic_gain[neuron].item(),
                "semantic_gain_bits": semantic_gain[neuron].item() / math.log(2.0),
                "prior_ce": prior_ce[neuron].item(),
                "e2_ce": e2_ce[neuron].item(),
                "best_semantic_ce": best_semantic_ce[neuron].item(),
                "best_semantic_control_idx": control_idx,
                "best_semantic_control": control_names[control_idx] if 0 <= control_idx < len(control_names) else "unknown",
                "activation_rate": activation_rate[neuron].item(),
            }
        )
    return "dynamic_semantic_gap_targets", rows


def probability_margin_candidate_rows(neuron_rows: List[Dict], count: int) -> Tuple[str, List[Dict]]:
    variable_rows = [row for row in neuron_rows if row.get("is_variable")]
    ranked = sorted(
        variable_rows,
        key=lambda row: row.get("probability_margin", float("nan")),
        reverse=True,
    )
    return "fallback_probability_margin_variable", ranked[: max(1, count)]


def export_intervention_interface(
    analysis_dir: str,
    all_ids: List[str],
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    train_tensors: Dict[str, torch.Tensor],
    val_tensors: Dict[str, torch.Tensor],
    neuron_rows: List[Dict],
    args: argparse.Namespace,
    binary_eval_meta: Optional[Dict[str, torch.Tensor]],
    prev_layer: int,
    next_layer: int,
    main_metrics: Dict,
    probe_metrics: Dict[str, Dict],
    leakage_metrics: Dict[str, Dict],
) -> Dict:
    export_count = args.intervention_top_neurons
    selection_source, candidate_rows = dynamic_intervention_candidate_rows(args.output_dir, export_count, args.dynamic_target_export_threshold)
    if not candidate_rows and not args.dynamic_target_weighting:
        selection_source, candidate_rows = probability_margin_candidate_rows(neuron_rows, export_count)
    selected_neurons = torch.tensor([int(row["neuron"]) for row in candidate_rows], dtype=torch.long)
    train_ids = subset_ids(all_ids, train_ds, args.probe_max_train_samples)
    val_ids = subset_ids(all_ids, val_ds, args.probe_max_val_samples)
    if len(val_ids) != val_tensors["x_i"].size(0):
        val_ids = val_ids[: val_tensors["x_i"].size(0)]

    probs = torch.sigmoid(val_tensors["main_logits"].float())
    target_binary = val_tensors["x_i"].float()
    if selected_neurons.numel() > 0:
        selected_target = target_binary[:, selected_neurons]
        selected_probs = probs[:, selected_neurons]
        true_score = selected_target.mean(dim=1)
        pred_score = selected_probs.mean(dim=1)
    else:
        true_score = torch.zeros(target_binary.size(0))
        pred_score = torch.zeros(target_binary.size(0))

    true_groups, true_group_cuts = tertile_groups(true_score)
    pred_groups, pred_group_cuts = tertile_groups(pred_score)
    sample_rows = []
    for idx, sample_id in enumerate(val_ids):
        sample_rows.append(
            {
                "val_index": idx,
                "id": sample_id,
                "selected_true_activation_score": true_score[idx].item(),
                "selected_pred_activation_score": pred_score[idx].item(),
                "true_score_group": group_name(int(true_groups[idx].item())),
                "pred_score_group": group_name(int(pred_groups[idx].item())),
            }
        )
    write_csv(os.path.join(analysis_dir, "intervention_sample_groups.csv"), sample_rows)

    target_rows = []
    metrics_by_neuron = {int(row["neuron"]): row for row in neuron_rows}
    for candidate in candidate_rows:
        neuron = int(candidate["neuron"])
        copied = dict(metrics_by_neuron.get(neuron, {}))
        copied.update(candidate)
        copied["intervention_selection_source"] = selection_source
        target_rows.append(copied)
    write_csv(os.path.join(analysis_dir, "intervention_neurons.csv"), target_rows)

    threshold_tensor = binary_eval_meta.get("threshold") if binary_eval_meta else None
    mean_tensor = binary_eval_meta.get("mean") if binary_eval_meta and "mean" in binary_eval_meta else None
    std_tensor = binary_eval_meta.get("std") if binary_eval_meta and "std" in binary_eval_meta else None
    tensor_bundle = {
        "schema_version": 1,
        "selected_neurons": selected_neurons,
        "val_target_binary": target_binary.to(torch.uint8),
        "val_main_logits": val_tensors["main_logits"].float(),
        "val_main_probs": probs,
        "val_selected_true_score": true_score,
        "val_selected_pred_score": pred_score,
        "val_true_score_group": true_groups,
        "val_pred_score_group": pred_groups,
        "binary_activation_threshold": threshold_tensor.cpu() if threshold_tensor is not None else None,
        "continuous_mean": mean_tensor.cpu() if mean_tensor is not None else None,
        "continuous_std": std_tensor.cpu() if std_tensor is not None else None,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "metadata": {
            "activation_dir": args.activation_dir,
            "model_label": args.model_label,
            "num_model_layers": getattr(args, "num_model_layers", None),
            "layer_fractions": getattr(args, "resolved_layer_fracs", None),
            "prev_layer": prev_layer,
            "layer_i": args.layer_i,
            "next_layer": next_layer,
            "prev_gap": args.layer_i - prev_layer,
            "next_gap": next_layer - args.layer_i,
            "target_mode": args.target_mode,
            "binary_quantile": args.binary_quantile,
            "binary_pred_threshold": args.binary_pred_threshold,
            "selection_source": selection_source,
        },
    }
    torch.save(tensor_bundle, os.path.join(analysis_dir, "intervention_targets.pt"))

    summary = {
        "schema_version": 1,
        "activation_dir": args.activation_dir,
        "model_label": args.model_label,
        "num_model_layers": getattr(args, "num_model_layers", None),
        "layer_fractions": getattr(args, "resolved_layer_fracs", None),
        "prev_layer": prev_layer,
        "layer_i": args.layer_i,
        "next_layer": next_layer,
        "hidden_state_note": "Layer ids are decoder block ids; intervention should modify hidden_states[layer_i + 1] / residual stream feature dimensions.",
        "target_mode": args.target_mode,
        "binary_quantile": args.binary_quantile,
        "binary_pred_threshold": args.binary_pred_threshold,
        "selection_source": selection_source,
        "selected_neuron_count": int(selected_neurons.numel()),
        "selected_neurons": selected_neurons.tolist(),
        "true_score_group_cuts": true_group_cuts,
        "pred_score_group_cuts": pred_group_cuts,
        "main_metrics": main_metrics,
        "probe_metric_names": list(probe_metrics.keys()),
        "leakage_metric_names": list(leakage_metrics.keys()),
        "artifacts": {
            "tensor_bundle": "intervention_targets.pt",
            "sample_groups_csv": "intervention_sample_groups.csv",
            "intervention_neurons_csv": "intervention_neurons.csv",
            "per_neuron_metrics_csv": "per_neuron_binary_metrics.csv",
            "variable_neuron_diagnostics_csv": "variable_neuron_diagnostics.csv",
            "dynamic_targets_csv": "../dynamic_targets/target_weights_latest.csv",
        },
        "suggested_interventions": [
            "Use selected_neurons as semantic-gap-supported targets for intervention or cluster/audit scripts.",
            "Prefer E2-latent or decoder-based interventions when testing semantic-independent control.",
        ],
    }
    with open(os.path.join(analysis_dir, "intervention_targets.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def evaluate_e2_recursive_purifier(
    purifier: E2RecursivePurifier,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    feature_weights: Optional[torch.Tensor],
    binary_eval_meta: Optional[Dict[str, torch.Tensor]],
    prediction_weight: Optional[float] = None,
    training_stage: str = "residual_meta",
) -> Dict[str, float]:
    purifier.eval()
    if prediction_weight is None:
        prediction_weight = args.lambda_purifier_pred
    feature_weights_device = feature_weights.to(device) if feature_weights is not None else None
    totals = {"loss": 0.0, "recon_z2": 0.0, "semantic_recon_z2": 0.0, "meta_residual": 0.0, "semantic_z1": 0.0, "semantic_prev": 0.0, "pred": 0.0, "orth": 0.0, "meta_sem_cov": 0.0, "var": 0.0, "meta_input_l2_ratio": 0.0}
    meta_chunks, semantic_chunks, pred_chunks, target_eval_chunks = [], [], [], []
    seen = 0
    with torch.no_grad():
        for z2, z1, x_prev, x_i_train, x_i_eval in loader:
            z2 = z2.to(device)
            z1 = z1.to(device)
            x_prev = x_prev.to(device)
            x_i_train = x_i_train.to(device)
            x_i_eval = x_i_eval.to(device)
            out = purifier(z2)
            loss_recon = F.mse_loss(out["z2_hat"], z2)
            loss_semantic_recon = F.mse_loss(out["semantic_z2_hat"], z2)
            loss_meta_residual = F.mse_loss(out["meta_residual_hat"], out["meta_input"].detach())
            loss_z1 = F.mse_loss(out["z1_hat"], z1)
            loss_prev = F.mse_loss(out["prev_hat"], x_prev)
            loss_pred = weighted_prediction_loss(out["i_hat"], x_i_train, args.target_mode, feature_weights_device)
            loss_orth = orthogonality_loss(out["semantic"], out["meta"])
            loss_cov = cross_cov_loss(out["meta"], z1) + cross_cov_loss(out["meta"], x_prev)
            if training_stage == "semantic_encoder":
                loss_var = variance_floor_loss(out["semantic"])
            elif training_stage == "semantic_z2_decoder":
                loss_var = out["semantic"].new_tensor(0.0)
            else:
                loss_var = variance_floor_loss(out["meta"])
            semantic_loss_weight = args.lambda_purifier_semantic if training_stage == "semantic_encoder" else 0.0
            semantic_recon_weight = args.lambda_purifier_semantic_recon_z2 if training_stage == "semantic_z2_decoder" else 0.0
            residual_stage_weight = 1.0 if training_stage == "residual_meta" else 0.0
            loss = (
                residual_stage_weight * args.lambda_purifier_recon_z2 * loss_recon
                + semantic_recon_weight * loss_semantic_recon
                + residual_stage_weight * args.lambda_purifier_meta_residual * loss_meta_residual
                + semantic_loss_weight * (loss_z1 + loss_prev)
                + prediction_weight * loss_pred
                + residual_stage_weight * args.lambda_purifier_orth * loss_orth
                + residual_stage_weight * args.lambda_purifier_meta_sem_cov * loss_cov
                + args.lambda_purifier_var * loss_var
            )
            bs = z2.size(0)
            totals["loss"] += loss.item() * bs
            totals["recon_z2"] += loss_recon.item() * bs
            totals["semantic_recon_z2"] += loss_semantic_recon.item() * bs
            totals["meta_residual"] += loss_meta_residual.item() * bs
            totals["semantic_z1"] += loss_z1.item() * bs
            totals["semantic_prev"] += loss_prev.item() * bs
            totals["pred"] += loss_pred.item() * bs
            totals["orth"] += loss_orth.item() * bs
            totals["meta_sem_cov"] += loss_cov.item() * bs
            totals["var"] += loss_var.item() * bs
            residual_ratio = out["meta_input"].float().norm(dim=1) / z2.float().norm(dim=1).clamp_min(1e-8)
            totals["meta_input_l2_ratio"] += residual_ratio.mean().item() * bs
            meta_chunks.append(out["meta"].detach().cpu())
            semantic_chunks.append(out["semantic"].detach().cpu())
            pred_chunks.append(out["i_hat"].detach().cpu())
            target_eval_chunks.append(x_i_eval.detach().cpu())
            seen += bs
    stats = {key: value / max(seen, 1) for key, value in totals.items()}
    meta = torch.cat(meta_chunks, dim=0) if meta_chunks else torch.empty(0)
    semantic = torch.cat(semantic_chunks, dim=0) if semantic_chunks else torch.empty(0)
    pred = torch.cat(pred_chunks, dim=0) if pred_chunks else torch.empty(0)
    target_eval = torch.cat(target_eval_chunks, dim=0) if target_eval_chunks else torch.empty(0)
    stats["meta_std_mean"] = meta.std(dim=0, unbiased=False).mean().item() if meta.numel() else float("nan")
    stats["semantic_std_mean"] = semantic.std(dim=0, unbiased=False).mean().item() if semantic.numel() else float("nan")
    if pred.numel():
        if is_binary_eval_mode(args.target_mode):
            logits = binary_eval_logits(pred, args.target_mode, binary_eval_meta)
            stats["pred_aux"] = -binary_ce_from_logits(logits, target_eval)
        else:
            pred_flat = pred.flatten()
            target_flat = target_eval.flatten()
            pred_z = (pred_flat - pred_flat.mean()) / pred_flat.std(unbiased=False).clamp_min(1e-6)
            target_z = (target_flat - target_flat.mean()) / target_flat.std(unbiased=False).clamp_min(1e-6)
            stats["pred_aux"] = (pred_z * target_z).mean().item()
    else:
        stats["pred_aux"] = float("nan")
    return stats


def collect_e2_purifier_codes(purifier: E2RecursivePurifier, tensors: Dict[str, torch.Tensor], device: torch.device, batch_size: int) -> Dict[str, torch.Tensor]:
    purifier.eval()
    outputs = {"semantic": [], "meta": [], "z2_hat": [], "i_hat": [], "z1_hat": [], "prev_hat": []}
    z2 = tensors["z2"]
    with torch.no_grad():
        for start in range(0, z2.size(0), batch_size):
            out = purifier(z2[start : start + batch_size].to(device))
            for key in outputs:
                outputs[key].append(out[key].detach().cpu())
    return {key: torch.cat(value, dim=0) for key, value in outputs.items()}


def compute_purifier_dynamic_target_weights(
    purifier: E2RecursivePurifier,
    train_tensors: Dict[str, torch.Tensor],
    val_tensors: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
    old_weights: Optional[torch.Tensor],
    inherited_weights: Optional[torch.Tensor],
    epoch: int,
    binary_eval_meta: Optional[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
    train_codes = collect_e2_purifier_codes(purifier, train_tensors, device, batch_size)
    val_codes = collect_e2_purifier_codes(purifier, val_tensors, device, batch_size)

    probe_epochs = args.purifier_dynamic_probe_epochs if args.purifier_dynamic_probe_epochs > 0 else (args.purifier_probe_epochs if args.purifier_probe_epochs > 0 else args.probe_epochs)
    semantic_probe_hidden = (
        args.purifier_dynamic_probe_hidden_dim
        if args.purifier_dynamic_probe_hidden_dim > 0
        else (args.purifier_probe_hidden_dim if args.purifier_probe_hidden_dim > 0 else (args.purifier_hidden_dim if args.purifier_hidden_dim > 0 else args.hidden_dim))
    )
    meta_logits = train_binary_linear_probe_logits_only(
        train_codes["meta"],
        train_tensors["x_i"],
        val_codes["meta"],
        device,
        probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Purifier dynamic meta target probe epoch {epoch}",
    )
    if args.purifier_dynamic_semantic_probe == "mlp":
        train_z1_recovered, val_z1_recovered = train_regression_mlp_probe_predict_only(
            train_codes["meta"],
            train_tensors["z1"],
            val_codes["meta"],
            device,
            probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            semantic_probe_hidden,
            args.dropout,
            f"Purifier dynamic MLP meta->Z1 leakage probe epoch {epoch}",
        )
        train_prev_recovered, val_prev_recovered = train_regression_mlp_probe_predict_only(
            train_codes["meta"],
            train_tensors["x_prev"],
            val_codes["meta"],
            device,
            probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            semantic_probe_hidden,
            args.dropout,
            f"Purifier dynamic MLP meta->prev leakage probe epoch {epoch}",
        )
    else:
        train_z1_recovered, val_z1_recovered = train_regression_linear_probe_predict_only(
            train_codes["meta"],
            train_tensors["z1"],
            val_codes["meta"],
            device,
            probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            f"Purifier dynamic linear meta->Z1 leakage probe epoch {epoch}",
        )
        train_prev_recovered, val_prev_recovered = train_regression_linear_probe_predict_only(
            train_codes["meta"],
            train_tensors["x_prev"],
            val_codes["meta"],
            device,
            probe_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            f"Purifier dynamic linear meta->prev leakage probe epoch {epoch}",
        )
    recovered_z1_logits = train_binary_linear_probe_logits_only(
        train_z1_recovered,
        train_tensors["x_i"],
        val_z1_recovered,
        device,
        probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Purifier dynamic recovered-Z1 target control epoch {epoch}",
    )
    recovered_prev_logits = train_binary_linear_probe_logits_only(
        train_prev_recovered,
        train_tensors["x_i"],
        val_prev_recovered,
        device,
        probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Purifier dynamic recovered-prev target control epoch {epoch}",
    )
    recovered_both_logits = train_binary_linear_probe_logits_only(
        torch.cat([train_z1_recovered, train_prev_recovered], dim=1),
        train_tensors["x_i"],
        torch.cat([val_z1_recovered, val_prev_recovered], dim=1),
        device,
        probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        f"Purifier dynamic recovered-Z1-prev target control epoch {epoch}",
    )

    prior_logits = binary_prior_logits(train_tensors["x_i"], val_tensors["x_i"])
    prior_ce = per_feature_binary_ce(prior_logits, val_tensors["x_i"])
    meta_ce = per_feature_binary_ce(meta_logits, val_tensors["x_i"])
    if binary_eval_meta is not None and is_binary_eval_mode(args.target_mode):
        direct_logits = binary_eval_logits(val_codes["i_hat"], args.target_mode, binary_eval_meta)
        direct_ce = per_feature_binary_ce(direct_logits, val_tensors["x_i"])
    else:
        direct_logits = meta_logits
        direct_ce = meta_ce
    recovered_z1_ce = per_feature_binary_ce(recovered_z1_logits, val_tensors["x_i"])
    recovered_prev_ce = per_feature_binary_ce(recovered_prev_logits, val_tensors["x_i"])
    recovered_both_ce = per_feature_binary_ce(recovered_both_logits, val_tensors["x_i"])
    semantic_stack = torch.stack([recovered_z1_ce, recovered_prev_ce, recovered_both_ce], dim=0)
    best_semantic_ce, best_semantic_control_idx = semantic_stack.min(dim=0)

    meta_gain = prior_ce - meta_ce
    direct_gain = prior_ce - direct_ce
    semantic_gain = prior_ce - best_semantic_ce
    semantic_gap = best_semantic_ce - meta_ce
    score = semantic_gap

    if args.purifier_dynamic_bootstrap_samples > 0:
        meta_loss = F.binary_cross_entropy_with_logits(meta_logits.float(), val_tensors["x_i"].float(), reduction="none")
        best_semantic_logits = torch.stack([recovered_z1_logits, recovered_prev_logits, recovered_both_logits], dim=0)
        gather_idx = best_semantic_control_idx.long().view(1, 1, -1).expand(1, best_semantic_logits.size(1), -1)
        gathered_semantic_logits = best_semantic_logits.gather(0, gather_idx).squeeze(0)
        best_semantic_loss = F.binary_cross_entropy_with_logits(gathered_semantic_logits.float(), val_tensors["x_i"].float(), reduction="none")
        gap_samples = best_semantic_loss - meta_loss
        generator = torch.Generator().manual_seed(args.seed + 9300 + epoch)
        boot_means = []
        for _ in tqdm(range(args.purifier_dynamic_bootstrap_samples), desc=f"Purifier dynamic bootstrap epoch {epoch}", leave=False):
            idx = torch.randint(0, gap_samples.size(0), (gap_samples.size(0),), generator=generator)
            boot_means.append(gap_samples[idx].mean(dim=0))
        gap_ci_low = torch.quantile(torch.stack(boot_means, dim=0), 0.025, dim=0)
    else:
        gap_ci_low = torch.full_like(score, float("nan"))

    train_activation_rate = train_tensors["x_i"].float().mean(dim=0)
    activation_rate = val_tensors["x_i"].float().mean(dim=0)
    activation_rate_drift = (train_activation_rate - activation_rate).abs()
    pos = val_tensors["x_i"].float().sum(dim=0)
    neg = val_tensors["x_i"].size(0) - pos
    variable = (
        (train_activation_rate >= args.min_neuron_activation_rate)
        & (train_activation_rate <= args.max_neuron_activation_rate)
        & (activation_rate >= args.min_neuron_activation_rate)
        & (activation_rate <= args.max_neuron_activation_rate)
        & (activation_rate_drift <= args.max_neuron_activation_rate_drift)
        & (pos >= args.min_neuron_positive)
        & (neg >= args.min_neuron_negative)
    )
    candidate_mask = torch.ones_like(variable)
    if inherited_weights is not None and args.purifier_dynamic_restrict_to_inherited_targets:
        candidate_mask = inherited_weights.float() > args.dynamic_target_export_threshold
    eligible = (
        variable
        & candidate_mask
        & (meta_gain >= args.purifier_dynamic_e2_gain_min)
        & (score >= args.purifier_dynamic_score_min)
    )
    if args.purifier_dynamic_require_positive_ci:
        eligible = eligible & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
    claim_score_min = (
        args.purifier_dynamic_score_min
        if args.purifier_dynamic_claim_score_min is None
        else args.purifier_dynamic_claim_score_min
    )
    claim_e2_gain_min = (
        args.purifier_dynamic_e2_gain_min
        if args.purifier_dynamic_claim_e2_gain_min is None
        else args.purifier_dynamic_claim_e2_gain_min
    )
    strict_eligible = (
        variable
        & candidate_mask
        & (meta_gain >= claim_e2_gain_min)
        & (score >= claim_score_min)
    )
    if args.purifier_dynamic_require_positive_ci:
        strict_eligible = strict_eligible & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
    strict_eligible_count = int(strict_eligible.sum().item())
    fallback_used = False
    strict_training_only = bool(args.purifier_gate_prediction_on_strict_targets)
    if not strict_training_only and eligible.sum().item() < args.purifier_dynamic_min_selected:
        fallback_mask = variable & candidate_mask & (meta_gain >= args.purifier_dynamic_e2_gain_min)
        if args.purifier_dynamic_require_positive_ci:
            fallback_mask = fallback_mask & torch.isfinite(gap_ci_low) & (gap_ci_low > 0.0)
        if fallback_mask.sum().item() > 0:
            eligible = fallback_mask
            fallback_used = True

    weights = torch.zeros_like(score)
    full_weight = torch.zeros_like(score, dtype=torch.bool)
    claim_selected = torch.zeros_like(score, dtype=torch.bool)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_indices.numel() > 0:
        ranked = eligible_indices[torch.argsort(score[eligible_indices], descending=True)]
        selection_min_selected = 0 if strict_training_only else args.purifier_dynamic_min_selected
        train_top_k = args.purifier_dynamic_train_top_k if args.purifier_dynamic_train_top_k > 0 else args.purifier_dynamic_top_k
        limit = dynamic_selection_limit(
            ranked.numel(),
            train_top_k,
            selection_min_selected,
            fallback_used,
        )
        selected = ranked[:limit]
        full_weight[selected] = True
        if args.purifier_dynamic_weight_mode == "topk":
            weights[selected] = 1.0
        elif args.purifier_dynamic_weight_mode == "soft":
            temp = max(args.purifier_dynamic_temperature, 1e-6)
            soft = torch.sigmoid((score[selected] - args.purifier_dynamic_score_min) / temp)
            weights[selected] = soft.clamp_min(args.purifier_dynamic_min_soft_weight)
        else:
            raise ValueError(args.purifier_dynamic_weight_mode)

    strict_indices = torch.nonzero(strict_eligible, as_tuple=False).flatten()
    if strict_indices.numel() > 0:
        ranked_strict = strict_indices[torch.argsort(score[strict_indices], descending=True)]
        claim_top_k = args.purifier_dynamic_claim_top_k if args.purifier_dynamic_claim_top_k > 0 else args.purifier_dynamic_top_k
        claim_limit = ranked_strict.numel() if claim_top_k <= 0 else min(ranked_strict.numel(), claim_top_k)
        claim_selected[ranked_strict[:claim_limit]] = True

    if old_weights is not None and args.purifier_dynamic_ema > 0.0:
        weights = args.purifier_dynamic_ema * old_weights.float() + (1.0 - args.purifier_dynamic_ema) * weights
        weights = torch.where(variable & candidate_mask, weights, torch.zeros_like(weights))
        full_weight = weights > args.dynamic_target_export_threshold

    selected_mask = weights > args.dynamic_target_export_threshold
    selected_count = int(selected_mask.sum().item())
    if selected_count > 0:
        selected_score = score[selected_mask]
        selected_meta_gain = meta_gain[selected_mask]
        selected_direct_gain = direct_gain[selected_mask]
        selected_semantic_gain = semantic_gain[selected_mask]
        selected_gap_ci_low = gap_ci_low[selected_mask]
        selected_activation = activation_rate[selected_mask]
        residual_fraction = (selected_score / selected_meta_gain.clamp_min(1e-8)).mean().item()
        leakage_fraction = (selected_semantic_gain / selected_meta_gain.clamp_min(1e-8)).mean().item()
    else:
        selected_score = torch.empty(0)
        selected_meta_gain = torch.empty(0)
        selected_direct_gain = torch.empty(0)
        selected_semantic_gain = torch.empty(0)
        selected_gap_ci_low = torch.empty(0)
        selected_activation = torch.empty(0)
        residual_fraction = float("nan")
        leakage_fraction = float("nan")

    tensors = {
        "weights": weights.cpu(),
        "score": score.cpu(),
        "semantic_gap_nats": semantic_gap.cpu(),
        "e2_gain_nats": meta_gain.cpu(),
        "main_e2_gain_nats": direct_gain.cpu(),
        "semantic_gain_nats": semantic_gain.cpu(),
        "prior_ce": prior_ce.cpu(),
        "e2_ce": meta_ce.cpu(),
        "main_e2_ce": direct_ce.cpu(),
        "best_semantic_ce": best_semantic_ce.cpu(),
        "recovered_z1_ce": recovered_z1_ce.cpu(),
        "recovered_prev_ce": recovered_prev_ce.cpu(),
        "recovered_both_ce": recovered_both_ce.cpu(),
        "best_semantic_control_idx": best_semantic_control_idx.cpu(),
        "semantic_gap_ci_low": gap_ci_low.cpu(),
        "activation_rate": activation_rate.cpu(),
        "train_activation_rate": train_activation_rate.cpu(),
        "activation_rate_drift": activation_rate_drift.cpu(),
        "variable": variable.cpu(),
        "eligible": eligible.cpu(),
        "strict_eligible": strict_eligible.cpu(),
        "selected": selected_mask.cpu(),
        "full_weight": full_weight.cpu(),
        "claim_selected": claim_selected.cpu(),
        "candidate_mask": candidate_mask.cpu(),
    }
    claim_score = score[claim_selected]
    full_score = score[full_weight]
    strict_score = score[strict_eligible]
    claim_meta_gain = meta_gain[claim_selected]
    full_meta_gain = meta_gain[full_weight]
    strict_meta_gain = meta_gain[strict_eligible]
    claim_positive_gap_sum = claim_score.clamp_min(0.0).sum().item() if claim_score.numel() else 0.0
    full_positive_gap_sum = full_score.clamp_min(0.0).sum().item() if full_score.numel() else 0.0
    strict_positive_gap_sum = strict_score.clamp_min(0.0).sum().item() if strict_score.numel() else 0.0
    claim_meta_gain_sum = claim_meta_gain.clamp_min(0.0).sum().item() if claim_meta_gain.numel() else 0.0
    full_meta_gain_sum = full_meta_gain.clamp_min(0.0).sum().item() if full_meta_gain.numel() else 0.0
    strict_meta_gain_sum = strict_meta_gain.clamp_min(0.0).sum().item() if strict_meta_gain.numel() else 0.0
    claim_residual_fraction = claim_positive_gap_sum / max(claim_meta_gain_sum, 1e-8)
    full_residual_fraction = full_positive_gap_sum / max(full_meta_gain_sum, 1e-8)
    strict_residual_fraction = strict_positive_gap_sum / max(strict_meta_gain_sum, 1e-8)
    summary = {
        "epoch": epoch,
        "target_type": "neuron",
        "selected_count": selected_count,
        "eligible_count": int(eligible.sum().item()),
        "strict_eligible_count": strict_eligible_count,
        "full_weight_count": int(full_weight.sum().item()),
        "claim_target_count": int(claim_selected.sum().item()),
        "strict_training_only": strict_training_only,
        "relaxed_min_selected_fallback_used": fallback_used,
        "variable_count": int(variable.sum().item()),
        "candidate_count": int(candidate_mask.sum().item()),
        "restrict_to_inherited_targets": bool(args.purifier_dynamic_restrict_to_inherited_targets),
        "allow_new_targets": not bool(args.purifier_dynamic_restrict_to_inherited_targets),
        "weight_sum": float(weights.sum().item()),
        "weight_mean": float(weights.mean().item()) if weights.numel() else float("nan"),
        "weight_nonzero": int((weights > 0).sum().item()),
        "score_mean_selected": selected_score.mean().item() if selected_score.numel() else float("nan"),
        "score_median_selected": selected_score.median().item() if selected_score.numel() else float("nan"),
        "semantic_gap_nats_mean_selected": selected_score.mean().item() if selected_score.numel() else float("nan"),
        "full_weight_positive_gap_sum": full_positive_gap_sum,
        "claim_positive_gap_sum": claim_positive_gap_sum,
        "strict_positive_gap_sum": strict_positive_gap_sum,
        "full_meta_gain_sum": full_meta_gain_sum,
        "claim_meta_gain_sum": claim_meta_gain_sum,
        "strict_meta_gain_sum": strict_meta_gain_sum,
        "full_residual_fraction": full_residual_fraction,
        "claim_residual_fraction": claim_residual_fraction,
        "strict_residual_fraction": strict_residual_fraction,
        "claim_gap_nats_mean": claim_score.mean().item() if claim_score.numel() else float("nan"),
        "full_gap_nats_mean": full_score.mean().item() if full_score.numel() else float("nan"),
        "semantic_gap_ci_low_mean_selected": selected_gap_ci_low[torch.isfinite(selected_gap_ci_low)].mean().item()
        if torch.isfinite(selected_gap_ci_low).any()
        else float("nan"),
        "e2_gain_nats_mean_selected": selected_meta_gain.mean().item() if selected_meta_gain.numel() else float("nan"),
        "direct_head_gain_nats_mean_selected": selected_direct_gain.mean().item() if selected_direct_gain.numel() else float("nan"),
        "semantic_gain_nats_mean_selected": selected_semantic_gain.mean().item() if selected_semantic_gain.numel() else float("nan"),
        "activation_rate_mean_selected": selected_activation.mean().item() if selected_activation.numel() else float("nan"),
        "dynamic_residual_fraction_mean_selected": residual_fraction,
        "dynamic_leakage_explained_fraction_mean_selected": leakage_fraction,
        "selection_rule": args.purifier_dynamic_weight_mode,
        "selection_objective": "purified_meta_vs_recovered_semantic_control_gap",
        "semantic_probe": args.purifier_dynamic_semantic_probe,
        "probe_epochs": int(probe_epochs),
        "semantic_probe_hidden_dim": int(semantic_probe_hidden),
        "top_k": args.purifier_dynamic_top_k,
        "train_top_k": args.purifier_dynamic_train_top_k,
        "claim_top_k": args.purifier_dynamic_claim_top_k,
        "score_min": args.purifier_dynamic_score_min,
        "e2_gain_min": args.purifier_dynamic_e2_gain_min,
        "claim_score_min": claim_score_min,
        "claim_e2_gain_min": claim_e2_gain_min,
        "bootstrap_samples": args.purifier_dynamic_bootstrap_samples,
        "require_positive_ci": args.purifier_dynamic_require_positive_ci,
    }
    return weights.cpu(), tensors, summary


def save_e2_recursive_purifier_plots(
    purifier_dir: str,
    history: List[Dict],
    probe_metrics: Dict[str, Dict],
    purifier_information: Optional[Dict],
    val_codes: Dict[str, torch.Tensor],
    val_target: torch.Tensor,
    val_meta_logits: torch.Tensor,
    feature_weights: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_nmi_style()
    plot_dir = os.path.join(purifier_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if history:
        epochs = [row["epoch"] for row in history]
        figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.9))
        paired_series = [
            ("recon_z2", "E2 reconstruction", COLORS["blue"]),
            ("semantic_recon_z2", "Semantic-only reconstruction", COLORS["orange"]),
            ("meta_residual", "Meta residual", COLORS["green"]),
        ]
        for stem, label, color in paired_series:
            axes[0].plot(
                epochs,
                [row.get(f"train_{stem}", float("nan")) for row in history],
                color=color,
                linestyle="--",
                alpha=0.65,
                label=f"{label}, train",
            )
            axes[0].plot(
                epochs,
                [row.get(f"val_{stem}", float("nan")) for row in history],
                color=color,
                label=f"{label}, validation",
            )
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Purifier epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Representation objectives", loc="left")
        axes[0].legend(ncol=1, fontsize=6.2)
        style_axis(axes[0])
        panel_label(axes[0], "a")

        for stem, label, color in [
            ("pred", "Target prediction", COLORS["vermillion"]),
            ("meta_sem_cov", "Meta-semantic covariance", COLORS["purple"]),
        ]:
            axes[1].plot(
                epochs,
                [row.get(f"train_{stem}", float("nan")) for row in history],
                color=color,
                linestyle="--",
                alpha=0.65,
                label=f"{label}, train",
            )
            axes[1].plot(
                epochs,
                [row.get(f"val_{stem}", float("nan")) for row in history],
                color=color,
                label=f"{label}, validation",
            )
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Purifier epoch")
        axes[1].set_ylabel("Loss or penalty")
        axes[1].set_title("Prediction and independence", loc="left")
        axes[1].legend(fontsize=6.2)
        style_axis(axes[1])
        panel_label(axes[1], "b")
        figure.tight_layout()
        save_figure(figure, os.path.join(plot_dir, "purifier_training_losses"))
        plt.close(figure)

        if getattr(args, "extended_diagnostic_plots", False):
            plt.figure(figsize=(9, 5))
            for key, label in [
                ("val_semantic_z1", "semantic->Z1"),
                ("val_semantic_prev", "semantic->prev"),
                ("val_semantic_recon_z2", "semantic->E2"),
                ("val_meta_residual", "meta->residual"),
                ("val_pred", "meta->i"),
                ("val_recon_z2", "semantic+meta->E2"),
            ]:
                values = [row.get(key, float("nan")) for row in history]
                plt.plot(epochs, values, label=label)
            plt.xlabel("purifier epoch")
            plt.ylabel("validation loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "purifier_validation_components.png"), dpi=180)
            plt.close()

            plt.figure(figsize=(8, 4.5))
            plt.plot(epochs, [row.get("val_meta_std_mean", float("nan")) for row in history], label="meta std")
            plt.plot(epochs, [row.get("val_semantic_std_mean", float("nan")) for row in history], label="semantic std")
            plt.xlabel("purifier epoch")
            plt.ylabel("mean latent std")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "purifier_code_std.png"), dpi=180)
            plt.close()

        if getattr(args, "extended_diagnostic_plots", False) and any("val_meta_input_l2_ratio" in row for row in history):
            plt.figure(figsize=(8, 4.5))
            plt.plot(epochs, [row.get("train_meta_input_l2_ratio", float("nan")) for row in history], label="train")
            plt.plot(epochs, [row.get("val_meta_input_l2_ratio", float("nan")) for row in history], label="validation")
            plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            plt.xlabel("purifier epoch")
            plt.ylabel("||meta input|| / ||E2||")
            plt.title("Semantic Residual Magnitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "purifier_semantic_residual_ratio.png"), dpi=180)
            plt.close()

        if any("purifier_pred_weight" in row for row in history):
            fig, ax_weight = plt.subplots(figsize=(3.5, 2.8))
            pred_weights = [row.get("purifier_pred_weight", float("nan")) for row in history]
            scheduled_weights = [row.get("purifier_pred_weight_scheduled", value) for row, value in zip(history, pred_weights)]
            target_counts = [row.get("purifier_dynamic_target_count", float("nan")) for row in history]
            ax_weight.plot(epochs, scheduled_weights, color=COLORS["blue"], linestyle="--", alpha=0.55, label="Scheduled weight")
            ax_weight.plot(epochs, pred_weights, color=COLORS["blue"], label="Effective weight")
            ax_weight.set_xlabel("Purifier epoch")
            ax_weight.set_ylabel("Prediction loss weight", color=COLORS["blue"])
            ax_weight.tick_params(axis="y", labelcolor=COLORS["blue"])
            ax_targets = ax_weight.twinx()
            ax_targets.plot(epochs, target_counts, color=COLORS["vermillion"], label="Selected targets")
            ax_targets.set_ylabel("Selected target count", color=COLORS["vermillion"])
            ax_targets.tick_params(axis="y", labelcolor=COLORS["vermillion"])
            ax_weight.legend(loc="upper left")
            style_axis(ax_weight)
            fig.tight_layout()
            save_figure(fig, os.path.join(plot_dir, "purifier_schedule_and_targets"))
            plt.close(fig)

    leakage_names = ["meta_to_z1_mlp", "meta_to_prev_mlp", "semantic_to_z1_mlp", "semantic_to_prev_mlp"]
    leakage_labels = ["meta->Z1", "meta->prev", "semantic->Z1", "semantic->prev"]
    leakage_r2 = [_plot_float(probe_metrics.get(name, {}).get("r2_debiased")) for name in leakage_names]
    if any(math.isfinite(value) for value in leakage_r2):
        figure, axis = plt.subplots(figsize=(3.5, 2.8))
        colors = [COLORS["vermillion"], COLORS["vermillion"], COLORS["green"], COLORS["green"]]
        bars = axis.bar(leakage_labels, leakage_r2, color=colors, alpha=0.82, width=0.68, zorder=3)
        axis.axhline(0, color=COLORS["black"], linewidth=0.8)
        axis.set_ylabel("Debiased $R^2$")
        axis.set_title("Nonlinear semantic leakage probes", loc="left")
        for bar, value in zip(bars, leakage_r2):
            if math.isfinite(value):
                axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        style_axis(axis)
        figure.tight_layout()
        save_figure(figure, os.path.join(plot_dir, "purifier_leakage_probe_r2"))
        plt.close(figure)

    meta_metrics = probe_metrics.get("purified_meta_to_i_mlp", probe_metrics.get("purified_meta_to_i_linear", {}))
    info_bits = _plot_float(meta_metrics.get("info_gain_bits_per_label_vs_prior"))
    ce = _plot_float(meta_metrics.get("ce_nats_per_label"))
    prior_ce = _plot_float(meta_metrics.get("prior_ce_nats_per_label"))
    if getattr(args, "extended_diagnostic_plots", False) and (math.isfinite(info_bits) or (math.isfinite(ce) and math.isfinite(prior_ce))):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].bar(["prior CE", "meta CE"], [prior_ce, ce], color=["tab:gray", "tab:blue"], alpha=0.82)
        axes[0].set_ylabel("nats / label")
        axes[0].set_title("Meta Prediction CE")
        axes[1].bar(["info gain"], [info_bits], color="tab:blue", alpha=0.82)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("bits / label")
        axes[1].set_title("Purified Meta -> i")
        if math.isfinite(info_bits):
            axes[1].text(0, info_bits, f"{info_bits:.3f}", ha="center", va="bottom" if info_bits >= 0 else "top", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "purifier_meta_prediction_information.png"), dpi=180)
        plt.close(fig)

    if purifier_information:
        total_bits = _plot_float(purifier_information.get("total_meta_info_bits_per_label"))
        semantic_bits = _plot_float(purifier_information.get("best_semantic_control_info_bits_per_label"))
        residual_bits = _plot_float(purifier_information.get("residual_meta_info_bits_per_label"))
        leakage_fraction = _plot_float(purifier_information.get("leakage_explained_fraction"))
        residual_fraction = _plot_float(purifier_information.get("residual_fraction"))

        def finite_last(history_key: str) -> float:
            for row in reversed(history):
                value = _plot_float(row.get(history_key))
                if math.isfinite(value):
                    return value
            return float("nan")

        def safe_fraction(numerator: float, denominator: float) -> float:
            if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-8:
                return float("nan")
            return numerator / denominator

        def csv_residual_fraction(path: str, weighted_positive: bool) -> float:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return float("nan")
            rows = []
            with open(path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    residual = _plot_float(row.get("residual_gap_bits"))
                    total = _plot_float(row.get("meta_info_bits_vs_prior"))
                    weight = _plot_float(row.get("training_weight"))
                    if not math.isfinite(weight):
                        weight = 1.0
                    if math.isfinite(residual) and math.isfinite(total):
                        rows.append((residual, total, weight))
            if not rows:
                return float("nan")
            if weighted_positive:
                numerator = sum(max(residual, 0.0) * weight for residual, _total, weight in rows)
                denominator = sum(max(total, 0.0) * weight for _residual, total, weight in rows)
                return safe_fraction(numerator, denominator)
            fractions = [safe_fraction(residual, total) for residual, total, _weight in rows if total > 1e-8]
            fractions = [value for value in fractions if math.isfinite(value)]
            return sum(fractions) / len(fractions) if fractions else float("nan")

        per_target_csv = os.path.join(purifier_dir, "purifier_per_target_information.csv")
        full_pool_csv = os.path.join(purifier_dir, "purifier_full_training_pool_information.csv")
        fraction_rows = [
            (
                "train claim\npositive-sum",
                finite_last("purifier_dynamic_checkpoint_claim_residual_fraction"),
                "from dynamic refresh: sum positive gap / sum positive meta gain over claim targets",
            ),
            (
                "train full\npositive-sum",
                finite_last("purifier_dynamic_full_residual_fraction"),
                "from dynamic refresh: sum positive gap / sum positive meta gain over full training targets",
            ),
            (
                "confirm selected\nglobal",
                residual_fraction,
                "from final probe: (meta info - best semantic control info) / meta info over selected target weights",
            ),
            (
                "confirm selected\npositive-sum",
                csv_residual_fraction(per_target_csv, weighted_positive=True),
                "from final per-target CSV: sum positive residual gap / sum positive meta gain",
            ),
            (
                "confirm full\npositive-sum",
                csv_residual_fraction(full_pool_csv, weighted_positive=True),
                "from final full-pool CSV: sum positive residual gap / sum positive meta gain",
            ),
        ]
        fraction_labels = [label for label, value, _note in fraction_rows if math.isfinite(value)]
        fraction_values = [value for _label, value, _note in fraction_rows if math.isfinite(value)]

        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.75))
        axes[0].bar(
            ["total meta", "semantic-related", "residual meta"],
            [total_bits, semantic_bits, residual_bits],
            color=[COLORS["blue"], COLORS["orange"], COLORS["green"]],
            alpha=0.82,
        )
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_ylabel("Bits per selected label")
        axes[0].set_title("Information decomposition", loc="left")
        axes[1].bar(
            ["semantic-related", "residual meta"],
            [leakage_fraction, residual_fraction],
            color=[COLORS["orange"], COLORS["green"]],
            alpha=0.82,
        )
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Fraction of total meta information")
        axes[1].set_title("Confirmatory fractions", loc="left")
        if fraction_values:
            colors = [COLORS["purple"], COLORS["sky"], COLORS["green"], COLORS["orange"], COLORS["blue"]][: len(fraction_values)]
            axes[2].bar(fraction_labels, fraction_values, color=colors, alpha=0.82)
            axes[2].axhline(0, color="black", linewidth=0.8)
            axes[2].axhline(1, color="gray", linestyle="--", linewidth=0.8)
            axes[2].set_ylabel("Residual fraction")
            axes[2].set_title("Measurement sensitivity", loc="left")
        else:
            axes[2].axis("off")
        for index, ax in enumerate(axes):
            for tick in ax.get_xticklabels():
                tick.set_rotation(25)
                tick.set_ha("right")
            style_axis(ax)
            panel_label(ax, chr(ord("a") + index))
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, "purifier_meta_semantic_residual_contributions"))
        plt.close(fig)

        notes_path = os.path.join(plot_dir, "purifier_meta_semantic_residual_contributions_notes.json")
        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "global_confirmatory": {
                        "total_meta_info_bits_per_label": total_bits,
                        "best_semantic_control_info_bits_per_label": semantic_bits,
                        "residual_meta_info_bits_per_label": residual_bits,
                        "leakage_explained_fraction": leakage_fraction,
                        "residual_fraction": residual_fraction,
                        "definition": "(meta info gain - best semantic-control info gain) / meta info gain, using the final confirmatory probe and final selected target weights.",
                    },
                    "residual_fraction_by_measurement": [
                        {"label": label, "value": value, "definition": note}
                        for label, value, note in fraction_rows
                    ],
                    "caution": "Training dynamic fractions, final confirmatory fractions, and posthoc clustering fractions are computed with different probes/splits unless explicitly aligned.",
                },
                f,
                indent=2,
            )

    if getattr(args, "extended_diagnostic_plots", False) and "meta" in val_codes and "semantic" in val_codes:
        meta_norm = val_codes["meta"].float().norm(dim=1) / math.sqrt(max(val_codes["meta"].size(1), 1))
        semantic_norm = val_codes["semantic"].float().norm(dim=1) / math.sqrt(max(val_codes["semantic"].size(1), 1))
        plt.figure(figsize=(8, 4.8))
        plt.hist(semantic_norm.numpy(), bins=40, alpha=0.55, label="semantic", density=True)
        plt.hist(meta_norm.numpy(), bins=40, alpha=0.55, label="meta", density=True)
        plt.xlabel("L2 norm / sqrt(dim)")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "purifier_code_norm_distribution.png"), dpi=180)
        plt.close()

    if getattr(args, "extended_diagnostic_plots", False) and is_binary_eval_mode(args.target_mode) and val_meta_logits.numel() and val_target.numel():
        selected = None
        if feature_weights is not None:
            selected = torch.nonzero(feature_weights.float().flatten() > 1e-8, as_tuple=False).flatten()
        if selected is not None and selected.numel() > 0:
            selected = selected[: max(1, args.binary_plot_features)]
            plot_tensors = {"x_i": val_target[:, selected], "x_i_hat": val_meta_logits[:, selected]}
            prefix = "purified_meta_selected_binary"
        else:
            plot_tensors = {"x_i": val_target, "x_i_hat": val_meta_logits}
            prefix = "purified_meta_binary"
        save_binary_activation_comparison(
            plot_tensors,
            plot_dir,
            args.binary_pred_threshold,
            args.binary_plot_samples,
            args.binary_plot_features,
            prefix=prefix,
        )


def run_e2_recursive_purifier(
    model: Decoupler,
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
    binary_eval_meta: Optional[Dict[str, torch.Tensor]],
    semantic_residual_state: Optional[Dict[str, List[torch.Tensor]]],
    feature_weights: Optional[torch.Tensor],
    test_ds: Optional[torch.utils.data.Dataset] = None,
) -> Optional[Dict]:
    if not args.run_e2_recursive_purifier:
        return None
    purifier_dir = os.path.join(args.output_dir, "e2_recursive_purifier")
    os.makedirs(purifier_dir, exist_ok=True)
    batch_size = args.purifier_batch_size if args.purifier_batch_size > 0 else args.batch_size
    legacy_code_dim = args.purifier_code_dim if args.purifier_code_dim > 0 else args.latent_dim
    semantic_dim = args.purifier_semantic_dim if args.purifier_semantic_dim > 0 else legacy_code_dim
    if args.purifier_meta_dim > 0:
        meta_dim = args.purifier_meta_dim
    else:
        # Keep the semantic branch roomy, but make meta a bottleneck by default.
        # This reduces duplicate semantic copying without adding an adversarial
        # objective that may merely hide recoverable information.
        meta_dim = min(legacy_code_dim, max(8, legacy_code_dim // 4)) if legacy_code_dim >= 32 else legacy_code_dim
    hidden_dim = args.purifier_hidden_dim if args.purifier_hidden_dim > 0 else args.hidden_dim
    lr = args.purifier_lr if args.purifier_lr > 0 else args.lr

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    train_tensors = collect_probe_tensors(
        model,
        train_loader,
        device,
        args.purifier_max_train_samples,
        "Collect purifier train tensors",
        args.target_mode,
        binary_eval_meta,
        semantic_residual_state,
        args.main_prediction_target,
    )
    val_tensors = collect_probe_tensors(
        model,
        val_loader,
        device,
        args.purifier_max_val_samples,
        "Collect purifier val tensors",
        args.target_mode,
        binary_eval_meta,
        semantic_residual_state,
        args.main_prediction_target,
    )
    train_dataset = TensorDataset(train_tensors["z2"], train_tensors["z1"], train_tensors["x_prev"], train_tensors["x_i_train"], train_tensors["x_i"])
    val_dataset = TensorDataset(val_tensors["z2"], val_tensors["z1"], val_tensors["x_prev"], val_tensors["x_i_train"], val_tensors["x_i"])
    purifier_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    purifier_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    purifier = E2RecursivePurifier(
        train_tensors["z2"].size(1),
        semantic_dim,
        meta_dim,
        hidden_dim,
        args.dropout,
        train_tensors["z1"].size(1),
        train_tensors["x_prev"].size(1),
        train_tensors["x_i_train"].size(1),
        args.purifier_meta_input_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(purifier.parameters(), lr=lr, weight_decay=args.weight_decay)
    inherited_feature_weights = feature_weights.detach().cpu().clone() if feature_weights is not None else None
    purifier_feature_weights = inherited_feature_weights.clone() if inherited_feature_weights is not None else None
    initial_selected_count = (
        int((purifier_feature_weights > args.dynamic_target_export_threshold).sum().item())
        if purifier_feature_weights is not None
        else int(train_tensors["x_i_train"].size(1))
    )
    purifier_dynamic_summary = {
        "enabled": bool(args.purifier_dynamic_target_weighting),
        "selected_count": initial_selected_count,
        "weight_sum": float(purifier_feature_weights.float().sum().item()) if purifier_feature_weights is not None else float(train_tensors["x_i_train"].size(1)),
        "score_mean_selected": float("nan"),
        "e2_gain_nats_mean_selected": float("nan"),
        "semantic_gain_nats_mean_selected": float("nan"),
        "semantic_gap_ci_low_mean_selected": float("nan"),
        "dynamic_residual_fraction_mean_selected": float("nan"),
        "dynamic_leakage_explained_fraction_mean_selected": float("nan"),
    }
    last_purifier_dynamic_refresh = 0

    history = []
    best_score = math.inf
    best_checkpoint_selection: Dict[str, Any] = {
        "mode": args.purifier_final_checkpoint,
        "best_epoch": None,
        "best_score": None,
        "best_objective": None,
        "best_total_positive_gap": None,
        "best_total_positive_gap_epoch": None,
        "best_residual_fraction": None,
        "best_residual_fraction_epoch": None,
        "best_claim_count": None,
        "best_full_weight_count": None,
    }
    purifier_progress = (
        _tqdm(
            range(1, args.purifier_epochs + 1),
            desc="E2 recursive purifier",
            unit="epoch",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        if compact_progress_enabled()
        else tqdm(range(1, args.purifier_epochs + 1), desc="E2 recursive purifier", leave=True)
    )
    for epoch in purifier_progress:
        semantic_encoder_end = args.purifier_semantic_warmup_epochs
        semantic_z2_end = semantic_encoder_end + args.purifier_semantic_z2_warmup_epochs
        if epoch <= semantic_encoder_end:
            purifier_stage = "semantic_encoder"
        elif epoch <= semantic_z2_end:
            purifier_stage = "semantic_z2_decoder"
        else:
            purifier_stage = "residual_meta"
        semantic_warmup = purifier_stage != "residual_meta"
        configure_purifier_training_stage(
            purifier,
            purifier_stage,
            args.purifier_freeze_semantic_after_warmup,
        )
        purifier_pred_weight_scheduled = ramp_weight(
            args.lambda_purifier_pred,
            epoch,
            args.purifier_pred_start_epoch,
            args.purifier_pred_ramp_epochs,
        )
        purifier_pred_weight = purifier_pred_weight_scheduled
        if semantic_warmup:
            purifier_pred_weight = 0.0
        should_refresh_purifier_targets = (
            args.purifier_dynamic_target_weighting
            and is_binary_eval_mode(args.target_mode)
            and not semantic_warmup
            and epoch >= args.purifier_dynamic_start_epoch
            and (
                last_purifier_dynamic_refresh == 0
                or epoch - last_purifier_dynamic_refresh >= args.purifier_dynamic_refresh_epochs
            )
        )
        if should_refresh_purifier_targets:
            if args.purifier_save_refresh_checkpoints:
                torch.save(
                    purifier.state_dict(),
                    os.path.join(purifier_dir, f"purifier_pre_refresh_epoch_{epoch:03d}.pt"),
                )
            purifier_feature_weights, purifier_dynamic_tensors, purifier_dynamic_summary = compute_purifier_dynamic_target_weights(
                purifier,
                train_tensors,
                val_tensors,
                args,
                device,
                batch_size,
                purifier_feature_weights,
                inherited_feature_weights,
                epoch,
                binary_eval_meta,
            )
            save_dynamic_target_update(purifier_dir, purifier_dynamic_tensors, purifier_dynamic_summary)
            last_purifier_dynamic_refresh = epoch
            if not compact_progress_enabled():
                print(json.dumps({"purifier_dynamic_target_update": purifier_dynamic_summary}, ensure_ascii=False))
        purifier_pred_gate_open = True
        if (
            args.purifier_gate_prediction_on_strict_targets
            and args.purifier_dynamic_target_weighting
            and last_purifier_dynamic_refresh > 0
            and int(purifier_dynamic_summary.get("strict_eligible_count", 0)) <= 0
        ):
            purifier_pred_gate_open = False
            purifier_pred_weight = 0.0
        feature_weights_device = purifier_feature_weights.to(device) if purifier_feature_weights is not None else None
        purifier.train()
        if purifier_stage == "semantic_z2_decoder":
            purifier.semantic_encoder.eval()
            purifier.semantic_to_z1.eval()
            purifier.semantic_to_prev.eval()
        elif not semantic_warmup and args.purifier_freeze_semantic_after_warmup:
            purifier.semantic_encoder.eval()
            purifier.semantic_to_z2.eval()
            purifier.semantic_to_z1.eval()
            purifier.semantic_to_prev.eval()
        totals = {"loss": 0.0, "recon_z2": 0.0, "semantic_recon_z2": 0.0, "meta_residual": 0.0, "semantic_z1": 0.0, "semantic_prev": 0.0, "pred": 0.0, "orth": 0.0, "meta_sem_cov": 0.0, "var": 0.0, "meta_input_l2_ratio": 0.0}
        seen = 0
        for z2, z1, x_prev, x_i_train, _ in tqdm(purifier_train_loader, desc=f"Purifier epoch {epoch}", leave=False):
            z2 = z2.to(device)
            z1 = z1.to(device)
            x_prev = x_prev.to(device)
            x_i_train = x_i_train.to(device)
            out = purifier(z2)
            loss_recon = F.mse_loss(out["z2_hat"], z2)
            loss_semantic_recon = F.mse_loss(out["semantic_z2_hat"], z2)
            loss_meta_residual = F.mse_loss(out["meta_residual_hat"], out["meta_input"].detach())
            loss_z1 = F.mse_loss(out["z1_hat"], z1)
            loss_prev = F.mse_loss(out["prev_hat"], x_prev)
            loss_pred = weighted_prediction_loss(out["i_hat"], x_i_train, args.target_mode, feature_weights_device)
            loss_orth = orthogonality_loss(out["semantic"], out["meta"])
            loss_cov = cross_cov_loss(out["meta"], z1) + cross_cov_loss(out["meta"], x_prev)
            if purifier_stage == "semantic_encoder":
                loss_var = variance_floor_loss(out["semantic"])
            elif purifier_stage == "semantic_z2_decoder":
                loss_var = out["semantic"].new_tensor(0.0)
            else:
                loss_var = variance_floor_loss(out["meta"])
            semantic_loss_weight = args.lambda_purifier_semantic if purifier_stage == "semantic_encoder" else 0.0
            semantic_recon_weight = args.lambda_purifier_semantic_recon_z2 if purifier_stage == "semantic_z2_decoder" else 0.0
            residual_stage_weight = 1.0 if purifier_stage == "residual_meta" else 0.0
            loss = (
                residual_stage_weight * args.lambda_purifier_recon_z2 * loss_recon
                + semantic_recon_weight * loss_semantic_recon
                + residual_stage_weight * args.lambda_purifier_meta_residual * loss_meta_residual
                + semantic_loss_weight * (loss_z1 + loss_prev)
                + purifier_pred_weight * loss_pred
                + residual_stage_weight * args.lambda_purifier_orth * loss_orth
                + residual_stage_weight * args.lambda_purifier_meta_sem_cov * loss_cov
                + args.lambda_purifier_var * loss_var
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(purifier.parameters(), 1.0)
            optimizer.step()
            bs = z2.size(0)
            totals["loss"] += loss.item() * bs
            totals["recon_z2"] += loss_recon.item() * bs
            totals["semantic_recon_z2"] += loss_semantic_recon.item() * bs
            totals["meta_residual"] += loss_meta_residual.item() * bs
            totals["semantic_z1"] += loss_z1.item() * bs
            totals["semantic_prev"] += loss_prev.item() * bs
            totals["pred"] += loss_pred.item() * bs
            totals["orth"] += loss_orth.item() * bs
            totals["meta_sem_cov"] += loss_cov.item() * bs
            totals["var"] += loss_var.item() * bs
            residual_ratio = out["meta_input"].float().norm(dim=1) / z2.float().norm(dim=1).clamp_min(1e-8)
            totals["meta_input_l2_ratio"] += residual_ratio.mean().item() * bs
            seen += bs
        val = evaluate_e2_recursive_purifier(
            purifier,
            purifier_val_loader,
            device,
            args,
            purifier_feature_weights,
            binary_eval_meta,
            purifier_pred_weight,
            purifier_stage,
        )
        row = {
            "epoch": epoch,
            "purifier_stage": purifier_stage,
            "purifier_pred_weight_scheduled": purifier_pred_weight_scheduled,
            "purifier_pred_weight": purifier_pred_weight,
            "purifier_pred_gate_open": purifier_pred_gate_open,
            "train_loss": totals["loss"] / max(seen, 1),
            "train_recon_z2": totals["recon_z2"] / max(seen, 1),
            "train_semantic_recon_z2": totals["semantic_recon_z2"] / max(seen, 1),
            "train_meta_residual": totals["meta_residual"] / max(seen, 1),
            "train_semantic_z1": totals["semantic_z1"] / max(seen, 1),
            "train_semantic_prev": totals["semantic_prev"] / max(seen, 1),
            "train_pred": totals["pred"] / max(seen, 1),
            "train_orth": totals["orth"] / max(seen, 1),
            "train_meta_sem_cov": totals["meta_sem_cov"] / max(seen, 1),
            "train_var": totals["var"] / max(seen, 1),
            "train_meta_input_l2_ratio": totals["meta_input_l2_ratio"] / max(seen, 1),
            **{f"val_{key}": value for key, value in val.items()},
            "purifier_dynamic_target_count": int(purifier_dynamic_summary.get("selected_count", 0)),
            "purifier_dynamic_target_weight_sum": float(purifier_dynamic_summary.get("weight_sum", float("nan"))),
            "purifier_dynamic_score_mean_selected": float(purifier_dynamic_summary.get("score_mean_selected", float("nan"))),
            "purifier_dynamic_e2_gain_mean_selected": float(purifier_dynamic_summary.get("e2_gain_nats_mean_selected", float("nan"))),
            "purifier_dynamic_semantic_gain_mean_selected": float(purifier_dynamic_summary.get("semantic_gain_nats_mean_selected", float("nan"))),
            "purifier_dynamic_semantic_gap_ci_low_mean_selected": float(purifier_dynamic_summary.get("semantic_gap_ci_low_mean_selected", float("nan"))),
            "purifier_dynamic_residual_fraction_mean_selected": float(purifier_dynamic_summary.get("dynamic_residual_fraction_mean_selected", float("nan"))),
            "purifier_dynamic_leakage_explained_fraction_mean_selected": float(purifier_dynamic_summary.get("dynamic_leakage_explained_fraction_mean_selected", float("nan"))),
            "purifier_dynamic_claim_residual_fraction": float(purifier_dynamic_summary.get("claim_residual_fraction", float("nan"))),
            "purifier_dynamic_full_residual_fraction": float(purifier_dynamic_summary.get("full_residual_fraction", float("nan"))),
            "purifier_dynamic_strict_residual_fraction": float(purifier_dynamic_summary.get("strict_residual_fraction", float("nan"))),
        }
        score = (
            args.lambda_purifier_recon_z2 * row["val_recon_z2"]
            + args.lambda_purifier_semantic_recon_z2 * row["val_semantic_recon_z2"]
            + args.lambda_purifier_meta_residual * row["val_meta_residual"]
            + args.lambda_purifier_semantic * (row["val_semantic_z1"] + row["val_semantic_prev"])
            + args.lambda_purifier_pred * row["val_pred"]
            + args.lambda_purifier_orth * row["val_orth"]
            + args.lambda_purifier_meta_sem_cov * row["val_meta_sem_cov"]
            + args.lambda_purifier_var * row["val_var"]
        )
        row["selection_score_final_objective"] = score
        claim_count = int(purifier_dynamic_summary.get("claim_target_count", purifier_dynamic_summary.get("strict_eligible_count", 0)))
        full_count = int(purifier_dynamic_summary.get("full_weight_count", purifier_dynamic_summary.get("selected_count", 0)))
        total_positive_gap = float(
            purifier_dynamic_summary.get(
                "claim_positive_gap_sum",
                purifier_dynamic_summary.get("strict_positive_gap_sum", 0.0),
            )
        )
        if not math.isfinite(total_positive_gap):
            total_positive_gap = 0.0
        claim_residual_fraction = float(purifier_dynamic_summary.get("claim_residual_fraction", 0.0))
        if not math.isfinite(claim_residual_fraction):
            claim_residual_fraction = 0.0
        if args.purifier_final_checkpoint == "objective":
            checkpoint_score = float(score)
        elif args.purifier_final_checkpoint == "target_count":
            checkpoint_score = -float(claim_count) - 1e-6 * total_positive_gap
        elif args.purifier_final_checkpoint == "total_positive_gap":
            checkpoint_score = -total_positive_gap
        elif args.purifier_final_checkpoint == "residual_fraction":
            checkpoint_score = -claim_residual_fraction - 1e-6 * total_positive_gap - 1e-9 * float(claim_count)
        elif args.purifier_final_checkpoint == "balanced":
            if claim_count < args.purifier_balanced_min_claim_count:
                checkpoint_score = float("inf")
            else:
                claim_top_k = max(1, int(getattr(args, "purifier_dynamic_claim_top_k", 0) or args.purifier_dynamic_top_k or claim_count))
                count_term = math.log1p(float(claim_count)) / math.log1p(float(max(claim_top_k, 1)))
                gap_term = math.log1p(max(0.0, float(total_positive_gap)))
                frac_term = max(0.0, float(claim_residual_fraction))
                balanced_score = (
                    args.purifier_balanced_gap_weight * gap_term
                    + args.purifier_balanced_fraction_weight * frac_term
                    + args.purifier_balanced_count_weight * count_term
                )
                checkpoint_score = -balanced_score
        else:
            raise ValueError(args.purifier_final_checkpoint)
        row["checkpoint_selection_mode"] = args.purifier_final_checkpoint
        row["checkpoint_selection_score"] = checkpoint_score
        row["purifier_dynamic_claim_target_count"] = claim_count
        row["purifier_dynamic_full_weight_count"] = full_count
        row["purifier_dynamic_claim_positive_gap_sum"] = total_positive_gap
        row["purifier_dynamic_checkpoint_residual_fraction"] = claim_residual_fraction
        history.append(row)
        if compact_progress_enabled():
            purifier_progress.set_postfix(
                stage=purifier_stage,
                recon=f"{row['val_recon_z2']:.4f}",
                sem=f"{row['val_semantic_z1'] + row['val_semantic_prev']:.4f}",
                targets=int(purifier_dynamic_summary.get("selected_count", 0)),
                claim=claim_count,
                gap=f"{float(purifier_dynamic_summary.get('score_mean_selected', float('nan'))):.4g}",
                frac=f"{claim_residual_fraction:.3g}",
                refresh=False,
            )
        else:
            print(json.dumps({"e2_recursive_purifier": row}, ensure_ascii=False))
        checkpoint_eligible = not semantic_warmup
        if checkpoint_eligible and checkpoint_score < best_score:
            best_score = checkpoint_score
            best_checkpoint_selection = {
                "mode": args.purifier_final_checkpoint,
                "best_epoch": int(epoch),
                "best_score": float(checkpoint_score),
                "best_objective": float(score),
                "best_total_positive_gap": float(total_positive_gap),
                "best_total_positive_gap_epoch": int(epoch),
                "best_residual_fraction": float(claim_residual_fraction),
                "best_residual_fraction_epoch": int(epoch),
                "best_claim_count": int(claim_count),
                "best_full_weight_count": int(full_count),
                "dynamic_summary": dict(purifier_dynamic_summary),
            }
            torch.save(purifier.state_dict(), os.path.join(purifier_dir, "best_purifier.pt"))
            torch.save(purifier.state_dict(), os.path.join(purifier_dir, f"best_{args.purifier_final_checkpoint}_purifier.pt"))
            torch.save(
                {
                    "feature_weights": purifier_feature_weights,
                    "dynamic_summary": purifier_dynamic_summary,
                    "epoch": epoch,
                    "selection_score": checkpoint_score,
                    "objective_score": score,
                    "checkpoint_selection": best_checkpoint_selection,
                },
                os.path.join(purifier_dir, "best_target_state.pt"),
            )
        if semantic_z2_end > 0 and epoch == semantic_z2_end:
            torch.save(purifier.state_dict(), os.path.join(purifier_dir, "semantic_warmup_purifier.pt"))
        if should_refresh_purifier_targets and args.purifier_save_refresh_checkpoints:
            torch.save(
                purifier.state_dict(),
                os.path.join(purifier_dir, f"purifier_post_refresh_epoch_{epoch:03d}.pt"),
            )
    write_csv(os.path.join(purifier_dir, "history.csv"), history)
    with open(os.path.join(purifier_dir, "history.jsonl"), "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    best_path = os.path.join(purifier_dir, "best_purifier.pt")
    if os.path.exists(best_path):
        purifier.load_state_dict(torch.load(best_path, map_location=device))
    best_target_state_path = os.path.join(purifier_dir, "best_target_state.pt")
    if os.path.exists(best_target_state_path):
        best_target_state = torch.load(best_target_state_path, map_location="cpu")
        purifier_feature_weights = best_target_state.get("feature_weights")
        purifier_dynamic_summary = best_target_state.get("dynamic_summary", purifier_dynamic_summary)
    torch.save(purifier.state_dict(), os.path.join(purifier_dir, "last_purifier.pt"))
    train_codes = collect_e2_purifier_codes(purifier, train_tensors, device, batch_size)
    val_codes = collect_e2_purifier_codes(purifier, val_tensors, device, batch_size)

    probe_epochs = args.purifier_probe_epochs if args.purifier_probe_epochs > 0 else args.probe_epochs
    probe_hidden = args.purifier_probe_hidden_dim if args.purifier_probe_hidden_dim > 0 else hidden_dim
    probe_metrics = {}
    meta_recovered_train: Dict[str, torch.Tensor] = {}
    meta_recovered_val: Dict[str, torch.Tensor] = {}
    for code_name in ["meta", "semantic"]:
        for target_name, train_target, val_target in [("z1", train_tensors["z1"], val_tensors["z1"]), ("prev", train_tensors["x_prev"], val_tensors["x_prev"])]:
            metrics, train_pred, val_pred = train_regression_mlp_probe_with_train_pred(
                train_codes[code_name],
                train_target,
                val_codes[code_name],
                val_target,
                device,
                probe_epochs,
                args.probe_batch_size,
                args.probe_lr,
                args.probe_weight_decay,
                probe_hidden,
                args.dropout,
                f"{code_name}_to_{target_name}_mlp",
                purifier_dir,
            )
            probe_metrics[f"{code_name}_to_{target_name}_mlp"] = metrics
            if code_name == "meta":
                meta_recovered_train[target_name] = train_pred
                meta_recovered_val[target_name] = val_pred
    meta_i_metrics, meta_i_logits = train_binary_linear_probe(
        train_codes["meta"],
        train_tensors["x_i"],
        val_codes["meta"],
        val_tensors["x_i"],
        device,
        probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        args.binary_pred_threshold,
        "purified_meta_to_i_linear",
        purifier_dir,
        purifier_feature_weights,
    )
    probe_metrics["purified_meta_to_i_linear"] = meta_i_metrics

    purifier_semantic_control_metrics: Dict[str, Dict] = {}
    purifier_semantic_control_logits: Dict[str, torch.Tensor] = {}
    purifier_information: Optional[Dict] = None
    if is_binary_eval_mode(args.target_mode):
        semantic_specs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        if "z1" in meta_recovered_train:
            semantic_specs.append(("purified_meta_recovered_z1_to_i", meta_recovered_train["z1"], meta_recovered_val["z1"]))
        if "prev" in meta_recovered_train:
            semantic_specs.append(("purified_meta_recovered_prev_to_i", meta_recovered_train["prev"], meta_recovered_val["prev"]))
        if "z1" in meta_recovered_train and "prev" in meta_recovered_train:
            semantic_specs.append(
                (
                    "purified_meta_recovered_z1_prev_to_i",
                    torch.cat([meta_recovered_train["z1"], meta_recovered_train["prev"]], dim=1),
                    torch.cat([meta_recovered_val["z1"], meta_recovered_val["prev"]], dim=1),
                )
            )
        for name, train_x, val_x in semantic_specs:
            metrics, logits = train_binary_linear_probe(
                train_x,
                train_tensors["x_i"],
                val_x,
                val_tensors["x_i"],
                device,
                probe_epochs,
                args.probe_batch_size,
                args.probe_lr,
                args.probe_weight_decay,
                args.binary_pred_threshold,
                name,
                purifier_dir,
                purifier_feature_weights,
            )
            metrics["control_type"] = "semantic_recovered_from_purified_meta"
            purifier_semantic_control_metrics[name] = metrics
            purifier_semantic_control_logits[name] = logits
        if purifier_semantic_control_metrics:
            best_name, best_metrics = min(
                purifier_semantic_control_metrics.items(),
                key=lambda item: item[1].get("ce_nats_per_label", float("inf")),
            )
            total_bits = float(meta_i_metrics.get("info_gain_bits_per_label_vs_prior", float("nan")))
            semantic_bits = float(best_metrics.get("info_gain_bits_per_label_vs_prior", float("nan")))
            residual_bits = total_bits - semantic_bits
            if math.isfinite(total_bits) and abs(total_bits) > 1e-8:
                leakage_fraction = semantic_bits / total_bits
                residual_fraction = residual_bits / total_bits
            else:
                leakage_fraction = float("nan")
                residual_fraction = float("nan")
            best_test = paired_ce_gap_test(
                "purified_meta_to_i_linear",
                best_name,
                meta_i_logits,
                purifier_semantic_control_logits[best_name],
                val_tensors["x_i"],
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed + 9100,
                args.information_alpha,
                "purified_meta_better_than_best_recovered_semantic_control",
                "purifier_selected" if purifier_feature_weights is not None else "all",
                purifier_feature_weights,
            )
            purifier_information = {
                "prior_is_activation_rate_baseline": True,
                "feature_scope": "purifier_selected" if purifier_feature_weights is not None else "all",
                "feature_weight_sum": None if purifier_feature_weights is None else float(purifier_feature_weights.float().sum().item()),
                "total_meta_info_bits_per_label": total_bits,
                "best_semantic_control": best_name,
                "best_semantic_control_info_bits_per_label": semantic_bits,
                "residual_meta_info_bits_per_label": residual_bits,
                "leakage_explained_fraction": leakage_fraction,
                "residual_fraction": residual_fraction,
                "best_control_test": best_test,
                "interpretation": "total_meta_info is CE(prior activation rate)-CE(meta probe); semantic-related info is the best prediction obtainable from Z1/prev variables recovered from purified meta; residual_meta_info is their difference.",
            }

    summary = {
        "enabled": True,
        "z2_dim": int(train_tensors["z2"].size(1)),
        "z1_dim": int(train_tensors["z1"].size(1)),
        "prev_dim": int(train_tensors["x_prev"].size(1)),
        "target_dim": int(train_tensors["x_i_train"].size(1)),
        "legacy_code_dim": legacy_code_dim,
        "semantic_code_dim": semantic_dim,
        "meta_code_dim": meta_dim,
        "total_code_dim": semantic_dim + meta_dim,
        "meta_input_mode": purifier.meta_input_mode,
        "semantic_warmup_epochs": args.purifier_semantic_warmup_epochs,
        "semantic_z2_warmup_epochs": args.purifier_semantic_z2_warmup_epochs,
        "freeze_semantic_after_warmup": bool(args.purifier_freeze_semantic_after_warmup),
        "meta_bottleneck_ratio_vs_z2": meta_dim / max(int(train_tensors["z2"].size(1)), 1),
        "hidden_dim": hidden_dim,
        "epochs": args.purifier_epochs,
        "best_score": best_score,
        "checkpoint_selection": best_checkpoint_selection,
        "inherited_feature_weight_sum": None if inherited_feature_weights is None else float(inherited_feature_weights.float().sum().item()),
        "feature_weight_sum": None if purifier_feature_weights is None else float(purifier_feature_weights.float().sum().item()),
        "target_type": "neuron",
        "dynamic_targeting": {
            "enabled": bool(args.purifier_dynamic_target_weighting),
            "gate_prediction_on_strict_targets": bool(args.purifier_gate_prediction_on_strict_targets),
            "save_refresh_checkpoints": bool(args.purifier_save_refresh_checkpoints),
            "start_epoch": args.purifier_dynamic_start_epoch,
            "refresh_epochs": args.purifier_dynamic_refresh_epochs,
            "probe_epochs": args.purifier_dynamic_probe_epochs,
            "semantic_probe": args.purifier_dynamic_semantic_probe,
            "probe_hidden_dim": args.purifier_dynamic_probe_hidden_dim,
            "restrict_to_inherited_targets": bool(args.purifier_dynamic_restrict_to_inherited_targets),
            "allow_new_targets": not bool(args.purifier_dynamic_restrict_to_inherited_targets),
            "final_summary": purifier_dynamic_summary,
            "weights_file": "e2_recursive_purifier/dynamic_targets/target_weights_latest.pt" if args.purifier_dynamic_target_weighting else None,
        },
        "objective": {
            "reconstruct_z2_from_semantic_meta": args.lambda_purifier_recon_z2,
            "semantic_only_reconstructs_z2": args.lambda_purifier_semantic_recon_z2,
            "meta_reconstructs_semantic_residual": args.lambda_purifier_meta_residual,
            "semantic_code_reconstructs_z1_prev": args.lambda_purifier_semantic,
            "meta_code_predicts_neurons": args.lambda_purifier_pred,
            "prediction_start_epoch": args.purifier_pred_start_epoch,
            "prediction_ramp_epochs": args.purifier_pred_ramp_epochs,
            "semantic_meta_orthogonality": args.lambda_purifier_orth,
            "meta_z1_prev_covariance_penalty": args.lambda_purifier_meta_sem_cov,
            "variance_floor": args.lambda_purifier_var,
        },
        "last_epoch": history[-1] if history else None,
        "nonlinear_leakage_probes": probe_metrics,
        "semantic_control_prediction_from_meta": purifier_semantic_control_metrics,
        "information_decomposition": purifier_information,
        "interpretation": {
            "architecture": "The recursive purifier uses an asymmetric bottleneck: a larger semantic code is encouraged to absorb Z1/prev-recoverable information, while a smaller meta code keeps only information useful for selected neuron prediction.",
            "meta_to_z1_prev": "Lower R2/debiased R2 for meta_to_z1/prev means less semantic leakage remains in the purified meta code.",
            "semantic_to_z1_prev": "Higher R2 for semantic_to_z1/prev means the second-stage semantic code absorbed recoverable semantic information.",
            "meta_to_i": "Meta-to-i probe measures whether purified meta code still predicts neuron activation targets.",
            "information_decomposition": "The purifier-specific decomposition compares purified meta prediction against semantic controls recovered from purified meta, both measured as CE improvement over activation-rate prior.",
        },
        "outputs": {
            "history_csv": "e2_recursive_purifier/history.csv",
            "best_purifier": "e2_recursive_purifier/best_purifier.pt",
            "best_target_state": "e2_recursive_purifier/best_target_state.pt",
            "semantic_warmup_purifier": "e2_recursive_purifier/semantic_warmup_purifier.pt" if args.purifier_semantic_warmup_epochs + args.purifier_semantic_z2_warmup_epochs > 0 else None,
            "codes": "e2_recursive_purifier/purified_codes.pt",
            "summary": "e2_recursive_purifier/summary.json",
            "semantic_controls_csv": "e2_recursive_purifier/purifier_semantic_controls.csv",
            "information_decomposition_json": "e2_recursive_purifier/purifier_information_decomposition.json",
            "plots_dir": None if args.no_plots else "e2_recursive_purifier/plots",
        },
    }
    if not args.no_plots:
        try:
            save_e2_recursive_purifier_plots(
                purifier_dir,
                history,
                probe_metrics,
                purifier_information,
                val_codes,
                val_tensors["x_i"],
                meta_i_logits,
                purifier_feature_weights,
                args,
            )
        except Exception as exc:
            summary["outputs"]["plots_error"] = str(exc)
    torch.save(
        {
            "train_meta": train_codes["meta"],
            "train_semantic": train_codes["semantic"],
            "val_meta": val_codes["meta"],
            "val_semantic": val_codes["semantic"],
            "val_i_logits_from_meta_probe": meta_i_logits,
            "val_target": val_tensors["x_i"],
            "inherited_feature_weights": inherited_feature_weights,
            "feature_weights": purifier_feature_weights,
            "summary": summary,
        },
        os.path.join(purifier_dir, "purified_codes.pt"),
    )
    write_csv(os.path.join(purifier_dir, "probe_metrics.csv"), list(probe_metrics.values()))
    write_csv(os.path.join(purifier_dir, "purifier_semantic_controls.csv"), list(purifier_semantic_control_metrics.values()))
    with open(os.path.join(purifier_dir, "purifier_information_decomposition.json"), "w", encoding="utf-8") as f:
        json.dump(purifier_information or {}, f, indent=2)
    with open(os.path.join(purifier_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_main_checkpoint_purifier_ablation(
    model: Decoupler,
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
    binary_eval_meta: Optional[Dict[str, torch.Tensor]],
    representation_checkpoint_path: str,
    checkpoint_dir: str,
    restore_bundle: Dict[str, Any],
    test_ds: Optional[torch.utils.data.Dataset] = None,
) -> Dict[str, Any]:
    """Run the same purifier from several checkpoints on one first-stage trajectory."""
    root_output_dir = args.output_dir
    ablation_dir = os.path.join(root_output_dir, "main_checkpoint_ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    purifier_seed = (
        args.main_checkpoint_ablation_seed
        if args.main_checkpoint_ablation_seed >= 0
        else args.seed + 17000
    )

    checkpoint_specs: List[Tuple[str, str]] = []
    if os.path.isfile(representation_checkpoint_path):
        checkpoint_specs.append(("best_representation", representation_checkpoint_path))
    for epoch in sorted(set(args.main_checkpoint_ablation_epochs)):
        path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing requested checkpoint-ablation bundle: {path}")
        checkpoint_specs.append((f"epoch_{epoch:03d}", path))

    rows: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []
    seen_epochs = set()
    try:
        for source_kind, checkpoint_path in checkpoint_specs:
            bundle = torch.load(checkpoint_path, map_location=device)
            epoch = int(bundle.get("epoch", -1))
            if epoch in seen_epochs:
                continue
            seen_epochs.add(epoch)
            label = f"representation_epoch_{epoch:03d}" if source_kind == "best_representation" else source_kind
            run_dir = os.path.join(ablation_dir, label)
            summary_path = os.path.join(run_dir, "e2_recursive_purifier", "summary.json")
            os.makedirs(run_dir, exist_ok=True)

            model.load_state_dict(bundle["model_state_dict"])
            feature_weights = bundle.get("target_feature_weights")
            semantic_state = bundle.get("semantic_residual_state")
            args.output_dir = run_dir
            if os.path.isfile(summary_path) and not args.main_checkpoint_ablation_overwrite:
                with open(summary_path, "r", encoding="utf-8") as handle:
                    purifier_result = json.load(handle)
            else:
                torch.manual_seed(purifier_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(purifier_seed)
                    torch.cuda.empty_cache()
                purifier_result = run_e2_recursive_purifier(
                    model,
                    train_ds,
                    val_ds,
                    args,
                    device,
                    binary_eval_meta,
                    semantic_state if args.dynamic_predict_on_semantic_residual else None,
                    feature_weights if args.dynamic_target_mode == "neuron" else None,
                    test_ds,
                )
                if purifier_result is None:
                    raise RuntimeError("Checkpoint ablation requires --run-e2-recursive-purifier.")

            source_stats = bundle.get("stats", {}) or {}
            information = purifier_result.get("information_decomposition") or {}
            per_target = information.get("per_target_analysis") or {}
            full_pool = information.get("full_training_pool_diagnostic") or {}
            checkpoint_selection = purifier_result.get("checkpoint_selection") or {}
            final_dynamic = purifier_result.get("dynamic_targeting", {}).get("final_summary", {}) or {}
            best_test = information.get("best_control_test") or {}
            val_next = source_stats.get("val_next_mse")
            val_prev = source_stats.get("val_prev_mse")
            reconstruction_feasible = (
                val_next is not None
                and val_prev is not None
                and float(val_next) <= float(args.recon_gate_next)
                and float(val_prev) <= float(args.recon_gate_prev)
            )
            row = {
                "checkpoint": label,
                "source_kind": source_kind,
                "first_stage_epoch": epoch,
                "representation_score": bundle.get("representation_score"),
                "val_next_mse": val_next,
                "val_prev_mse": val_prev,
                "reconstruction_feasible": reconstruction_feasible,
                "val_e2_prev_cov": source_stats.get("val_e2_prev_cov"),
                "val_gamma_hsic_penalty": source_stats.get("val_gamma_hsic_penalty"),
                "first_stage_candidate_count": source_stats.get("dynamic_target_candidate_count"),
                "first_stage_confirmed_count": source_stats.get("dynamic_target_confirmed_count"),
                "first_stage_claim_count": source_stats.get("dynamic_target_claim_count"),
                "purifier_best_total_positive_gap": checkpoint_selection.get("best_total_positive_gap"),
                "purifier_best_total_positive_gap_epoch": checkpoint_selection.get("best_total_positive_gap_epoch"),
                "purifier_final_claim_count": final_dynamic.get("claim_target_count"),
                "total_meta_info_bits_per_label": information.get("total_meta_info_bits_per_label"),
                "semantic_control_info_bits_per_label": information.get("best_semantic_control_info_bits_per_label"),
                "residual_meta_info_bits_per_label": information.get("residual_meta_info_bits_per_label"),
                "residual_fraction": information.get("residual_fraction"),
                "gap_ci_low_bits": best_test.get("bootstrap_ci_low_bits"),
                "gap_p": best_test.get("signflip_one_sided_p"),
                "claim_positive_gap_count": per_target.get("positive_gap_count"),
                "claim_fdr_significant_count": per_target.get("bh_fdr_significant_count"),
                "full_pool_positive_gap_count": full_pool.get("positive_gap_count"),
                "full_pool_fdr_significant_count": full_pool.get("bh_fdr_significant_count"),
                "purifier_seed": purifier_seed,
                "output_dir": os.path.relpath(run_dir, root_output_dir),
            }
            rows.append(row)
            completed.append(
                {
                    "checkpoint": label,
                    "checkpoint_path": os.path.relpath(checkpoint_path, root_output_dir),
                    "first_stage_epoch": epoch,
                    "purifier_summary": os.path.relpath(summary_path, root_output_dir),
                }
            )
            with open(os.path.join(run_dir, "checkpoint_source.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "checkpoint": label,
                        "source_checkpoint": os.path.relpath(checkpoint_path, root_output_dir),
                        "first_stage_epoch": epoch,
                        "representation_score": bundle.get("representation_score"),
                        "purifier_seed": purifier_seed,
                        "source_stats": source_stats,
                    },
                    handle,
                    indent=2,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        args.output_dir = root_output_dir
        model.load_state_dict(restore_bundle["model_state_dict"])

    write_csv(os.path.join(ablation_dir, "summary.csv"), rows)
    downstream_ranked = sorted(
        [
            row for row in rows
            if row.get("residual_meta_info_bits_per_label") is not None
            and math.isfinite(float(row["residual_meta_info_bits_per_label"]))
        ],
        key=lambda row: float(row["residual_meta_info_bits_per_label"]),
        reverse=True,
    )
    feasible_ranked = [row for row in downstream_ranked if row.get("reconstruction_feasible")]
    selected_row = next((row for row in rows if row["source_kind"] == "best_representation"), None)
    selected_rank = (
        next(
            (index + 1 for index, row in enumerate(downstream_ranked) if row["checkpoint"] == selected_row["checkpoint"]),
            None,
        )
        if selected_row is not None
        else None
    )
    selected_feasible_rank = (
        next(
            (index + 1 for index, row in enumerate(feasible_ranked) if row["checkpoint"] == selected_row["checkpoint"]),
            None,
        )
        if selected_row is not None
        else None
    )
    strategy_evaluation = {
        "primary_downstream_metric": "residual_meta_info_bits_per_label",
        "selected_checkpoint": None if selected_row is None else selected_row["checkpoint"],
        "selected_downstream_rank_all": selected_rank,
        "selected_downstream_rank_reconstruction_feasible": selected_feasible_rank,
        "best_downstream_checkpoint_all": None if not downstream_ranked else downstream_ranked[0]["checkpoint"],
        "best_downstream_checkpoint_reconstruction_feasible": None if not feasible_ranked else feasible_ranked[0]["checkpoint"],
        "selected_is_best_all": bool(selected_rank == 1),
        "selected_is_best_reconstruction_feasible": bool(selected_feasible_rank == 1),
        "reconstruction_feasible_count": len(feasible_ranked),
        "caution": "A single common purifier seed isolates checkpoints within this run; repeat the winning comparison across seeds before changing the checkpoint policy.",
    }
    summary = {
        "enabled": True,
        "design": "One first-stage trajectory; identical purifier configuration and seed for every checkpoint.",
        "selection_strategy_under_test": args.main_final_checkpoint,
        "selected_checkpoint_epoch": restore_bundle.get("epoch"),
        "common_purifier_seed": purifier_seed,
        "checkpoint_count": len(rows),
        "checkpoints": completed,
        "results": rows,
        "strategy_evaluation": strategy_evaluation,
        "interpretation": {
            "representation_score": "Lower is preferred by the current first-stage checkpoint rule.",
            "downstream_primary": "Higher purifier residual_meta_info_bits_per_label with positive CI and more FDR-significant targets supports a better purifier starting point.",
            "validation": "The current rule is supported only if best_representation is competitive on downstream metrics while preserving reconstruction and leakage constraints.",
        },
        "outputs": {
            "summary_csv": "main_checkpoint_ablation/summary.csv",
            "plot": None if args.no_plots else "main_checkpoint_ablation/checkpoint_comparison.png",
        },
    }
    with open(os.path.join(ablation_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not args.no_plots and rows:
        try:
            import matplotlib.pyplot as plt

            labels = [str(row["first_stage_epoch"]) for row in rows]

            def finite_values(key: str) -> List[float]:
                values = []
                for row in rows:
                    value = row.get(key)
                    values.append(float(value) if value is not None and math.isfinite(float(value)) else float("nan"))
                return values

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes[0, 0].plot(labels, finite_values("representation_score"), marker="o")
            axes[0, 0].set_title("First-stage representation score (lower is better)")
            axes[0, 0].set_ylabel("score")
            axes[0, 1].bar(labels, finite_values("residual_meta_info_bits_per_label"), color="#4f779d")
            axes[0, 1].set_title("Purifier residual information")
            axes[0, 1].set_ylabel("bits / label")
            axes[1, 0].bar(labels, finite_values("residual_fraction"), color="#5a9b68")
            axes[1, 0].set_title("Purifier residual contribution fraction")
            axes[1, 0].set_ylabel("fraction")
            axes[1, 1].plot(labels, finite_values("claim_fdr_significant_count"), marker="o", label="claim")
            axes[1, 1].plot(labels, finite_values("full_pool_fdr_significant_count"), marker="s", label="full pool")
            axes[1, 1].set_title("FDR-significant targets")
            axes[1, 1].set_ylabel("count")
            axes[1, 1].legend()
            for axis in axes.flat:
                axis.set_xlabel("first-stage epoch")
                axis.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(ablation_dir, "checkpoint_comparison.png"), dpi=180)
            plt.close(fig)
        except Exception as exc:
            summary["outputs"]["plot_error"] = str(exc)
            with open(os.path.join(ablation_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
    return summary


def run_binary_posthoc_analysis(
    model: Decoupler,
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
    binary_eval_meta: Optional[Dict[str, torch.Tensor]] = None,
    all_ids: Optional[List[str]] = None,
    prev_layer: Optional[int] = None,
    next_layer: Optional[int] = None,
    semantic_residual_state: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> Optional[Dict]:
    if not is_binary_eval_mode(args.target_mode) or args.skip_binary_posthoc:
        return None

    analysis_dir = os.path.join(args.output_dir, "binary_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    train_probe_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_probe_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    posthoc_steps = 8 if args.skip_information_analysis else 9
    posthoc_bar = _tqdm(total=posthoc_steps, desc="Binary posthoc", leave=True, dynamic_ncols=True)
    train_tensors = collect_probe_tensors(
        model,
        train_probe_loader,
        device,
        args.probe_max_train_samples,
        "Collect train probe tensors",
        args.target_mode,
        binary_eval_meta,
        semantic_residual_state,
        args.main_prediction_target,
    )
    val_tensors = collect_probe_tensors(
        model,
        val_probe_loader,
        device,
        args.probe_max_val_samples,
        "Collect val probe tensors",
        args.target_mode,
        binary_eval_meta,
        semantic_residual_state,
        args.main_prediction_target,
    )
    posthoc_bar.update(1)

    prior_logits = binary_prior_logits(train_tensors["x_i"], val_tensors["x_i"])
    prior_ce = binary_ce_from_logits(prior_logits, val_tensors["x_i"])
    main_ce = binary_ce_from_logits(val_tensors["main_logits"], val_tensors["x_i"])
    main_metrics = {
        "name": "main_e2_mlp",
        "ce_nats_per_label": main_ce,
        "prior_ce_nats_per_label": prior_ce,
        "info_gain_nats_per_label_vs_prior": prior_ce - main_ce,
        "info_gain_bits_per_label_vs_prior": (prior_ce - main_ce) / math.log(2.0),
        "note": "Global average accuracy/AUC is no longer used for target selection; dynamic target selection uses per-target semantic-control CE gap.",
    }
    posthoc_bar.update(1)

    probe_metrics = {}
    e1_metrics, e1_logits = train_binary_linear_probe(
        train_tensors["z1"],
        train_tensors["x_i"],
        val_tensors["z1"],
        val_tensors["x_i"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        args.binary_pred_threshold,
        "e1_to_i_linear",
        analysis_dir,
    )
    probe_metrics["e1_to_i_linear"] = e1_metrics
    posthoc_bar.update(1)
    e2_metrics, e2_logits = train_binary_linear_probe(
        train_tensors["z2"],
        train_tensors["x_i"],
        val_tensors["z2"],
        val_tensors["x_i"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        args.binary_pred_threshold,
        "e2_to_i_linear",
        analysis_dir,
    )
    probe_metrics["e2_to_i_linear"] = e2_metrics
    posthoc_bar.update(1)

    generator = torch.Generator().manual_seed(args.seed + 400)
    shuffle_idx = torch.randperm(train_tensors["x_prev"].size(0), generator=generator)
    random_train_z = torch.randn(train_tensors["z2"].shape, generator=generator)
    random_val_z = torch.randn(val_tensors["z2"].shape, generator=generator)

    leakage_metrics = {}
    e1_prev_metrics, e1_prev_pred = train_regression_linear_probe(
        train_tensors["z1"],
        train_tensors["x_prev"],
        val_tensors["z1"],
        val_tensors["x_prev"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        "e1_to_prev",
        analysis_dir,
    )
    leakage_metrics["e1_to_prev"] = e1_prev_metrics
    posthoc_bar.update(1)
    e2_prev_metrics, e2_prev_pred = train_regression_linear_probe(
        train_tensors["z2"],
        train_tensors["x_prev"],
        val_tensors["z2"],
        val_tensors["x_prev"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        "e2_to_prev",
        analysis_dir,
    )
    leakage_metrics["e2_to_prev"] = e2_prev_metrics
    posthoc_bar.update(1)
    shuffle_metrics, shuffle_pred = train_regression_linear_probe(
        train_tensors["z2"],
        train_tensors["x_prev"][shuffle_idx],
        val_tensors["z2"],
        val_tensors["x_prev"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        "e2_to_prev_shuffled_target",
        analysis_dir,
    )
    leakage_metrics["e2_to_prev_shuffled_target"] = shuffle_metrics
    random_metrics, random_pred = train_regression_linear_probe(
        random_train_z,
        train_tensors["x_prev"],
        random_val_z,
        val_tensors["x_prev"],
        device,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        "random_z_to_prev",
        analysis_dir,
    )
    leakage_metrics["random_z_to_prev"] = random_metrics
    posthoc_bar.update(1)
    mean_pred = train_tensors["x_prev"].mean(dim=0, keepdim=True).expand_as(val_tensors["x_prev"])
    leakage_metrics["mean_baseline"] = regression_metrics(mean_pred, val_tensors["x_prev"])
    leakage_metrics["mean_baseline"]["name"] = "mean_baseline"
    val_mean_pred = val_tensors["x_prev"].mean(dim=0, keepdim=True).expand_as(val_tensors["x_prev"])
    leakage_metrics["val_mean_baseline_oracle"] = regression_metrics(val_mean_pred, val_tensors["x_prev"])
    leakage_metrics["val_mean_baseline_oracle"]["name"] = "val_mean_baseline_oracle"
    add_relative_improvement(leakage_metrics)

    prev_leakage_predictions = {
        "e1_to_prev": e1_prev_pred,
        "e2_to_prev": e2_prev_pred,
        "e2_to_prev_shuffled_target": shuffle_pred,
        "random_z_to_prev": random_pred,
        "mean_baseline": mean_pred,
        "val_mean_baseline_oracle": val_mean_pred,
    }
    e2_sample_mse = per_sample_mse(e2_prev_pred, val_tensors["x_prev"])
    mean_sample_mse = per_sample_mse(mean_pred, val_tensors["x_prev"])
    shuffle_sample_mse = per_sample_mse(shuffle_pred, val_tensors["x_prev"])
    random_sample_mse = per_sample_mse(random_pred, val_tensors["x_prev"])
    e2_sample_mse_debiased = per_sample_debiased_mse(e2_prev_pred, val_tensors["x_prev"])
    mean_sample_mse_debiased = per_sample_debiased_mse(mean_pred, val_tensors["x_prev"])
    shuffle_sample_mse_debiased = per_sample_debiased_mse(shuffle_pred, val_tensors["x_prev"])
    random_sample_mse_debiased = per_sample_debiased_mse(random_pred, val_tensors["x_prev"])
    leakage_metrics["e2_to_prev"]["mse_minus_mean_baseline_ci"] = bootstrap_mean_ci(
        e2_sample_mse - mean_sample_mse, args.bootstrap_samples, args.seed + 500, "Bootstrap E2-prev vs mean"
    )
    leakage_metrics["e2_to_prev"]["mse_minus_shuffle_baseline_ci"] = bootstrap_mean_ci(
        e2_sample_mse - shuffle_sample_mse, args.bootstrap_samples, args.seed + 501, "Bootstrap E2-prev vs shuffled"
    )
    leakage_metrics["e2_to_prev"]["mse_minus_random_baseline_ci"] = bootstrap_mean_ci(
        e2_sample_mse - random_sample_mse, args.bootstrap_samples, args.seed + 502, "Bootstrap E2-prev vs random"
    )
    leakage_metrics["e2_to_prev"]["mse_minus_val_mean_oracle_ci"] = bootstrap_mean_ci(
        e2_sample_mse - per_sample_mse(val_mean_pred, val_tensors["x_prev"]),
        args.bootstrap_samples,
        args.seed + 507,
        "Bootstrap E2-prev vs val mean oracle",
    )
    leakage_metrics["e2_to_prev"]["mse_debiased_minus_mean_baseline_ci"] = bootstrap_mean_ci(
        e2_sample_mse_debiased - mean_sample_mse_debiased, args.bootstrap_samples, args.seed + 503, "Bootstrap debiased E2-prev vs mean"
    )
    leakage_metrics["e2_to_prev"]["mse_debiased_minus_val_mean_oracle_ci"] = bootstrap_mean_ci(
        e2_sample_mse_debiased - per_sample_debiased_mse(val_mean_pred, val_tensors["x_prev"]),
        args.bootstrap_samples,
        args.seed + 506,
        "Bootstrap debiased E2-prev vs val mean oracle",
    )
    leakage_metrics["e2_to_prev"]["mse_debiased_minus_shuffle_baseline_ci"] = bootstrap_mean_ci(
        e2_sample_mse_debiased - shuffle_sample_mse_debiased, args.bootstrap_samples, args.seed + 504, "Bootstrap debiased E2-prev vs shuffled"
    )
    leakage_metrics["e2_to_prev"]["mse_debiased_minus_random_baseline_ci"] = bootstrap_mean_ci(
        e2_sample_mse_debiased - random_sample_mse_debiased, args.bootstrap_samples, args.seed + 505, "Bootstrap debiased E2-prev vs random"
    )
    sample_raw_mse = {name: per_sample_mse(pred, val_tensors["x_prev"]) for name, pred in prev_leakage_predictions.items()}
    sample_debiased_mse = {name: per_sample_debiased_mse(pred, val_tensors["x_prev"]) for name, pred in prev_leakage_predictions.items()}
    sample_metric_rows = []
    for idx in range(val_tensors["x_prev"].size(0)):
        row = {"sample_index": idx}
        for name in prev_leakage_predictions:
            row[f"{name}_raw_mse"] = sample_raw_mse[name][idx].item()
            row[f"{name}_debiased_mse"] = sample_debiased_mse[name][idx].item()
        sample_metric_rows.append(row)
    write_csv(os.path.join(analysis_dir, "prev_leakage_sample_metrics.csv"), sample_metric_rows)
    posthoc_bar.update(1)

    neuron_rows = per_neuron_metrics_from_logits(
        val_tensors["main_logits"],
        val_tensors["x_i"],
        args.binary_pred_threshold,
        args.min_neuron_activation_rate,
        args.max_neuron_activation_rate,
        args.min_neuron_positive,
        args.min_neuron_negative,
    )
    unstable_rows = [row for row in neuron_rows if row["is_variable"]]
    neuron_fieldnames = list(neuron_rows[0].keys()) if neuron_rows else None
    write_csv(os.path.join(analysis_dir, "per_neuron_binary_metrics.csv"), neuron_rows)
    write_csv(os.path.join(analysis_dir, "variable_neuron_diagnostics.csv"), unstable_rows, neuron_fieldnames)
    write_csv(os.path.join(analysis_dir, "binary_probe_metrics.csv"), list(probe_metrics.values()))
    write_csv(os.path.join(analysis_dir, "prev_leakage_metrics.csv"), list(leakage_metrics.values()))

    information_summary = None
    if not args.skip_information_analysis:
        information_summary = run_information_analysis(
            analysis_dir,
            train_tensors,
            val_tensors,
            val_tensors["main_logits"],
            e1_logits,
            e2_logits,
            leakage_metrics,
            device,
            args,
        )
        posthoc_bar.update(1)
        write_csv(os.path.join(analysis_dir, "prev_leakage_metrics.csv"), list(leakage_metrics.values()))

    summary = {
        "main_e2_mlp": main_metrics,
        "probes": probe_metrics,
        "prev_leakage": leakage_metrics,
        "neuron_counts": {
            "total": len(neuron_rows),
            "variable": sum(1 for row in neuron_rows if row["is_variable"]),
        },
        "settings": {
            "binary_pred_threshold": args.binary_pred_threshold,
            "min_neuron_activation_rate": args.min_neuron_activation_rate,
            "max_neuron_activation_rate": args.max_neuron_activation_rate,
            "min_neuron_positive": args.min_neuron_positive,
            "min_neuron_negative": args.min_neuron_negative,
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_tests": args.permutation_tests,
            "intervention_top_neurons": args.intervention_top_neurons,
            "skip_information_analysis": args.skip_information_analysis,
            "information_alpha": args.information_alpha,
            "semantic_oracle_max_pca_dim": args.semantic_oracle_max_pca_dim,
            "semantic_oracle_budget_multiplier": args.semantic_oracle_budget_multiplier,
            "semantic_oracle_selection": args.semantic_oracle_selection,
        },
    }
    if information_summary is not None:
        summary["information_analysis"] = information_summary

    torch.save(
        {
            "val_target": val_tensors["x_i"],
            "main_logits": val_tensors["main_logits"],
            "e1_to_i_logits": e1_logits,
            "e2_to_i_linear_logits": e2_logits,
            "x_prev": val_tensors["x_prev"],
            "e1_to_prev_pred": e1_prev_pred,
            "e2_to_prev_pred": e2_prev_pred,
            "e2_to_prev_shuffled_target_pred": shuffle_pred,
            "random_z_to_prev_pred": random_pred,
            "mean_prev_pred": mean_pred,
            "val_mean_prev_pred": val_mean_pred,
            "information_analysis_available": information_summary is not None,
        },
        os.path.join(analysis_dir, "binary_analysis_predictions.pt"),
    )
    intervention_summary = None
    if all_ids is not None and prev_layer is not None and next_layer is not None:
        intervention_summary = export_intervention_interface(
            analysis_dir,
            all_ids,
            train_ds,
            val_ds,
            train_tensors,
            val_tensors,
            neuron_rows,
            args,
            binary_eval_meta,
            prev_layer,
            next_layer,
            main_metrics,
            probe_metrics,
            leakage_metrics,
        )
        summary["intervention_interface"] = {
            "summary": "intervention_targets.json",
            "tensor_bundle": "intervention_targets.pt",
            "sample_groups": "intervention_sample_groups.csv",
            "neurons": "intervention_neurons.csv",
            "selected_neuron_count": intervention_summary["selected_neuron_count"],
            "selection_source": intervention_summary["selection_source"],
        }
    with open(os.path.join(analysis_dir, "binary_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    posthoc_bar.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--model-label", default=None, help="Optional model label written to summaries, e.g. qwen2.5-7b or qwen2.5-72b-int4.")
    parser.add_argument("--layer-i", type=int, default=None)
    parser.add_argument("--prev-layer", type=int, default=None, help="Layer used as x_{i-k}. Defaults to layer-i - prev-offset.")
    parser.add_argument("--next-layer", type=int, default=None, help="Layer used as x_{i+k}. Defaults to layer-i + next-offset.")
    parser.add_argument("--prev-offset", type=int, default=1, help="Used only when --prev-layer is not set.")
    parser.add_argument("--next-offset", type=int, default=1, help="Used only when --next-layer is not set.")
    parser.add_argument("--layer-fracs", type=float, nargs=3, default=None, metavar=("PREV", "I", "NEXT"), help="Model-size-normalized layer triplet. Example: 0.125 0.5 0.875 maps 72B/80 layers to 10/40/70 and 7B/28 layers to 4/14/25.")
    parser.add_argument("--semantic-anchor-offsets", type=int, nargs="+", default=None, help="Optional i-n offsets used to construct a shared semantic target. Each anchor is standardized before weighted averaging.")
    parser.add_argument("--semantic-anchor-weights", type=float, nargs="+", default=None, help="Optional non-negative weights for --semantic-anchor-offsets; omitted uses uniform weights.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reuse-first-stage-dir",
        default="",
        help=(
            "Reuse an already trained first-stage representation checkpoint and skip gate calibration/main training. "
            "The source directory must contain best_representation_checkpoint.pt, config.json, and analysis_summary.json."
        ),
    )
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Optional held-out test ratio. With --split-by-base-id, this is applied to base sample groups.")
    parser.add_argument("--split-by-base-id", action="store_true", help="Group train/val/test splits by base example id so generated steps from the same prompt never cross splits.")
    parser.add_argument("--base-id-step-pattern", default=r"::step\d+$", help="Regex suffix removed from example ids when --split-by-base-id is enabled.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional train/validation split seed; omitted reuses --seed. This separates split stability from initialization stability.",
    )
    parser.add_argument("--lambda-next", type=float, default=1.0)
    parser.add_argument("--lambda-prev", type=float, default=1.0)
    parser.add_argument("--lambda-orth", type=float, default=0.1)
    parser.add_argument("--lambda-e2-prev", type=float, default=0.1)
    parser.add_argument("--lambda-e2-prev-adv", type=float, default=0.0, help="Adversarial penalty weight that discourages E2 from carrying reconstructable x_{i-1} information.")
    parser.add_argument("--e2-prev-adv-margin", type=float, default=0.9, help="Target lower bound for adversary MSE. Since x_{i-1} is standardized, mean-baseline MSE is usually near 1.")
    parser.add_argument("--e2-prev-adv-lr", type=float, default=1e-4, help="Learning rate for the z2->x_{i-1} adversary.")
    parser.add_argument("--e2-prev-adv-steps", type=int, default=1, help="Adversary update steps per main training batch.")
    parser.add_argument("--e2-prev-adv-start-epoch", type=int, default=1, help="Epoch at which the adversarial E2 leakage penalty starts.")
    parser.add_argument("--e2-prev-adv-ramp-epochs", type=int, default=10, help="Number of epochs used to ramp lambda-e2-prev-adv to its full value.")
    parser.add_argument("--lambda-var", type=float, default=0.01)
    parser.add_argument("--lambda-pred-max", type=float, default=0.2)
    parser.add_argument("--pred-ramp-start-epoch", type=int, default=10)
    parser.add_argument("--main-prediction-target", choices=["raw", "residual"], default="raw", help="raw lets E2 predict original targets; residual trains a Z1 semantic baseline and asks E2 to predict only the remaining target signal.")
    parser.add_argument("--main-semantic-pred-weight", type=float, default=-1.0, help="Weight for first-stage Z1 semantic target predictor. Negative reuses the current prediction ramp in residual mode and disables it in raw mode.")
    parser.add_argument("--fixed-target-file", default="", help="Optional .pt/.csv target weights from a previous purifier run. When set, prediction loss is restricted to these neuron targets unless dynamic target refresh later overwrites them.")
    parser.add_argument("--fixed-target-index-column", default="", help="CSV column for target indices; omitted tries feature_index/neuron/target_index/index.")
    parser.add_argument("--fixed-target-weight-column", default="", help="CSV column for target weights; omitted tries weight/target_weight/training_weight/residual_fraction/residual_gap_bits.")
    parser.add_argument("--fixed-target-top-k", type=int, default=0, help="If reading CSV fixed targets, keep only the top K by weight; 0 keeps all.")
    parser.add_argument("--fixed-target-min-weight", type=float, default=0.0, help="Drop fixed targets whose source weight is below this value.")
    parser.add_argument("--fixed-target-binarize", action="store_true", help="Convert all retained fixed target weights to 1.0.")
    parser.add_argument("--fixed-target-normalize", action=argparse.BooleanOptionalAction, default=True, help="Normalize retained fixed target weights to sum to the retained target count.")
    parser.add_argument("--dynamic-target-weighting", action="store_true", help="Dynamically weight i-layer prediction targets so E2 focuses on predictable, non-semantic-dominated neurons.")
    parser.add_argument("--dynamic-target-mode", choices=["neuron", "direction"], default="neuron", help="Select dynamic prediction targets as individual neurons or continuous activation directions.")
    parser.add_argument("--dynamic-target-start-epoch", type=int, default=0, help="Epoch to start dynamic target updates; 0 reuses --pred-ramp-start-epoch.")
    parser.add_argument("--dynamic-target-refresh-epochs", type=int, default=5, help="Refresh dynamic target weights every N epochs.")
    parser.add_argument("--dynamic-target-probe-epochs", type=int, default=3, help="Epochs for temporary target/leakage probes used during target refresh.")
    parser.add_argument("--dynamic-target-max-train-samples", type=int, default=2048, help="Max train samples used for each dynamic target refresh; 0 uses all.")
    parser.add_argument("--dynamic-target-max-val-samples", type=int, default=1024, help="Max validation samples used for each dynamic target refresh; 0 uses all.")
    parser.add_argument("--dynamic-target-top-k", type=int, default=256, help="Keep at most this many dynamically selected target neurons; 0 keeps all eligible.")
    parser.add_argument("--dynamic-target-min-selected", type=int, default=0, help="Optional relaxed warm-start floor for selected targets; 0 lets the model fully abandon unsupported targets.")
    parser.add_argument("--dynamic-target-weight-mode", choices=["topk", "soft"], default="topk", help="Use binary top-k weights or score-based soft weights.")
    parser.add_argument("--dynamic-target-temperature", type=float, default=0.02, help="Temperature for score-based soft dynamic target weights.")
    parser.add_argument("--dynamic-target-min-soft-weight", type=float, default=0.05, help="Minimum soft weight assigned to selected targets in soft mode.")
    parser.add_argument("--dynamic-target-e2-gain-min", type=float, default=0.0, help="Minimum per-neuron E2 information gain over activation-rate prior in nats.")
    parser.add_argument("--dynamic-target-score-min", type=float, default=0.0, help="Minimum semantic-control gap in nats: CE(best recovered semantic control) - CE(E2 target probe).")
    parser.add_argument("--dynamic-target-claim-e2-gain-min", type=float, default=None, help="Optional stricter E2-gain threshold for confirmation/checkpoint claims. Omitted reuses --dynamic-target-e2-gain-min.")
    parser.add_argument("--dynamic-target-claim-score-min", type=float, default=None, help="Optional stricter semantic-gap threshold for confirmation/checkpoint claims. Omitted reuses --dynamic-target-score-min.")
    parser.add_argument("--dynamic-target-bootstrap-samples", type=int, default=0, help="Bootstrap resamples for per-target semantic-control gap lower bounds during dynamic refresh; 0 disables.")
    parser.add_argument("--dynamic-target-require-positive-ci", action="store_true", help="Require the bootstrap 95% lower bound of semantic-control gap to be positive before selecting a target.")
    parser.add_argument("--dynamic-target-ema", type=float, default=0.0, help="EMA smoothing for dynamic target weights; 0 fully replaces old weights.")
    parser.add_argument("--dynamic-target-candidate-top-k", type=int, default=0, help="Maintain this many promising targets with exploration gradients between refreshes; 0 disables the persistent candidate policy.")
    parser.add_argument("--dynamic-target-exploration-weight", type=float, default=0.0, help="Relative prediction weight for unconfirmed targets in the persistent candidate pool.")
    parser.add_argument("--dynamic-target-score-ema", type=float, default=0.0, help="EMA coefficient for candidate potential scores in the persistent target policy.")
    parser.add_argument("--dynamic-target-stability-weight", type=float, default=0.0, help="Weight of cross-refresh stability score when ranking first-stage candidate/confirmed targets. 0 preserves score-only ranking.")
    parser.add_argument("--dynamic-target-candidate-gain-weight", type=float, default=0.25, help="Weight of positive E2 information gain when ranking promising candidates before their semantic gap is significant.")
    parser.add_argument("--dynamic-target-confirm-evals", type=int, default=1, help="Consecutive strict-positive refreshes required before a first-stage target receives full weight.")
    parser.add_argument("--dynamic-target-drop-patience-evals", type=int, default=1, help="Consecutive failed refreshes required before a confirmed first-stage target is removed.")
    parser.add_argument("--dynamic-target-export-threshold", type=float, default=1e-6, help="Weight threshold used when counting/exporting selected dynamic targets.")
    parser.add_argument("--main-final-checkpoint", choices=["last", "representation"], default="last", help="First-stage state passed to posthoc/purifier. representation uses a target-independent reconstruction/dependence checkpoint.")
    parser.add_argument("--main-checkpoint-start-epoch", type=int, default=0, help="Earliest epoch eligible for the representation checkpoint; 0 allows all epochs.")
    parser.add_argument("--main-checkpoint-cov-weight", type=float, default=1.0, help="Weight of validation E2-semantic covariance in the representation checkpoint score.")
    parser.add_argument("--main-checkpoint-orth-weight", type=float, default=0.1, help="Weight of validation Z1/Z2 orthogonality in the representation checkpoint score.")
    parser.add_argument("--main-checkpoint-z2-std-min", type=float, default=0.25, help="Minimum mean Z2 std encouraged by the representation checkpoint score.")
    parser.add_argument("--main-checkpoint-collapse-weight", type=float, default=1.0, help="Penalty weight when validation Z2 std falls below the checkpoint floor.")
    parser.add_argument(
        "--main-checkpoint-ablation-epochs",
        nargs="*",
        type=int,
        default=[],
        help="Save complete first-stage bundles at these epochs and compare identical purifier runs from each bundle plus the best representation checkpoint.",
    )
    parser.add_argument(
        "--main-checkpoint-ablation-only",
        action="store_true",
        help="Run only checkpoint-ablation purifiers and skip the ordinary root e2_recursive_purifier run.",
    )
    parser.add_argument(
        "--main-checkpoint-ablation-seed",
        type=int,
        default=-1,
        help="Common purifier seed for every checkpoint ablation; negative uses seed+17000.",
    )
    parser.add_argument(
        "--main-checkpoint-ablation-overwrite",
        action="store_true",
        help="Rerun checkpoint-ablation outputs even when a completed summary already exists.",
    )
    parser.add_argument("--dynamic-semantic-residual-rounds", type=int, default=0, help="Before dynamic target selection, iteratively remove linear E2 directions that recover Z1/prev. 0 keeps the original E2.")
    parser.add_argument("--dynamic-semantic-residual-rank", type=int, default=64, help="Max semantic row-space directions removed per residual round; 0 keeps all singular directions above the threshold.")
    parser.add_argument("--dynamic-semantic-residual-min-sv-ratio", type=float, default=1e-4, help="Keep semantic leakage singular directions with S >= max(S) * this ratio when residualizing E2.")
    parser.add_argument("--dynamic-predict-on-semantic-residual", action="store_true", help="When dynamic residualization is enabled, train/evaluate predict_i on residualized Z2 instead of raw Z2.")
    parser.add_argument("--dynamic-direction-candidates", type=int, default=256, help="Number of candidate directions evaluated when --dynamic-target-mode direction.")
    parser.add_argument("--dynamic-direction-source", choices=["residual_pls", "e2_pls", "pca", "random", "mixed"], default="residual_pls", help="How candidate directions are generated for direction-mode dynamic targets.")
    parser.add_argument("--dynamic-direction-min-variance", type=float, default=1e-6, help="Minimum validation variance of standardized projected direction targets.")
    parser.add_argument("--run-e2-recursive-purifier", action="store_true", help="After main training, freeze the decoupler and split E2 into semantic/meta codes while still predicting neuron targets from meta.")
    parser.add_argument("--purifier-epochs", type=int, default=30, help="Epochs for the posthoc E2 recursive purifier.")
    parser.add_argument("--purifier-code-dim", type=int, default=0, help="Legacy base code dim for the recursive purifier; 0 reuses --latent-dim. Used as semantic dim when --purifier-semantic-dim is unset.")
    parser.add_argument("--purifier-semantic-dim", type=int, default=0, help="Semantic-code dim for recursive purifier; 0 uses --purifier-code-dim/--latent-dim.")
    parser.add_argument("--purifier-meta-dim", type=int, default=0, help="Meta-code dim for recursive purifier; 0 uses a bottleneck around one quarter of the semantic/base code dim.")
    parser.add_argument("--purifier-hidden-dim", type=int, default=0, help="Hidden dim for purifier MLPs; 0 reuses --hidden-dim.")
    parser.add_argument("--purifier-batch-size", type=int, default=0, help="Batch size for purifier training; 0 reuses --batch-size.")
    parser.add_argument("--purifier-lr", type=float, default=0.0, help="Learning rate for purifier; 0 reuses --lr.")
    parser.add_argument("--purifier-max-train-samples", type=int, default=0, help="Max train samples for purifier; 0 uses all.")
    parser.add_argument("--purifier-max-val-samples", type=int, default=0, help="Max validation samples for purifier; 0 uses all.")
    parser.add_argument("--purifier-probe-epochs", type=int, default=10, help="Epochs for posthoc nonlinear leakage probes on purifier codes; 0 reuses --probe-epochs.")
    parser.add_argument("--purifier-probe-hidden-dim", type=int, default=0, help="Hidden dim for nonlinear purifier leakage probes; 0 reuses purifier hidden dim.")
    parser.add_argument("--purifier-meta-input-mode", choices=["z2", "semantic_residual"], default="z2", help="Feed the meta encoder either full E2 or the stop-gradient residual E2-semantic_to_z2(semantic).")
    parser.add_argument("--purifier-prediction-target", choices=["raw", "residual"], default="raw", help="raw trains meta_to_i directly on neuron states; residual trains semantic_to_i as a detached baseline and meta_to_i as the residual delta.")
    parser.add_argument("--purifier-semantic-warmup-epochs", type=int, default=0, help="Train semantic encoder only on Z1/prev targets for this many initial purifier epochs.")
    parser.add_argument("--purifier-semantic-z2-warmup-epochs", type=int, default=0, help="After semantic encoder warmup, freeze it and fit semantic-to-E2 for this many epochs before meta training.")
    parser.add_argument("--purifier-freeze-semantic-after-warmup", action=argparse.BooleanOptionalAction, default=False, help="Freeze semantic encoder/decoders after warmup so the residual reference cannot drift under prediction pressure.")
    parser.add_argument("--lambda-purifier-recon-z2", type=float, default=1.0, help="Weight for reconstructing original E2 from semantic+meta purifier codes.")
    parser.add_argument("--lambda-purifier-semantic-recon-z2", type=float, default=0.0, help="Weight for the independent semantic-only E2 reconstruction used to define residual meta input.")
    parser.add_argument("--lambda-purifier-meta-residual", type=float, default=0.0, help="Weight for reconstructing the stop-gradient semantic residual directly from purifier meta code.")
    parser.add_argument("--lambda-purifier-semantic", type=float, default=1.0, help="Weight for reconstructing Z1/x_prev from purifier semantic code.")
    parser.add_argument("--lambda-purifier-semantic-pred", type=float, default=-1.0, help="Semantic baseline prediction weight used when --purifier-prediction-target residual; negative reuses --lambda-purifier-pred.")
    parser.add_argument("--lambda-purifier-pred", type=float, default=0.2, help="Weight for predicting neuron targets from purifier meta code.")
    parser.add_argument("--purifier-pred-start-epoch", type=int, default=1, help="Epoch to enable purifier meta-to-neuron prediction; later starts provide a semantic/reconstruction warmup.")
    parser.add_argument("--purifier-pred-ramp-epochs", type=int, default=0, help="Linearly ramp purifier prediction weight over this many epochs; 0 enables the full weight immediately.")
    parser.add_argument("--lambda-purifier-orth", type=float, default=0.1, help="Orthogonality penalty between purifier semantic and meta codes.")
    parser.add_argument("--lambda-purifier-meta-sem-cov", type=float, default=0.3, help="Covariance penalty between purifier meta code and Z1/x_prev.")
    parser.add_argument("--lambda-purifier-var", type=float, default=0.01, help="Variance floor penalty for purifier codes.")
    parser.add_argument("--purifier-dynamic-target-weighting", action="store_true", help="During recursive purifier training, periodically reselect neuron targets whose purified meta code predicts better than semantic variables recovered from that same meta code.")
    parser.add_argument("--purifier-dynamic-start-epoch", type=int, default=5, help="Epoch to start purifier dynamic target reselection. Use >1 so the purifier has a warmup period.")
    parser.add_argument("--purifier-dynamic-refresh-epochs", type=int, default=5, help="Refresh purifier dynamic target weights every N purifier epochs.")
    parser.add_argument("--purifier-dynamic-probe-epochs", type=int, default=0, help="Probe epochs for purifier dynamic target reselection; 0 reuses --purifier-probe-epochs/--probe-epochs for alignment with final evaluation.")
    parser.add_argument("--purifier-dynamic-probe-repeats", type=int, default=1, help="Independent broad-screen target-probe fits averaged at each purifier target refresh.")
    parser.add_argument("--purifier-dynamic-confirmation-fraction", type=float, default=0.0, help="Fraction of the purifier validation split reserved for target confirmation. 0 preserves the legacy shared-validation behavior.")
    parser.add_argument("--purifier-dynamic-confirmation-shortlist-k", type=int, default=0, help="After broad screening, refit target-specialized probes for this many candidates on the disjoint confirmation partition. 0 disables screen-confirm probing.")
    parser.add_argument("--purifier-dynamic-confirmation-probe-epochs", type=int, default=0, help="Epochs for shortlist confirmation probes; 0 reuses --purifier-dynamic-probe-epochs.")
    parser.add_argument("--purifier-dynamic-confirmation-probe-repeats", type=int, default=0, help="Repeats for shortlist confirmation probes; 0 reuses --purifier-dynamic-probe-repeats.")
    parser.add_argument("--purifier-dynamic-freeze-shortlist-epoch", type=int, default=0, help="Freeze the first screen-confirm shortlist observed at or after this purifier epoch. 0 keeps refreshing it.")
    parser.add_argument("--purifier-dynamic-max-train-samples", type=int, default=0, help="Fixed training subset size used by repeated purifier target probes; 0 uses all collected purifier samples.")
    parser.add_argument("--purifier-dynamic-max-val-samples", type=int, default=0, help="Fixed validation subset size used by repeated purifier target probes; 0 uses all collected purifier samples.")
    parser.add_argument("--purifier-dynamic-semantic-probe", choices=["mlp", "linear"], default="mlp", help="Probe used to recover Z1/prev from purified meta during dynamic target selection. Default mlp matches final purifier semantic-control evaluation.")
    parser.add_argument("--purifier-dynamic-target-probe", choices=["linear", "mlp"], default="linear", help="Probe used for direct purified-meta to neuron prediction during target selection; mlp better matches the nonlinear semantic-control capacity.")
    parser.add_argument("--purifier-dynamic-probe-hidden-dim", type=int, default=0, help="Hidden dim for dynamic MLP semantic probes; 0 reuses --purifier-probe-hidden-dim/--purifier-hidden-dim/--hidden-dim.")
    parser.add_argument("--purifier-dynamic-top-k", type=int, default=64, help="Legacy purifier top-k used for both training and checkpoint claims unless the split top-k arguments are set.")
    parser.add_argument("--purifier-dynamic-train-top-k", type=int, default=0, help="Targets receiving full purifier prediction weight; 0 reuses --purifier-dynamic-top-k.")
    parser.add_argument("--purifier-dynamic-claim-top-k", type=int, default=0, help="Strongest strict targets used for checkpoint ranking and final hypothesis tests; 0 reuses --purifier-dynamic-top-k.")
    parser.add_argument("--purifier-target-group-size", type=int, default=0, help="Rotate non-claim full-weight targets in groups of this size; 0 keeps the original joint loss.")
    parser.add_argument("--purifier-group-claim-anchor", action="store_true", help="Include the current claim targets in every grouped prediction update while rotating the remaining full-weight targets.")
    parser.add_argument("--purifier-audit-full-training-pool", action="store_true", help="Run an additional final probe over every full-gradient target saved with the selected checkpoint. This is diagnostic, not confirmatory.")
    parser.add_argument("--purifier-final-reference-pool", choices=["claim", "full"], default="claim", help="Pool used to train final reference probes; claim/full statistics are then computed from the same logits.")
    parser.add_argument("--purifier-final-probe-repeats", type=int, default=1, help="Independent final reference-probe fits averaged before claim/full statistics.")
    parser.add_argument("--purifier-final-eval-split", choices=["validation", "test"], default="validation", help="Split used only for final purifier information decomposition. test is held out from target and checkpoint selection.")
    parser.add_argument("--purifier-final-max-eval-samples", type=int, default=0, help="Maximum rows in final purifier evaluation; 0 uses the complete selected split.")
    parser.add_argument("--purifier-dynamic-min-selected", type=int, default=0, help="Optional relaxed warm-start floor for purifier-selected targets.")
    parser.add_argument("--purifier-dynamic-weight-mode", choices=["topk", "soft"], default="topk")
    parser.add_argument("--purifier-dynamic-temperature", type=float, default=0.02)
    parser.add_argument("--purifier-dynamic-min-soft-weight", type=float, default=0.05)
    parser.add_argument("--purifier-dynamic-e2-gain-min", type=float, default=0.0, help="Minimum purified-meta information gain over activation-rate prior in nats.")
    parser.add_argument("--purifier-dynamic-score-min", type=float, default=0.0, help="Minimum residual semantic-control gap in nats: CE(best recovered semantic control) - CE(purified meta probe).")
    parser.add_argument("--purifier-dynamic-claim-e2-gain-min", type=float, default=None, help="Optional stricter purified-meta gain threshold for confirmation/checkpoint claims.")
    parser.add_argument("--purifier-dynamic-claim-score-min", type=float, default=None, help="Optional stricter residual-gap threshold for confirmation/checkpoint claims.")
    parser.add_argument("--purifier-dynamic-bootstrap-samples", type=int, default=0, help="Bootstrap resamples for purifier dynamic target gap lower bounds.")
    parser.add_argument("--purifier-dynamic-require-positive-ci", action="store_true", help="Require positive bootstrap lower bound for purifier target selection.")
    parser.add_argument("--purifier-dynamic-ema", type=float, default=0.0, help="EMA smoothing for purifier dynamic target weights.")
    parser.add_argument("--purifier-dynamic-confirm-evals", type=int, default=1, help="Require this many consecutive positive evaluations before a target is added to purifier training.")
    parser.add_argument("--purifier-dynamic-drop-patience-evals", type=int, default=1, help="Require this many consecutive failed evaluations before a confirmed target is removed.")
    parser.add_argument("--purifier-dynamic-candidate-top-k", type=int, default=0, help="Maintain this many purifier candidates with low exploration weight; 0 preserves the previous confirmed-only behavior.")
    parser.add_argument("--purifier-dynamic-exploration-weight", type=float, default=0.0, help="Relative purifier prediction weight assigned to unconfirmed candidate targets.")
    parser.add_argument("--purifier-dynamic-score-ema", type=float, default=0.0, help="EMA coefficient for purifier candidate potential scores.")
    parser.add_argument("--purifier-dynamic-stability-weight", type=float, default=0.0, help="Weight of cross-refresh stability score when ranking purifier candidate/full/claim targets. 0 preserves score-only ranking.")
    parser.add_argument("--purifier-dynamic-candidate-gain-weight", type=float, default=0.25, help="Weight of positive meta information gain in purifier candidate ranking.")
    parser.add_argument("--purifier-dynamic-restrict-to-inherited-targets", action="store_true", help="Ablation option: restrict purifier dynamic target selection to first-stage selected targets. By default all variable neurons are eligible.")
    parser.add_argument("--purifier-pin-inherited-targets", action="store_true", help="Keep every first-stage candidate in the purifier training pool even while its semantic-control gap is negative.")
    parser.add_argument("--purifier-inherited-min-weight", type=float, default=0.0, help="Minimum purifier training weight for pinned first-stage candidates.")
    parser.add_argument("--purifier-gate-prediction-on-strict-targets", action="store_true", help="Train only strict positive-gap targets after refresh; ignore min-selected fallback and set prediction weight to zero when no strict target remains.")
    parser.add_argument("--purifier-save-refresh-checkpoints", action="store_true", help="Save purifier states immediately before and after each dynamic target refresh epoch for leakage/prediction audits.")
    parser.add_argument("--purifier-final-checkpoint", choices=["target_count", "total_positive_gap", "residual_fraction", "balanced", "objective"], default="total_positive_gap", help="Purifier checkpoint used for final probes. balanced combines claim count, positive gap, and residual fraction.")
    parser.add_argument("--purifier-balanced-gap-weight", type=float, default=1.0, help="Weight of log positive residual-gap sum in balanced purifier checkpoint selection.")
    parser.add_argument("--purifier-balanced-fraction-weight", type=float, default=1.0, help="Weight of residual fraction in balanced purifier checkpoint selection.")
    parser.add_argument("--purifier-balanced-count-weight", type=float, default=0.25, help="Weight of normalized log claim count in balanced purifier checkpoint selection.")
    parser.add_argument("--purifier-balanced-min-claim-count", type=int, default=1, help="Minimum claim targets required before a balanced purifier checkpoint can be selected.")
    parser.add_argument("--purifier-checkpoint-rf-mode", choices=["positive_only", "net"], default="positive_only", help="Residual-fraction definition used for purifier checkpoint selection. net matches final information decomposition; positive_only preserves legacy behavior.")
    parser.add_argument("--purifier-balanced-min-meta-gain-sum", type=float, default=0.0, help="Minimum aggregate claim meta information gain in nats required by balanced checkpoint selection.")
    parser.add_argument("--recon-gate-next", type=float, default=0.03)
    parser.add_argument("--recon-gate-prev", type=float, default=0.03)
    parser.add_argument("--auto-recon-gate", action="store_true", help="Run a reconstruction-only calibration stage and use its best validation MSEs as recon gates.")
    parser.add_argument("--calibration-epochs", type=int, default=30, help="Epochs for reconstruction-only gate calibration.")
    parser.add_argument("--calibration-lr", type=float, default=0.0, help="Learning rate for gate calibration; 0 reuses --lr.")
    parser.add_argument("--calibration-batch-size", type=int, default=0, help="Batch size for gate calibration; 0 reuses --batch-size.")
    parser.add_argument("--calibration-latent-dim", type=int, default=0, help="Latent dim for gate calibration; 0 reuses --latent-dim.")
    parser.add_argument("--calibration-hidden-dim", type=int, default=0, help="Hidden dim for gate calibration; 0 reuses --hidden-dim.")
    parser.add_argument("--calibration-dropout", type=float, default=None, help="Dropout for gate calibration; omitted reuses --dropout.")
    parser.add_argument("--calibration-gate-factor", type=float, default=1.0, help="Multiplier applied to best calibration MSEs before setting recon gates.")
    parser.add_argument("--calibration-min-gate", type=float, default=0.0, help="Minimum recon gate after calibration.")
    parser.add_argument("--target-mode", choices=["continuous", "binary", "continuous_binary"], default="continuous")
    parser.add_argument("--binary-quantile", type=float, default=0.75)
    parser.add_argument("--binary-pred-threshold", type=float, default=0.5, help="Sigmoid probability threshold used to binarize predicted activations for binary-mode plots.")
    parser.add_argument("--binary-plot-samples", type=int, default=96, help="Max validation samples shown in binary activation comparison plots.")
    parser.add_argument("--binary-plot-features", type=int, default=256, help="Max neuron/features shown in binary activation comparison plots.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--compact-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show one compact progress bar per major stage and keep detailed epoch metrics in artifact files. Use --no-compact-progress for verbose JSON output.",
    )
    parser.add_argument("--log-batch-every", type=int, default=0, help="Write batch metrics every N steps; 0 disables batch-level logging.")
    parser.add_argument("--analysis-samples", type=int, default=512, help="Number of validation samples for latent/reconstruction analysis exports.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib plot generation.")
    parser.add_argument(
        "--extended-diagnostic-plots",
        action="store_true",
        help="Also emit legacy latent, hard-binary, and low-level purifier diagnostic plots.",
    )
    parser.add_argument("--save-analysis-tensors", action="store_true", help="Save validation latent/reconstruction tensors for later custom analysis.")
    parser.add_argument("--skip-binary-posthoc", action="store_true", help="Skip binary-mode posthoc probes, statistical tests, and per-neuron analysis.")
    parser.add_argument("--skip-information-analysis", action="store_true", help="Skip posthoc information-gain and semantic-leakage proxy analysis.")
    parser.add_argument("--information-alpha", type=float, default=0.05, help="Significance level for posthoc information-control hypothesis tests.")
    parser.add_argument("--semantic-oracle-max-pca-dim", type=int, default=128, help="Maximum PCA dimensions available to leakage-matched semantic oracle controls.")
    parser.add_argument("--semantic-oracle-budget-multiplier", type=float, default=1.0, help="Multiplier on measured E2 semantic leakage budget for semantic oracle controls; >1 gives semantic controls more information.")
    parser.add_argument("--semantic-oracle-selection", choices=["supervised", "variance"], default="supervised", help="How leakage-matched semantic oracle chooses PCA directions. supervised favors Yi-predictive semantic directions.")
    parser.add_argument("--probe-epochs", type=int, default=20, help="Epochs for posthoc linear probes.")
    parser.add_argument("--probe-batch-size", type=int, default=64, help="Batch size for posthoc linear probes.")
    parser.add_argument("--probe-lr", type=float, default=1e-3, help="Learning rate for posthoc linear probes.")
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4, help="Weight decay for posthoc linear probes.")
    parser.add_argument("--probe-max-train-samples", type=int, default=0, help="Maximum training examples for posthoc probes; 0 uses all.")
    parser.add_argument("--probe-max-val-samples", type=int, default=0, help="Maximum validation examples for posthoc probes; 0 uses all.")
    parser.add_argument("--bootstrap-samples", type=int, default=200, help="Bootstrap resamples for leakage and information-control hypothesis tests.")
    parser.add_argument("--permutation-tests", type=int, default=200, help="Sign-flip/permutation tests for information-control comparisons.")
    parser.add_argument("--min-neuron-activation-rate", type=float, default=0.05, help="Exclude neurons active below this rate from dynamic target eligibility and diagnostics.")
    parser.add_argument("--max-neuron-activation-rate", type=float, default=0.95, help="Exclude neurons active above this rate from dynamic target eligibility and diagnostics.")
    parser.add_argument(
        "--max-neuron-activation-rate-drift",
        type=float,
        default=1.0,
        help=(
            "Maximum absolute train/validation activation-rate difference for dynamic target "
            "eligibility. The compatibility default disables this cross-split stability gate."
        ),
    )
    parser.add_argument("--min-neuron-positive", type=int, default=20, help="Minimum positive validation samples for dynamic target eligibility.")
    parser.add_argument("--min-neuron-negative", type=int, default=20, help="Minimum negative validation samples for dynamic target eligibility.")
    parser.add_argument("--intervention-top-neurons", type=int, default=30, help="Number of semantic-gap dynamic targets exported for intervention.")
    args = parser.parse_args()
    set_compact_progress(args.compact_progress)

    torch.manual_seed(args.seed)
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.val_ratio < 0.0 or args.test_ratio < 0.0 or args.val_ratio + args.test_ratio >= 0.9:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and leave enough training data.")
    if args.main_semantic_pred_weight < -1.0:
        raise ValueError("--main-semantic-pred-weight must be >= -1.")
    if args.purifier_balanced_min_claim_count < 0:
        raise ValueError("--purifier-balanced-min-claim-count must be non-negative.")
    if not 0.0 <= args.purifier_dynamic_confirmation_fraction < 1.0:
        raise ValueError("--purifier-dynamic-confirmation-fraction must be in [0, 1).")
    if args.purifier_dynamic_confirmation_shortlist_k < 0:
        raise ValueError("--purifier-dynamic-confirmation-shortlist-k must be non-negative.")
    if args.purifier_dynamic_confirmation_probe_epochs < 0 or args.purifier_dynamic_confirmation_probe_repeats < 0:
        raise ValueError("Purifier confirmation probe epochs/repeats must be non-negative.")
    if args.purifier_dynamic_freeze_shortlist_epoch < 0:
        raise ValueError("--purifier-dynamic-freeze-shortlist-epoch must be non-negative.")
    if args.purifier_final_max_eval_samples < 0:
        raise ValueError("--purifier-final-max-eval-samples must be non-negative.")
    if args.purifier_balanced_min_meta_gain_sum < 0.0:
        raise ValueError("--purifier-balanced-min-meta-gain-sum must be non-negative.")
    if args.purifier_balanced_gap_weight < 0.0 or args.purifier_balanced_fraction_weight < 0.0 or args.purifier_balanced_count_weight < 0.0:
        raise ValueError("Balanced purifier checkpoint weights must be non-negative.")
    if not 0.0 < args.information_alpha < 1.0:
        raise ValueError("--information-alpha must be in (0, 1).")
    if args.semantic_oracle_max_pca_dim <= 0:
        raise ValueError("--semantic-oracle-max-pca-dim must be positive.")
    if args.semantic_oracle_budget_multiplier < 0.0:
        raise ValueError("--semantic-oracle-budget-multiplier must be non-negative.")
    if args.semantic_anchor_weights is not None and args.semantic_anchor_offsets is None:
        raise ValueError("--semantic-anchor-weights requires --semantic-anchor-offsets.")
    if args.dynamic_target_weighting and not is_binary_eval_mode(args.target_mode):
        raise ValueError("--dynamic-target-weighting currently requires --target-mode binary or continuous_binary.")
    if args.fixed_target_file and not is_binary_eval_mode(args.target_mode):
        raise ValueError("--fixed-target-file currently requires --target-mode binary or continuous_binary.")
    if args.fixed_target_file and args.dynamic_target_mode != "neuron":
        raise ValueError("--fixed-target-file currently supports neuron targets only.")
    if args.fixed_target_top_k < 0:
        raise ValueError("--fixed-target-top-k must be non-negative.")
    if args.fixed_target_min_weight < 0.0:
        raise ValueError("--fixed-target-min-weight must be non-negative.")
    if args.dynamic_target_mode == "direction" and args.target_mode != "continuous_binary":
        raise ValueError("--dynamic-target-mode direction currently requires --target-mode continuous_binary so continuous x_i is available for projections.")
    if args.dynamic_target_refresh_epochs <= 0:
        raise ValueError("--dynamic-target-refresh-epochs must be positive.")
    if args.dynamic_target_probe_epochs <= 0:
        raise ValueError("--dynamic-target-probe-epochs must be positive.")
    if args.dynamic_target_top_k < 0:
        raise ValueError("--dynamic-target-top-k must be non-negative.")
    if args.dynamic_target_min_selected < 0:
        raise ValueError("--dynamic-target-min-selected must be non-negative.")
    if args.dynamic_target_bootstrap_samples < 0:
        raise ValueError("--dynamic-target-bootstrap-samples must be non-negative.")
    if not 0.0 <= args.min_neuron_activation_rate < args.max_neuron_activation_rate <= 1.0:
        raise ValueError("Neuron activation-rate bounds must satisfy 0 <= min < max <= 1.")
    if not 0.0 <= args.max_neuron_activation_rate_drift <= 1.0:
        raise ValueError("--max-neuron-activation-rate-drift must be in [0, 1].")
    if args.main_checkpoint_start_epoch < 0:
        raise ValueError("--main-checkpoint-start-epoch must be non-negative.")
    if args.main_checkpoint_cov_weight < 0.0 or args.main_checkpoint_orth_weight < 0.0:
        raise ValueError("Main checkpoint dependence weights must be non-negative.")
    if args.main_checkpoint_z2_std_min < 0.0 or args.main_checkpoint_collapse_weight < 0.0:
        raise ValueError("Main checkpoint collapse controls must be non-negative.")
    if any(epoch <= 0 or epoch > args.epochs for epoch in args.main_checkpoint_ablation_epochs):
        raise ValueError("Every --main-checkpoint-ablation-epochs value must be in [1, --epochs].")
    if args.main_checkpoint_ablation_only and not args.main_checkpoint_ablation_epochs:
        raise ValueError("--main-checkpoint-ablation-only requires --main-checkpoint-ablation-epochs.")
    if args.main_checkpoint_ablation_epochs and not args.run_e2_recursive_purifier:
        raise ValueError("--main-checkpoint-ablation-epochs requires --run-e2-recursive-purifier.")
    if args.dynamic_target_require_positive_ci and args.dynamic_target_bootstrap_samples <= 0:
        raise ValueError("--dynamic-target-require-positive-ci requires --dynamic-target-bootstrap-samples > 0.")
    if not 0.0 <= args.dynamic_target_ema < 1.0:
        raise ValueError("--dynamic-target-ema must be in [0, 1).")
    if args.dynamic_target_candidate_top_k < 0:
        raise ValueError("--dynamic-target-candidate-top-k must be non-negative.")
    if not 0.0 <= args.dynamic_target_exploration_weight <= 1.0:
        raise ValueError("--dynamic-target-exploration-weight must be in [0, 1].")
    if not 0.0 <= args.dynamic_target_score_ema < 1.0:
        raise ValueError("--dynamic-target-score-ema must be in [0, 1).")
    if args.dynamic_target_stability_weight < 0.0:
        raise ValueError("--dynamic-target-stability-weight must be non-negative.")
    if args.dynamic_target_candidate_gain_weight < 0.0:
        raise ValueError("--dynamic-target-candidate-gain-weight must be non-negative.")
    if args.dynamic_target_confirm_evals <= 0 or args.dynamic_target_drop_patience_evals <= 0:
        raise ValueError("--dynamic-target-confirm-evals and --dynamic-target-drop-patience-evals must be positive.")
    if args.dynamic_target_mode == "direction" and args.dynamic_target_candidate_top_k > 0:
        raise ValueError("Persistent candidate tracking currently supports neuron targets only.")
    if args.dynamic_target_temperature <= 0.0:
        raise ValueError("--dynamic-target-temperature must be positive.")
    if args.dynamic_semantic_residual_rounds < 0:
        raise ValueError("--dynamic-semantic-residual-rounds must be non-negative.")
    if args.dynamic_semantic_residual_rank < 0:
        raise ValueError("--dynamic-semantic-residual-rank must be non-negative.")
    if args.dynamic_semantic_residual_min_sv_ratio < 0.0:
        raise ValueError("--dynamic-semantic-residual-min-sv-ratio must be non-negative.")
    if args.dynamic_direction_candidates <= 0:
        raise ValueError("--dynamic-direction-candidates must be positive.")
    if args.dynamic_direction_min_variance < 0.0:
        raise ValueError("--dynamic-direction-min-variance must be non-negative.")
    if args.purifier_epochs <= 0:
        raise ValueError("--purifier-epochs must be positive.")
    if args.purifier_pred_start_epoch <= 0:
        raise ValueError("--purifier-pred-start-epoch must be positive.")
    if args.purifier_pred_ramp_epochs < 0:
        raise ValueError("--purifier-pred-ramp-epochs must be non-negative.")
    if args.purifier_semantic_warmup_epochs < 0:
        raise ValueError("--purifier-semantic-warmup-epochs must be non-negative.")
    if args.purifier_semantic_z2_warmup_epochs < 0:
        raise ValueError("--purifier-semantic-z2-warmup-epochs must be non-negative.")
    if args.purifier_semantic_warmup_epochs + args.purifier_semantic_z2_warmup_epochs >= args.purifier_epochs:
        raise ValueError("Total purifier semantic warmup epochs must be smaller than --purifier-epochs.")
    if args.purifier_meta_input_mode == "semantic_residual" and args.lambda_purifier_semantic_recon_z2 <= 0.0:
        raise ValueError("--purifier-meta-input-mode semantic_residual requires --lambda-purifier-semantic-recon-z2 > 0.")
    if (
        args.purifier_code_dim < 0
        or args.purifier_semantic_dim < 0
        or args.purifier_meta_dim < 0
        or args.purifier_hidden_dim < 0
        or args.purifier_batch_size < 0
    ):
        raise ValueError("--purifier code/hidden/batch-size settings must be non-negative.")
    if args.purifier_max_train_samples < 0 or args.purifier_max_val_samples < 0:
        raise ValueError("--purifier max sample counts must be non-negative.")
    if args.purifier_probe_epochs < 0 or args.purifier_probe_hidden_dim < 0:
        raise ValueError("--purifier probe settings must be non-negative.")
    if args.purifier_dynamic_max_train_samples < 0 or args.purifier_dynamic_max_val_samples < 0:
        raise ValueError("--purifier dynamic probe sample limits must be non-negative.")
    if args.purifier_dynamic_confirm_evals <= 0 or args.purifier_dynamic_drop_patience_evals <= 0:
        raise ValueError("--purifier dynamic confirmation/drop patience must be positive.")
    if args.purifier_dynamic_target_weighting and not args.run_e2_recursive_purifier:
        raise ValueError("--purifier-dynamic-target-weighting requires --run-e2-recursive-purifier.")
    if args.purifier_dynamic_target_weighting and not is_binary_eval_mode(args.target_mode):
        raise ValueError("--purifier-dynamic-target-weighting currently requires --target-mode binary or continuous_binary.")
    if args.purifier_dynamic_start_epoch <= 0:
        raise ValueError("--purifier-dynamic-start-epoch must be positive.")
    if args.purifier_dynamic_refresh_epochs <= 0:
        raise ValueError("--purifier-dynamic-refresh-epochs must be positive.")
    if args.purifier_dynamic_probe_epochs < 0:
        raise ValueError("--purifier-dynamic-probe-epochs must be non-negative.")
    if args.purifier_dynamic_probe_repeats <= 0 or args.purifier_final_probe_repeats <= 0:
        raise ValueError("Purifier probe repeat counts must be positive.")
    if args.purifier_dynamic_probe_hidden_dim < 0:
        raise ValueError("--purifier-dynamic-probe-hidden-dim must be non-negative.")
    if args.purifier_dynamic_top_k < 0:
        raise ValueError("--purifier-dynamic-top-k must be non-negative.")
    if args.purifier_dynamic_train_top_k < 0:
        raise ValueError("--purifier-dynamic-train-top-k must be non-negative.")
    if args.purifier_dynamic_claim_top_k < 0:
        raise ValueError("--purifier-dynamic-claim-top-k must be non-negative.")
    if args.purifier_target_group_size < 0:
        raise ValueError("--purifier-target-group-size must be non-negative.")
    if args.purifier_dynamic_min_selected < 0:
        raise ValueError("--purifier-dynamic-min-selected must be non-negative.")
    if args.purifier_dynamic_temperature <= 0.0:
        raise ValueError("--purifier-dynamic-temperature must be positive.")
    if args.purifier_dynamic_min_soft_weight < 0.0:
        raise ValueError("--purifier-dynamic-min-soft-weight must be non-negative.")
    if args.purifier_dynamic_bootstrap_samples < 0:
        raise ValueError("--purifier-dynamic-bootstrap-samples must be non-negative.")
    if args.purifier_dynamic_require_positive_ci and args.purifier_dynamic_bootstrap_samples <= 0:
        raise ValueError("--purifier-dynamic-require-positive-ci requires --purifier-dynamic-bootstrap-samples > 0.")
    if not 0.0 <= args.purifier_dynamic_ema < 1.0:
        raise ValueError("--purifier-dynamic-ema must be in [0, 1).")
    if args.purifier_dynamic_candidate_top_k < 0:
        raise ValueError("--purifier-dynamic-candidate-top-k must be non-negative.")
    if not 0.0 <= args.purifier_dynamic_exploration_weight <= 1.0:
        raise ValueError("--purifier-dynamic-exploration-weight must be in [0, 1].")
    if not 0.0 <= args.purifier_dynamic_score_ema < 1.0:
        raise ValueError("--purifier-dynamic-score-ema must be in [0, 1).")
    if args.purifier_dynamic_stability_weight < 0.0:
        raise ValueError("--purifier-dynamic-stability-weight must be non-negative.")
    if args.purifier_dynamic_candidate_gain_weight < 0.0:
        raise ValueError("--purifier-dynamic-candidate-gain-weight must be non-negative.")
    if not 0.0 <= args.purifier_inherited_min_weight <= 1.0:
        raise ValueError("--purifier-inherited-min-weight must be in [0, 1].")
    if args.purifier_pin_inherited_targets and args.purifier_inherited_min_weight <= 0.0:
        raise ValueError("--purifier-pin-inherited-targets requires --purifier-inherited-min-weight > 0.")
    if args.dynamic_predict_on_semantic_residual and not args.dynamic_target_weighting:
        raise ValueError("--dynamic-predict-on-semantic-residual requires --dynamic-target-weighting.")
    if args.dynamic_predict_on_semantic_residual and args.dynamic_semantic_residual_rounds <= 0:
        raise ValueError("--dynamic-predict-on-semantic-residual requires --dynamic-semantic-residual-rounds > 0.")
    if args.probe_batch_size <= 0:
        args.probe_batch_size = args.batch_size
    os.makedirs(args.output_dir, exist_ok=True)

    reuse_first_stage_dir = os.path.abspath(args.reuse_first_stage_dir) if args.reuse_first_stage_dir else ""
    reuse_first_stage_config: Optional[Dict[str, Any]] = None
    reuse_first_stage_analysis: Optional[Dict[str, Any]] = None
    if reuse_first_stage_dir:
        required_reuse_files = (
            "best_representation_checkpoint.pt",
            "config.json",
            "analysis_summary.json",
        )
        missing_reuse_files = [
            name for name in required_reuse_files if not os.path.isfile(os.path.join(reuse_first_stage_dir, name))
        ]
        if missing_reuse_files:
            raise FileNotFoundError(
                f"--reuse-first-stage-dir={reuse_first_stage_dir} is missing: {', '.join(missing_reuse_files)}"
            )
        with open(os.path.join(reuse_first_stage_dir, "config.json"), "r", encoding="utf-8") as f:
            reuse_first_stage_config = json.load(f)
        with open(os.path.join(reuse_first_stage_dir, "analysis_summary.json"), "r", encoding="utf-8") as f:
            reuse_first_stage_analysis = json.load(f)

    activation_manifest = load_activation_manifest(args.activation_dir)
    prev_layer, layer_i, next_layer, layer_fracs, num_model_layers = resolve_layer_triplet(args, activation_manifest)
    args.layer_i = layer_i
    args.resolved_prev_layer = prev_layer
    args.resolved_next_layer = next_layer
    args.resolved_layer_fracs = layer_fracs
    args.num_model_layers = num_model_layers
    if prev_layer < 0:
        raise ValueError(f"Invalid prev layer {prev_layer}. Check --prev-layer or --prev-offset.")
    if next_layer <= args.layer_i:
        raise ValueError(f"Expected next layer to be greater than layer-i. Got layer-i={args.layer_i}, next_layer={next_layer}.")
    if prev_layer >= args.layer_i:
        raise ValueError(f"Expected prev layer to be smaller than layer-i. Got prev_layer={prev_layer}, layer-i={args.layer_i}.")
    if num_model_layers is not None and next_layer >= num_model_layers:
        raise ValueError(f"next_layer={next_layer} is out of range for num_model_layers={num_model_layers}.")
    ids_i, x_i = load_layer(os.path.join(args.activation_dir, f"layer_{args.layer_i:03d}.pt"))
    ids_next, x_next = load_layer(os.path.join(args.activation_dir, f"layer_{next_layer:03d}.pt"))
    ids_prev, x_prev, prev_mean, prev_std, semantic_anchor_meta = load_semantic_anchor_consensus(
        args.activation_dir,
        args.layer_i,
        prev_layer,
        args.semantic_anchor_offsets,
        args.semantic_anchor_weights,
    )
    if args.semantic_anchor_offsets:
        prev_layer = max(semantic_anchor_meta["layers"])
        args.prev_layer = prev_layer
        args.resolved_prev_layer = prev_layer
        args.semantic_anchor_offsets = semantic_anchor_meta["offsets"]
        args.semantic_anchor_weights = semantic_anchor_meta["weights"]
        if num_model_layers is not None:
            layer_fracs = [prev_layer / num_model_layers, args.layer_i / num_model_layers, next_layer / num_model_layers]
            args.resolved_layer_fracs = layer_fracs
    if ids_prev != ids_i or ids_i != ids_next:
        raise ValueError("Layer activation files have different example ids/order.")

    x_next, next_mean, next_std = standardize(x_next)
    x_i_raw = x_i
    binary_eval_meta = None
    if args.target_mode == "continuous":
        x_i, i_mean, i_std = standardize(x_i_raw)
        target_meta = {"mode": "continuous", "mean": i_mean, "std": i_std}
        dataset = TensorDataset(x_prev, x_i, x_next)
    elif args.target_mode == "binary":
        x_i, threshold = make_binary_targets(x_i_raw, args.binary_quantile)
        target_meta = {"mode": "binary", "binary_quantile": args.binary_quantile, "threshold": threshold}
        binary_eval_meta = {"threshold": threshold}
        dataset = TensorDataset(x_prev, x_i, x_next)
    elif args.target_mode == "continuous_binary":
        x_i_binary, threshold = make_binary_targets(x_i_raw, args.binary_quantile)
        x_i, i_mean, i_std = standardize(x_i_raw)
        binary_eval_meta = {"mean": i_mean, "std": i_std, "threshold": threshold}
        target_meta = {
            "mode": "continuous_binary",
            "train_target": "standardized_continuous",
            "eval_target": "hard_binary",
            "binary_quantile": args.binary_quantile,
            "mean": i_mean,
            "std": i_std,
            "threshold": threshold,
        }
        dataset = TensorDataset(x_prev, x_i, x_i_binary, x_next)
    else:
        raise ValueError(args.target_mode)

    dim = x_next.size(1)
    device = torch.device(args.device)
    train_ds, val_ds, test_ds, split_summary = split_dataset_with_manifest(dataset, ids_i, args)
    if reuse_first_stage_analysis is not None:
        expected_layers = {
            "prev_layer": prev_layer,
            "layer_i": args.layer_i,
            "next_layer": next_layer,
        }
        layer_mismatches = {
            key: (reuse_first_stage_analysis.get(key), value)
            for key, value in expected_layers.items()
            if int(reuse_first_stage_analysis.get(key, -1)) != int(value)
        }
        source_split = reuse_first_stage_analysis.get("split", {})
        split_keys = ("total_rows", "train_rows", "val_rows", "test_rows")
        split_mismatches = {
            key: (source_split.get(key), split_summary.get(key))
            for key in split_keys
            if source_split.get(key) != split_summary.get(key)
        }
        architecture_keys = ("latent_dim", "hidden_dim", "target_mode", "main_prediction_target")
        architecture_mismatches = {
            key: (reuse_first_stage_config.get(key), getattr(args, key))
            for key in architecture_keys
            if reuse_first_stage_config.get(key) != getattr(args, key)
        }
        if layer_mismatches or split_mismatches or architecture_mismatches:
            raise ValueError(
                "First-stage reuse is incompatible with the current run: "
                + json.dumps(
                    {
                        "layers": layer_mismatches,
                        "split": split_mismatches,
                        "architecture": architecture_mismatches,
                    },
                    ensure_ascii=False,
                )
            )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    with open(os.path.join(args.output_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    write_split_assignments(args.output_dir, ids_i, train_ds, val_ds, test_ds, args)

    gate_calibration_summary = None
    if reuse_first_stage_dir:
        source_gate_path = os.path.join(reuse_first_stage_dir, "gate_calibration", "gate_calibration_summary.json")
        if os.path.isfile(source_gate_path):
            with open(source_gate_path, "r", encoding="utf-8") as f:
                gate_calibration_summary = json.load(f)
            target_gate_dir = os.path.join(args.output_dir, "gate_calibration")
            os.makedirs(target_gate_dir, exist_ok=True)
            for name in ("gate_calibration_summary.json", "gate_calibration_history.csv", "gate_calibration.png"):
                source_path = os.path.join(reuse_first_stage_dir, "gate_calibration", name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, os.path.join(target_gate_dir, name))
        args.recon_gate_next = float(reuse_first_stage_analysis["recon_gate_next"])
        args.recon_gate_prev = float(reuse_first_stage_analysis["recon_gate_prev"])
        print(
            json.dumps(
                {
                    "reuse_first_stage": {
                        "source": reuse_first_stage_dir,
                        "recon_gate_next": args.recon_gate_next,
                        "recon_gate_prev": args.recon_gate_prev,
                        "main_training_skipped": True,
                    }
                },
                ensure_ascii=False,
            )
        )
    elif args.auto_recon_gate:
        gate_calibration_summary = run_recon_gate_calibration(train_ds, val_ds, dim, args, device, prev_layer, next_layer)
        args.recon_gate_next = gate_calibration_summary["calibrated_recon_gate_next"]
        args.recon_gate_prev = gate_calibration_summary["calibrated_recon_gate_prev"]
        if not compact_progress_enabled():
            print(
                json.dumps(
                    {
                        "auto_recon_gate": {
                            "recon_gate_next": args.recon_gate_next,
                            "recon_gate_prev": args.recon_gate_prev,
                            "source": "gate_calibration/gate_calibration_summary.json",
                        }
                    },
                    ensure_ascii=False,
                )
            )

    model = Decoupler(dim, args.latent_dim, args.hidden_dim, args.dropout, args.target_mode).to(device)
    main_parameters = [param for name, param in model.named_parameters() if not name.startswith("e2_prev_adversary.")]
    optimizer = torch.optim.AdamW(main_parameters, lr=args.lr, weight_decay=args.weight_decay)
    adv_enabled = args.lambda_e2_prev_adv > 0.0
    adv_optimizer = torch.optim.AdamW(model.e2_prev_adversary.parameters(), lr=args.e2_prev_adv_lr, weight_decay=args.weight_decay) if adv_enabled else None

    best_score = math.inf
    best_representation_score = math.inf
    best_representation_epoch: Optional[int] = None
    representation_checkpoint_path = os.path.join(args.output_dir, "best_representation_checkpoint.pt")
    checkpoint_ablation_epochs = set(args.main_checkpoint_ablation_epochs)
    checkpoint_ablation_checkpoint_dir = os.path.join(args.output_dir, "main_checkpoint_ablation", "checkpoints")
    if checkpoint_ablation_epochs:
        os.makedirs(checkpoint_ablation_checkpoint_dir, exist_ok=True)
    history = []
    batch_log_path = os.path.join(args.output_dir, "batch_metrics.csv")
    norm_meta = {
        "model_label": args.model_label,
        "activation_manifest": activation_manifest,
        "num_model_layers": num_model_layers,
        "layer_fractions": layer_fracs,
        "prev_layer": prev_layer,
        "layer_i": args.layer_i,
        "next_layer": next_layer,
        "prev_gap": args.layer_i - prev_layer,
        "next_gap": next_layer - args.layer_i,
        "prev_mean": prev_mean,
        "prev_std": prev_std,
        "semantic_anchors": semantic_anchor_meta,
        "next_mean": next_mean,
        "next_std": next_std,
        "target": target_meta,
        "split": split_summary,
        "main_prediction_target": args.main_prediction_target,
    }
    torch.save(norm_meta, os.path.join(args.output_dir, "normalization.pt"))
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    target_feature_weights: Optional[torch.Tensor] = None
    target_projection: Optional[torch.Tensor] = None
    semantic_residual_state: Optional[Dict[str, List[torch.Tensor]]] = None
    persistent_target_state: Optional[Dict[str, torch.Tensor]] = None
    dynamic_target_summary = {
        "epoch": 0,
        "target_type": args.dynamic_target_mode,
        "selected_count": 0 if args.dynamic_target_weighting else int(dim),
        "weight_sum": 0.0 if args.dynamic_target_weighting else float(dim),
        "prediction_weight_scale": 0.0 if args.dynamic_target_weighting else 1.0,
        "score_mean_selected": float("nan"),
        "e2_gain_nats_mean_selected": float("nan"),
        "e1_gain_nats_mean_selected": float("nan"),
        "semantic_gap_nats_mean_selected": float("nan"),
        "semantic_gain_nats_mean_selected": float("nan"),
        "semantic_gap_ci_low_mean_selected": float("nan"),
    }
    last_dynamic_target_refresh = 0
    fixed_target_summary = None
    fixed_target_tensors = None
    fixed_target_weights, fixed_target_summary, fixed_target_tensors = load_fixed_target_weights(args, x_i.size(1))
    if fixed_target_weights is not None:
        target_feature_weights = fixed_target_weights
        dynamic_target_summary.update(fixed_target_summary or {})
        save_dynamic_target_update(args.output_dir, fixed_target_tensors or {}, dynamic_target_summary)
        if not compact_progress_enabled():
            print(json.dumps({"fixed_target_weights": dynamic_target_summary}, ensure_ascii=False))
    if args.dynamic_target_weighting:
        dynamic_dir = os.path.join(args.output_dir, "dynamic_targets")
        os.makedirs(dynamic_dir, exist_ok=True)
        with open(os.path.join(dynamic_dir, "target_weight_history.jsonl"), "w", encoding="utf-8"):
            pass

    if reuse_first_stage_dir:
        source_bundle_path = os.path.join(reuse_first_stage_dir, "best_representation_checkpoint.pt")
        reused_bundle = torch.load(source_bundle_path, map_location="cpu")
        model.load_state_dict(reused_bundle["model_state_dict"])
        target_feature_weights = reused_bundle.get("target_feature_weights")
        target_projection = reused_bundle.get("target_projection")
        semantic_residual_state = reused_bundle.get("semantic_residual_state")
        dynamic_target_summary = reused_bundle.get("dynamic_target_summary", dynamic_target_summary)
        best_representation_epoch = int(reused_bundle.get("epoch", 0))
        best_representation_score = float(reused_bundle.get("representation_score", float("nan")))
        best_score = float(reuse_first_stage_analysis.get("final_best_constrained_score", float("nan")))
        torch.save(reused_bundle, representation_checkpoint_path)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

        source_history_jsonl = os.path.join(reuse_first_stage_dir, "history.jsonl")
        if os.path.isfile(source_history_jsonl):
            with open(source_history_jsonl, "r", encoding="utf-8") as f:
                history = [json.loads(line) for line in f if line.strip()]
            write_history_files(history, args.output_dir)
        with open(os.path.join(args.output_dir, "first_stage_reuse.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_dir": reuse_first_stage_dir,
                    "source_checkpoint": source_bundle_path,
                    "source_representation_epoch": best_representation_epoch,
                    "source_representation_score": best_representation_score,
                    "source_split": reuse_first_stage_analysis.get("split"),
                    "main_training_skipped": True,
                },
                f,
                indent=2,
            )

    main_progress = (
        _tqdm(
            range(1, 1 if reuse_first_stage_dir else args.epochs + 1),
            desc="Main decoupler",
            unit="epoch",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        if compact_progress_enabled()
        else range(1, 1 if reuse_first_stage_dir else args.epochs + 1)
    )
    for epoch in main_progress:
        model.train()
        running = 0.0
        train_totals = {
            "next": 0.0,
            "prev": 0.0,
            "orth": 0.0,
            "e2_prev_cov": 0.0,
            "e2_prev_adv_mse": 0.0,
            "e2_prev_adv_penalty": 0.0,
            "var": 0.0,
            "pred": 0.0,
            "semantic_pred": 0.0,
        }
        batch_rows = []
        regularizer_train_totals: Dict[str, float] = {}
        seen = 0
        val_probe = evaluate(
            model,
            val_loader,
            device,
            args.target_mode,
            args.binary_pred_threshold,
            adv_enabled,
            args.e2_prev_adv_margin,
            binary_eval_meta,
            semantic_residual_state if args.dynamic_predict_on_semantic_residual else None,
            cached_to_device(target_feature_weights, device) if target_feature_weights is not None else None,
            cached_to_device(target_projection, device) if target_projection is not None else None,
            regularizer_epoch=epoch,
            main_prediction_target=args.main_prediction_target,
        )
        gate_open = val_probe["next"] <= args.recon_gate_next and val_probe["prev"] <= args.recon_gate_prev
        if epoch < args.pred_ramp_start_epoch or not gate_open:
            pred_weight = 0.0
        else:
            ramp = min(1.0, (epoch - args.pred_ramp_start_epoch + 1) / 10.0)
            pred_weight = args.lambda_pred_max * ramp
        e2_prev_adv_weight = ramp_weight(args.lambda_e2_prev_adv, epoch, args.e2_prev_adv_start_epoch, args.e2_prev_adv_ramp_epochs)

        dynamic_start = dynamic_target_start_epoch(args)
        should_refresh_dynamic_targets = (
            args.dynamic_target_weighting
            and gate_open
            and epoch >= dynamic_start
            and (
                target_feature_weights is None
                or last_dynamic_target_refresh == 0
                or epoch - last_dynamic_target_refresh >= args.dynamic_target_refresh_epochs
            )
        )
        if should_refresh_dynamic_targets:
            raw_target_weights, dynamic_tensors, raw_dynamic_target_summary = compute_dynamic_target_weights(
                model,
                train_ds,
                val_ds,
                args,
                device,
                binary_eval_meta,
                None if persistent_target_policy_enabled(args) else target_feature_weights,
                epoch,
            )
            if persistent_target_policy_enabled(args):
                target_feature_weights, dynamic_tensors, dynamic_target_summary, persistent_target_state = apply_persistent_target_policy(
                    raw_target_weights,
                    dynamic_tensors,
                    raw_dynamic_target_summary,
                    persistent_target_state,
                    args,
                )
            else:
                target_feature_weights = raw_target_weights
                dynamic_target_summary = raw_dynamic_target_summary
                dynamic_target_summary["prediction_weight_scale"] = 1.0 if target_feature_weights.gt(args.dynamic_target_export_threshold).any() else 0.0
            last_dynamic_target_refresh = epoch
            if args.dynamic_predict_on_semantic_residual:
                semantic_residual_state = dynamic_tensors.get("semantic_residual_state")
            target_projection = dynamic_tensors.get("projection_matrix")
            save_dynamic_target_update(args.output_dir, dynamic_tensors, dynamic_target_summary)
            if not compact_progress_enabled():
                print(json.dumps({"dynamic_target_update": dynamic_target_summary}, ensure_ascii=False))
        pred_weight_scheduled = pred_weight
        if args.dynamic_target_weighting:
            if target_feature_weights is None:
                pred_weight = 0.0
            else:
                pred_weight *= float(dynamic_target_summary.get("prediction_weight_scale", 1.0))
        target_feature_weights_device = cached_to_device(target_feature_weights, device) if target_feature_weights is not None else None
        target_projection_device = cached_to_device(target_projection, device) if target_projection is not None else None

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1):
            x_prev_b, x_i_b, _, x_next_b = unpack_batch(batch)
            x_prev_b = move_to_device(x_prev_b, device)
            x_i_b = move_to_device(x_i_b, device)
            x_next_b = move_to_device(x_next_b, device)
            out = model(x_next_b)
            if args.dynamic_predict_on_semantic_residual and has_semantic_residual_state(semantic_residual_state):
                z2_signal = apply_semantic_residual_state(out["z2"], semantic_residual_state)
            else:
                z2_signal = out["z2"]
            x_i_hat_for_pred, x_i_semantic_hat, x_i_meta_hat = main_prediction_components(
                model,
                out["z1"],
                z2_signal,
                args.main_prediction_target,
            )

            if adv_enabled:
                set_requires_grad(model.e2_prev_adversary, True)
                adv_mse_train = torch.zeros((), device=device)
                for _ in range(max(1, args.e2_prev_adv_steps)):
                    adv_optimizer.zero_grad(set_to_none=True)
                    adv_pred = model.e2_prev_adversary(z2_signal.detach())
                    adv_loss = F.mse_loss(adv_pred, x_prev_b)
                    adv_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.e2_prev_adversary.parameters(), 1.0)
                    adv_optimizer.step()
                    adv_mse_train = adv_loss.detach()
                set_requires_grad(model.e2_prev_adversary, False)
                adv_pred_for_encoder = model.e2_prev_adversary(z2_signal)
                loss_e2_prev_adv_mse = F.mse_loss(adv_pred_for_encoder, x_prev_b)
                loss_e2_prev_adv = margin_adversary_penalty(loss_e2_prev_adv_mse, args.e2_prev_adv_margin)
            else:
                adv_mse_train = torch.tensor(float("nan"), device=device)
                loss_e2_prev_adv_mse = torch.tensor(float("nan"), device=device)
                loss_e2_prev_adv = torch.zeros((), device=device)

            loss_next = normalized_mse(out["x_next_hat"], x_next_b)
            loss_prev = normalized_mse(out["x_prev_hat"], x_prev_b)
            loss_orth = orthogonality_loss(out["z1"], z2_signal)
            loss_e2_prev = cross_cov_loss(z2_signal, x_prev_b)
            loss_var = variance_floor_loss(out["z1"]) + variance_floor_loss(z2_signal)
            loss_pred = weighted_prediction_loss(x_i_hat_for_pred, x_i_b, args.target_mode, target_feature_weights_device, target_projection_device)
            loss_semantic_pred = weighted_prediction_loss(x_i_semantic_hat, x_i_b, args.target_mode, target_feature_weights_device, target_projection_device)
            semantic_pred_weight = pred_weight if args.main_semantic_pred_weight < 0.0 else args.main_semantic_pred_weight
            if args.main_prediction_target != "residual":
                semantic_pred_weight = 0.0 if args.main_semantic_pred_weight < 0.0 else args.main_semantic_pred_weight
            extra_regularizer = first_stage_regularizer_metrics(out, z2_signal, x_prev_b, epoch, True)
            loss_extra_regularizer = extra_regularizer.get("penalty", loss_next.new_zeros(()))

            loss = (
                args.lambda_next * loss_next
                + args.lambda_prev * loss_prev
                + args.lambda_orth * loss_orth
                + args.lambda_e2_prev * loss_e2_prev
                + e2_prev_adv_weight * loss_e2_prev_adv
                + args.lambda_var * loss_var
                + pred_weight * loss_pred
                + semantic_pred_weight * loss_semantic_pred
                + loss_extra_regularizer
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x_next_b.size(0)
            running += loss.item() * bs
            train_totals["next"] += loss_next.item() * bs
            train_totals["prev"] += loss_prev.item() * bs
            train_totals["orth"] += loss_orth.item() * bs
            train_totals["e2_prev_cov"] += loss_e2_prev.item() * bs
            if adv_enabled:
                train_totals["e2_prev_adv_mse"] += loss_e2_prev_adv_mse.item() * bs
                train_totals["e2_prev_adv_penalty"] += loss_e2_prev_adv.item() * bs
            train_totals["var"] += loss_var.item() * bs
            train_totals["pred"] += loss_pred.item() * bs
            train_totals["semantic_pred"] = train_totals.get("semantic_pred", 0.0) + loss_semantic_pred.item() * bs
            for name, value in extra_regularizer.items():
                if name == "penalty":
                    continue
                regularizer_train_totals[name] = regularizer_train_totals.get(name, 0.0) + float(value.detach().item()) * bs
            seen += bs

            if args.log_batch_every > 0 and step % args.log_batch_every == 0:
                batch_rows.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": loss.item(),
                        "loss_next": loss_next.item(),
                        "loss_prev": loss_prev.item(),
                        "loss_orth": loss_orth.item(),
                        "loss_e2_prev_cov": loss_e2_prev.item(),
                        "loss_e2_prev_adv_mse": loss_e2_prev_adv_mse.item() if adv_enabled else float("nan"),
                        "loss_e2_prev_adv_penalty": loss_e2_prev_adv.item() if adv_enabled else float("nan"),
                        "adv_mse_train_detached_z2": adv_mse_train.item() if adv_enabled else float("nan"),
                        "e2_prev_adv_weight": e2_prev_adv_weight,
                        "loss_var": loss_var.item(),
                        "loss_pred": loss_pred.item(),
                        "loss_semantic_pred": loss_semantic_pred.item(),
                        "pred_weight": pred_weight,
                        "semantic_pred_weight": semantic_pred_weight,
                        "pred_weight_scheduled": pred_weight_scheduled,
                        "gate_open": gate_open,
                        "dynamic_target_count": dynamic_target_summary["selected_count"],
                        "dynamic_target_weight_sum": dynamic_target_summary["weight_sum"],
                        **{
                            f"train_{name}": float(value.detach().item())
                            for name, value in extra_regularizer.items()
                            if name != "penalty"
                        },
                    }
                )

        append_batch_metrics(batch_log_path, batch_rows)

        if adv_enabled:
            set_requires_grad(model.e2_prev_adversary, True)
        val = evaluate(
            model,
            val_loader,
            device,
            args.target_mode,
            args.binary_pred_threshold,
            adv_enabled,
            args.e2_prev_adv_margin,
            binary_eval_meta,
            semantic_residual_state if args.dynamic_predict_on_semantic_residual else None,
            target_feature_weights_device,
            target_projection_device,
            regularizer_epoch=epoch,
            main_prediction_target=args.main_prediction_target,
        )
        constrained_score = (
            val["next"]
            + val["prev"]
            + max(0.0, val["next"] - args.recon_gate_next) * 10.0
            + max(0.0, val["prev"] - args.recon_gate_prev) * 10.0
            + val.get("gamma_score_penalty", 0.0)
        )
        z2_collapse_penalty = max(0.0, args.main_checkpoint_z2_std_min - val["z2_std_mean"]) ** 2
        representation_score = (
            constrained_score
            + args.main_checkpoint_cov_weight * val["e2_prev_cov"]
            + args.main_checkpoint_orth_weight * val["orth"]
            + args.main_checkpoint_collapse_weight * z2_collapse_penalty
        )
        stats = TrainStats(
            epoch=epoch,
            train_loss=running / max(seen, 1),
            train_next_mse=train_totals["next"] / max(seen, 1),
            train_prev_mse=train_totals["prev"] / max(seen, 1),
            train_orth=train_totals["orth"] / max(seen, 1),
            train_e2_prev_cov=train_totals["e2_prev_cov"] / max(seen, 1),
            train_e2_prev_adv_mse=train_totals["e2_prev_adv_mse"] / max(seen, 1) if adv_enabled else float("nan"),
            train_e2_prev_adv_penalty=train_totals["e2_prev_adv_penalty"] / max(seen, 1) if adv_enabled else float("nan"),
            train_var_floor=train_totals["var"] / max(seen, 1),
            train_pred_loss=train_totals["pred"] / max(seen, 1),
            val_next_mse=val["next"],
            val_prev_mse=val["prev"],
            val_pred_loss=val["pred"],
            val_pred_aux=val["pred_aux"],
            val_orth=val["orth"],
            val_e2_prev_cov=val["e2_prev_cov"],
            val_e2_prev_adv_mse=val["e2_prev_adv_mse"],
            val_e2_prev_adv_penalty=val["e2_prev_adv_penalty"],
            val_z1_std_mean=val["z1_std_mean"],
            val_z2_std_mean=val["z2_std_mean"],
            val_z1_z2_cos_abs_mean=val["z1_z2_cos_abs_mean"],
            gate_open=gate_open,
            pred_weight=pred_weight,
            e2_prev_adv_weight=e2_prev_adv_weight,
            dynamic_target_count=int(dynamic_target_summary["selected_count"]),
            dynamic_target_weight_sum=float(dynamic_target_summary["weight_sum"]),
            dynamic_target_score_mean_selected=float(dynamic_target_summary["score_mean_selected"]),
            dynamic_target_e2_gain_mean_selected=float(dynamic_target_summary["e2_gain_nats_mean_selected"]),
            dynamic_target_e1_gain_mean_selected=float(dynamic_target_summary["e1_gain_nats_mean_selected"]),
            dynamic_target_semantic_gap_mean_selected=float(dynamic_target_summary.get("semantic_gap_nats_mean_selected", dynamic_target_summary["score_mean_selected"])),
            dynamic_target_semantic_gain_mean_selected=float(dynamic_target_summary.get("semantic_gain_nats_mean_selected", dynamic_target_summary["e1_gain_nats_mean_selected"])),
            dynamic_target_semantic_gap_ci_low_mean_selected=float(dynamic_target_summary.get("semantic_gap_ci_low_mean_selected", float("nan"))),
            dynamic_target_residual_fraction_mean_selected=float(dynamic_target_summary.get("dynamic_residual_fraction_mean_selected", float("nan"))),
            dynamic_target_leakage_explained_fraction_mean_selected=float(dynamic_target_summary.get("dynamic_leakage_explained_fraction_mean_selected", float("nan"))),
            constrained_score=constrained_score,
        )
        stats_row = asdict(stats)
        stats_row.update(
            {
                f"train_{name}": value / max(seen, 1)
                for name, value in regularizer_train_totals.items()
            }
        )
        stats_row.update(
            {
                "pred_weight_scheduled": pred_weight_scheduled,
                "train_semantic_pred_loss": train_totals["semantic_pred"] / max(seen, 1),
                "val_semantic_pred_loss": val.get("semantic_pred", float("nan")),
                "val_meta_pred_loss": val.get("meta_pred", float("nan")),
                "main_prediction_target": args.main_prediction_target,
                "representation_checkpoint_score": representation_score,
                "representation_z2_collapse_penalty": z2_collapse_penalty,
                "dynamic_target_candidate_count": int(dynamic_target_summary.get("candidate_count", 0)),
                "dynamic_target_confirmed_count": int(dynamic_target_summary.get("confirmed_count", 0)),
                "dynamic_target_confirmed_strict_count": int(dynamic_target_summary.get("confirmed_strict_count", 0)),
                "dynamic_target_full_weight_count": int(dynamic_target_summary.get("full_weight_count", dynamic_target_summary.get("selected_count", 0))),
                "dynamic_target_claim_count": int(dynamic_target_summary.get("claim_target_count", dynamic_target_summary.get("strict_eligible_count", 0))),
                "dynamic_target_stability_score_mean_selected": float(dynamic_target_summary.get("target_stability_score_mean_selected", float("nan"))),
                "dynamic_target_positive_rate_mean_selected": float(dynamic_target_summary.get("target_positive_rate_mean_selected", float("nan"))),
                "dynamic_target_selection_rate_mean_selected": float(dynamic_target_summary.get("target_selection_rate_mean_selected", float("nan"))),
                "dynamic_target_gap_std_mean_selected": float(dynamic_target_summary.get("target_gap_std_mean_selected", float("nan"))),
            }
        )
        stats_row.update(
            {
                f"val_{name}": value
                for name, value in val.items()
                if name.startswith("gamma_")
            }
        )
        for key in (
            "semantic_residual_rounds_applied",
            "semantic_residual_removed_dims",
            "semantic_residual_train_l2_ratio",
            "semantic_residual_val_l2_ratio",
        ):
            if key in dynamic_target_summary:
                stats_row[key] = dynamic_target_summary[key]
        history.append(stats_row)
        if compact_progress_enabled():
            main_progress.set_postfix(
                next=f"{val['next']:.4f}",
                prev=f"{val['prev']:.4f}",
                cand=int(dynamic_target_summary.get("candidate_count", 0)),
                conf=int(dynamic_target_summary.get("confirmed_count", 0)),
                claim=int(dynamic_target_summary.get("claim_target_count", dynamic_target_summary.get("strict_eligible_count", 0))),
                gap=f"{float(dynamic_target_summary.get('semantic_gap_nats_mean_selected', float('nan'))):.4g}",
                stab=f"{float(dynamic_target_summary.get('target_stability_score_mean_selected', float('nan'))):.3g}",
                pred=f"{pred_weight:.3g}",
                refresh=False,
            )
        else:
            print(json.dumps(stats_row, ensure_ascii=False))

        if constrained_score < best_score:
            best_score = constrained_score
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

        checkpoint_start = args.main_checkpoint_start_epoch if args.main_checkpoint_start_epoch > 0 else 1
        if epoch >= checkpoint_start and representation_score < best_representation_score:
            best_representation_score = representation_score
            best_representation_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_feature_weights": target_feature_weights,
                    "target_projection": target_projection,
                    "semantic_residual_state": semantic_residual_state,
                    "dynamic_target_summary": dynamic_target_summary,
                    "epoch": epoch,
                    "representation_score": representation_score,
                    "stats": stats_row,
                },
                representation_checkpoint_path,
            )

        if epoch in checkpoint_ablation_epochs:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_feature_weights": target_feature_weights,
                    "target_projection": target_projection,
                    "semantic_residual_state": semantic_residual_state,
                    "dynamic_target_summary": dynamic_target_summary,
                    "epoch": epoch,
                    "representation_score": representation_score,
                    "stats": stats_row,
                },
                os.path.join(checkpoint_ablation_checkpoint_dir, f"epoch_{epoch:03d}.pt"),
            )

        write_history_files(history, args.output_dir)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pt"))
    selected_main_checkpoint = "last"
    selected_main_bundle = {
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "target_feature_weights": target_feature_weights,
        "target_projection": target_projection,
        "semantic_residual_state": semantic_residual_state,
        "dynamic_target_summary": dynamic_target_summary,
        "epoch": args.epochs,
        "representation_score": history[-1].get("representation_checkpoint_score") if history else None,
        "stats": history[-1] if history else {},
    }
    if args.main_final_checkpoint == "representation":
        if not os.path.isfile(representation_checkpoint_path):
            raise RuntimeError("No eligible first-stage representation checkpoint was saved.")
        representation_bundle = torch.load(representation_checkpoint_path, map_location=device)
        model.load_state_dict(representation_bundle["model_state_dict"])
        target_feature_weights = representation_bundle.get("target_feature_weights")
        target_projection = representation_bundle.get("target_projection")
        semantic_residual_state = representation_bundle.get("semantic_residual_state")
        dynamic_target_summary = representation_bundle.get("dynamic_target_summary", dynamic_target_summary)
        selected_main_bundle = representation_bundle
        selected_main_checkpoint = "representation"
    analysis_tensors = collect_analysis_tensors(
        model,
        val_loader,
        device,
        args.analysis_samples,
        args.target_mode,
        binary_eval_meta,
        semantic_residual_state if args.dynamic_predict_on_semantic_residual else None,
        args.main_prediction_target,
    )
    binary_posthoc_summary = run_binary_posthoc_analysis(
        model,
        train_ds,
        val_ds,
        args,
        device,
        binary_eval_meta,
        ids_i,
        prev_layer,
        next_layer,
        semantic_residual_state if args.dynamic_predict_on_semantic_residual else None,
    )
    purifier_summary = None
    if not args.main_checkpoint_ablation_only:
        purifier_summary = run_e2_recursive_purifier(
            model,
            train_ds,
            val_ds,
            args,
            device,
            binary_eval_meta,
            semantic_residual_state if args.dynamic_predict_on_semantic_residual else None,
            target_feature_weights if args.dynamic_target_mode == "neuron" else None,
            test_ds,
        )
    checkpoint_ablation_summary = None
    if checkpoint_ablation_epochs:
        checkpoint_ablation_summary = run_main_checkpoint_purifier_ablation(
            model,
            train_ds,
            val_ds,
            args,
            device,
            binary_eval_meta,
            representation_checkpoint_path,
            checkpoint_ablation_checkpoint_dir,
            selected_main_bundle,
            test_ds,
        )
    analysis_summary = {
        "analysis_samples": int(next(iter(analysis_tensors.values())).size(0)) if analysis_tensors else 0,
        "model_label": args.model_label,
        "activation_manifest": activation_manifest,
        "num_model_layers": num_model_layers,
        "layer_fractions": layer_fracs,
        "semantic_anchors": {
            "mode": semantic_anchor_meta["mode"],
            "offsets": semantic_anchor_meta["offsets"],
            "layers": semantic_anchor_meta["layers"],
            "weights": semantic_anchor_meta["weights"],
        },
        "target_mode": args.target_mode,
        "main_prediction_target": args.main_prediction_target,
        "split": split_summary,
        "prev_layer": prev_layer,
        "layer_i": args.layer_i,
        "next_layer": next_layer,
        "prev_gap": args.layer_i - prev_layer,
        "next_gap": next_layer - args.layer_i,
        "recon_gate_next": args.recon_gate_next,
        "recon_gate_prev": args.recon_gate_prev,
        "gate_calibration": gate_calibration_summary,
        "binary_pred_threshold": args.binary_pred_threshold if is_binary_eval_mode(args.target_mode) else None,
        "final_best_constrained_score": best_score,
        "main_checkpoint": {
            "requested": args.main_final_checkpoint,
            "used": selected_main_checkpoint,
            "best_representation_epoch": best_representation_epoch,
            "best_representation_score": best_representation_score,
        },
        "dynamic_semantic_residual": {
            "target_mode": args.dynamic_target_mode,
            "direction_source": args.dynamic_direction_source if args.dynamic_target_mode == "direction" else None,
            "direction_candidates": args.dynamic_direction_candidates if args.dynamic_target_mode == "direction" else None,
            "rounds": args.dynamic_semantic_residual_rounds,
            "rank": args.dynamic_semantic_residual_rank,
            "min_sv_ratio": args.dynamic_semantic_residual_min_sv_ratio,
            "predict_on_semantic_residual": args.dynamic_predict_on_semantic_residual,
            "state_available": has_semantic_residual_state(semantic_residual_state),
            "basis_count": len(semantic_residual_state.get("bases", [])) if semantic_residual_state else 0,
            "latest_removed_dims": dynamic_target_summary.get("semantic_residual_removed_dims"),
            "latest_val_l2_ratio": dynamic_target_summary.get("semantic_residual_val_l2_ratio"),
            "training_signal": "z2_clean" if args.dynamic_predict_on_semantic_residual else "z2_raw",
            "note": "When predict_on_semantic_residual is true, prediction and E2 regularization losses use z2_clean = multi-round semantic residualization of raw z2.",
        },
        "outputs": {
            "history_jsonl": "history.jsonl",
            "history_csv": "history.csv",
            "split_manifest": "split_manifest.json",
            "split_assignments": "split_assignments.csv",
            "batch_metrics_csv": "batch_metrics.csv" if os.path.exists(batch_log_path) else None,
            "plots_dir": "plots" if not args.no_plots else None,
            "analysis_tensors": "analysis_tensors.pt" if args.save_analysis_tensors else None,
            "binary_analysis": "binary_analysis/binary_analysis_summary.json" if binary_posthoc_summary else None,
            "intervention_targets": "binary_analysis/intervention_targets.json" if binary_posthoc_summary else None,
            "dynamic_target_weights": "dynamic_targets/target_weights_latest.pt" if args.dynamic_target_weighting else None,
            "dynamic_target_weights_csv": "dynamic_targets/target_weights_latest.csv" if args.dynamic_target_weighting else None,
            "e2_recursive_purifier": "e2_recursive_purifier/summary.json" if purifier_summary else None,
            "main_checkpoint_ablation": "main_checkpoint_ablation/summary.json" if checkpoint_ablation_summary else None,
        },
    }
    if purifier_summary is not None:
        analysis_summary["e2_recursive_purifier"] = purifier_summary
    if checkpoint_ablation_summary is not None:
        analysis_summary["main_checkpoint_ablation"] = checkpoint_ablation_summary
    with open(os.path.join(args.output_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)
    if args.save_analysis_tensors:
        torch.save(analysis_tensors, os.path.join(args.output_dir, "analysis_tensors.pt"))
    if not args.no_plots:
        save_plots(
            history,
            analysis_tensors,
            args.output_dir,
            args.target_mode,
            args.binary_pred_threshold,
            args.binary_plot_samples,
            args.binary_plot_features,
            args.extended_diagnostic_plots,
        )
    print(f"Saved training artifacts to {args.output_dir}")


if __name__ == "__main__":
    raise SystemExit(
        "train_decoupler.py is an internal compatibility library. "
        "Use scripts/run_experiment.sh."
    )
