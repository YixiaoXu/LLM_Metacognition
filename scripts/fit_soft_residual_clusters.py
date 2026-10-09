#!/usr/bin/env python
"""Fit semantic-residual activation probes and produce fixed cluster artifacts.

This is deliberately posthoc: it never updates the decoupler or purifier.  A
semantic teacher predicts candidate layer-i activation labels from E1 and the
previous layer.  A second head predicts an additive logit correction from the
purified-meta code.  Clustering uses the continuous correction logits, not the
raw purified-meta vector or hard activation labels.
"""

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cluster_intervention import (
    centroid_silhouette_score,
    Decoupler,
    E2RecursivePurifier,
    infer_decoupler_arch,
    infer_purifier_arch,
    load_activation_ids,
    load_layer_matrix,
    raw_centroids_from_labels,
    read_json,
    resolve_purifier_checkpoint,
    run_kmeans,
    standardize_next_features,
)
from intervene_activations import safe_torch_load, write_csv
from publication_plot_style import (
    COLORS,
    DOUBLE_COLUMN_IN,
    apply_nmi_style,
    cluster_color,
    panel_label,
    save_figure,
    style_axis,
)


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecouplerFinalProbeMLP(nn.Module):
    """Probe architecture used by train_decoupler's final purifier audit."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_decoupler_split_indices(
    decoupler_dir: str,
    all_ids: Sequence[str],
    config: Dict[str, Any],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Reuse the exact decoupler split instead of reconstructing it approximately."""
    assignment_path = os.path.join(decoupler_dir, "split_assignments.csv")
    id_to_index = {str(sample_id): idx for idx, sample_id in enumerate(all_ids)}
    split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    missing_ids: List[str] = []

    if os.path.isfile(assignment_path):
        with open(assignment_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                split = str(row.get("split", "")).strip()
                sample_id = str(row.get("id", ""))
                if split not in split_indices:
                    continue
                idx = id_to_index.get(sample_id)
                if idx is None:
                    missing_ids.append(sample_id)
                    continue
                split_indices[split].append(idx)
        if not split_indices["train"] or not split_indices["val"]:
            raise ValueError(
                f"Invalid split assignments in {assignment_path}: "
                f"train={len(split_indices['train'])}, val={len(split_indices['val'])}."
            )
        source = "split_assignments.csv"
    else:
        if bool(config.get("split_by_base_id", False)):
            raise FileNotFoundError(
                f"{assignment_path} is required because the decoupler used --split-by-base-id."
            )
        n = len(all_ids)
        val_size = max(1, int(n * float(config.get("val_ratio", 0.1))))
        test_ratio = float(config.get("test_ratio", 0.0))
        test_size = max(1, int(n * test_ratio)) if test_ratio > 0.0 else 0
        train_size = n - val_size - test_size
        if train_size <= 0:
            raise ValueError("Saved train/val/test ratios leave no probe-training samples.")
        generator = torch.Generator().manual_seed(int(config.get("split_seed", config.get("seed", seed))))
        permutation = torch.randperm(n, generator=generator).tolist()
        split_indices["train"] = permutation[:train_size]
        split_indices["val"] = permutation[train_size : train_size + val_size]
        split_indices["test"] = permutation[train_size + val_size :]
        source = "reconstructed_row_random"

    tensors = {
        name: torch.tensor(indices, dtype=torch.long)
        for name, indices in split_indices.items()
    }
    summary = {
        "source": source,
        "assignment_path": assignment_path if os.path.isfile(assignment_path) else None,
        "train_n": int(tensors["train"].numel()),
        "val_n": int(tensors["val"].numel()),
        "test_n": int(tensors["test"].numel()),
        "missing_assignment_ids": len(missing_ids),
    }
    return tensors["train"], tensors["val"], tensors["test"], summary


def read_jsonl_by_id(path: str) -> Dict[str, Dict[str, Any]]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" in item:
                rows[str(item["id"])] = item
    return rows


def parse_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "safe"}:
        return True
    if text in {"false", "0", "no", "unsafe"}:
        return False
    return None


def active_categories(value: Any) -> List[str]:
    if isinstance(value, dict):
        active = [str(key) for key, flag in value.items() if parse_bool(flag) is True]
        return active if active else ["none"]
    if isinstance(value, list):
        return [str(item) for item in value] if value else ["none"]
    if value is None or value == "":
        return ["unknown"]
    return [str(value)]


def chi_square_binary_by_cluster(labels: List[int], positives: List[bool]) -> float:
    clusters = sorted(set(labels))
    observed = []
    for cluster in clusters:
        idx = [i for i, label in enumerate(labels) if label == cluster]
        safe = sum(1 for i in idx if positives[i])
        unsafe = len(idx) - safe
        observed.append([safe, unsafe])
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(row[col] for row in observed) for col in range(2)]
    total = sum(row_totals)
    stat = 0.0
    for row_idx, row in enumerate(observed):
        for col_idx, value in enumerate(row):
            expected = row_totals[row_idx] * col_totals[col_idx] / max(total, 1)
            if expected > 0:
                stat += (value - expected) ** 2 / expected
    return float(stat)


def mutual_information_bits(labels: List[int], positives: List[bool]) -> float:
    total = len(labels)
    if total == 0:
        return float("nan")
    clusters = sorted(set(labels))
    mi = 0.0
    for cluster in clusters:
        p_c = sum(1 for label in labels if label == cluster) / total
        for value in [False, True]:
            joint = sum(1 for label, pos in zip(labels, positives) if label == cluster and pos == value) / total
            if joint <= 0:
                continue
            p_s = sum(1 for pos in positives if pos == value) / total
            mi += joint * math.log(joint / max(p_c * p_s, 1e-12), 2)
    return float(mi)


def compute_safety_association(
    assignment_rows: List[Dict[str, Any]],
    data_path: str,
    permutation_tests: int,
    seed: int,
    output_dir: str,
    no_plots: bool,
) -> Dict[str, Any]:
    data_by_id = read_jsonl_by_id(data_path)
    usable = []
    for row in assignment_rows:
        item = data_by_id.get(str(row["id"]))
        if not item:
            continue
        safe = parse_bool(item.get("is_safe"))
        if safe is None:
            continue
        row["is_safe"] = bool(safe)
        row["safety_label"] = "safe" if safe else "unsafe"
        categories = active_categories(item.get("category"))
        row["safety_categories"] = " ".join(categories)
        usable.append(row)
    if not usable:
        return {"available": False, "reason": "no_rows_with_is_safe"}

    labels = [int(row["cluster"]) for row in usable]
    positives = [bool(row["is_safe"]) for row in usable]
    clusters = sorted(set(labels))
    cluster_rows = []
    for cluster in clusters:
        group = [row for row in usable if int(row["cluster"]) == cluster]
        safe_n = sum(1 for row in group if bool(row["is_safe"]))
        unsafe_n = len(group) - safe_n
        cluster_rows.append(
            {
                "cluster": cluster,
                "n": len(group),
                "safe_n": safe_n,
                "unsafe_n": unsafe_n,
                "safe_rate": safe_n / max(len(group), 1),
                "unsafe_rate": unsafe_n / max(len(group), 1),
            }
        )

    category_counts: Dict[Tuple[int, str], int] = {}
    for row in usable:
        cluster = int(row["cluster"])
        for category in str(row.get("safety_categories", "unknown")).split():
            category_counts[(cluster, category)] = category_counts.get((cluster, category), 0) + 1
    category_rows = []
    for (cluster, category), count in sorted(category_counts.items(), key=lambda item: (item[0][0], item[0][1])):
        n = next((row["n"] for row in cluster_rows if row["cluster"] == cluster), 1)
        category_rows.append({"cluster": cluster, "category": category, "count": count, "rate": count / max(n, 1)})

    observed_stat = chi_square_binary_by_cluster(labels, positives)
    rng = random.Random(seed)
    extreme = 0
    shuffled = positives[:]
    for _ in range(max(0, permutation_tests)):
        rng.shuffle(shuffled)
        if chi_square_binary_by_cluster(labels, shuffled) >= observed_stat - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (max(0, permutation_tests) + 1)
    mi = mutual_information_bits(labels, positives)
    majority_baseline = max(sum(positives), len(positives) - sum(positives)) / len(positives)
    cluster_majority_accuracy = sum(max(row["safe_n"], row["unsafe_n"]) for row in cluster_rows) / len(usable)

    write_csv(os.path.join(output_dir, "cluster_is_safe_association.csv"), cluster_rows)
    write_csv(os.path.join(output_dir, "cluster_safety_category_counts.csv"), category_rows)
    summary = {
        "available": True,
        "data_path": data_path,
        "n": len(usable),
        "cluster_rows": cluster_rows,
        "chi_square": observed_stat,
        "permutation_tests": permutation_tests,
        "permutation_p": p_value,
        "mutual_information_bits": mi,
        "majority_baseline_accuracy": majority_baseline,
        "cluster_majority_accuracy": cluster_majority_accuracy,
        "category_rows_csv": "cluster_safety_category_counts.csv",
        "association_csv": "cluster_is_safe_association.csv",
    }
    write_json(os.path.join(output_dir, "cluster_is_safe_association.json"), summary)
    if not no_plots:
        maybe_plot_safety_association(output_dir, cluster_rows, category_rows)
    return summary


def maybe_plot_safety_association(output_dir: str, cluster_rows: List[Dict[str, Any]], category_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    apply_nmi_style()
    plots = os.path.join(output_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    clusters = [int(row["cluster"]) for row in cluster_rows]
    unsafe_rates = [float(row["unsafe_rate"]) for row in cluster_rows]
    figure, axis = plt.subplots(figsize=(3.5, 2.8))
    axis.bar(
        [str(c) for c in clusters],
        unsafe_rates,
        color=[cluster_color(c) for c in clusters],
        width=0.62,
        zorder=3,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Dataset unsafe rate")
    axis.set_title("Cluster association with safety label", loc="left")
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, os.path.join(plots, "cluster_is_safe_unsafe_rate"))
    plt.close(figure)

    categories = sorted({str(row["category"]) for row in category_rows})
    if categories:
        matrix = torch.zeros(len(clusters), len(categories))
        cluster_to_idx = {cluster: idx for idx, cluster in enumerate(clusters)}
        category_to_idx = {category: idx for idx, category in enumerate(categories)}
        for row in category_rows:
            matrix[cluster_to_idx[int(row["cluster"])], category_to_idx[str(row["category"])]] = float(row["rate"])
        figure, axis = plt.subplots(figsize=(max(DOUBLE_COLUMN_IN, 0.34 * len(categories)), max(2.5, 0.38 * len(clusters) + 1.2)))
        image = axis.imshow(matrix.numpy(), aspect="auto", cmap="viridis", vmin=0.0)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label("Category rate within cluster")
        axis.set_xticks(range(len(categories)), categories, rotation=40, ha="right")
        axis.set_yticks(range(len(clusters)), [str(c) for c in clusters])
        axis.set_xlabel("Safety category")
        axis.set_ylabel("Cluster")
        figure.tight_layout()
        save_figure(figure, os.path.join(plots, "cluster_safety_category_heatmap"))
        plt.close(figure)


def unwrap_state(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        state = obj.get("model_state_dict") or obj.get("state_dict") or obj
    else:
        state = obj
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state_dict.")
    return state


def resolve_candidate_csv(args: argparse.Namespace) -> str:
    if args.candidate_csv:
        return args.candidate_csv
    purifier_dir = os.path.join(args.decoupler_dir, "e2_recursive_purifier")
    candidates = [
        os.path.join(purifier_dir, "purifier_training_candidate_pool.csv"),
        os.path.join(purifier_dir, "purifier_full_training_pool_information.csv"),
        os.path.join(purifier_dir, "purifier_per_target_information.csv"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    raise FileNotFoundError(
        "No purifier candidate CSV found. Expected one of: "
        + ", ".join(candidates)
    )


def read_candidate_rows(path: str, top_k: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in (
            "residual_gap_bits",
            "meta_info_bits_vs_prior",
            "semantic_info_bits_vs_prior",
            "training_weight",
            "claim_weight",
            "candidate_priority",
        ):
            try:
                row[key] = float(row.get(key, "nan"))
            except (TypeError, ValueError):
                row[key] = float("nan")
        row["neuron"] = int(row["neuron"])
        total = float(row.get("meta_info_bits_vs_prior", float("nan")))
        residual = float(row.get("residual_gap_bits", float("nan")))
        row["source_residual_fraction"] = residual / total if math.isfinite(total) and total > 1e-8 else float("nan")
    def rank_value(row: Dict[str, Any], key: str) -> float:
        value = float(row.get(key, float("nan")))
        return value if math.isfinite(value) else float("-inf")

    rows.sort(
        key=lambda row: (
            rank_value(row, "candidate_priority"),
            rank_value(row, "claim_weight"),
            rank_value(row, "training_weight"),
            rank_value(row, "residual_gap_bits"),
        ),
        reverse=True,
    )
    if top_k > 0:
        rows = rows[:top_k]
    if not rows:
        raise ValueError(f"No candidate targets found in {path}")
    return rows


def load_models(args: argparse.Namespace, next_dim: int, device: torch.device):
    config_path = os.path.join(args.decoupler_dir, "config.json")
    config = read_json(config_path)
    checkpoint_path = args.decoupler_checkpoint or os.path.join(args.decoupler_dir, "best_model.pt")
    normalization_path = args.decoupler_normalization or os.path.join(args.decoupler_dir, "normalization.pt")
    state = unwrap_state(safe_torch_load(checkpoint_path))
    latent_dim, hidden_dim, dropout, target_mode = infer_decoupler_arch(state, next_dim, config)
    decoupler = Decoupler(next_dim, latent_dim, hidden_dim, dropout, target_mode).to(device)
    decoupler.load_state_dict(state, strict=False)
    decoupler.eval()

    # ``main_direct`` intentionally skips the second-stage purifier.  Its
    # downstream representation is the raw E2 code, so posthoc clustering must
    # not require a nonexistent ``best_purifier.pt``.
    requested_purifier = str(args.purifier_checkpoint or "").strip()
    default_purifier = os.path.join(args.decoupler_dir, "e2_recursive_purifier", "best_purifier.pt")
    purifier_path: Optional[str] = requested_purifier or default_purifier
    if not os.path.exists(purifier_path):
        if config.get("second_stage") == "main_direct" and not requested_purifier:
            purifier_path = None
            print(
                "[posthoc] second_stage=main_direct has no purifier; using raw E2 as meta code",
                flush=True,
            )
        else:
            # Preserve the existing error path for purifier-based runs and for
            # an explicitly requested missing checkpoint.
            args.purifier_checkpoint = None
            purifier_path = resolve_purifier_checkpoint(args)

    purifier = None
    meta_dim = None
    if purifier_path is not None:
        purifier_state = unwrap_state(safe_torch_load(purifier_path))
        p_arch = infer_purifier_arch(purifier_state, config)
        z2_dim, semantic_dim, meta_dim, p_hidden, z1_dim, prev_dim, target_dim, p_dropout = p_arch
        purifier = E2RecursivePurifier(
            z2_dim, semantic_dim, meta_dim, p_hidden, p_dropout, z1_dim, prev_dim, target_dim
        ).to(device)
        purifier.load_state_dict(purifier_state, strict=False)
        purifier.eval()
    normalization = safe_torch_load(normalization_path)
    info = {
        "config_path": config_path,
        "decoupler_checkpoint": checkpoint_path,
        "normalization_path": normalization_path,
        "purifier_checkpoint": purifier_path,
        "latent_dim": latent_dim,
        "purifier_meta_dim": meta_dim if meta_dim is not None else latent_dim,
        "meta_source": "purified_meta" if purifier is not None else "raw_e2_main_direct",
        "target_mode": target_mode,
    }
    return decoupler, purifier, normalization, config, info


def encode_partition(
    activation_dir: str,
    ids: Sequence[str],
    prev_layer: int,
    layer_i: int,
    next_layer: int,
    candidate_neurons: torch.Tensor,
    decoupler: Decoupler,
    purifier: Optional[E2RecursivePurifier],
    normalization: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    label: str,
) -> Dict[str, torch.Tensor]:
    x_next = load_layer_matrix(activation_dir, next_layer, ids, None)
    x_next = standardize_next_features(x_next, normalization)
    z1_parts, meta_parts = [], []
    for start in tqdm(range(0, x_next.shape[0], batch_size), desc=f"Encode {label}"):
        batch = x_next[start : start + batch_size].to(device)
        with torch.inference_mode():
            z1 = decoupler.e1(batch)
            z2 = decoupler.e2(batch)
            meta = z2 if purifier is None else purifier(z2)["meta"]
        z1_parts.append(z1.float().cpu())
        meta_parts.append(meta.float().cpu())
    del x_next

    x_prev = load_layer_matrix(activation_dir, prev_layer, ids, None).float()
    prev_mean = normalization["prev_mean"].float().view(1, -1)
    prev_std = normalization["prev_std"].float().view(1, -1).clamp_min(1e-6)
    x_prev = (x_prev - prev_mean) / prev_std

    x_i = load_layer_matrix(activation_dir, layer_i, ids, candidate_neurons).float()
    threshold = normalization["target"]["threshold"].float().flatten()[candidate_neurons]
    y = (x_i >= threshold.view(1, -1)).float()
    del x_i
    return {
        "z1": torch.cat(z1_parts, dim=0),
        "prev": x_prev.contiguous(),
        "meta": torch.cat(meta_parts, dim=0),
        "y": y.contiguous(),
    }


def train_probe(
    model: nn.Module,
    tensors: Sequence[torch.Tensor],
    target: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label: str,
    semantic_logits: torch.Tensor = None,
    delta_l2: float = 0.0,
    feature_weights: torch.Tensor = None,
) -> List[float]:
    dataset_tensors = list(tensors) + [target]
    if semantic_logits is not None:
        dataset_tensors.append(semantic_logits)
    loader = DataLoader(TensorDataset(*dataset_tensors), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []
    model.train()
    progress = tqdm(range(1, epochs + 1), desc=label)
    for _epoch in progress:
        total, seen = 0.0, 0
        for batch in loader:
            if semantic_logits is None:
                *inputs, y = batch
                semantic = None
            else:
                *inputs, y, semantic = batch
            x = torch.cat([part.to(device) for part in inputs], dim=1)
            y = y.to(device)
            raw_logits = model(x)
            logits = raw_logits
            if semantic is not None:
                logits = logits + semantic.to(device)
            element_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            if feature_weights is None:
                loss = element_loss.mean()
            else:
                weights = feature_weights.to(device).float().view(1, -1)
                loss = (element_loss.mean(dim=0, keepdim=True) * weights).sum() / weights.sum().clamp_min(1e-8)
            if delta_l2 > 0.0 and semantic is not None:
                loss = loss + delta_l2 * raw_logits.pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * y.shape[0]
            seen += y.shape[0]
        value = total / max(seen, 1)
        history.append(value)
        progress.set_postfix(loss=f"{value:.4f}")
    model.eval()
    return history


def train_regression_probe(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_xs: Sequence[torch.Tensor],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label: str,
) -> List[torch.Tensor]:
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    progress = tqdm(range(1, epochs + 1), desc=label)
    for _epoch in progress:
        total, seen = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * y.shape[0]
            seen += y.shape[0]
        progress.set_postfix(loss=total / max(seen, 1))
    model.eval()
    outputs = []
    with torch.inference_mode():
        for eval_x in eval_xs:
            parts = []
            for start in range(0, eval_x.shape[0], batch_size):
                parts.append(model(eval_x[start : start + batch_size].to(device)).float().cpu())
            outputs.append(torch.cat(parts, dim=0))
    return outputs


def predict(model: nn.Module, tensors: Sequence[torch.Tensor], device: torch.device, batch_size: int) -> torch.Tensor:
    result = []
    n = tensors[0].shape[0]
    for start in range(0, n, batch_size):
        x = torch.cat([part[start : start + batch_size].to(device) for part in tensors], dim=1)
        with torch.inference_mode():
            result.append(model(x).float().cpu())
    return torch.cat(result, dim=0)


def fit_legacy_posthoc_probes(
    train: Dict[str, torch.Tensor],
    selection: Dict[str, torch.Tensor],
    evaluation: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    target_dim = train["y"].shape[1]
    if args.semantic_source == "recovered":
        recovered = {"train": [], "selection": [], "evaluation": []}
        for target_name in ("z1", "prev"):
            recovery_model = ProbeMLP(
                train["meta"].shape[1], train[target_name].shape[1], args.semantic_hidden_dim, args.probe_dropout
            ).to(device)
            pred_train, pred_selection, pred_eval = train_regression_probe(
                recovery_model,
                train["meta"],
                train[target_name],
                [train["meta"], selection["meta"], evaluation["meta"]],
                device,
                args.semantic_epochs,
                args.probe_batch_size,
                args.probe_lr,
                args.probe_weight_decay,
                f"Recover semantic {target_name} from meta",
            )
            recovered["train"].append(pred_train)
            recovered["selection"].append(pred_selection)
            recovered["evaluation"].append(pred_eval)
        semantic_train_parts = recovered["train"]
        semantic_selection_parts = recovered["selection"]
        semantic_eval_parts = recovered["evaluation"]
    else:
        semantic_train_parts = [train["z1"], train["prev"]]
        semantic_selection_parts = [selection["z1"], selection["prev"]]
        semantic_eval_parts = [evaluation["z1"], evaluation["prev"]]

    semantic_logits = {"train": [], "selection": [], "evaluation": []}
    semantic_history = []
    semantic_input_dim = sum(part.shape[1] for part in semantic_train_parts)
    for repeat in range(args.probe_repeats):
        torch.manual_seed(args.seed + 4100 + repeat)
        semantic_teacher = ProbeMLP(
            semantic_input_dim, target_dim, args.semantic_hidden_dim, args.probe_dropout
        ).to(device)
        history = train_probe(
            semantic_teacher,
            semantic_train_parts,
            train["y"],
            device,
            args.semantic_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            f"Semantic teacher r{repeat + 1}",
        )
        semantic_history.append(history[-1] if history else float("nan"))
        semantic_logits["train"].append(predict(semantic_teacher, semantic_train_parts, device, args.probe_batch_size))
        semantic_logits["selection"].append(
            predict(semantic_teacher, semantic_selection_parts, device, args.probe_batch_size)
        )
        semantic_logits["evaluation"].append(predict(semantic_teacher, semantic_eval_parts, device, args.probe_batch_size))
    semantic_train = torch.stack(semantic_logits["train"]).mean(dim=0)
    semantic_selection = torch.stack(semantic_logits["selection"]).mean(dim=0)
    semantic_eval = torch.stack(semantic_logits["evaluation"]).mean(dim=0)

    residual_logits = {"train": [], "selection": [], "evaluation": []}
    residual_history = []
    for repeat in range(args.probe_repeats):
        torch.manual_seed(args.seed + 5100 + repeat)
        residual_head = ProbeMLP(
            train["meta"].shape[1], target_dim, args.residual_hidden_dim, args.probe_dropout
        ).to(device)
        if args.residual_head_mode == "standalone_meta_delta":
            history = train_probe(
                residual_head,
                [train["meta"]],
                train["y"],
                device,
                args.residual_epochs,
                args.probe_batch_size,
                args.probe_lr,
                args.probe_weight_decay,
                f"Standalone meta head r{repeat + 1}",
            )
            residual_logits["train"].append(
                predict(residual_head, [train["meta"]], device, args.probe_batch_size) - semantic_train
            )
            residual_logits["selection"].append(
                predict(residual_head, [selection["meta"]], device, args.probe_batch_size) - semantic_selection
            )
            residual_logits["evaluation"].append(
                predict(residual_head, [evaluation["meta"]], device, args.probe_batch_size) - semantic_eval
            )
        else:
            history = train_probe(
                residual_head,
                [train["meta"]],
                train["y"],
                device,
                args.residual_epochs,
                args.probe_batch_size,
                args.probe_lr,
                args.probe_weight_decay,
                f"Residual head r{repeat + 1}",
                semantic_logits=semantic_train,
                delta_l2=args.delta_l2,
            )
            residual_logits["train"].append(predict(residual_head, [train["meta"]], device, args.probe_batch_size))
            residual_logits["selection"].append(
                predict(residual_head, [selection["meta"]], device, args.probe_batch_size)
            )
            residual_logits["evaluation"].append(
                predict(residual_head, [evaluation["meta"]], device, args.probe_batch_size)
            )
        residual_history.append(history[-1] if history else float("nan"))
    return {
        "semantic_train": semantic_train,
        "semantic_selection": semantic_selection,
        "semantic_eval": semantic_eval,
        "delta_train": torch.stack(residual_logits["train"]).mean(dim=0),
        "delta_selection": torch.stack(residual_logits["selection"]).mean(dim=0),
        "delta_eval": torch.stack(residual_logits["evaluation"]).mean(dim=0),
        "semantic_history": semantic_history,
        "residual_history": residual_history,
        "best_semantic_control_counts": None,
    }


def fit_decoupler_final_aligned_probes(
    train: Dict[str, torch.Tensor],
    selection: Dict[str, torch.Tensor],
    evaluation: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    feature_weights: torch.Tensor,
) -> Dict[str, Any]:
    """Reproduce the purifier dynamic/final information-decomposition probe family."""
    target_dim = train["y"].shape[1]
    recovered: Dict[str, Dict[str, torch.Tensor]] = {}
    for target_offset, target_name in enumerate(("z1", "prev")):
        torch.manual_seed(args.seed + 6100 + target_offset)
        if args.aligned_semantic_recovery_probe == "mlp":
            recovery_model = DecouplerFinalProbeMLP(
                train["meta"].shape[1],
                train[target_name].shape[1],
                args.semantic_hidden_dim,
                args.probe_dropout,
            ).to(device)
        else:
            recovery_model = ProbeMLP(
                train["meta"].shape[1], train[target_name].shape[1], 0, 0.0
            ).to(device)
        pred_train, pred_selection, pred_eval = train_regression_probe(
            recovery_model,
            train["meta"],
            train[target_name],
            [train["meta"], selection["meta"], evaluation["meta"]],
            device,
            args.semantic_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            f"Aligned recover meta->{target_name}",
        )
        recovered[target_name] = {
            "train": pred_train,
            "selection": pred_selection,
            "evaluation": pred_eval,
        }

    control_specs = {
        "recovered_z1": {split: recovered["z1"][split] for split in ("train", "selection", "evaluation")},
        "recovered_prev": {split: recovered["prev"][split] for split in ("train", "selection", "evaluation")},
        "recovered_z1_prev": {
            split: torch.cat([recovered["z1"][split], recovered["prev"][split]], dim=1)
            for split in ("train", "selection", "evaluation")
        },
    }
    control_logits: Dict[str, Dict[str, torch.Tensor]] = {}
    semantic_history = []
    for control_offset, (name, parts) in enumerate(control_specs.items()):
        repeats = {"train": [], "selection": [], "evaluation": []}
        for repeat in range(args.probe_repeats):
            torch.manual_seed(args.seed + 7100 + control_offset * 100 + repeat)
            probe = ProbeMLP(parts["train"].shape[1], target_dim, 0, 0.0).to(device)
            history = train_probe(
                probe,
                [parts["train"]],
                train["y"],
                device,
                args.semantic_epochs,
                args.probe_batch_size,
                args.probe_lr,
                args.probe_weight_decay,
                f"Aligned semantic control {name} r{repeat + 1}",
                feature_weights=feature_weights,
            )
            semantic_history.append(history[-1] if history else float("nan"))
            for split in repeats:
                repeats[split].append(predict(probe, [parts[split]], device, args.probe_batch_size))
        control_logits[name] = {split: torch.stack(values).mean(dim=0) for split, values in repeats.items()}

    control_names = list(control_logits)
    selection_ce = torch.stack(
        [
            F.binary_cross_entropy_with_logits(control_logits[name]["selection"], selection["y"], reduction="none").mean(dim=0)
            for name in control_names
        ],
        dim=0,
    )
    best_control = selection_ce.argmin(dim=0)

    def gather_best(split: str) -> torch.Tensor:
        stacked = torch.stack([control_logits[name][split] for name in control_names], dim=0)
        index = best_control.view(1, 1, -1).expand(1, stacked.shape[1], -1)
        return stacked.gather(0, index).squeeze(0)

    semantic_train = gather_best("train")
    semantic_selection = gather_best("selection")
    semantic_eval = gather_best("evaluation")

    meta_logits = {"train": [], "selection": [], "evaluation": []}
    residual_history = []
    for repeat in range(args.probe_repeats):
        torch.manual_seed(args.seed + 8100 + repeat)
        if args.aligned_target_probe == "mlp":
            meta_probe = DecouplerFinalProbeMLP(
                train["meta"].shape[1], target_dim, args.residual_hidden_dim, args.probe_dropout
            ).to(device)
        else:
            meta_probe = ProbeMLP(train["meta"].shape[1], target_dim, 0, 0.0).to(device)
        history = train_probe(
            meta_probe,
            [train["meta"]],
            train["y"],
            device,
            args.residual_epochs,
            args.probe_batch_size,
            args.probe_lr,
            args.probe_weight_decay,
            f"Aligned purified-meta target probe r{repeat + 1}",
            feature_weights=feature_weights,
        )
        residual_history.append(history[-1] if history else float("nan"))
        meta_logits["train"].append(predict(meta_probe, [train["meta"]], device, args.probe_batch_size))
        meta_logits["selection"].append(predict(meta_probe, [selection["meta"]], device, args.probe_batch_size))
        meta_logits["evaluation"].append(predict(meta_probe, [evaluation["meta"]], device, args.probe_batch_size))
    averaged_meta = {split: torch.stack(values).mean(dim=0) for split, values in meta_logits.items()}
    return {
        "semantic_train": semantic_train,
        "semantic_selection": semantic_selection,
        "semantic_eval": semantic_eval,
        "delta_train": averaged_meta["train"] - semantic_train,
        "delta_selection": averaged_meta["selection"] - semantic_selection,
        "delta_eval": averaged_meta["evaluation"] - semantic_eval,
        "semantic_history": semantic_history,
        "residual_history": residual_history,
        "best_semantic_control_counts": {
            name: int((best_control == idx).sum().item()) for idx, name in enumerate(control_names)
        },
        "aligned_target_probe": args.aligned_target_probe,
        "aligned_semantic_recovery_probe": args.aligned_semantic_recovery_probe,
    }


def _stat_chunk_size() -> int:
    """Bound temporary resampling tensors while amortizing Python overhead."""
    try:
        return max(1, int(os.environ.get("METACOG_STAT_CHUNK_SIZE", "64")))
    except ValueError:
        return 64


def bootstrap_ci(
    values: torch.Tensor,
    samples: int,
    generator: torch.Generator,
    progress_desc: str | None = None,
) -> Tuple[float, float]:
    values = values.detach().cpu()
    if samples <= 0 or values.numel() < 2:
        mean = float(values.mean().item())
        return mean, mean
    means = []
    progress = tqdm(total=samples, desc=progress_desc, leave=False) if progress_desc else None
    for start in range(0, samples, _stat_chunk_size()):
        count = min(_stat_chunk_size(), samples - start)
        idx = torch.stack(
            [
                torch.randint(values.numel(), (values.numel(),), generator=generator)
                for _ in range(count)
            ]
        )
        means.append(values[idx].mean(dim=1))
        if progress is not None:
            progress.update(count)
    if progress is not None:
        progress.close()
    stacked = torch.cat(means)
    return float(torch.quantile(stacked, 0.025).item()), float(torch.quantile(stacked, 0.975).item())


def signflip_p(
    values: torch.Tensor,
    samples: int,
    generator: torch.Generator,
    progress_desc: str | None = None,
) -> float:
    values = values.detach().cpu()
    observed = float(values.mean().item())
    if samples <= 0 or values.numel() == 0:
        return float("nan")
    count = 0
    progress = tqdm(total=samples, desc=progress_desc, leave=False) if progress_desc else None
    for start in range(0, samples, _stat_chunk_size()):
        batch = min(_stat_chunk_size(), samples - start)
        signs = torch.stack(
            [
                torch.randint(0, 2, values.shape, generator=generator)
                for _ in range(batch)
            ]
        ).to(values.dtype).mul_(2).sub_(1)
        means = (values.unsqueeze(0) * signs).mean(dim=1)
        count += int((means >= observed).sum().item())
        if progress is not None:
            progress.update(batch)
    if progress is not None:
        progress.close()
    return (count + 1.0) / (samples + 1.0)


def contribution_rows(
    candidate_rows: Sequence[Dict[str, Any]],
    y_train: torch.Tensor,
    y_eval: torch.Tensor,
    semantic_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    threshold: float,
    bootstrap_samples: int,
    permutation_tests: int,
    seed: int,
    progress_desc: str | None = None,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    prior = y_train.mean(dim=0).clamp(1e-5, 1.0 - 1e-5)
    prior_logits = torch.logit(prior).view(1, -1).expand_as(y_eval)
    combined_logits = semantic_logits + delta_logits
    prior_loss = F.binary_cross_entropy_with_logits(prior_logits, y_eval, reduction="none")
    semantic_loss = F.binary_cross_entropy_with_logits(semantic_logits, y_eval, reduction="none")
    combined_loss = F.binary_cross_entropy_with_logits(combined_logits, y_eval, reduction="none")
    generator = torch.Generator().manual_seed(seed)
    rows: List[Dict[str, Any]] = []
    d_values = []
    target_iterator = enumerate(candidate_rows)
    if progress_desc:
        target_iterator = tqdm(
            target_iterator,
            total=len(candidate_rows),
            desc=progress_desc,
            leave=False,
        )
    show_resample_detail = bool(progress_desc and len(candidate_rows) <= 8)
    for col, source in target_iterator:
        train_activation_rate = float(y_train[:, col].float().mean().item())
        eval_activation_rate = float(y_eval[:, col].float().mean().item())

        def binary_entropy(rate: float) -> float:
            probability = min(max(rate, 1e-8), 1.0 - 1e-8)
            return -probability * math.log(probability) - (1.0 - probability) * math.log(1.0 - probability)

        train_prior_entropy = binary_entropy(train_activation_rate)
        eval_prior_entropy = binary_entropy(eval_activation_rate)
        total_gain = float((prior_loss[:, col] - combined_loss[:, col]).mean().item())
        residual_gain = float((semantic_loss[:, col] - combined_loss[:, col]).mean().item())
        semantic_gain = float((prior_loss[:, col] - semantic_loss[:, col]).mean().item())
        legacy_ratio = residual_gain / total_gain if total_gain > 1e-8 else float("nan")
        positive_residual = max(residual_gain, 0.0)
        positive_semantic = max(semantic_gain, 0.0)
        positive_mass = positive_residual + positive_semantic
        # Use a bounded contribution share for selection.  The legacy
        # residual/total ratio can exceed one when the semantic control is
        # worse than the prior, and becomes unstable when total_gain is tiny.
        ratio = (
            positive_residual / positive_mass
            if total_gain > 1e-8 and residual_gain > 0.0 and positive_mass > 1e-8
            else float("nan")
        )
        d = (
            semantic_loss[:, col]
            - combined_loss[:, col]
            - threshold * (prior_loss[:, col] - combined_loss[:, col])
        )
        target_desc = (
            f"{progress_desc} target {col + 1}/{len(candidate_rows)}"
            if show_resample_detail
            else None
        )
        ci_low, ci_high = bootstrap_ci(
            d,
            bootstrap_samples,
            generator,
            f"{target_desc} bootstrap" if target_desc else None,
        )
        p = signflip_p(
            d,
            permutation_tests,
            generator,
            f"{target_desc} sign-flip" if target_desc else None,
        )
        d_values.append(d.mean())
        rows.append(
            {
                "neuron": int(source["neuron"]),
                "source_residual_fraction": source.get("source_residual_fraction"),
                "source_residual_gap_bits": source.get("residual_gap_bits"),
                "prior_ce_nats": float(prior_loss[:, col].mean().item()),
                "semantic_ce_nats": float(semantic_loss[:, col].mean().item()),
                "combined_ce_nats": float(combined_loss[:, col].mean().item()),
                "total_gain_nats": total_gain,
                "semantic_gain_nats": semantic_gain,
                "residual_gain_nats": residual_gain,
                "residual_fraction": ratio,
                "residual_fraction_bounded": ratio,
                "legacy_residual_over_total": legacy_ratio,
                "positive_information_mass_nats": positive_mass,
                "positive_total_gain": total_gain > 0.0,
                "positive_residual_gain": residual_gain > 0.0,
                "positive_semantic_gain": semantic_gain > 0.0,
                "threshold": threshold,
                "threshold_margin_nats": float(d.mean().item()),
                "threshold_margin_ci_low": ci_low,
                "threshold_margin_ci_high": ci_high,
                "threshold_signflip_p": p,
                "activation_rate": eval_activation_rate,
                "activation_rate_train": train_activation_rate,
                "activation_rate_eval": eval_activation_rate,
                "activation_rate_drift_abs": abs(train_activation_rate - eval_activation_rate),
                "prior_entropy_nats_train": train_prior_entropy,
                "prior_entropy_nats_eval": eval_prior_entropy,
                "prior_entropy_nats_min": min(train_prior_entropy, eval_prior_entropy),
            }
        )
    return rows, torch.stack(d_values)


def fit_transform_state(x: torch.Tensor, pca_dim: int, clip_quantile: float, l2_normalize: bool):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    transformed = (x - mean) / std
    clip_value = None
    if clip_quantile > 0.0:
        clip_value = torch.quantile(transformed.abs().flatten(), clip_quantile).clamp_min(1e-6)
        transformed = transformed.clamp(-float(clip_value), float(clip_value))
    components = None
    pca_mean = None
    if pca_dim > 0 and transformed.shape[1] > pca_dim:
        pca_mean = transformed.mean(dim=0, keepdim=True)
        centered = transformed - pca_mean
        _, _, v = torch.linalg.svd(centered, full_matrices=False)
        components = v[:pca_dim].t().contiguous()
        transformed = centered @ components
    if l2_normalize:
        transformed = transformed / transformed.norm(dim=1, keepdim=True).clamp_min(1e-6)
    state = {
        "mean": mean,
        "std": std,
        "clip_value": clip_value,
        "pca_mean": pca_mean,
        "components": components,
        "l2_normalize": bool(l2_normalize),
    }
    return transformed.contiguous(), state


def apply_transform(x: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
    transformed = (x - state["mean"]) / state["std"]
    if state["clip_value"] is not None:
        value = float(state["clip_value"])
        transformed = transformed.clamp(-value, value)
    if state["components"] is not None:
        transformed = (transformed - state["pca_mean"]) @ state["components"]
    if state["l2_normalize"]:
        transformed = transformed / transformed.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return transformed.contiguous()


def nearest_labels(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x.float(), centroids.float()).argmin(dim=1)


def balanced_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, k: int) -> float:
    recalls = []
    for label in range(k):
        mask = y_true == label
        if mask.any():
            recalls.append((y_pred[mask] == label).float().mean())
    return float(torch.stack(recalls).mean().item()) if recalls else float("nan")


def _batched_balanced_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """Balanced accuracy for [batch, samples] tensors without Python per-resample work."""
    recall_sum = torch.zeros(y_true.shape[0], dtype=torch.float32)
    valid_count = torch.zeros(y_true.shape[0], dtype=torch.float32)
    for label in range(num_clusters):
        mask = y_true == label
        denominator = mask.sum(dim=1)
        valid = denominator > 0
        numerator = ((y_pred == label) & mask).sum(dim=1)
        recall = numerator.float() / denominator.clamp_min(1).float()
        recall_sum += torch.where(valid, recall, torch.zeros_like(recall))
        valid_count += valid.float()
    return recall_sum / valid_count.clamp_min(1.0)


def semantic_cluster_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    num_clusters: int,
    device: torch.device,
    seed: int,
    epochs: int = 30,
    batch_size: int = 256,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = ProbeMLP(train_x.shape[1], num_clusters, min(128, max(32, train_x.shape[1] * 4)), 0.1).to(device)
    use_pinned_memory = device.type == "cuda"
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pinned_memory,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(max(1, epochs)):
        for x, y in loader:
            logits = model(x.to(device, non_blocking=use_pinned_memory))
            loss = F.cross_entropy(logits, y.to(device, non_blocking=use_pinned_memory))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    for start in range(0, eval_x.shape[0], 512):
        with torch.inference_mode():
            predictions.append(model(eval_x[start : start + 512].to(device)).argmax(dim=1).cpu())
    pred = torch.cat(predictions)
    return {
        "accuracy": float((pred == eval_y).float().mean().item()),
        "balanced_accuracy": balanced_accuracy(eval_y, pred, num_clusters),
        "predictions": pred,
    }


def semantic_predictability_test(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_clusters: int,
    bootstrap_samples: int,
    permutation_tests: int,
    seed: int,
    progress_desc: str | None = None,
) -> Dict[str, Any]:
    """Test semantic-cluster association on samples untouched by subset search."""
    labels = labels.long().cpu()
    predictions = predictions.long().cpu()
    observed = balanced_accuracy(labels, predictions, num_clusters)
    generator = torch.Generator().manual_seed(seed)
    bootstrap_values = []
    bootstrap_progress = (
        tqdm(total=bootstrap_samples, desc=f"{progress_desc} bootstrap", leave=False)
        if progress_desc and bootstrap_samples > 0
        else None
    )
    for start in range(0, max(0, bootstrap_samples), _stat_chunk_size()):
        count = min(_stat_chunk_size(), bootstrap_samples - start)
        indices = torch.stack(
            [
                torch.randint(0, labels.numel(), (labels.numel(),), generator=generator)
                for _ in range(count)
            ]
        )
        bootstrap_values.append(
            _batched_balanced_accuracy(labels[indices], predictions[indices], num_clusters)
        )
        if bootstrap_progress is not None:
            bootstrap_progress.update(count)
    if bootstrap_progress is not None:
        bootstrap_progress.close()
    if bootstrap_values:
        ordered = torch.cat(bootstrap_values).sort().values
        low_index = max(0, int(0.025 * (ordered.numel() - 1)))
        high_index = min(ordered.numel() - 1, int(0.975 * (ordered.numel() - 1)))
        ci = [float(ordered[low_index].item()), float(ordered[high_index].item())]
    else:
        ci = [float("nan"), float("nan")]

    exceedances = 0
    permutation_progress = (
        tqdm(total=permutation_tests, desc=f"{progress_desc} permutation", leave=False)
        if progress_desc and permutation_tests > 0
        else None
    )
    for start in range(0, max(0, permutation_tests), _stat_chunk_size()):
        count = min(_stat_chunk_size(), permutation_tests - start)
        # Keep torch.randperm and the original generator sequence; only metric
        # evaluation is vectorized, so the Monte Carlo test itself is unchanged.
        permutations = torch.stack(
            [torch.randperm(labels.numel(), generator=generator) for _ in range(count)]
        )
        shuffled = labels[permutations]
        batch_predictions = predictions.unsqueeze(0).expand(count, -1)
        scores = _batched_balanced_accuracy(shuffled, batch_predictions, num_clusters)
        exceedances += int((scores >= observed - 1e-12).sum().item())
        if permutation_progress is not None:
            permutation_progress.update(count)
    if permutation_progress is not None:
        permutation_progress.close()
    permutation_p = (
        (exceedances + 1) / (permutation_tests + 1)
        if permutation_tests > 0
        else float("nan")
    )
    chance = 1.0 / num_clusters
    return {
        "balanced_accuracy": observed,
        "bootstrap_ci95": ci,
        "permutation_p": permutation_p,
        "chance_balanced_accuracy": chance,
        "significant_above_chance_0p05": bool(
            permutation_tests > 0 and permutation_p < 0.05 and observed > chance
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "permutation_tests": int(permutation_tests),
    }


def _subset_candidates(
    pool: Sequence[int],
    subset_size: int,
    max_combinations: int,
    seed: int,
    selection_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Tuple[int, ...]], int]:
    total = math.comb(len(pool), subset_size)
    anchors = set()
    ranking_keys = ("threshold_margin_nats", "residual_fraction", "residual_gain_nats")
    for key in ranking_keys:
        ranked = sorted(
            pool,
            key=lambda col: float(selection_rows[col].get(key, -math.inf)),
            reverse=True,
        )
        anchors.add(tuple(sorted(ranked[:subset_size])))
    if total <= max_combinations:
        combinations = list(itertools.combinations(sorted(pool), subset_size))
    else:
        rng = random.Random(seed)
        combinations_set = set(anchors)
        while len(combinations_set) < max_combinations:
            combinations_set.add(tuple(sorted(rng.sample(list(pool), subset_size))))
        combinations = sorted(combinations_set)
    return combinations, total


def _conservative_semantic_bacc(value: float, num_clusters: int) -> float:
    if num_clusters == 2:
        return max(value, 1.0 - value)
    return max(value, 1.0 / num_clusters)


def _standardize_search_pair(
    train: torch.Tensor,
    validation: torch.Tensor,
    clip_quantile: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train = (train - mean) / std
    validation = (validation - mean) / std
    if 0.0 < clip_quantile < 1.0:
        clip_value = torch.quantile(train.abs().flatten(), clip_quantile).clamp_min(1e-6)
        train = train.clamp(-clip_value, clip_value)
        validation = validation.clamp(-clip_value, clip_value)
    return train, validation


def approximate_subset_rows(
    combinations: Sequence[Tuple[int, ...]],
    strict_pool: Sequence[int],
    selection_rows: Sequence[Dict[str, Any]],
    candidate_neurons: torch.Tensor,
    residual_features: torch.Tensor,
    semantic_features: torch.Tensor,
    search_train: torch.Tensor,
    search_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Vectorized proxy: residual first-PC margin plus a semantic ridge probe."""
    pool_columns = torch.tensor(strict_pool, dtype=torch.long)
    pool_position = {column: index for index, column in enumerate(strict_pool)}
    residual_train, residual_val = _standardize_search_pair(
        residual_features[search_train].index_select(1, pool_columns).to(device).float(),
        residual_features[search_val].index_select(1, pool_columns).to(device).float(),
        args.clip_quantile,
    )
    semantic_train, semantic_val = _standardize_search_pair(
        semantic_features[search_train].index_select(1, pool_columns).to(device).float(),
        semantic_features[search_val].index_select(1, pool_columns).to(device).float(),
        args.clip_quantile,
    )
    residual_fraction_pool = torch.tensor(
        [float(selection_rows[column]["residual_fraction"]) for column in strict_pool],
        device=device,
    )
    identity = torch.eye(args.subset_search_size, device=device).unsqueeze(0)
    rows: List[Dict[str, Any]] = []
    batch_size = max(1, int(args.subset_search_approx_batch_size))
    batches = range(0, len(combinations), batch_size)
    for start in tqdm(batches, desc="Approximate strict subset search"):
        batch_combinations = combinations[start : start + batch_size]
        local_columns = torch.tensor(
            [[pool_position[column] for column in combo] for combo in batch_combinations],
            dtype=torch.long,
            device=device,
        )
        residual_train_batch = residual_train[:, local_columns]
        residual_val_batch = residual_val[:, local_columns]
        if args.l2_normalize:
            residual_train_batch = residual_train_batch / residual_train_batch.norm(
                dim=2, keepdim=True
            ).clamp_min(1e-6)
            residual_val_batch = residual_val_batch / residual_val_batch.norm(
                dim=2, keepdim=True
            ).clamp_min(1e-6)

        centered = residual_train_batch - residual_train_batch.mean(dim=0, keepdim=True)
        covariance = torch.einsum("nbd,nbe->bde", centered, centered)
        covariance /= max(centered.shape[0] - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        principal_direction = eigenvectors[:, :, -1]
        train_scores = torch.einsum("nbd,bd->nb", centered, principal_direction)
        val_centered = residual_val_batch - residual_train_batch.mean(dim=0, keepdim=True)
        val_scores = torch.einsum("nbd,bd->nb", val_centered, principal_direction)

        threshold = train_scores.median(dim=0).values
        train_labels = train_scores >= threshold.unsqueeze(0)
        for _ in range(2):
            positive = train_labels.float()
            negative = 1.0 - positive
            centroid_one = (train_scores * positive).sum(dim=0) / positive.sum(dim=0).clamp_min(1.0)
            centroid_zero = (train_scores * negative).sum(dim=0) / negative.sum(dim=0).clamp_min(1.0)
            threshold = 0.5 * (centroid_zero + centroid_one)
            train_labels = train_scores >= threshold.unsqueeze(0)
        val_labels = val_scores >= threshold.unsqueeze(0)
        train_fraction_one = train_labels.float().mean(dim=0)
        val_fraction_one = val_labels.float().mean(dim=0)
        min_fraction = torch.minimum(
            torch.minimum(train_fraction_one, 1.0 - train_fraction_one),
            torch.minimum(val_fraction_one, 1.0 - val_fraction_one),
        )

        distance_zero = (val_scores - centroid_zero.unsqueeze(0)).abs()
        distance_one = (val_scores - centroid_one.unsqueeze(0)).abs()
        assigned_distance = torch.where(val_labels, distance_one, distance_zero)
        other_distance = torch.where(val_labels, distance_zero, distance_one)
        silhouette = ((other_distance - assigned_distance) / torch.maximum(
            assigned_distance, other_distance
        ).clamp_min(1e-6)).mean(dim=0)
        inertia = assigned_distance.square().mean(dim=0)

        semantic_train_batch = semantic_train[:, local_columns]
        semantic_val_batch = semantic_val[:, local_columns]
        train_target = train_labels.float() * 2.0 - 1.0
        semantic_mean = semantic_train_batch.mean(dim=0, keepdim=True)
        target_mean = train_target.mean(dim=0, keepdim=True)
        semantic_centered = semantic_train_batch - semantic_mean
        target_centered = train_target - target_mean
        gram = torch.einsum("nbd,nbe->bde", semantic_centered, semantic_centered)
        rhs = torch.einsum("nbd,nb->bd", semantic_centered, target_centered)
        gram = gram + args.subset_search_approx_ridge * identity
        weights = torch.linalg.solve(gram, rhs.unsqueeze(2)).squeeze(2)
        intercept = target_mean.squeeze(0) - torch.einsum(
            "bd,bd->b", semantic_mean.squeeze(0), weights
        )
        semantic_logits = torch.einsum("nbd,bd->nb", semantic_val_batch, weights)
        semantic_predictions = semantic_logits + intercept.unsqueeze(0) >= 0.0
        recall_zero = ((~semantic_predictions) & (~val_labels)).sum(dim=0).float()
        recall_zero /= (~val_labels).sum(dim=0).clamp_min(1)
        recall_one = (semantic_predictions & val_labels).sum(dim=0).float()
        recall_one /= val_labels.sum(dim=0).clamp_min(1)
        semantic_bacc = 0.5 * (recall_zero + recall_one)
        conservative_bacc = torch.maximum(semantic_bacc, 1.0 - semantic_bacc)
        mean_residual_fraction = residual_fraction_pool[local_columns].mean(dim=1)
        collapsed_penalty = (
            args.subset_search_min_cluster_fraction - min_fraction
        ).clamp_min(0.0) / max(args.subset_search_min_cluster_fraction, 1e-6)
        proxy_objective = (
            conservative_bacc
            + args.subset_search_collapse_penalty * collapsed_penalty
            - args.subset_search_silhouette_weight * silhouette
            - args.subset_search_residual_weight * mean_residual_fraction
        )

        batch_metrics = torch.stack(
            [
                mean_residual_fraction,
                min_fraction,
                silhouette,
                inertia,
                semantic_bacc,
                conservative_bacc,
                collapsed_penalty,
                proxy_objective,
            ],
            dim=1,
        ).detach().cpu()
        for offset, combo in enumerate(batch_combinations):
            metrics = batch_metrics[offset]
            rows.append(
                {
                    "combination_index": start + offset,
                    "candidate_columns": " ".join(str(column) for column in combo),
                    "neuron_indices": " ".join(
                        str(int(candidate_neurons[column])) for column in combo
                    ),
                    "mean_residual_fraction": float(metrics[0]),
                    "min_cluster_fraction": float(metrics[1]),
                    "centroid_silhouette": float(metrics[2]),
                    "inertia": float(metrics[3]),
                    "semantic_nearest_balanced_accuracy": float(metrics[4]),
                    "semantic_nearest_conservative_bacc": float(metrics[5]),
                    "collapsed_penalty": float(metrics[6]),
                    "passes_cluster_size_constraint": bool(
                        float(metrics[1]) >= args.subset_search_min_cluster_fraction
                    ),
                    "proxy_objective": float(metrics[7]),
                    "proxy_method": "first_pc_margin_plus_semantic_ridge",
                    "semantic_mlp_balanced_accuracy_mean": float("nan"),
                    "semantic_mlp_balanced_accuracy_std": float("nan"),
                    "semantic_mlp_conservative_bacc": float("nan"),
                    "final_objective": float("nan"),
                    "shortlisted": False,
                    "selected": False,
                }
            )
    return rows


def search_target_subset(
    strict_pool: Sequence[int],
    selection_rows: Sequence[Dict[str, Any]],
    candidate_neurons: torch.Tensor,
    residual_features: torch.Tensor,
    semantic_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any]]:
    """Choose a strict Top-K on a nested selection split, never final evaluation."""
    subset_size = int(args.subset_search_size)
    if len(strict_pool) < subset_size:
        raise ValueError(
            f"Subset search needs {subset_size} strict targets, but only {len(strict_pool)} passed."
        )
    budget = float(args.subset_search_budget)
    effective_max_combinations = max(
        64, int(round(args.subset_search_max_combinations * budget))
    )
    # Search quality settings stay fixed across budgets so budget sweeps isolate
    # the effect of evaluating more candidate combinations. Use every row in
    # the already-isolated target-selection split.
    effective_max_samples = int(residual_features.shape[0])
    effective_shortlist = max(1, int(args.subset_search_shortlist))
    effective_mlp_epochs = max(1, int(args.subset_search_mlp_epochs))
    effective_mlp_repeats = max(1, int(args.subset_search_mlp_repeats))
    print(
        f"[soft-residual] subset-search budget={budget:.2f} "
        f"combinations<={effective_max_combinations} samples=all({effective_max_samples}) "
        f"shortlist={effective_shortlist} mlp={effective_mlp_epochs}x{effective_mlp_repeats} "
        "(fixed quality settings)",
        flush=True,
    )
    combinations, total_combinations = _subset_candidates(
        strict_pool,
        subset_size,
        effective_max_combinations,
        args.seed + 6201,
        selection_rows,
    )
    generator = torch.Generator().manual_seed(args.seed + 6203)
    sample_indices = torch.randperm(residual_features.shape[0], generator=generator)
    split_at = int(round(sample_indices.numel() * args.subset_search_train_fraction))
    split_at = min(max(split_at, args.num_clusters * 2), sample_indices.numel() - args.num_clusters * 2)
    if split_at <= 0 or split_at >= sample_indices.numel():
        raise ValueError("Subset-search split is too small for nested train/validation clustering.")
    search_train = sample_indices[:split_at]
    search_val = sample_indices[split_at:]

    rows = approximate_subset_rows(
        combinations,
        strict_pool,
        selection_rows,
        candidate_neurons,
        residual_features,
        semantic_features,
        search_train,
        search_val,
        args,
        device,
    )

    valid_rows = [row for row in rows if row["passes_cluster_size_constraint"]]
    if not valid_rows:
        write_csv(os.path.join(args.output_dir, "target_subset_search.csv"), rows)
        raise RuntimeError(
            "No target subset satisfied --subset-search-min-cluster-fraction; "
            "lower the constraint only after inspecting target_subset_search.csv."
        )
    shortlist = sorted(valid_rows, key=lambda row: float(row["proxy_objective"]))[
        :effective_shortlist
    ]
    for row in tqdm(shortlist, desc="Validate subset semantic probes"):
        combo = tuple(int(value) for value in str(row["candidate_columns"]).split())
        columns = torch.tensor(combo, dtype=torch.long)
        semantic_train, semantic_transform = fit_transform_state(
            semantic_features[search_train].index_select(1, columns),
            0,
            args.clip_quantile,
            False,
        )
        semantic_val = apply_transform(
            semantic_features[search_val].index_select(1, columns), semantic_transform
        )
        residual_train, residual_transform = fit_transform_state(
            residual_features[search_train].index_select(1, columns),
            0,
            args.clip_quantile,
            args.l2_normalize,
        )
        residual_val = apply_transform(
            residual_features[search_val].index_select(1, columns), residual_transform
        )
        train_labels, centroids, _ = run_kmeans(
            residual_train,
            args.num_clusters,
            args.subset_search_kmeans_iters,
            args.subset_search_kmeans_restarts,
            args.seed + 8101 + int(row["combination_index"]),
        )
        val_labels = nearest_labels(residual_val, centroids)
        exact_train_fractions = torch.bincount(
            train_labels, minlength=args.num_clusters
        ).float() / max(train_labels.numel(), 1)
        exact_val_fractions = torch.bincount(
            val_labels, minlength=args.num_clusters
        ).float() / max(val_labels.numel(), 1)
        exact_min_fraction = min(
            float(exact_train_fractions.min()), float(exact_val_fractions.min())
        )
        exact_silhouette = centroid_silhouette_score(residual_val, val_labels, centroids)
        exact_collapsed_penalty = max(
            0.0, args.subset_search_min_cluster_fraction - exact_min_fraction
        ) / max(args.subset_search_min_cluster_fraction, 1e-6)
        mlp_baccs = []
        for repeat in range(effective_mlp_repeats):
            audit = semantic_cluster_probe(
                semantic_train,
                train_labels,
                semantic_val,
                val_labels,
                args.num_clusters,
                device,
                args.seed + 9101 + 97 * repeat + int(row["combination_index"]),
                epochs=effective_mlp_epochs,
                batch_size=args.subset_search_mlp_batch_size,
            )
            mlp_baccs.append(float(audit["balanced_accuracy"]))
        mean_bacc = sum(mlp_baccs) / len(mlp_baccs)
        std_bacc = float(torch.tensor(mlp_baccs).std(unbiased=False).item())
        conservative_bacc = _conservative_semantic_bacc(mean_bacc, args.num_clusters)
        worst_conservative_bacc = max(
            _conservative_semantic_bacc(value, args.num_clusters) for value in mlp_baccs
        )
        final_objective = (
            args.subset_search_semantic_weight * worst_conservative_bacc
            + args.subset_search_collapse_penalty * exact_collapsed_penalty
            - args.subset_search_silhouette_weight * exact_silhouette
            - args.subset_search_residual_weight * float(row["mean_residual_fraction"])
        )
        row.update(
            {
                "shortlisted": True,
                "semantic_mlp_balanced_accuracy_mean": mean_bacc,
                "semantic_mlp_balanced_accuracy_std": std_bacc,
                "semantic_mlp_conservative_bacc": conservative_bacc,
                "semantic_mlp_worst_conservative_bacc": worst_conservative_bacc,
                "passes_semantic_bacc": worst_conservative_bacc <= args.subset_search_max_semantic_bacc,
                "exact_min_cluster_fraction": exact_min_fraction,
                "exact_centroid_silhouette": exact_silhouette,
                "exact_collapsed_penalty": exact_collapsed_penalty,
                "final_objective": final_objective,
            }
        )

    semantic_safe_shortlist = [row for row in shortlist if row["passes_semantic_bacc"]]
    if args.subset_search_require_semantic_bacc and not semantic_safe_shortlist:
        raise RuntimeError(
            "No shortlisted target subset passed --subset-search-max-semantic-bacc. "
            "Increase the ceiling, search budget, or shortlist size."
        )
    selected_row = min(
        semantic_safe_shortlist or shortlist,
        key=lambda row: float(row["final_objective"]),
    )
    selected_row["selected"] = True
    selected_row["semantic_bacc_fallback_used"] = not bool(semantic_safe_shortlist)
    selected_columns = [int(value) for value in str(selected_row["candidate_columns"]).split()]
    write_csv(os.path.join(args.output_dir, "target_subset_search.csv"), rows)
    summary = {
        "enabled": True,
        "selection_only": True,
        "search_algorithm": "batched_first_pc_margin_plus_semantic_ridge_then_exact_shortlist",
        "search_budget": budget,
        "effective_settings": {
            "max_combinations": effective_max_combinations,
            "max_samples": effective_max_samples,
            "sample_policy": "all_available_target_selection_rows",
            "shortlist": effective_shortlist,
            "mlp_epochs": effective_mlp_epochs,
            "mlp_repeats": effective_mlp_repeats,
            "budget_controls": ["max_combinations"],
        },
        "strict_pool_size": len(strict_pool),
        "subset_size": subset_size,
        "total_possible_combinations": total_combinations,
        "evaluated_combinations": len(combinations),
        "shortlist_size": len(shortlist),
        "search_train_n": int(search_train.numel()),
        "search_validation_n": int(search_val.numel()),
        "selected_candidate_columns": selected_columns,
        "selected_neurons": [int(candidate_neurons[col]) for col in selected_columns],
        "selected_metrics": {
            key: value
            for key, value in selected_row.items()
            if key not in {"candidate_columns", "neuron_indices"}
        },
        "semantic_subset_guard": {
            "metric": "worst-repeat conservative semantic MLP balanced accuracy",
            "max_bacc": args.subset_search_max_semantic_bacc,
            "semantic_weight": args.subset_search_semantic_weight,
            "required": args.subset_search_require_semantic_bacc,
            "fallback_used": not bool(semantic_safe_shortlist),
        },
        "objective": (
            "minimize conservative semantic-cluster MLP balanced accuracy while penalizing "
            "cluster collapse and rewarding residual silhouette/fraction"
        ),
        "final_evaluation_used_for_search": False,
    }
    write_json(os.path.join(args.output_dir, "target_subset_search_summary.json"), summary)
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            apply_nmi_style()
            plots = os.path.join(args.output_dir, "plots")
            os.makedirs(plots, exist_ok=True)
            figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.8))
            axes[0].scatter(
                [float(row["mean_residual_fraction"]) for row in rows],
                [float(row["semantic_nearest_conservative_bacc"]) for row in rows],
                c=[float(row["centroid_silhouette"]) for row in rows],
                cmap="viridis",
                s=12,
                alpha=0.5,
            )
            axes[0].scatter(
                [float(selected_row["mean_residual_fraction"])],
                [float(selected_row["semantic_nearest_conservative_bacc"])],
                marker="*", s=100, color=COLORS["vermillion"], label="Selected",
            )
            axes[0].axhline(0.5, color=COLORS["black"], linestyle=":", linewidth=0.9)
            axes[0].set_xlabel("Mean residual fraction")
            axes[0].set_ylabel("Semantic centroid BAcc")
            axes[0].legend()
            style_axis(axes[0])
            panel_label(axes[0], "a")
            ranked_shortlist = sorted(shortlist, key=lambda row: float(row["final_objective"]))
            axes[1].errorbar(
                range(len(ranked_shortlist)),
                [float(row["semantic_mlp_balanced_accuracy_mean"]) for row in ranked_shortlist],
                yerr=[float(row["semantic_mlp_balanced_accuracy_std"]) for row in ranked_shortlist],
                fmt="o",
                markersize=4,
                linewidth=1,
            )
            axes[1].axhline(0.5, color=COLORS["black"], linestyle=":", linewidth=0.9)
            axes[1].set_xlabel("Shortlist rank")
            axes[1].set_ylabel("Nested-validation semantic MLP BAcc")
            style_axis(axes[1])
            panel_label(axes[1], "b")
            figure.tight_layout()
            save_figure(figure, os.path.join(plots, "target_subset_search"))
            plt.close(figure)
        except ImportError:
            pass
    return selected_columns, summary


def maybe_plot(
    output_dir: str,
    selection_rows: List[Dict[str, Any]],
    features: torch.Tensor,
    raw_residual_features: torch.Tensor,
    labels: torch.Tensor,
    semantic_audit: Dict[str, float],
    residual_fraction_threshold: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    apply_nmi_style()
    plots = os.path.join(output_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    ordered = sorted(selection_rows, key=lambda row: float(row["residual_fraction"]), reverse=True)
    selected = [row for row in ordered if row.get("selected_for_clustering")]
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.65))
    selected_neurons = {int(row["neuron"]) for row in selected}
    bar_colors = [
        COLORS["vermillion"] if int(row["neuron"]) in selected_neurons else COLORS["light_gray"]
        for row in ordered
    ]
    axis.bar(
        range(len(ordered)),
        [float(row["residual_fraction"]) for row in ordered],
        color=bar_colors,
        width=0.84,
        linewidth=0,
        zorder=3,
    )
    axis.axhline(
        float(ordered[0]["threshold"]),
        color=COLORS["black"],
        linestyle="--",
        linewidth=0.9,
        label="Selection threshold",
    )
    axis.set_xlabel("Candidate target rank")
    axis.set_ylabel("Conditional residual fraction")
    axis.legend(loc="upper right")
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, os.path.join(plots, "target_residual_fraction"))
    plt.close(figure)

    source_pairs = []
    for row in ordered:
        source = row.get("source_residual_fraction")
        posthoc = row.get("residual_fraction")
        try:
            source_value = float(source)
            posthoc_value = float(posthoc)
        except (TypeError, ValueError):
            continue
        if math.isfinite(source_value) and math.isfinite(posthoc_value):
            source_pairs.append((source_value, posthoc_value, bool(row.get("selected_for_clustering"))))
    if source_pairs:
        figure, axis = plt.subplots(figsize=(3.5, 3.15))
        xs = [item[0] for item in source_pairs]
        ys = [item[1] for item in source_pairs]
        colors = [COLORS["vermillion"] if item[2] else COLORS["blue"] for item in source_pairs]
        axis.scatter(xs, ys, c=colors, s=20, alpha=0.72, edgecolors="none", rasterized=True)
        lo = min(xs + ys + [0.0])
        hi = max(xs + ys + [1.0])
        axis.plot([lo, hi], [lo, hi], color=COLORS["black"], linestyle="--", linewidth=0.9)
        axis.axhline(residual_fraction_threshold, color=COLORS["gray"], linestyle=":", linewidth=0.9)
        axis.axvline(residual_fraction_threshold, color=COLORS["gray"], linestyle=":", linewidth=0.9)
        axis.set_xlabel("Training residual fraction")
        axis.set_ylabel("Posthoc residual fraction")
        style_axis(axis)
        figure.tight_layout()
        save_figure(figure, os.path.join(plots, "source_vs_posthoc_residual_fraction"))
        plt.close(figure)

    if selected:
        names = [str(row["neuron"]) for row in selected]
        semantic_gain = [float(row["semantic_gain_nats"]) for row in selected]
        residual_gain = [float(row["residual_gain_nats"]) for row in selected]
        y = list(range(len(selected)))
        figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, max(2.8, 0.22 * len(selected) + 1.1)))
        axis.barh(y, semantic_gain, label="Semantic", color=COLORS["blue"], height=0.72)
        axis.barh(y, residual_gain, left=semantic_gain, label="Residual", color=COLORS["orange"], height=0.72)
        axis.set_yticks(y, names)
        axis.set_xlabel("Information gain (nats per label)")
        axis.set_ylabel("Layer-i neuron")
        axis.legend(loc="lower right")
        style_axis(axis, grid="x")
        figure.tight_layout()
        save_figure(figure, os.path.join(plots, "selected_target_information_decomposition"))
        plt.close(figure)

        margins = [float(row["threshold_margin_nats"]) for row in selected]
        lower = [max(0.0, value - float(row["threshold_margin_ci_low"])) for value, row in zip(margins, selected)]
        upper = [max(0.0, float(row["threshold_margin_ci_high"]) - value) for value, row in zip(margins, selected)]
        colors = [COLORS["green"] if row.get("passes_strict_threshold") else COLORS["orange"] for row in selected]
        figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, max(2.8, 0.22 * len(selected) + 1.1)))
        axis.errorbar(margins, y, xerr=[lower, upper], fmt="none", ecolor=COLORS["gray"], capsize=2)
        axis.scatter(margins, y, c=colors, s=22, zorder=3)
        axis.axvline(0.0, color=COLORS["black"], linewidth=0.9, linestyle="--")
        axis.set_yticks(y, names)
        axis.set_xlabel("Residual-threshold margin (nats)")
        axis.set_ylabel("Layer-i neuron")
        style_axis(axis, grid="x")
        figure.tight_layout()
        save_figure(figure, os.path.join(plots, "selected_target_threshold_margin_ci"))
        plt.close(figure)

    centered = features - features.mean(dim=0, keepdim=True)
    if centered.shape[1] >= 2:
        _, _, v = torch.linalg.svd(centered, full_matrices=False)
        xy = centered @ v[:2].t()
    else:
        xy = torch.cat([centered, torch.zeros_like(centered)], dim=1)
    figure, axis = plt.subplots(figsize=(3.5, 3.05))
    for cluster_id in sorted(labels.unique().tolist()):
        mask = labels == int(cluster_id)
        axis.scatter(
            xy[mask, 0], xy[mask, 1], s=8, alpha=0.45,
            color=cluster_color(int(cluster_id)), edgecolors="none", rasterized=True,
            label=f"Cluster {cluster_id} (n={int(mask.sum())})",
        )
    axis.set_xlabel("Residual PC1")
    axis.set_ylabel("Residual PC2")
    axis.legend(markerscale=1.5)
    style_axis(axis, grid=None)
    figure.tight_layout()
    save_figure(figure, os.path.join(plots, "soft_residual_clusters"))
    plt.close(figure)

    counts = torch.bincount(labels, minlength=max(int(labels.max().item()) + 1, 1)).tolist()
    audit_values = [
        float(semantic_audit.get("nearest_centroid_balanced_accuracy", float("nan"))),
        float(semantic_audit.get("mlp_balanced_accuracy", float("nan"))),
    ]
    chance = float(semantic_audit.get("chance_balanced_accuracy", 1.0 / max(len(counts), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.75))
    axes[0].bar([str(i) for i in range(len(counts))], counts, color=[cluster_color(i) for i in range(len(counts))], width=0.68)
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Held-out samples")
    axes[0].set_title("Cluster balance", loc="left")
    style_axis(axes[0])
    panel_label(axes[0], "a")
    axes[1].bar(["Centroid", "MLP"], audit_values, color=[COLORS["sky"], COLORS["vermillion"]], width=0.62)
    axes[1].axhline(chance, color=COLORS["black"], linestyle="--", label="Chance")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Semantic predictability", loc="left")
    axes[1].legend()
    style_axis(axes[1])
    panel_label(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, os.path.join(plots, "cluster_balance_and_semantic_audit"))
    plt.close(fig)

    if raw_residual_features.numel() > 0:
        centroids = raw_centroids_from_labels(raw_residual_features, labels, len(counts))
        limit = max(float(centroids.abs().max().item()), 1e-6)
        figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.4))
        image = axis.imshow(centroids.numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label("Mean residual logit")
        axis.set_xlabel("Selected target rank")
        axis.set_ylabel("Cluster")
        figure.tight_layout()
        save_figure(figure, os.path.join(plots, "cluster_residual_logit_centroids"))
        plt.close(figure)


def maybe_plot_auto_k(output_dir: str, rows: List[Dict[str, Any]], selected_k: int) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    apply_nmi_style()
    plots = os.path.join(output_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    ks = [int(row["k"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    silhouettes = [float(row["centroid_silhouette"]) for row in rows]
    min_fractions = [float(row["min_cluster_fraction"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.75))
    axes[0].plot(ks, scores, marker="o", color=COLORS["blue"], label="Composite")
    axes[0].plot(ks, silhouettes, marker="s", color=COLORS["orange"], label="Silhouette")
    axes[0].axvline(selected_k, color=COLORS["black"], linestyle="--", label=f"Selected K={selected_k}")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("score")
    axes[0].legend()
    style_axis(axes[0])
    panel_label(axes[0], "a")
    axes[1].bar([str(k) for k in ks], min_fractions, color=COLORS["blue"])
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("smallest cluster fraction")
    axes[1].set_ylim(0.0, max(0.5, max(min_fractions) * 1.15))
    style_axis(axes[1])
    panel_label(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, os.path.join(plots, "auto_k_selection"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--decoupler-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", default=None, help="Optional prepared JSONL with is_safe/category fields for cluster-label association tests.")
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--decoupler-checkpoint", default=None)
    parser.add_argument("--decoupler-normalization", default=None)
    parser.add_argument("--purifier-checkpoint", default=None)
    parser.add_argument("--candidate-top-k", type=int, default=48)
    parser.add_argument("--residual-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--min-total-gain", type=float, default=0.002)
    parser.add_argument("--min-residual-gain", type=float, default=0.0)
    parser.add_argument("--min-semantic-gain", type=float, default=0.0)
    parser.add_argument(
        "--selection-alpha",
        type=float,
        default=0.05,
        help="Raw sign-flip p-value threshold for exploratory target selection.",
    )
    parser.add_argument("--require-positive-threshold-ci", action="store_true")
    parser.add_argument("--selected-top-k", type=int, default=16)
    parser.add_argument("--min-selected", type=int, default=4)
    parser.add_argument(
        "--search-target-subsets",
        action="store_true",
        help=(
            "Search strict target combinations on a nested target-selection split. The final "
            "cluster-evaluation split remains untouched until the selected combination is frozen."
        ),
    )
    parser.add_argument("--subset-search-size", type=int, default=4)
    parser.add_argument(
        "--subset-search-budget",
        type=float,
        default=1.0,
        help=(
            "Candidate-combination multiplier for subset search. Only the number of evaluated "
            "combinations is scaled; all available target-selection rows are used and shortlist/MLP "
            "settings remain fixed."
        ),
    )
    parser.add_argument("--subset-search-max-combinations", type=int, default=6000)
    parser.add_argument("--subset-search-shortlist", type=int, default=24)
    parser.add_argument(
        "--subset-search-max-samples",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option. Subset search always uses every available row in "
            "the isolated target-selection split."
        ),
    )
    parser.add_argument("--subset-search-train-fraction", type=float, default=0.6)
    parser.add_argument("--subset-search-min-cluster-fraction", type=float, default=0.15)
    parser.add_argument("--subset-search-kmeans-iters", type=int, default=20)
    parser.add_argument("--subset-search-kmeans-restarts", type=int, default=2)
    parser.add_argument("--subset-search-approx-batch-size", type=int, default=512)
    parser.add_argument("--subset-search-approx-ridge", type=float, default=1e-2)
    parser.add_argument("--subset-search-mlp-epochs", type=int, default=15)
    parser.add_argument("--subset-search-mlp-repeats", type=int, default=2)
    parser.add_argument("--subset-search-mlp-batch-size", type=int, default=256)
    parser.add_argument("--subset-search-semantic-weight", type=float, default=2.0)
    parser.add_argument(
        "--subset-search-max-semantic-bacc",
        type=float,
        default=0.55,
        help="Preferred ceiling on worst-repeat nested-validation semantic BAcc.",
    )
    parser.add_argument(
        "--subset-search-require-semantic-bacc",
        action="store_true",
        help="Fail subset search instead of falling back when no combination meets the BAcc ceiling.",
    )
    parser.add_argument("--subset-search-collapse-penalty", type=float, default=1.0)
    parser.add_argument("--subset-search-silhouette-weight", type=float, default=0.05)
    parser.add_argument("--subset-search-residual-weight", type=float, default=0.05)
    parser.add_argument("--allow-threshold-fallback", action="store_true")
    parser.add_argument(
        "--fallback-ranking",
        choices=["threshold_margin", "residual_fraction", "residual_gain"],
        default="threshold_margin",
        help="Ranking used only when no strict target set is available.",
    )
    parser.add_argument("--fit-max-samples", type=int, default=30000)
    parser.add_argument("--selection-fraction-of-val", type=float, default=0.5)
    parser.add_argument(
        "--probe-data-source",
        choices=["decoupler_train_val", "heldout_test"],
        default="decoupler_train_val",
        help=(
            "decoupler_train_val preserves the legacy split. heldout_test fits and selects posthoc "
            "probes only on disjoint partitions of the decoupler test split, keeping them isolated "
            "from all probe data used during decoupler/purifier training."
        ),
    )
    parser.add_argument("--heldout-probe-train-fraction", type=float, default=0.35)
    parser.add_argument("--heldout-selection-fraction", type=float, default=0.25)
    parser.add_argument(
        "--cluster-eval-scope",
        choices=["heldout_val", "all_val", "test", "all_cached"],
        default="heldout_val",
        help=(
            "Samples to assign clusters for after target selection. heldout_val keeps the original "
            "non-overlapping validation half when selection_fraction_of_val < 1. all_val assigns all "
            "validation samples. test uses the decoupler's untouched test split and uses all validation "
            "samples for target selection. all_cached assigns every cached activation row, useful for "
            "descriptive audits after target selection is fixed."
        ),
    )
    parser.add_argument(
        "--id-regex",
        default="",
        help=(
            "Optional regular expression used to keep only matching activation ids in train, "
            "selection, and cluster-evaluation splits. Useful for fixed-step generated-token "
            "activations such as --id-regex '::step16$'."
        ),
    )
    parser.add_argument("--semantic-hidden-dim", type=int, default=512)
    parser.add_argument("--residual-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--semantic-source",
        choices=["actual", "recovered"],
        default="actual",
        help=(
            "actual uses frozen Z1+prev as the semantic teacher input. recovered "
            "first learns meta->Z1/prev and matches the purifier final residual-fraction audit more closely."
        ),
    )
    parser.add_argument(
        "--residual-head-mode",
        choices=["additive_delta", "standalone_meta_delta"],
        default="additive_delta",
        help=(
            "additive_delta trains meta to add a correction on top of the semantic teacher. "
            "standalone_meta_delta trains a standalone meta predictor and uses meta_logits - semantic_logits as residual features."
        ),
    )
    parser.add_argument(
        "--probe-repeats",
        type=int,
        default=1,
        help="Train and average repeated semantic/meta probes; use 3 to match the purifier final audit.",
    )
    parser.add_argument("--probe-dropout", type=float, default=0.1)
    parser.add_argument("--semantic-epochs", type=int, default=15)
    parser.add_argument("--residual-epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--delta-l2", type=float, default=1e-4)
    parser.add_argument(
        "--probe-profile",
        choices=["legacy_posthoc", "purifier_dynamic", "decoupler_final"],
        default="legacy_posthoc",
        help=(
            "purifier_dynamic reproduces the probe used for target refresh during purifier training; "
            "decoupler_final reproduces the final purifier audit. Both use recovered-semantic controls."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--permutation-tests", type=int, default=500)
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--auto-k-metric", choices=["composite", "silhouette"], default="composite")
    parser.add_argument("--min-cluster-fraction", type=float, default=0.05)
    parser.add_argument("--auto-k-small-cluster-weight", type=float, default=1.0)
    parser.add_argument("--auto-k-imbalance-weight", type=float, default=0.25)
    parser.add_argument("--pca-dim", type=int, default=0)
    parser.add_argument("--clip-quantile", type=float, default=0.99)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--kmeans-restarts", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.residual_fraction_threshold < 1.0:
        raise ValueError("--residual-fraction-threshold must be in (0, 1).")
    if not 0.0 < args.selection_fraction_of_val <= 1.0:
        raise ValueError("--selection-fraction-of-val must be in (0, 1].")
    if args.probe_repeats <= 0:
        raise ValueError("--probe-repeats must be positive.")
    if args.search_target_subsets:
        if args.subset_search_size < 2:
            raise ValueError("--subset-search-size must be at least 2.")
        if args.selected_top_k != args.subset_search_size:
            raise ValueError(
                "Subset search requires --selected-top-k to equal --subset-search-size."
            )
        if args.subset_search_max_combinations <= 0 or args.subset_search_shortlist <= 0:
            raise ValueError("Subset-search combination and shortlist counts must be positive.")
        if not 0.0 < args.subset_search_budget <= 8.0:
            raise ValueError("--subset-search-budget must be in (0, 8].")
        if not 0.0 < args.subset_search_train_fraction < 1.0:
            raise ValueError("--subset-search-train-fraction must be in (0, 1).")
        if not 0.0 < args.subset_search_min_cluster_fraction < 0.5:
            raise ValueError("--subset-search-min-cluster-fraction must be in (0, 0.5).")
        if not 0.5 <= args.subset_search_max_semantic_bacc <= 1.0:
            raise ValueError("--subset-search-max-semantic-bacc must be in [0.5, 1.0].")
        if args.num_clusters != 2:
            raise ValueError("The current strict subset search is designed for --num-clusters 2.")
    if not 0.0 < args.heldout_probe_train_fraction < 1.0:
        raise ValueError("--heldout-probe-train-fraction must be in (0, 1).")
    if not 0.0 < args.heldout_selection_fraction < 1.0:
        raise ValueError("--heldout-selection-fraction must be in (0, 1).")
    if args.heldout_probe_train_fraction + args.heldout_selection_fraction >= 1.0:
        raise ValueError("Heldout probe-train and selection fractions must sum to less than 1.")
    if args.auto_k and (args.k_min < 2 or args.k_max < args.k_min):
        raise ValueError("Auto-K requires 2 <= --k-min <= --k-max.")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = read_json(os.path.join(args.decoupler_dir, "config.json"))
    requested_probe_settings = {
        "profile": args.probe_profile,
        "probe_repeats": int(args.probe_repeats),
        "probe_dropout": float(args.probe_dropout),
        "semantic_epochs": int(args.semantic_epochs),
        "residual_epochs": int(args.residual_epochs),
        "semantic_hidden_dim": int(args.semantic_hidden_dim),
        "residual_hidden_dim": int(args.residual_hidden_dim),
        "probe_batch_size": int(args.probe_batch_size),
        "probe_lr": float(args.probe_lr),
        "probe_weight_decay": float(args.probe_weight_decay),
        "semantic_source": args.semantic_source,
        "residual_head_mode": args.residual_head_mode,
    }
    args.aligned_target_probe = str(config.get("purifier_dynamic_target_probe") or "linear")
    args.aligned_semantic_recovery_probe = "mlp"
    if args.probe_profile in {"purifier_dynamic", "decoupler_final"}:
        if args.probe_profile == "purifier_dynamic":
            probe_epochs = int(
                config.get("purifier_dynamic_probe_epochs")
                or config.get("purifier_probe_epochs")
                or config.get("probe_epochs")
                or 10
            )
            probe_hidden = int(
                config.get("purifier_dynamic_probe_hidden_dim")
                or config.get("purifier_probe_hidden_dim")
                or config.get("purifier_hidden_dim")
                or config.get("hidden_dim")
                or 1024
            )
            args.probe_repeats = int(config.get("purifier_dynamic_probe_repeats") or args.probe_repeats)
            args.aligned_semantic_recovery_probe = str(
                config.get("purifier_dynamic_semantic_probe") or "linear"
            )
        else:
            probe_epochs = int(config.get("purifier_probe_epochs") or config.get("probe_epochs") or 10)
            probe_hidden = int(
                config.get("purifier_probe_hidden_dim")
                or config.get("purifier_hidden_dim")
                or config.get("hidden_dim")
                or 1024
            )
            args.probe_repeats = int(config.get("purifier_final_probe_repeats") or args.probe_repeats)
            args.aligned_semantic_recovery_probe = "mlp"
        args.probe_dropout = float(config.get("dropout", args.probe_dropout))
        args.semantic_epochs = probe_epochs
        args.residual_epochs = probe_epochs
        args.semantic_hidden_dim = probe_hidden
        args.residual_hidden_dim = probe_hidden
        args.probe_batch_size = int(config.get("probe_batch_size") or args.probe_batch_size)
        args.probe_lr = float(config.get("probe_lr") or args.probe_lr)
        args.probe_weight_decay = float(config.get("probe_weight_decay") or args.probe_weight_decay)
        args.semantic_source = "recovered"
        args.residual_head_mode = "standalone_meta_delta"
    prev_layer = int(config["prev_layer"])
    layer_i = int(config["layer_i"])
    next_layer = int(config["next_layer"])
    all_ids = load_activation_ids(args.activation_dir, layer_i)
    id_pattern = re.compile(args.id_regex) if args.id_regex else None
    n = len(all_ids)
    train_indices, val_indices, test_indices, split_info = load_decoupler_split_indices(
        args.decoupler_dir,
        all_ids,
        config,
        args.seed,
    )
    selection_eval_overlap = False
    posthoc_split_summary: Dict[str, Any] = {
        "probe_data_source": args.probe_data_source,
        "isolated_from_decoupler_train_val": args.probe_data_source == "heldout_test",
    }
    if args.probe_data_source == "heldout_test":
        if args.cluster_eval_scope != "test":
            raise ValueError("--probe-data-source heldout_test requires --cluster-eval-scope test.")
        if test_indices.numel() < 3:
            raise ValueError(
                "--probe-data-source heldout_test requires at least three rows in the decoupler test split."
            )
        shuffled_test = test_indices[
            torch.randperm(test_indices.numel(), generator=torch.Generator().manual_seed(args.seed + 991))
        ]
        probe_train_size = max(1, int(shuffled_test.numel() * args.heldout_probe_train_fraction))
        selection_size = max(1, int(shuffled_test.numel() * args.heldout_selection_fraction))
        if probe_train_size + selection_size >= shuffled_test.numel():
            selection_size = max(1, shuffled_test.numel() - probe_train_size - 1)
        train_indices = shuffled_test[:probe_train_size]
        selection_indices = shuffled_test[probe_train_size : probe_train_size + selection_size]
        eval_indices = shuffled_test[probe_train_size + selection_size :]
        posthoc_split_summary.update(
            {
                "source_test_n": int(test_indices.numel()),
                "probe_train_n": int(train_indices.numel()),
                "target_selection_n": int(selection_indices.numel()),
                "cluster_eval_n": int(eval_indices.numel()),
                "probe_train_fraction": float(args.heldout_probe_train_fraction),
                "selection_fraction": float(args.heldout_selection_fraction),
            }
        )
    else:
        val_shuffle = val_indices[
            torch.randperm(val_indices.numel(), generator=torch.Generator().manual_seed(args.seed + 991))
        ]
        if args.cluster_eval_scope == "test":
            if test_indices.numel() == 0:
                raise ValueError(
                    "--cluster-eval-scope test requires a non-empty decoupler test split; "
                    "train with --test-ratio > 0."
                )
            selection_indices = val_shuffle
            eval_indices = test_indices
        elif args.selection_fraction_of_val >= 1.0:
            selection_indices = val_shuffle
            eval_indices = val_shuffle
            selection_eval_overlap = True
        else:
            selection_size = max(1, int(val_shuffle.numel() * args.selection_fraction_of_val))
            selection_indices = val_shuffle[:selection_size]
            eval_indices = val_shuffle[selection_size:]
            if eval_indices.numel() == 0:
                raise ValueError("Validation split is too small to create a held-out clustering/evaluation set.")
        if args.cluster_eval_scope == "all_val":
            eval_indices = val_shuffle
            selection_eval_overlap = True
        elif args.cluster_eval_scope == "all_cached":
            eval_indices = torch.arange(n, dtype=torch.long)
            selection_eval_overlap = True
        posthoc_split_summary.update(
            {
                "probe_train_n": int(train_indices.numel()),
                "target_selection_n": int(selection_indices.numel()),
                "cluster_eval_n": int(eval_indices.numel()),
            }
        )
    if id_pattern is not None:
        def filter_indices(indices: torch.Tensor) -> torch.Tensor:
            kept = [int(idx) for idx in indices.tolist() if id_pattern.search(all_ids[int(idx)])]
            return torch.tensor(kept, dtype=torch.long)

        train_indices = filter_indices(train_indices)
        selection_indices = filter_indices(selection_indices)
        eval_indices = filter_indices(eval_indices)
        if train_indices.numel() == 0 or selection_indices.numel() == 0 or eval_indices.numel() == 0:
            raise ValueError(
                f"--id-regex {args.id_regex!r} removed all rows from at least one split: "
                f"train={train_indices.numel()}, selection={selection_indices.numel()}, eval={eval_indices.numel()}."
            )
    if args.fit_max_samples > 0 and train_indices.numel() > args.fit_max_samples:
        keep = torch.randperm(train_indices.numel(), generator=torch.Generator().manual_seed(args.seed + 313))[: args.fit_max_samples]
        train_indices = train_indices[keep]
    train_ids = [all_ids[int(idx)] for idx in train_indices.tolist()]
    selection_ids = [all_ids[int(idx)] for idx in selection_indices.tolist()]
    eval_ids = [all_ids[int(idx)] for idx in eval_indices.tolist()]
    posthoc_split_summary.update(
        {
            "probe_train_n_after_filters": len(train_ids),
            "target_selection_n_after_filters": len(selection_ids),
            "cluster_eval_n_after_filters": len(eval_ids),
        }
    )

    candidate_csv = resolve_candidate_csv(args)
    candidate_rows = read_candidate_rows(candidate_csv, args.candidate_top_k)
    candidate_neurons = torch.tensor([row["neuron"] for row in candidate_rows], dtype=torch.long)
    if args.probe_profile == "decoupler_final":
        probe_candidate_rows = read_candidate_rows(candidate_csv, 0)
    else:
        probe_candidate_rows = candidate_rows
    probe_candidate_neurons = torch.tensor(
        [row["neuron"] for row in probe_candidate_rows], dtype=torch.long
    )
    probe_feature_weights = torch.tensor(
        [
            float(row.get("training_weight", 1.0))
            if math.isfinite(float(row.get("training_weight", 1.0)))
            else 1.0
            for row in probe_candidate_rows
        ],
        dtype=torch.float32,
    )
    probe_column_by_neuron = {
        int(neuron): col for col, neuron in enumerate(probe_candidate_neurons.tolist())
    }
    candidate_probe_columns = torch.tensor(
        [probe_column_by_neuron[int(neuron)] for neuron in candidate_neurons.tolist()],
        dtype=torch.long,
    )
    print(
        f"[soft-residual] splits train={len(train_ids)} selection={len(selection_ids)} "
        f"heldout={len(eval_ids)} candidates={len(candidate_rows)} probe_targets={len(probe_candidate_rows)} "
        f"candidate_csv={candidate_csv}",
        flush=True,
    )

    # Load dimensions before constructing the frozen models.
    next_probe = load_layer_matrix(args.activation_dir, next_layer, train_ids[:1], None)
    decoupler, purifier, normalization, config, model_info = load_models(args, next_probe.shape[1], device)
    train = encode_partition(
        args.activation_dir, train_ids, prev_layer, layer_i, next_layer, probe_candidate_neurons,
        decoupler, purifier, normalization, device, args.encode_batch_size, "probe train",
    )
    selection = encode_partition(
        args.activation_dir, selection_ids, prev_layer, layer_i, next_layer, probe_candidate_neurons,
        decoupler, purifier, normalization, device, args.encode_batch_size, "target selection",
    )
    evaluation = encode_partition(
        args.activation_dir, eval_ids, prev_layer, layer_i, next_layer, probe_candidate_neurons,
        decoupler, purifier, normalization, device, args.encode_batch_size, "cluster evaluation",
    )

    if args.probe_profile in {"purifier_dynamic", "decoupler_final"}:
        probe_bundle = fit_decoupler_final_aligned_probes(
            train, selection, evaluation, args, device, probe_feature_weights
        )
    else:
        probe_bundle = fit_legacy_posthoc_probes(train, selection, evaluation, args, device)
    semantic_selection = probe_bundle["semantic_selection"].index_select(1, candidate_probe_columns)
    semantic_eval = probe_bundle["semantic_eval"].index_select(1, candidate_probe_columns)
    delta_selection = probe_bundle["delta_selection"].index_select(1, candidate_probe_columns)
    delta_eval = probe_bundle["delta_eval"].index_select(1, candidate_probe_columns)
    candidate_train_y = train["y"].index_select(1, candidate_probe_columns)
    candidate_selection_y = selection["y"].index_select(1, candidate_probe_columns)
    candidate_eval_y = evaluation["y"].index_select(1, candidate_probe_columns)
    semantic_history = probe_bundle["semantic_history"]
    residual_history = probe_bundle["residual_history"]
    print(
        f"[soft-residual] probes semantic_loss={semantic_history[-1]:.5f} "
        f"combined_loss={residual_history[-1]:.5f} repeats={args.probe_repeats} "
        f"profile={args.probe_profile} semantic_source={args.semantic_source} "
        f"residual_mode={args.residual_head_mode}",
        flush=True,
    )

    selection_rows, _ = contribution_rows(
        candidate_rows, candidate_train_y, candidate_selection_y, semantic_selection, delta_selection,
        args.residual_fraction_threshold, args.bootstrap_samples, args.permutation_tests, args.seed + 1777,
    )
    eligible = []
    for col, row in enumerate(selection_rows):
        passes = (
            float(row["total_gain_nats"]) >= args.min_total_gain
            and float(row["residual_gain_nats"]) >= args.min_residual_gain
            and float(row["semantic_gain_nats"]) >= args.min_semantic_gain
            and float(row["residual_fraction"]) >= args.residual_fraction_threshold
            and float(row["threshold_signflip_p"]) <= args.selection_alpha
        )
        if args.require_positive_threshold_ci:
            passes = passes and float(row["threshold_margin_ci_low"]) > 0.0
        row["passes_strict_threshold"] = bool(passes)
        if passes:
            eligible.append(col)
    eligible.sort(key=lambda col: float(selection_rows[col]["threshold_margin_nats"]), reverse=True)
    strict_count = len(eligible)
    strict_pool = list(eligible)
    fallback_used = False
    if len(eligible) < args.min_selected:
        if not args.allow_threshold_fallback:
            write_csv(os.path.join(args.output_dir, "residual_target_selection.csv"), selection_rows)
            raise RuntimeError(
                f"Only {len(eligible)} targets pass residual fraction {args.residual_fraction_threshold:.2f}; "
                "rerun with --allow-threshold-fallback for an explicitly exploratory top-margin analysis."
            )
        fallback_used = True
        fallback = [
            col for col, row in enumerate(selection_rows)
            if float(row["total_gain_nats"]) >= args.min_total_gain
            and float(row["residual_gain_nats"]) > 0.0
            and float(row["semantic_gain_nats"]) >= args.min_semantic_gain
        ]
        fallback_key = {
            "threshold_margin": "threshold_margin_nats",
            "residual_fraction": "residual_fraction",
            "residual_gain": "residual_gain_nats",
        }[args.fallback_ranking]
        fallback.sort(
            key=lambda col: (
                float(selection_rows[col][fallback_key])
                if math.isfinite(float(selection_rows[col][fallback_key]))
                else -math.inf,
                float(selection_rows[col]["residual_gain_nats"]),
            ),
            reverse=True,
        )
        eligible = fallback[: max(args.min_selected, args.selected_top_k)]
    subset_search_summary: Dict[str, Any] = {
        "enabled": False,
        "requested": bool(args.search_target_subsets),
    }
    subset_search_applied = False
    if args.search_target_subsets and len(strict_pool) >= args.subset_search_size:
        eligible, subset_search_summary = search_target_subset(
            strict_pool,
            selection_rows,
            candidate_neurons,
            delta_selection,
            semantic_selection,
            args,
            device,
        )
        fallback_used = False
        subset_search_applied = True
        print(
            f"[soft-residual] subset search selected neurons="
            f"{subset_search_summary['selected_neurons']} "
            f"from strict_pool={len(strict_pool)}",
            flush=True,
        )
    elif args.search_target_subsets:
        subset_search_summary.update(
            {
                "reason": "insufficient_strict_targets",
                "strict_pool_size": len(strict_pool),
                "required_subset_size": int(args.subset_search_size),
                "final_evaluation_used_for_search": False,
            }
        )
        write_json(
            os.path.join(args.output_dir, "target_subset_search_summary.json"),
            subset_search_summary,
        )
        print(
            f"[soft-residual] subset search skipped: strict={len(strict_pool)} "
            f"required={args.subset_search_size}; using configured fallback/ranking",
            flush=True,
        )
    if args.selected_top_k > 0 and not subset_search_applied:
        eligible = eligible[: args.selected_top_k]
    if not eligible:
        raise RuntimeError("No targets are available for residual clustering.")
    print(
        f"[soft-residual] target selection strict={strict_count} used={len(eligible)} "
        f"threshold={args.residual_fraction_threshold:.2f} fallback={fallback_used}",
        flush=True,
    )
    selected_cols = torch.tensor(eligible, dtype=torch.long)
    selected_neurons = candidate_neurons[selected_cols]
    selected_set = set(int(idx) for idx in eligible)
    for col, row in enumerate(selection_rows):
        row["selected_for_clustering"] = col in selected_set
        row["selection_fallback_used"] = fallback_used
        row["selected_by_subset_search"] = bool(subset_search_applied and col in selected_set)
    write_csv(os.path.join(args.output_dir, "residual_target_selection.csv"), selection_rows)

    selected_candidate_rows = [candidate_rows[col] for col in eligible]
    evaluation_rows, _ = contribution_rows(
        selected_candidate_rows,
        candidate_train_y.index_select(1, selected_cols),
        candidate_eval_y.index_select(1, selected_cols),
        semantic_eval.index_select(1, selected_cols),
        delta_eval.index_select(1, selected_cols),
        args.residual_fraction_threshold,
        args.bootstrap_samples,
        args.permutation_tests,
        args.seed + 2777,
    )
    for row, source_col in zip(evaluation_rows, eligible):
        row["selection_residual_fraction"] = selection_rows[source_col]["residual_fraction"]
        row["selection_threshold_margin_nats"] = selection_rows[source_col]["threshold_margin_nats"]
    write_csv(os.path.join(args.output_dir, "residual_target_heldout_evaluation.csv"), evaluation_rows)
    heldout_ratios = [
        float(row["residual_fraction"])
        for row in evaluation_rows
        if math.isfinite(float(row["residual_fraction"]))
    ]
    heldout_ratio_mean = sum(heldout_ratios) / len(heldout_ratios) if heldout_ratios else float("nan")
    print(
        f"[soft-residual] heldout confirmation targets={len(evaluation_rows)} "
        f"residual_fraction_mean={heldout_ratio_mean:.4f}",
        flush=True,
    )

    residual_selection = delta_selection[:, selected_cols]
    residual_eval = delta_eval[:, selected_cols]
    model_selection, transform_state = fit_transform_state(
        residual_selection, args.pca_dim, args.clip_quantile, args.l2_normalize
    )
    model_eval = apply_transform(residual_eval, transform_state)
    auto_k_candidates = []
    if args.auto_k:
        best = None
        k_max = min(int(args.k_max), max(2, model_selection.shape[0] - 1))
        for k in range(int(args.k_min), k_max + 1):
            candidate_labels, candidate_centroids, candidate_inertia = run_kmeans(
                model_selection,
                k,
                args.kmeans_iters,
                args.kmeans_restarts,
                args.seed + 17 * k,
            )
            counts = torch.bincount(candidate_labels, minlength=k).float()
            fractions = counts / max(candidate_labels.numel(), 1)
            min_fraction = float(fractions.min().item())
            silhouette = centroid_silhouette_score(
                model_selection, candidate_labels, candidate_centroids
            )
            small_penalty = max(
                0.0, float(args.min_cluster_fraction) - min_fraction
            ) / max(float(args.min_cluster_fraction), 1e-6)
            imbalance_penalty = float(fractions.std(unbiased=False).item())
            score = (
                silhouette
                if args.auto_k_metric == "silhouette"
                else silhouette
                - args.auto_k_small_cluster_weight * small_penalty
                - args.auto_k_imbalance_weight * imbalance_penalty
            )
            row = {
                "k": k,
                "score": float(score),
                "centroid_silhouette": float(silhouette),
                "inertia": float(candidate_inertia),
                "min_cluster_fraction": min_fraction,
                "max_cluster_fraction": float(fractions.max().item()),
                "small_cluster_penalty": float(small_penalty),
                "imbalance_penalty": imbalance_penalty,
                "cluster_counts": " ".join(str(int(value)) for value in counts.tolist()),
            }
            auto_k_candidates.append(row)
            if best is None or row["score"] > best["score"]:
                best = {
                    **row,
                    "labels": candidate_labels,
                    "centroids": candidate_centroids,
                }
        assert best is not None
        args.num_clusters = int(best["k"])
        selection_labels = best["labels"]
        centroids = best["centroids"]
        inertia = float(best["inertia"])
        write_csv(os.path.join(args.output_dir, "auto_k_candidates.csv"), auto_k_candidates)
        print(
            f"[soft-residual] auto-k selected k={args.num_clusters} "
            f"score={best['score']:.4f} silhouette={best['centroid_silhouette']:.4f}",
            flush=True,
        )
    else:
        selection_labels, centroids, inertia = run_kmeans(
            model_selection, args.num_clusters, args.kmeans_iters, args.kmeans_restarts, args.seed
        )
    eval_labels = nearest_labels(model_eval, centroids)

    # Conservative semantic-confound diagnostic: can semantic-teacher logits
    # alone reproduce the residual cluster assignment on held-out samples?
    semantic_selection_features, semantic_transform = fit_transform_state(
        semantic_selection[:, selected_cols], 0, args.clip_quantile, False
    )
    semantic_eval_features = apply_transform(semantic_eval[:, selected_cols], semantic_transform)
    semantic_centroids = raw_centroids_from_labels(semantic_selection_features, selection_labels, args.num_clusters)
    semantic_pred_labels = nearest_labels(semantic_eval_features, semantic_centroids)
    semantic_cluster_accuracy = float((semantic_pred_labels == eval_labels).float().mean().item())
    semantic_cluster_balanced_accuracy = balanced_accuracy(eval_labels, semantic_pred_labels, args.num_clusters)
    semantic_mlp_audit = semantic_cluster_probe(
        semantic_selection_features,
        selection_labels,
        semantic_eval_features,
        eval_labels,
        args.num_clusters,
        device,
        args.seed + 4111,
    )
    semantic_mlp_predictions = semantic_mlp_audit.pop("predictions")
    semantic_mlp_test = semantic_predictability_test(
        eval_labels,
        semantic_mlp_predictions,
        args.num_clusters,
        args.bootstrap_samples,
        args.permutation_tests,
        args.seed + 4127,
    )

    assignment_rows = []
    for idx, sample_id in enumerate(eval_ids):
        assignment_rows.append(
            {
                "id": sample_id,
                "activation_cluster": int(eval_labels[idx].item()),
                "cluster": int(eval_labels[idx].item()),
                "soft_residual_norm": float(residual_eval[idx].norm().item()),
                "model_feature_norm": float(model_eval[idx].norm().item()),
                "semantic_teacher_norm": float(semantic_eval[idx, selected_cols].norm().item()),
            }
        )
    assignment_path = os.path.join(args.output_dir, "soft_residual_cluster_assignments.csv")
    safety_association = None
    if args.data:
        safety_association = compute_safety_association(
            assignment_rows,
            args.data,
            args.permutation_tests,
            args.seed + 7919,
            args.output_dir,
            args.no_plots,
        )
    write_csv(assignment_path, assignment_rows)

    feature_path = os.path.join(args.output_dir, "soft_residual_cluster_features.pt")
    feature_summary = {
        "source": "soft_residual_logits_from_purified_meta",
        "probe_profile": args.probe_profile,
        "probe_data_source": args.probe_data_source,
        "target_probe": args.aligned_target_probe,
        "semantic_recovery_probe": args.aligned_semantic_recovery_probe,
        "semantic_source": args.semantic_source,
        "residual_head_mode": args.residual_head_mode,
        "probe_repeats": int(args.probe_repeats),
        "selected_target_count": int(selected_neurons.numel()),
        "strict_threshold_passed_count": strict_count,
        "selection_fallback_used": fallback_used,
        "target_subset_search": subset_search_summary,
        "residual_fraction_threshold": args.residual_fraction_threshold,
        "selection_split_n": len(selection_ids),
        "evaluation_split_n": len(eval_ids),
        "cluster_eval_scope": args.cluster_eval_scope,
        "id_regex": args.id_regex,
        "selection_eval_overlap": selection_eval_overlap,
        "decoupler_split": split_info,
    }
    torch.save(
        {
            "ids": eval_ids,
            "model_features": model_eval.float(),
            "raw_residual_features": residual_eval.float(),
            "labels": eval_labels.long(),
            "selected_neurons": selected_neurons.long(),
            "summary": feature_summary,
        },
        feature_path,
    )

    threshold = normalization["target"]["threshold"].float().cpu()
    target_path = os.path.join(args.output_dir, "soft_residual_intervention_targets.pt")
    torch.save(
        {
            "schema_version": 2,
            "selected_neurons": selected_neurons.long(),
            "val_target_binary": candidate_eval_y[:, selected_cols].to(torch.uint8),
            "binary_activation_threshold": threshold,
            "continuous_mean": normalization["target"].get("mean"),
            "continuous_std": normalization["target"].get("std"),
            "train_ids": train_ids,
            "selection_ids": selection_ids,
            "val_ids": eval_ids,
            "test_ids": eval_ids if args.cluster_eval_scope == "test" else [],
            "evaluation_ids": eval_ids,
            "metadata": {
                "activation_dir": args.activation_dir,
                "model_label": config.get("model_label"),
                "prev_layer": prev_layer,
                "layer_i": layer_i,
                "next_layer": next_layer,
                "target_mode": config.get("target_mode"),
                "selection_source": "soft_residual_posthoc",
                "probe_profile": args.probe_profile,
                "probe_data_source": args.probe_data_source,
                "target_probe": args.aligned_target_probe,
                "semantic_recovery_probe": args.aligned_semantic_recovery_probe,
                "semantic_source": args.semantic_source,
                "residual_head_mode": args.residual_head_mode,
                "probe_repeats": int(args.probe_repeats),
                "residual_fraction_threshold": args.residual_fraction_threshold,
                "strict_threshold_passed_count": strict_count,
                "selection_fallback_used": fallback_used,
                "target_subset_search": subset_search_summary,
                "selection_eval_overlap": selection_eval_overlap,
                "cluster_eval_scope": args.cluster_eval_scope,
                "decoupler_split": split_info,
                "id_regex": args.id_regex,
            },
        },
        target_path,
    )

    artifact_path = os.path.join(args.output_dir, "soft_residual_probe_artifact.pt")
    torch.save(
        {
            "probe_profile": args.probe_profile,
            "probe_data_source": args.probe_data_source,
            "probe_states_saved": False,
            "semantic_hidden_dim": args.semantic_hidden_dim,
            "residual_input_dim": int(train["meta"].shape[1]),
            "residual_hidden_dim": args.residual_hidden_dim,
            "best_semantic_control_counts": probe_bundle.get("best_semantic_control_counts"),
            "candidate_neurons": candidate_neurons,
            "selected_candidate_columns": selected_cols,
            "selected_neurons": selected_neurons,
            "cluster_transform": transform_state,
            "cluster_centroids": centroids,
            "cluster_selection_labels": selection_labels,
            "model_info": model_info,
            "args": vars(args),
        },
        artifact_path,
    )

    cluster_counts = torch.bincount(eval_labels, minlength=args.num_clusters).tolist()
    selected_ratios = [float(selection_rows[col]["residual_fraction"]) for col in eligible]
    selected_source_ratios = []
    for col in eligible:
        try:
            value = float(selection_rows[col].get("source_residual_fraction", float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            selected_source_ratios.append(value)
    summary = {
        "method": "semantic_residual_probe_aligned_posthoc",
        "probe_profile": args.probe_profile,
        "probe_data_source": args.probe_data_source,
        "target_probe": args.aligned_target_probe,
        "semantic_recovery_probe": args.aligned_semantic_recovery_probe,
        "requested_probe_settings": requested_probe_settings,
        "semantic_source": args.semantic_source,
        "residual_head_mode": args.residual_head_mode,
        "probe_repeats": int(args.probe_repeats),
        "probe_dropout": float(args.probe_dropout),
        "semantic_hidden_dim": int(args.semantic_hidden_dim),
        "residual_hidden_dim": int(args.residual_hidden_dim),
        "semantic_epochs": int(args.semantic_epochs),
        "residual_epochs": int(args.residual_epochs),
        "decoupler_unchanged": True,
        "model": model_info,
        "layers": {"prev": prev_layer, "i": layer_i, "next": next_layer},
        "candidate_count": len(candidate_rows),
        "probe_target_count": len(probe_candidate_rows),
        "probe_target_weight_sum": float(probe_feature_weights.sum().item()),
        "strict_threshold_passed_count": strict_count,
        "selected_target_count": int(selected_neurons.numel()),
        "selected_neurons": selected_neurons.tolist(),
        "target_subset_search": subset_search_summary,
        "selection_fallback_used": fallback_used,
        "residual_fraction_threshold": args.residual_fraction_threshold,
        "selected_residual_fraction_mean": sum(selected_ratios) / len(selected_ratios),
        "selected_residual_fraction_min": min(selected_ratios),
        "selected_residual_fraction_max": max(selected_ratios),
        "heldout_residual_fraction_mean": (
            sum(heldout_ratios) / len(heldout_ratios) if heldout_ratios else None
        ),
        "heldout_residual_fraction_min": min(heldout_ratios) if heldout_ratios else None,
        "heldout_residual_fraction_max": max(heldout_ratios) if heldout_ratios else None,
        "selected_source_residual_fraction_mean": (
            sum(selected_source_ratios) / len(selected_source_ratios) if selected_source_ratios else None
        ),
        "residual_fraction_definition": (
            "selected_residual_fraction is measured on the target-selection split; heldout_residual_fraction "
            "uses the same frozen probes and selected targets on the disjoint cluster-evaluation split. Both use "
            "(CE(semantic control) - CE(meta probe)) / (CE(prior) - CE(meta probe))."
        ),
        "source_residual_fraction_definition": (
            "source_residual_fraction is read from the decoupler candidate CSV and usually comes from the "
            "final confirmatory probe used when exporting purifier targets."
        ),
        "probe_train_n": len(train_ids),
        "selection_n": len(selection_ids),
        "heldout_cluster_eval_n": len(eval_ids),
        "decoupler_split": split_info,
        "posthoc_split": posthoc_split_summary,
        "cluster_eval_scope": args.cluster_eval_scope,
        "id_regex": args.id_regex,
        "selection_eval_overlap": selection_eval_overlap,
        "num_clusters": args.num_clusters,
        "auto_k": bool(args.auto_k),
        "auto_k_candidates": auto_k_candidates,
        "cluster_counts": cluster_counts,
        "cluster_inertia_selection": float(inertia),
        "semantic_cluster_predictability": {
            "nearest_centroid_accuracy": semantic_cluster_accuracy,
            "nearest_centroid_balanced_accuracy": semantic_cluster_balanced_accuracy,
            "mlp_accuracy": semantic_mlp_audit["accuracy"],
            "mlp_balanced_accuracy": semantic_mlp_audit["balanced_accuracy"],
            "mlp_balanced_accuracy_bootstrap_ci95": semantic_mlp_test["bootstrap_ci95"],
            "mlp_independence_permutation_p": semantic_mlp_test["permutation_p"],
            "mlp_significant_above_chance_0p05": semantic_mlp_test[
                "significant_above_chance_0p05"
            ],
            "chance_balanced_accuracy": 1.0 / args.num_clusters,
            "interpretation": (
                "This audit is evaluated only after the target subset is frozen. Held-out balanced "
                "accuracy significantly above chance indicates remaining measured-semantic cluster confounding."
            ),
        },
        "is_safe_association": safety_association,
        "training": {
            "semantic_final_loss": semantic_history[-1],
            "residual_final_loss": residual_history[-1],
            "best_semantic_control_counts": probe_bundle.get("best_semantic_control_counts"),
        },
        "artifacts": {
            "assignments": assignment_path,
            "features": feature_path,
            "intervention_targets": target_path,
            "probe_artifact": artifact_path,
            "target_selection": os.path.join(args.output_dir, "residual_target_selection.csv"),
            "target_heldout_evaluation": os.path.join(
                args.output_dir, "residual_target_heldout_evaluation.csv"
            ),
            "auto_k_candidates": (
                os.path.join(args.output_dir, "auto_k_candidates.csv")
                if args.auto_k
                else None
            ),
            "target_subset_search": (
                os.path.join(args.output_dir, "target_subset_search.csv")
                if subset_search_applied
                else None
            ),
            "target_subset_search_summary": (
                os.path.join(args.output_dir, "target_subset_search_summary.json")
                if args.search_target_subsets
                else None
            ),
            "plots": [
                "plots/target_residual_fraction.png",
                "plots/selected_target_information_decomposition.png",
                "plots/selected_target_threshold_margin_ci.png",
                "plots/target_subset_search.png",
                "plots/soft_residual_clusters.png",
                "plots/cluster_balance_and_semantic_audit.png",
                "plots/cluster_residual_logit_centroids.png",
                "plots/cluster_is_safe_unsafe_rate.png",
                "plots/cluster_safety_category_heatmap.png",
            ] if not args.no_plots else [],
        },
        "caution": (
            "A fallback target set is exploratory and does not satisfy the requested residual-fraction threshold."
            if fallback_used else
            "Targets satisfy the configured operational threshold on the selection split; semantic independence remains an empirical, not absolute, claim."
        ),
    }
    write_json(os.path.join(args.output_dir, "soft_residual_cluster_summary.json"), summary)
    if not args.no_plots:
        maybe_plot(
            args.output_dir,
            selection_rows,
            model_eval,
            residual_eval,
            eval_labels,
            summary["semantic_cluster_predictability"],
            args.residual_fraction_threshold,
        )
        maybe_plot_auto_k(args.output_dir, auto_k_candidates, args.num_clusters)
    console_summary = {
        "strict_targets": strict_count,
        "selected_targets": int(selected_neurons.numel()),
        "fallback_used": fallback_used,
        "selected_residual_fraction": {
            "mean": summary["selected_residual_fraction_mean"],
            "min": summary["selected_residual_fraction_min"],
            "max": summary["selected_residual_fraction_max"],
        },
        "cluster_counts": cluster_counts,
        "semantic_cluster_mlp_balanced_accuracy": semantic_mlp_audit["balanced_accuracy"],
        "semantic_cluster_mlp_permutation_p": semantic_mlp_test["permutation_p"],
        "chance_balanced_accuracy": 1.0 / args.num_clusters,
        "output_dir": args.output_dir,
    }
    print("[soft-residual] complete")
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(
        "fit_soft_residual_clusters.py is an internal training library. "
        "Use scripts/run_experiment.sh."
    )
