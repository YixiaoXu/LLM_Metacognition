#!/usr/bin/env python
"""Prepare a continuous, semantics-residualized module axis for causal audits.

The fitted axis is continuous. Low/high tails are exported only to orient the
sign of later dose interventions and to keep prototype discovery disjoint from
causal evaluation; they are not interpreted as latent classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from refined_residual_runtime import load_module_refiner
from semantic_residual_controls import (
    crossfit_strongest_semantic_prediction,
)


def safe_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ridge_fit(x: torch.Tensor, y: torch.Tensor, ridge: float) -> Dict[str, torch.Tensor]:
    x = x.float()
    y = y.float().view(-1, 1)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    design = torch.cat(
        [(x - mean) / std, torch.ones(x.size(0), 1)], dim=1
    )
    gram = design.T @ design
    penalty = torch.eye(gram.size(0), dtype=gram.dtype) * float(ridge)
    penalty[-1, -1] = 0.0
    weight = torch.linalg.solve(gram + penalty, design.T @ y)
    return {"mean": mean, "std": std, "weight": weight}


def ridge_predict(state: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    design = torch.cat(
        [
            (x.float() - state["mean"]) / state["std"],
            torch.ones(x.size(0), 1),
        ],
        dim=1,
    )
    return (design @ state["weight"]).flatten()


def regression_metrics(target: torch.Tensor, prediction: torch.Tensor) -> Dict[str, float]:
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


def refiner_codes(
    refiner: torch.nn.Module,
    meta: torch.Tensor,
    base_delta: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, meta.size(0), batch_size):
            stop = min(start + batch_size, meta.size(0))
            code, _ = refiner(
                meta[start:stop].to(device),
                base_delta[start:stop].to(device),
            )
            chunks.append(code.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def role_split(
    ids: List[str],
    seed: int,
    prototype_fraction: float,
    causal_fraction: float,
    base_id_step_pattern: str = "",
    association_fraction: float | None = None,
    trajectory_fraction: float | None = None,
) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for sample_id in ids:
        base_id = (
            re.sub(base_id_step_pattern, "", str(sample_id))
            if base_id_step_pattern
            else str(sample_id)
        )
        groups.setdefault(base_id, []).append(str(sample_id))
    group_ids = sorted(groups)
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(group_ids), generator=generator).tolist()
    prototype_n = int(round(len(group_ids) * prototype_fraction))
    causal_n = int(round(len(group_ids) * causal_fraction))
    explicit_confirmatory = (
        association_fraction is not None or trajectory_fraction is not None
    )
    if explicit_confirmatory and (
        association_fraction is None or trajectory_fraction is None
    ):
        raise ValueError(
            "Set both association_fraction and trajectory_fraction, or neither."
        )
    minimum_groups = 5
    if len(group_ids) < minimum_groups:
        raise ValueError(
            f"Need at least {minimum_groups} prompt groups for disjoint roles."
        )
    prototype_n = min(max(prototype_n, 1), len(group_ids) - 4)
    causal_n = min(max(causal_n, 2), len(group_ids) - prototype_n - 2)
    prototype = [group_ids[index] for index in order[:prototype_n]]
    causal = [group_ids[index] for index in order[prototype_n : prototype_n + causal_n]]
    confirmatory = [group_ids[index] for index in order[prototype_n + causal_n :]]
    if len(causal) < 2 or len(confirmatory) < 2:
        raise ValueError("Continuous-axis evaluation split is too small for disjoint screens.")
    causal_mid = len(causal) // 2
    causal_a, causal_b = causal[:causal_mid], causal[causal_mid:]
    if explicit_confirmatory:
        association_n = int(round(len(group_ids) * float(association_fraction)))
        association_n = min(max(association_n, 1), len(confirmatory) - 1)
        association = confirmatory[:association_n]
        trajectory = confirmatory[association_n:]
    else:
        confirmatory_mid = len(confirmatory) // 2
        association = confirmatory[:confirmatory_mid]
        trajectory = confirmatory[confirmatory_mid:]

    def expand(base_ids: List[str]) -> List[str]:
        return [sample_id for base_id in base_ids for sample_id in groups[base_id]]

    return {
        "prototype_discovery": expand(prototype),
        "causal_screen_a": expand(causal_a),
        "causal_screen_b": expand(causal_b),
        "association_confirmatory": expand(association),
        "trajectory_confirmatory": expand(trajectory),
        # Backward-compatible aliases. New confirmatory analyses should use the
        # disjoint leaf roles above.
        "causal_test": expand(causal),
        "style_confirmatory": expand(confirmatory),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a continuous semantics-residualized module axis."
    )
    parser.add_argument("--module-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--prototype-fraction", type=float, default=0.25)
    parser.add_argument("--causal-fraction", type=float, default=0.50)
    parser.add_argument("--association-fraction", type=float, default=None)
    parser.add_argument("--trajectory-fraction", type=float, default=None)
    parser.add_argument("--semantic-ridge", type=float, default=0.01)
    parser.add_argument(
        "--semantic-teacher",
        choices=["ridge", "strongest"],
        default="ridge",
        help=(
            "ridge preserves the legacy axis. strongest cross-fits ridge and "
            "MLP teachers, then removes the family with the highest "
            "selection-split R2."
        ),
    )
    parser.add_argument("--semantic-folds", type=int, default=5)
    parser.add_argument("--semantic-mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--semantic-mlp-epochs", type=int, default=16)
    parser.add_argument("--semantic-mlp-repeats", type=int, default=2)
    parser.add_argument("--semantic-mlp-dropout", type=float, default=0.10)
    parser.add_argument("--semantic-mlp-lr", type=float, default=1e-3)
    parser.add_argument("--semantic-mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--base-id-step-pattern",
        default="",
        help=(
            "Optional regex stripped from activation ids before role splitting. "
            "Use ::step\\d+$ for generated-step caches so one prompt never spans roles."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_threads = max(
        1,
        int(
            os.environ.get(
                "CONTINUOUS_AXIS_CPU_THREADS",
                os.environ.get("OMP_NUM_THREADS", "2"),
            )
        ),
    )
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits setting this before inter-op work starts.
        pass
    print(
        f"[continuous-axis] cpu_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()}",
        flush=True,
    )
    if not 0.05 <= args.tail_fraction < 0.5:
        raise ValueError("--tail-fraction must be in [0.05, 0.5).")
    if args.prototype_fraction <= 0 or args.causal_fraction <= 0:
        raise ValueError("Role fractions must be positive.")
    explicit_confirmatory = (
        args.association_fraction is not None
        or args.trajectory_fraction is not None
    )
    if explicit_confirmatory:
        if args.association_fraction is None or args.trajectory_fraction is None:
            raise ValueError(
                "Set both --association-fraction and --trajectory-fraction."
            )
        fractions = (
            args.prototype_fraction,
            args.causal_fraction,
            args.association_fraction,
            args.trajectory_fraction,
        )
        if any(value <= 0 for value in fractions):
            raise ValueError("All explicit role fractions must be positive.")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(
                "Explicit role fractions must sum to 1.0; got "
                f"{sum(fractions):.8f}."
            )
    elif args.prototype_fraction + args.causal_fraction >= 1:
        raise ValueError("Leave a positive fraction for confirmation roles.")

    module_dir = Path(args.module_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = module_dir / "aligned_probe_cache.pt"
    refiner_path = module_dir / "module_refiner.pt"
    summary_path = module_dir / "module_summary.json"
    for path in (cache_path, refiner_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    cache = safe_load(str(cache_path))
    missing = [
        split
        for split in ("train", "selection", "evaluation")
        if "semantic_controls" not in cache.get(split, {})
    ]
    if missing:
        raise ValueError(
            "aligned_probe_cache.pt predates continuous semantic-control export "
            f"for splits {missing}; rerun joint-v2 module export."
        )
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    refiner, refiner_info = load_module_refiner(str(refiner_path), device)
    train = cache["train"]
    selection = cache["selection"]
    evaluation = cache["evaluation"]
    train_code = refiner_codes(
        refiner,
        train["meta"],
        train["base_delta"],
        args.batch_size,
        device,
    )
    eval_code = refiner_codes(
        refiner,
        evaluation["meta"],
        evaluation["base_delta"],
        args.batch_size,
        device,
    )
    selection_code = refiner_codes(
        refiner,
        selection["meta"],
        selection["base_delta"],
        args.batch_size,
        device,
    )
    if train_code.size(1) != 1:
        centered = train_code - train_code.mean(dim=0, keepdim=True)
        _, _, components = torch.linalg.svd(centered, full_matrices=False)
        axis = components[0]
    else:
        axis = torch.ones(1)
    train_scalar = train_code @ axis
    selection_scalar = selection_code @ axis
    eval_scalar = eval_code @ axis
    if args.semantic_teacher == "strongest":
        all_controls = torch.cat(
            [
                train["semantic_controls"].float(),
                selection["semantic_controls"].float(),
                evaluation["semantic_controls"].float(),
            ],
            dim=0,
        )
        all_target = torch.cat(
            [train_scalar, selection_scalar, eval_scalar], dim=0
        )
        train_idx = torch.arange(train_scalar.numel())
        selection_idx = torch.arange(
            train_scalar.numel(),
            train_scalar.numel() + selection_scalar.numel(),
        )
        evaluation_idx = torch.arange(
            train_scalar.numel() + selection_scalar.numel(),
            all_target.numel(),
        )
        (
            all_semantic,
            semantic_teacher_diagnostics,
            teacher,
        ) = crossfit_strongest_semantic_prediction(
            all_controls,
            all_target,
            train_idx,
            selection_idx,
            evaluation_idx,
            families=("ridge", "mlp"),
            folds=args.semantic_folds,
            ridge=args.semantic_ridge,
            mlp_hidden_dim=args.semantic_mlp_hidden_dim,
            mlp_dropout=args.semantic_mlp_dropout,
            mlp_epochs=args.semantic_mlp_epochs,
            mlp_batch_size=args.batch_size,
            mlp_lr=args.semantic_mlp_lr,
            mlp_weight_decay=args.semantic_mlp_weight_decay,
            mlp_repeats=args.semantic_mlp_repeats,
            device=device,
            seed=args.seed + 301,
            progress_desc="Continuous-axis semantic teachers",
        )
        train_semantic = all_semantic.index_select(0, train_idx)
        selection_semantic = all_semantic.index_select(0, selection_idx)
        eval_semantic = all_semantic.index_select(0, evaluation_idx)
    else:
        teacher = ridge_fit(
            train["semantic_controls"], train_scalar, args.semantic_ridge
        )
        train_semantic = ridge_predict(
            teacher, train["semantic_controls"]
        )
        selection_semantic = ridge_predict(
            teacher, selection["semantic_controls"]
        )
        eval_semantic = ridge_predict(
            teacher, evaluation["semantic_controls"]
        )
        semantic_teacher_diagnostics = {
            "mode": "ridge",
            "selected_family": "ridge",
            "train_oof": regression_metrics(train_scalar, train_semantic),
            "selection": regression_metrics(
                selection_scalar, selection_semantic
            ),
            "evaluation": regression_metrics(eval_scalar, eval_semantic),
            "max_probe_r2": {
                "train_oof": regression_metrics(
                    train_scalar, train_semantic
                )["r2"],
                "selection": regression_metrics(
                    selection_scalar, selection_semantic
                )["r2"],
                "evaluation": regression_metrics(
                    eval_scalar, eval_semantic
                )["r2"],
            },
        }
    train_residual = train_scalar - train_semantic
    eval_residual = eval_scalar - eval_semantic
    residual_mean = train_residual.mean()
    residual_std = train_residual.std(unbiased=False).clamp_min(1e-6)
    train_score = (train_residual - residual_mean) / residual_std
    eval_score = (eval_residual - residual_mean) / residual_std
    low_threshold = float(torch.quantile(train_score, args.tail_fraction).item())
    high_threshold = float(
        torch.quantile(train_score, 1.0 - args.tail_fraction).item()
    )
    keep = (eval_score <= low_threshold) | (eval_score >= high_threshold)
    labels = (eval_score[keep] >= high_threshold).long()
    kept_ids = [
        sample_id
        for sample_id, selected in zip(evaluation["ids"], keep.tolist())
        if selected
    ]
    kept_code = eval_code[keep]
    kept_score = eval_score[keep]
    kept_raw_scalar = eval_scalar[keep]
    kept_semantic = eval_semantic[keep]
    all_roles = role_split(
        list(evaluation["ids"]),
        args.seed,
        args.prototype_fraction,
        args.causal_fraction,
        args.base_id_step_pattern,
        args.association_fraction,
        args.trajectory_fraction,
    )
    tail_id_set = set(kept_ids)
    roles = {
        key: [value for value in values if value in tail_id_set]
        for key, values in all_roles.items()
    }
    # Baseline conditional-association analysis uses the full continuous range.
    roles["style_confirmatory"] = list(all_roles["style_confirmatory"])
    roles["association_confirmatory"] = list(
        all_roles["association_confirmatory"]
    )
    leaf_roles = (
        "prototype_discovery",
        "causal_screen_a",
        "causal_screen_b",
        "association_confirmatory",
        "trajectory_confirmatory",
    )
    role_by_id = {
        sample_id: role
        for role in leaf_roles
        for sample_id in roles[role]
    }
    all_role_by_id = {
        sample_id: role
        for role in leaf_roles
        for sample_id in all_roles[role]
    }

    rows = []
    for index, sample_id in enumerate(kept_ids):
        rows.append(
            {
                "id": sample_id,
                "activation_cluster": int(labels[index].item()),
                "cluster": int(labels[index].item()),
                "continuous_meta_score": float(kept_score[index].item()),
                "raw_module_code": float(kept_raw_scalar[index].item()),
                "semantic_predicted_code": float(kept_semantic[index].item()),
                "analysis_role": role_by_id.get(sample_id, "tail_not_used"),
            }
        )
    assignment_path = output_dir / "continuous_axis_assignments.csv"
    feature_path = output_dir / "continuous_axis_features.pt"
    all_assignment_path = output_dir / "continuous_axis_all_assignments.csv"
    all_feature_path = output_dir / "continuous_axis_all_features.pt"
    artifact_path = output_dir / "continuous_axis.pt"
    write_csv(assignment_path, rows)
    torch.save(
        {
            "ids": kept_ids,
            "model_features": kept_code.float(),
            "raw_residual_features": kept_score.view(-1, 1).float(),
            "labels": labels,
            "selected_neurons": torch.as_tensor(cache["selected_neurons"]).long(),
            "module_loading": torch.as_tensor(cache["module_loading"]).float(),
            "summary": {
                "source": "continuous_semantics_residualized_module_axis",
                "tail_labels_are_orientation_only": True,
            },
        },
        feature_path,
    )
    all_labels = (eval_score >= 0.0).long()
    all_rows = [
        {
            "id": sample_id,
            "activation_cluster": int(all_labels[index].item()),
            "cluster": int(all_labels[index].item()),
            "continuous_meta_score": float(eval_score[index].item()),
            "raw_module_code": float(eval_scalar[index].item()),
            "semantic_predicted_code": float(eval_semantic[index].item()),
            "analysis_role": all_role_by_id.get(sample_id, "unknown"),
        }
        for index, sample_id in enumerate(evaluation["ids"])
    ]
    write_csv(all_assignment_path, all_rows)
    torch.save(
        {
            "ids": list(evaluation["ids"]),
            "model_features": eval_code.float(),
            "raw_residual_features": eval_score.view(-1, 1).float(),
            "labels": all_labels,
            "selected_neurons": torch.as_tensor(cache["selected_neurons"]).long(),
            "module_loading": torch.as_tensor(cache["module_loading"]).float(),
            "summary": {
                "source": "continuous_semantics_residualized_module_axis_full",
                "median_labels_are_visualization_only": True,
            },
        },
        all_feature_path,
    )
    torch.save(
        {
            "schema_version": 2,
            "module_dir": str(module_dir),
            "axis": axis.float(),
            "raw_residual_std": residual_std.float(),
            "raw_residual_mean": residual_mean.float(),
            "low_threshold_standardized": low_threshold,
            "high_threshold_standardized": high_threshold,
            "semantic_teacher": teacher,
            "semantic_teacher_diagnostics": semantic_teacher_diagnostics,
            "ids": list(evaluation["ids"]),
            "scores": eval_score.float(),
            "raw_codes": eval_code.float(),
            "semantic_controls": evaluation["semantic_controls"].float(),
            "labels": all_labels,
            "tail_ids": kept_ids,
            "tail_scores": kept_score.float(),
            "tail_labels": labels,
            "role_ids": roles,
            "all_role_ids": all_roles,
            "tail_labels_are_orientation_only": True,
        },
        artifact_path,
    )
    semantic_metrics = regression_metrics(eval_scalar, eval_semantic)
    module_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = torch.bincount(labels, minlength=2)
    summary = {
        "n_evaluation": len(evaluation["ids"]),
        "n_tail": len(kept_ids),
        "tail_fraction_each_side": args.tail_fraction,
        "tail_counts": counts.tolist(),
        "low_threshold_standardized": low_threshold,
        "high_threshold_standardized": high_threshold,
        "semantic_predictability_of_raw_code": semantic_metrics,
        "semantic_teacher": {
            "requested": args.semantic_teacher,
            "selected_family": semantic_teacher_diagnostics.get(
                "selected_family", "ridge"
            ),
            "diagnostics": semantic_teacher_diagnostics,
        },
        "heldout_residual_fraction": module_summary.get(
            "heldout_residual_fraction_mean"
        ),
        "heldout_residual_gain": (
            module_summary.get("information", {}).get("residual_gain_mse")
            or module_summary.get("information", {}).get(
                "residual_gain_sum_nats"
            )
        ),
        "discrete_state_interpretation": module_summary.get(
            "cluster_interpretation"
        ),
        "roles": {key: len(value) for key, value in roles.items()},
        "all_role_counts": {key: len(value) for key, value in all_roles.items()},
        "role_fractions": {
            "prototype": args.prototype_fraction,
            "causal_total": args.causal_fraction,
            "association": args.association_fraction,
            "trajectory": args.trajectory_fraction,
        },
        "scientific_contract": {
            "primary_variable": "continuous_meta_score",
            "tail_labels": (
                "orientation and sample-allocation device only; they do not imply "
                "two latent metacognitive states"
            ),
            "semantic_control": (
                "cross-fitted strongest-of-ridge-and-MLP prediction from the "
                "same compact z1+previous-layer controls exported by joint-v2"
                if args.semantic_teacher == "strongest"
                else "ridge prediction from the same compact "
                "z1+previous-layer controls exported by joint-v2"
            ),
        },
        "artifacts": {
            "assignments": str(assignment_path),
            "features": str(feature_path),
            "all_assignments": str(all_assignment_path),
            "all_features": str(all_feature_path),
            "axis": str(artifact_path),
        },
    }
    write_json(output_dir / "continuous_axis_summary.json", summary)

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            figure, axis_plot = plt.subplots(figsize=(6.4, 3.8))
            axis_plot.hist(
                train_score.numpy(),
                bins=60,
                density=True,
                alpha=0.35,
                color="#0072B2",
                label="Train",
            )
            axis_plot.hist(
                eval_score.numpy(),
                bins=60,
                density=True,
                histtype="step",
                linewidth=1.5,
                color="#D55E00",
                label="Held-out",
            )
            axis_plot.axvline(low_threshold, color="#333333", linestyle=":")
            axis_plot.axvline(high_threshold, color="#333333", linestyle=":")
            axis_plot.set_xlabel("Semantics-residualized module score (train SD)")
            axis_plot.set_ylabel("Density")
            axis_plot.set_title("Continuous meta axis; tails orient intervention dose")
            axis_plot.spines[["top", "right"]].set_visible(False)
            axis_plot.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(output_dir / "continuous_axis_distribution.png", dpi=220)
            plt.close(figure)
        except ImportError:
            pass
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
