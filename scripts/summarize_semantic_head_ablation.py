#!/usr/bin/env python
"""Summarize the external-semantic head capacity ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import itertools
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite(values: Iterable[Any]) -> List[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_condition(condition: Dict[str, str]) -> Dict[str, Any]:
    root = Path(condition["output_dir"])
    analyses = sorted(root.glob("**/decoupler_joint_v2/analysis_summary.json"))
    if len(analyses) != 1:
        return {
            **condition,
            "status": "missing" if not analyses else "ambiguous",
            "analysis_count": len(analyses),
        }
    analysis_path = analyses[0]
    analysis = read_json(analysis_path)
    main = analysis.get("main", {})
    main_metrics = main.get("best_metrics", {})
    semantic = main.get("semantic_probe", {})
    module_paths = sorted(
        analysis_path.parent.glob(
            "joint_residual_modules/module_*/module_summary.json"
        )
    )
    modules = [read_json(path) for path in module_paths]
    heldout_rf = finite(
        module.get("heldout_residual_fraction_mean") for module in modules
    )
    residual_gain = finite(
        module.get("information", {}).get("residual_gain_mse")
        for module in modules
    )
    semantic_r2 = finite(
        module.get("continuous_semantic_predictability", {})
        .get("evaluation", {})
        .get("r2")
        for module in modules
    )
    significant_gain = sum(
        bool(
            module.get("information", {}).get(
                "residual_gain_significant_positive_0p05", False
            )
        )
        for module in modules
    )
    weights = semantic.get("ensemble_weights")
    return {
        **condition,
        "status": "complete",
        "analysis_path": str(analysis_path),
        "semantic_validation_mse": semantic.get("best_validation_mse"),
        "semantic_heldout_mse": semantic.get("split_metrics", {})
        .get("heldout", {})
        .get("mse"),
        "semantic_head_parameters": semantic.get("trainable_parameters"),
        "ensemble_weight_linear": weights[0] if weights else None,
        "ensemble_weight_mlp": weights[1] if weights else None,
        "ensemble_weight_deep": weights[2] if weights else None,
        "main_semantic_gain": main_metrics.get("semantic_gain"),
        "main_residual_gain": main_metrics.get("residual_gain"),
        "main_total_gain": main_metrics.get("total_gain"),
        "main_residual_fraction": main_metrics.get("residual_fraction"),
        "main_positive_modules": main_metrics.get("positive_modules"),
        "exported_modules": len(modules),
        "heldout_rf_mean": mean(heldout_rf) if heldout_rf else None,
        "heldout_rf_median": median(heldout_rf) if heldout_rf else None,
        "heldout_residual_gain_mean": (
            mean(residual_gain) if residual_gain else None
        ),
        "heldout_semantic_r2_mean": mean(semantic_r2) if semantic_r2 else None,
        "heldout_positive_gain_modules": significant_gain,
    }


def make_plot(rows: List[Dict[str, Any]], output: Path) -> None:
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        return
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    labels = [row["condition"].replace("_", "\n") for row in complete]
    colors = ["#484878", "#7884B4", "#E4CCD8", "#F0C0CC"][: len(complete)]
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    panels = (
        ("semantic_heldout_mse", "Held-out semantic MSE", "Lower is stronger"),
        ("heldout_rf_mean", "Held-out residual fraction", "Module mean"),
        ("heldout_residual_gain_mean", "Held-out residual gain", "Module mean"),
    )
    for panel_label, axis, (field, ylabel, subtitle) in zip("abc", axes, panels):
        values = [
            float(row[field])
            if row.get(field) is not None
            else math.nan
            for row in complete
        ]
        axis.bar(range(len(values)), values, color=colors, width=0.72)
        axis.set_xticks(range(len(values)), labels, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.set_title(subtitle, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
        axis.set_axisbelow(True)
        axis.text(
            -0.16,
            1.06,
            panel_label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
    figure.suptitle("External semantic-head capacity ablation", fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "semantic_head_ablation.png", dpi=300)
    figure.savefig(output / "semantic_head_ablation.svg", bbox_inches="tight")
    figure.savefig(output / "semantic_head_ablation.pdf", bbox_inches="tight")
    figure.savefig(
        output / "semantic_head_ablation.tiff", dpi=600, bbox_inches="tight"
    )
    plt.close(figure)


def load_heldout_losses(output_dir: str) -> Dict[int, float]:
    paths = sorted(Path(output_dir).glob("**/external_semantic_probe_losses.csv"))
    if len(paths) != 1:
        return {}
    with paths[0].open(encoding="utf-8", newline="") as handle:
        return {
            int(row["row_index"]): float(row["external_semantic_mse"])
            for row in csv.DictReader(handle)
            if row.get("split") == "heldout"
        }


def paired_head_contrast(
    conditions: List[Dict[str, str]],
    bootstrap_samples: int,
    permutation_tests: int,
) -> Dict[str, Any]:
    import numpy as np

    by_name = {condition["condition"]: condition for condition in conditions}
    if "mlp" not in by_name or "ensemble" not in by_name:
        return {"status": "missing_condition"}
    baseline = load_heldout_losses(by_name["mlp"]["output_dir"])
    ensemble = load_heldout_losses(by_name["ensemble"]["output_dir"])
    common = sorted(set(baseline).intersection(ensemble))
    if not common:
        return {"status": "missing_paired_losses"}
    delta = np.asarray(
        [baseline[index] - ensemble[index] for index in common], dtype=np.float64
    )
    observed = float(delta.mean())
    rng = np.random.default_rng(42)
    boot = np.empty(max(1, int(bootstrap_samples)), dtype=np.float64)
    chunk = 100
    for start in range(0, boot.size, chunk):
        stop = min(start + chunk, boot.size)
        sampled = rng.integers(0, delta.size, size=(stop - start, delta.size))
        boot[start:stop] = delta[sampled].mean(axis=1)
    if delta.size <= 20:
        null = np.asarray(
            [
                np.mean(delta * np.asarray(signs, dtype=np.float64))
                for signs in itertools.product((-1.0, 1.0), repeat=delta.size)
            ]
        )
        p_value = float((null >= observed).mean())
        method = "exhaustive_signflip"
        draws = int(null.size)
    else:
        extreme = 0
        draws = max(1, int(permutation_tests))
        chunk = 256
        for start in range(0, draws, chunk):
            count = min(chunk, draws - start)
            signs = rng.choice((-1.0, 1.0), size=(count, delta.size))
            extreme += int(np.count_nonzero((signs * delta).mean(axis=1) >= observed))
        p_value = float((extreme + 1) / (draws + 1))
        method = "monte_carlo_signflip"
    return {
        "status": "complete",
        "contrast": "ensemble_minus_mlp_head_strength",
        "estimand": "mlp per-sample MSE minus ensemble per-sample MSE",
        "positive_means_ensemble_is_stronger": True,
        "n_paired_heldout_prompts": int(delta.size),
        "mean_mse_reduction": observed,
        "bootstrap_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "signflip_one_sided_p": p_value,
        "signflip_method": method,
        "signflip_draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutation-tests", type=int, default=20000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.conditions).open(encoding="utf-8", newline="") as handle:
        conditions = list(csv.DictReader(handle, delimiter="\t"))
    rows = [summarize_condition(condition) for condition in conditions]
    paired_contrast = paired_head_contrast(
        conditions, args.bootstrap_samples, args.permutation_tests
    )
    write_csv(output / "semantic_head_ablation_summary.csv", rows)
    with (output / "semantic_head_ablation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "conditions": rows,
                "paired_manipulation_check": paired_contrast,
                "primary_contrast": "ensemble_vs_mlp",
                "manipulation_check": "held-out external semantic activation MSE",
                "primary_robustness_readout": "held-out residual gain and residual fraction",
                "interpretation": (
                    "A stronger head must first lower held-out semantic MSE. "
                    "Residual evidence that survives this manipulation is less "
                    "likely to be an artifact of semantic-head underfitting."
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
    make_plot(rows, output)
    with (output / "ensemble_vs_mlp_paired_test.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(paired_contrast, handle, ensure_ascii=False, indent=2)
    for row in rows:
        print(
            "[semantic-head-summary] "
            f"{row['condition']} status={row['status']} "
            f"semantic_mse={row.get('semantic_validation_mse')} "
            f"heldout_rf={row.get('heldout_rf_mean')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
