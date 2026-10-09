#!/usr/bin/env python3
"""Aggregate fixed layer-topology runs into supplementary tables and figures."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values):
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for topology_dir in sorted(p for p in args.run_root.iterdir() if p.is_dir()):
        if topology_dir.name.startswith("support_"):
            continue
        pair_dirs = list(topology_dir.glob("llama32_3b__semantic_qwen3_0p6b_l*/"))
        if not pair_dirs:
            continue
        pair = pair_dirs[0]
        modules = []
        selected = pair / "adaptive_selection" / "selected_adaptive_modules.json"
        if selected.exists():
            payload = json.loads(selected.read_text(encoding="utf-8"))
            modules = payload.get("selected", [])
        topology = topology_dir.name
        for item in modules:
            records.append({
                "topology": topology,
                "module": str(item.get("module", "")),
                "support": item.get("support", ""),
                "heldout_rf": number(item.get("heldout_rf")),
                "heldout_residual_gain": number(item.get("heldout_residual_gain")),
                "heldout_gain_ci_low": number(item.get("heldout_residual_gain_ci_low")),
                "continuous_semantic_r2": number(item.get("continuous_semantic_r2_heldout")),
                "adaptive_score": number(item.get("adaptive_score")),
            })

        for axis_dir in sorted(pair.glob("continuous_modules/rank_*/")):
            rank = axis_dir.name
            info_path = axis_dir / "continuous_behavior" / "continuous_behavior_information_decomposition.csv"
            if info_path.exists():
                info = [r for r in rows(info_path) if r.get("meta_predictor") == "signed_direction"]
                for metric in ("generated_tokens", "line_count", "bullet_line_count", "style_hedging_score", "certainty_score"):
                    match = [r for r in info if r.get("metric") == metric]
                    if match:
                        r = match[0]
                        records.append({
                            "topology": topology,
                            "module": rank,
                            "support": "",
                            "heldout_rf": math.nan,
                            "heldout_residual_gain": math.nan,
                            "heldout_gain_ci_low": math.nan,
                            "continuous_semantic_r2": math.nan,
                            "adaptive_score": math.nan,
                            "behavior_metric": metric,
                            "conditional_meta_bits": number(r.get("conditional_meta_information_bits_per_sample")),
                            "shapley_meta_bits": number(r.get("shapley_meta_information_bits_per_sample")),
                            "shapley_meta_share": number(r.get("shapley_meta_share_of_joint_information")),
                            "behavior_p": number(r.get("conditional_information_signflip_one_sided")),
                        })

            dose_path = axis_dir / "causal_screen_a" / "continuous_dose_response.csv"
            if dose_path.exists():
                for r in rows(dose_path):
                    if r.get("metric") in {
                        "next_token_full_logit_delta_l2",
                        "next_token_js_divergence",
                        "next_token_total_variation",
                        "next_token_auto_logit_delta_projection_to_target",
                    }:
                        records.append({
                            "topology": topology,
                            "module": rank,
                            "support": "",
                            "heldout_rf": math.nan,
                            "heldout_residual_gain": math.nan,
                            "heldout_gain_ci_low": math.nan,
                            "continuous_semantic_r2": math.nan,
                            "adaptive_score": math.nan,
                            "intervention_metric": r.get("metric", ""),
                            "intervention_mean": number(r.get("mean")),
                            "intervention_p": number(r.get("signflip_two_sided_p")),
                            "intervention_q_legacy": number(r.get("bh_fdr_q")),
                        })

    if not records:
        raise SystemExit("No completed topology artifacts found")

    fields = sorted({key for record in records for key in record})
    with (args.output_dir / "layer_topology_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    # Plotting is intentionally optional at import time so table aggregation
    # still works on headless environments without matplotlib.
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[supplement] matplotlib unavailable; wrote CSV only")
        return

    order = [p.name for p in sorted(args.run_root.iterdir())
             if p.is_dir() and not p.name.startswith("support_")
             and any(p.glob("llama32_3b__semantic_qwen3_0p6b_l*/"))]
    metric_names = ["heldout_rf", "heldout_residual_gain", "continuous_semantic_r2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric in zip(axes, metric_names):
        values = [mean([r.get(metric, math.nan) for r in records if r.get("topology") == topology]) for topology in order]
        ax.plot(range(len(order)), values, marker="o", linewidth=2)
        ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Layer topology: decoupling and semantic leakage")
    fig.savefig(args.output_dir / "figS_layer_topology_decoupling.png", dpi=220)
    fig.savefig(args.output_dir / "figS_layer_topology_decoupling.svg")
    plt.close(fig)

    intervention_metrics = [
        "next_token_full_logit_delta_l2",
        "next_token_js_divergence",
        "next_token_total_variation",
        "next_token_auto_logit_delta_projection_to_target",
    ]
    matrix = []
    for topology in order:
        row = []
        for metric in intervention_metrics:
            row.append(mean([r.get("intervention_mean", math.nan) for r in records
                             if r.get("topology") == topology and r.get("intervention_metric") == metric]))
        matrix.append(row)
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(order)), order)
    ax.set_xticks(range(len(intervention_metrics)), [m.replace("next_token_", "").replace("_", " ") for m in intervention_metrics], rotation=30, ha="right")
    ax.set_title("Layer topology: next-token intervention response")
    fig.colorbar(im, ax=ax, label="mean dose response")
    fig.savefig(args.output_dir / "figS_layer_topology_intervention.png", dpi=220)
    fig.savefig(args.output_dir / "figS_layer_topology_intervention.svg")
    plt.close(fig)
    print(f"[supplement] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
