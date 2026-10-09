#!/usr/bin/env python3
"""Aggregate the independent support-size sweep for supplementary figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    condition_dirs = sorted(args.run_root.glob("support_*/"))
    for condition_dir in condition_dirs:
        try:
            support = int(condition_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        pair_dirs = list(condition_dir.glob("llama32_3b__semantic_qwen3_0p6b_l14/"))
        if not pair_dirs:
            continue
        pair = pair_dirs[0]

        selected_path = pair / "adaptive_selection" / "selected_adaptive_modules.json"
        if selected_path.exists():
            payload = json.loads(selected_path.read_text(encoding="utf-8"))
            for item in payload.get("selected", []):
                records.append({
                    "support": support,
                    "module": str(item.get("module", "")),
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
                info = [row for row in read_csv(info_path) if row.get("meta_predictor") == "signed_direction"]
                for metric in ("generated_tokens", "line_count", "bullet_line_count", "style_hedging_score", "certainty_score"):
                    matches = [row for row in info if row.get("metric") == metric]
                    if matches:
                        row = matches[0]
                        records.append({
                            "support": support,
                            "module": rank,
                            "behavior_metric": metric,
                            "conditional_meta_bits": number(row.get("conditional_meta_information_bits_per_sample")),
                            "shapley_meta_bits": number(row.get("shapley_meta_information_bits_per_sample")),
                            "shapley_meta_share": number(row.get("shapley_meta_share_of_joint_information")),
                            "behavior_p": number(row.get("conditional_information_signflip_one_sided")),
                        })

            dose_path = axis_dir / "causal_screen_a" / "continuous_dose_response.csv"
            if dose_path.exists():
                for row in read_csv(dose_path):
                    if row.get("metric") in {
                        "next_token_full_logit_delta_l2",
                        "next_token_js_divergence",
                        "next_token_total_variation",
                        "next_token_auto_logit_delta_projection_to_target",
                    }:
                        records.append({
                            "support": support,
                            "module": rank,
                            "intervention_metric": row.get("metric", ""),
                            "intervention_mean": number(row.get("mean")),
                            "intervention_p": number(row.get("signflip_two_sided_p")),
                        })

    if not records:
        raise SystemExit("No completed support artifacts found")

    fields = sorted({key for record in records for key in record})
    summary_path = args.output_dir / "support_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"[support-supplement] wrote {summary_path}; matplotlib unavailable")
        return

    supports = sorted({int(record["support"]) for record in records})
    decoupling_metrics = ["heldout_rf", "heldout_residual_gain", "continuous_semantic_r2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric in zip(axes, decoupling_metrics):
        values = [mean([record.get(metric, math.nan) for record in records
                        if record.get("support") == support]) for support in supports]
        ax.plot(supports, values, marker="o", linewidth=2)
        ax.set_xlabel("support size")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Support-size sweep: decoupling and semantic leakage")
    fig.savefig(args.output_dir / "figS_support_decoupling.png", dpi=220)
    fig.savefig(args.output_dir / "figS_support_decoupling.svg")
    plt.close(fig)

    intervention_metrics = [
        "next_token_full_logit_delta_l2",
        "next_token_js_divergence",
        "next_token_total_variation",
        "next_token_auto_logit_delta_projection_to_target",
    ]
    matrix = np.asarray([
        [mean([record.get("intervention_mean", math.nan) for record in records
               if record.get("support") == support and record.get("intervention_metric") == metric])
         for metric in intervention_metrics]
        for support in supports
    ], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(supports)), [str(value) for value in supports])
    ax.set_ylabel("support size")
    ax.set_xticks(range(len(intervention_metrics)),
                  [metric.replace("next_token_", "").replace("_", " ")
                   for metric in intervention_metrics], rotation=30, ha="right")
    ax.set_title("Support-size sweep: next-token intervention response")
    fig.colorbar(im, ax=ax, label="mean dose response")
    fig.savefig(args.output_dir / "figS_support_intervention.png", dpi=220)
    fig.savefig(args.output_dir / "figS_support_intervention.svg")
    plt.close(fig)
    print(f"[support-supplement] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
