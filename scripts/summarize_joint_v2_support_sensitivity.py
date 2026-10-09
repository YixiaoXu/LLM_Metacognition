#!/usr/bin/env python3
"""Summarize support size using continuous held-out evidence only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def aggregate(values: Iterable[float]) -> Dict[str, float]:
    ordered = sorted(finite(values))
    if not ordered:
        return {"mean": math.nan, "median": math.nan, "q25": math.nan, "q75": math.nan}

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "q25": quantile(0.25),
        "q75": quantile(0.75),
    }


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


def module_row(path: Path) -> Dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    module_dir = path.parent
    decoupler_dir = module_dir.parents[1]
    config = json.loads((decoupler_dir / "config.json").read_text(encoding="utf-8"))
    selection = summary.get("selection_metrics", {})
    information = summary.get("information", {})
    continuous = summary.get("continuous_semantic_predictability", {})
    return {
        "profile": decoupler_dir.parent.name,
        "support": int(config.get("module_support", 0)),
        "module": int(summary.get("module", -1)),
        "selection_rf": number(selection.get("residual_fraction")),
        "selection_residual_gain": number(selection.get("residual_gain_mse")),
        "heldout_rf": number(summary.get("heldout_residual_fraction_mean")),
        "heldout_residual_gain": number(information.get("residual_gain_mse")),
        "heldout_residual_gain_ci_low": number(
            information.get("residual_gain_bootstrap_ci95_low")
        ),
        "continuous_semantic_r2_train_oof": number(
            (continuous.get("train_oof") or {}).get("r2")
        ),
        "continuous_semantic_r2_selection": number(
            (continuous.get("selection") or {}).get("r2")
        ),
        "continuous_semantic_r2_heldout": number(
            (continuous.get("evaluation") or {}).get("r2")
        ),
        "module_dir": str(module_dir),
    }


def save_plot(output_dir: Path, rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = [
        ("heldout_rf", "Held-out residual fraction", "#0072B2"),
        ("heldout_residual_gain", "Held-out residual gain", "#009E73"),
        ("continuous_semantic_r2_heldout", "Continuous semantic R2", "#D55E00"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    supports = sorted({int(row["support"]) for row in rows})
    for axis, (metric, label, color) in zip(axes, metrics):
        for support in supports:
            values = [
                number(row[metric]) for row in rows
                if int(row["support"]) == support and math.isfinite(number(row[metric]))
            ]
            axis.scatter([support] * len(values), values, color=color, alpha=0.28, s=22)
        axis.plot(
            [row["support"] for row in summaries],
            [row[f"{metric}_median"] for row in summaries],
            color=color, marker="o", linewidth=1.8,
        )
        axis.set(xlabel="Neurons per module", ylabel=label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "support_continuous_evidence.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.run_root).glob(
        "jobs/*/decoupler_joint_v2/joint_residual_modules/module_*/module_summary.json"
    ))
    rows = [module_row(path) for path in paths]
    if not rows:
        raise SystemExit("No completed continuous module artifacts were found.")
    summaries: List[Dict[str, Any]] = []
    for support in sorted({int(row["support"]) for row in rows}):
        group = [row for row in rows if int(row["support"]) == support]
        output: Dict[str, Any] = {"support": support, "module_count": len(group)}
        for metric in (
            "selection_rf", "heldout_rf", "selection_residual_gain",
            "heldout_residual_gain", "continuous_semantic_r2_heldout",
        ):
            for statistic, value in aggregate(number(row[metric]) for row in group).items():
                output[f"{metric}_{statistic}"] = value
        summaries.append(output)
        print(
            f"[support-summary] k={support} modules={len(group)} "
            f"RF={output['heldout_rf_median']:.4f} "
            f"gain={output['heldout_residual_gain_median']:.4f} "
            f"semR2={output['continuous_semantic_r2_heldout_median']:.4f}",
            flush=True,
        )
    write_csv(output_dir / "module_support_diagnostics.csv", rows)
    write_csv(output_dir / "support_level_summary.csv", summaries)
    save_plot(output_dir, rows, summaries)
    payload = {
        "run_root": args.run_root,
        "module_count": len(rows),
        "selection_contract": "held-out RF/gain with continuous semantic R2 safeguard",
        "outputs": {
            "modules": "module_support_diagnostics.csv",
            "supports": "support_level_summary.csv",
            "plot": "support_continuous_evidence.png",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
