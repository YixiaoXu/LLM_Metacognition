#!/usr/bin/env python3
"""Collect module-specific proxy-confidence results for one model pair."""
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
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.pair_root.glob(
        "continuous_modules/rank_*/proxy_confidence/proxy_confidence_summary.csv"
    )):
        rank = path.parents[1].name
        for row in read_csv(path):
            row = dict(row)
            row["module_rank"] = rank
            row["source_file"] = str(path.relative_to(args.pair_root))
            rows.append(row)
    if not rows:
        raise SystemExit("No proxy-confidence module summaries found")

    fields = list(dict.fromkeys(key for row in rows for key in row))
    output_csv = args.output_dir / "proxy_confidence_module_summary.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    primary = [
        row for row in rows
        if str(row.get("confirmatory_primary", "")).lower() in {"true", "1"}
    ]
    payload = {
        "module_count": len({row["module_rank"] for row in primary}),
        "primary_proxy": primary[0].get("proxy") if primary else None,
        "primary_results": primary,
        "interpretation": (
            "Each row is module-specific. Positive conditional meta bits mean "
            "the module improves held-out correctness prediction after external "
            "semantics and the preregistered confidence proxy."
        ),
    }
    (args.output_dir / "proxy_confidence_module_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[proxy-confidence-summary] wrote {output_csv}")
        return
    labels = [row["module_rank"].replace("rank_", "M") for row in primary]
    values = [number(row.get("conditional_meta_information_bits_per_sample")) or 0.0 for row in primary]
    lower = [number(row.get("conditional_meta_bootstrap_ci95_low_bits")) for row in primary]
    upper = [number(row.get("conditional_meta_bootstrap_ci95_high_bits")) for row in primary]
    errors = [
        [max(value - (low if low is not None else value), 0.0) for value, low in zip(values, lower)],
        [max((high if high is not None else value) - value, 0.0) for value, high in zip(values, upper)],
    ]
    fig, ax = plt.subplots(figsize=(max(5.2, 0.9 * len(labels)), 4.5), constrained_layout=True)
    ax.bar(labels, values, color="#2878B5", yerr=errors, capsize=3)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_ylabel("Conditional meta information (bits/sample)")
    ax.set_title("Correctness information beyond semantics and confidence")
    fig.savefig(args.output_dir / "proxy_confidence_module_effects.png", dpi=220)
    fig.savefig(args.output_dir / "proxy_confidence_module_effects.svg")
    plt.close(fig)
    print(f"[proxy-confidence-summary] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
