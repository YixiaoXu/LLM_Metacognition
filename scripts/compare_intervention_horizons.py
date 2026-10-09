#!/usr/bin/env python3
"""Join matched single-step and continuous intervention summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def join_rows(
    single: Sequence[Mapping[str, str]],
    continuous: Sequence[Mapping[str, str]],
    key_fields: Tuple[str, ...],
) -> List[Dict[str, object]]:
    single_map = {tuple(row.get(field, "") for field in key_fields): row for row in single}
    continuous_map = {
        tuple(row.get(field, "") for field in key_fields): row for row in continuous
    }
    output: List[Dict[str, object]] = []
    for key in sorted(set(single_map) & set(continuous_map)):
        left, right = single_map[key], continuous_map[key]
        single_mean, continuous_mean = number(left.get("mean")), number(right.get("mean"))
        row: Dict[str, object] = dict(zip(key_fields, key))
        row.update(
            {
                "single_step_n": left.get("n", ""),
                "single_step_mean": single_mean,
                "single_step_ci95_low": number(left.get("ci95_low")),
                "single_step_ci95_high": number(left.get("ci95_high")),
                "single_step_exact_p": number(left.get("signflip_p")),
                "continuous_n": right.get("n", ""),
                "continuous_mean": continuous_mean,
                "continuous_ci95_low": number(right.get("ci95_low")),
                "continuous_ci95_high": number(right.get("ci95_high")),
                "continuous_exact_p": number(right.get("signflip_p")),
                "continuous_minus_single_effect": continuous_mean - single_mean,
                "continuous_to_single_abs_effect_ratio": (
                    abs(continuous_mean) / abs(single_mean)
                    if math.isfinite(single_mean) and abs(single_mean) > 1e-12
                    else math.nan
                ),
            }
        )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-step-dir", type=Path, required=True)
    parser.add_argument("--continuous-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    behavior = join_rows(
        read_csv(args.single_step_dir / "trajectory_behavior_effects.csv"),
        read_csv(args.continuous_dir / "trajectory_behavior_effects.csv"),
        ("population", "comparison", "metric"),
    )
    cumulative = join_rows(
        read_csv(args.single_step_dir / "trajectory_cumulative_logprob_effects.csv"),
        read_csv(args.continuous_dir / "trajectory_cumulative_logprob_effects.csv"),
        ("population", "comparison", "prefix_tokens"),
    )
    write_csv(args.output_dir / "single_vs_continuous_behavior.csv", behavior)
    write_csv(args.output_dir / "single_vs_continuous_cumulative_logprob.csv", cumulative)
    summary = {
        "design": "same module, evaluation ids, signed dose, hidden budget, and baseline",
        "single_step_patch_steps": 1,
        "continuous_patch_steps": "configured persistent patch steps",
        "behavior_comparisons": len(behavior),
        "cumulative_logprob_comparisons": len(cumulative),
        "note": "The table reports each paired intervention effect and their descriptive difference; exact p values belong to each pre-registered within-horizon contrast.",
    }
    with (args.output_dir / "intervention_horizon_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

