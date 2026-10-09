#!/usr/bin/env python3
"""Test a frozen behavior construct across signed persistent intervention doses."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

import _bootstrap  # noqa: F401
from analyze_continuous_meta_behavior import normalize_behavior_row
from metacog.evaluation import metric_families, metric_polarity, profile_names
from metacog.statistics import paired_bootstrap_and_signflip


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def dose_path(directory: Path, control: str, dose: float) -> Path:
    encoded = str(float(dose)).replace(".", "p")
    aliases = {
        "positive": ("main",),
        "negative": ("opposite-direction", "opposite_direction", "opposite"),
    }[control]
    matches: List[str] = []
    for alias in aliases:
        matches.extend(
            glob.glob(
                str(
                    directory
                    / f"continuous_dose_0_to_1_{alias}_a{encoded}_generations.jsonl"
                )
            )
        )
    if len(set(matches)) != 1:
        raise FileNotFoundError(
            f"Expected one {control} generation file for dose={dose}, found {matches}"
        )
    return Path(matches[0])


def one_sided_p(values: List[float], expected_sign: float, tests: int, seed: int) -> float:
    clean = torch.tensor(
        [expected_sign * value for value in values if math.isfinite(value)],
        dtype=torch.float32,
    )
    if clean.numel() == 0:
        return math.nan
    generator = torch.Generator().manual_seed(seed)
    observed = float(clean.mean().item())
    extreme = 0
    for start in range(0, tests, 512):
        current = min(512, tests - start)
        signs = torch.randint(
            0, 2, (current, clean.numel()), generator=generator
        ).float().mul_(2.0).sub_(1.0)
        permuted = (signs * clean.view(1, -1)).mean(dim=1)
        extreme += int((permuted >= observed).sum().item())
    return float((extreme + 1) / (tests + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-profile", choices=profile_names(), required=True)
    parser.add_argument("--construct", required=True)
    parser.add_argument("--expected-direction", choices=["positive", "negative"], required=True)
    parser.add_argument("--doses", nargs="+", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--permutation-tests", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    families = metric_families(args.metric_profile)
    if args.construct not in families:
        raise ValueError(
            f"Unknown construct {args.construct!r} for {args.metric_profile}: "
            f"{list(families)}"
        )
    generation_dir = Path(args.generation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = {
        str(row["id"]): normalize_behavior_row(row, args.max_new_tokens)
        for row in read_jsonl(generation_dir / "baseline_generations.jsonl")
    }
    modes: Dict[tuple[str, float], Dict[str, Dict[str, Any]]] = {}
    for dose in sorted(set(args.doses)):
        for direction in ("positive", "negative"):
            modes[(direction, dose)] = {
                str(row["id"]): normalize_behavior_row(row, args.max_new_tokens)
                for row in read_jsonl(dose_path(generation_dir, direction, dose))
            }

    shared_ids = set(baseline_rows)
    for rows in modes.values():
        shared_ids &= set(rows)
    shared = sorted(
        sample_id
        for sample_id in shared_ids
        if all(
            finite(rows[sample_id].get("style_truncated")) == 0.0
            for rows in [baseline_rows, *modes.values()]
        )
    )
    metrics = [
        metric
        for metric in families[args.construct]
        if sum(
            finite(baseline_rows[sample_id].get(metric)) is not None
            for sample_id in shared
        )
        >= 2
    ]
    if not metrics:
        raise ValueError(
            f"No finite component metrics for construct {args.construct!r}."
        )

    scales: Dict[str, float] = {}
    for metric in metrics:
        values = [
            finite(baseline_rows[sample_id].get(metric))
            for sample_id in shared
        ]
        tensor = torch.tensor(
            [value for value in values if value is not None], dtype=torch.float32
        )
        scales[metric] = max(float(tensor.std(unbiased=False).item()), 1e-6)

    component_rows: List[Dict[str, Any]] = []
    per_sample_effects: Dict[str, Dict[float, float]] = {
        sample_id: {} for sample_id in shared
    }
    for dose in sorted(set(args.doses)):
        positive = modes[("positive", dose)]
        negative = modes[("negative", dose)]
        for metric in metrics:
            component_values: List[float] = []
            for sample_id in shared:
                pos = finite(positive[sample_id].get(metric))
                neg = finite(negative[sample_id].get(metric))
                if pos is None or neg is None:
                    continue
                component_values.append(
                    metric_polarity(metric) * (pos - neg) / scales[metric]
                )
            test = paired_bootstrap_and_signflip(
                component_values,
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed + 101 * len(component_rows),
            )
            component_rows.append(
                {
                    "construct": args.construct,
                    "metric": metric,
                    "dose": dose,
                    "polarity": metric_polarity(metric),
                    "baseline_sd": scales[metric],
                    "n": test["n"],
                    "mean_standardized_positive_minus_negative": test["mean"],
                    "bootstrap_ci95_low": test["bootstrap_ci95"][0],
                    "bootstrap_ci95_high": test["bootstrap_ci95"][1],
                    "signflip_two_sided_p": test["signflip_two_sided_p"],
                }
            )
        for sample_id in shared:
            values = []
            for metric in metrics:
                pos = finite(positive[sample_id].get(metric))
                neg = finite(negative[sample_id].get(metric))
                if pos is None or neg is None:
                    continue
                values.append(
                    metric_polarity(metric) * (pos - neg) / scales[metric]
                )
            if values:
                per_sample_effects[sample_id][dose] = sum(values) / len(values)

    doses = sorted(set(args.doses))
    denominator = sum(dose * dose for dose in doses)
    slopes = [
        sum(dose * values[dose] for dose in doses) / max(denominator, 1e-12)
        for values in per_sample_effects.values()
        if all(dose in values for dose in doses)
    ]
    test = paired_bootstrap_and_signflip(
        slopes,
        args.bootstrap_samples,
        args.permutation_tests,
        args.seed + 9001,
    )
    mean_effect = test["mean"]
    expected_sign = 1.0 if args.expected_direction == "positive" else -1.0
    construct_rows = [
        {
            "population": "all_modes_untruncated",
            "comparison": "positive_minus_negative",
            "construct": args.construct,
            "component_metrics": " ".join(metrics),
            "expected_direction": args.expected_direction,
            "dose_levels": " ".join(str(dose) for dose in doses),
            "n": test["n"],
            "aligned_effect_per_dose": mean_effect,
            "bootstrap_ci95_low": test["bootstrap_ci95"][0],
            "bootstrap_ci95_high": test["bootstrap_ci95"][1],
            "signflip_two_sided_p": test["signflip_two_sided_p"],
            "directional_one_sided_p": one_sided_p(
                slopes,
                expected_sign,
                args.permutation_tests,
                args.seed + 9103,
            ),
            "direction_consistent": (
                mean_effect is not None and mean_effect * expected_sign > 0
            ),
            "estimand": (
                "per-sample slope of the polarity-aligned, equal-component-weight "
                "positive-minus-negative construct effect across frozen doses"
            ),
        }
    ]
    write_csv(output_dir / "construct_component_effects.csv", component_rows)
    write_csv(output_dir / "construct_dose_response.csv", construct_rows)
    summary = {
        "metric_profile": args.metric_profile,
        "construct": args.construct,
        "expected_direction": args.expected_direction,
        "n_shared_untruncated": len(shared),
        "doses": doses,
        "metrics": metrics,
        "result": construct_rows[0],
    }
    (output_dir / "construct_dose_response.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
