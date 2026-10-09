#!/usr/bin/env python3
"""Paired persistent-generation analysis for real vs orthogonal-random controls."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from metric_profiles import metrics_for_profile, profile_names


LOGPROB_METRICS = (
    "generated_cumulative_logprob",
    "generated_mean_logprob",
    "generated_min_logprob",
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def token_ids(row: Mapping[str, Any]) -> List[int]:
    try:
        return [int(value) for value in json.loads(row.get("generated_token_ids_json", "[]"))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def branch_changed(left: Sequence[int], right: Sequence[int]) -> float:
    for a, b in zip(left, right):
        if a != b:
            return 1.0
    return float(len(left) != len(right))


def stat(values: Sequence[float], bootstrap: int, permutations: int, seed: int) -> Dict[str, Any]:
    x = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not x.size:
        return {"n": 0, "mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "signflip_p": math.nan}
    rng = np.random.default_rng(seed)
    reps = max(int(bootstrap), 1)
    means = np.asarray([x[rng.integers(0, x.size, x.size)].mean() for _ in range(reps)])
    observed = abs(float(x.mean()))
    exceed = 0
    for _ in range(max(int(permutations), 1)):
        shuffled = x * rng.choice(np.asarray([-1.0, 1.0]), size=x.size)
        exceed += abs(float(shuffled.mean())) >= observed
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "signflip_p": float((exceed + 1) / (max(int(permutations), 1) + 1)),
    }


def resolve_mode(directory: str, control: str, dose: float) -> str:
    alpha = str(float(dose)).replace(".", "p")
    aliases = {
        "main": ["main"],
        "opposite_direction": ["opposite-direction", "opposite"],
        "random_direction": ["random-direction", "random"],
        "random_hidden_direction": [
            "random-hidden-direction",
            "random-hidden",
            "hidden-random",
        ],
    }[control]
    matches: List[str] = []
    for alias in aliases:
        matches.extend(
            glob.glob(
                os.path.join(
                    directory,
                    f"continuous_dose_0_to_1_{alias}_a{alpha}_generations.jsonl",
                )
            )
        )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {control} trajectory, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-profile", choices=profile_names(), default="conversation")
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--permutation-tests", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-truncated",
        action="store_true",
        help="Diagnostic override. The default paired analysis excludes truncated rows.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = {
        "baseline": os.path.join(args.generation_dir, "baseline_generations.jsonl"),
        "main": resolve_mode(args.generation_dir, "main", args.dose),
    }
    for control in ("random_hidden_direction", "random_direction"):
        try:
            paths[control] = resolve_mode(args.generation_dir, control, args.dose)
            break
        except FileNotFoundError:
            continue
    try:
        paths["opposite_direction"] = resolve_mode(
            args.generation_dir, "opposite_direction", args.dose
        )
    except FileNotFoundError:
        pass

    modes = {
        name: {str(row["id"]): row for row in read_jsonl(path)}
        for name, path in paths.items()
    }
    style_metrics = tuple(metrics_for_profile(args.metric_profile)) + LOGPROB_METRICS
    random_control = (
        "random_hidden_direction"
        if "random_hidden_direction" in modes
        else "random_direction"
    )
    comparisons = [("main", "baseline")]
    if random_control in modes:
        comparisons.append(("main", random_control))
    if "opposite_direction" in modes:
        comparisons.extend(
            [("opposite_direction", "baseline"), ("main", "opposite_direction")]
        )

    rows: List[Dict[str, Any]] = []
    for left, right in comparisons:
        shared = sorted(set(modes[left]) & set(modes[right]))
        untruncated = [
            sample_id
            for sample_id in shared
            if finite(modes[left][sample_id].get("style_truncated")) == 0.0
            and finite(modes[right][sample_id].get("style_truncated")) == 0.0
        ]
        populations = [("both_untruncated", untruncated)]
        if args.include_truncated:
            populations.append(("all", shared))
        for population, ids in populations:
            for metric_index, metric in enumerate(style_metrics):
                values = [
                    finite(modes[left][sample_id].get(metric))
                    - finite(modes[right][sample_id].get(metric))
                    for sample_id in ids
                    if math.isfinite(finite(modes[left][sample_id].get(metric)))
                    and math.isfinite(finite(modes[right][sample_id].get(metric)))
                ]
                rows.append(
                    {
                        "population": population,
                        "comparison": f"{left}_minus_{right}",
                        "metric": metric,
                        **stat(
                            values,
                            args.bootstrap_samples,
                            args.permutation_tests,
                            args.seed + len(rows) + metric_index,
                        ),
                    }
                )
            branch_values = [
                branch_changed(token_ids(modes[left][sample_id]), token_ids(modes[right][sample_id]))
                for sample_id in ids
            ]
            rows.append(
                {
                    "population": population,
                    "comparison": f"{left}_minus_{right}",
                    "metric": "response_branch_changed",
                    **stat(
                        branch_values,
                        args.bootstrap_samples,
                        args.permutation_tests,
                        args.seed + len(rows),
                    ),
                }
            )

    write_csv(os.path.join(args.output_dir, "random_orthogonal_trajectory_effects.csv"), rows)
    summary = {
        "generation_dir": args.generation_dir,
        "modes": sorted(modes),
        "comparisons": [f"{left}_minus_{right}" for left, right in comparisons],
        "exclude_truncated": not args.include_truncated,
        "primary_population": (
            "all" if args.include_truncated else "both_untruncated"
        ),
        "random_control": (
            "isotropic hidden-space direction orthogonal to the observed main "
            "hidden delta; hidden norm and semantic drift matched"
            if random_control == "random_hidden_direction"
            else "legacy code-space orthogonal control"
        ),
        "outputs": ["random_orthogonal_trajectory_effects.csv"],
    }
    with open(os.path.join(args.output_dir, "random_orthogonal_trajectory_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
