#!/usr/bin/env python3
"""Posthoc main/opposite/common-shift analysis for persistent style interventions."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


NON_METRIC_COLUMNS = {
    "id",
    "mode",
    "activation_cluster",
    "activation_cluster_name",
    "style_truncated",
}


def finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_source_cluster(path: str, main_mode: str) -> str | None:
    config_path = os.path.join(os.path.dirname(path), "run_config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        value = config.get("summary", {}).get("modes", {}).get(
            main_mode, {}
        ).get("source_cluster")
        return None if value is None else str(value)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


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


def paired_stats(
    values: Sequence[float],
    bootstrap_samples: int,
    permutation_tests: int,
    seed: int,
) -> Dict[str, Any]:
    values = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean_delta": math.nan,
            "median_delta": math.nan,
            "bootstrap_ci95_low": math.nan,
            "bootstrap_ci95_high": math.nan,
            "signflip_two_sided_p": math.nan,
        }

    rng = np.random.default_rng(seed)
    boot_n = max(int(bootstrap_samples), 1)
    indices = rng.integers(0, n, size=(boot_n, n))
    boot_means = values[indices].mean(axis=1)
    observed = abs(float(values.mean()))
    perm_n = max(int(permutation_tests), 1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(perm_n, n))
    null_means = (signs * values[None, :]).mean(axis=1)
    p_value = (float(np.count_nonzero(np.abs(null_means) >= observed)) + 1.0) / (
        perm_n + 1.0
    )
    return {
        "n": n,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "bootstrap_ci95_low": float(np.quantile(boot_means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot_means, 0.975)),
        "signflip_two_sided_p": p_value,
    }


def bh_adjust(values: Sequence[float]) -> List[float]:
    output = [math.nan] * len(values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if math.isfinite(float(value))
    ]
    valid.sort(key=lambda item: item[1])
    running = 1.0
    for reverse_index in range(len(valid) - 1, -1, -1):
        index, value = valid[reverse_index]
        rank = reverse_index + 1
        running = min(running, value * len(valid) / max(rank, 1))
        output[index] = running
    return output


def get_style_metrics(rows: Sequence[Mapping[str, str]]) -> List[str]:
    if not rows:
        return []
    return [
        key
        for key in rows[0]
        if (
            (key.startswith("style_") or key == "generated_tokens")
            and key not in NON_METRIC_COLUMNS
        )
    ]


def mode_rows(
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[str, Mapping[str, str]]]:
    result: Dict[str, Dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        sample_id = str(row.get("id", ""))
        mode = str(row.get("mode", ""))
        if sample_id and mode:
            result[mode][sample_id] = row
    return result


def analyze_file(
    path: str,
    bootstrap_samples: int,
    permutation_tests: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = read_csv(path)
    modes = mode_rows(rows)
    main_modes = sorted(mode for mode in modes if "_main_" in mode)
    opposite_modes = sorted(
        mode for mode in modes if "_opposite_" in mode
    )
    if "baseline" not in modes or not main_modes or not opposite_modes:
        return []

    main_mode = main_modes[0]
    opposite_mode = opposite_modes[0]
    baseline = modes["baseline"]
    main = modes[main_mode]
    opposite = modes[opposite_mode]
    source_cluster = read_source_cluster(path, main_mode)
    if source_cluster is not None:
        baseline = {
            sample_id: row
            for sample_id, row in baseline.items()
            if str(row.get("activation_cluster")) == source_cluster
        }
        main = {
            sample_id: row
            for sample_id, row in main.items()
            if str(row.get("activation_cluster")) == source_cluster
        }
        opposite = {
            sample_id: row
            for sample_id, row in opposite.items()
            if str(row.get("activation_cluster")) == source_cluster
        }
    shared = sorted(set(baseline) & set(main) & set(opposite))
    populations = {
        "all_paired": shared,
        "both_untruncated": [
            sample_id
            for sample_id in shared
            if finite(baseline[sample_id].get("style_truncated")) == 0.0
            and finite(main[sample_id].get("style_truncated")) == 0.0
            and finite(opposite[sample_id].get("style_truncated")) == 0.0
        ],
    }
    output: List[Dict[str, Any]] = []
    metrics = get_style_metrics(rows)
    for population, sample_ids in populations.items():
        for metric_index, metric in enumerate(metrics):
            arrays = {"baseline": [], "main": [], "opposite": []}
            for sample_id in sample_ids:
                arrays["baseline"].append(
                    finite(baseline[sample_id].get(metric))
                )
                arrays["main"].append(finite(main[sample_id].get(metric)))
                arrays["opposite"].append(
                    finite(opposite[sample_id].get(metric))
                )
            values = {
                "main_minus_baseline": [
                    main_value - base_value
                    for main_value, base_value in zip(
                        arrays["main"], arrays["baseline"]
                    )
                    if math.isfinite(main_value) and math.isfinite(base_value)
                ],
                "opposite_minus_baseline": [
                    opposite_value - base_value
                    for opposite_value, base_value in zip(
                        arrays["opposite"], arrays["baseline"]
                    )
                    if math.isfinite(opposite_value) and math.isfinite(base_value)
                ],
                "main_minus_opposite": [
                    main_value - opposite_value
                    for main_value, opposite_value in zip(
                        arrays["main"], arrays["opposite"]
                    )
                    if math.isfinite(main_value) and math.isfinite(opposite_value)
                ],
                "common_shift": [
                    0.5 * (main_value + opposite_value) - base_value
                    for main_value, opposite_value, base_value in zip(
                        arrays["main"], arrays["opposite"], arrays["baseline"]
                    )
                    if (
                        math.isfinite(main_value)
                        and math.isfinite(opposite_value)
                        and math.isfinite(base_value)
                    )
                ],
            }
            for comparison, comparison_values in values.items():
                stats = paired_stats(
                    comparison_values,
                    bootstrap_samples,
                    permutation_tests,
                    seed + 1009 * metric_index + 65537 * len(output),
                )
                output.append(
                    {
                        "module_dir": os.path.dirname(path),
                        "main_mode": main_mode,
                        "opposite_mode": opposite_mode,
                        "source_cluster": source_cluster,
                        "population": population,
                        "metric": metric,
                        "comparison": comparison,
                        **stats,
                    }
                )

    for population in populations:
        for comparison in (
            "main_minus_baseline",
            "opposite_minus_baseline",
            "main_minus_opposite",
            "common_shift",
        ):
            indices = [
                index
                for index, row in enumerate(output)
                if row["population"] == population
                and row["comparison"] == comparison
            ]
            q_values = bh_adjust(
                [output[index]["signflip_two_sided_p"] for index in indices]
            )
            for index, q_value in zip(indices, q_values):
                output[index]["fdr_family"] = (
                    f"{population}:{comparison}:style"
                )
                output[index]["fdr_family_size"] = len(indices)
                output[index]["bh_fdr_q"] = q_value
                p_value = output[index].get("signflip_two_sided_p")
                output[index]["significant_p_0p05"] = bool(
                    p_value is not None
                    and math.isfinite(float(p_value))
                    and float(p_value) < 0.05
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-tests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        args.run_root, "posthoc_direction_contrasts"
    )
    os.makedirs(output_dir, exist_ok=True)
    patterns = [
        os.path.join(
            args.run_root,
            "*",
            "continuous_modules",
            "rank_*",
            "persistent_trajectory",
            "all_generations_style_metrics.csv",
        ),
        os.path.join(
            args.run_root,
            "continuous_modules",
            "rank_*",
            "persistent_trajectory",
            "all_generations_style_metrics.csv",
        ),
    ]
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    print(f"[posthoc-direction] files={len(paths)}", flush=True)

    all_rows: List[Dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        rows = analyze_file(
            path,
            args.bootstrap_samples,
            args.permutation_tests,
            args.seed + index * 100003,
        )
        all_rows.extend(rows)
        print(
            f"[posthoc-direction] {index}/{len(paths)} rows={len(rows)} "
            f"path={os.path.dirname(path)}",
            flush=True,
        )

    csv_path = os.path.join(output_dir, "style_direction_contrasts.csv")
    write_csv(csv_path, all_rows)
    significant = [
        row for row in all_rows if row.get("significant_p_0p05")
    ]
    summary = {
        "run_root": args.run_root,
        "style_metric_file_count": len(paths),
        "contrast_row_count": len(all_rows),
        "significant_p_lt_0p05_count": len(significant),
        "output": csv_path,
        "comparisons": [
            "main_minus_baseline",
            "opposite_minus_baseline",
            "main_minus_opposite",
            "common_shift",
        ],
    }
    with open(
        os.path.join(output_dir, "style_direction_contrasts_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    for row in sorted(
        significant,
        key=lambda item: float(item.get("signflip_two_sided_p", math.inf)),
    )[:40]:
        module = os.path.basename(
            os.path.dirname(os.path.dirname(row["module_dir"]))
        )
        print(
            f"[significant] {module} {row['population']} "
            f"{row['comparison']} {row['metric']} "
            f"delta={row['mean_delta']:.4g} "
            f"p={row['signflip_two_sided_p']:.4g} n={row['n']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
