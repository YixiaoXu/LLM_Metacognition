#!/usr/bin/env python3
"""Freeze a pre-treatment critical-sample rule on screen A and test on screen B.

Argmax-change rows are exported separately as post-treatment, exploratory cases.
By default they do not enter the persistent-generation population or the
confirmatory critical-sample estimate.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


PRIMARY = "next_token_auto_logit_delta_projection_to_target"
METRICS = (
    PRIMARY,
    "next_token_auto_logit_margin_delta_toward_target",
    "next_token_js_divergence",
    "next_token_total_variation",
    "next_token_full_logit_delta_l2",
    "next_token_delta_entropy",
    "next_token_delta_max_probability",
    "next_token_argmax_changed",
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
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


def top2_gap(row: Mapping[str, Any]) -> float:
    payload = row.get("next_token_top_probability_tokens", "[]")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return math.nan
    if not isinstance(payload, list) or len(payload) < 2:
        return math.nan
    p1 = max(finite(payload[0].get("prob")), 1e-30)
    p2 = max(finite(payload[1].get("prob")), 1e-30)
    return math.log(p1) - math.log(p2)


def percentile(reference: Sequence[float], value: float, reverse: bool) -> float:
    clean = sorted(item for item in map(float, reference) if math.isfinite(item))
    if not clean or not math.isfinite(value):
        return math.nan
    rank = bisect.bisect_right(clean, value) / len(clean)
    return 1.0 - rank if reverse else rank


def mode_file(directory: str, direction: str, control: str, dose: float) -> str:
    source, target = direction.split(":")
    alpha = str(float(dose)).replace(".", "p")
    # Continuous intervention names use hyphens for controls, while the
    # critical-audit API historically used the shorter underscore label.
    # Accept both spellings so completed intervention outputs remain reusable.
    control_names = {
        "opposite": ("opposite", "opposite-direction", "opposite_direction"),
        "random": ("random", "random-direction", "random_direction"),
        "main": ("main",),
    }.get(control, (control, control.replace("_", "-")))
    matches = []
    for control_name in control_names:
        pattern = os.path.join(
            directory,
            f"continuous_dose_{source}_to_{target}_{control_name}_a{alpha}_generations.jsonl",
        )
        matches.extend(glob.glob(pattern))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {control!r} generation file in {directory}, found {matches}"
        )
    return matches[0]


def load_direction(directory: str, direction: str, dose: float) -> Dict[str, Dict[str, Any]]:
    baseline = {str(row["id"]): row for row in load_jsonl(os.path.join(directory, "baseline_generations.jsonl"))}
    main = {str(row["id"]): row for row in load_jsonl(mode_file(directory, direction, "main", dose))}
    opposite = {str(row["id"]): row for row in load_jsonl(mode_file(directory, direction, "opposite", dose))}
    shared = set(baseline) & set(main) & set(opposite)
    return {
        sample_id: {"baseline": baseline[sample_id], "main": main[sample_id], "opposite": opposite[sample_id]}
        for sample_id in shared
    }


def bootstrap_signflip(values: Sequence[float], samples: int, seed: int) -> Dict[str, Any]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "signflip_p": math.nan}
    rng = np.random.default_rng(seed)
    means = np.asarray([
        array[rng.integers(0, array.size, array.size)].mean() for _ in range(max(samples, 1))
    ])
    observed = abs(float(array.mean()))
    exceed = 0
    for _ in range(max(samples, 1)):
        exceed += abs(float((array * rng.choice([-1.0, 1.0], array.size)).mean())) >= observed
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "signflip_p": float((exceed + 1) / (max(samples, 1) + 1)),
    }


def bh_adjust(values: Sequence[float]) -> List[float]:
    output = [math.nan] * len(values)
    valid = [(index, value) for index, value in enumerate(values) if math.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    running = 1.0
    for reverse_rank in range(len(valid) - 1, -1, -1):
        index, value = valid[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, value * len(valid) / rank)
        output[index] = running
    return output


def score_rows(
    bundle: Mapping[str, Mapping[str, Any]],
    axis_scores: Mapping[str, float],
    references: Mapping[str, Sequence[float]] | None,
    weights: Tuple[float, float, float],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    raw = []
    for sample_id, modes in bundle.items():
        base = modes["baseline"]
        raw.append(
            {
                "id": sample_id,
                "top2_gap": top2_gap(base),
                "entropy": finite(base.get("next_token_entropy")),
                "abs_axis_score": abs(float(axis_scores.get(sample_id, math.nan))),
            }
        )
    if references is None:
        references = {key: [row[key] for row in raw] for key in ("top2_gap", "entropy", "abs_axis_score")}
    output = []
    for row in raw:
        components = (
            percentile(references["top2_gap"], row["top2_gap"], True),
            percentile(references["entropy"], row["entropy"], False),
            percentile(references["abs_axis_score"], row["abs_axis_score"], True),
        )
        if not all(math.isfinite(value) for value in components):
            continue
        score = sum(weight * value for weight, value in zip(weights, components)) / max(sum(weights), 1e-12)
        modes = bundle[row["id"]]
        result = {**row, "critical_score": score}
        for metric in METRICS:
            main = finite(modes["main"].get(metric))
            control = finite(modes["opposite"].get(metric))
            result[f"main__{metric}"] = main
            result[f"opposite__{metric}"] = control
            result[f"contrast__{metric}"] = main - control
        output.append(result)
    return output, {key: list(value) for key, value in references.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-a-dir", required=True)
    parser.add_argument("--screen-b-dir", required=True)
    parser.add_argument("--continuous-axis-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direction", default="0:1")
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--candidate-fractions", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--weights", type=float, nargs=3, default=[0.55, 0.45, 0.0], metavar=("LOGIT_GAP", "ENTROPY", "AXIS_CENTER"))
    parser.add_argument("--trajectory-max-samples", type=int, default=128)
    parser.add_argument(
        "--include-posttreatment-argmax-cases",
        action="store_true",
        help=(
            "Include post-intervention argmax-change cases in trajectory_ids.txt. "
            "Disabled by default so the persistent-generation population is "
            "selected only from pre-intervention quantities."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        axis = torch.load(args.continuous_axis_file, map_location="cpu", weights_only=False)
    except TypeError:
        axis = torch.load(args.continuous_axis_file, map_location="cpu")
    axis_scores = {str(sample_id): float(score) for sample_id, score in zip(axis["ids"], axis["scores"])}
    a, references = score_rows(load_direction(args.screen_a_dir, args.direction, args.dose), axis_scores, None, tuple(args.weights))
    b, _ = score_rows(load_direction(args.screen_b_dir, args.direction, args.dose), axis_scores, references, tuple(args.weights))

    candidates = []
    for fraction in args.candidate_fractions:
        cutoff = float(np.quantile([row["critical_score"] for row in a], 1.0 - fraction))
        values = [row[f"contrast__{PRIMARY}"] for row in a if row["critical_score"] >= cutoff]
        estimate = bootstrap_signflip(values, args.bootstrap_samples, args.seed + len(candidates))
        candidates.append({"fraction": fraction, "cutoff": cutoff, **estimate})
    # Screen A selects one declared rule; screen B is read exactly once below.
    selected = max(candidates, key=lambda row: (finite(row["mean"]), finite(row["ci95_low"])))
    cutoff = float(selected["cutoff"])
    for row in a:
        row["critical_preintervention"] = row["critical_score"] >= cutoff
    for row in b:
        row["critical_preintervention"] = row["critical_score"] >= cutoff
        row["argmax_changed_postintervention"] = bool(row["main__next_token_argmax_changed"])
        row["trajectory_selected"] = False

    critical = [row for row in b if row["critical_preintervention"]]
    argmax_rows = [row for row in b if row["argmax_changed_postintervention"]]
    ordered = sorted(critical, key=lambda row: row["critical_score"], reverse=True)
    if args.include_posttreatment_argmax_cases:
        posttreatment = sorted(
            argmax_rows, key=lambda row: row["critical_score"], reverse=True
        )
        seen = {row["id"] for row in posttreatment}
        ordered = posttreatment + [row for row in ordered if row["id"] not in seen]
    ordered = ordered[: args.trajectory_max_samples]
    selected_ids = {row["id"] for row in ordered}
    for row in b:
        row["trajectory_selected"] = row["id"] in selected_ids

    tests = []
    for population, rows in (("all", b), ("precritical_confirmatory", critical), ("argmax_changed_exploratory", argmax_rows)):
        for metric in METRICS:
            tests.append({"population": population, "metric": metric, **bootstrap_signflip([row[f"contrast__{metric}"] for row in rows], args.bootstrap_samples, args.seed + len(tests) + 101)})
    for population in {row["population"] for row in tests}:
        indices = [index for index, row in enumerate(tests) if row["population"] == population]
        adjusted = bh_adjust([tests[index]["signflip_p"] for index in indices])
        for index, q_value in zip(indices, adjusted):
            tests[index]["bh_fdr_q"] = q_value
    write_csv(os.path.join(args.output_dir, "screen_a_fraction_search.csv"), candidates)
    write_csv(os.path.join(args.output_dir, "screen_a_samples.csv"), a)
    write_csv(os.path.join(args.output_dir, "heldout_sample_groups.csv"), b)
    write_csv(os.path.join(args.output_dir, "heldout_effects.csv"), tests)
    for filename, rows in (
        ("heldout_critical_ids.txt", critical),
        ("heldout_argmax_changed_ids.txt", argmax_rows),
        ("trajectory_ids.txt", ordered),
    ):
        with open(os.path.join(args.output_dir, filename), "w", encoding="utf-8") as handle:
            handle.writelines(f"{row['id']}\n" for row in rows)
    summary = {
        "direction": args.direction,
        "dose_sd": args.dose,
        "screen_a_n": len(a),
        "screen_b_n": len(b),
        "selected_rule": selected,
        "screen_b_critical_n": len(critical),
        "screen_b_argmax_changed_n": len(argmax_rows),
        "trajectory_n": len(ordered),
        "confirmatory_contract": "critical status uses baseline-only quantities and a screen-A-frozen cutoff",
        "exploratory_contract": (
            "argmax-change status is post-treatment and is exported separately"
            if not args.include_posttreatment_argmax_cases
            else "argmax-change status is post-treatment and was explicitly included for descriptive trajectories"
        ),
        "trajectory_selection": (
            "preintervention_critical_plus_posttreatment_argmax"
            if args.include_posttreatment_argmax_cases
            else "preintervention_critical_only"
        ),
    }
    with open(os.path.join(args.output_dir, "critical_heldout_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(
        "[critical-selection] "
        f"screen_a={len(a)} screen_b={len(b)} "
        f"fraction={float(selected['fraction']):.3f} cutoff={cutoff:.4f} "
        f"heldout_critical={len(critical)} trajectory={len(ordered)} "
        f"policy={summary['trajectory_selection']}",
        flush=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
