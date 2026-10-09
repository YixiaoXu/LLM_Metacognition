#!/usr/bin/env python3
"""Test whether signed meta intervention has larger effects near decisions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


STYLE_METRICS = (
    "correct",
    "parse_success",
    "generated_tokens",
    "style_line_count",
    "style_number_count",
    "style_operator_count",
    "style_equation_line_count",
    "style_bullet_line_count",
    "style_step_marker_count",
    "style_explanation_marker_count",
    "style_self_correction_marker_count",
    "style_hash_answer",
    "style_answer_marker",
    "generated_mean_logprob",
)

PRIMARY_METRICS = {
    "generated_tokens",
    "style_line_count",
    "style_step_marker_count",
    "style_explanation_marker_count",
    "generated_mean_logprob",
}


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return float(value.lower() == "true")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def parse_tokens(value: Any) -> List[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    return [int(item) for item in parsed] if isinstance(parsed, list) else []


def first_divergence(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else -1


def mode_maps(directory: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    rows = read_csv(os.path.join(directory, "cluster_intervention_results.csv"))
    modes: Dict[str, Dict[str, Dict[str, str]]] = {
        "baseline": {}, "positive": {}, "negative": {}
    }
    for row in rows:
        mode = row.get("mode", "")
        if mode == "baseline":
            key = "baseline"
        elif "_main_a1p0" in mode:
            key = "positive"
        elif "_opposite_a1p0" in mode:
            key = "negative"
        else:
            continue
        modes[key][str(row["id"])] = row
    return modes


def sample_effects(
    modes: Mapping[str, Mapping[str, Mapping[str, str]]],
    selection: Mapping[str, Mapping[str, str]],
) -> Dict[str, Dict[str, float]]:
    shared = set(modes["baseline"]) & set(modes["positive"]) & set(modes["negative"])
    output: Dict[str, Dict[str, float]] = {}
    for sample_id in shared:
        base = modes["baseline"][sample_id]
        positive = modes["positive"][sample_id]
        negative = modes["negative"][sample_id]
        result: Dict[str, float] = {}
        for metric in STYLE_METRICS:
            b = finite(base.get(metric))
            p = finite(positive.get(metric))
            n = finite(negative.get(metric))
            result[f"directional__{metric}"] = p - n
            result[f"positive_delta__{metric}"] = p - b
            result[f"negative_delta__{metric}"] = n - b
            result[f"sensitivity__{metric}"] = 0.5 * (abs(p - b) + abs(n - b))
        token_sets = {
            "baseline": parse_tokens(base.get("generated_token_ids_json")),
            "positive": parse_tokens(positive.get("generated_token_ids_json")),
            "negative": parse_tokens(negative.get("generated_token_ids_json")),
        }
        critical_index = int(finite(selection.get(sample_id, {}).get("critical_token_index")))
        for label, left, right in (
            ("positive_vs_baseline", token_sets["positive"], token_sets["baseline"]),
            ("negative_vs_baseline", token_sets["negative"], token_sets["baseline"]),
            ("positive_vs_negative", token_sets["positive"], token_sets["negative"]),
        ):
            onset = first_divergence(left, right)
            denominator = max(len(left), len(right), 1)
            result[f"branch__{label}"] = float(onset >= 0)
            result[f"branch_by_critical_step__{label}"] = float(
                onset >= 0 and onset <= critical_index
            )
            result[f"branch_onset_normalized__{label}"] = (
                onset / denominator if onset >= 0 else 1.0
            )
        output[sample_id] = result
    return output


def stats(values: Sequence[float], samples: int, seed: int) -> Dict[str, Any]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "signflip_p": math.nan}
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        array[rng.integers(0, array.size, array.size)].mean()
        for _ in range(max(samples, 1))
    ])
    observed = abs(float(array.mean()))
    exceed = 0
    for _ in range(max(samples, 1)):
        permuted = array * rng.choice([-1.0, 1.0], array.size)
        exceed += abs(float(permuted.mean())) >= observed
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critical-generation-dir", required=True)
    parser.add_argument("--control-generation-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--example-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pairs = read_csv(os.path.join(args.selection_dir, "matched_pairs.csv"))
    selected_rows = [
        *read_csv(os.path.join(args.selection_dir, "selected_critical_samples.csv")),
        *read_csv(os.path.join(args.selection_dir, "selected_control_samples.csv")),
    ]
    selection = {str(row["id"]): row for row in selected_rows}
    critical_modes = mode_maps(args.critical_generation_dir)
    control_modes = mode_maps(args.control_generation_dir)
    critical_effects = sample_effects(critical_modes, selection)
    control_effects = sample_effects(control_modes, selection)

    interaction_rows: List[Dict[str, Any]] = []
    effect_keys = [
        *[f"directional__{metric}" for metric in STYLE_METRICS],
        *[f"sensitivity__{metric}" for metric in STYLE_METRICS],
        "branch__positive_vs_baseline",
        "branch__negative_vs_baseline",
        "branch__positive_vs_negative",
        "branch_by_critical_step__positive_vs_baseline",
        "branch_by_critical_step__negative_vs_baseline",
        "branch_by_critical_step__positive_vs_negative",
    ]
    for key in effect_keys:
        paired_values = []
        critical_values = []
        control_values = []
        for pair in pairs:
            critical_id = str(pair["critical_id"])
            control_id = str(pair["control_id"])
            if critical_id not in critical_effects or control_id not in control_effects:
                continue
            left = finite(critical_effects[critical_id].get(key))
            right = finite(control_effects[control_id].get(key))
            if math.isfinite(left) and math.isfinite(right):
                critical_values.append(left)
                control_values.append(right)
                paired_values.append(left - right)
        estimand, metric = key.split("__", 1)
        result = stats(paired_values, args.bootstrap_samples, args.seed + len(interaction_rows) + 1)
        interaction_rows.append(
            {
                "estimand": estimand,
                "metric": metric,
                "primary_metric": metric in PRIMARY_METRICS,
                "critical_mean": float(np.mean(critical_values)) if critical_values else math.nan,
                "matched_control_mean": float(np.mean(control_values)) if control_values else math.nan,
                "interaction_definition": "critical_effect_minus_matched_high_confidence_effect",
                **result,
            }
        )
    for estimand in sorted({row["estimand"] for row in interaction_rows}):
        indices = [index for index, row in enumerate(interaction_rows) if row["estimand"] == estimand]
        adjusted = bh_adjust([interaction_rows[index]["signflip_p"] for index in indices])
        for index, q_value in zip(indices, adjusted):
            interaction_rows[index]["bh_fdr_q"] = q_value
            p_value = interaction_rows[index].get("signflip_p")
            interaction_rows[index]["significant_p_0p05"] = bool(
                p_value is not None and float(p_value) < 0.05
            )
    write_csv(os.path.join(args.output_dir, "critical_vs_control_interactions.csv"), interaction_rows)

    # Save the per-sample effects to make every reported interaction auditable.
    effect_rows = []
    for group, effects in (("critical", critical_effects), ("matched_control", control_effects)):
        for sample_id, values in effects.items():
            effect_rows.append({"group": group, "id": sample_id, **values})
    write_csv(os.path.join(args.output_dir, "per_sample_effects.csv"), effect_rows)

    ranked_pairs = []
    for pair in pairs:
        critical_id = str(pair["critical_id"])
        control_id = str(pair["control_id"])
        if critical_id not in critical_effects or control_id not in control_effects:
            continue
        critical_branch = critical_effects[critical_id]["branch_by_critical_step__positive_vs_negative"]
        control_branch = control_effects[control_id]["branch_by_critical_step__positive_vs_negative"]
        ranked_pairs.append((critical_branch - control_branch, pair))
    ranked_pairs.sort(key=lambda item: item[0], reverse=True)
    with open(os.path.join(args.output_dir, "critical_examples.md"), "w", encoding="utf-8") as handle:
        handle.write("# Critical-sample intervention examples\n\n")
        for _, pair in ranked_pairs[: args.example_count]:
            critical_id = str(pair["critical_id"])
            handle.write(f"## {critical_id}\n\n")
            handle.write(
                f"- critical score: {selection[critical_id].get('critical_score')}\n"
                f"- critical token: {selection[critical_id].get('critical_token_text_json')} "
                f"at index {selection[critical_id].get('critical_token_index')}\n"
                f"- matched control: {pair['control_id']}\n\n"
            )
            for mode in ("baseline", "positive", "negative"):
                text = critical_modes[mode][critical_id].get("generated_text", "")
                handle.write(f"### {mode}\n\n{text[:4000]}\n\n")

    significant = [row for row in interaction_rows if row.get("significant_p_0p05")]
    summary = {
        "matched_pair_n": len(pairs),
        "critical_complete_n": len(critical_effects),
        "control_complete_n": len(control_effects),
        "primary_hypothesis": "signed or absolute meta-intervention effects are larger in pre-treatment critical samples",
        "interaction_test": "paired bootstrap CI and two-sided sign-flip test over matched pairs",
        "significant_interaction_n": len(significant),
        "significant_interactions": [
            {key: row[key] for key in ("estimand", "metric", "mean", "signflip_p", "bh_fdr_q")}
            for row in significant
        ],
    }
    with open(os.path.join(args.output_dir, "critical_interaction_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
