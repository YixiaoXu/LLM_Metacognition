#!/usr/bin/env python3
"""Describe persistent signed-dose effects for frozen critical/argmax groups."""

from __future__ import annotations

import argparse
import csv
import difflib
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
        writer.writeheader(); writer.writerows(rows)


def finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def stats(values: Sequence[float], samples: int, seed: int) -> Dict[str, Any]:
    x = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not x.size:
        return {"n": 0, "mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "signflip_p": math.nan}
    rng = np.random.default_rng(seed)
    means = np.asarray([x[rng.integers(0, x.size, x.size)].mean() for _ in range(max(samples, 1))])
    observed = abs(float(x.mean()))
    exceed = sum(abs(float((x * rng.choice([-1.0, 1.0], x.size)).mean())) >= observed for _ in range(max(samples, 1)))
    return {"n": int(x.size), "mean": float(x.mean()), "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975)), "signflip_p": float((exceed + 1) / (max(samples, 1) + 1))}


def bh_adjust(values: Sequence[float]) -> List[float]:
    output = [math.nan] * len(values)
    valid = sorted(
        [(index, value) for index, value in enumerate(values) if math.isfinite(value)],
        key=lambda item: item[1],
    )
    running = 1.0
    for reverse_rank in range(len(valid) - 1, -1, -1):
        index, value = valid[reverse_rank]
        running = min(running, value * len(valid) / (reverse_rank + 1))
        output[index] = running
    return output


def token_ids(row: Mapping[str, Any]) -> List[int]:
    try:
        return [int(value) for value in json.loads(row.get("generated_token_ids_json", "[]"))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else -1


def branch_metrics(a: Sequence[int], b: Sequence[int]) -> Dict[str, float]:
    divergence = first_divergence(a, b)
    changed = divergence >= 0
    shortest = max(min(len(a), len(b)), 1)
    longest = max(len(a), len(b), 1)
    common_prefix = divergence if changed else min(len(a), len(b))
    return {
        "response_branch_changed": float(changed),
        "response_branch_onset_token_conditional": (
            float(divergence) if changed else math.nan
        ),
        "response_branch_onset_normalized": (
            float(divergence / longest) if changed else 1.0
        ),
        "response_common_prefix_fraction": float(common_prefix / shortest),
        "response_token_sequence_similarity": float(
            difflib.SequenceMatcher(a=list(a), b=list(b), autojunk=False).ratio()
        ),
    }


def prefix_logprobs(row: Mapping[str, Any]) -> Dict[int, float]:
    try:
        payload = json.loads(row.get("generated_cumulative_logprob_prefix_json", "{}"))
        return {
            int(key): float(value)
            for key, value in payload.items()
            if math.isfinite(float(value))
        }
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def mode_path(directory: str, control: str, dose: float) -> str:
    alpha = str(float(dose)).replace(".", "p")
    aliases = {
        "main": ["main"],
        "opposite": ["opposite-direction", "opposite"],
        "random": ["random-direction", "random"],
    }.get(control, [control])
    matches = []
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
        raise FileNotFoundError(f"Expected one 0:1/{control} generation file, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--critical-dir",
        default="",
        help="Optional critical-sample metadata. Omit for an all-population trajectory.",
    )
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metric-profile", choices=profile_names(), default="safety"
    )
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--example-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-truncated",
        action="store_true",
        help="Diagnostic override. The default trajectory analysis uses only samples untruncated in every compared mode.",
    )
    args = parser.parse_args()
    style_metrics = tuple(metrics_for_profile(args.metric_profile)) + LOGPROB_METRICS
    os.makedirs(args.output_dir, exist_ok=True)
    modes = {
        "baseline": {str(row["id"]): row for row in read_jsonl(os.path.join(args.generation_dir, "baseline_generations.jsonl"))},
        "positive": {str(row["id"]): row for row in read_jsonl(mode_path(args.generation_dir, "main", args.dose))},
        "negative": {str(row["id"]): row for row in read_jsonl(mode_path(args.generation_dir, "opposite", args.dose))},
    }
    shared_modes = set(modes["baseline"]) & set(modes["positive"]) & set(modes["negative"])
    if args.critical_dir:
        groups = {
            row["id"]: row
            for row in read_csv(
                os.path.join(args.critical_dir, "heldout_sample_groups.csv")
            )
        }
        shared = sorted(set(groups) & shared_modes)
    else:
        shared = sorted(shared_modes)
        groups = {sample_id: {"id": sample_id} for sample_id in shared}
    all_untruncated = [
        sample_id
        for sample_id in shared
        if all(
            finite(modes[mode][sample_id].get("style_truncated")) == 0.0
            for mode in ("baseline", "positive", "negative")
        )
    ]
    analysis_base = shared if args.include_truncated else all_untruncated
    populations = {"all_modes_untruncated": all_untruncated}
    if args.include_truncated:
        populations["trajectory_all"] = shared
    if args.critical_dir:
        populations.update(
            {
                "precritical_confirmatory": [sample_id for sample_id in analysis_base if groups[sample_id].get("critical_preintervention", "").lower() == "true"],
                "argmax_changed_exploratory": [sample_id for sample_id in analysis_base if groups[sample_id].get("argmax_changed_postintervention", "").lower() == "true"],
            }
        )
    effects = []
    prefix_effects = []
    comparisons = (("positive", "baseline"), ("negative", "baseline"), ("positive", "negative"))
    for population, ids in populations.items():
        for left, right in comparisons:
            for metric in style_metrics:
                values = [finite(modes[left][sample_id].get(metric)) - finite(modes[right][sample_id].get(metric)) for sample_id in ids]
                effects.append({"population": population, "comparison": f"{left}_minus_{right}", "metric": metric, **stats(values, args.bootstrap_samples, args.seed + len(effects))})
            branches = [
                branch_metrics(
                    token_ids(modes[left][sample_id]),
                    token_ids(modes[right][sample_id]),
                )
                for sample_id in ids
            ]
            for metric in (
                "response_branch_changed",
                "response_branch_onset_normalized",
                "response_common_prefix_fraction",
                "response_token_sequence_similarity",
            ):
                effects.append(
                    {
                        "population": population,
                        "comparison": f"{left}_minus_{right}",
                        "metric": metric,
                        **stats(
                            [row[metric] for row in branches],
                            args.bootstrap_samples,
                            args.seed + len(effects),
                        ),
                    }
                )
            conditional = [
                row["response_branch_onset_token_conditional"] for row in branches
                if math.isfinite(row["response_branch_onset_token_conditional"])
            ]
            conditional_summary = stats(
                conditional, args.bootstrap_samples, args.seed + len(effects)
            )
            conditional_summary["signflip_p"] = math.nan
            effects.append(
                {
                    "population": population,
                    "comparison": f"{left}_minus_{right}",
                    "metric": "response_branch_onset_token_conditional",
                    **conditional_summary,
                    "conditional_post_divergence_statistic": True,
                }
            )
            all_steps = sorted(
                set().union(
                    *(set(prefix_logprobs(modes[mode][sample_id])) for mode in (left, right) for sample_id in ids)
                )
            )
            for step in all_steps:
                values = []
                for sample_id in ids:
                    left_prefix = prefix_logprobs(modes[left][sample_id])
                    right_prefix = prefix_logprobs(modes[right][sample_id])
                    if step in left_prefix and step in right_prefix:
                        values.append(left_prefix[step] - right_prefix[step])
                prefix_effects.append(
                    {
                        "population": population,
                        "comparison": f"{left}_minus_{right}",
                        "prefix_tokens": step,
                        **stats(
                            values,
                            args.bootstrap_samples,
                            args.seed + len(prefix_effects) + 5001,
                        ),
                    }
                )
    for population in populations:
        for comparison in {row["comparison"] for row in effects}:
            indices = [index for index, row in enumerate(effects) if row["population"] == population and row["comparison"] == comparison]
            adjusted = bh_adjust([effects[index]["signflip_p"] for index in indices])
            for index, q_value in zip(indices, adjusted):
                effects[index]["bh_fdr_q"] = q_value
    write_csv(os.path.join(args.output_dir, "trajectory_behavior_effects.csv"), effects)
    for population in populations:
        for comparison in {row["comparison"] for row in prefix_effects}:
            indices = [
                index for index, row in enumerate(prefix_effects)
                if row["population"] == population and row["comparison"] == comparison
            ]
            adjusted = bh_adjust([prefix_effects[index]["signflip_p"] for index in indices])
            for index, q_value in zip(indices, adjusted):
                prefix_effects[index]["bh_fdr_q"] = q_value
    write_csv(
        os.path.join(args.output_dir, "trajectory_cumulative_logprob_effects.csv"),
        prefix_effects,
    )

    fidelity = []
    for control, direction_name in (("main", "positive"), ("opposite", "negative")):
        alpha = str(float(args.dose)).replace(".", "p")
        pattern = os.path.join(args.generation_dir, f"refined_residual_intervention_audit_continuous_dose_0_to_1_{control}_a{alpha}.csv")
        if not os.path.exists(pattern):
            continue
        audit = read_csv(pattern)
        for population, ids in populations.items():
            id_set = set(ids)
            selected = [row for row in audit if row.get("id") in id_set]
            steps = sorted({int(float(row.get("step", -1))) for row in selected})
            for step in steps:
                rows = [row for row in selected if int(float(row.get("step", -1))) == step]
                for metric in ("actual_code_delta_l2", "code_margin_delta_toward_target", "dose_direction_cosine", "semantic_relative_delta", "hidden_relative_delta", "code_edit_fraction_achieved"):
                    values = [finite(row.get(metric)) for row in rows]
                    fidelity.append({"population": population, "direction": direction_name, "step": step, "metric": metric, "n": sum(math.isfinite(value) for value in values), "mean": float(np.nanmean(values)) if any(math.isfinite(value) for value in values) else math.nan})
                fidelity.append({"population": population, "direction": direction_name, "step": step, "metric": "code_correction_accepted_rate", "n": len(rows), "mean": sum(row.get("code_correction_accepted", "").lower() == "true" for row in rows) / len(rows) if rows else math.nan})
    write_csv(os.path.join(args.output_dir, "trajectory_code_fidelity.csv"), fidelity)

    if args.critical_dir:
        ranked = sorted(analysis_base, key=lambda sample_id: (groups[sample_id].get("argmax_changed_postintervention", "").lower() == "true", finite(groups[sample_id].get("critical_score"))), reverse=True)
    else:
        ranked = sorted(
            analysis_base,
            key=lambda sample_id: 1.0
            - branch_metrics(
                token_ids(modes["positive"][sample_id]),
                token_ids(modes["negative"][sample_id]),
            )["response_token_sequence_similarity"],
            reverse=True,
        )
    with open(os.path.join(args.output_dir, "trajectory_examples.md"), "w", encoding="utf-8") as handle:
        handle.write("# Persistent continuous-dose examples\n\n")
        if args.critical_dir:
            handle.write("Argmax-changed cases are exploratory post-treatment selections.\n\n")
        else:
            handle.write("Examples are ranked by positive-vs-negative branch divergence.\n\n")
        for sample_id in ranked[: args.example_count]:
            handle.write(f"## {sample_id}\n\n")
            handle.write(f"- precritical: {groups[sample_id].get('critical_preintervention')}\n")
            handle.write(f"- argmax changed: {groups[sample_id].get('argmax_changed_postintervention')}\n\n")
            for mode in ("baseline", "positive", "negative"):
                text = str(modes[mode][sample_id].get("generated_text", "")).strip()
                handle.write(
                    f"### {mode}\n\n"
                    f"- cumulative log-prob: {modes[mode][sample_id].get('generated_cumulative_logprob')}\n"
                    f"- mean token log-prob: {modes[mode][sample_id].get('generated_mean_logprob')}\n\n"
                    f"{text}\n\n"
                )
    summary = {
        "n_shared": len(shared),
        "exclude_truncated": not args.include_truncated,
        "truncated_rows_removed": len(shared) - len(all_untruncated),
        "analysis_population": (
            "all_rows" if args.include_truncated else "all_modes_untruncated"
        ),
        "population_counts": {key: len(value) for key, value in populations.items()},
        "primary_style_population": "all_modes_untruncated",
        "positive_negative_are_signed_continuous_axis_doses": True,
        "metric_profile": args.metric_profile,
        "argmax_changed_population_is_exploratory": True,
        "outputs": [
            "trajectory_behavior_effects.csv",
            "trajectory_cumulative_logprob_effects.csv",
            "trajectory_code_fidelity.csv",
            "trajectory_examples.md",
        ],
    }
    with open(os.path.join(args.output_dir, "trajectory_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
