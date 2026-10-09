#!/usr/bin/env python3
"""Evaluate continuous meta modules against confidence-like model behavior.

This is deliberately separate from task accuracy and ordinary response-style
summaries.  The primary outcomes are output-distribution concentration and
trajectory log-probability, treated as confidence/uncertainty proxies rather
than calibrated correctness probabilities.

The analysis uses held-out cross-fitted predictive information for baseline
associations and paired signed-dose contrasts for persistent interventions.
No outcome is used to select a module in this script.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

import _bootstrap  # noqa: F401
from analyze_continuous_meta_behavior import (
    crossfit_predictive_loss,
    make_fold_ids,
    normalize_behavior_row,
    predictive_information_decomposition,
    read_behavior_rows,
    safe_load,
)
from metacog.statistics import bh_adjust, paired_bootstrap_and_signflip


PRIMARY_METRICS = (
    "prompt_end_entropy",
    "prompt_end_normalized_entropy",
    "prompt_end_top1_top2_logit_margin",
    "prompt_end_max_probability",
    "generated_prefix16_entropy_mean",
    "generated_prefix16_logit_margin_mean",
    "generated_mean_logprob",
    "generated_min_logprob",
    "generated_cumulative_logprob_per_token",
)

# Keep the association estimate conservative, but do not turn a small
# held-out intersection into a failed module.  The downstream runner can then
# continue with intervention and the summary records why this diagnostic was
# skipped.
MIN_ASSOCIATION_SAMPLES = 50

EASY_SECONDARY_METRICS = (
    # Confidence transition during the first generated prefix.
    "confidence_entropy_shift_prompt_to_prefix16",
    "confidence_margin_shift_prompt_to_prefix16",
    # Whether the model allocates more computation when its internal state is
    # less concentrated. These are control/resource outcomes, not cognition by
    # themselves.
    "generated_tokens",
    # Simple linguistic readouts of uncertainty and monitoring.
    "style_hedging_score",
    "style_certainty_score",
    "style_self_correction_marker_count",
    "style_refusal_present",
    "style_refusal_onset_normalized",
)


def finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [row for row in read_behavior_rows(str(path))]


def prefix_payload(row: Mapping[str, Any]) -> Dict[int, float]:
    try:
        payload = json.loads(
            str(row.get("generated_cumulative_logprob_prefix_json", "{}"))
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    output: Dict[int, float] = {}
    for key, value in payload.items():
        number = finite(value)
        if number is not None:
            output[int(key)] = number
    return output


def metric_value(row: Mapping[str, Any], metric: str) -> Optional[float]:
    direct = finite(row.get(metric))
    if direct is not None:
        return direct
    fallback_fields = {
        "style_hedging_score": "hedging_score",
        "style_certainty_score": "certainty_score",
        "style_self_correction_marker_count": "self_correction_marker_count",
        "style_refusal_present": "model_refusal",
        "style_refusal_onset_normalized": "refusal_onset_normalized",
    }
    fallback = finite(row.get(fallback_fields.get(metric, "")))
    if fallback is not None:
        return fallback
    if metric == "generated_cumulative_logprob_per_token":
        total = finite(row.get("generated_cumulative_logprob"))
        count = finite(row.get("generated_logprob_token_count"))
        return total / count if total is not None and count and count > 0 else None
    if metric == "confidence_entropy_shift_prompt_to_prefix16":
        prompt = finite(row.get("prompt_end_entropy"))
        prefix = finite(row.get("generated_prefix16_entropy_mean"))
        return prefix - prompt if prompt is not None and prefix is not None else None
    if metric == "confidence_margin_shift_prompt_to_prefix16":
        prompt = finite(row.get("prompt_end_top1_top2_logit_margin"))
        prefix = finite(row.get("generated_prefix16_logit_margin_mean"))
        return prefix - prompt if prompt is not None and prefix is not None else None
    if metric.startswith("generated_cumulative_logprob_prefix_"):
        try:
            step = int(metric.rsplit("_", 1)[-1])
        except ValueError:
            return None
        value = prefix_payload(row).get(step)
        return value / step if value is not None and step > 0 else None
    return None


def available_metrics(
    rows: Sequence[Mapping[str, Any]], include_secondary: bool
) -> List[str]:
    metrics = list(PRIMARY_METRICS)
    if include_secondary:
        metrics.extend(EASY_SECONDARY_METRICS)
    checkpoints = sorted(
        {
            step
            for row in rows
            for step in prefix_payload(row)
            if step > 0
        }
    )
    metrics.extend(f"generated_cumulative_logprob_prefix_{step}" for step in checkpoints)
    return metrics


def role_ids(axis: Mapping[str, Any], role: str) -> Optional[set[str]]:
    if role == "all":
        return None
    roles = axis.get("all_role_ids", axis.get("role_ids", {}))
    values = roles.get(role)
    return {str(value) for value in values} if values is not None else None


def keep_ids(
    axis: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    role: str,
    exclude_truncated: bool,
) -> List[str]:
    allowed = role_ids(axis, role)
    output = []
    for sample_id in map(str, axis["ids"]):
        if allowed is not None and sample_id not in allowed:
            continue
        row = rows.get(sample_id)
        if row is None:
            continue
        if exclude_truncated and float(row.get("style_truncated") or 0.0) != 0.0:
            continue
        output.append(sample_id)
    return output


def paired_test(values: Sequence[float], args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    clean = [value for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": math.nan, "bootstrap_ci95": [math.nan, math.nan], "signflip_two_sided_p": math.nan}
    result = paired_bootstrap_and_signflip(
        clean, args.bootstrap_samples, args.permutation_tests, seed
    )
    return {"n": len(clean), **result}


def calibration_rows(
    rows: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Compute a small calibration diagnostic when a task has correctness labels.

    This is intentionally marked proxy calibration: top-1 probability is not
    the probability of a complete free-form answer being correct.
    """
    values = []
    for sample_id in ids:
        probability = finite(rows[sample_id].get("prompt_end_max_probability"))
        correct = finite(rows[sample_id].get("correct"))
        if probability is not None and correct is not None and correct in (0.0, 1.0):
            values.append((probability, correct))
    if len(values) < 50:
        return []
    values.sort(key=lambda item: item[0])
    output = []
    bins = min(10, len(values))
    for bin_index, chunk in enumerate(
        [values[i::bins] for i in range(bins)], start=1
    ):
        if not chunk:
            continue
        mean_confidence = sum(item[0] for item in chunk) / len(chunk)
        accuracy = sum(item[1] for item in chunk) / len(chunk)
        output.append(
            {
                "analysis": "prompt_end_top1_probability_proxy_calibration",
                "bin": bin_index,
                "n": len(chunk),
                "mean_prompt_end_max_probability": mean_confidence,
                "empirical_correct_rate": accuracy,
                "absolute_calibration_gap": abs(mean_confidence - accuracy),
            }
        )
    total_n = len(values)
    ece = sum(row["n"] / total_n * row["absolute_calibration_gap"] for row in output)
    brier = sum((probability - correct) ** 2 for probability, correct in values) / total_n
    output.append(
        {
            "analysis": "summary",
            "bin": "all",
            "n": total_n,
            "ece_equal_mass_10": ece,
            "brier_proxy": brier,
            "mean_prompt_end_max_probability": sum(item[0] for item in values) / total_n,
            "empirical_correct_rate": sum(item[1] for item in values) / total_n,
        }
    )
    return output


def run_association(
    axis: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ids = keep_ids(axis, rows, args.evaluation_role, args.exclude_truncated)
    if len(ids) < MIN_ASSOCIATION_SAMPLES:
        return [], {
            "status": "insufficient_heldout_samples",
            "n_shared": len(ids),
            "minimum_n": MIN_ASSOCIATION_SAMPLES,
            "metrics": [],
            "evaluation_role": args.evaluation_role,
            "exclude_truncated": bool(args.exclude_truncated),
            "reason": (
                "The module-specific held-out intersection is below the "
                "predeclared association minimum; no association estimate "
                "was produced."
            ),
        }
    index = {str(sample_id): i for i, sample_id in enumerate(axis["ids"])}
    indices = torch.tensor([index[sample_id] for sample_id in ids], dtype=torch.long)
    semantics = torch.as_tensor(axis["semantic_controls"]).float().index_select(0, indices)
    meta = torch.as_tensor(axis["scores"]).float().index_select(0, indices)
    folds = make_fold_ids(len(ids), args.crossfit_folds, args.seed)
    result_rows: List[Dict[str, Any]] = []
    all_metrics = available_metrics(
        [rows[sample_id] for sample_id in ids], args.include_secondary
    )
    for metric_index, metric in enumerate(all_metrics):
        values = [metric_value(rows[sample_id], metric) for sample_id in ids]
        valid = [i for i, value in enumerate(values) if value is not None]
        if len(valid) < 50:
            continue
        selected = torch.tensor(valid, dtype=torch.long)
        y = torch.tensor([values[i] for i in valid], dtype=torch.float32)
        if float(y.std(unbiased=False).item()) <= 1e-8:
            continue
        semantic = semantics.index_select(0, selected)
        x_meta = meta.index_select(0, selected).view(-1, 1)
        local_folds = folds.index_select(0, selected)
        prior_pred, prior_loss = crossfit_predictive_loss(None, y, local_folds, args.ridge, "gaussian")
        sem_pred, sem_loss = crossfit_predictive_loss(semantic, y, local_folds, args.ridge, "gaussian")
        meta_pred, meta_loss = crossfit_predictive_loss(x_meta, y, local_folds, args.ridge, "gaussian")
        joint_pred, joint_loss = crossfit_predictive_loss(
            torch.cat([semantic, x_meta], dim=1), y, local_folds, args.ridge, "gaussian"
        )
        info = predictive_information_decomposition(prior_loss, sem_loss, meta_loss, joint_loss)
        conditional = info["conditional_meta_bits"]
        shapley = info["shapley_meta_bits"]
        cond_test = paired_test(conditional.tolist(), args, args.seed + 1009 * metric_index)
        shapley_test = paired_test(shapley.tolist(), args, args.seed + 2003 * metric_index)
        sem_residual = y - sem_pred
        centered_meta = x_meta.flatten() - x_meta.flatten().mean()
        centered_residual = sem_residual - sem_residual.mean()
        denominator = centered_meta.norm() * centered_residual.norm()
        partial_r = float((centered_meta @ centered_residual / denominator).item()) if float(denominator) > 1e-12 else math.nan
        result_rows.append(
            {
                "metric": metric,
                "metric_group": (
                    "primary_confidence_proxy"
                    if metric in PRIMARY_METRICS
                    else "easy_secondary_behavior"
                ),
                "direction": (
                    "uncertainty"
                    if "entropy" in metric or "hedging" in metric
                    else "confidence_or_control"
                ),
                "n": len(valid),
                "prior_nll_nats": float(prior_loss.mean().item()),
                "semantic_nll_nats": float(sem_loss.mean().item()),
                "meta_nll_nats": float(meta_loss.mean().item()),
                "joint_nll_nats": float(joint_loss.mean().item()),
                "conditional_meta_information_bits": float(conditional.mean().item()),
                "conditional_meta_ci95_low_bits": cond_test["bootstrap_ci95"][0],
                "conditional_meta_ci95_high_bits": cond_test["bootstrap_ci95"][1],
                "conditional_meta_p": cond_test.get("signflip_two_sided_p"),
                "shapley_meta_information_bits": float(shapley.mean().item()),
                "shapley_meta_ci95_low_bits": shapley_test["bootstrap_ci95"][0],
                "shapley_meta_ci95_high_bits": shapley_test["bootstrap_ci95"][1],
                "shapley_meta_p": shapley_test.get("signflip_two_sided_p"),
                "partial_r_meta_vs_semantic_residual": partial_r,
                "mse_semantic": float(((y - sem_pred) ** 2).mean().item()),
                "mse_joint": float(((y - joint_pred) ** 2).mean().item()),
                "estimand": "cross-fitted predictive information beyond external semantics; MSE is auxiliary",
            }
        )
    for key in ("conditional_meta_p", "shapley_meta_p"):
        q_values = bh_adjust([row.get(key) for row in result_rows])
        for row, q_value in zip(result_rows, q_values):
            row[key.replace("_p", "_q")] = q_value
    summary = {
        "n": len(ids),
        "evaluation_role": args.evaluation_role,
        "exclude_truncated": args.exclude_truncated,
        "metrics": [row["metric"] for row in result_rows],
        "primary_outcomes": list(PRIMARY_METRICS),
        "easy_secondary_outcomes": list(EASY_SECONDARY_METRICS)
        if args.include_secondary
        else [],
        "interpretation": "Entropy is an uncertainty proxy; negative log-probability is a confidence/fit proxy, not a calibrated belief probability.",
    }
    return result_rows, summary


def find_generation(directory: Path, control: str, dose: float) -> Optional[Path]:
    alpha = str(float(dose)).replace(".", "p")
    aliases = {"main": ("main",), "opposite": ("opposite-direction", "opposite")}[control]
    matches: List[str] = []
    for alias in aliases:
        matches.extend(glob.glob(str(directory / f"continuous_dose_0_to_1_{alias}_a{alpha}_generations.jsonl")))
    return Path(matches[0]) if len(matches) == 1 else None


def run_intervention(
    axis: Mapping[str, Any],
    trajectory_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    files = {
        "baseline": trajectory_dir / "baseline_generations.jsonl",
        "positive": find_generation(trajectory_dir, "main", args.dose),
        "negative": find_generation(trajectory_dir, "opposite", args.dose),
    }
    if any(path is None or not path.is_file() for path in files.values()):
        raise FileNotFoundError(f"Incomplete persistent trajectory files: {files}")
    mode_rows = {
        mode: {
            str(row["id"]): normalize_behavior_row(row, args.max_new_tokens)
            for row in read_jsonl(path)
        }
        for mode, path in files.items()
    }
    shared = set(mode_rows["baseline"]) & set(mode_rows["positive"]) & set(mode_rows["negative"])
    allowed = role_ids(axis, args.evaluation_role)
    if allowed is not None:
        shared &= allowed
    if args.exclude_truncated:
        shared = {
            sample_id
            for sample_id in shared
            if all(float(mode_rows[mode][sample_id].get("style_truncated") or 0.0) == 0.0 for mode in mode_rows)
        }
    if len(shared) < 30:
        raise ValueError(f"Only {len(shared)} paired persistent rows are available.")
    ids = sorted(shared)
    metrics = available_metrics(
        [mode_rows["baseline"][sample_id] for sample_id in ids],
        args.include_secondary,
    )
    rows: List[Dict[str, Any]] = []
    for metric_index, metric in enumerate(metrics):
        for left, right in (("positive", "baseline"), ("negative", "baseline"), ("positive", "negative")):
            values = [
                (
                    metric_value(mode_rows[left][sample_id], metric)
                    if metric_value(mode_rows[left][sample_id], metric) is not None
                    else math.nan
                )
                - (
                    metric_value(mode_rows[right][sample_id], metric)
                    if metric_value(mode_rows[right][sample_id], metric) is not None
                    else math.nan
                )
                for sample_id in ids
            ]
            test = paired_test(values, args, args.seed + 5003 * metric_index + len(rows))
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{left}_minus_{right}",
                    "dose": args.dose,
                    "metric_group": (
                        "primary_confidence_proxy"
                        if metric in PRIMARY_METRICS
                        else "easy_secondary_behavior"
                    ),
                    "population": "paired_persistent_untruncated" if args.exclude_truncated else "paired_persistent_all",
                    **test,
                }
            )
    for comparison in {row["comparison"] for row in rows}:
        selected = [row for row in rows if row["comparison"] == comparison]
        q_values = bh_adjust([row.get("signflip_two_sided_p") for row in selected])
        for row, q_value in zip(selected, q_values):
            row["fdr_q"] = q_value
    return rows, {"n_shared": len(ids), "metrics": metrics, "dose": args.dose}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-axis-file", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-role", default="association_confirmatory")
    parser.add_argument("--trajectory-role", default="trajectory_confirmatory")
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-tests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42017)
    parser.add_argument("--exclude-truncated", action="store_true")
    parser.add_argument(
        "--include-secondary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include low-cost monitoring/control readouts in addition to primary confidence proxies.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    axis = safe_load(args.continuous_axis_file)
    baseline_rows = {
        str(row["id"]): normalize_behavior_row(row, args.max_new_tokens)
        for row in read_jsonl(args.baseline_file)
    }
    association, association_summary = run_association(axis, baseline_rows, args)
    write_csv(args.output_dir / "metacognitive_baseline_associations.csv", association)
    baseline_ids = keep_ids(
        axis, baseline_rows, args.evaluation_role, args.exclude_truncated
    )
    calibration = calibration_rows(baseline_rows, baseline_ids)
    write_csv(args.output_dir / "metacognitive_calibration_proxy.csv", calibration)
    summary: Dict[str, Any] = {
        "baseline": association_summary,
        "calibration_proxy": {
            "available": bool(calibration),
            "interpretation": "Diagnostic only: prompt-end top-1 probability is not complete-answer probability.",
        },
        "outputs": [
            "metacognitive_baseline_associations.csv",
            "metacognitive_calibration_proxy.csv",
        ],
    }
    if association_summary.get("status") == "insufficient_heldout_samples":
        print(
            "[metacognitive-behavior] skip association: "
            f"n={association_summary['n_shared']} "
            f"< minimum={association_summary['minimum_n']}",
            flush=True,
        )
    if args.trajectory_dir:
        intervention, intervention_summary = run_intervention(axis, args.trajectory_dir, args)
        write_csv(args.output_dir / "metacognitive_intervention_effects.csv", intervention)
        summary["intervention"] = intervention_summary
        summary["outputs"].append("metacognitive_intervention_effects.csv")
    (args.output_dir / "metacognitive_behavior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
