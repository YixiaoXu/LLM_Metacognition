#!/usr/bin/env python3
"""Reanalyse cached MathQA generations after removing truncated rows.

This utility intentionally uses only JSONL/CSV artifacts, so it can be run in
the lightweight analysis environment.  It evaluates the already cross-fitted
confidence predictions on the untruncated subset; it does not refit the
semantic/meta calibrators.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(
    "runs/mathqa_proxy_confidence_qwen3_4b_20260822_202705"
) / "qwen3_4b__semantic_llama31_8b_l16" / "continuous_modules"
RANKS = ["rank_1_k128_m2", "rank_2_k128_m1", "rank_3_k128_m0", "rank_4_k128_m4"]
EPS = 1e-7


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values: Iterable[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else None


def parse_json_list(value: Any) -> List[float]:
    try:
        result = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [float(x) for x in result if finite(x) is not None]


def parse_prefix(value: Any) -> Dict[int, float]:
    try:
        result = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {int(k): float(v) for k, v in result.items() if finite(v) is not None}


def bootstrap_ci(values: Sequence[float], seed: int, n_boot: int = 2000) -> List[float | None]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return [None, None]
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        boot.append(sum(rng.choice(values) for _ in values) / len(values))
    boot.sort()
    return [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]


def bootstrap_difference_ci(
    group_a: Sequence[float], group_b: Sequence[float], seed: int, n_boot: int = 2000
) -> List[float | None]:
    a = [x for x in group_a if math.isfinite(x)]
    b = [x for x in group_b if math.isfinite(x)]
    if not a or not b:
        return [None, None]
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        mean_a = sum(rng.choice(a) for _ in a) / len(a)
        mean_b = sum(rng.choice(b) for _ in b) / len(b)
        boot.append(mean_a - mean_b)
    boot.sort()
    return [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]


def permutation_p(group_a: Sequence[float], group_b: Sequence[float], seed: int, n_perm: int = 5000) -> float | None:
    a = [x for x in group_a if math.isfinite(x)]
    b = [x for x in group_b if math.isfinite(x)]
    if not a or not b:
        return None
    observed = abs(sum(a) / len(a) - sum(b) / len(b))
    pooled = a + b
    n_a = len(a)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        delta = abs(sum(pooled[:n_a]) / n_a - sum(pooled[n_a:]) / len(b))
        exceed += delta >= observed
    return (exceed + 1) / (n_perm + 1)


def auroc(target: Sequence[int], score: Sequence[float]) -> float | None:
    pairs = sorted(zip(score, target), key=lambda pair: pair[0])
    pos = sum(target)
    neg = len(target) - pos
    if not pos or not neg:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        rank_sum += rank * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def confidence_restricted(
    rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
) -> Dict[str, Any]:
    keep = {str(row["id"]) for row in rows}
    pred = [row for row in prediction_rows if str(row["id"]) in keep]
    target = [int(float(row["correct"])) for row in pred]
    result: Dict[str, Any] = {"n": len(pred)}
    for name in ("proxy_calibrated_correctness", "semantic_proxy_correctness", "semantic_proxy_meta_correctness"):
        values = [float(row[name]) for row in pred]
        losses = [-(y * math.log(max(p, EPS)) + (1 - y) * math.log(max(1 - p, EPS))) for y, p in zip(target, values)]
        result[name] = {
            "nll_nats": mean(losses),
            "brier": mean([(p - y) ** 2 for p, y in zip(values, target)]),
            "auroc": auroc(target, values),
        }
    base = [
        -(y * math.log(max(float(row["semantic_proxy_correctness"]), EPS))
          + (1 - y) * math.log(max(1 - float(row["semantic_proxy_correctness"]), EPS)))
        for row, y in zip(pred, target)
    ]
    joint = [
        -(y * math.log(max(float(row["semantic_proxy_meta_correctness"]), EPS))
          + (1 - y) * math.log(max(1 - float(row["semantic_proxy_meta_correctness"]), EPS)))
        for row, y in zip(pred, target)
    ]
    delta_bits = [(b - j) / math.log(2.0) for b, j in zip(base, joint)]
    result["restricted_existing_crossfit_conditional_meta_bits"] = {
        "mean": mean(delta_bits),
        "bootstrap_ci95": bootstrap_ci(delta_bits, 1301, bootstrap_samples),
        "note": "evaluation of cached cross-fitted predictions; not refitted after filtering",
    }
    return result


def row_metrics(row: Mapping[str, Any]) -> Dict[str, float | None]:
    logprobs = parse_json_list(row.get("generated_token_logprobs_json"))
    prefix = parse_prefix(row.get("generated_cumulative_logprob_prefix_json"))
    return {
        "correct": float(bool(row.get("correct"))),
        "prompt_end_entropy": finite(row.get("prompt_end_entropy")),
        "prompt_end_top1_top2_logit_margin": finite(row.get("prompt_end_top1_top2_logit_margin")),
        "generated_prefix16_entropy_mean": finite(row.get("generated_prefix16_entropy_mean")),
        "generated_prefix16_logit_margin_mean": finite(row.get("generated_prefix16_logit_margin_mean")),
        "generated_mean_logprob": finite(row.get("generated_mean_logprob")),
        "generated_min_logprob": finite(row.get("generated_min_logprob")),
        "generated_cumulative_logprob": finite(row.get("generated_cumulative_logprob")),
        "generated_tokens": finite(row.get("generated_tokens")),
        "self_correction_marker_count": finite(row.get("style_self_correction_marker_count")),
        "self_correction_present": float((finite(row.get("style_self_correction_marker_count")) or 0.0) > 0),
        "parse_success": float(bool(row.get("parse_success"))),
        "prefix_logprob_1": prefix.get(1),
        "prefix_logprob_4": prefix.get(4),
        "prefix_logprob_8": prefix.get(8),
        "prefix_logprob_16": prefix.get(16),
        "prefix_logprob_32": prefix.get(32),
        "prefix_logprob_64": prefix.get(64),
        "prefix_logprob_128": prefix.get(128),
        "prefix_logprob_256": prefix.get(256),
        "prefix_logprob_512": prefix.get(512),
        "generated_token_count_from_logprobs": float(len(logprobs)),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    permutation_tests: int,
) -> Dict[str, Any]:
    cached = [row_metrics(row) for row in rows]
    metrics = sorted(cached[0]) if cached else []
    output: Dict[str, Any] = {"n": len(rows), "accuracy": mean([float(bool(r.get("correct"))) for r in rows])}
    for metric in metrics:
        values = [metrics_row[metric] for metrics_row in cached]
        values = [x for x in values if x is not None]
        correct = [
            metrics_row[metric]
            for row, metrics_row in zip(rows, cached)
            if bool(row.get("correct")) and metrics_row[metric] is not None
        ]
        incorrect = [
            metrics_row[metric]
            for row, metrics_row in zip(rows, cached)
            if not bool(row.get("correct")) and metrics_row[metric] is not None
        ]
        if not values:
            continue
        delta = (mean(correct) - mean(incorrect)) if correct and incorrect else None
        output[metric] = {
            "n": len(values),
            "mean": mean(values),
            "correct_mean": mean(correct),
            "incorrect_mean": mean(incorrect),
            "correct_minus_incorrect": delta,
            "correct_minus_incorrect_bootstrap_ci95": bootstrap_difference_ci(
                correct, incorrect, 2000, bootstrap_samples
            ),
            "correct_vs_incorrect_permutation_p": permutation_p(
                correct, incorrect, 3000, permutation_tests
            ) if correct and incorrect else None,
        }
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ranks", nargs="+", default=RANKS)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-tests", type=int, default=5000)
    args = parser.parse_args()
    root = args.root
    output_root = root / "proxy_confidence_untruncated"
    output_root.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    for rank in args.ranks:
        rank_dir = root / rank
        generation_path = rank_dir / "baseline_association_confirmatory" / "baseline_generations.jsonl"
        prediction_path = rank_dir / "proxy_confidence" / "proxy_confidence_predictions.csv"
        rows = read_jsonl(generation_path)
        untruncated = [row for row in rows if float(row.get("style_truncated") or 0.0) == 0.0]
        predictions = read_csv(prediction_path)
        report = {
            "rank": rank,
            "generation_rows": len(rows),
            "truncated_rows": len(rows) - len(untruncated),
            "untruncated_rows": len(untruncated),
            "truncated_rate": (len(rows) - len(untruncated)) / max(len(rows), 1),
            "behavior": summarize(
                untruncated, args.bootstrap_samples, args.permutation_tests
            ),
            "confidence": confidence_restricted(
                untruncated, predictions, args.bootstrap_samples
            ),
            "limitations": {
                "answer_token_probability_or_margin": "not recorded in baseline_generations.jsonl",
                "per_step_entropy": "not recorded; only prefix-16 entropy aggregate is available",
                "answer_revision": "no explicit revision event field; self-correction marker count is used as a proxy",
            },
        }
        all_reports[rank] = report
        (output_root / f"{rank}_untruncated_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[untruncated] {rank} n={len(untruncated)} "
            f"truncated={len(rows)-len(untruncated)} "
            f"accuracy={report['behavior']['accuracy']:.4f}"
        )
    (output_root / "untruncated_proxy_trajectory_summary.json").write_text(
        json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[untruncated] wrote {output_root}")


if __name__ == "__main__":
    main()
