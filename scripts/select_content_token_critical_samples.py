#!/usr/bin/env python3
"""Select pre-treatment low-confidence generations and matched controls.

The criticality threshold is calibrated on one baseline-only split and then
applied unchanged to an independent evaluation split.  Only generated token
probabilities observed before intervention are used for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


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


def parse_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else []


def finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def is_content_piece(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"<think>", "</think>", "<assistant>", "assistant"}:
        return False
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in text)


def load_axis_scores(path: str) -> Dict[str, float]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return {
        str(sample_id): float(score)
        for sample_id, score in zip(payload["ids"], payload["scores"])
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    axis_scores: Mapping[str, float],
    window: int,
    surprisal_top_k: int,
    max_generated_token_index: int,
) -> List[Dict[str, Any]]:
    special_ids = set(tokenizer.all_special_ids)
    output: List[Dict[str, Any]] = []
    for source in rows:
        sample_id = str(source["id"])
        token_ids = [int(value) for value in parse_list(source.get("generated_token_ids_json"))]
        logprobs = [finite(value) for value in parse_list(source.get("generated_token_logprobs_json"))]
        count = min(len(token_ids), len(logprobs))
        decisions: List[Tuple[int, int, str, float]] = []
        for index, (token_id, logprob) in enumerate(zip(token_ids[:count], logprobs[:count])):
            if max_generated_token_index >= 0 and index > max_generated_token_index:
                break
            if token_id in special_ids or not math.isfinite(logprob):
                continue
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            if not is_content_piece(piece):
                continue
            decisions.append((index, token_id, piece, max(0.0, -logprob)))
            if len(decisions) >= window:
                break
        if not decisions:
            continue
        ranked = sorted(decisions, key=lambda item: item[3], reverse=True)
        selected = ranked[: max(1, min(surprisal_top_k, len(ranked)))]
        critical_index, critical_token_id, critical_text, max_surprisal = ranked[0]
        critical_score = float(np.mean([item[3] for item in selected]))
        output.append(
            {
                "id": sample_id,
                "critical_score": critical_score,
                "early_topk_surprisal_mean": critical_score,
                "early_max_surprisal": max_surprisal,
                "early_min_max_probability": math.exp(-max_surprisal),
                "first_content_surprisal": decisions[0][3],
                "first_content_index": decisions[0][0],
                "critical_token_index": critical_index,
                "critical_token_id": critical_token_id,
                "critical_token_text_json": json.dumps(critical_text, ensure_ascii=False),
                "content_token_count": len(decisions),
                "axis_score": finite(axis_scores.get(sample_id)),
                "prompt_tokens": finite(source.get("prompt_tokens")),
                "baseline_generated_tokens": finite(source.get("generated_tokens")),
            }
        )
    return output


def robust_scale(values: Sequence[float]) -> float:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not clean.size:
        return 1.0
    scale = float(np.quantile(clean, 0.75) - np.quantile(clean, 0.25))
    return scale if scale > 1e-8 else max(float(clean.std()), 1.0)


def match_controls(
    critical: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    features = ("axis_score", "prompt_tokens", "first_content_index")
    scales = {
        key: robust_scale([finite(row.get(key)) for row in [*critical, *candidates]])
        for key in features
    }
    available = {str(row["id"]): row for row in candidates}
    pairs: List[Dict[str, Any]] = []
    controls: List[Dict[str, Any]] = []
    for pair_id, target in enumerate(sorted(critical, key=lambda row: float(row["critical_score"]), reverse=True)):
        best_id = None
        best_distance = math.inf
        for candidate_id, candidate in available.items():
            distance = 0.0
            observed = 0
            for key in features:
                left = finite(target.get(key))
                right = finite(candidate.get(key))
                if math.isfinite(left) and math.isfinite(right):
                    distance += ((left - right) / scales[key]) ** 2
                    observed += 1
            if observed and distance < best_distance:
                best_id = candidate_id
                best_distance = distance
        if best_id is None:
            break
        control = dict(available.pop(best_id))
        control["group"] = "matched_high_confidence"
        control["match_pair_id"] = pair_id
        controls.append(control)
        pairs.append(
            {
                "match_pair_id": pair_id,
                "critical_id": target["id"],
                "control_id": control["id"],
                "match_distance": best_distance,
                "critical_score": target["critical_score"],
                "control_score": control["critical_score"],
                "critical_axis_score": target.get("axis_score"),
                "control_axis_score": control.get("axis_score"),
                "critical_prompt_tokens": target.get("prompt_tokens"),
                "control_prompt_tokens": control.get("prompt_tokens"),
            }
        )
    return controls, pairs


def write_ids(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(f"{row['id']}\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-baseline", required=True)
    parser.add_argument("--evaluation-baseline", required=True)
    parser.add_argument("--continuous-axis-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--critical-fraction", type=float, default=0.20)
    parser.add_argument("--content-window", type=int, default=32)
    parser.add_argument("--surprisal-top-k", type=int, default=3)
    parser.add_argument(
        "--max-generated-token-index",
        type=int,
        default=-1,
        help="Only define criticality from tokens at or before this zero-based index.",
    )
    parser.add_argument("--control-max-quantile", type=float, default=0.50)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--min-pairs", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.critical_fraction < 0.5:
        raise ValueError("--critical-fraction must be between 0 and 0.5")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    axis_scores = load_axis_scores(args.continuous_axis_file)
    calibration = summarize_rows(
        read_jsonl(args.calibration_baseline), tokenizer, axis_scores,
        args.content_window, args.surprisal_top_k, args.max_generated_token_index,
    )
    evaluation = summarize_rows(
        read_jsonl(args.evaluation_baseline), tokenizer, axis_scores,
        args.content_window, args.surprisal_top_k, args.max_generated_token_index,
    )
    if not calibration or not evaluation:
        raise ValueError("No usable baseline rows with saved token ids and log-probabilities.")
    calibration_scores = np.asarray([row["critical_score"] for row in calibration])
    threshold = float(np.quantile(calibration_scores, 1.0 - args.critical_fraction))
    control_ceiling = float(np.quantile(calibration_scores, args.control_max_quantile))
    for row in calibration:
        row["critical_preintervention"] = row["critical_score"] >= threshold
    for row in evaluation:
        row["critical_preintervention"] = row["critical_score"] >= threshold
        row["control_candidate"] = row["critical_score"] <= control_ceiling
    critical = [dict(row) for row in evaluation if row["critical_preintervention"]]
    critical.sort(key=lambda row: row["critical_score"], reverse=True)
    critical = critical[: args.max_pairs]
    for pair_id, row in enumerate(critical):
        row["group"] = "precritical_confirmatory"
        row["match_pair_id"] = pair_id
    controls, pairs = match_controls(
        critical,
        [row for row in evaluation if row["control_candidate"]],
    )
    usable = min(len(critical), len(controls))
    critical = critical[:usable]
    controls = controls[:usable]
    pairs = pairs[:usable]
    if usable < args.min_pairs:
        raise ValueError(
            f"Only {usable} matched pairs were selected; require at least {args.min_pairs}."
        )
    write_csv(os.path.join(args.output_dir, "calibration_samples.csv"), calibration)
    write_csv(os.path.join(args.output_dir, "evaluation_samples.csv"), evaluation)
    write_csv(os.path.join(args.output_dir, "matched_pairs.csv"), pairs)
    write_csv(os.path.join(args.output_dir, "selected_critical_samples.csv"), critical)
    write_csv(os.path.join(args.output_dir, "selected_control_samples.csv"), controls)
    write_ids(os.path.join(args.output_dir, "critical_ids.txt"), critical)
    write_ids(os.path.join(args.output_dir, "matched_control_ids.txt"), controls)
    write_ids(os.path.join(args.output_dir, "trajectory_union_ids.txt"), [*critical, *controls])
    summary = {
        "selection_is_preintervention": True,
        "threshold_frozen_on_calibration_split": True,
        "criticality": "mean of the largest token surprisals among the first content tokens",
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "critical_fraction_requested": args.critical_fraction,
        "critical_threshold": threshold,
        "control_score_ceiling": control_ceiling,
        "evaluation_above_frozen_threshold_n": sum(row["critical_preintervention"] for row in evaluation),
        "matched_pair_n": usable,
        "content_window": args.content_window,
        "surprisal_top_k": args.surprisal_top_k,
        "max_generated_token_index": args.max_generated_token_index,
        "critical_score_mean": float(np.mean([row["critical_score"] for row in critical])),
        "control_score_mean": float(np.mean([row["critical_score"] for row in controls])),
        "critical_min_probability_median": float(np.median([row["early_min_max_probability"] for row in critical])),
        "control_min_probability_median": float(np.median([row["early_min_max_probability"] for row in controls])),
        "matching_variables": ["continuous_axis_score", "prompt_tokens", "first_content_index"],
    }
    with open(os.path.join(args.output_dir, "critical_selection_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
