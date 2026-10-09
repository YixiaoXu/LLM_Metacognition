#!/usr/bin/env python3
"""Find a deterministic baseline compatible with the current generation contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CONFIDENCE_FIELDS = {
    "prompt_end_entropy",
    "prompt_end_top1_top2_logit_margin",
    "generated_mean_logprob",
    "generated_prefix16_negative_entropy_mean",
}


def normalized_path(value: str) -> str:
    path = Path(value).expanduser()
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def first_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return {}


def candidate_baseline(config_path: Path, config: dict) -> Path | None:
    configured = config.get("baseline_file")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(config_path.parent / "baseline_generations.jsonl")
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", default="runs")
    parser.add_argument("--exclude-root", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-profile", required=True)
    parser.add_argument("--metric-profile", required=True)
    parser.add_argument("--answer-extraction", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--require-confidence", action="store_true")
    args = parser.parse_args()

    search_root = Path(args.search_root)
    excluded = normalized_path(args.exclude_root) if args.exclude_root else ""
    expected_model = normalized_path(args.model_path)
    matches: list[tuple[float, Path]] = []
    for config_path in search_root.glob("**/baseline_*/run_config.json"):
        if excluded and normalized_path(str(config_path)).startswith(excluded):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not config.get("baseline_only", False):
            continue
        if normalized_path(str(config.get("model_path", ""))) != expected_model:
            continue
        if config.get("metric_profile") != args.metric_profile:
            continue
        if config.get("answer_extraction") != args.answer_extraction:
            continue
        if int(config.get("max_new_tokens", -1)) != args.max_new_tokens:
            continue
        if int(config.get("seed", -1)) != args.seed:
            continue
        if float(config.get("temperature", -1.0)) != 0.0:
            continue
        data_path = str(config.get("data", "")).lower()
        if args.dataset_profile.lower() not in data_path:
            continue
        baseline = candidate_baseline(config_path, config)
        if baseline is None:
            continue
        try:
            row = first_row(baseline)
        except (OSError, json.JSONDecodeError):
            continue
        if args.require_confidence and not CONFIDENCE_FIELDS.issubset(row):
            continue
        matches.append((baseline.stat().st_mtime, baseline))

    if not matches:
        print("[baseline-search] no compatible historical baseline", file=sys.stderr)
        raise SystemExit(1)
    matches.sort(reverse=True)
    selected = matches[0][1]
    print(f"[baseline-search] selected {selected}", file=sys.stderr)
    print(selected)


if __name__ == "__main__":
    main()
