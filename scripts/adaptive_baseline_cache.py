#!/usr/bin/env python3
"""Plan and merge an adaptive deterministic-generation baseline.

The planner keeps the frozen assignment order, reuses only compatible,
untruncated historical rows, and emits an allowlist for rows that still need
generation.  The merger adds newly generated rows to the persistent state and
reports the number of usable (untruncated) examples.  This lets the shell
driver top up only when the initial baseline is underpowered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path, tolerate_trailing_partial: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file() or path.stat().st_size == 0:
        return rows
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_trailing_partial and index == len(lines) - 1:
                print(f"[baseline-cache] ignore incomplete trailing row in {path}", file=sys.stderr)
                break
            raise
        if isinstance(value, dict):
            rows.append(value)
    return rows


def assignment_ids(path: Path) -> List[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        ids = [str(row.get("id", "")).strip() for row in csv.DictReader(handle)]
    ids = [sample_id for sample_id in ids if sample_id]
    if not ids:
        raise ValueError(f"No ids in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {path}")
    return ids


def model_identity(value: Any) -> tuple[str, str]:
    text = str(value or "").strip().rstrip("/")
    path = Path(text).expanduser()
    if text and path.exists():
        try:
            text = str(path.resolve())
        except OSError:
            pass
    return text.lower(), Path(text).name.lower()


def same_model(left: Any, right: Any) -> bool:
    left_full, left_name = model_identity(left)
    right_full, right_name = model_identity(right)
    return bool(left_full and right_full and (left_full == right_full or left_name == right_name))


def load_config_for_baseline(path: Path) -> Dict[str, Any] | None:
    config_path = path.parent / "run_config.json"
    if not config_path.is_file():
        return None
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def config_is_compatible(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    if not same_model(config.get("model_path"), args.model_path):
        return False
    if str(config.get("metric_profile", "")) != args.metric_profile:
        return False
    if str(config.get("answer_extraction", "")) != args.answer_extraction:
        return False
    if str(config.get("prompt_style", "data")) != args.prompt_style:
        return False
    if bool(config.get("use_chat_template", False)) != bool(args.use_chat_template):
        return False
    if str(config.get("chat_template_enable_thinking", "auto")) != args.thinking:
        return False
    if str(config.get("system_prompt", "")) != args.system_prompt:
        return False
    if float(config.get("temperature", -1.0)) != 0.0:
        return False
    if int(config.get("max_length", -1)) != int(args.max_length):
        return False
    data_value = str(config.get("data", "")).lower()
    if args.data_marker and args.data_marker.lower() not in data_value:
        return False
    return True


def row_is_reusable(
    row: Dict[str, Any], required_fields: Sequence[str], max_new_tokens: int
) -> bool:
    sample_id = str(row.get("id", "")).strip()
    if not sample_id or any(field not in row for field in required_fields):
        return False
    try:
        truncated = float(row.get("style_truncated", 1.0)) > 0.0
        generated_tokens = int(float(row.get("generated_tokens", max_new_tokens)))
    except (TypeError, ValueError):
        return False
    # An older, shorter generation contract is reusable only when generation
    # naturally terminated before both its own cap and the current cap.
    return not truncated and generated_tokens < int(max_new_tokens)


def historical_candidates(search_roots: Sequence[str], exclude_root: str) -> Iterator[Path]:
    excluded = ""
    if exclude_root:
        try:
            excluded = str(Path(exclude_root).resolve())
        except OSError:
            excluded = str(Path(exclude_root))
    candidates: List[Path] = []
    for raw_root in search_roots:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        for path in root.glob("**/baseline_generations.jsonl"):
            try:
                resolved = str(path.resolve())
            except OSError:
                resolved = str(path)
            if excluded and resolved.startswith(excluded):
                continue
            try:
                if path.stat().st_size > 0:
                    candidates.append(path)
            except OSError:
                continue
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    yield from candidates


def valid_count(rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        try:
            count += int(float(row.get("style_truncated", 1.0)) <= 0.0)
        except (TypeError, ValueError):
            continue
    return count


def load_state(path: Path, required_fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path, tolerate_trailing_partial=True):
        sample_id = str(row.get("id", "")).strip()
        if sample_id and all(field in row for field in required_fields):
            by_id[sample_id] = row
    return by_id


def plan(args: argparse.Namespace) -> None:
    all_ids = assignment_ids(Path(args.assignments))
    limit = min(max(int(args.limit), 1), len(all_ids))
    selected_ids = all_ids[:limit]
    selected_set = set(selected_ids)
    state_path = Path(args.state_file)
    by_id = load_state(state_path, args.required_field)
    by_id = {sample_id: row for sample_id, row in by_id.items() if sample_id in selected_set}
    provenance: Dict[str, str] = {sample_id: "adaptive_state" for sample_id in by_id}

    history_files_scanned = 0
    history_rows_reused = 0
    missing_set = selected_set - set(by_id)
    for candidate in historical_candidates(args.search_root, args.exclude_root):
        if not missing_set:
            break
        config = load_config_for_baseline(candidate)
        if config is None or not config_is_compatible(config, args):
            continue
        history_files_scanned += 1
        try:
            rows = read_jsonl(candidate, tolerate_trailing_partial=True)
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            sample_id = str(row.get("id", "")).strip()
            if sample_id not in missing_set:
                continue
            if not row_is_reusable(row, args.required_field, args.max_new_tokens):
                continue
            by_id[sample_id] = row
            provenance[sample_id] = str(candidate)
            missing_set.remove(sample_id)
            history_rows_reused += 1

    cached_rows = [by_id[sample_id] for sample_id in selected_ids if sample_id in by_id]
    missing_ids = [sample_id for sample_id in selected_ids if sample_id not in by_id]
    atomic_write_jsonl(state_path, cached_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in selected_ids), encoding="utf-8"
    )
    (output_dir / "missing_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in missing_ids), encoding="utf-8"
    )
    with (output_dir / "missing_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id"])
        writer.writeheader()
        writer.writerows({"id": sample_id} for sample_id in missing_ids)
    plan_hash = hashlib.sha256(
        json.dumps(
            {
                "selected_ids": selected_ids,
                "missing_ids": missing_ids,
                "model": args.model_path,
                "max_new_tokens": args.max_new_tokens,
                "max_length": args.max_length,
                "metric_profile": args.metric_profile,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        "schema_version": 1,
        "limit": limit,
        "pool_size": len(all_ids),
        "cached_rows": len(cached_rows),
        "cached_untruncated_rows": valid_count(cached_rows),
        "missing_rows": len(missing_ids),
        "history_files_scanned": history_files_scanned,
        "history_rows_reused_this_round": history_rows_reused,
        "plan_sha256": plan_hash,
        "state_file": str(state_path),
        "selected_ids_file": str(output_dir / "selected_ids.txt"),
        "missing_ids_file": str(output_dir / "missing_ids.txt"),
        "missing_assignments": str(output_dir / "missing_assignments.csv"),
    }
    atomic_write_json(output_dir / "plan_summary.json", summary)
    atomic_write_json(
        output_dir / "history_provenance.json",
        {sample_id: provenance[sample_id] for sample_id in selected_ids if sample_id in provenance},
    )
    print(json.dumps(summary, ensure_ascii=False))


def merge(args: argparse.Namespace) -> None:
    selected_ids = [
        line.strip()
        for line in Path(args.selected_ids).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_set = set(selected_ids)
    if not selected_ids or len(selected_ids) != len(selected_set):
        raise ValueError(f"Invalid selected-id file: {args.selected_ids}")
    state_path = Path(args.state_file)
    by_id = load_state(state_path, args.required_field)
    generated_rows = 0
    for raw_path in args.generated_file:
        path = Path(raw_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing generated baseline file: {path}")
        for row in read_jsonl(path, tolerate_trailing_partial=True):
            sample_id = str(row.get("id", "")).strip()
            if sample_id not in selected_set:
                continue
            missing_fields = [field for field in args.required_field if field not in row]
            if missing_fields:
                raise ValueError(f"Generated row {sample_id} misses {missing_fields}: {path}")
            by_id[sample_id] = row
            generated_rows += 1
    missing = [sample_id for sample_id in selected_ids if sample_id not in by_id]
    if missing:
        raise ValueError(
            f"Adaptive baseline coverage mismatch: missing={len(missing)} examples={missing[:5]}"
        )
    rows = [by_id[sample_id] for sample_id in selected_ids]
    atomic_write_jsonl(state_path, rows)
    if args.output:
        atomic_write_jsonl(Path(args.output), rows)
    summary = {
        "schema_version": 1,
        "selected_rows": len(rows),
        "untruncated_rows": valid_count(rows),
        "truncated_rows": len(rows) - valid_count(rows),
        "generated_rows_merged": generated_rows,
        "state_file": str(state_path),
        "output": args.output or "",
    }
    if args.summary:
        atomic_write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False))


def add_contract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metric-profile", required=True)
    parser.add_argument("--answer-extraction", required=True)
    parser.add_argument("--data-marker", default="")
    parser.add_argument("--prompt-style", default="data")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--thinking", default="auto")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--required-field", action="append", default=[])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--assignments", required=True)
    plan_parser.add_argument("--limit", type=int, required=True)
    plan_parser.add_argument("--state-file", required=True)
    plan_parser.add_argument("--output-dir", required=True)
    plan_parser.add_argument("--search-root", action="append", default=[])
    plan_parser.add_argument("--exclude-root", default="")
    add_contract_args(plan_parser)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--selected-ids", required=True)
    merge_parser.add_argument("--state-file", required=True)
    merge_parser.add_argument("--generated-file", action="append", default=[])
    merge_parser.add_argument("--output", default="")
    merge_parser.add_argument("--summary", default="")
    merge_parser.add_argument("--required-field", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.required_field:
        raise ValueError("At least one --required-field is required")
    if args.command == "plan":
        if not args.search_root:
            args.search_root = ["runs"]
        plan(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
