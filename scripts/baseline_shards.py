#!/usr/bin/env python3
"""Split a frozen baseline population and merge independently generated shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_assignment_ids(path: Path) -> List[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        ids = [str(row.get("id", "")).strip() for row in csv.DictReader(handle)]
    ids = [sample_id for sample_id in ids if sample_id]
    if not ids:
        raise ValueError(f"No ids found in baseline assignment file: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in baseline assignment file: {path}")
    return ids


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


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


def split(args: argparse.Namespace) -> None:
    assignments = Path(args.assignments)
    output_dir = Path(args.output_dir)
    ids = read_assignment_ids(assignments)
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.num_shards > len(ids):
        raise ValueError("--num-shards cannot exceed the number of baseline ids")

    contract = sorted(str(value) for value in args.contract)
    digest_payload = json.dumps(
        {"ids": ids, "contract": contract},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_entries = []
    for shard_index in range(args.num_shards):
        shard_ids = ids[shard_index :: args.num_shards]
        shard_path = output_dir / f"shard_{shard_index:02d}.txt"
        shard_path.write_text("".join(f"{sample_id}\n" for sample_id in shard_ids), encoding="utf-8")
        shard_entries.append(
            {
                "index": shard_index,
                "path": str(shard_path),
                "count": len(shard_ids),
            }
        )

    manifest = {
        "schema_version": 1,
        "assignments": str(assignments),
        "plan_sha256": digest,
        "total_ids": len(ids),
        "num_shards": args.num_shards,
        "generation_contract": contract,
        "ordered_ids": ids,
        "shards": shard_entries,
    }
    manifest_path = output_dir / "baseline_shard_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "plan_sha256": digest,
                "total_ids": len(ids),
                "shard_counts": [entry["count"] for entry in shard_entries],
            },
            ensure_ascii=False,
        )
    )


def merge(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = [str(value) for value in manifest.get("ordered_ids", [])]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"Invalid ordered ids in shard manifest: {manifest_path}")
    if len(args.shard_file) != int(manifest.get("num_shards", -1)):
        raise ValueError(
            f"Expected {manifest.get('num_shards')} shard files, got {len(args.shard_file)}"
        )

    by_id: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, str] = {}
    for raw_path in args.shard_file:
        path = Path(raw_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty baseline shard: {path}")
        for row in read_jsonl(path):
            sample_id = str(row.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"Baseline row has no id: {path}")
            if sample_id in by_id:
                raise ValueError(
                    f"Duplicate baseline id {sample_id!r} in {sources[sample_id]} and {path}"
                )
            missing_fields = [field for field in args.required_field if field not in row]
            if missing_fields:
                raise ValueError(
                    f"Baseline row {sample_id!r} in {path} misses fields: {missing_fields}"
                )
            by_id[sample_id] = row
            sources[sample_id] = str(path)

    expected_set = set(expected_ids)
    missing = [sample_id for sample_id in expected_ids if sample_id not in by_id]
    extra = sorted(set(by_id) - expected_set)
    if missing or extra:
        raise ValueError(
            f"Baseline shard coverage mismatch: missing={len(missing)} examples={missing[:5]}, "
            f"extra={len(extra)} examples={extra[:5]}"
        )

    ordered_rows = [by_id[sample_id] for sample_id in expected_ids]
    output = Path(args.output)
    atomic_write_jsonl(output, ordered_rows)
    summary = {
        "schema_version": 1,
        "output": str(output),
        "manifest": str(manifest_path),
        "plan_sha256": manifest.get("plan_sha256"),
        "rows": len(ordered_rows),
        "shard_files": [str(Path(value)) for value in args.shard_file],
        "required_fields": list(args.required_field),
    }
    atomic_write_json(output.with_suffix(".merge.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--assignments", required=True)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--num-shards", type=int, required=True)
    split_parser.add_argument(
        "--contract",
        action="append",
        default=[],
        help="Generation-contract field included in the shard cache key.",
    )

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--manifest", required=True)
    merge_parser.add_argument("--shard-file", action="append", default=[], required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.add_argument("--required-field", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "split":
        split(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
