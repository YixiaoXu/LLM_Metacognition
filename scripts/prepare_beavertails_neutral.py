#!/usr/bin/env python
"""Prepare BeaverTails prompts without adding task or safety instructions.

The output ``prompt`` is exactly the stripped source prompt.  Role markers are
added later by the target model's own chat template, so this file does not
encode a requested refusal policy, answer format, or response length.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from tqdm import tqdm


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "safe"}:
        return True
    if text in {"false", "0", "no", "unsafe"}:
        return False
    raise ValueError(f"Cannot parse is_safe={value!r} as boolean.")


def active_categories(category: Any) -> List[str]:
    if not isinstance(category, dict):
        return []
    return sorted(str(name) for name, active in category.items() if bool(active))


def normalized_prompt(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def source_key(path: str) -> str:
    stem = Path(path).stem.replace(" ", "_")
    digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


SUPPORTED_SUFFIXES = {".jsonl", ".json", ".parquet"}


def expand_sources(paths: Sequence[str]) -> List[str]:
    """Resolve explicit files, directories, and shell-style globs deterministically."""
    resolved: List[str] = []
    for value in paths:
        matches = sorted(glob.glob(value, recursive=True))
        if not matches and os.path.exists(value):
            matches = [value]
        for match in matches:
            if os.path.isdir(match):
                for root, _, names in os.walk(match):
                    for name in sorted(names):
                        path = os.path.join(root, name)
                        if Path(path).suffix.lower() in SUPPORTED_SUFFIXES:
                            resolved.append(os.path.abspath(path))
            elif Path(match).suffix.lower() in SUPPORTED_SUFFIXES:
                resolved.append(os.path.abspath(match))
    return list(dict.fromkeys(resolved))


def iter_source_records(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                line = line.strip()
                if line:
                    yield row_number, json.loads(line)
        return
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            for key in ("data", "train", "rows", "records"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError(f"JSON source must contain a list of records: {path}")
        for row_number, item in enumerate(value, start=1):
            if isinstance(item, dict):
                yield row_number, item
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Reading BeaverTails parquet files requires pyarrow. "
                "Install it in the experiment environment with `pip install pyarrow`."
            ) from exc
        row_number = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=8192):
            for item in batch.to_pylist():
                row_number += 1
                yield row_number, item
        return
    raise ValueError(f"Unsupported BeaverTails source format: {path}")


def infer_split(path: str, item: Dict[str, Any]) -> str:
    explicit = str(item.get("split") or item.get("source_split") or "").strip().lower()
    if explicit:
        return explicit
    lowered = path.lower()
    if "test" in lowered:
        return "test"
    if "validation" in lowered or "valid" in lowered or "/val" in lowered:
        return "validation"
    if "train" in lowered:
        return "train"
    return "unknown"


def read_sources(paths: Sequence[str]) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    resolved_paths = expand_sources(paths)
    if not resolved_paths:
        raise FileNotFoundError(
            "No supported BeaverTails files were found. Inputs may be JSONL, JSON, "
            "Parquet, directories, or glob patterns."
        )
    rows: List[Dict[str, Any]] = []
    duplicate_count = 0
    seen = set()
    for path in resolved_paths:
        key = source_key(path)
        iterator = iter_source_records(path)
        for line_number, item in tqdm(iterator, desc=f"Read {Path(path).name}"):
            prompt = str(item.get("prompt") or "").strip()
            if not prompt:
                continue
            dedup_key = normalized_prompt(prompt)
            if dedup_key in seen:
                duplicate_count += 1
                continue
            seen.add(dedup_key)
            rows.append(
                {
                    "source_path": os.path.abspath(path),
                    "source_key": key,
                    "source_line": line_number,
                    "source_split": infer_split(path, item),
                    "prompt": prompt,
                    "response": str(item.get("response") or ""),
                    "is_safe": parse_bool(item.get("is_safe")),
                    "category": item.get("category") if isinstance(item.get("category"), dict) else {},
                }
            )
    return rows, duplicate_count, resolved_paths


def select_rows(rows: Sequence[Dict[str, Any]], max_samples: int, mode: str, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)
    if max_samples <= 0 or max_samples >= len(rows):
        selected = rows
    elif mode == "proportional":
        selected = rng.sample(rows, max_samples)
    else:
        safe = [row for row in rows if row["is_safe"]]
        unsafe = [row for row in rows if not row["is_safe"]]
        each = max_samples // 2
        selected = rng.sample(safe, min(each, len(safe))) + rng.sample(unsafe, min(each, len(unsafe)))
        selected_ids = {(row["source_key"], row["source_line"]) for row in selected}
        remaining = [row for row in rows if (row["source_key"], row["source_line"]) not in selected_ids]
        if len(selected) < max_samples:
            selected.extend(rng.sample(remaining, min(max_samples - len(selected), len(remaining))))
    rng.shuffle(selected)
    return selected


def write_json(path: str, value: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="BeaverTails JSONL/JSON/Parquet files, directories, or glob patterns.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=20000, help="0 keeps every deduplicated prompt.")
    parser.add_argument("--sampling", choices=["proportional", "balanced"], default="proportional")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")

    rows, duplicate_count, resolved_paths = read_sources(args.input)
    selected = select_rows(rows, args.max_samples, args.sampling, args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    label_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in selected:
            categories = active_categories(row["category"])
            label_counts["safe" if row["is_safe"] else "unsafe"] += 1
            category_counts.update(categories or ["none"])
            sample_id = f"beavertails/neutral-{row['source_key']}-{row['source_line']:06d}"
            record = {
                "id": sample_id,
                "question": row["prompt"],
                "prompt": row["prompt"],
                "answer": row["response"],
                "reference_response": row["response"],
                "numeric_answer": None,
                "is_safe": row["is_safe"],
                "category": row["category"],
                "safety_categories": categories,
                "source": "beavertails/round0",
                "source_file": row["source_path"],
                "source_line": row["source_line"],
                "source_split": row["source_split"],
                "prompt_template": "natural_neutral_user_only",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "output": os.path.abspath(args.output),
        "input": resolved_paths,
        "seed": args.seed,
        "sampling": args.sampling,
        "requested_max_samples": args.max_samples,
        "source_unique_prompts": len(rows),
        "duplicates_removed": duplicate_count,
        "written_samples": len(selected),
        "label_counts": dict(label_counts),
        "category_counts": dict(category_counts.most_common()),
        "template_contract": {
            "prompt_equals_source_prompt": True,
            "system_prompt": "",
            "added_task_instruction": False,
            "added_safety_instruction": False,
            "added_output_format": False,
            "role_rendering": "deferred to tokenizer.apply_chat_template",
        },
    }
    manifest_path = args.output + ".manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
