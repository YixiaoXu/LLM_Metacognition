"""Prepare local UltraChat shards for metacognition experiments."""

from __future__ import annotations

import glob
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from metacog.io import write_json, write_jsonl


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".parquet"}


def expand_sources(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value, recursive=True))
        if not matches and Path(value).exists():
            matches = [value]
        for match in matches:
            path = Path(match).resolve()
            if path.is_dir():
                paths.extend(
                    child
                    for child in sorted(path.rglob("*"))
                    if child.suffix.lower() in SUPPORTED_SUFFIXES
                )
            elif path.suffix.lower() in SUPPORTED_SUFFIXES:
                paths.append(path)
    return list(dict.fromkeys(paths))


def _iter_file(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield index, value
        return
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("data", "train", "rows", "records"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError(f"Expected a list in {path}.")
        for index, row in enumerate(value):
            if isinstance(row, dict):
                yield index, row
        return
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "UltraChat Parquet input requires pyarrow (`pip install pyarrow`)."
            ) from exc
        parquet = pq.ParquetFile(path)
        offset = 0
        for batch in parquet.iter_batches(batch_size=8192):
            for index, row in enumerate(batch.to_pylist(), start=offset):
                if isinstance(row, dict):
                    yield index, row
            offset += batch.num_rows
        return
    raise ValueError(f"Unsupported UltraChat input: {path}")


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or row.get("conversation") or row.get("conversations")
    return [item for item in (messages or []) if isinstance(item, dict)]


def _message_role(message: dict[str, Any]) -> str:
    return str(message.get("role") or message.get("from") or "").strip().lower()


def _message_text(message: dict[str, Any]) -> str:
    return str(message.get("content") or message.get("value") or "").strip()


def first_user_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        return prompt
    for message in _messages(row):
        if _message_role(message) in {"user", "human"}:
            text = _message_text(message)
            if text:
                return text
    return ""


def first_assistant_response(row: dict[str, Any]) -> str:
    for message in _messages(row):
        if _message_role(message) in {"assistant", "gpt", "bot"}:
            text = _message_text(message)
            if text:
                return text
    return str(row.get("response") or "").strip()


def _normalized_prompt(text: str) -> str:
    return " ".join(text.split()).casefold()


def _source_id(path: Path, index: int, row: dict[str, Any]) -> str:
    explicit = row.get("prompt_id") or row.get("id")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}-{digest}-{index:07d}"


def iter_records(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for path in paths:
        for index, row in _iter_file(path):
            prompt = first_user_prompt(row)
            if not prompt:
                continue
            normalized = _normalized_prompt(prompt)
            if normalized in seen:
                continue
            seen.add(normalized)
            source_id = _source_id(path, index, row)
            reference_response = first_assistant_response(row)
            yield {
                "id": f"ultrachat/train_sft/{source_id}",
                "question": prompt,
                "prompt": prompt,
                "answer": reference_response,
                "reference_response": reference_response,
                "source": "HuggingFaceH4/ultrachat_200k",
                "source_file": str(path),
                "source_row": index,
                "prompt_template": "natural_user_request",
            }


def reservoir_sample(
    records: Iterable[dict[str, Any]], max_samples: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    valid = 0
    for record in records:
        valid += 1
        if max_samples <= 0:
            selected.append(record)
        elif len(selected) < max_samples:
            selected.append(record)
        else:
            replacement = rng.randrange(valid)
            if replacement < max_samples:
                selected[replacement] = record
    rng.shuffle(selected)
    return selected, valid


def prepare_ultrachat(
    inputs: Sequence[str], output: str | Path, max_samples: int, seed: int
) -> dict[str, Any]:
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative.")
    paths = expand_sources(inputs)
    if not paths:
        raise FileNotFoundError("No UltraChat JSON/JSONL/Parquet shards were found.")
    selected, valid = reservoir_sample(iter_records(paths), max_samples, seed)
    output_path = Path(output)
    write_jsonl(output_path, selected)
    manifest = {
        "schema_version": 1,
        "dataset_profile": "ultrachat",
        "output": str(output_path.resolve()),
        "inputs": [str(path) for path in paths],
        "seed": seed,
        "requested_max_samples": max_samples,
        "valid_unique_prompts": valid,
        "written_samples": len(selected),
        "prompt_contract": "first user request; model chat template applied downstream",
        "reference_response_is_not_used_as_generation_input": True,
    }
    write_json(f"{output_path}.manifest.json", manifest)
    return manifest
