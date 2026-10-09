"""Small, deterministic JSON and JSONL helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            yield value


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, value: Any) -> None:
    _atomic_text_write(
        Path(path), json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False) + "\n" for row in materialized
    )
    _atomic_text_write(Path(path), text)
    return len(materialized)
