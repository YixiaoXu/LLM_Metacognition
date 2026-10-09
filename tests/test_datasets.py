from __future__ import annotations

import json
from pathlib import Path

import pytest

from metacog.datasets import get_dataset
from metacog.datasets.mathqa import prepare_mathqa
from metacog.datasets.ultrachat import prepare_ultrachat


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_ultrachat_preparation_is_deduplicated_and_valid(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    _write_jsonl(
        source,
        [
            {
                "id": "a",
                "messages": [
                    {"role": "user", "content": "Explain rainbows."},
                    {"role": "assistant", "content": "Light is refracted."},
                ],
            },
            {"id": "duplicate", "prompt": "  Explain   rainbows. "},
            {"id": "b", "prompt": "Write a short email."},
        ],
    )
    output = tmp_path / "prepared.jsonl"
    manifest = prepare_ultrachat([str(source)], output, max_samples=0, seed=42)
    assert manifest["written_samples"] == 2
    report = get_dataset("ultrachat").validate(output, min_samples=2)
    assert report.metric_profile == "conversation"


def test_dataset_validation_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    _write_jsonl(path, [{"id": "same", "prompt": "a"}, {"id": "same", "prompt": "b"}])
    with pytest.raises(ValueError, match="Duplicate id"):
        get_dataset("ultrachat").validate(path)


def test_mathqa_preparation_and_contract(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    item = {
        "Problem": "What is 1 + 1?",
        "options": "a) 1, b) 2, c) 3, d) 4, e) 5",
        "correct": "b",
        "Rationale": "One plus one is two.",
    }
    for filename in ("train.json", "dev.json", "test.json"):
        (source / filename).write_text(json.dumps([item]), encoding="utf-8")
    output = tmp_path / "prepared"
    manifest = prepare_mathqa(
        source, output, split_mode="resplit", train_size=1, validation_size=1
    )
    assert manifest["counts"]["all"] == 3
    report = get_dataset("mathqa").validate(output / "all.jsonl", min_samples=3)
    assert report.answer_extraction == "mathqa_choice"
