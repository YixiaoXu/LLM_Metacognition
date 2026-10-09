"""MathQA conversion to the shared prompt JSONL contract."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from metacog.io import write_json, write_jsonl


FILES = {"train": "train.json", "validation": "dev.json", "test": "test.json"}
LETTERS = "abcde"


def _locate(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Cannot find {name} under {root}.")
    return matches[0]


def parse_options(raw: str) -> Optional[list[tuple[str, str]]]:
    text = str(raw or "").strip()
    markers = list(re.finditer(r"(?i)(?:^|,\s*)([a-e])\s*\)\s*", text))
    if len(markers) != 5:
        return None
    values = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        values.append((marker.group(1).lower(), text[marker.end() : end].strip(" ,")))
    if [label for label, _ in values] != list(LETTERS) or any(
        not value for _, value in values
    ):
        return None
    return values


def render_prompt(problem: str, options: Sequence[tuple[str, str]]) -> str:
    choices = "\n".join(
        f"{index}. {value}" for index, (_, value) in enumerate(options, 1)
    )
    return (
        "Solve the following multiple-choice math problem. Show concise step-by-step "
        "reasoning. At the end, write the selected option number exactly as "
        "#### <option number>, where the option number is 1, 2, 3, 4, or 5.\n\n"
        f"Question: {problem.strip()}\n\nOptions:\n{choices}\n\nAnswer:"
    )


def convert_row(row: dict[str, Any], split: str, index: int) -> Optional[dict[str, Any]]:
    problem = str(row.get("Problem", row.get("problem", ""))).strip()
    options = parse_options(row.get("options", ""))
    correct = str(row.get("correct", "")).strip().lower()
    if not problem or options is None or correct not in LETTERS:
        return None
    answer = LETTERS.index(correct) + 1
    return {
        "id": f"mathqa/{split}-{index}",
        "source": "mathqa",
        "original_split": split,
        "question": problem,
        "prompt": render_prompt(problem, options),
        "answer": f"#### {answer}",
        "numeric_answer": str(answer),
        "correct_option_number": answer,
        "correct_letter": correct.upper(),
        "correct_option_text": options[answer - 1][1],
        "options": [value for _, value in options],
        "rationale": row.get("Rationale", row.get("rationale", "")),
        "category": row.get("category", ""),
        "prompt_template": "reasoning_choice_v2",
    }


def prepare_mathqa(
    input_dir: str | Path,
    output_dir: str | Path,
    split_mode: str = "resplit",
    train_size: int = 22000,
    validation_size: int = 7000,
    seed: int = 42,
) -> dict[str, Any]:
    source_root = Path(input_dir)
    converted: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, int] = {}
    for split, filename in FILES.items():
        raw = json.loads(_locate(source_root, filename).read_text(encoding="utf-8"))
        rows = [
            value
            for index, item in enumerate(raw)
            if (value := convert_row(item, split, index))
        ]
        converted[split] = rows
        skipped[split] = len(raw) - len(rows)
    if split_mode == "resplit":
        all_rows = [row for split in FILES for row in converted[split]]
        random.Random(seed).shuffle(all_rows)
        if train_size + validation_size >= len(all_rows):
            raise ValueError("Requested train/validation sizes leave no test examples.")
        converted = {
            "train": all_rows[:train_size],
            "validation": all_rows[train_size : train_size + validation_size],
            "test": all_rows[train_size + validation_size :],
        }
    output = Path(output_dir)
    for split, rows in converted.items():
        for row in rows:
            row["split"] = split
        write_jsonl(output / f"{split}.jsonl", rows)
    all_rows = [
        row
        for split in ("train", "validation", "test")
        for row in converted[split]
    ]
    write_jsonl(output / "all.jsonl", all_rows)
    manifest = {
        "schema_version": 1,
        "dataset_profile": "mathqa",
        "input_dir": str(source_root),
        "split_mode": split_mode,
        "seed": seed,
        "counts": {split: len(rows) for split, rows in converted.items()}
        | {"all": len(all_rows)},
        "skipped": skipped,
        "answer_contract": "1-based option number, emitted as #### <1..5>",
        "chat_contract": "prompt is user content; target tokenizer applies chat template",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
