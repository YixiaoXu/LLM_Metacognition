"""Dataset contracts used by activation extraction and downstream evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from metacog.io import read_jsonl


@dataclass(frozen=True)
class DatasetValidationReport:
    profile: str
    path: str
    rows: int
    unique_ids: int
    metric_profile: str
    answer_extraction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAdapter:
    """Stable task boundary for prompt data and outcome evaluation metadata."""

    name = "base"
    metric_profile = "general"
    answer_extraction = "none"

    def validate_row(self, row: dict[str, Any], line_number: int) -> None:
        sample_id = str(row.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"Missing id at line {line_number}.")
        if not str(row.get("prompt") or "").strip():
            raise ValueError(f"Missing prompt for {sample_id!r} at line {line_number}.")

    def validate(self, path: str | Path, min_samples: int = 0) -> DatasetValidationReport:
        seen: set[str] = set()
        rows = 0
        for line_number, row in enumerate(read_jsonl(path), start=1):
            self.validate_row(row, line_number)
            sample_id = str(row["id"])
            if sample_id in seen:
                raise ValueError(f"Duplicate id {sample_id!r} at line {line_number}.")
            seen.add(sample_id)
            rows += 1
        if rows < min_samples:
            raise ValueError(
                f"Dataset {path} has {rows} rows, fewer than required {min_samples}."
            )
        return DatasetValidationReport(
            profile=self.name,
            path=str(Path(path)),
            rows=rows,
            unique_ids=len(seen),
            metric_profile=self.metric_profile,
            answer_extraction=self.answer_extraction,
        )


class MathQAAdapter(DatasetAdapter):
    name = "mathqa"
    metric_profile = "math"
    answer_extraction = "mathqa_choice"

    def validate_row(self, row: dict[str, Any], line_number: int) -> None:
        super().validate_row(row, line_number)
        try:
            answer = int(row.get("correct_option_number", -1))
        except (TypeError, ValueError):
            answer = -1
        if answer not in range(1, 6):
            raise ValueError(f"Invalid MathQA option at line {line_number}: {answer!r}.")


class BeaverTailsAdapter(DatasetAdapter):
    name = "beavertails"
    metric_profile = "safety"
    answer_extraction = "none"

    def validate_row(self, row: dict[str, Any], line_number: int) -> None:
        super().validate_row(row, line_number)
        if not isinstance(row.get("is_safe"), bool):
            raise ValueError(f"Expected boolean is_safe at line {line_number}.")


class UltraChatAdapter(DatasetAdapter):
    name = "ultrachat"
    metric_profile = "conversation"
    answer_extraction = "none"
