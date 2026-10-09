"""Dataset adapters and the common JSONL contract."""

from .base import DatasetAdapter, DatasetValidationReport
from .registry import get_dataset, list_datasets

__all__ = [
    "DatasetAdapter",
    "DatasetValidationReport",
    "get_dataset",
    "list_datasets",
]
