"""Dataset adapter registry."""

from __future__ import annotations

from .base import BeaverTailsAdapter, DatasetAdapter, MathQAAdapter, UltraChatAdapter


_ADAPTERS: dict[str, type[DatasetAdapter]] = {
    adapter.name: adapter
    for adapter in (BeaverTailsAdapter, MathQAAdapter, UltraChatAdapter)
}


def get_dataset(name: str) -> DatasetAdapter:
    try:
        return _ADAPTERS[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset profile {name!r}; available: {', '.join(_ADAPTERS)}"
        ) from exc


def list_datasets() -> list[DatasetAdapter]:
    return [adapter() for adapter in _ADAPTERS.values()]
