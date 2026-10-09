"""Declarative model profiles used by shell and Python entry points."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "models.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    path_env: str
    dtype: str = "bfloat16"
    extraction_batch_size: int = 8
    trust_remote_code: bool = False
    chat_template_enable_thinking: str = "auto"

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "ModelSpec":
        required = {"path", "path_env"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Model {name!r} misses fields: {sorted(missing)}")
        return cls(name=name, **value)

    @property
    def resolved_path(self) -> str:
        return os.environ.get(self.path_env, self.path)

    def field(self, name: str) -> str:
        if name == "path":
            return self.resolved_path
        if not hasattr(self, name):
            raise KeyError(f"Unknown model field {name!r}.")
        value = getattr(self, name)
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)


@lru_cache(maxsize=4)
def _load_registry(path: str) -> dict[str, ModelSpec]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No models found in registry: {registry_path}")
    return {
        name: ModelSpec.from_dict(name, value)
        for name, value in sorted(models.items())
    }


def model_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    configured = path or os.environ.get("METACOG_MODEL_REGISTRY") or DEFAULT_REGISTRY
    return dict(_load_registry(str(Path(configured).resolve())))


def get_model(name: str, path: str | Path | None = None) -> ModelSpec:
    registry = model_registry(path)
    try:
        return registry[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model profile {name!r}; available: {', '.join(registry)}"
        ) from exc


def list_models(path: str | Path | None = None) -> list[ModelSpec]:
    return list(model_registry(path).values())
