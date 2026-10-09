"""Declarative model registry and environment-resolved metadata."""

from .registry import ModelSpec, get_model, list_models

__all__ = ["ModelSpec", "get_model", "list_models"]
