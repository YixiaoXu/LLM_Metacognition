from __future__ import annotations

from metacog.models import get_model, list_models


def test_registry_contains_supported_families() -> None:
    names = {model.name for model in list_models()}
    assert {"llama31_8b", "qwen3_4b", "qwen25_math7b"} <= names


def test_environment_overrides_model_path(monkeypatch) -> None:
    monkeypatch.setenv("QWEN3_4B_MODEL_PATH", "/models/custom-qwen")
    assert get_model("qwen3_4b").resolved_path == "/models/custom-qwen"


def test_boolean_fields_are_shell_friendly() -> None:
    assert get_model("qwen3_4b").field("trust_remote_code") == "1"
    assert get_model("llama31_8b").field("trust_remote_code") == "0"
