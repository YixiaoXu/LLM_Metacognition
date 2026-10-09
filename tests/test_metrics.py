from __future__ import annotations

from metacog.evaluation import (
    compute_style_metrics,
    diagnostic_metric_families,
    family_for_metric,
    metric_polarity,
    metric_families,
    metrics_for_profile,
    primary_construct_count,
)


def test_profiles_do_not_mix_task_families() -> None:
    conversation = set(metrics_for_profile("conversation"))
    assert "correct" not in conversation
    assert "style_refusal_score" not in conversation
    assert "style_paragraph_count" in conversation
    assert "correct" not in metrics_for_profile("math")
    assert "correct" in metrics_for_profile("math", include_diagnostics=True)
    assert "correct" in diagnostic_metric_families("math")["task_performance_control"]


def test_conversation_style_metrics() -> None:
    text = "Sure, here is an example:\n\n## Plan\n- First item\n- Second item?"
    metrics = compute_style_metrics(text, generated_tokens=15, max_new_tokens=128)
    assert metrics["style_paragraph_count"] == 2
    assert metrics["style_heading_count"] == 1
    assert metrics["style_bullet_line_count"] == 2
    assert metrics["style_question_count"] == 1
    assert metrics["style_truncated"] == 0


def test_conversation_fdr_families_are_prespecified() -> None:
    families = metric_families("conversation")
    assert set(families) == {
        "response_extent",
        "paragraph_organization",
        "formatted_structure",
        "explanatory_scaffolding",
        "interaction_orientation",
        "epistemic_monitoring",
        "initial_uncertainty",
        "early_confidence_update",
        "sequence_confidence",
        "local_sequence_fit",
    }
    assert family_for_metric("paragraph_count", "conversation") == "paragraph_organization"
    assert primary_construct_count("conversation") == 10


def test_all_primary_profiles_have_ten_constructs() -> None:
    assert all(primary_construct_count(profile) == 10 for profile in (
        "conversation", "safety", "math"
    ))
    assert family_for_metric("compliance_score", "safety") == "refusal_compliance"
    assert metric_polarity("prompt_end_max_probability") == -1.0


def test_safety_compliance_style_metric() -> None:
    metrics = compute_style_metrics(
        "Sure, I can explain a safer alternative.",
        generated_tokens=9,
        max_new_tokens=128,
    )
    assert metrics["style_compliance_score"] >= 1
