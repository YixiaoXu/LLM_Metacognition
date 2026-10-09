"""Prespecified behavior constructs for each supported research task.

Raw measurements remain available for localization and diagnostics. Primary
inference is performed at the construct level so correlated measurements such
as line/operator/equation counts or nested log-probability summaries do not
enter a multiplicity correction as separate scientific hypotheses.
"""

from __future__ import annotations

from typing import Dict, List


SHARED_MONITORING_FAMILIES: Dict[str, List[str]] = {
    "initial_uncertainty": [
        "prompt_end_entropy",
        "prompt_end_normalized_entropy",
        "prompt_end_max_probability",
        "prompt_end_top1_top2_logit_margin",
    ],
    "early_confidence_update": [
        "confidence_entropy_shift_prompt_to_prefix16",
        "confidence_margin_shift_prompt_to_prefix16",
    ],
    "sequence_confidence": [
        "generated_prefix16_entropy_mean",
        "generated_prefix16_max_probability_mean",
        "generated_prefix16_logit_margin_mean",
        "generated_mean_logprob",
        "generated_cumulative_logprob_per_token",
    ],
    "local_sequence_fit": [
        "generated_min_logprob",
    ],
}


SAFETY_METRIC_FAMILIES: Dict[str, List[str]] = {
    "refusal_compliance": [
        "style_refusal_present",
        "style_refusal_score",
        "style_compliance_score",
    ],
    "refusal_timing": [
        "style_refusal_onset_normalized",
    ],
    "safe_redirection": [
        "style_redirect_score",
    ],
    "safety_framing": [
        "style_policy_language_score",
        "style_harmful_detail_score",
    ],
    "response_organization": [
        "generated_tokens",
        "style_line_count",
    ],
    "epistemic_monitoring": [
        "style_hedging_score",
        "style_certainty_score",
        "style_self_correction_marker_count",
    ],
    **SHARED_MONITORING_FAMILIES,
}


MATH_METRIC_FAMILIES: Dict[str, List[str]] = {
    "response_extent": [
        "generated_tokens",
    ],
    "reasoning_organization": [
        "style_line_count",
        "style_bullet_line_count",
        "style_step_marker_count",
        "style_explanation_marker_count",
    ],
    "mathematical_formalization": [
        "style_number_count",
        "style_operator_count",
        "style_equation_line_count",
    ],
    "answer_presentation": [
        "style_hash_answer",
        "style_answer_marker",
    ],
    "self_correction": [
        "style_self_correction_marker_count",
    ],
    "epistemic_stance": [
        "style_hedging_score",
        "style_certainty_score",
    ],
    **SHARED_MONITORING_FAMILIES,
}


CONVERSATION_METRIC_FAMILIES: Dict[str, List[str]] = {
    "response_extent": [
        "generated_tokens",
    ],
    "paragraph_organization": [
        "style_line_count",
        "style_paragraph_count",
    ],
    "formatted_structure": [
        "style_bullet_line_count",
        "style_heading_count",
        "style_code_block_count",
    ],
    "explanatory_scaffolding": [
        "style_explanation_marker_count",
        "style_example_marker_count",
    ],
    "interaction_orientation": [
        "style_question_count",
        "style_politeness_score",
        "style_first_person_count",
        "style_second_person_count",
    ],
    "epistemic_monitoring": [
        "style_hedging_score",
        "style_certainty_score",
        "style_self_correction_marker_count",
    ],
    **SHARED_MONITORING_FAMILIES,
}


# Negative controls and data-quality outcomes do not enter the ten-construct
# primary family.
DIAGNOSTIC_METRIC_FAMILIES: Dict[str, Dict[str, List[str]]] = {
    "math": {
        "task_performance_control": [
            "correct",
            "parse_success",
            "strict_hash_correct",
            "strict_hash_parse_success",
            "valid_choice",
        ],
    },
    "conversation": {},
    "safety": {},
}

QUALITY_METRICS = ["style_truncated"]


# Canonical generation fields map to the unprefixed fields emitted by the
# task-specific behavior tables.
BEHAVIOR_FIELD_ALIASES = {
    "style_line_count": "line_count",
    "style_number_count": "number_count",
    "style_operator_count": "operator_count",
    "style_equation_line_count": "equation_line_count",
    "style_bullet_line_count": "bullet_line_count",
    "style_step_marker_count": "step_marker_count",
    "style_explanation_marker_count": "explanation_marker_count",
    "style_self_correction_marker_count": "self_correction_marker_count",
    "style_hash_answer": "hash_answer",
    "style_answer_marker": "answer_marker",
    "style_paragraph_count": "paragraph_count",
    "style_heading_count": "heading_count",
    "style_code_block_count": "code_block_count",
    "style_question_count": "question_count",
    "style_example_marker_count": "example_marker_count",
    "style_certainty_score": "certainty_score",
    "style_politeness_score": "politeness_score",
    "style_first_person_count": "first_person_count",
    "style_second_person_count": "second_person_count",
    "style_refusal_present": "model_refusal",
    "style_refusal_score": "refusal_score",
    "style_compliance_score": "compliance_score",
    "style_redirect_score": "redirect_score",
    "style_refusal_onset_normalized": "refusal_onset_normalized",
    "style_policy_language_score": "policy_language_score",
    "style_hedging_score": "hedging_score",
    "style_harmful_detail_score": "harmful_detail_score",
}


# Signs orient construct effects for behavior-matched intervention tests.
# Information tests are sign-free; this table is used only after a construct
# passes its held-out association test.
METRIC_POLARITY = {
    "prompt_end_entropy": 1.0,
    "prompt_end_normalized_entropy": 1.0,
    "prompt_end_max_probability": -1.0,
    "prompt_end_top1_top2_logit_margin": -1.0,
    "confidence_entropy_shift_prompt_to_prefix16": 1.0,
    "confidence_margin_shift_prompt_to_prefix16": -1.0,
    "generated_prefix16_entropy_mean": -1.0,
    "generated_prefix16_max_probability_mean": 1.0,
    "generated_prefix16_logit_margin_mean": 1.0,
    "generated_mean_logprob": 1.0,
    "generated_cumulative_logprob_per_token": 1.0,
    "generated_min_logprob": 1.0,
    "style_certainty_score": -1.0,
    "style_compliance_score": -1.0,
    "style_harmful_detail_score": -1.0,
}


_PROFILES = {
    "safety": SAFETY_METRIC_FAMILIES,
    "math": MATH_METRIC_FAMILIES,
    "conversation": CONVERSATION_METRIC_FAMILIES,
}


# The strict three-ring analysis treats response behavior and internal
# monitoring as two prespecified scientific families.  Their five hypotheses
# are corrected separately; changing these groups after inspecting a model's
# results would invalidate the confirmatory contract.
PRIMARY_CONSTRUCT_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "conversation": {
        "behavior": [
            "response_extent",
            "paragraph_organization",
            "formatted_structure",
            "explanatory_scaffolding",
            "interaction_orientation",
        ],
        "monitoring": [
            "epistemic_monitoring",
            "initial_uncertainty",
            "early_confidence_update",
            "sequence_confidence",
            "local_sequence_fit",
        ],
    },
    "safety": {
        "behavior": [
            "refusal_compliance",
            "refusal_timing",
            "safe_redirection",
            "safety_framing",
            "response_organization",
        ],
        "monitoring": [
            "epistemic_monitoring",
            "initial_uncertainty",
            "early_confidence_update",
            "sequence_confidence",
            "local_sequence_fit",
        ],
    },
    "math": {
        "behavior": [
            "response_extent",
            "reasoning_organization",
            "mathematical_formalization",
            "answer_presentation",
            "self_correction",
        ],
        "monitoring": [
            "epistemic_stance",
            "initial_uncertainty",
            "early_confidence_update",
            "sequence_confidence",
            "local_sequence_fit",
        ],
    },
}


def profile_names() -> list[str]:
    return list(_PROFILES)


def _flatten(families: Dict[str, List[str]]) -> List[str]:
    output: List[str] = []
    for metrics in families.values():
        for metric in metrics:
            if metric not in output:
                output.append(metric)
    return output


def metric_families(profile: str) -> Dict[str, List[str]]:
    try:
        families = _PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown metric profile {profile!r}; choose one of {profile_names()}."
        ) from exc
    return {name: list(values) for name, values in families.items()}


def primary_construct_groups(profile: str) -> Dict[str, List[str]]:
    """Return the frozen five-behavior/five-monitoring hypothesis families."""
    if profile not in _PROFILES:
        raise ValueError(
            f"Unknown metric profile {profile!r}; choose one of {profile_names()}."
        )
    groups = PRIMARY_CONSTRUCT_GROUPS[profile]
    flattened = [construct for values in groups.values() for construct in values]
    expected = list(metric_families(profile))
    if len(flattened) != len(set(flattened)) or set(flattened) != set(expected):
        raise RuntimeError(
            f"Primary construct groups for {profile!r} do not partition its "
            "metric families exactly."
        )
    return {name: list(values) for name, values in groups.items()}


def diagnostic_metric_families(profile: str) -> Dict[str, List[str]]:
    if profile not in _PROFILES:
        raise ValueError(
            f"Unknown metric profile {profile!r}; choose one of {profile_names()}."
        )
    return {
        name: list(values)
        for name, values in DIAGNOSTIC_METRIC_FAMILIES.get(profile, {}).items()
    }


def metrics_for_profile(
    profile: str,
    include_quality: bool = False,
    include_diagnostics: bool = False,
) -> List[str]:
    metrics = _flatten(metric_families(profile))
    if include_diagnostics:
        metrics.extend(
            metric
            for metric in _flatten(diagnostic_metric_families(profile))
            if metric not in metrics
        )
    if include_quality:
        metrics.extend(metric for metric in QUALITY_METRICS if metric not in metrics)
    return metrics


def behavior_field_aliases() -> dict[str, str]:
    """Map unprefixed behavior fields to canonical generated-row fields."""
    return {behavior: style for style, behavior in BEHAVIOR_FIELD_ALIASES.items()}


def behavior_metrics_for_profile(profile: str) -> List[str]:
    return [
        BEHAVIOR_FIELD_ALIASES.get(metric, metric)
        for metric in metrics_for_profile(profile)
    ]


def family_for_metric(metric: str, profile: str) -> str:
    canonical_metric = behavior_field_aliases().get(metric, metric)
    for family, metrics in metric_families(profile).items():
        if canonical_metric in metrics:
            return family
    for family, metrics in diagnostic_metric_families(profile).items():
        if canonical_metric in metrics:
            return family
    if canonical_metric in QUALITY_METRICS:
        return "quality_diagnostic"
    return "unclassified"


def metric_polarity(metric: str) -> float:
    canonical_metric = behavior_field_aliases().get(metric, metric)
    return float(METRIC_POLARITY.get(canonical_metric, 1.0))


def primary_construct_count(profile: str) -> int:
    return len(metric_families(profile))
