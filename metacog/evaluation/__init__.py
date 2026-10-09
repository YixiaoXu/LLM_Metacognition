"""Task-aware outcome profiles and generation-style measurements."""

from .profiles import (
    behavior_field_aliases,
    behavior_metrics_for_profile,
    diagnostic_metric_families,
    family_for_metric,
    metric_polarity,
    metric_families,
    metrics_for_profile,
    primary_construct_count,
    primary_construct_groups,
    profile_names,
)
from .styles import compute_style_metrics

__all__ = [
    "behavior_field_aliases",
    "behavior_metrics_for_profile",
    "compute_style_metrics",
    "diagnostic_metric_families",
    "family_for_metric",
    "metric_polarity",
    "metric_families",
    "metrics_for_profile",
    "primary_construct_count",
    "primary_construct_groups",
    "profile_names",
]
