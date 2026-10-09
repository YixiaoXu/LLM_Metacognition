"""Compatibility imports for legacy script entry points.

New code should import from :mod:`metacog.evaluation`.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metacog.evaluation.profiles import (  # noqa: F401
    BEHAVIOR_FIELD_ALIASES,
    CONVERSATION_METRIC_FAMILIES,
    DIAGNOSTIC_METRIC_FAMILIES,
    MATH_METRIC_FAMILIES,
    METRIC_POLARITY,
    QUALITY_METRICS,
    SAFETY_METRIC_FAMILIES,
    behavior_field_aliases,
    behavior_metrics_for_profile,
    diagnostic_metric_families,
    family_for_metric,
    metric_polarity,
    metric_families,
    metrics_for_profile,
    primary_construct_count,
    profile_names,
)

__all__ = [name for name in globals() if not name.startswith("_")]
