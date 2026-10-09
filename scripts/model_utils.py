"""Compatibility exports for legacy script entry points.

New code should import from :mod:`metacog.models.loading`.
"""

import _bootstrap  # noqa: F401
from metacog.models.loading import *  # noqa: F403
