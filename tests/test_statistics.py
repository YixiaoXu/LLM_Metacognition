from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from metacog.clustering import run_kmeans  # noqa: E402
from metacog.statistics import bh_adjust, paired_bootstrap_and_signflip  # noqa: E402


def test_bh_adjust_is_monotone_in_rank() -> None:
    adjusted = bh_adjust([0.01, 0.04, 0.03, None])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.04)
    assert adjusted[2] == pytest.approx(0.04)
    assert adjusted[3] is None


def test_paired_test_and_kmeans_are_deterministic() -> None:
    result = paired_bootstrap_and_signflip([1.0, 1.5, 2.0], 32, 64, seed=7)
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(1.5)
    values = torch.tensor([[0.0], [0.1], [10.0], [10.1]])
    labels_a, _, inertia_a = run_kmeans(values, 2, 20, 3, seed=11)
    labels_b, _, inertia_b = run_kmeans(values, 2, 20, 3, seed=11)
    assert torch.equal(labels_a, labels_b)
    assert inertia_a == pytest.approx(inertia_b)
