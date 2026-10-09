"""Reusable statistical tests for paired experimental outcomes."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch


def bh_adjust(p_values: Sequence[Optional[float]]) -> list[Optional[float]]:
    valid = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    ]
    adjusted: list[Optional[float]] = [None for _ in p_values]
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    running = 1.0
    total = len(ordered)
    for rank_from_end in range(total - 1, -1, -1):
        original_index, value = ordered[rank_from_end]
        rank = rank_from_end + 1
        running = min(running, value * total / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted


def paired_bootstrap_and_signflip(
    values: Sequence[float], bootstrap_samples: int, permutation_tests: int, seed: int
) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "mean": None,
            "bootstrap_ci95": [None, None],
            "signflip_two_sided_p": None,
            "n": 0,
        }
    tensor = torch.tensor(clean, dtype=torch.float32)
    observed = float(tensor.mean().item())
    generator = torch.Generator().manual_seed(int(seed))
    chunk_size = 512
    bootstrap_means: list[float] = []
    for start in range(0, max(int(bootstrap_samples), 0), chunk_size):
        current = min(chunk_size, int(bootstrap_samples) - start)
        indices = torch.randint(
            0, tensor.numel(), (current, tensor.numel()), generator=generator
        )
        bootstrap_means.extend(tensor[indices].mean(dim=1).tolist())
    if bootstrap_means:
        bootstrap_tensor = torch.tensor(bootstrap_means)
        ci = [
            float(torch.quantile(bootstrap_tensor, 0.025).item()),
            float(torch.quantile(bootstrap_tensor, 0.975).item()),
        ]
    else:
        ci = [None, None]
    tests = max(int(permutation_tests), 0)
    extreme = 0
    for start in range(0, tests, chunk_size):
        current = min(chunk_size, tests - start)
        signs = (
            torch.randint(0, 2, (current, tensor.numel()), generator=generator)
            .float()
            .mul_(2.0)
            .sub_(1.0)
        )
        permuted = (signs * tensor.view(1, -1)).mean(dim=1).abs()
        extreme += int((permuted >= abs(observed)).sum().item())
    return {
        "mean": observed,
        "bootstrap_ci95": ci,
        "signflip_two_sided_p": float((extreme + 1) / (tests + 1)) if tests else None,
        "n": int(tensor.numel()),
    }
