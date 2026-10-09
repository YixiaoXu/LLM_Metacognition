#!/usr/bin/env python
"""CPU smoke test with known synthetic information structure, no LLM needed."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import numpy as np

from metacog.audits.semantic_leakage import information_bounds, leakage_ceiling


def main() -> None:
    rng = np.random.default_rng(7319)
    n, train = 30000, 5000
    c = rng.integers(0, 2, n)
    v = c ^ (rng.random(n) < .03)
    independent_q = np.full((n - train, 2), .5)
    signal = information_bounds(c[:train], v[:train], c[train:], v[train:],
             {"known_independent": independent_q}, 2, .001, .05, 71)
    if signal["remaining_information_lcb_at_zero_density_kl"] <= .5:
        raise AssertionError("Known residual signal should pass the known-density synthetic check.")

    s = rng.integers(0, 2, n)
    c = s ^ (rng.random(n) < .05)
    v = s ^ (rng.random(n) < .03)
    q = np.column_stack([np.where(s[train:] == 0, .95, .05), np.where(s[train:] == 1, .95, .05)])
    null = information_bounds(c[:train], v[:train], c[train:], v[train:],
             {"known_semantic_common_cause": q}, 2, .001, .05, 71)
    if null["remaining_information_lcb_at_zero_density_kl"] >= 0:
        raise AssertionError("Pure-semantic common cause must not produce a positive gap here.")

    misleading = information_bounds(s[:train], s[:train], s[train:], s[train:],
             {"deliberately_bad_uniform_density": independent_q}, 2, .001, .05, 71)
    if misleading["absolute_semantic_freeness_proven"]:
        raise AssertionError("An incompetent probe must never produce an unconditional proof.")
    if misleading["state_information_lcb_bits"] > leakage_ceiling(misleading, 1.):
        raise AssertionError("Accounting for the actual one-bit density error must remove the apparent gap.")
    print(json.dumps({"statistics_smoke": "passed", "synthetic_only": True,
          "known_residual_signal_gap_bits": signal["remaining_information_lcb_at_zero_density_kl"],
          "known_semantic_null_gap_bits": null["remaining_information_lcb_at_zero_density_kl"],
          "bad_probe_apparent_zero_kl_gap_bits": misleading["remaining_information_lcb_at_zero_density_kl"],
          "bad_probe_gap_with_actual_kl_bits": misleading["state_information_lcb_bits"] - leakage_ceiling(misleading, 1.),
          "note": "Synthetic checks validate code, not the real LLM semantic leakage assumption."}, indent=2))


if __name__ == "__main__":
    main()
