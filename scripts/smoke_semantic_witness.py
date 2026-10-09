#!/usr/bin/env python
"""Offline CPU smoke: direct-bound identities plus frozen fit/evaluation I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import _bootstrap  # noqa: F401
import numpy as np
import torch

from metacog.audits.conditional_witness import density_statement, evaluate_witness, fit_witness, paired_gain
from run_semantic_witness import validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Optional directory; otherwise remove synthetic artifacts after testing.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/semantic_witness_v2.json").read_text())
    config.update({"smoke_only": True, "bin_candidates": [2], "min_train_per_bin": 8,
                   "min_calibration_per_bin": 4, "min_test_prompts": 32, "max_test_prompts": 128,
                   "bootstrap_samples": 64, "continuous_auxiliary": True})
    config["probes"].update({"device": "cpu", "epochs": 2, "patience": 2, "hidden_dim": 16,
                              "families": ["linear", "mlp", "ensemble"]})
    validate_config(config)
    torch.set_num_threads(1)
    rng = np.random.default_rng(9327)
    n = 20000
    m = rng.integers(0, 2, n)
    y = m ^ (rng.random(n) < .05)
    sem = np.full((n, 2), .5)
    joint = np.column_stack([np.where(m == 0, .95, .05), np.where(m == 1, .95, .05)])
    known = paired_gain(y, sem, joint, .5, .05, 1., .05, [f"p{i}" for i in range(n)], 64, 2)
    assert known["gain_lcb_bits"] > .5
    misleading = density_statement(known["gain_lcb_bits"], None)
    assert misleading["conditional_information_lcb_under_assumption_bits"] is None
    assert not misleading["unconditional_shannon_information_certified"]
    assert not density_statement(known["gain_lcb_bits"], {"upper_bits": 1., "rationale": "synthetic error"})["positive_conditional_information_under_assumption"]

    def bundle(prefix, seed):
        g = torch.Generator().manual_seed(seed)
        x, z = torch.randn(160, 4, generator=g), torch.randn(160, 1, generator=g)
        return {"ids": [f"{prefix}{i}::step000" for i in range(160)], "steps": [0] * 160,
                "module_scores": z, "target_scores": z + x[:, :1], "module_names": ["synthetic_module"],
                "semantics": {"training_reference": x, "independent_model": x.square(),
                              "lexical_controls": torch.sin(x)}, "forbidden_ids": []}
    temp = tempfile.TemporaryDirectory(prefix="semantic_witness_smoke_") if not args.output_dir else None
    output = Path(args.output_dir or temp.name)
    pilot, test = bundle("pilot", 1), bundle("test", 2)
    fit_witness(pilot, "synthetic_module", output / "fit", config, .05)
    result = evaluate_witness(test, "synthetic_module", output / "fit", output / "test", config, .05)
    assert evaluate_witness(test, "synthetic_module", output / "fit", output / "test", config, .05) == result
    try:
        evaluate_witness(pilot, "synthetic_module", output / "fit", output / "overlap", config, .05)
    except ValueError as exc:
        assert "overlaps pilot" in str(exc)
    else:
        raise AssertionError("Historical prompt overlap was not rejected.")
    print(json.dumps({"statistics_and_pipeline_smoke": "passed", "synthetic_only": True,
                      "known_density_direct_gain_lcb_bits": known["gain_lcb_bits"],
                      "frozen_predictor_reload": True, "resume": True, "overlap_rejected": True,
                      "unknown_density_error_not_assumed_zero": True}, indent=2))
    if temp:
        temp.cleanup()


if __name__ == "__main__":
    main()
