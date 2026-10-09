from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import scripts.summarize_strict_module_chain as strict
from metacog.evaluation import metric_families, primary_construct_groups


def write_csv(path: Path, rows: list[dict[str, object]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def make_module(
    pair_root: Path,
    module_index: int,
    behavior_p: float,
    screen_p: float,
) -> tuple[dict[str, object], Path]:
    module_dir = pair_root / "jobs" / "support_32" / "modules" / f"module_{module_index:02d}"
    module_dir.mkdir(parents=True)
    (module_dir / "module_summary.json").write_text(
        json.dumps(
            {
                "information": {
                    "residual_gain_mse": 0.1,
                    "residual_gain_bootstrap_ci95_low": 0.05,
                    "residual_gain_signflip_one_sided_p": 0.001,
                }
            }
        ),
        encoding="utf-8",
    )
    rank = module_index + 1
    output = pair_root / "continuous_modules" / f"rank_{rank}_k32_m{module_index}"
    behavior_rows = []
    for construct in metric_families("math"):
        behavior_rows.append(
            {
                "meta_predictor": "signed_direction",
                "metric_family": construct,
                "metrics": "synthetic_metric",
                "mean_conditional_meta_information_bits_per_sample": 0.02,
                "aligned_standardized_slope": 0.1,
                "signflip_one_sided_p": behavior_p,
                "aligned_slope_signflip_two_sided_p": behavior_p,
            }
        )
    write_csv(
        output / "continuous_behavior" / "continuous_behavior_family_omnibus.csv",
        behavior_rows,
    )
    screen_rows = [
        {
            "metric": "next_token_auto_logit_delta_projection_to_target",
            "mean": 0.2,
            "signflip_two_sided_p": screen_p,
        },
        {
            "metric": "next_token_auto_logit_margin_delta_toward_target",
            "mean": 0.1,
            "signflip_two_sided_p": screen_p,
        },
    ]
    for role in ("causal_screen_a", "causal_screen_b"):
        write_csv(output / role / "continuous_signed_dose_response.csv", screen_rows)
    return (
        {
            "rank": rank,
            "profile": "support_32",
            "support": 32,
            "module": module_index,
            "module_dir": str(module_dir),
        },
        output,
    )


def test_only_ring_1_2_passers_enter_next_token_and_trajectory(
    tmp_path: Path, monkeypatch,
) -> None:
    pair_root = tmp_path / "pair"
    selected_rows = [
        make_module(pair_root, 0, behavior_p=0.001, screen_p=0.001)[0],
        make_module(pair_root, 1, behavior_p=0.20, screen_p=0.001)[0],
    ]
    write_csv(
        pair_root / "adaptive_selection" / "selected_adaptive_modules.tsv",
        selected_rows,
        delimiter="\t",
    )
    output = pair_root / "strict_module_chain"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_strict_module_chain.py",
            "--pair-root",
            str(pair_root),
            "--output-dir",
            str(output),
            "--metric-profile",
            "math",
            "--phase",
            "final",
        ],
    )
    strict.main()

    next_token = strict.read_csv(output / "next_token_all_modules.csv")
    eligible = strict.read_csv(
        output / "trajectory_eligible_modules.tsv", delimiter="\t"
    )
    summary = strict.read_json(output / "strict_module_chain_summary.json")
    assert len(next_token) == 2
    assert next_token[0]["next_token_scheduled"] == "True"
    assert next_token[1]["next_token_scheduled"] == "False"
    assert next_token[1]["next_token_complete"] == "False"
    assert len(eligible) == 1
    assert eligible[0]["module_index"] == "0"
    assert summary["next_token_complete"] == 1
    assert summary["trajectory_eligible"] == 1
    assert summary["three_ring_pass"] == 1
    chain = strict.read_csv(output / "strict_module_chain.csv")
    assert "three_ring_p_holm_across_modules" not in chain[0]
    assert "three_ring_p_raw" not in chain[0]


def test_second_ring_uses_separate_five_construct_holm_families(
    tmp_path: Path,
) -> None:
    output = tmp_path / "module"
    rows = []
    special = {
        "interaction_orientation": 0.00900,
        "initial_uncertainty": 0.00940,
    }
    for construct in metric_families("conversation"):
        rows.append(
            {
                "meta_predictor": "signed_direction",
                "metric_family": construct,
                "metrics": "synthetic_metric",
                "mean_conditional_meta_information_bits_per_sample": 0.001,
                "aligned_standardized_slope": 0.1,
                # This diagnostic p-value must not enter the primary test.
                "signflip_one_sided_p": 0.99,
                "aligned_slope_signflip_two_sided_p": special.get(construct, 0.5),
            }
        )
    write_csv(
        output / "continuous_behavior" / "continuous_behavior_family_omnibus.csv",
        rows,
    )

    candidates = strict.behavior_candidates(output, "conversation")
    by_construct = {candidate["construct"]: candidate for candidate in candidates}
    assert by_construct["interaction_orientation"]["construct_group"] == "behavior"
    assert by_construct["initial_uncertainty"]["construct_group"] == "monitoring"
    assert by_construct["interaction_orientation"][
        "holm_p_within_construct_group"
    ] == 0.045
    assert by_construct["initial_uncertainty"][
        "holm_p_within_construct_group"
    ] == 0.047
    assert all(
        candidate["holm_family_size"] == 5 for candidate in candidates
    )
    assert primary_construct_groups("conversation")["behavior"][-1] == (
        "interaction_orientation"
    )
