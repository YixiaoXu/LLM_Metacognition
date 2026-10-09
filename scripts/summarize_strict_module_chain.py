#!/usr/bin/env python3
"""Freeze and summarize the three-stage evidence chain for each module."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import _bootstrap  # noqa: F401  # Direct ``python scripts/...`` execution.
except ModuleNotFoundError:  # Imported as ``scripts.summarize_strict_module_chain``.
    from scripts import _bootstrap  # type: ignore  # noqa: F401

from metacog.evaluation import (
    primary_construct_count,
    primary_construct_groups,
    profile_names,
)


def number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_csv(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm(values: List[float]) -> List[float]:
    n = len(values)
    order = sorted(range(n), key=lambda index: values[index])
    adjusted = [1.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (n - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def behavior_candidates(output: Path, profile: str) -> List[Dict[str, Any]]:
    construct_groups = primary_construct_groups(profile)
    group_by_construct = {
        construct: group
        for group, constructs in construct_groups.items()
        for construct in constructs
    }
    rows = read_csv(
        output / "continuous_behavior" / "continuous_behavior_family_omnibus.csv"
    )
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("meta_predictor") != "signed_direction":
            continue
        bits = number(row.get("mean_conditional_meta_information_bits_per_sample"))
        slope = number(row.get("aligned_standardized_slope"))
        information_p = number(row.get("signflip_one_sided_p"), 1.0)
        direction_p = number(row.get("aligned_slope_signflip_two_sided_p"), 1.0)
        valid = bits > 0 and math.isfinite(slope) and slope != 0
        candidates.append(
            {
                "construct": row.get("metric_family", ""),
                "construct_group": group_by_construct.get(
                    row.get("metric_family", ""), "unclassified"
                ),
                "component_metrics": row.get("metrics", ""),
                "conditional_bits": bits,
                "aligned_slope": slope,
                "aligned_direction": "positive" if slope > 0 else "negative",
                "information_p": information_p,
                "direction_p": direction_p,
                # Positive conditional information is an effect gate, not a
                # second null-hypothesis p-value.  The sole primary p-value is
                # the semantic-residualized directional slope test.
                "primary_direction_p": direction_p if valid else 1.0,
                "positive_information_gate": bits > 0,
                "valid_primary_test": valid,
            }
        )

    # Correct the two prespecified scientific families independently. Missing
    # constructs enter as p=1 so an incomplete output cannot receive a more
    # permissive correction by silently shrinking the family.
    by_construct: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        by_construct.setdefault(candidate["construct"], []).append(candidate)
    for group, expected_constructs in construct_groups.items():
        primary_ps = [
            (
                by_construct[construct][0]["primary_direction_p"]
                if len(by_construct.get(construct, [])) == 1
                else 1.0
            )
            for construct in expected_constructs
        ]
        adjusted = holm(primary_ps)
        for construct, value in zip(expected_constructs, adjusted):
            for candidate in by_construct.get(construct, []):
                candidate["holm_p_within_construct_group"] = value
                candidate["holm_family_size"] = len(expected_constructs)
                # Compatibility alias for historical notebooks. Its meaning is
                # now the prespecified five-construct group, not all ten tests.
                candidate["holm_p_within_module"] = value
    for candidate in candidates:
        candidate.setdefault("holm_p_within_construct_group", 1.0)
        candidate.setdefault("holm_family_size", 0)
        candidate.setdefault("holm_p_within_module", 1.0)
    return candidates


def next_token_screen(output: Path, role: str, alpha: float) -> Dict[str, Any]:
    rows = read_csv(output / role / "continuous_signed_dose_response.csv")
    by_metric = {row.get("metric", ""): row for row in rows}
    projection = by_metric.get(
        "next_token_auto_logit_delta_projection_to_target", {}
    )
    margin = by_metric.get(
        "next_token_auto_logit_margin_delta_toward_target", {}
    )
    projection_mean = number(projection.get("mean"))
    projection_p = number(projection.get("signflip_two_sided_p"), 1.0)
    margin_mean = number(margin.get("mean"))
    margin_p = number(margin.get("signflip_two_sided_p"), 1.0)
    return {
        "available": bool(rows),
        "projection_mean": projection_mean,
        "projection_p": projection_p,
        "margin_mean": margin_mean,
        "margin_p": margin_p,
        "directional_pass": projection_mean > 0 and projection_p < alpha,
    }


def persistent_candidate(
    output: Path, construct: str, association_direction: str
) -> Optional[Dict[str, Any]]:
    rows = read_csv(
        output / "trajectory_analysis" / "construct_dose_response.csv"
    )
    matching = [
        row
        for row in rows
        if row.get("construct") == construct
        and row.get("comparison") == "positive_minus_negative"
        and row.get("population") == "all_modes_untruncated"
    ]
    if not matching:
        return None
    row = matching[0]
    effect = number(row.get("aligned_effect_per_dose"))
    p = number(row.get("directional_one_sided_p"), 1.0)
    expected_sign = 1.0 if association_direction == "positive" else -1.0
    direction_ok = math.isfinite(effect) and effect * expected_sign > 0
    return {
        "effect": effect,
        "p": p if direction_ok else 1.0,
        "direction_ok": direction_ok,
        "n": int(number(row.get("n"), 0)),
        "dose_levels": row.get("dose_levels", ""),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], delimiter: str = ",") -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-profile", choices=profile_names(), required=True)
    parser.add_argument("--phase", choices=["eligibility", "final"], default="final")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    pair_root = Path(args.pair_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = pair_root / "adaptive_selection" / "selected_adaptive_modules.tsv"
    selected = read_csv(selected_path, delimiter="\t")
    if not selected:
        raise ValueError(f"No frozen modules found in {selected_path}")

    rows: List[Dict[str, Any]] = []
    eligibility_rows: List[Dict[str, Any]] = []
    next_token_rows: List[Dict[str, Any]] = []
    stage2_rows: List[Dict[str, Any]] = []
    expected_constructs = primary_construct_count(args.metric_profile)
    construct_groups = primary_construct_groups(args.metric_profile)
    expected_construct_names = {
        construct
        for constructs in construct_groups.values()
        for construct in constructs
    }
    for module_index, selected_row in enumerate(selected):
        rank = int(float(selected_row["rank"]))
        support = int(float(selected_row["support"]))
        module = int(float(selected_row["module"]))
        module_key = f"rank_{rank}_k{support}_m{module}"
        output = pair_root / "continuous_modules" / module_key
        module_dir = Path(selected_row["module_dir"])
        summary_path = module_dir / "module_summary.json"
        association_path = (
            output
            / "continuous_behavior"
            / "continuous_behavior_family_omnibus.csv"
        )
        summary = read_json(summary_path)
        information = summary.get("information", {})

        transmission_p = number(
            information.get("residual_gain_signflip_one_sided_p"), 1.0
        )
        transmission_lcb = number(
            information.get("residual_gain_bootstrap_ci95_low")
        )
        transmission_pass = transmission_lcb > 0 and transmission_p < args.alpha

        candidates = behavior_candidates(output, args.metric_profile)
        behavior = (
            min(
                candidates,
                key=lambda item: item["holm_p_within_construct_group"],
            )
            if candidates
            else None
        )
        behavior_p = (
            behavior["holm_p_within_construct_group"] if behavior else 1.0
        )
        candidate_construct_names = [candidate["construct"] for candidate in candidates]
        construct_family_complete = (
            len(candidate_construct_names) == expected_constructs
            and set(candidate_construct_names) == expected_construct_names
        )
        behavior_pass = (
            behavior is not None
            and construct_family_complete
            and behavior["positive_information_gate"]
            and behavior_p < args.alpha
        )
        for candidate in candidates:
            stage2_rows.append(
                {
                    "module_index": module_index,
                    "module_key": module_key,
                    "rank": rank,
                    "support": support,
                    "module": module,
                    "construct_group": candidate["construct_group"],
                    "construct": candidate["construct"],
                    "component_metrics": candidate["component_metrics"],
                    "conditional_bits": candidate["conditional_bits"],
                    "positive_information_gate": candidate[
                        "positive_information_gate"
                    ],
                    "aligned_slope": candidate["aligned_slope"],
                    "aligned_direction": candidate["aligned_direction"],
                    "direction_p_raw": candidate["direction_p"],
                    "direction_p_holm_within_construct_group": candidate[
                        "holm_p_within_construct_group"
                    ],
                    "holm_family_size": candidate["holm_family_size"],
                    "information_p_diagnostic_only": candidate["information_p"],
                    "selected_primary_construct": candidate is behavior,
                    "construct_pass": (
                        candidate["positive_information_gate"]
                        and candidate["holm_p_within_construct_group"] < args.alpha
                    ),
                    "construct_family_complete": construct_family_complete,
                }
            )
        trajectory_eligible = transmission_pass and behavior_pass

        if trajectory_eligible:
            screen_a = next_token_screen(output, "causal_screen_a", args.alpha)
            screen_b = next_token_screen(output, "causal_screen_b", args.alpha)
        else:
            # Ignore stale intervention files from an older all-module run.
            # Non-eligible modules do not enter the third ring.
            screen_a = {
                "available": False,
                "projection_mean": math.nan,
                "projection_p": 1.0,
                "margin_mean": math.nan,
                "margin_p": 1.0,
                "directional_pass": False,
            }
            screen_b = dict(screen_a)
        next_token_complete = screen_a["available"] and screen_b["available"]
        next_token_replicated = (
            next_token_complete
            and screen_a["directional_pass"]
            and screen_b["directional_pass"]
        )
        # The third ring is an intersection-union replication test: both
        # independently held-out screens must move the next-token distribution
        # toward the precomputed target direction. Taking the larger p-value is
        # already conservative for this conjunction and does not require an
        # additional A/B multiplicity correction.
        next_token_p = (
            max(screen_a["projection_p"], screen_b["projection_p"])
            if next_token_replicated
            else 1.0
        )
        next_token_rows.append(
            {
                "module_index": module_index,
                "module_key": module_key,
                "next_token_scheduled": trajectory_eligible,
                "screen_a_projection_mean": screen_a["projection_mean"],
                "screen_a_projection_p": screen_a["projection_p"],
                "screen_b_projection_mean": screen_b["projection_mean"],
                "screen_b_projection_p": screen_b["projection_p"],
                "screen_a_margin_mean": screen_a["margin_mean"],
                "screen_a_margin_p": screen_a["margin_p"],
                "screen_b_margin_mean": screen_b["margin_mean"],
                "screen_b_margin_p": screen_b["margin_p"],
                "next_token_complete": next_token_complete,
                "next_token_replicated_directional_pass": next_token_replicated,
                "next_token_replication_p": next_token_p,
            }
        )

        persistent = (
            persistent_candidate(
                output,
                behavior["construct"],
                behavior["aligned_direction"],
            )
            if trajectory_eligible and behavior is not None
            else None
        )
        persistent_p = persistent["p"] if persistent else 1.0
        persistent_pass = (
            trajectory_eligible
            and persistent is not None
            and persistent["direction_ok"]
            and persistent_p < args.alpha
        )
        three_ring_pass = (
            transmission_pass and behavior_pass and next_token_replicated
        )
        row = {
            "module_index": module_index,
            "rank": rank,
            "support": support,
            "module": module,
            "module_key": module_key,
            "module_dir": str(module_dir),
            "module_summary_sha256": sha256(summary_path),
            "association_family_sha256": sha256(association_path),
            "transmission_gain": number(information.get("residual_gain_mse")),
            "transmission_lcb": transmission_lcb,
            "transmission_p": transmission_p,
            "transmission_pass": transmission_pass,
            "construct_count_tested": len(candidates),
            "construct_count_expected": expected_constructs,
            "construct_family_complete": construct_family_complete,
            "primary_construct_group": behavior["construct_group"] if behavior else "",
            "primary_behavior_construct": behavior["construct"] if behavior else "",
            "primary_component_metrics": behavior["component_metrics"] if behavior else "",
            "conditional_bits": behavior["conditional_bits"] if behavior else math.nan,
            "aligned_behavior_slope": behavior["aligned_slope"] if behavior else math.nan,
            "aligned_behavior_direction": behavior["aligned_direction"] if behavior else "",
            "behavior_information_p": behavior["information_p"] if behavior else 1.0,
            "behavior_direction_p": behavior["direction_p"] if behavior else 1.0,
            "behavior_p_holm_within_construct_group": behavior_p,
            # Compatibility alias; correction now covers five constructs in
            # the selected behavior/monitoring group, not all ten constructs.
            "behavior_p_holm_within_module": behavior_p,
            "behavior_pass": behavior_pass,
            "trajectory_eligible": trajectory_eligible,
            "next_token_complete": next_token_complete,
            "next_token_replicated_directional_pass": next_token_replicated,
            "next_token_replication_p": next_token_p,
            "persistent_effect": persistent["effect"] if persistent else math.nan,
            "persistent_direction_ok": persistent["direction_ok"] if persistent else False,
            "persistent_p": persistent_p if persistent else math.nan,
            "persistent_pass": persistent_pass,
            "three_ring_pass": three_ring_pass,
            # Backward-compatible decision alias. There is no additional
            # cross-module test after the three prespecified rings.
            "complete_chain_pass": three_ring_pass,
            "three_ring_status": (
                "complete"
                if next_token_complete
                else "pending_next_token"
                if trajectory_eligible
                else "stages_1_2_not_both_passed"
            ),
            "trajectory_status": (
                "complete"
                if persistent is not None
                else "eligible_pending_trajectory"
                if trajectory_eligible
                else "not_trajectory_eligible"
            ),
        }
        rows.append(row)
        if trajectory_eligible:
            eligibility_rows.append(
                {
                    "module_index": module_index,
                    "module_key": module_key,
                    "rank": rank,
                    "support": support,
                    "module": module,
                    "module_dir": str(module_dir),
                    "output_dir": str(output),
                    "primary_behavior_construct": behavior["construct"],
                    "primary_component_metrics": behavior["component_metrics"],
                    "aligned_behavior_direction": behavior["aligned_direction"],
                    # Keep the legacy positional columns above unchanged: the
                    # Bash trajectory scheduler reads them as a TSV tuple.
                    "primary_construct_group": behavior["construct_group"],
                    "aligned_behavior_slope": behavior["aligned_slope"],
                    "transmission_p": transmission_p,
                    "behavior_p_holm_within_construct_group": behavior_p,
                    "behavior_p_holm_within_module": behavior_p,
                    "module_summary_sha256": sha256(summary_path),
                    "association_family_sha256": sha256(association_path),
                }
            )

    write_csv(output_dir / "strict_module_chain.csv", rows)
    write_csv(output_dir / "next_token_all_modules.csv", next_token_rows)
    write_csv(output_dir / "stage2_construct_tests.csv", stage2_rows)
    write_csv(
        output_dir / "trajectory_eligible_modules.tsv",
        eligibility_rows,
        delimiter="\t",
    )
    payload = {
        "phase": args.phase,
        "pair_root": str(pair_root),
        "metric_profile": args.metric_profile,
        "module_count": len(rows),
        "constructs_per_module": expected_constructs,
        "construct_groups": construct_groups,
        "stage2_test_contract": (
            "conditional information must be positive; the sole primary p-value "
            "is the semantic-residualized directional slope test; Holm correction "
            "is applied separately to the five behavior and five monitoring "
            "constructs; either group may pass"
        ),
        "transmission_pass": sum(bool(row["transmission_pass"]) for row in rows),
        "behavior_pass": sum(bool(row["behavior_pass"]) for row in rows),
        "trajectory_eligible": len(eligibility_rows),
        "next_token_complete": sum(bool(row["next_token_complete"]) for row in rows),
        "next_token_replicated_directional_pass": sum(
            bool(row["next_token_replicated_directional_pass"]) for row in rows
        ),
        "persistent_pass": sum(bool(row["persistent_pass"]) for row in rows),
        "three_ring_pass": sum(bool(row["three_ring_pass"]) for row in rows),
        "complete_chain_pass": sum(bool(row["complete_chain_pass"]) for row in rows),
        "selection_contract": (
            "persistent generation eligibility uses only held-out transmission and "
            "a positive-information association passing Holm correction within "
            "one prespecified five-construct behavior or monitoring group; all "
            "eligible modules receive independent next-token A/B audits; a module "
            "passes when each of the three prespecified rings passes, with no "
            "additional cross-module significance test; persistent generation is "
            "a focused trajectory validation"
        ),
        "rows": rows,
    }
    (output_dir / "strict_module_chain_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "module_count",
                    "transmission_pass",
                    "behavior_pass",
                    "trajectory_eligible",
                    "next_token_complete",
                    "three_ring_pass",
                    "persistent_pass",
                    "complete_chain_pass",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
