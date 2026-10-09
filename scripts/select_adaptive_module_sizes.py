#!/usr/bin/env python3
"""Select sparse modules and their support sizes from split-specific diagnostics.

The support-sweep jobs are independent: module_0 from k=4 is not assumed to
be the same latent module as module_0 from k=32.  Candidates are therefore
ranked globally, with per-job and total caps. Continuous semantic R2 and
residual evidence are the semantic safeguards. Strict experiments can rank on
the selection split and reserve held-out rows for downstream evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostics-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-modules", type=int, default=8)
    p.add_argument("--max-per-profile", type=int, default=2)
    p.add_argument(
        "--fill-to-max",
        action="store_true",
        help=(
            "After ranking on the declared selection split, fill the frozen "
            "family to --max-modules with the next best candidates. The later "
            "held-out transmission gate remains authoritative."
        ),
    )
    p.add_argument(
        "--all-heldout-gate",
        action="store_true",
        help=(
            "Export every candidate passing the held-out evidence gate. "
            "This disables module/profile caps and never uses a q-value gate."
        ),
    )
    p.add_argument(
        "--all-candidates",
        action="store_true",
        help=(
            "Export every usable candidate in pre-registered order. Intended "
            "for null controls so failed gates remain observable."
        ),
    )
    p.add_argument("--min-heldout-rf", type=float, default=0.03)
    p.add_argument("--min-gain-ci-low", type=float, default=-0.005)
    p.add_argument("--max-continuous-semantic-r2", type=float, default=0.75)
    p.add_argument("--size-penalty", type=float, default=0.01)
    p.add_argument(
        "--evidence-split",
        choices=["heldout", "selection"],
        default="heldout",
        help=(
            "Split used to rank and gate modules. Use selection for strict "
            "experiments that reserve heldout rows for downstream tests."
        ),
    )
    p.add_argument("--rf-weight", type=float, default=0.40)
    p.add_argument("--gain-weight", type=float, default=0.45)
    p.add_argument("--semantic-weight", type=float, default=0.15)
    return p.parse_args()


def num(row: Dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def scale(value: float, values: List[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite or not math.isfinite(value):
        return 0.0
    lo, hi = min(finite), max(finite)
    if hi <= lo:
        return 0.5
    return (value - lo) / (hi - lo)


def main() -> None:
    cfg = args()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with Path(cfg.diagnostics_csv).open("r", encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    if not rows:
        raise SystemExit("No module diagnostics found.")

    if cfg.evidence_split == "selection":
        rf_key = "selection_rf"
        gain_key = "selection_residual_gain"
        semantic_key = "continuous_semantic_r2_selection"
        gain_low_key = None
    else:
        rf_key = "heldout_rf"
        gain_key = "heldout_residual_gain"
        semantic_key = "continuous_semantic_r2_heldout"
        gain_low_key = "heldout_residual_gain_ci_low"

    rf_values = [num(r, rf_key) for r in rows]
    gain_values = [num(r, gain_key) for r in rows]
    semantic_values = [num(r, semantic_key) for r in rows]

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        rf = num(row, rf_key)
        gain = num(row, gain_key)
        gain_low = num(row, gain_low_key) if gain_low_key else math.nan
        semantic = num(row, semantic_key)
        support = int(float(row.get("support", 0) or 0))
        if not row.get("module_dir") or support <= 0:
            continue
        semantic_ok = (not math.isfinite(semantic)) or semantic <= cfg.max_continuous_semantic_r2
        evidence_ok = math.isfinite(rf) and rf >= cfg.min_heldout_rf
        gain_ok = (not math.isfinite(gain_low)) or gain_low >= cfg.min_gain_ci_low
        eligible = evidence_ok and gain_ok and semantic_ok
        semantic_score = 1.0 - scale(semantic, semantic_values) if math.isfinite(semantic) else 0.5
        score = (
            cfg.rf_weight * scale(rf, rf_values)
            + cfg.gain_weight * scale(gain, gain_values)
            + cfg.semantic_weight * semantic_score
            - cfg.size_penalty * support / 32.0
        )
        item = dict(row)
        item.update({
            "support": support,
            "heldout_rf_num": num(row, "heldout_rf"),
            "heldout_residual_gain_num": num(row, "heldout_residual_gain"),
            "heldout_gain_ci_low_num": num(row, "heldout_residual_gain_ci_low"),
            "continuous_semantic_r2_num": num(
                row, "continuous_semantic_r2_heldout"
            ),
            "adaptive_score": score,
            "adaptive_eligible": eligible,
            "selection_fallback_used": False,
            "evidence_split": cfg.evidence_split,
            "evidence_rf": rf,
            "evidence_residual_gain": gain,
            "evidence_gain_ci_low": gain_low,
            "evidence_continuous_semantic_r2": semantic,
        })
        candidates.append(item)

    if not candidates:
        raise SystemExit("No usable module candidates found.")
    eligible = [r for r in candidates if r["adaptive_eligible"]]
    pool = eligible if eligible else candidates
    fallback = not bool(eligible)
    pool.sort(
        key=lambda r: (float(r["adaptive_score"]), float(r["evidence_rf"])),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    profile_counts: Dict[str, int] = {}
    if cfg.all_candidates:
        # Keep random-null module ids in their pre-registered order. No gate or
        # outcome-derived score selects them, but the same module-count budget
        # as the learned condition still applies.
        ordered = sorted(
            candidates,
            key=lambda r: (
                str(r.get("profile", "")),
                int(float(r.get("module", 0) or 0)),
            ),
        )
        for row in ordered:
            profile = str(row.get("profile", "unknown"))
            if (
                cfg.max_per_profile > 0
                and profile_counts.get(profile, 0) >= cfg.max_per_profile
            ):
                continue
            selected.append(row)
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
            if cfg.max_modules > 0 and len(selected) >= cfg.max_modules:
                break
        fallback = False
    elif cfg.all_heldout_gate:
        if not eligible:
            raise SystemExit(
                "No module passes the held-out gate; refusing an unvalidated fallback."
            )
        selected = list(eligible)
        selected.sort(
            key=lambda r: (float(r["adaptive_score"]), float(r["evidence_rf"])),
            reverse=True,
        )
        for row in selected:
            row["selection_fallback_used"] = False
    else:
        for row in pool:
            profile = str(row.get("profile", "unknown"))
            if cfg.max_per_profile > 0 and profile_counts.get(profile, 0) >= cfg.max_per_profile:
                continue
            row["selection_fallback_used"] = fallback
            selected.append(row)
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
            if cfg.max_modules > 0 and len(selected) >= cfg.max_modules:
                break
    if not selected:
        row = pool[0]
        row["selection_fallback_used"] = True
        selected = [row]
    if cfg.fill_to_max and cfg.max_modules > 0 and len(selected) < cfg.max_modules:
        selected_keys = {
            (str(row.get("profile", "")), str(row.get("module", "")))
            for row in selected
        }
        remaining = sorted(
            candidates,
            key=lambda row: (
                float(row["adaptive_score"]), float(row["evidence_rf"])
            ),
            reverse=True,
        )
        for row in remaining:
            key = (str(row.get("profile", "")), str(row.get("module", "")))
            if key in selected_keys:
                continue
            profile = str(row.get("profile", "unknown"))
            if (
                cfg.max_per_profile > 0
                and profile_counts.get(profile, 0) >= cfg.max_per_profile
            ):
                continue
            row["selection_fallback_used"] = True
            selected.append(row)
            selected_keys.add(key)
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
            if len(selected) >= cfg.max_modules:
                break

    fields = [
        "rank", "profile", "support", "module", "module_dir", "adaptive_score",
        "adaptive_eligible", "selection_fallback_used", "heldout_rf_num",
        "heldout_residual_gain_num", "heldout_gain_ci_low_num",
        "continuous_semantic_r2_num", "evidence_split", "evidence_rf",
        "evidence_residual_gain", "evidence_gain_ci_low",
        "evidence_continuous_semantic_r2",
    ]
    with (out / "selected_adaptive_modules.tsv").open("w", encoding="utf-8", newline="") as h:
        h.write("\t".join(fields) + "\n")
        for rank, row in enumerate(selected, 1):
            values = {
                "rank": rank,
                "profile": row.get("profile", ""),
                "support": row["support"],
                "module": row.get("module", ""),
                "module_dir": row.get("module_dir", ""),
                "adaptive_score": row["adaptive_score"],
                "adaptive_eligible": int(bool(row["adaptive_eligible"])),
                "selection_fallback_used": int(bool(row["selection_fallback_used"])),
                "heldout_rf_num": row["heldout_rf_num"],
                "heldout_residual_gain_num": row["heldout_residual_gain_num"],
                "heldout_gain_ci_low_num": row["heldout_gain_ci_low_num"],
                "continuous_semantic_r2_num": row["continuous_semantic_r2_num"],
                "evidence_split": row["evidence_split"],
                "evidence_rf": row["evidence_rf"],
                "evidence_residual_gain": row["evidence_residual_gain"],
                "evidence_gain_ci_low": row["evidence_gain_ci_low"],
                "evidence_continuous_semantic_r2": row[
                    "evidence_continuous_semantic_r2"
                ],
            }
            h.write("\t".join(str(values[f]) for f in fields) + "\n")

    payload = {
        "selection_policy": (
            f"global {cfg.evidence_split} RF/gain/continuous-semantic tradeoff"
        ),
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "fallback_used": fallback,
        "fill_to_max_used": bool(cfg.fill_to_max),
        "config": vars(cfg),
        "selected": selected,
    }
    (out / "selected_adaptive_modules.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[adaptive-select] candidates={len(candidates)} eligible={len(eligible)} "
        f"selected={len(selected)} fallback={fallback}", flush=True
    )
    for rank, row in enumerate(selected, 1):
        print(
            f"[adaptive-select] #{rank} profile={row.get('profile')} k={row['support']} "
            f"module={row.get('module')} score={row['adaptive_score']:.4f} "
            f"split={row['evidence_split']} RF={row['evidence_rf']:.4f} "
            f"gain={row['evidence_residual_gain']:.4f} "
            f"semR2={row['evidence_continuous_semantic_r2']:.4f}", flush=True
        )


if __name__ == "__main__":
    main()
