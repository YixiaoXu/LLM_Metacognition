#!/usr/bin/env python3
"""Summarize module-specific incremental behavior effects without pooling them."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_finite(row: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = finite(row.get(key))
        if value is not None:
            return value
    return None


def quantile(values: Sequence[float], probability: float) -> Optional[float]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = probability * (len(clean) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def module_identity(run_root: Path, summary_path: Path) -> Tuple[str, str]:
    relative = summary_path.relative_to(run_root)
    parts = relative.parts
    if "continuous_modules" not in parts:
        return "unknown_pair", summary_path.parent.parent.name
    index = parts.index("continuous_modules")
    pair = parts[index - 1] if index > 0 else run_root.name
    module = parts[index + 1] if index + 1 < len(parts) else summary_path.parent.name
    return pair, module


def bonferroni_min_p(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [finite(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return min(1.0, min(values) * len(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a module-by-behavior map from continuous held-out analyses."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted(
        run_root.glob("**/continuous_behavior/continuous_behavior_summary.json")
    )
    if not summary_paths:
        raise FileNotFoundError(
            f"No continuous behavior summaries found below {run_root}"
        )

    module_rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = read_json(summary_path)
        pair, module = module_identity(run_root, summary_path)
        current_family_rows = list(summary.get("module_family_omnibus") or [])
        module_p = bonferroni_min_p(
            current_family_rows, "signflip_one_sided_p"
        )
        best_family_gate = min(
            current_family_rows,
            key=lambda row: finite(row.get("signflip_one_sided_p"))
            if finite(row.get("signflip_one_sided_p")) is not None
            else math.inf,
            default={},
        )
        module_rows.append(
            {
                "model_pair": pair,
                "module": module,
                "n": summary.get("n"),
                "metric_profile": summary.get("metric_profile"),
                "module_omnibus_p": module_p,
                "best_predictor": best_family_gate.get("meta_predictor"),
                "best_gate_family": best_family_gate.get("metric_family"),
                "best_family_conditional_information_bits": first_finite(
                    best_family_gate,
                    "mean_conditional_meta_information_bits_per_sample",
                    "mean_normalized_mse_reduction",
                ),
                "best_family_ci95_low": best_family_gate.get("bootstrap_ci95_low"),
                "best_family_ci95_high": best_family_gate.get("bootstrap_ci95_high"),
                "summary_path": str(summary_path),
            }
        )
        for row in current_family_rows:
            family_rows.append(
                {
                    "model_pair": pair,
                    "module": module,
                    **row,
                }
            )
        for row in summary.get("incremental_results") or []:
            metric_rows.append(
                {
                    "model_pair": pair,
                    "module": module,
                    **row,
                }
            )

    # Keep inference local to each module. Modules are heterogeneous
    # candidates, so this summary uses the original prespecified p-values for
    # descriptive evidence only. q-values are never used as a gate or module
    # selection condition in the current pipeline.
    by_pair: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(module_rows):
        by_pair[str(row["model_pair"])].append(index)

    family_rows_by_module: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in family_rows:
        family_rows_by_module[(str(row["model_pair"]), str(row["module"]))].append(row)

    metric_rows_by_module: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        metric_rows_by_module[(str(row["model_pair"]), str(row["module"]))].append(row)

    for module in module_rows:
        key = (str(module["model_pair"]), str(module["module"]))
        positive_family_pass = any(
            first_finite(
                row,
                "mean_conditional_meta_information_bits_per_sample",
                "mean_normalized_mse_reduction",
            ) is not None
            and first_finite(
                row,
                "mean_conditional_meta_information_bits_per_sample",
                "mean_normalized_mse_reduction",
            ) > 0.0
            and finite(row.get("signflip_one_sided_p")) is not None
            and float(row["signflip_one_sided_p"]) < args.alpha
            for row in family_rows_by_module[key]
        )
        positive_metric_pass = any(
            first_finite(
                row,
                "conditional_meta_information_bits_per_sample",
                "incremental_partial_r2",
            ) is not None
            and first_finite(
                row,
                "conditional_meta_information_bits_per_sample",
                "incremental_partial_r2",
            ) > 0.0
            and first_finite(
                row,
                "conditional_information_one_sided_p",
                "conditional_information_signflip_one_sided_p",
            ) is not None
            and float(
                first_finite(
                    row,
                    "conditional_information_one_sided_p",
                    "conditional_information_signflip_one_sided_p",
                )
            ) < args.alpha
            for row in metric_rows_by_module[key]
        )
        module["within_module_family_evidence_pass"] = bool(positive_family_pass)
        module["within_module_metric_evidence_pass"] = bool(positive_metric_pass)
        module["within_module_evidence_pass"] = bool(
            positive_family_pass or positive_metric_pass
        )

    module_lookup = {
        (str(row["model_pair"]), str(row["module"])): row for row in module_rows
    }
    for row in family_rows:
        module = module_lookup[(str(row["model_pair"]), str(row["module"]))]
        row["within_module_evidence_pass"] = module.get(
            "within_module_evidence_pass", False
        )
        row["within_module_family_evidence_pass"] = module.get(
            "within_module_family_evidence_pass", False
        )
        family_p = finite(row.get("signflip_one_sided_p"))
        effect = first_finite(
            row,
            "mean_conditional_meta_information_bits_per_sample",
            "mean_normalized_mse_reduction",
        )
        row["confirmatory_family_pass"] = bool(
            row["within_module_evidence_pass"]
            and family_p is not None
            and family_p < args.alpha
            and effect is not None
            and effect > 0.0
        )

    behavior_map: List[Dict[str, Any]] = []
    for module in module_rows:
        key = (str(module["model_pair"]), str(module["module"]))
        candidates = [
            row
            for row in family_rows
            if (str(row["model_pair"]), str(row["module"])) == key
            and (
                first_finite(
                    row,
                    "mean_conditional_meta_information_bits_per_sample",
                    "mean_normalized_mse_reduction",
                )
                or -math.inf
            ) > 0.0
        ]
        best_family = min(
            candidates,
            key=lambda row: finite(row.get("signflip_one_sided_p"))
            if finite(row.get("signflip_one_sided_p")) is not None
            else math.inf,
            default={},
        )
        metric_candidates = [
            row
            for row in metric_rows
            if (str(row["model_pair"]), str(row["module"])) == key
            and (
                first_finite(
                    row,
                    "conditional_meta_information_bits_per_sample",
                    "incremental_partial_r2",
                )
                or -math.inf
            ) > 0.0
        ]
        best_metric = min(
            metric_candidates,
            key=lambda row: first_finite(
                row,
                "conditional_information_one_sided_p",
                "conditional_information_signflip_one_sided_p",
            )
            if first_finite(
                row,
                "conditional_information_one_sided_p",
                "conditional_information_signflip_one_sided_p",
            ) is not None
            else math.inf,
            default={},
        )
        if module.get("within_module_metric_evidence_pass"):
            tier = "within_module_metric_significant"
        elif module.get("within_module_family_evidence_pass"):
            tier = "within_module_family_significant"
        else:
            tier = "exploratory"
        behavior_map.append(
            {
                **module,
                "evidence_tier": tier,
                "best_behavior_family": best_family.get("metric_family"),
                "best_family_predictor": best_family.get("meta_predictor"),
                "best_family_conditional_information_bits": first_finite(
                    best_family,
                    "mean_conditional_meta_information_bits_per_sample",
                    "mean_normalized_mse_reduction",
                ),
                "best_family_p": best_family.get("signflip_one_sided_p"),
                "best_family_q": best_family.get("within_module_family_fdr_q"),
                "localized_metric": best_metric.get("metric"),
                "localized_metric_predictor": best_metric.get("meta_predictor"),
                "localized_metric_conditional_information_bits": first_finite(
                    best_metric,
                    "conditional_meta_information_bits_per_sample",
                    "incremental_partial_r2",
                ),
                "localized_metric_partial_r2": best_metric.get(
                    "incremental_partial_r2"
                ),
                "localized_metric_p": first_finite(
                    best_metric,
                    "conditional_information_one_sided_p",
                    "conditional_information_signflip_one_sided_p",
                ),
                "localized_metric_q": best_metric.get("fdr_q"),
            }
        )

    matrix_rows: List[Dict[str, Any]] = []
    for module in module_rows:
        key = (str(module["model_pair"]), str(module["module"]))
        output: Dict[str, Any] = {
            "model_pair": key[0],
            "module": key[1],
            "within_module_evidence_pass": module.get(
                "within_module_evidence_pass"
            ),
            "within_module_family_evidence_pass": module.get(
                "within_module_family_evidence_pass"
            ),
            "within_module_metric_evidence_pass": module.get(
                "within_module_metric_evidence_pass"
            ),
        }
        for row in metric_rows:
            if (str(row["model_pair"]), str(row["module"])) != key:
                continue
            prefix = f"{row.get('meta_predictor')}__{row.get('metric')}"
            output[f"{prefix}__conditional_information_bits"] = row.get(
                "conditional_meta_information_bits_per_sample"
            )
            output[f"{prefix}__shapley_meta_bits"] = row.get(
                "shapley_meta_information_bits_per_sample"
            )
            output[f"{prefix}__partial_r2"] = row.get("incremental_partial_r2")
            output[f"{prefix}__q"] = row.get("fdr_q")
        matrix_rows.append(output)

    heterogeneity_rows: List[Dict[str, Any]] = []
    grouped_metrics: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped_metrics[(str(row.get("meta_predictor")), str(row.get("metric")))].append(row)
    for (predictor, metric), rows in sorted(grouped_metrics.items()):
        effects = [
            first_finite(
                row,
                "conditional_meta_information_bits_per_sample",
                "incremental_partial_r2",
            )
            for row in rows
        ]
        effects = [effect for effect in effects if effect is not None]
        heterogeneity_rows.append(
            {
                "meta_predictor": predictor,
                "metric": metric,
                "module_count": len(effects),
                "positive_module_count": sum(effect > 0.0 for effect in effects),
                "positive_module_rate": (
                    sum(effect > 0.0 for effect in effects) / len(effects)
                    if effects
                    else None
                ),
                "metric_p_lt_0p05_count": sum(
                    p_value is not None and p_value < args.alpha for p_value in [
                        first_finite(
                            row,
                            "conditional_information_one_sided_p",
                            "conditional_information_signflip_one_sided_p",
                        )
                        for row in rows
                    ]
                ),
                "median_conditional_information_bits_descriptive": quantile(
                    effects, 0.5
                ),
                "conditional_information_bits_q25": quantile(effects, 0.25),
                "conditional_information_bits_q75": quantile(effects, 0.75),
                "interpretation": (
                    "descriptive heterogeneity only; modules need not share an effect"
                ),
            }
        )

    write_csv(output_dir / "module_any_family_gates.csv", module_rows)
    write_csv(output_dir / "module_family_omnibus.csv", family_rows)
    write_csv(output_dir / "module_behavior_effects_long.csv", metric_rows)
    write_csv(output_dir / "module_behavior_effect_matrix.csv", matrix_rows)
    write_csv(output_dir / "module_behavior_map.csv", behavior_map)
    write_csv(output_dir / "metric_heterogeneity_descriptive.csv", heterogeneity_rows)

    summary = {
        "run_root": str(run_root),
        "module_count": len(module_rows),
        "model_pair_count": len(by_pair),
        "within_module_evidence_pass_count": sum(
            bool(row.get("within_module_evidence_pass")) for row in module_rows
        ),
        "within_module_significant_module_count": sum(
            row.get("evidence_tier")
            in {
                "within_module_metric_significant",
                "within_module_family_significant",
            }
            for row in behavior_map
        ),
        "within_module_family_significant_module_count": sum(
            bool(row.get("within_module_family_evidence_pass"))
            for row in module_rows
        ),
        "within_module_metric_significant_module_count": sum(
            bool(row.get("within_module_metric_evidence_pass"))
            for row in module_rows
        ),
        "primary_analysis": (
            "module-specific within-module family and metric evidence, then "
            "metric localization; cross-module averages are descriptive only"
        ),
        "module_multiple_testing": (
            "Current module gates use held-out evidence and original p-values; "
            "q-values are retained only in legacy diagnostic files and are not "
            "used for inference, selection, or downstream execution"
        ),
        "interpretation_warning": (
            "Modules passing the held-out gate are retained. Since modules were "
            "searched and compared, the strongest module-level finding remains "
            "exploratory until replicated on an independent held-out dataset "
            "with a frozen module and outcome hypothesis."
        ),
        "outputs": {
            "module_behavior_map": "module_behavior_map.csv",
            "module_effect_matrix": "module_behavior_effect_matrix.csv",
            "family_omnibus": "module_family_omnibus.csv",
            "heterogeneity": "metric_heterogeneity_descriptive.csv",
        },
    }
    (output_dir / "module_behavior_map_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_plots and metric_rows:
        try:
            import matplotlib.pyplot as plt

            columns = sorted(
                {(str(row["meta_predictor"]), str(row["metric"])) for row in metric_rows}
            )
            modules = [(str(row["model_pair"]), str(row["module"])) for row in module_rows]
            values = []
            effect_lookup = {
                (
                    str(row["model_pair"]),
                    str(row["module"]),
                    str(row["meta_predictor"]),
                    str(row["metric"]),
                ): first_finite(
                    row,
                    "conditional_meta_information_bits_per_sample",
                    "incremental_partial_r2",
                )
                for row in metric_rows
            }
            for pair, module in modules:
                values.append(
                    [
                        effect_lookup.get((pair, module, predictor, metric), math.nan)
                        for predictor, metric in columns
                    ]
                )
            figure, axis = plt.subplots(
                figsize=(max(10.0, 0.48 * len(columns)), max(5.0, 0.25 * len(modules)))
            )
            finite_values = [
                abs(float(value))
                for row in values
                for value in row
                if math.isfinite(float(value))
            ]
            limit = max(finite_values, default=0.01)
            image = axis.imshow(
                values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
            )
            axis.set_xticks(
                range(len(columns)),
                [f"{predictor}\n{metric}" for predictor, metric in columns],
                rotation=55,
                ha="right",
            )
            axis.set_yticks(
                range(len(modules)),
                [f"{pair} / {module}" for pair, module in modules],
            )
            axis.set_title("Module-specific conditional predictive information")
            figure.colorbar(image, ax=axis, label="Bits per held-out sample")
            figure.tight_layout()
            figure.savefig(output_dir / "module_behavior_effect_matrix.png", dpi=220)
            plt.close(figure)
        except ImportError:
            pass

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
