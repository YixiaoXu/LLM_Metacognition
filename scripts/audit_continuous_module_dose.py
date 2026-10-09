#!/usr/bin/env python
"""Run continuous refined-code dose interventions through the mature core runner.

Low/high tail labels orient the axis only. The scientific treatment is a
signed continuous dose measured in training residual-code standard deviations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import torch

import cluster_intervention as core


AXIS: Dict[str, Any] = {}
EVALUATION_ROLE = "association_confirmatory"
CONTROL_MODES: Tuple[str, ...] = ("opposite_direction",)
CONTINUOUS_DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1),)
LAST_SPECS: List[core.InterventionSpec] = []
ORIGINAL_AVAILABLE_IDS = core.fixed_cluster_available_ids
ORIGINAL_APPLY_MASK = core.build_apply_mask_for_batch
CONTINUOUS_LOGIT_PROTOTYPE = False
CONTINUOUS_POPULATION = "full"
AXIS_SCORE_BY_ID: Dict[str, float] = {}

# Public core flags accepted by the cleaned continuous runner. The underlying
# mature generation engine contains private compatibility code, but historical
# intervention modes cannot be re-enabled through this entry point.
ACTIVE_CORE_FLAGS = {
    "--activation-dir", "--alphas", "--answer-extraction",
    "--answer-tolerance", "--baseline-file", "--baseline-only", "--batch-size",
    "--chat-template-enable-thinking", "--clear-cuda-cache-between-modes",
    "--clear-cuda-cache-every", "--log-cuda-memory",
    "--cluster-sample-strategy", "--cluster-source", "--data",
    "--decoupler-dir", "--device-map", "--dtype", "--eval-scope",
    "--evaluation-id-file", "--fixed-cluster-assignments",
    "--fixed-cluster-features", "--fixed-cluster-ids-only",
    "--generated-patch-steps", "--id-regex", "--intervention-space",
    "--generation-progress-seconds", "--generation-checkpoint-file",
    "--max-delta-rel-norm",
    "--max-length", "--max-new-tokens", "--max-samples",
    "--max-samples-per-cluster", "--max-truncated-rate", "--model-path",
    "--next-token-audit", "--next-token-auto-logit-top-k",
    "--next-token-auto-min-mean-prob", "--next-token-auto-min-selected",
    "--next-token-bootstrap-samples", "--next-token-permutation-tests",
    "--next-token-prototype-file", "--next-token-top-k", "--num-clusters",
    "--metric-profile", "--no-plots", "--output-dir", "--prompt-style",
    "--record-layers", "--record-prompt-confidence",
    "--refined-code-max-semantic-rel-delta",
    "--refined-direct-hidden-penalty",
    "--refined-direct-step-scale", "--refined-direct-steps",
    "--refined-code-semantic-penalty", "--refined-code-z1-penalty",
    "--refined-code-dose-match-controls",
    "--refined-code-dose-match-tolerance",
    "--refined-code-dose-hidden-tolerance",
    "--refined-code-dose-min-direction-cosine",
    "--refined-code-dose-semantic-tolerance",
    "--refined-code-random-dose-mode",
    "--refined-code-random-axis-penalty",
    "--refined-code-random-max-real-axis-cosine",
    "--refined-code-random-hidden-candidates",
    "--refined-module-dir",
    "--save-generated-token-ids", "--save-generated-token-logprobs", "--seed",
    "--style-bootstrap-samples", "--style-permutation-tests",
    "--style-untruncated-only", "--target-file", "--temperature", "--top-p",
    "--trust-remote-code", "--use-chat-template",
}


def safe_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ContinuousDoseIntervention(core.RefinedResidualCodeIntervention):
    """Move the current code by a fixed signed dose along the module axis."""

    def _axis_direction(self, current_code: torch.Tensor) -> torch.Tensor:
        axis = torch.as_tensor(AXIS["axis"], device=current_code.device).to(
            current_code.dtype
        )
        axis = axis.flatten()
        if axis.numel() != current_code.size(1):
            raise ValueError(
                f"Continuous axis dimension {axis.numel()} != runtime code "
                f"dimension {current_code.size(1)}."
            )
        axis = axis / axis.norm().clamp_min(1e-8)
        centers = self.code_centroids.to(current_code)
        centroid_direction = (
            centers[self.target_cluster] - centers[self.source_cluster]
        )
        orientation = torch.sign((centroid_direction * axis).sum())
        if float(orientation.abs().item()) < 0.5:
            orientation = torch.tensor(
                1.0 if self.target_cluster > self.source_cluster else -1.0,
                device=current_code.device,
                dtype=current_code.dtype,
            )
        direction = axis * orientation
        if self.control == "opposite_direction":
            direction = -direction
        return direction.view(1, -1)

    def _target_code(
        self,
        current_code: torch.Tensor,
        dose_magnitudes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        direction = self._axis_direction(current_code)
        if dose_magnitudes is not None:
            magnitude = dose_magnitudes.view(-1, 1).to(current_code)
        else:
            residual_std = float(torch.as_tensor(AXIS["raw_residual_std"]).item())
            magnitude = torch.full(
                (current_code.size(0), 1),
                abs(float(self.alpha)) * residual_std,
                device=current_code.device,
                dtype=current_code.dtype,
            )
        return current_code + magnitude * direction


class ContinuousLogitAuditState(core.NextTokenAuditState):
    """Discover next-token directions by regressing log-probability on axis score."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continuous_n = 0
        self.continuous_score_sum = 0.0
        self.continuous_score_square_sum = 0.0
        self.continuous_log_prob_sum: torch.Tensor | None = None
        self.continuous_score_log_prob_sum: torch.Tensor | None = None
        self.continuous_prob_sum: torch.Tensor | None = None

    def record_baseline(
        self,
        ids: Sequence[str],
        labels: Sequence[int],
        logits: torch.Tensor,
    ) -> None:
        super().record_baseline(ids, labels, logits)
        if self.external_prototypes or not CONTINUOUS_LOGIT_PROTOTYPE:
            return
        logits_cpu = logits.detach().float().cpu()
        log_prob = torch.log_softmax(logits_cpu, dim=-1)
        probability = log_prob.exp()
        valid_rows = [
            index
            for index, sample_id in enumerate(ids)
            if str(sample_id) in AXIS_SCORE_BY_ID
        ]
        if not valid_rows:
            return
        index = torch.tensor(valid_rows, dtype=torch.long)
        scores = torch.tensor(
            [AXIS_SCORE_BY_ID[str(ids[row])] for row in valid_rows],
            dtype=torch.float32,
        )
        selected_log_prob = log_prob.index_select(0, index)
        selected_probability = probability.index_select(0, index)
        if self.continuous_log_prob_sum is None:
            self.continuous_log_prob_sum = torch.zeros(
                logits_cpu.size(1), dtype=torch.float32
            )
            self.continuous_score_log_prob_sum = torch.zeros_like(
                self.continuous_log_prob_sum
            )
            self.continuous_prob_sum = torch.zeros_like(
                self.continuous_log_prob_sum
            )
        self.continuous_n += scores.numel()
        self.continuous_score_sum += float(scores.sum().item())
        self.continuous_score_square_sum += float(scores.square().sum().item())
        self.continuous_log_prob_sum.add_(selected_log_prob.sum(dim=0))
        assert self.continuous_score_log_prob_sum is not None
        assert self.continuous_prob_sum is not None
        self.continuous_score_log_prob_sum.add_(
            (scores.view(-1, 1) * selected_log_prob).sum(dim=0)
        )
        self.continuous_prob_sum.add_(selected_probability.sum(dim=0))

    def finalize(self) -> Dict[str, Any]:
        if (
            self.external_prototypes
            or not CONTINUOUS_LOGIT_PROTOTYPE
            or self.continuous_n < 8
            or self.continuous_log_prob_sum is None
        ):
            return super().finalize()
        assert self.continuous_score_log_prob_sum is not None
        assert self.continuous_prob_sum is not None
        count = float(self.continuous_n)
        mean_score = self.continuous_score_sum / count
        score_variance = max(
            self.continuous_score_square_sum / count - mean_score**2,
            1e-8,
        )
        mean_log_prob = self.continuous_log_prob_sum / count
        mean_probability = self.continuous_prob_sum / count
        covariance = (
            self.continuous_score_log_prob_sum / count
            - mean_score * mean_log_prob
        )
        slope = covariance / score_variance
        token_score = slope.abs() * mean_probability.sqrt()
        eligible = mean_probability >= self.min_mean_probability
        ranked_score = token_score.masked_fill(~eligible, float("-inf"))
        special_ids = {
            int(value)
            for value in getattr(self.tokenizer, "all_special_ids", [])
        }
        candidate_count = min(
            ranked_score.numel(),
            max(self.top_k * 80, self.top_k + 1024, 4096),
        )
        candidates = torch.topk(
            ranked_score, k=max(candidate_count, 1)
        ).indices.tolist()
        selected = []
        rows = []
        for token_id in candidates:
            token_id = int(token_id)
            if token_id in special_ids:
                continue
            text = self.tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )
            if not text.strip():
                continue
            selected.append(token_id)
            rows.append(
                {
                    "rank": len(selected),
                    "token_id": token_id,
                    "token_text": text,
                    "continuous_log_prob_slope": float(slope[token_id].item()),
                    "discriminative_score": float(
                        token_score[token_id].item()
                    ),
                    "discriminative_score_mode": (
                        "absolute_continuous_logprob_slope_x_sqrt_mean_prob"
                    ),
                    "mean_probability": float(
                        mean_probability[token_id].item()
                    ),
                    "mean_log_prob_at_axis_zero": float(
                        mean_log_prob[token_id].item()
                    ),
                }
            )
            if len(selected) >= self.top_k:
                break
        self.token_ids = torch.tensor(selected, dtype=torch.long)
        if self.token_ids.numel():
            selected_mean = mean_log_prob.index_select(0, self.token_ids)
            selected_slope = slope.index_select(0, self.token_ids)
            self.prototypes = torch.stack(
                [selected_mean - selected_slope, selected_mean + selected_slope],
                dim=0,
            )
        else:
            self.prototypes = torch.empty(self.num_clusters, 0)
        self.token_rows = rows
        return {
            "enabled": bool(self.token_ids.numel()),
            "prototype_source": "continuous_log_probability_regression",
            "prototype_kind": "axis_minus1sd_vs_plus1sd",
            "baseline_sample_count": len(self.baseline_logits),
            "continuous_regression_sample_count": self.continuous_n,
            "selected_token_count": int(self.token_ids.numel()),
            "blank_and_special_tokens_excluded": True,
            "score_mode": (
                "absolute_continuous_logprob_slope_x_sqrt_mean_prob"
            ),
            "min_mean_probability": self.min_mean_probability,
            "prototype_space": "log_probability",
        }


def continuous_specs(
    main_cluster: core.ClusterResult,
    _controls: Dict[str, core.ClusterResult],
    args: argparse.Namespace,
) -> List[core.InterventionSpec]:
    global LAST_SPECS
    transitions = list(CONTINUOUS_DIRECTIONS)
    specs: List[core.InterventionSpec] = []
    for alpha in args.alphas:
        for source, target in transitions:
            alpha_name = str(alpha).replace(".", "p")
            specs.append(
                core.InterventionSpec(
                    name=(
                        f"continuous_dose_{source}_to_{target}_"
                        f"main_a{alpha_name}"
                    ),
                    cluster_result=main_cluster,
                    source_cluster=source,
                    target_cluster=target,
                    alpha=float(alpha),
                    patch_mode="continuous_axis_sd",
                    control="main",
                )
            )
            for control in CONTROL_MODES:
                if control not in {
                    "opposite_direction",
                    "random_direction",
                    "random_hidden_direction",
                }:
                    raise ValueError(f"Unsupported continuous control: {control}")
                specs.append(
                    core.InterventionSpec(
                        name=(
                            f"continuous_dose_{source}_to_{target}_"
                            f"{control.replace('_', '-')}_a{alpha_name}"
                        ),
                        cluster_result=main_cluster,
                        source_cluster=source,
                        target_cluster=target,
                        alpha=float(alpha),
                        patch_mode="continuous_axis_sd",
                        control=control,
                    )
                )
    if args.max_intervention_modes > 0:
        specs = specs[: args.max_intervention_modes]
    LAST_SPECS = specs
    return specs


def role_available_ids(assignment_path: str, feature_path: str) -> set[str]:
    available = ORIGINAL_AVAILABLE_IDS(assignment_path, feature_path)
    role_map = AXIS.get(
        "all_role_ids" if CONTINUOUS_POPULATION == "full" else "role_ids",
        {},
    )
    if EVALUATION_ROLE == "all":
        role_values = [value for values in role_map.values() for value in values]
    else:
        role_values = role_map.get(EVALUATION_ROLE, [])
    role_ids = {str(value) for value in role_values}
    if not role_ids:
        raise ValueError(
            f"Continuous-axis artifact has no ids for role {EVALUATION_ROLE!r}."
        )
    overlap = available & role_ids
    if not overlap:
        raise ValueError(
            f"No fixed-axis ids overlap evaluation role {EVALUATION_ROLE!r}."
        )
    print(
        f"[continuous-dose] role={EVALUATION_ROLE} ids={len(overlap)} "
        f"fixed={len(available)}",
        flush=True,
    )
    return overlap


def continuous_apply_mask(
    batch_rows: List[Dict],
    row_index_by_id: Dict[str, int],
    spec: core.InterventionSpec,
    meta_group: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if CONTINUOUS_POPULATION != "full":
        return ORIGINAL_APPLY_MASK(batch_rows, row_index_by_id, spec, meta_group)
    apply_mask = torch.tensor(
        [meta_group == "all" or row.get("meta_group") == meta_group for row in batch_rows],
        dtype=torch.bool,
    )
    return torch.empty(len(batch_rows), 0), apply_mask


def spec_file(output_dir: str, spec: core.InterventionSpec) -> str:
    return os.path.join(output_dir, f"{spec.name}_generations.jsonl")


def paired_dose_statistics(
    output_dir: str,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> None:
    grouped: Dict[Tuple[int, int, float], Dict[str, core.InterventionSpec]] = {}
    for spec in LAST_SPECS:
        grouped.setdefault(
            (spec.source_cluster, spec.target_cluster, float(spec.alpha)), {}
        )[spec.control] = spec

    contrasts: List[Dict[str, Any]] = []
    by_direction_metric: Dict[
        Tuple[int, int, str], Dict[float, Dict[str, float]]
    ] = {}
    for key, modes in grouped.items():
        main = modes.get("main")
        if main is None:
            continue
        main_rows = {
            str(row["id"]): row
            for row in read_jsonl(spec_file(output_dir, main))
            if bool(row.get("intervention_applied"))
        }
        for control_name, comparison_name in (
            ("opposite_direction", "axis_direction_minus_opposite_direction"),
            ("random_direction", "axis_direction_minus_random_orthogonal_direction"),
            ("random_hidden_direction", "axis_direction_minus_random_hidden_direction"),
        ):
            control = modes.get(control_name)
            if control is None:
                continue
            control_rows = {
                str(row["id"]): row
                for row in read_jsonl(spec_file(output_dir, control))
                if bool(row.get("intervention_applied"))
            }
            matched = sorted(set(main_rows) & set(control_rows))
            strict = [
                sample_id
                for sample_id in matched
                if bool(main_rows[sample_id].get("code_correction_accepted"))
                and bool(control_rows[sample_id].get("dose_match_accepted"))
            ]
            for population, sample_ids in (
                ("intention_to_treat", matched),
                ("accepted_dose_matched", strict),
            ):
                for metric_index, metric in enumerate(core.NEXT_TOKEN_AUDIT_METRICS):
                    values = [
                        float(main_rows[sample_id][metric])
                        - float(control_rows[sample_id][metric])
                        for sample_id in sample_ids
                        if main_rows[sample_id].get(metric) is not None
                        and control_rows[sample_id].get(metric) is not None
                        and math.isfinite(float(main_rows[sample_id][metric]))
                        and math.isfinite(float(control_rows[sample_id][metric]))
                    ]
                    result = core.paired_bootstrap_and_signflip(
                        values,
                        bootstrap,
                        permutations,
                        seed + 1009 * len(contrasts) + metric_index,
                    )
                    contrasts.append(
                        {
                            "source_tail": key[0],
                            "target_tail": key[1],
                            "dose_sd": key[2],
                            "comparison": comparison_name,
                            "population": population,
                            "metric": metric,
                            **result,
                        }
                    )
                    if population == "intention_to_treat" and control_name == "opposite_direction":
                        for sample_id in sample_ids:
                            value = main_rows[sample_id].get(metric)
                            if value is None or not math.isfinite(float(value)):
                                continue
                            by_direction_metric.setdefault(
                                (key[0], key[1], metric), {}
                            ).setdefault(key[2], {})[sample_id] = float(value)

    for population in sorted({row["population"] for row in contrasts}):
        indices = [
            index
            for index, row in enumerate(contrasts)
            if row["population"] == population
        ]
        q_values = core.bh_adjust(
            [contrasts[index]["signflip_two_sided_p"] for index in indices]
        )
        for index, q_value in zip(indices, q_values):
            contrasts[index]["bh_fdr_q"] = q_value
    write_csv(
        os.path.join(output_dir, "continuous_dose_paired_contrasts.csv"),
        contrasts,
    )

    trends: List[Dict[str, Any]] = []
    for key, dose_map in sorted(by_direction_metric.items()):
        doses = sorted(dose_map)
        common_ids = sorted(
            set.intersection(*(set(dose_map[dose]) for dose in doses))
        )
        slopes = []
        denominator = sum(dose * dose for dose in doses)
        null_value = (
            1.0 if key[2] == "next_token_top10_overlap" else 0.0
        )
        for sample_id in common_ids:
            slopes.append(
                sum(
                    dose
                    * (dose_map[dose][sample_id] - null_value)
                    for dose in doses
                )
                / max(denominator, 1e-12)
            )
        result = core.paired_bootstrap_and_signflip(
            slopes,
            bootstrap,
            permutations,
            seed + 7919 * len(trends),
        )
        trends.append(
            {
                "source_tail": key[0],
                "target_tail": key[1],
                "metric": key[2],
                "dose_levels_sd": " ".join(str(value) for value in doses),
                "estimand": (
                    "per_sample_slope_of_change_from_metric_null_per_1sd"
                ),
                "metric_null_value": null_value,
                "positive_slope_rate": (
                    sum(value > 0 for value in slopes) / len(slopes)
                    if slopes
                    else None
                ),
                **result,
            }
        )
    q_values = core.bh_adjust(
        [row["signflip_two_sided_p"] for row in trends]
    )
    for row, q_value in zip(trends, q_values):
        row["bh_fdr_q"] = q_value
    write_csv(
        os.path.join(output_dir, "continuous_dose_response.csv"), trends
    )

    # Pool both directions on one signed axis. Target-oriented prototype
    # movement is re-oriented so positive always means movement toward +axis.
    signed_rows: List[Dict[str, Any]] = []
    for metric in core.NEXT_TOKEN_AUDIT_METRICS:
        per_id: Dict[str, List[Tuple[float, float]]] = {}
        null_value = 1.0 if metric == "next_token_top10_overlap" else 0.0
        for (source, target, dose), modes in grouped.items():
            sign = 1.0 if target > source else -1.0
            for control, dose_sign in (("main", sign), ("opposite_direction", -sign)):
                spec = modes.get(control)
                if spec is None:
                    continue
                for row in read_jsonl(spec_file(output_dir, spec)):
                    if not bool(row.get("intervention_applied")):
                        continue
                    value = row.get(metric)
                    if value is None or not math.isfinite(float(value)):
                        continue
                    oriented = float(value) - null_value
                    if metric in {
                        "next_token_auto_logit_margin_delta_toward_target",
                        "next_token_auto_logit_delta_projection_to_target",
                    }:
                        oriented *= sign
                    per_id.setdefault(str(row["id"]), []).append((dose_sign * dose, oriented))
        slopes = []
        for values in per_id.values():
            if len(values) < 2:
                continue
            denominator = sum(dose * dose for dose, _ in values)
            slopes.append(sum(dose * value for dose, value in values) / max(denominator, 1e-12))
        result = core.paired_bootstrap_and_signflip(
            slopes, bootstrap, permutations, seed + 17011 * (len(signed_rows) + 1)
        )
        signed_rows.append(
            {
                "metric": metric,
                "estimand": "within_sample_slope_per_signed_axis_sd",
                "positive_slope_rate": (
                    sum(value > 0 for value in slopes) / len(slopes) if slopes else None
                ),
                **result,
            }
        )
    q_values = core.bh_adjust([row["signflip_two_sided_p"] for row in signed_rows])
    for row, q_value in zip(signed_rows, q_values):
        row["bh_fdr_q"] = q_value
    write_csv(os.path.join(output_dir, "continuous_signed_dose_response.csv"), signed_rows)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--continuous-axis-file", required=True)
    parser.add_argument(
        "--continuous-evaluation-role",
        choices=[
            "prototype_discovery", "causal_screen_a", "causal_screen_b",
            "association_confirmatory", "style_confirmatory",
            "trajectory_confirmatory", "all",
        ],
        default="association_confirmatory",
    )
    parser.add_argument(
        "--continuous-directions",
        nargs="+",
        choices=["0:1", "1:0"],
        default=["0:1"],
        help="Axis orientations to instantiate; one direction plus opposite control spans both signed doses.",
    )
    parser.add_argument(
        "--continuous-controls",
        nargs="+",
        choices=[
            "opposite_direction",
            "random_direction",
            "random_hidden_direction",
        ],
        default=["opposite_direction"],
        help="Continuous refined-code controls to generate in addition to the real module.",
    )
    parser.add_argument(
        "--continuous-logit-prototype",
        action="store_true",
        help=(
            "Discover token prototypes from continuous log-probability slopes "
            "instead of artificial low/high tail means."
        ),
    )
    if "--help" in sys.argv[1:]:
        parser.print_help()
        return
    wrapper_args, remaining = parser.parse_known_args()
    unknown_flags = sorted(
        {value for value in remaining if value.startswith("--")}
        - ACTIVE_CORE_FLAGS
        - {"--help"}
    )
    if unknown_flags:
        raise ValueError(
            "Unsupported by the cleaned continuous interface: "
            + ", ".join(unknown_flags)
        )
    global AXIS, EVALUATION_ROLE, CONTROL_MODES, CONTINUOUS_DIRECTIONS
    global CONTINUOUS_LOGIT_PROTOTYPE, AXIS_SCORE_BY_ID
    AXIS = safe_load(wrapper_args.continuous_axis_file)
    AXIS_SCORE_BY_ID = {
        str(sample_id): float(score)
        for sample_id, score in zip(AXIS["ids"], AXIS["scores"])
    }
    CONTINUOUS_LOGIT_PROTOTYPE = bool(
        wrapper_args.continuous_logit_prototype
    )
    EVALUATION_ROLE = wrapper_args.continuous_evaluation_role
    CONTROL_MODES = tuple(dict.fromkeys(wrapper_args.continuous_controls))
    # The cleaned wrapper owns continuous controls; they are intentionally not
    # forwarded as the legacy core ``args.controls`` option.  Expose them to
    # the core validation layer so dose matching can validate the actual specs.
    core.CONTINUOUS_CONTROL_MODES = CONTROL_MODES
    CONTINUOUS_DIRECTIONS = tuple(
        tuple(map(int, value.split(":"))) for value in wrapper_args.continuous_directions
    )
    if "--intervention-source-only" in remaining:
        remaining.remove("--intervention-source-only")
        print("[continuous-dose] removed --intervention-source-only for full-range treatment", flush=True)
    sys.argv = [sys.argv[0], *remaining]
    core.RefinedResidualCodeIntervention = ContinuousDoseIntervention
    core.NextTokenAuditState = ContinuousLogitAuditState
    core.build_intervention_specs = continuous_specs
    core.fixed_cluster_available_ids = role_available_ids
    core.build_apply_mask_for_batch = continuous_apply_mask
    print(
        f"[continuous-dose] axis={wrapper_args.continuous_axis_file} "
        f"role={EVALUATION_ROLE} controls={','.join(CONTROL_MODES)} population=full "
        f"continuous_logit_prototype={CONTINUOUS_LOGIT_PROTOTYPE}",
        flush=True,
    )
    core.main()

    output_dir = None
    bootstrap = 2000
    permutations = 2000
    seed = 42
    for index, value in enumerate(remaining):
        if value == "--output-dir" and index + 1 < len(remaining):
            output_dir = remaining[index + 1]
        elif value == "--next-token-bootstrap-samples" and index + 1 < len(remaining):
            bootstrap = int(remaining[index + 1])
        elif value == "--next-token-permutation-tests" and index + 1 < len(remaining):
            permutations = int(remaining[index + 1])
        elif value == "--seed" and index + 1 < len(remaining):
            seed = int(remaining[index + 1])
    if output_dir and "--next-token-audit" in remaining and not (
        "--baseline-only" in remaining
    ):
        paired_dose_statistics(output_dir, bootstrap, permutations, seed)
    if output_dir:
        write_json(
            os.path.join(output_dir, "continuous_dose_config.json"),
            {
                "continuous_axis_file": wrapper_args.continuous_axis_file,
                "evaluation_role": EVALUATION_ROLE,
                "population": CONTINUOUS_POPULATION,
                "dose_unit": "training semantics-residualized module-code SD",
                "controls": CONTROL_MODES,
                "directions": CONTINUOUS_DIRECTIONS,
                "tail_labels_are_orientation_only": True,
                "continuous_logit_prototype": (
                    CONTINUOUS_LOGIT_PROTOTYPE
                ),
                "primary_outputs": [
                    "continuous_dose_paired_contrasts.csv",
                    "continuous_dose_response.csv",
                    "continuous_signed_dose_response.csv",
                ],
            },
        )


if __name__ == "__main__":
    main()
