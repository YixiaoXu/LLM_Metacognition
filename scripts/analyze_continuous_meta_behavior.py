#!/usr/bin/env python
"""Cross-fitted behavior information decomposition for continuous meta scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def finite(value: Any) -> Optional[float]:
    """Return a finite float for optional statistical fields."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  # Legacy direct-script compatibility.

from metacog.statistics import bh_adjust, paired_bootstrap_and_signflip
from metacog.evaluation import (
    behavior_field_aliases,
    behavior_metrics_for_profile,
    compute_style_metrics,
    family_for_metric,
    metric_polarity,
    metric_families,
    profile_names,
)


def safe_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_behavior_rows(path: str) -> List[Dict[str, Any]]:
    """Read generation JSONL or the richer natural-behavior CSV."""
    if str(path).lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return read_jsonl(path)


def normalize_behavior_row(
    row: Dict[str, Any], max_new_tokens: int
) -> Dict[str, Any]:
    """Expose a common outcome schema for JSONL generations and behavior CSVs."""
    row = dict(row)
    text = row.get("generated_text")
    if text is None:
        text = row.get("visible_text", row.get("response", ""))
    try:
        tokens = int(float(row.get("generated_tokens") or 0))
    except (TypeError, ValueError):
        tokens = 0
    generated = compute_style_metrics(str(text or ""), tokens, max_new_tokens)
    if finite(row.get("style_truncated")) is None:
        row["style_truncated"] = generated.get("style_truncated", 0.0)
    aliases = behavior_field_aliases()
    for behavior_name, style_name in aliases.items():
        if behavior_name not in row:
            row[behavior_name] = generated.get(style_name)
    if "generated_tokens" not in row:
        row["generated_tokens"] = generated.get("generated_tokens", tokens)
    if "refusal_score" not in row:
        row.update(
            {
                "model_refusal": generated.get("style_refusal_present"),
                "refusal_score": generated.get("style_refusal_score"),
                "redirect_score": generated.get("style_redirect_score"),
                "policy_language_score": generated.get("style_policy_language_score"),
                "hedging_score": generated.get("style_hedging_score"),
                "harmful_detail_score": generated.get("style_harmful_detail_score"),
                "refusal_onset_normalized": generated.get(
                    "style_refusal_onset_normalized"
                ),
            }
        )
    if "model_refusal" in row and isinstance(row["model_refusal"], str):
        row["model_refusal"] = float(
            row["model_refusal"].strip().lower() in {"1", "true", "yes"}
        )
    cumulative = finite(row.get("generated_cumulative_logprob"))
    token_count = finite(row.get("generated_tokens"))
    if finite(row.get("generated_cumulative_logprob_per_token")) is None:
        row["generated_cumulative_logprob_per_token"] = (
            cumulative / max(token_count, 1.0)
            if cumulative is not None and token_count is not None
            else None
        )
    prompt_entropy = finite(row.get("prompt_end_entropy"))
    prefix_entropy = finite(row.get("generated_prefix16_entropy_mean"))
    if finite(row.get("confidence_entropy_shift_prompt_to_prefix16")) is None:
        row["confidence_entropy_shift_prompt_to_prefix16"] = (
            prefix_entropy - prompt_entropy
            if prefix_entropy is not None and prompt_entropy is not None
            else None
        )
    prompt_margin = finite(row.get("prompt_end_top1_top2_logit_margin"))
    prefix_margin = finite(row.get("generated_prefix16_logit_margin_mean"))
    if finite(row.get("confidence_margin_shift_prompt_to_prefix16")) is None:
        row["confidence_margin_shift_prompt_to_prefix16"] = (
            prefix_margin - prompt_margin
            if prefix_margin is not None and prompt_margin is not None
            else None
        )
    return row


def aggregate_sample_contributions(
    contribution_maps: Sequence[Dict[int, float]],
) -> List[float]:
    """Equal-weight outcomes, then average within sample for an omnibus test."""
    by_sample: Dict[int, List[float]] = defaultdict(list)
    for contribution_map in contribution_maps:
        for sample_index, value in contribution_map.items():
            if math.isfinite(float(value)):
                by_sample[int(sample_index)].append(float(value))
    return [
        float(sum(values) / len(values))
        for _, values in sorted(by_sample.items())
        if values
    ]


def add_fdr(rows: List[Dict[str, Any]], p_key: str, q_key: str) -> None:
    q_values = bh_adjust([row.get(p_key) for row in rows])
    for row, q_value in zip(rows, q_values):
        row[q_key] = q_value
        p_value = finite(row.get(p_key))
        row["significant_p_0p05"] = bool(
            p_value is not None and p_value < 0.05
        )


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


def ridge_fit_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train = torch.cat(
        [
            (train_x - mean) / std,
            torch.ones(
                train_x.size(0), 1, dtype=train_x.dtype, device=train_x.device
            ),
        ],
        dim=1,
    )
    test = torch.cat(
        [
            (test_x - mean) / std,
            torch.ones(
                test_x.size(0), 1, dtype=test_x.dtype, device=test_x.device
            ),
        ],
        dim=1,
    )
    gram = train.T @ train
    penalty = torch.eye(
        gram.size(0), dtype=gram.dtype, device=gram.device
    ) * float(ridge)
    penalty[-1, -1] = 0.0
    weight = torch.linalg.solve(
        gram + penalty, train.T @ train_y.view(-1, 1)
    )
    return (test @ weight).flatten()


def crossfit_residual(
    semantics: torch.Tensor,
    target: torch.Tensor,
    folds: int,
    ridge: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(target.numel(), generator=generator)
    fold_ids = torch.arange(target.numel()) % max(2, min(folds, target.numel()))
    fold_ids = fold_ids[torch.argsort(order)].to(target.device)
    prediction = torch.zeros_like(target)
    for fold in range(int(fold_ids.max().item()) + 1):
        test = fold_ids == fold
        train = ~test
        prediction[test] = ridge_fit_predict(
            semantics[train],
            target[train],
            semantics[test],
            ridge,
        )
    return target - prediction, prediction


def make_fold_ids(
    n: int, folds: int, seed: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(n, generator=generator)
    fold_count = max(2, min(int(folds), n))
    fold_ids = torch.arange(n) % fold_count
    return fold_ids[torch.argsort(order)].to(device)


def crossfit_predict(
    features: torch.Tensor,
    target: torch.Tensor,
    fold_ids: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    prediction = torch.zeros_like(target)
    for fold in range(int(fold_ids.max().item()) + 1):
        test = fold_ids == fold
        train = ~test
        prediction[test] = ridge_fit_predict(
            features[train], target[train], features[test], ridge
        )
    return prediction


def ridge_fit_train_and_test_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit once and return train/test predictions for variance calibration."""
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train = torch.cat(
        [
            (train_x - mean) / std,
            torch.ones(
                train_x.size(0), 1, dtype=train_x.dtype, device=train_x.device
            ),
        ],
        dim=1,
    )
    test = torch.cat(
        [
            (test_x - mean) / std,
            torch.ones(
                test_x.size(0), 1, dtype=test_x.dtype, device=test_x.device
            ),
        ],
        dim=1,
    )
    gram = train.T @ train
    penalty = torch.eye(
        gram.size(0), dtype=train.dtype, device=train.device
    ) * float(ridge)
    penalty[-1, -1] = 0.0
    weight = torch.linalg.solve(
        gram + penalty, train.T @ train_y.view(-1, 1)
    )
    return (train @ weight).flatten(), (test @ weight).flatten()


def logistic_fit_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    ridge: float,
    max_iter: int = 50,
) -> torch.Tensor:
    """Fit the ridge-logistic objective without materializing a dense Hessian."""
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train = torch.cat(
        [
            (train_x - mean) / std,
            torch.ones(
                train_x.size(0), 1, dtype=train_x.dtype, device=train_x.device
            ),
        ],
        dim=1,
    ).double()
    test = torch.cat(
        [
            (test_x - mean) / std,
            torch.ones(
                test_x.size(0), 1, dtype=test_x.dtype, device=test_x.device
            ),
        ],
        dim=1,
    ).double()
    y = train_y.double()
    prevalence = y.mean().clamp(1e-5, 1.0 - 1e-5)
    weight = torch.zeros(
        train.size(1), dtype=torch.float64, device=train.device
    )
    weight[-1] = torch.logit(prevalence)
    weight.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [weight],
        lr=1.0,
        max_iter=max(1, int(max_iter)),
        max_eval=max(2, int(max_iter) * 2),
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        objective = F.binary_cross_entropy_with_logits(
            train @ weight, y, reduction="sum"
        )
        objective = objective + 0.5 * float(ridge) * weight[:-1].square().sum()
        objective.backward()
        return objective

    optimizer.step(closure)
    return (
        torch.sigmoid(test @ weight)
        .float()
        .clamp(1e-6, 1.0 - 1e-6)
        .detach()
    )


def bootstrap_crossfit_r2(
    target: torch.Tensor,
    prediction: torch.Tensor,
    samples: int,
    seed: int,
    chunk_size: int = 256,
) -> List[float]:
    """Vectorize the prompt bootstrap while bounding temporary memory."""
    generator = torch.Generator().manual_seed(int(seed))
    values: List[float] = []
    sample_count = max(int(samples), 0)
    for start in range(0, sample_count, int(chunk_size)):
        current = min(int(chunk_size), sample_count - start)
        indices = torch.randint(
            0,
            target.numel(),
            (current, target.numel()),
            generator=generator,
        ).to(target.device)
        sampled_target = target[indices]
        sampled_prediction = prediction[indices]
        mse = (sampled_target - sampled_prediction).square().mean(dim=1)
        baseline = (
            sampled_target - sampled_target.mean(dim=1, keepdim=True)
        ).square().mean(dim=1).clamp_min(1e-12)
        values.extend((1.0 - mse / baseline).detach().cpu().tolist())
    return values


def infer_outcome_family(target: torch.Tensor) -> str:
    """Use Bernoulli scoring only for genuinely binary 0/1 outcomes."""
    unique = torch.unique(target)
    if unique.numel() <= 2 and bool(
        torch.all((unique == 0.0) | (unique == 1.0)).item()
    ):
        return "bernoulli"
    return "gaussian"


def crossfit_predictive_loss(
    features: Optional[torch.Tensor],
    target: torch.Tensor,
    fold_ids: torch.Tensor,
    ridge: float,
    outcome_family: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return held-out predictions and proper predictive losses in nats.

    Bernoulli outcomes use logistic cross-entropy. Continuous outcomes use a
    Gaussian log score whose mean and residual variance are fitted only on the
    training portion of each fold.
    """
    prediction = torch.zeros_like(target)
    loss = torch.zeros_like(target)
    for fold in range(int(fold_ids.max().item()) + 1):
        test = fold_ids == fold
        train = ~test
        train_y = target[train]
        test_y = target[test]
        if outcome_family == "bernoulli":
            if features is None:
                test_prediction = train_y.mean().clamp(1e-6, 1.0 - 1e-6).expand_as(
                    test_y
                )
            else:
                test_prediction = logistic_fit_predict(
                    features[train], train_y, features[test], ridge
                )
            prediction[test] = test_prediction
            loss[test] = F.binary_cross_entropy(
                test_prediction, test_y, reduction="none"
            )
            continue

        if features is None:
            train_prediction = train_y.mean().expand_as(train_y)
            test_prediction = train_y.mean().expand_as(test_y)
        else:
            train_prediction, test_prediction = ridge_fit_train_and_test_predict(
                features[train], train_y, features[test], ridge
            )
        prediction[test] = test_prediction
        train_variance = train_y.var(unbiased=False).clamp_min(1e-8)
        residual_variance = (train_y - train_prediction).square().mean()
        residual_variance = residual_variance.clamp_min(train_variance * 1e-6)
        loss[test] = 0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(residual_variance)
            + (test_y - test_prediction).square() / residual_variance
        )
    return prediction, loss


def predictive_information_decomposition(
    prior_loss: torch.Tensor,
    semantic_loss: torch.Tensor,
    meta_loss: torch.Tensor,
    joint_loss: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Per-sample conditional information and two-player Shapley allocation."""
    nats_to_bits = 1.0 / math.log(2.0)
    marginal_semantic = (prior_loss - semantic_loss) * nats_to_bits
    marginal_meta = (prior_loss - meta_loss) * nats_to_bits
    conditional_semantic = (meta_loss - joint_loss) * nats_to_bits
    conditional_meta = (semantic_loss - joint_loss) * nats_to_bits
    joint = (prior_loss - joint_loss) * nats_to_bits
    shapley_semantic = 0.5 * (marginal_semantic + conditional_semantic)
    shapley_meta = 0.5 * (marginal_meta + conditional_meta)
    interaction = joint - marginal_semantic - marginal_meta
    return {
        "marginal_semantic_bits": marginal_semantic,
        "marginal_meta_bits": marginal_meta,
        "conditional_semantic_bits": conditional_semantic,
        "conditional_meta_bits": conditional_meta,
        "joint_bits": joint,
        "shapley_semantic_bits": shapley_semantic,
        "shapley_meta_bits": shapley_meta,
        "interaction_bits": interaction,
    }


def one_sided_signflip_p(
    values: Sequence[float], permutation_tests: int, seed: int
) -> float | None:
    clean = torch.tensor(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=torch.float32,
    )
    if clean.numel() == 0 or permutation_tests <= 0:
        return None
    observed = float(clean.mean().item())
    generator = torch.Generator().manual_seed(int(seed))
    extreme = 0
    for start in range(0, int(permutation_tests), 512):
        current = min(512, int(permutation_tests) - start)
        signs = torch.randint(
            0, 2, (current, clean.numel()), generator=generator
        ).float().mul_(2.0).sub_(1.0)
        permuted = (signs * clean.view(1, -1)).mean(dim=1)
        extreme += int((permuted >= observed).sum().item())
    return float((extreme + 1) / (int(permutation_tests) + 1))


def rank_values(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(
        values.numel(), dtype=values.dtype, device=values.device
    )
    return ranks


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator.item()) <= 1e-12:
        return math.nan
    return float((left * right).sum().item() / denominator.item())


def slope_values(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    centered = x - x.mean()
    mean_square = centered.square().mean().clamp_min(1e-12)
    per_sample_slope = centered * y / mean_square
    return float(per_sample_slope.mean().item()), per_sample_slope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous conditional association between module score and behavior."
    )
    parser.add_argument("--continuous-axis-file", required=True)
    parser.add_argument(
        "--baseline-file",
        required=False,
        help="Generation JSONL; used when --behavior-file is not supplied.",
    )
    parser.add_argument(
        "--behavior-file",
        default=None,
        help="Optional natural_behavior_by_sample.csv with refusal outcomes.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metric-profile",
        choices=profile_names(),
        default="safety",
        help="Use only the prespecified outcomes for this task family.",
    )
    parser.add_argument(
        "--evaluation-role",
        choices=[
            "association_confirmatory",
            "style_confirmatory",
            "trajectory_confirmatory",
            "prototype_discovery",
            "causal_screen_a",
            "causal_screen_b",
            "all",
        ],
        default="association_confirmatory",
    )
    parser.add_argument("--semantic-ridge", type=float, default=0.1)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-tests", type=int, default=2000)
    parser.add_argument(
        "--semantic-r2-equivalence-margin",
        type=float,
        default=0.20,
        help=(
            "Report equivalence only when the bootstrap upper CI for semantic "
            "prediction R2 is below this prespecified bound."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--include-truncated",
        dest="exclude_truncated",
        action="store_false",
        help="Diagnostic override. The default analysis excludes truncated generations.",
    )
    parser.set_defaults(exclude_truncated=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_threads = max(
        1,
        int(
            os.environ.get(
                "CONTINUOUS_BEHAVIOR_CPU_THREADS",
                os.environ.get("OMP_NUM_THREADS", "4"),
            )
        ),
    )
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    requested_device = os.environ.get("CONTINUOUS_BEHAVIOR_DEVICE", "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[continuous-behavior] CUDA requested but unavailable; use CPU",
            flush=True,
        )
        requested_device = "cpu"
    compute_device = torch.device(requested_device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = safe_load(args.continuous_axis_file)
    behavior_path = args.behavior_file or args.baseline_file
    if not behavior_path:
        raise ValueError("Provide --baseline-file or --behavior-file.")
    baseline = {
        str(row["id"]): normalize_behavior_row(row, args.max_new_tokens)
        for row in read_behavior_rows(behavior_path)
    }
    ids = [str(value) for value in axis["ids"]]
    if args.evaluation_role != "all":
        role_ids = axis.get("all_role_ids", axis.get("role_ids", {}))
        allowed = {
            str(value)
            for value in role_ids.get(args.evaluation_role, [])
        }
        ids = [sample_id for sample_id in ids if sample_id in allowed]
    ids = [sample_id for sample_id in ids if sample_id in baseline]
    n_before_truncation_filter = len(ids)
    if args.exclude_truncated:
        ids = [
            sample_id
            for sample_id in ids
            if finite(baseline[sample_id].get("style_truncated")) == 0.0
        ]
    truncated_rows_removed = n_before_truncation_filter - len(ids)
    print(
        "[continuous-behavior] truncation filter: "
        f"enabled={args.exclude_truncated} kept={len(ids)} "
        f"removed={truncated_rows_removed}",
        flush=True,
    )
    if len(ids) < 50:
        raise ValueError(
            f"Only {len(ids)} continuous-axis ids overlap the baseline file."
        )
    index_by_id = {
        str(sample_id): index for index, sample_id in enumerate(axis["ids"])
    }
    indices = torch.tensor([index_by_id[sample_id] for sample_id in ids])
    score = (
        torch.as_tensor(axis["scores"])
        .float()
        .index_select(0, indices)
        .to(compute_device)
    )
    semantics = (
        torch.as_tensor(axis["semantic_controls"])
        .float()
        .index_select(0, indices)
        .to(compute_device)
    )
    rows = [baseline[sample_id] for sample_id in ids]
    candidate_metrics = [
        metric
        for metric in behavior_metrics_for_profile(args.metric_profile)
        if any(row.get(metric) is not None for row in rows)
    ]
    analysis_started = time.monotonic()
    print(
        "[continuous-behavior] setup: "
        f"rows={len(ids)} semantic_dim={semantics.size(1)} "
        f"metrics={len(candidate_metrics)} folds={args.crossfit_folds} "
        f"bootstrap={args.bootstrap_samples} permutations={args.permutation_tests} "
        f"device={compute_device} cpu_threads={torch.get_num_threads()}",
        flush=True,
    )

    print("[continuous-behavior] residualize signed module score", flush=True)
    score_residual, score_semantic_prediction = crossfit_residual(
        semantics,
        score,
        args.crossfit_folds,
        args.semantic_ridge,
        args.seed,
    )
    semantic_mse = float(
        (score - score_semantic_prediction).square().mean().item()
    )
    mean_mse = float((score - score.mean()).square().mean().clamp_min(1e-12).item())
    score_semantic_r2 = 1.0 - semantic_mse / mean_mse
    semantic_r2_bootstrap = bootstrap_crossfit_r2(
        score,
        score_semantic_prediction,
        args.bootstrap_samples,
        args.seed + 7001,
    )
    if semantic_r2_bootstrap:
        semantic_r2_ci = [
            float(torch.quantile(torch.tensor(semantic_r2_bootstrap), q).item())
            for q in (0.025, 0.975)
        ]
    else:
        semantic_r2_ci = [None, None]
    print("[continuous-behavior] residualize absolute module strength", flush=True)
    absolute_strength = score.abs()
    absolute_strength_residual, absolute_strength_semantic_prediction = (
        crossfit_residual(
            semantics,
            absolute_strength,
            args.crossfit_folds,
            args.semantic_ridge,
            args.seed + 17,
        )
    )
    strength_semantic_mse = float(
        (
            absolute_strength - absolute_strength_semantic_prediction
        ).square().mean().item()
    )
    strength_mean_mse = float(
        (
            absolute_strength - absolute_strength.mean()
        ).square().mean().clamp_min(1e-12).item()
    )
    strength_semantic_r2 = 1.0 - strength_semantic_mse / strength_mean_mse
    predictors = {
        "signed_direction": score_residual,
        "absolute_strength": absolute_strength_residual,
    }
    raw_predictors = {
        "signed_direction": score,
        "absolute_strength": absolute_strength,
    }

    results: List[Dict[str, Any]] = []
    quantile_rows: List[Dict[str, Any]] = []
    conditional_information_contributions: Dict[
        Tuple[str, str], Dict[int, float]
    ] = {}
    slope_contribution_maps: Dict[
        Tuple[str, str], Dict[int, float]
    ] = {}
    quantile = torch.bucketize(
        score_residual,
        torch.quantile(
            score_residual,
            torch.tensor(
                [0.2, 0.4, 0.6, 0.8], device=compute_device
            ),
        ),
    )
    for metric_index, metric in enumerate(candidate_metrics):
        metric_started = time.monotonic()
        values = []
        valid_indices = []
        for index, row in enumerate(rows):
            value = row.get(metric)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
                valid_indices.append(index)
        if len(values) < 50:
            continue
        selected = torch.tensor(
            valid_indices, dtype=torch.long, device=compute_device
        )
        y = torch.tensor(values, dtype=torch.float32, device=compute_device)
        semantic_subset = semantics.index_select(0, selected)
        y_residual, _ = crossfit_residual(
            semantic_subset,
            y,
            args.crossfit_folds,
            args.semantic_ridge,
            args.seed + 101 * metric_index,
        )
        standardized_y = y_residual / y_residual.std(
            unbiased=False
        ).clamp_min(1e-6)
        fold_ids = make_fold_ids(
            len(values),
            args.crossfit_folds,
            args.seed + 4001 * metric_index,
            compute_device,
        )
        outcome_family = infer_outcome_family(y)
        print(
            "[continuous-behavior] metric "
            f"{metric_index + 1}/{len(candidate_metrics)} name={metric} "
            f"n={len(values)} family={outcome_family}",
            flush=True,
        )
        prior_prediction, prior_predictive_loss = crossfit_predictive_loss(
            None,
            y,
            fold_ids,
            args.semantic_ridge,
            outcome_family,
        )
        semantic_prediction, semantic_predictive_loss = crossfit_predictive_loss(
            semantic_subset,
            y,
            fold_ids,
            args.semantic_ridge,
            outcome_family,
        )
        for predictor_index, (predictor_name, predictor) in enumerate(
            predictors.items()
        ):
            x_subset = predictor.index_select(0, selected)
            x_standardized = (
                x_subset - x_subset.mean()
            ) / x_subset.std(unbiased=False).clamp_min(1e-6)
            raw_slope, _ = slope_values(x_standardized, y_residual)
            (
                standardized_slope,
                standardized_slope_contributions,
            ) = slope_values(x_standardized, standardized_y)
            test = paired_bootstrap_and_signflip(
                standardized_slope_contributions.detach().cpu().tolist(),
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed
                + 1009 * metric_index
                + 17011 * predictor_index,
            )
            slope_contribution_maps[(predictor_name, metric)] = {
                int(sample_index): float(value)
                for sample_index, value in zip(
                    valid_indices,
                    standardized_slope_contributions.detach().cpu().tolist(),
                )
            }
            results.append(
                {
                    "meta_predictor": predictor_name,
                    "metric": metric,
                    "n": len(values),
                    "raw_behavior_change_per_meta_sd": raw_slope,
                    "standardized_behavior_change_per_meta_sd": (
                        standardized_slope
                    ),
                    "partial_pearson_r": correlation(
                        x_standardized, y_residual
                    ),
                    "partial_spearman_r": correlation(
                        rank_values(x_standardized),
                        rank_values(y_residual),
                    ),
                    "standardized_slope_bootstrap_mean": test["mean"],
                    "standardized_slope_bootstrap_ci95_low": (
                        test["bootstrap_ci95"][0]
                    ),
                    "standardized_slope_bootstrap_ci95_high": (
                        test["bootstrap_ci95"][1]
                    ),
                    "signflip_two_sided_p": test["signflip_two_sided_p"],
                }
            )
            # Primary estimand: proper held-out predictive information from
            # prior, semantic-only, meta-only and joint models on identical folds.
            raw_predictor = raw_predictors[predictor_name].index_select(0, selected)
            meta_prediction, meta_predictive_loss = crossfit_predictive_loss(
                raw_predictor.view(-1, 1),
                y,
                fold_ids,
                args.semantic_ridge,
                outcome_family,
            )
            combined_features = torch.cat(
                [semantic_subset, raw_predictor.view(-1, 1)], dim=1
            )
            combined_prediction, combined_predictive_loss = (
                crossfit_predictive_loss(
                    combined_features,
                    y,
                    fold_ids,
                    args.semantic_ridge,
                    outcome_family,
                )
            )
            information = predictive_information_decomposition(
                prior_predictive_loss,
                semantic_predictive_loss,
                meta_predictive_loss,
                combined_predictive_loss,
            )
            conditional_meta = information["conditional_meta_bits"]
            shapley_meta = information["shapley_meta_bits"]
            conditional_test = paired_bootstrap_and_signflip(
                conditional_meta.detach().cpu().tolist(),
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed + 5003 * metric_index + 193 * predictor_index,
            )
            shapley_test = paired_bootstrap_and_signflip(
                shapley_meta.detach().cpu().tolist(),
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed + 5501 * metric_index + 197 * predictor_index,
            )
            prior_squared_error = (y - prior_prediction).square()
            semantic_squared_error = (y - semantic_prediction).square()
            meta_squared_error = (y - meta_prediction).square()
            combined_squared_error = (y - combined_prediction).square()
            mse_improvement = semantic_squared_error - combined_squared_error
            mse_test = paired_bootstrap_and_signflip(
                mse_improvement.detach().cpu().tolist(),
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed + 5701 * metric_index + 199 * predictor_index,
            )
            total_variance = float((y - y.mean()).square().mean().item())
            semantic_mse = float(semantic_squared_error.mean().item())
            combined_mse = float(combined_squared_error.mean().item())
            joint_information = float(information["joint_bits"].mean().item())
            conditional_meta_information = float(conditional_meta.mean().item())
            shapley_meta_information = float(shapley_meta.mean().item())
            conditional_information_contributions[(predictor_name, metric)] = {
                int(sample_index): float(value)
                for sample_index, value in zip(
                    valid_indices, conditional_meta.detach().cpu().tolist()
                )
            }
            results.append(
                {
                    "analysis": "nested_incremental_prediction",
                    "meta_predictor": predictor_name,
                    "metric": metric,
                    "n": len(values),
                    "predictive_loss_family": (
                        "bernoulli_cross_entropy"
                        if outcome_family == "bernoulli"
                        else "gaussian_negative_log_likelihood"
                    ),
                    "prior_nll_nats_per_sample": float(
                        prior_predictive_loss.mean().item()
                    ),
                    "meta_only_nll_nats_per_sample": float(
                        meta_predictive_loss.mean().item()
                    ),
                    "semantic_only_nll_nats_per_sample": float(
                        semantic_predictive_loss.mean().item()
                    ),
                    "semantic_plus_meta_nll_nats_per_sample": float(
                        combined_predictive_loss.mean().item()
                    ),
                    "joint_predictive_information_bits_per_sample": joint_information,
                    "marginal_semantic_information_bits_per_sample": float(
                        information["marginal_semantic_bits"].mean().item()
                    ),
                    "marginal_meta_information_bits_per_sample": float(
                        information["marginal_meta_bits"].mean().item()
                    ),
                    "conditional_semantic_information_bits_per_sample": float(
                        information["conditional_semantic_bits"].mean().item()
                    ),
                    "conditional_meta_information_bits_per_sample": (
                        conditional_meta_information
                    ),
                    "shapley_semantic_information_bits_per_sample": float(
                        information["shapley_semantic_bits"].mean().item()
                    ),
                    "shapley_meta_information_bits_per_sample": (
                        shapley_meta_information
                    ),
                    "shapley_meta_share_of_joint_information": (
                        shapley_meta_information / joint_information
                        if joint_information > 1e-12
                        else None
                    ),
                    "information_interaction_bits_per_sample": float(
                        information["interaction_bits"].mean().item()
                    ),
                    "conditional_information_bootstrap_ci95_low_bits": (
                        conditional_test["bootstrap_ci95"][0]
                    ),
                    "conditional_information_bootstrap_ci95_high_bits": (
                        conditional_test["bootstrap_ci95"][1]
                    ),
                    "conditional_information_signflip_two_sided_p": (
                        conditional_test["signflip_two_sided_p"]
                    ),
                    "conditional_information_signflip_one_sided_p": (
                        one_sided_signflip_p(
                            conditional_meta.detach().cpu().tolist(),
                            args.permutation_tests,
                            args.seed
                            + 6007 * metric_index
                            + 211 * predictor_index,
                        )
                    ),
                    "shapley_meta_bootstrap_ci95_low_bits": shapley_test[
                        "bootstrap_ci95"
                    ][0],
                    "shapley_meta_bootstrap_ci95_high_bits": shapley_test[
                        "bootstrap_ci95"
                    ][1],
                    "shapley_meta_signflip_two_sided_p": shapley_test[
                        "signflip_two_sided_p"
                    ],
                    "shapley_meta_signflip_one_sided_p": one_sided_signflip_p(
                        shapley_meta.detach().cpu().tolist(),
                        args.permutation_tests,
                        args.seed + 6503 * metric_index + 223 * predictor_index,
                    ),
                    # MSE/Brier metrics are retained as intuitive prediction-error
                    # diagnostics; they are not interpreted as information amounts.
                    "prior_mse": float(prior_squared_error.mean().item()),
                    "meta_only_mse": float(meta_squared_error.mean().item()),
                    "semantic_only_mse": semantic_mse,
                    "semantic_plus_meta_mse": combined_mse,
                    "incremental_mse_reduction": float(mse_improvement.mean().item()),
                    "incremental_partial_r2": float(
                        mse_improvement.mean().item() / max(semantic_mse, 1e-12)
                    ),
                    "semantic_only_r2": float(
                        1.0 - semantic_mse / max(total_variance, 1e-12)
                    ),
                    "semantic_plus_meta_r2": float(
                        1.0 - combined_mse / max(total_variance, 1e-12)
                    ),
                    "incremental_bootstrap_ci95_low": mse_test[
                        "bootstrap_ci95"
                    ][0],
                    "incremental_bootstrap_ci95_high": mse_test[
                        "bootstrap_ci95"
                    ][1],
                    "incremental_signflip_two_sided_p": mse_test[
                        "signflip_two_sided_p"
                    ],
                    "incremental_signflip_one_sided_p": one_sided_signflip_p(
                        mse_improvement.detach().cpu().tolist(),
                        args.permutation_tests,
                        args.seed + 6701 * metric_index + 227 * predictor_index,
                    ),
                    "interpretation": (
                        "conditional_meta_information_bits_per_sample is the primary "
                        "operational information estimate; MSE fields are auxiliary "
                        "prediction-error diagnostics and are not information amounts"
                    ),
                }
            )
        signed_subset = score_residual.index_select(0, selected)
        signed_standardized = (
            signed_subset - signed_subset.mean()
        ) / signed_subset.std(unbiased=False).clamp_min(1e-6)
        selected_quantile = quantile.index_select(0, selected)
        for group in range(5):
            mask = selected_quantile == group
            quantile_rows.append(
                {
                    "metric": metric,
                    "meta_score_quintile": group + 1,
                    "n": int(mask.sum().item()),
                    "behavior_mean": (
                        float(y[mask].mean().item()) if mask.any() else None
                    ),
                    "behavior_semantic_residual_mean": (
                        float(y_residual[mask].mean().item())
                        if mask.any()
                        else None
                    ),
                    "meta_score_mean": (
                        float(signed_standardized[mask].mean().item())
                        if mask.any()
                        else None
                    ),
                    "interpretation": "descriptive bin of a continuous axis",
                }
            )
        print(
            "[continuous-behavior] metric complete "
            f"{metric_index + 1}/{len(candidate_metrics)} name={metric} "
            f"elapsed={time.monotonic() - metric_started:.1f}s",
            flush=True,
        )

    family_omnibus_rows: List[Dict[str, Any]] = []
    all_outcome_diagnostic_rows: List[Dict[str, Any]] = []
    for predictor_index, predictor_name in enumerate(predictors):
        predictor_metrics = [
            metric
            for metric in candidate_metrics
            if (predictor_name, metric) in conditional_information_contributions
        ]
        available_families = {
            family_for_metric(metric, args.metric_profile)
            for metric in predictor_metrics
        }
        families = [
            family
            for family in metric_families(args.metric_profile)
            if family in available_families
        ]
        for family_index, family in enumerate(families):
            family_metrics = [
                metric
                for metric in predictor_metrics
                if family_for_metric(metric, args.metric_profile) == family
            ]
            contributions = aggregate_sample_contributions(
                [
                    conditional_information_contributions[(predictor_name, metric)]
                    for metric in family_metrics
                ]
            )
            test = paired_bootstrap_and_signflip(
                contributions,
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed + 7103 * predictor_index + 307 * family_index,
            )
            oriented_slopes = aggregate_sample_contributions(
                [
                    {
                        sample_index: metric_polarity(metric) * value
                        for sample_index, value in slope_contribution_maps[
                            (predictor_name, metric)
                        ].items()
                    }
                    for metric in family_metrics
                ]
            )
            direction_test = paired_bootstrap_and_signflip(
                oriented_slopes,
                args.bootstrap_samples,
                args.permutation_tests,
                args.seed + 7603 * predictor_index + 353 * family_index,
            )
            direction_mean = direction_test["mean"]
            family_omnibus_rows.append(
                {
                    "meta_predictor": predictor_name,
                    "metric_family": family,
                    "metrics": " ".join(family_metrics),
                    "metric_count": len(family_metrics),
                    "n": test["n"],
                    "mean_conditional_meta_information_bits_per_sample": test["mean"],
                    "bootstrap_ci95_low": test["bootstrap_ci95"][0],
                    "bootstrap_ci95_high": test["bootstrap_ci95"][1],
                    "signflip_two_sided_p": test["signflip_two_sided_p"],
                    "signflip_one_sided_p": one_sided_signflip_p(
                        contributions,
                        args.permutation_tests,
                        args.seed + 8101 * predictor_index + 401 * family_index,
                    ),
                    "aligned_standardized_slope": direction_mean,
                    "aligned_slope_bootstrap_ci95_low": direction_test[
                        "bootstrap_ci95"
                    ][0],
                    "aligned_slope_bootstrap_ci95_high": direction_test[
                        "bootstrap_ci95"
                    ][1],
                    "aligned_slope_signflip_two_sided_p": direction_test[
                        "signflip_two_sided_p"
                    ],
                    "aligned_direction": (
                        "positive"
                        if direction_mean is not None and direction_mean > 0
                        else "negative"
                        if direction_mean is not None and direction_mean < 0
                        else "zero"
                    ),
                    "estimand": (
                        "equal-outcome-weight mean of per-sample conditional "
                        "predictive information in bits after semantic controls; "
                        "aligned slope averages prespecified signed standardized "
                        "component effects"
                    ),
                }
            )
        global_contributions = aggregate_sample_contributions(
            [
                conditional_information_contributions[(predictor_name, metric)]
                for metric in predictor_metrics
            ]
        )
        global_test = paired_bootstrap_and_signflip(
            global_contributions,
            args.bootstrap_samples,
            args.permutation_tests,
            args.seed + 9103 * predictor_index,
        )
        all_outcome_diagnostic_rows.append(
            {
                "meta_predictor": predictor_name,
                "metric_count": len(predictor_metrics),
                "metrics": " ".join(predictor_metrics),
                "n": global_test["n"],
                "mean_conditional_meta_information_bits_per_sample": global_test[
                    "mean"
                ],
                "bootstrap_ci95_low": global_test["bootstrap_ci95"][0],
                "bootstrap_ci95_high": global_test["bootstrap_ci95"][1],
                "signflip_two_sided_p": global_test["signflip_two_sided_p"],
                "signflip_one_sided_p": one_sided_signflip_p(
                    global_contributions,
                    args.permutation_tests,
                    args.seed + 10103 * predictor_index,
                ),
                "estimand": (
                    "equal-outcome-weight mean of per-sample conditional "
                    "predictive information in bits across prespecified outcomes"
                ),
            }
        )
    add_fdr(
        family_omnibus_rows,
        "signflip_one_sided_p",
        "within_module_family_fdr_q",
    )
    add_fdr(
        all_outcome_diagnostic_rows,
        "signflip_one_sided_p",
        "diagnostic_predictor_fdr_q",
    )
    for row in results:
        row["fdr_test"] = (
            "conditional_information_signflip_one_sided"
            if row.get("analysis") == "nested_incremental_prediction"
            else "residualized_slope_two_sided"
        )
        row["fdr_family"] = (
            f"{row['fdr_test']}:{family_for_metric(row['metric'], args.metric_profile)}"
        )
    for family in sorted({row["fdr_family"] for row in results}):
        indices = [
            index for index, row in enumerate(results)
            if row["fdr_family"] == family
        ]
        q_values = bh_adjust(
            [
                results[index].get(
                    "conditional_information_signflip_one_sided_p",
                    results[index].get("signflip_two_sided_p"),
                )
                for index in indices
            ]
        )
        for index, q_value in zip(indices, q_values):
            results[index]["fdr_family_size"] = len(indices)
            results[index]["bh_fdr_q"] = q_value
            p_value = finite(
                results[index].get(
                    "conditional_information_signflip_one_sided_p",
                    results[index].get("signflip_two_sided_p"),
                )
            )
            results[index]["significant_p_0p05"] = bool(
                p_value is not None and p_value < 0.05
            )
    write_csv(output_dir / "continuous_behavior_associations.csv", results)
    write_csv(
        output_dir / "continuous_behavior_information_decomposition.csv",
        [
            row
            for row in results
            if row.get("analysis") == "nested_incremental_prediction"
        ],
    )
    write_csv(output_dir / "continuous_behavior_quintiles.csv", quantile_rows)
    write_csv(
        output_dir / "continuous_behavior_family_omnibus.csv",
        family_omnibus_rows,
    )
    write_csv(
        output_dir / "continuous_behavior_all_outcome_diagnostic.csv",
        all_outcome_diagnostic_rows,
    )
    print(
        "[continuous-behavior] analysis complete "
        f"elapsed={time.monotonic() - analysis_started:.1f}s",
        flush=True,
    )
    summary = {
        "n": len(ids),
        "compute_device": str(compute_device),
        "cpu_threads": int(torch.get_num_threads()),
        "logistic_solver": "ridge_lbfgs_strong_wolfe",
        "exclude_truncated": bool(args.exclude_truncated),
        "n_before_truncation_filter": n_before_truncation_filter,
        "truncated_rows_removed": truncated_rows_removed,
        "analysis_population": (
            "untruncated_only" if args.exclude_truncated else "all_rows"
        ),
        "metric_profile": args.metric_profile,
        "reported_metrics": candidate_metrics,
        "evaluation_role": args.evaluation_role,
        "continuous_meta_score_semantic_crossfit_r2": score_semantic_r2,
        "absolute_meta_strength_semantic_crossfit_r2": strength_semantic_r2,
        "continuous_meta_score_semantic_crossfit_r2_bootstrap_ci95": (
            semantic_r2_ci
        ),
        "incremental_results": [
            {
                "meta_predictor": row["meta_predictor"],
                "metric": row["metric"],
                "predictive_loss_family": row["predictive_loss_family"],
                "conditional_meta_information_bits_per_sample": row[
                    "conditional_meta_information_bits_per_sample"
                ],
                "conditional_information_bootstrap_ci95_bits": [
                    row["conditional_information_bootstrap_ci95_low_bits"],
                    row["conditional_information_bootstrap_ci95_high_bits"],
                ],
                "conditional_information_one_sided_p": row[
                    "conditional_information_signflip_one_sided_p"
                ],
                "joint_predictive_information_bits_per_sample": row[
                    "joint_predictive_information_bits_per_sample"
                ],
                "shapley_meta_information_bits_per_sample": row[
                    "shapley_meta_information_bits_per_sample"
                ],
                "shapley_semantic_information_bits_per_sample": row[
                    "shapley_semantic_information_bits_per_sample"
                ],
                "shapley_meta_share_of_joint_information": row[
                    "shapley_meta_share_of_joint_information"
                ],
                "shapley_meta_bootstrap_ci95_bits": [
                    row["shapley_meta_bootstrap_ci95_low_bits"],
                    row["shapley_meta_bootstrap_ci95_high_bits"],
                ],
                "shapley_meta_one_sided_p": row[
                    "shapley_meta_signflip_one_sided_p"
                ],
                "incremental_mse_reduction": row["incremental_mse_reduction"],
                "incremental_partial_r2": row["incremental_partial_r2"],
                "mse_reduction_bootstrap_ci95": [
                    row["incremental_bootstrap_ci95_low"],
                    row["incremental_bootstrap_ci95_high"],
                ],
                "mse_reduction_one_sided_p": row[
                    "incremental_signflip_one_sided_p"
                ],
                "fdr_q": row["bh_fdr_q"],
            }
            for row in results
            if row.get("analysis") == "nested_incremental_prediction"
        ],
        "module_family_omnibus": family_omnibus_rows,
        "all_outcome_average_diagnostic": all_outcome_diagnostic_rows,
        "semantic_r2_equivalence_margin": args.semantic_r2_equivalence_margin,
        "semantic_r2_equivalent_below_margin": bool(
            semantic_r2_ci[1] is not None
            and semantic_r2_ci[1] < args.semantic_r2_equivalence_margin
        ),
        "primary_estimand": (
            "module-specific cross-fitted conditional predictive information beyond "
            "semantics, measured by proper held-out log loss; "
            "the primary gate asks whether the module predicts at least one "
            "prespecified behavior family, followed by metric localization"
        ),
        "information_decomposition": (
            "prior, semantic-only, meta-only and semantic-plus-meta models are fit on "
            "identical folds. Conditional information is semantic loss minus joint "
            "loss. Two-player Shapley values average marginal and conditional entry "
            "orders and sum to the joint predictive information. Values are operational, "
            "model-dependent estimates in bits per held-out sample."
        ),
        "mse_guardrail": (
            "MSE/Brier and partial R2 are retained only as intuitive prediction-error "
            "diagnostics; they are not interpreted as information quantities."
        ),
        "multiple_testing": (
            "Individual metrics use Benjamini-Hochberg FDR within prespecified "
            "behavior families. Family omnibus tests use FDR across behavior "
            "families and signed/absolute predictors within the module; the run-level "
            "summary additionally corrects across modules within each model pair."
        ),
        "significant_metrics": [
            f"{row['meta_predictor']}:{row['metric']}"
            for row in results
            if row["significant_p_0p05"]
            and row.get("analysis") == "nested_incremental_prediction"
        ],
        "interpretation_guardrail": (
            "Quintiles visualize a continuous relationship and are not latent classes. "
            "Modules are allowed to control different behaviors. Cross-module mean "
            "effects are descriptive heterogeneity summaries, not the primary claim."
        ),
    }
    (output_dir / "continuous_behavior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    if not args.no_plots and results:
        try:
            import matplotlib.pyplot as plt

            association_results = [
                row
                for row in results
                if row.get("analysis") != "nested_incremental_prediction"
            ]
            ordered = sorted(
                association_results,
                key=lambda row: abs(
                    row["standardized_behavior_change_per_meta_sd"]
                ),
            )[-12:]
            if ordered:
                figure, plot = plt.subplots(
                    figsize=(7.0, max(3.8, 0.34 * len(ordered) + 1.2))
                )
                values = [
                    row["standardized_behavior_change_per_meta_sd"]
                    for row in ordered
                ]
                colors = [
                    "#0072B2" if value >= 0 else "#D55E00" for value in values
                ]
                plot.barh(
                    [
                        (
                            "signed"
                            if row["meta_predictor"] == "signed_direction"
                            else "|score|"
                        )
                        + " · "
                        + row["metric"].replace("style_", "")
                        for row in ordered
                    ],
                    values,
                    color=colors,
                    alpha=0.85,
                )
                plot.axvline(0.0, color="#222222", linewidth=0.8)
                plot.set_xlabel("Semantic-adjusted behavior change per meta-score SD")
                plot.set_title("Continuous conditional behavior associations")
                plot.spines[["top", "right"]].set_visible(False)
                figure.tight_layout()
                figure.savefig(
                    output_dir / "continuous_behavior_effects.png", dpi=220
                )
                plt.close(figure)

            incremental = [
                row
                for row in results
                if row.get("analysis") == "nested_incremental_prediction"
            ]
            ordered_incremental = sorted(
                incremental,
                key=lambda row: abs(
                    row["conditional_meta_information_bits_per_sample"]
                ),
            )[-12:]
            if ordered_incremental:
                figure, plot = plt.subplots(
                    figsize=(7.0, max(3.8, 0.34 * len(ordered_incremental) + 1.2))
                )
                values = [
                    row["conditional_meta_information_bits_per_sample"]
                    for row in ordered_incremental
                ]
                lower = [
                    row["conditional_information_bootstrap_ci95_low_bits"]
                    for row in ordered_incremental
                ]
                upper = [
                    row["conditional_information_bootstrap_ci95_high_bits"]
                    for row in ordered_incremental
                ]
                plot.errorbar(
                    values,
                    range(len(ordered_incremental)),
                    xerr=[
                        [value - lo for value, lo in zip(values, lower)],
                        [hi - value for value, hi in zip(values, upper)],
                    ],
                    fmt="o",
                    color="#0072B2",
                    ecolor="#555555",
                    capsize=3,
                )
                plot.axvline(0.0, color="#222222", linewidth=0.8)
                plot.set_yticks(
                    range(len(ordered_incremental)),
                    [
                        f"{row['meta_predictor']} · {row['metric'].replace('style_', '')}"
                        for row in ordered_incremental
                    ],
                )
                plot.set_xlabel("Conditional meta information (bits / held-out sample)")
                plot.set_title("Predictive information beyond semantic controls")
                plot.spines[["top", "right"]].set_visible(False)
                figure.tight_layout()
                figure.savefig(
                    output_dir / "continuous_behavior_incremental_effects.png",
                    dpi=220,
                )
                figure.savefig(
                    output_dir / "continuous_behavior_information_effects.png",
                    dpi=220,
                )
                plt.close(figure)
        except ImportError:
            pass
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
