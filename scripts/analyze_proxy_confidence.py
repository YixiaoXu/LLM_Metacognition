#!/usr/bin/env python3
"""Held-out calibration of output-distribution confidence proxies.

The primary question is whether a continuous residual-module score predicts
MathQA correctness beyond both external semantic controls and an ordinary
output-confidence proxy.  All probability models are cross-fitted; no row is
evaluated by a calibrator trained on that row.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  # Direct-script compatibility.
from analyze_continuous_meta_behavior import (
    logistic_fit_predict,
    make_fold_ids,
    one_sided_signflip_p,
)
from metacog.statistics import paired_bootstrap_and_signflip


PROXY_FIELDS = {
    "generated_prefix16_negative_entropy": "generated_prefix16_negative_entropy_mean",
    "prompt_end_negative_entropy": "prompt_end_negative_entropy",
    "prompt_end_max_probability": "prompt_end_max_probability",
    "prompt_end_logit_margin": "prompt_end_top1_top2_logit_margin",
    "generated_mean_logprob": "generated_mean_logprob",
    "generated_min_logprob": "generated_min_logprob",
    "generated_prefix16_mean_logprob": "generated_cumulative_logprob_prefix_json",
}


def safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    output = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                output.append(json.loads(line))
    return output


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def proxy_value(row: Mapping[str, Any], proxy: str) -> float | None:
    field = PROXY_FIELDS[proxy]
    if proxy != "generated_prefix16_mean_logprob":
        return finite(row.get(field))
    try:
        prefix = json.loads(str(row.get(field, "{}")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    value = finite(prefix.get("16"))
    return value / 16.0 if value is not None else None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def crossfit_binary(
    features: torch.Tensor | None,
    target: torch.Tensor,
    fold_ids: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.zeros_like(target)
    for fold in range(int(fold_ids.max().item()) + 1):
        test = fold_ids == fold
        train = ~test
        train_target = target[train]
        if features is None or torch.unique(train_target).numel() < 2:
            current = train_target.mean().clamp(1e-6, 1.0 - 1e-6)
            prediction[test] = current
            continue
        train_features = features[train].float()
        test_features = features[test].float()
        center = train_features.mean(dim=0, keepdim=True)
        scale = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        prediction[test] = logistic_fit_predict(
            (train_features - center) / scale,
            train_target,
            (test_features - center) / scale,
            ridge,
        )
    prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
    loss = F.binary_cross_entropy(prediction, target, reduction="none")
    return prediction, loss


def auroc(target: torch.Tensor, score: torch.Tensor) -> float:
    pairs = sorted(zip(score.tolist(), target.tolist()), key=lambda pair: pair[0])
    positive_n = int(target.sum().item())
    negative_n = target.numel() - positive_n
    if positive_n == 0 or negative_n == 0:
        return math.nan
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = 0.5 * ((index + 1) + end)
        rank_sum += average_rank * sum(pair[1] for pair in pairs[index:end])
        index = end
    return float(
        (rank_sum - positive_n * (positive_n + 1) / 2.0)
        / (positive_n * negative_n)
    )


def auprc(target: torch.Tensor, score: torch.Tensor) -> float:
    positive_n = int(target.sum().item())
    if positive_n == 0:
        return math.nan
    order = torch.argsort(score, descending=True)
    ordered = target.index_select(0, order)
    true_positive = torch.cumsum(ordered, dim=0)
    precision = true_positive / torch.arange(1, ordered.numel() + 1).float()
    return float((precision * ordered).sum().item() / positive_n)


def equal_mass_ece(
    target: torch.Tensor, probability: torch.Tensor, bins: int
) -> Tuple[float, List[Dict[str, Any]]]:
    order = torch.argsort(probability)
    chunks = torch.tensor_split(order, max(1, min(int(bins), target.numel())))
    ece = 0.0
    rows = []
    for bin_index, indices in enumerate(chunks, start=1):
        if indices.numel() == 0:
            continue
        confidence = float(probability.index_select(0, indices).mean().item())
        accuracy = float(target.index_select(0, indices).mean().item())
        weight = indices.numel() / target.numel()
        ece += weight * abs(confidence - accuracy)
        rows.append(
            {
                "bin": bin_index,
                "n": int(indices.numel()),
                "mean_predicted_correctness": confidence,
                "empirical_accuracy": accuracy,
            }
        )
    return float(ece), rows


def prediction_metrics(target: torch.Tensor, probability: torch.Tensor) -> Dict[str, float]:
    loss = F.binary_cross_entropy(probability, target, reduction="none")
    ece, _ = equal_mass_ece(target, probability, 10)
    return {
        "nll_nats": float(loss.mean().item()),
        "brier": float((probability - target).square().mean().item()),
        "auroc": auroc(target, probability),
        "auprc": auprc(target, probability),
        "ece_equal_mass_10": ece,
    }


def bootstrap_delta(
    target: torch.Tensor,
    baseline: torch.Tensor,
    expanded: torch.Tensor,
    metric,
    samples: int,
    seed: int,
) -> Tuple[float | None, float | None]:
    if samples <= 0:
        return None, None
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(samples):
        indices = torch.randint(0, target.numel(), (target.numel(),), generator=generator)
        sampled_target = target.index_select(0, indices)
        values.append(
            metric(sampled_target, expanded.index_select(0, indices))
            - metric(sampled_target, baseline.index_select(0, indices))
        )
    clean = torch.tensor([value for value in values if math.isfinite(value)])
    if clean.numel() == 0:
        return None, None
    return (
        float(torch.quantile(clean, 0.025).item()),
        float(torch.quantile(clean, 0.975).item()),
    )


def concatenate(*features: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [feature if feature.ndim == 2 else feature.view(-1, 1) for feature in features],
        dim=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-axis-file", type=Path, required=True)
    parser.add_argument("--generation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-role", default="association_confirmatory")
    parser.add_argument(
        "--primary-proxy",
        choices=sorted(PROXY_FIELDS),
        default="generated_prefix16_negative_entropy",
    )
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-tests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-truncated",
        dest="exclude_truncated",
        action="store_true",
        help=(
            "Restrict every proxy analysis to rows with style_truncated == 0. "
            "The generation JSONL must contain this field; filtering is done "
            "before proxy-specific missing-value filtering."
        ),
    )
    parser.add_argument(
        "--include-truncated",
        dest="exclude_truncated",
        action="store_false",
        help="Diagnostic override. The default analysis excludes truncated generations.",
    )
    parser.set_defaults(exclude_truncated=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    axis = safe_load(args.continuous_axis_file)
    generation_by_id = {
        str(row["id"]): row for row in read_jsonl(args.generation_file)
    }
    role_ids = axis.get("all_role_ids", axis.get("role_ids", {}))
    allowed = {
        str(value) for value in role_ids.get(args.evaluation_role, axis.get("ids", []))
    }
    axis_ids = [str(value) for value in axis["ids"]]
    index_by_id = {sample_id: index for index, sample_id in enumerate(axis_ids)}
    common_ids = [
        sample_id for sample_id in axis_ids
        if sample_id in allowed and sample_id in generation_by_id
    ]
    common_ids_before_truncation_filter = len(common_ids)
    truncated_removed = 0
    if args.exclude_truncated:
        missing_truncation = [
            sample_id
            for sample_id in common_ids
            if "style_truncated" not in generation_by_id[sample_id]
        ]
        if missing_truncation:
            examples = ", ".join(missing_truncation[:3])
            raise ValueError(
                "--exclude-truncated requires style_truncated in every "
                f"generation row; missing {len(missing_truncation)} rows "
                f"(examples: {examples})."
            )
        common_ids = [
            sample_id
            for sample_id in common_ids
            if float(generation_by_id[sample_id].get("style_truncated") or 0.0) == 0.0
        ]
        truncated_removed = common_ids_before_truncation_filter - len(common_ids)
        print(
            f"[proxy-confidence] excluding truncated rows: "
            f"kept={len(common_ids)} removed={truncated_removed}"
        )
    if len(common_ids) < 100:
        raise ValueError(
            f"Only {len(common_ids)} held-out ids overlap axis and generations."
        )

    summary_rows = []
    reliability_rows = []
    primary_predictions = []
    primary_result = None
    for proxy_index, proxy in enumerate(PROXY_FIELDS):
        valid_ids = [
            sample_id for sample_id in common_ids
            if proxy_value(generation_by_id[sample_id], proxy) is not None
            and generation_by_id[sample_id].get("correct") is not None
        ]
        if len(valid_ids) < 100:
            print(f"[proxy-confidence] skip {proxy}: n={len(valid_ids)}")
            continue
        indices = torch.tensor([index_by_id[sample_id] for sample_id in valid_ids])
        semantics = torch.as_tensor(axis["semantic_controls"]).float().index_select(0, indices)
        meta = torch.as_tensor(axis["scores"]).float().index_select(0, indices).view(-1, 1)
        proxy_tensor = torch.tensor(
            [proxy_value(generation_by_id[sample_id], proxy) for sample_id in valid_ids],
            dtype=torch.float32,
        ).view(-1, 1)
        target = torch.tensor(
            [float(bool(generation_by_id[sample_id]["correct"])) for sample_id in valid_ids],
            dtype=torch.float32,
        )
        folds = make_fold_ids(len(valid_ids), args.crossfit_folds, args.seed + 1009 * proxy_index)

        features = {
            "prior": None,
            "semantic": semantics,
            "proxy": proxy_tensor,
            "meta": meta,
            "semantic_proxy": concatenate(semantics, proxy_tensor),
            "semantic_meta": concatenate(semantics, meta),
            "proxy_meta": concatenate(proxy_tensor, meta),
            "semantic_proxy_meta": concatenate(semantics, proxy_tensor, meta),
        }
        predictions = {}
        losses = {}
        metrics = {}
        for name, feature in features.items():
            predictions[name], losses[name] = crossfit_binary(
                feature, target, folds, args.ridge
            )
            metrics[name] = prediction_metrics(target, predictions[name])

        conditional_meta = (
            losses["semantic_proxy"] - losses["semantic_proxy_meta"]
        ) / math.log(2.0)
        conditional_test = paired_bootstrap_and_signflip(
            conditional_meta.tolist(),
            args.bootstrap_samples,
            args.permutation_tests,
            args.seed + 5003 * proxy_index,
        )
        shapley_meta = (
            (losses["prior"] - losses["meta"]) / 3.0
            + (losses["semantic"] - losses["semantic_meta"]) / 6.0
            + (losses["proxy"] - losses["proxy_meta"]) / 6.0
            + (losses["semantic_proxy"] - losses["semantic_proxy_meta"]) / 3.0
        ) / math.log(2.0)
        shapley_test = paired_bootstrap_and_signflip(
            shapley_meta.tolist(),
            args.bootstrap_samples,
            args.permutation_tests,
            args.seed + 7001 * proxy_index,
        )
        auc_ci = bootstrap_delta(
            target,
            predictions["semantic_proxy"],
            predictions["semantic_proxy_meta"],
            auroc,
            args.bootstrap_samples,
            args.seed + 9001 * proxy_index,
        )
        row = {
            "proxy": proxy,
            "confirmatory_primary": proxy == args.primary_proxy,
            "n": len(valid_ids),
            "n_before_truncation_filter": common_ids_before_truncation_filter,
            "truncated_rows_removed": truncated_removed,
            "accuracy": float(target.mean().item()),
            "parse_rate": float(
                sum(bool(generation_by_id[sample_id].get("parse_success")) for sample_id in valid_ids)
                / len(valid_ids)
            ),
            "truncated_rate": float(
                sum(float(generation_by_id[sample_id].get("style_truncated") or 0.0) > 0.0 for sample_id in valid_ids)
                / len(valid_ids)
            ),
            **{f"semantic_proxy_{key}": value for key, value in metrics["semantic_proxy"].items()},
            **{f"joint_{key}": value for key, value in metrics["semantic_proxy_meta"].items()},
            "delta_auroc_meta_over_semantic_proxy": (
                metrics["semantic_proxy_meta"]["auroc"] - metrics["semantic_proxy"]["auroc"]
            ),
            "delta_auroc_bootstrap_ci95_low": auc_ci[0],
            "delta_auroc_bootstrap_ci95_high": auc_ci[1],
            "delta_brier_meta_over_semantic_proxy": (
                metrics["semantic_proxy"]["brier"] - metrics["semantic_proxy_meta"]["brier"]
            ),
            "conditional_meta_information_bits_per_sample": float(conditional_meta.mean().item()),
            "conditional_meta_bootstrap_ci95_low_bits": conditional_test["bootstrap_ci95"][0],
            "conditional_meta_bootstrap_ci95_high_bits": conditional_test["bootstrap_ci95"][1],
            "conditional_meta_signflip_one_sided_p": one_sided_signflip_p(
                conditional_meta.tolist(),
                args.permutation_tests,
                args.seed + 6007 * proxy_index,
            ),
            "conditional_meta_signflip_two_sided_p": conditional_test.get("signflip_two_sided_p"),
            "three_player_shapley_meta_bits_per_sample": float(shapley_meta.mean().item()),
            "shapley_meta_bootstrap_ci95_low_bits": shapley_test["bootstrap_ci95"][0],
            "shapley_meta_bootstrap_ci95_high_bits": shapley_test["bootstrap_ci95"][1],
            "shapley_meta_signflip_two_sided_p": shapley_test.get("signflip_two_sided_p"),
            "interpretation": (
                "Primary estimand is held-out log-loss information gained by meta "
                "after external semantics and the confidence proxy. MSE/Brier is "
                "an auxiliary calibration error, not an information amount."
            ),
        }
        summary_rows.append(row)

        if proxy == args.primary_proxy:
            primary_result = row
            for model_name in ("proxy", "semantic_proxy", "semantic_proxy_meta"):
                _, bins = equal_mass_ece(
                    target, predictions[model_name], args.calibration_bins
                )
                for bin_row in bins:
                    reliability_rows.append(
                        {"model": model_name, **bin_row}
                    )
            for sample_index, sample_id in enumerate(valid_ids):
                primary_predictions.append(
                    {
                        "id": sample_id,
                        "correct": int(target[sample_index].item()),
                        "proxy_value": float(proxy_tensor[sample_index].item()),
                        "module_score": float(meta[sample_index].item()),
                        "proxy_calibrated_correctness": float(predictions["proxy"][sample_index].item()),
                        "semantic_proxy_correctness": float(predictions["semantic_proxy"][sample_index].item()),
                        "semantic_proxy_meta_correctness": float(predictions["semantic_proxy_meta"][sample_index].item()),
                    }
                )
        print(
            f"[proxy-confidence] {proxy} n={len(valid_ids)} "
            f"auc={metrics['semantic_proxy_meta']['auroc']:.4f} "
            f"delta_auc={row['delta_auroc_meta_over_semantic_proxy']:.4f} "
            f"cond_meta_bits={row['conditional_meta_information_bits_per_sample']:.5f}"
        )

    if primary_result is None:
        raise ValueError(
            f"Primary proxy {args.primary_proxy!r} was unavailable. Regenerate the "
            "baseline with --record-prompt-confidence."
        )

    write_csv(args.output_dir / "proxy_confidence_summary.csv", summary_rows)
    write_csv(args.output_dir / "proxy_confidence_reliability.csv", reliability_rows)
    write_csv(args.output_dir / "proxy_confidence_predictions.csv", primary_predictions)
    summary = {
        "primary_proxy": args.primary_proxy,
        "exclude_truncated": bool(args.exclude_truncated),
        "common_ids_before_truncation_filter": common_ids_before_truncation_filter,
        "truncated_rows_removed": truncated_removed,
        "primary_result": primary_result,
        "secondary_proxies_are_exploratory": True,
        "crossfit_folds": args.crossfit_folds,
        "calibration": "fold-specific standardized logistic calibration",
        "multiplicity": (
            "No multiplicity correction is applied to the single preregistered "
            "primary proxy; secondary proxies are consistency diagnostics."
        ),
        "outputs": {
            "summary": "proxy_confidence_summary.csv",
            "reliability": "proxy_confidence_reliability.csv",
            "predictions": "proxy_confidence_predictions.csv",
        },
    }
    (args.output_dir / "proxy_confidence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[proxy-confidence] matplotlib unavailable; wrote tables only")
        else:
            fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
            labels = {
                "proxy": "Proxy only",
                "semantic_proxy": "Semantic + proxy",
                "semantic_proxy_meta": "Semantic + proxy + meta",
            }
            for model_name in labels:
                model_rows = [row for row in reliability_rows if row["model"] == model_name]
                ax.plot(
                    [row["mean_predicted_correctness"] for row in model_rows],
                    [row["empirical_accuracy"] for row in model_rows],
                    marker="o",
                    linewidth=1.8,
                    label=labels[model_name],
                )
            ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1)
            ax.set(xlabel="Predicted probability correct", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
            ax.legend(frameon=False)
            fig.savefig(args.output_dir / "proxy_confidence_reliability.png", dpi=220)
            fig.savefig(args.output_dir / "proxy_confidence_reliability.svg")
            plt.close(fig)
    print(f"[proxy-confidence] complete: {args.output_dir}")


if __name__ == "__main__":
    main()
