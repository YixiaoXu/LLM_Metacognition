#!/usr/bin/env python3
"""Discover reportable neurons, then test reporting and semantic readout on held-out prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import _bootstrap  # noqa: F401
from audit_internal_activation_report import load_features, report_forward, single_token_options
from metacog.audits.internal_report_stats import direct_report_classification
from metacog.models.capture import get_decoder_layers
from metacog.models.chat_tokens import diagnostic_turn_ids
from metacog.models.loading import load_model_and_tokenizer, model_input_device
from metacog.models.registry import get_model


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_rows(axis_path: Path, expanded_path: Path, seed: int, max_prefix: int) -> dict[str, list[dict]]:
    with axis_path.open(encoding="utf-8") as handle:
        roles = {row["id"]: row["analysis_role"] for row in csv.DictReader(handle)
                 if row["id"].endswith("::step000")}
    wanted = {"association_confirmatory", "prototype_discovery", "trajectory_confirmatory"}
    output = {role: [] for role in wanted}
    with expanded_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            role = roles.get(row["id"])
            if role in wanted and 0 < len(row.get("prompt_token_ids", [])) <= max_prefix:
                output[role].append(row)
    for role, rows in output.items():
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())
        if not rows:
            raise ValueError(f"No usable step-zero rows for {role}")
    sets = [{row["id"].split("::step")[0] for row in rows} for rows in output.values()]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise ValueError("Report discovery and evaluation roles overlap by base prompt")
    return output


def r2_columns(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    denominator = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
    numerator = np.sum((observed - predicted) ** 2, axis=0)
    return 1 - numerator / np.maximum(denominator, 1e-12)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def neuron_scores(model, layer, layer_id: int, device, tokenizer, row: dict, neuron: int, threshold: float,
                  tokens: tuple[int, int], patch: bool, std: float, budget: float,
                  replay_mode: str = "legacy_full") -> dict:
    prefix = list(map(int, row["prompt_token_ids"]))
    marker = len(prefix) - 1
    normal, changed, captured = [], [], []
    flipped = []
    a_token, b_token = tokens
    for high_is_a in (True, False):
        question = (f"Diagnostic question about the immediately preceding assistant token: "
                    f"was decoder layer {layer_id} neuron {neuron} above or below "
                    f"its frozen reference threshold ({threshold:.8g})? Answer with exactly one letter. "
                    + ("A = above; B = below." if high_is_a else "A = below; B = above."))
        turn = diagnostic_turn_ids(tokenizer, question)
        full = prefix + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []) + turn
        logits, state = report_forward(model, layer, device, full, neuron, marker,
                                       replay_mode=replay_mode)
        sign = 1 if high_is_a else -1
        normal.append(float((logits[a_token] - logits[b_token]).item()) * sign)
        captured.append(state["before"])
        if patch:
            delta = max(0.5 * std, abs(state["before"] - threshold) + 0.25 * std)
            destination = threshold - delta if state["before"] >= threshold else threshold + delta
            clipped = state["before"] + float(np.clip(destination - state["before"],
                                                       -budget * state["hidden_norm"],
                                                       budget * state["hidden_norm"]))
            logits_after, state_after = report_forward(model, layer, device, full, neuron, marker,
                                                       clipped, replay_mode=replay_mode)
            changed.append(float((logits_after[a_token] - logits_after[b_token]).item()) * sign)
            flipped.append(state_after["after"])
    result = {"id": row["id"], "report_high_logodds": float(np.mean(normal)),
              "captured_activation": float(np.mean(captured))}
    if patch:
        result.update({"flipped_activation": float(np.mean(flipped)),
                       "flipped_report_high_logodds": float(np.mean(changed))})
    return result


def report_set(model, layer, layer_id, device, tokenizer, rows, neuron, threshold, std, budget,
               tokens, recorded, flip_rows=0, replay_mode="legacy_full"):
    nearest = set(np.argsort(np.abs(recorded - threshold))[:flip_rows].tolist())
    output = []
    for index, row in enumerate(rows):
        result = neuron_scores(model, layer, layer_id, device, tokenizer, row, neuron, threshold,
                               tokens, index in nearest, std, budget, replay_mode)
        result["recorded_activation"] = float(recorded[index])
        result["recorded_high"] = bool(recorded[index] > threshold)
        result["replay_mode"] = replay_mode
        result["replay_absolute_error"] = abs(result["captured_activation"] - recorded[index])
        result["replay_label_matches_cache"] = bool(
            (result["captured_activation"] > threshold) == result["recorded_high"])
        tolerance = max(0.05, 0.1 * std)
        if abs(result["captured_activation"] - recorded[index]) > tolerance:
            raise ValueError(f"Cached activation does not reproduce for {row['id']} neuron {neuron}: "
                             f"cached={recorded[index]:.6g} replay={result['captured_activation']:.6g} "
                             f"tolerance={tolerance:.6g}")
        output.append(result)
    return output


def report_statistics(rows: list[dict], seed: int) -> dict:
    truth = np.array([int(row["recorded_high"]) for row in rows])
    scores = np.array([row["report_high_logodds"] for row in rows])
    result = direct_report_classification(truth, scores, seed)
    flips = [row for row in rows if "flipped_activation" in row]
    effective = [row for row in flips if (row["flipped_activation"] > row["threshold"])
                 != row["recorded_high"]]
    result["flip_attempted"] = len(flips)
    result["flip_effective"] = len(effective)
    if effective:
        movements = [
            (row["flipped_report_high_logodds"] - row["report_high_logodds"])
            * (1 if row["flipped_activation"] > row["threshold"] else -1)
            for row in effective
        ]
        result["flip_mean_shift_toward_new_state"] = float(np.mean(movements))
        result["flip_positive_shift_rate"] = float(np.mean(np.asarray(movements) > 0))
    else:
        result["flip_mean_shift_toward_new_state"] = None
        result["flip_positive_shift_rate"] = None
    return result


def semantic_r2_interval(observed: np.ndarray, predicted: np.ndarray, seed: int) -> list[float | None]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(500):
        sample = rng.integers(0, len(observed), len(observed))
        if np.var(observed[sample]) > 1e-12:
            values.append(r2_score(observed[sample], predicted[sample]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else [None, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "activation-dir", "semantic-cache", "axis-csv", "module-dirs", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--semantic-fit-rows", type=int, default=1200)
    parser.add_argument("--screen-rows", type=int, default=96)
    parser.add_argument("--evaluation-rows", type=int, default=240)
    parser.add_argument("--candidate-neurons", type=int, default=24)
    parser.add_argument("--selected-per-group", type=int, default=4)
    parser.add_argument("--flip-rows", type=int, default=16)
    parser.add_argument("--threshold-quantile", type=float, default=0.5)
    parser.add_argument("--flip-norm-budget", type=float, default=0.03)
    parser.add_argument("--max-prefix-tokens", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()
    if not 0 < args.threshold_quantile < 1 or args.selected_per_group > args.candidate_neurons:
        raise ValueError("Invalid threshold quantile or candidate counts")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    modules = [Path(value) for value in args.module_dirs.split(",")]
    configs = [read_json(path.parent.parent / "config.json") for path in modules]
    layer_ids = {int(config["layer_i"]) for config in configs}
    if len(layer_ids) != 1:
        raise ValueError("Neuron report groups require one earlier layer per target/domain")
    layer_id = layer_ids.pop()
    support = set()
    for path in modules:
        support.update(map(int, read_json(path / "module_summary.json")["module_member_neurons"]))
    roles = stable_rows(Path(args.axis_csv), Path(args.activation_dir) / "expanded_generation_data.jsonl",
                        args.seed, args.max_prefix_tokens)
    association = roles["association_confirmatory"][:args.semantic_fit_rows]
    screen = roles["prototype_discovery"][:args.screen_rows]
    evaluation = roles["trajectory_confirmatory"][:args.evaluation_rows]
    if len(association) < 200 or len(screen) < 60 or len(evaluation) < 100:
        raise ValueError("Too few prompt-disjoint rows for report discovery/evaluation")
    combined = association + screen + evaluation
    ids = [row["id"] for row in combined]
    activations = load_features(Path(args.activation_dir) / f"layer_{layer_id:03d}.pt", ids)
    semantics = load_features(Path(args.semantic_cache), ids)
    n_assoc, n_screen = len(association), len(screen)
    a, b = activations[:n_assoc], activations[n_assoc:n_assoc + n_screen]
    c = activations[n_assoc + n_screen:]
    s_assoc = semantics[:n_assoc]
    s_eval = semantics[n_assoc + n_screen:]
    split = n_assoc // 2
    scaler = StandardScaler().fit(s_assoc[:split])
    pca = PCA(n_components=min(64, split - 2, s_assoc.shape[1]), random_state=args.seed).fit(
        scaler.transform(s_assoc[:split]))
    def semantic_projection(x):
        return pca.transform(scaler.transform(x))
    teacher = Ridge(alpha=10.0).fit(semantic_projection(s_assoc[:split]), a[:split])
    validation_r2 = r2_columns(a[split:], teacher.predict(semantic_projection(s_assoc[split:])))
    evaluation_prediction = teacher.predict(semantic_projection(s_eval))
    thresholds = np.quantile(a[:split], args.threshold_quantile, axis=0)
    train_std = np.std(a[:split], axis=0)
    stable = train_std >= max(1e-6, float(np.quantile(train_std, 0.25)))
    counts = np.minimum((b > thresholds).sum(axis=0), (b <= thresholds).sum(axis=0))
    eligible = stable & (counts >= 8) & np.isfinite(validation_r2)
    high = [int(i) for i in np.argsort(-validation_r2) if eligible[i] and validation_r2[i] >= 0.05]
    low = [int(i) for i in np.argsort(np.abs(validation_r2))
           if eligible[i] and validation_r2[i] <= 0.03]
    low.sort(key=lambda i: (i not in support, abs(validation_r2[i])))
    high = high[:args.candidate_neurons]
    low = [i for i in low if i not in high][:args.candidate_neurons]
    selection_path = output / "frozen_selection.json"
    contract = {"target": args.target, "layer": layer_id, "support": sorted(support),
                "association_ids": [row["id"] for row in association],
                "screen_ids": [row["id"] for row in screen],
                "evaluation_ids": [row["id"] for row in evaluation],
                "threshold_quantile": args.threshold_quantile,
                "candidate_neurons": args.candidate_neurons,
                "selected_per_group": args.selected_per_group}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    previous = read_json(selection_path) if selection_path.exists() else None
    if previous and previous["contract_sha256"] != fingerprint:
        raise ValueError("Frozen reporting roles or configuration changed; use a new output directory")
    spec = get_model(args.target)
    model_args = argparse.Namespace(model_path=spec.resolved_path, dtype=spec.dtype,
                                    device_map="single", trust_remote_code=spec.trust_remote_code,
                                    gptq_backend=None, attn_implementation=None)
    model, tokenizer = load_model_and_tokenizer(model_args)
    device = model_input_device(model)
    layer = get_decoder_layers(model)[layer_id]
    tokens = single_token_options(tokenizer)
    if previous is None:
        screened = []
        for group, neuron in tqdm([("high_semantic", n) for n in high] +
                                  [("low_semantic", n) for n in low], desc="Screen direct reports"):
            rows = report_set(model, layer, layer_id, device, tokenizer, screen, neuron,
                              float(thresholds[neuron]), float(train_std[neuron]),
                              args.flip_norm_budget, tokens, b[:, neuron])
            truth = np.array([row["recorded_high"] for row in rows], dtype=int)
            scores = np.array([row["report_high_logodds"] for row in rows])
            auc = float(roc_auc_score(truth, scores)) if min(np.bincount(truth, minlength=2)) >= 2 else None
            screened.append({"group": group, "neuron": neuron, "screen_auc": auc,
                             "screen_decoded_auc": max(auc, 1 - auc) if auc is not None else None,
                             "report_orientation": 1 if auc is None or auc >= 0.5 else -1,
                             "screen_counts": [int((truth == 0).sum()), int(truth.sum())],
                             "validation_semantic_r2": float(validation_r2[neuron]),
                             "threshold": float(thresholds[neuron]), "train_std": float(train_std[neuron]),
                             "in_frozen_module_support": neuron in support})
        selected = []
        for group in ("high_semantic", "low_semantic"):
            available = [row for row in screened if row["group"] == group and row["screen_auc"] is not None]
            if group == "high_semantic":
                available.sort(key=lambda row: -row["screen_decoded_auc"])
            else:
                available.sort(key=lambda row: abs(row["screen_auc"] - 0.5))
            selected.extend(available[:args.selected_per_group])
        previous = {"contract_sha256": fingerprint, "screened": screened, "selected": selected,
                    "selection_note": "AUC selection used prototype_discovery; final evaluation uses disjoint trajectory_confirmatory"}
        selection_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    results = []
    for candidate in tqdm(previous["selected"], desc="Held-out direct reports"):
        neuron = candidate["neuron"]
        rows = report_set(model, layer, layer_id, device, tokenizer, evaluation, neuron,
                          candidate["threshold"], candidate["train_std"],
                          args.flip_norm_budget, tokens, c[:, neuron], args.flip_rows)
        for row in rows:
            row["threshold"] = candidate["threshold"]
        stats = report_statistics(rows, args.seed)
        decoded = direct_report_classification(
            np.array([int(row["recorded_high"]) for row in rows]),
            np.array([row["report_high_logodds"] * candidate["report_orientation"] for row in rows]),
            args.seed)
        observed = c[:, neuron]
        predicted = evaluation_prediction[:, neuron]
        semantic_r2 = float(r2_score(observed, predicted)) if np.var(observed) > 1e-12 else None
        semantic_ci = semantic_r2_interval(observed, predicted, args.seed)
        candidate_result = {**candidate, "n_evaluation": len(rows), "semantic_r2_heldout": semantic_r2,
                            "semantic_r2_ci95": semantic_ci,
                            "high_semantic_confirmed": semantic_ci[0] is not None and
                                                       candidate["group"] == "high_semantic" and
                                                       semantic_ci[0] > 0.05,
                            "low_semantic_confirmed": semantic_ci[1] is not None and
                                                      candidate["group"] == "low_semantic" and
                                                      semantic_ci[1] < 0.05,
                            "report": stats, "decoded_report": decoded,
                            "near_chance_95ci": (decoded["auc_bootstrap_ci95"] is not None and
                                                 decoded["auc_bootstrap_ci95"][0] >= 0.4 and
                                                 decoded["auc_bootstrap_ci95"][1] <= 0.6)}
        results.append(candidate_result)
        with (output / f"neuron_{neuron:04d}_report_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    high_results = [row for row in results if row["group"] == "high_semantic" and
                    row["decoded_report"]["auc_permutation_one_sided_p"] is not None]
    adjusted = holm([row["decoded_report"]["auc_permutation_one_sided_p"] for row in high_results])
    for row, p_holm in zip(high_results, adjusted):
        row["within_condition_holm_p"] = p_holm
    summary = {"target": args.target, "layer": layer_id,
               "n_association": n_assoc, "n_screen": n_screen, "n_evaluation": len(evaluation),
               "n_high_screened": len(high), "n_low_screened": len(low),
               "n_frozen_module_support_neurons": len(support), "selected": results,
               "interpretation": "Raw A/B agreement and screen-oriented one-dimensional decoding are separate. "
                                 "High reportability requires held-out decoded AUC>0.5 and corrected p<0.05; "
                                 "near-chance claims require a narrow held-out AUC interval. "
                                 "Semantic R2 is relative to the supplied reference, not all semantics."}
    (output / "report_group_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                    encoding="utf-8")
    print(json.dumps({"target": args.target, "n_selected": len(results),
                      "high_reportable": sum(row.get("within_condition_holm_p", 1) < 0.05 and
                                             row["decoded_report"]["auc"] > 0.5 for row in high_results),
                      "low_near_chance": sum(row["near_chance_95ci"] for row in results
                                             if row["group"] == "low_semantic")}, indent=2))


if __name__ == "__main__":
    main()
