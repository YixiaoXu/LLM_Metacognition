#!/usr/bin/env python3
"""Search for reportable neurons, then evaluate frozen winners and controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import _bootstrap  # noqa: F401
from audit_internal_activation_report import load_features, report_forward, single_token_options
from audit_neuron_report_groups import r2_columns, report_set, stable_rows
from metacog.audits.neuron_report_measurement import measure_report_activations
from metacog.audits.neuron_report_replay import generation_step_hidden_state
from metacog.models.capture import get_decoder_layers
from metacog.models.chat_tokens import diagnostic_turn_ids
from metacog.models.loading import load_model_and_tokenizer, model_input_device
from metacog.models.registry import get_model


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stratified_candidates(r2: np.ndarray, eligible: np.ndarray, count: int,
                          bins: int, seed: int) -> list[dict]:
    ordered = np.flatnonzero(eligible)
    ordered = ordered[np.argsort(r2[ordered], kind="stable")]
    if len(ordered) < count:
        raise ValueError(f"Only {len(ordered)} stable, label-balanced neurons; need {count}")
    rng = np.random.default_rng(seed)
    selected = []
    for stratum, group in enumerate(np.array_split(ordered, bins)):
        quota = count // bins + (stratum < count % bins)
        if len(group) < quota:
            raise ValueError(f"Semantic-R2 stratum {stratum} has {len(group)} neurons; need {quota}")
        selected.extend({"neuron": int(neuron), "stratum": stratum,
                         "semantic_r2_selection": float(r2[neuron])}
                        for neuron in rng.choice(group, quota, replace=False))
    return selected


def balanced_indices(values: np.ndarray, threshold: float, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    low = np.flatnonzero(values <= threshold)
    high = np.flatnonzero(values > threshold)
    half = count // 2
    if len(low) < half or len(high) < half:
        raise ValueError(f"Cannot draw {half} examples per label: low={len(low)} high={len(high)}")
    return sorted(map(int, np.r_[rng.choice(low, half, replace=False),
                                  rng.choice(high, half, replace=False)]))


def report_auc(rows: list[dict], threshold: float) -> float:
    truth = np.asarray([row["recorded_activation"] > threshold for row in rows], dtype=int)
    scores = np.asarray([row["report_high_logodds"] for row in rows], dtype=float)
    return float(roc_auc_score(truth, scores))


def conditional_report_gain(semantic: np.ndarray, activation: np.ndarray,
                            scores: np.ndarray, seed: int) -> dict:
    if len(scores) < 40 or np.var(scores) < 1e-10:
        return {"semantic_mse": None, "semantic_plus_activation_mse": None,
                "added_report_r2": None, "status": "insufficient_report_variance"}
    activation = np.asarray(activation, dtype=np.float64).reshape(-1, 1)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = [np.empty(len(scores)), np.empty(len(scores))]
    for train, test in KFold(5, shuffle=True, random_state=seed).split(semantic):
        scaler = StandardScaler().fit(semantic[train])
        sem_train, sem_test = scaler.transform(semantic[train]), scaler.transform(semantic[test])
        scale_activation = StandardScaler().fit(activation[train])
        joint_train = np.column_stack([sem_train, scale_activation.transform(activation[train])])
        joint_test = np.column_stack([sem_test, scale_activation.transform(activation[test])])
        predictions[0][test] = Ridge(alpha=10.0).fit(sem_train, scores[train]).predict(sem_test)
        predictions[1][test] = Ridge(alpha=10.0).fit(joint_train, scores[train]).predict(joint_test)
    sem_mse = float(np.mean((scores - predictions[0]) ** 2))
    joint_mse = float(np.mean((scores - predictions[1]) ** 2))
    return {"semantic_mse": sem_mse, "semantic_plus_activation_mse": joint_mse,
            "added_report_r2": float((sem_mse - joint_mse) / np.var(scores)),
            "status": "five_fold_out_of_fold"}


def matched_control_movement(model, layer_id: int, device, tokenizer, source: dict,
                             target_neuron: int, control_neuron: int, threshold: float,
                             delta: float, budget: float, tokens: tuple[int, int],
                             new_high: bool, replay_mode: str = "generation_step_kv") -> float:
    prefix = list(map(int, source["prompt_token_ids"]))
    marker = len(prefix) - 1
    layer = get_decoder_layers(model)[layer_id]
    a_token, b_token = tokens
    changes = []
    for high_is_a in (True, False):
        question = (f"Diagnostic question about the immediately preceding assistant token: "
                    f"was decoder layer {layer_id} neuron {target_neuron} above or below "
                    f"its frozen reference threshold ({threshold:.8g})? Answer with exactly one letter. "
                    + ("A = above; B = below." if high_is_a else "A = below; B = above."))
        full = prefix + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
        full += diagnostic_turn_ids(tokenizer, question)
        before, state = report_forward(model, layer, device, full, control_neuron, marker,
                                       replay_mode=replay_mode)
        amount = float(np.clip(delta, -budget * state["hidden_norm"],
                               budget * state["hidden_norm"]))
        after, _ = report_forward(model, layer, device, full, control_neuron,
                                  marker, state["before"] + amount, replay_mode=replay_mode)
        sign = 1 if high_is_a else -1
        changes.append(float(((after[a_token] - after[b_token]) -
                              (before[a_token] - before[b_token])).item()) * sign)
    return float(np.mean(changes)) * (1 if new_high else -1)


def evaluate_neuron(model, layer, device, tokenizer, item: dict, rows: list[dict],
                    activations: np.ndarray, predictions: np.ndarray,
                    semantic_projection: np.ndarray, control_neuron: int | None,
                    tokens: tuple[int, int], args) -> dict:
    neuron = item["neuron"]
    threshold = item["threshold"]
    indices = balanced_indices(activations[:, neuron], threshold, args.confirm_rows,
                               args.seed + neuron + 300000)
    chosen = [rows[index] for index in indices]
    observed = activations[indices, neuron]
    report = report_set(model, layer, args.layer_id, device, tokenizer, chosen, neuron,
                        threshold, item["train_std"], args.flip_norm_budget, tokens,
                        observed, args.flip_rows if item["kind"] == "winner" else 0,
                        replay_mode=args.replay_mode)
    truth = observed > threshold
    scores = np.asarray([row["report_high_logodds"] for row in report], dtype=float)
    auc = float(roc_auc_score(truth, scores))
    p = float(mannwhitneyu(scores[truth], scores[~truth], alternative="greater").pvalue)
    rng = np.random.default_rng(args.seed + neuron)
    boot = []
    for _ in range(args.bootstrap_samples):
        sample = rng.integers(0, len(truth), len(truth))
        if np.unique(truth[sample]).size == 2:
            boot.append(float(roc_auc_score(truth[sample], scores[sample])))
    control_movements = {}
    for row, source in zip(report, chosen):
        if control_neuron is None or "flipped_activation" not in row:
            continue
        new_high = row["flipped_activation"] > threshold
        if new_high == (row["recorded_activation"] > threshold):
            continue
        control_movements[row["id"]] = matched_control_movement(
            model, args.layer_id, device, tokenizer, source, neuron, control_neuron,
            threshold, row["flipped_activation"] - row["captured_activation"],
            args.flip_norm_budget, tokens, new_high, args.replay_mode)
    effective = [row for row in report if "flipped_activation" in row and
                 (row["flipped_activation"] > threshold) != (row["recorded_activation"] > threshold)]
    movements = [(row["flipped_report_high_logodds"] - row["report_high_logodds"])
                 * (1 if row["flipped_activation"] > threshold else -1) for row in effective]
    paired = [(movement, control_movements[row["id"]]) for row, movement in zip(effective, movements)
              if row["id"] in control_movements]
    predicted = predictions[indices, neuron]
    semantic_r2 = float(1 - np.sum((observed - predicted) ** 2) /
                        max(np.sum((observed - observed.mean()) ** 2), 1e-12))
    conditional = conditional_report_gain(semantic_projection[indices, :32], observed,
                                          scores, args.seed + neuron)
    return {"neuron": neuron, "kind": item["kind"], "stratum": item["stratum"],
            "matched_control_neuron": control_neuron,
            "semantic_r2_selection": item["semantic_r2_selection"],
            "semantic_r2_heldout": semantic_r2,
            "semantic_label_auc_heldout": float(roc_auc_score(truth, predicted)),
            "report_auc": auc, "report_auc_ci95": list(map(float, np.quantile(boot, [0.025, 0.975]))),
            "report_auc_one_sided_p": p,
            "report_balanced_accuracy": float(balanced_accuracy_score(truth, scores > 0)),
            "replay_mode": args.replay_mode,
            "activation_source": args.activation_source,
            "replay_absolute_error_max": float(max(row["replay_absolute_error"] for row in report)),
            "replay_label_agreement": float(np.mean([
                row["replay_label_matches_cache"] for row in report])),
            "conditional_report_gain": conditional,
            "flip_attempted": sum("flipped_activation" in row for row in report),
            "flip_effective": len(effective),
            "flip_mean_toward_new_state": float(np.mean(movements)) if movements else None,
            "flip_positive_rate": float(np.mean(np.asarray(movements) > 0)) if movements else None,
            "target_minus_control_movement": float(np.mean([a - b for a, b in paired])) if paired else None,
            "evaluation_ids": [row["id"] for row in report], "rows": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "activation-dir", "semantic-cache", "axis-csv", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--layer-id", type=int, required=True)
    parser.add_argument("--semantic-fit-rows", type=int, default=1200)
    parser.add_argument("--screen-a-pool", type=int, default=160)
    parser.add_argument("--candidate-neurons", type=int, default=256)
    parser.add_argument("--semantic-bins", type=int, default=8)
    parser.add_argument("--screen-a-rows", type=int, default=24)
    parser.add_argument("--screen-b-rows", type=int, default=80)
    parser.add_argument("--screen-b-candidates", type=int, default=24)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--evaluation-pool-rows", type=int, default=512)
    parser.add_argument("--confirm-rows", type=int, default=160)
    parser.add_argument("--flip-rows", type=int, default=24)
    parser.add_argument("--flip-norm-budget", type=float, default=0.03)
    parser.add_argument("--max-prefix-tokens", type=int, default=3000)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--replay-mode", choices=("legacy_full", "generation_step_kv"),
                        default="generation_step_kv")
    parser.add_argument("--activation-source", choices=("cache", "runtime_remeasured"),
                        default="cache")
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()
    if args.activation_source == "runtime_remeasured" and args.replay_mode != "generation_step_kv":
        parser.error("Fresh measurements require --replay-mode generation_step_kv")
    if (args.semantic_fit_rows < 200 or args.candidate_neurons < args.screen_b_candidates or
            args.screen_b_candidates < args.finalists or args.semantic_bins < 2 or
            args.candidate_neurons % args.semantic_bins or args.finalists < 1 or
            any(count < 8 or count % 2 for count in (args.screen_a_rows, args.screen_b_rows,
                                                    args.confirm_rows)) or
            args.screen_a_pool < args.screen_a_rows or args.evaluation_pool_rows < args.confirm_rows or
            args.flip_rows > args.confirm_rows or not 0 < args.flip_norm_budget < 1):
        parser.error("Invalid stage sizes, neuron counts, or flip budget")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    roles = stable_rows(Path(args.axis_csv),
                        Path(args.activation_dir) / "expanded_generation_data.jsonl",
                        args.seed, args.max_prefix_tokens)
    association = roles["association_confirmatory"][:args.semantic_fit_rows]
    discovery = roles["prototype_discovery"]
    evaluation = roles["trajectory_confirmatory"][:args.evaluation_pool_rows]
    if (len(association) < args.semantic_fit_rows or
            len(discovery) < args.screen_a_pool + args.screen_b_rows or
            len(evaluation) < args.evaluation_pool_rows):
        raise ValueError(f"Insufficient disjoint rows: association={len(association)} "
                         f"discovery={len(discovery)} evaluation={len(evaluation)}")
    screen_a = discovery[:args.screen_a_pool]
    screen_b = discovery[args.screen_a_pool:]
    groups = association + screen_a + screen_b + evaluation
    ids = [row["id"] for row in groups]
    semantic = load_features(Path(args.semantic_cache), ids)
    spec = get_model(args.target)
    model_args = argparse.Namespace(model_path=spec.resolved_path, dtype=spec.dtype,
                                    device_map="single", trust_remote_code=spec.trust_remote_code,
                                    gptq_backend=None, attn_implementation=None)
    model, tokenizer = load_model_and_tokenizer(model_args)
    device = model_input_device(model)
    layers = get_decoder_layers(model)
    if not 0 <= args.layer_id < len(layers):
        raise ValueError(f"Target layer {args.layer_id} is outside {len(layers)} decoder layers")
    layer = layers[args.layer_id]
    tokens = single_token_options(tokenizer)
    if args.activation_source == "runtime_remeasured":
        execution = {"target": args.target, "layer_id": args.layer_id,
                     "model_path": str(spec.resolved_path), "model_class": type(model).__name__,
                     "dtype": str(model.dtype), "replay_mode": args.replay_mode,
                     "torch": importlib.metadata.version("torch"),
                     "transformers": importlib.metadata.version("transformers"),
                     "model_config_sha256": hashlib.sha256(json.dumps(
                         model.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest()}
        activation = measure_report_activations(
            output, groups, int(model.config.hidden_size), execution,
            lambda prefix: generation_step_hidden_state(model, layer, device, prefix).numpy())
    else:
        activation = load_features(Path(args.activation_dir) / f"layer_{args.layer_id:03d}.pt", ids)
    n_a, n_1, n_2 = len(association), len(screen_a), len(screen_b)
    a = activation[:n_a]
    b1 = activation[n_a:n_a + n_1]
    b2 = activation[n_a + n_1:n_a + n_1 + n_2]
    e = activation[n_a + n_1 + n_2:]
    split = n_a // 2
    scaler = StandardScaler().fit(semantic[:split])
    pca = PCA(n_components=min(64, split - 2, semantic.shape[1]), random_state=args.seed).fit(
        scaler.transform(semantic[:split]))
    projection = pca.transform(scaler.transform(semantic))
    teacher = Ridge(alpha=10.0).fit(projection[:split], a[:split])
    selection_r2 = r2_columns(a[split:], teacher.predict(projection[split:n_a]))
    evaluation_prediction = teacher.predict(projection[n_a + n_1 + n_2:])
    evaluation_projection = projection[n_a + n_1 + n_2:]
    thresholds = np.median(a[:split], axis=0)
    std = np.std(a[:split], axis=0)
    stable = std >= max(1e-6, float(np.quantile(std, 0.25)))
    def enough(values: np.ndarray, half: int) -> np.ndarray:
        return np.minimum((values > thresholds).sum(0),
                          (values <= thresholds).sum(0)) >= half
    eligible = (stable & np.isfinite(selection_r2) & enough(a[split:], 8) &
                enough(b1, args.screen_a_rows // 2) & enough(b2, args.screen_b_rows // 2) &
                enough(e, args.confirm_rows // 2))
    candidates = stratified_candidates(selection_r2, eligible, args.candidate_neurons,
                                       args.semantic_bins, args.seed)
    for item in candidates:
        item["threshold"] = float(thresholds[item["neuron"]])
        item["train_std"] = float(std[item["neuron"]])
    contract = {"target": args.target, "layer_id": args.layer_id, "options": vars(args),
                "activation_table_sha256": hashlib.sha256(activation.tobytes()).hexdigest(),
                "association_ids": [row["id"] for row in association],
                "screen_a_ids": [row["id"] for row in screen_a],
                "screen_b_ids": [row["id"] for row in screen_b],
                "evaluation_ids": [row["id"] for row in evaluation], "candidates": candidates}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    candidate_file = output / "frozen_candidates.json"
    if candidate_file.exists():
        if read_json(candidate_file)["contract_sha256"] != fingerprint:
            raise ValueError("Candidate/prompt contract changed; use a new output directory")
    else:
        write_json(candidate_file, {"contract_sha256": fingerprint, "candidates": candidates,
                                    "n_eligible": int(eligible.sum()),
                                    "selection_rule": "uniform random sample from semantic-R2 strata; "
                                                      "reportability not used"})
    print(f"[report-search] target={args.target} eligible={eligible.sum()} "
          f"candidates={len(candidates)} screen_a={len(screen_a)} "
          f"screen_b={len(screen_b)} confirm_pool={len(evaluation)}", flush=True)

    def screen(stage: str, items: list[dict], rows: list[dict], values: np.ndarray,
               sample_size: int, offset: int) -> list[dict]:
        result = []
        directory = output / stage
        for item in tqdm(items, desc=f"Report {stage}"):
            neuron = item["neuron"]
            file = directory / f"neuron_{neuron:04d}.json"
            if file.exists():
                row = read_json(file)
            else:
                indices = balanced_indices(values[:, neuron], item["threshold"], sample_size,
                                           args.seed + neuron + offset)
                reports = report_set(model, layer, args.layer_id, device, tokenizer,
                                     [rows[index] for index in indices], neuron,
                                     item["threshold"], item["train_std"],
                                     args.flip_norm_budget, tokens, values[indices, neuron],
                                     replay_mode=args.replay_mode)
                row = {**item, "report_auc": report_auc(reports, item["threshold"]),
                       "n": len(reports), "evaluation_ids": [entry["id"] for entry in reports],
                       "replay_mode": args.replay_mode}
                write_json(file, row)
            result.append(row)
        return result

    first = screen("screen_a", candidates, screen_a, b1, args.screen_a_rows, 100000)
    first.sort(key=lambda item: (-item["report_auc"], item["neuron"]))
    second = screen("screen_b", first[:args.screen_b_candidates], screen_b, b2,
                    args.screen_b_rows, 200000)
    second.sort(key=lambda item: (-item["report_auc"], item["neuron"]))
    winners = second[:args.finalists]
    used = {item["neuron"] for item in winners}
    controls = []
    for winner in winners:
        pool = [item for item in candidates if item["neuron"] not in used and
                item["stratum"] == winner["stratum"]]
        control = min(pool, key=lambda item: (abs(item["semantic_r2_selection"] -
                                                winner["semantic_r2_selection"]), item["neuron"]))
        controls.append(control)
        used.add(control["neuron"])
    frozen = {"contract_sha256": fingerprint, "primary_neuron": winners[0]["neuron"],
              "winners": [{**item, "kind": "winner"} for item in winners],
              "controls": [{**item, "kind": "matched_control"} for item in controls],
              "screen_a_size": len(first), "screen_b_size": len(second),
              "selection_rule": "highest raw AUC on independent screen_b after screen_a preselection; "
                                "final tests use trajectory_confirmatory only"}
    selection_file = output / "frozen_selection.json"
    if selection_file.exists() and read_json(selection_file) != frozen:
        raise ValueError("Frozen reportability selection changed; use a new output directory")
    if not selection_file.exists():
        write_json(selection_file, frozen)
    print(f"[report-search] frozen winner={frozen['primary_neuron']} "
          f"screen_auc={winners[0]['report_auc']:.3f} controls={len(controls)}", flush=True)
    results = []
    for index, item in enumerate(tqdm(frozen["winners"] + frozen["controls"],
                                      desc="Held-out report confirmation")):
        neuron = item["neuron"]
        file = output / "heldout" / f"neuron_{neuron:04d}.json"
        if file.exists():
            value = read_json(file)
        else:
            control_neuron = controls[index]["neuron"] if index < len(winners) else None
            value = evaluate_neuron(model, layer, device, tokenizer, item, evaluation, e,
                                    evaluation_prediction, evaluation_projection,
                                    control_neuron, tokens, args)
            report_rows = value.pop("rows")
            rows_file = output / "heldout" / f"neuron_{neuron:04d}_rows.jsonl"
            rows_file.parent.mkdir(parents=True, exist_ok=True)
            rows_file.write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                         for row in report_rows), encoding="utf-8")
            write_json(file, value)
        results.append(value)
    winner_results = [row for row in results if row["kind"] == "winner"]
    control_results = [row for row in results if row["kind"] == "matched_control"]
    candidate_r2 = np.asarray([item["semantic_r2_selection"] for item in candidates])
    winner_percentiles = [float(np.mean(candidate_r2 <= item["semantic_r2_selection"]))
                          for item in winner_results]
    x = np.asarray([item["semantic_r2_selection"] for item in first])
    y = np.asarray([item["report_auc"] for item in first])
    rho = float(spearmanr(x, y).statistic) if len(np.unique(x)) > 1 and len(np.unique(y)) > 1 else None
    write_json(output / "report_search_summary.json", {
        "target": args.target, "layer": args.layer_id, "n_eligible": int(eligible.sum()),
        "replay_mode": args.replay_mode,
        "activation_source": args.activation_source,
        "activation_table_sha256": contract["activation_table_sha256"],
        "threshold_source": "median of semantic-fitting training half only",
        "n_screened": len(first), "n_rescreened": len(second),
        "n_heldout_winners": len(winner_results), "n_heldout_controls": len(control_results),
        "primary_neuron": frozen["primary_neuron"],
        "primary_report_auc": winner_results[0]["report_auc"],
        "primary_report_auc_one_sided_p": winner_results[0]["report_auc_one_sided_p"],
        "mean_winner_report_auc": float(np.mean([row["report_auc"] for row in winner_results])),
        "mean_control_report_auc": float(np.mean([row["report_auc"] for row in control_results])),
        "mean_winner_minus_control_report_auc": float(np.mean([
            winner["report_auc"] - control["report_auc"]
            for winner, control in zip(winner_results, control_results)])),
        "winner_semantic_r2_percentiles_in_screen_pool": winner_percentiles,
        "screen_population_semantic_r2_vs_report_auc_spearman": rho,
        "winners": winner_results, "matched_controls": control_results,
        "scope": "Discovery A/B and final confirmation are prompt-disjoint. Primary AUC uses the "
                 "literal above/below direction; no post-hoc orientation calibration. Search AUCs "
                 "are descriptive only. Semantic R2 is relative to the supplied reference."})
    print(f"[report-search] complete primary_auc={winner_results[0]['report_auc']:.3f} "
          f"p={winner_results[0]['report_auc_one_sided_p']:.4g}", flush=True)


if __name__ == "__main__":
    main()
