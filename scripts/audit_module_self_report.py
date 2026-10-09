#!/usr/bin/env python3
"""Test whether a target model reports a frozen residual module state.

The primary contrast changes the module in two directions on the same prompt.
The semantic predictor and report examples are fit on disjoint prompt roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import _bootstrap  # noqa: F401
import audit_continuous_module_dose as dose_core
import cluster_intervention as core
from audit_internal_activation_report import load_features, single_token_options
from audit_neuron_report_search import conditional_report_gain
from audit_neuron_report_groups import stable_rows
from metacog.models.capture import get_decoder_layers, last_token_logits_kwargs
from metacog.models.chat_tokens import diagnostic_turn_ids
from metacog.models.loading import load_model_and_tokenizer, model_input_device
from metacog.models.registry import get_model


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def role_rows(roles: dict, name: str, score_by_id: dict) -> list[dict]:
    return [row for row in roles[name] if row["id"] in score_by_id]


def frozen_axis_scalars(axis: dict) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    codes = torch.as_tensor(axis["raw_codes"]).float()
    direction = torch.as_tensor(axis["axis"]).float().flatten()
    scores = torch.as_tensor(axis["scores"]).float().flatten()
    ids = list(map(str, axis["ids"]))
    if codes.ndim != 2 or direction.numel() != codes.shape[1]:
        raise ValueError(
            f"Frozen axis shape mismatch: raw_codes={tuple(codes.shape)}, "
            f"axis={tuple(direction.shape)}"
        )
    if len(ids) != codes.shape[0] or len(ids) != scores.numel() or len(set(ids)) != len(ids):
        raise ValueError("Frozen axis IDs, raw codes and scores are not aligned")
    if (not torch.isfinite(codes).all() or not torch.isfinite(direction).all()
            or not torch.isfinite(scores).all()
            or not torch.isclose(direction.norm(), direction.new_tensor(1.0), atol=1e-3)):
        raise ValueError("Frozen axis contains non-finite values or is not unit length")
    raw_scalars = codes @ direction
    return (direction, dict(zip(ids, map(float, scores))),
            dict(zip(ids, map(float, raw_scalars))))


def balanced(rows: list[dict], score_by_id: dict, count: int) -> list[dict]:
    low = [row for row in rows if score_by_id[row["id"]] < 0]
    high = [row for row in rows if score_by_id[row["id"]] >= 0]
    if min(len(low), len(high)) < count // 2:
        raise ValueError(f"Need {count // 2} per module-score sign; low={len(low)} high={len(high)}")
    return sorted(low[:count // 2] + high[:count // 2], key=lambda row: row["id"])


def replay_generation_step(model, layer, device, prefix: list[int],
                           replacement: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, object]:
    if len(prefix) < 2:
        raise ValueError("A generated-step state requires prompt tokens and one generated token")
    prompt = torch.tensor([prefix[:-1]], device=device)
    step = torch.tensor([[prefix[-1]]], device=device)
    state = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        state["hidden"] = hidden[0, -1].detach().float().clone()
        if replacement is None:
            return None
        updated = hidden.clone()
        updated[0, -1] = replacement.to(updated)
        return (updated, *output[1:]) if isinstance(output, tuple) else updated

    with torch.inference_mode():
        prefill = model(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        use_cache=True, **last_token_logits_kwargs(model))
        past = prefill.past_key_values
        del prefill
        handle = layer.register_forward_hook(hook)
        try:
            generated = model(
                input_ids=step,
                attention_mask=torch.ones((1, len(prefix)), dtype=torch.long, device=device),
                past_key_values=past, use_cache=True,
                **last_token_logits_kwargs(model),
            )
        finally:
            handle.remove()
    if "hidden" not in state:
        raise RuntimeError("Target layer did not return a hidden state")
    return state["hidden"], generated.past_key_values


def capture_hidden(model, layer, device, ids: list[int], marker: int) -> torch.Tensor:
    if marker != len(ids) - 1:
        raise ValueError("Only the final generated token can be replayed")
    hidden, _ = replay_generation_step(model, layer, device, ids)
    return hidden


def report_logits(model, layer, device, tokenizer, prefix: list[int], question: str,
                  tokens: tuple[int, int], replacement: torch.Tensor | None,
                  expected_hidden: torch.Tensor,
                  enable_thinking: bool | None) -> tuple[float, float, float]:
    suffix = ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
    suffix += diagnostic_turn_ids(tokenizer, question, enable_thinking=enable_thinking)
    before, past = replay_generation_step(model, layer, device, prefix, replacement)
    replay_error = float(((before - expected_hidden).norm() /
                          expected_hidden.norm().clamp_min(1e-6)).item())
    with torch.inference_mode():
        result = model(
            input_ids=torch.tensor([suffix], device=device),
            attention_mask=torch.ones((1, len(prefix) + len(suffix)),
                                      dtype=torch.long, device=device),
            past_key_values=past, use_cache=True,
            **last_token_logits_kwargs(model),
        )
        logits = result.logits[0, -1].float()
        logodds = float((logits[tokens[0]] - logits[tokens[1]]).item())
        option_mass = float(torch.softmax(logits, dim=-1)[list(tokens)].sum().item())
    return logodds, replay_error, option_mass


def make_question(examples: list[tuple[str, bool]], high_is_a: bool) -> str:
    def letter(high: bool) -> str:
        return "A" if high == high_is_a else "B"

    lines = ["Calibration examples for a fixed internal signal M in this model."]
    for snippet, high in examples:
        lines.append(f"Earlier input: {snippet}\nSignal M: {letter(high)}")
    lines.append("For the assistant state at the final token immediately before this question, "
                 "is signal M above or below its reference level? Answer one letter only. "
                 + ("A = above; B = below." if high_is_a else "A = below; B = above."))
    return "\n".join(lines)


def prompt_examples(rows: list[dict], scores: dict, shots_per_label: int) -> list[tuple[str, bool]]:
    selected = balanced(rows, scores, shots_per_label * 2)
    result = []
    for row in selected:
        source = row.get("question") or row.get("prompt") or ""
        text = " ".join(str(source).split())[-180:]
        result.append((text, scores[row["id"]] >= 0))
    return result


def code_state(hidden: torch.Tensor, decoupler, purifier, refiner,
               normalization: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = normalization["next_mean"].to(hidden).view(1, -1)
    std = normalization["next_std"].to(hidden).view(1, -1).clamp_min(1e-6)
    with torch.inference_mode():
        standardized = (hidden.view(1, -1) - mean) / std
        z1 = decoupler.e1(standardized)
        result = purifier(decoupler.e2(standardized))
        code = core.runtime_code(refiner, result["meta"])
    return code, z1, result["semantic"]


def relative_change(after: torch.Tensor, before: torch.Tensor) -> float:
    return float(((after - before).norm() / before.norm().clamp_min(1e-6)).item())


def random_pair(original: torch.Tensor, main: dict, before_semantic: torch.Tensor,
                before_code: torch.Tensor, solvers: dict, axis: torch.Tensor,
                seed: int, candidates: int, semantic_tolerance: float,
                max_axis_leakage: float) -> dict:
    generator = torch.Generator(device=original.device).manual_seed(seed)
    delta_plus = main["positive"]["hidden"] - original
    delta_minus = main["negative"]["hidden"] - original
    direction = delta_plus - delta_minus
    direction = direction / direction.norm().clamp_min(1e-8)
    target_norms = {key: (main[key]["hidden"] - original).norm() for key in ("positive", "negative")}
    best = None
    for _ in range(candidates):
        vector = torch.randn(original.shape, generator=generator,
                             device=original.device, dtype=original.dtype)
        vector = vector - vector.dot(direction) * direction
        vector = vector / vector.norm().clamp_min(1e-8)
        candidate = {}
        loss = 0.0
        for name, sign in (("random_positive", 1), ("random_negative", -1)):
            reference = "positive" if sign > 0 else "negative"
            hidden = original + sign * vector * target_norms[reference]
            code, _, semantic = code_state(hidden, solvers["decoupler"], solvers["purifier"],
                                           solvers["refiner"], solvers["normalization"])
            drift = relative_change(semantic, before_semantic)
            code_shift = float(((code - before_code) @ axis.view(-1, 1)).item())
            candidate[name] = {"hidden": hidden.detach(), "code": code.detach(),
                               "semantic_drift": drift, "code_shift": code_shift,
                               "hidden_relative": relative_change(hidden, original)}
            loss += abs(drift - main[reference]["semantic_drift"])
            loss += 0.05 * abs(code_shift) / max(float(solvers["residual_std"]), 1e-6)
        if best is None or loss < best[0]:
            best = (loss, candidate)
    assert best is not None
    for name, reference in (("random_positive", "positive"),
                            ("random_negative", "negative")):
        candidate = best[1][name]
        target = main[reference]
        drift_gap = abs(candidate["semantic_drift"] - target["semantic_drift"])
        candidate["semantic_drift_gap"] = drift_gap
        candidate["matched"] = bool(
            drift_gap <= semantic_tolerance and
            abs(candidate["code_shift"]) <= max_axis_leakage *
            max(abs(target["code_shift"]), 1e-6)
        )
    return best[1]


def paired_test(values: np.ndarray, seed: int, repeats: int) -> dict:
    if len(values) < 12:
        return {"n": len(values), "mean": None, "ci95": None, "p_one_sided": None}
    rng = np.random.default_rng(seed)
    bootstrap = [float(np.mean(values[rng.integers(0, len(values), len(values))]))
                 for _ in range(repeats)]
    null = [float(np.mean(values * rng.choice([-1, 1], size=len(values))))
            for _ in range(repeats)]
    return {"n": len(values), "mean": float(values.mean()),
            "ci95": list(map(float, np.quantile(bootstrap, [0.025, 0.975]))),
            "p_one_sided": float((1 + sum(value >= values.mean() for value in null)) /
                                 (repeats + 1))}


def semantic_label_auc(cache: Path, train: list[dict], test: list[dict],
                       scores: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = [row["id"] for row in train + test]
    features = load_features(cache, ids)
    count = len(train)
    scaler = StandardScaler().fit(features[:count])
    scaled = scaler.transform(features)
    pca = PCA(n_components=min(32, count - 2, scaled.shape[1]), random_state=seed)
    projected = pca.fit_transform(scaled[:count])
    test_projected = pca.transform(scaled[count:])
    labels = np.asarray([scores[row["id"]] >= 0 for row in train], dtype=int)
    if np.unique(labels).size != 2:
        raise ValueError("Semantic-only training role contains one module-score sign")
    classifier = LogisticRegression(C=0.1, max_iter=1000).fit(projected, labels)
    return classifier.predict_proba(test_projected)[:, 1], test_projected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "module-dir", "axis-file", "activation-dir", "semantic-cache", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--n-prompts", type=int, default=80)
    parser.add_argument("--semantic-train-rows", type=int, default=400)
    parser.add_argument("--shots-per-label", type=int, default=2)
    parser.add_argument("--dose", type=float, default=0.5)
    parser.add_argument("--max-prefix-tokens", type=int, default=2048)
    parser.add_argument("--max-delta-rel-norm", type=float, default=0.05)
    parser.add_argument("--max-semantic-drift", type=float, default=0.08)
    parser.add_argument("--direct-steps", type=int, default=12)
    parser.add_argument("--random-candidates", type=int, default=8)
    parser.add_argument("--random-semantic-tolerance", type=float, default=0.02)
    parser.add_argument("--random-max-axis-leakage", type=float, default=0.3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--max-cache-hidden-error", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()
    if args.n_prompts < 8 or args.n_prompts % 2 or args.semantic_train_rows < 40:
        parser.error("n-prompts must be even and >=8; semantic-train-rows must be >=40")
    if args.random_semantic_tolerance < 0 or not 0 <= args.random_max_axis_leakage < 1:
        parser.error("Invalid random-control matching tolerances")
    module_dir = Path(args.module_dir)
    axis_path = Path(args.axis_file)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    axis = torch.load(axis_path, map_location="cpu", weights_only=False)
    frozen_direction, score_by_id, raw_by_id = frozen_axis_scalars(axis)
    print(f"[module-report] frozen code dimension={frozen_direction.numel()} "
          "report target=one-dimensional axis projection", flush=True)
    roles = stable_rows(axis_path.parent / "continuous_axis_all_assignments.csv",
                        Path(args.activation_dir) / "expanded_generation_data.jsonl",
                        args.seed, args.max_prefix_tokens)
    test = balanced(role_rows(roles, "trajectory_confirmatory", score_by_id),
                    score_by_id, args.n_prompts)
    train = role_rows(roles, "association_confirmatory", score_by_id)[:args.semantic_train_rows]
    demos = prompt_examples(role_rows(roles, "prototype_discovery", score_by_id),
                            score_by_id, args.shots_per_label)
    if len(train) < args.semantic_train_rows:
        raise ValueError("Insufficient prompt-disjoint semantic-only training rows")
    semantic_probabilities, semantic_test = semantic_label_auc(
        Path(args.semantic_cache), train, test, score_by_id, args.seed)
    config = read_json(module_dir.parent.parent / "config.json")
    layer_id = int(config["next_layer"])
    ids = [row["id"] for row in test]
    cached = load_features(Path(args.activation_dir) / f"layer_{layer_id:03d}.pt", ids)
    spec = get_model(args.target)
    enable_thinking = {"true": True, "false": False}.get(spec.chat_template_enable_thinking)
    model_args = argparse.Namespace(model_path=spec.resolved_path, dtype=spec.dtype,
                                    device_map="single", trust_remote_code=spec.trust_remote_code,
                                    gptq_backend=None, attn_implementation=None)
    model, tokenizer = load_model_and_tokenizer(model_args)
    model.eval()
    device = model_input_device(model)
    layer = get_decoder_layers(model)[layer_id]
    component_args = argparse.Namespace(decoupler_dir=str(module_dir.parent.parent),
                                        decoupler_checkpoint=None, decoupler_normalization=None,
                                        purifier_checkpoint=None, refined_module_dir=str(module_dir))
    decoupler, purifier, normalization, refiner, _ = core.load_refined_residual_components(
        component_args, cached.shape[1], device)
    axis_vector = frozen_direction.to(device)
    residual_mean = float(torch.as_tensor(axis["raw_residual_mean"]).item())
    residual_std = float(torch.as_tensor(axis["raw_residual_std"]).item())
    dose_core.AXIS = axis
    centers = torch.stack([-axis_vector, axis_vector])
    solvers = {"decoupler": decoupler, "purifier": purifier, "refiner": refiner,
               "normalization": normalization, "residual_std": residual_std}
    for name, source, target in (("positive", 0, 1), ("negative", 1, 0)):
        solvers[name] = dose_core.ContinuousDoseIntervention(
            context=core.GenerationContext(), layer_module=torch.nn.Identity(),
            decoupler=decoupler, purifier=purifier, normalization=normalization,
            refiner=refiner, code_centroids=centers, source_cluster=source,
            target_cluster=target, alpha=args.dose, code_intervention="direction",
            max_delta_rel_norm=args.max_delta_rel_norm, apply_to_generated=False,
            generated_patch_steps=0, control="main", semantic_penalty=5.0,
            z1_penalty=5.0, max_semantic_relative_delta=args.max_semantic_drift,
            direct_steps=args.direct_steps, direct_step_scale=1.5,
            direct_hidden_penalty=0.05)
    tokens = single_token_options(tokenizer)
    existing = {}
    rows_path = output / "report_rows.jsonl"
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = row
    contract = {"report_method_version": 2,
                "target": args.target, "module_dir": str(module_dir.resolve()),
                "axis_file": str(axis_path.resolve()), "test_ids": ids,
                "examples": demos, "diagnostic_enable_thinking": enable_thinking,
                "options": vars(args)}
    frozen = output / "frozen_report_plan.json"
    if frozen.exists() and read_json(frozen) != contract:
        raise ValueError("Frozen report plan changed; use a new output directory")
    if not frozen.exists():
        write_json(frozen, contract)
    print(f"[module-report] target={args.target} module={module_dir.name} "
          f"test={len(test)} replay=generation_step_kv_cache", flush=True)
    for index, sample in enumerate(tqdm(test, desc="Module self-report")):
        if sample["id"] in existing:
            continue
        prefix = list(map(int, sample["prompt_token_ids"]))
        hidden = capture_hidden(model, layer, device, prefix, len(prefix) - 1)
        error = float(np.linalg.norm(cached[index] - hidden.cpu().numpy()) /
                      max(np.linalg.norm(cached[index]), 1e-6))
        if not np.isfinite(error) or error > args.max_cache_hidden_error:
            raise ValueError(f"Cached later-layer activation does not reproduce for {sample['id']}: "
                             f"relative_l2_error={error:.6g} "
                             f"max={args.max_cache_hidden_error:.6g}")
        before_code, before_z1, before_sem = code_state(
            hidden, decoupler, purifier, refiner, normalization)
        raw_code = float((before_code @ axis_vector.view(-1, 1)).item())
        if not np.isfinite(raw_code):
            raise ValueError(f"Runtime module code is non-finite for {sample['id']}")
        fixed_semantic_prediction = raw_by_id[sample["id"]] - (
            score_by_id[sample["id"]] * residual_std + residual_mean)
        observed_score = (raw_code - fixed_semantic_prediction - residual_mean) / residual_std
        mean = normalization["next_mean"].to(hidden).view(1, -1)
        std = normalization["next_std"].to(hidden).view(1, -1).clamp_min(1e-6)
        interventions = {"baseline": {"hidden": hidden, "code": before_code,
                                      "semantic_drift": 0.0, "hidden_relative": 0.0,
                                      "accepted": True}}
        for name in ("positive", "negative"):
            result = solvers[name]._direct_optimize_hidden(
                hidden.view(1, -1), mean, std, before_code, before_z1, before_sem,
                False, torch.zeros(1, dtype=torch.bool, device=device),
                torch.zeros(1, device=device))
            replacement = result["best_replacement"][0].detach()
            interventions[name] = {
                "hidden": replacement, "code": result["best_code"].detach(),
                "semantic_drift": relative_change(result["best_semantic_after"], before_sem),
                "hidden_relative": relative_change(replacement, hidden),
                "code_shift": float(((result["best_code"] - before_code) @
                                      axis_vector.view(-1, 1)).item()),
                "accepted": bool(result["best_accepted"][0].item())}
        seed = args.seed + int(hashlib.sha256(sample["id"].encode()).hexdigest()[:8], 16)
        controls = random_pair(hidden, interventions, before_sem, before_code, solvers,
                               axis_vector, seed, args.random_candidates,
                               args.random_semantic_tolerance, args.random_max_axis_leakage)
        interventions.update(controls)
        result_row = {"id": sample["id"], "score_before": observed_score,
                      "cached_score": score_by_id[sample["id"]],
                      "cache_hidden_relative_error": error,
                      "cache_score_difference": observed_score - score_by_id[sample["id"]],
                      "cache_label_agrees": bool((observed_score >= 0) ==
                                                 (score_by_id[sample["id"]] >= 0)),
                      "modes": {}}
        for mode, item in interventions.items():
            code_value = float((item["code"] @ axis_vector.view(-1, 1)).item())
            delivered_score = (code_value - fixed_semantic_prediction - residual_mean) / residual_std
            report_scores = []
            replay_errors = []
            option_masses = []
            for high_is_a in (True, False):
                question = make_question(demos, high_is_a)
                logodds, replay_error, option_mass = report_logits(
                    model, layer, device, tokenizer, prefix, question, tokens,
                    None if mode == "baseline" else item["hidden"], hidden,
                    enable_thinking)
                report_scores.append(logodds * (1 if high_is_a else -1))
                replay_errors.append(replay_error)
                option_masses.append(option_mass)
            if max(replay_errors) > 0.01:
                raise ValueError(f"Diagnostic turn changed cached prefix state for {sample['id']}")
            result_row["modes"][mode] = {"replay_relative_error": max(replay_errors)}
            result_row["modes"][mode].update({
                "report_high_logodds": float(np.mean(report_scores)),
                "answer_option_mass": float(np.mean(option_masses)),
                "delivered_score": delivered_score,
                "score_delta": delivered_score - observed_score,
                "semantic_drift": item["semantic_drift"],
                "hidden_relative": item["hidden_relative"],
                "accepted": bool(item.get("accepted", True)),
                "random_matched": item.get("matched")})
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result_row, allow_nan=False) + "\n")
            handle.flush()
        existing[sample["id"]] = result_row

    for name in ("positive", "negative"):
        solvers[name].close()
    rows = [existing[sid] for sid in ids]
    truth = np.asarray([row["score_before"] >= 0 for row in rows], dtype=int)
    if np.unique(truth).size != 2:
        raise ValueError("Runtime replay has only one module-score sign; report AUC is undefined")
    reports = np.asarray([row["modes"]["baseline"]["report_high_logodds"] for row in rows])
    conditional = conditional_report_gain(semantic_test[:, :16],
                                          np.asarray([row["score_before"] for row in rows]),
                                          reports, args.seed)
    natural_auc = float(roc_auc_score(truth, reports))
    natural_bacc = float(balanced_accuracy_score(truth, reports > 0))
    semantic_auc = float(roc_auc_score(truth, semantic_probabilities))
    delivered = [row for row in rows if row["modes"]["positive"]["accepted"] and
                 row["modes"]["negative"]["accepted"] and
                 row["modes"]["positive"]["score_delta"] >= 0.1 and
                 row["modes"]["negative"]["score_delta"] <= -0.1]
    matched = [row for row in delivered if row["modes"]["random_positive"]["random_matched"]
               and row["modes"]["random_negative"]["random_matched"]]
    real = np.asarray([row["modes"]["positive"]["report_high_logodds"] -
                       row["modes"]["negative"]["report_high_logodds"] for row in delivered])
    matched_real = np.asarray([row["modes"]["positive"]["report_high_logodds"] -
                               row["modes"]["negative"]["report_high_logodds"]
                               for row in matched])
    random_effect = np.asarray([row["modes"]["random_positive"]["report_high_logodds"] -
                                row["modes"]["random_negative"]["report_high_logodds"]
                                for row in matched])
    scores = np.asarray([row["score_before"] for row in rows])
    cache_hidden_errors = np.asarray([row["cache_hidden_relative_error"] for row in rows])
    cache_score_differences = np.asarray([row["cache_score_difference"] for row in rows])
    cache_sign_agreement = float(np.mean([row["cache_label_agrees"] for row in rows]))
    cache_score_error_p95 = float(np.quantile(np.abs(cache_score_differences), 0.95))
    baseline_option_masses = [
        row["modes"]["baseline"]["answer_option_mass"] for row in rows
        if "answer_option_mass" in row["modes"]["baseline"]
    ]
    report_rho = (float(spearmanr(scores, reports).statistic)
                  if len(np.unique(scores)) > 1 and len(np.unique(reports)) > 1 else None)
    matched_drift_gap = [
        abs(row["modes"][random_name]["semantic_drift"] -
            row["modes"][reference_name]["semantic_drift"])
        for row in matched
        for random_name, reference_name in (("random_positive", "positive"),
                                            ("random_negative", "negative"))
    ]
    matched_hidden_gap = [
        abs(row["modes"][random_name]["hidden_relative"] -
            row["modes"][reference_name]["hidden_relative"])
        for row in matched
        for random_name, reference_name in (("random_positive", "positive"),
                                            ("random_negative", "negative"))
    ]
    summary = {
        "report_method_version": 2,
        "n_test": len(rows), "n_delivered_pairs": len(delivered),
        "n_matched_random_pairs": len(matched),
        "cache_hidden_relative_error_median": float(np.median(cache_hidden_errors)),
        "cache_hidden_relative_error_max": float(np.max(cache_hidden_errors)),
        "cache_score_absolute_error_median": float(np.median(np.abs(cache_score_differences))),
        "cache_score_absolute_error_p95": cache_score_error_p95,
        "cache_score_sign_agreement_rate": cache_sign_agreement,
        "measurement_quality": ("pass" if cache_sign_agreement >= 0.95 and
                                cache_score_error_p95 <= 0.25 else "review"),
        "diagnostic_replay_relative_error_max": float(max(
            row["modes"][mode]["replay_relative_error"]
            for row in rows for mode in row["modes"])),
        "natural_report_auc": natural_auc, "natural_report_balanced_accuracy": natural_bacc,
        "natural_report_score_spearman": report_rho,
        "semantic_only_label_auc": semantic_auc,
        "baseline_answer_option_mass_mean": (
            float(np.mean(baseline_option_masses)) if baseline_option_masses else None),
        "baseline_answer_option_mass_below_1pct": (
            float(np.mean(np.asarray(baseline_option_masses) < 0.01))
            if baseline_option_masses else None),
        "module_added_report_fit": conditional,
        "paired_real": paired_test(real, args.seed, args.bootstrap_samples),
        "paired_random": paired_test(random_effect, args.seed + 1, args.bootstrap_samples),
        "paired_real_minus_random": paired_test(matched_real - random_effect, args.seed + 2,
                                                 args.bootstrap_samples),
        "delivered_positive_rate": len(delivered) / len(rows),
        "matched_random_rate_given_delivery": len(matched) / max(len(delivered), 1),
        "matched_semantic_drift_gap_mean": (float(np.mean(matched_drift_gap))
                                             if matched_drift_gap else None),
        "matched_hidden_relative_gap_mean": (float(np.mean(matched_hidden_gap))
                                              if matched_hidden_gap else None),
        "mean_positive_score_delta": float(np.mean([row["modes"]["positive"]["score_delta"]
                                                for row in rows])),
        "mean_negative_score_delta": float(np.mean([row["modes"]["negative"]["score_delta"]
                                                for row in rows])),
        "mean_random_absolute_score_delta": float(np.mean([
            (abs(row["modes"]["random_positive"]["score_delta"]) +
             abs(row["modes"]["random_negative"]["score_delta"])) / 2 for row in rows])),
        "mean_semantic_drift_positive": float(np.mean([row["modes"]["positive"]["semantic_drift"]
                                                    for row in rows])),
        "mean_semantic_drift_negative": float(np.mean([row["modes"]["negative"]["semantic_drift"]
                                                    for row in rows])),
        "mean_semantic_drift_random": float(np.mean([
            (row["modes"]["random_positive"]["semantic_drift"] +
             row["modes"]["random_negative"]["semantic_drift"]) / 2 for row in rows])),
        "scope": "Fixed semantic prediction for each prompt; direct trust-region module edits; "
                 "two counterbalanced A/B mappings; paired hidden-norm and semantic-drift-matched "
                 "random control; step-zero prompt-disjoint heldout role. No model was trained "
                 "to expose this analyst-defined score."
    }
    write_json(output / "module_report_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
