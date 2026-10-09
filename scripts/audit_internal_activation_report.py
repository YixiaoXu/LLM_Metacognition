#!/usr/bin/env python3
"""Ask a target LLM about one frozen earlier-layer neuron on held-out prompts.

The state is measured before the diagnostic question is appended. A/B answer
labels are counterbalanced; a small optional earlier-layer flip tests whether
the report follows the actual hidden state rather than the prompt alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
from metacog.audits.internal_report_stats import direct_report_classification, predict_checks
from metacog.audits.neuron_report_replay import generation_step_report_forward
from metacog.models.capture import get_decoder_layers, last_token_logits_kwargs
from metacog.models.chat_tokens import diagnostic_turn_ids
from metacog.models.loading import load_model_and_tokenizer, model_input_device
from metacog.models.registry import get_model


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(path: Path, ids: list[str]) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    index = {str(sid): i for i, sid in enumerate(payload["ids"])}
    missing = [sid for sid in ids if sid not in index]
    if missing:
        raise ValueError(f"{path}: {len(missing)} selected IDs are absent")
    return torch.as_tensor(payload["features"])[[index[sid] for sid in ids]].float().numpy()


def choose_rows(axis_path: Path, expanded_path: Path, role: str, maximum: int, seed: int,
                max_prefix_tokens: int):
    with axis_path.open(encoding="utf-8") as handle:
        axis = {r["id"]: r for r in csv.DictReader(handle)
                if r["analysis_role"] == role and r["id"].endswith("::step000")}
    if not axis:
        raise ValueError(f"No step-zero {role} rows in {axis_path}")
    rows = []
    with expanded_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in axis and len(row.get("prompt_token_ids", [])) <= max_prefix_tokens:
                rows.append(row)
    if not rows:
        raise ValueError("No eligible exact-prefix rows remain after the length limit")
    rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())
    return rows[:maximum], axis


def single_token_options(tokenizer) -> tuple[int, int]:
    for a, b in ((" A", " B"), ("A", "B")):
        ta = tokenizer.encode(a, add_special_tokens=False)
        tb = tokenizer.encode(b, add_special_tokens=False)
        if len(ta) == len(tb) == 1 and ta[0] != tb[0]:
            return ta[0], tb[0]
    raise ValueError("A/B report options are not single tokens for this tokenizer")


def report_forward(model, layer, device, ids: list[int], neuron: int, marker: int,
                   patch_value: float | None = None, *, replay_mode: str = "legacy_full"):
    if replay_mode == "generation_step_kv":
        return generation_step_report_forward(model, layer, device, ids, neuron, marker, patch_value)
    if replay_mode != "legacy_full":
        raise ValueError(f"Unknown neuron-report replay mode: {replay_mode}")
    observed = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        value = float(hidden[0, marker, neuron].float().item())
        observed["before"] = value
        observed["hidden_norm"] = float(hidden[0, marker].float().norm().item())
        if patch_value is None:
            observed["after"] = value
            return None
        updated = hidden.clone()
        updated[0, marker, neuron] = patch_value
        observed["after"] = float(updated[0, marker, neuron].float().item())
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                            **last_token_logits_kwargs(model))
        logits = outputs.logits[0, -1].float()
        return logits, observed
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--module-dir", required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--axis-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--role", default="trajectory_confirmatory")
    parser.add_argument("--max-rows", type=int, default=160)
    parser.add_argument("--max-prefix-tokens", type=int, default=3000)
    parser.add_argument("--flip-rows", type=int, default=16)
    parser.add_argument("--flip-norm-budget", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()
    if not 0 <= args.flip_rows <= args.max_rows or args.max_rows < 60:
        raise ValueError("Invalid report or flip sample count")
    if not 0 < args.flip_norm_budget < 1:
        raise ValueError("Flip hidden-state norm budget must lie in (0, 1)")
    module_dir = Path(args.module_dir)
    dec_dir = module_dir.parent.parent
    config = read_json(dec_dir / "config.json")
    module = read_json(module_dir / "module_summary.json")
    loading = np.abs(module["module_loading"])
    neuron = int(module["module_member_neurons"][int(np.argmax(loading))])
    target_file = module_dir / "soft_residual_intervention_targets.pt"
    target = torch.load(target_file, map_location="cpu", weights_only=False)
    if list(map(int, target["selected_neurons"])) != list(map(int, module["module_member_neurons"])):
        raise ValueError("Frozen refiner neuron order differs from module summary")
    mean = float(torch.as_tensor(target["continuous_mean"]).flatten()[neuron])
    std = float(torch.as_tensor(target["continuous_std"]).flatten()[neuron])
    layer_id = int(config["layer_i"])
    activation_dir = Path(args.activation_dir)
    rows, axis = choose_rows(Path(args.axis_csv), activation_dir / "expanded_generation_data.jsonl",
                             args.role, args.max_rows, args.seed, args.max_prefix_tokens)
    if len(rows) < 60:
        raise ValueError(f"Only {len(rows)} held-out prefixes survive length filtering; need at least 60")
    ids = [row["id"] for row in rows]
    measured = load_features(activation_dir / f"layer_{layer_id:03d}.pt", ids)[:, neuron]
    semantic = load_features(Path(args.semantic_cache), ids)
    meta = np.array([[float(axis[sid]["continuous_meta_score"])] for sid in ids])
    spec = get_model(args.target)
    model_args = argparse.Namespace(model_path=spec.resolved_path, dtype=spec.dtype,
                                    device_map="single", trust_remote_code=spec.trust_remote_code,
                                    gptq_backend=None, attn_implementation=None)
    model, tokenizer = load_model_and_tokenizer(model_args)
    device = model_input_device(model)
    layer = get_decoder_layers(model)[layer_id]
    a_token, b_token = single_token_options(tokenizer)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_rows = []
    flip_indices = set(np.argsort(np.abs(measured - mean))[:args.flip_rows].tolist())
    for index, row in enumerate(tqdm(rows, desc="Direct activation report")):
        prefix = list(map(int, row["prompt_token_ids"]))
        if not prefix:
            raise ValueError(f"No frozen prompt tokens: {row['id']}")
        marker = len(prefix) - 1
        recorded = float(measured[index])
        scores, captures = [], []
        patched_scores, patched_states = [], []
        for high_is_a in (True, False):
            question = (f"Diagnostic question about the immediately preceding assistant token: "
                        f"was decoder layer {layer_id} neuron {neuron} above or below its "
                        f"reference mean ({mean:.5f})? Answer with exactly one letter. "
                        + ("A = above; B = below." if high_is_a else "A = below; B = above."))
            turn = diagnostic_turn_ids(tokenizer, question)
            full = prefix + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []) + turn
            logits, state = report_forward(model, layer, device, full, neuron, marker)
            scores.append(float((logits[a_token] - logits[b_token]).item()) * (1 if high_is_a else -1))
            captures.append(state["before"])
            if index in flip_indices:
                delta = max(0.5 * std, abs(state["before"] - mean) + 0.25 * std)
                patch = mean - delta if state["before"] >= mean else mean + delta
                max_delta = args.flip_norm_budget * state["hidden_norm"]
                patch = state["before"] + float(np.clip(patch - state["before"], -max_delta, max_delta))
                patched_logits, patched = report_forward(model, layer, device, full, neuron, marker, patch)
                patched_scores.append(float((patched_logits[a_token] - patched_logits[b_token]).item()) *
                                      (1 if high_is_a else -1))
                patched_states.append(patched["after"])
        if max(abs(value - recorded) for value in captures) > max(0.05, 0.1 * std):
            raise ValueError(f"Cached earlier activation does not reproduce for {row['id']}; no report claim is valid")
        item = {"id": row["id"], "recorded_activation": recorded,
                "recorded_high": recorded > mean, "report_high_logodds": float(np.mean(scores)),
                "baseline_report_high": float(np.mean(scores)) > 0,
                "module_score": float(meta[index, 0])}
        if patched_scores:
            item.update({"flipped_activation": float(np.mean(patched_states)),
                         "flipped_high": float(np.mean(patched_states)) > mean,
                         "flipped_report_high_logodds": float(np.mean(patched_scores)),
                         "report_logodds_change": float(np.mean(patched_scores) - np.mean(scores))})
        result_rows.append(item)
    with (output / "report_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    checks = predict_checks(result_rows, semantic, meta, args.seed)
    truth = np.array([int(r["recorded_high"]) for r in result_rows])
    report_scores = np.array([r["report_high_logodds"] for r in result_rows])
    checks["direct_report"] = direct_report_classification(truth, report_scores, args.seed)
    flipped = [r for r in result_rows if "flipped_activation" in r]
    if flipped:
        effective = [r for r in flipped if r["recorded_high"] != r["flipped_high"]]
        checks["activation_flip"] = {"n_attempted": len(flipped), "n_effective": len(effective),
                                     "observed_label_flip_rate": len(effective) / len(flipped)}
        if effective:
            changes = np.array([r["report_logodds_change"] * (1 if r["flipped_high"] else -1)
                                for r in effective])
            checks["activation_flip"].update({"mean_report_shift_toward_new_state": float(changes.mean()),
                                               "positive_shift_rate": float((changes > 0).mean())})
    checks.update({"report_statistics_version": 2, "neuron": neuron, "layer": layer_id, "frozen_threshold": mean,
                   "n": len(result_rows), "scope": "held-out report diagnostic; not a fourth three-ring gate"})
    (output / "report_summary.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
