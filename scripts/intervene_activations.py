#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm

from model_utils import (
    build_generation_prompt as shared_build_generation_prompt,
    generation_eos_token_id,
    get_decoder_layers as shared_get_decoder_layers,
    load_model_and_tokenizer,
    model_input_device as shared_model_input_device,
    parse_max_memory as shared_parse_max_memory,
    prompt_content as shared_prompt_content,
    resolve_device_map as shared_resolve_device_map,
)


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:(?:\d[\d,]*(?:\.\d+)?)|(?:\.\d+))(?:\s*/\s*[-+]?(?:(?:\d[\d,]*(?:\.\d+)?)|(?:\.\d+)))?"
)


def decimal_from_number_text(text: Optional[str]) -> Optional[Decimal]:
    if text is None:
        return None
    value = re.sub(r"\s+", "", str(text).strip().replace(",", "").replace("$", ""))
    if not value:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = Decimal(denominator)
            return None if denominator_value == 0 else Decimal(numerator) / denominator_value
        return Decimal(value)
    except InvalidOperation:
        return None


def normalize_number(text: Optional[str]) -> Optional[str]:
    number = decimal_from_number_text(text)
    if number is not None:
        value = format(number.normalize(), "f")
        return value.rstrip("0").rstrip(".") if "." in value else value
    if text is None:
        return None
    value = str(text).strip().replace(",", "")
    return value or None


def _choice_number(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    label = str(value).strip().strip("()[]{}.:#").upper()
    if label in {"1", "2", "3", "4", "5"}:
        return label
    if label in {"A", "B", "C", "D", "E"}:
        return str(ord(label) - ord("A") + 1)
    return None


def extract_mathqa_choice(text: str) -> Dict:
    """Parse an explicitly stated 1-based MathQA option without last-number guessing."""
    cleaned = str(text or "").replace("\u0120", " ").replace("\u010a", "\n")
    candidates = []
    patterns = (
        (r"#{4,}\s*(?:option(?:\s+number)?\s*)?(?:is\s*|[:=]\s*)?[#(\[]*([1-5A-E])\b", "hash_choice"),
        (r"\b(?:selected|correct|final)\s+(?:answer|option(?:\s+number)?)\s*(?:is\s*|[:=]\s*)?(?:option\s*)?[#(\[]*([1-5A-E])\b", "explicit_choice"),
        (r"\b(?:correct\s+)?option(?:\s+number)?\s*(?:is\s*|[:=]\s*)?[#(\[]*([1-5A-E])\b", "option_choice"),
        (r"\\boxed\s*\{\s*([1-5A-E])\s*\}", "boxed_choice"),
    )
    for pattern, source in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE | re.MULTILINE):
            choice = _choice_number(match.group(1))
            if choice is not None:
                candidates.append((match.start(), source, choice))
    if not candidates:
        return {"answer": None, "source": "mathqa_choice_missing", "candidates": []}
    candidates.sort(key=lambda item: item[0])
    _, source, answer = candidates[-1]
    return {
        "answer": answer,
        "source": source,
        "candidates": [item[2] for item in candidates],
    }


def extract_strict_hash(text: str) -> Dict:
    cleaned = str(text or "").replace("\u0120", " ").replace("\u010a", "\n")
    matches = []
    for marker in re.finditer(r"#{4,}", cleaned):
        tail = cleaned[marker.end() :].lstrip()
        match = NUMBER_PATTERN.match(tail)
        if match is not None:
            matches.append(normalize_number(match.group(0)))
    answer = matches[-1] if matches else None
    return {"answer": answer, "source": "strict_hash" if answer else "strict_hash_missing", "candidates": matches}


def evaluate_generated_answer(
    text: str, row: Dict, method: str = "none", tolerance: float = 0.0
) -> Dict:
    if method == "none":
        return {}
    if method != "mathqa_choice":
        raise ValueError(f"Unsupported answer extraction method: {method}")
    primary = extract_mathqa_choice(text)
    strict = extract_strict_hash(text)
    gold = normalize_number(
        row.get("numeric_answer")
        or row.get("correct_option_number")
        or row.get("gold_answer")
    )
    prediction = normalize_number(primary["answer"])
    return {
        "gold_answer": gold,
        "answer_extraction": method,
        "pred_answer": prediction,
        "pred_answer_source": primary["source"],
        "pred_answer_candidates": primary["candidates"],
        "parse_success": prediction is not None,
        "correct": prediction is not None and gold is not None and prediction == gold,
        "strict_hash_pred_answer": strict["answer"],
        "strict_hash_parse_success": strict["answer"] is not None,
        "strict_hash_correct": (
            strict["answer"] is not None
            and gold is not None
            and normalize_number(strict["answer"]) == gold
        ),
        "valid_choice": prediction in {"1", "2", "3", "4", "5"},
    }


def group_name(group_id: int) -> str:
    return ["low", "mid", "high"][int(group_id)] if int(group_id) in {0, 1, 2} else "unknown"


def parse_record_layers(spec: str, layer_i: int, next_layer: int) -> List[int]:
    spec = spec.strip().lower()
    if spec in {"", "none"}:
        return []
    if spec == "auto":
        return sorted({layer_i, next_layer})
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "layer_i":
            layers.add(layer_i)
        elif part == "next":
            layers.add(next_layer)
        elif "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def parse_max_memory(items: Optional[List[str]]) -> Optional[Dict[int, str]]:
    return shared_parse_max_memory(items)


def resolve_device_map(name: str):
    return shared_resolve_device_map(name)


def get_decoder_layers(model) -> Sequence[torch.nn.Module]:
    return shared_get_decoder_layers(model)


def model_input_device(model) -> torch.device:
    return shared_model_input_device(model)


def split_layer_output(output):
    if isinstance(output, tuple):
        return output[0], lambda hidden: (hidden,) + output[1:]
    if isinstance(output, list):
        return output[0], lambda hidden: [hidden] + output[1:]
    return output, lambda hidden: hidden


class GenerationContext:
    def __init__(self):
        self.mode = "baseline"
        self.current_ids: List[str] = []
        self.apply_mask: Optional[torch.Tensor] = None
        self.patch_values: Optional[torch.Tensor] = None
        self.generated_step: int = 0
        self.current_step: int = -1


class HiddenFeatureIntervention:
    def __init__(
        self,
        context: GenerationContext,
        layer_module: torch.nn.Module,
        feature_indices: torch.Tensor,
        mode: str,
        mean_values: Optional[torch.Tensor],
        threshold_values: Optional[torch.Tensor],
        clamp_factor: float,
        scale_factor: float,
        noise_scale: float,
        apply_to_generated: bool,
    ):
        self.context = context
        self.feature_indices = feature_indices.long()
        self.mode = mode
        self.mean_values = mean_values
        self.threshold_values = threshold_values
        self.clamp_factor = clamp_factor
        self.scale_factor = scale_factor
        self.noise_scale = noise_scale
        self.apply_to_generated = apply_to_generated
        self.handle = layer_module.register_forward_hook(self.hook)

    def close(self) -> None:
        self.handle.remove()

    def hook(self, _module, _inputs, output):
        hidden, rebuild = split_layer_output(output)
        if hidden.ndim != 3 or self.feature_indices.numel() == 0:
            return output
        seq_len = hidden.size(1)
        is_prefill = seq_len > 1
        if not is_prefill and not self.apply_to_generated:
            return output

        batch = hidden.size(0)
        apply_mask = self.context.apply_mask
        if apply_mask is None or apply_mask.numel() != batch:
            apply_mask = torch.ones(batch, dtype=torch.bool)
        rows = torch.nonzero(apply_mask.bool(), as_tuple=False).flatten()
        if rows.numel() == 0:
            return output

        features = self.feature_indices.to(hidden.device)
        rows = rows.to(hidden.device)
        pos = hidden.size(1) - 1 if is_prefill else 0
        edited = hidden.clone()
        current = edited[rows, pos][:, features]
        replacement = self.make_replacement(current, rows, hidden.device, hidden.dtype)
        edited[rows[:, None], pos, features[None, :]] = replacement
        return rebuild(edited)

    def make_replacement(self, current: torch.Tensor, rows: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.mode in {"zero_ablate", "inactive_clamp"}:
            return torch.zeros_like(current)
        if self.mode == "mean_ablate":
            if self.mean_values is None:
                return torch.zeros_like(current)
            return self.mean_values.to(device=device, dtype=dtype).view(1, -1).expand_as(current)
        if self.mode == "active_clamp":
            if self.threshold_values is None:
                return current
            threshold = self.threshold_values.to(device=device, dtype=dtype).view(1, -1)
            sign = current.sign()
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            return sign * threshold * self.clamp_factor
        if self.mode == "sign_flip":
            return -current
        if self.mode == "scale":
            return current * self.scale_factor
        if self.mode == "noise":
            scale = self.noise_scale
            if self.threshold_values is not None:
                scale_tensor = self.threshold_values.to(device=device, dtype=dtype).view(1, -1) * scale
                return torch.randn_like(current) * scale_tensor
            return torch.randn_like(current) * scale
        if self.mode == "sample_patch":
            if self.context.patch_values is None:
                return current
            patch = self.context.patch_values.to(device=device, dtype=dtype)
            return patch[rows]
        raise ValueError(f"Unknown intervention mode: {self.mode}")


class ActivationRecorder:
    def __init__(self, context: GenerationContext, layers: Sequence[torch.nn.Module], layer_ids: Sequence[int], feature_indices: torch.Tensor):
        self.context = context
        self.feature_indices = feature_indices.long()
        self.layer_ids = list(layer_ids)
        self.records: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
        self.handles = [
            layers[layer_id].register_forward_hook(self.make_hook(layer_id))
            for layer_id in self.layer_ids
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def make_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            hidden, _ = split_layer_output(output)
            if hidden.ndim != 3 or hidden.size(1) <= 1 or self.feature_indices.numel() == 0:
                return output
            if not self.context.current_ids:
                return output
            features = self.feature_indices.to(hidden.device)
            values = hidden[:, -1, features].detach().float().cpu()
            mode_records = self.records.setdefault(self.context.mode, {})
            for row_idx, sample_id in enumerate(self.context.current_ids[: values.size(0)]):
                mode_records.setdefault(sample_id, {})[layer_id] = values[row_idx]
            return output
        return hook


def load_intervention_targets(path: str) -> Dict:
    bundle = safe_torch_load(path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected a dict in {path}. Use binary_analysis/intervention_targets.pt from train_decoupler.py.")
    return bundle


def build_eval_rows(data_path: str, targets: Dict, group_basis: str, eval_group: str, max_samples: int) -> List[Dict]:
    data_rows = {row["id"]: row for row in read_jsonl(data_path)}
    val_ids = list(targets.get("val_ids", []))
    group_key = "val_true_score_group" if group_basis == "true" else "val_pred_score_group"
    group_tensor = targets.get(group_key)
    if group_tensor is None:
        group_tensor = torch.ones(len(val_ids), dtype=torch.long)

    rows = []
    for idx, sample_id in enumerate(val_ids):
        if sample_id not in data_rows:
            continue
        group = group_name(int(group_tensor[idx].item()))
        if eval_group != "all" and group != eval_group:
            continue
        item = dict(data_rows[sample_id])
        item["val_index"] = idx
        item["meta_group"] = group
        rows.append(item)
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return rows


def prompt_content(row: Dict, prompt_style: str) -> str:
    return shared_prompt_content(row, prompt_style)


def build_generation_prompt(row: Dict, tokenizer, args: argparse.Namespace) -> str:
    return shared_build_generation_prompt(row, tokenizer, args)


def load_layer_feature_map(activation_dir: str, layer_i: int, ids: Sequence[str], feature_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    path = os.path.join(activation_dir, f"layer_{layer_i:03d}.pt")
    obj = safe_torch_load(path)
    all_ids = list(obj["ids"])
    features = obj["features"].float()
    id_to_index = {sample_id: idx for idx, sample_id in enumerate(all_ids)}
    feature_indices = feature_indices.long()
    result = {}
    for sample_id in ids:
        if sample_id in id_to_index:
            result[sample_id] = features[id_to_index[sample_id], feature_indices].clone()
    return result


def build_patch_values(
    batch_rows: List[Dict],
    donor_ids: List[str],
    feature_map: Dict[str, torch.Tensor],
    feature_count: int,
    offset: int,
) -> torch.Tensor:
    values = []
    for row_idx, _row in enumerate(batch_rows):
        if donor_ids:
            donor_id = donor_ids[(offset + row_idx) % len(donor_ids)]
            values.append(feature_map[donor_id])
        else:
            values.append(torch.zeros(feature_count))
    return torch.stack(values, dim=0)


def batch_iter(rows: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def generate_for_mode(
    model,
    tokenizer,
    rows: List[Dict],
    context: GenerationContext,
    mode: str,
    args: argparse.Namespace,
    targets: Dict,
    donor_ids: List[str],
    feature_map: Dict[str, torch.Tensor],
    feature_count: int,
) -> List[Dict]:
    context.mode = mode
    output_rows = []
    input_device = model_input_device(model)
    do_sample = args.temperature > 0.0
    patch_offset = 0
    group_key = "meta_group"
    for batch in tqdm(list(batch_iter(rows, args.batch_size)), desc=f"Generate {mode}", leave=True):
        prompts = [build_generation_prompt(row, tokenizer, args) for row in batch]
        ids = [row["id"] for row in batch]
        context.current_ids = ids
        if args.target_group == "all":
            apply_mask = torch.ones(len(batch), dtype=torch.bool)
        else:
            apply_mask = torch.tensor([row[group_key] == args.target_group for row in batch], dtype=torch.bool)
        context.apply_mask = apply_mask
        if mode == "sample_patch":
            context.patch_values = build_patch_values(batch, donor_ids, feature_map, feature_count, patch_offset)
            patch_offset += len(batch)
        else:
            context.patch_values = None

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        prompt_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        eos_token_id = generation_eos_token_id(model, tokenizer)
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )
        prompt_width = encoded["input_ids"].size(1)
        generated_token_ids = generated[:, prompt_width:]
        decoded = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        generated_token_counts = (generated_token_ids != tokenizer.pad_token_id).sum(dim=1).detach().cpu().tolist()
        for row, prompt, prompt_tokens, generated_tokens, text, apply in zip(
            batch,
            prompts,
            prompt_token_counts,
            generated_token_counts,
            decoded,
            apply_mask.tolist(),
        ):
            answer_eval = evaluate_generated_answer(text, row, args.answer_extraction, args.answer_tolerance)
            output_rows.append(
                {
                    "id": row["id"],
                    "val_index": row.get("val_index"),
                    "meta_group": row.get("meta_group"),
                    "mode": mode,
                    "intervention_applied": bool(apply and mode != "baseline"),
                    "question": row.get("question"),
                    "prompt": prompt,
                    "source_prompt": row.get("prompt"),
                    "prompt_style": args.prompt_style,
                    "use_chat_template": args.use_chat_template,
                    "prompt_tokens": int(prompt_tokens),
                    "generated_tokens": int(generated_tokens),
                    "generated_text": text,
                    **answer_eval,
                }
            )
    context.current_ids = []
    context.apply_mask = None
    context.patch_values = None
    return output_rows


def activation_delta_columns(
    row: Dict,
    baseline_records: Dict[str, Dict[int, torch.Tensor]],
    mode_records: Dict[str, Dict[int, torch.Tensor]],
    record_layers: Sequence[int],
) -> Dict:
    sample_id = row["id"]
    values = {}
    for layer in record_layers:
        base = baseline_records.get(sample_id, {}).get(layer)
        other = mode_records.get(sample_id, {}).get(layer)
        key = f"act_delta_l2_layer_{layer:03d}"
        cos_key = f"act_delta_cos_layer_{layer:03d}"
        if base is None or other is None:
            values[key] = float("nan")
            values[cos_key] = float("nan")
            continue
        diff = other - base
        values[key] = diff.pow(2).sum().sqrt().item()
        denom = base.norm().clamp_min(1e-8) * other.norm().clamp_min(1e-8)
        values[cos_key] = (base * other).sum().div(denom).item()
    return values


def summarize_accuracy_rows(rows: List[Dict]) -> Dict:
    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    parseable = sum(1 for row in rows if row.get("parse_success"))
    result = {
        "n": n,
        "correct_n": correct,
        "accuracy": correct / max(n, 1),
        "parseable_n": parseable,
        "parse_rate": parseable / max(n, 1),
        "unparseable_n": n - parseable,
        "accuracy_given_parse": correct / parseable if parseable else None,
    }
    return result


def mcnemar_exact_p(correct_to_wrong: int, wrong_to_correct: int) -> Optional[float]:
    discordant = int(correct_to_wrong) + int(wrong_to_correct)
    if discordant == 0:
        return None
    k = min(int(correct_to_wrong), int(wrong_to_correct))
    log_two = math.log(2.0)
    tail = 0.0
    for i in range(k + 1):
        log_prob = math.lgamma(discordant + 1) - math.lgamma(i + 1) - math.lgamma(discordant - i + 1) - discordant * log_two
        tail += math.exp(log_prob)
    return min(1.0, 2.0 * tail)


def bootstrap_accuracy_delta_ci(
    rows: List[Dict],
    baseline_by_id: Dict[str, Dict],
    bootstrap_samples: int,
    seed: int,
) -> Optional[List[float]]:
    if bootstrap_samples <= 0 or not rows:
        return None
    deltas = []
    for row in rows:
        base = baseline_by_id.get(row["id"], {})
        deltas.append(float(bool(row.get("correct"))) - float(bool(base.get("correct"))))
    delta_tensor = torch.tensor(deltas, dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sample_indices = torch.randint(
        low=0,
        high=len(deltas),
        size=(bootstrap_samples, len(deltas)),
        generator=generator,
    )
    sample_means = delta_tensor[sample_indices].mean(dim=1)
    low, high = torch.quantile(sample_means, torch.tensor([0.025, 0.975])).tolist()
    return [float(low), float(high)]


def summarize_mode(
    rows: List[Dict],
    baseline_by_id: Dict[str, Dict],
    record_layers: Sequence[int],
    bootstrap_samples: int,
    seed: int,
) -> Dict:
    n = len(rows)
    baseline_rows = [baseline_by_id[row["id"]] for row in rows if row["id"] in baseline_by_id]
    has_accuracy = bool(rows) and all("correct" in row for row in rows)
    has_baseline_accuracy = bool(baseline_rows) and all(
        "correct" in row for row in baseline_rows
    )
    has_parseability = bool(rows) and all("parse_success" in row for row in rows)
    has_answers = bool(rows) and all(
        "pred_answer" in row and "pred_answer" in baseline_by_id.get(row["id"], {})
        for row in rows
    )
    changed_text = 0
    changed_answer = 0
    correct_to_wrong = 0
    wrong_to_correct = 0
    applied = sum(1 for row in rows if row.get("intervention_applied"))
    deltas = {f"act_delta_l2_layer_{layer:03d}": [] for layer in record_layers}
    for row in rows:
        base = baseline_by_id.get(row["id"], {})
        if row.get("generated_text") != base.get("generated_text"):
            changed_text += 1
        if has_answers and row.get("pred_answer") != base.get("pred_answer"):
            changed_answer += 1
        if has_accuracy and has_baseline_accuracy:
            if base.get("correct") and not row.get("correct"):
                correct_to_wrong += 1
            if not base.get("correct") and row.get("correct"):
                wrong_to_correct += 1
        for key in list(deltas):
            value = row.get(key)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                deltas[key].append(float(value))
    summary = {
        "n": n,
        "applied_n": applied,
        "changed_text_rate": changed_text / max(n, 1),
    }
    if has_answers:
        summary["changed_answer_rate"] = changed_answer / max(n, 1)
    if has_parseability:
        parseable = sum(1 for row in rows if row.get("parse_success"))
        summary.update(
            {
                "parseable_n": parseable,
                "parse_rate": parseable / max(n, 1),
                "unparseable_n": n - parseable,
            }
        )
    if has_accuracy and has_baseline_accuracy:
        correct = sum(1 for row in rows if row.get("correct"))
        baseline_correct = sum(1 for row in baseline_rows if row.get("correct"))
        accuracy = correct / max(n, 1)
        baseline_accuracy = baseline_correct / max(len(baseline_rows), 1)
        summary.update(
            {
                "correct_n": correct,
                "accuracy": accuracy,
                "baseline_correct_n": baseline_correct,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_delta_vs_baseline": accuracy - baseline_accuracy,
                "correct_to_wrong": correct_to_wrong,
                "wrong_to_correct": wrong_to_correct,
                "net_correct_change": wrong_to_correct - correct_to_wrong,
                "mcnemar_exact_p": mcnemar_exact_p(correct_to_wrong, wrong_to_correct),
                "accuracy_delta_ci95": bootstrap_accuracy_delta_ci(
                    rows, baseline_by_id, bootstrap_samples, seed
                ),
            }
        )
    for key, values in deltas.items():
        summary[f"mean_{key}"] = sum(values) / max(len(values), 1) if values else None
    return summary


def write_summary(
    path: str,
    all_rows: Dict[str, List[Dict]],
    record_layers: Sequence[int],
    bootstrap_samples: int,
    seed: int,
) -> Dict:
    baseline_rows = all_rows["baseline"]
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    summary = {
        "baseline": summarize_accuracy_rows(baseline_rows),
        "baseline_groups": {},
        "modes": {},
        "groups": {},
    }
    for group in ["low", "mid", "high"]:
        group_rows = [row for row in baseline_rows if row.get("meta_group") == group]
        if group_rows:
            summary["baseline_groups"][group] = summarize_accuracy_rows(group_rows)
    for mode, rows in all_rows.items():
        if mode == "baseline":
            continue
        summary["modes"][mode] = summarize_mode(rows, baseline_by_id, record_layers, bootstrap_samples, seed)
        for group in ["low", "mid", "high"]:
            group_rows = [row for row in rows if row.get("meta_group") == group]
            if group_rows:
                summary["groups"].setdefault(mode, {})[group] = summarize_mode(
                    group_rows,
                    baseline_by_id,
                    record_layers,
                    bootstrap_samples,
                    seed,
                )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True, help="Prepared jsonl used by extract_activations.py / train_decoupler.py.")
    parser.add_argument("--target-file", required=True, help="binary_analysis/intervention_targets.pt exported by train_decoupler.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modes", nargs="+", default=["zero_ablate", "mean_ablate", "active_clamp"], choices=["zero_ablate", "mean_ablate", "active_clamp", "inactive_clamp", "sign_flip", "scale", "noise", "sample_patch"])
    parser.add_argument("--eval-group", choices=["all", "low", "mid", "high"], default="all", help="Only evaluate samples in this meta group.")
    parser.add_argument("--target-group", choices=["all", "low", "mid", "high"], default="all", help="Only apply intervention to this group within evaluated samples.")
    parser.add_argument("--group-basis", choices=["true", "pred"], default="true", help="Use true or predicted selected-neuron score groups from intervention_targets.pt.")
    parser.add_argument("--patch-source-group", choices=["low", "mid", "high"], default="high", help="Donor group for sample_patch.")
    parser.add_argument("--activation-dir", default=None, help="Activation cache dir; required for sample_patch unless present in target metadata.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-style", choices=["data", "direct", "cot"], default="data", help="data keeps the jsonl prompt; cot/direct rebuild prompts from question for stronger GSM8K evaluation.")
    parser.add_argument("--use-chat-template", action="store_true", help="Wrap the prompt with tokenizer.apply_chat_template for instruct/chat models.")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt used only with --use-chat-template.")
    parser.add_argument("--chat-template-enable-thinking", choices=["auto", "true", "false"], default="auto", help="For Qwen3-style templates, optionally pass enable_thinking.")
    parser.set_defaults(answer_extraction="none")
    parser.add_argument("--answer-tolerance", type=float, default=0.0, help="Absolute numeric tolerance for answer matching.")
    parser.add_argument("--accuracy-bootstrap-samples", type=int, default=1000, help="Bootstrap samples for paired accuracy-delta CI; set 0 to disable.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--clamp-factor", type=float, default=1.2)
    parser.add_argument("--scale-factor", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--apply-to-generated", action="store_true", help="Also intervene on generated-token hidden states, not only the prompt prefill.")
    parser.add_argument("--record-layers", default="auto", help="Comma list, ranges, layer_i, next, auto, or none. Records selected-feature deltas vs baseline.")
    parser.add_argument("--save-recorded-activations", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto", choices=["auto", "balanced", "balanced_low_0", "sequential", "single", "none"], help="Transformers device_map. Use single for one-GPU 7B, balanced_low_0 with --max-memory for split 72B.")
    parser.add_argument("--max-memory", nargs="*", default=None, help="Optional per-GPU memory caps, e.g. --max-memory 0:36GiB 1:36GiB.")
    parser.add_argument("--attn-implementation", default=None, help="Optional transformers attention implementation, e.g. eager, sdpa, flash_attention_2.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    targets = load_intervention_targets(args.target_file)
    metadata = targets.get("metadata", {})
    layer_i = int(metadata.get("layer_i"))
    next_layer = int(metadata.get("next_layer", layer_i + 1))
    activation_dir = args.activation_dir or metadata.get("activation_dir")
    feature_indices = targets["selected_neurons"].long()
    if feature_indices.numel() == 0:
        raise ValueError("No selected_neurons found in intervention target file.")
    record_layers = parse_record_layers(args.record_layers, layer_i, next_layer)

    eval_rows = build_eval_rows(args.data, targets, args.group_basis, args.eval_group, args.max_samples)
    if not eval_rows:
        raise ValueError("No evaluation rows selected. Check --data, --eval-group, and intervention target ids.")

    group_tensor = targets["val_true_score_group"] if args.group_basis == "true" else targets["val_pred_score_group"]
    val_ids = list(targets.get("val_ids", []))
    donor_ids = [
        sample_id
        for sample_id, group_id in zip(val_ids, group_tensor.tolist())
        if group_name(int(group_id)) == args.patch_source_group
    ]
    feature_map = {}
    if "sample_patch" in args.modes:
        if not activation_dir:
            raise ValueError("--activation-dir is required for sample_patch when target metadata has no activation_dir.")
        feature_map = load_layer_feature_map(activation_dir, layer_i, donor_ids, feature_indices)
        donor_ids = [sample_id for sample_id in donor_ids if sample_id in feature_map]
        if not donor_ids:
            raise ValueError(f"No donor activations found for patch source group {args.patch_source_group}.")

    model, tokenizer = load_model_and_tokenizer(args, padding_side="left")
    layers = get_decoder_layers(model)
    if layer_i < 0 or layer_i >= len(layers):
        raise ValueError(f"layer_i={layer_i} is out of range for model with {len(layers)} layers.")
    bad_record_layers = [layer for layer in record_layers if layer < 0 or layer >= len(layers)]
    if bad_record_layers:
        raise ValueError(f"record layers out of range: {bad_record_layers}")

    selected_mean = targets.get("continuous_mean")
    selected_threshold = targets.get("binary_activation_threshold")
    mean_values = selected_mean[0, feature_indices].float() if isinstance(selected_mean, torch.Tensor) else None
    threshold_values = selected_threshold[0, feature_indices].float() if isinstance(selected_threshold, torch.Tensor) else None

    context = GenerationContext()
    recorder = ActivationRecorder(context, layers, record_layers, feature_indices) if record_layers else None
    all_generations: Dict[str, List[Dict]] = {}
    try:
        baseline_rows = generate_for_mode(model, tokenizer, eval_rows, context, "baseline", args, targets, [], {}, feature_indices.numel())
        all_generations["baseline"] = baseline_rows
        baseline_records = recorder.records.get("baseline", {}) if recorder else {}

        for mode in args.modes:
            intervention = HiddenFeatureIntervention(
                context,
                layers[layer_i],
                feature_indices,
                mode,
                mean_values,
                threshold_values,
                args.clamp_factor,
                args.scale_factor,
                args.noise_scale,
                args.apply_to_generated,
            )
            try:
                rows = generate_for_mode(model, tokenizer, eval_rows, context, mode, args, targets, donor_ids, feature_map, feature_indices.numel())
            finally:
                intervention.close()
            mode_records = recorder.records.get(mode, {}) if recorder else {}
            baseline_by_id = {row["id"]: row for row in baseline_rows}
            for row in rows:
                base = baseline_by_id.get(row["id"], {})
                row["baseline_pred_answer"] = base.get("pred_answer")
                row["baseline_correct"] = base.get("correct")
                row["changed_text"] = row.get("generated_text") != base.get("generated_text")
                row["changed_answer"] = row.get("pred_answer") != base.get("pred_answer")
                row.update(activation_delta_columns(row, baseline_records, mode_records, record_layers))
            all_generations[mode] = rows
    finally:
        if recorder is not None:
            recorder.close()

    write_jsonl(os.path.join(args.output_dir, "baseline_generations.jsonl"), all_generations["baseline"])
    flat_rows = []
    for mode, rows in all_generations.items():
        write_jsonl(os.path.join(args.output_dir, f"{mode}_generations.jsonl"), rows)
        flat_rows.extend(rows)
    write_csv(os.path.join(args.output_dir, "intervention_results.csv"), flat_rows)
    summary = write_summary(
        os.path.join(args.output_dir, "intervention_summary.json"),
        all_generations,
        record_layers,
        args.accuracy_bootstrap_samples,
        args.seed,
    )

    run_config = {
        "model_path": args.model_path,
        "data": args.data,
        "target_file": args.target_file,
        "output_dir": args.output_dir,
        "layer_i": layer_i,
        "next_layer": next_layer,
        "feature_count": int(feature_indices.numel()),
        "selected_neurons": feature_indices.tolist(),
        "record_layers": record_layers,
        "modes": args.modes,
        "eval_group": args.eval_group,
        "target_group": args.target_group,
        "group_basis": args.group_basis,
        "max_samples": args.max_samples,
        "prompt_style": args.prompt_style,
        "use_chat_template": args.use_chat_template,
        "answer_extraction": args.answer_extraction,
        "answer_tolerance": args.answer_tolerance,
        "accuracy_bootstrap_samples": args.accuracy_bootstrap_samples,
        "summary": summary,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    if args.save_recorded_activations and recorder is not None:
        torch.save(recorder.records, os.path.join(args.output_dir, "recorded_selected_activations.pt"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(
        "intervene_activations.py is an internal generation library. "
        "Use scripts/run_experiment.sh."
    )
