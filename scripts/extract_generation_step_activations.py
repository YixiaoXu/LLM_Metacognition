#!/usr/bin/env python3
"""Cache hidden states at selected autoregressive generation steps.

The expanded JSONL is the alignment contract for an external semantic model:
it contains the target model's exact rendered prompt and generated prefix, so
the reference model encodes the same text rather than generating its own path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_utils import (
    DecoderLayerCapture,
    build_generation_prompt,
    generation_eos_token_id,
    last_token_logits_kwargs,
    load_model_and_tokenizer,
    model_input_device,
    num_hidden_layers_from_config,
    parse_dtype,
)


class PromptDataset(Dataset):
    def __init__(self, path: str, max_samples: int, strategy: str, seed: int):
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if max_samples > 0 and len(rows) > max_samples:
            if strategy == "random":
                indices = sorted(random.Random(seed).sample(range(len(rows)), max_samples))
                rows = [rows[index] for index in indices]
            else:
                rows = rows[:max_samples]
        ids = [str(row.get("id", "")) for row in rows]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Input rows must have unique nonempty ids.")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        return self.rows[index]


def collate(rows: List[Dict]) -> Dict:
    return {"ids": [str(row["id"]) for row in rows], "rows": rows}


def parse_layers(values: Sequence[str], count: int) -> List[int]:
    layers = set()
    for value in values:
        if value.lower() == "all":
            layers.update(range(count))
        elif "-" in value:
            start, stop = map(int, value.split("-", 1))
            layers.update(range(start, stop + 1))
        else:
            layers.add(int(value))
    invalid = [layer for layer in layers if not 0 <= layer < count]
    if invalid:
        raise ValueError(f"Layers outside [0, {count - 1}]: {invalid}")
    return sorted(layers)


def parse_steps(value: str, maximum: int) -> List[int]:
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not steps or any(step < 0 or step >= maximum for step in steps):
        raise ValueError(f"--record-steps must be inside [0, {maximum - 1}].")
    return steps


def last_nonpad(mask: torch.Tensor) -> torch.Tensor:
    position = torch.arange(mask.size(1), device=mask.device).view(1, -1)
    return position.masked_fill(~mask.bool(), 0).max(dim=1).values.long()


def choose(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probability = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
    return torch.multinomial(probability, 1).squeeze(1)


def append_hidden(
    buffers: Dict[int, List[torch.Tensor]],
    layers: Sequence[int],
    states: Sequence[torch.Tensor] | Mapping[int, torch.Tensor],
    rows: torch.Tensor,
    positions: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    for layer in layers:
        hidden = states[layer] if isinstance(states, Mapping) else states[layer + 1]
        pooled = hidden[rows, 0] if hidden.size(1) == 1 else hidden[rows, positions[rows]]
        buffers[layer].append(pooled.detach().to(device="cpu", dtype=dtype))


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def id_digest(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def flush(
    buffers: Dict[int, List[torch.Tensor]], ids: List[str], directory: Path, shard: int
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for layer, chunks in buffers.items():
        torch.save(
            {"layer": layer, "ids": list(ids), "features": torch.cat(chunks, 0).contiguous()},
            directory / f"layer_{layer:03d}_shard_{shard:05d}.pt",
        )
        chunks.clear()


def merge(
    layers: Sequence[int], ids: List[str], shard_dir: Path, output: Path, shards: int
) -> None:
    for layer in tqdm(layers, desc="Merge generation-step shards"):
        paths = [shard_dir / f"layer_{layer:03d}_shard_{index:05d}.pt" for index in range(shards)]
        objects = [torch.load(path, map_location="cpu") for path in paths]
        shard_ids = [sample_id for obj in objects for sample_id in obj["ids"]]
        if shard_ids != ids:
            raise RuntimeError(f"Shard id order mismatch for layer {layer}.")
        features = torch.cat([obj["features"] for obj in objects], 0).contiguous()
        torch.save({"layer": layer, "ids": ids, "features": features}, output / f"layer_{layer:03d}.pt")
        for path in paths:
            path.unlink()
    shard_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--layers", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-generation-steps", type=int, default=21)
    parser.add_argument("--record-steps", default="0,4,8,12,16,20")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", choices=["auto", "single", "none"], default="single")
    parser.add_argument("--prompt-style", choices=["data", "direct", "cot"], default="data")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--chat-template-enable-thinking", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Capture only requested decoder layers and request last-token logits when supported.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.shard_size <= 0 or args.max_generation_steps <= 0:
        raise ValueError("Batch, shard and generation-step counts must be positive.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args, padding_side="left")
    device = model_input_device(model)
    layer_count = num_hidden_layers_from_config(model.config)
    layers = parse_layers(args.layers, layer_count)
    if args.memory_efficient and layers[-1] == layer_count - 1:
        raise ValueError(
            "--memory-efficient does not support the final decoder layer because "
            "Transformers records its post-normalization state."
        )
    record_steps = parse_steps(args.record_steps, args.max_generation_steps)
    record_set = set(record_steps)
    save_dtype = parse_dtype(args.dtype)
    dataset = PromptDataset(args.data, args.max_samples, args.sample_strategy, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    eos_ids = generation_eos_token_id(model, tokenizer)
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else ([] if eos_ids is None else [eos_ids]))

    buffers = {layer: [] for layer in layers}
    all_ids: List[str] = []
    pending_ids: List[str] = []
    metadata: List[Dict] = []
    expanded: List[Dict] = []
    semantic_inputs: List[Dict] = []
    shard_dir = output / "_activation_shards"
    shard_index = 0
    logit_kwargs = last_token_logits_kwargs(model) if args.memory_efficient else {}
    print(
        f"Memory-efficient extraction={args.memory_efficient}; "
        f"last-token logits optimization={bool(logit_kwargs)}"
    )

    def maybe_flush(force: bool = False) -> None:
        nonlocal shard_index
        if pending_ids and (force or len(pending_ids) >= args.shard_size):
            flush(buffers, pending_ids, shard_dir, shard_index)
            pending_ids.clear()
            shard_index += 1

    with torch.inference_mode():
        progress = tqdm(loader, desc="Extract generated-step activations")
        for batch in progress:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            rendered = [build_generation_prompt(row, tokenizer, args) for row in batch["rows"]]
            encoded = tokenizer(
                rendered, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention = encoded["attention_mask"].to(device)
            size = input_ids.size(0)
            base_token_ids = [
                input_ids[index][attention[index].bool()].detach().cpu().tolist()
                for index in range(size)
            ]
            row_index = torch.arange(size, device=device)
            output_state = model(
                input_ids=input_ids, attention_mask=attention,
                output_hidden_states=False, use_cache=True, **logit_kwargs,
            )
            logits = (
                output_state.logits[:, -1]
                if output_state.logits.size(1) == 1
                else output_state.logits[row_index, last_nonpad(attention)]
            )
            next_tokens = choose(logits, args.temperature)
            past = output_state.past_key_values
            active = torch.ones(size, dtype=torch.bool, device=device)
            prefixes: List[List[int]] = [[] for _ in range(size)]
            del output_state

            capture_context = (
                DecoderLayerCapture(model, layers)
                if args.memory_efficient
                else nullcontext(None)
            )
            with capture_context as capture:
                for step in range(args.max_generation_steps):
                    current = next_tokens.view(size, 1)
                    attention = torch.cat(
                        [attention, torch.ones(size, 1, dtype=attention.dtype, device=device)], dim=1
                    )
                    if capture is not None:
                        capture.clear()
                    output_state = model(
                        input_ids=current, attention_mask=attention, past_key_values=past,
                        output_hidden_states=not args.memory_efficient, use_cache=True,
                        **logit_kwargs,
                    )
                    past = output_state.past_key_values
                    ended = torch.zeros_like(active)
                    for token_id in eos_ids:
                        ended |= current[:, 0] == int(token_id)
                    active_rows = torch.nonzero(active & ~ended, as_tuple=False).flatten()
                    for index in active_rows.tolist():
                        prefixes[index].append(int(current[index, 0]))
                    if step in record_set and active_rows.numel():
                        states = capture.outputs if capture is not None else output_state.hidden_states
                        append_hidden(
                            buffers, layers, states, active_rows,
                            torch.zeros(size, dtype=torch.long, device=device), save_dtype,
                        )
                        for index in active_rows.tolist():
                            record_id = f"{batch['ids'][index]}::step{step:03d}"
                            prefix = tokenizer.decode(prefixes[index], skip_special_tokens=False)
                            all_ids.append(record_id)
                            pending_ids.append(record_id)
                            metadata.append({
                                "record_id": record_id,
                                "sample_id": batch["ids"][index],
                                "step": step,
                                "generated_token_id": int(current[index, 0]),
                                "generated_prefix": prefix,
                            })
                            row = dict(batch["rows"][index])
                            row.update({
                                "id": record_id,
                                "source_id": batch["ids"][index],
                                "generation_step": step,
                                "generated_prefix": prefix,
                                "prompt": rendered[index] + prefix,
                                "prompt_token_ids": (
                                    base_token_ids[index] + list(prefixes[index])
                                ),
                                "prompt_is_fully_rendered": True,
                            })
                            expanded.append(row)
                            semantic_row = dict(batch["rows"][index])
                            original_prompt = str(
                                batch["rows"][index].get("prompt")
                                or batch["rows"][index].get("question")
                                or ""
                            ).strip()
                            semantic_row.update({
                                "id": record_id,
                                "source_id": batch["ids"][index],
                                "generation_step": step,
                                "generated_prefix": prefix,
                                "prompt": original_prompt,
                                "assistant_prefix": prefix,
                                "semantic_reference_input": True,
                            })
                            semantic_inputs.append(semantic_row)
                        maybe_flush()
                    next_tokens = choose(output_state.logits[:, -1], args.temperature)
                    active &= ~ended
                    del output_state
                    if not bool(active.any()):
                        break
            if device.type == "cuda":
                progress.set_postfix(
                    seq=int(input_ids.shape[1]),
                    alloc_gib=f"{torch.cuda.memory_allocated(device) / 2**30:.1f}",
                    reserved_gib=f"{torch.cuda.memory_reserved(device) / 2**30:.1f}",
                    peak_gib=f"{torch.cuda.max_memory_allocated(device) / 2**30:.1f}",
                )
            del past, input_ids, attention, logits, next_tokens, active, prefixes

    maybe_flush(force=True)
    if not all_ids:
        raise RuntimeError(
            "No generated-step states were recorded. Check EOS behavior and "
            "--record-steps/--max-generation-steps."
        )
    merge(layers, all_ids, shard_dir, output, shard_index)
    write_jsonl(output / "generation_step_metadata.jsonl", metadata)
    write_jsonl(output / "expanded_generation_data.jsonl", expanded)
    write_jsonl(output / "external_semantic_data.jsonl", semantic_inputs)
    manifest = {
        "model_path": args.model_path,
        "data": args.data,
        "layers": layers,
        "num_model_layers": layer_count,
        "num_prompt_examples": len(dataset),
        "num_examples": len(all_ids),
        "record_type": "generated_token_states",
        "record_steps": record_steps,
        "record_step_counts": {
            str(step): sum(int(row["step"]) == step for row in metadata)
            for step in record_steps
        },
        "id_format": "<sample_id>::stepNNN",
        "id_sha256": id_digest(all_ids),
        "expanded_data": "expanded_generation_data.jsonl",
        "expanded_data_is_fully_rendered": True,
        "expanded_data_has_exact_prompt_token_ids": True,
        "external_semantic_data": "external_semantic_data.jsonl",
        "external_semantic_data_uses_raw_prompt_plus_assistant_prefix": True,
        "external_semantic_chat_contract": (
            "reference tokenizer renders the raw user prompt and then appends "
            "the target model's decoded assistant prefix"
        ),
        "prompt_style": args.prompt_style,
        "use_chat_template": bool(args.use_chat_template),
        "chat_template_enable_thinking": args.chat_template_enable_thinking,
        "temperature": args.temperature,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "memory_efficient": bool(args.memory_efficient),
        "last_token_logits_optimization": bool(logit_kwargs),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "records": len(all_ids), "id_sha256": manifest["id_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
