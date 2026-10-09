#!/usr/bin/env python
import argparse
import json
import os
from contextlib import nullcontext
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_utils import (
    DecoderLayerCapture,
    build_generation_prompt,
    last_token_logits_kwargs,
    load_model_and_tokenizer,
    model_input_device,
    num_hidden_layers_from_config,
    parse_dtype,
)


class PromptDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def collate_prompts(batch: List[Dict]) -> Dict:
    return {
        "ids": [x["id"] for x in batch],
        "rows": batch,
    }


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    attention_mask = attention_mask.to(hidden.device)
    if mode == "last_token":
        lengths = attention_mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    raise ValueError(f"Unknown pooling mode: {mode}")


def parse_layer_specs(specs: Sequence[str], num_layers: int) -> List[int]:
    layers = set()
    for spec in specs:
        spec = spec.strip().lower()
        if spec == "all":
            layers.update(range(num_layers))
            continue
        if "-" in spec:
            start_text, end_text = spec.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range {spec}: end is smaller than start.")
            layers.update(range(start, end + 1))
            continue
        layers.add(int(spec))

    bad_layers = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if bad_layers:
        raise ValueError(f"Layer ids out of range 0-{num_layers - 1}: {bad_layers}")
    return sorted(layers)


def flush_shard(
    layer_buffers: Dict[int, List[torch.Tensor]],
    shard_ids: List[str],
    shard_dir: str,
    shard_index: int,
) -> None:
    os.makedirs(shard_dir, exist_ok=True)
    for layer, chunks in layer_buffers.items():
        if not chunks:
            continue
        tensor = torch.cat(chunks, dim=0).contiguous()
        torch.save(
            {"layer": layer, "ids": list(shard_ids), "features": tensor},
            os.path.join(shard_dir, f"layer_{layer:03d}_shard_{shard_index:05d}.pt"),
        )
        chunks.clear()


def merge_shards(
    layers: Sequence[int],
    all_ids: List[str],
    shard_dir: str,
    output_dir: str,
    num_shards: int,
    keep_shards: bool,
) -> None:
    for layer in tqdm(layers, desc="Merging layer shards"):
        chunks = []
        layer_shard_paths = []
        for shard_index in range(num_shards):
            path = os.path.join(shard_dir, f"layer_{layer:03d}_shard_{shard_index:05d}.pt")
            obj = torch.load(path, map_location="cpu")
            chunks.append(obj["features"])
            layer_shard_paths.append(path)
        tensor = torch.cat(chunks, dim=0).contiguous()
        torch.save({"layer": layer, "ids": all_ids, "features": tensor}, os.path.join(output_dir, f"layer_{layer:03d}.pt"))
        del tensor, chunks
        if not keep_shards:
            for path in layer_shard_paths:
                os.remove(path)

    if not keep_shards:
        for filename in os.listdir(shard_dir):
            os.remove(os.path.join(shard_dir, filename))
        os.rmdir(shard_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        help="Decoder block layers to cache. Supports examples like: all, 0 1 2, 0-10, or 0 10 20-30.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["last_token", "mean"], default="last_token")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto", choices=["auto", "balanced", "balanced_low_0", "sequential", "single", "none"])
    parser.add_argument("--max-memory", nargs="*", default=None, help="Optional per-GPU memory caps, e.g. --max-memory 0:36GiB 1:36GiB.")
    parser.add_argument("--attn-implementation", default=None, help="Optional transformers attention implementation, e.g. eager, sdpa, flash_attention_2.")
    parser.add_argument("--gptq-backend", default=None, help="Override GPTQ backend, e.g. gptq_triton or gptq_torch, to avoid incompatible auto-selected kernels.")
    parser.add_argument("--prompt-style", choices=["data", "direct", "cot"], default="data", help="data keeps the jsonl prompt; cot/direct rebuild prompts from question.")
    parser.add_argument("--use-chat-template", action="store_true", help="Cache activations under the chat/instruct prompt format used for generation.")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--chat-template-enable-thinking", choices=["auto", "true", "false"], default="auto", help="For Qwen3-style templates, optionally pass enable_thinking.")
    parser.add_argument("--shard-size", type=int, default=1024, help="Flush activation shards every N examples before final merge. Use 0 to keep everything in RAM.")
    parser.add_argument("--keep-shards", action="store_true", help="Keep intermediate shard files after final layer files are written.")
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Capture only requested decoder layers and request last-token logits when supported.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_dtype = parse_dtype(args.dtype)

    model, tokenizer = load_model_and_tokenizer(args, padding_side="right")
    input_device = model_input_device(model)
    num_layers = num_hidden_layers_from_config(model.config)
    layers = parse_layer_specs(args.layers, num_layers)
    if args.memory_efficient and layers[-1] == num_layers - 1:
        raise ValueError(
            "--memory-efficient does not support the final decoder layer because "
            "Transformers records its post-normalization state."
        )
    print(f"Caching {len(layers)} decoder block layer(s): {layers[0]}..{layers[-1]}" if len(layers) > 1 else f"Caching layer {layers[0]}")

    dataset = PromptDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_prompts)

    layer_buffers = {layer: [] for layer in layers}
    all_ids: List[str] = []
    shard_ids: List[str] = []
    shard_dir = os.path.join(args.output_dir, "_activation_shards")
    shard_index = 0

    capture_context = (
        DecoderLayerCapture(model, layers) if args.memory_efficient else nullcontext(None)
    )
    logit_kwargs = last_token_logits_kwargs(model) if args.memory_efficient else {}
    print(
        f"Memory-efficient extraction={args.memory_efficient}; "
        f"last-token logits optimization={bool(logit_kwargs)}"
    )
    with capture_context as capture, torch.inference_mode():
        progress = tqdm(loader, desc="Extracting activations")
        for batch in progress:
            if input_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(input_device)
            prompts = [build_generation_prompt(row, tokenizer, args) for row in batch["rows"]]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {k: v.to(input_device) for k, v in encoded.items()}
            if capture is not None:
                capture.clear()
            outputs = model(
                **encoded,
                output_hidden_states=not args.memory_efficient,
                use_cache=False,
                **logit_kwargs,
            )
            hidden_states = capture.outputs if capture is not None else outputs.hidden_states

            for layer in layers:
                if capture is not None:
                    if layer not in hidden_states:
                        raise RuntimeError(f"Decoder hook did not capture layer {layer}.")
                    hidden = hidden_states[layer]
                else:
                    hf_hidden_index = layer + 1
                    if hf_hidden_index >= len(hidden_states):
                        raise ValueError(
                            f"Layer {layer} is out of range. Model returned {len(hidden_states) - 1} decoder layers."
                        )
                    hidden = hidden_states[hf_hidden_index]
                pooled = pool_hidden(hidden, encoded["attention_mask"], args.pooling)
                layer_buffers[layer].append(pooled.detach().to("cpu", dtype=save_dtype))
            all_ids.extend(batch["ids"])
            shard_ids.extend(batch["ids"])
            if input_device.type == "cuda":
                progress.set_postfix(
                    seq=int(encoded["input_ids"].shape[1]),
                    alloc_gib=f"{torch.cuda.memory_allocated(input_device) / 2**30:.1f}",
                    reserved_gib=f"{torch.cuda.memory_reserved(input_device) / 2**30:.1f}",
                    peak_gib=f"{torch.cuda.max_memory_allocated(input_device) / 2**30:.1f}",
                )
            del outputs, hidden_states, encoded

            if args.shard_size > 0 and len(shard_ids) >= args.shard_size:
                flush_shard(layer_buffers, shard_ids, shard_dir, shard_index)
                shard_ids.clear()
                shard_index += 1

    if args.shard_size > 0:
        if shard_ids:
            flush_shard(layer_buffers, shard_ids, shard_dir, shard_index)
            shard_index += 1
        merge_shards(layers, all_ids, shard_dir, args.output_dir, shard_index, args.keep_shards)
    else:
        for layer, chunks in layer_buffers.items():
            tensor = torch.cat(chunks, dim=0).contiguous()
            torch.save({"layer": layer, "ids": all_ids, "features": tensor}, os.path.join(args.output_dir, f"layer_{layer:03d}.pt"))

    manifest = {
        "model_path": args.model_path,
        "data": args.data,
        "requested_layers": args.layers,
        "layers": layers,
        "num_model_layers": num_layers,
        "num_examples": len(all_ids),
        "semantic_reference_assistant_prefix_rows": sum(
            bool(row.get("semantic_reference_input"))
            and "assistant_prefix" in row
            for row in dataset.rows
        ),
        "pooling": args.pooling,
        "prompt_style": args.prompt_style,
        "use_chat_template": args.use_chat_template,
        "system_prompt": args.system_prompt,
        "chat_template_enable_thinking": args.chat_template_enable_thinking,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "attn_implementation": args.attn_implementation,
        "model_type": getattr(model.config, "model_type", None),
        "max_length": args.max_length,
        "shard_size": args.shard_size,
        "keep_shards": args.keep_shards,
        "memory_efficient": bool(args.memory_efficient),
        "last_token_logits_optimization": bool(logit_kwargs),
        "note": "Layer ids are decoder block ids; saved vectors are hidden_states[layer_id + 1].",
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved activations to {args.output_dir}")


if __name__ == "__main__":
    main()
