#!/usr/bin/env python3
"""Fail fast when target and external-semantic activation caches are misaligned."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def digest(ids) -> str:
    value = hashlib.sha256()
    for sample_id in ids:
        value.update(str(sample_id).encode("utf-8")); value.update(b"\n")
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--semantic-dir", required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument("--semantic-layer", type=int, required=True)
    args = parser.parse_args()
    target = Path(args.target_dir)
    semantic = Path(args.semantic_dir)
    target_obj = load(target / f"layer_{args.target_layer:03d}.pt")
    semantic_obj = load(semantic / f"layer_{args.semantic_layer:03d}.pt")
    target_ids = [str(value) for value in target_obj["ids"]]
    semantic_ids = [str(value) for value in semantic_obj["ids"]]
    if target_ids != semantic_ids:
        target_set, semantic_set = set(target_ids), set(semantic_ids)
        raise ValueError(
            "Activation id/order mismatch: "
            f"target={len(target_ids)} semantic={len(semantic_ids)} "
            f"target_only={len(target_set - semantic_set)} "
            f"semantic_only={len(semantic_set - target_set)}"
        )
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Aligned cache contains duplicate record ids.")
    target_manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    semantic_manifest = json.loads((semantic / "manifest.json").read_text(encoding="utf-8"))
    if target_manifest.get("record_type") != "generated_token_states":
        raise ValueError("Target cache is not a generated-step cache.")
    if not target_manifest.get(
        "external_semantic_data_uses_raw_prompt_plus_assistant_prefix"
    ):
        raise ValueError(
            "Target cache predates the assistant-prefix semantic-reference "
            "contract; regenerate it before training."
        )
    if not target_manifest.get("expanded_data_has_exact_prompt_token_ids"):
        raise ValueError(
            "Target cache lacks exact downstream prompt token ids; regenerate it."
        )
    if semantic_manifest.get("pooling") != "last_token":
        raise ValueError("External semantic cache must use last-token pooling.")
    if not semantic_manifest.get("use_chat_template"):
        raise ValueError(
            "External semantic cache must use the reference model's chat template."
        )
    semantic_layers = [int(value) for value in semantic_manifest.get("layers", [])]
    if args.semantic_layer not in semantic_layers:
        raise ValueError(
            f"Semantic layer {args.semantic_layer} absent from cache {semantic_layers}."
        )
    if int(semantic_manifest.get("num_examples", -1)) != len(target_ids):
        raise ValueError("Semantic manifest row count disagrees with aligned tensors.")
    if int(
        semantic_manifest.get("semantic_reference_assistant_prefix_rows", -1)
    ) != len(target_ids):
        raise ValueError(
            "External semantic cache was not built from assistant-prefix rows; "
            "regenerate it with the current extraction contract."
        )
    report = {
        "aligned": True,
        "rows": len(target_ids),
        "id_sha256": digest(target_ids),
        "target_layer": args.target_layer,
        "semantic_layer": args.semantic_layer,
        "target_model": target_manifest.get("model_path"),
        "semantic_model": semantic_manifest.get("model_path"),
        "record_steps": target_manifest.get("record_steps"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
