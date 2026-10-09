#!/usr/bin/env python3
"""Build one aligned semantic-reference cache from several frozen LLMs.

Each reference is aligned by sample id, standardized using only the upstream
training split, compressed independently with a train-only PCA basis, and then
concatenated.  The output follows the regular activation-cache contract, so the
existing decoupler can consume it without changing the main training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from train_decoupler import load_layer


def deterministic_split(
    ids: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_by_base_id: bool = True,
    base_id_step_pattern: str = r"::step\d+$",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Mirror the grouped split contract without importing the training stack."""
    if not split_by_base_id:
        raise ValueError("The semantic ensemble requires base-prompt grouped splitting.")
    groups: Dict[str, List[int]] = {}
    for row_index, sample_id in enumerate(ids):
        base_id = re.sub(base_id_step_pattern, "", str(sample_id))
        groups.setdefault(base_id, []).append(row_index)
    group_ids = list(groups)
    if len(group_ids) < 10:
        raise ValueError("At least 10 base prompt groups are required.")
    random.Random(seed).shuffle(group_ids)
    test_n = max(1, int(len(group_ids) * test_ratio)) if test_ratio > 0 else 0
    val_n = max(1, int(len(group_ids) * val_ratio))
    if test_n + val_n >= len(group_ids):
        raise ValueError("Validation/test ratios leave no training groups.")
    test_groups = set(group_ids[:test_n])
    val_groups = set(group_ids[test_n : test_n + val_n])
    split_rows: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
    for base_id, row_indices in groups.items():
        key = "test" if base_id in test_groups else "validation" if base_id in val_groups else "train"
        split_rows[key].extend(row_indices)
    train = torch.tensor(sorted(split_rows["train"]), dtype=torch.long)
    val = torch.tensor(sorted(split_rows["validation"]), dtype=torch.long)
    test = torch.tensor(sorted(split_rows["test"]), dtype=torch.long)
    summary = {
        "mode": "group_by_base_id",
        "seed": seed,
        "base_id_step_pattern": base_id_step_pattern,
        "total_rows": len(ids),
        "total_groups": len(group_ids),
        "train_groups": len(group_ids) - len(test_groups) - len(val_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_rows": int(train.numel()),
        "val_rows": int(val.numel()),
        "test_rows": int(test.numel()),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    return train, val, test, summary


def parse_source(value: str) -> Tuple[Path, int, str]:
    parts = value.rsplit(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--source must have the form ACTIVATION_DIR:LAYER:LABEL"
        )
    directory, layer, label = parts
    return Path(directory), int(layer), label


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--target-layer", required=True, type=int)
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-layer", type=int, default=0)
    parser.add_argument("--components-per-source", type=int, default=512)
    parser.add_argument("--pca-fit-max-rows", type=int, default=8192)
    parser.add_argument("--pca-oversample", type=int, default=16)
    parser.add_argument("--pca-niter", type=int, default=4)
    parser.add_argument("--projection-batch-size", type=int, default=4096)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-id-step-pattern", default=r"::step\d+$")
    return parser.parse_args()


def align(ids: Sequence[str], features: torch.Tensor, target_ids: Sequence[str]) -> torch.Tensor:
    index: Dict[str, int] = {}
    for row, sample_id in enumerate(ids):
        key = str(sample_id)
        if key in index:
            raise ValueError(f"Duplicate semantic-reference id: {key}")
        index[key] = row
    missing = [str(sample_id) for sample_id in target_ids if str(sample_id) not in index]
    if missing:
        raise ValueError(
            f"Semantic reference misses {len(missing)} target ids; examples={missing[:5]}"
        )
    order = torch.tensor([index[str(sample_id)] for sample_id in target_ids])
    return features.index_select(0, order).float().contiguous()


def train_rows_subset(train_idx: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    if maximum <= 0 or train_idx.numel() <= maximum:
        return train_idx
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(train_idx.numel(), generator=generator)[:maximum]
    return train_idx.index_select(0, chosen)


def project_reference(
    values: torch.Tensor,
    train_idx: torch.Tensor,
    components: int,
    fit_max_rows: int,
    oversample: int,
    niter: int,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    train = values.index_select(0, train_idx)
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    fit_idx = train_rows_subset(train_idx, fit_max_rows, seed)
    fit = (values.index_select(0, fit_idx) - mean) / std
    output_dim = min(components, fit.size(0) - 1, fit.size(1))
    if output_dim < 1:
        raise ValueError("Not enough rows or dimensions for semantic PCA.")
    q = min(fit.size(0), fit.size(1), output_dim + max(0, oversample))
    torch.manual_seed(seed)
    _, singular, basis = torch.pca_lowrank(fit, q=q, center=False, niter=niter)
    basis = basis[:, :output_dim].contiguous()

    chunks: List[torch.Tensor] = []
    for start in range(0, values.size(0), batch_size):
        batch = values[start : start + batch_size]
        chunks.append((((batch - mean) / std) @ basis).cpu())
    projected = torch.cat(chunks, dim=0).contiguous()
    projected_train = projected.index_select(0, train_idx)
    projected_mean = projected_train.mean(dim=0, keepdim=True)
    projected_std = projected_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    projected = ((projected - projected_mean) / projected_std).contiguous()
    explained_proxy = singular[:output_dim].square()
    explained_ratio = float(
        explained_proxy.sum().item() / max(singular.square().sum().item(), 1e-12)
    )
    return projected, {
        "raw_dim": int(values.size(1)),
        "projected_dim": int(projected.size(1)),
        "fit_rows": int(fit.size(0)),
        "pca_q": int(q),
        "captured_variance_within_computed_subspace": explained_ratio,
        "standardization": "train-only before and after PCA",
    }


def main() -> None:
    args = arguments()
    target_path = Path(args.target_dir) / f"layer_{args.target_layer:03d}.pt"
    target_ids, _ = load_layer(str(target_path))
    train_idx, val_idx, test_idx, split = deterministic_split(
        target_ids,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        split_by_base_id=True,
        base_id_step_pattern=args.base_id_step_pattern,
    )

    projections: List[torch.Tensor] = []
    source_manifest: List[Dict[str, object]] = []
    for source_index, (directory, layer, label) in enumerate(args.source):
        path = directory / f"layer_{layer:03d}.pt"
        ids, raw = load_layer(str(path))
        values = align(ids, raw, target_ids)
        projected, diagnostics = project_reference(
            values,
            train_idx,
            args.components_per_source,
            args.pca_fit_max_rows,
            args.pca_oversample,
            args.pca_niter,
            args.projection_batch_size,
            args.seed + 1009 * source_index,
        )
        projections.append(projected)
        source_manifest.append(
            {
                "label": label,
                "activation_dir": str(directory),
                "layer": layer,
                **diagnostics,
            }
        )
        print(
            f"[semantic-ensemble] {label}: rows={projected.size(0)} "
            f"raw={values.size(1)} projected={projected.size(1)}",
            flush=True,
        )

    ensemble = torch.cat(projections, dim=1).to(torch.float16).contiguous()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"layer": args.output_layer, "ids": list(target_ids), "features": ensemble},
        output / f"layer_{args.output_layer:03d}.pt",
    )
    digest = hashlib.sha256()
    for sample_id in target_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
    manifest = {
        "schema_version": 1,
        "model_path": "semantic_ensemble",
        "record_type": "aligned_multi_reference_semantic_features",
        "pooling": "last_token",
        "prompt_style": "data",
        "use_chat_template": True,
        "semantic_reference_assistant_prefix_rows": len(target_ids),
        "layers": [args.output_layer],
        "num_examples": len(target_ids),
        "id_sha256": digest.hexdigest(),
        "output_dim": int(ensemble.size(1)),
        "sources": source_manifest,
        "split": split,
        "train_rows": int(train_idx.numel()),
        "validation_rows": int(val_idx.numel()),
        "test_rows": int(test_idx.numel()),
        "fit_contract": "all transforms fitted on upstream train ids only",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output_dir": str(output), "rows": len(target_ids), "dim": ensemble.size(1)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
