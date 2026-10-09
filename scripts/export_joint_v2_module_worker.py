#!/usr/bin/env python
"""Refine and audit one frozen-module shard on a shared GPU.

The parent process writes one mmap-compatible CPU cache after joint direction
discovery. Workers read that immutable cache, write only their own module
directories, and return rows for the parent's global ranking step.
"""

from __future__ import annotations

import argparse
import os

import torch

import _bootstrap  # noqa: F401  # Direct-script compatibility.
import train_decoupler_joint_v2 as joint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--worker-id", required=True, type=int)
    parser.add_argument("--module-indices", nargs="+", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    payload = torch.load(
        cli.cache,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    args = argparse.Namespace(**payload["args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Parallel module-refiner workers require a visible CUDA GPU.")
    seed = int(args.seed) + 100_000 + cli.worker_id
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(
        f"[module-worker] id={cli.worker_id} device={device} "
        f"modules={cli.module_indices} seed={seed}",
        flush=True,
    )
    result = joint.export_modules(
        payload["ids"],
        payload["raw_i"],
        payload["candidate_values"],
        payload["candidate_neurons"],
        payload["threshold"],
        payload["target_mean"],
        payload["target_std"],
        payload["train_idx"],
        payload["val_idx"],
        payload["test_idx"],
        payload["prev_features"],
        payload["encoded"],
        payload["semantic_prediction"],
        payload["directions"],
        args,
        device,
        payload["config"],
        module_indices=cli.module_indices,
        finalize=False,
        write_shared_artifacts=False,
    )
    os.makedirs(os.path.dirname(cli.result) or ".", exist_ok=True)
    torch.save(result, cli.result)
    print(
        f"[module-worker] complete id={cli.worker_id} "
        f"modules={len(result['module_rows'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
