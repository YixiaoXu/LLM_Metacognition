"""Fresh, resumable activation measurements for the direct-report experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def measure_report_activations(output: Path, rows: list[dict], width: int,
                               execution: dict, measure, checkpoint_rows: int = 32) -> np.ndarray:
    if not rows or width < 1 or checkpoint_rows < 1:
        raise ValueError("Activation measurement requires rows, hidden width and checkpoint size")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Activation measurement contains duplicate prompt IDs")
    prefix_hash = hashlib.sha256()
    for row in rows:
        prefix = list(map(int, row["prompt_token_ids"]))
        if len(prefix) < 2:
            raise ValueError(f"Missing generated-token prefix: {row['id']}")
        prefix_hash.update(json.dumps([row["id"], prefix], separators=(",", ":")).encode())
    contract = {"schema_version": 1, "ids": ids, "width": width,
                "prefix_sha256": prefix_hash.hexdigest(), "execution": execution}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    file = output / "runtime_activation_measurements.npz"
    values = np.zeros((len(rows), width), dtype=np.float32)
    completed = 0
    if file.exists():
        with np.load(file, allow_pickle=False) as saved:
            if (str(saved["contract_sha256"].item()) != fingerprint or
                    saved["ids"].tolist() != ids):
                raise ValueError("Runtime measurement contract changed; use a new output directory")
            values = saved["activations"].copy()
            completed = int(saved["n_complete"].item())
        if (values.shape != (len(rows), width) or values.dtype != np.float32 or
                not 0 <= completed <= len(rows) or not np.isfinite(values[:completed]).all()):
            raise ValueError("Invalid runtime activation checkpoint")
    manifest = output / "runtime_activation_contract.json"
    manifest.write_text(json.dumps({**contract, "contract_sha256": fingerprint}, indent=2) + "\n")

    def save() -> None:
        temporary = file.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, activations=values, ids=np.asarray(ids),
                     n_complete=np.asarray(completed), contract_sha256=np.asarray(fingerprint))
        temporary.replace(file)

    print(f"[report-measurement] source=runtime_remeasured rows={len(rows)} "
          f"completed={completed} batch_size=1 replay=generation_step_kv", flush=True)
    for index in tqdm(range(completed, len(rows)), total=len(rows), initial=completed,
                      desc="Measure report activations"):
        hidden = np.asarray(measure(list(map(int, rows[index]["prompt_token_ids"]))), dtype=np.float32)
        if hidden.shape != (width,) or not np.isfinite(hidden).all():
            raise ValueError(f"Invalid fresh activation for {ids[index]}: shape={hidden.shape}")
        values[index] = hidden
        completed = index + 1
        if completed % checkpoint_rows == 0 or completed == len(rows):
            save()
    return values
