#!/usr/bin/env python3
"""Validate the frozen result deposit without importing ML dependencies."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def boolean(value):
    return str(value).lower() == "true"


def identity(row):
    return tuple(row[k] for k in ("dataset", "target", "module_key"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    data = root / "source_data"
    modules = rows(data / "modules.csv")
    assert len(modules) == len({identity(r) for r in modules}) == 384
    assert len(rows(data / "conditions.csv")) == 24
    assert sum(boolean(r["transmission_pass"]) for r in modules) == 384
    assert sum(boolean(r["behavior_pass"]) for r in modules) == 89
    passing = {identity(r) for r in modules if boolean(r["three_ring_pass"])}
    assert len(passing) == 46
    for row in modules:
        assert boolean(row["three_ring_pass"]) == all(boolean(row[k]) for k in (
            "transmission_pass", "behavior_pass", "next_token_replicated_directional_pass"))
    constructs = rows(data / "constructs.csv")
    assert len(constructs) == 3840
    grouped = {}
    for row in constructs:
        grouped.setdefault(identity(row), []).append(row)
    assert set(grouped) == {identity(r) for r in modules}
    for values in grouped.values():
        assert len({r["construct"] for r in values}) == 10
        assert sum(r["group"] == "behavior" for r in values) == 5
        assert sum(r["group"] == "monitoring" for r in values) == 5
    reports = rows(data / "report_conditions.csv")
    assert len(reports) == len({(r["dataset"], r["target"]) for r in reports}) == 24
    assert all(r["activation_source"] == "runtime_remeasured" and not boolean(r["reused"])
               for r in reports)
    assert len(rows(data / "report_neurons.csv")) == 384
    trajectories = rows(data / "trajectory.csv")
    assert len(trajectories) == 46 and {identity(r) for r in trajectories} == passing
    assert not rows(data / "trajectory_pending.csv")
    index = rows(root / "metadata/source_file_index.csv")
    assert len(index) == len({r["deposited_path"] for r in index})
    compressed_count = 0
    for entry in index:
        file = root / entry["deposited_path"]
        assert file.is_file(), file
        assert sha(file) == entry["deposited_sha256"], file
        if file.suffix == ".gz":
            compressed_count += 1
            if entry["transformation"].startswith("gzip;"):
                with gzip.open(file, "rb") as handle:
                    digest = hashlib.sha256()
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                assert digest.hexdigest() == entry["source_sha256"], file
            else:
                with gzip.open(file, "rt", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    assert not {"question", "options", "prompt", "source_prompt"}.intersection(reader.fieldnames)
                    assert sum(1 for _ in reader) == int(entry["rows"]), file
        elif entry["transformation"] == "byte-identical original result":
            assert entry["deposited_sha256"] == entry["source_sha256"]
    assert compressed_count == 384 + 24 + 46
    prohibited = {".pt", ".pth", ".bin", ".safetensors", ".npy", ".npz", ".pkl", ".pickle"}
    assert not [p for p in root.rglob("*") if p.suffix in prohibited]
    manifest = root / "MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            assert sha(root / name) == digest, name
    return {"status": "passed", "main_conditions": 24, "frozen_modules": 384,
            "construct_estimates": 3840, "behavior_pass": 89, "three_ring_pass": 46,
            "report_conditions": 24, "report_neurons": 384, "persistent_followups": 46,
            "source_files_verified": len(index), "compressed_sample_tables": compressed_count,
            "gpu_experiments_rerun": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
