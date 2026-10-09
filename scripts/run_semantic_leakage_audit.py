#!/usr/bin/env python
"""Run a fresh-prompt semantic audit of already selected residual modules.

One process owns one GPU and one complete target-model task at a time. LLMs
run only to extract short trajectories and independent semantic references.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        return ROOT / path
    for marker in ("/runs/", "/activations/", "/data/"):
        if marker in value:
            return ROOT / (marker.strip("/") + "/" + value.split(marker, 1)[1])
    return path


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def prompt_text(row: dict) -> str:
    return str(row.get("prompt") or row.get("question") or "").strip()


def discover_tasks(source: Path, patterns: list[str], max_modules: int = 0) -> list[dict]:
    tasks = []
    for task_dir in sorted((source / "tasks").glob("main_*")):
        if not any(fnmatch.fnmatch(task_dir.name, pattern) for pattern in patterns):
            continue
        manifests = sorted(task_dir.glob("*/adaptive_selection/selected_adaptive_modules.json"))
        for manifest in manifests:
            selected = read_json(manifest)["selected"]
            if max_modules:
                selected = selected[:max_modules]
            pair_dir = manifest.parent.parent
            target, reference = pair_dir.name.split("__semantic_", 1)
            reference, layer = reference.rsplit("_l", 1)
            modules = []
            for rank, item in enumerate(selected, 1):
                module = local_path(item["module_dir"])
                # Resolve relocated submission/run trees from the actual manifest.
                if not module.exists():
                    module = pair_dir / "jobs" / item["profile"] / "decoupler_joint_v2" / "joint_residual_modules" / f"module_{int(item['module']):02d}"
                decoupler = module.parent.parent
                config = read_json(decoupler / "config.json")
                if config.get("second_stage") != "main_direct" or not config.get("external_semantic_activation_dir"):
                    raise ValueError(f"Expected current external-semantic main_direct checkpoint: {decoupler}")
                if config.get("module_direction_mode", "learned") != "learned":
                    raise ValueError(f"Main audit cannot silently use a random-support checkpoint: {decoupler}")
                modules.append({"name": f"rank_{rank}_k{item['support']}_m{item['module']}",
                                "module_dir": str(module), "decoupler_dir": str(decoupler)})
            if modules:
                config = read_json(Path(modules[0]["decoupler_dir"]) / "config.json")
                tasks.append({"name": task_dir.name, "target": target, "reference": reference,
                              "reference_layer": int(layer), "dataset": task_dir.name.split("_")[1],
                              "modules": modules, "source_manifest": str(manifest),
                              "source_manifest_sha256": digest(manifest),
                              "old_activation_dir": str(local_path(config["activation_dir"]))})
    if not tasks:
        raise ValueError(f"No selected modules found under {source}/tasks for {patterns}.")
    return tasks


def strict_tasks_from_plan(plan: dict) -> tuple[list[dict], list[dict]]:
    """Adapt frozen strict-chain modules without rerunning module selection."""
    source = local_path(plan["source_root"])
    tasks = []
    for item in plan["conditions"]:
        dataset, target = item["dataset"], item["target"]
        summary_path = source / dataset / target / "strict_chain_summary" / "strict_module_chain_summary.json"
        if item.get("summary_sha256") and digest(summary_path) != item["summary_sha256"]:
            raise ValueError(f"Frozen strict-chain summary changed: {summary_path}")
        summary = read_json(summary_path)
        row = next((r for r in summary["rows"] if r["module_key"] == item["module_key"]), None)
        if row is None or not row["three_ring_pass"]:
            raise ValueError(f"Frozen module is absent or lacks three-ring passage: {item}")
        module = local_path(row["module_dir"])
        if item.get("module_summary_sha256") and digest(module / "module_summary.json") != item["module_summary_sha256"]:
            raise ValueError(f"Frozen module definition changed: {module}")
        decoupler = module.parent.parent
        old = read_json(decoupler / "config.json")
        ensemble_path = local_path(old["external_semantic_activation_dir"]) / "manifest.json"
        ensemble = read_json(ensemble_path)
        sources = [{"model": s["label"], "layers": [int(s["layer"])]}
                   for s in ensemble["sources"]]
        if len(sources) < 2 or target in {s["model"] for s in sources}:
            raise ValueError(f"Invalid independent training-reference ensemble: {ensemble_path}")
        tasks.append({"name": f"strict_{dataset}_{target}", "dataset": dataset,
                      "target": target, "reference": sources[0]["model"],
                      "reference_layer": sources[0]["layers"][0],
                      "training_references": sources,
                      "reference_feature_mode": "raw_original_models_without_training_pca",
                      "modules": [{"name": row["module_key"], "module_dir": str(module),
                                   "decoupler_dir": str(decoupler)}],
                      "source_manifest": str(summary_path),
                      "source_manifest_sha256": digest(summary_path),
                      "old_activation_dir": str(local_path(old["activation_dir"]))})
    exclusions = []
    for summary_path in sorted(source.glob("*/*/strict_chain_summary/strict_module_chain_summary.json")):
        summary = read_json(summary_path)
        if not summary["rows"]:
            continue
        module = local_path(summary["rows"][0]["module_dir"])
        old = read_json(module.parent.parent / "config.json")
        exclusions.append({"name": summary_path.parent.parent.name,
                           "dataset": summary_path.parent.parent.parent.name,
                           "old_activation_dir": str(local_path(old["activation_dir"]))})
    return tasks, exclusions


def fresh_prompts(task: dict, all_tasks: list[dict], config: dict, output: Path) -> Path:
    """Exclude IDs and normalized texts used by any main task in this domain."""
    from metacog.audits.semantic_leakage import base_id, fingerprint

    destination = output / "fresh_prompts.jsonl"
    receipt = output / "fresh_prompt_manifest.json"
    if receipt.exists():
        old = read_json(receipt)
        valid_inputs = (Path(old["source"]).exists() and
                        digest(Path(old["source"])) == old["source_sha256"] and
                        all(Path(s["path"]).exists() and digest(Path(s["path"])) == s["sha256"]
                            for s in old["exclusion_sources"]))
        if valid_inputs and destination.exists() and digest(destination) == old["output_sha256"]:
            return destination
        raise ValueError(f"Fresh prompt source/exclusions/cache changed: {destination}; use a new RUN_ROOT.")
    forbidden_ids, forbidden_texts, sources = set(), set(), []
    for directory in sorted({t["old_activation_dir"] for t in all_tasks if t["dataset"] == task["dataset"]}):
        path = Path(directory) / "external_semantic_data.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Need original raw-prompt metadata for text-overlap checks: {path}")
        for row in read_rows(path):
            forbidden_ids.add(base_id(str(row["id"])))
            forbidden_texts.add(prompt_hash(prompt_text(row)))
        sources.append({"path": str(path), "sha256": digest(path)})
    for extra in config.get("additional_exclusion_jsonl", []):
        path = local_path(extra)
        for row in read_rows(path):
            forbidden_ids.add(base_id(str(row["id"])))
            forbidden_texts.add(prompt_hash(prompt_text(row)))
        sources.append({"path": str(path), "sha256": digest(path)})
    data_override = config.get("data_paths", {}).get(task["dataset"])
    old_manifest = read_json(Path(task["old_activation_dir"]) / "manifest.json")
    data = local_path(data_override or old_manifest["data"])
    if not data.exists():
        raise FileNotFoundError(f"Full prepared data missing: {data}. Set data_paths.{task['dataset']} in the audit JSON.")
    candidates, seen_ids, duplicate, excluded = {}, {}, 0, 0
    for row in read_rows(data):
        sid, text = str(row["id"]), prompt_text(row)
        key = prompt_hash(text)
        if sid in seen_ids and seen_ids[sid] != key:
            raise ValueError(f"Input ID {sid} identifies different prompt texts in {data}.")
        seen_ids[sid] = key
        if not text or row.get("prompt_is_fully_rendered") or row.get("assistant_prefix"):
            raise ValueError("Fresh audit input must contain raw base prompts, not expanded trajectory rows.")
        if base_id(sid) in forbidden_ids or key in forbidden_texts:
            excluded += 1
            continue
        if key in candidates:
            duplicate += 1
            continue
        row = dict(row)
        # Only prompt/input annotations are available to the independent LLMs.
        for field in ("answer", "rationale", "reference_response", "generated_text"):
            row.pop(field, None)
        candidates[key] = row
    count = min(config["fresh_prompt_count"], len(candidates))
    if count < config["minimum_fresh_prompts"]:
        raise ValueError(f"Only {count} unused prompts remain in {data}; need {config['minimum_fresh_prompts']}. "
                         "Provide a larger same-domain raw-prompt JSONL in data_paths; no old rows will be reused.")
    selected = sorted(candidates, key=lambda key: fingerprint([config["seed"], key]))[:count]
    with destination.open("w", encoding="utf-8") as handle:
        for key in selected:
            handle.write(json.dumps(candidates[key], ensure_ascii=False) + "\n")
    write_json(output / "forbidden_base_ids.json", sorted(forbidden_ids))
    write_json(receipt, {"n_fresh_prompts": count, "remaining_available": len(candidates),
               "excluded_rows": excluded, "duplicates_removed": duplicate, "source": str(data),
               "source_sha256": digest(data), "exclusion_sources": sources,
               "zero_base_id_overlap": True, "zero_normalized_prompt_overlap": True,
               "output_sha256": digest(destination)})
    print(f"[audit-data] {task['name']} fresh={count} available={len(candidates)} "
          f"excluded={excluded} old_prompt_overlap=0", flush=True)
    return destination


def run_stage(command: list[str], output: Path, expected: list[Path], dependencies: list[Path]) -> None:
    """Reuse only a successful extraction under the exact command/input contract."""
    from metacog.audits.semantic_leakage import fingerprint

    validate_extraction_command(command)
    receipt = output / "audit_stage_receipt.json"
    contract = fingerprint({"command": command, "inputs": {str(p): digest(p) for p in dependencies}})
    if receipt.exists():
        old = read_json(receipt)
        if old["contract"] != contract:
            raise ValueError(f"Extraction configuration changed under {output}; use a new RUN_ROOT.")
        if all(p.exists() and p.stat().st_size for p in expected):
            print(f"[audit-reuse] {output}", flush=True)
            return
    print("[audit-stage] " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not all(p.exists() and p.stat().st_size for p in expected):
        raise RuntimeError(f"Extraction returned without required artifacts: {expected}")
    write_json(receipt, {"contract": contract, "command": command,
                         "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


def supported_flags(script: str) -> set[str]:
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    return {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
            for arg in node.args if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str) and arg.value.startswith("--")}


def memory_flags(script: str) -> list[str]:
    if "--memory-efficient" in supported_flags(script):
        return ["--memory-efficient"]
    print(f"[audit-compatibility] {script} lacks --memory-efficient; using its supported extraction path.", flush=True)
    return []


def validate_extraction_command(command: list[str]) -> None:
    unknown = {arg for arg in command[2:] if arg.startswith("--")} - supported_flags(command[1])
    if unknown:
        raise ValueError(f"Unsupported extraction flags for {command[1]}: {sorted(unknown)}")


def extraction_flags(spec, template: str, old_manifest: dict | None = None) -> list[str]:
    old = old_manifest or {}
    flags = ["--model-path", spec.resolved_path, "--dtype", old.get("dtype", spec.dtype),
             "--device-map", "single", "--prompt-style", old.get("prompt_style", "data"),
             "--system-prompt", old.get("system_prompt", ""),
             "--chat-template-enable-thinking", str(template).lower()]
    if old.get("use_chat_template", True):
        flags += ["--use-chat-template"]
    if spec.trust_remote_code:
        flags += ["--trust-remote-code"]
    return flags


def load_aligned(path: Path, ids: list[str] | None = None):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    names = list(map(str, payload["ids"]))
    x = torch.as_tensor(payload["features"])
    if len(names) != len(x) or len(set(names)) != len(names) or x.ndim != 2:
        raise ValueError(f"Invalid activation cache: {path}")
    if ids is None:
        return names, x.float()
    index = {sid: i for i, sid in enumerate(names)}
    missing = [sid for sid in ids if sid not in index]
    if missing:
        raise ValueError(f"{path}: {len(missing)} missing state IDs, e.g. {missing[:3]}")
    return ids, x[torch.tensor([index[sid] for sid in ids])].float()


def lexical_features(rows: list[dict]):
    import torch

    result = []
    for row in rows:
        text = prompt_text(row) + " " + str(row.get("assistant_prefix", ""))
        tokens = re.findall(r"\w+|[^\w\s]", text)
        result.append([math.log1p(len(text)), math.log1p(len(tokens)),
                       math.log1p(sum(c.isdigit() for c in text)),
                       math.log1p(text.count("\n")), math.log1p(text.count("?")),
                       math.log1p(text.count("!")), math.log1p(text.count("(")),
                       math.log1p(text.count(":")), math.log1p(text.count("=")),
                       sum(c.isupper() for c in text) / max(1, len(text)),
                       len(set(t.lower() for t in tokens)) / max(1, len(tokens))])
    return torch.tensor(result, dtype=torch.float32)


def encode_frozen_modules(task: dict, target_dir: Path, semantic_blocks: dict,
                          config: dict, output: Path) -> dict:
    import torch
    from metacog.intervention.direct import load_module_refiner
    from train_decoupler import MLP

    # Audit the frozen residual prediction of the selected neuron combination.
    # The target is the REAL earlier-layer projection, never its residual
    # predictor and never a label manufactured from the later-layer code.
    modules, targets, module_names, provenance = [], [], [], []
    ids = None
    encodings = {}
    device = torch.device(config["probes"]["device"])
    for item in task["modules"]:
        module_dir, dec_dir = Path(item["module_dir"]), Path(item["decoupler_dir"])
        old = read_json(dec_dir / "config.json")
        if old.get("second_stage") != "main_direct":
            raise ValueError("This audit adapter supports the active main_direct checkpoints only.")
        norm = torch.load(dec_dir / "normalization.pt", map_location="cpu", weights_only=False)
        target = torch.load(module_dir / "soft_residual_intervention_targets.pt", map_location="cpu", weights_only=False)
        if str(dec_dir) not in encodings:
            next_ids, raw = load_aligned(target_dir / f"layer_{int(old['next_layer']):03d}.pt", ids)
            ids = next_ids
            checkpoint = torch.load(dec_dir / "best_model.pt", map_location="cpu", weights_only=False)
            state = {key[len("e2."):]: value for key, value in checkpoint.items() if key.startswith("e2.")}
            encoder = MLP(raw.shape[1], old["latent_dim"], old["hidden_dim"], old["dropout"]).to(device)
            encoder.load_state_dict(state, strict=True)
            encoder.eval()
            batch = config["encode_batch_size"]
            parts = []
            with torch.inference_mode():
                for start in range(0, len(raw), batch):
                    standardized = (raw[start:start + batch] - norm["next_mean"]) / norm["next_std"].clamp_min(1e-6)
                    parts.append(encoder(standardized.to(device)).cpu())
            encodings[str(dec_dir)] = torch.cat(parts)
            del encoder, checkpoint, raw
        _, earlier = load_aligned(target_dir / f"layer_{int(old['layer_i']):03d}.pt", ids)
        neuron = torch.as_tensor(target["selected_neurons"]).long()
        loading = torch.as_tensor(target["module_loading"]).float()
        mean = torch.as_tensor(target["continuous_mean"]).flatten()[neuron]
        scale = torch.as_tensor(target["continuous_std"]).flatten()[neuron].clamp_min(1e-6)
        observed = ((earlier[:, neuron] - mean) / scale) @ loading
        refiner, ref_info = load_module_refiner(str(module_dir / "module_refiner.pt"), device)
        if ref_info.get("latent_source") != "main_z2":
            raise ValueError("Refiner latent source differs from frozen main-stage Z2.")
        meta = encodings[str(dec_dir)]
        parts = []
        with torch.inference_mode():
            for start in range(0, len(meta), config["encode_batch_size"]):
                _, prediction = refiner(meta[start:start + config["encode_batch_size"]].to(device))
                parts.append(prediction.cpu() @ loading)
        modules.append(torch.cat(parts))
        targets.append(observed)
        module_names.append(item["name"])
        provenance.append({**item, "checkpoint_sha256": digest(dec_dir / "best_model.pt"),
                           "normalization_sha256": digest(dec_dir / "normalization.pt"),
                           "config_sha256": digest(dec_dir / "config.json"),
                           "refiner_sha256": digest(module_dir / "module_refiner.pt"),
                           "targets_sha256": digest(module_dir / "soft_residual_intervention_targets.pt"),
                           "earlier_layer": old["layer_i"], "later_layer": old["next_layer"],
                           "n_support": len(neuron)})
        del refiner, earlier
    # Align all independently rendered reference activations by exact step IDs.
    semantics = {}
    for name, paths in semantic_blocks.items():
        semantics[name] = torch.cat([load_aligned(path, ids)[1] for path in paths], 1)
    rows_by_id = {str(row["id"]): row for row in read_rows(target_dir / "external_semantic_data.jsonl")}
    semantics["lexical_controls"] = lexical_features([rows_by_id[i] for i in ids])
    forbidden = read_json(output / "forbidden_base_ids.json")
    return {"ids": ids, "steps": [int(rows_by_id[i]["generation_step"]) for i in ids],
            "module_scores": torch.stack(modules, 1), "target_scores": torch.stack(targets, 1),
            "module_names": module_names, "semantics": semantics, "forbidden_ids": forbidden,
            "provenance": {"task": task["name"], "modules": provenance,
                "reference_feature_mode": task.get("reference_feature_mode", "single_original_reference"),
                "signal_definition": "frozen refiner residual prediction projected onto frozen module loading",
                "target_definition": "recorded earlier-layer activation, original train normalization and loading",
                "fresh_prompt_manifest": read_json(output / "fresh_prompt_manifest.json"),
                "all_semantic_dimensions_retained": True,
                "behavior_axis_refitted": False}}


def prepare_bundle(task: dict, all_tasks: list[dict], config: dict, output: Path) -> dict:
    import torch
    from metacog.models.registry import get_model

    torch.set_num_threads(config["cpu_threads"])
    output.mkdir(parents=True, exist_ok=True)
    print(f"[audit-task] {task['name']} started {time.strftime('%F %T')}", flush=True)
    data = fresh_prompts(task, all_tasks, config, output)
    bundle_path = output / "audit_bundle.pt"
    if bundle_path.exists():
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        for frozen in bundle["provenance"]["modules"]:
            if digest(Path(frozen["decoupler_dir"]) / "best_model.pt") != frozen["checkpoint_sha256"]:
                raise ValueError("Source checkpoint changed since bundle extraction.")
            if digest(Path(frozen["module_dir"]) / "module_refiner.pt") != frozen["refiner_sha256"]:
                raise ValueError("Source refiner changed since bundle extraction.")
            if digest(Path(frozen["module_dir"]) / "soft_residual_intervention_targets.pt") != frozen["targets_sha256"]:
                raise ValueError("Frozen module loading changed since bundle extraction.")
            for filename, key in (("normalization.pt", "normalization_sha256"), ("config.json", "config_sha256")):
                if digest(Path(frozen["decoupler_dir"]) / filename) != frozen[key]:
                    raise ValueError(f"Frozen source {filename} changed since bundle extraction.")
    else:
        old_manifest = read_json(Path(task["old_activation_dir"]) / "manifest.json")
        layers = set()
        for module in task["modules"]:
            old = read_json(Path(module["decoupler_dir"]) / "config.json")
            layers.update([old["layer_i"], old["next_layer"]])
        target_dir = output / "activations" / "target"
        target_spec = get_model(task["target"])
        command = [sys.executable, "scripts/extract_generation_step_activations.py",
                   *extraction_flags(target_spec, old_manifest.get("chat_template_enable_thinking", "auto"), old_manifest),
                   "--data", str(data), "--output-dir", str(target_dir), "--layers", *map(str, sorted(layers)),
                   "--record-steps", ",".join(map(str, config["record_steps"])),
                   "--max-generation-steps", str(max(config["record_steps"]) + 1), "--max-samples", "0",
                   "--temperature", "0", "--seed", str(config["seed"]),
                   "--max-length", str(old_manifest.get("max_length", 2048)),
                   "--batch-size", str(config["target_extraction_batch_size"]),
                   *memory_flags("scripts/extract_generation_step_activations.py")]
        run_stage(command, target_dir, [target_dir / f"layer_{l:03d}.pt" for l in layers] +
                  [target_dir / "external_semantic_data.jsonl"], [data])
        training_references = task.get("training_references") or [
            {"model": task["reference"], "layers": [task["reference_layer"]]}]
        references = [{**reference, "name": (f"training_{reference['model']}"
                                           if task.get("training_references") else "training_reference")}
                      for reference in training_references]
        for candidate in config["audit_references"]:
            if candidate["model"] in {task["target"]} | {r["model"] for r in training_references}:
                continue
            references.append({**candidate, "name": f"independent_{candidate['model']}"})
            if len(references) == config["independent_reference_count"] + len(training_references):
                break
        if len(references) != config["independent_reference_count"] + len(training_references):
            raise ValueError("Not enough non-self, independent audit models in config.")
        semantic_blocks = {}
        for reference in references:
            spec = get_model(reference["model"])
            ref_dir = output / "activations" / reference["name"]
            if "layers" in reference:
                ref_layers = reference["layers"]
            else:
                model_config = read_json(local_path(spec.resolved_path) / "config.json")
                count = (model_config.get("text_config") or model_config)["num_hidden_layers"]
                ref_layers = sorted({max(0, min(count - 2, int(count * f))) for f in reference["layer_fractions"]})
            input_path = target_dir / "external_semantic_data.jsonl"
            if reference["name"].startswith("training_"):
                old = read_json(Path(task["modules"][0]["decoupler_dir"]) / "config.json")
                if task.get("training_references"):
                    source = next(s for s in read_json(local_path(old["external_semantic_activation_dir"]) / "manifest.json")["sources"]
                                  if s["label"] == reference["model"])
                    ref_manifest_path = local_path(source["activation_dir"]) / "manifest.json"
                else:
                    ref_manifest_path = local_path(old["external_semantic_activation_dir"]) / "manifest.json"
                ref_manifest = read_json(ref_manifest_path)
                template = ref_manifest.get("chat_template_enable_thinking", "auto")
                max_length = ref_manifest.get("max_length", old_manifest.get("max_length", 2048))
            else:
                ref_manifest = None
                template = spec.chat_template_enable_thinking
                max_length = old_manifest.get("max_length", 2048)
            command = [sys.executable, "scripts/extract_activations.py", *extraction_flags(spec, template, ref_manifest),
                       "--data", str(input_path), "--output-dir", str(ref_dir), "--layers", *map(str, ref_layers),
                       "--max-length", str(max_length), "--batch-size", str(config["semantic_extraction_batch_size"]),
                       "--pooling", "last_token", *memory_flags("scripts/extract_activations.py")]
            paths = [ref_dir / f"layer_{l:03d}.pt" for l in ref_layers]
            run_stage(command, ref_dir, paths, [input_path])
            semantic_blocks[reference["name"]] = paths
        if task.get("training_references"):
            semantic_blocks["training_reference"] = [path for reference in training_references
                                                      for path in semantic_blocks.pop(f"training_{reference['model']}")]
        bundle = encode_frozen_modules(task, target_dir, semantic_blocks, config, output)
        torch.save(bundle, bundle_path)
    return bundle


def run_task(task: dict, all_tasks: list[dict], config: dict, output: Path) -> None:
    from metacog.audits.semantic_leakage import audit_bundle

    bundle = prepare_bundle(task, all_tasks, config, output)
    result = audit_bundle(bundle, output / "analysis", config)
    export_tables(result, output / "analysis")
    print(f"[audit-task] COMPLETE {task['name']} {time.strftime('%F %T')}", flush=True)


def export_tables(summary: dict, output: Path) -> None:
    tables = {"continuous_information.csv": [], "information_bound_sensitivity.csv": [], "density_diagnostics.csv": []}
    for module in summary["modules"]:
        for bank, value in module["continuous_audits"].items():
            tables["continuous_information.csv"].append({"module": module["module"], "bank": bank, **value})
        for bound in module["bounds"]:
            if bound["status"] != "estimated":
                continue
            basic = {"module": module["module"], **{k: v for k, v in bound.items()
                      if not isinstance(v, (list, dict))}}
            for sensitivity in bound["sensitivity"]:
                tables["information_bound_sensitivity.csv"].append({**basic, **sensitivity})
            for density in bound["density_models"]:
                tables["density_diagnostics.csv"].append({"module": module["module"], "step": bound["step"], **density})
    for name, rows in tables.items():
        if rows:
            with (output / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def validate_config(config: dict) -> None:
    required_positive = ("fresh_prompt_count", "minimum_fresh_prompts", "min_bound_prompts",
                         "target_extraction_batch_size", "semantic_extraction_batch_size",
                         "encode_batch_size", "cpu_threads", "independent_reference_count", "bootstrap_samples")
    for name in required_positive:
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config["minimum_fresh_prompts"] > config["fresh_prompt_count"]:
        raise ValueError("minimum_fresh_prompts exceeds fresh_prompt_count.")
    if not 0 < config["train_fraction"] < 1 or not 0 < config["validation_fraction"] < 1 - config["train_fraction"]:
        raise ValueError("Invalid prompt-disjoint split fractions.")
    for field in ("record_steps", "bound_steps"):
        values = config[field]
        if not values or len(set(values)) != len(values) or any(not isinstance(s, int) or s < 0 for s in values):
            raise ValueError(f"{field} must contain unique, nonnegative integer steps.")
    if not set(config["bound_steps"]) <= set(config["record_steps"]):
        raise ValueError("bound_steps must be a subset of record_steps.")
    if not isinstance(config["bins"], int) or not 2 <= config["bins"] <= 8:
        raise ValueError("bins must be an integer from 2 to 8.")
    if not 0 < config["probability_floor"] < 1 / config["bins"] or not 0 < config["alpha"] < 1:
        raise ValueError("Invalid probability floor or family confidence level.")
    budgets = list(config["density_kl_budgets_bits"])
    if config.get("assumed_density_kl_budget_bits") is not None:
        budgets.append(config["assumed_density_kl_budget_bits"])
    if not budgets or any(not math.isfinite(b) or b < 0 for b in budgets):
        raise ValueError("Density KL sensitivity budgets must be finite and nonnegative.")
    probes = config["probes"]
    families = probes["families"]
    if len(set(families)) != len(families) or not set(families) <= {"linear", "mlp", "deep_residual", "ensemble"}:
        raise ValueError("Unknown or duplicate probe families.")
    if not families or ("ensemble" in families and len(families) < 3):
        raise ValueError("Ensemble requires at least two fixed member families.")
    for name in ("epochs", "patience", "hidden_dim", "batch_size"):
        if not isinstance(probes[name], int) or probes[name] < 1:
            raise ValueError(f"probes.{name} must be a positive integer.")
    if not 0 <= probes["dropout"] < 1 or probes["lr"] <= 0 or probes["weight_decay"] < 0:
        raise ValueError("Invalid probe optimizer parameters.")
    names = [r["model"] for r in config["audit_references"]]
    if len(set(names)) != len(names):
        raise ValueError("Independent audit model names must be unique.")
    for reference in config["audit_references"]:
        if not reference.get("layer_fractions") or any(not 0 <= f < 1 for f in reference["layer_fractions"]):
            raise ValueError("Each independent reference requires layer_fractions in [0,1).")


def preflight(tasks: list[dict], config: dict, exclusion_tasks: list[dict] | None = None) -> list[str]:
    from metacog.models.registry import get_model

    missing = []
    for task in tasks:
        models = {task["target"]} | {r["model"] for r in task.get("training_references", [])}
        if not task.get("training_references"):
            models.add(task["reference"])
        refs = [r["model"] for r in config["audit_references"] if r["model"] not in models]
        if len(refs) < config["independent_reference_count"]:
            raise ValueError(f"Insufficient non-self audit models for {task['name']}.")
        models.update(refs[:config["independent_reference_count"]])
        for name in models:
            p = local_path(get_model(name).resolved_path) / "config.json"
            if not p.exists():
                missing.append(str(p))
        for module in task["modules"]:
            old = read_json(Path(module["decoupler_dir"]) / "config.json")
            ref_manifest = local_path(old["external_semantic_activation_dir"]) / "manifest.json"
            if not ref_manifest.exists():
                missing.append(str(ref_manifest))
            elif task.get("training_references"):
                for source in read_json(ref_manifest)["sources"]:
                    p = local_path(source["activation_dir"]) / "manifest.json"
                    if not p.exists():
                        missing.append(str(p))
            for file in ("best_model.pt", "normalization.pt"):
                p = Path(module["decoupler_dir"]) / file
                if not p.exists():
                    missing.append(str(p))
            for file in ("module_refiner.pt", "soft_residual_intervention_targets.pt"):
                p = Path(module["module_dir"]) / file
                if not p.exists():
                    missing.append(str(p))
        p = Path(task["old_activation_dir"]) / "external_semantic_data.jsonl"
        if not p.exists():
            missing.append(str(p))
        manifest_path = Path(task["old_activation_dir"]) / "manifest.json"
        if not manifest_path.exists():
            missing.append(str(manifest_path))
        else:
            data = local_path(config.get("data_paths", {}).get(task["dataset"]) or read_json(manifest_path)["data"])
            if not data.exists():
                missing.append(str(data))
    domains = {t["dataset"] for t in tasks}
    for t in exclusion_tasks or tasks:
        p = Path(t["old_activation_dir"]) / "external_semantic_data.jsonl"
        if t["dataset"] in domains and not p.exists():
            missing.append(str(p))
    for extra in config.get("additional_exclusion_jsonl", []):
        if not local_path(extra).exists():
            missing.append(str(local_path(extra)))
    return sorted(set(missing))


def aggregate_results(output: Path, tasks: list[dict]) -> None:
    """Publish all modules, including null/negative results, without reranking."""
    for filename in ("continuous_information.csv", "information_bound_sensitivity.csv", "density_diagnostics.csv"):
        rows = []
        for task in tasks:
            path = output / task["name"] / "analysis" / filename
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    rows.extend({"task": task["name"], "dataset": task["dataset"], **r} for r in csv.DictReader(handle))
        if rows:
            with (output / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                writer.writeheader()
                writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/semantic_leakage_audit_v1.json")
    parser.add_argument("--source-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", help="Task names or glob patterns; defaults to main_*.")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--selected-plan", help="Frozen strict-chain follow-up plan JSON.")
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path(args.output_dir).resolve()
    if args.worker_index is not None:
        manifest = read_json(output / "audit_manifest.json")
        run_task(manifest["tasks"][args.worker_index], manifest["exclusion_tasks"], manifest["config"],
                 output / manifest["tasks"][args.worker_index]["name"])
        return
    config = read_json(local_path(args.config))
    source = local_path(args.source_root or config["source_root"])
    patterns = args.tasks or config["task_patterns"]
    if args.smoke:
        patterns = args.tasks or ["main_mathqa_qwen3_4b"]
        config.update({"fresh_prompt_count": 192, "minimum_fresh_prompts": 160,
                       "record_steps": [0, 4], "bound_steps": [0], "min_bound_prompts": 16,
                       "bootstrap_samples": 64, "maximum_modules": 1})
        config["probes"].update({"epochs": 2, "patience": 2, "hidden_dim": 32,
                                 "families": ["linear", "mlp", "ensemble"]})
    validate_config(config)
    if args.selected_plan:
        tasks, exclusion_tasks = strict_tasks_from_plan(read_json(local_path(args.selected_plan)))
    else:
        tasks = discover_tasks(source, patterns, config.get("maximum_modules", 0))
        # A one-task pilot must still exclude prompts used by the other main tasks.
        exclusion_tasks = [{k: t[k] for k in ("name", "dataset", "old_activation_dir")}
                           for t in discover_tasks(source, ["main_*"], 1)]
    config["planned_comparisons"] = sum(len(t["modules"]) for t in tasks) * len(config["bound_steps"])
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / ".pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {"schema_version": 1, "source_root": str(source), "tasks": tasks,
                "exclusion_tasks": exclusion_tasks,
                "implementation_sha256": {p: digest(ROOT / p) for p in
                    ("scripts/run_semantic_leakage_audit.py", "metacog/audits/semantic_leakage.py",
                     "scripts/extract_generation_step_activations.py", "scripts/extract_activations.py",
                     "metacog/intervention/direct.py", "scripts/train_decoupler.py")},
                "config": config, "smoke_only": args.smoke,
                "preflight_missing": preflight(tasks, config, exclusion_tasks)}
    manifest_path = output / "audit_manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if any(previous.get(k) != manifest[k] for k in
               ("config", "tasks", "exclusion_tasks", "implementation_sha256")):
            raise ValueError("RUN_ROOT already contains a different frozen audit contract. Use a new RUN_ROOT.")
    write_json(manifest_path, manifest)
    print(f"[audit-plan] tasks={len(tasks)} modules={sum(len(t['modules']) for t in tasks)} "
          f"fresh_prompts_per_target={config['fresh_prompt_count']} planned_bounds={config['planned_comparisons']} "
          f"smoke={args.smoke}", flush=True)
    for task in tasks:
        print(f"  {task['name']}: {len(task['modules'])} frozen modules", flush=True)
    if manifest["preflight_missing"]:
        print(f"[preflight] {len(manifest['preflight_missing'])} server artifacts absent; "
              f"full list: {manifest_path}", flush=True)
    if args.plan_only:
        return
    if manifest["preflight_missing"]:
        raise FileNotFoundError("Server preflight failed. Inspect audit_manifest.json preflight_missing before extraction.")
    if config.get("assumed_density_kl_budget_bits") is not None:
        print("[audit-assumption] A density KL budget was supplied. Any positive bound remains conditional on that assumption.", flush=True)
    gpus = list(dict.fromkeys(args.gpus))
    write_json(output / "launcher.json", {"pid": os.getpid(), "gpus": gpus})
    stopped = threading.Event()
    active, active_lock = {}, threading.Lock()

    def stop_children(signum, frame):
        stopped.set()
        with active_lock:
            running = list(active.values())
        for proc in running:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        print("[audit-queue] Stop requested; terminating only this launcher's task process groups.", flush=True)

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    def worker(gpu: str, indices: list[int]) -> list[dict]:
        status = []
        for index in indices:
            if stopped.is_set():
                break
            task = tasks[index]
            directory = output / task["name"]
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "task.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1",
                   "OMP_NUM_THREADS": str(config["cpu_threads"])}
            with log_path.open("a", encoding="utf-8") as handle:
                with active_lock:
                    if stopped.is_set():
                        break
                    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                         "--output-dir", str(output), "--worker-index", str(index)], cwd=ROOT,
                         env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    active[proc.pid] = proc
                write_json(directory / "process.json", {"pid": proc.pid, "gpu": gpu, "log": str(log_path)})
                print(f"[audit-queue] START gpu={gpu} {task['name']} pid={proc.pid} log={log_path}", flush=True)
                rc = proc.wait()
                with active_lock:
                    active.pop(proc.pid, None)
            row = {"task": task["name"], "gpu": gpu, "exit_code": rc,
                   "status": "complete" if rc == 0 else "stopped" if stopped.is_set() else "failed", "log": str(log_path)}
            write_json(directory / "status.json", row)
            status.append(row)
            print(f"[audit-queue] {row['status'].upper()} gpu={gpu} {task['name']} exit={rc}", flush=True)
        return status

    status = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, list(range(i, len(tasks), len(gpus)))) for i, gpu in enumerate(gpus)]
        try:
            for future in concurrent.futures.as_completed(futures):
                status.extend(future.result())
        except BaseException:
            stop_children(None, None)
            raise
    write_json(output / "status_summary.json", {"tasks": sorted(status, key=lambda r: r["task"])})
    aggregate_results(output, tasks)
    if stopped.is_set():
        raise SystemExit(130)
    if any(s["exit_code"] for s in status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
