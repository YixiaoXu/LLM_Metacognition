#!/usr/bin/env python
"""Freeze pilot-selected witnesses or all completed candidates, then test on new prompts."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import fnmatch
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import _bootstrap  # noqa: F401
import run_semantic_leakage_audit as common

ROOT = common.ROOT
read_json, write_json, digest, local_path = common.read_json, common.write_json, common.digest, common.local_path


def validate_config(config: dict) -> None:
    selection_mode = config.get("witness_selection_mode", "top_positive")
    if selection_mode not in {"top_positive", "all_completed_candidates"}:
        raise ValueError("witness_selection_mode must be top_positive or all_completed_candidates.")
    for key in ("witness_count", "min_train_per_bin", "min_calibration_per_bin", "min_test_prompts",
                "max_test_prompts", "bootstrap_samples", "target_extraction_batch_size",
                "semantic_extraction_batch_size", "encode_batch_size", "cpu_threads"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if selection_mode == "top_positive" and config["witness_count"] > 3:
        raise ValueError("This protocol freezes at most three witnesses, globally, not per target model.")
    if config["min_test_prompts"] < 16 or config["max_test_prompts"] < config["min_test_prompts"]:
        raise ValueError("Invalid independent-test sample limits.")
    count = config["test_prompt_count"]
    if count is not None and (not isinstance(count, int) or not config["min_test_prompts"] <= count <= config["max_test_prompts"]):
        raise ValueError("test_prompt_count must be null (pilot-based planning) or within the configured limits.")
    if not 0 < config["alpha"] < 1 or not 0 < config["calibration_fraction"] < 1:
        raise ValueError("Invalid confidence or calibration fraction.")
    bins = config["bin_candidates"]
    if not bins or len(set(bins)) != len(bins) or any(not isinstance(k, int) or not 2 <= k <= 8 for k in bins):
        raise ValueError("bin_candidates must contain unique integer resolutions in 2..8.")
    floors = config["probability_floor_candidates"]
    if not floors or any(not math.isfinite(f) or not 0 < f < 1 / max(bins) for f in floors):
        raise ValueError("Smoothing candidates must lie in (0,1/max_bins).")
    if not 0 < config["upper_cap_quantile"] < 1 or config["target_bound_penalty_bits"] <= 0:
        raise ValueError("Invalid pilot cap quantile or precision target.")
    if not isinstance(config["primary_step"], int) or config["primary_step"] < 0:
        raise ValueError("primary_step must be a nonnegative integer, fixed before test.")
    if not config["semantic_null_families"] or not set(config["semantic_null_families"]) <= {"linear", "nonlinear"}:
        raise ValueError("Provide at least one known semantic null: linear and/or nonlinear.")
    if not math.isfinite(config["null_noise_std"]) or config["null_noise_std"] < 0:
        raise ValueError("Null noise must have a nonnegative finite standard deviation.")
    assumption = config.get("target_density_error_assumption")
    if assumption is not None and (not isinstance(assumption, dict) or not assumption.get("rationale") or
            not math.isfinite(assumption.get("upper_bits", float("nan"))) or assumption["upper_bits"] < 0):
        raise ValueError("Density error is unknown by default; an assumption requires upper_bits and rationale.")
    # Reuse optimizer validation without coupling the witness protocol to v1 bounds.
    template = read_json(ROOT / "configs/semantic_leakage_audit_v1.json")
    template["probes"] = config["probes"]
    common.validate_config(template)


def select_witnesses(pilot: Path, manifest: dict, config: dict) -> tuple[list[dict], list[dict]]:
    candidates = []
    for task in manifest["tasks"]:
        if not any(fnmatch.fnmatch(task["name"], pat) for pat in config["task_patterns"]):
            continue
        for module in task["modules"]:
            path = pilot / task["name"] / "analysis" / module["name"] / "audit_result.json"
            if not path.exists():
                continue
            result = read_json(path)
            audit = result.get("continuous_audits", {}).get("expanded_bank", {})
            score = audit.get("ci95_low_bits")
            if score is None or not math.isfinite(score):
                continue
            candidates.append({"task": task["name"], "module": module["name"], "pilot_ranking_score": score,
                               "pilot_result": str(path), "pilot_result_sha256": digest(path)})
    candidates.sort(key=lambda r: (-r["pilot_ranking_score"], r["task"], r["module"]))
    requested = config.get("frozen_witnesses", [])
    selection_mode = config.get("witness_selection_mode", "top_positive")
    if requested and selection_mode == "all_completed_candidates":
        raise ValueError("Explicit frozen_witnesses cannot be combined with all_completed_candidates.")
    if requested:
        keys = [(r["task"], r["module"]) for r in requested]
        if len(set(keys)) != len(keys) or not 1 <= len(keys) <= 3:
            raise ValueError("Specify one to three distinct frozen task/module pairs.")
        lookup = {(r["task"], r["module"]): r for r in candidates}
        missing = [key for key in keys if key not in lookup]
        if missing:
            raise ValueError(f"Requested witnesses lack completed pilot results: {missing}")
        selected = [lookup[key] for key in keys]
    elif selection_mode == "all_completed_candidates":
        selected = list(candidates)
        if not selected:
            raise ValueError("The pilot contains no completed candidates with finite expanded-bank scores.")
    else:
        selected = [r for r in candidates if r["pilot_ranking_score"] > 0][:config["witness_count"]]
        if len(selected) < config["witness_count"] and config.get("smoke_only"):
            selected = candidates[:config["witness_count"]]
        if len(selected) < config["witness_count"]:
            raise ValueError("Too few completed positive pilot candidates; no automatic fallback to a different criterion.")
    for i, row in enumerate(selected, 1):
        row.update({"witness_id": f"witness_{i}", "alpha": config["alpha"] / len(selected)})
    return selected, candidates


def collect_exclusions(pilot: Path, config: dict, domains: set[str], output: Path,
                       inherited: list[str] | None = None) -> dict:
    """Snapshot every visible historical audit population, not just winners."""
    roots = {pilot.resolve()}
    for pattern in config["historical_audit_globs"]:
        roots.update(p.resolve() for p in ROOT.glob(pattern) if p.is_dir() and p.resolve() != output)
    result = {domain: {} for domain in domains}
    for root in sorted(roots):
        paths = list(root.glob("*/fresh_prompts.jsonl")) + list(root.glob("*/test/fresh_prompts.jsonl"))
        for path in paths:
            task_name = path.parent.name if path.parent.name != "test" else path.parent.parent.name
            parts = task_name.split("_")
            if len(parts) > 1 and parts[1] in result:
                result[parts[1]][str(path.resolve())] = digest(path)
    for value in [*config.get("additional_exclusion_jsonl", []), *(inherited or [])]:
        p = local_path(value)
        for domain in domains:
            result[domain][str(p)] = digest(p)
    for domain, paths in result.items():
        if not paths:
            raise ValueError(f"Cannot verify old audit prompt texts for {domain}; sync the pilot fresh_prompts.jsonl files.")
    return result


def verify_snapshot(manifest: dict) -> None:
    for path, sha in manifest["implementation_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Implementation changed after witness freeze: {path}; use a new RUN_ROOT.")
    for row in manifest["witnesses"]:
        if digest(Path(row["pilot_result"])) != row["pilot_result_sha256"]:
            raise ValueError("Selected pilot result changed after freeze.")
    for task in manifest["tasks"]:
        if digest(Path(task["pilot_bundle"])) != task["pilot_bundle_sha256"]:
            raise ValueError("Pilot activation bundle changed after freeze.")
    for sources in manifest["exclusions_by_domain"].values():
        for path, sha in sources.items():
            if digest(Path(path)) != sha:
                raise ValueError(f"Historical prompt population changed: {path}")


def verify_source_modules(bundle: dict, task: dict) -> None:
    lookup = {m["name"]: m for m in bundle["provenance"]["modules"]}
    for module in task["modules"]:
        old = lookup[module["name"]]
        for directory, file, key in (
            ("decoupler_dir", "best_model.pt", "checkpoint_sha256"),
            ("decoupler_dir", "normalization.pt", "normalization_sha256"),
            ("decoupler_dir", "config.json", "config_sha256"),
            ("module_dir", "module_refiner.pt", "refiner_sha256"),
            ("module_dir", "soft_residual_intervention_targets.pt", "targets_sha256"),
        ):
            if digest(Path(module[directory]) / file) != old[key]:
                raise ValueError(f"Frozen upstream artifact no longer matches pilot: {module[directory]}/{file}")


def extraction_config(manifest: dict, task: dict, count: int) -> dict:
    config = copy.deepcopy(manifest["pilot_config"])
    current = manifest["config"]
    config.update({k: current[k] for k in ("target_extraction_batch_size", "semantic_extraction_batch_size",
                                           "encode_batch_size", "cpu_threads", "seed", "probes")})
    config.update({"record_steps": [current["primary_step"]], "bound_steps": [current["primary_step"]],
                   "fresh_prompt_count": count, "minimum_fresh_prompts": current["min_test_prompts"],
                   "additional_exclusion_jsonl": list(manifest["exclusions_by_domain"][task["dataset"]]),
                   "data_paths": {**config.get("data_paths", {}), **current.get("data_paths", {})}})
    return config


def run_task(manifest: dict, index: int, output: Path) -> None:
    import torch
    from metacog.audits.conditional_witness import fit_witness, evaluate_witness

    verify_snapshot(manifest)
    config, task = manifest["config"], manifest["tasks"][index]
    torch.set_num_threads(config["cpu_threads"])
    directory = output / task["name"]
    witnesses = [w for w in manifest["witnesses"] if w["task"] == task["name"]]
    bundle = torch.load(task["pilot_bundle"], map_location="cpu", weights_only=False)
    verify_source_modules(bundle, task)
    print(f"[witness-task] {task['name']} witnesses={len(witnesses)} reused_pilot={task['pilot_bundle']}", flush=True)
    plans = []
    for witness in witnesses:
        fit = fit_witness(bundle, witness["module"], directory / witness["witness_id"] / "fit", config, witness["alpha"])
        plans.append(fit["planned_test_prompts"])
    del bundle
    count = config["test_prompt_count"] or max(plans)
    effective = extraction_config(manifest, task, count)
    write_json(directory / "test_extraction_config.json", effective)
    print(f"[witness-test-plan] fresh_prompts={count}; all predictors and design choices already frozen", flush=True)
    fresh = common.prepare_bundle(task, manifest["exclusion_tasks"], effective, directory / "test")
    verify_snapshot(manifest)
    # Source check after extraction also catches a checkpoint changed during a long job.
    pilot = torch.load(task["pilot_bundle"], map_location="cpu", weights_only=False)
    verify_source_modules(pilot, task)
    del pilot
    for witness in witnesses:
        result = evaluate_witness(fresh, witness["module"], directory / witness["witness_id"] / "fit",
                                  directory / witness["witness_id"] / "analysis", config, witness["alpha"])
        print(f"[witness-complete] {witness['witness_id']} {witness['module']} "
              f"gain_LCB={result['signals']['real']['robust_gain_lcb_bits']:.6f} "
              f"null_alerts={result['semantic_null_alerts']} density_error_measured=False", flush=True)


def export_results(output: Path, manifest: dict) -> dict:
    rows, detail, witnesses = [], [], []
    for w in manifest["witnesses"]:
        path = output / w["task"] / w["witness_id"] / "analysis/witness_result.json"
        if not path.exists():
            witnesses.append({**w, "status": "not_completed"})
            continue
        r = read_json(path)
        real = r["signals"]["real"]
        summary = {**w, "status": "complete", "n_test_prompts": r["n_test_prompts"], "bins": r["bins"],
                   "minimum_mean_gain_bits": real["minimum_mean_gain_bits"],
                   "gain_lcb_bits": real["robust_gain_lcb_bits"],
                   "positive_vs_frozen_predictor_bank": real["positive_vs_all_frozen_semantic_predictors"],
                   "semantic_null_alerts": r["semantic_null_alerts"],
                   "predictive_witness_with_null_sanity": r["predictive_witness_with_null_sanity"],
                   **r["conditional_information"]}
        witnesses.append(summary)
        rows.append({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in summary.items()})
        for name, signal in r["signals"].items():
            for family, comparison in signal["comparisons"].items():
                detail.append({"task": w["task"], "module": w["module"], "signal": name,
                               "semantic_comparator": family, **comparison})
    result = {"schema_version": 2, "planned_witnesses": len(manifest["witnesses"]), "family_alpha": manifest["config"]["alpha"],
              "per_witness_alpha": manifest["witnesses"][0]["alpha"] if manifest["witnesses"] else None,
              "witness_selection_mode": manifest["config"].get("witness_selection_mode", "top_positive"),
              "multiplicity_control": "Bonferroni across every frozen witness in this final-test family",
              "completed_witnesses": len(rows), "witnesses": witnesses,
              "at_least_one_predictive_witness_with_null_sanity": any(w.get("predictive_witness_with_null_sanity", False) for w in witnesses),
              "at_least_one_positive_CMI_bound_under_assumption": any(w.get("positive_conditional_information_under_assumption", False) for w in witnesses),
              "unconditional_semantic_leakage_exclusion_proven": False, "smoke_only": manifest["config"]["smoke_only"],
              "interpretation": "Existence is scoped to the frozen predictor bank. Shannon CMI requires a stated, justified target-density KL bound."}
    if result["smoke_only"]:
        result["at_least_one_predictive_witness_with_null_sanity"] = False
        result["at_least_one_positive_CMI_bound_under_assumption"] = False
    write_json(output / "existence_summary.json", result)
    for name, values in (("witness_summary.csv", rows), ("predictor_comparisons.csv", detail)):
        if values:
            with (output / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in values for k in r)))
                writer.writeheader()
                writer.writerows(values)
    return result


def launch(manifest: dict, output: Path, gpus: list[str]) -> None:
    active, lock, stopped = {}, threading.Lock(), threading.Event()
    def stop(signum, frame):
        stopped.set()
        with lock:
            for proc in active.values():
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    write_json(output / "launcher.json", {"pid": os.getpid(), "gpus": gpus})
    def worker(gpu, indices):
        statuses = []
        for index in indices:
            if stopped.is_set():
                break
            task = manifest["tasks"][index]
            directory = output / task["name"]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "task.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1",
                   "OMP_NUM_THREADS": str(manifest["config"]["cpu_threads"])}
            with path.open("a", encoding="utf-8") as handle:
                with lock:
                    if stopped.is_set():
                        break
                    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--output-dir", str(output),
                            "--worker-index", str(index)], cwd=ROOT, env=env, stdout=handle,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    active[proc.pid] = proc
                write_json(directory / "process.json", {"pid": proc.pid, "gpu": gpu})
                print(f"[witness-queue] START gpu={gpu} {task['name']} pid={proc.pid} log={path}", flush=True)
                rc = proc.wait()
                with lock:
                    active.pop(proc.pid, None)
            row = {"task": task["name"], "gpu": gpu, "exit_code": rc,
                   "status": "complete" if rc == 0 else "stopped" if stopped.is_set() else "failed"}
            write_json(directory / "status.json", row)
            statuses.append(row)
            print(f"[witness-queue] {row['status']} {task['name']} exit={rc}", flush=True)
        return statuses
    statuses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, range(i, len(manifest["tasks"]), len(gpus))) for i, gpu in enumerate(gpus)]
        try:
            for future in concurrent.futures.as_completed(futures):
                statuses.extend(future.result())
        except BaseException:
            stop(None, None)
            raise
    write_json(output / "status_summary.json", {"tasks": statuses})
    export_results(output, manifest)
    if stopped.is_set():
        raise SystemExit(130)
    if any(row["exit_code"] for row in statuses):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/semantic_witness_v2.json")
    parser.add_argument("--pilot-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--witness", nargs="+", help="Exact task:module names, selected on pilot data only.")
    parser.add_argument("--witness-count", type=int)
    parser.add_argument("--all-candidates", action="store_true",
                        help="Freeze every completed finite-score pilot candidate before final extraction.")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path(args.output_dir).resolve()
    if args.worker_index is not None:
        run_task(read_json(output / "witness_manifest.json"), args.worker_index, output)
        return
    config = read_json(local_path(args.config))
    if args.pilot_root:
        config["pilot_root"] = args.pilot_root
    if args.witness_count is not None:
        config["witness_count"] = args.witness_count
    if args.all_candidates:
        config["witness_selection_mode"] = "all_completed_candidates"
    if args.tasks:
        config["task_patterns"] = args.tasks
    if args.witness:
        config["frozen_witnesses"] = [dict(zip(("task", "module"), value.split(":", 1))) for value in args.witness]
        if any("module" not in row for row in config["frozen_witnesses"]):
            parser.error("--witness requires task:module")
    config["smoke_only"] = args.smoke
    if args.smoke:
        config.update({"witness_count": 1, "min_train_per_bin": 8, "min_calibration_per_bin": 4,
                       "min_test_prompts": 64, "max_test_prompts": 128, "test_prompt_count": 128,
                       "bootstrap_samples": 64, "continuous_auxiliary": False})
        config["probes"].update({"epochs": 2, "patience": 2, "hidden_dim": 32,
                                 "families": ["linear", "mlp", "ensemble"]})
    validate_config(config)
    for script in ("scripts/extract_generation_step_activations.py", "scripts/extract_activations.py"):
        required = {"--model-path", "--data", "--layers", "--output-dir", "--batch-size", "--use-chat-template"}
        if required - common.supported_flags(script):
            raise ValueError(f"Extractor interface is incompatible: {script}")
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / ".pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = output / "witness_manifest.json"
    if path.exists():
        manifest = read_json(path)
        if config != manifest["config"]:
            raise ValueError("Frozen witness configuration differs; do not change settings after test. Use a new RUN_ROOT.")
        verify_snapshot(manifest)
    else:
        pilot = local_path(config["pilot_root"])
        pilot_manifest_path = pilot / "audit_manifest.json"
        if not pilot_manifest_path.exists():
            candidates = sorted(str(path.parent.relative_to(ROOT)) for path in (ROOT / "runs").glob("**/audit_manifest.json"))
            available = "\n  ".join(candidates[-12:]) if candidates else "none found"
            raise FileNotFoundError(
                f"PILOT_ROOT does not contain audit_manifest.json: {pilot}.\n"
                f"Available audit roots:\n  {available}\n"
                "Use scripts/run_semantic_witness_full.sh to create a new pilot automatically."
            )
        pilot_manifest = read_json(pilot_manifest_path)
        if pilot_manifest.get("smoke_only") and not args.smoke:
            raise ValueError("A smoke audit cannot select publication witnesses.")
        selected, candidates = select_witnesses(pilot, pilot_manifest, config)
        tasks = []
        for source_task in pilot_manifest["tasks"]:
            names = {r["module"] for r in selected if r["task"] == source_task["name"]}
            if not names:
                continue
            task = copy.deepcopy(source_task)
            task["modules"] = [{**m, "module_dir": str(local_path(m["module_dir"])),
                                 "decoupler_dir": str(local_path(m["decoupler_dir"]))}
                                for m in task["modules"] if m["name"] in names]
            task["old_activation_dir"] = str(local_path(task["old_activation_dir"]))
            bp = pilot / task["name"] / "audit_bundle.pt"
            task.update({"pilot_bundle": str(bp), "pilot_bundle_sha256": digest(bp)})
            tasks.append(task)
        for task in tasks:
            if not (pilot / task["name"] / "fresh_prompts.jsonl").exists():
                raise FileNotFoundError(f"Need selected pilot prompt texts for overlap checks: {pilot / task['name'] / 'fresh_prompts.jsonl'}")
        exclusions = collect_exclusions(pilot, config, {t["dataset"] for t in tasks}, output,
                                        pilot_manifest["config"].get("additional_exclusion_jsonl", []))
        old_tasks = [{**t, "old_activation_dir": str(local_path(t["old_activation_dir"]))}
                     for t in pilot_manifest["exclusion_tasks"]]
        implementation = ["scripts/run_semantic_witness.py", "scripts/run_semantic_leakage_audit.py",
                          "metacog/audits/conditional_witness.py", "metacog/audits/semantic_leakage.py",
                          "scripts/extract_generation_step_activations.py", "scripts/extract_activations.py",
                          "metacog/intervention/direct.py", "scripts/train_decoupler.py"]
        selection_rule = ("all_completed_finite_expanded_bank_candidates"
                          if config.get("witness_selection_mode") == "all_completed_candidates"
                          else "largest_old_audit_expanded_bank_CI_lower_or_explicit_pilot_choice")
        manifest = {"schema_version": 2, "config": config, "pilot_config": pilot_manifest["config"],
                    "tasks": tasks, "witnesses": selected, "exclusion_tasks": old_tasks,
                    "exclusions_by_domain": exclusions, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "selection_rule": selection_rule,
                    "implementation_sha256": {p: digest(ROOT / p) for p in implementation}}
        write_json(output / "pilot_candidates.json", {"candidates": candidates})
        write_json(path, manifest)
    missing = []
    for task in manifest["tasks"]:
        effective = extraction_config(manifest, task, config["test_prompt_count"] or config["min_test_prompts"])
        missing += common.preflight([task], effective, manifest["exclusion_tasks"])
    write_json(output / "preflight.json", {"missing": sorted(set(missing))})
    print(f"[witness-plan] frozen={len(manifest['witnesses'])} target_tasks={len(manifest['tasks'])} "
          f"selection={config.get('witness_selection_mode', 'top_positive')} "
          f"family_alpha={config['alpha']} per_witness_alpha={manifest['witnesses'][0]['alpha']:.6g} "
          f"step={config['primary_step']} smoke={args.smoke}", flush=True)
    for row in manifest["witnesses"]:
        print(f"  {row['witness_id']} {row['task']} {row['module']} alpha={row['alpha']:.6g} "
              f"pilot_score={row['pilot_ranking_score']:.6f}", flush=True)
    if args.plan_only:
        print(f"[witness-plan] missing_server_artifacts={len(set(missing))}; see {output / 'preflight.json'}", flush=True)
        return
    if missing:
        raise FileNotFoundError(f"Preflight failed before model loading. See {output / 'preflight.json'}")
    launch(manifest, output, list(dict.fromkeys(args.gpus)))


if __name__ == "__main__":
    main()
