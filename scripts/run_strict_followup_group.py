#!/usr/bin/env python3
"""Three-stage, resumable follow-up of frozen strict-chain modules.

Global stage barrier: direct report -> critical persistent generation ->
fresh-prompt independent semantic audit. One condition owns one GPU slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from metacog.models.registry import get_model

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("direct_report", "persistent_generation", "independent_semantic_audit")
ACTIVE = set()
ACTIVE_LOCK = threading.Lock()
STOPPED = threading.Event()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def make_plan(config: dict, source: Path) -> dict:
    conditions = []
    for item in config["conditions"]:
        dataset, target, key = item["dataset"], item["target"], item["module_key"]
        summary_path = source / dataset / target / "strict_chain_summary" / "strict_module_chain_summary.json"
        summary = read_json(summary_path)
        if summary.get("phase") != "final":
            raise ValueError(f"Strict-chain result is not final: {summary_path}")
        matched = [row for row in summary["rows"] if row["module_key"] == key]
        if len(matched) != 1 or not matched[0]["three_ring_pass"]:
            raise ValueError(f"Only one previously frozen three-ring module may be selected: {item}")
        row = matched[0]
        module_dir = path(row["module_dir"])
        dec_dir = module_dir.parent.parent
        old = read_json(dec_dir / "config.json")
        if old.get("second_stage") != "main_direct" or old.get("module_direction_mode", "learned") != "learned":
            raise ValueError(f"Expected a frozen learned main_direct module: {dec_dir}")
        pair_root = path(summary["pair_root"])
        output_dir = pair_root / "continuous_modules" / key
        ensemble_dir = path(old["external_semantic_activation_dir"])
        conditions.append({"dataset": dataset, "target": target, "module_key": key,
                           "module_dir": str(module_dir), "decoupler_dir": str(dec_dir),
                           "activation_dir": str(path(old["activation_dir"])),
                           "semantic_cache": str(ensemble_dir / "layer_000.pt"),
                           "old_output_dir": str(output_dir),
                           "summary_sha256": sha(summary_path),
                           "module_summary_sha256": sha(module_dir / "module_summary.json"),
                           "primary_behavior_construct": row["primary_behavior_construct"],
                           "primary_behavior_direction": row["aligned_behavior_direction"]})
    if len({(c["dataset"], c["target"]) for c in conditions}) != len(conditions):
        raise ValueError("Use at most one frozen module per target/domain in this focused group")
    return {"schema_version": 1, "source_root": str(source), "conditions": conditions,
            "stage_order": list(STAGES), "config": config}


def check_prerequisites(condition: dict, stage: str) -> list[str]:
    old = Path(condition["old_output_dir"])
    common = [Path(condition["module_dir"]) / "soft_residual_intervention_targets.pt",
              Path(condition["activation_dir"]) / "expanded_generation_data.jsonl",
              old / "continuous_axis" / "continuous_axis_all_assignments.csv"]
    if stage == "direct_report":
        config = read_json(Path(condition["decoupler_dir"]) / "config.json")
        common += [Path(condition["activation_dir"]) / f"layer_{int(config['layer_i']):03d}.pt",
                   Path(condition["semantic_cache"])]
    if stage == "persistent_generation":
        common += [old / "continuous_axis" / "continuous_axis.pt",
                   old / "continuous_axis" / "continuous_axis_all_features.pt",
                   old / "prototype_discovery" / "next_token_logit_prototypes.pt",
                   old / "causal_screen_a" / "continuous_dose_response.csv",
                   old / "causal_screen_b" / "continuous_dose_response.csv"]
    return [str(p) for p in common if not p.is_file()]


def stop_children(_signum, _frame):
    STOPPED.set()
    with ACTIVE_LOCK:
        children = list(ACTIVE)
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    print("[followup] stop requested; only this group's active subprocesses were signalled", flush=True)


def run_logged(command: list[str], log: Path, env: dict | None = None):
    if STOPPED.is_set():
        raise RuntimeError("Follow-up launch was stopped")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{time.strftime('%F %T')}] $ {' '.join(command)}\n")
        handle.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        with ACTIVE_LOCK:
            ACTIVE.add(child)
        if STOPPED.is_set() and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        try:
            code = child.wait()
            if code:
                with log.open("rb") as error_log:
                    error_log.seek(max(0, error_log.seek(0, os.SEEK_END) - 8192))
                    tail = error_log.read().decode("utf-8", errors="replace").splitlines()[-12:]
                raise RuntimeError(f"Exit {code}; log={log}\n" + "\n".join(tail))
        finally:
            with ACTIVE_LOCK:
                ACTIVE.discard(child)


def run_direct_report(condition: dict, config: dict, output: Path, gpu: str):
    directory = output / "direct_report" / condition["dataset"] / condition["target"]
    expected = directory / "report_summary.json"
    receipt = directory / "stage_complete.json"
    if (expected.is_file() and receipt.is_file() and
            read_json(receipt).get("report_statistics_version") == 2):
        print(f"[reuse] {expected}", flush=True)
        return
    missing = check_prerequisites(condition, "direct_report")
    if missing:
        raise FileNotFoundError(f"Direct-report prerequisites missing: {missing}")
    spec = get_model(condition["target"])
    if not (Path(spec.resolved_path) / "config.json").is_file():
        raise FileNotFoundError(f"Target model unavailable: {spec.resolved_path}")
    old = Path(condition["old_output_dir"])
    opts = config["report"]
    command = [sys.executable, "scripts/audit_internal_activation_report.py",
               "--target", condition["target"], "--module-dir", condition["module_dir"],
               "--activation-dir", condition["activation_dir"],
               "--semantic-cache", condition["semantic_cache"],
               "--axis-csv", str(old / "continuous_axis" / "continuous_axis_all_assignments.csv"),
               "--output-dir", str(directory), "--role", opts["role"],
               "--max-rows", str(opts["max_rows"]), "--flip-rows", str(opts["flip_rows"]),
               "--flip-norm-budget", str(opts["flip_norm_budget"]), "--seed", str(opts["seed"])]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    run_logged(command, output / "logs" / f"direct_report__{condition['dataset']}__{condition['target']}.log", env)
    if not expected.is_file():
        raise RuntimeError(f"Report audit produced no summary: {expected}")
    write_json(receipt, {"stage": "direct_report", "module_key": condition["module_key"],
                         "report_statistics_version": 2})


def run_trajectory(condition: dict, config: dict, output: Path, gpu: str):
    directory = Path(condition.get("trajectory_output_dir") or
                     output / "persistent_generation" / condition["dataset"] / condition["target"])
    expected = directory / "trajectory_analysis" / "construct_dose_response.csv"
    receipt = directory / "stage_complete.json"
    if expected.is_file() and receipt.is_file():
        print(f"[reuse] {expected}", flush=True)
        return
    missing = check_prerequisites(condition, "persistent_generation")
    if missing:
        raise FileNotFoundError(f"Persistent-generation prerequisites missing: {missing}")
    old = Path(condition["old_output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("continuous_axis", "prototype_discovery", "causal_screen_a", "causal_screen_b"):
        link = directory / name
        target = old / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"Frozen upstream link conflicts with existing output: {link}")
        link.symlink_to(target.resolve(), target_is_directory=True)
    dataset_config = f"configs/{condition['dataset']}_strict_chain_main_v2.env"
    config_script = ("set -a; source \"$1\"; set +a; "
                     "export DATA_PATH=\"$FOLLOWUP_DATA_PATH\"; "
                     "export RUN_PERSISTENT_TRAJECTORY=1 CONTINUOUS_STAGE=trajectory FORCE=0 "
                     "FORCE_AXIS=0 FORCE_INTERVENTION=0; "
                     "export CRITICAL_MAX_SAMPLES=\"$FOLLOWUP_CRITICAL_MAX_SAMPLES\" "
                     "CONTINUOUS_DOSES=\"$FOLLOWUP_DOSES\"; "
                     "exec bash scripts/run_continuous_module.sh")
    env = {**os.environ, "GPU": gpu, "MODULE_DIR": condition["module_dir"],
           "OUTPUT_DIR": str(directory), "MODEL_PATH": get_model(condition["target"]).resolved_path,
           "FOLLOWUP_DATA_PATH": str(Path(condition["activation_dir"]) / "expanded_generation_data.jsonl"),
           "ACTIVATION_DIR": condition["activation_dir"], "DECOUPLER_DIR": condition["decoupler_dir"],
           "PRIMARY_BEHAVIOR_CONSTRUCT": condition["primary_behavior_construct"],
           "PRIMARY_BEHAVIOR_DIRECTION": condition["primary_behavior_direction"],
           "FOLLOWUP_CRITICAL_MAX_SAMPLES": str(config["trajectory"]["critical_max_samples"]),
           "FOLLOWUP_DOSES": config["trajectory"]["doses"]}
    run_logged(["bash", "-c", config_script, "strict-followup", dataset_config],
               output / "logs" /
               f"persistent_generation__{condition['dataset']}__{condition['target']}__{condition['module_key']}.log",
               env)
    if not expected.is_file():
        raise RuntimeError(f"Trajectory analysis missing: {expected}")
    write_json(receipt, {"stage": "persistent_generation", "module_key": condition["module_key"]})


def run_parallel(stage: str, plan: dict, output: Path, gpus: list[str]):
    work = queue.Queue()
    for condition in plan["conditions"]:
        work.put(condition)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str):
        while not STOPPED.is_set():
            try:
                condition = work.get_nowait()
            except queue.Empty:
                return
            label = f"{condition['dataset']}__{condition['target']}"
            print(f"[{stage}] start {label} gpu={gpu}", flush=True)
            try:
                if stage == "direct_report":
                    run_direct_report(condition, plan["config"], output, gpu)
                else:
                    run_trajectory(condition, plan["config"], output, gpu)
                print(f"[{stage}] complete {label} gpu={gpu}", flush=True)
            except Exception as exc:
                with lock:
                    failures.append(f"{label}: {exc}")
                print(f"[{stage}] FAILED {label}: {exc}", flush=True)
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"{stage}: {len(failures)} condition(s) failed: {failures}")


def semantic_audit_command(plan_path: Path, plan: dict, output: Path, gpus: list[str]) -> list[str]:
    opts = plan["config"]["semantic_audit"]
    audit_dir = output / "independent_semantic_audit"
    base = read_json(path(opts["base_config"]))
    base.update({"source_root": plan["source_root"], "fresh_prompt_count": opts["fresh_prompt_count"],
                 "minimum_fresh_prompts": opts["minimum_fresh_prompts"],
                 "record_steps": opts["record_steps"], "bound_steps": opts["record_steps"],
                 "independent_reference_count": opts["independent_reference_count"],
                 "bootstrap_samples": opts["bootstrap_samples"], "maximum_modules": 1})
    base["probes"]["epochs"] = opts["probe_epochs"]
    audit_config = output / "semantic_audit_config.json"
    if audit_config.exists() and read_json(audit_config) != base:
        raise ValueError("Semantic-audit config changed in an existing run root")
    write_json(audit_config, base)
    command = [sys.executable, "scripts/run_semantic_leakage_audit.py",
               "--config", str(audit_config), "--source-root", plan["source_root"],
               "--selected-plan", str(plan_path), "--output-dir", str(audit_dir),
               "--gpus", *gpus]
    return command


def preflight_semantic_audit(plan_path: Path, plan: dict, output: Path, gpus: list[str]):
    command = semantic_audit_command(plan_path, plan, output, gpus)
    run_logged(command + ["--plan-only"], output / "logs" / "independent_semantic_audit_preflight.log")
    manifest = read_json(output / "independent_semantic_audit" / "audit_manifest.json")
    missing = manifest["preflight_missing"]
    if missing:
        raise FileNotFoundError(f"Independent semantic audit has {len(missing)} missing server inputs: {missing[:10]}")
    from run_semantic_leakage_audit import fresh_prompts

    for task in manifest["tasks"]:
        task_output = output / "independent_semantic_audit" / task["name"]
        task_output.mkdir(parents=True, exist_ok=True)
        selected = fresh_prompts(task, manifest["exclusion_tasks"], manifest["config"], task_output)
        print(f"[preflight] frozen fresh prompts: {task['name']} {selected}", flush=True)


def run_semantic_audit(plan_path: Path, plan: dict, output: Path, gpus: list[str]):
    audit_dir = output / "independent_semantic_audit"
    expected = audit_dir / "continuous_information.csv"
    status_path = audit_dir / "status_summary.json"
    status = read_json(status_path).get("tasks", []) if status_path.is_file() else []
    if expected.is_file() and len(status) == len(plan["conditions"]) and all(
            row.get("status") == "complete" for row in status):
        print(f"[reuse] {expected}", flush=True)
        return
    command = semantic_audit_command(plan_path, plan, output, gpus)
    run_logged(command, output / "logs" / "independent_semantic_audit.log")
    if not expected.is_file():
        raise RuntimeError(f"Independent semantic audit produced no aggregate: {expected}")


def write_group_summary(plan: dict, output: Path):
    rows = []
    for condition in plan["conditions"]:
        dataset, target, key = condition["dataset"], condition["target"], condition["module_key"]
        report = output / "direct_report" / dataset / target / "report_summary.json"
        trajectory = output / "persistent_generation" / dataset / target / "trajectory_analysis" / "construct_dose_response.json"
        audit = output / "independent_semantic_audit" / f"strict_{dataset}_{target}" / "analysis" / key / "audit_result.json"
        for expected in (report, trajectory, audit):
            if not expected.is_file():
                raise FileNotFoundError(f"Group result incomplete: {expected}")
        rows.append({"dataset": dataset, "target": target, "module_key": key,
                     "primary_construct": condition["primary_behavior_construct"],
                     "direct_report": read_json(report),
                     "persistent_construct": read_json(trajectory)["result"],
                     "independent_semantic_audit": read_json(audit)["continuous_audits"],
                     "artifact_paths": {"report": str(report), "trajectory": str(trajectory), "audit": str(audit)}})
    write_json(output / "group_results.json", {"stage_order": list(STAGES), "conditions": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/strict_followup_group_v1.json")
    parser.add_argument("--source-root", help="Override strict-main run root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config_path = path(args.config)
    config = read_json(config_path)
    source = path(args.source_root or config["source_root"]).resolve()
    output = path(args.output_dir).resolve()
    plan = make_plan(config, source)
    plan_path = output / "frozen_plan.json"
    if plan_path.exists():
        old = read_json(plan_path)
        if old != plan:
            raise ValueError("The frozen plan differs from this run root; use another output dir")
    else:
        write_json(plan_path, plan)
    print(f"[followup] source={source} conditions={len(plan['conditions'])} gpus={args.gpus}", flush=True)
    for condition in plan["conditions"]:
        print(f"  {condition['dataset']} {condition['target']} {condition['module_key']} "
              f"construct={condition['primary_behavior_construct']}", flush=True)
        for stage in STAGES[:2]:
            missing = check_prerequisites(condition, stage)
            if missing:
                print(f"  [preflight:{stage}] {len(missing)} artifacts absent (see server sync): {missing[:3]}", flush=True)
    if args.plan_only:
        return
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU IDs must be distinct")
    missing = [item for condition in plan["conditions"] for stage in STAGES[:2]
               for item in check_prerequisites(condition, stage)]
    if missing:
        raise FileNotFoundError(f"Follow-up cannot start: {len(set(missing))} upstream artifacts absent: {sorted(set(missing))[:10]}")
    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    lock_path = output / ".followup.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight_semantic_audit(plan_path, plan, output, args.gpus)
        for stage in STAGES:
            print(f"\n[{time.strftime('%F %T')}] ===== {stage} =====", flush=True)
            if stage == "independent_semantic_audit":
                run_semantic_audit(plan_path, plan, output, args.gpus)
            else:
                run_parallel(stage, plan, output, args.gpus)
            print(f"[{time.strftime('%F %T')}] ===== {stage} COMPLETE =====", flush=True)
    write_group_summary(plan, output)
    write_json(output / "group_complete.json", {"stages": list(STAGES), "conditions": len(plan["conditions"]),
                                                       "completed_at": time.strftime("%F %T")})


if __name__ == "__main__":
    main()
