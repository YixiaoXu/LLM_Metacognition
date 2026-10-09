#!/usr/bin/env python3
"""Resumable two-GPU reportability and persistent-generation follow-up."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from run_strict_followup_group import (check_prerequisites, path, read_json, run_logged,
                                       run_trajectory, sha, stop_children, write_json)

ROOT = Path(__file__).resolve().parents[1]


def frozen_pair(source: Path, dataset: str, target: str) -> dict:
    summary_path = source / dataset / target / "strict_chain_summary" / "strict_module_chain_summary.json"
    summary = read_json(summary_path)
    if summary.get("phase") != "final":
        raise ValueError(f"Strict chain is not final: {summary_path}")
    passing = [row for row in summary["rows"] if row.get("three_ring_pass")]
    if not summary["rows"]:
        raise ValueError(f"No frozen modules: {dataset}/{target}")
    return {"dataset": dataset, "target": target, "pair_root": str(path(summary["pair_root"])),
            "summary_sha256": sha(summary_path), "passing": passing,
            "report_anchor": passing[0] if passing else summary["rows"][0]}


def conditions(config: dict, source: Path, key: str) -> list[dict]:
    requested = config[key]
    if requested == "all_pairs":
        return [{"dataset": file.parents[2].name, "target": file.parents[1].name}
                for file in sorted(source.glob("*/*/strict_chain_summary/strict_module_chain_summary.json"))]
    if requested == "all_passing":
        output = []
        for file in sorted(source.glob("*/*/strict_chain_summary/strict_module_chain_summary.json")):
            summary = read_json(file)
            if summary.get("phase") != "final":
                raise ValueError(f"Strict chain is not final: {file}")
            output.extend({"dataset": file.parents[2].name, "target": file.parents[1].name,
                           "module_key": row["module_key"]}
                          for row in summary["rows"] if row.get("three_ring_pass"))
        return output
    if not isinstance(requested, list):
        raise ValueError(f"{key} must be a condition list, all_pairs, or all_passing")
    return requested


def make_plan(config: dict, source: Path) -> dict:
    report_conditions = conditions(config, source, "report_conditions")
    trajectory_conditions = conditions(config, source, "trajectory_conditions")
    if not report_conditions or not trajectory_conditions:
        raise ValueError("No report conditions or no three-ring trajectory conditions")
    pairs = {}
    for item in report_conditions + trajectory_conditions:
        key = (item["dataset"], item["target"])
        if key not in pairs:
            pairs[key] = frozen_pair(source, *key)
    reports = []
    for item in report_conditions:
        pair = pairs[(item["dataset"], item["target"])]
        first = pair["report_anchor"]
        module_dirs = [str(path(row["module_dir"])) for row in pair["passing"]] or [str(path(first["module_dir"]))]
        old = read_json(path(first["module_dir"]).parent.parent / "config.json")
        reports.append({"dataset": item["dataset"], "target": item["target"],
                        "module_dirs": module_dirs, "activation_dir": str(path(old["activation_dir"])),
                        "semantic_cache": str(path(old["external_semantic_activation_dir"]) / "layer_000.pt"),
                        "layer_i": int(old["layer_i"]),
                        "axis_csv": str(Path(pair["pair_root"]) / "continuous_modules" /
                                        first["module_key"] / "continuous_axis" /
                                        "continuous_axis_all_assignments.csv")})
    trajectories = []
    for item in trajectory_conditions:
        pair = pairs[(item["dataset"], item["target"])]
        matches = [row for row in pair["passing"] if row["module_key"] == item["module_key"]]
        if len(matches) != 1:
            raise ValueError(f"Module is not a unique frozen three-ring pass: {item}")
        row = matches[0]
        module = path(row["module_dir"])
        old = read_json(module.parent.parent / "config.json")
        trajectories.append({"dataset": item["dataset"], "target": item["target"],
                             "module_key": item["module_key"], "module_dir": str(module),
                             "decoupler_dir": str(module.parent.parent),
                             "activation_dir": str(path(old["activation_dir"])),
                             "old_output_dir": str(Path(pair["pair_root"]) / "continuous_modules" /
                                                   item["module_key"]),
                             "primary_behavior_construct": row["primary_behavior_construct"],
                             "primary_behavior_direction": row["aligned_behavior_direction"]})
    return {"schema_version": 1, "source_root": str(source), "config": config,
            "report_conditions": reports, "trajectory_conditions": trajectories,
            "source_summary_sha256": {f"{d}/{t}": pair["summary_sha256"]
                                      for (d, t), pair in pairs.items()}}


def missing_inputs(plan: dict) -> list[str]:
    missing = []
    for item in plan["report_conditions"]:
        files = [Path(item["axis_csv"]), Path(item["semantic_cache"]),
                 Path(item["activation_dir"]) / "expanded_generation_data.jsonl",
                 Path(item["activation_dir"]) / f"layer_{item['layer_i']:03d}.pt"]
        files += [Path(directory) / "module_summary.json" for directory in item["module_dirs"]]
        missing.extend(str(file) for file in files if not file.is_file())
    for item in plan["trajectory_conditions"]:
        missing.extend(check_prerequisites(item, "persistent_generation"))
    return sorted(set(missing))


def run_report(item: dict, config: dict, output: Path, gpu: str):
    directory = output / "direct_report" / item["dataset"] / item["target"]
    if config["report"].get("mode", "grouped") != "grouped":
        raise ValueError("Continuous random-neuron baseline is archived; use neuron_report_search_v1")
    expected = directory / "report_group_summary.json"
    if expected.is_file():
        print(f"[reuse] {expected}", flush=True)
        return
    options = config["report"]
    command = [sys.executable, "scripts/audit_neuron_report_groups.py", "--target", item["target"],
               "--activation-dir", item["activation_dir"],
               "--semantic-cache", item["semantic_cache"], "--axis-csv", item["axis_csv"],
               "--output-dir", str(directory), "--module-dirs", ",".join(item["module_dirs"])]
    for key, value in options.items():
        if key == "mode":
            continue
        command.extend(["--" + key.replace("_", "-"), str(value)])
    run_logged(command, output / "logs" / f"report__{item['dataset']}__{item['target']}.log",
               {**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
    if not expected.is_file():
        raise RuntimeError(f"Report task did not produce {expected}")


def run_jobs(name: str, items: list[dict], config: dict, output: Path, gpus: list[str]):
    pending = queue.Queue()
    for item in items:
        pending.put(item)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str):
        while True:
            try:
                item = pending.get_nowait()
            except queue.Empty:
                return
            key = f"{item['dataset']}/{item['target']}/{item.get('module_key', 'neuron_groups')}"
            print(f"[{name}] start {key} gpu={gpu}", flush=True)
            try:
                if name == "direct_report":
                    run_report(item, config, output, gpu)
                else:
                    condition = {**item, "trajectory_output_dir": str(
                        output / "persistent_generation" / item["dataset"] /
                        item["target"] / item["module_key"])}
                    if not reuse_trajectory(condition, config, output):
                        run_trajectory(condition, config, output, gpu)
                print(f"[{name}] complete {key} gpu={gpu}", flush=True)
            except Exception as exc:
                with lock:
                    failures.append(f"{key}: {exc}")
                print(f"[{name}] FAILED {key}: {exc}", flush=True)
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def reuse_trajectory(item: dict, config: dict, output: Path) -> bool:
    prior_root = config.get("reuse_trajectory_root")
    if not prior_root:
        return False
    prior_root = path(prior_root).resolve()
    prior_plan_file = prior_root / "frozen_plan.json"
    if not prior_plan_file.is_file() or read_json(prior_plan_file)["config"]["trajectory"] != config["trajectory"]:
        return False
    relative = Path("persistent_generation") / item["dataset"] / item["target"] / item["module_key"]
    prior = prior_root / relative
    expected = prior / "trajectory_analysis" / "construct_dose_response.csv"
    if not (expected.is_file() and (prior / "stage_complete.json").is_file()):
        return False
    destination = output / relative
    if destination.is_symlink():
        if destination.resolve() != prior.resolve():
            raise ValueError(f"Conflicting reused trajectory: {destination}")
    elif destination.exists():
        return False
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(prior, target_is_directory=True)
    print(f"[reuse] complete trajectory {relative} from {prior_root}", flush=True)
    return True


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    output = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[index]))
        output[index] = running
    return output


def summarize(plan: dict, output: Path):
    reports = []
    high = []
    for item in plan["report_conditions"]:
        file = (output / "direct_report" / item["dataset"] / item["target"] /
                "report_group_summary.json")
        if not file.is_file():
            reports.append({"dataset": item["dataset"], "target": item["target"], "status": "missing"})
            continue
        value = read_json(file)
        report_entry = {"dataset": item["dataset"], "target": item["target"],
                        "status": "complete", "path": str(file)}
        report_entry["result"] = value
        reports.append(report_entry)
        for row in value["selected"]:
            if row["group"] == "high_semantic" and row["decoded_report"]["auc_permutation_one_sided_p"] is not None:
                high.append(row)
    for row, p in zip(high, holm([row["decoded_report"]["auc_permutation_one_sided_p"] for row in high])):
        row["across_frozen_neurons_holm_p"] = p
    trajectories = []
    for item in plan["trajectory_conditions"]:
        file = (output / "persistent_generation" / item["dataset"] / item["target"] /
                item["module_key"] / "trajectory_analysis" / "construct_dose_response.json")
        if not file.is_file():
            trajectories.append({"dataset": item["dataset"], "target": item["target"],
                                 "module_key": item["module_key"], "status": "missing"})
            continue
        value = read_json(file)
        trajectories.append({"dataset": item["dataset"], "target": item["target"],
                             "module_key": item["module_key"], "status": "complete",
                             "path": str(file), "result": value.get("result", value)})
    write_json(output / "experiment_summary.json", {"report_conditions": reports,
                                                     "trajectory_conditions": trajectories,
                                                     "global_report_holm_family_size": len(high)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/report_trajectory_followup_v1.json")
    parser.add_argument("--source-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = read_json(path(args.config))
    if config["report"].get("mode", "grouped") != "grouped":
        raise ValueError("Continuous random-neuron baseline is archived; use neuron_report_search_v1")
    source = path(args.source_root or config["source_root"]).resolve()
    output = path(args.output_dir).resolve()
    plan = make_plan(config, source)
    frozen = output / "frozen_plan.json"
    if frozen.exists() and read_json(frozen) != plan:
        raise ValueError("Frozen follow-up plan changed; use a new RUN_ROOT")
    if not frozen.exists():
        write_json(frozen, plan)
    print(f"[followup-v2] reports={len(plan['report_conditions'])} trajectories="
          f"{len(plan['trajectory_conditions'])} gpus={args.gpus}", flush=True)
    for item in plan["trajectory_conditions"]:
        print(f"  {item['dataset']} {item['target']} {item['module_key']} "
              f"construct={item['primary_behavior_construct']}", flush=True)
    if args.plan_only:
        return
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU IDs must be distinct")
    missing = missing_inputs(plan)
    if missing:
        print(f"[preflight] {len(missing)} artifacts missing; affected conditions will fail individually: "
              f"{missing[:8]}", flush=True)
    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    lock_file = output / ".followup_v2.lock"
    failures = []
    with lock_file.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, items in (("direct_report", plan["report_conditions"]),
                            ("persistent_generation", plan["trajectory_conditions"])):
            print(f"\n[{time.strftime('%F %T')}] ===== {name} =====", flush=True)
            stage_failures = run_jobs(name, items, config, output, args.gpus)
            failures.extend(f"{name}: {failure}" for failure in stage_failures)
            print(f"[{name}] finished {len(items) - len(stage_failures)}/{len(items)}", flush=True)
    summarize(plan, output)
    write_json(output / "group_status.json", {"completed_at": time.strftime("%F %T"),
                                               "report_conditions": len(plan["report_conditions"]),
                                               "trajectory_conditions": len(plan["trajectory_conditions"]),
                                               "failures": failures})
    if failures:
        raise RuntimeError(f"Follow-up finished with {len(failures)} failed conditions; see group_status.json")
    write_json(output / "group_complete.json", {"completed_at": time.strftime("%F %T"),
                                                 "report_conditions": len(plan["report_conditions"]),
                                                 "trajectory_conditions": len(plan["trajectory_conditions"])})


if __name__ == "__main__":
    main()
