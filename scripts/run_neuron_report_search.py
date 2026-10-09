#!/usr/bin/env python3
"""Schedule prompt-disjoint neuron report search on one worker per GPU."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import queue
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from run_strict_followup_group import STOPPED, path, read_json, run_logged, sha, stop_children, write_json


def compatible_report_options(current: dict, previous: dict) -> bool:
    # Completed legacy trials passed their original activation-consistency check.
    ignored = {"replay_mode"}
    return ({key: value for key, value in current.items() if key not in ignored} ==
            {key: value for key, value in previous.items() if key not in ignored})


def reusable_condition(item: dict, plan: dict, reuse_root: Path | None) -> Path | None:
    if reuse_root is None:
        return None
    directory = reuse_root / "direct_report" / item["dataset"] / item["target"]
    summary_file = directory / "report_search_summary.json"
    if not summary_file.is_file():
        return None
    previous = read_json(reuse_root / "frozen_plan.json")
    if not compatible_report_options(plan["config"]["report"], previous["config"]["report"]):
        raise ValueError(f"Completed report has different sample/search settings: {directory}")
    matched = [entry for entry in previous["conditions"]
               if (entry["dataset"], entry["target"]) == (item["dataset"], item["target"])]
    if len(matched) != 1 or matched[0] != item:
        raise ValueError(f"Completed report has different frozen upstream inputs: {directory}")
    value = read_json(summary_file)
    report = plan["config"]["report"]
    if (value["target"] != item["target"] or value["layer"] != item["layer_i"] or
            value["n_screened"] != report["candidate_neurons"] or
            value["n_rescreened"] != report["screen_b_candidates"] or
            len(value["winners"]) != report["finalists"] or
            len(value["matched_controls"]) != report["finalists"]):
        raise ValueError(f"Completed report summary is incomplete or incompatible: {directory}")
    return directory


def copy_completed_report(source: Path, directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing to mix an old completed report with partial new results: {directory}")
    shutil.copytree(source, directory, dirs_exist_ok=True)
    write_json(directory / "reuse_receipt.json", {
        "source": str(source), "summary_sha256": sha(source / "report_search_summary.json"),
        "replay_mode": read_json(source / "report_search_summary.json").get(
            "replay_mode", "legacy_full"),
        "reuse_rule": "complete checked condition with identical frozen upstream and search settings",
    })


def preflight_condition(item: dict, report: dict) -> None:
    activation_dir = Path(item["activation_dir"])
    expanded = activation_dir / "expanded_generation_data.jsonl"
    required = [Path(item["axis_csv"]), Path(item["semantic_cache"]), expanded]
    if report.get("activation_source", "cache") == "cache":
        required.append(activation_dir / f"layer_{item['layer_i']:03d}.pt")
    missing = [str(file) for file in required if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisite files: {missing}")
    from metacog.models.registry import get_model
    model_config = path(get_model(item["target"]).resolved_path) / "config.json"
    if not model_config.is_file():
        raise FileNotFoundError(f"Target model is unavailable: {model_config}")
    wanted = {"association_confirmatory", "prototype_discovery", "trajectory_confirmatory"}
    with Path(item["axis_csv"]).open(encoding="utf-8") as handle:
        roles = {row["id"]: row["analysis_role"] for row in csv.DictReader(handle)
                 if row["id"].endswith("::step000") and row["analysis_role"] in wanted}
    counts = {role: 0 for role in wanted}
    max_prefix = report.get("max_prefix_tokens", 3000)
    seen = set()
    with expanded.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sid = row["id"]
            if sid in roles and 0 < len(row.get("prompt_token_ids", [])) <= max_prefix:
                if sid in seen:
                    raise ValueError(f"Duplicate report prompt ID: {sid}")
                counts[roles[sid]] += 1
                seen.add(sid)
    required_counts = {
        "association_confirmatory": report["semantic_fit_rows"],
        "prototype_discovery": report["screen_a_pool"] + report["screen_b_rows"],
        "trajectory_confirmatory": report["evaluation_pool_rows"],
    }
    insufficient = {role: (counts[role], count) for role, count in required_counts.items()
                    if counts[role] < count}
    if insufficient:
        raise ValueError(f"Insufficient prompt-disjoint report pools: {insufficient}")
    print(f"[report-preflight] {item['dataset']}/{item['target']} pools={counts}", flush=True)


def make_plan(config: dict, source: Path, only: set[str]) -> dict:
    if config.get("schema_version") != 1 or config.get("conditions") != "all_pairs":
        raise ValueError("Expected schema_version=1 and conditions=all_pairs")
    entries = []
    excluded_targets = set(config.get("exclude_targets", []))
    files = sorted(source.glob("*/*/strict_chain_summary/strict_module_chain_summary.json"))
    for file in files:
        dataset, target = file.parents[2].name, file.parents[1].name
        if target in excluded_targets:
            continue
        if only and f"{dataset}/{target}" not in only:
            continue
        summary = read_json(file)
        if summary.get("phase") != "final" or not summary.get("rows"):
            raise ValueError(f"No final frozen modules: {file}")
        anchor = next((row for row in summary["rows"] if row.get("three_ring_pass")),
                      summary["rows"][0])
        module = path(anchor["module_dir"])
        old = read_json(module.parent.parent / "config.json")
        pair_root = path(summary["pair_root"])
        entries.append({
            "dataset": dataset, "target": target, "layer_i": int(old["layer_i"]),
            "activation_dir": str(path(old["activation_dir"])),
            "semantic_cache": str(path(old["external_semantic_activation_dir"]) / "layer_000.pt"),
            "axis_csv": str(pair_root / "continuous_modules" / anchor["module_key"] /
                            "continuous_axis" / "continuous_axis_all_assignments.csv"),
            "strict_summary_sha256": sha(file),
        })
    missing_pairs = only - {f"{item['dataset']}/{item['target']}" for item in entries}
    if missing_pairs:
        raise ValueError(f"Unknown pairs: {sorted(missing_pairs)}")
    if not entries:
        raise ValueError(f"No source conditions found in {source}")
    expected = config.get("expected_conditions")
    if expected is not None and not only and len(entries) != expected:
        raise ValueError(f"Expected {expected} model-domain conditions, found {len(entries)}")
    return {"schema_version": 1, "source_root": str(source), "config": config,
            "conditions": entries, "stages": ["screen_a", "screen_b", "heldout"]}


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def summarize(plan: dict, output: Path, failures: list[dict]) -> None:
    rows = []
    for item in plan["conditions"]:
        file = output / "direct_report" / item["dataset"] / item["target"] / "report_search_summary.json"
        if not file.exists():
            rows.append({"dataset": item["dataset"], "target": item["target"], "status": "missing"})
            continue
        value = read_json(file)
        primary = value["winners"][0]
        rows.append({"dataset": item["dataset"], "target": item["target"], "status": "complete",
                     "primary_neuron": primary["neuron"], "primary_auc": primary["report_auc"],
                     "primary_p": primary["report_auc_one_sided_p"],
                     "primary_semantic_r2": primary["semantic_r2_heldout"],
                     "primary_semantic_label_auc": primary["semantic_label_auc_heldout"],
                     "primary_conditional_report_r2": primary["conditional_report_gain"]["added_report_r2"],
                     "mean_winner_auc": value["mean_winner_report_auc"],
                     "mean_matched_control_auc": value["mean_control_report_auc"],
                     "replay_mode": value.get("replay_mode", "legacy_full"),
                     "activation_source": value.get("activation_source", "cache"),
                     "reused": (file.parent / "reuse_receipt.json").is_file(),
                     "path": str(file)})
    completed = [row for row in rows if row["status"] == "complete"]
    adjusted_all = holm([row["primary_p"] if row["status"] == "complete" else 1.0
                         for row in rows])
    for row, adjusted in zip(rows, adjusted_all):
        if row["status"] == "complete":
            row["primary_across_condition_holm_p"] = adjusted
    write_json(output / "report_search_study_summary.json", {
        "planned_conditions": len(rows), "completed_conditions": len(completed),
        "failures": failures, "conditions": rows,
        "primary_family": "one neuron selected before held-out testing per planned model-domain pair; "
                          "Holm across all planned conditions (missing tests assigned p=1)",
        "secondary_winners_are_descriptive": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neuron_report_search_v1.json")
    parser.add_argument("--source-root")
    parser.add_argument("--reuse-root", help="Reuse complete compatible conditions; empty disables reuse")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--only", action="append", default=[], help="dataset/target; repeatable")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--require-fresh-measurement", action="store_true")
    args = parser.parse_args()
    config = read_json(path(args.config))
    reuse_value = args.reuse_root if args.reuse_root is not None else config.get("reuse_completed_root")
    reuse_root = path(reuse_value).resolve() if reuse_value else None
    if args.require_fresh_measurement and (
            config["report"].get("activation_source") != "runtime_remeasured" or
            config["report"].get("replay_mode") != "generation_step_kv" or reuse_root is not None):
        raise ValueError("Fresh 24-condition run requires runtime_remeasured, generation_step_kv "
                         "and no imported report results")
    if reuse_root is not None and not (reuse_root / "frozen_plan.json").is_file():
        raise FileNotFoundError(f"Reuse root has no frozen_plan.json: {reuse_root}")
    source = path(args.source_root or config["source_root"]).resolve()
    output = path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = make_plan(config, source, set(args.only))
    frozen = output / "frozen_plan.json"
    if frozen.exists() and read_json(frozen) != plan:
        raise ValueError("Frozen search plan changed; use a new RUN_ROOT")
    if not frozen.exists():
        write_json(frozen, plan)
    print(f"[report-search] conditions={len(plan['conditions'])} gpus={args.gpus} "
          f"persistent_generation=disabled", flush=True)
    for item in plan["conditions"]:
        print(f"  {item['dataset']}/{item['target']} layer={item['layer_i']}", flush=True)
    reusable = {}
    preflight_errors = []
    for item in plan["conditions"]:
        name = f"{item['dataset']}/{item['target']}"
        expected = output / "direct_report" / item["dataset"] / item["target"] / "report_search_summary.json"
        if expected.is_file():
            continue
        try:
            previous = reusable_condition(item, plan, reuse_root)
            if previous is not None:
                reusable[name] = previous
            else:
                preflight_condition(item, config["report"])
        except Exception as exc:
            preflight_errors.append({"condition": name, "error": str(exc)})
    counts = {"planned": len(plan["conditions"]), "reuse_completed": len(reusable),
              "already_completed": sum((output / "direct_report" / item["dataset"] / item["target"] /
                                        "report_search_summary.json").is_file() for item in plan["conditions"])}
    counts["pending"] = counts["planned"] - counts["reuse_completed"] - counts["already_completed"]
    write_json(output / "preflight_status.json", {**counts, "errors": preflight_errors})
    print(f"[report-search] plan={counts['planned']} reused={counts['reuse_completed']} "
          f"done={counts['already_completed']} pending={counts['pending']}", flush=True)
    if preflight_errors:
        raise ValueError(f"{len(preflight_errors)} preflight failures; see {output / 'preflight_status.json'}")
    if args.plan_only:
        return
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU IDs must be distinct")
    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    pending = queue.Queue()
    for item in plan["conditions"]:
        pending.put(item)
    failures = []
    mutex = threading.Lock()
    def worker(gpu: str) -> None:
        while not STOPPED.is_set():
            try:
                item = pending.get_nowait()
            except queue.Empty:
                return
            name = f"{item['dataset']}/{item['target']}"
            directory = output / "direct_report" / item["dataset"] / item["target"]
            expected = directory / "report_search_summary.json"
            try:
                if expected.exists():
                    print(f"[report-search] reuse {name}", flush=True)
                    continue
                if name in reusable:
                    copy_completed_report(reusable[name], directory)
                    print(f"[report-search] imported completed {name}", flush=True)
                    continue
                required = [Path(item["axis_csv"]), Path(item["semantic_cache"]),
                            Path(item["activation_dir"]) / "expanded_generation_data.jsonl"]
                if config["report"].get("activation_source", "cache") == "cache":
                    required.append(Path(item["activation_dir"]) / f"layer_{item['layer_i']:03d}.pt")
                missing = [str(file) for file in required if not file.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing prerequisite files: {missing}")
                print(f"[report-search] start {name} gpu={gpu}", flush=True)
                command = [sys.executable, "scripts/audit_neuron_report_search.py",
                           "--target", item["target"], "--layer-id", str(item["layer_i"]),
                           "--activation-dir", item["activation_dir"],
                           "--semantic-cache", item["semantic_cache"],
                           "--axis-csv", item["axis_csv"], "--output-dir", str(directory)]
                for key, value in config["report"].items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                run_logged(command, output / "logs" /
                           f"report__{item['dataset']}__{item['target']}.log",
                           {**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
                if not expected.exists():
                    raise RuntimeError(f"No report summary produced: {expected}")
                print(f"[report-search] complete {name} gpu={gpu}", flush=True)
            except Exception as exc:
                with mutex:
                    failures.append({"condition": name, "error": str(exc)})
                print(f"[report-search] FAILED {name}: {exc}", flush=True)
            finally:
                pending.task_done()
    lock = output / ".report_search.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    summarize(plan, output, failures)
    write_json(output / "group_status.json", {"completed_at": time.strftime("%F %T"),
                                                "failures": failures})
    if STOPPED.is_set():
        raise RuntimeError("Report search stopped; completed conditions are preserved")
    if failures:
        raise RuntimeError(f"{len(failures)} report conditions failed; see group_status.json")
    write_json(output / "group_complete.json", {"completed_at": time.strftime("%F %T"),
                                                  "conditions": len(plan["conditions"])})


if __name__ == "__main__":
    main()
