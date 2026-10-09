#!/usr/bin/env python3
"""Run frozen-module self-report pilots with one condition per GPU."""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import signal
import sys
import threading
from pathlib import Path

import _bootstrap  # noqa: F401
from run_strict_followup_group import (
    make_plan, path, read_json, run_logged, stop_children, write_json,
)


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/module_self_report_pilot_v1.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--only", action="append", default=[], help="dataset/target")
    parser.add_argument("--n-prompts", type=int, help="Small smoke-run override")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = read_json(path(args.config))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported module-report pilot config")
    source = path(config["source_root"]).resolve()
    output = path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = make_plan(config, source)
    if args.only:
        requested = set(args.only)
        plan["conditions"] = [item for item in plan["conditions"]
                              if f"{item['dataset']}/{item['target']}" in requested]
        if len(plan["conditions"]) != len(requested):
            raise ValueError(f"Unknown or repeated --only entries: {sorted(requested)}")
    if not plan["conditions"]:
        raise ValueError("No frozen module conditions selected")
    report_options = dict(config["report"])
    if args.n_prompts is not None:
        report_options["n_prompts"] = args.n_prompts
    frozen = {"source": plan, "report_options": report_options}
    manifest = output / "frozen_plan.json"
    if manifest.exists() and read_json(manifest) != frozen:
        raise ValueError("Frozen plan changed; use a new RUN_ROOT")
    if not manifest.exists():
        write_json(manifest, frozen)
    print(f"[module-report] planned={len(plan['conditions'])} gpus={args.gpus} "
          f"n_prompts={report_options['n_prompts']}", flush=True)
    for item in plan["conditions"]:
        print(f"  {item['dataset']}/{item['target']} {item['module_key']}", flush=True)
    if args.plan_only:
        return
    prior_root = path(config["reuse_results_root"]).resolve() if config.get("reuse_results_root") else None

    def reusable_result(item: dict) -> tuple[Path, Path] | None:
        if prior_root is None or args.n_prompts is not None:
            return None
        directory = prior_root / item["dataset"] / item["target"] / item["module_key"]
        summary = directory / "module_report_summary.json"
        frozen_report = directory / "frozen_report_plan.json"
        if not summary.is_file():
            return None
        if not frozen_report.is_file():
            raise ValueError(f"Prior module report has no frozen plan: {summary}")
        old = read_json(frozen_report)
        if old.get("report_method_version") != 2:
            return None
        expected = {"target": item["target"], "module_dir": item["module_dir"],
                    "activation_dir": item["activation_dir"],
                    "semantic_cache": item["semantic_cache"],
                    "axis_file": str(Path(item["old_output_dir"]) / "continuous_axis" /
                                     "continuous_axis.pt")}
        for key, value in expected.items():
            saved = old.get(key, old.get("options", {}).get(key))
            if saved != value:
                raise ValueError(f"Prior module report has a different {key}: {summary}")
        for key, value in report_options.items():
            if old["options"].get(key) != value:
                raise ValueError(f"Prior module report has a different {key}: {summary}")
        return summary, frozen_report

    def required_artifacts(item: dict) -> list[Path]:
        axis_dir = Path(item["old_output_dir"]) / "continuous_axis"
        old = read_json(Path(item["decoupler_dir"]) / "config.json")
        return [Path(item["module_dir"]) / "module_refiner.pt",
                Path(item["decoupler_dir"]) / "best_model.pt",
                Path(item["decoupler_dir"]) / "normalization.pt",
                axis_dir / "continuous_axis.pt",
                axis_dir / "continuous_axis_all_assignments.csv",
                Path(item["activation_dir"]) / "expanded_generation_data.jsonl",
                Path(item["activation_dir"]) / f"layer_{int(old['next_layer']):03d}.pt",
                Path(item["semantic_cache"])]

    if args.preflight_only:
        import torch
        from audit_module_self_report import frozen_axis_scalars

        for item in plan["conditions"]:
            name = f"{item['dataset']}/{item['target']}"
            prior = reusable_result(item)
            if prior is not None:
                print(f"[preflight] reuse prior {name}", flush=True)
                continue
            missing = [str(file) for file in required_artifacts(item) if not file.is_file()]
            if missing:
                raise FileNotFoundError(f"{name}: missing frozen artifacts: {missing}")
            axis_file = Path(item["old_output_dir"]) / "continuous_axis" / "continuous_axis.pt"
            frozen_direction, scores, _ = frozen_axis_scalars(
                torch.load(axis_file, map_location="cpu", weights_only=False))
            print(f"[preflight] {name} {item['module_key']} "
                  f"code_dim={frozen_direction.numel()} rows={len(scores)}", flush=True)
        print("[preflight] all frozen module-report inputs are ready", flush=True)
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
        while True:
            try:
                item = pending.get_nowait()
            except queue.Empty:
                return
            name = f"{item['dataset']}/{item['target']}"
            directory = output / item["dataset"] / item["target"] / item["module_key"]
            expected = directory / "module_report_summary.json"
            try:
                if expected.is_file():
                    print(f"[module-report] reuse {name}", flush=True)
                    continue
                prior = reusable_result(item)
                if prior is not None:
                    directory.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(prior[0], expected)
                    shutil.copy2(prior[1], directory / "reused_frozen_report_plan.json")
                    write_json(directory / "reuse_provenance.json", {"summary": str(prior[0])})
                    print(f"[module-report] reuse prior {name}: {prior[0]}", flush=True)
                    continue
                axis_dir = Path(item["old_output_dir"]) / "continuous_axis"
                missing = [str(file) for file in required_artifacts(item) if not file.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing frozen artifacts: {missing}")
                command = [sys.executable, "scripts/audit_module_self_report.py",
                           "--target", item["target"], "--module-dir", item["module_dir"],
                           "--axis-file", str(axis_dir / "continuous_axis.pt"),
                           "--activation-dir", item["activation_dir"],
                           "--semantic-cache", item["semantic_cache"],
                           "--output-dir", str(directory)]
                for key, value in report_options.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                print(f"[module-report] start {name} gpu={gpu}", flush=True)
                run_logged(command, output / "logs" /
                           f"report__{item['dataset']}__{item['target']}.log",
                           {**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
                if not expected.is_file():
                    raise RuntimeError(f"No module report summary produced: {expected}")
                print(f"[module-report] complete {name} gpu={gpu}", flush=True)
            except Exception as exc:
                with mutex:
                    failures.append({"condition": name, "error": str(exc)})
                print(f"[module-report] FAILED {name}: {exc}", flush=True)
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results = []
    for item in plan["conditions"]:
        file = output / item["dataset"] / item["target"] / item["module_key"] / "module_report_summary.json"
        if file.is_file():
            value = read_json(file)
            reused = output / item["dataset"] / item["target"] / item["module_key"] / "reuse_provenance.json"
            results.append({"dataset": item["dataset"], "target": item["target"],
                            "module_key": item["module_key"], "status": "complete",
                            "reused_from": read_json(reused)["summary"] if reused.is_file() else None,
                            "natural_report_auc": value["natural_report_auc"],
                            "report_method_version": value.get("report_method_version"),
                            "natural_report_balanced_accuracy": value["natural_report_balanced_accuracy"],
                            "semantic_only_label_auc": value["semantic_only_label_auc"],
                            "module_added_report_r2": value["module_added_report_fit"]["added_report_r2"],
                            "baseline_answer_option_mass_mean": value.get("baseline_answer_option_mass_mean"),
                            "baseline_answer_option_mass_below_1pct": value.get(
                                "baseline_answer_option_mass_below_1pct"),
                            "cache_hidden_relative_error_max": value.get(
                                "cache_hidden_relative_error_max"),
                            "cache_score_sign_agreement_rate": value.get(
                                "cache_score_sign_agreement_rate"),
                            "measurement_quality": value.get("measurement_quality"),
                            "n_delivered_pairs": value["n_delivered_pairs"],
                            "n_matched_random_pairs": value["n_matched_random_pairs"],
                            "paired_real": value["paired_real"],
                            "paired_effect": value["paired_real_minus_random"],
                            "result": str(file)})
        else:
            results.append({"dataset": item["dataset"], "target": item["target"],
                            "module_key": item["module_key"], "status": "failed"})
    adjusted = holm([row["paired_effect"]["p_one_sided"]
                     if row["status"] == "complete" and
                     row["paired_effect"]["p_one_sided"] is not None else 1.0
                     for row in results])
    for row, value in zip(results, adjusted):
        if row["status"] == "complete":
            row["paired_holm_p"] = value
    write_json(output / "module_report_study_summary.json", {
        "planned": len(results), "completed": sum(row["status"] == "complete" for row in results),
        "reused": sum(row.get("reused_from") is not None for row in results),
        "conditions": results, "failures": failures,
        "primary_family": "paired report log-odds contrast of real positive-vs-negative dose "
                          "minus matched random dose; Holm across frozen modules"})
    if failures:
        raise RuntimeError(f"{len(failures)} module-report conditions failed")


if __name__ == "__main__":
    main()
