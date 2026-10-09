"""Plan and adapter checks that require no model weights or GPUs."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_strict_followup_group import STAGES, make_plan  # noqa: E402
from run_semantic_leakage_audit import strict_tasks_from_plan  # noqa: E402
import run_strict_followup_group as followup  # noqa: E402
import run_semantic_leakage_audit as leakage  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path):
    source = tmp_path / "strict"
    condition = source / "mathqa" / "llama2_7b"
    pair = condition / "pair"
    decoupler = pair / "jobs" / "support_32" / "decoupler_joint_v2"
    module = decoupler / "joint_residual_modules" / "module_03"
    activation = tmp_path / "activations" / "target"
    ensemble = condition / "semantic_ensemble_cache"
    write(module / "module_summary.json", {"module_member_neurons": [1], "module_loading": [1.0]})
    write(decoupler / "config.json", {"activation_dir": str(activation),
                                      "external_semantic_activation_dir": str(ensemble),
                                      "second_stage": "main_direct", "module_direction_mode": "learned"})
    write(ensemble / "manifest.json", {"sources": [
        {"label": "qwen3_8b", "layer": 18, "activation_dir": "activations/qwen3_8b"},
        {"label": "llama31_8b", "layer": 16, "activation_dir": "activations/llama31_8b"}]})
    summary = condition / "strict_chain_summary" / "strict_module_chain_summary.json"
    write(summary, {"phase": "final", "pair_root": str(pair), "rows": [{
        "module_key": "rank_1_k32_m3", "module_dir": str(module),
        "primary_behavior_construct": "initial_uncertainty", "aligned_behavior_direction": "positive",
        "three_ring_pass": True}]})
    config = {"conditions": [{"dataset": "mathqa", "target": "llama2_7b",
                              "module_key": "rank_1_k32_m3"}]}
    return source, config, summary


def test_stage_order_and_frozen_multi_reference_adapter(tmp_path):
    source, config, _ = fixture(tmp_path)
    plan = make_plan(config, source)
    assert STAGES == ("direct_report", "persistent_generation", "independent_semantic_audit")
    assert plan["conditions"][0]["primary_behavior_construct"] == "initial_uncertainty"
    tasks, exclusions = strict_tasks_from_plan(plan)
    assert len(tasks) == len(exclusions) == 1
    assert [r["model"] for r in tasks[0]["training_references"]] == ["qwen3_8b", "llama31_8b"]
    assert tasks[0]["modules"][0]["name"] == "rank_1_k32_m3"


def test_plan_rejects_nonpassing_or_unfinished_source(tmp_path):
    source, config, summary = fixture(tmp_path)
    data = json.loads(summary.read_text())
    data["rows"][0]["three_ring_pass"] = False
    write(summary, data)
    with pytest.raises(ValueError, match="three-ring"):
        make_plan(config, source)
    data["rows"][0]["three_ring_pass"] = True
    data["phase"] = "screen"
    write(summary, data)
    with pytest.raises(ValueError, match="not final"):
        make_plan(config, source)


def test_preflight_creates_fresh_prompt_directory_before_write(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "semantic_audit_command", lambda *args: ["audit"])

    def fake_run(_command, _log):
        write(tmp_path / "independent_semantic_audit" / "audit_manifest.json", {
            "preflight_missing": [], "tasks": [{"name": "strict_mathqa_llama2_7b"}],
            "exclusion_tasks": [], "config": {},
        })

    def fake_fresh(_task, _exclusions, _config, output):
        assert output.is_dir()
        return output / "fresh_prompts.jsonl"

    monkeypatch.setattr(followup, "run_logged", fake_run)
    monkeypatch.setattr(leakage, "fresh_prompts", fake_fresh)
    followup.preflight_semantic_audit(tmp_path / "frozen_plan.json", {}, tmp_path, ["0"])


def test_failed_child_reports_log_tail(tmp_path):
    log = tmp_path / "logs" / "failed.log"
    with pytest.raises(RuntimeError, match="direct-report marker") as error:
        followup.run_logged([sys.executable, "-c", "raise RuntimeError('direct-report marker')"], log)
    assert str(log) in str(error.value)
