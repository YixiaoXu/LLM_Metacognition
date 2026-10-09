"""Plan and multiplicity checks without model weights or a GPU."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_report_trajectory_followup import holm, make_plan  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_plan_keeps_multiple_modules_in_distinct_trajectory_outputs(tmp_path):
    source = tmp_path / "strict"
    pair = source / "mathqa" / "llama2_7b" / "pair"
    decoupler = pair / "jobs" / "support_32" / "decoupler_joint_v2"
    write(decoupler / "config.json", {"activation_dir": str(tmp_path / "activations"),
                                      "external_semantic_activation_dir": str(tmp_path / "semantic"),
                                      "layer_i": 16})
    rows = []
    for index in (1, 2):
        module = decoupler / "joint_residual_modules" / f"module_{index:02d}"
        write(module / "module_summary.json", {"module_member_neurons": [index]})
        rows.append({"module_key": f"rank_{index}_k32_m{index}", "module_dir": str(module),
                     "three_ring_pass": True, "primary_behavior_construct": "initial_uncertainty",
                     "aligned_behavior_direction": "positive"})
    write(source / "mathqa" / "llama2_7b" / "strict_chain_summary" /
          "strict_module_chain_summary.json", {"phase": "final", "pair_root": str(pair), "rows": rows})
    config = {"report_conditions": [{"dataset": "mathqa", "target": "llama2_7b"}],
              "trajectory_conditions": [{"dataset": "mathqa", "target": "llama2_7b",
                                         "module_key": row["module_key"]} for row in rows]}
    plan = make_plan(config, source)
    assert len(plan["report_conditions"][0]["module_dirs"]) == 2
    assert len(plan["trajectory_conditions"]) == 2
    assert plan["trajectory_conditions"][0]["old_output_dir"] != plan["trajectory_conditions"][1]["old_output_dir"]


def test_plan_rejects_missing_condition(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_plan({"report_conditions": [{"dataset": "missing", "target": "model"}],
                   "trajectory_conditions": [{"dataset": "missing", "target": "model",
                                              "module_key": "rank_1_k32_m0"}]}, tmp_path)


def test_plan_rejects_empty_trajectory_conditions(tmp_path):
    with pytest.raises(ValueError, match="No report conditions or no three-ring"):
        make_plan({"report_conditions": [{"dataset": "missing", "target": "model"}],
                   "trajectory_conditions": []}, tmp_path)


def test_holm_adjustment_is_monotone_in_p_order():
    assert holm([0.03, 0.001, 0.04]) == pytest.approx([0.06, 0.003, 0.06])
