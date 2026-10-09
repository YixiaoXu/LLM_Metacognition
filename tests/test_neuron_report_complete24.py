"""Check report-family coverage, compatible reuse and preflight without model weights."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_neuron_report_search import (  # noqa: E402
    compatible_report_options, copy_completed_report, make_plan,
    preflight_condition, reusable_condition, summarize,
)


def write(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value))


@pytest.fixture
def full_plan(tmp_path):
    config = json.loads((ROOT / "configs/neuron_report_search_complete24_v2.json").read_text())
    source = tmp_path / "main"
    for dataset in ("ultrachat", "beavertails", "mathqa"):
        for target in ("llama2_7b", "llama31_8b", "llama32_3b", "deepseek_llama8b",
                       "qwen25_7b", "qwen3_4b", "qwen3_8b", "deepseek_qwen7b"):
            pair = source / dataset / target / "pair"
            dec = pair / "jobs/support_32/decoupler_joint_v2"
            module = dec / "joint_residual_modules/module_00"
            write(dec / "config.json", {"layer_i": 14, "activation_dir": str(tmp_path / target),
                                        "external_semantic_activation_dir": str(pair / "semantic")})
            write(source / dataset / target / "strict_chain_summary/strict_module_chain_summary.json",
                  {"phase": "final", "pair_root": str(pair), "rows": [
                      {"module_dir": str(module), "module_key": "rank_1_k32_m0", "three_ring_pass": True}]})
    return make_plan(config, source, set())


def summary_for(item, options):
    def neuron(index):
        return {"neuron": index, "report_auc": 0.7, "report_auc_one_sided_p": 0.001,
                "semantic_r2_heldout": 0.8, "semantic_label_auc_heldout": 0.9,
                "conditional_report_gain": {"added_report_r2": 0.01}}
    return {"target": item["target"], "layer": item["layer_i"],
            "n_screened": options["candidate_neurons"],
            "n_rescreened": options["screen_b_candidates"],
            "winners": [neuron(i) for i in range(options["finalists"])],
            "matched_controls": [neuron(i+10) for i in range(options["finalists"])],
            "mean_winner_report_auc": 0.7, "mean_control_report_auc": 0.5}


def test_full_plan_includes_eight_models_in_each_domain(full_plan):
    assert len(full_plan["conditions"]) == 24
    assert sum(item["target"] == "deepseek_qwen7b" for item in full_plan["conditions"]) == 3


def test_fresh_24_config_cannot_import_cached_report_conditions(full_plan, tmp_path):
    config = json.loads((ROOT / "configs/neuron_report_search_fresh24_v3.json").read_text())
    fresh = make_plan(config, Path(full_plan["source_root"]), set())
    assert len(fresh["conditions"]) == 24
    assert config["reuse_completed_root"] is None
    assert config["report"]["activation_source"] == "runtime_remeasured"
    assert not compatible_report_options(config["report"], full_plan["config"]["report"])
    assert reusable_condition(fresh["conditions"][0], fresh, None) is None


def test_missing_main_condition_is_detected(full_plan):
    source = Path(full_plan["source_root"])
    (source / "mathqa/deepseek_qwen7b/strict_chain_summary/strict_module_chain_summary.json").unlink()
    with pytest.raises(ValueError, match="Expected 24.*found 23"):
        make_plan(full_plan["config"], source, set())


def test_report_sample_settings_must_match_before_reuse(full_plan):
    current = full_plan["config"]["report"]
    legacy = {key: value for key, value in current.items() if key != "replay_mode"}
    assert compatible_report_options(current, legacy)
    assert not compatible_report_options(current, {**legacy, "confirm_rows": 80})


def test_only_complete_compatible_conditions_are_reused(tmp_path, full_plan):
    old = tmp_path / "old"
    item = full_plan["conditions"][0]
    directory = old / "direct_report" / item["dataset"] / item["target"]
    assert reusable_condition(item, full_plan, old) is None
    write(old / "frozen_plan.json", full_plan)
    write(directory / "report_search_summary.json", summary_for(item, full_plan["config"]["report"]))
    assert reusable_condition(item, full_plan, old) == directory
    with pytest.raises(ValueError, match="different frozen upstream"):
        reusable_condition({**item, "strict_summary_sha256": "changed"}, full_plan, old)
    dest = tmp_path / "new/direct_report" / item["dataset"] / item["target"]
    source_bytes = (directory / "report_search_summary.json").read_bytes()
    copy_completed_report(directory, dest)
    assert (dest / "report_search_summary.json").read_bytes() == source_bytes
    assert (directory / "report_search_summary.json").read_bytes() == source_bytes
    assert json.loads((dest / "reuse_receipt.json").read_text())["replay_mode"] == "legacy_full"


def test_summary_recalculates_holm_over_all_24_conditions(tmp_path, full_plan):
    for item in full_plan["conditions"][:13]:
        write(tmp_path / "direct_report" / item["dataset"] / item["target"] /
              "report_search_summary.json", summary_for(item, full_plan["config"]["report"]))
    summarize(full_plan, tmp_path, [])
    value = json.loads((tmp_path / "report_search_study_summary.json").read_text())
    assert value["planned_conditions"] == 24
    assert value["completed_conditions"] == 13
    assert value["conditions"][0]["primary_across_condition_holm_p"] == pytest.approx(0.024)


def test_preflight_detects_too_few_usable_prompts(tmp_path, monkeypatch):
    import metacog.models.registry as registry
    model = tmp_path / "model"
    write(model / "config.json", {})
    monkeypatch.setattr(registry, "get_model", lambda _: SimpleNamespace(resolved_path=str(model)))
    activation = tmp_path / "activation"
    write(activation / "layer_014.pt", {})
    semantic = tmp_path / "semantic.pt"
    write(semantic, {})
    axis = tmp_path / "axis.csv"
    ids = [f"p{i}::step000" for i in range(3)]
    roles = ["association_confirmatory", "prototype_discovery", "trajectory_confirmatory"]
    axis.write_text("id,analysis_role\n" + "".join(f"{sid},{role}\n" for sid, role in zip(ids, roles)))
    expanded = activation / "expanded_generation_data.jsonl"
    expanded.write_text("".join(json.dumps({"id": sid, "prompt_token_ids": [1, 2]}) + "\n" for sid in ids))
    item = {"dataset": "mathqa", "target": "test", "activation_dir": str(activation),
            "semantic_cache": str(semantic), "axis_csv": str(axis), "layer_i": 14}
    with pytest.raises(ValueError, match="Insufficient prompt-disjoint report pools"):
        preflight_condition(item, {"semantic_fit_rows": 2, "screen_a_pool": 1,
                                  "screen_b_rows": 1, "evaluation_pool_rows": 2})


def test_fresh_preflight_does_not_require_old_target_activation_tensor(tmp_path, monkeypatch):
    import metacog.models.registry as registry
    model = tmp_path / "model"
    write(model / "config.json", {})
    monkeypatch.setattr(registry, "get_model", lambda _: SimpleNamespace(resolved_path=str(model)))
    activation = tmp_path / "activation"
    semantic = tmp_path / "semantic.pt"
    write(semantic, {})
    roles = ["association_confirmatory", "prototype_discovery", "trajectory_confirmatory"]
    ids = [f"p{i}::step000" for i in range(3)]
    axis = tmp_path / "axis.csv"
    axis.write_text("id,analysis_role\n" + "".join(f"{sid},{role}\n" for sid, role in zip(ids, roles)))
    activation.mkdir()
    (activation / "expanded_generation_data.jsonl").write_text("".join(
        json.dumps({"id": sid, "prompt_token_ids": [1, 2]}) + "\n" for sid in ids))
    item = {"dataset": "mathqa", "target": "test", "activation_dir": str(activation),
            "semantic_cache": str(semantic), "axis_csv": str(axis), "layer_i": 14}
    options = {"activation_source": "runtime_remeasured", "semantic_fit_rows": 1,
               "screen_a_pool": 0, "screen_b_rows": 1, "evaluation_pool_rows": 1}
    preflight_condition(item, options)
    with pytest.raises(FileNotFoundError, match="layer_014.pt"):
        preflight_condition(item, {**options, "activation_source": "cache"})
