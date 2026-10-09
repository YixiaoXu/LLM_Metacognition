from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from metacog.audits.conditional_witness import (
    choose_floor, density_statement, evaluate_witness, fit_null, fit_witness,
    fixed_step, null_scores, paired_gain, pilot_split, select_bins,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_semantic_witness as runner
import run_semantic_leakage_audit as common
import stop_semantic_audit as stopper


def options():
    value = json.loads((ROOT / "configs/semantic_witness_v2.json").read_text())
    value.update({"smoke_only": True, "min_train_per_bin": 8, "min_calibration_per_bin": 4,
                  "min_test_prompts": 32, "max_test_prompts": 128, "bin_candidates": [2],
                  "bootstrap_samples": 32, "continuous_auxiliary": True})
    value["probes"].update({"device": "cpu", "families": ["linear", "mlp", "ensemble"],
                            "epochs": 2, "patience": 2, "hidden_dim": 8, "batch_size": 64})
    return value


def test_direct_conditional_gain_with_known_true_density():
    rng = np.random.default_rng(51)
    n = 20000
    z = rng.integers(0, 2, n)
    y = z ^ (rng.random(n) < .05)
    q = np.full((n, 2), .5)
    joint = np.column_stack([np.where(z == 0, .95, .05), np.where(z == 1, .95, .05)])
    result = paired_gain(y, q, joint, .5, .05, 1., .05, [f"p{i}" for i in range(n)], 32, 3)
    true_mi = 1 + .05 * np.log2(.05) + .95 * np.log2(.95)
    assert .6 < result["gain_lcb_bits"] < true_mi
    statement = density_statement(result["gain_lcb_bits"], {"upper_bits": 0., "rationale": "synthetic exact conditional"})
    assert statement["positive_conditional_information_under_assumption"]
    assert not statement["unconditional_shannon_information_certified"]


def test_semantic_underfitting_never_becomes_an_unconditional_information_proof():
    n = 10000
    y = np.tile([0, 1], n // 2)
    sem = np.full((n, 2), .5)
    joint = np.column_stack([np.where(y == 0, .99, .01), np.where(y == 1, .99, .01)])
    result = paired_gain(y, sem, joint, .5, .01, 1., .05, [f"p{i}" for i in range(n)], 32, 3)
    assert result["gain_lcb_bits"] > .8
    unknown = density_statement(result["gain_lcb_bits"], None)
    assert unknown["conditional_information_lcb_under_assumption_bits"] is None
    assert not unknown["positive_conditional_information_under_assumption"]
    # U is determined by S, so uniform q(U|S) has exactly one bit of KL error.
    actual = density_statement(result["gain_lcb_bits"], {"upper_bits": 1., "rationale": "known semantic null"})
    assert not actual["positive_conditional_information_under_assumption"]


def test_upper_only_cap_is_conservative_and_does_not_discard_negative_tail():
    y = np.tile([0, 1], 100)
    q = np.full((len(y), 2), .5)
    joint = np.tile([.99, .01], (len(y), 1))
    result = paired_gain(y, q, joint, .5, .01, .1, .05, [str(i) for i in range(len(y))], 32, 8)
    raw = np.log2(joint[np.arange(len(y)), y] / .5)
    assert result["capped_mean_bits"] == pytest.approx(np.minimum(raw, .1).mean())
    assert result["capped_mean_bits"] < result["mean_bits"]
    assert result["observed_statistic_min_bits"] < -5
    with pytest.raises(ValueError, match="independent"):
        paired_gain(y, q, joint, .5, .01, .1, .05, ["same"] * len(y), 32, 8)


def test_floor_rule_and_bin_rule_use_pilot_data_only():
    q = np.full((120, 4), .25)
    y = np.tile(np.arange(4), 30)
    chosen = choose_floor(q, y, [.001, .01, .05])
    assert chosen["floor"] == .05
    opt = options()
    opt.update({"bin_candidates": [2, 4, 8], "min_train_per_bin": 25, "min_calibration_per_bin": 8})
    u = np.linspace(-1, 1, 160)
    splits = pilot_split([f"p{i}" for i in range(len(u))], .25, 9)
    choice = select_bins(u, splits, opt)
    assert choice["bins"] in (2, 4)
    assert all(min(v) >= (25 if r == "train" else 8) for r, v in choice["counts"].items())


def test_known_semantic_null_generators_are_batch_and_order_invariant():
    g = torch.Generator().manual_seed(3)
    x, z = torch.randn(100, 10, generator=g), torch.randn(100, 1, generator=g)
    ids = [f"p{i}" for i in range(100)]
    for family in ("linear", "nonlinear"):
        spec = fit_null(x, z, ids, np.arange(75), family, 22, .2)
        whole = null_scores(x, ids, spec)
        batch = null_scores(x[70:], ids[70:], spec)
        assert torch.allclose(whole[70:], batch, atol=1e-6)
        assert whole.std() > .1


def synthetic_bundle(prefix, n=128):
    gen = torch.Generator().manual_seed(17 if prefix == "pilot" else 25)
    s = torch.randn(n, 3, generator=gen)
    z = torch.randn(n, 1, generator=gen)
    return {"ids": [f"{prefix}{i}::step000" for i in range(n)], "steps": [0] * n,
            "module_scores": z, "target_scores": z + .5 * s[:, :1], "module_names": ["module_a"],
            "semantics": {"training_reference": s, "independent_audit": s.square(),
                          "lexical_controls": torch.sin(s)}, "forbidden_ids": []}


def test_full_frozen_fit_reload_and_new_prompt_evaluation(tmp_path):
    torch.set_num_threads(1)
    opt = options()
    pilot, test = synthetic_bundle("pilot"), synthetic_bundle("new")
    fitdir, out = tmp_path / "fit", tmp_path / "out"
    receipt = fit_witness(pilot, "module_a", fitdir, opt, .05)
    assert receipt["freeze_before_final_extraction"]
    assert fit_witness(pilot, "module_a", fitdir, opt, .05) == receipt
    result = evaluate_witness(test, "module_a", fitdir, out, opt, .05)
    assert set(result["signals"]) == {"real", "semantic_null_linear", "semantic_null_nonlinear"}
    assert len(result["signals"]["real"]["comparisons"]) == 3
    assert result["continuous_auxiliary"]["not_a_shannon_information_bound"]
    assert result["conditional_information"]["conditional_information_lcb_under_assumption_bits"] is None
    assert evaluate_witness(test, "module_a", fitdir, out, opt, .05) == result
    with pytest.raises(ValueError, match="overlaps pilot"):
        evaluate_witness(pilot, "module_a", fitdir, tmp_path / "bad", opt, .05)
    changed = copy.deepcopy(opt)
    changed["primary_step"] = 4
    with pytest.raises(ValueError, match="configuration differs"):
        evaluate_witness(test, "module_a", fitdir, out, changed, .05)
    with pytest.raises(ValueError, match="changed"):
        fit_witness(pilot, "module_a", fitdir, changed, .05)
    weights = fitdir / "real/joint/predictors.pt"
    weights.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="changed"):
        evaluate_witness(test, "module_a", fitdir, out, opt, .05)


def test_test_bank_cannot_silently_drop_or_reorder_semantic_features():
    data = synthetic_bundle("pilot")
    schema = fixed_step(data, "module_a", 0)["schema"]
    data["semantics"]["independent_audit"] = data["semantics"]["independent_audit"][:, :2]
    with pytest.raises(ValueError, match="dimensions"):
        fixed_step(data, "module_a", 0, schema)


def test_witness_selection_is_global_and_alpha_tracks_only_frozen_witnesses(tmp_path):
    opt = options()
    opt["witness_count"] = 2
    tasks = []
    for i, score in enumerate([.05, .1, .07]):
        name = f"main_mathqa_m{i}"
        tasks.append({"name": name, "modules": [{"name": "module_a"}]})
        path = tmp_path / name / "analysis/module_a/audit_result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"continuous_audits": {"expanded_bank": {"ci95_low_bits": score}}}))
    selected, candidates = runner.select_witnesses(tmp_path, {"tasks": tasks}, opt)
    assert len(selected) == 2 and len(candidates) == 3
    assert [r["task"] for r in selected] == ["main_mathqa_m1", "main_mathqa_m2"]
    assert all(r["alpha"] == .025 for r in selected)
    opt["frozen_witnesses"] = [{"task": "main_mathqa_m0", "module": "module_a"}]
    selected, _ = runner.select_witnesses(tmp_path, {"tasks": tasks}, opt)
    assert selected[0]["alpha"] == .05


def test_smoke_selection_can_fallback_without_creating_a_formal_positive_result(tmp_path):
    opt = options()
    opt.update({"witness_count": 1, "smoke_only": True})
    task = {"name": "main_mathqa_m0", "modules": [{"name": "module_a"}]}
    path = tmp_path / task["name"] / "analysis/module_a/audit_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"continuous_audits": {"expanded_bank": {"ci95_low_bits": -.1}}}))
    selected, _ = runner.select_witnesses(tmp_path, {"tasks": [task]}, opt)
    assert selected[0]["module"] == "module_a"


def test_all_candidate_selection_freezes_positive_and_negative_pilot_results(tmp_path):
    opt = options()
    opt.update({"witness_selection_mode": "all_completed_candidates", "witness_count": 144})
    tasks = []
    for i, score in enumerate([.05, -.02, .0]):
        name = f"main_mathqa_m{i}"
        tasks.append({"name": name, "modules": [{"name": "module_a"}]})
        path = tmp_path / name / "analysis/module_a/audit_result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"continuous_audits": {"expanded_bank": {"ci95_low_bits": score}}}))
    runner.validate_config(opt)
    selected, candidates = runner.select_witnesses(tmp_path, {"tasks": tasks}, opt)
    assert len(selected) == len(candidates) == 3
    assert {row["pilot_ranking_score"] for row in selected} == {.05, -.02, .0}
    assert all(row["alpha"] == pytest.approx(.05 / 3) for row in selected)


def test_historical_exclusion_collects_all_pilot_tasks_not_just_winner(tmp_path):
    root = tmp_path / "pilot"
    for task in ("main_mathqa_m1", "main_mathqa_m2", "main_ultrachat_m1"):
        directory = root / task
        directory.mkdir(parents=True)
        (directory / "fresh_prompts.jsonl").write_text(json.dumps({"id": task, "prompt": "sample"}) + "\n")
    opt = options()
    opt["historical_audit_globs"] = []
    excluded = runner.collect_exclusions(root, opt, {"mathqa"}, tmp_path / "out")
    assert len(excluded["mathqa"]) == 2


def test_config_and_extractor_interfaces():
    runner.validate_config(options())
    bad = options()
    bad["witness_count"] = 144
    with pytest.raises(ValueError, match="at most three"):
        runner.validate_config(bad)
    bad = options()
    bad["target_density_error_assumption"] = {"upper_bits": .01}
    with pytest.raises(ValueError, match="rationale"):
        runner.validate_config(bad)
    for script in ("scripts/extract_generation_step_activations.py", "scripts/extract_activations.py"):
        common.validate_extraction_command(["python", script, "--model-path", "m", "--data", "d",
                                            "--layers", "1", "--output-dir", "o", *common.memory_flags(script)])
        with pytest.raises(ValueError, match="Unsupported"):
            common.validate_extraction_command(["python", script, "--not-a-real-argument"])


def test_worker_freezes_before_new_extraction_and_does_not_reselect_on_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    opt = options()
    opt["continuous_auxiliary"] = False
    pilot, test = synthetic_bundle("pilot"), synthetic_bundle("new")
    bp = tmp_path / "audit_bundle.pt"
    torch.save(pilot, bp)
    task = {"name": "main_mathqa_fake", "dataset": "mathqa", "modules": [{"name": "module_a"}],
            "pilot_bundle": str(bp), "pilot_bundle_sha256": runner.digest(bp)}
    witness = {"task": task["name"], "module": "module_a", "witness_id": "witness_1", "alpha": .05}
    manifest = {"tasks": [task], "witnesses": [witness], "config": opt, "pilot_config": {},
                "exclusion_tasks": [], "exclusions_by_domain": {"mathqa": {}}}
    monkeypatch.setattr(runner, "verify_snapshot", lambda m: None)
    monkeypatch.setattr(runner, "verify_source_modules", lambda b, t: None)
    calls = []
    def extract(t, excluded, effective, directory):
        frozen = directory.parent / "witness_1/fit/frozen_fit_manifest.json"
        assert frozen.exists(), "New evaluation data must not precede predictor freezing"
        assert effective["record_steps"] == [0]
        calls.append(effective["fresh_prompt_count"])
        return test
    monkeypatch.setattr(common, "prepare_bundle", extract)
    out = tmp_path / "run"
    runner.run_task(manifest, 0, out)
    runner.run_task(manifest, 0, out)
    assert len(calls) == 2 and calls[0] == calls[1]
    result = runner.export_results(out, manifest)
    assert result["completed_witnesses"] == 1
    assert result["planned_witnesses"] == 1
    assert (out / "predictor_comparisons.csv").exists()


def test_final_prompts_exclude_pilot_text_under_different_ids(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "external_semantic_data.jsonl").write_text(json.dumps({"id": "old", "prompt": "Original"}) + "\n")
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text(json.dumps({"id": "seen1", "prompt": "Already audited"}) + "\n")
    data = tmp_path / "data.jsonl"
    rows = [{"id": "new_id", "prompt": "ALREADY  AUDITED"}, {"id": "old_id", "prompt": "original"}]
    rows += [{"id": f"new{i}", "prompt": f"Fresh question {i}"} for i in range(20)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (old / "manifest.json").write_text(json.dumps({"data": str(data)}))
    task = {"name": "main_mathqa_fake", "dataset": "mathqa", "old_activation_dir": str(old)}
    out = tmp_path / "test"
    out.mkdir()
    config = {"seed": 7, "fresh_prompt_count": 20, "minimum_fresh_prompts": 16,
              "additional_exclusion_jsonl": [str(pilot)]}
    path = common.fresh_prompts(task, [task], config, out)
    assert len(list(common.read_rows(path))) == 20
    assert runner.read_json(out / "fresh_prompt_manifest.json")["excluded_rows"] == 2


def test_stop_command_requires_exact_repo_script_and_exact_run(tmp_path):
    repo = tmp_path / "repo"
    run = repo / "runs/old_audit"
    argv = ["python", "scripts/run_semantic_leakage_audit.py", "--output-dir", "runs/old_audit", "--worker-index", "0"]
    assert stopper.matches_run(argv, repo, repo, run)
    assert not stopper.matches_run(argv, repo, repo, repo / "runs/old_audit_extra")
    assert not stopper.matches_run(["python", "/other/scripts/run_semantic_leakage_audit.py", "--output-dir", str(run)], repo, repo, run)
    assert not stopper.matches_run(["python", "scripts/train_decoupler.py", "--output-dir", str(run)], repo, repo, run)
    assert stopper.matches_run(["python", str(repo / "scripts/run_semantic_witness.py"), f"--output-dir={run}"], repo, repo, run)
