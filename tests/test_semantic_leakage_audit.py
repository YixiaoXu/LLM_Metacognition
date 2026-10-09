from __future__ import annotations

import importlib.util
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from metacog.audits.semantic_leakage import (
    audit_bundle, assert_disjoint, categorize, club_blocks, entropy_lower_bound,
    fit_bins, floor_probabilities, grouped_split, information_bounds,
    entropy_continuity_penalty, leakage_ceiling,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("semantic_audit_runner", ROOT / "scripts/run_semantic_leakage_audit.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_grouped_splits_and_historical_exclusion():
    ids = [f"p{i}::step{s:03d}" for i in range(100) for s in (0, 4, 8)]
    split = grouped_split(ids, 71)
    groups = [{ids[i].split("::")[0] for i in idx} for idx in split.values()]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:])
    assert sum(map(len, split.values())) == len(ids)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(ids, ["p20::step020"])
    with pytest.raises(ValueError, match="Duplicate"):
        grouped_split(ids + ids[:1], 71)


def test_bins_do_not_refit_on_test_and_fail_on_collapse():
    edges = fit_bins(np.arange(100.), 4)
    assert np.array_equal(categorize(np.array([-10., 10., 99., 120.]), edges), [0, 0, 3, 3])
    with pytest.raises(ValueError, match="Degenerate"):
        fit_bins(np.ones(30), 4)


def test_entropy_bound_and_probability_normalization():
    labels = np.tile(np.arange(4), 2500)
    lower = entropy_lower_bound(labels, 4, .01)
    assert 1.9 < lower <= 2
    assert entropy_lower_bound(np.zeros(300, dtype=int), 4, .05) == pytest.approx(0)
    probs = floor_probabilities(np.array([[100., 0., 0., 0.]]), .01)
    assert probs.sum() == pytest.approx(1)
    assert probs.min() == pytest.approx(.01)


def test_club_exact_joint_product_expectation():
    rng = np.random.default_rng(5)
    s = rng.integers(0, 2, 100000)
    c = s ^ (rng.random(len(s)) < .1)
    probs = np.column_stack([np.where(s == 0, .9, .1), np.where(s == 1, .9, .1)])
    estimated = club_blocks(c, probs, 52).mean()
    expected = .4 * np.log2(9)
    assert estimated == pytest.approx(expected, abs=.025)


def test_nonsemantic_signal_can_exceed_semantic_functional_with_known_density():
    rng = np.random.default_rng(111)
    train_c = rng.integers(0, 2, 5000)
    train_v = train_c ^ (rng.random(len(train_c)) < .03)
    c = rng.integers(0, 2, 20000)
    v = c ^ (rng.random(len(c)) < .03)
    result = information_bounds(train_c, train_v, c, v, {"known_independent": np.full((len(c), 2), .5)},
                                2, .001, .05, 51)
    assert result["density_kl_break_even_bits"] > .5
    assert result["remaining_information_lcb_at_zero_density_kl"] > .5
    assert result["max_supported_density_kl_budget_bits"] > 0
    assert not result["absolute_semantic_freeness_proven"]
    assert result["unconditional_gap_lcb_bits"] <= 0


def test_pure_semantic_common_cause_does_not_pass_gap():
    rng = np.random.default_rng(313)
    s = rng.integers(0, 2, 30000)
    c = s ^ (rng.random(len(s)) < .05)
    v = s ^ (rng.random(len(s)) < .03)
    q = np.column_stack([np.where(s[5000:] == 0, .95, .05), np.where(s[5000:] == 1, .95, .05)])
    result = information_bounds(c[:5000], v[:5000], c[5000:], v[5000:], {"known_semantic": q}, 2, .001, .05, 42)
    assert result["density_kl_break_even_bits"] < 0
    assert result["remaining_information_lcb_at_zero_density_kl"] < 0


def test_bad_density_cannot_be_called_unconditional_upper_bound():
    rng = np.random.default_rng(71)
    s = rng.integers(0, 2, 20000)
    # A deliberately incompetent uniform probe hides a perfect semantic copy.
    result = information_bounds(s[:1000], s[:1000], s[1000:], s[1000:],
                                {"misspecified": np.full((19000, 2), .5)}, 2, .001, .05, 2)
    assert result["density_kl_break_even_bits"] > 0
    assert result["upper_bound_status"] == "requires_additive_conditional_density_KL_budget"
    assert not result["absolute_semantic_freeness_proven"]
    # The true KL to the uniform probe is 1 bit, which eliminates the apparent gap.
    assert result["density_kl_break_even_bits"] - 1 < 0
    assert result["state_information_lcb_bits"] - leakage_ceiling(result, 1.) < 0


def test_entropy_continuity_budget_is_monotonic_and_reaches_trivial_bound():
    values = [entropy_continuity_penalty(b, 4) for b in (0., .01, .1, .5, 2., 5.)]
    assert values[0] == 0
    assert values[-1] == pytest.approx(2)
    assert all(a <= b for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError, match="nonnegative"):
        entropy_continuity_penalty(-.1, 4)


def test_fixed_target_table_range_remains_a_valid_small_sample_bound():
    rng = np.random.default_rng(9)
    c = rng.integers(0, 2, 2000)
    v = c ^ (rng.random(len(c)) < .2)
    bound = information_bounds(c[:1000], v[:1000], c[1000:], v[1000:],
            {"independent": np.full((1000, 2), .5)}, 2, .001, .01, 1)
    true_mi = 1 + .2 * np.log2(.2) + .8 * np.log2(.8)
    assert 0 <= bound["state_information_lcb_bits"] <= true_mi
    assert bound["target_cross_entropy_fixed_range_bits"] < 3


def test_load_alignment_does_not_join_by_position(tmp_path):
    path = tmp_path / "layer_001.pt"
    torch.save({"ids": ["b", "a"], "features": torch.tensor([[2., 3.], [0., 1.]])}, path)
    ids, x = runner.load_aligned(path, ["a", "b"])
    assert ids == ["a", "b"]
    assert torch.equal(x, torch.tensor([[0., 1.], [2., 3.]]))
    with pytest.raises(ValueError, match="missing"):
        runner.load_aligned(path, ["c"])


def test_fresh_prompts_exclude_text_even_when_ids_differ(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    existing = [{"id": f"old{i}::step000", "prompt": f"Question {i}"} for i in range(10)]
    (old / "external_semantic_data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in existing))
    data = tmp_path / "all.jsonl"
    rows = [{"id": f"new{i}", "prompt": f"QUESTION   {i}"} for i in range(30)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (old / "manifest.json").write_text(json.dumps({"data": str(data)}))
    task = {"name": "fake", "dataset": "mathqa", "old_activation_dir": str(old)}
    out = tmp_path / "out"
    out.mkdir()
    result = runner.fresh_prompts(task, [task], {"seed": 3, "fresh_prompt_count": 15,
                                  "minimum_fresh_prompts": 15}, out)
    picked = list(runner.read_rows(result))
    assert len(picked) == 15
    assert all(int(r["id"][3:]) >= 10 for r in picked)
    assert runner.fresh_prompts(task, [task], {"seed": 3, "fresh_prompt_count": 15,
                                "minimum_fresh_prompts": 15}, out) == result
    data.write_text(data.read_text() + json.dumps({"id": "extra", "prompt": "new question"}) + "\n")
    with pytest.raises(ValueError, match="source/exclusions/cache changed"):
        runner.fresh_prompts(task, [task], {"seed": 3, "fresh_prompt_count": 15,
                                "minimum_fresh_prompts": 15}, out)


def test_config_validation_rejects_invalid_audit_contract():
    config = json.loads((ROOT / "configs/semantic_leakage_audit_v1.json").read_text())
    runner.validate_config(config)
    config["bound_steps"] = [1]
    with pytest.raises(ValueError, match="subset"):
        runner.validate_config(config)
    config["bound_steps"] = [0]
    config["density_kl_budgets_bits"] = [-1]
    with pytest.raises(ValueError, match="nonnegative"):
        runner.validate_config(config)


def test_extraction_flags_match_current_interfaces_and_preserve_template():
    model = SimpleNamespace(resolved_path="Qwen3-4B", dtype="bfloat16", trust_remote_code=False)
    args = runner.extraction_flags(model, "false", {"use_chat_template": False,
                  "system_prompt": "original system", "dtype": "float16", "prompt_style": "data"})
    assert "--use-chat-template" not in args
    assert args[args.index("--system-prompt") + 1] == "original system"
    assert args[args.index("--dtype") + 1] == "float16"
    assert "--use-chat-template" in runner.extraction_flags(model, "auto")
    for script in ("extract_activations.py", "extract_generation_step_activations.py"):
        tree = ast.parse((ROOT / "scripts" / script).read_text())
        supported = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                     for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        assert {v for v in args if v.startswith("--")} <= supported


def test_frozen_encoder_reads_later_state_but_target_uses_earlier_state(tmp_path):
    from train_decoupler import MLP
    from metacog.intervention.direct import ModuleRefiner

    torch.manual_seed(18)
    dec = tmp_path / "dec"
    module = dec / "joint_residual_modules/module_00"
    module.mkdir(parents=True)
    act = tmp_path / "act"
    act.mkdir()
    ids = [f"new{i}::step000" for i in range(4)]
    earlier, later = torch.randn(4, 5), torch.randn(4, 5)
    torch.save({"ids": ids, "features": earlier}, act / "layer_001.pt")
    torch.save({"ids": ids, "features": later}, act / "layer_003.pt")
    encoder = MLP(5, 3, 7, 0.).eval()
    torch.save({f"e2.{k}": v for k, v in encoder.state_dict().items()}, dec / "best_model.pt")
    torch.save({"next_mean": torch.zeros(1, 5), "next_std": torch.ones(1, 5)}, dec / "normalization.pt")
    runner.write_json(dec / "config.json", {"second_stage": "main_direct", "latent_dim": 3,
                       "hidden_dim": 7, "dropout": 0., "layer_i": 1, "next_layer": 3})
    refiner = ModuleRefiner(3, 2, 5, 2, 0.).eval()
    torch.save({"latent_source": "main_z2", "input_dim": 3, "code_dim": 2, "hidden_dim": 5,
                "output_dim": 2, "dropout": 0., "state_dict": refiner.state_dict()}, module / "module_refiner.pt")
    loading = torch.tensor([.6, -.8])
    torch.save({"selected_neurons": torch.tensor([0, 2]), "module_loading": loading,
                "continuous_mean": torch.zeros(5), "continuous_std": torch.ones(5)},
               module / "soft_residual_intervention_targets.pt")
    (act / "external_semantic_data.jsonl").write_text("".join(json.dumps({"id": i, "prompt": "Question",
                           "generation_step": 0}) + "\n" for i in ids))
    runner.write_json(tmp_path / "forbidden_base_ids.json", [])
    task = {"name": "fake", "modules": [{"name": "module_0", "module_dir": str(module), "decoupler_dir": str(dec)}]}
    runner.write_json(tmp_path / "fresh_prompt_manifest.json", {"zero_base_id_overlap": True})
    result = runner.encode_frozen_modules(task, act, {"training_reference": [act / "layer_003.pt"]},
                       {"probes": {"device": "cpu"}, "encode_batch_size": 2}, tmp_path)
    with torch.inference_mode():
        expected_m = refiner(encoder(later))[1] @ loading
    assert torch.allclose(result["module_scores"][:, 0], expected_m, atol=1e-6)
    assert torch.allclose(result["target_scores"][:, 0], earlier[:, [0, 2]] @ loading)
    assert not torch.allclose(result["module_scores"], result["target_scores"])


def test_complete_probe_pipeline_resume_and_contract(tmp_path):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(61)
    n = 120
    s, z = torch.randn(n, 3, generator=generator), torch.randn(n, 1, generator=generator)
    bundle = {"ids": [f"fresh{i}::step000" for i in range(n)], "steps": [0] * n,
              "module_scores": z, "target_scores": z + s[:, :1], "module_names": ["module_0"],
              "semantics": {"training_reference": s, "independent_audit": s.square()}, "forbidden_ids": []}
    options = json.loads((ROOT / "configs/semantic_leakage_audit_v1.json").read_text())
    options.update({"bootstrap_samples": 32, "min_bound_prompts": 10, "planned_comparisons": 1})
    options["probes"].update({"device": "cpu", "families": ["linear", "mlp", "ensemble"],
                              "epochs": 2, "hidden_dim": 8})
    result = audit_bundle(bundle, tmp_path, options)
    assert result["modules"][0]["bounds"][0]["status"] == "estimated"
    assert not result["modules"][0]["bounds"][0]["positive_gap_under_assumed_budget"]
    runner.export_tables(result, tmp_path)
    assert (tmp_path / "information_bound_sensitivity.csv").exists()
    assert (tmp_path / "continuous_information.csv").exists()
    assert audit_bundle(bundle, tmp_path, options)["modules"] == result["modules"]
    options["probes"]["epochs"] += 1
    with pytest.raises(ValueError, match="changed"):
        audit_bundle(bundle, tmp_path, options)
