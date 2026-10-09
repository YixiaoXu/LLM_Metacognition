"""Frozen-witness tests of predictive gain and assumption-indexed CMI bounds.

All design choices use pilot data. Final prompts are used only for evaluation.
For fixed q(U|S), r(U|S,M), D=E log2(r/q) satisfies I(U;M|S)>=D-KL(p||q).
The unknown conditional-density error is never estimated from NLL alone.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .semantic_leakage import (
    base_id, categorize, empirical_bernstein, fingerprint, fit_bins, fit_probes,
    floor_probabilities, make_probe, predict, prompt_bootstrap, write_json,
)


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(bundle: dict) -> str:
    h = hashlib.sha256()
    h.update(fingerprint({k: bundle[k] for k in ("ids", "steps", "module_names")}).encode())
    items = {"module": bundle["module_scores"], "target": bundle["target_scores"], **bundle["semantics"]}
    for name, values in sorted(items.items()):
        arr = torch.as_tensor(values).float().contiguous().numpy()
        h.update(fingerprint([name, arr.shape]).encode())
        h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def fixed_step(bundle: dict, module: str, step: int, schema: list | None = None) -> dict:
    if module not in bundle["module_names"]:
        raise ValueError(f"Frozen module absent from activation bundle: {module}")
    names = ["training_reference"] + sorted(k for k in bundle["semantics"] if k != "training_reference")
    if len(names) < 3 or "training_reference" not in bundle["semantics"]:
        raise ValueError("Need training reference, independent reference and semantic/lexical controls.")
    actual_schema = [[name, int(torch.as_tensor(bundle["semantics"][name]).shape[1])] for name in names]
    if schema is not None and schema != actual_schema:
        raise ValueError("Test semantic bank names/dimensions differ from frozen pilot bank.")
    idx = np.flatnonzero(np.asarray(bundle["steps"]) == step)
    ids = [str(bundle["ids"][i]) for i in idx]
    if not ids or len({base_id(i) for i in ids}) != len(ids):
        raise ValueError("Witness test needs exactly one fixed-step state per independent prompt.")
    m = bundle["module_names"].index(module)
    x = torch.cat([torch.as_tensor(bundle["semantics"][name]).float()[idx] for name in names], 1)
    z = torch.as_tensor(bundle["module_scores"]).float()[idx, m:m + 1]
    u = torch.as_tensor(bundle["target_scores"]).float()[idx, m]
    if not all(torch.isfinite(v).all() for v in (x, z, u)):
        raise ValueError("Nonfinite witness features.")
    return {"ids": ids, "x": x, "z": z, "u": u, "schema": actual_schema}


def pilot_split(ids: list[str], fraction: float, seed: int) -> dict:
    if not 0 < fraction < 1:
        raise ValueError("Pilot calibration fraction must be between zero and one.")
    order = sorted(range(len(ids)), key=lambda i: fingerprint([seed, base_id(ids[i])]))
    cut = int(len(ids) * (1 - fraction))
    if min(cut, len(ids) - cut) < 16:
        raise ValueError("Need at least 16 pilot prompts in both fit and calibration roles.")
    return {"train": np.asarray(order[:cut]), "validation": np.asarray(order[cut:]),
            "test": np.empty(0, dtype=int)}


def select_bins(u: np.ndarray, splits: dict, options: dict) -> dict:
    """Largest resolution satisfying pilot occupancy, never a test-effect rule."""
    records, eligible = [], []
    for k in options["bin_candidates"]:
        try:
            edges = fit_bins(u[splits["train"]], k)
        except ValueError as exc:
            records.append({"bins": k, "eligible": False, "reason": str(exc)})
            continue
        counts = {role: np.bincount(categorize(u[idx], edges), minlength=k).tolist()
                  for role, idx in splits.items() if role != "test"}
        ok = (min(counts["train"]) >= options["min_train_per_bin"] and
              min(counts["validation"]) >= options["min_calibration_per_bin"])
        record = {"bins": k, "edges": edges.tolist(), "counts": counts, "eligible": ok}
        records.append(record)
        if ok:
            eligible.append(record)
    if not eligible:
        raise ValueError("No bin resolution meets pilot occupancy requirements; increase pilot data.")
    selected = max(eligible, key=lambda r: r["bins"])
    return {**selected, "rule": "largest_pilot_occupancy_supported_resolution", "candidates": records}


def choose_floor(probabilities: np.ndarray, truth: np.ndarray, candidates: list[float]) -> dict:
    """Largest smoothing floor within one paired SE of best calibration NLL."""
    losses = {f: -np.log2(floor_probabilities(probabilities, f)[np.arange(len(truth)), truth])
              for f in candidates}
    best = min(losses, key=lambda f: float(losses[f].mean()))
    records, allowed = [], []
    for f, values in losses.items():
        difference = values - losses[best]
        se = float(difference.std(ddof=1) / math.sqrt(len(truth)))
        ok = float(difference.mean()) <= se + 1e-12
        records.append({"floor": f, "calibration_nll_bits": float(values.mean()),
                        "paired_se_from_best": se, "eligible": ok})
        if ok:
            allowed.append(f)
    return {"floor": max(allowed), "best_nll_floor": best, "rule": "paired_one_SE_then_largest_floor",
            "candidates": records}


def fit_bank(x: torch.Tensor, y: torch.Tensor, splits: dict, options: dict,
             directory: Path, label: str, classes: int = 0) -> dict:
    result = fit_probes(x, y, splits, options, directory, label, classes)
    write_json(directory / "predictor_spec.json", {"options": options, "classes": classes})
    return result


def predict_bank(directory: Path, x: torch.Tensor, device: str) -> dict[str, np.ndarray]:
    spec = json.loads((directory / "predictor_spec.json").read_text())
    opts, classes = spec["options"], spec["classes"]
    states = torch.load(directory / "predictors.pt", map_location="cpu", weights_only=False)
    result = {}
    for family, state in states.items():
        model = make_probe(x.shape[1], classes or 1, family, opts["hidden_dim"], opts["dropout"])
        model.load_state_dict(state["state_dict"], strict=True)
        model.to(device).eval()
        normed = (x - state["input_mean"]) / state["input_std"]
        pred = predict(model, normed, opts["batch_size"], torch.device(device))
        pred = (torch.softmax(pred / state["temperature"], 1) if classes
                else pred * state["target_std"] + state["target_mean"])
        result[family] = pred.numpy()
        del model, normed
    if "ensemble" in opts["families"]:
        result["ensemble"] = np.mean(list(result.values()), axis=0)
    return result


def _null_raw(x: torch.Tensor, ids: list[str], spec: dict) -> np.ndarray:
    a = ((x - spec["mean"]) / spec["std"]) @ spec["weights"]
    if spec["family"] == "linear":
        raw = a[:, 0].numpy()
    else:
        raw = (torch.sin(a[:, 0]) + .5 * a[:, 1] * a[:, 2] + torch.tanh(a[:, 3])).numpy()
    raw = (raw - spec["raw_mean"]) / spec["raw_std"]
    # Per-ID independent pseudorandom noise is stable across resume and batching.
    noise = np.array([np.random.default_rng(int(fingerprint([spec["seed"], base_id(i)])[:16], 16)).normal()
                      for i in ids])
    return raw + spec["noise_std"] * noise


def fit_null(x: torch.Tensor, z: torch.Tensor, ids: list[str], train: np.ndarray,
             family: str, seed: int, noise_std: float) -> dict:
    gen = torch.Generator().manual_seed(seed)
    spec = {"family": family, "seed": seed, "noise_std": noise_std,
            "mean": x[train].mean(0), "std": x[train].std(0, unbiased=False).clamp_min(1e-6),
            "weights": torch.randn(x.shape[1], 4, generator=gen) / math.sqrt(x.shape[1]),
            "raw_mean": 0., "raw_std": 1.}
    no_noise = {**spec, "noise_std": 0.}
    raw = _null_raw(x[train], [ids[i] for i in train], no_noise)
    spec.update({"raw_mean": float(raw.mean()), "raw_std": max(float(raw.std()), 1e-6)})
    raw = _null_raw(x[train], [ids[i] for i in train], spec)
    quantiles = np.linspace(0, 1, 257)
    spec["map_from"] = np.quantile(raw, quantiles)
    spec["map_to"] = np.quantile(z[train].numpy().flatten(), quantiles)
    return spec


def null_scores(x: torch.Tensor, ids: list[str], spec: dict) -> torch.Tensor:
    raw = _null_raw(x, ids, spec)
    return torch.as_tensor(np.interp(raw, spec["map_from"], spec["map_to"]), dtype=torch.float32).view(-1, 1)


def ratio_span(classes: int, sem_floor: float, joint_floor: float, cap: float) -> tuple[float, float]:
    lo = math.log2(joint_floor / (1 - (classes - 1) * sem_floor))
    hi = math.log2((1 - (classes - 1) * joint_floor) / sem_floor)
    return lo, min(cap, hi)


def paired_gain(y: np.ndarray, sem: np.ndarray, joint: np.ndarray, sem_floor: float,
                joint_floor: float, cap: float, alpha: float, ids: list[str],
                bootstrap: int, seed: int) -> dict:
    if len(set(map(base_id, ids))) != len(ids) or len(ids) != len(y):
        raise ValueError("Bounds require one independent fixed-step row per base prompt.")
    if not 0 < alpha < 1 or cap <= 0 or not math.isfinite(cap):
        raise ValueError("Invalid confidence allocation or frozen positive log-ratio cap.")
    for q, f in ((sem, sem_floor), (joint, joint_floor)):
        if (q.ndim != 2 or len(q) != len(y) or not np.isfinite(q).all() or
                not np.allclose(q.sum(1), 1) or q.min() < f - 1e-10 or f <= 0):
            raise ValueError("Invalid probabilities or smoothing-floor contract.")
    idx = np.arange(len(y))
    raw = np.log2(joint[idx, y]) - np.log2(sem[idx, y])
    capped = np.minimum(raw, cap)
    lo, hi = ratio_span(sem.shape[1], sem_floor, joint_floor, cap)
    lower, _ = empirical_bernstein(capped, alpha, hi - lo)
    boot = prompt_bootstrap(raw, ids, bootstrap, seed)
    return {**boot, "capped_mean_bits": float(capped.mean()), "gain_lcb_bits": lower,
            "alpha": alpha, "upper_cap_bits": cap, "upper_capped_fraction": float((raw > cap).mean()),
            "guaranteed_statistic_min_bits": lo, "guaranteed_statistic_max_bits": hi,
            "guaranteed_statistic_span_bits": hi - lo,
            "observed_statistic_min_bits": float(raw.min()), "observed_statistic_max_bits": float(raw.max()),
            "semantic_nll_bits": float(-np.log2(sem[idx, y]).mean()),
            "joint_nll_bits": float(-np.log2(joint[idx, y]).mean()),
            "bound_method": "one_sided_empirical_Bernstein_on_upper_capped_log_ratio",
            "cap_validity": "E[min(D,cap)]<=E[D]; negative tail is not clipped",
            "bootstrap_scope": "approximate_fixed_predictor_diagnostic_not_the_primary_certificate"}


def density_statement(lower: float, assumption: dict | None) -> dict:
    if assumption is not None and (not math.isfinite(assumption["upper_bits"]) or
                                  assumption["upper_bits"] < 0 or not assumption.get("rationale")):
        raise ValueError("An explicit target-density KL assumption needs a nonnegative budget and rationale.")
    value = None if assumption is None else lower - assumption["upper_bits"]
    return {"target_density_error_definition": "E_S KL(p(U_bin|S)||q_selected(U_bin|S)) in bits",
            "target_density_error_estimated": False,
            "density_error_assumption": assumption,
            "conditional_information_lcb_under_assumption_bits": value,
            "positive_conditional_information_under_assumption": value is not None and value > 0,
            "density_error_break_even_bits": max(0., lower),
            "positive_budget_exists": lower > 0,
            "unconditional_shannon_information_certified": False,
            "absolute_semantic_freeness_proven": False,
            "scope": "frozen binned target, supplied semantic bank, fixed step and prompt population"}


def sample_plan(variance: float, span: float, alpha: float, options: dict) -> dict:
    def penalty(n):
        return math.sqrt(2 * variance * math.log(2 / alpha) / n) + 7 * span * math.log(2 / alpha) / (3 * (n - 1))
    low, high = options["min_test_prompts"], options["max_test_prompts"]
    while low < high:
        mid = (low + high) // 2
        if penalty(mid) <= options["target_bound_penalty_bits"]:
            high = mid
        else:
            low = mid + 1
    return {"planned_prompts": low, "pilot_estimated_penalty_bits": penalty(low),
            "precision_target_met_in_pilot": penalty(low) <= options["target_bound_penalty_bits"],
            "sample_size_is_power_planning_not_a_guarantee": True}


def verify_fit(directory: Path) -> dict:
    receipt = json.loads((directory / "frozen_fit_manifest.json").read_text())
    for name, sha in receipt["artifact_sha256"].items():
        if not (directory / name).exists() or file_digest(directory / name) != sha:
            raise ValueError(f"Frozen witness predictor changed: {directory / name}")
    return receipt


def fit_witness(bundle: dict, module: str, output: Path, options: dict, alpha: float) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    contract = {"pilot_signature": signature(bundle), "module": module, "options": options, "alpha": alpha}
    done = output / "frozen_fit_manifest.json"
    if done.exists():
        receipt = verify_fit(output)
        if receipt["contract"] != contract:
            raise ValueError("Pilot data or frozen witness fit contract changed. Use a new RUN_ROOT.")
        print(f"[witness-reuse] frozen predictors {output}", flush=True)
        return receipt
    data = fixed_step(bundle, module, options["primary_step"])
    x, z, u, ids = (data[k] for k in ("x", "z", "u", "ids"))
    splits = pilot_split(ids, options["calibration_fraction"], options["seed"])
    tr, cal = splits["train"], splits["validation"]
    binned = select_bins(u.numpy(), splits, options)
    k = binned["bins"]
    y = torch.as_tensor(categorize(u.numpy(), np.asarray(binned["edges"])))
    probe = {**options["probes"], "seed": options["seed"]}
    print(f"[witness-fit] {module} step={options['primary_step']} fit={len(tr)} calibration={len(cal)} "
          f"bins={k} alpha={alpha:.6g} full_semantic_dim={x.shape[1]}", flush=True)
    sem = fit_bank(x, y, splits, probe, output / "semantic", f"{module}/U|S", k)
    floors = {name: choose_floor(p[cal], y[cal].numpy(), options["probability_floor_candidates"])
              for name, p in sem["predictions"].items()}
    sem_probs = {name: floor_probabilities(p, floors[name]["floor"]) for name, p in sem["predictions"].items()}
    selected_s = min(sem_probs, key=lambda name: float(-np.log2(sem_probs[name][cal, y[cal].numpy()]).mean()))
    nulls = {name: fit_null(x, z, ids, tr, name, options["seed"] + 801 + i, options["null_noise_std"])
             for i, name in enumerate(options["semantic_null_families"])}
    signals = {"real": z, **{f"semantic_null_{name}": null_scores(x, ids, spec) for name, spec in nulls.items()}}
    settings, plans = {}, []
    for name, signal in signals.items():
        fit = fit_bank(torch.cat([x, signal], 1), y, splits, probe, output / name / "joint", f"{module}/{name}/U|S,M", k)
        jfloors = {family: choose_floor(p[cal], y[cal].numpy(), options["probability_floor_candidates"])
                   for family, p in fit["predictions"].items()}
        jprobs = {f: floor_probabilities(p, jfloors[f]["floor"]) for f, p in fit["predictions"].items()}
        selected_j = min(jprobs, key=lambda f: float(-np.log2(jprobs[f][cal, y[cal].numpy()]).mean()))
        pairs = {}
        for sf, sp in sem_probs.items():
            delta = np.log2(jprobs[selected_j][cal, y[cal].numpy()]) - np.log2(sp[cal, y[cal].numpy()])
            cap = max(.01, float(np.quantile(delta, options["upper_cap_quantile"])))
            pairs[sf] = {"upper_cap_bits": cap, "calibration_mean_gain_bits": float(delta.mean())}
            if name == "real":
                lo, hi = ratio_span(k, floors[sf]["floor"], jfloors[selected_j]["floor"], cap)
                plans.append(sample_plan(float(np.minimum(delta, cap).var(ddof=1)), hi - lo, alpha, options))
        settings[name] = {"selected_joint_family": selected_j, "joint_floors": jfloors, "comparisons": pairs}
        print(f"[witness-calibration] {name} semantic={selected_s} joint={selected_j} "
              f"calibration_gain={pairs[selected_s]['calibration_mean_gain_bits']:.5f} "
              f"floor_sem={floors[selected_s]['floor']} floor_joint={jfloors[selected_j]['floor']}", flush=True)
    regression = None
    if options["continuous_auxiliary"]:
        rs = fit_bank(x, u, splits, probe, output / "continuous/semantic", f"{module}/continuous/U|S")
        rj = fit_bank(torch.cat([x, z], 1), u, splits, probe, output / "continuous/joint", f"{module}/continuous/U|S,M")
        sf, jf = rs["selected_family"], rj["selected_family"]
        regression = {"semantic_family": sf, "joint_family": jf,
                      "semantic_variance": max(float(np.square(u[cal].numpy() - rs["predictions"][sf][cal].flatten()).mean()), 1e-8),
                      "joint_variance": max(float(np.square(u[cal].numpy() - rj["predictions"][jf][cal].flatten()).mean()), 1e-8)}
    frozen = {"schema": data["schema"], "bins": binned, "semantic_floors": floors, "selected_semantic_family": selected_s,
              "signals": settings, "null_generators": nulls, "regression": regression,
              "pilot_base_ids": sorted(set(map(base_id, bundle["ids"]))),
              "split_ids": {r: [ids[i] for i in idx] for r, idx in splits.items() if r != "test"}}
    torch.save(frozen, output / "frozen_audit.pt")
    planned = max(plan["planned_prompts"] for plan in plans)
    receipt = {"contract": contract, "planned_test_prompts": planned, "sample_plans_by_semantic_comparator": plans,
               "module": module, "bins": binned, "selected_semantic_family": selected_s,
               "semantic_floors": floors, "signal_settings": settings,
               "freeze_before_final_extraction": True,
               "artifact_sha256": {str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob("*"))
                                    if p.is_file() and p.name in {"predictors.pt", "predictor_spec.json", "frozen_audit.pt"}}}
    write_json(done, receipt)
    print(f"[witness-frozen] {module} planned_new_test_prompts={planned}; no final test inspected", flush=True)
    return receipt


def evaluate_witness(bundle: dict, module: str, fit_dir: Path, output: Path, options: dict, alpha: float) -> dict:
    receipt = verify_fit(fit_dir)
    if receipt["contract"]["options"] != options or receipt["contract"]["alpha"] != alpha:
        raise ValueError("Evaluation configuration differs from the frozen fit.")
    frozen = torch.load(fit_dir / "frozen_audit.pt", map_location="cpu", weights_only=False)
    contract = {"test_signature": signature(bundle), "fit_manifest_sha256": file_digest(fit_dir / "frozen_fit_manifest.json")}
    output.mkdir(parents=True, exist_ok=True)
    done = output / "witness_result.json"
    if done.exists():
        saved = json.loads(done.read_text())
        if saved["contract"] != contract:
            raise ValueError("Final evaluation data changed; refusing to overwrite a witnessed result.")
        return saved
    if set(map(base_id, bundle["ids"])) & set(frozen["pilot_base_ids"]):
        raise ValueError("Final test overlaps pilot prompts; new prompt IDs/texts are required.")
    if set(map(base_id, bundle["ids"])) & set(map(base_id, bundle.get("forbidden_ids", []))):
        raise ValueError("Final test overlaps excluded historical prompts.")
    data = fixed_step(bundle, module, options["primary_step"], frozen["schema"])
    x, z, u, ids = (data[k] for k in ("x", "z", "u", "ids"))
    if len(ids) < options["min_test_prompts"]:
        raise ValueError("Too few independent fixed-step test prompts.")
    y = categorize(u.numpy(), np.asarray(frozen["bins"]["edges"]))
    sem = predict_bank(fit_dir / "semantic", x, options["probes"]["device"])
    sem = {f: floor_probabilities(p, frozen["semantic_floors"][f]["floor"]) for f, p in sem.items()}
    signals = {"real": z, **{f"semantic_null_{f}": null_scores(x, ids, spec)
                            for f, spec in frozen["null_generators"].items()}}
    results, predictions = {}, {"ids": ids, "target_bins": y, "semantic_probabilities": sem}
    for name, signal in tqdm(signals.items(), desc="Evaluate frozen witness and semantic nulls", unit="signal"):
        setting = frozen["signals"][name]
        jf = setting["selected_joint_family"]
        all_joint = predict_bank(fit_dir / name / "joint", torch.cat([x, signal], 1), options["probes"]["device"])
        joint_floor = setting["joint_floors"][jf]["floor"]
        joint = floor_probabilities(all_joint[jf], joint_floor)
        comparisons = {sf: paired_gain(y, sp, joint, frozen["semantic_floors"][sf]["floor"], joint_floor,
                        setting["comparisons"][sf]["upper_cap_bits"], alpha, ids,
                        options["bootstrap_samples"], options["seed"]) for sf, sp in sem.items()}
        lower = min(r["gain_lcb_bits"] for r in comparisons.values())
        results[name] = {"comparisons": comparisons, "robust_gain_lcb_bits": lower,
                         "positive_vs_all_frozen_semantic_predictors": lower > 0,
                         "minimum_mean_gain_bits": min(r["mean_bits"] for r in comparisons.values()),
                         "iut_bootstrap_p_diagnostic": max(r["p_one_sided_centered_bootstrap"] for r in comparisons.values()),
                         "inference": "intersection_union_against_fixed_finite_predictor_bank",
                         "known_true_conditional_information_bits": None if name == "real" else 0.}
        predictions[name] = {"signal": signal, "joint_probabilities": joint}
        print(f"[witness-test] {module} {name} n={len(ids)} gain_LCB={lower:.6f} "
              f"predictor_bank_pass={lower > 0}", flush=True)
    lower = results["real"]["robust_gain_lcb_bits"]
    statement = density_statement(lower, options.get("target_density_error_assumption"))
    budgets = sorted({0., *[max(0., lower) * f for f in (0.25, .5, .75, 1., 1.5)]})
    sensitivity = [{"hypothetical_target_density_KL_bits": b, "conditional_information_lcb_bits": lower - b,
                    "budget_is_measured": False} for b in budgets]
    auxiliary = None
    if frozen["regression"]:
        reg = frozen["regression"]
        sp = predict_bank(fit_dir / "continuous/semantic", x, options["probes"]["device"])[reg["semantic_family"]].flatten()
        jp = predict_bank(fit_dir / "continuous/joint", torch.cat([x, z], 1), options["probes"]["device"])[reg["joint_family"]].flatten()
        es, ej = np.square(u.numpy() - sp), np.square(u.numpy() - jp)
        vs, vj = reg["semantic_variance"], reg["joint_variance"]
        inc = .5 * (np.log(vs / vj) + es / vs - ej / vj) / np.log(2)
        auxiliary = {**prompt_bootstrap(inc, ids, options["bootstrap_samples"], options["seed"]),
                     "semantic_mse": float(es.mean()), "semantic_plus_module_mse": float(ej.mean()),
                     "relative_mse_reduction": float(1 - ej.mean() / max(es.mean(), 1e-12)),
                     "not_a_shannon_information_bound": True}
        predictions["continuous_auxiliary"] = {"target": u, "semantic": sp, "joint": jp}
    flags = [name for name, r in results.items() if name != "real" and r["positive_vs_all_frozen_semantic_predictors"]]
    result = {"schema_version": 2, "module": module, "contract": contract, "n_test_prompts": len(ids),
              "primary_step": options["primary_step"], "bins": frozen["bins"]["bins"], "alpha": alpha,
              "signals": results, "conditional_information": statement, "density_sensitivity": sensitivity,
              "continuous_auxiliary": auxiliary, "semantic_null_alerts": flags,
              "predictive_witness_with_null_sanity": lower > 0 and not flags,
              "nulls_are_diagnostics_not_a_density_error_upper_bound": True,
              "semantic_bank_schema": data["schema"],
              "statistical_scope": "per frozen witness; alpha already allocated across planned witnesses",
              "interpretation": "Positive finite-bank gain is not unconditional proof of nonsemantic or second-order information."}
    torch.save(predictions, output / "heldout_predictions.pt")
    write_json(done, result)
    return result
