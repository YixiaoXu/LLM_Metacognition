"""Semantic audit statistics and probes, without changing frozen representations.

The categorical CLUB functional is an upper bound only after adding the
unknown conditional-density KL error. We report that error as a sensitivity
budget; a confidence bound on the fitted functional alone is not an MI bound.
All information quantities below use base-2 logarithms.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def base_id(value: str) -> str:
    return re.sub(r"::step\d+$", "", str(value))


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def grouped_split(ids: list[str], seed: int, train_fraction: float = .4,
                  validation_fraction: float = .2) -> dict[str, np.ndarray]:
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate activation IDs are not permitted.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Invalid train/validation fractions.")
    groups = sorted({base_id(i) for i in ids}, key=lambda x: fingerprint([seed, x]))
    n_train = int(len(groups) * train_fraction)
    n_val = int(len(groups) * validation_fraction)
    if min(n_train, n_val, len(groups) - n_train - n_val) < 8:
        raise ValueError("Audit needs at least eight independent prompts in each split.")
    role = {g: ("train" if i < n_train else "validation" if i < n_train + n_val
                else "test") for i, g in enumerate(groups)}
    return {name: np.array([i for i, sid in enumerate(ids) if role[base_id(sid)] == name])
            for name in ("train", "validation", "test")}


def assert_disjoint(ids: list[str], forbidden_ids: list[str]) -> None:
    overlap = {base_id(i) for i in ids} & {base_id(i) for i in forbidden_ids}
    if overlap:
        raise ValueError(f"Audit prompts overlap historical data: {len(overlap)}; "
                         f"examples={sorted(overlap)[:5]}")


def fit_bins(values: np.ndarray, count: int) -> np.ndarray:
    if not 2 <= count <= 8:
        raise ValueError("Use 2..8 bins for the finite-alphabet information audit.")
    edges = np.quantile(np.asarray(values), np.arange(1, count) / count)
    if len(np.unique(edges)) != count - 1 or np.std(values) < 1e-10:
        raise ValueError("Degenerate module or target: quantile bins cannot be frozen.")
    return edges


def categorize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right")


def floor_probabilities(values: np.ndarray, floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    k = values.shape[1]
    if not 0 < floor < 1 / k:
        raise ValueError("Probability floor must lie between zero and 1/K.")
    values = np.maximum(values, 0)
    values = values / values.sum(axis=1, keepdims=True).clip(1e-30)
    return (1 - k * floor) * values + floor


def empirical_bernstein(values: np.ndarray, alpha: float, span: float) -> tuple[float, float]:
    """One-sided finite-sample empirical Bernstein bounds (Maurer--Pontil).

    Values must be independent, bounded in an interval of known length span,
    and produced by a predictor frozen before evaluation.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 or not 0 < alpha < 1 or not np.isfinite(values).all():
        raise ValueError("Invalid inputs to finite-sample concentration bound.")
    log_term = math.log(2 / alpha)
    radius = math.sqrt(2 * values.var(ddof=1) * log_term / n)
    radius += 7 * span * log_term / (3 * (n - 1))
    return float(values.mean() - radius), float(values.mean() + radius)


def entropy_lower_bound(labels: np.ndarray, k: int, alpha: float) -> float:
    """Minimize categorical entropy in simultaneous Hoeffding intervals.

    Entropy is concave, so the minimum is at a simplex-box vertex: all but
    at most one probability are on their lower/upper interval endpoints.
    """
    frequencies = np.bincount(labels, minlength=k) / len(labels)
    radius = math.sqrt(math.log(2 * k / alpha) / (2 * len(labels)))
    lower = np.maximum(0, frequencies - radius)
    upper = np.minimum(1, frequencies + radius)
    answer = math.log2(k)
    for free in range(k):
        fixed = [i for i in range(k) if i != free]
        for bits in itertools.product((0, 1), repeat=k - 1):
            p = np.zeros(k)
            p[fixed] = [upper[i] if bit else lower[i] for i, bit in zip(fixed, bits)]
            p[free] = 1 - p.sum()
            if lower[free] - 1e-12 <= p[free] <= upper[free] + 1e-12:
                positive = p[p > 0]
                answer = min(answer, float(-(positive * np.log2(positive)).sum()))
    return max(0., answer)


def club_blocks(labels: np.ndarray, probabilities: np.ndarray, seed: int) -> np.ndarray:
    """Independent two-prompt blocks for the joint-minus-product expectation.

    The swapped labels come from different independent prompts. They are not
    a conditional shuffle, a within-batch reuse or same-trajectory negatives.
    """
    order = np.random.default_rng(seed).permutation(len(labels))
    first, second = order[:2 * (len(order) // 2)].reshape(-1, 2).T
    logs = np.log2(probabilities)
    return .5 * (logs[first, labels[first]] + logs[second, labels[second]]
                 - logs[first, labels[second]] - logs[second, labels[first]])


def entropy_continuity_penalty(kl_bits: float, k: int) -> float:
    """Pinsker plus finite-alphabet entropy continuity; KL is in bits."""
    if kl_bits < 0:
        raise ValueError("Density KL budgets must be nonnegative.")
    tv = min(math.sqrt(math.log(2) * kl_bits / 2), 1 - 1 / k)
    binary_entropy = -(tv * math.log2(tv) + (1 - tv) * math.log2(1 - tv)) if tv > 0 else 0.
    return min(math.log2(k), tv * math.log2(k - 1) + binary_entropy)


def leakage_ceiling(bound: dict, budget: float) -> float:
    """Upper bound conditional on at least one retained density having KL<=budget."""
    return min(bound["unconditional_semantic_mi_upper_bound_bits"],
               max(0., bound["club_functional_envelope_ucb_bits"] + budget),
               max(0., bound["entropy_ceiling_envelope_ucb_bits"] +
                   entropy_continuity_penalty(budget, bound["bins"])))


def information_bounds(train_m: np.ndarray, train_u: np.ndarray, test_m: np.ndarray,
                       test_u: np.ndarray, densities: dict[str, np.ndarray],
                       k: int, floor: float, alpha: float, seed: int) -> dict:
    """I(C;V) lower bound and an upper bound on the fitted CLUB functional.

    C and V are train-binned module and target scores. Densities approximate
    p(C|S). For EVERY such q,
      I(C;S) <= CLUB(q) + E_S KL[p(C|S) || q(C|S)].
    A positive gap is therefore evidence conditional on a stated KL budget,
    never an unconditional semantic-freeness certificate.
    """
    if len(test_m) != len(test_u) or len(train_m) != len(train_u) or not densities:
        raise ValueError("Mismatched samples or an empty semantic density audit.")
    for name, value in densities.items():
        if value.shape != (len(test_m), k) or not np.isfinite(value).all():
            raise ValueError(f"Invalid conditional density {name}.")
    count = np.ones((k, k))
    np.add.at(count, (train_m, train_u), 1)
    target_table = floor_probabilities(count, floor)
    q_target = target_table[test_m]
    ce = -np.log2(q_target[np.arange(len(test_u)), test_u])
    alpha_part = alpha / (2 + 2 * len(densities))
    entropy_lcb = entropy_lower_bound(test_u, k, alpha_part)
    # The entire lookup table is frozen on train, so its exact range is known
    # before test. Using that range is tighter than the global probability floor.
    ce_span = float(np.ptp(-np.log2(target_table)))
    ce_ucb = empirical_bernstein(ce, alpha_part, ce_span)[1]
    lower = max(0., entropy_lcb - ce_ucb)
    p = np.bincount(test_u, minlength=k) / len(test_u)
    h = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    rows = []
    for name, q in densities.items():
        q = floor_probabilities(q, floor)
        block = club_blocks(test_m, q, seed)
        lo, hi = empirical_bernstein(block, alpha_part, -2 * math.log2(floor))
        nll = float(-np.log2(q[np.arange(len(test_m)), test_m]).mean())
        conditional_entropy = -(q * np.log2(q)).sum(1)
        entropy_lcb_q = empirical_bernstein(conditional_entropy, alpha_part, math.log2(k))[0]
        freq = np.bincount(test_m, minlength=k) / len(test_m)
        brier = np.square(q - np.eye(k)[test_m]).sum(1).mean()
        rows.append({"family": name, "club_functional_bits": float(block.mean()),
                     "club_functional_lcb_bits": lo, "club_functional_ucb_bits": hi,
                     "conditional_q_entropy_lcb_bits": entropy_lcb_q,
                     "entropy_ceiling_ucb_bits_at_zero_kl": math.log2(k) - entropy_lcb_q,
                     "density_test_nll_bits": nll, "density_brier_score": float(brier),
                     "class_marginal_max_abs_error": float(np.abs(q.mean(0) - freq).max()),
                     "independent_blocks": len(block)})
    # A maximum avoids gaining a positive result by choosing a weak audit probe.
    envelope = max(row["club_functional_ucb_bits"] for row in rows)
    result = {"state_information_lcb_bits": lower,
            "state_binned_variational_estimate_bits": h - float(ce.mean()),
            "target_entropy_lcb_bits": entropy_lcb, "target_cross_entropy_ucb_bits": ce_ucb,
            "target_cross_entropy_fixed_range_bits": ce_span,
            "club_functional_envelope_ucb_bits": envelope,
            "density_kl_break_even_bits": lower - envelope,
            "unconditional_semantic_mi_upper_bound_bits": math.log2(k),
            "unconditional_gap_lcb_bits": lower - math.log2(k),
            "entropy_ceiling_envelope_ucb_bits": max(r["entropy_ceiling_ucb_bits_at_zero_kl"] for r in rows),
            "bins": k,
            "density_models": rows, "alpha_for_this_comparison": alpha,
            "absolute_semantic_freeness_proven": False,
            "population": "one fixed generation step per independent base prompt",
            "upper_bound_status": "requires_additive_conditional_density_KL_budget",
            "scope": "train-binned frozen module score and the complete supplied semantic bank"}
    result["remaining_information_lcb_at_zero_density_kl"] = lower - leakage_ceiling(result, 0.)
    if result["remaining_information_lcb_at_zero_density_kl"] > 0:
        low_budget, high_budget = 0., math.log2(k)
        for _ in range(60):
            midpoint = (low_budget + high_budget) / 2
            if lower > leakage_ceiling(result, midpoint):
                low_budget = midpoint
            else:
                high_budget = midpoint
        result["max_supported_density_kl_budget_bits"] = low_budget
    else:
        result["max_supported_density_kl_budget_bits"] = None
    return result


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) / math.sqrt(2)


def make_probe(in_dim: int, out_dim: int, family: str, width: int, dropout: float) -> nn.Module:
    if family == "linear":
        return nn.Linear(in_dim, out_dim)
    if family == "mlp":
        return nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(width, width), nn.GELU(), nn.Linear(width, out_dim))
    if family == "deep_residual":
        return nn.Sequential(nn.Linear(in_dim, width), nn.GELU(),
                             *(ResidualBlock(width, dropout) for _ in range(3)),
                             nn.LayerNorm(width), nn.Linear(width, out_dim))
    raise ValueError(f"Unknown audit family {family}")


def predict(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        return torch.cat([model(x[i:i + batch_size].to(device)).float().cpu()
                          for i in range(0, len(x), batch_size)])


def fit_probes(x: torch.Tensor, y: torch.Tensor, splits: dict[str, np.ndarray],
               options: dict, output_dir: Path, label: str, classes: int = 0) -> dict:
    """Same capacity, split and optimizer contract for every compared input set.

    Validation alone selects epochs and probability temperature. Predictors
    stay frozen for the final test. No PCA or coordinate truncation is used.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(options["device"])
    train, val = splits["train"], splits["validation"]
    mean, scale = x[train].mean(0), x[train].std(0, unbiased=False).clamp_min(1e-6)
    x = ((x - mean) / scale).contiguous()
    y = y.long().view(-1) if classes else y.float().view(-1, 1)
    y_mean = torch.tensor(0.) if classes else y[train].mean()
    y_std = torch.tensor(1.) if classes else y[train].std(unbiased=False).clamp_min(1e-6)
    fit_y = y if classes else (y - y_mean) / y_std
    outputs, histories, states = {}, [], {}
    batch = options["batch_size"]
    families = options["families"]
    for family_index, family in enumerate(families):
        if family == "ensemble":
            continue
        torch.manual_seed(options["seed"] + family_index)
        model = make_probe(x.shape[1], classes or 1, family, options["hidden_dim"],
                           options["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"],
                                      weight_decay=options["weight_decay"])
        best, best_state, stale = float("inf"), None, 0
        gen = torch.Generator().manual_seed(options["seed"])
        bar = tqdm(range(1, options["epochs"] + 1), desc=f"{label} {family}", unit="epoch")
        for epoch in bar:
            order = train[torch.randperm(len(train), generator=gen).numpy()]
            model.train()
            total = 0.
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                prediction = model(x[idx].to(device))
                truth = fit_y[idx].to(device)
                loss = F.cross_entropy(prediction, truth) if classes else F.mse_loss(prediction, truth)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite {label} loss; no result exported.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            model.eval()
            pred_val = predict(model, x[val], batch, device)
            value = float(F.cross_entropy(pred_val, fit_y[val]) if classes
                          else F.mse_loss(pred_val, fit_y[val]))
            if value < best - 1e-7:
                best, stale = value, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            else:
                stale += 1
            histories.append({"family": family, "epoch": epoch,
                              "train_loss": total / len(train), "validation_loss": value})
            bar.set_postfix(train=f"{total / len(train):.4f}", val=f"{value:.4f}", best=best_epoch)
            if epoch == 1 or epoch % 5 == 0:
                print(f"[audit-probe] {label} {family} epoch={epoch} "
                      f"train={total / len(train):.5f} validation={value:.5f}", flush=True)
            if stale >= options["patience"]:
                break
        model.load_state_dict(best_state)
        model.eval()
        all_pred = predict(model, x, batch, device)
        if classes:
            temperatures = [.5, .75, 1., 1.5, 2., 3., 5.]
            temperature = min(temperatures, key=lambda t: float(F.cross_entropy(all_pred[val] / t, y[val])))
            all_pred = torch.softmax(all_pred / temperature, dim=1)
        else:
            temperature = None
            all_pred = all_pred * y_std + y_mean
        outputs[family] = all_pred.numpy()
        states[family] = {"state_dict": best_state, "input_mean": mean, "input_std": scale,
                          "target_mean": y_mean, "target_std": y_std,
                          "temperature": temperature, "best_epoch": best_epoch}
        del model, optimizer
    if "ensemble" in families:
        if len(outputs) < 2:
            raise ValueError("Ensemble needs at least two fitted member families.")
        outputs["ensemble"] = np.mean(list(outputs.values()), axis=0)
    if classes:
        val_loss = {k: float(-np.log(v[val, y[val].numpy()].clip(1e-12)).mean())
                    for k, v in outputs.items()}
    else:
        val_loss = {k: float(np.square(v[val] - y[val].numpy()).mean())
                    for k, v in outputs.items()}
    selected = min(val_loss, key=val_loss.get)
    torch.save(states, output_dir / "predictors.pt")
    write_json(output_dir / "fit_summary.json", {"selected_family": selected,
               "selection_rule": "minimum_validation_loss", "validation_losses": val_loss,
               "input_dim": x.shape[1], "history": histories})
    return {"predictions": outputs, "selected_family": selected}


def prompt_bootstrap(values: np.ndarray, groups: list[str], samples: int, seed: int) -> dict:
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=values) / counts
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for i in range(samples):
        boot[i] = means[rng.integers(0, len(means), len(means))].mean()
    # Centered bootstrap tests the mean without a symmetry assumption.
    p = (1 + np.count_nonzero(boot - means.mean() >= means.mean())) / (samples + 1)
    return {"mean_bits": float(means.mean()), "ci95_low_bits": float(np.quantile(boot, .025)),
            "ci95_high_bits": float(np.quantile(boot, .975)),
            "p_one_sided_centered_bootstrap": float(p), "n_prompts": len(unique),
            "n_rows": len(values), "inference": "prompt_bootstrap_fixed_predictors"}


def regression_comparison(target: np.ndarray, sem: np.ndarray, joint: np.ndarray,
                          validation: np.ndarray, test: np.ndarray, ids: list[str],
                          samples: int, seed: int) -> tuple[dict, np.ndarray]:
    sem, joint, target = sem.flatten(), joint.flatten(), target.flatten()
    vs = max(float(np.square(target[validation] - sem[validation]).mean()), 1e-8)
    vj = max(float(np.square(target[validation] - joint[validation]).mean()), 1e-8)
    # These are fitted Gaussian log-score gains, not distribution-free MI bounds.
    es, ej = np.square(target[test] - sem[test]), np.square(target[test] - joint[test])
    increment = .5 * (np.log(vs / vj) + es / vs - ej / vj) / np.log(2)
    result = prompt_bootstrap(increment, [base_id(ids[i]) for i in test], samples, seed)
    result.update({"semantic_mse": float(es.mean()), "semantic_plus_module_mse": float(ej.mean()),
                   "relative_mse_reduction": float(1 - ej.mean() / max(es.mean(), 1e-12)),
                   "semantic_variance_calibrated_on_validation": vs,
                   "joint_variance_calibrated_on_validation": vj,
                   "not_a_shannon_information_bound": True})
    return result, increment


DEFAULT_PROBES = {"families": ["linear", "mlp", "deep_residual", "ensemble"],
                  "epochs": 40, "patience": 10, "hidden_dim": 256, "dropout": .1,
                  "lr": .001, "weight_decay": .0001, "batch_size": 512,
                  "seed": 7319, "device": "cpu"}


def audit_bundle(bundle: dict, output_dir: Path, options: dict) -> dict:
    """Audit all frozen modules in a manifest using disjoint audit splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(i) for i in bundle["ids"]]
    assert_disjoint(ids, bundle["forbidden_ids"])
    splits = grouped_split(ids, options["seed"], options["train_fraction"], options["validation_fraction"])
    z = torch.as_tensor(bundle["module_scores"]).float()
    u = torch.as_tensor(bundle["target_scores"]).float()
    blocks = {k: torch.as_tensor(v).float() for k, v in bundle["semantics"].items()}
    if z.ndim != 2 or z.shape != u.shape or len(z) != len(ids):
        raise ValueError("Module/target scores must be aligned [rows, modules] arrays.")
    for name, value in {"module": z, "target": u, **blocks}.items():
        if len(value) != len(ids) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid/unaligned/nonfinite {name} data.")
    if "training_reference" not in blocks or len(blocks) < 2:
        raise ValueError("Require original reference plus an independent semantic audit bank.")
    if len(bundle["module_names"]) != z.shape[1] or len(set(bundle["module_names"])) != z.shape[1]:
        raise ValueError("Module names must be unique and match the score columns.")
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in bundle["module_names"]):
        raise ValueError("Module names must be safe local path components.")
    digests = {name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
               for name, value in {"module": z, "target": u, **blocks}.items()}
    contract = {"fingerprint": fingerprint({"ids": ids, "data": digests,
                 "options": options, "provenance": bundle.get("provenance", {})})}
    contract_path = output_dir / "audit_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Audit inputs/options changed. Use a new output directory.")
    write_json(contract_path, contract)
    steps = np.asarray(bundle["steps"], dtype=int)
    if len(steps) != len(ids) or not set(options["bound_steps"]).issubset(set(steps)):
        raise ValueError("Requested bound steps are absent from the aligned audit bundle.")
    base = blocks["training_reference"]
    # Full dimensions are retained. Step is an explicit positional control.
    base = torch.cat([base, torch.as_tensor(steps).float().view(-1, 1)], 1)
    bank = torch.cat([base] + [v for k, v in blocks.items() if k != "training_reference"], 1)
    write_json(output_dir / "split_manifest.json", {k: [ids[i] for i in v] for k, v in splits.items()})
    k, floor = options["bins"], options["probability_floor"]
    probes = {**DEFAULT_PROBES, **options["probes"], "seed": options["seed"]}
    test, val, train = (splits[s] for s in ("test", "validation", "train"))
    families_n = len(bundle["module_names"]) * len(options["bound_steps"])
    # Global over all planned modules and steps. Every per-bound component
    # receives a further Bonferroni allocation in information_bounds().
    effective_alpha = options["alpha"] / max(1, options.get("planned_comparisons", families_n))
    results, capacity_rows, predictions = [], [], {}
    for m, name in enumerate(bundle["module_names"]):
        module_dir = output_dir / name
        done_path = module_dir / "audit_result.json"
        if done_path.exists():
            print(f"[audit] reuse completed {name}", flush=True)
            saved = json.loads(done_path.read_text())
            results.append(saved)
            continue
        print(f"[audit] module={name} fresh_prompts={len({base_id(i) for i in ids})} "
              f"rows={len(ids)} semantic_dimensions={bank.shape[1]}", flush=True)
        module_result = {"module": name, "continuous_audits": {}, "bounds": []}
        for bank_name, x in (("training_reference", base), ("expanded_bank", bank)):
            fit_sem = fit_probes(x, u[:, m], splits, probes, module_dir / bank_name / "semantic",
                                 f"{name}/{bank_name}/U|S")
            fit_joint = fit_probes(torch.cat([x, z[:, m:m + 1]], 1), u[:, m], splits, probes,
                                   module_dir / bank_name / "joint", f"{name}/{bank_name}/U|S,M")
            sf, jf = fit_sem["selected_family"], fit_joint["selected_family"]
            sem, joint = fit_sem["predictions"][sf], fit_joint["predictions"][jf]
            comparison, per_row = regression_comparison(u[:, m].numpy(), sem, joint, val, test,
                                                         ids, options["bootstrap_samples"], options["seed"])
            comparison.update({"semantic_family": sf, "joint_family": jf})
            module_result["continuous_audits"][bank_name] = comparison
            for family in fit_sem["predictions"]:
                row, _ = regression_comparison(u[:, m].numpy(), fit_sem["predictions"][family],
                                                fit_joint["predictions"][family], val, test, ids,
                                                options["bootstrap_samples"], options["seed"])
                capacity_rows.append({"module": name, "bank": bank_name, "family": family, **row})
            predictions[f"{name}/{bank_name}"] = {"test_ids": [ids[i] for i in test],
                                                    "target": u[test, m], "semantic": sem[test],
                                                    "joint": joint[test], "increment_bits": per_row}
            print(f"[audit] {name} {bank_name} incremental_log_score={comparison['mean_bits']:.5f} "
                  f"CI=[{comparison['ci95_low_bits']:.5f},{comparison['ci95_high_bits']:.5f}] "
                  f"MSE={comparison['semantic_mse']:.5f}->{comparison['semantic_plus_module_mse']:.5f}", flush=True)
        for step in options["bound_steps"] if options.get("run_information_bounds", True) else []:
            idx = np.flatnonzero(steps == step)
            if len({base_id(ids[i]) for i in idx}) != len(idx):
                raise ValueError("Finite-sample bounds need one observation per prompt and step.")
            lookup = {old: new for new, old in enumerate(idx)}
            split_step = {role: np.array([lookup[i] for i in indices if i in lookup], dtype=int)
                          for role, indices in splits.items()}
            if min(map(len, split_step.values())) < options["min_bound_prompts"]:
                module_result["bounds"].append({"step": step, "status": "insufficient_independent_prompts",
                                                "split_counts": {s: len(v) for s, v in split_step.items()}})
                continue
            tr, te = split_step["train"], split_step["test"]
            zm, um = z[idx, m].numpy(), u[idx, m].numpy()
            try:
                me, ue = fit_bins(zm[tr], k), fit_bins(um[tr], k)
            except ValueError as exc:
                module_result["bounds"].append({"step": step, "status": "degenerate", "reason": str(exc)})
                continue
            c, v = categorize(zm, me), categorize(um, ue)
            density = fit_probes(bank[idx], torch.as_tensor(c), split_step, probes,
                                  module_dir / f"bound_step_{step}" / "density", f"{name}/C|S/step{step}", k)
            densities = {f: p[te] for f, p in density["predictions"].items()}
            bound = information_bounds(c[tr], v[tr], c[te], v[te], densities, k, floor,
                                        effective_alpha, options["seed"] + step)
            bound.update({"step": step, "n_test_prompts": len(te), "module_bin_edges": me.tolist(),
                          "target_bin_edges": ue.tolist(), "status": "estimated"})
            budget = options.get("assumed_density_kl_budget_bits")
            gap = bound["remaining_information_lcb_at_zero_density_kl"]
            bound["assumed_density_kl_budget_bits"] = budget
            bound["positive_gap_under_assumed_budget"] = budget is not None and bound["state_information_lcb_bits"] > leakage_ceiling(bound, budget)
            bound["sensitivity"] = [{"density_kl_budget_bits": b,
                                     "semantic_mi_ceiling_bits_under_budget": leakage_ceiling(bound, b),
                                     "remaining_information_lcb_bits": bound["state_information_lcb_bits"] - leakage_ceiling(bound, b)}
                                     for b in options["density_kl_budgets_bits"]]
            module_result["bounds"].append(bound)
            predictions[f"{name}/bound_step_{step}"] = {"test_ids": [ids[idx[i]] for i in te],
                 "c_test": c[te], "v_test": v[te], "c_train": c[tr], "v_train": v[tr],
                 "densities": densities}
            print(f"[audit-bound] {name} step={step} state_LCB={bound['state_information_lcb_bits']:.5f} "
                  f"CLUB_functional_UCB={bound['club_functional_envelope_ucb_bits']:.5f} "
                  f"zero_KL_gap={gap:.5f} max_density_KL_budget={bound['max_supported_density_kl_budget_bits']} "
                  f"assumed_budget={budget}", flush=True)
        write_json(module_dir / "capacity_curve.json", {"rows": [r for r in capacity_rows if r["module"] == name]})
        torch.save({key: value for key, value in predictions.items() if key.startswith(name + "/")},
                   module_dir / "heldout_predictions.pt")
        write_json(done_path, module_result)
        results.append(module_result)
        predictions.clear()
    summary = {"schema_version": 1, "modules": results, "source": bundle.get("provenance", {}),
               "options": options, "interpretation": {
                 "continuous": "held-out Gaussian predictive log-score gains, not Shannon MI bounds",
                 "bounds": "finite-sample binned-state lower bound versus CLUB and entropy-continuity semantic ceilings",
                 "density_error": "semantic ceilings require a justified conditional-density KL budget; calibration alone cannot certify it",
                 "absolute_semantic_freeness_proven": False}}
    write_json(output_dir / "semantic_leakage_audit_summary.json", summary)
    return summary
