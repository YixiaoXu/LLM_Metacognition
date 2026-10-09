"""Statistics for the held-out internal-activation report diagnostic."""

import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def predict_checks(rows: list[dict], semantic: np.ndarray, meta: np.ndarray, seed: int) -> dict:
    n = len(rows)
    if n < 60:
        raise ValueError("At least 60 measured prompts are needed for a report audit")
    y = np.array([int(row["recorded_high"]) for row in rows])
    report = np.array([float(row["report_high_logodds"]) for row in rows])
    counts = np.bincount(y, minlength=2)
    if counts.min() >= 2:
        train, test = train_test_split(np.arange(n), test_size=0.4, stratify=y, random_state=seed)
        activation_status = "estimable"
    else:
        order = np.random.default_rng(seed).permutation(n)
        train, test = order[: int(0.6 * n)], order[int(0.6 * n):]
        activation_status = "not_estimable_fewer_than_two_in_one_class"

    scaler = StandardScaler().fit(semantic[train])
    sem = scaler.transform(semantic)
    pca = PCA(n_components=min(32, len(train) - 2, sem.shape[1]), random_state=seed).fit(sem[train])
    sem = pca.transform(sem)
    both = np.column_stack([sem, StandardScaler().fit(meta[train]).transform(meta)])
    results = {}
    for name, x in (("semantic_only", sem), ("semantic_plus_module", both)):
        regressor = Ridge(alpha=10.0).fit(x[train], report[train])
        value = {"reported_score_r2": float(r2_score(report[test], regressor.predict(x[test]))),
                 "activation_log_loss_bits": None, "activation_auc": None}
        if activation_status == "estimable":
            classifier = LogisticRegression(C=0.1, max_iter=1000).fit(x[train], y[train])
            prob = classifier.predict_proba(x[test])[:, 1].clip(1e-6, 1 - 1e-6)
            value["activation_log_loss_bits"] = float(log_loss(y[test], prob, labels=[0, 1]) / math.log(2))
            value["activation_auc"] = float(roc_auc_score(y[test], prob))
        results[name] = value
    return {
        "n_fit": len(train), "n_heldout": len(test), "activation_label_counts":
            {"below_or_equal": int(counts[0]), "above": int(counts[1])},
        "activation_classification_status": activation_status,
        "model_checks": results,
        "module_added_activation_log_loss_bits": (
            results["semantic_only"]["activation_log_loss_bits"] -
            results["semantic_plus_module"]["activation_log_loss_bits"]
            if activation_status == "estimable" else None),
        "module_added_report_r2": results["semantic_plus_module"]["reported_score_r2"] -
            results["semantic_only"]["reported_score_r2"],
    }


def direct_report_classification(truth: np.ndarray, scores: np.ndarray, seed: int) -> dict:
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)
    counts = np.bincount(truth, minlength=2)
    result = {"n_below_or_equal": int(counts[0]), "n_above": int(counts[1]),
              "balanced_accuracy": None, "auc": None, "auc_bootstrap_ci95": None,
              "auc_permutation_one_sided_p": None}
    if counts.min() < 2:
        result["status"] = "not_estimable_fewer_than_two_in_one_class"
        return result
    result["status"] = "estimable"
    result["balanced_accuracy"] = float(balanced_accuracy_score(truth, scores > 0))
    result["auc"] = float(roc_auc_score(truth, scores))
    rng = np.random.default_rng(seed)
    boot_auc = []
    for _ in range(1000):
        sample = rng.integers(0, len(truth), len(truth))
        if len(np.unique(truth[sample])) == 2:
            boot_auc.append(roc_auc_score(truth[sample], scores[sample]))
    result["auc_bootstrap_ci95"] = [float(x) for x in np.quantile(boot_auc, [0.025, 0.975])]
    null_auc = [roc_auc_score(rng.permutation(truth), scores) for _ in range(1000)]
    result["auc_permutation_one_sided_p"] = float(
        (1 + sum(value >= result["auc"] for value in null_auc)) / (1 + len(null_auc)))
    return result
