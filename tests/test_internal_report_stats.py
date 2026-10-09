"""Degenerate frozen-neuron labels must not turn a diagnostic into a pipeline gate."""

import numpy as np

from metacog.audits.internal_report_stats import direct_report_classification, predict_checks


def example_rows(n_above: int):
    labels = [True] * n_above + [False] * (160 - n_above)
    rows = [{"recorded_high": label, "report_high_logodds": float(i % 19) / 10}
            for i, label in enumerate(labels)]
    rng = np.random.default_rng(1)
    return rows, rng.normal(size=(160, 12)), rng.normal(size=(160, 1))


def test_one_class_keeps_regression_but_marks_classification_unavailable():
    rows, semantic, meta = example_rows(0)
    checks = predict_checks(rows, semantic, meta, 7319)
    assert checks["activation_label_counts"] == {"below_or_equal": 160, "above": 0}
    assert checks["module_added_activation_log_loss_bits"] is None
    assert checks["module_added_report_r2"] is not None
    report = direct_report_classification(np.zeros(160), np.arange(160), 7319)
    assert report["status"].startswith("not_estimable")
    assert report["auc"] is None


def test_rare_label_uses_stratified_split():
    rows, semantic, meta = example_rows(2)
    checks = predict_checks(rows, semantic, meta, 7319)
    assert checks["activation_classification_status"] == "estimable"
    assert checks["module_added_activation_log_loss_bits"] is not None
