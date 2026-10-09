"""Fresh labels and resume contracts must not depend on historical target activations."""

import json

import numpy as np
import pytest

from metacog.audits.neuron_report_measurement import measure_report_activations


def rows(count=5):
    return [{"id": f"p{i}::step000", "prompt_token_ids": [1, 10+i]} for i in range(count)]


def test_measurements_come_from_current_prefix_not_old_cache(tmp_path):
    measured = measure_report_activations(
        tmp_path, rows(), 2, {"layer": 14}, lambda prefix: [prefix[-1], -prefix[-1]])
    np.testing.assert_array_equal(measured[:, 0], [10, 11, 12, 13, 14])
    assert measured.dtype == np.float32
    with np.load(tmp_path / "runtime_activation_measurements.npz", allow_pickle=False) as saved:
        assert saved["n_complete"].item() == 5
        assert saved["ids"].tolist() == [row["id"] for row in rows()]
    assert json.loads((tmp_path / "runtime_activation_contract.json").read_text())["width"] == 2


def test_complete_measurements_resume_without_new_forwards(tmp_path):
    first = measure_report_activations(tmp_path, rows(), 2, {}, lambda prefix: [prefix[-1], 0])
    def forbidden(_prefix):
        raise AssertionError("Completed measurements must not be recomputed")
    second = measure_report_activations(tmp_path, rows(), 2, {}, forbidden)
    np.testing.assert_array_equal(first, second)


def test_partial_checkpoint_preserves_row_alignment_on_resume(tmp_path):
    def interrupt(prefix):
        if prefix[-1] == 13:
            raise RuntimeError("interrupted")
        return [prefix[-1], 0]
    with pytest.raises(RuntimeError, match="interrupted"):
        measure_report_activations(tmp_path, rows(), 2, {}, interrupt, checkpoint_rows=2)
    seen = []
    def continuation(prefix):
        seen.append(prefix[-1])
        return [prefix[-1], 0]
    result = measure_report_activations(tmp_path, rows(), 2, {}, continuation)
    assert seen == [12, 13, 14]
    np.testing.assert_array_equal(result[:, 0], [10, 11, 12, 13, 14])


@pytest.mark.parametrize("change", ["prefix", "execution"])
def test_changed_prefix_or_execution_cannot_reuse_measurements(tmp_path, change):
    original = rows()
    measure_report_activations(tmp_path, original, 2, {"dtype": "bfloat16"}, lambda _: [1, 2])
    if change == "prefix":
        original[0]["prompt_token_ids"][0] = 99
    with pytest.raises(ValueError, match="contract changed"):
        measure_report_activations(tmp_path, original, 2,
                                  {"dtype": "float32" if change == "execution" else "bfloat16"},
                                  lambda _: [1, 2])


def test_nonfinite_measurements_and_duplicate_prompts_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid fresh activation"):
        measure_report_activations(tmp_path, rows(), 2, {}, lambda _: [np.nan, 0])
    duplicate = [rows()[0], rows()[0]]
    with pytest.raises(ValueError, match="duplicate prompt"):
        measure_report_activations(tmp_path, duplicate, 2, {}, lambda _: [0, 0])
