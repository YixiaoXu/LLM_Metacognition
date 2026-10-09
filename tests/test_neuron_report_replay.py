"""A small random decoder checks replay and cached propagation, without downloaded weights."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from metacog.audits.neuron_report_replay import (  # noqa: E402
    generation_step_hidden_state, generation_step_report_forward,
)


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = transformers.LlamaConfig(vocab_size=128, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2,
                                      num_key_value_heads=2, max_position_embeddings=128)
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


def test_question_suffix_does_not_recompute_the_measured_prefix(model):
    layer = model.model.layers[0]
    prefix = [2, 3, 4, 5]
    logits, first = generation_step_report_forward(model, layer, "cpu", prefix+[6, 7], 0, 3)
    _, second = generation_step_report_forward(model, layer, "cpu", prefix+[8, 9, 10], 0, 3)
    assert first["before"] == second["before"]
    with torch.inference_mode():
        full = model(input_ids=torch.tensor([prefix+[6, 7]]), use_cache=False).logits[0, -1]
    torch.testing.assert_close(logits, full, atol=1e-5, rtol=1e-5)
    assert not layer._forward_hooks


def test_fresh_hidden_measurement_and_reporting_use_identical_state(model):
    layer = model.model.layers[0]
    prefix = [2, 3, 4, 5]
    hidden = generation_step_hidden_state(model, layer, "cpu", prefix)
    for suffix in ([6, 7], [8, 9, 10]):
        for neuron in (0, 3, 15):
            _, state = generation_step_report_forward(model, layer, "cpu", prefix+suffix, neuron, 3)
            assert state["before"] == hidden[neuron].item()
    assert not layer._forward_hooks


def test_neuron_flip_is_retained_in_the_question_cache(model):
    layer = model.model.layers[0]
    ids = [2, 3, 4, 5, 6, 7]
    before, state = generation_step_report_forward(model, layer, "cpu", ids, 0, 3)
    target = state["before"] + 1.0
    after, patched = generation_step_report_forward(model, layer, "cpu", ids, 0, 3, target)
    assert patched["before"] == state["before"]
    assert patched["after"] == pytest.approx(target)
    assert not torch.allclose(before, after, atol=1e-7, rtol=1e-7)
    assert not layer._forward_hooks
