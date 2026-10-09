from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from metacog.models.capture import (  # noqa: E402
    DecoderLayerCapture,
    last_token_logits_kwargs,
)


class _Block(torch.nn.Module):
    def __init__(self, increment: float):
        super().__init__()
        self.increment = increment

    def forward(self, hidden):
        return (hidden + self.increment, None)


class _Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(1.0), _Block(2.0), _Block(3.0)])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Core()

    def forward(self, hidden, logits_to_keep=0):
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


def test_selected_layer_capture_matches_layer_outputs() -> None:
    model = _Model()
    initial = torch.tensor([[5.0]])
    with DecoderLayerCapture(model, [0, 2]) as capture:
        result = model(initial)
        assert torch.equal(capture.outputs[0], torch.tensor([[6.0]]))
        assert torch.equal(capture.outputs[2], result)
        assert 1 not in capture.outputs
    assert capture.outputs == {}


def test_last_token_logits_capability_detection() -> None:
    assert last_token_logits_kwargs(_Model()) == {"logits_to_keep": 1}
