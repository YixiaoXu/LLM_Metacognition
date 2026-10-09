"""Memory-efficient decoder-layer capture independent of Transformers internals."""

from __future__ import annotations

import inspect
from typing import Dict, Sequence

import torch


def get_decoder_layers(model) -> Sequence[torch.nn.Module]:
    candidates = [
        lambda m: getattr(getattr(m, "model", None), "layers", None),
        lambda m: getattr(
            getattr(getattr(m, "language_model", None), "model", None),
            "layers",
            None,
        ),
        lambda m: getattr(
            getattr(getattr(m, "model", None), "decoder", None), "layers", None
        ),
        lambda m: getattr(getattr(m, "decoder", None), "layers", None),
        lambda m: getattr(getattr(m, "transformer", None), "h", None),
        lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None),
    ]
    for getter in candidates:
        layers = getter(model)
        if layers is not None:
            return layers
    raise AttributeError(
        "Cannot find decoder layers. Expected model.layers, model.decoder.layers, "
        "transformer.h, or gpt_neox.layers."
    )


def _layer_hidden_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(hidden):
        return hidden
    raise TypeError(f"Cannot extract a hidden-state tensor from {type(output).__name__}.")


class DecoderLayerCapture:
    """Capture only requested decoder outputs without retaining every layer."""

    def __init__(self, model, layer_ids: Sequence[int]):
        decoder_layers = get_decoder_layers(model)
        self.layer_ids = sorted({int(layer) for layer in layer_ids})
        invalid = [layer for layer in self.layer_ids if not 0 <= layer < len(decoder_layers)]
        if invalid:
            raise ValueError(f"Decoder layer ids out of range: {invalid}")
        self._layers = {layer: decoder_layers[layer] for layer in self.layer_ids}
        self._handles = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def clear(self) -> None:
        self.outputs.clear()

    def _hook(self, layer_id: int):
        def capture(_module, _inputs, output) -> None:
            self.outputs[layer_id] = _layer_hidden_tensor(output)

        return capture

    def __enter__(self):
        self.clear()
        self._handles = [
            module.register_forward_hook(self._hook(layer_id))
            for layer_id, module in self._layers.items()
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()


def last_token_logits_kwargs(model) -> Dict[str, int]:
    """Use a model's supported last-logit option, falling back conservatively."""
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: 1}
    return {}
