"""Replay a generated token before appending an internal-state question."""

from __future__ import annotations

import torch

from metacog.models.capture import last_token_logits_kwargs


def _generation_step_state(model, layer, device, prefix: list[int],
                           neuron: int | None = None, patch_value: float | None = None):
    if len(prefix) < 2:
        raise ValueError("Generation-step reporting requires a prompt and generated token")
    observed = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        observed["hidden"] = hidden[0, -1].detach().float().clone()
        if patch_value is None:
            return None
        if neuron is None:
            raise ValueError("A neuron index is required for a single-neuron patch")
        updated = hidden.clone()
        updated[0, -1, neuron] = patch_value
        observed["after"] = float(updated[0, -1, neuron].float().item())
        return (updated, *output[1:]) if isinstance(output, tuple) else updated

    kwargs = last_token_logits_kwargs(model)
    with torch.inference_mode():
        prompt = torch.tensor([prefix[:-1]], dtype=torch.long, device=device)
        prefill = model(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        use_cache=True, **kwargs)
        past = prefill.past_key_values
        del prefill
        handle = layer.register_forward_hook(hook)
        try:
            generated = model(
                input_ids=torch.tensor([[prefix[-1]]], dtype=torch.long, device=device),
                attention_mask=torch.ones((1, len(prefix)), dtype=torch.long, device=device),
                past_key_values=past, use_cache=True, **kwargs,
            )
        finally:
            handle.remove()
        if "hidden" not in observed:
            raise RuntimeError("Generated-token replay did not capture the requested layer")
        past = generated.past_key_values
        del generated
    return observed, past


def generation_step_hidden_state(model, layer, device, prefix: list[int]):
    observed, _past = _generation_step_state(model, layer, device, prefix)
    return observed["hidden"].cpu()


def generation_step_report_forward(model, layer, device, ids: list[int], neuron: int,
                                   marker: int, patch_value: float | None = None):
    prefix, suffix = ids[:marker + 1], ids[marker + 1:]
    if not suffix:
        raise ValueError("Generation-step reporting requires a diagnostic question")
    observed, past = _generation_step_state(model, layer, device, prefix, neuron, patch_value)
    hidden = observed.pop("hidden")
    observed["before"] = float(hidden[neuron].item())
    observed["hidden_norm"] = float(hidden.norm().item())
    observed.setdefault("after", observed["before"])
    with torch.inference_mode():
        # The question consumes the cached prefix without recomputing its activation.
        result = model(
            input_ids=torch.tensor([suffix], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device),
            past_key_values=past, use_cache=True, **last_token_logits_kwargs(model),
        )
        logits = result.logits[0, -1].float().clone()
    return logits, observed
