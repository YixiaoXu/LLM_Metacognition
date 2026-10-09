#!/usr/bin/env python
"""Runtime components for causal edits in a refined residual-code space."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class ModuleRefiner(nn.Module):
    def __init__(self, input_dim: int, code_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
        )
        self.head = nn.Linear(code_dim, output_dim)

    def forward(
        self, meta: torch.Tensor, base_delta: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.adapter(meta)
        return code, self.head(code)


class TaskAlignedModuleRefiner(ModuleRefiner):
    """Expose only the residual prediction projected onto a module loading.

    The internal bottleneck can remain expressive, but clustering and causal
    editing see a one-dimensional, task-aligned coordinate. This prevents
    unused bottleneck dimensions from carrying arbitrary semantic variation.
    """

    def __init__(
        self,
        input_dim: int,
        internal_code_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        module_loading: torch.Tensor,
    ):
        super().__init__(
            input_dim,
            internal_code_dim,
            hidden_dim,
            output_dim,
            dropout,
        )
        loading = torch.as_tensor(module_loading).float().view(1, -1)
        if loading.size(1) != output_dim:
            raise ValueError(
                f"module_loading dim {loading.size(1)} does not match output_dim {output_dim}"
            )
        self.register_buffer("module_loading", loading)

    def forward(
        self, meta: torch.Tensor, base_delta: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        internal_code = self.adapter(meta)
        prediction = self.head(internal_code)
        loading = self.module_loading.to(prediction)
        denominator = loading.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
        task_code = (prediction * loading).sum(dim=1, keepdim=True) / denominator
        return task_code, prediction


class AffineCodeRefiner(nn.Module):
    """Apply a train-fitted scalar calibration to an exported module code.

    The wrapped prediction head remains unchanged. Only the code used for
    module discovery and causal intervention is calibrated, so runtime code
    exactly matches the conditional-incremental code evaluated during export.
    """

    def __init__(
        self,
        base: nn.Module,
        scale: torch.Tensor | float,
        bias: torch.Tensor | float,
    ):
        super().__init__()
        self.base = base
        self.register_buffer("code_affine_scale", torch.as_tensor(scale).float())
        self.register_buffer("code_affine_bias", torch.as_tensor(bias).float())

    def forward(
        self, meta: torch.Tensor, base_delta: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        code, prediction = self.base(meta, base_delta)
        scale = self.code_affine_scale.to(code).view(1, -1)
        bias = self.code_affine_bias.to(code).view(1, -1)
        return code * scale + bias, prediction


def load_module_refiner(path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mode = str(payload.get("refinement_mode", "module"))
    if mode == "task_aligned_module":
        model = TaskAlignedModuleRefiner(
            int(payload["input_dim"]),
            int(payload.get("internal_code_dim", payload["code_dim"])),
            int(payload["hidden_dim"]),
            int(payload["output_dim"]),
            float(payload.get("dropout", 0.0)),
            torch.as_tensor(payload["module_loading"]),
        ).to(device)
    else:
        model = ModuleRefiner(
            int(payload["input_dim"]),
            int(payload["code_dim"]),
            int(payload["hidden_dim"]),
            int(payload["output_dim"]),
            float(payload.get("dropout", 0.0)),
        ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if "code_affine_scale" in payload or "code_affine_bias" in payload:
        model = AffineCodeRefiner(
            model,
            payload.get("code_affine_scale", 1.0),
            payload.get("code_affine_bias", 0.0),
        ).to(device)
    model.eval()
    return model, payload


def runtime_code(
    refiner: nn.Module,
    meta: torch.Tensor,
) -> torch.Tensor:
    code, _ = refiner(meta)
    return code
