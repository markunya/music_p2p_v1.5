"""Placeholder for P2P-style cross-attention replacement (to be implemented)."""

from __future__ import annotations

import torch

from src.attention_injection.controllers.base import AttentionControllerBase


class ReplacementAttentionController(AttentionControllerBase):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        _ = (args, kwargs)

    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ReplacementAttentionController: not implemented yet; use controller=dummy or reweight")
