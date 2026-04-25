from __future__ import annotations

import abc
from typing import Any

from omegaconf import DictConfig

import torch


class AttentionControllerBase(abc.ABC):
    @abc.abstractmethod
    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def build(self, *, handler: Any, cfg: DictConfig) -> None:
        return None

    def __call__(self, attn_weight: torch.Tensor) -> torch.Tensor:
        return self.forward(attn_weight)


class DummyAttentionController(AttentionControllerBase):
    def build(self, cfg, handler):
        pass
    
    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        return attn_weight
