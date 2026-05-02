import abc
from typing import Any

from omegaconf import DictConfig

import torch


class AttentionControllerBase(abc.ABC):
    @abc.abstractmethod
    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def build(self, *, handler: Any, cfg: DictConfig, writer: Any | None = None) -> None:
        _ = writer
        return None

    def __call__(self, attn_weight: torch.Tensor) -> torch.Tensor:
        return self.forward(attn_weight)


class DummyAttentionController(AttentionControllerBase):
    def build(self, *, handler: Any, cfg: DictConfig, writer: Any | None = None) -> None:
        _ = (handler, cfg, writer)
        pass
    
    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        return attn_weight
