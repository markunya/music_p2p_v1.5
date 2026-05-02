from abc import ABC, abstractmethod
from typing import Tuple

import torch

from src.attention_injection.validate import assert_row_stochastic


class InjectionMapper(ABC):

    def __init__(self, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> None:
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._matrix = self._build_matrix().to(device=self._device, dtype=self._dtype)
        assert_row_stochastic(self._matrix)

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self._matrix.shape[0]), int(self._matrix.shape[1]))

    @abstractmethod
    def _build_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, attn_src: torch.Tensor) -> torch.Tensor:
        m_t = self._matrix.to(device=attn_src.device, dtype=attn_src.dtype).t()
        return torch.matmul(attn_src, m_t)
