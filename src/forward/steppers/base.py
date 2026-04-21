"""Локальный шаг диффузии: общий ODE-драйвер вызывает ``step`` на каждой паре (t_curr, t_prev)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class LocalDiffusionStepper(ABC):
    """Один шаг \(x_{t_\text{curr}} \to x_{t_\text{prev}}\) (Euler ODE); цикл — в ``diffusion_driver``."""

    @abstractmethod
    def step(self, xt: torch.Tensor, t_curr: float, t_prev: float, step_idx: int) -> torch.Tensor:
        """Вернуть состояние на времени ``t_prev``."""
