from __future__ import annotations

import torch
from loguru import logger

from src.steppers.guidance import GuidanceStepper, StepperPayload
from src.utils.conditioning import ModelCondition


class UniGuidanceStepper(GuidanceStepper):
    def step(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        *,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
        model_condition: ModelCondition,
    ) -> StepperPayload:
        dt = self._segment_dt_tensor(
            dt=t_curr - t_next,
            bsz=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        v_curr = super().step(model, x, t_curr=t_curr, t_next=t_next, model_condition=model_condition).v
        x_corr = x - v_curr * dt
        v_next = super().step(model, x_corr, t_curr=t_next, t_next=t_next, model_condition=model_condition).v
        x_new = x - v_next * dt
        
        return StepperPayload(x=x_new, v=v_next)
