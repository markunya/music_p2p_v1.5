import torch

from src.steppers.base import BaseStepper, StepperPayload
from src.utils.conditioning import ModelCondition


class Euler(BaseStepper):
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
        v = self.velocity_with_side_cache(model, x, t_curr, model_condition)
        return StepperPayload(x=x - v * dt, v=v)
