import torch

from src.steppers.base import BaseStepper, StepperPayload
from src.utils.conditioning import ModelCondition


class Heun(BaseStepper):
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
        v1 = self.velocity_with_side_cache(model, x, t_curr, model_condition)
        x_pred = x - v1 * dt
        v2 = self.velocity_fresh_cache(model, x_pred, t_next, model_condition)
        v_avg = 0.5 * (v1 + v2)
        return StepperPayload(x=x - v_avg * dt, v=v_avg)
