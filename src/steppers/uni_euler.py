import torch

from src.steppers.base import BaseStepper, StepperPayload
from src.utils.conditioning import ModelCondition


class UniEuler(BaseStepper):
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
        v_curr = self.velocity_with_side_cache(model, x, t_curr, model_condition)
        x_corr = x - v_curr * dt
        v_next = self.velocity_fresh_cache(model, x_corr, t_next, model_condition)
        x_new = x - v_next * dt
        return StepperPayload(x=x_new, v=v_next)
