import torch

from src.steppers.base import BaseStepper, StepperPayload
from src.utils.conditioning import ModelCondition


class UniHeun(BaseStepper):
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

        v_lo = self.velocity_fresh_cache(model, x_corr, t_next, model_condition)
        x_e = x_corr - v_lo * dt
        v_hi = self.velocity_fresh_cache(model, x_e, t_curr, model_condition)
        v_heun = 0.5 * (v_lo + v_hi)

        x_new = x - v_heun * dt
        return StepperPayload(x=x_new, v=v_heun)
