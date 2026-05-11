from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import Adam

from src.steppers.base import BaseStepper, StepperPayload
from src.utils.conditioning import ModelCondition


class OptStepper(BaseStepper):
    """One ``init_stepper`` step, then optimize ``delta`` so ``opt_stepper`` with swapped time fits ``x``.

    Returns ``x_init + delta`` (detached); ``v`` from the init step.
    Inner optimization can add i.i.d. Gaussian noise (mean 0, variance ``reg_noise_variance``) to ``x_opt`` each step.
    """

    def __init__(
        self,
        init_stepper: BaseStepper,
        opt_stepper: BaseStepper,
        lr: float,
        num_steps: int,
        norm_boundary: float = 0.05,
        epsilon: float = 1e-8,
        lr_end: float | None = None,
        reg_noise_std: float = 0.0,
        use_best_delta: bool = True,
    ) -> None:
        self._init_stepper = init_stepper
        self._opt_stepper = opt_stepper
        self._lr = float(lr)
        self._lr_end = float(lr_end) if lr_end is not None else float(lr)
        self._num_steps = int(num_steps)
        self._norm_boundary = float(norm_boundary)
        self._epsilon = float(epsilon)
        self._reg_noise_std = float(reg_noise_std)
        self._use_best_delta = bool(use_best_delta)

    def _collapse_if_cfg(self, stepper: BaseStepper, cond: ModelCondition) -> None:
        from src.steppers.guidance import CFG_GUIDANCE_STEPPERS

        if isinstance(stepper, CFG_GUIDANCE_STEPPERS):
            stepper.collapse_cfg_batch_layout(cond)

    def _opt_forward_x(
        self,
        model: torch.nn.Module,
        x_opt: torch.Tensor,
        *,
        t_lo: torch.Tensor,
        t_hi: torch.Tensor,
        cond_base: ModelCondition,
    ) -> torch.Tensor:
        cond = cond_base.clone()
        x_pred = self._opt_stepper.step(
            model,
            x_opt,
            t_curr=t_lo,
            t_next=t_hi,
            model_condition=cond,
        ).x
        self._collapse_if_cfg(self._opt_stepper, cond)
        return x_pred

    def step(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        *,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
        model_condition: ModelCondition,
    ) -> StepperPayload:
        model_condition.past_key_values = None
        cond_init = model_condition.clone()
        cond_opt = model_condition.clone()

        with torch.no_grad():
            payload0 = self._init_stepper.step(
                model,
                x,
                t_curr=t_curr,
                t_next=t_next,
                model_condition=cond_init,
            )
            self._collapse_if_cfg(self._init_stepper, cond_init)

        x_init = payload0.x.detach()
        v_out = payload0.v.detach()
        x_ref = x.detach()

        if self._num_steps <= 0:
            return StepperPayload(x=x_init, v=v_out)

        t_lo, t_hi = t_next, t_curr

        if self._epsilon > 0.0:
            with torch.no_grad():
                x_pred0 = self._opt_forward_x(
                    model, x_init, t_lo=t_lo, t_hi=t_hi, cond_base=cond_opt
                )
                m0 = float(F.mse_loss(x_pred0.float(), x_ref.float()).item())
            if m0 < self._epsilon:
                return StepperPayload(x=x_init, v=v_out)

        # Inversion / sampling often wraps the trajectory in ``torch.no_grad()``; re-enable
        # autograd locally so ``loss`` depends on ``delta`` and ``backward`` works.
        with torch.enable_grad():
            delta = torch.zeros_like(x_init, requires_grad=True)
            r_scalar = float(
                self._norm_boundary * torch.linalg.vector_norm(x_init.reshape(-1)).item()
            )
            logger.info("OptStepper r_scalar={:.6e}", r_scalar)
            adam = Adam([delta], lr=self._lr, weight_decay=0.0)

            best_loss = float("inf")
            best_delta: Optional[torch.Tensor] = None

            denom = max(self._num_steps - 1, 1)

            for it in range(self._num_steps):
                t_lin = float(it) / float(denom) if self._num_steps > 1 else 0.0
                cur_lr = self._lr + (self._lr_end - self._lr) * t_lin
                adam.param_groups[0]["lr"] = cur_lr

                adam.zero_grad(set_to_none=True)
                x_opt = x_init + delta
                if self._reg_noise_std > 0.0:
                    x_opt  += torch.randn_like(x_opt) * self._reg_noise_std

                x_pred = self._opt_forward_x(
                    model, x_opt, t_lo=t_lo, t_hi=t_hi, cond_base=cond_opt
                )
                loss = F.mse_loss(x_pred.float(), x_ref.float())
                loss_v = float(loss.item())
                logger.info(
                    "OptStepper[{}, {}] loss={:.6e} lr={:.6e}",
                    type(self._init_stepper).__name__,
                    type(self._opt_stepper).__name__,
                    loss_v,
                    cur_lr,
                )
                if loss_v < best_loss:
                    best_loss = loss_v
                    best_delta = delta.detach().clone()
                if self._epsilon > 0.0 and loss_v < self._epsilon:
                    break
                loss.backward()
                adam.step()
                if r_scalar > 0.0:
                    n = float(torch.linalg.vector_norm(delta.data.reshape(-1)).item())
                    if n > r_scalar:
                        delta.data.mul_(r_scalar / (n + 1e-12))

        if self._use_best_delta and best_delta is not None:
            delta_final = best_delta
            delta_src = "best"
        else:
            delta_final = delta.detach()
            delta_src = "last"
        delta_final.requires_grad_(False)
        logger.info(
            "OptStepper delta_final_norm={:.6e} best_loss={:.6e} delta={}",
            float(torch.linalg.vector_norm(delta_final.reshape(-1)).item()),
            best_loss,
            delta_src,
        )
        return StepperPayload(x=x_init + delta_final, v=v_out)
