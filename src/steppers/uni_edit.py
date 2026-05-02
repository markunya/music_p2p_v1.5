import torch
from typing import Literal

from src.steppers.base import BaseStepper, StepperPayload
from src.utils.conditioning import ModelCondition


class UniEditStepper(BaseStepper):
    def __init__(
        self,
        omega: float = 0.0,
        eps: float = 1e-8,
        do_abs: bool = False,
        mask_mode: Literal["image", "audio"] = "image",
    ):
        super().__init__()
        self.omega = omega
        self.eps = eps
        self.do_abs = do_abs
        self.mask_mode = mask_mode

    @staticmethod
    def _slice_model_condition(
        model_condition: ModelCondition,
        start: int,
        end: int,
    ) -> ModelCondition:
        return ModelCondition(
            encoder_hidden_states=model_condition.encoder_hidden_states[start:end],
            encoder_attention_mask=model_condition.encoder_attention_mask[start:end],
            context_latents=model_condition.context_latents[start:end],
            attention_mask=model_condition.attention_mask[start:end],
            past_key_values=model_condition.past_key_values,
        )

    def _minmax_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        flat = x.reshape(x.shape[0], -1)
        min_val = flat.min(dim=-1, keepdim=True).values
        max_val = flat.max(dim=-1, keepdim=True).values
        norm = (flat - min_val) / (max_val - min_val + self.eps)
        return norm.reshape_as(x)

    def _mask(self, v_diff: torch.Tensor) -> torch.Tensor:
        values = v_diff.abs() if self.do_abs else v_diff

        if self.mask_mode == "image":
            mean_map = values.mean(dim=1, keepdim=True)
            return self._minmax_per_sample(mean_map)

        if self.mask_mode == "audio":
            return self._minmax_per_sample(values)

        raise ValueError(
            f"Unknown mask_mode={self.mask_mode!r}. Expected 'image' or 'audio'."
        )

    def step(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        *,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
        model_condition: ModelCondition,
    ) -> StepperPayload:
        if x.shape[0] != 2:
            raise ValueError(f"UniEditStepper expects batch size 2, got {x.shape[0]}")

        dt = self._segment_dt_tensor(
            dt=t_curr - t_next,
            bsz=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        x_src = x[0:1]
        x_edit = x[1:2]

        dt_src = dt[0:1]
        dt_edit = dt[1:2]

        cond_src = self._slice_model_condition(model_condition, 0, 1)
        cond_tgt = self._slice_model_condition(model_condition, 1, 2)

        v_src = self.velocity_fresh_cache(model, x_src, t_curr, cond_src)
        x_src_new = x_src - dt_src * v_src

        v_s = self.velocity_fresh_cache(model, x_edit, t_curr, cond_src)
        v_t = self.velocity_fresh_cache(model, x_edit, t_curr, cond_tgt)

        v_diff = v_t - v_s
        m = self._mask(v_diff)

        stride = -self.omega * dt_edit * (1.0 + m) * v_diff
        x_corr = x_edit + stride

        v_fused = m * v_t + (1.0 - m) * v_s
        x_edit_new = x_corr - dt_edit * v_fused

        x_new = torch.cat([x_src_new, x_edit_new], dim=0)
        v_out = torch.cat([v_src, v_fused], dim=0)
        return StepperPayload(x=x_new, v=v_out)
