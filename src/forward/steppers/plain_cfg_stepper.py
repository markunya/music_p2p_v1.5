"""Euler ODE + CFG (learned null или NTI null по шагу), как в ACE без cover/repaint."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
from acestep.models.base.apg_guidance import MomentumBuffer, adg_forward, apg_forward
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from src.forward.steppers.base import LocalDiffusionStepper


class PlainCfgEulerStepper(LocalDiffusionStepper):
    """Один Euler-шаг: decoder + APG/ADG + ``xt -= vt * dt``. Состояние и константы — поля объекта."""

    def __init__(
        self,
        model: Any,
        enc_cond: torch.Tensor,
        enc_mask: torch.Tensor,
        context_lat: torch.Tensor,
        attention_mask: torch.Tensor,
        bsz: int,
        *,
        use_adg: bool,
        guidance_scale: float,
        cfg_interval_start: float,
        cfg_interval_end: float,
        infer_method: str,
        sampler_mode: str,
        null_encoder_hidden_states_per_step: Optional[List[torch.Tensor]] = None,
    ) -> None:
        if infer_method != "ode":
            raise ValueError(f"PlainCfgEulerStepper: only infer_method='ode', got {infer_method!r}")
        if sampler_mode != "euler":
            raise ValueError(f"PlainCfgEulerStepper: only sampler_mode='euler' for now, got {sampler_mode!r}")
        self._model = model
        self._enc_cond = enc_cond
        self._enc_mask = enc_mask
        self._context_lat = context_lat
        self._attention_mask = attention_mask
        self._bsz = int(bsz)
        self._device = enc_cond.device
        self._dtype = enc_cond.dtype
        self._use_adg = use_adg
        self._guidance_scale = float(guidance_scale)
        self._cfg_interval_start = float(cfg_interval_start)
        self._cfg_interval_end = float(cfg_interval_end)
        self._double_batch = self._guidance_scale > 1.0
        self._null_per_step = null_encoder_hidden_states_per_step
        self._past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        self._momentum_buffer = MomentumBuffer()

    def _encoder_for_step(self, step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._double_batch:
            return self._enc_cond, self._enc_mask
        if self._null_per_step is not None:
            null_t = self._null_per_step[step_idx]
            if not isinstance(null_t, torch.Tensor):
                raise TypeError(f"null_encoder_hidden_states_per_step[{step_idx}] must be Tensor")
            null_t = null_t.to(device=self._device, dtype=self._enc_cond.dtype)
            if null_t.shape != self._enc_cond.shape:
                raise ValueError(
                    f"null_encoder_hidden_states_per_step[{step_idx}] shape {tuple(null_t.shape)} "
                    f"!= conditional {tuple(self._enc_cond.shape)}"
                )
            enc = torch.cat([self._enc_cond, null_t], dim=0)
        else:
            enc = torch.cat(
                [self._enc_cond, self._model.null_condition_emb.expand_as(self._enc_cond)],
                dim=0,
            )
        enc_mask = torch.cat([self._enc_mask, self._enc_mask], dim=0)
        return enc, enc_mask

    def step(self, xt: torch.Tensor, t_curr: float, t_prev: float, step_idx: int) -> torch.Tensor:
        enc_hs, enc_mask = self._encoder_for_step(step_idx)
        if self._double_batch:
            x = torch.cat([xt, xt], dim=0)
        else:
            x = xt
        t_tensor = t_curr * torch.ones((x.shape[0],), device=self._device, dtype=self._dtype)
        dec = self._model.decoder(
            hidden_states=x,
            timestep=t_tensor,
            timestep_r=t_tensor,
            attention_mask=self._attention_mask,
            encoder_hidden_states=enc_hs,
            encoder_attention_mask=enc_mask,
            context_latents=self._context_lat,
            use_cache=True,
            past_key_values=self._past_key_values,
        )
        vt = dec[0]
        self._past_key_values = dec[1]

        apply_cfg_blend = self._cfg_interval_start <= t_curr <= self._cfg_interval_end
        if self._double_batch:
            pred_cond, pred_null = vt.chunk(2)
            if apply_cfg_blend:
                if not self._use_adg:
                    vt = apg_forward(
                        pred_cond=pred_cond,
                        pred_uncond=pred_null,
                        guidance_scale=self._guidance_scale,
                        momentum_buffer=self._momentum_buffer,
                        dims=[1],
                    )
                else:
                    sigma_b = torch.full((self._bsz,), t_curr, device=self._device, dtype=self._dtype)
                    vt = adg_forward(
                        latents=xt,
                        noise_pred_cond=pred_cond,
                        noise_pred_uncond=pred_null,
                        sigma=sigma_b,
                        guidance_scale=self._guidance_scale,
                    )
            else:
                vt = pred_cond
        else:
            vt = vt

        dt = t_curr - t_prev
        dt_tensor = dt * torch.ones((self._bsz,), device=self._device, dtype=self._dtype).unsqueeze(-1).unsqueeze(-1)
        return xt - vt * dt_tensor
