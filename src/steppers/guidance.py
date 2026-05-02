from enum import StrEnum
from typing import Any

import torch
from loguru import logger
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from acestep.models.common.apg_guidance import (
    MomentumBuffer,
    adg_forward,
    apg_forward,
    cfg_forward,
)

from src.steppers.base import BaseStepper, StepperPayload
from src.steppers.euler import Euler
from src.steppers.heun import Heun
from src.utils.conditioning import ModelCondition


class GuidanceMode(StrEnum):
    CFG = "cfg"
    APG = "apg"
    ADG = "adg"


class GuidanceStepper(BaseStepper):

    def __init__(
        self,
        base_stepper: BaseStepper,
        guidance_scale: float = 7.0,
        guidance_mode: GuidanceMode | str = GuidanceMode.CFG,
        cfg_t_start: float = 0.0,
        cfg_t_end: float = 1.0,
        apg_momentum: float = -0.75,
        apg_eta: float = 0.0,
        apg_norm_threshold: float = 2.5,
        apg_dims: list[int] | None = None,
        adg_angle_clip: float = 3.14159265 / 6.0,
        adg_apply_norm: bool = False,
        adg_apply_clip: bool = True,
    ) -> None:
        if not isinstance(base_stepper, (Euler, Heun)):
            raise TypeError(f"GuidanceStepper expects Euler or Heun, got {type(base_stepper).__name__}")
        if isinstance(guidance_mode, GuidanceMode):
            self.guidance_mode = guidance_mode
        else:
            try:
                self.guidance_mode = GuidanceMode(str(guidance_mode))
            except ValueError as exc:
                allowed = ", ".join(m.value for m in GuidanceMode)
                raise ValueError(f"guidance_mode must be one of {allowed}, got {guidance_mode!r}") from exc
        self._base = base_stepper
        self.guidance_scale = float(guidance_scale)
        self.cfg_t_start = float(cfg_t_start)
        self.cfg_t_end = float(cfg_t_end)
        self.apg_eta = float(apg_eta)
        self.apg_norm_threshold = float(apg_norm_threshold)
        self.apg_dims = apg_dims if apg_dims is not None else [1]
        self.adg_angle_clip = float(adg_angle_clip)
        self.adg_apply_norm = bool(adg_apply_norm)
        self.adg_apply_clip = bool(adg_apply_clip)
        self._cfg_batch_expanded = False
        self._null_warned = False
        self._apg_buffer: MomentumBuffer | None
        self._apg_momentum = float(apg_momentum)
        if self.guidance_mode is GuidanceMode.APG:
            self._apg_buffer = MomentumBuffer(momentum=self._apg_momentum)
        else:
            self._apg_buffer = None
        self._null_encoder_override: torch.Tensor | None = None

    def set_null_encoder_override(self, emb: torch.Tensor | None) -> None:
        self._null_encoder_override = emb

    def reset_guidance_layout(self) -> None:
        self._cfg_batch_expanded = False
        self._null_encoder_override = None

    def collapse_cfg_batch_layout(self, model_condition: ModelCondition) -> None:
        if not self._cfg_batch_expanded:
            return
        enc = model_condition.encoder_hidden_states
        b = enc.shape[0] // 2
        if b < 1:
            self.reset_guidance_layout()
            return
        model_condition.encoder_hidden_states = enc[:b].contiguous()
        model_condition.encoder_attention_mask = model_condition.encoder_attention_mask[:b].contiguous()
        model_condition.context_latents = model_condition.context_latents[:b].contiguous()
        model_condition.attention_mask = model_condition.attention_mask[:b].contiguous()
        model_condition.past_key_values = None
        self.reset_guidance_layout()

    def reset_apg_momentum_for_nti_inner(self) -> None:
        if self.guidance_mode is GuidanceMode.APG:
            self._apg_buffer = MomentumBuffer(momentum=self._apg_momentum)

    @staticmethod
    def _expand_null_to_match_cond(null_tensor: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if null_tensor.shape[0] == cond.shape[0]:
            return null_tensor
        if null_tensor.shape[0] == 1:
            return null_tensor.expand_as(cond)
        raise ValueError(
            f"null encoder override batch {null_tensor.shape[0]} does not match cond batch {cond.shape[0]}"
        )

    def _active_guidance(self, model: torch.nn.Module) -> bool:
        if self.guidance_scale <= 1.0:
            return False
        if not hasattr(model, "null_condition_emb"):
            if not self._null_warned:
                logger.warning("GuidanceStepper: model has no null_condition_emb; running without CFG")
                self._null_warned = True
            return False
        return True

    def _in_cfg_interval(self, t_scalar: torch.Tensor) -> bool:
        t = float(t_scalar)
        return self.cfg_t_start <= t <= self.cfg_t_end

    def _ensure_cfg_batch_layout(self, model: torch.nn.Module, model_condition: ModelCondition) -> bool:
        if self._cfg_batch_expanded:
            return True
        null_emb_weight = getattr(model, "null_condition_emb", None)
        if null_emb_weight is None:
            return False
        enc = model_condition.encoder_hidden_states
        enc_mask = model_condition.encoder_attention_mask
        ctx = model_condition.context_latents
        attn = model_condition.attention_mask
        if self._null_encoder_override is not None:
            null_half = self._expand_null_to_match_cond(self._null_encoder_override, enc)
        else:
            null_half = null_emb_weight.expand_as(enc)
        model_condition.encoder_hidden_states = torch.cat([enc, null_half], dim=0)
        model_condition.encoder_attention_mask = torch.cat([enc_mask, enc_mask], dim=0)
        model_condition.context_latents = torch.cat([ctx, ctx], dim=0)
        model_condition.attention_mask = torch.cat([attn, attn], dim=0)
        model_condition.past_key_values = None
        self._cfg_batch_expanded = True
        return True

    def _refresh_encoder_uncond_half(self, model: torch.nn.Module, model_condition: ModelCondition) -> None:
        if not self._cfg_batch_expanded:
            return
        enc_full = model_condition.encoder_hidden_states
        bsz = enc_full.shape[0] // 2
        cond = enc_full[:bsz]
        if self._null_encoder_override is not None:
            null_half = self._expand_null_to_match_cond(self._null_encoder_override, cond)
        else:
            null_w = getattr(model, "null_condition_emb", None)
            if null_w is None:
                raise RuntimeError("model.null_condition_emb required for CFG layout")
            null_half = null_w.expand_as(cond)
        model_condition.encoder_hidden_states = torch.cat([cond, null_half], dim=0)

    def _guided_velocity(
        self,
        model: torch.nn.Module,
        x_b: torch.Tensor,
        t_scalar: torch.Tensor,
        model_condition: ModelCondition,
        *,
        latents_for_adg: torch.Tensor,
        use_cache: bool,
        past_key_values: Any,
        is_heun_corrector: bool,
    ) -> tuple[torch.Tensor, Any]:
        bsz = x_b.shape[0]
        x2 = torch.cat([x_b, x_b], dim=0)
        device, dtype = x_b.device, x_b.dtype
        t_tensor = t_scalar * torch.ones((x2.shape[0],), device=device, dtype=dtype)
        out = model.decoder(
            hidden_states=x2,
            timestep=t_tensor,
            timestep_r=t_tensor,
            attention_mask=model_condition.attention_mask,
            encoder_hidden_states=model_condition.encoder_hidden_states,
            encoder_attention_mask=model_condition.encoder_attention_mask,
            context_latents=model_condition.context_latents,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        vt2, new_pkv = out[0], out[1]
        pred_cond, pred_null = vt2.chunk(2, dim=0)
        apply_cfg = self._in_cfg_interval(t_scalar)
        if not apply_cfg:
            v_b = pred_cond
            return v_b, new_pkv

        if self.guidance_mode is GuidanceMode.CFG:
            v_b = cfg_forward(pred_cond, pred_null, self.guidance_scale)
        elif self.guidance_mode is GuidanceMode.APG:
            if is_heun_corrector:
                v_b = cfg_forward(pred_cond, pred_null, self.guidance_scale)
            else:
                v_b = apg_forward(
                    pred_cond=pred_cond,
                    pred_uncond=pred_null,
                    guidance_scale=self.guidance_scale,
                    momentum_buffer=self._apg_buffer,
                    eta=self.apg_eta,
                    norm_threshold=self.apg_norm_threshold,
                    dims=self.apg_dims,
                )
        elif self.guidance_mode is GuidanceMode.ADG:
            if is_heun_corrector and float(t_scalar) <= 0.0:
                v_b = cfg_forward(pred_cond, pred_null, self.guidance_scale)
            else:
                v_b = adg_forward(
                    latents_for_adg,
                    pred_cond,
                    pred_null,
                    t_scalar,
                    self.guidance_scale,
                    angle_clip=self.adg_angle_clip,
                    apply_norm=self.adg_apply_norm,
                    apply_clip=self.adg_apply_clip,
                )
        else:
            raise ValueError(f"Unknown guidance_mode: {self.guidance_mode}")
        return v_b, new_pkv

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

        if not self._active_guidance(model) or not self._ensure_cfg_batch_layout(model, model_condition):
            if isinstance(self._base, Euler):
                v = self.velocity_with_side_cache(model, x, t_curr, model_condition)
                return StepperPayload(x=x - v * dt, v=v)
            v1 = self.velocity_with_side_cache(model, x, t_curr, model_condition)
            x_pred = x - v1 * dt
            v2 = self.velocity_fresh_cache(model, x_pred, t_next, model_condition)
            v_avg = 0.5 * (v1 + v2)
            return StepperPayload(x=x - v_avg * dt, v=v_avg)

        self._refresh_encoder_uncond_half(model, model_condition)

        if isinstance(self._base, Euler):
            if model_condition.past_key_values is None:
                model_condition.past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            v, new_pkv = self._guided_velocity(
                model,
                x,
                t_curr,
                model_condition,
                latents_for_adg=x,
                use_cache=True,
                past_key_values=model_condition.past_key_values,
                is_heun_corrector=False,
            )
            model_condition.past_key_values = new_pkv
            return StepperPayload(x=x - v * dt, v=v)

        if model_condition.past_key_values is None:
            model_condition.past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        v1, new_pkv = self._guided_velocity(
            model,
            x,
            t_curr,
            model_condition,
            latents_for_adg=x,
            use_cache=True,
            past_key_values=model_condition.past_key_values,
            is_heun_corrector=False,
        )
        model_condition.past_key_values = new_pkv
        x_pred = x - v1 * dt

        corrector_cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        v2, _ = self._guided_velocity(
            model,
            x_pred,
            t_next,
            model_condition,
            latents_for_adg=x_pred,
            use_cache=False,
            past_key_values=corrector_cache,
            is_heun_corrector=True,
        )
        v_avg = 0.5 * (v1 + v2)
        return StepperPayload(x=x - v_avg * dt, v=v_avg)
