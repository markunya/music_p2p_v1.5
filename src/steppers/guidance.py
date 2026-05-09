from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from acestep.models.common.apg_guidance import cfg_forward

from src.steppers.base import BaseStepper, StepperPayload
from src.utils.conditioning import ModelCondition


class _GuidanceCfgCore:
    def _init_guidance_cfg(self, guidance_scale: float, cfg_t_start: float, cfg_t_end: float) -> None:
        self.guidance_scale = float(guidance_scale)
        self.cfg_t_start = float(cfg_t_start)
        self.cfg_t_end = float(cfg_t_end)
        self._cfg_batch_expanded = False
        self._null_warned = False
        self._null_encoder_override: torch.Tensor | None = None
        self._null_seq_align_warned = False
        self._forbid_decoder_kv_cache = False

    def set_forbid_decoder_kv_cache(self, forbid: bool) -> None:
        """When True, CFG decoder uses no KV cache (needed for NTI per-layer torch.checkpoint)."""
        self._forbid_decoder_kv_cache = bool(forbid)

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

    def _expand_null_to_match_cond(self, null_tensor: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        n = null_tensor
        if n.shape[0] == 1 and cond.shape[0] != 1:
            n = n.expand(cond.shape[0], -1, -1).contiguous()
        elif n.shape[0] != cond.shape[0]:
            raise ValueError(
                f"null encoder override batch {n.shape[0]} does not match cond batch {cond.shape[0]}"
            )
        ln, lc = int(n.shape[1]), int(cond.shape[1])
        if ln != lc:
            if not self._null_seq_align_warned:
                logger.warning(
                    "GuidanceStepper: null encoder override seq_len {} != cond {}; padding or truncating",
                    ln,
                    lc,
                )
                self._null_seq_align_warned = True
            if ln < lc:
                n = F.pad(n, (0, 0, 0, lc - ln, 0, 0))
            else:
                n = n[:, :lc, :].contiguous()
        return n

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
        null_mask = enc_mask[:1].expand_as(enc_mask).contiguous()
        model_condition.encoder_hidden_states = torch.cat([enc, null_half], dim=0)
        model_condition.encoder_attention_mask = torch.cat([enc_mask, null_mask], dim=0)
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

    def _guided_velocity_cfg(
        self,
        model: torch.nn.Module,
        x_b: torch.Tensor,
        t_scalar: torch.Tensor,
        model_condition: ModelCondition,
        *,
        use_cache: bool,
        past_key_values: Any,
    ) -> tuple[torch.Tensor, Any]:
        x2 = torch.cat([x_b, x_b], dim=0)
        device, dtype = x_b.device, x_b.dtype
        t_tensor = t_scalar * torch.ones((x2.shape[0],), device=device, dtype=dtype)
        use_cache_eff = use_cache and not self._forbid_decoder_kv_cache
        # Cross-attention still mutates EncoderDecoderCache whenever it is non-None (see AceStepAttention:
        # the cross path keys off past_key_value, not use_cache). Per-layer checkpoint requires no cache.
        pkv_for_decoder = past_key_values if use_cache_eff else None
        out = model.decoder(
            hidden_states=x2,
            timestep=t_tensor,
            timestep_r=t_tensor,
            attention_mask=model_condition.attention_mask,
            encoder_hidden_states=model_condition.encoder_hidden_states,
            encoder_attention_mask=model_condition.encoder_attention_mask,
            context_latents=model_condition.context_latents,
            use_cache=use_cache_eff,
            past_key_values=pkv_for_decoder,
        )
        vt2, new_pkv = out[0], out[1]
        pred_cond, pred_null = vt2.chunk(2, dim=0)
        if not self._in_cfg_interval(t_scalar):
            return pred_cond, new_pkv
        return cfg_forward(pred_cond, pred_null, self.guidance_scale), new_pkv


class GuidanceStepperEuler(_GuidanceCfgCore, BaseStepper):
    def __init__(
        self,
        guidance_scale: float = 3.0,
        cfg_t_start: float = 0.0,
        cfg_t_end: float = 1.0,
    ) -> None:
        self._init_guidance_cfg(guidance_scale, cfg_t_start, cfg_t_end)

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
            v = self.velocity_with_side_cache(model, x, t_curr, model_condition)
            return StepperPayload(x=x - v * dt, v=v)

        self._refresh_encoder_uncond_half(model, model_condition)

        if model_condition.past_key_values is None:
            model_condition.past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        v, new_pkv = self._guided_velocity_cfg(
            model,
            x,
            t_curr,
            model_condition,
            use_cache=True,
            past_key_values=model_condition.past_key_values,
        )
        model_condition.past_key_values = new_pkv
        return StepperPayload(x=x - v * dt, v=v)


class UniEulerGuidanceStepper(GuidanceStepperEuler):
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

        old_pkv = model_condition.past_key_values

        model_condition.past_key_values = None
        v_curr = super().step(
            model,
            x,
            t_curr=t_curr,
            t_next=t_next,
            model_condition=model_condition,
        ).v

        x_corr = x - v_curr * dt

        model_condition.past_key_values = None
        v_next = super().step(
            model,
            x_corr,
            t_curr=t_next,
            t_next=t_next,
            model_condition=model_condition,
        ).v

        model_condition.past_key_values = old_pkv

        x_new = x - v_next * dt
        return StepperPayload(x=x_new, v=v_next)


class GuidanceStepperHeun(_GuidanceCfgCore, BaseStepper):
    def __init__(
        self,
        guidance_scale: float = 7.0,
        cfg_t_start: float = 0.0,
        cfg_t_end: float = 1.0,
    ) -> None:
        self._init_guidance_cfg(guidance_scale, cfg_t_start, cfg_t_end)

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
            v1 = self.velocity_with_side_cache(model, x, t_curr, model_condition)
            x_pred = x - v1 * dt
            v2 = self.velocity_fresh_cache(model, x_pred, t_next, model_condition)
            v_avg = 0.5 * (v1 + v2)
            return StepperPayload(x=x - v_avg * dt, v=v_avg)

        self._refresh_encoder_uncond_half(model, model_condition)

        if model_condition.past_key_values is None:
            model_condition.past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        v1, new_pkv = self._guided_velocity_cfg(
            model,
            x,
            t_curr,
            model_condition,
            use_cache=True,
            past_key_values=model_condition.past_key_values,
        )
        model_condition.past_key_values = new_pkv
        x_pred = x - v1 * dt

        corrector_cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        v2, _ = self._guided_velocity_cfg(
            model,
            x_pred,
            t_next,
            model_condition,
            use_cache=False,
            past_key_values=corrector_cache,
        )
        v_avg = 0.5 * (v1 + v2)
        return StepperPayload(x=x - v_avg * dt, v=v_avg)


class UniHeunGuidanceStepper(GuidanceStepperHeun):
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

        old_pkv = model_condition.past_key_values

        # 1) Heun velocity at current point:
        # u_curr = H[v_guided](x_t, t_curr -> t_next)
        model_condition.past_key_values = None
        u_curr = super().step(
            model,
            x,
            t_curr=t_curr,
            t_next=t_next,
            model_condition=model_condition,
        ).v

        # 2) Uni-Inv correction
        x_corr = x - dt * u_curr

        # 3) Corrected velocity at predicted point.
        # We use zero-length step to extract velocity at t_next.
        model_condition.past_key_values = None
        u_next = super().step(
            model,
            x_corr,
            t_curr=t_next,
            t_next=t_next,
            model_condition=model_condition,
        ).v

        # 4) Uni update
        x_new = x - dt * u_next

        model_condition.past_key_values = old_pkv

        return StepperPayload(x=x_new, v=u_next)


class GuidanceContinuationInversionStepper(_GuidanceCfgCore, BaseStepper):
    _START_GUIDANCE_SCALE = 1.0

    def __init__(
        self,
        guidance_scale: float = 2.0,
        cfg_t_start: float = 0.0,
        cfg_t_end: float = 1.0,
        continuation_steps: int = 10,
        j_approx: bool = False,
        j_eps: float = 1e-3,
    ) -> None:
        self._init_guidance_cfg(guidance_scale, cfg_t_start, cfg_t_end)
        self.continuation_steps = int(continuation_steps)
        self.j_approx = bool(j_approx)
        self.j_eps = float(j_eps)

        self.base_solver = UniEulerGuidanceStepper(
            guidance_scale=self._START_GUIDANCE_SCALE,
            cfg_t_start=cfg_t_start,
            cfg_t_end=cfg_t_end,
        )

    def _collapse_cfg_batch_layout_keep_override(
        self,
        model_condition: ModelCondition,
    ) -> None:
        if not self._cfg_batch_expanded:
            return

        enc = model_condition.encoder_hidden_states
        b = enc.shape[0] // 2

        model_condition.encoder_hidden_states = enc[:b].contiguous()
        model_condition.encoder_attention_mask = (
            model_condition.encoder_attention_mask[:b].contiguous()
        )
        model_condition.context_latents = model_condition.context_latents[:b].contiguous()
        model_condition.attention_mask = model_condition.attention_mask[:b].contiguous()
        model_condition.past_key_values = None

        self._cfg_batch_expanded = False

    def _cond_null_velocity(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        model_condition: ModelCondition,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        old_pkv = model_condition.past_key_values
        model_condition.past_key_values = None

        try:
            self._ensure_cfg_batch_layout(model, model_condition)
            self._refresh_encoder_uncond_half(model, model_condition)

            x2 = torch.cat([x, x], dim=0)
            t_tensor = t * torch.ones((x2.shape[0],), device=x.device, dtype=x.dtype)

            out = model.decoder(
                hidden_states=x2,
                timestep=t_tensor,
                timestep_r=t_tensor,
                attention_mask=model_condition.attention_mask,
                encoder_hidden_states=model_condition.encoder_hidden_states,
                encoder_attention_mask=model_condition.encoder_attention_mask,
                context_latents=model_condition.context_latents,
                use_cache=False,
                past_key_values=None,
            )

            v2 = out[0]
            v_cond, v_null = v2.chunk(2, dim=0)
            return v_cond, v_null

        finally:
            self._collapse_cfg_batch_layout_keep_override(model_condition)
            model_condition.past_key_values = old_pkv

    def _velocity_from_cond_null(
        self,
        v_cond: torch.Tensor,
        v_null: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        return cfg_forward(v_cond, v_null, float(scale))

    def _velocity_at_scale(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        model_condition: ModelCondition,
        scale: float,
    ) -> torch.Tensor:
        v_cond, v_null = self._cond_null_velocity(
            model,
            x,
            t,
            model_condition,
        )
        return self._velocity_from_cond_null(v_cond, v_null, scale)

    def _jvp_approx(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        model_condition: ModelCondition,
        scale: float,
        direction: torch.Tensor,
        v_x: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.j_eps

        v_shifted = self._velocity_at_scale(
            model,
            x + eps * direction,
            t,
            model_condition,
            scale,
        )

        return (v_shifted - v_x) / eps

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

        target_scale = self.guidance_scale
        old_pkv = model_condition.past_key_values

        self._collapse_cfg_batch_layout_keep_override(model_condition)

        self.base_solver.guidance_scale = self._START_GUIDANCE_SCALE
        self.base_solver.set_null_encoder_override(self._null_encoder_override)
        self.base_solver.set_forbid_decoder_kv_cache(self._forbid_decoder_kv_cache)

        model_condition.past_key_values = None
        payload = self.base_solver.step(
            model,
            x,
            t_curr=t_curr,
            t_next=t_next,
            model_condition=model_condition,
        )

        x_s = payload.x
        v_s = payload.v

        prev_scale = self._START_GUIDANCE_SCALE

        for k in range(1, self.continuation_steps + 1):
            scale = self._START_GUIDANCE_SCALE + (
                target_scale - self._START_GUIDANCE_SCALE
            ) * k / self.continuation_steps

            ds = scale - prev_scale

            v_cond, v_null = self._cond_null_velocity(
                model,
                x_s,
                t_next,
                model_condition,
            )

            dv_ds = v_cond - v_null
            v_s = self._velocity_from_cond_null(v_cond, v_null, scale)

            if self.j_approx:
                jvp = self._jvp_approx(
                    model,
                    x_s,
                    t_next,
                    model_condition,
                    scale,
                    dv_ds,
                    v_s,
                )
                direction = dv_ds - dt * jvp
            else:
                direction = dv_ds

            x_s = x_s - dt * ds * direction
            prev_scale = scale

        self.guidance_scale = target_scale
        model_condition.past_key_values = old_pkv

        return StepperPayload(x=x_s, v=v_s)

CFG_GUIDANCE_STEPPERS = (
    GuidanceStepperEuler,
    GuidanceStepperHeun,
    UniEulerGuidanceStepper,
    UniHeunGuidanceStepper,
    GuidanceContinuationInversionStepper
)
