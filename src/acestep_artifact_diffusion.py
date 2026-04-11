# Vendored logic derived from ACE-Step 1.5 acestep/models/base/modeling_acestep_v15_base.py
# (AceStepConditionGenerationModel.generate_audio). Sync when upgrading acestep.
# Simplified for prompt-to-prompt / text2music: CFG only, no cover/non-cover, no repaint, no cover_noise init.
# Adds: per-step null encoder states (NTI). Fixed start latent = artifact ``noise`` + prepare_noise patch.

from __future__ import annotations

import time
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.utils import logging as hf_logging

from acestep.models.base.apg_guidance import MomentumBuffer, adg_forward, apg_forward, cfg_forward

hf_logger = hf_logging.get_logger(__name__)


def generate_audio_artifact_impl(
    self,
    null_encoder_hidden_states_per_step: Optional[List[torch.Tensor]],
    **kwargs,
):
    if "diffusion_guidance_sale" in kwargs:
        hf_logger.warning(
            "generate_audio() received deprecated kwarg 'diffusion_guidance_sale'; "
            "please rename it to 'diffusion_guidance_scale'."
        )
        kwargs["diffusion_guidance_scale"] = kwargs.pop("diffusion_guidance_sale")

    if kwargs.pop("timesteps", None) is not None:
        hf_logger.warning(
            "artifact diffusion: custom timesteps are not supported; use infer_steps/shift from cfg."
        )

    text_hidden_states = kwargs.pop("text_hidden_states")
    text_attention_mask = kwargs.pop("text_attention_mask")
    lyric_hidden_states = kwargs.pop("lyric_hidden_states")
    lyric_attention_mask = kwargs.pop("lyric_attention_mask")
    refer_audio_acoustic_hidden_states_packed = kwargs.pop("refer_audio_acoustic_hidden_states_packed")
    refer_audio_order_mask = kwargs.pop("refer_audio_order_mask")
    src_latents = kwargs.pop("src_latents")
    chunk_masks = kwargs.pop("chunk_masks")
    is_covers = kwargs.pop("is_covers")
    silence_latent = kwargs.pop("silence_latent", None)
    attention_mask = kwargs.pop("attention_mask", None)
    seed = kwargs.pop("seed", None)
    infer_method = kwargs.pop("infer_method", "ode")
    kwargs.pop("use_cache", True)
    infer_steps = kwargs.pop("infer_steps", 30)
    diffusion_guidance_scale = kwargs.pop("diffusion_guidance_scale", 7.0)
    audio_cover_strength = kwargs.pop("audio_cover_strength", 1.0)
    kwargs.pop("non_cover_text_hidden_states", None)
    kwargs.pop("non_cover_text_attention_mask", None)
    cfg_interval_start = kwargs.pop("cfg_interval_start", 0.0)
    cfg_interval_end = kwargs.pop("cfg_interval_end", 1.0)
    precomputed_lm_hints_25Hz = kwargs.pop("precomputed_lm_hints_25Hz", None)
    audio_codes = kwargs.pop("audio_codes", None)
    use_progress_bar = kwargs.pop("use_progress_bar", True)
    use_adg = kwargs.pop("use_adg", False)
    shift = kwargs.pop("shift", 1.0)
    cover_noise_strength = kwargs.pop("cover_noise_strength", 0.0)
    kwargs.pop("repaint_mask", None)
    kwargs.pop("clean_src_latents", None)
    kwargs.pop("repaint_crossfade_frames", None)
    kwargs.pop("repaint_injection_ratio", None)
    sampler_mode = kwargs.pop("sampler_mode", "euler")
    velocity_norm_threshold = kwargs.pop("velocity_norm_threshold", 0.0)
    velocity_ema_factor = kwargs.pop("velocity_ema_factor", 0.0)

    if kwargs:
        hf_logger.warning(f"generate_audio_artifact_impl: ignoring unknown kwargs {list(kwargs.keys())}")

    if diffusion_guidance_scale <= 1.0:
        raise ValueError(
            "artifact diffusion supports only classifier-free guidance: set diffusion_guidance_scale > 1."
        )
    if audio_cover_strength != 1.0:
        raise ValueError(
            "artifact diffusion does not implement cover/non-cover switching; use audio_cover_strength=1.0."
        )
    if cover_noise_strength != 0.0:
        raise ValueError(
            "artifact diffusion does not implement cover_noise_strength; use 0.0."
        )

    if attention_mask is None:
        latent_length = src_latents.shape[1]
        attention_mask = torch.ones(
            src_latents.shape[0], latent_length, device=src_latents.device, dtype=src_latents.dtype
        )

    if null_encoder_hidden_states_per_step is None:
        raise ValueError("null_encoder_hidden_states_per_step is required for artifact diffusion path.")
    if len(null_encoder_hidden_states_per_step) != infer_steps:
        raise ValueError(
            f"len(null_encoder_hidden_states_per_step)={len(null_encoder_hidden_states_per_step)} "
            f"must equal infer_steps={infer_steps}"
        )

    time_costs = {}
    start_time = time.time()
    total_start_time = start_time
    encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        lyric_hidden_states=lyric_hidden_states,
        lyric_attention_mask=lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask=refer_audio_order_mask,
        hidden_states=src_latents,
        attention_mask=attention_mask,
        silence_latent=silence_latent,
        src_latents=src_latents,
        chunk_masks=chunk_masks,
        is_covers=is_covers,
        precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
        audio_codes=audio_codes,
    )
    end_time = time.time()
    time_costs["encoder_time_cost"] = end_time - start_time
    start_time = end_time

    device, dtype = context_latents.device, context_latents.dtype
    t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    if use_progress_bar:
        iterator = tqdm(zip(t[:-1], t[1:]), total=infer_steps)
    else:
        iterator = zip(t[:-1], t[1:])

    noise = self.prepare_noise(context_latents, seed)
    bsz, device, dtype = context_latents.shape[0], context_latents.device, context_latents.dtype
    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
    momentum_buffer = MomentumBuffer()
    xt = noise

    do_cfg_guidance = True
    enc_hs_cond = encoder_hidden_states
    enc_mask_cond = encoder_attention_mask
    context_latents = torch.cat([context_latents, context_latents], dim=0)
    attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

    use_heun = sampler_mode == "heun"
    use_norm_clamp = velocity_norm_threshold > 0.0
    use_ema = velocity_ema_factor > 0.0
    prev_vt = None
    if use_heun and infer_method == "sde":
        hf_logger.warning("Heun sampler is not compatible with SDE; falling back to Euler.")
        use_heun = False

    with torch.no_grad():
        for step_idx, (t_curr, t_prev) in enumerate(iterator):
            assert enc_hs_cond is not None and enc_mask_cond is not None
            null_step = null_encoder_hidden_states_per_step[step_idx]
            if not isinstance(null_step, torch.Tensor):
                raise TypeError(
                    f"null_encoder_hidden_states_per_step[{step_idx}] must be Tensor, got {type(null_step)}"
                )
            null_step = null_step.to(device=device, dtype=enc_hs_cond.dtype)
            if null_step.shape != enc_hs_cond.shape:
                raise ValueError(
                    f"null_encoder_hidden_states_per_step[{step_idx}] shape {tuple(null_step.shape)} "
                    f"!= conditional {tuple(enc_hs_cond.shape)}"
                )
            encoder_hidden_states = torch.cat([enc_hs_cond, null_step], dim=0)
            encoder_attention_mask = torch.cat([enc_mask_cond, enc_mask_cond], dim=0)

            x = torch.cat([xt, xt], dim=0)
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), device=device, dtype=dtype)
            decoder_outputs = self.decoder(
                hidden_states=x,
                timestep=t_curr_tensor,
                timestep_r=t_curr_tensor,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=True,
                past_key_values=past_key_values,
            )

            vt = decoder_outputs[0]
            past_key_values = decoder_outputs[1]
            apply_cfg_guidance = t_curr >= cfg_interval_start and t_curr <= cfg_interval_end
            pred_cond, pred_null_cond = vt.chunk(2)
            if apply_cfg_guidance:
                if not use_adg:
                    vt = apg_forward(
                        pred_cond=pred_cond,
                        pred_uncond=pred_null_cond,
                        guidance_scale=diffusion_guidance_scale,
                        momentum_buffer=momentum_buffer,
                        dims=[1],
                    )
                else:
                    vt = adg_forward(
                        latents=xt,
                        noise_pred_cond=pred_cond,
                        noise_pred_uncond=pred_null_cond,
                        sigma=t_curr,
                        guidance_scale=diffusion_guidance_scale,
                    )
            else:
                vt = pred_cond

            if use_norm_clamp:
                vt_norm = torch.norm(vt, dim=(1, 2), keepdim=True)
                xt_norm = torch.norm(xt, dim=(1, 2), keepdim=True) + 1e-10
                scale = torch.clamp(velocity_norm_threshold * xt_norm / (vt_norm + 1e-10), max=1.0)
                vt = vt * scale

            if use_ema and prev_vt is not None:
                vt = (1.0 - velocity_ema_factor) * vt + velocity_ema_factor * prev_vt

            if infer_method == "sde":
                t_curr_bsz = t_curr * torch.ones((bsz,), device=device, dtype=dtype)
                pred_clean = self.get_x0_from_noise(xt, vt, t_curr_bsz)
                next_timestep = 1.0 - (float(step_idx + 1) / infer_steps)
                xt = self.renoise(pred_clean, next_timestep)
            elif use_heun and infer_method == "ode":
                dt = t_curr - t_prev
                dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt_predicted = xt - vt * dt_tensor
                x2 = torch.cat([xt_predicted, xt_predicted], dim=0)
                t_prev_tensor = t_prev * torch.ones((x2.shape[0],), device=device, dtype=dtype)
                corrector_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
                decoder_outputs2 = self.decoder(
                    hidden_states=x2,
                    timestep=t_prev_tensor,
                    timestep_r=t_prev_tensor,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                    use_cache=False,
                    past_key_values=corrector_kv,
                )
                vt2 = decoder_outputs2[0]
                pred_cond2, pred_null_cond2 = vt2.chunk(2)
                apply_cfg_corrector = t_prev >= cfg_interval_start and t_prev <= cfg_interval_end
                if apply_cfg_corrector:
                    if not use_adg:
                        vt2 = cfg_forward(pred_cond2, pred_null_cond2, diffusion_guidance_scale)
                    elif t_prev > 0:
                        vt2 = adg_forward(
                            latents=xt_predicted,
                            noise_pred_cond=pred_cond2,
                            noise_pred_uncond=pred_null_cond2,
                            sigma=t_prev,
                            guidance_scale=diffusion_guidance_scale,
                        )
                    else:
                        vt2 = cfg_forward(pred_cond2, pred_null_cond2, diffusion_guidance_scale)
                else:
                    vt2 = pred_cond2
                if use_norm_clamp:
                    vt2_norm = torch.norm(vt2, dim=(1, 2), keepdim=True)
                    xt_pred_norm = torch.norm(xt_predicted, dim=(1, 2), keepdim=True) + 1e-10
                    scale2 = torch.clamp(velocity_norm_threshold * xt_pred_norm / (vt2_norm + 1e-10), max=1.0)
                    vt2 = vt2 * scale2
                if use_ema:
                    vt2 = (1.0 - velocity_ema_factor) * vt2 + velocity_ema_factor * vt
                vt_avg = 0.5 * (vt + vt2)
                xt = xt - vt_avg * dt_tensor
                vt = vt_avg
            elif infer_method == "ode":
                dt = t_curr - t_prev
                dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt = xt - vt * dt_tensor

            prev_vt = vt

    end_time = time.time()
    time_costs["diffusion_time_cost"] = end_time - start_time
    time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / infer_steps
    time_costs["total_time_cost"] = end_time - total_start_time
    return {
        "target_latents": xt,
        "time_costs": time_costs,
    }


def bind_generate_audio_patch(
    model: torch.nn.Module,
    orig_generate_audio,
    *,
    null_encoder_hidden_states_per_step: Optional[List[torch.Tensor]],
):
    """Return (patched_method, orig) for assignment to model.generate_audio."""

    import types

    null_list = null_encoder_hidden_states_per_step

    def patched(self, **kwargs):
        return generate_audio_artifact_impl(self, null_list, **kwargs)

    return types.MethodType(patched, model), orig_generate_audio
