"""Per-step null encoder optimization so CFG steps match a fixed pivot trajectory (music_p2p-style NTI)."""

from __future__ import annotations

from src.mps_adg_patch import apply_adg_mps_patch

apply_adg_mps_patch()

from typing import Any, Dict, List, Literal

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from acestep.models.base.apg_guidance import MomentumBuffer, adg_forward, apg_forward

from src.logging import utils as logging
from src.logging.writer import BaseWriter, DummyWriter
from src.nti.schedule import acestep_sigma_grid

NtiLatentIntegrator = Literal["euler", "heun"]


def _dt_tensor(dt: torch.Tensor, bsz: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return dt * torch.ones((bsz,), device=device, dtype=dtype).view(-1, 1, 1)


def _predict_latent_cfg_step(
    model: torch.nn.Module,
    latent_cur: torch.Tensor,
    t_curr: torch.Tensor,
    t_prev: torch.Tensor,
    null_emb: torch.Tensor,
    dt_tensor: torch.Tensor,
    enc_hs_cond: torch.Tensor,
    enc_mask_cond: torch.Tensor,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    diffusion_guidance_scale: float,
    use_adg: bool,
    integrator: NtiLatentIntegrator,
) -> torch.Tensor:
    """One cond+CFG velocity step along decreasing ``σ``: ``t_curr`` → ``t_prev`` (``dt = t_curr - t_prev`` > 0)."""
    vt1 = _decoder_velocity_cfg_trainable_null(
        model,
        latent_cur,
        t_curr,
        enc_hs_cond,
        null_emb,
        enc_mask_cond,
        context_latents,
        attention_mask,
        diffusion_guidance_scale=diffusion_guidance_scale,
        use_adg=use_adg,
        apply_cfg_guidance=True,
        momentum_buffer=MomentumBuffer(),
    )
    if integrator == "euler":
        return latent_cur - vt1 * dt_tensor
    lat_e = latent_cur - vt1 * dt_tensor
    vt2 = _decoder_velocity_cfg_trainable_null(
        model,
        lat_e,
        t_prev,
        enc_hs_cond,
        null_emb,
        enc_mask_cond,
        context_latents,
        attention_mask,
        diffusion_guidance_scale=diffusion_guidance_scale,
        use_adg=use_adg,
        apply_cfg_guidance=True,
        momentum_buffer=MomentumBuffer(),
    )
    return latent_cur - 0.5 * (vt1 + vt2) * dt_tensor


def _prepare_condition_tensors(
    model: torch.nn.Module,
    handler: Any,
    payload: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_latents = payload["src_latents"]
    device, dtype = src_latents.device, src_latents.dtype
    attention_mask = torch.ones(
        src_latents.shape[0],
        src_latents.shape[1],
        device=device,
        dtype=dtype,
    )
    encoder_hidden_states, encoder_attention_mask, context_latents = model.prepare_condition(
        text_hidden_states=payload["text_hidden_states"],
        text_attention_mask=payload["text_attention_mask"],
        lyric_hidden_states=payload["lyric_hidden_states"],
        lyric_attention_mask=payload["lyric_attention_mask"],
        refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
        refer_audio_order_mask=payload["refer_audio_order_mask"],
        hidden_states=src_latents,
        attention_mask=attention_mask,
        silence_latent=handler.silence_latent,
        src_latents=src_latents,
        chunk_masks=payload["chunk_mask"],
        is_covers=payload["is_covers"],
        precomputed_lm_hints_25Hz=payload["precomputed_lm_hints_25Hz"],
    )
    return encoder_hidden_states, encoder_attention_mask, context_latents, attention_mask


@torch.no_grad()
def _velocity_cond_only(
    model: torch.nn.Module,
    xt: torch.Tensor,
    t_scalar: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    bsz = xt.shape[0]
    device, dtype = xt.device, xt.dtype
    t_tensor = t_scalar * torch.ones((bsz,), device=device, dtype=dtype)
    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
    out = model.decoder(
        hidden_states=xt,
        timestep=t_tensor,
        timestep_r=t_tensor,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
        use_cache=False,
        past_key_values=past_key_values,
    )
    return out[0]


def _decoder_velocity_cfg_trainable_null(
    model: torch.nn.Module,
    xt: torch.Tensor,
    t_scalar: torch.Tensor,
    enc_hs_cond: torch.Tensor,
    null_emb: torch.Tensor,
    enc_mask_cond: torch.Tensor,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    diffusion_guidance_scale: float,
    use_adg: bool,
    apply_cfg_guidance: bool,
    momentum_buffer: MomentumBuffer,
) -> torch.Tensor:
    device, dtype = xt.device, xt.dtype
    encoder_hidden_states = torch.cat([enc_hs_cond, null_emb], dim=0)
    encoder_attention_mask = torch.cat([enc_mask_cond, enc_mask_cond], dim=0)
    ctx2 = torch.cat([context_latents, context_latents], dim=0)
    attn2 = torch.cat([attention_mask, attention_mask], dim=0)
    x2 = torch.cat([xt, xt], dim=0)
    t_tensor = t_scalar * torch.ones((x2.shape[0],), device=device, dtype=dtype)
    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
    vt, _ = model.decoder(
        hidden_states=x2,
        timestep=t_tensor,
        timestep_r=t_tensor,
        attention_mask=attn2,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=ctx2,
        use_cache=False,
        past_key_values=past_key_values,
    )
    pred_cond, pred_null = vt.chunk(2)
    if not apply_cfg_guidance:
        return pred_cond
    if not use_adg:
        return apg_forward(
            pred_cond=pred_cond,
            pred_uncond=pred_null,
            guidance_scale=diffusion_guidance_scale,
            momentum_buffer=momentum_buffer,
            dims=[1],
        )
    return adg_forward(
        latents=xt,
        noise_pred_cond=pred_cond,
        noise_pred_uncond=pred_null,
        sigma=t_scalar,
        guidance_scale=diffusion_guidance_scale,
    )


class NullTextInversionAceStep:
    """Optimize unconditional encoder states per diffusion step (same role as ``music_p2p.nti.NullTextOptimization``)."""

    def __init__(
        self,
        *,
        lr: float = 1e-2,
        num_inner_steps: int = 15,
        epsilon: float = 1e-7,
        latent_integrator: NtiLatentIntegrator = "euler",
        writer: BaseWriter | None = None,
        debug_mode: bool = False,
    ):
        self._lr = lr
        self._num_inner_steps = num_inner_steps
        self._epsilon = epsilon
        assert latent_integrator in ("euler", "heun")
        self._latent_integrator = latent_integrator
        self._writer = writer or DummyWriter()
        self._debug_mode = debug_mode

    def run(
        self,
        model: torch.nn.Module,
        handler: Any,
        payload: Dict[str, Any],
        trajectory: List[torch.Tensor],
        *,
        infer_steps: int,
        shift: float,
        diffusion_guidance_scale: float,
        cfg_interval_start: float,
        cfg_interval_end: float,
        use_adg: bool,
        use_progress_bar: bool = True,
    ) -> List[torch.Tensor]:
        if len(trajectory) != infer_steps + 1:
            raise ValueError(
                f"trajectory length {len(trajectory)} != infer_steps + 1 ({infer_steps + 1})"
            )
        if infer_steps < 1:
            raise ValueError("NTI requires infer_steps >= 1")
        if diffusion_guidance_scale <= 1.0:
            raise ValueError("NTI requires diffusion_guidance_scale > 1.0")

        # LR schedule like null-text-w-ptp: lr *= (1 - step / ref) with ref = 2 * inference_steps.
        lr_decay_ref = 2.0 * float(infer_steps)

        enc_hs_cond, enc_mask_cond, context_latents, attention_mask = _prepare_condition_tensors(
            model, handler, payload
        )
        base_null = model.null_condition_emb.expand_as(enc_hs_cond).detach()

        device = enc_hs_cond.device
        dtype = enc_hs_cond.dtype
        t = acestep_sigma_grid(infer_steps, shift, device=device, dtype=dtype)

        # Pivot is often built under ``torch.inference_mode()``; those tensors cannot mix with
        # autograd. Detach+clone yields ordinary tensors for targets and rolling state.
        trajectory = [x.detach().clone().to(device=device, dtype=dtype) for x in trajectory]

        null_list: List[torch.Tensor] = []
        latent_cur = trajectory[0]
        bsz = latent_cur.shape[0]
        null_carry: torch.Tensor | None = None

        outer_it = range(infer_steps)
        if use_progress_bar:
            outer_it = tqdm(
                outer_it,
                total=infer_steps,
                desc=f"Null-text inversion ({self._latent_integrator})",
            )

        for step_idx in outer_it:
            t_curr, t_prev = t[step_idx], t[step_idx + 1]
            dt = t_curr - t_prev
            latent_next = trajectory[step_idx + 1]
            apply_cfg = cfg_interval_start <= float(t_curr) <= cfg_interval_end

            if not isinstance(self._writer, DummyWriter):
                self._writer.set_step(step_idx)
                self._writer.add_scalar("nti/cfg_active", 1.0 if apply_cfg else 0.0)

            with torch.no_grad():
                vt_cond = _velocity_cond_only(
                    model,
                    latent_cur,
                    t_curr,
                    enc_hs_cond,
                    enc_mask_cond,
                    context_latents,
                    attention_mask,
                )

            if not apply_cfg:
                null_list.append(base_null.clone())
                dt_tensor = _dt_tensor(dt, bsz, device, dtype)
                if self._latent_integrator == "euler":
                    latent_cur = latent_cur - vt_cond * dt_tensor
                else:
                    lat_e = latent_cur - vt_cond * dt_tensor
                    vt2 = _velocity_cond_only(
                        model,
                        lat_e,
                        t_prev,
                        enc_hs_cond,
                        enc_mask_cond,
                        context_latents,
                        attention_mask,
                    )
                    latent_cur = latent_cur - 0.5 * (vt_cond + vt2) * dt_tensor
                continue

            if null_carry is not None:
                null_emb = null_carry.clone().detach().requires_grad_(True)
            else:
                null_emb = base_null.clone().detach().requires_grad_(True)

            step_lr = float(self._lr) * max(0.0, 1.0 - float(step_idx) / lr_decay_ref)
            step_lr = max(step_lr, 1e-12)
            if not isinstance(self._writer, DummyWriter):
                self._writer.add_scalar("nti/lr", step_lr, step=step_idx)

            optimizer = Adam([null_emb], lr=step_lr)

            dt_tensor = _dt_tensor(dt, bsz, device, dtype)
            # ACE-Step ``MomentumBuffer`` has no ``detach()`` (unlike music_p2p). Fresh buffer
            # per inner step avoids carrying ``running_average`` across Adam steps / autograd.
            for inner_i in range(self._num_inner_steps):
                lat_prev = _predict_latent_cfg_step(
                    model,
                    latent_cur,
                    t_curr,
                    t_prev,
                    null_emb,
                    dt_tensor,
                    enc_hs_cond,
                    enc_mask_cond,
                    context_latents,
                    attention_mask,
                    diffusion_guidance_scale=diffusion_guidance_scale,
                    use_adg=use_adg,
                    integrator=self._latent_integrator,
                )
                loss = F.mse_loss(lat_prev, latent_next)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                inner_global = step_idx * self._num_inner_steps + inner_i
                loss_f = float(loss.detach().item())
                if not isinstance(self._writer, DummyWriter):
                    self._writer.add_scalar("nti/inner/loss", loss_f, step=inner_global)
                if loss_f < self._epsilon:
                    break

            if self._debug_mode or not isinstance(self._writer, DummyWriter):
                norm_f = float(torch.norm(null_emb.detach()).item())
                if not isinstance(self._writer, DummyWriter):
                    self._writer.add_scalar("nti/outer/loss_final", loss_f)
                    self._writer.add_scalar("nti/outer/null_emb_norm", norm_f)
                if self._debug_mode:
                    logging.info(
                        f"NTI step {step_idx} nti/outer/loss_final={loss_f:.6f} "
                        f"nti/outer/null_emb_norm={norm_f:.6f}"
                    )

            null_carry = null_emb.detach().clone()
            null_list.append(null_carry.clone())

            with torch.no_grad():
                latent_cur = _predict_latent_cfg_step(
                    model,
                    latent_cur,
                    t_curr,
                    t_prev,
                    null_carry,
                    dt_tensor,
                    enc_hs_cond,
                    enc_mask_cond,
                    context_latents,
                    attention_mask,
                    diffusion_guidance_scale=diffusion_guidance_scale,
                    use_adg=use_adg,
                    integrator=self._latent_integrator,
                )

        return null_list
