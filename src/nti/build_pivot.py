"""Inverse ODE along the ACE-Step schedule: clean latent → approximate initial noise (pivot trajectory)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from src.nti.schedule import acestep_sigma_grid

PivotIntegrator = Literal["euler", "heun", "uni_inv"]


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


def _segment_dt_tensor(
    dt: torch.Tensor, bsz: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return dt * torch.ones((bsz,), device=device, dtype=dtype).view(-1, 1, 1)


def _pivot_step_euler(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    t_curr: torch.Tensor,
    dt_tensor: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    v = _velocity_cond_only(
        model, x, t_curr, encoder_hidden_states, encoder_attention_mask, context_latents, attention_mask
    )
    return x + v * dt_tensor


def _pivot_step_heun(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    dt_tensor: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    v_lo = _velocity_cond_only(
        model, x, t_next, encoder_hidden_states, encoder_attention_mask, context_latents, attention_mask
    )
    x_e = x + v_lo * dt_tensor
    v_hi = _velocity_cond_only(
        model, x_e, t_curr, encoder_hidden_states, encoder_attention_mask, context_latents, attention_mask
    )
    return x + 0.5 * (v_lo + v_hi) * dt_tensor


def _pivot_step_uni_inv(
    model: torch.nn.Module,
    x: torch.Tensor,
    v_hat_prev: torch.Tensor,
    *,
    t_curr: torch.Tensor,
    dt_tensor: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictor-corrector inverse step (Uni-Inv / UniEdit-Flow style; see ``uninv.md``)."""
    z_bar = x + v_hat_prev * dt_tensor
    v_hat = _velocity_cond_only(
        model, z_bar, t_curr, encoder_hidden_states, encoder_attention_mask, context_latents, attention_mask
    )
    x_next = x + v_hat * dt_tensor
    return x_next, v_hat


def _invert_payload_prepare_condition(
    model: torch.nn.Module,
    handler: Any,
    payload: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_latents = payload["src_latents"]
    device, dtype = src_latents.device, src_latents.dtype
    attention_mask = torch.ones(
        src_latents.shape[0], src_latents.shape[1], device=device, dtype=dtype
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
def build_pivot_trajectory(
    model: torch.nn.Module,
    handler: Any,
    payload: Dict[str, Any],
    *,
    clean_latents: torch.Tensor,
    infer_steps: int,
    shift: float,
    pivot_integrator: PivotIntegrator = "euler",
    use_progress_bar: bool = True,
) -> List[torch.Tensor]:
    """Trajectory length ``infer_steps + 1``: noise-like → … → clean (same indexing as cond-only forward).

    Integrates ``dx/dσ = v(x, σ)`` from ``σ≈0`` (``clean_latents``) toward ``σ≈1`` on ``acestep_sigma_grid``.
    ``pivot_integrator``: ``euler`` (one ``v`` eval per segment, velocity at ``t[i]``), ``heun`` (two evals),
    or ``uni_inv`` (predictor-corrector; carries velocity between segments).
    """
    assert pivot_integrator in ("euler", "heun", "uni_inv")

    src = payload["src_latents"]
    device, dtype = src.device, src.dtype
    bsz = src.shape[0]
    cl = clean_latents.to(device=device, dtype=dtype)
    if cl.shape != src.shape:
        raise ValueError(f"clean_latents {tuple(cl.shape)} must match src_latents {tuple(src.shape)}")

    enc_hs, enc_mask, ctx_lat, attn_mask = _invert_payload_prepare_condition(model, handler, payload)
    t = acestep_sigma_grid(infer_steps, shift, device=device, dtype=dtype)

    x = cl.detach().clone()
    back: List[torch.Tensor] = [x.clone()]
    v_hat_uni: torch.Tensor | None = None
    if pivot_integrator == "uni_inv":
        v_hat_uni = _velocity_cond_only(
            model, x, t[infer_steps], enc_hs, enc_mask, ctx_lat, attn_mask
        )
    indices = range(infer_steps - 1, -1, -1)
    if use_progress_bar:
        indices = tqdm(indices, total=infer_steps, desc=f"Pivot ({pivot_integrator})")

    for i in indices:
        t_curr, t_next = t[i], t[i + 1]
        dt = t_curr - t_next
        if dt.abs() < 1e-12:
            continue
        dt_t = _segment_dt_tensor(dt, bsz, device, dtype)
        if pivot_integrator == "euler":
            x = _pivot_step_euler(
                model, x, t_curr=t_curr, dt_tensor=dt_t,
                encoder_hidden_states=enc_hs, encoder_attention_mask=enc_mask,
                context_latents=ctx_lat, attention_mask=attn_mask,
            )
        elif pivot_integrator == "heun":
            x = _pivot_step_heun(
                model, x, t_curr=t_curr, t_next=t_next, dt_tensor=dt_t,
                encoder_hidden_states=enc_hs, encoder_attention_mask=enc_mask,
                context_latents=ctx_lat, attention_mask=attn_mask,
            )
        else:
            assert v_hat_uni is not None
            x, v_hat_uni = _pivot_step_uni_inv(
                model,
                x,
                v_hat_uni,
                t_curr=t_curr,
                dt_tensor=dt_t,
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_mask,
                context_latents=ctx_lat,
                attention_mask=attn_mask,
            )
        back.append(x.detach().clone())

    traj = list(reversed(back))
    assert len(traj) == infer_steps + 1
    return traj
