"""Inverse ODE along the ACE-Step schedule: clean latent → approximate initial noise (pivot trajectory)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from src.nti.schedule import acestep_sigma_grid


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


@torch.no_grad()
def build_pivot_trajectory(
    model: torch.nn.Module,
    handler: Any,
    payload: Dict[str, Any],
    *,
    clean_latents: torch.Tensor,
    infer_steps: int,
    shift: float,
    infer_method: str = "ode",
    use_progress_bar: bool = True,
) -> List[torch.Tensor]:
    """Return ``trajectory`` length ``infer_steps + 1``: ``[x_{t=1}, …, x_{t=0}]`` (noise → clean).

    ``prepare_condition`` берётся из ``payload`` (как при генерации). Конец ODE — ``clean_latents``
    (латент целевого аудио), форма как у ``payload['src_latents']``.

    Inverse integration mirrors the forward Euler step ``x_{next} = x - v(x,t_{curr}) * dt``
    with explicit approximation ``x_{curr} ≈ x_{next} + v(x_{next}, t_{curr}) * dt`` along
    decreasing time indices (cf. ``music_p2p`` pivot + Flow-Match inverse idea).

    Only ``infer_method == \"ode\"`` is supported for the pivot.
    """
    if infer_method != "ode":
        raise ValueError(f"build_pivot_trajectory: only infer_method='ode' is supported, got {infer_method!r}")

    src_latents = payload["src_latents"]
    device, dtype = src_latents.device, src_latents.dtype

    cl = clean_latents.to(device=device, dtype=dtype)
    if cl.shape != src_latents.shape:
        raise ValueError(
            f"clean_latents shape {tuple(cl.shape)} must match payload src_latents {tuple(src_latents.shape)}"
        )

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

    t = acestep_sigma_grid(infer_steps, shift, device=device, dtype=dtype)

    x = cl.detach().clone()
    back_states: List[torch.Tensor] = [x.clone()]

    indices = range(infer_steps - 1, -1, -1)
    if use_progress_bar:
        indices = tqdm(indices, total=infer_steps, desc="Pivot (inverse ODE)")

    for i in indices:
        t_curr = t[i]
        t_next = t[i + 1]
        dt = t_curr - t_next
        if dt.abs() < 1e-12:
            continue
        v = _velocity_cond_only(
            model,
            x,
            t_curr,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
            attention_mask,
        )
        dt_tensor = dt * torch.ones((x.shape[0],), device=device, dtype=dtype).view(-1, 1, 1)
        x = x + v * dt_tensor
        back_states.append(x.detach().clone())

    trajectory = list(reversed(back_states))
    if len(trajectory) != infer_steps + 1:
        raise RuntimeError(f"internal: expected {infer_steps + 1} pivot states, got {len(trajectory)}")
    return trajectory
