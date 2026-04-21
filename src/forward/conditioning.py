"""Подготовка cond для ``model.decoder`` из ACE ``prepare_condition`` (text2music payload)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def prepare_condition_from_payload(
    model: torch.nn.Module,
    handler: Any,
    payload: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_latents = payload["src_latents"]
    device, dtype = src_latents.device, src_latents.dtype
    attention_mask = torch.ones(
        src_latents.shape[0],
        src_latents.shape[1],
        device=device,
        dtype=dtype,
    )
    enc_hs, enc_mask, ctx_lat = model.prepare_condition(
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
    return enc_hs, enc_mask, ctx_lat, attention_mask
