"""Batch text conditioning → ``prepare_condition`` tensors for DiT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf

from src.schemas import PromptConfig


@dataclass
class ModelCondition:
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    context_latents: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Any | None = None
    clean_latents: torch.Tensor | None = None  # set when ``prepare_conditions(..., source_stereo_wav=...)``


def _unpack_preprocess_tuple(processed_data: Tuple[Any, ...]) -> dict[str, Any]:
    """Mirror ``ServiceGenerateExecuteMixin._unpack_service_processed_data``."""
    (
        keys,
        text_inputs,
        src_latents,
        target_latents,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        _audio_attention_mask,
        refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask,
        chunk_mask,
        spans,
        is_covers,
        _audio_codes,
        lyric_token_idss,
        precomputed_lm_hints_25Hz,
        non_cover_text_hidden_states,
        non_cover_text_attention_masks,
        repaint_mask,
    ) = processed_data
    return {
        "keys": keys,
        "text_inputs": text_inputs,
        "src_latents": src_latents,
        "target_latents": target_latents,
        "text_hidden_states": text_hidden_states,
        "text_attention_mask": text_attention_mask,
        "lyric_hidden_states": lyric_hidden_states,
        "lyric_attention_mask": lyric_attention_mask,
        "refer_audio_acoustic_hidden_states_packed": refer_audio_acoustic_hidden_states_packed,
        "refer_audio_order_mask": refer_audio_order_mask,
        "chunk_mask": chunk_mask,
        "spans": spans,
        "is_covers": is_covers,
        "lyric_token_idss": lyric_token_idss,
        "precomputed_lm_hints_25Hz": precomputed_lm_hints_25Hz,
        "non_cover_text_hidden_states": non_cover_text_hidden_states,
        "non_cover_text_attention_masks": non_cover_text_attention_masks,
        "repaint_mask": repaint_mask,
    }


def _prompts_to_lists(prompts: List[PromptConfig]) -> tuple[list[str], list[str], list[str]]:
    captions = [p.captions for p in prompts]
    lyrics = [p.lyrics for p in prompts]
    langs = [p.vocal_language for p in prompts]
    return captions, lyrics, langs


def _silent_target_wavs(handler: Any, batch_size: int, duration: float) -> torch.Tensor:
    """Placeholder stereo waveform batch: ``_prepare_batch`` requires a tensor, not ``None``.

    Silent audio yields silence latents inside ACE-Step ``_prepare_target_latents_and_wavs``.
    """
    sr = int(getattr(handler, "sample_rate", 48000))
    seconds = float(duration) if duration is not None and float(duration) > 0 else 30.0
    n_samples = max(int(seconds * sr), sr // 10)
    return torch.zeros(
        batch_size,
        2,
        n_samples,
        device=handler.device,
        dtype=torch.float32,
    )


def _pad_latent_time(lat: torch.Tensor, target_t: int) -> torch.Tensor:
    """Pad or crop latent time dim ``T`` to ``target_t``. ``lat`` is ``[B, T, C]``."""
    if lat.dim() != 3:
        raise ValueError(f"Expected latents [B, T, C], got shape {tuple(lat.shape)}")
    _b, t, _c = lat.shape
    if t == target_t:
        return lat
    if t > target_t:
        return lat[:, :target_t].contiguous()
    pad_len = target_t - t
    return F.pad(lat, (0, 0, 0, pad_len))


@torch.inference_mode()
def prepare_conditions(
    handler: Any,
    prompts: List[PromptConfig],
    duration: float,
    *,
    source_stereo_wav: torch.Tensor | None = None,
) -> ModelCondition:
    """Build batched ``ModelCondition`` from prompts (B = len(prompts)).

    If ``source_stereo_wav`` is set (stereo ``[2, samples]`` at handler sample rate), the temporal grid
    uses the clip length (``samples / sample_rate``), ignoring ``duration`` for that purpose, and
    ``clean_latents`` on the result holds VAE latents padded to match ``context_latents``. Requires
    ``batch_size == 1`` in v1.
    """
    if not prompts:
        raise ValueError("prepare_conditions: prompts list is empty")

    bsz = len(prompts)
    captions, lyrics, vocal_languages = _prompts_to_lists(prompts)

    grid_duration = float(duration)
    clean_out: torch.Tensor | None = None
    if source_stereo_wav is not None:
        if bsz != 1:
            raise ValueError("prepare_conditions with source_stereo_wav: batch size 1 only in v1")
        if source_stereo_wav.dim() != 2 or source_stereo_wav.shape[0] != 2:
            raise ValueError(f"source_stereo_wav must be [2, T], got {tuple(source_stereo_wav.shape)}")
        sr = float(getattr(handler, "sample_rate", 48000))
        grid_duration = float(source_stereo_wav.shape[-1]) / sr
        if grid_duration <= 0:
            raise ValueError("Non-positive duration derived from source_stereo_wav")

    handler._ensure_silence_latent_on_device()

    metas: list[Any] = [None] * bsz
    if grid_duration > 0:
        metas = [{"duration": float(grid_duration)} for _ in range(bsz)]

    target_wavs = _silent_target_wavs(handler, bsz, grid_duration)

    batch = handler._prepare_batch(
        captions=captions,
        lyrics=lyrics,
        keys=[f"sample_{i}" for i in range(bsz)],
        target_wavs=target_wavs,
        refer_audios=None,
        metas=metas,
        vocal_languages=vocal_languages,
        repainting_start=None,
        repainting_end=None,
        instructions=None,
        audio_code_hints=None,
        audio_cover_strength=1.0,
        cover_noise_strength=0.0,
        chunk_mask_modes=None,
    )
    processed = handler.preprocess_batch(batch)
    payload = _unpack_preprocess_tuple(processed)

    src = payload["src_latents"]
    attn = torch.ones(
        src.shape[0],
        src.shape[1],
        device=src.device,
        dtype=src.dtype,
    )

    enc_hs, enc_am, ctx = handler.model.prepare_condition(
        text_hidden_states=payload["text_hidden_states"],
        text_attention_mask=payload["text_attention_mask"],
        lyric_hidden_states=payload["lyric_hidden_states"],
        lyric_attention_mask=payload["lyric_attention_mask"],
        refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
        refer_audio_order_mask=payload["refer_audio_order_mask"],
        hidden_states=src,
        attention_mask=attn,
        silence_latent=handler.silence_latent,
        src_latents=src,
        chunk_masks=payload["chunk_mask"],
        is_covers=payload["is_covers"],
        precomputed_lm_hints_25Hz=payload.get("precomputed_lm_hints_25Hz"),
        audio_codes=None,
    )
    if source_stereo_wav is not None:
        target_t = int(ctx.shape[1])
        out_dtype = ctx.dtype
        music_on_dev = source_stereo_wav.to(handler.device).to(handler._get_vae_dtype())
        music_lat = handler._encode_audio_to_latents(music_on_dev)
        if music_lat.dim() == 2:
            music_lat = music_lat.unsqueeze(0)
        clean_out = _pad_latent_time(music_lat, target_t).to(dtype=out_dtype)
        if clean_out.shape[0] != bsz or clean_out.shape[1] != target_t:
            raise RuntimeError(
                f"clean latents shape {tuple(clean_out.shape)} vs ctx {tuple(ctx.shape)}"
            )

    logger.info(
        "prepare_condition: batch={}, enc_hs.shape={}, ctx.shape={}",
        bsz,
        tuple(enc_hs.shape),
        tuple(ctx.shape),
    )
    if clean_out is not None:
        logger.info("prepare_condition: clean_latents.shape={}", tuple(clean_out.shape))
    return ModelCondition(
        encoder_hidden_states=enc_hs,
        encoder_attention_mask=enc_am,
        context_latents=ctx,
        attention_mask=attn,
        past_key_values=None,
        clean_latents=clean_out,
    )


def prompts_from_hydra_prompt_node(prompt_cfg: Any, batch_size: int) -> List[PromptConfig]:
    """Expand a single Hydra ``prompt`` config node to ``batch_size`` `PromptConfig` instances."""
    d = OmegaConf.to_container(prompt_cfg, resolve=True)
    if not isinstance(d, dict):
        raise TypeError("prompt config must resolve to a dict")
    base = PromptConfig(
        captions=str(d.get("captions", "")),
        lyrics=str(d.get("lyrics", "")),
        vocal_language=str(d.get("vocal_language", "en")),
    )
    return [PromptConfig(captions=base.captions, lyrics=base.lyrics, vocal_language=base.vocal_language) for _ in range(batch_size)]


def prompt_config_from_hydra_node(prompt_node: Any) -> PromptConfig:
    """Single ``PromptConfig`` from a Hydra prompt subnode (e.g. ``cfg.p2p_task.src``)."""
    d = OmegaConf.to_container(prompt_node, resolve=True)
    if not isinstance(d, dict):
        raise TypeError("prompt node must resolve to a dict")
    return PromptConfig(
        captions=str(d.get("captions", "")),
        lyrics=str(d.get("lyrics", "")),
        vocal_language=str(d.get("vocal_language", "en")),
    )


def p2p_src_tgt_prompt_configs(p2p_task_cfg: Any) -> List[PromptConfig]:
    """``[src, tgt]`` from Hydra ``p2p_task`` (nodes ``src`` / ``tgt`` from ``prompt@p2p_task.*``)."""
    src = OmegaConf.select(p2p_task_cfg, "src", default=None)
    tgt = OmegaConf.select(p2p_task_cfg, "tgt", default=None)
    if src is None or tgt is None:
        raise ValueError("p2p_task must contain src and tgt prompt nodes (use prompt@p2p_task.src / prompt@p2p_task.tgt in defaults)")
    return [prompt_config_from_hydra_node(src), prompt_config_from_hydra_node(tgt)]
