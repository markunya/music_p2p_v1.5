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

    def clone(self) -> "ModelCondition":
        return ModelCondition(
            encoder_hidden_states=self.encoder_hidden_states.clone(),
            encoder_attention_mask=self.encoder_attention_mask.clone(),
            context_latents=self.context_latents.clone(),
            attention_mask=self.attention_mask.clone(),
            past_key_values=None,
        )

    def slice(self, start: int, end: int) -> "ModelCondition":
        return ModelCondition(
            encoder_hidden_states=self.encoder_hidden_states[start:end].clone(),
            encoder_attention_mask=self.encoder_attention_mask[start:end].clone(),
            context_latents=self.context_latents[start:end].clone(),
            attention_mask=self.attention_mask[start:end].clone(),
            past_key_values=None,
        )


def _unpack_preprocess_tuple(processed_data: Tuple[Any, ...]) -> dict[str, Any]:
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
    if lat.dim() != 3:
        raise ValueError(f"Expected latents [B, T, C], got shape {tuple(lat.shape)}")
    _b, t, _c = lat.shape
    if t == target_t:
        return lat
    if t > target_t:
        return lat[:, :target_t].contiguous()
    pad_len = target_t - t
    return F.pad(lat, (0, 0, 0, pad_len))


@torch.no_grad()
def prepare_conditions(
    handler: Any,
    prompts: List[PromptConfig],
    duration: float,
    *,
    return_unpack: bool = False,
) -> ModelCondition | tuple[ModelCondition, dict[str, Any]]:
    if not prompts:
        raise ValueError("prepare_conditions: prompts list is empty")

    bsz = len(prompts)
    captions, lyrics, vocal_languages = _prompts_to_lists(prompts)

    grid_duration = float(duration)
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

    logger.info(
        "prepare_condition: batch={}, enc_hs.shape={}, ctx.shape={}",
        bsz,
        tuple(enc_hs.shape),
        tuple(ctx.shape),
    )
    out = ModelCondition(
        encoder_hidden_states=enc_hs,
        encoder_attention_mask=enc_am,
        context_latents=ctx,
        attention_mask=attn,
    )
    if return_unpack:
        return out, payload
    return out


def prompts_from_hydra_prompt_node(prompt_cfg: Any, batch_size: int) -> List[PromptConfig]:
    d = OmegaConf.to_container(prompt_cfg, resolve=True)
    base = PromptConfig(
        captions=str(d.get("captions", "")),
        lyrics=str(d.get("lyrics", "")),
        vocal_language=str(d.get("vocal_language", "en")),
    )
    return [PromptConfig(captions=base.captions, lyrics=base.lyrics, vocal_language=base.vocal_language) for _ in range(batch_size)]


def prompt_config_from_hydra_node(prompt_node: Any) -> PromptConfig:
    d = OmegaConf.to_container(prompt_node, resolve=True)
    return PromptConfig(
        captions=str(d.get("captions", "")),
        lyrics=str(d.get("lyrics", "")),
        vocal_language=str(d.get("vocal_language", "en")),
    )


def p2p_src_tgt_prompt_configs(p2p_task_cfg: Any) -> List[PromptConfig]:
    src = OmegaConf.select(p2p_task_cfg, "src", default=None)
    tgt = OmegaConf.select(p2p_task_cfg, "tgt", default=None)
    if src is None or tgt is None:
        raise ValueError("p2p_task must contain src and tgt prompt nodes (use prompt@p2p_task.src / prompt@p2p_task.tgt in defaults)")
    return [prompt_config_from_hydra_node(src), prompt_config_from_hydra_node(tgt)]
