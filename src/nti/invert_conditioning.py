"""Сборка кондиционирования для инверсии в духе music_p2p / text2music.

ACE-Step смешивает текст и **src_latents** в ``prepare_condition``. Если инвертировать с
латентами **реального трека** в ``src_latents``, а ``generate`` из артефакта идёт с **тишиной**
в ``src_latents``, условие другое → NTI и шум не согласованы с генерацией.

Здесь батч строится как у обычного text2music: **тишина** той же длины, что и клип
(те же метаданные длительности). Латенты **музыки** кодируются отдельно и возвращаются как
``clean_acoustic_latents`` — только конечная точка pivot (что восстанавливаем), не путь в
``prepare_condition``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch


def _pad_latent_time(
    handler: Any,
    lat: torch.Tensor,
    target_T: int,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``lat`` (T0, C) → (1, target_T, C), паддинг срезом тишины."""
    if lat.dim() != 2:
        raise ValueError(f"expected latent (T, C), got {tuple(lat.shape)}")
    lat = lat.to(device=handler.device, dtype=out_dtype)
    t0 = lat.shape[0]
    if t0 > target_T:
        lat = lat[:target_T].contiguous()
        t0 = target_T
    if t0 < target_T:
        pad = handler._get_silence_latent_slice(target_T - t0).to(device=handler.device, dtype=out_dtype)
        lat = torch.cat([lat, pad], dim=0)
    return lat.unsqueeze(0)


def build_invert_payload(
    handler: Any,
    *,
    captions: str,
    lyrics: str,
    vocal_language: str,
    music_stereo_48k: torch.Tensor,
    infer_steps: int,
    seed: Union[int, None] = 0,
    chunk_mask_modes: Optional[list[str]] = None,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Как ``service_generate`` для **text2music + тишина** той же длины, что клип.

    Returns:
        ``payload`` — unpacked preprocess (кондиционирование как при ``generate`` из артефакта).
        ``clean_acoustic_latents`` — VAE-латент входного трека, (1, T, C), T совпадает с
        ``payload['src_latents'].shape[1]`` (конец pivot).
    """
    handler._ensure_silence_latent_on_device()

    if music_stereo_48k.dim() != 2 or music_stereo_48k.shape[0] != 2:
        raise ValueError(f"Expected music tensor [2, T], got {tuple(music_stereo_48k.shape)}")

    pure_caption = handler.extract_caption_from_sft_format(captions)
    instruction = handler.generate_instruction("text2music")
    duration_sec = float(music_stereo_48k.shape[-1]) / float(handler.sample_rate)
    meta = handler._build_metadata_dict(None, "", "", duration_sec)

    normalized = handler._normalize_service_generate_inputs(
        captions=pure_caption,
        lyrics=lyrics,
        keys=None,
        metas=meta,
        vocal_languages=vocal_language,
        repainting_start=None,
        repainting_end=None,
        instructions=instruction,
        audio_code_hints=None,
        infer_steps=infer_steps,
        seed=seed,
        return_intermediate=False,
    )

    # Тишина, тот же число сэмплов → тот же ``max_latent_length``, что дал бы клип нулевой громкости.
    silent_wav = torch.zeros_like(music_stereo_48k)
    target_wavs = silent_wav.unsqueeze(0).to(handler.device).to(handler._get_vae_dtype())

    batch = handler._prepare_batch(
        captions=normalized["captions"],
        global_captions=None,
        lyrics=normalized["lyrics"],
        keys=normalized["keys"],
        target_wavs=target_wavs,
        refer_audios=None,
        metas=normalized["metas"],
        vocal_languages=normalized["vocal_languages"],
        repainting_start=normalized["repainting_start"],
        repainting_end=normalized["repainting_end"],
        instructions=normalized["instructions"],
        audio_code_hints=normalized["audio_code_hints"],
        audio_cover_strength=1.0,
        cover_noise_strength=0.0,
        chunk_mask_modes=chunk_mask_modes,
    )

    target_T = int(batch["src_latents"].shape[1])
    out_dtype = batch["src_latents"].dtype

    music_on_dev = music_stereo_48k.to(handler.device).to(handler._get_vae_dtype())
    music_lat = handler._encode_audio_to_latents(music_on_dev)
    clean_acoustic_latents = _pad_latent_time(handler, music_lat, target_T, out_dtype=out_dtype)

    if clean_acoustic_latents.shape != batch["src_latents"].shape:
        raise RuntimeError(
            f"internal: clean latents {tuple(clean_acoustic_latents.shape)} != batch src "
            f"{tuple(batch['src_latents'].shape)}"
        )

    processed = handler.preprocess_batch(batch)
    payload = handler._unpack_service_processed_data(processed)
    return payload, clean_acoustic_latents
