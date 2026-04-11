"""Один код-путь: cfg → ``generate_music``; опционально патчи ``prepare_noise`` / ``generate_audio`` для артефакта."""

from __future__ import annotations

import types
from typing import Any

import torch
from omegaconf import DictConfig

from src.acestep_artifact_diffusion import bind_generate_audio_patch
from src.artifact_bundle import GenerationArtifactPayload


def generate_music_kwargs_from_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Маппинг структуры generate-конфига в kwargs ``AceStepHandler.generate_music``."""
    audio_duration = float(cfg.duration) if float(cfg.duration) > 0 else None
    seed_arg: str | int = -1 if bool(cfg.use_random_seed) else int(cfg.seed)
    return {
        "captions": str(cfg.prompt.captions),
        "lyrics": str(cfg.prompt.lyrics),
        "vocal_language": str(cfg.vocal_language),
        "inference_steps": int(cfg.inference_steps),
        "guidance_scale": float(cfg.guidance_scale),
        "use_random_seed": bool(cfg.use_random_seed),
        "seed": seed_arg,
        "audio_duration": audio_duration,
        "batch_size": int(cfg.batch_size),
        "task_type": str(cfg.task_type),
        "shift": float(cfg.shift),
        "infer_method": str(cfg.infer_method),
        "sampler_mode": str(cfg.sampler_mode),
        "use_adg": bool(cfg.use_adg),
        "cfg_interval_start": float(cfg.cfg_interval_start),
        "cfg_interval_end": float(cfg.cfg_interval_end),
        "audio_code_string": "",
    }


def _patch_prepare_noise(model: Any, fixed: torch.Tensor) -> Any:
    orig = model.prepare_noise

    def patched(self: Any, context_latents: torch.Tensor, seed=None) -> torch.Tensor:
        b, t, q = context_latents.shape[0], context_latents.shape[1], context_latents.shape[-1]
        expected_c = q // 2
        exp_shape = (b, t, expected_c)
        x = fixed
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape != exp_shape:
            raise ValueError(
                f"Artifact noise shape {tuple(x.shape)} != DiT expected {exp_shape} "
                f"(from context_latents {tuple(context_latents.shape)})"
            )
        return x.to(device=context_latents.device, dtype=context_latents.dtype)

    model.prepare_noise = types.MethodType(patched, model)
    return orig


def run_generate(
    handler: Any,
    work_cfg: DictConfig,
    artifact: GenerationArtifactPayload | None,
) -> dict[str, Any]:
    """kwargs из ``work_cfg``; при непустом payload — патчи на ``handler.model``."""
    payload = artifact or GenerationArtifactPayload()
    mlx = bool(getattr(handler, "use_mlx_dit", False)) and getattr(handler, "mlx_decoder", None) is not None
    if payload.uses_mlx_incompatible_override() and mlx:
        raise RuntimeError(
            "Артефакт с noise или null-text по шагам поддерживается только с PyTorch DiT "
            "(acestep.use_mlx_dit=false)."
        )

    kwargs = generate_music_kwargs_from_cfg(work_cfg)
    model = handler.model

    null_list = payload.null_encoder_hidden_states_per_step
    needs_nti_path = null_list is not None

    orig_prepare_noise = None
    orig_generate_audio = None
    try:
        if payload.noise is not None:
            orig_prepare_noise = _patch_prepare_noise(model, payload.noise)
        if needs_nti_path:
            orig_generate_audio = model.generate_audio
            patched_ga, _ = bind_generate_audio_patch(
                model,
                orig_generate_audio,
                null_encoder_hidden_states_per_step=null_list,
            )
            model.generate_audio = patched_ga

        return handler.generate_music(**kwargs)
    finally:
        if orig_generate_audio is not None:
            model.generate_audio = orig_generate_audio
        if orig_prepare_noise is not None:
            model.prepare_noise = orig_prepare_noise
