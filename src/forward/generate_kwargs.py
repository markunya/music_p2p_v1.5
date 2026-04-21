"""Маппинг Hydra ``generate``-конфига в kwargs ``AceStepHandler.generate_music`` (для CLI/логов)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig


def generate_music_kwargs_from_cfg(cfg: DictConfig) -> dict[str, Any]:
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
