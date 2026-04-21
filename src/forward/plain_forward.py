"""Text2music / артефакт: свой ODE-цикл + ``PlainCfgEulerStepper`` (без ``handler.generate_music``)."""

from __future__ import annotations

import types
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from src.artifact_bundle import GenerationArtifactPayload
from src.forward.conditioning import prepare_condition_from_payload
from src.forward.diffusion_driver import run_euler_ode_loop
from src.forward.steppers.plain_cfg_stepper import PlainCfgEulerStepper
from src.nti.invert_conditioning import build_invert_payload

_LATENT_FRAMES_PER_AUDIO_SECOND = 25.0
_DEFAULT_DURATION_SEC = 60.0


def _resolve_duration_sec(work_cfg: DictConfig, artifact: GenerationArtifactPayload) -> float:
    if artifact.noise is not None:
        return float(artifact.noise.shape[1]) / _LATENT_FRAMES_PER_AUDIO_SECOND
    d = float(work_cfg.duration)
    if d > 0:
        return d
    return _DEFAULT_DURATION_SEC


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
            if (
                x.shape[0] == b
                and x.shape[2] == expected_c
                and abs(x.shape[1] - t) == 1
            ):
                t_x = int(x.shape[1])
                if t_x > t:
                    warnings.warn(
                        f"Artifact noise time T={t_x} trimmed to DiT T={t} (duration→latent "
                        f"rounding vs T/25 mismatch).",
                        UserWarning,
                        stacklevel=2,
                    )
                    x = x[:, :t, :].contiguous()
                else:
                    pad_t = t - t_x
                    warnings.warn(
                        f"Artifact noise time T={t_x} padded to DiT T={t} (zero tail).",
                        UserWarning,
                        stacklevel=2,
                    )
                    x = F.pad(x, (0, 0, 0, pad_t))
            else:
                raise ValueError(
                    f"Artifact noise shape {tuple(x.shape)} != DiT expected {exp_shape} "
                    f"(from context_latents {tuple(context_latents.shape)})"
                )
        return x.to(device=context_latents.device, dtype=context_latents.dtype)

    model.prepare_noise = types.MethodType(patched, model)
    return orig


def run_plain_forward(
    handler: Any,
    work_cfg: DictConfig,
    artifact: GenerationArtifactPayload,
) -> dict[str, Any]:
    mlx = bool(getattr(handler, "use_mlx_dit", False)) and getattr(handler, "mlx_decoder", None) is not None
    if artifact.uses_mlx_incompatible_override() and mlx:
        raise RuntimeError(
            "Артефакт с noise или null-text по шагам поддерживается только с PyTorch DiT "
            "(acestep.use_mlx_dit=false)."
        )

    infer_steps = int(work_cfg.inference_steps)
    null_list = artifact.null_encoder_hidden_states_per_step
    guidance_scale = float(work_cfg.guidance_scale)
    if getattr(handler, "config", None) is not None and bool(getattr(handler.config, "is_turbo", False)):
        guidance_scale = 1.0

    if null_list is not None and guidance_scale <= 1.0:
        raise ValueError("null_encoder_hidden_states_per_step (NTI) requires guidance_scale > 1.0")
    if null_list is not None and guidance_scale > 1.0 and len(null_list) != infer_steps:
        raise ValueError(
            f"len(null_encoder_hidden_states_per_step)={len(null_list)} must equal inference_steps={infer_steps}"
        )

    if int(work_cfg.batch_size) != 1:
        raise NotImplementedError("run_plain_forward supports batch_size=1 only.")

    if artifact.noise is not None:
        t_noise = int(artifact.noise.shape[1])
        if t_noise < 128:
            raise ValueError(
                f"Artifact noise time dimension {t_noise} < 128: задайте T >= 128 или увеличьте duration."
            )

    duration_sec = _resolve_duration_sec(work_cfg, artifact)
    n_samples = max(1, int(duration_sec * float(handler.sample_rate)))
    silent = torch.zeros(2, n_samples, dtype=torch.float32)
    seed = int(work_cfg.seed) if not bool(work_cfg.use_random_seed) else 0

    payload, _clean = build_invert_payload(
        handler,
        captions=str(work_cfg.prompt.captions),
        lyrics=str(work_cfg.prompt.lyrics),
        vocal_language=str(work_cfg.vocal_language),
        music_stereo_48k=silent,
        infer_steps=infer_steps,
        seed=seed,
    )

    model = handler.model
    model.eval()

    enc_cond, enc_mask, ctx_lat_s, attn_s = prepare_condition_from_payload(model, handler, payload)
    if guidance_scale > 1.0:
        context_lat = torch.cat([ctx_lat_s, ctx_lat_s], dim=0)
        attention_mask = torch.cat([attn_s, attn_s], dim=0)
    else:
        context_lat = ctx_lat_s
        attention_mask = attn_s

    seed_for_noise = seed if not bool(work_cfg.use_random_seed) else None
    orig_prepare_noise = None
    try:
        if artifact.noise is not None:
            orig_prepare_noise = _patch_prepare_noise(model, artifact.noise)
        xt = model.prepare_noise(ctx_lat_s, seed_for_noise)
    finally:
        if orig_prepare_noise is not None:
            model.prepare_noise = orig_prepare_noise

    if xt.dim() == 2:
        xt = xt.unsqueeze(0)
    bsz = int(xt.shape[0])

    stepper = PlainCfgEulerStepper(
        model,
        enc_cond,
        enc_mask,
        context_lat,
        attention_mask,
        bsz,
        use_adg=bool(work_cfg.use_adg),
        guidance_scale=guidance_scale,
        cfg_interval_start=float(work_cfg.cfg_interval_start),
        cfg_interval_end=float(work_cfg.cfg_interval_end),
        infer_method=str(work_cfg.infer_method),
        sampler_mode=str(work_cfg.sampler_mode),
        null_encoder_hidden_states_per_step=null_list,
    )

    with torch.inference_mode():
        pred_latents = run_euler_ode_loop(
            xt,
            infer_steps,
            float(work_cfg.shift),
            stepper,
            use_progress_bar=not bool(getattr(handler, "disable_tqdm", False)),
            desc="diffusion (plain)",
        )

    pred_latents_for_decode = pred_latents.transpose(1, 2).contiguous().to(handler.vae.dtype)
    with torch.inference_mode():
        with handler._load_model_context("vae"):
            pred_wavs = handler.tiled_decode(pred_latents_for_decode)
    if pred_wavs.dtype != torch.float32:
        pred_wavs = pred_wavs.float()
    peak = pred_wavs.abs().amax(dim=[1, 2], keepdim=True)
    if torch.any(peak > 1.0):
        pred_wavs = pred_wavs / peak.clamp(min=1.0)

    sr = int(getattr(handler, "sample_rate", 48_000))
    audios = [{"tensor": pred_wavs[i].cpu(), "sample_rate": sr} for i in range(pred_wavs.shape[0])]
    return {
        "success": True,
        "audios": audios,
        "status_message": "done",
        "extra_outputs": {},
        "error": None,
    }
