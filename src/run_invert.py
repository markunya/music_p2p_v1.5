"""Orchestration: payload → pivot trajectory → NTI → ``save_generation_artifact``."""

from __future__ import annotations

from src.mps_adg_patch import apply_adg_mps_patch

apply_adg_mps_patch()

from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.artifact_bundle import save_generation_artifact
from src.logging import utils as logging
from src.logging.writer import BaseWriter, setup_writer
from src.nti.build_pivot import PivotIntegrator, build_pivot_trajectory
from src.nti.invert_conditioning import build_invert_payload
from src.nti.null_text_inversion import NtiLatentIntegrator, NullTextInversionAceStep
from src.nti.pivot_trajectory_viz import log_pivot_trajectory_comet


def run_invert(
    handler: Any,
    work_cfg: DictConfig,
    *,
    music_path: str,
    artifact_out: Path,
    writer: BaseWriter | None = None,
) -> None:
    mlx = bool(getattr(handler, "use_mlx_dit", False)) and getattr(handler, "mlx_decoder", None) is not None
    if mlx:
        raise RuntimeError("Null-text inversion requires PyTorch DiT (acestep.use_mlx_dit=false).")
    if getattr(handler, "config", None) is not None and bool(getattr(handler.config, "is_turbo", False)):
        raise RuntimeError("Turbo checkpoints do not use CFG; NTI is not applicable (use a non-turbo config).")

    if str(work_cfg.infer_method) != "ode":
        raise ValueError(f"invert: only infer_method='ode' is supported, got {work_cfg.infer_method!r}")
    sampler_mode = str(work_cfg.sampler_mode)
    if sampler_mode != "euler":
        raise ValueError(f"invert: only sampler_mode='euler' is supported for now, got {sampler_mode!r}")

    pivot_integrator = str(OmegaConf.select(work_cfg, "pivot_integrator", default="euler"))
    if pivot_integrator not in ("euler", "heun"):
        raise ValueError(f"invert: pivot_integrator must be 'euler' or 'heun', got {pivot_integrator!r}")

    nti_latent_integrator = str(OmegaConf.select(work_cfg, "nti_latent_integrator", default="euler"))
    if nti_latent_integrator not in ("euler", "heun"):
        raise ValueError(
            f"invert: nti_latent_integrator must be 'euler' or 'heun', got {nti_latent_integrator!r}"
        )

    infer_steps = int(work_cfg.inference_steps)
    guidance = float(work_cfg.guidance_scale)
    if guidance <= 1.0:
        raise ValueError("invert: guidance_scale must be > 1.0 for NTI")

    wav = handler.process_target_audio(music_path)
    if wav is None:
        raise RuntimeError(f"Failed to load or process audio: {music_path}")

    seed = int(work_cfg.seed) if not bool(work_cfg.use_random_seed) else 0
    payload, clean_latents = build_invert_payload(
        handler,
        captions=str(work_cfg.prompt.captions),
        lyrics=str(work_cfg.prompt.lyrics),
        vocal_language=str(work_cfg.vocal_language),
        music_stereo_48k=wav,
        infer_steps=infer_steps,
        seed=seed,
    )

    model = handler.model
    model.eval()

    with torch.inference_mode():
        trajectory = build_pivot_trajectory(
            model,
            handler,
            payload,
            clean_latents=clean_latents,
            infer_steps=infer_steps,
            shift=float(work_cfg.shift),
            pivot_integrator=cast(PivotIntegrator, pivot_integrator),
            use_progress_bar=not bool(getattr(handler, "disable_tqdm", False)),
        )

    own_writer = writer is None
    track = setup_writer(work_cfg) if own_writer else writer
    debug_mode = bool(OmegaConf.select(work_cfg, "debug_mode", default=False))

    if bool(OmegaConf.select(work_cfg, "log_pivot_trajectory_to_comet", default=True)):
        max_edge = int(OmegaConf.select(work_cfg, "log_pivot_trajectory_max_edge", default=4096))
        log_pivot_trajectory_comet(track, trajectory, max_edge=max_edge)

    try:
        nti = NullTextInversionAceStep(
            lr=float(work_cfg.nti_learning_rate),
            num_inner_steps=int(work_cfg.nti_num_inner_steps),
            epsilon=float(work_cfg.nti_epsilon),
            latent_integrator=cast(NtiLatentIntegrator, nti_latent_integrator),
            writer=track,
            debug_mode=debug_mode,
        )

        with torch.set_grad_enabled(True):
            null_list = nti.run(
                model,
                handler,
                payload,
                trajectory,
                infer_steps=infer_steps,
                shift=float(work_cfg.shift),
                diffusion_guidance_scale=guidance,
                cfg_interval_start=float(work_cfg.cfg_interval_start),
                cfg_interval_end=float(work_cfg.cfg_interval_end),
                use_adg=bool(work_cfg.use_adg),
                use_progress_bar=not bool(getattr(handler, "disable_tqdm", False)),
            )
    finally:
        if own_writer:
            track.end()

    noise = trajectory[0].detach().cpu()
    null_cpu = [t.detach().cpu() for t in null_list]

    container = OmegaConf.to_container(work_cfg, resolve=True)
    if isinstance(container, dict) and "hydra" in container:
        container = {k: v for k, v in container.items() if k != "hydra"}
    save_cfg = OmegaConf.create(container)

    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    save_generation_artifact(
        artifact_out,
        save_cfg,
        noise=noise,
        null_encoder_hidden_states_per_step=null_cpu,
    )
    logging.info(f"Saved inversion artifact to {artifact_out.resolve()}")
