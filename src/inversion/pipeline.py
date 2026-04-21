"""Pivot trajectory + optional NTI → ``InversionArtifact`` (no disk I/O)."""

from __future__ import annotations

from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.inversion.artifact import InversionArtifact
from src.logging.writer import BaseWriter, setup_writer
from src.nti.build_pivot import PivotIntegrator, build_pivot_trajectory
from src.nti.invert_conditioning import build_invert_payload
from src.nti.null_text_inversion import NtiLatentIntegrator, NullTextInversionAceStep
from src.nti.pivot_trajectory_viz import log_pivot_trajectory_comet


def _nti_float(cfg: DictConfig, key: str, default: float) -> float:
    v = OmegaConf.select(cfg, f"nti.{key}")
    return float(v) if v is not None else default


def _nti_int(cfg: DictConfig, key: str, default: int) -> int:
    v = OmegaConf.select(cfg, f"nti.{key}")
    return int(v) if v is not None else default


class InversionPipeline:
    """Инверсия по Hydra-конфигу: ``run`` возвращает ``InversionArtifact`` (без записи на диск)."""

    __slots__ = ("_cfg",)

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def run(
        self,
        handler: Any,
        *,
        music_path: str,
        writer: BaseWriter | None = None,
    ) -> InversionArtifact:
        cfg = self._cfg
        nti_enabled = bool(OmegaConf.select(cfg, "nti.enabled", default=True))
        mlx = bool(getattr(handler, "use_mlx_dit", False)) and getattr(handler, "mlx_decoder", None) is not None

        if nti_enabled and mlx:
            raise RuntimeError("Null-text inversion requires PyTorch DiT (acestep.use_mlx_dit=false).")
        if nti_enabled and getattr(handler, "config", None) is not None and bool(getattr(handler.config, "is_turbo", False)):
            raise RuntimeError("Turbo checkpoints do not use CFG; NTI is not applicable (use a non-turbo config).")

        if str(cfg.infer_method) != "ode":
            raise ValueError(f"invert: only infer_method='ode' is supported, got {cfg.infer_method!r}")
        sampler_mode = str(cfg.sampler_mode)
        if sampler_mode != "euler":
            raise ValueError(f"invert: only sampler_mode='euler' is supported for now, got {sampler_mode!r}")

        pivot_integrator = str(OmegaConf.select(cfg, "pivot_integrator", default="euler"))
        if pivot_integrator not in ("euler", "heun", "uni_inv"):
            raise ValueError(
                f"invert: pivot_integrator must be 'euler', 'heun', or 'uni_inv', got {pivot_integrator!r}"
            )

        nti_latent_integrator = str(OmegaConf.select(cfg, "nti.latent_integrator", default="euler"))
        if nti_enabled and nti_latent_integrator not in ("euler", "heun"):
            raise ValueError(
                f"invert: nti latent integrator must be 'euler' or 'heun', got {nti_latent_integrator!r}"
            )

        infer_steps = int(cfg.inference_steps)
        guidance = float(cfg.guidance_scale)
        if nti_enabled and guidance <= 1.0:
            raise ValueError("invert: with nti.enabled=true, guidance_scale must be > 1.0 for NTI")

        wav = handler.process_target_audio(music_path)
        if wav is None:
            raise RuntimeError(f"Failed to load or process audio: {music_path}")

        seed = int(cfg.seed) if not bool(cfg.use_random_seed) else 0
        payload, clean_latents = build_invert_payload(
            handler,
            captions=str(cfg.prompt.captions),
            lyrics=str(cfg.prompt.lyrics),
            vocal_language=str(cfg.vocal_language),
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
                shift=float(cfg.shift),
                pivot_integrator=cast(PivotIntegrator, pivot_integrator),
                use_progress_bar=not bool(getattr(handler, "disable_tqdm", False)),
            )

        own_writer = writer is None
        track = setup_writer(cfg) if own_writer else writer
        debug_mode = bool(OmegaConf.select(cfg, "debug_mode", default=False))

        if bool(OmegaConf.select(cfg, "log_pivot_trajectory_to_comet", default=True)):
            max_edge = int(OmegaConf.select(cfg, "log_pivot_trajectory_max_edge", default=4096))
            log_pivot_trajectory_comet(track, trajectory, max_edge=max_edge)

        null_list: list[torch.Tensor] | None = None
        try:
            if nti_enabled:
                nti_lr = _nti_float(cfg, "learning_rate", 1.0e-2)
                nti_steps = _nti_int(cfg, "num_inner_steps", 15)
                nti_eps = _nti_float(cfg, "epsilon", 1.0e-7)

                nti = NullTextInversionAceStep(
                    lr=nti_lr,
                    num_inner_steps=nti_steps,
                    epsilon=nti_eps,
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
                        shift=float(cfg.shift),
                        diffusion_guidance_scale=guidance,
                        cfg_interval_start=float(cfg.cfg_interval_start),
                        cfg_interval_end=float(cfg.cfg_interval_end),
                        use_adg=bool(cfg.use_adg),
                        use_progress_bar=not bool(getattr(handler, "disable_tqdm", False)),
                    )
        finally:
            if own_writer:
                track.end()

        noise = trajectory[0].detach().cpu()
        null_cpu = [t.detach().cpu() for t in null_list] if null_list is not None else None

        container = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(container, dict) and "hydra" in container:
            container = {k: v for k, v in container.items() if k != "hydra"}
        save_cfg = OmegaConf.create(container)

        return InversionArtifact(
            noise=noise,
            null_encoder_hidden_states_per_step=null_cpu,
            cfg_snapshot=save_cfg,
        )
