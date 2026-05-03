from typing import Any

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.inversion.artifact import InversionArtifact
from src.logging.trajectory_logging import log_latent_trajectory, trajectory_image_flags
from src.logging.writer import BaseWriter, DummyWriter
from src.steppers.base import BaseStepper
from src.steppers.guidance import CFG_GUIDANCE_STEPPERS
from src.utils.conditioning import ModelCondition


class InversionPipeline:

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        stepper_cfg = OmegaConf.select(cfg, "invert_stepper", default=None)
        if stepper_cfg is None:
            stepper_cfg = cfg.stepper
        self._stepper: BaseStepper = instantiate(stepper_cfg)
        self._infer_steps = int(cfg.inference_steps)
        self._alpha: float = float(OmegaConf.select(cfg, "alpha", default=1.0))
        if not (0.0 <= self._alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self._alpha}")
        if isinstance(self._stepper, CFG_GUIDANCE_STEPPERS):
            logger.warning(
                "InversionPipeline: CFG guidance stepper is not recommended for inversion in v1 (2B CFG + KV); "
                "prefer euler/heun/uni_*."
            )

    def _inversion_steps_count(self) -> int:
        k = int(round(self._alpha * self._infer_steps))
        return max(0, min(self._infer_steps, k))

    def _build_inversion_trajectory(
        self,
        model: torch.nn.Module,
        *,
        clean_latents: torch.Tensor,
        model_condition: ModelCondition,
    ) -> list[torch.Tensor]:
        device = clean_latents.device
        dtype = clean_latents.dtype
        x = clean_latents.detach().clone().to(device=device, dtype=dtype)

        n = self._infer_steps
        k = self._inversion_steps_count()
        t = torch.linspace(1.0, 0.0, n + 1, device=device, dtype=dtype)
        traj: list[torch.Tensor] = [x.detach().clone()]
        desc = f"Inversion ({type(self._stepper).__name__})"
        for i in tqdm(range(n - 1, n - 1 - k, -1), total=k, desc=desc):
            t_curr = t[i + 1]
            t_next = t[i]
            payload = self._stepper.step(
                model=model,
                x=x,
                t_curr=t_curr,
                t_next=t_next,
                model_condition=model_condition,
            )
            x = payload.x
            traj.append(x.detach().clone())
        return traj

    def run(
        self,
        model: Any,
        *,
        clean_latents: torch.Tensor,
        model_condition: ModelCondition,
        writer: BaseWriter | None = None,
    ) -> tuple[InversionArtifact, list[torch.Tensor]]:
        model.eval()
        nti_on = bool(OmegaConf.select(self._cfg, "nti.enabled", default=False))
        k = self._inversion_steps_count()
        forward_start = self._infer_steps - k
        logger.info(
            "InversionPipeline: alpha={} -> K={}/{} steps, forward_start_step_index={}",
            self._alpha,
            k,
            self._infer_steps,
            forward_start,
        )
        if k == 0:
            logger.warning("InversionPipeline: alpha=0 — no inversion steps; artifact stores clean latents")

        model_condition_for_nti = model_condition.clone() if nti_on else None

        with torch.no_grad():
            trajectory = self._build_inversion_trajectory(
                model,
                clean_latents=clean_latents,
                model_condition=model_condition,
            )
        noise = trajectory[-1].detach().cpu()
        stepper_name = type(self._stepper).__name__
        null_per_step: list[torch.Tensor] | None = None

        log_w = writer if writer is not None else DummyWriter()

        if nti_on:
            from src.inversion.nti import NullTextOptimization, validate_nti_prerequisites

            assert model_condition_for_nti is not None
            if k == 0:
                logger.warning("NTI enabled but alpha=0 (no inversion steps); skipping NTI")
                null_per_step = None
            else:
                fwd_stepper = instantiate(self._cfg.stepper)
                validate_nti_prerequisites(model, fwd_stepper)
                nti = NullTextOptimization(self._cfg, writer=log_w)
                null_per_step = nti.run(
                    model=model,
                    trajectory=trajectory,
                    model_condition=model_condition_for_nti,
                    guidance_stepper=fwd_stepper,
                    infer_steps=self._infer_steps,
                    forward_start_step_index=forward_start,
                )

        artifact = InversionArtifact(
            noise=noise,
            forward_start_step_index=forward_start,
            inference_steps=self._infer_steps,
            stepper_class_name=stepper_name,
            null_embeddings_per_step=null_per_step,
        )

        traj_prefix = f"{OmegaConf.select(self._cfg, 'comet_run_prefix', default='inv')}/inversion_latent"
        li, me = trajectory_image_flags(self._cfg)
        log_latent_trajectory(log_w, trajectory, prefix=traj_prefix, log_images=li, max_edge=me)

        return artifact, trajectory
