from __future__ import annotations

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
from src.steppers.guidance import GuidanceStepper
from src.utils.conditioning import ModelCondition


class InversionPipeline:
    """Backward ODE; stepper from ``cfg.invert_stepper`` (Hydra: ``stepper@invert_stepper: …``), else ``cfg.stepper``."""

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        stepper_cfg = OmegaConf.select(cfg, "invert_stepper", default=None)
        if stepper_cfg is None:
            stepper_cfg = cfg.stepper
        self._stepper: BaseStepper = instantiate(stepper_cfg)
        self._infer_steps = int(cfg.inference_steps)
        if type(self._stepper) is GuidanceStepper:
            logger.warning(
                "InversionPipeline: GuidanceStepper is not recommended for inversion in v1 (2B CFG + KV); "
                "prefer euler/heun/uni_*."
            )

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

        t = torch.linspace(1.0, 0.0, self._infer_steps + 1, device=device, dtype=dtype)
        traj: list[torch.Tensor] = [x.detach().clone()]
        desc = f"Inversion ({type(self._stepper).__name__})"
        for i in tqdm(range(self._infer_steps - 1, -1, -1), total=self._infer_steps, desc=desc):
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
            fwd_stepper = instantiate(self._cfg.stepper)
            validate_nti_prerequisites(model, fwd_stepper)
            nti = NullTextOptimization(self._cfg, writer=log_w)
            null_per_step = nti.run(
                model=model,
                trajectory=trajectory,
                model_condition=model_condition_for_nti,
                guidance_stepper=fwd_stepper,
                infer_steps=self._infer_steps,
            )

        artifact = InversionArtifact(
            noise=noise,
            forward_start_step_index=0,
            inference_steps=self._infer_steps,
            stepper_class_name=stepper_name,
            null_embeddings_per_step=null_per_step,
        )

        traj_prefix = f"{OmegaConf.select(self._cfg, 'comet_run_prefix', default='inv')}/inversion_latent"
        li, me = trajectory_image_flags(self._cfg)
        log_latent_trajectory(log_w, trajectory, prefix=traj_prefix, log_images=li, max_edge=me)

        return artifact, trajectory
