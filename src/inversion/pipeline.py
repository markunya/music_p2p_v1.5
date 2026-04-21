from __future__ import annotations

from typing import Any

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.inversion.artifact import InversionArtifact
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
        if isinstance(self._stepper, GuidanceStepper):
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
        model_condition.past_key_values = None

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
    ) -> InversionArtifact:
        model.eval()
        with torch.inference_mode():
            trajectory = self._build_inversion_trajectory(
                model,
                clean_latents=clean_latents,
                model_condition=model_condition,
            )
        noise = trajectory[-1].detach().cpu()
        stepper_name = type(self._stepper).__name__
        return InversionArtifact(
            noise=noise,
            forward_start_step_index=0,
            inference_steps=self._infer_steps,
            stepper_class_name=stepper_name,
        )
