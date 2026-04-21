from __future__ import annotations

from typing import Any

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.steppers.base import BaseStepper
from src.utils.conditioning import ModelCondition


class ForwardPipeline:
    """ODE-style forward diffusion using a Hydra-configured stepper.

    Plain **Euler** / **Heun** use a single ``decoder`` call per velocity eval. For classifier-free
    (or APG/ADG) guidance, set ``cfg.stepper`` to ``stepper/guidance_euler`` or ``stepper/guidance_heun``
    (see ``src.steppers.guidance.GuidanceStepper``).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._infer_steps = int(cfg.inference_steps)
        self._stepper: BaseStepper = instantiate(cfg.stepper)

    def run(
        self,
        model: torch.nn.Module,
        *,
        initial_latents: torch.Tensor,
        model_condition: ModelCondition,
    ) -> dict[str, Any]:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        x = initial_latents.to(device=device, dtype=dtype)
        model_condition.encoder_hidden_states = model_condition.encoder_hidden_states.to(
            device=device, dtype=dtype
        )
        model_condition.encoder_attention_mask = model_condition.encoder_attention_mask.to(
            device=device, dtype=dtype
        )
        model_condition.context_latents = model_condition.context_latents.to(device=device, dtype=dtype)
        model_condition.attention_mask = model_condition.attention_mask.to(device=device, dtype=dtype)

        model_condition.past_key_values = None

        t = torch.linspace(1.0, 0.0, self._infer_steps + 1, device=device, dtype=dtype)
        traj: list[torch.Tensor] = [x.detach().clone()]
        indices = range(self._infer_steps)
        step_name = type(self._stepper).__name__
        for i in tqdm(indices, total=self._infer_steps, desc=f"Forward ({step_name})"):
            t_curr, t_next = t[i], t[i + 1]
            payload = self._stepper.step(
                model=model,
                x=x,
                t_curr=t_curr,
                t_next=t_next,
                model_condition=model_condition,
            )
            x = payload.x
            traj.append(x.detach().clone())

        logger.info("Forward diffusion done, final x.shape={}", tuple(x.shape))
        return {"final_latents": x, "trajectory": traj}
