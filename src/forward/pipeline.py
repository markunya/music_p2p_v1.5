from __future__ import annotations

from typing import Any

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.inversion.artifact import InversionArtifact
from src.steppers.base import BaseStepper
from src.steppers.guidance import GuidanceStepper
from src.utils.conditioning import ModelCondition


class ForwardPipeline:
    """ODE-style forward diffusion using a Hydra-configured stepper.

    Initial noise always comes from ``inversion_artifact.noise`` (load from disk, or
    :meth:`InversionArtifact.from_noise` for fresh ``prepare_noise``). Plain **Euler** / **Heun**
    use one ``decoder`` call per velocity eval; for CFG / APG / ADG use ``stepper/guidance_*``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._infer_steps = int(cfg.inference_steps)
        self._stepper: BaseStepper = instantiate(cfg.stepper)

    def run(
        self,
        model: torch.nn.Module,
        *,
        model_condition: ModelCondition,
        inversion_artifact: InversionArtifact,
    ) -> dict[str, Any]:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x = inversion_artifact.noise.to(device=device, dtype=dtype)
        bsz_mc = model_condition.encoder_hidden_states.shape[0]
        if x.shape[0] == 1 and bsz_mc > 1:
            x = x.expand(bsz_mc, *x.shape[1:])
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
        npe = inversion_artifact.null_embeddings_per_step
        if npe is not None and len(npe) != self._infer_steps:
            logger.warning(
                "null_embeddings_per_step has length {} but inference_steps={}; "
                "indices beyond min will skip override",
                len(npe),
                self._infer_steps,
            )

        for i in tqdm(indices, total=self._infer_steps, desc=f"Forward ({step_name})"):
            if isinstance(self._stepper, GuidanceStepper):
                if npe is not None and i < len(npe):
                    ne = npe[i].to(device=device, dtype=dtype)
                    # Do not use ``encoder_hidden_states.shape[0]``: under CFG it is ``2 * latent_bsz``.
                    latent_bsz = int(x.shape[0])
                    if ne.shape[0] == 1 and latent_bsz > 1:
                        ne = ne.expand(latent_bsz, *ne.shape[1:])
                    self._stepper.set_null_encoder_override(ne)
                else:
                    self._stepper.set_null_encoder_override(None)

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

        if isinstance(self._stepper, GuidanceStepper):
            self._stepper.collapse_cfg_batch_layout(model_condition)

        logger.info("Forward diffusion done, final x.shape={}", tuple(x.shape))
        return {"final_latents": x, "trajectory": traj}
