from __future__ import annotations

from typing import Any, Optional

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.inversion.artifact import InversionArtifact
from src.attention_injection.controllers.base import AttentionControllerBase, DummyAttentionController
from src.attention_injection.eager_hook import (
    clear_runtime_controller,
    install_eager_attention_control_patch,
    rebind_eager_hook_for_decoder_submodule,
    set_runtime_controller,
)
from src.steppers.base import BaseStepper
from src.steppers.guidance import GuidanceStepper
from src.utils.conditioning import ModelCondition


class _DecoderWithRuntimeController(torch.nn.Module):
    """``model.decoder`` must stay an ``nn.Module``; we only wrap ``forward`` to set the attention controller."""

    def __init__(self, inner: torch.nn.Module, controller: AttentionControllerBase) -> None:
        super().__init__()
        self.inner = inner
        self._controller = controller

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        set_runtime_controller(self._controller)
        try:
            return self.inner(*args, **kwargs)
        finally:
            clear_runtime_controller()


class ForwardPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        *,
        attention_controller: Optional[AttentionControllerBase] = None,
    ) -> None:
        self._cfg = cfg
        self._infer_steps = int(cfg.inference_steps)
        self._stepper: BaseStepper = instantiate(cfg.stepper)
        install_eager_attention_control_patch()
        self._controller: AttentionControllerBase = (
            attention_controller
            if attention_controller is not None
            else instantiate(cfg.controller)  # type: ignore[assignment]
        )
        self._decoder_wrap_done: bool = False
        self._decoder_orig: torch.nn.Module | None = None

    def _ensure_decoder_wrapped(self, model: torch.nn.Module) -> None:
        if self._decoder_wrap_done and self._decoder_orig is not None:
            return
        self._decoder_orig = model.decoder
        model.decoder = _DecoderWithRuntimeController(self._decoder_orig, self._controller)
        self._decoder_wrap_done = True

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
        self._ensure_decoder_wrapped(model)
        dit = model.decoder.inner if hasattr(model.decoder, "inner") else model.decoder
        rebind_eager_hook_for_decoder_submodule(dit)

        prev_attn_impl: str | int | None = None
        need_injection = not isinstance(self._controller, DummyAttentionController)
        if need_injection and hasattr(dit, "config") and hasattr(dit.config, "_attn_implementation"):
            prev_attn_impl = dit.config._attn_implementation
            if str(prev_attn_impl) != "eager":
                dit.config._attn_implementation = "eager"
                logger.info(
                    "Attention injection: decoder _attn_implementation {!r} -> 'eager' "
                    "(SDPA/flash do not go through the hooked eager path)",
                    prev_attn_impl,
                )
        npe = inversion_artifact.null_embeddings_per_step
        if npe is not None and len(npe) != self._infer_steps:
            logger.warning(
                "null_embeddings_per_step has length {} but inference_steps={}; "
                "indices beyond min will skip override",
                len(npe),
                self._infer_steps,
            )

        try:
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
        finally:
            if prev_attn_impl is not None and dit is not None and hasattr(dit, "config"):
                dit.config._attn_implementation = prev_attn_impl
                logger.info("Attention injection: restored decoder _attn_implementation to {!r}", prev_attn_impl)

        logger.info("Forward diffusion done, final x.shape={}", tuple(x.shape))
        return {"final_latents": x, "trajectory": traj}
