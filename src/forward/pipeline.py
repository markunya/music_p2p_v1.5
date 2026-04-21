"""Forward pass: plain ODE vs velocity edit (общий ``diffusion_driver`` внутри)."""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.artifact_bundle import GenerationArtifactPayload
from src.forward.generate_kwargs import generate_music_kwargs_from_cfg
from src.forward.plain_forward import run_plain_forward
from src.forward.steppers.velocity_fusion import VelocityFusionEditRunner


class ForwardPipeline:
    """Форвард по Hydra-конфигу: ``run(handler, artifact)``."""

    __slots__ = ("_cfg",)

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def run(
        self,
        handler: Any,
        artifact: GenerationArtifactPayload,
        *,
        velocity_fusion_runner: VelocityFusionEditRunner | None = None,
    ) -> dict[str, Any]:
        cfg = self._cfg
        mode = str(OmegaConf.select(cfg, "forward.mode", default="plain"))
        if mode == "velocity_fusion":
            runner = velocity_fusion_runner or instantiate(cfg.p2p_strategy)
            if not isinstance(runner, VelocityFusionEditRunner):
                raise TypeError(
                    f"forward.mode=velocity_fusion requires VelocityFusionEditRunner, got {type(runner).__name__}"
                )
            return runner.run(handler, cfg, artifact)
        return run_plain_forward(handler, cfg, artifact)


__all__ = ["ForwardPipeline", "generate_music_kwargs_from_cfg"]
