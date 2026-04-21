"""Artifact / cfg → audio (explicit ODE loops + steppers)."""

from src.forward.artifact_resolve import forward_artifact_and_work_cfg
from src.forward.generate_kwargs import generate_music_kwargs_from_cfg
from src.forward.pipeline import ForwardPipeline

__all__ = ["ForwardPipeline", "forward_artifact_and_work_cfg", "generate_music_kwargs_from_cfg"]
