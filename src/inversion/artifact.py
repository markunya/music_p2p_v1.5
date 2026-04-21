from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ARTIFACT_VERSION = 1


@dataclass
class InversionArtifact:
    """Inverted noise state and minimal metadata for ``generate.py`` warm-start."""

    noise: torch.Tensor
    forward_start_step_index: int = 0
    inference_steps: int = 0
    stepper_class_name: str = ""

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "version": ARTIFACT_VERSION,
            "noise": self.noise,
            "forward_start_step_index": int(self.forward_start_step_index),
            "inference_steps": int(self.inference_steps),
            "stepper_class_name": str(self.stepper_class_name),
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_state_dict(), str(p))

    @staticmethod
    def from_state_dict(data: dict[str, Any]) -> "InversionArtifact":
        ver = int(data.get("version", 0))
        if ver != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported artifact version {ver}, expected {ARTIFACT_VERSION}")
        noise = data["noise"]
        if not torch.is_tensor(noise):
            raise TypeError("artifact missing tensor 'noise'")
        return InversionArtifact(
            noise=noise,
            forward_start_step_index=int(data.get("forward_start_step_index", 0)),
            inference_steps=int(data.get("inference_steps", 0)),
            stepper_class_name=str(data.get("stepper_class_name", "")),
        )

    @classmethod
    def load(cls, path: Path | str) -> "InversionArtifact":
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(str(path), map_location="cpu")
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in artifact file, got {type(data)}")
        return cls.from_state_dict(data)
