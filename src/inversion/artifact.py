from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ARTIFACT_VERSION = 2


@dataclass
class InversionArtifact:

    noise: torch.Tensor
    forward_start_step_index: int = 0
    inference_steps: int = 0
    stepper_class_name: str = ""
    null_embeddings_per_step: list[torch.Tensor] | None = None

    @classmethod
    def from_noise(cls, noise: torch.Tensor) -> "InversionArtifact":
        return cls(noise=noise.detach(), null_embeddings_per_step=None)

    def to_state_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": ARTIFACT_VERSION,
            "noise": self.noise,
            "forward_start_step_index": int(self.forward_start_step_index),
            "inference_steps": int(self.inference_steps),
            "stepper_class_name": str(self.stepper_class_name),
        }
        if self.null_embeddings_per_step is not None:
            d["null_embeddings_per_step"] = [t.detach().cpu() for t in self.null_embeddings_per_step]
        else:
            d["null_embeddings_per_step"] = None
        return d

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_state_dict(), str(p))

    @staticmethod
    def from_state_dict(data: dict[str, Any]) -> "InversionArtifact":
        ver = int(data.get("version", 1))
        if ver not in (1, 2):
            raise ValueError(f"Unsupported artifact version {ver}, expected 1 or 2")
        noise = data["noise"]
        if not torch.is_tensor(noise):
            raise TypeError("artifact missing tensor 'noise'")
        null_list: list[torch.Tensor] | None = None
        if ver == 2:
            raw = data.get("null_embeddings_per_step")
            if raw is not None:
                if not isinstance(raw, (list, tuple)):
                    raise TypeError("null_embeddings_per_step must be a list of tensors")
                null_list = []
                for i, t in enumerate(raw):
                    if not torch.is_tensor(t):
                        raise TypeError(f"null_embeddings_per_step[{i}] is not a tensor")
                    null_list.append(t)
        return InversionArtifact(
            noise=noise,
            forward_start_step_index=int(data.get("forward_start_step_index", 0)),
            inference_steps=int(data.get("inference_steps", 0)),
            stepper_class_name=str(data.get("stepper_class_name", "")),
            null_embeddings_per_step=null_list,
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
