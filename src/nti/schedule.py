from __future__ import annotations

import torch


def acestep_sigma_grid(infer_steps: int, shift: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Same ``t`` schedule as ``AceStepConditionGenerationModel.generate_audio`` (length ``infer_steps + 1``)."""
    t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t
