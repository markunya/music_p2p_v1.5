"""In-memory result of ``InversionPipeline`` (disk I/O lives in ``run_invert``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from omegaconf import DictConfig


@dataclass
class InversionArtifact:
    """Pivot start noise and optional null-text embeddings per diffusion step."""

    noise: torch.Tensor
    null_encoder_hidden_states_per_step: Optional[List[torch.Tensor]]
    cfg_snapshot: DictConfig
