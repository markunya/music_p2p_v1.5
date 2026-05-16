from __future__ import annotations

import math
from typing import Any

import torch
from loguru import logger
from omegaconf import DictConfig

from src.attention_injection.controllers.base import AttentionControllerBase
from src.attention_injection.injection_utils import infer_attention_head_query_key


def alpha_ppae_fuser(
    t: float,
    t_s: float,
    t_e: float,
    eta_min: float,
    eta_max: float,
) -> float | None:
    if t < t_e or t > t_s:
        return None
    span = t_s - t_e
    if span <= 0.0:
        raise ValueError(f"ppae fuser: need t_s > t_e, got t_s={t_s}, t_e={t_e}")
    p = (t_s - t) / span
    p = min(1.0, max(0.0, float(p)))
    return float(eta_min + 0.5 * (eta_max - eta_min) * (1.0 - math.cos(math.pi * p)))


class PPAEReplacementController(AttentionControllerBase):
    def __init__(
        self,
        *,
        t_s: float = 0.8,
        t_e: float = 0.2,
        eta_min: float = 0.0,
        eta_max: float = 1.0,
    ) -> None:
        super().__init__()
        self._t_s = float(t_s)
        self._t_e = float(t_e)
        self._eta_min = float(eta_min)
        self._eta_max = float(eta_max)
        self._built = False
        self._logged_cfg_batch = False

    def build(self, *, handler: Any, cfg: DictConfig, writer: Any | None = None) -> None:
        _ = (handler, cfg, writer)
        if self._t_s <= self._t_e:
            raise ValueError(
                f"PPAEReplacementController: expected t_s > t_e (diffusion time in [0,1] "
                f"same as controller ctx['t']), got t_s={self._t_s}, t_e={self._t_e}"
            )
        for name, v in (
            ("eta_min", self._eta_min),
            ("eta_max", self._eta_max),
        ):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"PPAEReplacementController: {name} must be in [0, 1], got {v}")
        if self._eta_min > self._eta_max:
            raise ValueError(
                f"PPAEReplacementController: need eta_min <= eta_max, got "
                f"eta_min={self._eta_min}, eta_max={self._eta_max}"
            )
        self._built = True
        logger.info(
            "PPAEReplacementController: t_s={} t_e={} eta_min={} eta_max={}",
            self._t_s,
            self._t_e,
            self._eta_min,
            self._eta_max,
        )

    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        if not self._built:
            logger.warning("PPAEReplacementController.forward: build() was not called; pass-through")
            return attn_weight
        if attn_weight.dim() != 4:
            return attn_weight
        b, _, _, _ = infer_attention_head_query_key(attn_weight)
        if b not in (2, 4):
            if not self._logged_cfg_batch:
                self._logged_cfg_batch = True
                logger.warning(
                    "PPAEReplacementController: attention batch {} not in (2, 4); pass-through",
                    b,
                )
            return attn_weight

        raw_t = self.ctx.get("t")
        if raw_t is None:
            return attn_weight
        t = float(raw_t)

        alpha = alpha_ppae_fuser(t, self._t_s, self._t_e, self._eta_min, self._eta_max)
        out = attn_weight.clone()

        def _apply_pair(src_row: torch.Tensor, tgt_row: torch.Tensor) -> torch.Tensor:
            if alpha is None:
                return src_row.clone()
            a = torch.as_tensor(alpha, device=attn_weight.device, dtype=torch.float32)
            return (a * tgt_row.float() + (1.0 - a) * src_row.float()).to(dtype=out.dtype)

        if b == 2:
            out[1] = _apply_pair(out[0], out[1])
        else:
            out[1] = _apply_pair(out[0], out[1])
            out[3] = _apply_pair(out[2], out[3])
        return out
