"""Rescale cross-attention weights along the key axis (no renormalization; PTP null-text reweight style)."""

from __future__ import annotations

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.attention_injection.controllers.base import AttentionControllerBase
from src.attention_injection.injection_utils import infer_attention_head_query_key
from src.utils.conditioning import (
    p2p_src_tgt_prompt_configs,
    prepare_conditions,
)
from src.attention_injection.reweight_utils import (
    build_2d_equalizer_for_p2p,
    build_p2p_key_boost,
    parse_reweight_from_tgt,
)


class ReweightAttentionController(AttentionControllerBase):
    def __init__(
        self,
        *,
        reweight_strength: float = 1.0,
    ) -> None:
        self.reweight_strength = reweight_strength
        self._equalizer: torch.Tensor | None = None
        self._n_forward: int = 0

    def build(self, *, handler, cfg: DictConfig) -> None:
        src_raw, tgt_raw = p2p_src_tgt_prompt_configs(cfg.p2p_task)
        clean_tgt, reweight_targets = parse_reweight_from_tgt(tgt_raw)
        prompts = [src_raw, clean_tgt]
        duration = float(OmegaConf.select(cfg, "duration", default=-1.0))
        cond_fwd, prep_unpack = prepare_conditions(
            handler,
            prompts,
            duration,
            return_unpack=True,
        )

        k_dim = int(cond_fwd.encoder_hidden_states.shape[1])
        strength = float(OmegaConf.select(cfg, "controller.reweight_strength", default=self.reweight_strength))
        metas = [{"duration": float(duration)} for _ in range(len(prompts))]
        if reweight_targets:
            mask1 = build_p2p_key_boost(
                handler,
                metas=metas,
                captions=[p.captions for p in prompts],
                lyrics=[p.lyrics for p in prompts],
                vocal_languages=[p.vocal_language for p in prompts],
                model_condition=cond_fwd,
                unpack=prep_unpack,
                clean_tgt=clean_tgt,
                targets=reweight_targets,
                batch_tgt_index=1,
            )
        else:
            mask1 = [0] * k_dim
            logger.warning("ReweightAttentionController.build: no reweight marks in tgt — using identity equalizer.")

        eq2d = build_2d_equalizer_for_p2p(
            strength,
            mask1,
            batch_size=len(prompts),
            k=k_dim,
            tgt_batch_index=1,
        )
        self._equalizer = torch.tensor(eq2d, dtype=torch.float32)
        self._n_forward = 0

        n_boosted = int(sum(int(x) for x in mask1))
        logger.info(
            "ReweightAttentionController.build: strength={} K={} boosted_keys={}/{}",
            strength,
            k_dim,
            n_boosted,
            k_dim,
        )
        if n_boosted > 0 and abs(strength - 1.0) < 1e-5:
            logger.warning(
                "ReweightAttentionController.build: reweight_strength=1.0 on masked keys means no change."
            )

    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        w = attn_weight
        b, _, _, k = infer_attention_head_query_key(w)
        if self._equalizer is None:
            return w
        if self._equalizer.shape != (b, k):
            raise ValueError(f"equalizer shape {(b, k)} expected, got {tuple(self._equalizer.shape)}")

        m = self._equalizer.to(device=w.device, dtype=w.dtype)
        if w.dim() == 4:
            m = m.view(b, 1, 1, k)
        else:
            m = m.view(b, 1, k)

        out = w * m
        self._n_forward += 1
        if self._n_forward == 1:
            w_abs_mean = w.detach().abs().mean().clamp_min(1e-12)
            delta = (out - w).detach().abs().mean()
            k_mass = out.detach().sum(dim=-1).float().mean().item()
            rel = (delta / w_abs_mean).item()
            logger.info(
                "ReweightAttentionController: first cross-attn apply | w.shape={} "
                "mean|Δattn|/mean|w|={:.4g} post_mul mean(sum_K)={:.4g} (softmax was ~1.0 per row)",
                tuple(w.shape),
                rel,
                k_mass,
            )
            if rel < 1e-6:
                logger.warning(
                    "ReweightAttentionController: first forward change is ~0 — equalizer is all-ones on this head "
                    "or K mismatch; check key_boost / tgt row."
                )
        return out

