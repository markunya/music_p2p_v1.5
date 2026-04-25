"""Tensor helpers for attention key layout (ACE-Step eager: ``(B, H, Q, K)`` or ``(H, Q, K)``)."""

from __future__ import annotations

import torch


def infer_attention_head_query_key(x: torch.Tensor) -> tuple[int, int, int, int]:
    if x.dim() == 4:
        b, h, q, k = x.shape
        return b, h, q, k
    if x.dim() == 3:
        h, q, k = x.shape
        return 1, h, q, k
    raise ValueError(f"expected attention tensor (B, H, Q, K) or (H, Q, K), got {tuple(x.shape)}")


def verify_key_dim_against_M(M: torch.Tensor, K: int) -> None:
    s, t = M.shape[0], M.shape[1]
    if s > K or t > K:
        raise ValueError(f"Mapper shape {(s, t)} is larger than key length K={K}")
