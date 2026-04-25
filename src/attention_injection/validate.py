"""Row-stochastic checks for injection matrices ``M`` with shape ``(len_tgt, len_src)``."""

from __future__ import annotations

import torch


def assert_row_stochastic(M: torch.Tensor, *, atol: float = 1e-5) -> None:
    """Each row sums to 1, entries non-negative."""
    if M.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(M.shape)}")
    if (M < -atol).any():
        raise ValueError("Matrix contains negative entries")
    sums = M.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0.0):
        bad = (sums - 1.0).abs().max().item()
        raise ValueError(f"Rows do not sum to 1 (max deviation {bad})")


def is_row_stochastic(M: torch.Tensor, *, atol: float = 1e-5) -> bool:
    try:
        assert_row_stochastic(M, atol=atol)
    except ValueError:
        return False
    return True


def pad_square_identity(M: torch.Tensor, max_len: int, *, device=None, dtype=None) -> torch.Tensor:
    """Pad ``(m, n)`` to ``(max_len, max_len)`` with identity tail for empty rows."""
    if max_len < M.shape[0] or max_len < M.shape[1]:
        raise ValueError(f"max_len {max_len} smaller than M {tuple(M.shape)}")
    dev = device or M.device
    dt = dtype or M.dtype
    out = torch.zeros(max_len, max_len, device=dev, dtype=dt)
    m, n = M.shape
    out[:m, :n] = M.to(device=dev, dtype=dt)
    eps = 1e-12
    for i in range(max_len):
        if float(out[i].sum()) <= eps:
            out[i, i] = 1.0
    return out
