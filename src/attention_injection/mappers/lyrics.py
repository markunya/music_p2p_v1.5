"""Lyrics replacement mapper: ``SequenceMatcher`` on lyric token ids → row-stochastic ``(tgt, src)``."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List

import torch

from src.attention_injection.mappers.base import InjectionMapper
from src.attention_injection.tokenize import TextTokenizerLike, tokenize_lyrics_for_mapper


def _build_src_tgt_stochastic_matrix(
    idx_src: List[int],
    idx_tgt: List[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``A`` with shape ``(len_src, len_tgt)``, each **src** row sums to 1 (replacement-style)."""
    m, n = len(idx_src), len(idx_tgt)
    A = torch.zeros(m, n, device=device, dtype=dtype)
    sm = SequenceMatcher(None, idx_src, idx_tgt)
    for opcode, i1, i2, j1, j2 in sm.get_opcodes():
        if opcode == "equal":
            for k in range(i2 - i1):
                A[i1 + k, j1 + k] = 1.0
        elif opcode == "insert":
            tgt_inds = list(range(j1, j2))
            if tgt_inds:
                for j in tgt_inds:
                    A[:, j] = 1.0 / float(max(m, 1))
        elif opcode == "delete":
            src_inds = list(range(i1, i2))
            if src_inds:
                for i in src_inds:
                    A[i, :] = 1.0 / float(max(n, 1))
        elif opcode == "replace":
            src_inds = list(range(i1, i2))
            tgt_inds = list(range(j1, j2))
            if not src_inds or not tgt_inds:
                continue
            weight = 1.0 / float(len(tgt_inds))
            for i in src_inds:
                for j in tgt_inds:
                    A[i, j] = weight
        else:
            raise RuntimeError(f"unknown opcode {opcode}")
    row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    A = A / row_sums
    # ``(tgt, src)``: row ``j`` proportional to column ``j`` of ``A`` (mass from src keys to tgt ``j``).
    M_tgt_src = A.transpose(0, 1).contiguous()
    for j in range(n):
        csum = M_tgt_src[j].sum()
        if float(csum) < 1e-12:
            M_tgt_src[j] = 1.0 / float(m)
        else:
            M_tgt_src[j] /= csum
    return M_tgt_src


class LyricReplacementMapper(InjectionMapper):
    """Replacement alignment of two lyric strings in the same token space as ACE-Step ``lyric_token_idss``."""

    def __init__(
        self,
        src_lyrics: str,
        tgt_lyrics: str,
        vocal_language: str,
        tokenizer: TextTokenizerLike,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        max_length: int = 2048,
    ) -> None:
        self._src_lyrics = src_lyrics
        self._tgt_lyrics = tgt_lyrics
        self._vocal_language = vocal_language
        self._tokenizer = tokenizer
        self._max_length = max_length
        super().__init__(device=device, dtype=dtype)

    def _build_matrix(self) -> torch.Tensor:
        src_t = tokenize_lyrics_for_mapper(
            self._src_lyrics, self._vocal_language, self._tokenizer, max_length=self._max_length
        )
        tgt_t = tokenize_lyrics_for_mapper(
            self._tgt_lyrics, self._vocal_language, self._tokenizer, max_length=self._max_length
        )
        idx_src = src_t.active_ids()
        idx_tgt = tgt_t.active_ids()
        if not idx_src or not idx_tgt:
            raise ValueError("empty lyric token sequence after tokenization")
        return _build_src_tgt_stochastic_matrix(
            idx_src,
            idx_tgt,
            device=self._device,
            dtype=self._dtype,
        )
