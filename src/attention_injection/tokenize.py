"""Tokenization aligned with ACE-Step handler (``conditioning_text``): same ``text_tokenizer`` for caption + lyrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


class TextTokenizerLike(Protocol):
    """Minimal interface of ACE-Step ``handler.text_tokenizer``."""

    pad_token_id: int

    def __call__(
        self,
        text: str,
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> Any: ...


def format_lyrics_for_dit(lyrics: str, vocal_language: str) -> str:
    """Match ``PromptMixin._format_lyrics`` in ACE-Step."""
    return f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"


@dataclass(frozen=True)
class TokenizedText:
    """1D token ids (no batch dim) with optional mask for non-pad positions."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None

    def active_ids(self) -> list[int]:
        ids = self.input_ids.tolist()
        if self.attention_mask is not None:
            mask = self.attention_mask.tolist()
            return [ids[i] for i in range(len(ids)) if mask[i]]
        return ids


def tokenize_with_handler_tokenizer(
    tokenizer: TextTokenizerLike,
    text: str,
    *,
    max_length: int,
) -> TokenizedText:
    """Single sequence; mirrors ``conditioning_text`` tokenizer call."""
    out = tokenizer(
        text,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = out.input_ids[0]
    am = getattr(out, "attention_mask", None)
    attention_mask = am[0].bool() if am is not None else None
    return TokenizedText(input_ids=input_ids, attention_mask=attention_mask)


def tokenize_lyrics_for_mapper(
    raw_lyrics: str,
    vocal_language: str,
    tokenizer: TextTokenizerLike,
    *,
    max_length: int = 2048,
) -> TokenizedText:
    """Lyrics string → ids as in ACE-Step ``conditioning_text`` (2048 cap)."""
    formatted = format_lyrics_for_dit(raw_lyrics, vocal_language)
    return tokenize_with_handler_tokenizer(tokenizer, formatted, max_length=max_length)


def tokenize_caption_raw(
    caption: str,
    tokenizer: TextTokenizerLike,
    *,
    max_length: int = 256,
) -> TokenizedText:
    """Raw user caption (comma-separated tags, etc.) tokenized alone (not full SFT block)."""
    return tokenize_with_handler_tokenizer(tokenizer, caption.strip(), max_length=max_length)
