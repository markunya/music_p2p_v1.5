from dataclasses import dataclass
from typing import Any, Protocol

import torch


class TextTokenizerLike(Protocol):

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
    return f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"


@dataclass(frozen=True)
class TokenizedText:

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
    formatted = format_lyrics_for_dit(raw_lyrics, vocal_language)
    return tokenize_with_handler_tokenizer(tokenizer, formatted, max_length=max_length)


def tokenize_caption_raw(
    caption: str,
    tokenizer: TextTokenizerLike,
    *,
    max_length: int = 256,
) -> TokenizedText:
    return tokenize_with_handler_tokenizer(tokenizer, caption.strip(), max_length=max_length)
