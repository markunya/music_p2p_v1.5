"""Bundle lyric + caption replacement mappers; encoder key slices when wiring cross-attention."""

from __future__ import annotations

from dataclasses import dataclass

from src.attention_injection.mappers.captions import CaptionsReplacementMapper
from src.attention_injection.mappers.lyrics import LyricReplacementMapper
from src.attention_injection.tokenize import TextTokenizerLike
from src.schemas import PromptConfig


@dataclass
class EncoderSegmentLayout:
    """Offsets of ``lyric || timbre || text`` in ``encoder_hidden_states`` (fill when integrating)."""

    lyric_slice: slice | None = None
    timbre_slice: slice | None = None
    text_slice: slice | None = None


@dataclass(frozen=True)
class AttentionInjectionBundle:
    """Paired :class:`LyricReplacementMapper` and :class:`CaptionsReplacementMapper` for src→tgt edit."""

    lyrics: LyricReplacementMapper
    captions: CaptionsReplacementMapper
    layout: EncoderSegmentLayout | None = None


def build_mappers(
    src: PromptConfig,
    tgt: PromptConfig,
    tokenizer: TextTokenizerLike,
    *,
    device=None,
    dtype=None,
    lyric_max_length: int = 2048,
    caption_max_length: int = 256,
) -> AttentionInjectionBundle:
    """Build lyric + caption replacement mappers from two :class:`PromptConfig` values."""
    import torch as _torch

    dev = device or _torch.device("cpu")
    dt = dtype or _torch.float32
    lyrics = LyricReplacementMapper(
        src.lyrics,
        tgt.lyrics,
        src.vocal_language,
        tokenizer,
        device=dev,
        dtype=dt,
        max_length=lyric_max_length,
    )
    captions = CaptionsReplacementMapper(
        src.captions,
        tgt.captions,
        tokenizer,
        device=dev,
        dtype=dt,
        max_length=caption_max_length,
    )
    return AttentionInjectionBundle(lyrics=lyrics, captions=captions, layout=None)
