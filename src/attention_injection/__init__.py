"""Attention injection: mappers, DiT cross-attn controllers, eager hook."""

from src.attention_injection.bundle import (
    AttentionInjectionBundle,
    EncoderSegmentLayout,
    build_mappers,
)
from src.attention_injection.controllers import (
    AttentionControllerBase,
    DummyAttentionController,
    ReplacementAttentionController,
    ReweightAttentionController,
)
from src.attention_injection.eager_hook import clear_runtime_controller, install_eager_attention_control_patch, set_runtime_controller
from src.attention_injection.injection_utils import infer_attention_head_query_key, verify_key_dim_against_M
from src.attention_injection.mappers import (
    CaptionsReplacementMapper,
    InjectionMapper,
    LyricReplacementMapper,
)
from src.attention_injection.tokenize import (
    TokenizedText,
    format_lyrics_for_dit,
    tokenize_caption_raw,
    tokenize_lyrics_for_mapper,
    tokenize_with_handler_tokenizer,
)
from src.attention_injection.reweight_utils import (
    REWEIGHT_DOWN,
    REWEIGHT_UP,
    ReweightTarget,
    build_2d_equalizer_for_p2p,
    build_p2p_key_boost,
    parse_reweight_from_tgt,
    strip_reweight_marks,
)
from src.attention_injection.validate import assert_row_stochastic, is_row_stochastic, pad_square_identity

__all__ = [
    "InjectionMapper",
    "LyricReplacementMapper",
    "CaptionsReplacementMapper",
    "AttentionInjectionBundle",
    "EncoderSegmentLayout",
    "build_mappers",
    "TokenizedText",
    "format_lyrics_for_dit",
    "tokenize_caption_raw",
    "tokenize_lyrics_for_mapper",
    "tokenize_with_handler_tokenizer",
    "assert_row_stochastic",
    "is_row_stochastic",
    "pad_square_identity",
    "AttentionControllerBase",
    "DummyAttentionController",
    "ReplacementAttentionController",
    "ReweightAttentionController",
    "install_eager_attention_control_patch",
    "set_runtime_controller",
    "clear_runtime_controller",
    "infer_attention_head_query_key",
    "verify_key_dim_against_M",
    "REWEIGHT_UP",
    "REWEIGHT_DOWN",
    "ReweightTarget",
    "parse_reweight_from_tgt",
    "strip_reweight_marks",
    "build_p2p_key_boost",
    "build_2d_equalizer_for_p2p",
]
