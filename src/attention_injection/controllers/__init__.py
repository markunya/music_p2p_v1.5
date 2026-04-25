"""ACE-Step cross-attention controllers (post-softmax, hook in ``eager_hook``)."""

from __future__ import annotations

from src.attention_injection.controllers.base import AttentionControllerBase, DummyAttentionController
from src.attention_injection.controllers.replacement import ReplacementAttentionController
from src.attention_injection.controllers.reweight import ReweightAttentionController

__all__ = [
    "AttentionControllerBase",
    "DummyAttentionController",
    "ReplacementAttentionController",
    "ReweightAttentionController",
]
