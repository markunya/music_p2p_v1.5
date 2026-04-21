"""Пара ``src`` / ``tgt`` промптов для P2P-edit из узла Hydra ``p2p_task``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class P2PPromptPair:
    """Исходное условие (референс / инверсия) и целевое (желаемое редактирование)."""

    src_captions: str
    src_lyrics: str
    tgt_captions: str
    tgt_lyrics: str
    vocal_language: str = "en"

    @classmethod
    def from_cfg_node(cls, p2p_task: Mapping[str, Any] | DictConfig) -> P2PPromptPair:
        if isinstance(p2p_task, DictConfig):
            p2p_task = cast(dict[str, Any], OmegaConf.to_container(p2p_task, resolve=True))
        src = p2p_task["src"]
        tgt = p2p_task["tgt"]
        return cls(
            src_captions=str(src["captions"]),
            src_lyrics=str(src["lyrics"]),
            tgt_captions=str(tgt["captions"]),
            tgt_lyrics=str(tgt["lyrics"]),
            vocal_language=str(p2p_task.get("vocal_language", "en")),
        )
