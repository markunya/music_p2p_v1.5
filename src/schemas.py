from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class PromptConfig:
    captions: str = MISSING
    lyrics: str = MISSING
    vocal_language: str = "en"
