from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING


@dataclass
class AceStepInitConfig:
    project_root: str = MISSING
    config_path: str = "acestep-v15-base"
    use_mlx_dit: bool = False
    device: str = "auto"
    offload_to_cpu: bool = False


@dataclass
class PromptConfig:
    captions: str = MISSING
    lyrics: str = MISSING
    vocal_language: str = "en"


@dataclass
class ArtifactRefConfig:
    path: str | None = None


@dataclass
class GenerateConfig:
    defaults: List[Any] = field(default_factory=list)

    acestep: AceStepInitConfig = MISSING
    prompt: PromptConfig = MISSING
    stepper: Any = MISSING
    artifact: ArtifactRefConfig = field(default_factory=ArtifactRefConfig)

    exp_name: str = "gen_run"
    save_dir: str = "_exps/gen"
    seed: int = 1
    debug_mode: bool = False

    duration: float = -1.0
    inference_steps: int = 50
    use_random_seed: bool = False
    batch_size: int = 1
    audio_format: str = "wav"
