"""Structured config nodes for Hydra (optional validation via ConfigStore)."""

from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING


@dataclass
class AceStepInitConfig:
    project_root: str = MISSING
    config_path: str = "acestep-v15-turbo"
    use_mlx_dit: bool = False
    device: str = "auto"
    offload_to_cpu: bool = False


@dataclass
class PromptConfig:
    captions: str = MISSING
    lyrics: str = MISSING


@dataclass
class GenerateConfig:
    """Top-level generate config; composed with acestep + prompt groups."""

    defaults: List[Any] = field(default_factory=list)

    acestep: AceStepInitConfig = MISSING
    prompt: PromptConfig = MISSING

    exp_name: str = "gen_run"
    save_dir: str = "_exps/gen"
    seed: int = 1
    debug_mode: bool = False

    duration: float = -1.0
    vocal_language: str = "en"
    inference_steps: int = 8
    guidance_scale: float = 7.0
    use_random_seed: bool = False
    batch_size: int = 1
    task_type: str = "text2music"
    shift: float = 1.0
    infer_method: str = "ode"
    sampler_mode: str = "euler"
    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
