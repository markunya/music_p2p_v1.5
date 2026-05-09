import os
import random
from pathlib import Path
import math
import numpy as np
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_exp_dir(cfg: DictConfig) -> str:
    base = Path(get_original_cwd()) / str(cfg.save_dir)
    exp_dir = base / str(cfg.exp_name)
    if exp_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {exp_dir}. "
            "Pick a new exp_name (Hydra override exp_name=...) so runs do not overwrite each other."
        )
    exp_dir.mkdir(parents=True, exist_ok=False)
    cfg_path = exp_dir / "config.yaml"
    OmegaConf.save(config=cfg, f=str(cfg_path), resolve=True)
    logger.info("Experiment directory: {}", exp_dir)
    return str(exp_dir)


def resolve_against_original_cwd(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p.resolve())
    return str((Path(get_original_cwd()) / p).resolve())

def make_time_grid(n: int, device, dtype, ratio: float = 1.0) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=dtype)
    a = math.log(ratio)
    tau = (torch.exp(a * u) - 1.0) / (math.exp(a) - 1.0)
    t = 1.0 - tau
    return t
