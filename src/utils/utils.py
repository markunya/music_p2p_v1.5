import os
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.logging import utils as logging


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_exp_dir(cfg: DictConfig) -> str:
    """Create experiment dir under ``save_dir`` (relative to original cwd)."""
    base = Path(get_original_cwd()) / str(cfg.save_dir)
    exp_dir = base / str(cfg.exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = exp_dir / "config.yaml"
    OmegaConf.save(config=cfg, f=str(cfg_path), resolve=True)
    logging.info(f"Experiment directory: {exp_dir}")
    return str(exp_dir)


def resolve_against_original_cwd(path: str) -> str:
    """Resolve a path from config as relative to Hydra original cwd."""
    p = Path(path)
    if p.is_absolute():
        return str(p.resolve())
    return str((Path(get_original_cwd()) / p).resolve())
