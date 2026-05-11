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
    if abs(ratio - 1.0) < 1e-8:
        return torch.linspace(1.0, 0.0, n + 1, device=device, dtype=dtype)
        
    u = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=dtype)
    a = math.log(ratio)
    tau = (torch.exp(a * u) - 1.0) / (math.exp(a) - 1.0)
    t = 1.0 - tau
    return t

def make_time_grid_piecewise_dense_start(
    n: int,
    device,
    dtype,
    split_t: float = 0.6,
    dense_frac: float = 0.7,
) -> torch.Tensor:
    if not (0.0 < split_t < 1.0):
        raise ValueError("split_t must be in (0, 1)")
    if not (0.0 < dense_frac < 1.0):
        raise ValueError("dense_frac must be in (0, 1)")
    if n < 2:
        raise ValueError("n must be >= 2")

    n_dense = int(round(n * dense_frac))
    n_dense = max(1, min(n - 1, n_dense))
    n_sparse = n - n_dense

    t_dense = torch.linspace(1.0, split_t, n_dense + 1, device=device, dtype=dtype)
    t_sparse = torch.linspace(split_t, 0.0, n_sparse + 1, device=device, dtype=dtype)

    return torch.cat([t_dense[:-1], t_sparse], dim=0)

def make_time_grid_edm(
    n: int,
    device,
    dtype,
    p: float = 1.3,
) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=dtype)
    t = 1.0 - u.pow(p)
    return t
