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


def infer_attention_head_query_key(x: torch.Tensor) -> tuple[int, int, int, int]:
    if x.dim() == 4:
        b, h, q, k = x.shape
        return b, h, q, k
    if x.dim() == 3:
        h, q, k = x.shape
        return 1, h, q, k
    raise ValueError(f"expected attention tensor (B, H, Q, K) or (H, Q, K), got {tuple(x.shape)}")


def assert_row_stochastic(M: torch.Tensor, *, atol: float = 1e-5) -> None:
    if M.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(M.shape)}")
    if (M < -atol).any():
        raise ValueError("Matrix contains negative entries")
    sums = M.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0.0):
        bad = (sums - 1.0).abs().max().item()
        raise ValueError(f"Rows do not sum to 1 (max deviation {bad})")
