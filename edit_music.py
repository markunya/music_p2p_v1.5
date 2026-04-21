"""P2P edit: velocity fusion via ``ForwardPipeline``; optional on-the-fly inversion when no artifact file."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import torch
from src.forward.edit_forward import UnifiedEditForwardRunner
from src.forward.pipeline import ForwardPipeline, ForwardStrategy
from src.inversion.pipeline import InversionPipeline
from src.logging import utils as logging
from src.p2p import P2PPromptPair
from src.runtime.cli_bootstrap import build_cfg_for_edit_inversion, init_acestep_handler, save_audios_to_exp_dir
from src.utils.utils import resolve_against_original_cwd, set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="p2p_edit")
def main(cli_cfg: DictConfig) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
