"""Invert a clip: pivot (+ optional NTI) → ``InversionArtifact``; save only if ``artifact_out`` is set."""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.logging import utils as logging
from src.logging.writer import setup_writer
from src.inversion.run_invert import run_invert
from src.runtime.cli_bootstrap import init_acestep_handler, resolve_invert_artifact_out
from src.utils.utils import resolve_against_original_cwd, set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="invert_music")
def main(cli_cfg: DictConfig) -> None:
    OmegaConf.resolve(cli_cfg)

    set_random_seed(int(cli_cfg.seed))

    exp_dir = Path(setup_exp_dir(cli_cfg))
    music_path = resolve_against_original_cwd(str(cli_cfg.music_path))
    artifact_path = resolve_invert_artifact_out(cli_cfg, exp_dir)

    handler, status = init_acestep_handler(cli_cfg)
    logging.info(status)

    writer = setup_writer(cli_cfg)
    try:
        run_invert(
            handler,
            cli_cfg,
            music_path=str(music_path),
            artifact_out=artifact_path,
            writer=writer,
        )
    finally:
        writer.end()


if __name__ == "__main__":
    main()
