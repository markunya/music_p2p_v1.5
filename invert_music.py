"""Invert a clip: pivot trajectory + null-text optimization → ``artifact.pt`` for ``generate.py``."""

from __future__ import annotations

from src.mps_adg_patch import apply_adg_mps_patch

apply_adg_mps_patch()

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from acestep.handler import AceStepHandler

from src.logging import utils as logging
from src.logging.writer import setup_writer
from src.run_invert import run_invert
from src.utils.utils import resolve_against_original_cwd, set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="invert_music")
def main(cli_cfg: DictConfig) -> None:
    OmegaConf.resolve(cli_cfg)

    set_random_seed(int(cli_cfg.seed))

    project_root = resolve_against_original_cwd(str(cli_cfg.acestep.project_root))
    exp_dir = Path(setup_exp_dir(cli_cfg))
    music_path = resolve_against_original_cwd(str(cli_cfg.music_path))
    artifact_name = str(cli_cfg.artifact_out)
    artifact_path = Path(artifact_name)
    if not artifact_path.is_absolute():
        artifact_path = exp_dir / artifact_name

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path=str(cli_cfg.acestep.config_path),
        device=str(cli_cfg.acestep.device),
        use_mlx_dit=bool(cli_cfg.acestep.use_mlx_dit),
        offload_to_cpu=bool(cli_cfg.acestep.offload_to_cpu),
    )
    if not ok:
        raise RuntimeError(f"initialize_service failed: {status}")
    logging.info(status)

    writer = setup_writer(cli_cfg)
    try:
        run_invert(handler, cli_cfg, music_path=str(music_path), artifact_out=artifact_path, writer=writer)
    finally:
        writer.end()


if __name__ == "__main__":
    main()
