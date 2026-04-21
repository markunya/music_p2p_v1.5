"""Text-to-music: cfg + optional artifact → ``ForwardPipeline`` → WAV."""

from __future__ import annotations

from src.mps_adg_patch import apply_adg_mps_patch

apply_adg_mps_patch()

import warnings

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.forward.artifact_resolve import forward_artifact_and_work_cfg
from src.forward.pipeline import ForwardPipeline
from src.logging import utils as logging
from src.runtime.cli_bootstrap import init_acestep_handler, save_audios_to_exp_dir
from src.utils.utils import set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="generate")
def main(cli_cfg: DictConfig) -> None:
    OmegaConf.resolve(cli_cfg)

    work_cfg, artifact_payload = forward_artifact_and_work_cfg(cli_cfg)
    OmegaConf.resolve(work_cfg)

    if work_cfg.debug_mode:
        logger.info("Resolved work_cfg:\n{}", OmegaConf.to_yaml(work_cfg))

    set_random_seed(int(work_cfg.seed))

    exp_dir = setup_exp_dir(work_cfg)

    handler, status = init_acestep_handler(work_cfg)
    logging.info(status)

    result = ForwardPipeline(work_cfg).run(handler, artifact_payload)

    if not result.get("success"):
        raise RuntimeError(result.get("error") or result.get("status_message", "generation failed"))

    save_audios_to_exp_dir(result["audios"], exp_dir)
    logging.info(result.get("status_message", "done"))


if __name__ == "__main__":
    main()
