"""Text-to-music: один путь — cfg задаёт генерацию; шум из артефакта или случайный."""

from __future__ import annotations

import warnings

import hydra
import torchaudio
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from acestep.handler import AceStepHandler

from src.artifact_bundle import load_generation_bundle
from src.logging import utils as logging
from src.run_generate import run_generate
from src.utils.utils import resolve_against_original_cwd, set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


def _save_wavs(audios: list, exp_dir: str) -> None:
    for i, item in enumerate(audios):
        tensor = item["tensor"]
        sr = int(item.get("sample_rate", 48_000))
        path = f"{exp_dir}/sample_{i}.wav"
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(path, tensor, sr)
        logging.info(f"Saved {path}")


@hydra.main(version_base=None, config_path="src/configs", config_name="generate")
def main(cli_cfg: DictConfig) -> None:
    OmegaConf.resolve(cli_cfg)

    work_cfg, artifact_payload = load_generation_bundle(cli_cfg)
    OmegaConf.resolve(work_cfg)

    if work_cfg.debug_mode:
        logger.info("Resolved work_cfg:\n{}", OmegaConf.to_yaml(work_cfg))

    set_random_seed(int(work_cfg.seed))

    project_root = resolve_against_original_cwd(str(work_cfg.acestep.project_root))
    exp_dir = setup_exp_dir(work_cfg)

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path=str(work_cfg.acestep.config_path),
        device=str(work_cfg.acestep.device),
        use_mlx_dit=bool(work_cfg.acestep.use_mlx_dit),
        offload_to_cpu=bool(work_cfg.acestep.offload_to_cpu),
    )
    if not ok:
        raise RuntimeError(f"initialize_service failed: {status}")
    logging.info(status)

    result = run_generate(handler, work_cfg, artifact_payload)

    if not result.get("success"):
        raise RuntimeError(result.get("error") or result.get("status_message", "generation failed"))

    _save_wavs(result["audios"], exp_dir)
    logging.info(result.get("status_message", "done"))


if __name__ == "__main__":
    main()
