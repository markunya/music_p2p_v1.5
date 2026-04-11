"""Text-to-music generation via ACE-Step 1.5 + Hydra (scaffold)."""

from __future__ import annotations

import warnings

import hydra
import torchaudio
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from acestep.handler import AceStepHandler

from src.logging import utils as logging
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
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    if cfg.debug_mode:
        logger.info("Resolved config:\n{}", OmegaConf.to_yaml(cfg))

    set_random_seed(int(cfg.seed))

    project_root = resolve_against_original_cwd(str(cfg.acestep.project_root))
    exp_dir = setup_exp_dir(cfg)

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path=str(cfg.acestep.config_path),
        device=str(cfg.acestep.device),
        use_mlx_dit=bool(cfg.acestep.use_mlx_dit),
        offload_to_cpu=bool(cfg.acestep.offload_to_cpu),
    )
    if not ok:
        raise RuntimeError(f"initialize_service failed: {status}")
    logging.info(status)

    audio_duration = float(cfg.duration) if float(cfg.duration) > 0 else None
    seed_arg: str | int = -1 if bool(cfg.use_random_seed) else int(cfg.seed)

    result = handler.generate_music(
        captions=str(cfg.prompt.captions),
        lyrics=str(cfg.prompt.lyrics),
        vocal_language=str(cfg.vocal_language),
        inference_steps=int(cfg.inference_steps),
        guidance_scale=float(cfg.guidance_scale),
        use_random_seed=bool(cfg.use_random_seed),
        seed=seed_arg,
        audio_duration=audio_duration,
        batch_size=int(cfg.batch_size),
        task_type=str(cfg.task_type),
        shift=float(cfg.shift),
        infer_method=str(cfg.infer_method),
        sampler_mode=str(cfg.sampler_mode),
        use_adg=bool(cfg.use_adg),
        cfg_interval_start=float(cfg.cfg_interval_start),
        cfg_interval_end=float(cfg.cfg_interval_end),
        audio_code_string="",
    )

    if not result.get("success"):
        raise RuntimeError(result.get("error") or result.get("status_message", "generation failed"))

    _save_wavs(result["audios"], exp_dir)
    logging.info(result.get("status_message", "done"))


if __name__ == "__main__":
    main()
