"""Invert a clip to latent noise and optionally save a ``.pt`` artifact (Hydra)."""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.inversion.artifact import InversionArtifact
from src.inversion.pipeline import InversionPipeline
from src.logging.writer import setup_writer
from src.mps_adg_patch import apply_adg_mps_patch
from src.utils.initialization import init_dit_handler
from src.utils.conditioning import prepare_conditions, prompts_from_hydra_prompt_node
from src.utils.utils import resolve_against_original_cwd, set_random_seed, setup_exp_dir

apply_adg_mps_patch()

warnings.filterwarnings("ignore", category=UserWarning)


def _load_stereo_wav(path: str, target_sr: int) -> torch.Tensor:
    """Load WAV/FLAC, mix to stereo, resample to ``target_sr``. Returns ``[2, T]`` float32."""
    wav, sr = torchaudio.load(path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    wav = wav.to(dtype=torch.float32)
    if int(sr) != int(target_sr):
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=int(target_sr))
    return wav


def _artifact_out_path(cli_cfg: DictConfig, exp_dir: Path) -> Path | None:
    raw = cli_cfg.get("artifact_out")
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("null", "none", ""):
        return None
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return exp_dir / p


@hydra.main(version_base=None, config_path="src/configs", config_name="invert_music")
def main(cli_cfg: DictConfig) -> None:
    OmegaConf.resolve(cli_cfg)

    if cli_cfg.debug_mode:
        logger.info("Resolved cfg:\n{}", OmegaConf.to_yaml(cli_cfg))

    use_random = bool(cli_cfg.use_random_seed)
    if not use_random:
        set_random_seed(int(cli_cfg.seed))

    exp_dir = Path(setup_exp_dir(cli_cfg))
    music_path = resolve_against_original_cwd(str(cli_cfg.music_path))
    out_path = _artifact_out_path(cli_cfg, exp_dir)

    handler, status = init_dit_handler(cli_cfg)
    logger.info("{}", status.strip())

    bsz = int(cli_cfg.batch_size)
    if bsz != 1:
        raise ValueError("invert_music v1: batch_size must be 1")

    prompts = prompts_from_hydra_prompt_node(cli_cfg.prompt, bsz)
    wav = _load_stereo_wav(music_path, target_sr=int(handler.sample_rate))

    model = handler.model
    writer = setup_writer(cli_cfg)
    try:
        with torch.inference_mode():
            cond = prepare_conditions(handler, prompts, float(cli_cfg.duration), source_stereo_wav=wav)
            clean_latents = cond.clean_latents
            if clean_latents is None:
                raise RuntimeError("prepare_conditions: expected clean_latents when source_stereo_wav is set")
            pipe = InversionPipeline(cli_cfg)
            artifact = pipe.run(model, clean_latents=clean_latents, model_condition=cond)

        if out_path is not None:
            artifact.save(out_path)
            logger.info("Saved inversion artifact to {}", out_path)
        else:
            logger.info("artifact_out is null — artifact not written to disk")
    finally:
        writer.end()


if __name__ == "__main__":
    main()
