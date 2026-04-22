"""P2P edit MVP: invert track with ``src`` prompt, then batched forward with same noise and ``[src, tgt]`` conditioning."""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.forward.pipeline import ForwardPipeline
from src.inversion.pipeline import InversionPipeline
from src.logging.trajectory_logging import log_latent_trajectory, trajectory_image_flags
from src.logging.writer import setup_writer
from src.mps_adg_patch import apply_adg_mps_patch
from src.utils.initialization import init_dit_handler
from src.utils.conditioning import (
    p2p_src_tgt_prompt_configs,
    prepare_conditions,
)
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


@hydra.main(version_base=None, config_path="src/configs", config_name="edit_music")
def main(cli_cfg: DictConfig) -> None:
    OmegaConf.resolve(cli_cfg)

    if cli_cfg.debug_mode:
        logger.info("Resolved cfg:\n{}", OmegaConf.to_yaml(cli_cfg))

    use_random = bool(cli_cfg.use_random_seed)
    if not use_random:
        set_random_seed(int(cli_cfg.seed))

    exp_dir = Path(setup_exp_dir(cli_cfg))
    music_path = resolve_against_original_cwd(str(cli_cfg.music_path))
    inv_artifact_path = _artifact_out_path(cli_cfg, exp_dir)

    handler, status = init_dit_handler(cli_cfg)
    logger.info("{}", status.strip())

    src_tgt = p2p_src_tgt_prompt_configs(cli_cfg.p2p_task)
    prompt_src_only = [src_tgt[0]]

    model = handler.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    wav = _load_stereo_wav(music_path, target_sr=int(handler.sample_rate))
    grid_duration_sec = float(wav.shape[-1]) / float(handler.sample_rate)

    writer = setup_writer(cli_cfg)
    li, me = trajectory_image_flags(cli_cfg)
    try:
        writer.add_audio("edit/input/source_stereo", wav, sample_rate=int(handler.sample_rate))
        with torch.inference_mode():
            cond_inv = prepare_conditions(
                handler,
                prompt_src_only,
                float(cli_cfg.duration),
                source_stereo_wav=wav,
            )
            clean_latents = cond_inv.clean_latents
            if clean_latents is None:
                raise RuntimeError("prepare_conditions: expected clean_latents when source_stereo_wav is set")

            pipe_inv = InversionPipeline(cli_cfg)
            artifact, inv_traj = pipe_inv.run(model, clean_latents=clean_latents, model_condition=cond_inv)

            log_latent_trajectory(writer, inv_traj, prefix="edit/inversion_latent", log_images=li, max_edge=me)

            if inv_artifact_path is not None:
                artifact.save(inv_artifact_path)
                logger.info("Saved inversion artifact to {}", inv_artifact_path)

            noise = artifact.noise.to(device=device, dtype=dtype)
            noise_b2 = noise.repeat(2, 1, 1)

            cond_fwd = prepare_conditions(handler, src_tgt, grid_duration_sec)
            fwd = ForwardPipeline(cli_cfg)
            out = fwd.run(model, initial_latents=noise_b2, model_condition=cond_fwd)

        fwd_traj = out.get("trajectory")
        if isinstance(fwd_traj, list) and fwd_traj:
            log_latent_trajectory(writer, fwd_traj, prefix="edit/forward_latent")

        x = out["final_latents"]
        latents_decode = x.transpose(1, 2).contiguous().to(handler.vae.dtype)
        wavs = handler.tiled_decode(latents_decode)

        sr = int(handler.sample_rate)
        fmt = str(cli_cfg.audio_format).lower()
        for i in range(wavs.shape[0]):
            out_path = exp_dir / f"sample_{i}.{fmt}"
            clip = wavs[i].detach().cpu().float()
            torchaudio.save(str(out_path), clip, sr)
            label = "src_recon" if i == 0 else "tgt_edit"
            writer.add_audio(f"edit/output/sample_{i}_{label}", clip, sample_rate=sr)

        logger.info("Saved {} file(s) under {} (sample_0=src recon, sample_1=tgt from shared noise)", wavs.shape[0], exp_dir)
    finally:
        writer.end()


if __name__ == "__main__":
    main()
