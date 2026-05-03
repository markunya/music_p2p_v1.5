import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.attention_injection.controllers.base import AttentionControllerBase
from src.forward.pipeline import ForwardPipeline
from src.inversion.pipeline import InversionPipeline
from src.logging.trajectory_logging import log_latent_trajectory, trajectory_image_flags
from src.logging.writer import setup_writer
from src.mps_adg_patch import apply_adg_mps_patch
from src.utils.initialization import encode_clean_latents, init_dit_handler
from src.utils.conditioning import (
    p2p_src_tgt_prompt_configs,
    prepare_conditions,
)
from src.utils.utils import resolve_against_original_cwd, set_random_seed, setup_exp_dir
from src.attention_injection.reweight_utils import parse_reweight_from_tgt

apply_adg_mps_patch()

warnings.filterwarnings("ignore", category=UserWarning)


def _load_stereo_wav(path: str, target_sr: int) -> torch.Tensor:
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

    src_raw, tgt_raw = p2p_src_tgt_prompt_configs(cli_cfg.p2p_task)
    clean_tgt, _ = parse_reweight_from_tgt(tgt_raw)
    src_tgt = [src_raw, clean_tgt]

    model = handler.model

    wav = _load_stereo_wav(music_path, target_sr=int(handler.sample_rate))
    duration = float(wav.shape[-1]) / float(handler.sample_rate)
    OmegaConf.update(cli_cfg, "duration", duration, force_add=True)

    writer = setup_writer(cli_cfg)
    try:
        writer.add_audio("edit/input/source_stereo", wav, sample_rate=int(handler.sample_rate))
        with torch.no_grad():
            cond_fwd = prepare_conditions(handler, src_tgt, float(cli_cfg.duration))
            cond_inv = cond_fwd.slice(0, 1)
            clean_latents = encode_clean_latents(
                handler,
                wav,
                target_t=int(cond_inv.context_latents.shape[1]),
                out_dtype=cond_inv.context_latents.dtype,
            )

        pipe_inv = InversionPipeline(cli_cfg)
        artifact, _ = pipe_inv.run(
            model, clean_latents=clean_latents, model_condition=cond_inv, writer=writer
        )

        if inv_artifact_path is not None:
            artifact.save(inv_artifact_path)
            logger.info("Saved inversion artifact to {}", inv_artifact_path)

        controller: AttentionControllerBase = instantiate(cli_cfg.controller)
        controller.build(handler=handler, cfg=cli_cfg, writer=writer)

        with torch.inference_mode():
            fwd = ForwardPipeline(cli_cfg, attention_controller=controller)
            out = fwd.run(model, model_condition=cond_fwd, inversion_artifact=artifact)

        fwd_traj = out.get("trajectory")
        if isinstance(fwd_traj, list) and fwd_traj:
            li, me = trajectory_image_flags(cli_cfg)
            log_latent_trajectory(writer, fwd_traj, prefix="edit/forward_latent", log_images=li, max_edge=me)

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
