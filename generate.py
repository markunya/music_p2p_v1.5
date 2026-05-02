import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.mps_adg_patch import apply_adg_mps_patch

apply_adg_mps_patch()

from src.forward.pipeline import ForwardPipeline
from src.inversion.artifact import InversionArtifact
from src.logging.trajectory_logging import log_latent_trajectory
from src.logging.writer import setup_writer
from src.utils.conditioning import prepare_conditions, prompts_from_hydra_prompt_node
from src.utils.initialization import init_dit_handler
from src.utils.utils import resolve_against_original_cwd, set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="generate")
def main(cli_cfg: DictConfig) -> None:
    OmegaConf.resolve(cli_cfg)

    if cli_cfg.debug_mode:
        logger.info("Resolved cfg:\n{}", OmegaConf.to_yaml(cli_cfg))

    use_random = bool(cli_cfg.use_random_seed)
    if not use_random:
        set_random_seed(int(cli_cfg.seed))

    exp_dir = Path(setup_exp_dir(cli_cfg))
    writer = setup_writer(cli_cfg)

    handler, status = init_dit_handler(cli_cfg)
    logger.info("{}", status.strip())

    bsz = int(cli_cfg.batch_size)
    prompts = prompts_from_hydra_prompt_node(cli_cfg.prompt, bsz)
    duration = float(cli_cfg.duration)

    model = handler.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    artifact_path_raw = OmegaConf.select(cli_cfg, "artifact.path", default=None)
    artifact_path = str(artifact_path_raw).strip() if artifact_path_raw is not None else ""

    try:
        with torch.inference_mode():
            cond = prepare_conditions(handler, prompts, duration)
            seeds, seed_msg = handler.prepare_seeds(bsz, int(cli_cfg.seed), use_random)
            logger.info("Seeds: {}", seed_msg)

            if artifact_path and artifact_path.lower() not in ("null", "none", ""):
                ap = resolve_against_original_cwd(artifact_path)
                inversion_artifact = InversionArtifact.load(ap)
                if inversion_artifact.inference_steps and int(inversion_artifact.inference_steps) != int(
                    cli_cfg.inference_steps
                ):
                    logger.warning(
                        "Artifact inference_steps={} != cfg inference_steps={}; continuing with cfg grid",
                        inversion_artifact.inference_steps,
                        int(cli_cfg.inference_steps),
                    )
                logger.info(
                    "Loaded inversion artifact from {} (noise.shape={}, stepper={}, forward_start={})",
                    ap,
                    tuple(inversion_artifact.noise.shape),
                    inversion_artifact.stepper_class_name or "?",
                    inversion_artifact.forward_start_step_index,
                )
            else:
                inversion_artifact = InversionArtifact.from_noise(
                    model.prepare_noise(cond.context_latents, seed=seeds)
                )

            pipe = ForwardPipeline(cli_cfg)
            out = pipe.run(model, model_condition=cond, inversion_artifact=inversion_artifact)

        traj = out.get("trajectory")
        if isinstance(traj, list) and traj:
            log_latent_trajectory(writer, traj, prefix="gen/forward_latent")

        x = out["final_latents"]
        latents_decode = x.transpose(1, 2).contiguous().to(handler.vae.dtype)
        wavs = handler.tiled_decode(latents_decode)

        sr = int(handler.sample_rate)
        fmt = str(cli_cfg.audio_format).lower()
        for i in range(wavs.shape[0]):
            out_path = exp_dir / f"sample_{i}.{fmt}"
            clip = wavs[i].detach().cpu().float()
            torchaudio.save(str(out_path), clip, sr)
            writer.add_audio(f"gen/output/sample_{i}", clip, sample_rate=sr)

        logger.info("Saved {} file(s) under {}", wavs.shape[0], exp_dir)
    finally:
        writer.end()


if __name__ == "__main__":
    main()
