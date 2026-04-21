"""P2P edit: velocity fusion via ``ForwardPipeline``; optional on-the-fly inversion when no artifact file."""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from src.artifact_bundle import GenerationArtifactPayload
from src.forward.pipeline import ForwardPipeline
from src.inversion.pipeline import InversionPipeline
from src.logging import utils as logging
from src.p2p import P2PPromptPair
from src.forward.steppers.velocity_fusion import VelocityFusionEditRunner
from src.runtime.cli_bootstrap import build_cfg_for_edit_inversion, init_acestep_handler, save_audios_to_exp_dir
from src.utils.utils import resolve_against_original_cwd, set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


def _cfg_nonempty(x: object) -> bool:
    return x is not None and str(x).strip() not in ("", "null", "None")


@hydra.main(version_base=None, config_path="src/configs", config_name="p2p_edit")
def main(cli_cfg: DictConfig) -> None:
    OmegaConf.resolve(cli_cfg)
    set_random_seed(int(cli_cfg.seed))

    runner = instantiate(cli_cfg.p2p_strategy)
    prompts = P2PPromptPair.from_cfg_node(cli_cfg.p2p_task)

    src_audio = OmegaConf.select(cli_cfg, "source_audio_path", default=None)
    art = OmegaConf.select(cli_cfg, "artifact_path", default=None)
    has_src = _cfg_nonempty(src_audio)
    has_art = _cfg_nonempty(art)

    if not isinstance(runner, VelocityFusionEditRunner):
        logging.info(
            f"P2P runner {type(runner).__name__!r}: only ``VelocityFusionEditRunner`` is wired here "
            f"(prompts parsed OK, vocal_language={prompts.vocal_language!r})."
        )
        return

    if not has_src:
        logging.info(
            "velocity_fusion: set ``source_audio_path`` (and ``artifact_path`` or rely on on-the-fly inversion) "
            "via Hydra overrides."
        )
        return

    if bool(cli_cfg.debug_mode):
        logger.info("Resolved cli_cfg:\n{}", OmegaConf.to_yaml(cli_cfg))

    exp_dir = setup_exp_dir(cli_cfg)
    handler, status = init_acestep_handler(cli_cfg)
    logging.info(status)

    with open_dict(cli_cfg):
        if OmegaConf.select(cli_cfg, "forward") is None:
            cli_cfg.forward = OmegaConf.create({})
        cli_cfg.forward.mode = "velocity_fusion"

    if has_art:
        artifact_payload = GenerationArtifactPayload()
    else:
        inv_cfg = build_cfg_for_edit_inversion(cli_cfg)
        music_full = resolve_against_original_cwd(str(src_audio))
        inv = InversionPipeline(inv_cfg).run(handler, music_path=music_full)
        artifact_payload = GenerationArtifactPayload(
            noise=inv.noise,
            null_encoder_hidden_states_per_step=inv.null_encoder_hidden_states_per_step,
        )

    result = ForwardPipeline(cli_cfg).run(handler, artifact_payload, velocity_fusion_runner=runner)
    if not result.get("success"):
        raise RuntimeError(result.get("error") or result.get("status_message", "edit failed"))

    save_audios_to_exp_dir(result["audios"], exp_dir)
    logging.info(f"velocity_fusion edit finished → {Path(exp_dir).resolve()}")


if __name__ == "__main__":
    main()
