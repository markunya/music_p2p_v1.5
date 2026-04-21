"""Call ``InversionPipeline`` and optionally ``save_generation_artifact``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.artifact_bundle import save_generation_artifact
from src.inversion.pipeline import InversionPipeline
from src.logging import utils as logging
from src.logging.writer import BaseWriter


def run_invert(
    handler: Any,
    work_cfg: DictConfig,
    *,
    music_path: str,
    artifact_out: Path | None,
    writer: BaseWriter | None = None,
) -> None:
    """Run inversion; persist ``.pt`` only when ``artifact_out`` is not ``None``."""
    inv = InversionPipeline(work_cfg).run(handler, music_path=music_path, writer=writer)

    if artifact_out is None:
        logging.info("artifact_out is null — skipping save (in-memory inversion only).")
        return

    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    save_generation_artifact(
        artifact_out,
        inv.cfg_snapshot,
        noise=inv.noise,
        null_encoder_hidden_states_per_step=inv.null_encoder_hidden_states_per_step,
    )
    logging.info(f"Saved inversion artifact to {artifact_out.resolve()}")
