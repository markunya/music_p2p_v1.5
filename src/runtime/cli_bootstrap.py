"""Hydra entrypoints: handler init, WAV export, inversion paths, edit→invert cfg."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torchaudio
from acestep.handler import AceStepHandler
from omegaconf import DictConfig, OmegaConf

from src.logging import utils as logging
from src.utils.utils import resolve_against_original_cwd


def init_acestep_handler(work_cfg: DictConfig) -> Tuple[AceStepHandler, str]:
    project_root = resolve_against_original_cwd(str(work_cfg.acestep.project_root))
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
    return handler, status


def save_audios_to_exp_dir(audios: list, exp_dir: str) -> None:
    for i, item in enumerate(audios):
        tensor = item["tensor"]
        sr = int(item.get("sample_rate", 48_000))
        path = f"{exp_dir}/sample_{i}.wav"
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(path, tensor, sr)
        logging.info(f"Saved {path}")


def resolve_invert_artifact_out(cli_cfg: DictConfig, exp_dir: Path) -> Path | None:
    raw = OmegaConf.select(cli_cfg, "artifact_out")
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in ("null", "none"):
        return None
    p = Path(s)
    return p.resolve() if p.is_absolute() else (exp_dir / p).resolve()


def build_cfg_for_edit_inversion(edit_cfg: DictConfig) -> DictConfig:
    """Subset of fields for ``InversionPipeline`` using ``p2p_task.src`` text + source clip."""
    cfg = OmegaConf.create(OmegaConf.to_container(edit_cfg, resolve=True))
    p2p = edit_cfg.p2p_task
    cfg.prompt = OmegaConf.create(
        {
            "captions": str(p2p.src.captions),
            "lyrics": str(p2p.src.lyrics),
        }
    )
    vl = OmegaConf.select(p2p, "vocal_language", default=None)
    if vl is not None:
        cfg.vocal_language = str(vl)
    return cfg
