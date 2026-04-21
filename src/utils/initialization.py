"""ACE-Step DiT handler initialization (no LLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.utils.utils import resolve_against_original_cwd

if TYPE_CHECKING:
    from omegaconf import DictConfig


def init_dit_handler(cfg: "DictConfig"):
    """Return ``(AceStepHandler, status_message)`` after ``initialize_service``."""
    from acestep.handler import AceStepHandler

    project_root = resolve_against_original_cwd(str(cfg.acestep.project_root))
    use_flash_attention = False
    try:
        import flash_attn  # noqa: F401

        use_flash_attention = True
    except ImportError:
        pass

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path=str(cfg.acestep.config_path),
        device=str(cfg.acestep.device),
        use_flash_attention=use_flash_attention,
        compile_model=False,
        offload_to_cpu=bool(cfg.acestep.offload_to_cpu),
        offload_dit_to_cpu=bool(getattr(cfg.acestep, "offload_dit_to_cpu", False)),
        quantization=None,
        use_mlx_dit=bool(cfg.acestep.use_mlx_dit),
    )
    if not ok:
        logger.error("DiT init failed: {}", status)
        raise RuntimeError(status)
    return handler, status
