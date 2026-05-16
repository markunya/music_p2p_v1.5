from typing import TYPE_CHECKING, Any

from loguru import logger
import torch

from src.utils.utils import resolve_against_original_cwd

if TYPE_CHECKING:
    from omegaconf import DictConfig


def init_dit_handler(cfg: "DictConfig"):
    from acestep.handler import AceStepHandler

    project_root = resolve_against_original_cwd(str(cfg.acestep.project_root))
    use_flash_attention = False
    try:
        import flash_attn

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


def pad_latent_time(lat: torch.Tensor, target_t: int) -> torch.Tensor:
    if lat.dim() != 3:
        raise ValueError(f"Expected latents [B, T, C], got shape {tuple(lat.shape)}")
    _b, t, _c = lat.shape
    if t == target_t:
        return lat
    if t > target_t:
        return lat[:, :target_t].contiguous()
    pad_len = target_t - t
    return torch.nn.functional.pad(lat, (0, 0, 0, pad_len))


def encode_clean_latents(
    handler: Any,
    wav: torch.Tensor,
    *,
    target_t: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    wav_on_dev = wav.to(handler.device).to(handler._get_vae_dtype())
    music_lat = handler._encode_audio_to_latents(wav_on_dev)
    if music_lat.dim() == 2:
        music_lat = music_lat.unsqueeze(0)
    return pad_latent_time(music_lat, target_t).to(dtype=out_dtype)
