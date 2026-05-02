import torch
from omegaconf import DictConfig, OmegaConf

from src.logging.writer import BaseWriter


def trajectory_image_flags(cfg: DictConfig) -> tuple[bool, int]:
    return (
        bool(OmegaConf.select(cfg, "log_trajectory_images", default=False)),
        int(OmegaConf.select(cfg, "log_trajectory_max_edge", default=4096)),
    )


def log_latent_trajectory(
    writer: BaseWriter,
    trajectory: list[torch.Tensor],
    *,
    prefix: str,
    log_images: bool = False,
    max_edge: int = 4096,
) -> None:
    if not trajectory:
        return
    prev: torch.Tensor | None = None
    for i, xt in enumerate(trajectory):
        x = xt.detach().float().cpu()
        writer.set_step(i)
        writer.add_scalar(f"{prefix}/latent_mean", float(x.mean().item()), step=i)
        writer.add_scalar(f"{prefix}/latent_std", float(x.std().item()), step=i)
        writer.add_scalar(f"{prefix}/latent_l2", float(torch.linalg.vector_norm(x).item()), step=i)
        if prev is not None:
            delta = (x - prev).abs().mean()
            writer.add_scalar(f"{prefix}/delta_mean_abs", float(delta.item()), step=i)
        prev = x

    if log_images:
        from src.logging.trajectory_images import log_latent_trajectory_images

        log_latent_trajectory_images(writer, trajectory, prefix=f"{prefix}/grid", max_edge=max_edge)
