"""Pivot trajectory for Comet: one (T, C) image per diffusion step (step slider in UI).

Values are shown as a **pseudocolor heatmap** (``matplotlib`` **Viridis**: dark blue → teal → green → yellow).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.logging import utils as logging
from src.logging.writer import BaseWriter, DummyWriter


def _scalar_to_rgb_viridis(t01: np.ndarray) -> np.ndarray:
    """Map scalar field ``t01`` in ``[0, 1]`` to RGB float ``H×W×3`` in ``[0, 1]`` (Viridis)."""
    t01 = np.clip(t01.astype(np.float64), 0.0, 1.0)
    try:
        from matplotlib import colormaps

        rgba = colormaps["viridis"](t01)
    except Exception:
        try:
            import matplotlib.cm as cm  # type: ignore[import-not-found]

            rgba = cm.get_cmap("viridis")(t01)
        except Exception:
            z = np.zeros_like(t01, dtype=np.float32)
            g = t01.astype(np.float32)
            return np.stack([z, g, z], axis=-1)

    return np.asarray(rgba[..., :3], dtype=np.float32)


def _latent_matrix_falsecolor_rgb(mat: np.ndarray, *, max_edge: int) -> np.ndarray:
    """``(T, C)`` float → min-max → Viridis ``uint8`` ``H×W×3``, optional downscale."""
    lo = float(mat.min())
    hi = float(mat.max())
    if hi - lo < 1e-8:
        t01 = np.zeros(mat.shape, dtype=np.float32)
    else:
        t01 = np.clip((mat.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

    rgb = _scalar_to_rgb_viridis(t01)
    h, w = rgb.shape[:2]
    if max_edge > 0 and max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        tt = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
        tt = F.interpolate(tt, size=(nh, nw), mode="bilinear", align_corners=False)
        rgb = tt.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy()

    return np.clip(rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def log_pivot_trajectory_comet(
    writer: BaseWriter,
    trajectory: List[torch.Tensor],
    *,
    max_edge: int = 4096,
    prefix: str = "nti/pivot_trajectory",
) -> None:
    """Log ``(T, C)`` per diffusion index ``s`` at ``step=s`` (same name per batch → Comet step slider)."""
    if isinstance(writer, DummyWriter) or not trajectory:
        return
    bsz = int(trajectory[0].shape[0])
    for s, x in enumerate(trajectory):
        for b in range(bsz):
            try:
                mat = x[b].detach().float().cpu().numpy()
                rgb = _latent_matrix_falsecolor_rgb(mat, max_edge=max_edge)
                writer.add_image(f"{prefix}/batch_{b}", rgb, step=s)
            except Exception as exc:
                logging.info(f"{prefix}: batch_{b} step_{s} Comet image skipped: {exc}")
