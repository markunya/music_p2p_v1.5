"""Общий ODE-Euler цикл по сетке \(\sigma\) (как в ACE ``generate_audio``, без cover/repaint)."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from tqdm import tqdm

from src.forward.steppers.base import LocalDiffusionStepper
from src.nti.schedule import acestep_sigma_grid


def pairs_from_sigma_grid(
    infer_steps: int,
    shift: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    t_full = acestep_sigma_grid(infer_steps, shift, device=device, dtype=dtype)
    return list(zip(t_full[:-1], t_full[1:]))


def run_euler_ode_on_pairs(
    xt_init: torch.Tensor,
    pairs: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    stepper: LocalDiffusionStepper,
    *,
    use_progress_bar: bool,
    desc: str,
) -> torch.Tensor:
    pair_list = list(pairs)
    iterator = tqdm(pair_list, total=len(pair_list), desc=desc) if use_progress_bar else pair_list
    xt = xt_init
    for step_idx, (t_curr, t_prev) in enumerate(iterator):
        tc = float(t_curr.item()) if isinstance(t_curr, torch.Tensor) else float(t_curr)
        tp = float(t_prev.item()) if isinstance(t_prev, torch.Tensor) else float(t_prev)
        xt = stepper.step(xt, tc, tp, step_idx)
    return xt


def run_euler_ode_loop(
    xt_init: torch.Tensor,
    infer_steps: int,
    shift: float,
    stepper: LocalDiffusionStepper,
    *,
    use_progress_bar: bool,
    desc: str,
) -> torch.Tensor:
    device, dtype = xt_init.device, xt_init.dtype
    pairs = pairs_from_sigma_grid(infer_steps, shift, device=device, dtype=dtype)
    return run_euler_ode_on_pairs(xt_init, pairs, stepper, use_progress_bar=use_progress_bar, desc=desc)
