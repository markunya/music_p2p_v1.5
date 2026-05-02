from typing import Any, List

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm

from src.logging.writer import BaseWriter, DummyWriter
from src.steppers.guidance import GuidanceStepper
from src.utils.conditioning import ModelCondition


def _lr_outer(j: int, n_infer: int, lr: float) -> float:
    denom = max(n_infer - 1, 1)
    return float(lr + (lr / 2.0 - lr) * (j / denom))


class NullTextOptimization:
    def __init__(
        self,
        cfg: DictConfig,
        *,
        writer: BaseWriter | None = None,
    ) -> None:
        self._cfg = cfg
        nti = cfg.nti
        self._lr = float(nti.lr)
        self._num_inner = int(nti.num_inner_steps)
        self._epsilon = float(nti.epsilon)
        self._init_from_previous_outer = bool(
            OmegaConf.select(nti, "init_from_previous_outer", default=True)
        )
        self._writer = writer if writer is not None else DummyWriter()

    def run(
        self,
        model: torch.nn.Module,
        *,
        trajectory: List[torch.Tensor],
        model_condition: ModelCondition,
        guidance_stepper: GuidanceStepper,
        infer_steps: int,
        forward_start_step_index: int = 0,
    ) -> List[torch.Tensor]:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        cond_base = model_condition.clone()
        cond_ref = cond_base.encoder_hidden_states
        if cond_ref.shape[0] != 1:
            raise ValueError(f"NTI v1 expects batch size 1, got {cond_ref.shape[0]}")

        start = int(forward_start_step_index)
        if start < 0 or start > infer_steps:
            raise ValueError(f"forward_start_step_index must be in [0, {infer_steps}], got {start}")

        rev_traj = trajectory[::-1]
        expected_len = infer_steps - start + 1
        if len(rev_traj) != expected_len:
            raise ValueError(
                f"NTI: trajectory length {len(trajectory)} (rev {len(rev_traj)}) != "
                f"infer_steps - forward_start + 1 = {expected_len} (infer_steps={infer_steps}, start={start})"
            )

        t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
        null_stored: List[torch.Tensor] = []
        latent_cur = rev_traj[0].detach().to(device=device, dtype=dtype)
        null_emb_prev: torch.Tensor | None = None

        n_outer = len(rev_traj) - 1
        if n_outer == 0:
            logger.info("NTI: zero outer steps (trajectory length 1); skipping null-text optimization")
            return []

        if not self._init_from_previous_outer:
            logger.info(
                "NTI: init_from_previous_outer=False — each outer step starts from model.null_condition_emb"
            )

        outer = tqdm(range(n_outer), desc="NTI (null-text)", leave=False)
        for j in outer:
            target = rev_traj[j + 1].detach().to(device=device, dtype=dtype)
            t_curr, t_next = t[start + j], t[start + j + 1]
            lr_j = _lr_outer(j, n_outer, self._lr)

            use_prev = self._init_from_previous_outer and j > 0 and null_emb_prev is not None
            if use_prev:
                null_emb = null_emb_prev.detach().clone().requires_grad_(True)
            else:
                null_emb = (
                    model.null_condition_emb.expand_as(cond_ref).clone().detach().requires_grad_(True)
                )

            guidance_stepper.reset_guidance_layout()
            optimizer = AdamW([null_emb], lr=lr_j, weight_decay=0.1)

            for k in range(self._num_inner):
                guidance_stepper.reset_guidance_layout()
                cond_nti = cond_base.clone()
                cond_nti.past_key_values = None
                guidance_stepper.reset_apg_momentum_for_nti_inner()
                guidance_stepper.set_null_encoder_override(null_emb)

                payload = guidance_stepper.step(
                    model=model,
                    x=latent_cur,
                    t_curr=t_curr,
                    t_next=t_next,
                    model_condition=cond_nti,
                )
                loss = F.mse_loss(payload.x, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                global_step = j * self._num_inner + k
                self._writer.add_scalar("nti/loss", float(loss.item()), step=global_step)
                self._writer.add_scalar("nti/lr_outer", lr_j, step=global_step)

                if float(loss.item()) < self._epsilon:
                    break

            self._writer.add_scalar(
                "nti/null_emb_norm",
                float(null_emb.detach().float().norm().item()),
                step=j,
            )

            null_stored.append(null_emb.detach().cpu())
            null_emb_prev = null_emb.detach()

            guidance_stepper.reset_guidance_layout()
            cond_final = cond_base.clone()
            cond_final.past_key_values = None
            guidance_stepper.set_null_encoder_override(null_emb_prev)
            with torch.no_grad():
                p = guidance_stepper.step(
                    model=model,
                    x=latent_cur,
                    t_curr=t_curr,
                    t_next=t_next,
                    model_condition=cond_final,
                )
                latent_cur = p.x.detach()

            guidance_stepper.set_null_encoder_override(None)
            guidance_stepper.reset_guidance_layout()

        logger.info("NTI done: {} null embeddings stored", len(null_stored))
        return null_stored


def validate_nti_prerequisites(model: Any, stepper: Any) -> None:
    if not isinstance(stepper, GuidanceStepper):
        raise TypeError("NTI requires cfg.stepper to instantiate GuidanceStepper")
    if float(stepper.guidance_scale) <= 1.0:
        raise ValueError("NTI requires guidance_scale > 1")
    if not hasattr(model, "null_condition_emb"):
        raise ValueError("NTI requires model.null_condition_emb")
