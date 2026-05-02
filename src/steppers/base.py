from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from src.utils.conditioning import ModelCondition


@dataclass
class StepperPayload:
    x: torch.Tensor
    v: torch.Tensor


class BaseStepper(ABC):
    @staticmethod
    def _segment_dt_tensor(
        dt: torch.Tensor,
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return dt * torch.ones((bsz,), device=device, dtype=dtype).view(-1, 1, 1)

    @staticmethod
    def decoder_velocity(
        model: torch.nn.Module,
        xt: torch.Tensor,
        t_scalar: torch.Tensor,
        model_condition: ModelCondition,
        *,
        use_cache: bool,
        past_key_values: Any,
    ) -> tuple[torch.Tensor, Any]:
        bsz = xt.shape[0]
        device, dtype = xt.device, xt.dtype
        t_tensor = t_scalar * torch.ones((bsz,), device=device, dtype=dtype)
        out = model.decoder(
            hidden_states=xt,
            timestep=t_tensor,
            timestep_r=t_tensor,
            attention_mask=model_condition.attention_mask,
            encoder_hidden_states=model_condition.encoder_hidden_states,
            encoder_attention_mask=model_condition.encoder_attention_mask,
            context_latents=model_condition.context_latents,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        return out[0], out[1]

    @staticmethod
    def velocity_with_side_cache(
        model: torch.nn.Module,
        xt: torch.Tensor,
        t_scalar: torch.Tensor,
        model_condition: ModelCondition,
    ) -> torch.Tensor:
        if model_condition.past_key_values is None:
            model_condition.past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        vt, new_pkv = BaseStepper.decoder_velocity(
            model,
            xt,
            t_scalar,
            model_condition,
            use_cache=True,
            past_key_values=model_condition.past_key_values,
        )
        model_condition.past_key_values = new_pkv
        return vt

    @staticmethod
    def velocity_fresh_cache(
        model: torch.nn.Module,
        xt: torch.Tensor,
        t_scalar: torch.Tensor,
        model_condition: ModelCondition,
    ) -> torch.Tensor:
        fresh = EncoderDecoderCache(DynamicCache(), DynamicCache())
        vt, _ = BaseStepper.decoder_velocity(
            model,
            xt,
            t_scalar,
            model_condition,
            use_cache=False,
            past_key_values=fresh,
        )
        return vt

    @abstractmethod
    def step(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        *,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
        model_condition: ModelCondition,
    ) -> StepperPayload:
        raise NotImplementedError
