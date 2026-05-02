import sys
import threading
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger

from src.attention_injection.controllers.base import AttentionControllerBase

_controller_var: ContextVar[AttentionControllerBase | None] = ContextVar("ac_controller", default=None)
_patch_lock = threading.Lock()


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _instrumented_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    is_cross = bool(getattr(module, "is_cross_attention", False))
    if is_cross:
        ctrl = _controller_var.get()
        if ctrl is not None:
            attn_weights = ctrl.forward(attn_weights)

    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _rebind_eager_in_acestep_modeling_modules() -> int:
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or "acestep" not in name or "modeling" not in name:
            continue
        if not hasattr(mod, "eager_attention_forward"):
            continue
        mod.eager_attention_forward = _instrumented_eager_attention_forward  # type: ignore[assignment]
        n += 1
    return n


def rebind_eager_hook_for_decoder_submodule(obj: object) -> bool:
    mod_name = getattr(obj, "__class__", type(obj)).__module__
    m = sys.modules.get(mod_name)
    if m is not None and hasattr(m, "eager_attention_forward"):
        m.eager_attention_forward = _instrumented_eager_attention_forward  # type: ignore[assignment]
        logger.debug("eager attention hook: re-bound {} (DiT class module)", mod_name)
        return True
    return False


def install_eager_attention_control_patch() -> None:
    with _patch_lock:
        try:
            import transformers.models.qwen3.modeling_qwen3 as m
        except Exception:  # noqa: BLE001
            return
        m.eager_attention_forward = _instrumented_eager_attention_forward  # type: ignore[assignment]
        n_ace = _rebind_eager_in_acestep_modeling_modules()
        if n_ace:
            logger.info(
                "eager attention hook: patched qwen3 + re-bound eager_attention_forward in {} acestep module(s)",
                n_ace,
            )
        else:
            logger.debug(
                "eager attention hook: no acestep modeling modules in sys.modules at patch time (rebind in ForwardPipeline.run if needed)"
            )


def set_runtime_controller(controller: AttentionControllerBase) -> None:
    _controller_var.set(controller)


def clear_runtime_controller() -> None:
    _controller_var.set(None)
