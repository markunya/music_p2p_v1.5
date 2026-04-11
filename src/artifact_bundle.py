"""Артефакт генерации: снимок ``cfg`` + опциональные тензоры диффузии."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.utils import resolve_against_original_cwd


@dataclass
class GenerationArtifactPayload:
    """Данные из ``torch.save`` bundle (не путать с отсутствием файла артефакта)."""

    noise: Optional[torch.Tensor] = None
    null_encoder_hidden_states_per_step: Optional[List[torch.Tensor]] = None

    def uses_mlx_incompatible_override(self) -> bool:
        return self.noise is not None or self.null_encoder_hidden_states_per_step is not None


def save_generation_artifact(
    path: str | Path,
    cfg: DictConfig,
    *,
    noise: torch.Tensor | None = None,
    null_encoder_hidden_states_per_step: list[torch.Tensor] | torch.Tensor | None = None,
) -> None:
    """Сохранить ``cfg`` (plain dict без ``hydra``) и опциональные тензоры."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(container, dict) and "hydra" in container:
        container = {k: v for k, v in container.items() if k != "hydra"}
    bundle: dict[str, Any] = {"cfg": container}
    if noise is not None:
        bundle["noise"] = noise.detach().cpu()
    if null_encoder_hidden_states_per_step is not None:
        if isinstance(null_encoder_hidden_states_per_step, torch.Tensor):
            bundle["null_encoder_hidden_states_per_step"] = null_encoder_hidden_states_per_step.detach().cpu()
        else:
            bundle["null_encoder_hidden_states_per_step"] = [
                t.detach().cpu() for t in null_encoder_hidden_states_per_step
            ]
    torch.save(bundle, path)


def _optional_noise(data: dict[str, Any]) -> torch.Tensor | None:
    if "noise" not in data:
        return None
    t = data["noise"]
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"artifact['noise'] must be torch.Tensor, got {type(t)}")
    return t


def _legacy_initial_latents_as_noise(data: dict[str, Any]) -> torch.Tensor | None:
    """Старые артефакты с ключом ``initial_latents`` — трактуем как ``noise``."""
    if "initial_latents" not in data:
        return None
    t = data["initial_latents"]
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"artifact['initial_latents'] must be torch.Tensor, got {type(t)}")
    warnings.warn(
        "Ключ артефакта 'initial_latents' устарел; сохраняйте тот же тензор под ключом 'noise'.",
        DeprecationWarning,
        stacklevel=3,
    )
    return t


def _normalize_null_per_step(raw: Any) -> list[torch.Tensor] | None:
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        if raw.dim() == 0:
            raise ValueError("null_encoder_hidden_states_per_step tensor must have leading step dimension")
        return [raw[i].contiguous() for i in range(raw.shape[0])]
    if isinstance(raw, list):
        out: list[torch.Tensor] = []
        for i, t in enumerate(raw):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"null_encoder_hidden_states_per_step[{i}] must be torch.Tensor, got {type(t)}")
            out.append(t)
        return out
    raise TypeError(
        f"null_encoder_hidden_states_per_step must be list[Tensor] or stacked Tensor, got {type(raw)}"
    )


def _optional_null_per_step(data: dict[str, Any]) -> list[torch.Tensor] | None:
    if "null_encoder_hidden_states_per_step" not in data:
        return None
    return _normalize_null_per_step(data["null_encoder_hidden_states_per_step"])


def load_generation_bundle(cli_cfg: DictConfig) -> tuple[DictConfig, GenerationArtifactPayload | None]:
    """Без ``artifact.path`` — исходный конфиг и ``None``. Иначе — ``cfg`` из файла и payload."""
    raw_path = OmegaConf.select(cli_cfg, "artifact.path")
    if raw_path is None or str(raw_path).strip() == "":
        return cli_cfg, None

    path = resolve_against_original_cwd(str(raw_path))
    bundle = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(bundle, dict):
        raise TypeError(f"Artifact must be a dict, got {type(bundle)}")
    if "cfg" not in bundle:
        raise KeyError(
            "Artifact must contain key 'cfg' — снимок конфига (dict), тот же вид, что у Hydra generate."
        )

    work_cfg = OmegaConf.create(bundle["cfg"])
    noise = _optional_noise(bundle)
    if noise is None:
        noise = _legacy_initial_latents_as_noise(bundle)

    payload = GenerationArtifactPayload(
        noise=noise,
        null_encoder_hidden_states_per_step=_optional_null_per_step(bundle),
    )
    return work_cfg, payload
