"""Артефакт генерации: снимок ``cfg`` + опциональные тензоры диффузии."""

from __future__ import annotations

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


def _hydra_task_overrides_touch_prefix(prefix: str) -> bool:
    """True if the Hydra CLI included a task override for this config prefix (e.g. ``prompt`` or ``prompt.captions``)."""
    try:
        from hydra.core.hydra_config import HydraConfig

        ovr = HydraConfig.get().overrides.task
    except (ImportError, ValueError, AttributeError):
        return False
    if ovr is None:
        return False
    starters = (
        f"{prefix}=",
        f"{prefix}.",
        f"+{prefix}=",
        f"++{prefix}=",
        f"~{prefix}",
    )
    for item in ovr:
        s = str(item).strip()
        if any(s.startswith(st) for st in starters):
            return True
    return False


def _apply_cli_task_overrides_for_artifact_work_cfg(cli_cfg: DictConfig, work_cfg: DictConfig) -> None:
    """Подставить в ``work_cfg`` поля из ``cli_cfg``, если для них были Hydra-оверрайды в командной строке."""
    if _hydra_task_overrides_touch_prefix("prompt"):
        work_cfg.prompt = OmegaConf.create(OmegaConf.to_container(cli_cfg.prompt, resolve=True))
    if _hydra_task_overrides_touch_prefix("vocal_language"):
        work_cfg.vocal_language = cli_cfg.vocal_language
    if _hydra_task_overrides_touch_prefix("guidance_scale"):
        work_cfg.guidance_scale = cli_cfg.guidance_scale


def load_generation_bundle(cli_cfg: DictConfig) -> tuple[DictConfig, GenerationArtifactPayload | None]:
    """Без ``artifact.path`` — исходный конфиг и ``None``. Иначе — ``cfg`` из файла и payload.

    Для артефакта ``cfg`` берётся из файла; при оверрайдах в CLI подставляются из ``cli_cfg`` (см.
    ``_apply_cli_task_overrides_for_artifact_work_cfg``): ``prompt`` / ``prompt.*``, ``vocal_language``,
    ``guidance_scale``.
    """
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

    payload = GenerationArtifactPayload(
        noise=noise,
        null_encoder_hidden_states_per_step=_optional_null_per_step(bundle),
    )
    _apply_cli_task_overrides_for_artifact_work_cfg(cli_cfg, work_cfg)
    return work_cfg, payload
