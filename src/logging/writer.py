"""Comet ML (optional) и заглушка — по образцу ``music_p2p``."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.logging import utils as logging


class ExperimentMode(Enum):
    Online = auto()
    Offline = auto()


@dataclass
class CometMLConfig:
    project_name: str
    workspace: str | None
    run_name: str | None
    mode: ExperimentMode


class BaseWriter(ABC):
    @abstractmethod
    def set_step(self, step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_scalar(self, scalar_name: str, scalar: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_scalars(self, scalars: dict[str, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_audio(self, audio_name: str, audio: Any, sample_rate: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_image(self, image_name: str, image: Any) -> None:
        raise NotImplementedError

    def end(self) -> None:
        """Завершить эксперимент (Comet: ``experiment.end()``)."""
        pass


class DummyWriter(BaseWriter):
    def set_step(self, step: int) -> None:
        pass

    def add_scalar(self, scalar_name: str, scalar: float) -> None:
        pass

    def add_scalars(self, scalars: dict[str, float]) -> None:
        pass

    def add_audio(self, audio_name: str, audio: Any, sample_rate: int | None = None) -> None:
        pass

    def add_image(self, image_name: str, image: Any) -> None:
        pass

    def end(self) -> None:
        pass


class CometMLWriter(BaseWriter):
    """Трекинг через Comet ML (``pip install comet_ml``, переменные окружения / ``comet_ml.login()``)."""

    def __init__(self, project_config: DictConfig):
        import comet_ml

        comet_ml.login()

        w = OmegaConf.select(project_config, "writer")
        if w is None:
            raise ValueError("CometMLWriter: cfg.writer is missing")
        w_d = OmegaConf.to_container(w, resolve=True)
        if not isinstance(w_d, dict):
            raise TypeError("cfg.writer must resolve to a dict")
        mode_raw = w_d.get("mode", "Online")
        if isinstance(mode_raw, str):
            mode = ExperimentMode[mode_raw]
        else:
            mode = mode_raw

        match mode:
            case ExperimentMode.Offline:
                exp_class = comet_ml.OfflineExperiment
            case ExperimentMode.Online:
                exp_class = comet_ml.Experiment
            case _:
                raise ValueError(f"Invalid writer.mode: {mode}")

        self.exp = exp_class(
            project_name=str(w_d["project_name"]),
            workspace=w_d.get("workspace"),
            experiment_key=None,
            log_code=False,
            log_graph=False,
            auto_metric_logging=False,
            auto_param_logging=False,
        )
        run_name = w_d.get("run_name")
        if run_name:
            self.exp.set_name(str(run_name))

        try:
            params = OmegaConf.to_container(project_config, resolve=True)
            if isinstance(params, dict):
                self.exp.log_parameters(parameters=_flatten_config_for_comet(params))
        except Exception as exc:
            logging.info(f"Comet log_parameters пропущен: {exc}")

        self.step = 0
        self.timer = datetime.now()

        try:
            self.exp.log_metric("nti/invert_started", 1.0, step=0)
        except Exception:
            pass
        url = getattr(self.exp, "url", None)
        if url:
            logging.info(f"Comet: {url}")

    def set_step(self, step: int) -> None:
        previous_step = self.step
        self.step = step
        try:
            self.exp.set_step(step)
        except Exception:
            pass
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", (self.step - previous_step) / max(duration.total_seconds(), 1e-9))
            self.timer = datetime.now()

    def add_scalar(self, scalar_name: str, scalar: float) -> None:
        v = float(scalar)
        try:
            self.exp.log_metric(scalar_name, v, step=self.step)
        except TypeError:
            self.exp.log_metrics({scalar_name: v}, step=self.step)

    def add_scalars(self, scalars: dict[str, float]) -> None:
        for name, val in scalars.items():
            self.add_scalar(name, float(val))

    def add_image(self, image_name: str, image: Any) -> None:
        self.exp.log_image(image_data=image, name=image_name, step=self.step)

    def add_audio(self, audio_name: str, audio: Any, sample_rate: int | None = 48_000) -> None:
        import numpy as np

        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        if isinstance(audio, np.ndarray):
            audio = audio.T
        self.exp.log_audio(file_name=audio_name, audio_data=audio, sample_rate=int(sample_rate or 48_000))

    def end(self) -> None:
        try:
            flush = getattr(self.exp, "flush", None)
            if callable(flush):
                flush()
            self.exp.end()
        except Exception as exc:
            logging.info(f"Comet experiment.end() failed: {exc}")


def _flatten_config_for_comet(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Плоский dict строковых значений для ``log_parameters``."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "hydra":
                continue
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_config_for_comet(v, key))
    elif isinstance(obj, (list, tuple)):
        out[prefix or "list"] = str(obj)[:2000]
    else:
        try:
            out[prefix] = obj if isinstance(obj, (int, float, bool, str)) else str(obj)[:500]
        except Exception:
            out[prefix] = "<unserializable>"
    return out


def _dummy_writer(reason: str) -> DummyWriter:
    logging.info(f"Writer: DummyWriter — {reason}")
    return DummyWriter()


def setup_writer(cfg: DictConfig) -> BaseWriter:
    """По ``cfg.writer``: ``null`` / отсутствует → ``DummyWriter``; иначе ``CometMLWriter``."""
    if isinstance(cfg, DictConfig) and OmegaConf.is_missing(cfg, "writer"):
        return _dummy_writer("в конфиге нет ключа writer")
    w = OmegaConf.select(cfg, "writer")
    if w is None:
        return _dummy_writer("cfg.writer == None (например оверрайд writer=null)")
    d = OmegaConf.to_container(w, resolve=True)
    if d is None or d == {}:
        return _dummy_writer("writer после resolve пустой")
    if not isinstance(d, dict):
        return _dummy_writer(f"writer не dict, а {type(d).__name__}")
    if not d.get("project_name"):
        return _dummy_writer("writer.project_name пуст — задайте имя проекта Comet")
    try:
        import comet_ml as _comet_check  # noqa: F401
    except ImportError as exc:
        logging.info(
            f"comet_ml не импортируется для этого интерпретатора:\n  {sys.executable}\n"
            f"Причина: {exc!r}\n"
            "Установи пакет тем же Python (не голый «pip», если он от другой версии): "
            f'"{sys.executable}" -m pip install comet_ml\n'
            "Если «pip install» пишет путь …/Python/3.xx/site-packages, а «python» другой 3.yy — "
            "как раз эта ситуация: только «python -m pip» привязывает установку к нужному интерпретатору."
        )
        return _dummy_writer(f"импорт comet_ml: {exc}")
    try:
        writer = CometMLWriter(cfg)
    except Exception as exc:
        logging.info(
            f"Comet writer init failed (python={sys.executable}): {exc!r}"
        )
        return _dummy_writer(f"ошибка инициализации Comet: {exc}")
    logging.info("Writer: CometMLWriter (эксперимент создан)")
    return writer
