#!/usr/bin/env python3
"""Собрать тестовый artifact.pt без импорта acestep: снимок Hydra-cfg + случайный noise + нулевые null по шагам.

Формы тензоров нужно задать вручную (или подогнать под свой прогон). Если не совпадут с реальным
prepare_condition / encoder, generate.py упадёт на проверке формы — это ожидаемо для «глупого» дамми.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifact_bundle import save_generation_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy artifact: только cfg + тензоры, без загрузки модели.")
    parser.add_argument("--out", type=Path, default=ROOT / "_dummy_artifact.pt")
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Переопределить cfg.acestep.config_path (для NTI с guidance>1 — non-turbo, напр. acestep-v15-base)",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--latent-t",
        type=int,
        default=250,
        help="T в noise (B,T,C); типичный порядок для ~10s @ 25Hz латентов, подстрой под свой кейс",
    )
    parser.add_argument(
        "--noise-c",
        type=int,
        default=64,
        help="Каналы шума; в модели это context_latents.shape[-1] // 2",
    )
    parser.add_argument(
        "--null-b",
        type=int,
        default=None,
        help="Batch для null encoder states (по умолчанию = --batch)",
    )
    parser.add_argument("--null-len", type=int, default=512, help="Длина последовательности null embeds")
    parser.add_argument("--null-dim", type=int, default=2048, help="Размерность null embeds (encoder hidden)")
    parser.add_argument(
        "--infer-steps",
        type=int,
        default=None,
        help="Число шагов null-листа; по умолчанию cfg.inference_steps",
    )
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src" / "configs")):
        cfg = compose(config_name="generate")
    OmegaConf.resolve(cfg)

    OmegaConf.set_struct(cfg, False)
    if args.config_path:
        cfg.acestep.config_path = args.config_path
    elif "turbo" in str(cfg.acestep.config_path).lower():
        print(
            "Предупреждение: в cfg turbo; для NTI + guidance>1 лучше сохранить пресет acestep-v15-base в артефакте.",
            file=sys.stderr,
        )
        cfg.acestep.config_path = "acestep-v15-base"

    infer_steps = int(args.infer_steps if args.infer_steps is not None else cfg.inference_steps)
    nb = args.null_b if args.null_b is not None else args.batch

    noise = torch.randn(args.batch, args.latent_t, args.noise_c)
    null_shape = (nb, args.null_len, args.null_dim)
    null_list = [torch.zeros(null_shape) for _ in range(infer_steps)]

    save_generation_artifact(
        args.out,
        cfg,
        noise=noise,
        null_encoder_hidden_states_per_step=null_list,
    )
    print(
        f"OK: {args.out.resolve()}\n"
        f"  inference_steps={infer_steps}, noise {tuple(noise.shape)}, "
        f"null/step {null_shape} × {infer_steps}"
    )


if __name__ == "__main__":
    main()
