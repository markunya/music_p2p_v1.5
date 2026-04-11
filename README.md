# music_p2p_v1.5

Эксперименты с prompt-to-prompt / инверсией на **ACE-Step 1.5**. Репозиторий рядом с клоном [`ACE-Step-1.5`](../ACE-Step-1.5) (или поправьте `acestep.project_root` в конфиге).

## Установка

```bash
cd music_p2p_v1.5
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Скачайте веса ACE-Step 1.5 в каталог `ACE-Step-1.5/checkpoints` (см. документацию upstream: `acestep-download`).

## Генерация (Hydra)

Из корня **этого** репозитория:

```bash
python generate.py
```

Переопределения:

```bash
python generate.py acestep.config_path=acestep-v15-base acestep.project_root=/abs/path/to/ACE-Step-1.5 inference_steps=50 prompt=example
```

`defaults` в `src/configs/generate.yaml` ставит **`_self_` последним**, чтобы локальные поля не перетирались группами `acestep` / `prompt`.

Пути вроде `../ACE-Step-1.5` резолвятся от **исходного** cwd (до смены директории Hydra в `outputs/`).

### Артефакт (тот же сценарий, что и обычная генерация)

Один код-путь в `generate.py`: из **рабочего cfg** собираются kwargs для `generate_music`. Если в Hydra задан `artifact.path`, рабочий cfg берётся **целиком из файла** (`bundle["cfg"]`), иначе — это текущий merged Hydra-конфиг.

Файл артефакта — dict, сохранённый через `torch.save`:

| Ключ | Назначение |
|------|------------|
| `cfg` | обязательный снимок конфига (как у Hydra `generate`) |
| `noise` | опционально: фиксированный тензор для `prepare_noise` (та же форма, что у DiT-шума) |
| `null_encoder_hidden_states_per_step` | опционально: список тензоров **по одному на шаг** диффузии (CFG uncond в стиле null-text inversion) **или** один stacked tensor с ведущей осью `T == inference_steps` |
| `initial_latents` | опционально: стартовый `xt` вместо `noise` (не сочетать с `cover_noise_strength > 0` в cfg) |

Режимы с `noise` / null-text / `initial_latents` требуют **PyTorch DiT** (`acestep.use_mlx_dit: false`). Реализация null-text и `initial_latents` — в [`src/acestep_artifact_diffusion.py`](src/acestep_artifact_diffusion.py) (vendored-логика из upstream `generate_audio`); при обновлении пакета `acestep` этот файл нужно сверять с `modeling_acestep_v15_base.py`. Кастомные `timesteps` в этом пути не поддерживаются (используйте `inference_steps` и `shift` из cfg). Для **XL/SFT** с отличающимся `generate_audio` может понадобиться отдельная копия цикла.

Сохранение снимка конфига (например из инверсии или после настройки в Hydra):

```python
from omegaconf import DictConfig
from src.artifact_bundle import save_generation_artifact

# cfg — тот же DictConfig, что в @hydra.main (save_generation_artifact уберёт узел hydra)
save_generation_artifact(
    "out.pt",
    cfg,
    noise=noise_tensor,  # опционально
    null_encoder_hidden_states_per_step=null_list_or_stacked,  # опционально
    initial_latents=xt0,  # опционально
)
```

Запуск только с указанием пути (остальное внутри `cfg` в файле):

```bash
python generate.py artifact.path=/path/to/out.pt
```

## Структура

- `generate.py` — загрузка bundle → `initialize_service` → `run_generate` (один путь).
- `src/run_generate.py` — `generate_music_kwargs_from_cfg` + `run_generate` (патчи `prepare_noise` / `generate_audio`).
- `src/artifact_bundle.py` — `load_generation_bundle` / `save_generation_artifact` / `GenerationArtifactPayload`.
- `src/acestep_artifact_diffusion.py` — диффузия с NTI и `initial_latents` без правок установленного `acestep`.
- `src/configs/` — Hydra и группы `acestep/`, `prompt/`.
- `src/schemas.py` — зеркало полей для типизации.
