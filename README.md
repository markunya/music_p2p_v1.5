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
| `noise` | опционально: фиксированный латент/шум для `prepare_noise` — с него начинается диффузия (форма как у выхода DiT `prepare_noise`) |
| `null_encoder_hidden_states_per_step` | опционально: список тензоров **по одному на шаг** (CFG uncond, null-text inversion) **или** stacked tensor с ведущей осью `T == inference_steps` |

Режимы с `noise` или null-text требуют **PyTorch DiT** (`acestep.use_mlx_dit: false`). Диффузия с фиксированным шумом и per-step null из артефакта идёт через **`ForwardPipeline`** → [`src/forward/plain_forward.py`](src/forward/plain_forward.py) + [`PlainCfgEulerStepper`](src/forward/steppers/plain_cfg_stepper.py) + [`diffusion_driver`](src/forward/diffusion_driver.py) (CFG при `guidance_scale > 1`). При обновлении ACE-Step имеет смысл сверять поведение с upstream `generate_audio` в `acestep/models/base/modeling_acestep_v15_base.py`. Кастомные `timesteps` не поддерживаются.

Сохранение снимка конфига (например из инверсии или после настройки в Hydra):

```python
from omegaconf import DictConfig
from src.artifact_bundle import save_generation_artifact

# cfg — тот же DictConfig, что в @hydra.main (save_generation_artifact уберёт узел hydra)
save_generation_artifact(
    "out.pt",
    cfg,
    noise=noise_tensor,  # опционально: старт диффузии
    null_encoder_hidden_states_per_step=null_list_or_stacked,  # опционально
)
```

Запуск только с указанием пути (остальное внутри `cfg` в файле):

```bash
python generate.py artifact.path=/path/to/out.pt
```

## Структура

- `generate.py` — `forward_artifact_and_work_cfg` → `init_acestep_handler` → **`ForwardPipeline(work_cfg).run(...)`** → WAV.
- `invert_music.py` — **`InversionPipeline(cfg)`** (через `run_invert`): pivot + опциональный NTI (`nti.enabled` в конфиге); **`artifact_out=null`** — только в памяти, без `.pt`.
- `edit_music.py` — **`ForwardPipeline(cli_cfg).run`** с `forward.mode=velocity_fusion`; если задан только `source_audio_path`, сначала **`InversionPipeline(inv_cfg).run`**, затем редактирование с полученным шумом.
- `src/inversion/` — **`InversionArtifact`** (dataclass), **`InversionPipeline`**, `run_invert` (сохранение артефакта только при не-null `artifact_out`).
- `src/forward/` — **`ForwardPipeline`**, **`run_plain_forward`** (ODE + ``PlainCfgEulerStepper`` + ``diffusion_driver`` для CFG и per-step null из артефакта), **`VelocityFusionEditRunner`** (edit + stepper UniEdit + ``diffusion_driver``), `forward_artifact_and_work_cfg`.
- `src/p2p/prompts.py` — **`P2PPromptPair`** из узла ``p2p_task``; раннер — ``VelocityFusionEditRunner`` ([`p2p_strategy/velocity_fusion.yaml`](src/configs/p2p_strategy/velocity_fusion.yaml): ``fusion_mode`` = ``time`` | ``spectral_time`` | ``spectral_spectral``; STFT **вдоль оси токена латента L**, не по WAV).

Пример абляции edit:

```bash
python edit_music.py source_audio_path=../real_music/the-beatles-her-majesty.mp3 \
  p2p_strategy.fusion_mode=spectral_time p2p_strategy.stft_n_fft=512
```
- `src/runtime/cli_bootstrap.py` — общий bootstrap handler / WAV / пути инверсии.
- `src/artifact_bundle.py` — `load_generation_bundle` / `save_generation_artifact` / `GenerationArtifactPayload`.
- `src/configs/` — Hydra и группы `acestep/`, `prompt/`; `forward.mode` в `generate.yaml`; блок **`nti:`** только в [`invert_nti.yaml`](src/configs/invert_nti.yaml) (подключается из `invert_music` / `p2p_edit*`, не из чистого generate).
- `src/schemas.py` — зеркало полей для типизации.
