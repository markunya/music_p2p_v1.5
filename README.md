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

Переопределения (путь к ACE-Step, число шагов, пресет промпта, стэппер):

```bash
python generate.py acestep.config_path=acestep-v15-base acestep.project_root=/abs/path/to/ACE-Step-1.5 inference_steps=50 prompt=example stepper=euler
```

**Guidance (CFG / APG / ADG):** подключите группу `stepper/guidance_euler` или `stepper/guidance_heun` (или через CLI: `stepper=guidance_euler`). Параметры в [`src/configs/stepper/guidance_euler.yaml`](src/configs/stepper/guidance_euler.yaml): `guidance_scale` (> 1 включает удвоение батча), `guidance_mode` — `cfg` | `apg` | `adg`, интервал `cfg_t_start` / `cfg_t_end` по шкале времени как в цикле (`1 → 0`). Нулевое условие берётся с весов модели: `model.null_condition_emb` (как в ACE-Step XL).

Поток: **`src/utils/initialization.init_dit_handler`** (только DiT, без LLM) → **`prepare_conditions`** (список промптов → батч после `prepare_condition`) → **`model.prepare_noise`** → **`ForwardPipeline`**, внутри которого **`hydra.utils.instantiate(cfg.stepper)`** (см. `src/configs/stepper/*.yaml`, поле `_target_` на класс, например `src.steppers.euler.Euler` или `src.steppers.guidance.GuidanceStepper`) → **`handler.tiled_decode`** → файлы `sample_0.wav`, … в каталоге эксперимента.

Расписание диффузии по времени соответствует upstream при **`shift = 1`** (только `linspace` от 1 до 0); параметр `shift` в конфиг **не выносится**.

Без **`GuidanceStepper`** на шаг делается один вызов ``decoder`` на оценку скорости; с ним — удвоенный батч `[cond, null]` и смешивание предиктов по выбранному `guidance_mode`.

`defaults` в [`src/configs/generate.yaml`](src/configs/generate.yaml) ставит **`_self_` последним** и подключает группы `acestep`, `prompt`, **`stepper`**.

Пути вроде `../ACE-Step-1.5` резолвятся от **исходного** cwd (до смены директории Hydra в `outputs/`).

**Генерация из инверсии:** если задан `artifact.path`, `generate.py` подставляет `noise` из артефакта вместо `prepare_noise`. Промпт, `batch_size` и `duration` должны соответствовать инверсии (иначе ошибка уже внутри DiT / декодера).

## Инверсия (Hydra)

Конфиг: [`src/configs/invert_music.yaml`](src/configs/invert_music.yaml) (наследует `generate` — `seed`, `inference_steps`, `duration` и т.д. не дублируйте; группу **`stepper`** в `defaults` повторно не подключайте — иначе Hydra: «stepper appears more than once»; другой степпер только через CLI, например `stepper=heun` или `stepper=uni_heun`).

```bash
python invert_music.py \
  acestep.project_root=/abs/path/to/ACE-Step-1.5 \
  music_path=/abs/path/to/track.wav \
  artifact_out=inverted_artifact.pt
```

- `artifact_out: null` (или CLI `artifact_out=null`) — не писать `.pt` на диск.

## Структура

- `generate.py` — Hydra → `init_dit_handler` → `prepare_conditions` → **`prepare_noise` или шум из `artifact.path`** → **`ForwardPipeline`** → `tiled_decode` → WAV.
- `invert_music.py` — `init_dit_handler` → **`prepare_conditions(..., source_stereo_wav=…)`** → **`InversionPipeline`** (``clean_latents`` из поля ``ModelCondition``) → опционально **`InversionArtifact.save`** (`artifact_out`).
- `edit_music.py` — **`ForwardPipeline(cli_cfg).run`** с узлом Hydra **`forward`** (по умолчанию в `p2p_edit` — [`forward/edit_unified.yaml`](src/configs/forward/edit_unified.yaml) → ``UnifiedEditForwardRunner``); если задан только `source_audio_path`, сначала **`InversionPipeline(inv_cfg).run`**, затем редактирование с полученным шумом.
- `src/inversion/` — **`InversionArtifact`** (`torch.save` dict `version=1`), **`InversionPipeline`** (обратная дискретизация по сетке `t` относительно forward).
- `src/forward/` — **`ForwardPipeline`** (`instantiate(cfg.stepper)`, цикл ODE).
- `src/steppers/` — **Euler** / **Heun** / **`GuidanceStepper`** (обёртка над Euler или Heun); выбор через Hydra-группу `stepper`.
- `src/utils/initialization.py` — инициализация **только** `AceStepHandler` (DiT).
- `src/utils/conditioning.py` — **`prepare_conditions`**: батч промптов → `prepare_condition`; опционально **`source_stereo_wav`** — сетка по длине клипа и **`ModelCondition.clean_latents`** (VAE исходного трека).
- `src/p2p/` — черновики под P2P (см. конфиги `forward/` для edit-сценариев).

Смена стратегии edit (например delayed injection вместо UniEdit):

```bash
python edit_music.py source_audio_path=../real_music/stay.mp3 forward=edit_delayed forward.alpha=0.6
```

- `src/configs/` — Hydra: `acestep/`, `prompt/`, **`stepper/`** для `generate.py`, а также `forward/` для черновиков edit.
- `src/schemas.py` — зеркало полей для типизации.
