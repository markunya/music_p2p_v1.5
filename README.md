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

`defaults` в [`src/configs/generate.yaml`](src/configs/generate.yaml) ставит **`_self_` последним** и подключает группы `acestep`, `prompt`, **`stepper`**, **`writer`**.

Пути вроде `../ACE-Step-1.5` резолвятся от **исходного** cwd (до смены директории Hydra в `outputs/`).

**Comet:** в `defaults` есть `writer` ([`src/configs/writer/default.yaml`](src/configs/writer/default.yaml)). Имя рана в Comet — ``run_name: ${comet_run_prefix}_${exp_name}``; в конфигах заданы `comet_run_prefix`: **`gen`** ([`generate.yaml`](src/configs/generate.yaml)), **`inv`** ([`invert_music.yaml`](src/configs/invert_music.yaml)), **`edit`** ([`edit_music.yaml`](src/configs/edit_music.yaml)). Без Comet: `writer=null`. Если Comet включён, но `run_name` после резолва не начинается с `gen_`, `edit_` или `inv_`, конфиг отвергается с `ValueError`.

**Каталог эксперимента:** `setup_exp_dir` создаёт `save_dir/exp_name` только если папки ещё нет; иначе `FileExistsError` (новый запуск — новый `exp_name=...`).

`generate.py` / `invert_music.py` / `edit_music.py` логируют в Comet выходные WAV (и входной трек для invert/edit), а также скалярные сводки по латентной траектории (форвард и/или инверсия). Опционально **`log_trajectory_images=true`** только для **инверсии** ([`invert_music.yaml`](src/configs/invert_music.yaml), наследует `edit`): false-color сетка латента (Viridis), **`log_trajectory_max_edge`** — даунскейл. Генерация (`generate.py`) картинки траектории не логирует.

**Генерация из инверсии:** если задан `artifact.path`, `generate.py` подставляет `noise` из артефакта вместо `prepare_noise`. Промпт, `batch_size` и `duration` должны соответствовать инверсии (иначе ошибка уже внутри DiT / декодера).

## Инверсия (Hydra)

Конфиг: [`src/configs/invert_music.yaml`](src/configs/invert_music.yaml) (наследует `generate`; в `defaults` стоит **`stepper@invert_stepper: euler`** — тот же каталог пресетов [`src/configs/stepper/`](src/configs/stepper), второй узел в дереве конфига). В CLI: **`invert_stepper=uni_heun`** для инверсии, **`stepper=...`** для форварда / `generate`. **`InversionPipeline`** берёт **`cfg.invert_stepper`**, при отсутствии — **`cfg.stepper`**.

```bash
python invert_music.py \
  acestep.project_root=/abs/path/to/ACE-Step-1.5 \
  music_path=/abs/path/to/track.wav \
  artifact_out=inverted_artifact.pt
```

- `artifact_out: null` (или CLI `artifact_out=null`) — не писать `.pt` на диск.

## Структура

- `generate.py` — Hydra → `init_dit_handler` → `prepare_conditions` → **`prepare_noise` или шум из `artifact.path`** → **`ForwardPipeline`** → `tiled_decode` → WAV.
- `invert_music.py` — `init_dit_handler` → **`prepare_conditions(..., source_stereo_wav=…)`** → **`InversionPipeline`** (`instantiate(cfg.invert_stepper)`, пресеты из **`stepper/`**) → опционально **`InversionArtifact.save`** (`artifact_out`).
- `edit_music.py` — конфиг [`src/configs/edit_music.yaml`](src/configs/edit_music.yaml): **`defaults: invert_music`** + **`prompt@p2p_task.src` / `prompt@p2p_task.tgt`** (корневой **`prompt`** из `generate` может остаться в резолве, скрипт читает только **`p2p_task`**); **`music_path`** и **`artifact_out`** как у инверсии; инверсия с промптом **src**, затем **`ForwardPipeline`** с **`noise.repeat(2,1,1)`** и **[src, tgt]** → **`sample_0.wav`** / **`sample_1.wav`**.
- `src/inversion/` — **`InversionArtifact`** (`torch.save` dict `version=1`), **`InversionPipeline`** (обратная дискретизация по сетке `t` относительно forward).
- `src/forward/` — **`ForwardPipeline`** (`instantiate(cfg.stepper)`, цикл ODE).
- `src/steppers/` — **Euler** / **Heun** / **`GuidanceStepper`** (обёртка над Euler или Heun); выбор через Hydra-группу `stepper`.
- `src/utils/initialization.py` — инициализация **только** `AceStepHandler` (DiT).
- `src/utils/conditioning.py` — **`prepare_conditions`**: батч промптов → `prepare_condition`; опционально **`source_stereo_wav`** — сетка по длине клипа и **`ModelCondition.clean_latents`** (VAE исходного трека).
- `src/p2p/` — черновики под P2P (см. конфиги `forward/` для edit-сценариев).

**Edit (baseline):**

```bash
python edit_music.py \
  acestep.project_root=/abs/path/to/ACE-Step-1.5 \
  music_path=/abs/path/to/track.wav
```

Пресеты **src** / **tgt** задаются в `edit_music.yaml` (`prompt@p2p_task.*`); при необходимости не писать `.pt` инверсии: `artifact_out=null` (как у `invert_music`).

- `src/configs/` — Hydra: `acestep/`, `prompt/`, **`stepper/`** (и форвард, и инверсия — второй экземпляр через **`stepper@invert_stepper`** в `invert_music`); **`edit_music.yaml`**, **`invert_music.yaml`**.
- `src/schemas.py` — зеркало полей для типизации.
