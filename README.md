# music_p2p_v1.5

Экспериментальный код для **prompt-to-prompt** и **инверсии диффузии** поверх **ACE-Step 1.5**: генерация, инверсия аудио в латентное шумовое состояние и редактирование с парой промптов (src → tgt). Конфигурация через **Hydra**; точки входа — `generate.py`, `invert_music.py`, `edit_music.py`.

Ожидаемая раскладка репозиториев (можно поправить пути в конфиге `acestep`):

```text
diploma/
  ACE-Step-1.5/          # upstream, веса в checkpoints/
  music_p2p_v1.5/        # этот проект
  real_music/            # опционально: *.mp3 для sweep-скриптов (см. ниже)
```

---

## Требования

- Python 3.10+ (рекомендуется виртуальное окружение).
- Клон **ACE-Step 1.5** рядом с этим репо (см. editable install в `requirements.txt`).
- Веса модели в `ACE-Step-1.5/checkpoints` (команда загрузки — в документации upstream, например `acestep-download`).
- Для Apple Silicon при необходимости включён патч MPS в коде (`apply_adg_mps_patch` в точках входа).

---

## Установка

```bash
cd music_p2p_v1.5
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` тянет ACE-Step как editable (`-e ../ACE-Step-1.5`). Если репо лежит в другом месте — поправьте эту строку или установите ACE-Step вручную и укажите `acestep.project_root` при запуске.

---

## Быстрый старт

Все команды выполняются **из корня `music_p2p_v1.5`** (где лежат `generate.py`, `invert_music.py`, `edit_music.py`).

### Генерация

```bash
python generate.py
```

Типичные переопределения:

```bash
python generate.py \
  acestep.config_path=acestep-v15-base \
  acestep.project_root=/abs/path/to/ACE-Step-1.5 \
  inference_steps=50 \
  prompt=example \
  stepper=euler
```

### Инверсия

Конфиг по умолчанию: [`src/configs/invert_music.yaml`](src/configs/invert_music.yaml) (наследует `generate`; второй стэппер монтируется как `invert_stepper`).

```bash
python invert_music.py \
  acestep.project_root=/abs/path/to/ACE-Step-1.5 \
  music_path=/abs/path/to/track.wav \
  artifact_out=inverted_artifact.pt
```

- `artifact_out=null` — не сохранять `.pt` на диск.
- В CLI инверсия: `invert_stepper=...`, форвард при необходимости: `stepper=...`. **`InversionPipeline`** использует `cfg.invert_stepper`, при отсутствии — `cfg.stepper`.

### Редактирование (p2p baseline)

[`src/configs/edit_music.yaml`](src/configs/edit_music.yaml): `defaults: invert_music` и пара промптов `prompt@p2p_task.src` / `prompt@p2p_task.tgt`.

```bash
python edit_music.py \
  acestep.project_root=/abs/path/to/ACE-Step-1.5 \
  music_path=/abs/path/to/track.wav
```

Пайплайн: инверсия с промптом **src**, затем форвард с артефактом инверсии и батчем **[src, tgt]** → выходные `sample_0.wav` / `sample_1.wav` в каталоге эксперимента.

---

## Guidance (CFG / APG / ADG)

Подключение через группы `stepper/guidance_euler` или `stepper/guidance_heun`, либо CLI: `stepper=guidance_euler`.

Параметры — в [`src/configs/stepper/guidance_euler.yaml`](src/configs/stepper/guidance_euler.yaml): `guidance_scale` (при > 1 включается удвоение батча cond/null), `guidance_mode`: `cfg` | `apg` | `adg`, интервал `cfg_t_start` / `cfg_t_end` по шкале времени как в цикле (`1 → 0`). Нулевое условие: `model.null_condition_emb` (как в ACE-Step XL).

Без **`GuidanceStepper`** на один шаг делается один вызов `decoder`; с ним — батч `[cond, null]` и смешивание по выбранному режиму.

---

## Поток данных (кратко)

1. **`init_dit_handler`** — только DiT (без LLM).
2. **`prepare_conditions`** — список промптов → батч после `prepare_condition`; опционально исходный WAV → сетка по длине и **`ModelCondition.clean_latents`**.
3. **`model.prepare_noise`** (или шум из артефакта инверсии).
4. **`ForwardPipeline`**: **`hydra.utils.instantiate(cfg.stepper)`** — см. [`src/configs/stepper/*.yaml`](src/configs/stepper/) (`_target_` → классы вроде `src.steppers.euler.Euler`, `src.steppers.guidance.GuidanceStepper`).
5. **`handler.tiled_decode`** → WAV в каталоге эксперимента.

Расписание времени диффузии совпадает с upstream при **`shift = 1`** (`linspace` от 1 до 0); параметр `shift` в конфиг не выносится.

Пути вида `../ACE-Step-1.5` резолвятся от **исходного** cwd (до того как Hydra переключится в `outputs/`) — см. `resolve_against_original_cwd`.

---

## Конфигурация Hydra

| Файл | Назначение |
|------|------------|
| [`src/configs/generate.yaml`](src/configs/generate.yaml) | Генерация: `acestep`, `prompt`, `stepper`, `writer`, … |
| [`src/configs/invert_music.yaml`](src/configs/invert_music.yaml) | Инверсия: `stepper@invert_stepper`, NTI, `music_path`, артефакт |
| [`src/configs/edit_music.yaml`](src/configs/edit_music.yaml) | Edit: наследует invert + p2p промпты, `save_dir` под p2p |
| [`src/configs/stepper/*.yaml`](src/configs/stepper/) | Пресеты стэпперов (Euler, Heun, guidance, **gci**, …) |

В `defaults` порядок важен: **`_self_` последним** в группах, где это задано.

### Имена экспериментов и CLI

- Каталог прогона: `save_dir/exp_name`. **`setup_exp_dir`** создаёт папку только если её ещё нет; иначе **`FileExistsError`** — для нового прогона задайте другой **`exp_name=...`**.
- В одном аргументе командной строки Hydra допустим только **один** разделитель `ключ=значение`. В **значении** не используйте символ **`=`** (иначе ошибка вида `mismatched input '=' expecting <EOF>`). Для сложных имён используйте подчёркивания или другие разделители (`track_foo_gs_1_5`, не `track=foo`).

---

## Логирование и Comet

- В `defaults` подключается **`writer`** ([`src/configs/writer/default.yaml`](src/configs/writer/default.yaml)). Имя рана: `run_name: ${comet_run_prefix}_${exp_name}`.
- Префиксы: **`gen`** ([`generate.yaml`](src/configs/generate.yaml)), **`inv`** ([`invert_music.yaml`](src/configs/invert_music.yaml)), **`edit`** ([`edit_music.yaml`](src/configs/edit_music.yaml)).
- Отключить логгер внешнего сервиса: `writer=null`.
- Если Comet включён, после резолва `run_name` должен начинаться с `gen_`, `edit_` или `inv_` — иначе конфиг отвергается (`ValueError`).

Скрипты логируют выходные WAV (и входной трек для invert/edit), скаляры по траектории; опционально **`log_trajectory_images=true`** для инверсии/edit — false-color сетка латента (Viridis), **`log_trajectory_max_edge`** ограничивает размер.

Опционально для edit: **`log_edit_latent_recon_mse`** — MSE между чистыми латентами и форвардом по src (loguru + Comet scalar).

---

## Null-text optimization (NTI)

Группа [`src/configs/nti/default.yaml`](src/configs/nti/default.yaml) подключена из `invert_music.yaml`.

- **`nti.enabled: true`** — после инверсии оптимизируются «null»-эмбеддинги под **`cfg.stepper`** (нужен **`GuidanceStepper`** с **`guidance_scale > 1`**). Инверсия по-прежнему через **`invert_stepper`** (часто без guidance).
- Поля: **`nti.lr`**, **`nti.num_inner_steps`**, **`nti.epsilon`**; внешний LR линейно снижается от `lr` к `lr/2`.

**`InversionArtifact`** версии **2** (`torch.save`): опционально **`null_embeddings_per_step`**. Старые артефакты v1 загружаются. **`ForwardPipeline.run(..., inversion_artifact=...)`** использует шум из **`artifact.noise`** и при наличии NTI-поля.

---

## Скрипты серийных запусков (`scripts/`)

Запуск из корня **`music_p2p_v1.5`**:

### [`scripts/run_edit_music_sweeps.py`](scripts/run_edit_music_sweeps.py)

Последовательный sweep вызовов `edit_music.py` по всем `*.mp3`:

- **Группа 1:** форвард `guidance_euler` × несколько `guidance_scale` × инвертеры `uni_euler`, `uni_guidance_euler`, `gci`.
- **Группа 2:** форвард `euler` × инвертеры `euler`, `heun`, `uni_euler`.

Один упавший подпроцесс **не останавливает** остальные; в конце код выхода **1**, если были ошибки.

```bash
python scripts/run_edit_music_sweeps.py
python scripts/run_edit_music_sweeps.py --group 1 --dry-run
```

Аргументы: `--repo-root`, `--real-music-dir`, `--group` (0 | 1 | 2), `--dry-run`.

### [`scripts/run_edit_music_gci_jeps_sweep.py`](scripts/run_edit_music_gci_jeps_sweep.py)

Для каждого трека — GCI (`invert_stepper=gci`), форвард `guidance_euler`, перебор **`invert_stepper.j_eps`** (сетка задаётся константами в начале файла), **`invert_stepper.j_approx=true`**. GUIDANCE-настройки и `inference_steps` синхронизированы с замыслом sweep (см. комментарии в скрипте).

### Папка с треками

По умолчанию скрипты ищут аудио в первой существующей из:

1. `<repo-root>/real_music`
2. `<repo-root>/../real_music` (например `diploma/real_music` при репо внутри `diploma`)

Иначе укажите явно:

```bash
python scripts/run_edit_music_sweeps.py --real-music-dir /path/to/mp3s
```

Имя промпта для Hydra строится из имени файла: stem `gods-plan.mp3` → `detailed/gods_plan`; целевой промпт — `detailed/gender/gods_plan` (наличие yaml в [`src/configs/prompt/`](src/configs/prompt/) — на вашей стороне).

---

## Структура каталогов (исходники)

| Путь | Содержимое |
|------|------------|
| `generate.py`, `invert_music.py`, `edit_music.py` | Точки входа Hydra |
| [`src/forward/`](src/forward/) | **`ForwardPipeline`** — цикл ODE, инстанцирование `cfg.stepper` |
| [`src/inversion/`](src/inversion/) | **`InversionPipeline`**, **`InversionArtifact`**, **`NullTextOptimization`** |
| [`src/steppers/`](src/steppers/) | Euler, Heun, Guidance, **GuidanceContinuationInversionStepper (GCI)** и др. |
| [`src/utils/`](src/utils/) | Инициализация DiT, conditioning, `setup_exp_dir`, пути cwd |
| [`src/configs/`](src/configs/) | Группы Hydra: `acestep/`, `prompt/`, `stepper/`, `writer/`, … |
| [`src/schemas.py`](src/schemas.py) | Поля конфига для типизации |
| [`notebooks/`](notebooks/) | Нотбуки для анализа экспериментов |

Артефакты Hydra по умолчанию попадают в `outputs/`; эксперименты — в `_exps/` согласно `save_dir` в конфигах.

---

## Генерация из сохранённой инверсии

Если в конфиге задан **`artifact.path`**, `generate.py` подставляет **noise** из файла артефакта вместо `prepare_noise`. Промпт, `batch_size` и длительность должны быть согласованы с инверсией.

---

## Полезные ссылки на код

- Стэпперы и guidance: [`src/steppers/guidance.py`](src/steppers/guidance.py)
- Условия и чистые латенты: [`src/utils/conditioning.py`](src/utils/conditioning.py)
- Разрешение путей относительно исходного cwd: [`src/utils/utils.py`](src/utils/utils.py)
