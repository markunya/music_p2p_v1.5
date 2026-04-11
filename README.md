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

## Структура

- `generate.py` — точка входа, `AceStepHandler.initialize_service` + `generate_music` (text2music, без LM / audio codes).
- `src/configs/` — Hydra-конфиги и группы `acestep/`, `prompt/`.
- `src/schemas.py` — зеркало полей конфига для будущего `ConfigStore` / типизации (сейчас Hydra читает только YAML).
