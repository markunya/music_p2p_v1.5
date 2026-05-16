# music_p2p_v1.5

Research codebase for **prompt-to-prompt (P2P) editing** and **diffusion inversion** on top of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). Hydra drives all experiments; entry points are `generate.py`, `invert_music.py`, and `edit_music.py`.

## What this repo does

| Capability | Entry / config | Description |
|------------|----------------|-------------|
| Text-to-audio generation | `generate.py` | Sample from noise (or from a saved inversion artifact). |
| Audio inversion | `invert_music.py` | Encode a track to latent noise / `InversionArtifact`. |
| P2P edit | `edit_music.py` | Invert with **source** prompt, forward with **source + target** prompts. |
| ODE integrators | `src/configs/stepper/` | Euler, Heun, uni variants, CFG guidance, GCI inversion, OptStepper, UniEdit. |
| Attention control | `src/configs/controller/` | Dummy (no-op), token replacement, reweight, PPAE map fusion. |
| Null-text optimization | `src/configs/nti/` | Optional per-step null embedding optimization after inversion. |
| Batch sweeps | `scripts/` | Sequential `edit_music.py` runs over tracks and hyperparameters. |

Outputs land under `_exps/<save_dir>/<exp_name>/` (WAV, resolved config, optional Comet logs). Hydra run logs also go to `outputs/`.

## Configuration documentation

**Every Hydra preset is documented in the header comment block at the top of its YAML file** (before `_target_` or `defaults`). That header explains what the component does and lists all parameters with their meaning. There are no per-field inline comments.

| Directory | Contents |
|-----------|----------|
| [`src/configs/generate.yaml`](src/configs/generate.yaml) | Base generation run |
| [`src/configs/invert_music.yaml`](src/configs/invert_music.yaml) | Inversion (+ inherits generate) |
| [`src/configs/edit_music.yaml`](src/configs/edit_music.yaml) | P2P edit (+ inherits invert) |
| [`src/configs/acestep/`](src/configs/acestep/) | ACE-Step handler paths |
| [`src/configs/stepper/`](src/configs/stepper/) | Forward / invert integrators |
| [`src/configs/controller/`](src/configs/controller/) | Cross-attention injection |
| [`src/configs/nti/`](src/configs/nti/) | Null-text optimization |
| [`src/configs/writer/`](src/configs/writer/) | Comet ML logging |
| [`src/configs/prompt/`](src/configs/prompt/) | Prompt presets (`captions`, `lyrics`, `vocal_language`) |

Override any field from the CLI, e.g. `stepper=guidance_euler controller=ppae_replacement nti.enabled=true`.

**CLI note:** use only one `=` per argument. Do not put `=` inside `exp_name` values (Hydra will fail to parse them).

## Repository layout

```text
parent/
  ACE-Step-1.5/       # upstream (installed separately)
  music_p2p_v1.5/     # this repo
  real_music/         # optional: *.mp3 for sweep scripts
```

Set `acestep.project_root` in config or CLI if ACE-Step is not at `../ACE-Step-1.5` relative to this repo.

## Installation

Use **one Python environment** (3.11–3.12 recommended, matching ACE-Step). Install ACE-Step first, then this repo’s requirements.

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5
git checkout 5bb7daef978ec452e0a6b01b4dbf41a594b7297c

python -m pip install -U pip uv
uv pip install -e .

cd ../
git clone https://github.com/markunya/music_p2p_v1.5.git
cd music_p2p_v1.5
pip install -r requirements.txt
```

Download ACE-Step checkpoints into `ACE-Step-1.5/checkpoints` (see upstream docs, e.g. `acestep-download`).

All commands below are run from **`music_p2p_v1.5`** (where `generate.py` lives).

## Quick start

### Generation

```bash
python generate.py \
  acestep.project_root=../ACE-Step-1.5 \
  prompt=example \
  stepper=euler
```

### Inversion

```bash
python invert_music.py \
  acestep.project_root=../ACE-Step-1.5 \
  music_path=/path/to/track.wav \
  artifact_out=inverted_artifact.pt
```

Use `artifact_out=null` to skip writing the `.pt` file. Inversion uses `invert_stepper` (default in `invert_music.yaml`); forward sampling in other entry points uses `stepper`.

### P2P edit

```bash
python edit_music.py \
  acestep.project_root=../ACE-Step-1.5 \
  music_path=/path/to/track.wav \
  prompt@p2p_task.src=detailed/her_majesty \
  prompt@p2p_task.tgt=detailed/gender/her_majesty \
  controller=ppae_replacement
```

Writes `sample_0.wav` (source reconstruction) and `sample_1.wav` (target branch). See [`src/configs/edit_music.yaml`](src/configs/edit_music.yaml) for defaults.

## Guidance and inversion steppers

- Forward CFG: `stepper=guidance_euler` or `stepper=guidance_heun` — see [`src/configs/stepper/guidance_euler.yaml`](src/configs/stepper/guidance_euler.yaml).
- Invert with GCI: `invert_stepper=gci` — see [`src/configs/stepper/gci.yaml`](src/configs/stepper/gci.yaml).
- NTI after inversion: `nti.enabled=true` with a CFG forward stepper — see [`src/configs/nti/default.yaml`](src/configs/nti/default.yaml).

## Logging

- Default: Comet ML via [`src/configs/writer/default.yaml`](src/configs/writer/default.yaml). Disable with `writer=null`.
- Run names must resolve to `gen_*`, `inv_*`, or `edit_*` (from `comet_run_prefix` in each entry config).
- Each experiment directory is created once; reuse requires a new `exp_name=...`.

## Sweep scripts

From repo root:

```bash
python scripts/run_edit_music_sweeps.py
python scripts/run_edit_music_nti_sweep.py
python scripts/run_edit_music_gci_jeps_sweep.py
```

Audio is read from `<repo>/real_music` or `<repo>/../real_music`, or pass `--real-music-dir`. Use `--dry-run` to print commands only.

## Source layout

| Path | Role |
|------|------|
| [`src/forward/pipeline.py`](src/forward/pipeline.py) | Forward ODE loop + optional attention controllers |
| [`src/inversion/`](src/inversion/) | Inversion pipeline, artifacts, NTI |
| [`src/steppers/`](src/steppers/) | Integrator implementations |
| [`src/attention_injection/`](src/attention_injection/) | Eager attention hook + controllers |
| [`src/utils/`](src/utils/) | DiT init, conditioning, experiment dirs |
| [`scripts/`](scripts/) | Batch experiment runners |
