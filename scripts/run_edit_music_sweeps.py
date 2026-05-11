#!/usr/bin/env python3
"""
Sequential sweeps for ``edit_music.py`` (run from repo root ``music_p2p_v1.5``).

Group 1: guidance forward + invert variants × tracks × guidance_scale.
Group 2: euler forward + invert variants × tracks.

A failed run does not stop the sweep; exit code 1 means at least one run failed (see stderr summary).

Track id for Hydra prompts: ``*.mp3`` stem with ``-`` replaced by ``_``
(e.g. ``gods-plan.mp3`` → ``detailed/gods_plan``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def continuation_steps_for_gci(gs: float) -> int:
    """Match (GS - 1) / 0.1 as integer step count (1.5→5, 2.0→10, 3.0→20)."""
    return max(1, int(round((gs - 1.0) / 0.1)))


def track_prompt_id(mp3: Path) -> str:
    return mp3.stem.replace("-", "_")


def _hydra_music_dir_rel(repo_root: Path, music_dir: Path) -> str:
    rr = repo_root.resolve()
    d = music_dir.resolve()
    try:
        return d.relative_to(rr).as_posix()
    except ValueError:
        return Path(os.path.relpath(d, rr)).as_posix()


def resolve_real_music_dir(
    repo_root: Path, override: Path | None
) -> tuple[Path, str] | None:
    """(absolute dir with *.mp3, Hydra music_path directory relative to repo_root)."""
    rr = repo_root.resolve()
    if override is not None:
        d = override.expanduser().resolve()
        if not d.is_dir():
            return None
        return d, _hydra_music_dir_rel(rr, d)

    in_repo = rr / "real_music"
    sibling = rr.parent / "real_music"
    if in_repo.is_dir():
        return in_repo, _hydra_music_dir_rel(rr, in_repo)
    if sibling.is_dir():
        return sibling, _hydra_music_dir_rel(rr, sibling)
    return None


def run_edit_music(
    repo_root: Path,
    args: list[str],
    *,
    dry_run: bool,
) -> int:
    cmd = [sys.executable, str(repo_root / "edit_music.py"), *args]
    line = subprocess.list2cmdline(cmd)
    print(line, flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=repo_root)
    return proc.returncode


def group_1(
    repo_root: Path, mp3s: list[Path], music_dir_rel: str, *, dry_run: bool
) -> list[str]:
    guidance_scales = (1.5, 2.0, 3.0)
    invert_methods = ("uni_euler", "uni_guidance_euler", "gci")
    failed: list[str] = []

    for mp3 in mp3s:
        tid = track_prompt_id(mp3)
        music_arg = f"{music_dir_rel}/{mp3.name}"

        for gs in guidance_scales:
            for inv in invert_methods:
                hydra: list[str] = [
                    f"stepper@invert_stepper={inv}",
                    "stepper=guidance_euler",
                    f"prompt@p2p_task.src=detailed/{tid}",
                    f"prompt@p2p_task.tgt=detailed/gender/{tid}",
                    f"exp_name=___GUIDANCE_INVERSION___track_{tid}___gs_{gs}___method_{inv}",
                    f"music_path={music_arg}",
                    "log_local_error=true",
                    "inference_steps=50",
                    f"stepper.guidance_scale={gs}",
                    "log_trajectory_images=true",
                ]
                if inv == "gci":
                    hydra.append(
                        f"invert_stepper.continuation_steps={continuation_steps_for_gci(gs)}"
                    )
                if inv in ("uni_guidance_euler", "gci"):
                    hydra.append(f"invert_stepper.guidance_scale={gs}")

                rc = run_edit_music(repo_root, hydra, dry_run=dry_run)
                if rc != 0:
                    msg = f"[group1] rc={rc} track={tid} gs={gs} inv={inv}"
                    print(msg, file=sys.stderr)
                    failed.append(msg)
    return failed


def group_2(
    repo_root: Path, mp3s: list[Path], music_dir_rel: str, *, dry_run: bool
) -> list[str]:
    invert_methods = ("euler", "heun", "uni_euler")
    failed: list[str] = []

    for mp3 in mp3s:
        tid = track_prompt_id(mp3)
        music_arg = f"{music_dir_rel}/{mp3.name}"

        for inv in invert_methods:
            hydra: list[str] = [
                f"stepper@invert_stepper={inv}",
                "stepper=euler",
                f"prompt@p2p_task.src=detailed/{tid}",
                f"prompt@p2p_task.tgt=detailed/gender/{tid}",
                f"exp_name=INVERSION___track_{tid}___method_{inv}",
                f"music_path={music_arg}",
                "log_local_error=true",
                "inference_steps=50",
                "log_trajectory_images=true",
            ]
            rc = run_edit_music(repo_root, hydra, dry_run=dry_run)
            if rc != 0:
                msg = f"[group2] rc={rc} track={tid} inv={inv}"
                print(msg, file=sys.stderr)
                failed.append(msg)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run edit_music.py sweep groups sequentially.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to music_p2p_v1.5 (default: parent of scripts/).",
    )
    parser.add_argument(
        "--real-music-dir",
        type=Path,
        default=None,
        help="Folder with *.mp3. Default: first existing among <repo>/real_music, <repo>/../real_music.",
    )
    parser.add_argument(
        "--group",
        type=int,
        choices=(1, 2, 0),
        default=0,
        help="Run only group 1, only group 2, or 0 = both in order (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = (args.repo_root or script_dir.parent).resolve()
    resolved = resolve_real_music_dir(repo_root, args.real_music_dir)

    if not (repo_root / "edit_music.py").is_file():
        print(f"edit_music.py not found under {repo_root}", file=sys.stderr)
        return 2
    if resolved is None:
        if args.real_music_dir is not None:
            print(f"--real-music-dir is not a directory: {args.real_music_dir}", file=sys.stderr)
        else:
            a, b = repo_root / "real_music", repo_root.parent / "real_music"
            print("No folder with tracks found. Tried:", file=sys.stderr)
            print(f"  {a}", file=sys.stderr)
            print(f"  {b}", file=sys.stderr)
            print("Put *.mp3 in one of these or pass: --real-music-dir /path/to/mp3s", file=sys.stderr)
        return 2

    real_music, music_dir_rel = resolved

    mp3s = sorted(real_music.glob("*.mp3"))
    if not mp3s:
        print(f"No *.mp3 in {real_music}", file=sys.stderr)
        return 2

    print(f"repo_root={repo_root}", flush=True)
    print(f"real_music={real_music} (music_path prefix: {music_dir_rel}/)", flush=True)
    print(f"tracks ({len(mp3s)}): {[p.name for p in mp3s]}", flush=True)

    all_failed: list[str] = []

    if args.group in (0, 1):
        print("=== Group 1: guidance_euler forward × GS × invert ===", flush=True)
        all_failed.extend(group_1(repo_root, mp3s, music_dir_rel, dry_run=args.dry_run))

    if args.group in (0, 2):
        print("=== Group 2: euler forward × invert ===", flush=True)
        all_failed.extend(group_2(repo_root, mp3s, music_dir_rel, dry_run=args.dry_run))

    if all_failed:
        print(f"\n=== Sweep finished with {len(all_failed)} failed run(s) ===", file=sys.stderr)
        for m in all_failed:
            print(m, file=sys.stderr)
        return 1

    print("=== All runs completed successfully ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
