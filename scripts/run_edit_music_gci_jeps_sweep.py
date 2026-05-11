#!/usr/bin/env python3
"""
Per-track sweep: GCI inversion with guidance_scale=1.5, j_approx=true, varying j_eps.

Same Hydra layout as GUIDANCE_INVERSION in ``run_edit_music_sweeps.py`` (guidance_euler
forward, gci invert, detailed prompts). Run from repo root ``music_p2p_v1.5`` or pass
``--repo-root``.

j_eps grid: 1e-1, 1e-2, 1e-3, 1e-4 (Hydra CLI overrides, no ``=`` inside exp_name).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

GS = 2.0
J_EPS_CLI = ("1",)


def continuation_steps_for_gci(gs: float) -> int:
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


def run_edit_music(repo_root: Path, args: list[str], *, dry_run: bool) -> int:
    cmd = [sys.executable, str(repo_root / "edit_music.py"), *args]
    print(subprocess.list2cmdline(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=repo_root).returncode


def sweep(
    repo_root: Path, mp3s: list[Path], music_dir_rel: str, *, dry_run: bool
) -> list[str]:
    failed: list[str] = []
    cont = continuation_steps_for_gci(GS)

    for mp3 in mp3s:
        tid = track_prompt_id(mp3)
        music_arg = f"{music_dir_rel}/{mp3.name}"

        for jeps_tag in J_EPS_CLI:
            hydra: list[str] = [
                "stepper@invert_stepper=gci",
                "stepper=guidance_euler",
                f"prompt@p2p_task.src=detailed/{tid}",
                f"prompt@p2p_task.tgt=detailed/gender/{tid}",
                f"exp_name=GUIDANCE_INVERSION_GCI_JAPPROX___track_{tid}___gs_{GS}___jeps_{jeps_tag}_",
                f"music_path={music_arg}",
                "log_local_error=true",
                "inference_steps=50",
                f"stepper.guidance_scale={GS}",
                "log_trajectory_images=true",
                f"invert_stepper.continuation_steps={cont}",
                f"invert_stepper.guidance_scale={GS}",
                "invert_stepper.j_approx=true",
                f"invert_stepper.j_eps={jeps_tag}",
            ]
            rc = run_edit_music(repo_root, hydra, dry_run=dry_run)
            if rc != 0:
                msg = f"[gci_jeps] rc={rc} track={tid} j_eps={jeps_tag}"
                print(msg, file=sys.stderr)
                failed.append(msg)
    return failed


def main() -> int:
    p = argparse.ArgumentParser(description="GCI j_eps sweep (gs=1.5, j_approx=true) per track.")
    p.add_argument("--repo-root", type=Path, default=None, help="music_p2p_v1.5 (default: parent of scripts/).")
    p.add_argument(
        "--real-music-dir",
        type=Path,
        default=None,
        help="Folder with *.mp3. Default: first existing among <repo>/real_music, <repo>/../real_music.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = p.parse_args()

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
    print(f"GCI gs={GS} j_approx=true j_eps in {list(J_EPS_CLI)}", flush=True)

    failed = sweep(repo_root, mp3s, music_dir_rel, dry_run=args.dry_run)
    if failed:
        print(f"\n=== Sweep finished with {len(failed)} failed run(s) ===", file=sys.stderr)
        for m in failed:
            print(m, file=sys.stderr)
        return 1
    print("=== All runs completed successfully ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
