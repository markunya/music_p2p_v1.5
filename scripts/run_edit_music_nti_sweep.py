#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

INV_METHODS = ("gci", "uni_euler", "uni_guidance_euler")


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


def build_hydra_args(
    *,
    inv_method: str,
    track_id: str,
    mp3_name: str,
    music_dir_rel: str,
    gs: float,
    inference_steps: int,
    gci_j_eps: str,
    nti_num_inner: int,
    nti_optimize_first_outer: int,
) -> list[str]:
    exp = f"__NTI___track_{track_id}___method_{inv_method}___gs_{gs}"
    cont = continuation_steps_for_gci(gs)
    args: list[str] = [
        f"stepper@invert_stepper={inv_method}",
        "stepper=guidance_euler",
        f"prompt@p2p_task.src=detailed/{track_id}",
        f"prompt@p2p_task.tgt=detailed/gender/{track_id}",
        f"exp_name={exp}",
        f"music_path={music_dir_rel}/{mp3_name}",
        "log_local_error=true",
        f"inference_steps={inference_steps}",
        f"stepper.guidance_scale={gs}",
        "log_trajectory_images=true",
        "nti.enabled=true",
        "nti.epsilon=1.0e-8",
        f"nti.num_inner_steps={nti_num_inner}",
        f"nti.optimize_first_outer_steps={nti_optimize_first_outer}",
    ]
    if inv_method == "gci":
        args += [
            f"invert_stepper.continuation_steps={cont}",
            f"invert_stepper.guidance_scale={gs}",
            "invert_stepper.j_approx=true",
            f"invert_stepper.j_eps={gci_j_eps}",
        ]
    elif inv_method == "uni_guidance_euler":
        args.append(f"invert_stepper.guidance_scale={gs}")
    return args


def run_edit_music(
    repo_root: Path,
    hydra_args: list[str],
    *,
    dry_run: bool,
    use_caffeinate: bool,
) -> int:
    cmd = [sys.executable, str(repo_root / "edit_music.py"), *hydra_args]
    if use_caffeinate and sys.platform == "darwin":
        cmd = ["caffeinate", "-i", *cmd]
    print(subprocess.list2cmdline(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=repo_root).returncode


def sweep(
    repo_root: Path,
    mp3s: list[Path],
    music_dir_rel: str,
    *,
    gs: float,
    inference_steps: int,
    gci_j_eps: str,
    nti_num_inner: int,
    nti_optimize_first_outer: int,
    dry_run: bool,
    use_caffeinate: bool,
) -> list[str]:
    failed: list[str] = []
    for mp3 in mp3s:
        tid = track_prompt_id(mp3)
        for method in INV_METHODS:
            hydra = build_hydra_args(
                inv_method=method,
                track_id=tid,
                mp3_name=mp3.name,
                music_dir_rel=music_dir_rel,
                gs=gs,
                inference_steps=inference_steps,
                gci_j_eps=gci_j_eps,
                nti_num_inner=nti_num_inner,
                nti_optimize_first_outer=nti_optimize_first_outer,
            )
            rc = run_edit_music(
                repo_root, hydra, dry_run=dry_run, use_caffeinate=use_caffeinate
            )
            if rc != 0:
                msg = f"[nti_sweep] rc={rc} track={tid} method={method}"
                print(msg, file=sys.stderr)
                failed.append(msg)
    return failed


def main() -> int:
    p = argparse.ArgumentParser(
        description="NTI sweep: guidance_euler forward × gci/uni_euler/uni_guidance_euler per track."
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="music_p2p_v1.5 (default: parent of scripts/).",
    )
    p.add_argument(
        "--real-music-dir",
        type=Path,
        default=None,
        help="Folder with *.mp3. Default: <repo>/real_music or <repo>/../real_music.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    p.add_argument(
        "--no-caffeinate",
        action="store_true",
        help="Do not wrap with caffeinate -i (default: use on macOS).",
    )
    p.add_argument("--guidance-scale", type=float, default=2.0, help="Forward + invert GS where applicable.")
    p.add_argument("--inference-steps", type=int, default=50)
    p.add_argument("--gci-j-eps", type=str, default="1.0", help="invert_stepper.j_eps for gci only.")
    p.add_argument("--nti-num-inner-steps", type=int, default=10)
    p.add_argument("--nti-optimize-first-outer-steps", type=int, default=20)
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
            print("Pass --real-music-dir /path/to/mp3s", file=sys.stderr)
        return 2

    real_music, music_dir_rel = resolved
    mp3s = sorted(real_music.glob("*.mp3"))
    if not mp3s:
        print(f"No *.mp3 in {real_music}", file=sys.stderr)
        return 2

    use_caf = not args.no_caffeinate

    print(f"repo_root={repo_root}", flush=True)
    print(f"real_music={real_music} (music_path prefix: {music_dir_rel}/)", flush=True)
    print(f"tracks ({len(mp3s)}): {[p.name for p in mp3s]}", flush=True)
    print(f"methods: {INV_METHODS}", flush=True)
    print(f"guidance_scale={args.guidance_scale} NTI inner={args.nti_num_inner_steps} "
          f"optimize_first_outer={args.nti_optimize_first_outer_steps}", flush=True)
    print(f"dry_run={args.dry_run} caffeinate={use_caf and sys.platform == 'darwin'}", flush=True)

    failed = sweep(
        repo_root,
        mp3s,
        music_dir_rel,
        gs=args.guidance_scale,
        inference_steps=args.inference_steps,
        gci_j_eps=args.gci_j_eps,
        nti_num_inner=args.nti_num_inner_steps,
        nti_optimize_first_outer=args.nti_optimize_first_outer_steps,
        dry_run=args.dry_run,
        use_caffeinate=use_caf,
    )
    if failed:
        print(f"\n=== Sweep finished with {len(failed)} failed run(s) ===", file=sys.stderr)
        for m in failed:
            print(m, file=sys.stderr)
        return 1
    print("=== All runs completed successfully ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
