#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""Hybrid -> Fix -> Smooth -> Plot (optional) pipeline runner.

Pipeline per triplet:
  1) Run hybrid_dual_final_patched.py (import + set globals) to create RAW CSV (with ball)
  2) Split ball rows out of RAW CSV
  3) Run fix_pose_csv_adaptive.py on people-only CSV -> FIXED
  4) Run smooth_csv.py on FIXED -> SMOOTHED
  5) Merge ball rows back into FIXED/SMOOTHED outputs
  6) Optionally run plot_3d_simple_compare.py to create a compare MP4

Input modes:
  A) Directory mode: --input_dir points to files like:
       1match{game}_{match}.mp4, 2match{game}_{match}.mp4, 3match{game}_{match}.mp4
  B) Single mode: --video1/--video2/--video3 given explicitly

Notes:
  - Paths like \\wsl.localhost\Ubuntu\home\... are accepted.
  - This script does NOT require a 3-cam calibration video; it uses the provided NPZ.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import pandas as pd


VIDEO_RE = re.compile(r"^[123]match(?P<game>\d+)_(?P<match>\d+)\.mp4$")
MATCH_STEM_RE = re.compile(r"(?:^|/|\\\\)2?match(?P<game>\d+)_(?P<match>\d+)\.mp4$", re.IGNORECASE)


def normalize_wsl_path(p: str) -> str:
    """Convert UNC WSL path to a normal Linux path if needed."""
    if p is None:
        return p
    p = str(p).strip().strip('"').strip("'")
    if not p:
        return p
    p2 = p.replace('\\\\', '/')
    lower = p2.lower()
    if lower.startswith('//wsl.localhost/ubuntu/'):
        rest = p2[len('//wsl.localhost/Ubuntu/'):]
        return '/' + rest.lstrip('/')
    if lower.startswith('/wsl.localhost/ubuntu/'):
        rest = p2[len('/wsl.localhost/Ubuntu/'):]
        return '/' + rest.lstrip('/')
    return p2


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_int_list(spec: Optional[str]) -> Optional[List[int]]:
    """Parse '1-15' or '1,2,5' or '1 2 5' into sorted unique list.

    If spec is None / '' / 'all' -> returns None (meaning "no filter").
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s or s.lower() in {'all', '*'}:
        return None
    parts = re.split(r"[\s,]+", s)
    out: Set[int] = set()
    for part in parts:
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            lo = int(a.strip())
            hi = int(b.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def find_available_pairs(input_dir: str) -> Set[Tuple[int, int]]:
    p = Path(input_dir)
    pairs: Set[Tuple[int, int]] = set()
    if not p.exists():
        return pairs
    for name in os.listdir(p):
        m = VIDEO_RE.match(name)
        if not m:
            continue
        pairs.add((int(m.group('game')), int(m.group('match'))))
    return pairs


def derive_stem_from_center_video(video2: str) -> str:
    v = normalize_wsl_path(video2)
    m = MATCH_STEM_RE.search(v)
    if m:
        return f"match{int(m.group('game'))}_{int(m.group('match'))}"
    # fallback: strip leading '2'
    base = Path(v).name
    if base.startswith('2'):
        base = base[1:]
    return Path(base).stem


def load_hybrid_module(hybrid_script: str):
    hybrid_script = normalize_wsl_path(hybrid_script)
    if not Path(hybrid_script).exists():
        raise FileNotFoundError(f"Hybrid script not found: {hybrid_script}")
    spec = importlib.util.spec_from_file_location("hybrid_mod", hybrid_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import hybrid script: {hybrid_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, 'main'):
        raise RuntimeError("Hybrid script has no main()")
    return mod


def run_hybrid_one(mod, video1: str, video2: str, video3: str, npz: str, out_csv: str, out_video: str) -> None:
    """Run hybrid main() by overwriting globals."""
    mod.VIDEO_LEFT = normalize_wsl_path(video1)
    mod.VIDEO_CENTER = normalize_wsl_path(video2)
    mod.VIDEO_RIGHT = normalize_wsl_path(video3)
    mod.CALIB_NPZ = normalize_wsl_path(npz)
    mod.OUT_CSV = normalize_wsl_path(out_csv)
    mod.OUT_VIDEO = normalize_wsl_path(out_video)

    # Ensure output directories exist (hybrid will fail otherwise)
    ensure_dir(str(Path(mod.OUT_CSV).parent))
    ensure_dir(str(Path(mod.OUT_VIDEO).parent))

    mod.main()


def split_ball_rows(raw_csv: str, people_csv: str, ball_csv: str) -> None:
    df = pd.read_csv(raw_csv)
    if not {'frame', 'person_id', 'joint', 'X', 'Y', 'Z'}.issubset(df.columns):
        raise ValueError(f"Unexpected CSV columns in {raw_csv}")
    ball_mask = (df['person_id'] == -1) | (df['joint'].astype(str) == 'ball')
    df_ball = df[ball_mask].copy()
    df_people = df[~ball_mask].copy()

    # Keep stable ordering
    df_people = df_people.sort_values(['frame', 'person_id', 'joint']).reset_index(drop=True)
    df_ball = df_ball.sort_values(['frame']).reset_index(drop=True)

    df_people.to_csv(people_csv, index=False)
    df_ball.to_csv(ball_csv, index=False)


def merge_ball_rows(people_csv: str, ball_csv: str, out_csv: str) -> None:
    df_people = pd.read_csv(people_csv) if Path(people_csv).exists() else pd.DataFrame()
    df_ball = pd.read_csv(ball_csv) if Path(ball_csv).exists() else pd.DataFrame()

    if df_people.empty and df_ball.empty:
        raise RuntimeError("Both people and ball CSVs are empty.")

    if df_people.empty:
        df = df_ball
    elif df_ball.empty:
        df = df_people
    else:
        df = pd.concat([df_people, df_ball], ignore_index=True)

    # Sort for readability
    sort_cols = [c for c in ['frame', 'person_id', 'joint'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def run_cmd(cmd: Sequence[str], log_path: Optional[str] = None) -> None:
    """Run a subprocess, optionally tee stdout/stderr to log."""
    if log_path:
        log_file = open(log_path, 'a', encoding='utf-8')
    else:
        log_file = None

    try:
        proc = subprocess.run(cmd, stdout=log_file or None, stderr=log_file or None, check=False, text=True)
    finally:
        if log_file:
            log_file.close()

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (code={proc.returncode}): {' '.join(cmd)}")


def pipeline_one(
    hybrid_mod,
    video1: str,
    video2: str,
    video3: str,
    calib_npz: str,
    out_dir: str,
    stem: str,
    fix_script: str,
    smooth_script: str,
    plot_script: str,
    base_anchor_thr: float,
    force_reset_frames: int,
    window: int,
    jump_thresh: float,
    neighbor_thresh: float,
    max_gap: int,
    compare_3d: bool,
    log_path: str,
) -> None:
    out_dir = normalize_wsl_path(out_dir)
    ensure_dir(out_dir)

    raw_csv = str(Path(out_dir) / f"{stem}_raw.csv")
    det_video = str(Path(out_dir) / f"{stem}_det.mp4")

    people_raw = str(Path(out_dir) / f"{stem}_people_raw.csv")
    ball_raw = str(Path(out_dir) / f"{stem}_ball.csv")

    people_fixed = str(Path(out_dir) / f"{stem}_people_fixed.csv")
    people_smoothed = str(Path(out_dir) / f"{stem}_people_smoothed.csv")

    fixed_csv = str(Path(out_dir) / f"{stem}_fixed.csv")
    smoothed_csv = str(Path(out_dir) / f"{stem}_smoothed.csv")

    # Step 1: hybrid detection
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"=== [HYBRID] {stem} ===\n")
        f.write(f"video1={video1}\nvideo2={video2}\nvideo3={video3}\nnpz={calib_npz}\n")

    run_hybrid_one(hybrid_mod, video1, video2, video3, calib_npz, raw_csv, det_video)

    # Step 2: split ball vs people
    split_ball_rows(raw_csv, people_raw, ball_raw)

    # Step 3: fix
    fix_cmd = [
        sys.executable,
        normalize_wsl_path(fix_script),
        '--input', people_raw,
        '--output', people_fixed,
        '--base_anchor_thr', str(base_anchor_thr),
        '--force_reset_frames', str(force_reset_frames),
    ]
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n=== [FIX] {stem} ===\n")
        f.write(' '.join(fix_cmd) + "\n")
    run_cmd(fix_cmd, log_path=log_path)

    # merge ball back (fixed)
    merge_ball_rows(people_fixed, ball_raw, fixed_csv)

    # Step 4: smooth
    smooth_cmd = [
        sys.executable,
        normalize_wsl_path(smooth_script),
        '--input', people_fixed,
        '--output', people_smoothed,
        '--window', str(window),
        '--jump_thresh', str(jump_thresh),
        '--neighbor_thresh', str(neighbor_thresh),
        '--max_gap', str(max_gap),
    ]
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n=== [SMOOTH] {stem} ===\n")
        f.write(' '.join(smooth_cmd) + "\n")
    run_cmd(smooth_cmd, log_path=log_path)

    # merge ball back (smoothed)
    merge_ball_rows(people_smoothed, ball_raw, smoothed_csv)

    # Step 5: optional compare plot
    if compare_3d:
        compare_mp4 = str(Path(out_dir) / f"{stem}_compare3d.mp4")
        plot_cmd = [
            sys.executable,
            normalize_wsl_path(plot_script),
            '--inputs', raw_csv, fixed_csv, smoothed_csv,
            '--output', compare_mp4,
        ]
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n=== [PLOT] {stem} ===\n")
            f.write(' '.join(plot_cmd) + "\n")
        run_cmd(plot_cmd, log_path=log_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run hybrid pipeline for matches.')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input_dir', type=str, help='Directory containing 1matchG_M.mp4,2matchG_M.mp4,3matchG_M.mp4')
    src.add_argument('--video2', type=str, help='Center video (single mode)')

    # Single mode requires these when --video2 is used
    p.add_argument('--video1', type=str, help='Left video (single mode)')
    p.add_argument('--video3', type=str, help='Right video (single mode)')

    p.add_argument('--calib_npz', required=True, type=str, help='3-cam calib npz')
    p.add_argument('--out_dir', required=True, type=str, help='Output directory')

    # Filters for directory mode
    p.add_argument('--game', type=int, default=None, help='Single game number filter (directory mode)')
    p.add_argument('--games', type=str, default=None, help='Game list/range (e.g. 1,2 or 1-3)')
    p.add_argument('--matches', type=str, default='all', help='Match list/range (e.g. 1-15). Default=all')

    # Script paths
    p.add_argument('--hybrid_script', type=str, default=str(Path('code') / 'hybrid_dual_final_patched.py'))
    p.add_argument('--fix_script', type=str, default=str(Path('code') / 'fix_pose_csv_adaptive.py'))
    p.add_argument('--smooth_script', type=str, default=str(Path('code') / 'smooth_csv.py'))
    p.add_argument('--plot_script', type=str, default=str(Path('code') / 'plot_3d_simple_compare.py'))

    # Postprocess params
    p.add_argument('--base_anchor_thr', type=float, default=0.30)
    p.add_argument('--force_reset_frames', type=int, default=10)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--jump_thresh', type=float, default=0.0)
    p.add_argument('--neighbor_thresh', type=float, default=0.25)
    p.add_argument('--max_gap', type=int, default=2)

    p.add_argument('--compare_3d', action='store_true', help='Create compare3d mp4')

    return p


def main() -> None:
    args = build_argparser().parse_args()

    out_dir = normalize_wsl_path(args.out_dir)
    ensure_dir(out_dir)

    calib_npz = normalize_wsl_path(args.calib_npz)
    hybrid_script = normalize_wsl_path(args.hybrid_script)
    fix_script = normalize_wsl_path(args.fix_script)
    smooth_script = normalize_wsl_path(args.smooth_script)
    plot_script = normalize_wsl_path(args.plot_script)

    # Load hybrid module once (faster)
    hybrid_mod = load_hybrid_module(hybrid_script)

    if args.input_dir:
        input_dir = normalize_wsl_path(args.input_dir)
        available = find_available_pairs(input_dir)
        if not available:
            raise RuntimeError(f"No match videos found in: {input_dir}")

        games: Optional[List[int]] = None
        if args.games:
            games = parse_int_list(args.games)
        elif args.game is not None:
            games = [int(args.game)]

        matches = parse_int_list(args.matches)

        targets = sorted(available)
        if games is not None:
            gset = set(games)
            targets = [t for t in targets if t[0] in gset]
        if matches is not None:
            mset = set(matches)
            targets = [t for t in targets if t[1] in mset]

        if not targets:
            print('[WARN] No targets matched your filters.')
            print(f'  available pairs (first 20): {sorted(list(available))[:20]}')
            return

        ok = 0
        ng = 0
        for g, m in targets:
            base = f"match{g}_{m}.mp4"
            v1 = str(Path(input_dir) / f"1{base}")
            v2 = str(Path(input_dir) / f"2{base}")
            v3 = str(Path(input_dir) / f"3{base}")
            missing = [p for p in (v1, v2, v3) if not Path(p).exists()]
            if missing:
                print(f"[SKIP] match{g}_{m} missing videos: {missing}")
                ng += 1
                continue

            stem = f"match{g}_{m}"
            log_path = str(Path(out_dir) / f"_log_{stem}.txt")
            print(f"\n=== {stem} ===")
            print(f"log: {log_path}")

            try:
                pipeline_one(
                    hybrid_mod=hybrid_mod,
                    video1=v1,
                    video2=v2,
                    video3=v3,
                    calib_npz=calib_npz,
                    out_dir=out_dir,
                    stem=stem,
                    fix_script=fix_script,
                    smooth_script=smooth_script,
                    plot_script=plot_script,
                    base_anchor_thr=args.base_anchor_thr,
                    force_reset_frames=args.force_reset_frames,
                    window=args.window,
                    jump_thresh=args.jump_thresh,
                    neighbor_thresh=args.neighbor_thresh,
                    max_gap=args.max_gap,
                    compare_3d=bool(args.compare_3d),
                    log_path=log_path,
                )
                ok += 1
            except Exception as e:
                print(f"[FAIL] {stem}: {e}")
                ng += 1

        print(f"\nDone. OK={ok} NG={ng} out_dir={out_dir}")
        return

    # Single mode
    if not args.video2 or not args.video1 or not args.video3:
        raise ValueError("Single mode requires --video1 --video2 --video3")

    v1 = normalize_wsl_path(args.video1)
    v2 = normalize_wsl_path(args.video2)
    v3 = normalize_wsl_path(args.video3)
    stem = derive_stem_from_center_video(v2)
    log_path = str(Path(out_dir) / f"_log_{stem}.txt")
    print(f"\n=== {stem} ===")
    print(f"log: {log_path}")

    pipeline_one(
        hybrid_mod=hybrid_mod,
        video1=v1,
        video2=v2,
        video3=v3,
        calib_npz=calib_npz,
        out_dir=out_dir,
        stem=stem,
        fix_script=fix_script,
        smooth_script=smooth_script,
        plot_script=plot_script,
        base_anchor_thr=args.base_anchor_thr,
        force_reset_frames=args.force_reset_frames,
        window=args.window,
        jump_thresh=args.jump_thresh,
        neighbor_thresh=args.neighbor_thresh,
        max_gap=args.max_gap,
        compare_3d=bool(args.compare_3d),
        log_path=log_path,
    )


if __name__ == '__main__':
    main()
