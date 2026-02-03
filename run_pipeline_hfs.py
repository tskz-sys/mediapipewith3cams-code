#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_pipeline_multi.py

sync1 のようなディレクトリ内の
  1match{game}_{match}.mp4 / 2match... / 3match... 
を、指定した game / match だけ一括処理するラッパー。

パイプライン
------------
1) batch3_3dposeestimation.py で 3D推定 → match{g}_{m}_raw.csv を作る
2) (任意) fix_pose_csv_adaptive.py でゲート追跡 → match{g}_{m}_fixed.csv
3) (任意) smooth_csv.py で補間＋移動平均 → match{g}_{m}_smoothed.csv
4) (任意) plot_3d_simple_compare.py で raw/fixed/smoothed の比較動画

さらに、calib npz が batch3 互換(R1/t1/R3/t3/P1/P3...)じゃない場合は
自動で互換npzを out_dir に生成して使う。

例
--
uv run python code/run_pipeline_multi.py \
  --input_dir ../ffmpeg/movie/1208experiment/sync1 \
  --calib_npz ../calibrationwith3cams/output/1208_3cams_center_cam2.npz \
  --out_dir ./output/pipeline_match1 \
  --game 1 --matches 1-15 \
  --postprocess external \
  --compare_3d

Windowsの UNC も受け付け(※bashでは必ずクォート推奨):
  --input_dir "\\\\wsl.localhost\\Ubuntu\\home\\nagas\\research\\ffmpeg\\movie\\1208experiment\\sync1"

"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np


VIDEO_RE = re.compile(r"^[123]match(?P<game>\d+)_(?P<match>\d+)\.mp4$")


def normalize_wsl_path(p: str) -> str:
    """Accept //wsl.localhost/Ubuntu/home/... and convert to /home/..."""
    p = p.strip().strip('"').strip("'")
    if not p:
        return p
    p2 = p.replace('\\', '/')
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
            lo = int(a.strip()); hi = int(b.strip())
            if hi < lo:
                lo, hi = hi, lo
            for x in range(lo, hi + 1):
                out.add(x)
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


def _pick_key(data: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in data:
            return k
    return None


def maybe_convert_npz_for_batch3(src_npz: str, out_npz: str) -> str:
    """batch3 が読むキー(R1/t1/R3/t3/P1/P3/...)が無ければ out_npz を生成して返す。"""
    src_npz = normalize_wsl_path(src_npz)
    out_npz = normalize_wsl_path(out_npz)

    d = np.load(src_npz, allow_pickle=True)
    keys = set(d.files)

    if {'R1', 'R3'}.issubset(keys) and (('t1' in keys) or ('T1' in keys)) and (('t3' in keys) or ('T3' in keys)):
        return src_npz

    if Path(out_npz).exists():
        return out_npz

    # Intrinsics
    k1_key = _pick_key(d, ['K1', 'cam_matrix1'])
    k2_key = _pick_key(d, ['K2', 'cam_matrix2'])
    k3_key = _pick_key(d, ['K3', 'cam_matrix3'])
    if not (k1_key and k2_key and k3_key):
        raise KeyError(f"Cannot find K matrices in {src_npz}. keys={sorted(keys)}")
    K1 = np.array(d[k1_key], dtype=np.float64)
    K2 = np.array(d[k2_key], dtype=np.float64)
    K3 = np.array(d[k3_key], dtype=np.float64)

    # Distortion
    d1_key = _pick_key(d, ['dist1', 'dist_coeffs1'])
    d2_key = _pick_key(d, ['dist2', 'dist_coeffs2'])
    d3_key = _pick_key(d, ['dist3', 'dist_coeffs3'])
    if not (d1_key and d2_key and d3_key):
        raise KeyError(f"Cannot find distortion arrays in {src_npz}. keys={sorted(keys)}")
    D1 = np.array(d[d1_key], dtype=np.float64)
    D2 = np.array(d[d2_key], dtype=np.float64)
    D3 = np.array(d[d3_key], dtype=np.float64)

    # Extrinsics relative to cam2 world
    R1_key = _pick_key(d, ['R_w2c1', 'R21', 'R12'])
    t1_key = _pick_key(d, ['t_w2c1', 'T21', 'T12'])
    R3_key = _pick_key(d, ['R_w2c3', 'R23', 'R32'])
    t3_key = _pick_key(d, ['t_w2c3', 'T23', 'T32'])
    if not (R1_key and t1_key and R3_key and t3_key):
        raise KeyError(
            "Cannot find world-to-camera extrinsics for cam1/cam3. "
            "Need R_w2c1/t_w2c1 & R_w2c3/t_w2c3 (or R21/T21, R23/T23). "
            f"keys={sorted(keys)}"
        )

    R1 = np.array(d[R1_key], dtype=np.float64)
    t1 = np.array(d[t1_key], dtype=np.float64).reshape(3, 1)
    R3 = np.array(d[R3_key], dtype=np.float64)
    t3 = np.array(d[t3_key], dtype=np.float64).reshape(3, 1)

    # Projection matrices: choose_extrinsics を direct に誘導するため入れておく
    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P3 = K3 @ np.hstack([R3, t3])

    payload = {
        'K1': K1,
        'K2': K2,
        'K3': K3,
        'dist1': D1,
        'dist2': D2,
        'dist3': D3,
        'R1': R1,
        't1': t1,
        'R3': R3,
        't3': t3,
        'P1': P1,
        'P2': P2,
        'P3': P3,
        'coord_system': d['coord_system'] if 'coord_system' in d else 'cam2_world',
        'units': d['units'] if 'units' in d else 'm',
        'source_npz': np.array([src_npz], dtype=object),
    }

    ensure_dir(str(Path(out_npz).parent))
    np.savez_compressed(out_npz, **payload)
    print(f"[npz] Converted calib npz for batch3: {out_npz}")
    return out_npz


def _fmt_cmd(cmd: List[str]) -> str:
    return ' '.join([repr(c) if (' ' in c or '\\' in c) else c for c in cmd])


def run_step(cmd: List[str], log_f, dry_run: bool) -> int:
    print(_fmt_cmd(cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return int(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description='Bulk pipeline runner for match{game}_{match} triplets')

    ap.add_argument('--input_dir', required=True, help='Directory containing 1matchX_Y.mp4 / 2matchX_Y.mp4 / 3matchX_Y.mp4')
    ap.add_argument('--calib_npz', required=True, help='Calibration npz (auto-convert if needed)')
    ap.add_argument('--out_dir', required=True, help='Output directory')

    ap.add_argument('--game', type=int, default=None, help='num1: game number (e.g., 1). If omitted, uses all found.')
    ap.add_argument('--games', type=str, default=None, help='Multiple games: e.g., "1,2" or "1-3"')
    ap.add_argument('--matches', type=str, default=None, help='num2 list/range: e.g., "1-15" or "1,3,5". If omitted, uses all found for selected games.')

    ap.add_argument('--postprocess', choices=['none', 'external', 'batch3'], default='external',
                    help='none=3D推定のみ / external=fix+smoothを外部スクリプトで実行 / batch3=batch3の--run_smoothingを使う')
    ap.add_argument('--compare_3d', action='store_true', help='raw/fixed/smoothed を1つの3D比較動画にする (external時のみ推奨)')

    ap.add_argument('--dry_run', action='store_true', help='Print commands only')

    # External postprocess params
    ap.add_argument('--base_anchor_thr', type=float, default=0.5, help='fix_pose_csv_adaptive: 通常の移動許容範囲(m)')
    ap.add_argument('--force_reset_frames', type=int, default=10, help='fix_pose_csv_adaptive: 強制リセットの欠損フレ数')

    ap.add_argument('--smooth_window', type=int, default=5, help='smooth_csv: rolling window')
    ap.add_argument('--jump_thresh', type=float, default=0.0, help='smooth_csv: jump suppression threshold (m); 0=off')
    ap.add_argument('--neighbor_thresh', type=float, default=0.25, help='smooth_csv: neighbor threshold (m)')
    ap.add_argument('--max_gap', type=int, default=2, help='smooth_csv: max frame gap')

    # Script paths
    ap.add_argument('--batch3_script', default=None, help='Path to batch3_3dposeestimation.py')
    ap.add_argument('--fix_script', default=None, help='Path to fix_pose_csv_adaptive.py')
    ap.add_argument('--smooth_script', default=None, help='Path to smooth_csv.py')
    ap.add_argument('--plot_script', default=None, help='Path to plot_3d_simple_compare.py')

    # Pass-through args to batch3 (after "--")
    ap.add_argument('batch3_args', nargs=argparse.REMAINDER, help='Arguments passed through to batch3 (put them after --)')

    args = ap.parse_args()

    input_dir = normalize_wsl_path(args.input_dir)
    calib_npz = normalize_wsl_path(args.calib_npz)
    out_dir = normalize_wsl_path(args.out_dir)
    ensure_dir(out_dir)

    # Script defaults (run from repo root)
    batch3_script = normalize_wsl_path(args.batch3_script or str(Path('code') / 'batch3_3dposeestimation.py'))
    fix_script = normalize_wsl_path(args.fix_script or str(Path('code') / 'fix_pose_csv_adaptive.py'))
    smooth_script = normalize_wsl_path(args.smooth_script or str(Path('code') / 'smooth_csv.py'))
    plot_script = normalize_wsl_path(args.plot_script or str(Path('code') / 'plot_3d_simple_compare.py'))

    for p in [batch3_script]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Script not found: {p}")

    # Prepare calib npz for batch3
    legacy_npz = str(Path(out_dir) / '_calib_legacy_for_batch3.npz')
    calib_for_batch3 = maybe_convert_npz_for_batch3(calib_npz, legacy_npz)

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
        targets = [p for p in targets if p[0] in gset]
    if matches is not None:
        mset = set(matches)
        targets = [p for p in targets if p[1] in mset]

    if not targets:
        print('[WARN] No targets matched your filters.')
        print(f'  available pairs (first 20): {sorted(list(available))[:20]}')
        return

    passthrough = list(args.batch3_args)
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]

    ok = 0
    ng = 0

    for g, m in targets:
        base = f"match{g}_{m}.mp4"
        v1 = Path(input_dir) / f"1{base}"
        v2 = Path(input_dir) / f"2{base}"
        v3 = Path(input_dir) / f"3{base}"
        missing = [str(p) for p in (v1, v2, v3) if not p.exists()]
        if missing:
            print(f"[SKIP] match{g}_{m} missing videos: {missing}")
            ng += 1
            continue

        log_path = Path(out_dir) / f"_log_match{g}_{m}.txt"
        print(f"\n=== match{g}_{m} ===")
        print(f"log: {log_path}")

        with open(log_path, 'w', encoding='utf-8') as log_f:
            # Step 1: batch3 (3D estimation)
            cmd_batch3 = [
                sys.executable, batch3_script,
                '--input_dir', input_dir,
                '--calib_npz', calib_for_batch3,
                '--out_dir', out_dir,
                '--only_game', str(g),
                '--only_match', str(m),
            ]
            if args.postprocess == 'batch3':
                # make sure smoothing is enabled
                if '--run_smoothing' not in passthrough:
                    cmd_batch3.append('--run_smoothing')
            cmd_batch3.extend(passthrough)

            rc = run_step(cmd_batch3, log_f, args.dry_run)
            if rc != 0:
                print(f"[ERR] batch3 failed for match{g}_{m}. See log: {log_path}")
                ng += 1
                continue

            raw_csv = Path(out_dir) / f"match{g}_{m}_raw.csv"
            fixed_csv = Path(out_dir) / f"match{g}_{m}_fixed.csv"
            smoothed_csv = Path(out_dir) / f"match{g}_{m}_smoothed.csv"

            # Step 2-3: external postprocess
            if args.postprocess == 'external':
                if not Path(fix_script).exists():
                    raise FileNotFoundError(f"fix_script not found: {fix_script}")
                if not Path(smooth_script).exists():
                    raise FileNotFoundError(f"smooth_script not found: {smooth_script}")

                if not raw_csv.exists() and not args.dry_run:
                    print(f"[ERR] raw csv not found: {raw_csv}")
                    ng += 1
                    continue

                cmd_fix = [
                    sys.executable, fix_script,
                    '--input', str(raw_csv),
                    '--output', str(fixed_csv),
                    '--base_anchor_thr', str(args.base_anchor_thr),
                    '--force_reset_frames', str(args.force_reset_frames),
                ]
                rc = run_step(cmd_fix, log_f, args.dry_run)
                if rc != 0:
                    print(f"[ERR] fix step failed for match{g}_{m}. See log: {log_path}")
                    ng += 1
                    continue

                cmd_smooth = [
                    sys.executable, smooth_script,
                    '--input', str(fixed_csv),
                    '--output', str(smoothed_csv),
                    '--window', str(args.smooth_window),
                    '--jump_thresh', str(args.jump_thresh),
                    '--neighbor_thresh', str(args.neighbor_thresh),
                    '--max_gap', str(args.max_gap),
                ]
                rc = run_step(cmd_smooth, log_f, args.dry_run)
                if rc != 0:
                    print(f"[ERR] smooth step failed for match{g}_{m}. See log: {log_path}")
                    ng += 1
                    continue

                if args.compare_3d:
                    if not Path(plot_script).exists():
                        raise FileNotFoundError(f"plot_script not found: {plot_script}")
                    out_mp4 = Path(out_dir) / f"match{g}_{m}_compare3d.mp4"
                    cmd_plot = [
                        sys.executable, plot_script,
                        '--inputs', str(raw_csv), str(fixed_csv), str(smoothed_csv),
                        '--output', str(out_mp4),
                        '--cols', '3',
                    ]
                    rc = run_step(cmd_plot, log_f, args.dry_run)
                    if rc != 0:
                        print(f"[ERR] compare_3d failed for match{g}_{m}. See log: {log_path}")
                        ng += 1
                        continue

        print(f"[OK] match{g}_{m} done")
        ok += 1

    print("\n=== SUMMARY ===")
    print(f"input_dir : {input_dir}")
    print(f"out_dir   : {out_dir}")
    print(f"calib_npz : {calib_for_batch3}")
    print(f"targets   : {len(targets)} (ok={ok}, failed/skip={ng})")


if __name__ == '__main__':
    main()
