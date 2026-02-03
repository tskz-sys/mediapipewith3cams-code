#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run YOLO+MediaPipe 3D pose estimation for match1_1..1_6,
then compute metrics and build a grid 3D animation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from calc_metrics import compute_metrics, load_goal_json


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _invert_extrinsics(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    T_inv = -R_inv @ T.reshape(3, 1)
    return R_inv, T_inv


def _projection_error(P_ref: np.ndarray, K: np.ndarray, R: np.ndarray, T: np.ndarray) -> float:
    P_est = K @ np.hstack([R, T.reshape(3, 1)])
    denom = float(np.linalg.norm(P_ref))
    if denom < 1e-9:
        return float("inf")
    return float(np.linalg.norm(P_est - P_ref) / denom)


def prepare_calib_npz(calib_npz: str, out_dir: str) -> str:
    try:
        with np.load(calib_npz) as data:
            files = set(data.files)
            needed = {"P1", "P3", "K1", "K3", "R1", "R3", "T1", "T3"}
            if not needed.issubset(files):
                return calib_npz

            def needs_preinvert(cam_idx: int) -> tuple[bool, float, float] | None:
                try:
                    K = data[f"K{cam_idx}"]
                    R = data[f"R{cam_idx}"]
                    T = data[f"T{cam_idx}"]
                    P = data[f"P{cam_idx}"]
                except KeyError:
                    return None
                err_direct = _projection_error(P, K, R, T)
                R_inv, T_inv = _invert_extrinsics(R, T)
                err_inv = _projection_error(P, K, R_inv, T_inv)
                return (err_direct < err_inv * 0.5), err_direct, err_inv

            checks = [needs_preinvert(1), needs_preinvert(3)]
            checks = [c for c in checks if c is not None]
            if not checks:
                return calib_npz

            if not any(flag for flag, _, _ in checks):
                return calib_npz

            data_dict = {k: data[k] for k in data.files}
    except Exception as exc:
        print(f"Warning: calibration check failed ({exc}), using raw npz.", file=sys.stderr)
        return calib_npz

    for cam_idx in (1, 2, 3):
        r_key = f"R{cam_idx}"
        t_key = f"T{cam_idx}"
        if r_key in data_dict and t_key in data_dict:
            R_inv, T_inv = _invert_extrinsics(data_dict[r_key], data_dict[t_key])
            data_dict[r_key] = R_inv
            data_dict[t_key] = T_inv

    compat_name = f"calib_preinverted_{os.path.basename(calib_npz)}"
    compat_path = os.path.join(out_dir, compat_name)
    np.savez(compat_path, **data_dict)
    print(f"Adjusted calib npz for batch: {compat_path}")
    return compat_path


def build_default_paths():
    input_dir = os.path.abspath(os.path.join(ROOT_DIR, "..", "ffmpeg", "output"))
    calib_npz = os.path.abspath(os.path.join(ROOT_DIR, "npz", "11253cams_fixedd.npz"))
    output_root = os.path.abspath(os.path.join(ROOT_DIR, "output"))
    goal_json = os.path.abspath(os.path.join(ROOT_DIR, "output", "3dposeestimation", "goal_position.json"))
    return input_dir, calib_npz, output_root, goal_json


def main() -> int:
    input_dir, calib_npz, output_root, default_goal_json = build_default_paths()

    parser = argparse.ArgumentParser(description="Run match1_1..1_6 pipeline and metrics.")
    parser.add_argument("--game", type=int, default=1, help="Game number in filenames")
    parser.add_argument("--matches", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Match numbers")
    parser.add_argument("--input_dir", type=str, default=input_dir, help="Input videos dir")
    parser.add_argument("--calib_npz", type=str, default=calib_npz, help="Calibration npz")
    parser.add_argument("--output_root", type=str, default=output_root, help="Base output dir")
    parser.add_argument("--goal_json", type=str, default=default_goal_json, help="Goal position json")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: timestamped)")
    parser.add_argument("--reproj_err_thresh", type=float, default=0.08, help="Drop joints with reprojection error over this threshold")
    parser.add_argument("--jump_center_thresh", type=float, default=0.9, help="Drop frames with center jump vs prev/next (m)")
    parser.add_argument("--release_tail", type=int, default=0, help="Search release within the last N ball frames (0=use latter half)")
    parser.add_argument("--fps", type=int, default=30, help="Grid video FPS")
    parser.add_argument("--grid_cols", type=int, default=3, help="Grid columns")
    parser.add_argument("--grid_elev", type=int, default=20, help="Grid view elevation")
    parser.add_argument("--grid_azim", type=int, default=45, help="Grid view azimuth")
    args = parser.parse_args()

    matches = sorted(set(args.matches))
    if not matches:
        print("Error: No matches specified.", file=sys.stderr)
        return 1

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{min(matches)}_{max(matches)}"
        out_dir = os.path.join(os.path.abspath(args.output_root), f"match{args.game}_{label}_run_{ts}")

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir not found: {args.input_dir}", file=sys.stderr)
        return 1
    if not os.path.exists(args.calib_npz):
        print(f"Error: calib_npz not found: {args.calib_npz}", file=sys.stderr)
        return 1

    calib_npz = prepare_calib_npz(args.calib_npz, out_dir)

    goal_pos = None
    if args.goal_json:
        if not os.path.exists(args.goal_json):
            print(f"Error: goal_json not found: {args.goal_json}", file=sys.stderr)
            return 1
        goal_pos, goal_err = load_goal_json(args.goal_json)
        if goal_err:
            print(goal_err, file=sys.stderr)
            return 1
        shutil.copy2(args.goal_json, os.path.join(out_dir, os.path.basename(args.goal_json)))

    batch_script = os.path.join(ROOT_DIR, "code", "batch_3dposeestimation.py")
    for match in matches:
        cmd = [
            sys.executable,
            batch_script,
            "--input_dir",
            args.input_dir,
            "--calib_npz",
            calib_npz,
            "--out_dir",
            out_dir,
            "--run_smoothing",
            "--reassign_ids_by_ball",
            "--only_game",
            str(args.game),
            "--only_match",
            str(match),
        ]
        if args.reproj_err_thresh is not None:
            cmd += ["--reproj_err_thresh", str(args.reproj_err_thresh)]
        if args.jump_center_thresh is not None:
            cmd += ["--jump_center_thresh", str(args.jump_center_thresh)]
        run_cmd(cmd)

    metrics_rows = []
    smoothed_paths = []
    missing = []
    for match in matches:
        csv_path = os.path.join(out_dir, f"match{args.game}_{match}_smoothed.csv")
        if not os.path.exists(csv_path):
            missing.append(csv_path)
            continue
        smoothed_paths.append(csv_path)
        df = pd.read_csv(csv_path)
        metrics = compute_metrics(
            df,
            release_tail=args.release_tail,
            goal_pos=goal_pos,
        )
        row = {
            "match": f"{args.game}_{match}",
            "csv": os.path.basename(csv_path),
            "release_frame": metrics.get("release_frame"),
            "avg_spacing": metrics.get("avg_spacing"),
            "contest_dist": metrics.get("contest_dist"),
            "contest_error": metrics.get("contest_error"),
            "offense_goal_dist": metrics.get("offense_goal_dist"),
            "goal_error": metrics.get("goal_error"),
            "shooter_id": metrics.get("shooter_id"),
            "defender_id": metrics.get("defender_id"),
            "spacing_frames": metrics.get("spacing_frames"),
            "frame_count": metrics.get("frame_count"),
            "start_frame": metrics.get("start_frame"),
            "errors": "; ".join(metrics.get("errors", [])),
            "warnings": "; ".join(metrics.get("warnings", [])),
        }
        if metrics.get("goal_pos") is not None:
            gx, gy, gz = metrics["goal_pos"]
            row["goal_x"] = float(gx)
            row["goal_y"] = float(gy)
            row["goal_z"] = float(gz)
        metrics_rows.append(row)

    summary_csv = os.path.join(out_dir, "metrics_summary.csv")
    pd.DataFrame(metrics_rows).to_csv(summary_csv, index=False)

    summary_json = os.path.join(out_dir, "metrics_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "game": args.game,
                "matches": matches,
                "goal_json": os.path.basename(args.goal_json) if args.goal_json else None,
                "metrics": metrics_rows,
                "missing_csv": missing,
            },
            f,
            indent=2,
        )

    if missing:
        print("Warning: Missing smoothed CSVs:")
        for p in missing:
            print("  -", p)

    if smoothed_paths:
        grid_script = os.path.join(ROOT_DIR, "code", "create_3d_grid_animation.py")
        grid_out = os.path.join(out_dir, "video", f"match{args.game}_{min(matches)}_{max(matches)}_grid.mp4")
        cmd = [
            sys.executable,
            grid_script,
            "--inputs",
            *smoothed_paths,
            "--output",
            grid_out,
            "--fps",
            str(args.fps),
            "--cols",
            str(args.grid_cols),
            "--elev",
            str(args.grid_elev),
            "--azim",
            str(args.grid_azim),
            "--release_tail",
            str(args.release_tail),
        ]
        run_cmd(cmd)
    else:
        print("Warning: No smoothed CSVs found, skipping grid video.")

    print(f"Done. Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
