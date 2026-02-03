#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix ID swaps so the release shooter matches the initial offense ID.

If shooter_id at release != initial offense ID (farther from goal on the first
valid two-person frame), swap IDs starting from the frame where the two players
are closest when searching backward from release.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from calc_metrics import (
    CORE_JOINTS,
    calculate_centroid,
    compute_metrics,
    load_goal_json,
    select_primary_pids,
)


def load_pose_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ["X", "Y", "Z", "frame", "person_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame", "person_id"])
    df["frame"] = df["frame"].astype(int)
    df["person_id"] = df["person_id"].astype(int)
    return df


def get_center(df_person: pd.DataFrame, frame: int, pid: int) -> Optional[np.ndarray]:
    rows = df_person[(df_person["frame"] == frame) & (df_person["person_id"] == pid)]
    if rows.empty:
        return None
    return calculate_centroid(rows, CORE_JOINTS)


def find_initial_offense(
    df_person: pd.DataFrame,
    pids: list[int],
    goal_pos: np.ndarray,
    max_frame: Optional[int],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    frames = sorted(df_person["frame"].unique())
    if max_frame is not None:
        frames = [f for f in frames if f <= max_frame]
    for f in frames:
        c0 = get_center(df_person, f, pids[0])
        c1 = get_center(df_person, f, pids[1])
        if c0 is None or c1 is None:
            continue
        d0 = float(np.linalg.norm(c0 - goal_pos))
        d1 = float(np.linalg.norm(c1 - goal_pos))
        if d0 <= d1:
            defense_id, offense_id = pids[0], pids[1]
        else:
            defense_id, offense_id = pids[1], pids[0]
        return f, offense_id, defense_id
    return None, None, None


def find_closest_frame(
    df_person: pd.DataFrame,
    pids: list[int],
    start_frame: Optional[int],
    release_frame: int,
) -> Tuple[Optional[int], Optional[float]]:
    frames = sorted(df_person["frame"].unique())
    frames = [f for f in frames if f <= release_frame]
    if start_frame is not None:
        frames = [f for f in frames if f >= start_frame]

    best_frame = None
    best_dist = None
    for f in frames:
        c0 = get_center(df_person, f, pids[0])
        c1 = get_center(df_person, f, pids[1])
        if c0 is None or c1 is None:
            continue
        dist = float(np.linalg.norm(c0 - c1))
        if best_dist is None or dist < best_dist - 1e-9:
            best_dist = dist
            best_frame = f
        elif best_dist is not None and abs(dist - best_dist) <= 1e-9 and best_frame is not None:
            if f > best_frame:
                best_frame = f
    return best_frame, best_dist


def swap_ids_after_frame(df: pd.DataFrame, pid_a: int, pid_b: int, swap_frame: int) -> pd.DataFrame:
    out = df.copy()
    mask = out["frame"] >= swap_frame
    pid_vals = out["person_id"].copy()
    pid_vals[mask & (out["person_id"] == pid_a)] = pid_b
    pid_vals[mask & (out["person_id"] == pid_b)] = pid_a
    out["person_id"] = pid_vals
    return out


def process_csv(
    input_csv: str,
    output_csv: str,
    goal_pos: np.ndarray,
    release_tail: int,
) -> None:
    df = load_pose_csv(input_csv)
    if df.empty:
        print(f"Skip (empty): {input_csv}")
        return

    metrics = compute_metrics(df, release_tail=release_tail, goal_pos=goal_pos)
    release_frame = metrics.get("release_frame")
    shooter_id = metrics.get("shooter_id")
    if release_frame is None or shooter_id is None:
        print(f"Skip (release/shooter missing): {os.path.basename(input_csv)}")
        df.to_csv(output_csv, index=False)
        return

    df_person = df[(df["person_id"] != -1) & (df["joint"] != "ball")].copy()
    pids = select_primary_pids(df_person, limit=2)
    if len(pids) < 2:
        print(f"Skip (less than 2 people): {os.path.basename(input_csv)}")
        df.to_csv(output_csv, index=False)
        return

    init_frame, initial_offense_id, _ = find_initial_offense(
        df_person,
        pids,
        goal_pos,
        max_frame=release_frame,
    )
    if init_frame is None or initial_offense_id is None:
        print(f"Skip (initial offense not found): {os.path.basename(input_csv)}")
        df.to_csv(output_csv, index=False)
        return

    if int(shooter_id) == int(initial_offense_id):
        print(f"No swap needed: {os.path.basename(input_csv)}")
        df.to_csv(output_csv, index=False)
        return

    swap_frame, swap_dist = find_closest_frame(
        df_person,
        pids,
        start_frame=init_frame,
        release_frame=int(release_frame),
    )
    if swap_frame is None:
        print(f"Skip (no swap frame): {os.path.basename(input_csv)}")
        df.to_csv(output_csv, index=False)
        return

    fixed = swap_ids_after_frame(df, pids[0], pids[1], swap_frame)
    fixed.to_csv(output_csv, index=False)
    print(
        f"Swap applied: {os.path.basename(input_csv)} "
        f"release={release_frame} shooter={shooter_id} "
        f"init_offense={initial_offense_id} swap_frame={swap_frame} "
        f"closest_dist={swap_dist:.4f}"
    )


def build_default_goal_path(input_dir: str) -> Optional[str]:
    candidate = os.path.join(input_dir, "goal_position.json")
    if os.path.exists(candidate):
        return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix ID swaps using release and initial offense.")
    parser.add_argument("--input", type=str, default=None, help="Single input CSV")
    parser.add_argument("--output", type=str, default=None, help="Single output CSV")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing match*_smoothed.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for fixed CSVs")
    parser.add_argument("--game", type=int, default=1, help="Game number")
    parser.add_argument("--matches", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Match numbers")
    parser.add_argument("--release_tail", type=int, default=0, help="Release search tail length (0=use latter half)")
    parser.add_argument("--goal_json", type=str, default=None, help="Goal position json")
    parser.add_argument("--suffix", type=str, default="idfix", help="Suffix for output CSVs")
    args = parser.parse_args()

    if args.input:
        if not args.output:
            print("Error: --output is required when using --input.", file=sys.stderr)
            return 1
        goal_path = args.goal_json
        if goal_path is None:
            goal_path = build_default_goal_path(os.path.dirname(args.input))
        if goal_path is None or not os.path.exists(goal_path):
            print("Error: goal_json not found.", file=sys.stderr)
            return 1
        goal_pos, goal_err = load_goal_json(goal_path)
        if goal_err:
            print(goal_err, file=sys.stderr)
            return 1
        process_csv(args.input, args.output, goal_pos, args.release_tail)
        return 0

    if not args.input_dir or not args.output_dir:
        print("Error: --input_dir and --output_dir are required.", file=sys.stderr)
        return 1

    goal_path = args.goal_json or build_default_goal_path(args.input_dir)
    if goal_path is None or not os.path.exists(goal_path):
        print("Error: goal_json not found.", file=sys.stderr)
        return 1
    goal_pos, goal_err = load_goal_json(goal_path)
    if goal_err:
        print(goal_err, file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    for match in sorted(set(args.matches)):
        name = f"match{args.game}_{match}_smoothed.csv"
        input_csv = os.path.join(args.input_dir, name)
        if not os.path.exists(input_csv):
            print(f"Skip (missing): {input_csv}")
            continue
        base = name.replace(".csv", f"_{args.suffix}.csv")
        output_csv = os.path.join(args.output_dir, base)
        process_csv(input_csv, output_csv, goal_pos, args.release_tail)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
