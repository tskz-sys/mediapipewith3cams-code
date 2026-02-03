#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity checks for 3D CSV output:
- basic ranges and outliers
- per-camera reprojection bounds / bbox coverage
- motion speed stats
- bone length stability
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd


IGNORE_JOINTS = {
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
}

BONES = [
    ("left_shoulder", "right_shoulder", "shoulder_width"),
    ("left_hip", "right_hip", "hip_width"),
    ("left_shoulder", "left_hip", "torso_left"),
    ("right_shoulder", "right_hip", "torso_right"),
    ("left_hip", "left_knee", "thigh_left"),
    ("right_hip", "right_knee", "thigh_right"),
    ("left_knee", "left_ankle", "shin_left"),
    ("right_knee", "right_ankle", "shin_right"),
]


def load_csv(path: str, include_face: bool) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"frame", "person_id", "joint", "X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    for col in ["frame", "person_id", "X", "Y", "Z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame", "person_id", "X", "Y", "Z"])
    df["frame"] = df["frame"].astype(int)
    df["person_id"] = df["person_id"].astype(int)
    if not include_face:
        df = df[~df["joint"].isin(IGNORE_JOINTS)]
    return df


def load_bbox(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    required = {"frame", "person_id", "cam_idx", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BBox CSV missing columns: {', '.join(sorted(missing))}")
    for col in ["frame", "person_id", "cam_idx", "x1", "y1", "x2", "y2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame", "person_id", "cam_idx", "x1", "y1", "x2", "y2"])
    df["frame"] = df["frame"].astype(int)
    df["person_id"] = df["person_id"].astype(int)
    df["cam_idx"] = df["cam_idx"].astype(int)
    return df


@dataclass
class CamParams:
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    t: np.ndarray
    w: int
    h: int
    fps: float


def _get_video_wh_fps(video_path: str) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return w, h, fps


def get_inverse_transform(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return R.T, -R.T @ T


def scale_camera_matrix(K: np.ndarray, dist: np.ndarray, target_w: int, target_h: int) -> tuple[np.ndarray, np.ndarray]:
    orig_w, orig_h = K[0, 2] * 2, K[1, 2] * 2
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05:
        return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx
    K_new[1, 1] *= sy
    K_new[0, 2] *= sx
    K_new[1, 2] *= sy
    return K_new, dist


def load_params_BR(npz_path: str, v1: str, v2: str, v3: str) -> list[CamParams]:
    d = np.load(npz_path, allow_pickle=True)

    def get_k(k_new, k_old):
        return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1", "cam_matrix1"), get_k("dist1", "dist_coeffs1")
    K2, D2 = get_k("K2", "cam_matrix2"), get_k("dist2", "dist_coeffs2")
    K3, D3 = get_k("K3", "cam_matrix3"), get_k("dist3", "dist_coeffs3")

    w1, h1, fps1 = _get_video_wh_fps(v1)
    w2, h2, fps2 = _get_video_wh_fps(v2)
    w3, h3, fps3 = _get_video_wh_fps(v3)

    K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    R1_raw = d["R1"]
    t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3, 1))

    R2 = np.eye(3)
    t2 = np.zeros((3, 1))

    R3_raw = d["R3"]
    t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3, 1))

    return [
        CamParams(K=K1, D=D1, R=R1, t=t1, w=w1, h=h1, fps=fps1),
        CamParams(K=K2, D=D2, R=R2, t=t2, w=w2, h=h2, fps=fps2),
        CamParams(K=K3, D=D3, R=R3, t=t3, w=w3, h=h3, fps=fps3),
    ]


def project_points(points: np.ndarray, cam: CamParams) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    rvec, _ = cv2.Rodrigues(cam.R)
    img_points, _ = cv2.projectPoints(points, rvec, cam.t, cam.K, cam.D)
    return img_points.reshape(-1, 2)


def compute_ranges(df: pd.DataFrame, max_abs: float) -> dict:
    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    outliers = np.any(np.abs(xyz) > max_abs, axis=1)
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "outlier_count": int(outliers.sum()),
        "outlier_ratio": float(outliers.mean()) if len(outliers) else 0.0,
    }


def compute_motion(df: pd.DataFrame, fps: float) -> dict:
    if fps <= 0:
        fps = 30.0
    centers = df.groupby(["frame", "person_id"])[["X", "Y", "Z"]].median().reset_index()
    speeds = []
    for _, group in centers.groupby("person_id"):
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy()
        coords = group[["X", "Y", "Z"]].to_numpy()
        if len(frames) < 2:
            continue
        dframe = np.diff(frames)
        dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        speed = dist * fps / np.maximum(dframe, 1)
        speeds.append(speed)
    if not speeds:
        return {"count": 0}
    speeds_all = np.concatenate(speeds)
    return {
        "count": int(speeds_all.size),
        "median": float(np.median(speeds_all)),
        "p95": float(np.percentile(speeds_all, 95)),
        "max": float(speeds_all.max()),
    }


def compute_bone_stats(df: pd.DataFrame) -> list[dict]:
    stats = []
    for joint_a, joint_b, label in BONES:
        a = df[df["joint"] == joint_a][["frame", "person_id", "X", "Y", "Z"]].rename(
            columns={"X": "Xa", "Y": "Ya", "Z": "Za"}
        )
        b = df[df["joint"] == joint_b][["frame", "person_id", "X", "Y", "Z"]].rename(
            columns={"X": "Xb", "Y": "Yb", "Z": "Zb"}
        )
        merged = a.merge(b, on=["frame", "person_id"], how="inner")
        if merged.empty:
            continue
        dist = np.sqrt(
            (merged["Xa"] - merged["Xb"]) ** 2
            + (merged["Ya"] - merged["Yb"]) ** 2
            + (merged["Za"] - merged["Zb"]) ** 2
        ).to_numpy()
        median = float(np.median(dist))
        mean = float(np.mean(dist))
        std = float(np.std(dist))
        cv = float(std / mean) if mean > 1e-9 else 0.0
        stats.append(
            {
                "bone": label,
                "count": int(dist.size),
                "median": median,
                "p95": float(np.percentile(dist, 95)),
                "cv": cv,
            }
        )
    return stats


def compute_reprojection(
    df: pd.DataFrame,
    cams: list[CamParams],
    bbox_df: pd.DataFrame | None,
    margin_ratio: float,
) -> list[dict]:
    points = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    frames = df["frame"].to_numpy()
    pids = df["person_id"].to_numpy()
    results = []
    for cam_idx, cam in enumerate(cams):
        uv = project_points(points, cam)
        in_bounds = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < cam.w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < cam.h)
        )
        entry = {
            "cam_idx": cam_idx,
            "in_bounds_ratio": float(in_bounds.mean()) if len(in_bounds) else 0.0,
            "in_bounds_count": int(in_bounds.sum()),
            "total_points": int(len(in_bounds)),
        }

        if bbox_df is not None and not bbox_df.empty:
            bbox_cam = bbox_df[bbox_df["cam_idx"] == cam_idx]
            if not bbox_cam.empty:
                tmp = pd.DataFrame(
                    {
                        "idx": np.arange(len(df)),
                        "frame": frames,
                        "person_id": pids,
                    }
                )
                tmp = tmp.merge(bbox_cam, on=["frame", "person_id"], how="left")
                has_bbox = tmp["x1"].notna().to_numpy()
                if has_bbox.any():
                    idxs = tmp.loc[has_bbox, "idx"].to_numpy()
                    u = uv[idxs, 0]
                    v = uv[idxs, 1]
                    x1 = tmp.loc[has_bbox, "x1"].to_numpy()
                    y1 = tmp.loc[has_bbox, "y1"].to_numpy()
                    x2 = tmp.loc[has_bbox, "x2"].to_numpy()
                    y2 = tmp.loc[has_bbox, "y2"].to_numpy()
                    margin_x = (x2 - x1) * margin_ratio
                    margin_y = (y2 - y1) * margin_ratio
                    in_bbox = (
                        (u >= x1 - margin_x)
                        & (u <= x2 + margin_x)
                        & (v >= y1 - margin_y)
                        & (v <= y2 + margin_y)
                    )
                    entry.update(
                        {
                            "in_bbox_ratio": float(in_bbox.mean()),
                            "in_bbox_count": int(in_bbox.sum()),
                            "bbox_points": int(in_bbox.size),
                        }
                    )
        results.append(entry)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 3D CSV quality")
    parser.add_argument("--input_csv", required=True, help="Input 3D CSV (smoothed)")
    parser.add_argument("--input_dir", required=True, help="Input video dir")
    parser.add_argument("--calib_npz", required=True, help="Calibration npz")
    parser.add_argument("--game", type=int, required=True, help="Game number")
    parser.add_argument("--match", type=int, required=True, help="Match number")
    parser.add_argument("--bbox_csv", default=None, help="BBox CSV (optional)")
    parser.add_argument("--max_abs", type=float, default=20.0, help="Outlier threshold for |coord|")
    parser.add_argument("--bbox_margin", type=float, default=0.1, help="BBox margin ratio")
    parser.add_argument("--include_face", action="store_true", help="Include face joints")
    parser.add_argument("--out_json", default=None, help="Write summary JSON")
    args = parser.parse_args()

    stem = f"match{args.game}_{args.match}"
    v1 = os.path.join(args.input_dir, f"1{stem}.mp4")
    v2 = os.path.join(args.input_dir, f"2{stem}.mp4")
    v3 = os.path.join(args.input_dir, f"3{stem}.mp4")
    for v in (v1, v2, v3):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Video not found: {v}")

    df_all = load_csv(args.input_csv, include_face=args.include_face)
    ball_mask = (df_all["person_id"] == -1) | (df_all["joint"] == "ball")
    df_people = df_all[~ball_mask].reset_index(drop=True)
    df_ball = df_all[ball_mask].reset_index(drop=True)

    bbox_csv = args.bbox_csv
    if bbox_csv is None:
        guess = os.path.join(os.path.dirname(args.input_csv), f"{stem}_bbox.csv")
        if os.path.exists(guess):
            bbox_csv = guess
    bbox_df = load_bbox(bbox_csv)

    cams = load_params_BR(args.calib_npz, v1, v2, v3)
    fps = next((c.fps for c in cams if c.fps > 0), 30.0)

    summary = {
        "csv": os.path.abspath(args.input_csv),
        "frames": int(df_all["frame"].nunique()),
        "points_total": int(len(df_all)),
        "points_people": int(len(df_people)),
        "points_ball": int(len(df_ball)),
        "ranges_people": compute_ranges(df_people, args.max_abs) if not df_people.empty else None,
        "motion_people": compute_motion(df_people, fps) if not df_people.empty else None,
        "bones": compute_bone_stats(df_people) if not df_people.empty else [],
        "reprojection_people": compute_reprojection(df_people, cams, bbox_df, args.bbox_margin)
        if not df_people.empty
        else [],
    }

    if not df_ball.empty:
        summary["reprojection_ball"] = compute_reprojection(df_ball, cams, None, 0.0)

    print("=== 3D CSV Quality Check ===")
    print(f"CSV: {summary['csv']}")
    print(f"Frames: {summary['frames']}")
    print(f"Points (people/ball/total): {summary['points_people']} / {summary['points_ball']} / {summary['points_total']}")

    if summary["ranges_people"]:
        r = summary["ranges_people"]
        print(f"XYZ min: {r['min']}")
        print(f"XYZ max: {r['max']}")
        print(f"Outliers (|coord|>{args.max_abs}): {r['outlier_count']} ({r['outlier_ratio']:.2%})")

    if summary["motion_people"] and summary["motion_people"]["count"] > 0:
        m = summary["motion_people"]
        print(f"Speed m/s (median/p95/max): {m['median']:.3f} / {m['p95']:.3f} / {m['max']:.3f}")

    if summary["bones"]:
        print("Bone length stability (median m / p95 m / cv):")
        for b in summary["bones"]:
            print(f"  {b['bone']}: {b['median']:.3f} / {b['p95']:.3f} / {b['cv']:.3f} (n={b['count']})")

    if summary["reprojection_people"]:
        print("Reprojection in-bounds (people):")
        for cam in summary["reprojection_people"]:
            line = f"  cam{cam['cam_idx']+1}: {cam['in_bounds_ratio']:.2%} ({cam['in_bounds_count']}/{cam['total_points']})"
            if "in_bbox_ratio" in cam:
                line += f", in_bbox {cam['in_bbox_ratio']:.2%} ({cam['in_bbox_count']}/{cam['bbox_points']})"
            print(line)

    if summary.get("reprojection_ball"):
        print("Reprojection in-bounds (ball):")
        for cam in summary["reprojection_ball"]:
            print(
                f"  cam{cam['cam_idx']+1}: {cam['in_bounds_ratio']:.2%} "
                f"({cam['in_bounds_count']}/{cam['total_points']})"
            )

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
