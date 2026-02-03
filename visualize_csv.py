#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproject 3D CSV onto 3 camera videos and save a side-by-side MP4."""

import argparse
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


COLORS = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]
BALL_COLOR = (0, 165, 255)

IGNORE_JOINTS = {
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
}

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

JOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


def project_3d_to_2d(pts_3d, K, R, t, dist):
    if len(pts_3d) == 0:
        return []
    pts_3d = np.array(pts_3d, dtype=np.float64).reshape(-1, 1, 3)
    rvec, _ = cv2.Rodrigues(R)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, t, K, dist)
    return pts_2d.reshape(-1, 2)


def to_int_point(pt, limit=1_000_000):
    arr = np.asarray(pt, dtype=float).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    if abs(arr[0]) > limit or abs(arr[1]) > limit:
        return None
    return (int(arr[0]), int(arr[1]))


def get_inverse_transform(R, T):
    return R.T, -R.T @ T


def projection_error(P_ref, K, R, T):
    P_est = K @ np.hstack([R, T.reshape(3, 1)])
    denom = float(np.linalg.norm(P_ref))
    if denom < 1e-9:
        return float("inf")
    return float(np.linalg.norm(P_est - P_ref) / denom)


def choose_extrinsics(data, cam_idx, K_raw, R_raw, t_raw):
    t_raw = t_raw.reshape(3, 1)
    P_key = f"P{cam_idx}"
    if P_key in data:
        err_direct = projection_error(data[P_key], K_raw, R_raw, t_raw)
        R_inv, t_inv = get_inverse_transform(R_raw, t_raw)
        err_inv = projection_error(data[P_key], K_raw, R_inv, t_inv)
        if err_direct <= err_inv:
            return R_raw, t_raw, "direct", err_direct, err_inv
        return R_inv, t_inv, "inverse", err_direct, err_inv
    R_inv, t_inv = get_inverse_transform(R_raw, t_raw)
    return R_inv, t_inv, "inverse", None, None


def scale_camera_matrix(K, dist, target_w, target_h):
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


def load_params_br(npz_path, v1, v2, v3, invert_both=False):
    d = np.load(npz_path, allow_pickle=True)

    def get_k(k_new, k_old):
        return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1", "cam_matrix1"), get_k("dist1", "dist_coeffs1")
    K2, D2 = get_k("K2", "cam_matrix2"), get_k("dist2", "dist_coeffs2")
    K3, D3 = get_k("K3", "cam_matrix3"), get_k("dist3", "dist_coeffs3")

    K1_raw = K1.copy()
    K3_raw = K3.copy()

    def get_wh(v):
        c = cv2.VideoCapture(v)
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        return w, h

    w1, h1 = get_wh(v1)
    K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    w2, h2 = get_wh(v2)
    K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    w3, h3 = get_wh(v3)
    K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    R1_raw = d["R1"]
    t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1, mode1, err_d1, err_i1 = choose_extrinsics(d, 1, K1_raw, R1_raw, t1_raw)

    R2 = np.eye(3)
    t2 = np.zeros((3, 1))

    R3_raw = d["R3"]
    t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3, mode3, err_d3, err_i3 = choose_extrinsics(d, 3, K3_raw, R3_raw, t3_raw)

    if err_d1 is not None and err_i1 is not None:
        print(f"[calib] cam1 extrinsics: {mode1} (direct={err_d1:.3g}, inv={err_i1:.3g})")
    if err_d3 is not None and err_i3 is not None:
        print(f"[calib] cam3 extrinsics: {mode3} (direct={err_d3:.3g}, inv={err_i3:.3g})")

    if invert_both:
        R1, t1 = get_inverse_transform(R1, t1)
        R3, t3 = get_inverse_transform(R3, t3)
        print("[calib] invert_both applied")

    return [
        {"K": K1, "D": D1, "R": R1, "t": t1},
        {"K": K2, "D": D2, "R": R2, "t": t2},
        {"K": K3, "D": D3, "R": R3, "t": t3},
    ]


def main():
    parser = argparse.ArgumentParser(description="Reproject 3D CSV onto 3 camera videos")
    parser.add_argument("--input_csv", required=True, help="Input 3D CSV")
    parser.add_argument("--input_dir", required=True, help="Input video dir")
    parser.add_argument("--calib_npz", required=True, help="Calibration npz")
    parser.add_argument("--invert_both", action="store_true", help="apply extra invert to calibration extrinsics")
    parser.add_argument("--game", type=int, required=True, help="Game number")
    parser.add_argument("--match", type=int, required=True, help="Match number")
    parser.add_argument("--output", default=None, help="Output MP4 (default: next to CSV)")
    parser.add_argument("--show_face", action="store_true", help="Show face joints")
    args = parser.parse_args()

    stem = f"match{args.game}_{args.match}"
    video_paths = [
        os.path.join(args.input_dir, f"1{stem}.mp4"),
        os.path.join(args.input_dir, f"2{stem}.mp4"),
        os.path.join(args.input_dir, f"3{stem}.mp4"),
    ]

    for v in video_paths:
        if not os.path.exists(v):
            print(f"Error: {v} not found.")
            return
    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} not found.")
        return
    if not os.path.exists(args.calib_npz):
        print(f"Error: {args.calib_npz} not found.")
        return

    if args.output is None:
        out_dir = os.path.join(os.path.dirname(args.input_csv), "video")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"{stem}_reproject.mp4")
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Reading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if not args.show_face:
        df = df[~df["joint"].isin(IGNORE_JOINTS)]

    for col in ["frame", "person_id", "X", "Y", "Z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame", "person_id"])
    df["frame"] = df["frame"].astype(int)

    frame_groups = {int(k): v for k, v in df.groupby("frame")}

    cams_params = load_params_br(
        args.calib_npz, video_paths[0], video_paths[1], video_paths[2], invert_both=args.invert_both
    )
    caps = [cv2.VideoCapture(p) for p in video_paths]

    fps = caps[0].get(cv2.CAP_PROP_FPS)
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(min([c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps]))

    temp_out = args.output + ".temp.mp4"
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W * 3, H))

    print(f"Processing {total_frames} frames...")
    line_error_logged = False
    for f_idx in tqdm(range(total_frames)):
        frames = []
        for c in caps:
            ret, f = c.read()
            frames.append(f if ret else np.zeros((H, W, 3), dtype=np.uint8))

        frame_data = frame_groups.get(f_idx)
        if frame_data is None or frame_data.empty:
            vw.write(np.hstack(frames))
            continue

        for cam_idx in range(3):
            cam = cams_params[cam_idx]
            img = frames[cam_idx]

            # 1. people
            for pid in frame_data['person_id'].unique():
                if int(pid) == -1:
                    continue
                p_df = frame_data[frame_data['person_id'] == pid]
                col = COLORS[int(pid) % len(COLORS)]

                kps_3d = p_df[['X', 'Y', 'Z']].values
                kps_2d = project_3d_to_2d(kps_3d, cam['K'], cam['R'], cam['t'], cam['D'])

                pts_map = {}
                for name, pt in zip(p_df['joint'], kps_2d):
                    if np.all(np.isfinite(pt)):
                        pts_map[name] = pt

                for u_idx, v_idx in POSE_CONNECTIONS:
                    u_name = JOINT_NAMES[u_idx]
                    v_name = JOINT_NAMES[v_idx]
                    if u_name in pts_map and v_name in pts_map:
                        pt1 = to_int_point(pts_map[u_name])
                        pt2 = to_int_point(pts_map[v_name])
                        if pt1 is not None and pt2 is not None:
                            try:
                                cv2.line(img, pt1, pt2, col, 2)
                            except cv2.error as exc:
                                if not line_error_logged:
                                    print(f"[warn] cv2.line failed: {exc}")
                                    print(f"[warn] pt1={pt1} pt2={pt2}")
                                    line_error_logged = True

                for pt in pts_map.values():
                    center = to_int_point(pt)
                    if center is not None:
                        cv2.circle(img, center, 3, col, -1)

            # 2. ball
            ball_df = frame_data[frame_data['joint'] == 'ball']
            if not ball_df.empty:
                b_3d = ball_df[['X', 'Y', 'Z']].values
                b_2d = project_3d_to_2d(b_3d, cam['K'], cam['R'], cam['t'], cam['D'])
                for pt in b_2d:
                    center = to_int_point(pt)
                    if center is not None:
                        cv2.circle(img, center, 10, BALL_COLOR, -1)

        vw.write(np.hstack(frames))

    vw.release()
    for c in caps:
        c.release()

    if os.path.exists(temp_out):
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_out,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", args.output
        ], check=False)
        os.remove(temp_out)
        print(f"Finished: {args.output}")


if __name__ == "__main__":
    main()
