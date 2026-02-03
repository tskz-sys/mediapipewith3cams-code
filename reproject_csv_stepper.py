#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step through frames and overlay reprojected 3D CSV points on the 3 camera videos.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd


IGNORE_JOINTS = {
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
}

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_ankle", "right_foot_index"),
    ("left_wrist", "left_pinky"), ("left_wrist", "left_index"), ("left_wrist", "left_thumb"),
    ("right_wrist", "right_pinky"), ("right_wrist", "right_index"), ("right_wrist", "right_thumb"),
]

COLORS = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    -1: (0, 165, 255),
}


def get_inverse_transform(R: np.ndarray, T: np.ndarray):
    return R.T, -R.T @ T


def scale_camera_matrix(K: np.ndarray, dist: np.ndarray, target_w: int, target_h: int):
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


def load_params_BR(npz_path: str, v1: str, v2: str, v3: str):
    d = np.load(npz_path, allow_pickle=True)

    def get_k(k_new, k_old):
        return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1", "cam_matrix1"), get_k("dist1", "dist_coeffs1")
    K2, D2 = get_k("K2", "cam_matrix2"), get_k("dist2", "dist_coeffs2")
    K3, D3 = get_k("K3", "cam_matrix3"), get_k("dist3", "dist_coeffs3")

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
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3, 1))

    R2 = np.eye(3)
    t2 = np.zeros((3, 1))

    R3_raw = d["R3"]
    t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3, 1))

    return [(K1, D1, R1, t1), (K2, D2, R2, t2), (K3, D3, R3, t3)]


def project_point_distorted(X_3d: np.ndarray, K, D, R, t):
    object_points = np.array([X_3d], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    img_points, _ = cv2.projectPoints(object_points, rvec, t, K, D)
    u, v = img_points[0][0]
    return int(u), int(v)


def load_csv_data(csv_path: str, show_face: bool):
    df = pd.read_csv(csv_path)
    data = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        pid = int(row["person_id"])
        jname = str(row["joint"])
        if not show_face and jname in IGNORE_JOINTS:
            continue
        x, y, z = row["X"], row["Y"], row["Z"]
        if not np.isfinite([x, y, z]).all():
            continue
        data.setdefault(f, {}).setdefault(pid, {})[jname] = np.array([x, y, z], dtype=float)
    return data


def draw_skeleton(img, proj_2d, color):
    for j1, j2 in SKELETON_CONNECTIONS:
        if j1 in proj_2d and j2 in proj_2d:
            cv2.line(img, proj_2d[j1], proj_2d[j2], color, 2)
    for jname, pt in proj_2d.items():
        if jname == "ball":
            cv2.circle(img, pt, 8, (0, 165, 255), -1)
            cv2.putText(img, "Ball", (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.circle(img, pt, 4, color, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproject 3D CSV onto videos (stepper)")
    parser.add_argument("--input_csv", required=True, help="Input 3D CSV")
    parser.add_argument("--input_dir", required=True, help="Input video dir")
    parser.add_argument("--calib_npz", required=True, help="Calibration npz")
    parser.add_argument("--game", type=int, required=True, help="Game number")
    parser.add_argument("--match", type=int, required=True, help="Match number")
    parser.add_argument("--start", type=int, default=None, help="Start frame (default: first)")
    parser.add_argument("--show_face", action="store_true", help="Show face joints")
    args = parser.parse_args()

    stem = f"match{args.game}_{args.match}"
    v1 = os.path.join(args.input_dir, f"1{stem}.mp4")
    v2 = os.path.join(args.input_dir, f"2{stem}.mp4")
    v3 = os.path.join(args.input_dir, f"3{stem}.mp4")

    for v in (v1, v2, v3):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Video not found: {v}")
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")
    if not os.path.exists(args.calib_npz):
        raise FileNotFoundError(f"NPZ not found: {args.calib_npz}")

    cams = load_params_BR(args.calib_npz, v1, v2, v3)
    caps = [cv2.VideoCapture(v) for v in (v1, v2, v3)]
    total_frames = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps))

    csv_data = load_csv_data(args.input_csv, args.show_face)
    frames = sorted(csv_data.keys()) if csv_data else list(range(total_frames))

    if args.start is None:
        idx = 0
    else:
        target = int(args.start)
        idx = int(np.argmin([abs(f - target) for f in frames]))

    cv2.namedWindow("Reprojection Viewer", cv2.WINDOW_NORMAL)

    while True:
        frame_idx = int(frames[idx])
        frame_idx = min(frame_idx, total_frames - 1)
        imgs = []
        for cam_i, cap in enumerate(caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame = np.zeros((h, w, 3), dtype=np.uint8)

            if frame_idx in csv_data:
                persons = csv_data[frame_idx]
                K, D, R, t = cams[cam_i]
                for pid, joints in persons.items():
                    col = COLORS.get(pid, (0, 255, 255))
                    proj_2d = {}
                    for jname, pos_3d in joints.items():
                        uv = project_point_distorted(pos_3d, K, D, R, t)
                        proj_2d[jname] = uv
                    draw_skeleton(frame, proj_2d, col)

            cv2.putText(frame, f"Cam{cam_i+1} Frame:{frame_idx}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            imgs.append(frame)

        combined = np.hstack(imgs)
        cv2.imshow("Reprojection Viewer", combined)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("d"), ord("l")) or key == 83:
            idx = min(idx + 1, len(frames) - 1)
        elif key in (ord("a"), ord("j")) or key == 81:
            idx = max(idx - 1, 0)
        elif key == ord("n"):
            idx = min(idx + 10, len(frames) - 1)
        elif key == ord("p"):
            idx = max(idx - 10, len(frames) - 1)

    for c in caps:
        c.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
