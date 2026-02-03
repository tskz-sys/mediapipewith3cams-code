#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_3dposeestimation.py

要件:
- 入力動画: ../ffmpeg/output/ 内の mp4
  ファイル名: [camnumber]match[gamenumber]_[matchnumber].mp4
    例: 1match1_2.mp4, 2match1_2.mp4, 3match1_2.mp4
  left=cam1, center=cam2, right=cam3 をセットで処理

- 出力: ./output/3dposeestimation/ に全てまとめて保存
  例:
    match1_2_raw.csv
    match1_2_fixed.csv
    match1_2_smoothed.csv
    match1_2_bbox.csv
    match1_2_viz.mp4

- 平滑化: 添付の2本のスクリプト（fix_pose_csv_adaptive.py相当 + smooth_csv.py相当）を
  この1本の中で連続実行できるように内蔵（オプションでON/OFF）

実行例:
  python batch_3dposeestimation.py \
    --input_dir ../ffmpeg/output \
    --calib_npz ./npz/11253cams_fixedd.npz \
    --out_dir ./output/3dposeestimation \
    --run_smoothing \
    --base_anchor_thr 0.5 \
    --force_reset_frames 10 \
    --smooth_window 5

注意:
- このスクリプトは「hybrid_dual_final_with_bbox.py（v5相当）」の処理をベースにしています。
- モデルやMediaPipe設定は下の USER CONFIG で変更できます。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# ==========================================================
# USER CONFIG (必要なら変更)
# ==========================================================
DET_MODEL = "yolo11x-pose.pt"  # Person
BALL_MODEL = "yolo11x.pt"      # Ball

INFERENCE_SIZE = 1920
CONF_LOW_LIMIT = 0.10

CONF_PERSON = 0.25
CONF_BALL = 0.15

MAX_PEOPLE = 2

MP_COMPLEXITY = 1
MIN_MP_CONF = 0.3

DRAW_DEBUG_BBOX = True

# MediaPipe Joint Names
# MediaPipe Joint Names (33 landmarks: 0..32)
# NOTE: 以前の版はリスト長が33未満になり得て IndexError の原因になっていました。
try:
    MP_JOINTS = [lm.name.lower() for lm in mp.solutions.pose.PoseLandmark]
except Exception:
    MP_JOINTS = [f"kp{i}" for i in range(33)]

# NOTE: 元コードに合わせて使用関節（2視点コスト計算用）
USE_JOINTS_IDX = [11, 12, 23, 24, 25, 26]

# 2カメラ一致を優先する設定
PAIR_COST_MAX = 0.2
ATTACH_COST_MAX = 0.2
MIN_JOINT_CONF = 0.5
CAM2_REPROJ_MAX = 0.08
PARALLAX_MIN_DEG = 2.0
# Default pair priority (tie-breaker)
PAIR_PRIORITY = [(0, 2), (0, 1), (1, 2)]

# Re-ID helpers
CORE_JOINTS = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
HAND_JOINTS = ['left_wrist', 'right_wrist', 'left_index', 'right_index']


# ==========================================================
# Calibration helpers
# ==========================================================
def get_inverse_transform(R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return R.T, -R.T @ T

def projection_error(P_ref: np.ndarray, K: np.ndarray, R: np.ndarray, T: np.ndarray) -> float:
    P_est = K @ np.hstack([R, T.reshape(3, 1)])
    denom = float(np.linalg.norm(P_ref))
    if denom < 1e-9:
        return float("inf")
    return float(np.linalg.norm(P_est - P_ref) / denom)

def choose_extrinsics(
    data: np.lib.npyio.NpzFile,
    cam_idx: int,
    K_raw: np.ndarray,
    R_raw: np.ndarray,
    t_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str, Optional[float], Optional[float]]:
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

def scale_camera_matrix(
    K: np.ndarray,
    dist: np.ndarray,
    target_w: int,
    target_h: int,
    orig_size: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    if orig_size is None:
        return K, dist
    orig_w, orig_h = orig_size
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05:
        return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx
    K_new[1, 1] *= sy
    K_new[0, 2] *= sx
    K_new[1, 2] *= sy
    return K_new, dist

def load_params_BR(
    npz_path: str,
    v1: str,
    v2: str,
    v3: str,
    invert_both: bool = False,
    invert_cam1: bool = False,
    invert_cam3: bool = False,
):
    d = np.load(npz_path, allow_pickle=True)

    def get_k(k_new, k_old):
        return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1", "cam_matrix1"), get_k("dist1", "dist_coeffs1")
    K2, D2 = get_k("K2", "cam_matrix2"), get_k("dist2", "dist_coeffs2")
    K3, D3 = get_k("K3", "cam_matrix3"), get_k("dist3", "dist_coeffs3")

    K1_raw = K1.copy()
    K3_raw = K3.copy()

    def get_calib_size(cam_idx: int) -> Optional[Tuple[int, int]]:
        for key in (f"image_size{cam_idx}", "image_size"):
            if key in d:
                size = d[key]
                if size is None:
                    continue
                w, h = int(size[0]), int(size[1])
                if w > 0 and h > 0:
                    return (w, h)
        return None

    def get_wh(v):
        c = cv2.VideoCapture(v)
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        return w, h

    w1, h1 = get_wh(v1)
    K1, D1 = scale_camera_matrix(K1, D1, w1, h1, get_calib_size(1))
    w2, h2 = get_wh(v2)
    K2, D2 = scale_camera_matrix(K2, D2, w2, h2, get_calib_size(2))
    w3, h3 = get_wh(v3)
    K3, D3 = scale_camera_matrix(K3, D3, w3, h3, get_calib_size(3))

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
        invert_cam1 = True
        invert_cam3 = True
        print("[calib] invert_both applied")

    if invert_cam1:
        R1, t1 = get_inverse_transform(R1, t1)
        print("[calib] invert_cam1 applied")
    if invert_cam3:
        R3, t3 = get_inverse_transform(R3, t3)
        print("[calib] invert_cam3 applied")

    P1_ext = np.hstack([R1, t1])
    P2_ext = np.hstack([R2, t2])
    P3_ext = np.hstack([R3, t3])

    return [(K1, D1, P1_ext), (K2, D2, P2_ext), (K3, D3, P3_ext)], [(R1, t1), (R2, t2), (R3, t3)]

def undistort_points(kps: List[Tuple[float, float]], K: np.ndarray, dist: np.ndarray):
    if len(kps) == 0:
        return []
    pts = np.array(kps, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=None).reshape(-1, 2)

def triangulate_DLT(Ps: List[np.ndarray], pts: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    A = []
    for P, (x, y) in zip(Ps, pts):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    X = X[:3] / X[3]

    errs = []
    for P, (x, y) in zip(Ps, pts):
        xh = P @ np.hstack([X, 1.0])
        if abs(xh[2]) < 1e-9:
            errs.append(1000.0)
            continue
        xp, yp = xh[0] / xh[2], xh[1] / xh[2]
        errs.append((xp - x) ** 2 + (yp - y) ** 2)
    return X, float(np.mean(errs))

def _camera_center_from_P(P: np.ndarray) -> np.ndarray:
    R = P[:, :3]
    t = P[:, 3].reshape(3, 1)
    return (-R.T @ t).reshape(3)

def _parallax_deg(C1: np.ndarray, C2: np.ndarray, X: np.ndarray) -> float:
    v1 = X - C1
    v2 = X - C2
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

def _select_best_cams(
    persons: List[Optional[dict]],
    cams: List[int],
    joint_idx: int,
    min_joint_conf: float,
    max_cams: int = 2,
) -> List[int]:
    candidates: List[Tuple[float, int]] = []
    for cam_idx in cams:
        p = persons[cam_idx]
        if p is None:
            continue
        conf = float(p["conf"][joint_idx])
        if not np.isfinite(conf):
            continue
        if conf < min_joint_conf:
            continue
        candidates.append((conf, cam_idx))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [cam_idx for _, cam_idx in candidates[:max_cams]]

def project_point_distorted(X_3d: np.ndarray, K: np.ndarray, D: np.ndarray, R: np.ndarray, t: np.ndarray) -> Optional[np.ndarray]:
    if np.any(np.isnan(X_3d)):
        return None
    object_points = np.array([X_3d], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    img_points, _ = cv2.projectPoints(object_points, rvec, t, K, D)
    return img_points[0][0]


# ==========================================================
# Detection helpers (padding + edge filter)
# Mask pixels outside the original bbox when running MediaPipe.
MASK_OUTSIDE_BBOX = True
BBOX_MARGIN_RATIO = 0.0
CLAMP_OUTSIDE_BBOX = True
# ==========================================================
def get_pose_from_crop_padded(full_img: np.ndarray, box: np.ndarray, pose_model):
    H_img, W_img = full_img.shape[:2]
    x1_raw, y1_raw, x2_raw, y2_raw = map(int, box)
    x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw

    margin_ratio = BBOX_MARGIN_RATIO
    w_box = x2 - x1
    h_box = y2 - y1
    pad_w = int(w_box * margin_ratio)
    pad_h = int(h_box * margin_ratio)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(W_img, x2 + pad_w)
    y2 = min(H_img, y2 + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None, None, (x1, y1, x2, y2)

    crop = full_img[y1:y2, x1:x2].copy()
    h_crop, w_crop = crop.shape[:2]

    if MASK_OUTSIDE_BBOX:
        bx1 = max(0, x1_raw - x1)
        by1 = max(0, y1_raw - y1)
        bx2 = min(w_crop, x2_raw - x1)
        by2 = min(h_crop, y2_raw - y1)
        masked = np.zeros_like(crop)
        if bx2 > bx1 and by2 > by1:
            masked[by1:by2, bx1:bx2] = crop[by1:by2, bx1:bx2]
        crop = masked

    size = max(h_crop, w_crop)
    padded_img = np.zeros((size, size, 3), dtype=np.uint8)

    ox = (size - w_crop) // 2
    oy = (size - h_crop) // 2
    padded_img[oy:oy + h_crop, ox:ox + w_crop] = crop

    crop_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(crop_rgb)

    if not results.pose_landmarks:
        return None, None, (x1, y1, x2, y2)

    kps = []
    confs = []
    for lm in results.pose_landmarks.landmark:
        px = lm.x * size
        py = lm.y * size

        cx = px - ox
        cy = py - oy

        global_x = cx + x1
        global_y = cy + y1

        conf = lm.visibility
        if global_x < x1_raw or global_x > x2_raw or global_y < y1_raw or global_y > y2_raw:
            conf = 0.0
            if CLAMP_OUTSIDE_BBOX:
                global_x = min(max(global_x, x1_raw), x2_raw)
                global_y = min(max(global_y, y1_raw), y2_raw)

        kps.append([global_x, global_y])
        confs.append(conf)

    return np.array(kps), np.array(confs), (x1, y1, x2, y2)


def process_person_yolo(img: np.ndarray, pose_model, mp_model, K: np.ndarray, D: np.ndarray, max_p: int):
    H, W = img.shape[:2]
    res = pose_model.predict(img, conf=CONF_LOW_LIMIT, imgsz=INFERENCE_SIZE, verbose=False, classes=[0])[0]

    people = []
    if res.boxes is None or len(res.boxes) == 0:
        return []

    indices = np.argsort(-res.boxes.conf.cpu().numpy())

    p_count = 0
    for idx in indices:
        conf = float(res.boxes.conf[idx].cpu().numpy())
        if conf < CONF_PERSON:
            continue
        if p_count >= max_p + 1:
            break

        box = res.boxes.xyxy[idx].cpu().numpy()
        yolo_kps = res.keypoints.xy[idx].cpu().numpy()

        mp_kps, mp_confs, _ = get_pose_from_crop_padded(img, box, mp_model)

        if mp_kps is not None:
            # edge filter
            for i in range(len(mp_kps)):
                kx, ky = mp_kps[i]
                margin = 5
                if kx < margin or kx > W - margin or ky < margin or ky > H - margin:
                    mp_confs[i] = 0.0

            norm_mp_kps = undistort_points(mp_kps, K, D)
            people.append({
                "type": "person",
                "box": box,
                "yolo_kps": yolo_kps,
                "kps_raw": mp_kps,
                "kps_norm": norm_mp_kps,
                "conf": mp_confs,
                "score": conf
            })
            p_count += 1

    return people


def detect_ball_yolo(img: np.ndarray, det_model, K: np.ndarray, D: np.ndarray):
    res = det_model.predict(img, conf=CONF_LOW_LIMIT, imgsz=INFERENCE_SIZE, verbose=False, classes=[32])[0]
    balls = []
    if res.boxes is None or len(res.boxes) == 0:
        return []

    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    for idx in indices:
        conf = float(res.boxes.conf[idx].cpu().numpy())
        if conf > CONF_BALL:
            box = res.boxes.xyxy[idx].cpu().numpy()
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            norm_pt = undistort_points([(cx, cy)], K, D)[0]
            balls.append({
                "center_raw": np.array([cx, cy]),
                "center_norm": norm_pt,
                "box": box,
                "type": "ball",
                "conf": conf
            })
    return balls


# ==========================================================
# Matching & 3D
# ==========================================================
def find_best_matches(
    ppl_lists: List[List[dict]],
    Ps: List[np.ndarray],
    pair_cost_max: float,
    attach_cost_max: float,
    min_joint_conf: float,
    cam2_reproj_max: Optional[float],
):
    # 2-view matching prioritized; choose the most consistent camera pair, attach 3rd if consistent.
    def calc_cost(persons, cam_indices):
        total_err = 0.0
        valid_joints = 0
        for j in USE_JOINTS_IDX:
            pts = []
            proj_matrices = []
            is_good = True
            for p, cam_idx in zip(persons, cam_indices):
                if p["conf"][j] < min_joint_conf:
                    is_good = False
                    break
                pts.append(p["kps_norm"][j])
                proj_matrices.append(Ps[cam_idx])
            if is_good:
                _, err = triangulate_DLT(proj_matrices, pts)
                total_err += err
                valid_joints += 1
        if valid_joints == 0:
            return 1000.0
        return total_err / valid_joints

    def calc_attach_reproj_error(p1, p2, p3, c1: int, c2: int, attach_cam: int) -> Optional[float]:
        errs = []
        for j in USE_JOINTS_IDX:
            if p1["conf"][j] < min_joint_conf:
                continue
            if p2["conf"][j] < min_joint_conf:
                continue
            if p3["conf"][j] < min_joint_conf:
                continue
            X, _ = triangulate_DLT(
                [Ps[c1], Ps[c2]],
                [p1["kps_norm"][j], p2["kps_norm"][j]],
            )
            proj = _project_normalized(Ps[attach_cam], X)
            if proj is None:
                continue
            obs = p3["kps_norm"][j]
            errs.append(float(np.linalg.norm(obs - proj)))
        if len(errs) < 2:
            return None
        return float(np.mean(errs))

    def build_candidates(c1: int, c2: int):
        candidates = []
        for i1, p1 in enumerate(ppl_lists[c1]):
            for i2, p2 in enumerate(ppl_lists[c2]):
                cost = calc_cost([p1, p2], [c1, c2])
                if cost < pair_cost_max:
                    candidates.append({'ids': (i1, i2), 'cost': cost})
        candidates.sort(key=lambda x: x['cost'])
        return candidates

    def estimate_pair_quality(candidates):
        used1 = set()
        used2 = set()
        selected = []
        for cand in candidates:
            i1, i2 = cand['ids']
            if i1 in used1 or i2 in used2:
                continue
            selected.append(cand)
            used1.add(i1)
            used2.add(i2)
        if not selected:
            return 0, float("inf")
        mean_cost = sum(c['cost'] for c in selected) / len(selected)
        return len(selected), mean_cost

    matched_people = []
    used_indices = [set() for _ in range(3)]

    pair_infos = []
    for priority, (c1, c2) in enumerate(PAIR_PRIORITY):
        candidates = build_candidates(c1, c2)
        match_count, mean_cost = estimate_pair_quality(candidates)
        pair_infos.append({
            'pair': (c1, c2),
            'candidates': candidates,
            'match_count': match_count,
            'mean_cost': mean_cost,
            'priority': priority,
        })

    pair_infos.sort(key=lambda x: (-x['match_count'], x['mean_cost'], x['priority']))

    for info in pair_infos:
        c1, c2 = info['pair']
        attach_cam = 3 - c1 - c2
        for cand in info['candidates']:
            idx1, idx2 = cand['ids']
            if idx1 in used_indices[c1] or idx2 in used_indices[c2]:
                continue

            p1 = ppl_lists[c1][idx1]
            p2 = ppl_lists[c2][idx2]
            persons = [None, None, None]
            persons[c1] = p1
            persons[c2] = p2
            cams = [c1, c2]

            best_attach_idx = None
            best_score = 999.0
            best_p = None
            for i3, p3 in enumerate(ppl_lists[attach_cam]):
                if i3 in used_indices[attach_cam]:
                    continue
                cost_a = calc_cost([p1, p3], [c1, attach_cam])
                cost_b = calc_cost([p2, p3], [c2, attach_cam])
                if cost_a < attach_cost_max and cost_b < attach_cost_max:
                    if cam2_reproj_max is not None and cam2_reproj_max > 0 and attach_cam == 1:
                        reproj_err = calc_attach_reproj_error(p1, p2, p3, c1, c2, attach_cam)
                        if reproj_err is None or reproj_err > cam2_reproj_max:
                            continue
                    score = cost_a + cost_b
                    if score < best_score:
                        best_score = score
                        best_attach_idx = i3
                        best_p = p3
            if best_attach_idx is not None:
                persons[attach_cam] = best_p
                cams = sorted([c1, c2, attach_cam])
                used_indices[attach_cam].add(best_attach_idx)

            matched_people.append({'persons': persons, 'cams': cams})
            used_indices[c1].add(idx1)
            used_indices[c2].add(idx2)

    return matched_people


def solve_3d_joints(
    persons: List[Optional[dict]],
    cams: List[int],
    Ps: List[np.ndarray],
    min_joint_conf: float,
    max_triang_err: Optional[float],
    parallax_min_deg: Optional[float],
) -> np.ndarray:
    # Hybrid-style triangulation: use all available cameras for each joint.
    kps_3d = np.full((33, 3), np.nan)
    for j in range(33):
        valid_pts = []
        valid_Ps = []
        for cam_idx in cams:
            p = persons[cam_idx]
            if p is None:
                continue
            conf = float(p["conf"][j])
            if not np.isfinite(conf) or conf < min_joint_conf:
                continue
            valid_pts.append(p["kps_norm"][j])
            valid_Ps.append(Ps[cam_idx])
        if len(valid_pts) >= 2:
            X, _ = triangulate_DLT(valid_Ps, valid_pts)
            if np.linalg.norm(X) < 50.0:
                kps_3d[j] = X
    return kps_3d


# ==========================================================
# Smoothing (内蔵)
#   - fix step: 閾値ゲート + 長欠損の強制復帰
#   - smooth step: 補間(短欠損) + rolling mean
# ==========================================================
def fix_pose_csv_adaptive_like(input_csv: str, output_csv: str, base_anchor_thr: float = 0.5, force_reset_frames: int = 10):
    df = pd.read_csv(input_csv)

    required_cols = ['frame', 'person_id', 'joint', 'X', 'Y', 'Z']
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"Missing required columns in {input_csv}. expected={required_cols}, got={list(df.columns)}")

    df = df.sort_values(['frame', 'person_id']).reset_index(drop=True)
    frames = sorted(df['frame'].unique())
    pids = sorted(df['person_id'].unique())
    joints = df['joint'].unique()

    fixed_rows = []

    for pid in pids:
        df_p = df[df['person_id'] == pid].copy()

        data_map: Dict[int, Dict[str, np.ndarray]] = {}
        for _, row in df_p.iterrows():
            f = int(row['frame'])
            j = row['joint']
            if not np.isnan(row['X']):
                data_map.setdefault(f, {})[j] = np.array([row['X'], row['Y'], row['Z']], dtype=float)

        anchors: Dict[str, np.ndarray] = {}
        missing_counts: Dict[str, int] = {}

        start_frame_idx = 0
        for i, f in enumerate(frames):
            if f in data_map:
                anchors = data_map[f].copy()
                for j in joints:
                    missing_counts[j] = 0
                start_frame_idx = i
                break

        if not anchors:
            continue

        for i in range(len(frames)):
            f = frames[i]
            if i < start_frame_idx:
                continue

            current_frame_data = data_map.get(f, {})

            for j in joints:
                has_input = (j in current_frame_data)
                valid_val = False
                out_val = np.array([np.nan, np.nan, np.nan], dtype=float)

                if has_input:
                    raw_pos = current_frame_data[j]

                    if j in anchors:
                        prev_pos = anchors[j]
                        dist = float(np.linalg.norm(raw_pos - prev_pos))

                        if dist < base_anchor_thr:
                            valid_val = True
                        elif missing_counts[j] > force_reset_frames:
                            valid_val = True
                    else:
                        valid_val = True

                    if valid_val:
                        out_val = raw_pos
                        anchors[j] = raw_pos
                        missing_counts[j] = 0
                    else:
                        missing_counts[j] += 1
                else:
                    missing_counts[j] += 1

                if not np.isnan(out_val[0]):
                    fixed_rows.append([f, pid, j, out_val[0], out_val[1], out_val[2]])

    out_df = pd.DataFrame(fixed_rows, columns=['frame', 'person_id', 'joint', 'X', 'Y', 'Z'])
    out_df.to_csv(output_csv, index=False)


def smooth_csv_like(input_csv: str, output_csv: str, window: int = 5, interpolate_limit: int = 3):
    df = pd.read_csv(input_csv)
    df = df.sort_values(['person_id', 'joint', 'frame'])

    smoothed_dfs = []

    all_frames = sorted(df['frame'].unique())
    full_idx = pd.Index(all_frames, name='frame')

    for (pid, joint), group in df.groupby(['person_id', 'joint']):
        g = group.set_index('frame')
        g = g.reindex(full_idx)

        g[['X', 'Y', 'Z']] = g[['X', 'Y', 'Z']].interpolate(method='linear', limit=interpolate_limit)
        g[['X', 'Y', 'Z']] = g[['X', 'Y', 'Z']].rolling(window=window, center=True, min_periods=1).mean()

        g['person_id'] = pid
        g['joint'] = joint
        g = g.reset_index()

        g = g.dropna(subset=['X', 'Y', 'Z'])
        smoothed_dfs.append(g)

    final_df = pd.concat(smoothed_dfs, ignore_index=True)
    final_df = final_df.sort_values(['frame', 'person_id', 'joint'])
    final_df.to_csv(output_csv, index=False)

def _project_normalized(P: np.ndarray, X: np.ndarray) -> Optional[np.ndarray]:
    Xh = np.hstack([X, 1.0])
    proj = P @ Xh
    if abs(proj[2]) < 1e-9:
        return None
    return np.array([proj[0] / proj[2], proj[1] / proj[2]], dtype=float)

def filter_by_reprojection_error(
    kps_3d: np.ndarray,
    persons: List[Optional[dict]],
    cams: List[int],
    Ps: List[np.ndarray],
    thresh: float,
    min_joint_conf: float = MIN_JOINT_CONF,
    use_all_cams: bool = False,
) -> np.ndarray:
    if thresh is None or thresh <= 0:
        return kps_3d
    if use_all_cams:
        cams_to_check = [idx for idx, p in enumerate(persons) if p is not None]
    else:
        cams_to_check = cams
    filtered = kps_3d.copy()
    for j in range(filtered.shape[0]):
        if np.isnan(filtered[j]).any():
            continue
        best_cams = _select_best_cams(persons, cams_to_check, j, min_joint_conf, max_cams=2)
        if not best_cams:
            continue
        errs = []
        for cam_idx in best_cams:
            p = persons[cam_idx]
            if p is None:
                continue
            obs = p["kps_norm"][j]
            proj = _project_normalized(Ps[cam_idx], filtered[j])
            if proj is None:
                continue
            err = float(np.linalg.norm(obs - proj))
            errs.append(err)
        if errs and max(errs) > thresh:
            filtered[j] = np.array([np.nan, np.nan, np.nan], dtype=float)
    return filtered

def apply_bbox_gate(
    kps_3d: np.ndarray,
    persons: List[Optional[dict]],
    cams_rt: List[dict],
    margin_ratio: float,
    min_joint_conf: float,
) -> np.ndarray:
    if margin_ratio is None or margin_ratio < 0:
        return kps_3d
    gated = kps_3d.copy()
    cams_to_check = [idx for idx, p in enumerate(persons) if p is not None]
    for j in range(gated.shape[0]):
        if np.isnan(gated[j]).any():
            continue
        best_cams = _select_best_cams(persons, cams_to_check, j, min_joint_conf, max_cams=2)
        if not best_cams:
            continue
        for cam_idx in best_cams:
            p = persons[cam_idx]
            if p is None:
                continue
            box = p["box"]
            x1, y1, x2, y2 = [float(v) for v in box]
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w_m = w * (1.0 + margin_ratio)
            h_m = h * (1.0 + margin_ratio)
            x1_m = cx - w_m / 2.0
            x2_m = cx + w_m / 2.0
            y1_m = cy - h_m / 2.0
            y2_m = cy + h_m / 2.0

            cam = cams_rt[cam_idx]
            proj = project_point_distorted(gated[j], cam["K"], cam["D"], cam["R"], cam["t"])
            if proj is None:
                continue
            px, py = float(proj[0]), float(proj[1])
            if not (x1_m <= px <= x2_m and y1_m <= py <= y2_m):
                gated[j] = np.array([np.nan, np.nan, np.nan], dtype=float)
                break
    return gated

def render_3d_plot_video(input_csv: str, output_mp4: str, fps: float) -> None:
    script = os.path.join(os.path.dirname(__file__), "plot_3d_simple_compare.py")
    fps_out = int(round(fps)) if fps and fps > 1 else 30
    out_dir = os.path.dirname(output_mp4)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        script,
        "-i",
        input_csv,
        "-o",
        output_mp4,
        "--cols",
        "1",
        "--fps",
        str(fps_out),
    ]
    subprocess.run(cmd, check=False)
    if not os.path.exists(output_mp4):
        raise RuntimeError(f"3D plot video not created: {output_mp4}")

def _compute_center(df_person: pd.DataFrame) -> Optional[np.ndarray]:
    subset = df_person[df_person['joint'].isin(CORE_JOINTS)][['X', 'Y', 'Z']]
    arr = subset.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        arr = df_person[['X', 'Y', 'Z']].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        return None
    return arr.mean(axis=0)

def _get_ball_pos(df_ball: pd.DataFrame, frame: int) -> Optional[np.ndarray]:
    rows = df_ball[df_ball['frame'] == frame]
    if rows.empty:
        return None
    for _, row in rows.iterrows():
        pos = np.array([row['X'], row['Y'], row['Z']], dtype=float)
        if np.isfinite(pos).all():
            return pos
    return None

def _hand_ball_dist(df_person: pd.DataFrame, ball_pos: Optional[np.ndarray]) -> float:
    if ball_pos is None or not np.isfinite(ball_pos).all():
        return float('inf')
    subset = df_person[df_person['joint'].isin(HAND_JOINTS)][['X', 'Y', 'Z']]
    arr = subset.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        return float('inf')
    dists = np.linalg.norm(arr - ball_pos, axis=1)
    return float(np.min(dists))

def _assign_by_continuity(center_a: np.ndarray, center_b: np.ndarray, last_centers: Dict[int, np.ndarray]):
    d_a0 = np.linalg.norm(center_a - last_centers[0])
    d_a1 = np.linalg.norm(center_a - last_centers[1])
    d_b0 = np.linalg.norm(center_b - last_centers[0])
    d_b1 = np.linalg.norm(center_b - last_centers[1])
    if d_a0 + d_b1 <= d_a1 + d_b0:
        return 0, 1
    return 1, 0
def reassign_ids_by_ball_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Empty CSV data: {input_csv}")

    for col in ['X', 'Y', 'Z', 'frame', 'person_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['frame', 'person_id'])
    df['frame'] = df['frame'].astype(int)
    df['person_id'] = df['person_id'].astype(int)

    ball_mask = (df['joint'] == 'ball') | (df['person_id'] == -1)
    df_ball = df[ball_mask].copy()
    df_person = df[~ball_mask].copy()

    if df_person.empty:
        df.to_csv(output_csv, index=False)
        return

    frames = sorted(df_person['frame'].unique())
    output_parts = []
    pending = []
    last_centers: Optional[Dict[int, np.ndarray]] = None

    def select_two_candidates(frame_df: pd.DataFrame) -> List[int]:
        pids = sorted(frame_df['person_id'].unique())
        if len(pids) <= 2:
            return pids
        counts = frame_df.groupby('person_id')['joint'].count()
        pids = sorted(counts.index.tolist(), key=lambda pid: (-counts.loc[pid], pid))
        return pids[:2]

    for f in frames:
        frame_df = df_person[df_person['frame'] == f]
        if frame_df.empty:
            continue
        pids = select_two_candidates(frame_df)
        if len(pids) == 0:
            continue

        if len(pids) == 1:
            pid = pids[0]
            person_df = frame_df[frame_df['person_id'] == pid].copy()
            center = _compute_center(person_df)
            pending.append({'frame': f, 'df': person_df, 'center': center})
            continue

        person_a = frame_df[frame_df['person_id'] == pids[0]].copy()
        person_b = frame_df[frame_df['person_id'] == pids[1]].copy()
        center_a = _compute_center(person_a)
        center_b = _compute_center(person_b)

        ball_pos = _get_ball_pos(df_ball, f)
        dist_a = _hand_ball_dist(person_a, ball_pos)
        dist_b = _hand_ball_dist(person_b, ball_pos)
        if np.isfinite(dist_a) and np.isfinite(dist_b):
            if dist_a <= dist_b:
                id_a, id_b = 1, 0
            else:
                id_a, id_b = 0, 1
        elif last_centers is not None and center_a is not None and center_b is not None:
            id_a, id_b = _assign_by_continuity(center_a, center_b, last_centers)
        else:
            id_a, id_b = 0, 1

        if pending:
            last_pending_center = pending[-1]['center']
            pending_id = None
            if last_pending_center is not None and center_a is not None and center_b is not None:
                d_pa = np.linalg.norm(last_pending_center - center_a)
                d_pb = np.linalg.norm(last_pending_center - center_b)
                pending_id = id_a if d_pa <= d_pb else id_b
            elif last_centers is not None and last_pending_center is not None:
                d0 = np.linalg.norm(last_pending_center - last_centers[0])
                d1 = np.linalg.norm(last_pending_center - last_centers[1])
                pending_id = 0 if d0 <= d1 else 1
            if pending_id is None:
                pending_id = id_a
            for item in pending:
                df_p = item['df'].copy()
                df_p['person_id'] = pending_id
                output_parts.append(df_p)
            pending.clear()

        person_a['person_id'] = id_a
        person_b['person_id'] = id_b
        output_parts.append(person_a)
        output_parts.append(person_b)

        if center_a is not None and center_b is not None:
            if id_a == 0:
                last_centers = {0: center_a, 1: center_b}
            else:
                last_centers = {0: center_b, 1: center_a}

    if pending:
        pending_id = 0
        last_pending_center = pending[-1]['center']
        if last_centers is not None and last_pending_center is not None:
            d0 = np.linalg.norm(last_pending_center - last_centers[0])
            d1 = np.linalg.norm(last_pending_center - last_centers[1])
            pending_id = 0 if d0 <= d1 else 1
        for item in pending:
            df_p = item['df'].copy()
            df_p['person_id'] = pending_id
            output_parts.append(df_p)

    out_df = pd.concat([df_ball] + output_parts, ignore_index=True)
    sort_cols = [c for c in ['frame', 'person_id', 'joint'] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols)
    out_df.to_csv(output_csv, index=False)

def filter_pose_jump_frames(input_csv: str, output_csv: str, center_thresh: float):
    df = pd.read_csv(input_csv)
    if df.empty:
        df.to_csv(output_csv, index=False)
        return

    for col in ['X', 'Y', 'Z', 'frame', 'person_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['frame', 'person_id'])
    df['frame'] = df['frame'].astype(int)
    df['person_id'] = df['person_id'].astype(int)

    ball_mask = (df['joint'] == 'ball') | (df['person_id'] == -1)
    df_ball = df[ball_mask].copy()
    df_person = df[~ball_mask].copy()

    if df_person.empty:
        df.to_csv(output_csv, index=False)
        return

    drop_frames_by_pid: Dict[int, List[int]] = {}
    for pid in sorted(df_person['person_id'].unique()):
        if pid < 0:
            continue
        dfp = df_person[df_person['person_id'] == pid]
        frames = sorted(dfp['frame'].unique())
        if len(frames) < 3:
            continue

        centers: List[Optional[np.ndarray]] = []
        for f in frames:
            frame_df = dfp[dfp['frame'] == f]
            centers.append(_compute_center(frame_df))

        drop_frames: List[int] = []
        for i in range(1, len(frames) - 1):
            c_prev = centers[i - 1]
            c_cur = centers[i]
            c_next = centers[i + 1]
            if c_prev is None or c_cur is None or c_next is None:
                continue
            d_prev = float(np.linalg.norm(c_cur - c_prev))
            d_next = float(np.linalg.norm(c_cur - c_next))
            if d_prev > center_thresh and d_next > center_thresh:
                drop_frames.append(frames[i])

        if drop_frames:
            drop_frames_by_pid[pid] = drop_frames

    if drop_frames_by_pid:
        drop_mask = np.zeros(len(df_person), dtype=bool)
        for pid, frames in drop_frames_by_pid.items():
            if not frames:
                continue
            drop_mask |= ((df_person['person_id'] == pid) & (df_person['frame'].isin(frames)))
            preview = ",".join(str(f) for f in frames[:5])
            suffix = "..." if len(frames) > 5 else ""
            print(f"[jump_filter] pid={pid} drop={len(frames)} frames: {preview}{suffix}")
        df_person = df_person[~drop_mask]

    out_df = pd.concat([df_ball, df_person], ignore_index=True)
    sort_cols = [c for c in ['frame', 'person_id', 'joint'] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols)
    out_df.to_csv(output_csv, index=False)


# ==========================================================
# One-run pipeline
# ==========================================================
@dataclass
class RunOutputs:
    raw_csv: str
    reid_csv: Optional[str]
    jump_filtered_csv: Optional[str]
    fixed_csv: Optional[str]
    smoothed_csv: Optional[str]
    bbox_csv: str
    viz_mp4: str
    plot3d_mp4: Optional[str]


def run_one_triplet(
    video_left: str,
    video_center: str,
    video_right: str,
    calib_npz: str,
    invert_both: bool,
    invert_cam1: bool,
    invert_cam3: bool,
    out_dir: str,
    out_stem: str,
    run_smoothing: bool,
    base_anchor_thr: float,
    force_reset_frames: int,
    smooth_window: int,
    interpolate_limit: int,
    reassign_ids_by_ball: bool,
    reproj_err_thresh: Optional[float],
    jump_center_thresh: Optional[float],
    pair_cost_max: float,
    attach_cost_max: float,
    min_joint_conf: float,
    max_triang_err: Optional[float],
    cam2_reproj_max: Optional[float],
    parallax_min_deg: Optional[float],
    reproj_all_cams: bool,
    bbox_gate_margin: Optional[float],
    min_cams: int,
) -> RunOutputs:
    os.makedirs(out_dir, exist_ok=True)

    out_raw_csv = os.path.join(out_dir, f"{out_stem}_raw.csv")
    out_reid_csv = os.path.join(out_dir, f"{out_stem}_reid.csv")
    out_jump_csv = os.path.join(out_dir, f"{out_stem}_jump_filtered.csv")
    out_fixed_csv = os.path.join(out_dir, f"{out_stem}_fixed.csv")
    out_smoothed_csv = os.path.join(out_dir, f"{out_stem}_smoothed.csv")
    out_bbox_csv = os.path.join(out_dir, f"{out_stem}_bbox.csv")
    out_video = os.path.join(out_dir, f"{out_stem}_viz.mp4")
    out_3d_video = os.path.join(out_dir, "video", f"{out_stem}_3d.mp4")

    if not os.path.exists(calib_npz):
        raise FileNotFoundError(f"CALIB_NPZ not found: {calib_npz}")

    cam_params_full, cam_rt = load_params_BR(
        calib_npz,
        video_left,
        video_center,
        video_right,
        invert_both=invert_both,
        invert_cam1=invert_cam1,
        invert_cam3=invert_cam3,
    )
    Ps = [p[2] for p in cam_params_full]
    cams_rt = [
        {"K": cam_params_full[i][0], "D": cam_params_full[i][1], "R": cam_rt[i][0], "t": cam_rt[i][1]}
        for i in range(3)
    ]

    caps = [cv2.VideoCapture(v) for v in [video_left, video_center, video_right]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Models
    yolo_pose = YOLO(DET_MODEL)
    yolo_det = YOLO(BALL_MODEL)
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(static_image_mode=True, model_complexity=MP_COMPLEXITY, min_detection_confidence=MIN_MP_CONF)

    # Writers
    f_csv = open(out_raw_csv, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z"])

    f_bbox = open(out_bbox_csv, 'w', newline='')
    writer_bbox = csv.writer(f_bbox)
    writer_bbox.writerow(["frame", "person_id", "cam_idx", "x1", "y1", "x2", "y2"])

    temp_out = out_video + ".temp.mp4"
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W * 3, H))

    mp_conn = mp.solutions.pose.POSE_CONNECTIONS
    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]

    prev_centers: Dict[int, np.ndarray] = {}

    for i in tqdm(range(total_frames), desc=out_stem):
        frames = []
        ok = True
        for c in caps:
            ret, f = c.read()
            if not ret or f is None:
                ok = False
                break
            frames.append(f)
        if not ok or len(frames) < 3:
            break

        ppl_lists = []
        ball_lists = []

        for cam_i, f in enumerate(frames):
            K, D, _ = cam_params_full[cam_i]
            p = process_person_yolo(f, yolo_pose, pose_estimator, K, D, MAX_PEOPLE)
            ppl_lists.append(p)
            b = detect_ball_yolo(f, yolo_det, K, D)
            ball_lists.append(b)

            if DRAW_DEBUG_BBOX:
                for per in p:
                    bx = list(map(int, per["box"]))
                    cv2.rectangle(f, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 2)
                for bal in b:
                    bx = list(map(int, bal["box"]))
                    cv2.rectangle(f, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)

        matched_results = find_best_matches(
            ppl_lists,
            Ps,
            pair_cost_max=pair_cost_max,
            attach_cost_max=attach_cost_max,
            min_joint_conf=min_joint_conf,
            cam2_reproj_max=cam2_reproj_max,
        )
        if min_cams > 1:
            matched_results = [m for m in matched_results if len(m.get("cams", [])) >= min_cams]

        final_persons = [None] * MAX_PEOPLE

        candidates = []
        for match in matched_results:
            persons = match['persons']
            active_cams = match['cams']
            kps_3d = solve_3d_joints(
                persons,
                active_cams,
                Ps,
                min_joint_conf=min_joint_conf,
                max_triang_err=max_triang_err,
                parallax_min_deg=parallax_min_deg,
            )
            kps_3d = filter_by_reprojection_error(
                kps_3d,
                persons,
                active_cams,
                Ps,
                reproj_err_thresh,
                min_joint_conf=min_joint_conf,
                use_all_cams=reproj_all_cams,
            )
            kps_3d = apply_bbox_gate(
                kps_3d,
                persons,
                cams_rt,
                margin_ratio=bbox_gate_margin,
                min_joint_conf=min_joint_conf,
            )

            valid_pts = kps_3d[~np.isnan(kps_3d).any(axis=1)]
            center = np.mean(valid_pts, axis=0) if len(valid_pts) > 0 else np.array([0.0, 0.0, 0.0], dtype=float)

            candidates.append({'match_data': match, 'center': center, 'kps_3d': kps_3d})

        if i == 0 or len(prev_centers) == 0:
            for idx, cand in enumerate(candidates):
                if idx < MAX_PEOPLE:
                    final_persons[idx] = cand
                    prev_centers[idx] = cand['center']
        else:
            used_candidates = set()
            for pid in range(MAX_PEOPLE):
                if pid not in prev_centers:
                    continue
                last_pos = prev_centers[pid]
                best_dist = 200.0
                best_idx = -1
                for c_idx, cand in enumerate(candidates):
                    if c_idx in used_candidates:
                        continue
                    dist = float(np.linalg.norm(cand['center'] - last_pos))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = c_idx
                if best_idx != -1:
                    final_persons[pid] = candidates[best_idx]
                    prev_centers[pid] = candidates[best_idx]['center']
                    used_candidates.add(best_idx)

            for c_idx, cand in enumerate(candidates):
                if c_idx not in used_candidates:
                    for pid in range(MAX_PEOPLE):
                        if final_persons[pid] is None:
                            final_persons[pid] = cand
                            prev_centers[pid] = cand['center']
                            break

        # write persons
        for pid in range(MAX_PEOPLE):
            cand = final_persons[pid]
            if cand is None:
                continue

            kps_3d = cand['kps_3d']
            match = cand['match_data']
            persons = match['persons']
            col = colors[pid % len(colors)]

            for j in range(33):
                if not np.isnan(kps_3d[j][0]):
                    jname = MP_JOINTS[j] if j < len(MP_JOINTS) else f"kp{j}"
                    writer.writerow([i, pid, jname, float(kps_3d[j][0]), float(kps_3d[j][1]), float(kps_3d[j][2])])

            for cam_idx in range(3):
                p_data = persons[cam_idx]
                if p_data is None:
                    continue
                box = p_data["box"]
                writer_bbox.writerow([i, pid, cam_idx, float(box[0]), float(box[1]), float(box[2]), float(box[3])])

            # draw skeleton
            for cam_idx in range(3):
                p_data = persons[cam_idx]
                if p_data is None:
                    continue
                frame = frames[cam_idx]
                kps_raw = p_data["kps_raw"]
                cam = cams_rt[cam_idx]
                rvec, _ = cv2.Rodrigues(cam["R"])
                tvec = cam["t"]
                K = cam["K"]
                D = cam["D"]

                if "yolo_kps" in p_data:
                    for ykp in p_data["yolo_kps"]:
                        cv2.circle(frame, (int(ykp[0]), int(ykp[1])), 3, (0, 255, 255), -1)

                for u, v in mp_conn:
                    if u < len(kps_raw) and v < len(kps_raw):
                        if p_data["conf"][u] > 0.3 and p_data["conf"][v] > 0.3:
                            pt1 = (int(kps_raw[u][0]), int(kps_raw[u][1]))
                            pt2 = (int(kps_raw[v][0]), int(kps_raw[v][1]))
                            cv2.line(frame, pt1, pt2, col, 2)

                for j in range(33):
                    X, Y, Z = kps_3d[j]
                    if not np.isfinite([X, Y, Z]).all():
                        continue
                    cam_pt = cam["R"] @ np.array([X, Y, Z], dtype=np.float64).reshape(3, 1) + tvec
                    if cam_pt[2, 0] <= 0:
                        continue
                    img_pts, _ = cv2.projectPoints(
                        np.array([[X, Y, Z]], dtype=np.float64),
                        rvec,
                        tvec,
                        K,
                        D,
                    )
                    rx_f = float(img_pts[0][0][0])
                    ry_f = float(img_pts[0][0][1])
                    if not np.isfinite(rx_f) or not np.isfinite(ry_f):
                        continue
                    rx = int(round(rx_f))
                    ry = int(round(ry_f))
                    try:
                        cv2.circle(frame, (rx, ry), 4, (0, 0, 255), -1)
                    except cv2.error:
                        continue

                box = p_data["box"]
                cv2.putText(frame, f"ID:{pid}", (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

        # ball 3D
        valid_balls = []
        valid_ball_Ps = []
        ball_boxes_for_draw = [None] * 3
        for cam_idx in range(3):
            if len(ball_lists[cam_idx]) > 0:
                b = ball_lists[cam_idx][0]
                valid_balls.append(b["center_norm"])
                valid_ball_Ps.append(Ps[cam_idx])
                ball_boxes_for_draw[cam_idx] = b["box"]

        if len(valid_balls) >= 2:
            X_b, err_b = triangulate_DLT(valid_ball_Ps, valid_balls)
            if np.linalg.norm(X_b) < 50.0 and err_b < 20.0:
                writer.writerow([i, -1, "ball", float(X_b[0]), float(X_b[1]), float(X_b[2])])
                for cam_idx in range(3):
                    box = ball_boxes_for_draw[cam_idx]
                    if box is not None:
                        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                        cv2.circle(frames[cam_idx], (cx, cy), 10, (0, 165, 255), -1)
                        cv2.putText(frames[cam_idx], "BALL 3D", (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        vw.write(np.hstack(frames))

    # close
    f_csv.close()
    f_bbox.close()
    pose_estimator.close()
    vw.release()
    for c in caps:
        c.release()

    # finalize mp4
    if os.path.exists(temp_out):
        subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", out_video], check=False)
        os.remove(temp_out)

    reid_csv_path = None
    fixed_csv_path = None
    smoothed_csv_path = None
    smoothing_input_csv = out_raw_csv
    jump_csv_path: Optional[str] = None

    if reassign_ids_by_ball:
        reassign_ids_by_ball_csv(out_raw_csv, out_reid_csv)
        reid_csv_path = out_reid_csv
        smoothing_input_csv = out_reid_csv

    if jump_center_thresh is not None and jump_center_thresh > 0:
        filter_pose_jump_frames(smoothing_input_csv, out_jump_csv, jump_center_thresh)
        jump_csv_path = out_jump_csv
        smoothing_input_csv = out_jump_csv

    if run_smoothing:
        fix_pose_csv_adaptive_like(smoothing_input_csv, out_fixed_csv, base_anchor_thr=base_anchor_thr, force_reset_frames=force_reset_frames)
        smooth_csv_like(out_fixed_csv, out_smoothed_csv, window=smooth_window, interpolate_limit=interpolate_limit)
        fixed_csv_path = out_fixed_csv
        smoothed_csv_path = out_smoothed_csv

    csv_for_3d = smoothed_csv_path or smoothing_input_csv
    plot3d_mp4 = None
    if csv_for_3d and os.path.exists(csv_for_3d):
        render_3d_plot_video(csv_for_3d, out_3d_video, fps=fps)
        plot3d_mp4 = out_3d_video

    return RunOutputs(
        raw_csv=out_raw_csv,
        reid_csv=reid_csv_path,
        jump_filtered_csv=jump_csv_path,
        fixed_csv=fixed_csv_path,
        smoothed_csv=smoothed_csv_path,
        bbox_csv=out_bbox_csv,
        viz_mp4=out_video,
        plot3d_mp4=plot3d_mp4,
    )


# ==========================================================
# Batch discovery
# ==========================================================
PATTERN = re.compile(r"^(?P<cam>[123])match(?P<game>\d+)_(?P<match>\d+)\.mp4$", re.IGNORECASE)

def discover_triplets(input_dir: str) -> List[Tuple[int, int, str, str, str]]:
    """
    return list of (game, match, left_path, center_path, right_path)
    """
    groups: Dict[Tuple[int, int], Dict[int, str]] = {}

    for name in os.listdir(input_dir):
        m = PATTERN.match(name)
        if not m:
            continue
        cam = int(m.group("cam"))
        game = int(m.group("game"))
        match = int(m.group("match"))
        key = (game, match)
        groups.setdefault(key, {})[cam] = os.path.join(input_dir, name)

    triplets = []
    for (game, match), d in groups.items():
        if 1 in d and 2 in d and 3 in d:
            triplets.append((game, match, d[1], d[2], d[3]))
    triplets.sort(key=lambda x: (x[0], x[1]))
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="../ffmpeg/output", help="input videos dir")
    ap.add_argument("--calib_npz", type=str, required=True, help="calibration npz")
    ap.add_argument("--invert_both", action="store_true", help="apply extra invert to calibration extrinsics")
    ap.add_argument("--invert_cam1", action="store_true", help="invert only cam1 extrinsics")
    ap.add_argument("--invert_cam3", action="store_true", help="invert only cam3 extrinsics")
    ap.add_argument("--out_dir", type=str, default="./output/3dposeestimation", help="output base dir")
    ap.add_argument("--run_smoothing", action="store_true", help="also run fix+smooth after raw csv")
    ap.add_argument("--base_anchor_thr", type=float, default=0.5, help="fix step: normal move tolerance (m)")
    ap.add_argument("--force_reset_frames", type=int, default=10, help="fix step: force reset after missing this many frames")
    ap.add_argument("--smooth_window", type=int, default=5, help="smooth step: rolling mean window")
    ap.add_argument("--interpolate_limit", type=int, default=3, help="smooth step: linear interpolate limit (frames)")
    ap.add_argument("--reassign_ids_by_ball", action="store_true", help="reassign IDs after 3D (ball-nearest=ID1)")
    ap.add_argument("--reproj_err_thresh", type=float, default=None, help="drop joints with reprojection error over this threshold")
    ap.add_argument("--jump_center_thresh", type=float, default=None, help="drop frames with center jump vs prev/next (m)")
    ap.add_argument("--pair_cost_max", type=float, default=PAIR_COST_MAX, help="max cost for 2-view match (normalized)")
    ap.add_argument("--attach_cost_max", type=float, default=ATTACH_COST_MAX, help="max cost when attaching 3rd view (normalized)")
    ap.add_argument("--min_joint_conf", type=float, default=MIN_JOINT_CONF, help="min joint confidence for matching/triangulation")
    ap.add_argument("--max_triang_err", type=float, default=None, help="drop joints with DLT reprojection error over this (normalized)")
    ap.add_argument("--cam2_reproj_max", type=float, default=CAM2_REPROJ_MAX, help="attach cam2 only if reprojection error is below this (normalized)")
    ap.add_argument("--parallax_min_deg", type=float, default=PARALLAX_MIN_DEG, help="drop joints with parallax angle below this (deg)")
    ap.add_argument("--reproj_all_cams", action="store_true", help="also validate reprojection against non-triangulated cameras")
    ap.add_argument("--bbox_gate_margin", type=float, default=None, help="drop joints projected outside bbox (margin ratio, e.g. 0.1)")
    ap.add_argument("--min_cams", type=int, default=2, help="minimum cameras required per match (2 or 3)")

    ap.add_argument("--only_game", type=int, default=None, help="process only this gamenumber")
    ap.add_argument("--only_match", type=int, default=None, help="process only this matchnumber")

    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"input_dir not found: {args.input_dir}")

    triplets = discover_triplets(args.input_dir)
    if args.only_game is not None:
        triplets = [t for t in triplets if t[0] == args.only_game]
    if args.only_match is not None:
        triplets = [t for t in triplets if t[1] == args.only_match]

    if len(triplets) == 0:
        print("No valid triplets found. Expected filenames like 1match1_2.mp4, 2match1_2.mp4, 3match1_2.mp4")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Found {len(triplets)} triplets in {args.input_dir}")
    for game, match, v1, v2, v3 in triplets:
        stem = f"match{game}_{match}"
        print(f"\n=== Processing {stem} ===")
        print(f"  left  : {os.path.basename(v1)}")
        print(f"  center: {os.path.basename(v2)}")
        print(f"  right : {os.path.basename(v3)}")

        outs = run_one_triplet(
            video_left=v1,
            video_center=v2,
            video_right=v3,
            calib_npz=args.calib_npz,
            invert_both=args.invert_both,
            invert_cam1=args.invert_cam1,
            invert_cam3=args.invert_cam3,
            out_dir=args.out_dir,
            out_stem=stem,
            run_smoothing=args.run_smoothing,
            base_anchor_thr=args.base_anchor_thr,
            force_reset_frames=args.force_reset_frames,
            smooth_window=args.smooth_window,
            interpolate_limit=args.interpolate_limit,
            reassign_ids_by_ball=args.reassign_ids_by_ball,
            reproj_err_thresh=args.reproj_err_thresh,
            jump_center_thresh=args.jump_center_thresh,
            pair_cost_max=args.pair_cost_max,
            attach_cost_max=args.attach_cost_max,
            min_joint_conf=args.min_joint_conf,
            max_triang_err=args.max_triang_err,
            cam2_reproj_max=args.cam2_reproj_max,
            parallax_min_deg=args.parallax_min_deg,
            reproj_all_cams=args.reproj_all_cams,
            bbox_gate_margin=args.bbox_gate_margin,
            min_cams=args.min_cams,
        )

        print(f"  -> raw     : {outs.raw_csv}")
        if outs.reid_csv:
            print(f"  -> reid    : {outs.reid_csv}")
        if outs.jump_filtered_csv:
            print(f"  -> jumpflt : {outs.jump_filtered_csv}")
        print(f"  -> bbox    : {outs.bbox_csv}")
        print(f"  -> viz     : {outs.viz_mp4}")
        if outs.plot3d_mp4:
            print(f"  -> 3dplot  : {outs.plot3d_mp4}")
        if args.run_smoothing:
            print(f"  -> fixed   : {outs.fixed_csv}")
            print(f"  -> smoothed: {outs.smoothed_csv}")

if __name__ == "__main__":
    main()
