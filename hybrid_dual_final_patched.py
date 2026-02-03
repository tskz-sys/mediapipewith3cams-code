#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# YOLO + MediaPipe Hybrid [v5 PADDING & EDGE FILTER]
#   - 修正1: 画面端で切れた画像を無理やり伸ばさず、黒帯(パディング)を入れてアスペクト比を維持
#            → これにより「潰れた姿勢」や「太った姿勢」になるのを防ぐ
#   - 修正2: 画面外(0未満やW以上)にあると推定された関節の信頼度を強制的に0にする
#            → これにより、見えていない足を無理やり3D化して精度が落ちるのを防ぐ
# ==========================================================

import os
import csv
import argparse
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from dataclasses import dataclass
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# ★ ユーザー設定
# ==========================================
VIDEO_LEFT   = "../ffmpeg/movie/1208experiment/sync1/1match1_1.mp4"
VIDEO_CENTER = "../ffmpeg/movie/1208experiment/sync1/2match1_1.mp4"
VIDEO_RIGHT  = "../ffmpeg/movie/1208experiment/sync1/3match1_1.mp4"

CALIB_NPZ = "../calibrationwith3cams/output/1208_3cams_center_cam2.npz"

OUT_CSV   = "./output/old_test3/1_1.csv"
OUT_VIDEO = "./output/old_test3/1_1.mp4"

# --- モデル設定 ---
DET_MODEL = "yolo11x-pose.pt" # Person用
BALL_MODEL = "yolo11x.pt"     # Ball用

# --- 検出パラメータ ---
INFERENCE_SIZE = 1920  # 高解像度維持
CONF_LOW_LIMIT = 0.03  # 足切りライン (検出漏れ減少のため緩和)
CONF_BALL_LOW_LIMIT = 0.005  # Ball用 足切りライン (より低信頼を拾う)

CONF_PERSON    = 0.25  # 人採用閾値
DEFAULT_CONF_BALL = 0.02  # ボール採用閾値のデフォルト (検出漏れ減少のため緩和)

MAX_PEOPLE     = 2     # 最大人数

# --- ボール検出パラメータ ---
BALL_MAX_CANDIDATES = 60   # 候補数を増やして真の候補を拾う (旧: 3)
BALL_REPROJ_THR = 50.0     # 再投影誤差閾値を緩和 (旧: 12.0)
BALL_REPROJ_THR_LOOSE = 120.0  # さらに緩い閾値 (fallback用)
BALL_MAX_JUMP_M = 6.0      # 連続フレームの最大ジャンプ距離 (m)
BALL_MAX_JUMP_M_LOOSE = 12.0  # fallback用ジャンプ距離 (m)
BALL_CONF_WEIGHT = 5.0     # 検出信頼度の重み (高信頼度ほどスコア低=良)
BALL_THIRD_CAM_WEIGHT = 0.1  # 第3視点再投影誤差の重み

# --- ボールゲーティング(2D) ---
BALL_GATE_ENABLE = True
BALL_GATE_EXPAND = 2.0
BALL_GATE_CAM3_RIGHT_CUT = 1.0
BALL_GATE_MAX_JUMP_PX = 800.0
BALL_GATE_OUTSIDE_FOCUS_PX = 500.0
BALL_RESET_FRAMES = 30
BALL_FILL_MAX_GAP = 30

# MediaPipe設定
MP_COMPLEXITY = 1
MIN_MP_CONF   = 0.3

# Debug描画（TrueにするとYOLOの生BBoxとパディング状況を確認可能）
DRAW_DEBUG_BBOX = True

# MediaPipe Joint Names
MP_JOINTS = [
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
USE_JOINTS_IDX = [11, 12, 23, 24, 25, 26] 

# ==========================================
# キャリブレーション関数群
# ==========================================
def get_inverse_transform(R, T):
    return R.T, -R.T @ T

def scale_camera_matrix(K, dist, target_w, target_h):
    orig_w, orig_h = K[0, 2] * 2, K[1, 2] * 2
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05: return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx; K_new[1, 1] *= sy
    K_new[0, 2] *= sx; K_new[1, 2] *= sy
    return K_new, dist

def load_params_BR(npz_path, v1, v2, v3):
    """Load intrinsics/extrinsics for 3-cam triangulation.

    This loader supports multiple NPZ formats:
      - New format (recommended):
          cam_matrix1/2/3, dist_coeffs1/2/3
          R_w2c1,t_w2c1, R_w2c2,t_w2c2, R_w2c3,t_w2c3  (world=cam2)
          (optionally C1,C2,C3 camera centers in world)

      - Legacy format:
          cam_matrix1/2/3, dist_coeffs1/2/3
          R1,t1, R2,t2, R3,t3  (direction may vary)

    The returned projection matrices are *normalized* (no K): P = [R|t] mapping world->camera.
    """
    d = np.load(npz_path, allow_pickle=True)

    def get_k(k_new, k_old):
        return d[k_new] if k_new in d else d[k_old]

    # --- intrinsics ---
    K1, D1 = get_k("K1", "cam_matrix1"), get_k("dist1", "dist_coeffs1")
    K2, D2 = get_k("K2", "cam_matrix2"), get_k("dist2", "dist_coeffs2")
    K3, D3 = get_k("K3", "cam_matrix3"), get_k("dist3", "dist_coeffs3")

    def get_wh(v):
        c = cv2.VideoCapture(v)
        w, h = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        c.release()
        return w, h

    # Scale K if video resolution differs from calibration resolution
    w1, h1 = get_wh(v1); K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    w2, h2 = get_wh(v2); K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    w3, h3 = get_wh(v3); K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    def as_colvec(t):
        t = np.asarray(t, dtype=np.float64).reshape(3, 1)
        return t

    def inv_rt(R, t):
        R = np.asarray(R, dtype=np.float64)
        t = as_colvec(t)
        return R.T, -R.T @ t

    def camera_center_from_w2c(R, t):
        # C = -R^T t
        R = np.asarray(R, dtype=np.float64)
        t = as_colvec(t)
        return (-R.T @ t).reshape(3)

    def decide_w2c_from_legacy(R_raw, t_raw, C_key=None):
        """Return (R_w2c, t_w2c) from legacy stored R/t.

        If camera center C is present in NPZ, use it to infer direction.
        Otherwise assume legacy stored is c2w and invert it (previous behavior).
        """
        R_raw = np.asarray(R_raw, dtype=np.float64)
        t_raw = as_colvec(t_raw)

        if C_key is not None and C_key in d:
            C_ref = np.asarray(d[C_key], dtype=np.float64).reshape(3)

            # Hypothesis A: raw is w2c
            C_a = camera_center_from_w2c(R_raw, t_raw)
            err_a = float(np.linalg.norm(C_a - C_ref))

            # Hypothesis B: raw is c2w
            # For Xw = Rcw Xc + tcw, camera center in world is tcw.
            C_b = t_raw.reshape(3)
            err_b = float(np.linalg.norm(C_b - C_ref))

            if err_a <= err_b:
                return R_raw, t_raw  # w2c
            else:
                return inv_rt(R_raw, t_raw)  # c2w -> w2c

        # No reference: keep previous assumption (legacy stored is c2w)
        return inv_rt(R_raw, t_raw)

    # --- extrinsics ---
    # World is cam2. We need world(cam2)->camX.
    if "R_w2c1" in d and "t_w2c1" in d:
        R1 = np.asarray(d["R_w2c1"], dtype=np.float64)
        t1 = as_colvec(d["t_w2c1"])
    elif "R1" in d:
        t1_raw = d["T1"] if "T1" in d else d.get("t1")
        if t1_raw is None:
            raise KeyError("Legacy NPZ missing t1/T1")
        R1, t1 = decide_w2c_from_legacy(d["R1"], t1_raw, C_key="C1")
    else:
        raise KeyError("NPZ missing cam1 extrinsics (R_w2c1/t_w2c1 or R1/t1)")

    # cam2 is world origin
    if "R_w2c2" in d and "t_w2c2" in d:
        R2 = np.asarray(d["R_w2c2"], dtype=np.float64)
        t2 = as_colvec(d["t_w2c2"])
    else:
        R2 = np.eye(3, dtype=np.float64)
        t2 = np.zeros((3, 1), dtype=np.float64)

    if "R_w2c3" in d and "t_w2c3" in d:
        R3 = np.asarray(d["R_w2c3"], dtype=np.float64)
        t3 = as_colvec(d["t_w2c3"])
    elif "R3" in d:
        t3_raw = d["T3"] if "T3" in d else d.get("t3")
        if t3_raw is None:
            raise KeyError("Legacy NPZ missing t3/T3")
        R3, t3 = decide_w2c_from_legacy(d["R3"], t3_raw, C_key="C3")
    else:
        raise KeyError("NPZ missing cam3 extrinsics (R_w2c3/t_w2c3 or R3/t3)")

    P1_ext = np.hstack([R1, t1])
    P2_ext = np.hstack([R2, t2])
    P3_ext = np.hstack([R3, t3])

    return [(K1, D1, P1_ext), (K2, D2, P2_ext), (K3, D3, P3_ext)], [(R1, t1), (R2, t2), (R3, t3)]

def undistort_points(kps, K, dist):
    if len(kps) == 0: return []
    pts = np.array(kps, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=None).reshape(-1, 2)

def triangulate_DLT(Ps, pts):
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
        xp, yp = xh[0]/xh[2], xh[1]/xh[2]
        errs.append((xp-x)**2 + (yp-y)**2)
    return X, np.mean(errs)

def str2bool(value):
    if isinstance(value, bool): 
        return value
    val = str(value).strip().lower()
    if val in ("yes", "true", "t", "1"):
        return True
    if val in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")

@dataclass
class BallSelectorConfig:
    pair_select: bool
    reproj_thr: float
    reproj_thr_loose: float
    max_jump_m: float
    max_jump_m_loose: float
    jump_lambda: float
    conf_weight: float = 0.0       # 検出信頼度の重み (高いほど高信頼度を優遇)
    third_cam_weight: float = 0.0  # 第3視点再投影誤差の重み

def triangulate_dlt_pair(Pa, ua, Pb, ub):
    return triangulate_DLT([Pa, Pb], [ua, ub])

def project_norm(P, X):
    xh = P @ np.append(X, 1.0)
    if abs(xh[2]) < 1e-9:
        return None
    return np.array([xh[0] / xh[2], xh[1] / xh[2]], dtype=np.float64)

def norm_to_pixel(K, pt_norm):
    uv = K @ np.array([pt_norm[0], pt_norm[1], 1.0], dtype=np.float64)
    return uv[:2]

def reproj_error_px(P, K, X, target_norm):
    # Pixel error in undistorted pixel coordinates (K * normalized).
    pt_norm = project_norm(P, X)
    if pt_norm is None:
        return float("inf")
    proj_px = norm_to_pixel(K, pt_norm)
    target_px = norm_to_pixel(K, target_norm)
    return float(np.linalg.norm(proj_px - target_px))

def choose_best_ball_pair(ball_lists, Ps, Ks, prev_ball_3d, config, use_loose=False):
    """Pair selection with improved scoring.

    New scoring formula:
        score = reproj_err
              + (jump_lambda * jump_dist)
              - (conf_weight * avg_conf)           # 高信頼度ほどスコア低=良
              + (third_cam_weight * third_err)     # 第3視点整合
    """
    best_candidate = None
    best_score = float("inf")
    reject_reason = "no_valid_pair"
    combos = [(0, 1), (0, 2), (1, 2)]
    cam_indices = {0, 1, 2}

    reproj_thr = config.reproj_thr_loose if use_loose else config.reproj_thr
    max_jump_m = config.max_jump_m_loose if use_loose else config.max_jump_m

    for cam_a, cam_b in combos:
        if not ball_lists[cam_a] or not ball_lists[cam_b]:
            continue

        # 第3カメラを特定
        third_cam = next(iter(cam_indices.difference({cam_a, cam_b})))

        for cand_a in ball_lists[cam_a]:
            for cand_b in ball_lists[cam_b]:
                X, _ = triangulate_dlt_pair(Ps[cam_a], cand_a["center_norm"],
                                            Ps[cam_b], cand_b["center_norm"])
                if np.isnan(X).any():
                    continue
                err_a = reproj_error_px(Ps[cam_a], Ks[cam_a], X, cand_a["center_norm"])
                err_b = reproj_error_px(Ps[cam_b], Ks[cam_b], X, cand_b["center_norm"])
                if not np.isfinite(err_a) or not np.isfinite(err_b):
                    continue
                reproj_err = 0.5 * (err_a + err_b)
                if reproj_err > reproj_thr:
                    reject_reason = f"reproj_above_thr_{cam_a+1}{cam_b+1}"
                    continue

                # ジャンプ距離チェック
                jump_dist = 0.0
                if prev_ball_3d is not None:
                    jump_dist = float(np.linalg.norm(X - prev_ball_3d))
                    if jump_dist > max_jump_m:
                        reject_reason = f"jump_too_large_{cam_a+1}{cam_b+1}"
                        continue

                # 検出信頼度の平均を計算
                avg_conf = 0.5 * (float(cand_a["conf"]) + float(cand_b["conf"]))

                # 第3視点の再投影誤差を計算 (ループ内で評価)
                third_err = None
                if ball_lists[third_cam]:
                    third_errs = []
                    for cand_c in ball_lists[third_cam]:
                        err_c = reproj_error_px(Ps[third_cam], Ks[third_cam], X, cand_c["center_norm"])
                        if np.isfinite(err_c):
                            third_errs.append(err_c)
                    if third_errs:
                        third_err = float(min(third_errs))

                # 新スコアリング式
                total_score = reproj_err
                total_score += config.jump_lambda * jump_dist
                total_score -= config.conf_weight * avg_conf  # 高信頼度ほどスコア低=良
                if third_err is not None and config.third_cam_weight > 0:
                    total_score += config.third_cam_weight * third_err  # 第3視点整合

                if total_score < best_score:
                    best_score = total_score
                    best_candidate = {
                        "X": X,
                        "pair": (cam_a, cam_b),
                        "pair_label": f"{cam_a+1}{cam_b+1}",
                        "reproj_err": reproj_err,
                        "jump_dist": jump_dist,
                        "cand_a": cand_a,
                        "cand_b": cand_b,
                        "avg_conf": avg_conf,
                        "third_cam": third_cam,
                        "third_reproj_err": third_err,
                    }
                    reject_reason = "ok"

    if not best_candidate:
        return None, reject_reason, None

    return best_candidate, reject_reason, best_score
# ==========================================
# 検出ロジック (パディング追加版)
# ==========================================

def get_pose_from_crop_padded(full_img, box, pose_model):
    """
    アスペクト比を維持したまま正方形にパディングし、座標を正確に元画像へ戻す関数
    """
    H_img, W_img = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # 1. 枠を少し広げる（マージン）: 体が見切れるのを防ぐため重要
    margin_ratio = 0.2  # 20%広げる
    w_box = x2 - x1
    h_box = y2 - y1
    pad_w = int(w_box * margin_ratio)
    pad_h = int(h_box * margin_ratio)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(W_img, x2 + pad_w)
    y2 = min(H_img, y2 + pad_h)
    
    if x2 <= x1 or y2 <= y1: return None, None, (x1, y1, x2, y2)

    crop = full_img[y1:y2, x1:x2]
    h_crop, w_crop = crop.shape[:2]
    
    # 2. 正方形の黒画像（キャンバス）を作成
    # MediaPipeは正方形入力が最も精度が出ます
    size = max(h_crop, w_crop)
    padded_img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # 3. 画像を中央に配置するためのオフセット計算
    ox = (size - w_crop) // 2
    oy = (size - h_crop) // 2
    padded_img[oy:oy+h_crop, ox:ox+w_crop] = crop
    
    # 4. 推論実行
    crop_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(crop_rgb)
    
    if not results.pose_landmarks: return None, None, (x1, y1, x2, y2)
    
    kps = []
    confs = []
    
    # 5. 座標変換 (ここが最も重要)
    # MediaPipeの正規化座標(0~1) -> パディング画像座標 -> 元のクロップ座標 -> 全体画像座標
    for lm in results.pose_landmarks.landmark:
        # パディング画像上のピクセル座標 (0 ~ size)
        px = lm.x * size
        py = lm.y * size
        
        # パディング(ox, oy)を除去してクロップ画像上の座標へ
        cx = px - ox
        cy = py - oy
        
        # 全体画像の座標へ (x1, y1 を足す)
        global_x = cx + x1
        global_y = cy + y1
        
        kps.append([global_x, global_y])
        confs.append(lm.visibility)
        
    return np.array(kps), np.array(confs), (x1, y1, x2, y2)

def process_person_yolo(img, pose_model, mp_model, K, D, max_p, conf_low_limit):
    H, W = img.shape[:2]
    res = pose_model.predict(img, conf=conf_low_limit, imgsz=INFERENCE_SIZE, verbose=False, classes=[0])[0]
    
    people = []
    if res.boxes is None or len(res.boxes) == 0: return []

    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    
    p_count = 0
    for idx in indices:
        conf = res.boxes.conf[idx].cpu().numpy()
        if conf < CONF_PERSON: continue
        if p_count >= max_p + 1: break

        box = res.boxes.xyxy[idx].cpu().numpy()
        yolo_kps = res.keypoints.xy[idx].cpu().numpy()

        # パディング付き推論を実行
        mp_kps, mp_confs, _ = get_pose_from_crop_padded(img, box, mp_model)
        
        if mp_kps is not None:
            # --- 画面外判定フィルタ ---
            # 関節座標が画像の端(0やW)を超えている場合、信頼度を強制的に0にする
            # これにより「切れている足」を「そこにある」と誤認するのを防ぐ
            for i in range(len(mp_kps)):
                kx, ky = mp_kps[i]
                margin = 5 # 5ピクセル余裕を見る
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

def detect_ball_yolo(img, det_model, K, D, conf_ball, conf_low_limit, max_candidates=3, allow_low_conf_fallback=True):
    res = det_model.predict(img, conf=conf_low_limit, imgsz=INFERENCE_SIZE, verbose=False, classes=[32])[0]
    balls = []
    if res.boxes is None or len(res.boxes) == 0:
        return []

    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    for idx in indices:
        conf = float(res.boxes.conf[idx].cpu().numpy())
        if conf < conf_ball and not allow_low_conf_fallback:
            continue

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
        if len(balls) >= max_candidates:
            break
    return balls


def _people_union_bbox(people):
    """Return union bbox (x1,y1,x2,y2) of people list. None if empty."""
    if not people:
        return None
    xs1, ys1, xs2, ys2 = [], [], [], []
    for p in people:
        if "box" not in p:
            continue
        x1, y1, x2, y2 = map(float, p["box"])
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
    if not xs1:
        return None
    return (min(xs1), min(ys1), max(xs2), max(ys2))


def _expand_bbox(b, expand, W, H):
    """Expand bbox by (expand * w, expand * h) on each side and clamp."""
    x1, y1, x2, y2 = b
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    x1e = max(0.0, x1 - w * expand)
    y1e = max(0.0, y1 - h * expand)
    x2e = min(float(W - 1), x2 + w * expand)
    y2e = min(float(H - 1), y2 + h * expand)
    return (x1e, y1e, x2e, y2e)


def _point_bbox_distance(px, py, b):
    """Euclidean distance from point to bbox (0 if inside)."""
    x1, y1, x2, y2 = b
    dx = 0.0
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    dy = 0.0
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2
    return float(np.hypot(dx, dy))


def select_ball_candidate(
    balls,
    people,
    prev_center_raw,
    frame_shape,
    cam_idx,
    expand,
    cam3_right_cut,
    max_jump_px,
    outside_focus_px,
    soft_fallback=True,
):
    """Pick 0..1 ball candidate per camera to reduce static false positives."""
    if not balls:
        return []

    H, W = int(frame_shape[0]), int(frame_shape[1])

    # cam3 right-cut (optional)
    if 0.0 < cam3_right_cut < 1.0 and cam_idx == 2:
        x_max = float(W) * float(cam3_right_cut)
        balls = [b for b in balls if float(0.5 * (b["box"][0] + b["box"][2])) <= x_max]
        if not balls:
            return []

    # focus region from people union bbox (expanded)
    focus = _people_union_bbox(people)
    if focus is not None and expand is not None and expand > 0:
        focus = _expand_bbox(focus, float(expand), W, H)

    hard_focus = (focus is not None) and (outside_focus_px is not None) and (outside_focus_px > 0)

    best = None
    best_score = -1e18
    for b in balls:
        conf = float(b.get("conf", 0.0))
        cx = float(b["center_raw"][0])
        cy = float(b["center_raw"][1])

        # HARD REJECT: outside focus
        d_focus = None
        if hard_focus:
            d_focus = _point_bbox_distance(cx, cy, focus)
            if d_focus > float(outside_focus_px):
                continue

        score = conf

        if prev_center_raw is not None and np.all(np.isfinite(prev_center_raw)):
            d_prev = float(np.hypot(cx - float(prev_center_raw[0]), cy - float(prev_center_raw[1])))
            score -= 0.003 * d_prev

        # small preference: closer to focus is better (tie-break)
        if focus is not None:
            if d_focus is None:
                d_focus = _point_bbox_distance(cx, cy, focus)
            score -= 0.001 * float(d_focus)

        if score > best_score:
            best_score = score
            best = b

    if best is None:
        if soft_fallback:
            best = max(balls, key=lambda x: float(x.get("conf", 0.0)))
        else:
            return []

    cx = float(best["center_raw"][0])
    cy = float(best["center_raw"][1])

    if prev_center_raw is not None and np.all(np.isfinite(prev_center_raw)):
        d = float(np.hypot(cx - float(prev_center_raw[0]), cy - float(prev_center_raw[1])))
        if max_jump_px is not None and max_jump_px > 0 and d > float(max_jump_px):
            return []

    if hard_focus:
        d_focus = _point_bbox_distance(cx, cy, focus)
        if d_focus > float(outside_focus_px):
            if soft_fallback:
                return [best]
            return []

    return [best]

# ==========================================
# マッチング & 3D化
# ==========================================
def find_best_matches(ppl_lists, Ps):
    # 2視点マッチングを許容するロジック
    def calc_cost(p_list, p_indices, cam_indices):
        total_err = 0
        valid_joints = 0
        for j in USE_JOINTS_IDX:
            pts = []
            proj_matrices = []
            is_good = True
            for i, p_idx in enumerate(p_indices):
                if p_list[i]["conf"][j] < 0.5:
                    is_good = False
                    break
                pts.append(p_list[i]["kps_norm"][j])
                proj_matrices.append(Ps[cam_indices[i]])
            
            if is_good:
                _, err = triangulate_DLT(proj_matrices, pts)
                total_err += err
                valid_joints += 1
        
        if valid_joints == 0: return 1000.0
        return total_err / valid_joints

    matched_people = []
    used_indices = [set() for _ in range(3)]

    # 1. 3視点マッチング
    candidates3 = []
    for i1, p1 in enumerate(ppl_lists[0]):
        for i2, p2 in enumerate(ppl_lists[1]):
            for i3, p3 in enumerate(ppl_lists[2]):
                cost = calc_cost([p1, p2, p3], [i1, i2, i3], [0, 1, 2])
                if cost < 0.3:
                    candidates3.append({'ids': (i1, i2, i3), 'cost': cost, 'cams': (0,1,2)})
    
    candidates3.sort(key=lambda x: x['cost'])
    
    for cand in candidates3:
        i1, i2, i3 = cand['ids']
        if i1 in used_indices[0] or i2 in used_indices[1] or i3 in used_indices[2]: continue
        matched_people.append({
            'persons': [ppl_lists[0][i1], ppl_lists[1][i2], ppl_lists[2][i3]],
            'cams': [0, 1, 2]
        })
        used_indices[0].add(i1); used_indices[1].add(i2); used_indices[2].add(i3)

    # 2. 2視点マッチング
    cam_pairs = [(0, 1), (1, 2), (0, 2)]
    for c1, c2 in cam_pairs:
        candidates2 = []
        for i1, p1 in enumerate(ppl_lists[c1]):
            if i1 in used_indices[c1]: continue
            for i2, p2 in enumerate(ppl_lists[c2]):
                if i2 in used_indices[c2]: continue
                cost = calc_cost([p1, p2], [i1, i2], [c1, c2])
                if cost < 0.2:
                    candidates2.append({'ids': (i1, i2), 'cost': cost, 'cams': (c1, c2)})
        
        candidates2.sort(key=lambda x: x['cost'])
        
        for cand in candidates2:
            idx1, idx2 = cand['ids']
            if idx1 in used_indices[c1] or idx2 in used_indices[c2]: continue
            p_res = [None, None, None]
            p_res[c1] = ppl_lists[c1][idx1]
            p_res[c2] = ppl_lists[c2][idx2]
            matched_people.append({'persons': p_res, 'cams': [c1, c2]})
            used_indices[c1].add(idx1); used_indices[c2].add(idx2)

    return matched_people

def solve_3d_joints(persons, cams, Ps):
    kps_3d = np.full((33, 3), np.nan)
    for j in range(33):
        valid_pts = []
        valid_Ps = []
        for cam_idx in cams:
            p = persons[cam_idx]
            # 信頼度チェック（画面外フィルタ済みの値）
            if p is not None and p["conf"][j] > 0.5:
                valid_pts.append(p["kps_norm"][j])
                valid_Ps.append(Ps[cam_idx])
        if len(valid_pts) >= 2:
            X, _ = triangulate_DLT(valid_Ps, valid_pts)
            if np.linalg.norm(X) < 50.0:
                kps_3d[j] = X
    return kps_3d

# ==========================================
# Main
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid v5 padding + edge filter with ball pair selection")
    parser.add_argument("--video_left", default=VIDEO_LEFT, help="Left view video (default from config)")
    parser.add_argument("--video_center", default=VIDEO_CENTER, help="Center view video")
    parser.add_argument("--video_right", default=VIDEO_RIGHT, help="Right view video")
    parser.add_argument("--calib_npz", default=CALIB_NPZ, help="Camera calibration NPZ path")
    parser.add_argument("--out_csv", default=OUT_CSV, help="Output CSV path")
    parser.add_argument("--out_video", default=OUT_VIDEO, help="Output video path")
    parser.add_argument("--ball_pair_select", type=str2bool, default=True, help="Enable pair selection gating")
    parser.add_argument("--ball_reproj_thr", type=float, default=BALL_REPROJ_THR, help="Reprojection error threshold in undistorted pixels")
    parser.add_argument("--ball_reproj_thr_loose", type=float, default=BALL_REPROJ_THR_LOOSE, help="Loose reprojection error threshold")
    parser.add_argument("--ball_max_jump_m", type=float, default=BALL_MAX_JUMP_M, help="Max allowed jump distance for sequential frames")
    parser.add_argument("--ball_max_jump_m_loose", type=float, default=BALL_MAX_JUMP_M_LOOSE, help="Loose max jump distance for sequential frames")
    parser.add_argument("--ball_jump_lambda", type=float, default=0.0, help="Distance penalty weight when selecting best pair")
    parser.add_argument("--ball_verbose", action="store_true", help="Print ball debug info at 1s intervals or rejection")
    parser.add_argument("--ball_max_candidates", type=int, default=BALL_MAX_CANDIDATES, help="Max ball detections to keep per camera")
    parser.add_argument("--ball_conf_weight", type=float, default=BALL_CONF_WEIGHT, help="Weight for detection confidence in scoring (higher favors confident detections)")
    parser.add_argument("--ball_third_cam_weight", type=float, default=BALL_THIRD_CAM_WEIGHT, help="Weight for third camera reprojection error in scoring")
    parser.add_argument("--conf_ball", type=float, default=DEFAULT_CONF_BALL, help="Minimum confidence to consider a ball detection")
    parser.add_argument("--conf_ball_low_limit", type=float, default=CONF_BALL_LOW_LIMIT, help="Low confidence cutoff for ball YOLO inference")
    parser.add_argument("--conf_low_limit", type=float, default=CONF_LOW_LIMIT, help="Low confidence cutoff for YOLO inference")
    parser.add_argument("--ball_gate_disable", action="store_true", help="Disable per-camera ball gating")
    parser.add_argument("--ball_gate_expand", type=float, default=BALL_GATE_EXPAND, help="Expand ratio for people-union bbox when gating balls")
    parser.add_argument("--ball_gate_cam3_right_cut", type=float, default=BALL_GATE_CAM3_RIGHT_CUT, help="Cam3: drop balls with cx > W*cut (0-1). Set 1.0 to disable")
    parser.add_argument("--ball_gate_max_jump_px", type=float, default=BALL_GATE_MAX_JUMP_PX, help="Reject ball 2D jumps larger than this (px/frame)")
    parser.add_argument("--ball_gate_outside_focus_px", type=float, default=BALL_GATE_OUTSIDE_FOCUS_PX, help="Drop balls far outside focus bbox (px)")
    parser.add_argument("--ball_reset_frames", type=int, default=BALL_RESET_FRAMES, help="Reset ball tracking after this many misses")
    parser.add_argument("--ball_fill_max_gap", type=int, default=BALL_FILL_MAX_GAP, help="Carry forward last ball for up to this many missing frames")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"=== Hybrid v5 Padding & Edge Filter ===")

    video_paths = [args.video_left, args.video_center, args.video_right]
    if not os.path.exists(args.calib_npz): 
        print(f"calibration file missing: {args.calib_npz}")
        return

    cam_params_full, _ = load_params_BR(args.calib_npz, *video_paths)
    Ks = [p[0] for p in cam_params_full]
    P1, P2, P3 = [p[2] for p in cam_params_full]
    Ps = [P1, P2, P3]

    caps = [cv2.VideoCapture(v) for v in video_paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Loading Models...")
    yolo_pose = YOLO(DET_MODEL) 
    yolo_det = YOLO(BALL_MODEL) 
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(static_image_mode=True, model_complexity=MP_COMPLEXITY, min_detection_confidence=MIN_MP_CONF)

    f_csv = open(args.out_csv, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z",
                     "ball_pair", "ball_reproj_err", "ball_jump_dist"])

    temp_out = args.out_video + ".temp.mp4"
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    
    mp_conn = mp.solutions.pose.POSE_CONNECTIONS
    colors = [(0,255,0), (0,0,255), (255,255,0)]

    print(f"Processing {total_frames} frames...")

    prev_centers = {}
    prev_ball_3d = None
    prev_ball_center_raw = [None, None, None]
    ball_miss_streak = 0
    ball_config = BallSelectorConfig(
        pair_select=args.ball_pair_select,
        reproj_thr=args.ball_reproj_thr,
        reproj_thr_loose=args.ball_reproj_thr_loose,
        max_jump_m=args.ball_max_jump_m,
        max_jump_m_loose=args.ball_max_jump_m_loose,
        jump_lambda=args.ball_jump_lambda,
        conf_weight=args.ball_conf_weight,
        third_cam_weight=args.ball_third_cam_weight,
    )
    log_interval = max(1, int(round(fps))) if fps > 0 else 1

    for i in tqdm(range(total_frames)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break

        ppl_lists = []
        ball_lists = []

        for cam_i, f in enumerate(frames):
            K, D, _ = cam_params_full[cam_i]
            p = process_person_yolo(f, yolo_pose, pose_estimator, K, D, MAX_PEOPLE, args.conf_low_limit)
            ppl_lists.append(p)
            b_all = detect_ball_yolo(
                f, yolo_det, K, D,
                args.conf_ball,
                args.conf_ball_low_limit,
                args.ball_max_candidates,
                allow_low_conf_fallback=True,
            )

            if not args.ball_gate_disable:
                b_sel = select_ball_candidate(
                    b_all,
                    p,
                    prev_ball_center_raw[cam_i],
                    f.shape,
                    cam_i,
                    args.ball_gate_expand,
                    args.ball_gate_cam3_right_cut,
                    args.ball_gate_max_jump_px,
                    args.ball_gate_outside_focus_px,
                    soft_fallback=True,
                )
                if b_sel:
                    prev_ball_center_raw[cam_i] = b_sel[0]["center_raw"]
                b = b_sel
            else:
                b = b_all

            ball_lists.append(b)

            if DRAW_DEBUG_BBOX:
                for per in p:
                    bx = list(map(int, per["box"]))
                    cv2.rectangle(f, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 2)
                for bal in b:
                    bx = list(map(int, bal["box"]))
                    cv2.rectangle(f, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)

        matched_results = find_best_matches(ppl_lists, Ps)

        final_persons = [None] * MAX_PEOPLE # ID順に格納するリスト
        
        # 現在のフレームで検出された候補リストを作成
        candidates = []
        for match in matched_results:
            persons = match['persons']
            active_cams = match['cams']
            # 仮の3D座標を計算して重心を取得
            kps_3d = solve_3d_joints(persons, active_cams, Ps)
            
            # 有効な関節の平均座標(重心)を計算
            valid_pts = kps_3d[~np.isnan(kps_3d).any(axis=1)]
            if len(valid_pts) > 0:
                center = np.mean(valid_pts, axis=0)
            else:
                center = np.array([0,0,0]) # 検出なし
            
            candidates.append({
                'match_data': match,
                'center': center,
                'kps_3d': kps_3d
            })

        # 最初のフレーム、または誰もいなかった後は、そのままリスト順で登録
        if i == 0 or len(prev_centers) == 0:
            for idx, cand in enumerate(candidates):
                if idx < MAX_PEOPLE:
                    final_persons[idx] = cand
                    prev_centers[idx] = cand['center']
        else:
            # 2フレーム目以降: 前回の位置に近い順に割り当て
            used_candidates = set()
            
            for pid in range(MAX_PEOPLE):
                if pid not in prev_centers: continue
                last_pos = prev_centers[pid]
                
                best_dist = 200.0 # 閾値 (2m以上飛んだら別人扱い)
                best_idx = -1
                
                for c_idx, cand in enumerate(candidates):
                    if c_idx in used_candidates: continue
                    
                    dist = np.linalg.norm(cand['center'] - last_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = c_idx
                
                if best_idx != -1:
                    final_persons[pid] = candidates[best_idx]
                    prev_centers[pid] = candidates[best_idx]['center']
                    used_candidates.add(best_idx)
            
            # 余った候補があれば、空いているIDに入れる (新規出現など)
            for c_idx, cand in enumerate(candidates):
                if c_idx not in used_candidates:
                    for pid in range(MAX_PEOPLE):
                        if final_persons[pid] is None:
                            final_persons[pid] = cand
                            prev_centers[pid] = cand['center']
                            break

        # ---------------------------------------------------------
        # CSV書き込み & 描画 (final_persons を使用)
        # ---------------------------------------------------------
        for pid in range(MAX_PEOPLE):
            cand = final_persons[pid]
            if cand is None: continue # このIDの人は今回いない

            kps_3d = cand['kps_3d']
            match = cand['match_data']
            persons = match['persons']
            col = colors[pid % len(colors)]

            # CSV出力
            for j in range(33):
                if not np.isnan(kps_3d[j][0]):
                    writer.writerow([
                        i, pid, MP_JOINTS[j], kps_3d[j][0], kps_3d[j][1], kps_3d[j][2],
                        "", "", ""
                    ])

            # 描画
            # (描画コードは元の matched_results ループの中身と同じですが、
            #  pid が固定されるため色が安定します)
            for cam_idx in range(3):
                p_data = persons[cam_idx]
                if p_data is None: continue
                
                frame = frames[cam_idx]
        
        # for pid, match in enumerate(matched_results):
        #     persons = match['persons']
        #     active_cams = match['cams']
        #     col = colors[pid % len(colors)]
        #     kps_3d = solve_3d_joints(persons, active_cams, Ps)
            
        #     for j in range(33):
        #         if not np.isnan(kps_3d[j][0]):
        #             writer.writerow([i, pid, MP_JOINTS[j], kps_3d[j][0], kps_3d[j][1], kps_3d[j][2]])

            for cam_idx in range(3):
                p_data = persons[cam_idx]
                if p_data is None: continue
                
                frame = frames[cam_idx]
                kps_raw = p_data["kps_raw"]
                
                # YOLO点
                if "yolo_kps" in p_data:
                    for ykp in p_data["yolo_kps"]:
                        cv2.circle(frame, (int(ykp[0]), int(ykp[1])), 3, (0, 255, 255), -1)

                # MediaPipe線 (信頼度0.5以上のみ描画)
                for u, v in mp_conn:
                    if u < len(kps_raw) and v < len(kps_raw):
                        if p_data["conf"][u] > 0.3 and p_data["conf"][v] > 0.3:
                            pt1 = (int(kps_raw[u][0]), int(kps_raw[u][1]))
                            pt2 = (int(kps_raw[v][0]), int(kps_raw[v][1]))
                            cv2.line(frame, pt1, pt2, col, 2)
                
                box = p_data["box"]
                cv2.putText(frame, f"ID:{pid}", (int(box[0]), int(box[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

        # Ball
        ball_centers_for_draw = [None] * 3
        for cam_idx in range(3):
            if ball_lists[cam_idx]:
                ball_centers_for_draw[cam_idx] = ball_lists[cam_idx][0]["box"]

        best_ball = None
        ball_reason = "no_pair"
        if args.ball_pair_select:
            best_ball, ball_reason, _ = choose_best_ball_pair(ball_lists, Ps, Ks, prev_ball_3d, ball_config, use_loose=False)
            if not best_ball:
                best_ball, loose_reason, _ = choose_best_ball_pair(ball_lists, Ps, Ks, prev_ball_3d, ball_config, use_loose=True)
                if best_ball:
                    ball_reason = f"loose_{loose_reason}"
            if best_ball:
                prev_ball_3d = best_ball["X"]
                ball_miss_streak = 0
                a_cam, b_cam = best_ball["pair"]
                ball_centers_for_draw[a_cam] = best_ball["cand_a"]["box"]
                ball_centers_for_draw[b_cam] = best_ball["cand_b"]["box"]
            else:
                ball_miss_streak += 1
        else:
            valid_ball_pts = []
            valid_ball_Ps = []
            ball_reason = "legacy"
            for cam_idx in range(3):
                if ball_lists[cam_idx]:
                    cand = ball_lists[cam_idx][0]
                    valid_ball_pts.append(cand["center_norm"])
                    valid_ball_Ps.append(Ps[cam_idx])
            if len(valid_ball_pts) >= 2:
                X_b, err_b = triangulate_DLT(valid_ball_Ps, valid_ball_pts)
                if np.linalg.norm(X_b) < 50.0 and err_b < 20.0:
                    best_ball = {
                        "X": X_b,
                        "pair_label": "legacy",
                        "reproj_err": err_b,
                        "jump_dist": 0.0
                    }
                    prev_ball_3d = X_b
                    ball_miss_streak = 0
                else:
                    ball_reason = "legacy_reproj"
            if best_ball is None:
                ball_miss_streak += 1

        # Fill missing frames by carrying the last known ball position (limited gap)
        if best_ball is None and prev_ball_3d is not None:
            if args.ball_fill_max_gap is not None and ball_miss_streak <= int(args.ball_fill_max_gap):
                best_ball = {
                    "X": prev_ball_3d,
                    "pair_label": "carry",
                    "reproj_err": float("nan"),
                    "jump_dist": 0.0,
                }
                ball_reason = "carry"

        if best_ball is None and ball_miss_streak >= max(1, int(args.ball_reset_frames)):
            prev_ball_3d = None
            prev_ball_center_raw = [None, None, None]
            ball_miss_streak = 0

        should_log = args.ball_verbose or (i % log_interval == 0) or (best_ball is None) or (ball_reason not in ("ok", "legacy"))
        if should_log:
            pair_label = best_ball["pair_label"] if best_ball else "N/A"
            reproj_val = f"{best_ball['reproj_err']:.3f}" if best_ball else "N/A"
            jump_val = f"{best_ball['jump_dist']:.3f}" if best_ball else "N/A"
            third_val = "N/A"
            if best_ball and best_ball.get("third_reproj_err") is not None:
                third_val = f"{best_ball['third_reproj_err']:.3f}"
            print(f"[Ball] frame={i}, pair={pair_label}, reproj={reproj_val}, jump={jump_val}, third={third_val}, reason={ball_reason}")

        if best_ball:
            writer.writerow([
                i, -1, "ball", best_ball["X"][0], best_ball["X"][1], best_ball["X"][2],
                best_ball["pair_label"], f"{best_ball['reproj_err']:.4f}", f"{best_ball['jump_dist']:.4f}"
            ])
            for cam_idx in range(3):
                box = ball_centers_for_draw[cam_idx]
                if box is not None:
                    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    cv2.circle(frames[cam_idx], (cx, cy), 10, (0, 165, 255), -1)
                    cv2.putText(frames[cam_idx], "BALL 3D", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        vw.write(np.hstack(frames))

    f_csv.close()
    pose_estimator.close()
    vw.release()
    for c in caps: c.release()

    if os.path.exists(temp_out):
        subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out_video], check=False)
        os.remove(temp_out)
        print(f"Saved: {args.out_video}")

if __name__ == "__main__":
    main()
