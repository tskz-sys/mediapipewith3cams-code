#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + MediaPipe Hybrid v4
- "v2 のロジック"（= 検出→MP Pose→3カメラ対応付け→DLT三角測量→描画/CSV）を維持
- 入力/出力を "v3 っぽく"（= dictで動画パスを渡せる / カメラ別出力も可能）に整理

想定キー:
  video_paths = {"L": "...", "C": "...", "R": "..."}   # Left / Center / Right
  あるいは {"B": "...", "C": "...", "R": "..."} でもOK（キー名は任意だが 3本であること）

出力:
  - out_csv: 3D点（MediaPipe 33点 + ball）を frame, person_id, joint, X,Y,Z で保存
  - out_video_montage: 3枚横並び（L|C|R）の描画動画（任意）
  - out_videos_per_cam: 各カメラごとの描画動画（任意、dictで指定）
"""

from __future__ import annotations

import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ultralytics import YOLO
from tqdm import tqdm


# =========================
# Defaults (v2ロジック寄り)
# =========================
DET_MODEL_DEFAULT  = "yolo11x-pose.pt"  # Person (Pose)
BALL_MODEL_DEFAULT = "yolo11x.pt"       # Ball (Detect, COCO class 32)

INFERENCE_SIZE_DEFAULT = 1280
CONF_LOW_LIMIT_DEFAULT = 0.15

CONF_PERSON_DEFAULT = 0.25
CONF_BALL_DEFAULT   = 0.15

MAX_PEOPLE_DEFAULT  = 2

MP_COMPLEXITY_DEFAULT = 1
MIN_MP_CONF_DEFAULT   = 0.5

# MediaPipe Pose 33 joints (index aligns with mediapipe pose landmarks)
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

# マッチング用（肩/腰/膝など：取り違えに強い）
USE_JOINTS_IDX = [11, 12, 23, 24, 25, 26]  # L/R shoulder, L/R hip, L/R knee


# =========================
# Calibration helpers
# =========================
def get_inverse_transform(R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(R, T) を逆変換 (R^T, -R^T T) にする"""
    return R.T, -R.T @ T

def scale_camera_matrix(K: np.ndarray, dist: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Kが別解像度で推定されている場合に、動画解像度にスケール合わせする"""
    orig_w, orig_h = K[0, 2] * 2, K[1, 2] * 2
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05:
        return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx; K_new[1, 1] *= sy
    K_new[0, 2] *= sx; K_new[1, 2] *= sy
    return K_new, dist

def _get_video_wh(video_path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return w, h, fps, n

def load_params_BR(npz_path: str, v1: str, v2: str, v3: str):
    """
    npz: (K1,D1,R1,t1, K2,D2,R2,t2, K3,D3,R3,t3) 相当を読み、
    「中央を基準（R=I, t=0）」で外部を組み立てる想定。
    """
    d = np.load(npz_path, allow_pickle=True)

    def get_k(k_new, k_old):
        if k_new in d:
            return d[k_new]
        if k_old in d:
            return d[k_old]
        raise KeyError(f"Missing key in npz: {k_new} / {k_old}")

    K1, D1 = get_k("K1", "cam_matrix1"), get_k("dist1", "dist_coeffs1")
    K2, D2 = get_k("K2", "cam_matrix2"), get_k("dist2", "dist_coeffs2")
    K3, D3 = get_k("K3", "cam_matrix3"), get_k("dist3", "dist_coeffs3")

    # スケール合わせ（動画解像度に合わせる）
    w1, h1, _, _ = _get_video_wh(v1); K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    w2, h2, _, _ = _get_video_wh(v2); K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    w3, h3, _, _ = _get_video_wh(v3); K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    # BRパターン補正（あなたの既存npzと同じ解釈）
    R1_raw = d["R1"]; t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3, 1))

    R2 = np.eye(3); t2 = np.zeros((3, 1))

    R3_raw = d["R3"]; t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3, 1))

    P1_ext = np.hstack([R1, t1])
    P2_ext = np.hstack([R2, t2])
    P3_ext = np.hstack([R3, t3])

    return [(K1, D1, P1_ext), (K2, D2, P2_ext), (K3, D3, P3_ext)], [(R1, t1), (R2, t2), (R3, t3)]


def undistort_points(kps: List[Tuple[float, float]] | np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    if len(kps) == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.array(kps, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=None).reshape(-1, 2)


def triangulate_DLT(Ps: List[np.ndarray], pts: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """3視点以上に対応したDLT。ptsは正規化画像座標（undistortPointsの出力）を想定。"""
    A = []
    for P, (x, y) in zip(Ps, pts):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    X = X[:3] / X[3]

    # reprojection error (normalized plane)
    errs = []
    for P, (x, y) in zip(Ps, pts):
        xh = P @ np.hstack([X, 1.0])
        if abs(xh[2]) < 1e-9:
            errs.append(1e9)
            continue
        xp, yp = xh[0] / xh[2], xh[1] / xh[2]
        errs.append((xp - x) ** 2 + (yp - y) ** 2)
    return X, float(np.mean(errs))


# =========================
# Detection / Pose helpers
# =========================
def get_pose_from_crop(full_img: np.ndarray, box_xyxy: np.ndarray, mp_pose_model) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[int,int,int,int]]:
    """
    YOLO bboxで切り出し → MediaPipe Pose
    return:
      mp_kps (33,2) in global pixel coords, mp_confs (33,), padded_box (x1,y1,x2,y2)
    """
    H, W = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box_xyxy)

    pad_w = int((x2 - x1) * 0.15)
    pad_h = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - pad_w); y1 = max(0, y1 - pad_h)
    x2 = min(W, x2 + pad_w); y2 = min(H, y2 + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None, None, (x1, y1, x2, y2)

    crop = full_img[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = mp_pose_model.process(crop_rgb)
    if not results.pose_landmarks:
        return None, None, (x1, y1, x2, y2)

    kps = []
    confs = []
    ch, cw = crop.shape[:2]
    for lm in results.pose_landmarks.landmark:
        gx = lm.x * cw + x1
        gy = lm.y * ch + y1
        kps.append([gx, gy])
        confs.append(lm.visibility)
    return np.array(kps, dtype=np.float32), np.array(confs, dtype=np.float32), (x1, y1, x2, y2)


def process_person_yolo(
    img: np.ndarray,
    yolo_pose_model,
    mp_pose_model,
    K: np.ndarray,
    D: np.ndarray,
    max_people: int,
    conf_low_limit: float,
    conf_person: float,
    imgsz: int
) -> List[dict]:
    """
    1枚の画像から人物を最大max_people人まで拾う。
    返り値は person dict のリスト：
      {
        "box": xyxy,
        "yolo_kps": (17,2) pixel,
        "kps_raw": (33,2) pixel (MP),
        "kps_norm": (33,2) normalized (undistorted),
        "conf": (33,) MP visibility
      }
    """
    res = yolo_pose_model.predict(img, conf=conf_low_limit, imgsz=imgsz, verbose=False, classes=[0])[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    people = []
    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    p_count = 0

    for idx in indices:
        det_conf = float(res.boxes.conf[idx].cpu().numpy())
        if det_conf < conf_person:
            continue
        if p_count >= max_people:
            break

        box = res.boxes.xyxy[idx].cpu().numpy()
        yolo_kps = res.keypoints.xy[idx].cpu().numpy() if res.keypoints is not None else None

        mp_kps, mp_confs, _ = get_pose_from_crop(img, box, mp_pose_model)
        if mp_kps is None:
            continue

        norm_mp_kps = undistort_points(mp_kps, K, D)

        people.append({
            "type": "person",
            "box": box,
            "yolo_kps": yolo_kps,
            "kps_raw": mp_kps,
            "kps_norm": norm_mp_kps,
            "conf": mp_confs,
        })
        p_count += 1

    return people


def detect_ball_yolo(
    img: np.ndarray,
    yolo_det_model,
    K: np.ndarray,
    D: np.ndarray,
    conf_low_limit: float,
    conf_ball: float,
    imgsz: int
) -> Optional[dict]:
    """
    COCO class 32 (sports ball) を1つだけ拾う。
    return dict or None
    """
    res = yolo_det_model.predict(img, conf=conf_low_limit, imgsz=imgsz, verbose=False, classes=[32])[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None

    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    for idx in indices:
        det_conf = float(res.boxes.conf[idx].cpu().numpy())
        if det_conf < conf_ball:
            continue
        box = res.boxes.xyxy[idx].cpu().numpy()
        cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
        norm_pt = undistort_points([(cx, cy)], K, D)[0]
        return {
            "type": "ball",
            "conf": det_conf,
            "box": box,
            "center_raw": np.array([cx, cy], dtype=np.float32),
            "center_norm": norm_pt,
        }
    return None


# =========================
# Main runner (v3-ish I/O)
# =========================
@dataclass
class HybridConfig:
    calib_npz: str
    out_csv: str
    out_video_montage: Optional[str] = None
    out_videos_per_cam: Optional[Dict[str, str]] = None

    det_model: str = DET_MODEL_DEFAULT
    ball_model: str = BALL_MODEL_DEFAULT

    imgsz: int = INFERENCE_SIZE_DEFAULT
    conf_low_limit: float = CONF_LOW_LIMIT_DEFAULT
    conf_person: float = CONF_PERSON_DEFAULT
    conf_ball: float = CONF_BALL_DEFAULT

    max_people: int = MAX_PEOPLE_DEFAULT
    mp_complexity: int = MP_COMPLEXITY_DEFAULT
    mp_min_conf: float = MIN_MP_CONF_DEFAULT

    # triangulation filters
    match_err_thresh: float = 0.2
    match_min_joints: int = 3
    max_3d_norm: float = 50.0

    # video options
    show: bool = False
    save_h264_reencode: bool = True  # ffmpegで再エンコードして互換性を上げる


def run_hybrid_v4(video_paths: Dict[str, str], cfg: HybridConfig) -> None:
    """
    video_paths: dict of 3 video paths (keys are used for output naming)
    """
    if len(video_paths) != 3:
        raise ValueError(f"video_paths must have 3 items, got {len(video_paths)}")

    keys = list(video_paths.keys())
    v1, v2, v3 = [video_paths[k] for k in keys]

    for p in [v1, v2, v3, cfg.calib_npz]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)
    if cfg.out_video_montage:
        os.makedirs(os.path.dirname(cfg.out_video_montage) or ".", exist_ok=True)
    if cfg.out_videos_per_cam:
        for _, op in cfg.out_videos_per_cam.items():
            os.makedirs(os.path.dirname(op) or ".", exist_ok=True)

    # Video info
    w1, h1, fps1, n1 = _get_video_wh(v1)
    w2, h2, fps2, n2 = _get_video_wh(v2)
    w3, h3, fps3, n3 = _get_video_wh(v3)
    W, H = w1, h1
    fps = fps1 or fps2 or fps3 or 30.0

    # total_frames may be 0 on some codecs/VFR; handle by while-loop fallback
    total_frames = min([n for n in [n1, n2, n3] if n > 0], default=0)

    print("=== Hybrid v4: v2-logic + v3-like IO ===")
    print(f"Videos: {keys[0]}={v1}, {keys[1]}={v2}, {keys[2]}={v3}")
    print(f"Resolution: {(W, H)}  FPS: {fps:.3f}  Frames(meta): {total_frames if total_frames>0 else 'unknown'}")
    print(f"Calib: {cfg.calib_npz}")
    print(f"Out CSV: {cfg.out_csv}")
    if cfg.out_video_montage:
        print(f"Out Montage: {cfg.out_video_montage}")
    if cfg.out_videos_per_cam:
        print(f"Out PerCam: {cfg.out_videos_per_cam}")

    # Load calib
    cam_params_full, _ = load_params_BR(cfg.calib_npz, v1, v2, v3)
    P1, P2, P3 = [p[2] for p in cam_params_full]

    # Open videos
    caps = [cv2.VideoCapture(v) for v in [v1, v2, v3]]
    for c, v in zip(caps, [v1, v2, v3]):
        if not c.isOpened():
            raise RuntimeError(f"Cannot open: {v}")

    # Models
    print(f"Loading YOLO Pose: {cfg.det_model}")
    yolo_pose = YOLO(cfg.det_model)
    print(f"Loading YOLO Detect: {cfg.ball_model}")
    yolo_det = YOLO(cfg.ball_model)

    print("Loading MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=cfg.mp_complexity,
        min_detection_confidence=cfg.mp_min_conf
    )

    # CSV
    f_csv = open(cfg.out_csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z"])

    # Writers
    montage_writer = None
    temp_montage = None
    if cfg.out_video_montage:
        temp_montage = cfg.out_video_montage + ".temp.mp4"
        montage_writer = cv2.VideoWriter(
            temp_montage, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 3, H)
        )

    per_cam_writers = {}
    if cfg.out_videos_per_cam:
        for cam_key, out_path in cfg.out_videos_per_cam.items():
            # cam_key must exist
            if cam_key not in video_paths:
                raise ValueError(f"out_videos_per_cam key {cam_key} not in video_paths keys {list(video_paths.keys())}")
            temp_out = out_path + ".temp.mp4" if cfg.save_h264_reencode else out_path
            per_cam_writers[cam_key] = (cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)), temp_out, out_path)

    mp_conn = mp.solutions.pose.POSE_CONNECTIONS
    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]

    def _read_triplet():
        frames = []
        rets = []
        for cap in caps:
            ret, f = cap.read()
            rets.append(ret)
            frames.append(f)
        if not all(rets):
            return None
        # resize if needed to match first video
        fixed = []
        for f in frames:
            if f is None:
                return None
            if f.shape[1] != W or f.shape[0] != H:
                f = cv2.resize(f, (W, H))
            fixed.append(f)
        return fixed

    # Main loop
    frame_iter = range(total_frames) if total_frames > 0 else iter(int, 1)  # infinite
    i = 0
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="frames")
    try:
        for _ in frame_iter:
            triplet = _read_triplet()
            if triplet is None:
                break
            frames = triplet

            ppl_lists = []
            ball_list = []

            # 1) Detection
            for cam_i, f in enumerate(frames):
                K, D, _ = cam_params_full[cam_i]
                ppl = process_person_yolo(
                    f, yolo_pose, pose_estimator, K, D,
                    cfg.max_people, cfg.conf_low_limit, cfg.conf_person, cfg.imgsz
                )
                ppl_lists.append(ppl)

                b = detect_ball_yolo(
                    f, yolo_det, K, D, cfg.conf_low_limit, cfg.conf_ball, cfg.imgsz
                )
                ball_list.append(b)

            # 2) Person matching (3 cameras)
            candidates = []
            for idx1 in range(len(ppl_lists[0])):
                for idx2 in range(len(ppl_lists[1])):
                    for idx3 in range(len(ppl_lists[2])):
                        p1, p2, p3 = ppl_lists[0][idx1], ppl_lists[1][idx2], ppl_lists[2][idx3]
                        cost_sum, count = 0.0, 0
                        for j in USE_JOINTS_IDX:
                            if p1["conf"][j] > 0.5 and p2["conf"][j] > 0.5 and p3["conf"][j] > 0.5:
                                pts = [p1["kps_norm"][j], p2["kps_norm"][j], p3["kps_norm"][j]]
                                _, err = triangulate_DLT([P1, P2, P3], pts)
                                if err < cfg.match_err_thresh:
                                    cost_sum += err
                                    count += 1
                        if count >= cfg.match_min_joints:
                            candidates.append({"ids": (idx1, idx2, idx3), "cost": cost_sum / max(count, 1)})

            candidates.sort(key=lambda x: x["cost"])
            used1, used2, used3 = set(), set(), set()
            pid_counter = 0

            # 3) Triangulate matched persons & draw
            for cand in candidates:
                i1, i2, i3 = cand["ids"]
                if i1 in used1 or i2 in used2 or i3 in used3:
                    continue

                persons = [ppl_lists[0][i1], ppl_lists[1][i2], ppl_lists[2][i3]]
                col = colors[pid_counter % len(colors)]

                # 3D (MediaPipe 33)
                kps_3d = np.full((33, 3), np.nan, dtype=np.float32)
                for j in range(33):
                    if all(p["conf"][j] > 0.5 for p in persons):
                        pts = [p["kps_norm"][j] for p in persons]
                        X, err = triangulate_DLT([P1, P2, P3], pts)
                        if np.linalg.norm(X) < cfg.max_3d_norm:
                            kps_3d[j] = X.astype(np.float32)
                            writer.writerow([i, pid_counter, MP_JOINTS[j], float(X[0]), float(X[1]), float(X[2])])

                # Draw each camera
                for cam_i in range(3):
                    frame = frames[cam_i]
                    pdata = persons[cam_i]

                    # A) YOLO pose keypoints (yellow dots) if available
                    if pdata.get("yolo_kps") is not None:
                        for ykp in pdata["yolo_kps"]:
                            x, y = int(ykp[0]), int(ykp[1])
                            if 0 <= x < W and 0 <= y < H:
                                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

                    # B) MediaPipe skeleton lines
                    kps_raw = pdata["kps_raw"]
                    for u, v in mp_conn:
                        if u < len(kps_raw) and v < len(kps_raw):
                            pt1 = (int(kps_raw[u][0]), int(kps_raw[u][1]))
                            pt2 = (int(kps_raw[v][0]), int(kps_raw[v][1]))
                            cv2.line(frame, pt1, pt2, col, 2)

                    # C) Finger-ish landmarks (MP Pose indices 17-22)
                    for tip in [17, 18, 19, 20, 21, 22]:
                        pt = (int(kps_raw[tip][0]), int(kps_raw[tip][1]))
                        cv2.circle(frame, pt, 4, (255, 255, 0), -1)

                used1.add(i1); used2.add(i2); used3.add(i3)
                pid_counter += 1

            # 4) Ball triangulation (if present in all cams)
            b1, b2, b3 = ball_list
            if b1 and b2 and b3:
                pts_b = [b1["center_norm"], b2["center_norm"], b3["center_norm"]]
                Xb, errb = triangulate_DLT([P1, P2, P3], pts_b)
                if np.linalg.norm(Xb) < cfg.max_3d_norm and errb < 5.0:
                    writer.writerow([i, -1, "ball", float(Xb[0]), float(Xb[1]), float(Xb[2])])
                    for cam_i, b in enumerate([b1, b2, b3]):
                        box = b["box"]
                        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                        cv2.circle(frames[cam_i], (cx, cy), 8, (0, 165, 255), -1)
                        cv2.putText(frames[cam_i], "BALL", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # Write outputs
            if montage_writer is not None:
                montage_writer.write(np.hstack(frames))

            if per_cam_writers:
                for cam_i, cam_key in enumerate(keys):
                    if cam_key in per_cam_writers:
                        w, _, _ = per_cam_writers[cam_key]
                        w.write(frames[cam_i])

            if cfg.show:
                show_img = cv2.resize(np.hstack(frames), (1280, int(1280 * H / (W * 3))))
                cv2.imshow("Hybrid v4 (L|C|R)", show_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            i += 1
            pbar.update(1)

    finally:
        pbar.close()
        f_csv.close()
        pose_estimator.close()
        for c in caps:
            c.release()
        if montage_writer is not None:
            montage_writer.release()
        for cam_key, (w, _, _) in per_cam_writers.items():
            w.release()
        if cfg.show:
            cv2.destroyAllWindows()

    # ffmpeg re-encode for compatibility
    def _reencode(in_path: str, out_path: str):
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path], check=False)
        try:
            os.remove(in_path)
        except OSError:
            pass

    if cfg.out_video_montage and montage_writer is not None and temp_montage:
        if cfg.save_h264_reencode:
            _reencode(temp_montage, cfg.out_video_montage)

    if per_cam_writers and cfg.save_h264_reencode:
        for cam_key, (_, temp_path, final_path) in per_cam_writers.items():
            # temp_path == final_path if reencode off
            if temp_path != final_path:
                _reencode(temp_path, final_path)

    print(f"Done. Frames processed: {i}")
    print(f"Saved CSV: {cfg.out_csv}")
    if cfg.out_video_montage:
        print(f"Saved montage: {cfg.out_video_montage}")
    if cfg.out_videos_per_cam:
        print(f"Saved per-cam: {cfg.out_videos_per_cam}")


# =========================
# CLI
# =========================
def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Hybrid v4 (v2 logic + v3-like IO)")
    ap.add_argument("--left",   required=True, help="Left video path")
    ap.add_argument("--center", required=True, help="Center video path")
    ap.add_argument("--right",  required=True, help="Right video path")
    ap.add_argument("--calib",  required=True, help="Calibration npz path")

    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_montage", default=None, help="Output montage mp4 path (L|C|R). Optional.")
    ap.add_argument("--out_left", default=None, help="Output per-cam mp4 (left). Optional.")
    ap.add_argument("--out_center", default=None, help="Output per-cam mp4 (center). Optional.")
    ap.add_argument("--out_right", default=None, help="Output per-cam mp4 (right). Optional.")

    ap.add_argument("--det_model", default=DET_MODEL_DEFAULT)
    ap.add_argument("--ball_model", default=BALL_MODEL_DEFAULT)
    ap.add_argument("--imgsz", type=int, default=INFERENCE_SIZE_DEFAULT)

    ap.add_argument("--conf_low", type=float, default=CONF_LOW_LIMIT_DEFAULT)
    ap.add_argument("--conf_person", type=float, default=CONF_PERSON_DEFAULT)
    ap.add_argument("--conf_ball", type=float, default=CONF_BALL_DEFAULT)
    ap.add_argument("--max_people", type=int, default=MAX_PEOPLE_DEFAULT)

    ap.add_argument("--mp_complexity", type=int, default=MP_COMPLEXITY_DEFAULT)
    ap.add_argument("--mp_min_conf", type=float, default=MIN_MP_CONF_DEFAULT)

    ap.add_argument("--show", action="store_true", help="Show preview window (press q to quit)")
    ap.add_argument("--no_reencode", action="store_true", help="Do not ffmpeg re-encode outputs")

    return ap.parse_args()

def main():
    args = _parse_args()
    video_paths = {"L": args.left, "C": args.center, "R": args.right}

    out_per_cam = None
    if args.out_left or args.out_center or args.out_right:
        out_per_cam = {}
        if args.out_left:   out_per_cam["L"] = args.out_left
        if args.out_center: out_per_cam["C"] = args.out_center
        if args.out_right:  out_per_cam["R"] = args.out_right

    cfg = HybridConfig(
        calib_npz=args.calib,
        out_csv=args.out_csv,
        out_video_montage=args.out_montage,
        out_videos_per_cam=out_per_cam,
        det_model=args.det_model,
        ball_model=args.ball_model,
        imgsz=args.imgsz,
        conf_low_limit=args.conf_low,
        conf_person=args.conf_person,
        conf_ball=args.conf_ball,
        max_people=args.max_people,
        mp_complexity=args.mp_complexity,
        mp_min_conf=args.mp_min_conf,
        show=args.show,
        save_h264_reencode=(not args.no_reencode),
    )

    run_hybrid_v4(video_paths, cfg)

if __name__ == "__main__":
    main()
