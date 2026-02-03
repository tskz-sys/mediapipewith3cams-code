#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# YOLO + MediaPipe Hybrid [v3 OPTIMIZED]
#   - 修正: NameErrorの解消
#   - 修正: ボール検出精度の向上 (低閾値 + 高解像度)
#   - 機能: Dual 2D Pose (YOLO黄点 + MP線)
# ==========================================================

import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# ★ ユーザー設定
# ==========================================
VIDEO_LEFT   = "../ffmpeg/output/1match1_2.mp4"
VIDEO_CENTER = "../ffmpeg/output/2match1_2.mp4"
VIDEO_RIGHT  = "../ffmpeg/output/3match1_2.mp4"

CALIB_NPZ = "./npz/11253cams_fixedd.npz"

OUT_CSV   = "./csv/kakunin.csv"
OUT_VIDEO = "./movie/kakunin.mp4"

# --- モデル設定 ---
DET_MODEL = "yolo11x-pose.pt" # Person用 (Pose)
BALL_MODEL = "yolo11x.pt"     # Ball用 (Detect)

# --- 検出パラメータ (精度向上のため修正) ---
INFERENCE_SIZE = 1920  # 画像を縮小せずフルサイズで見る (ボール検出に重要)
CONF_LOW_LIMIT = 0.10  # 足切りライン（低めに設定して取りこぼしを防ぐ）

CONF_PERSON    = 0.25  # 人として採用する閾値
CONF_BALL      = 0.15  # ボールとして採用する閾値

MAX_PEOPLE     = 2

# MediaPipe設定
MP_COMPLEXITY = 1
MIN_MP_CONF   = 0.5

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
# マッチング用
USE_JOINTS_IDX = [11, 12, 23, 24, 25, 26] 

# ==========================================
# キャリブレーション補正 (Pattern BR)
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
    d = np.load(npz_path, allow_pickle=True)
    def get_k(k_new, k_old): return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1","cam_matrix1"), get_k("dist1","dist_coeffs1")
    K2, D2 = get_k("K2","cam_matrix2"), get_k("dist2","dist_coeffs2")
    K3, D3 = get_k("K3","cam_matrix3"), get_k("dist3","dist_coeffs3")

    def get_wh(v):
        c = cv2.VideoCapture(v); w, h = int(c.get(3)), int(c.get(4)); c.release()
        return w, h
    
    w1, h1 = get_wh(v1); K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    w2, h2 = get_wh(v2); K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    w3, h3 = get_wh(v3); K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    R1_raw = d["R1"]; t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3,1))

    R2 = np.eye(3); t2 = np.zeros((3, 1))

    R3_raw = d["R3"]; t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3,1))

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
        A.append(x * P[2] - P[0]); A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]; X = X[:3] / X[3]
    
    errs = []
    for P, (x, y) in zip(Ps, pts):
        xh = P @ np.hstack([X, 1.0])
        if abs(xh[2]) < 1e-9: errs.append(1000.0); continue
        xp, yp = xh[0]/xh[2], xh[1]/xh[2]
        errs.append((xp-x)**2 + (yp-y)**2)
    return X, np.mean(errs)

# ==========================================
# 検出ロジック (最適化版)
# ==========================================

def get_pose_from_crop(full_img, box, pose_model):
    """ YOLOの枠を切り抜いてMediaPipeにかける """
    H, W = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    pad_w = int((x2 - x1) * 0.15)
    pad_h = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - pad_w); y1 = max(0, y1 - pad_h)
    x2 = min(W, x2 + pad_w); y2 = min(H, y2 + pad_h)
    
    if x2 <= x1 or y2 <= y1: return None, None, (x1, y1, x2, y2)

    crop = full_img[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    results = pose_model.process(crop_rgb)
    
    if not results.pose_landmarks: return None, None, (x1, y1, x2, y2)
    
    kps = []
    confs = []
    crop_h, crop_w = crop.shape[:2]
    
    for lm in results.pose_landmarks.landmark:
        gx = lm.x * crop_w + x1
        gy = lm.y * crop_h + y1
        kps.append([gx, gy])
        confs.append(lm.visibility)
        
    return np.array(kps), np.array(confs), (x1, y1, x2, y2)

def process_person_yolo(img, pose_model, mp_model, K, D, max_p):
    # 人の検出 (YOLO Pose)
    # 低い閾値で広く拾う
    res = pose_model.predict(img, conf=CONF_LOW_LIMIT, imgsz=INFERENCE_SIZE, verbose=False, classes=[0])[0]
    
    people = []
    if res.boxes is None or len(res.boxes) == 0: return []

    # 信頼度順
    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    
    p_count = 0
    for idx in indices:
        conf = res.boxes.conf[idx].cpu().numpy()
        
        # 人用閾値チェック
        if conf < CONF_PERSON: continue
        if p_count >= max_p: break

        box = res.boxes.xyxy[idx].cpu().numpy()
        yolo_kps = res.keypoints.xy[idx].cpu().numpy() # YOLO 2D Raw

        # MediaPipe Pose
        mp_kps, mp_confs, padded_box = get_pose_from_crop(img, box, mp_model)
        
        if mp_kps is not None:
            norm_mp_kps = undistort_points(mp_kps, K, D)
            
            people.append({
                "type": "person",
                "box": box,
                "yolo_kps": yolo_kps,    # YOLO Raw
                "kps_raw": mp_kps,       # MP Raw
                "kps_norm": norm_mp_kps, # MP Norm
                "conf": mp_confs
            })
            p_count += 1
    return people

def detect_ball_yolo(img, det_model, K, D):
    # ボールの検出 (YOLO Detect)
    # クラス32のみ、低閾値、高解像度
    res = det_model.predict(img, conf=CONF_LOW_LIMIT, imgsz=INFERENCE_SIZE, verbose=False, classes=[32])[0]
    
    balls = []
    if res.boxes is None or len(res.boxes) == 0: return []

    # 信頼度順
    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    
    for idx in indices:
        conf = res.boxes.conf[idx].cpu().numpy()
        
        # ボール用閾値チェック (CONF_BALL = 0.15)
        if conf > CONF_BALL:
            # 1つ見つかればOK (一番信頼度が高いもの)
            box = res.boxes.xyxy[idx].cpu().numpy()
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            norm_pt = undistort_points([(cx, cy)], K, D)[0]
            
            balls.append({
                "center_raw": np.array([cx, cy]),
                "center_norm": norm_pt,
                "box": box,
                "type": "ball",
                "conf": conf
            })
            break # 1フレーム1個と仮定
            
    return balls

# ==========================================
# Main
# ==========================================

def main():
    print(f"=== Hybrid v3 Optimized: Dual Pose + Ball ===")
    
    if not os.path.exists(CALIB_NPZ): return

    cam_params_full, extrinsics = load_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    P1, P2, P3 = [p[2] for p in cam_params_full]

    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Models
    print("Loading YOLO Pose (Person)...")
    yolo_pose = YOLO(DET_MODEL) 
    
    print("Loading YOLO Detect (Ball)...")
    yolo_det = YOLO(BALL_MODEL) 
    
    print("Loading MediaPipe...")
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=MP_COMPLEXITY,
        min_detection_confidence=MIN_MP_CONF
    )

    f_csv = open(OUT_CSV, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z"])

    temp_out = OUT_VIDEO + ".temp.mp4"
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    
    mp_conn = mp.solutions.pose.POSE_CONNECTIONS
    colors = [(0,255,0), (0,0,255), (255,255,0)]
    LIMIT = 5000 

    print(f"Processing {total_frames} frames...")
    
    for i in tqdm(range(total_frames)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break

        ppl_lists = []
        ball_lists = []
        
        # --- 1. Detection ---
        for cam_i, f in enumerate(frames):
            K, D, _ = cam_params_full[cam_i]
            # Person Detection
            p = process_person_yolo(f, yolo_pose, pose_estimator, K, D, MAX_PEOPLE)
            ppl_lists.append(p)
            # Ball Detection
            b = detect_ball_yolo(f, yolo_det, K, D)
            ball_lists.append(b)

        # --- 2. Person Matching ---
        candidates = []
        for idx1 in range(len(ppl_lists[0])):
            for idx2 in range(len(ppl_lists[1])):
                for idx3 in range(len(ppl_lists[2])):
                    p1, p2, p3 = ppl_lists[0][idx1], ppl_lists[1][idx2], ppl_lists[2][idx3]
                    cost_sum, count = 0, 0
                    for j in USE_JOINTS_IDX:
                        if p1["conf"][j] > 0.5 and p2["conf"][j] > 0.5 and p3["conf"][j] > 0.5:
                            pts = [p1["kps_norm"][j], p2["kps_norm"][j], p3["kps_norm"][j]]
                            _, err = triangulate_DLT([P1, P2, P3], pts)
                            if err < 0.2:
                                cost_sum += err; count += 1
                    if count >= 3:
                        candidates.append({'ids': (idx1, idx2, idx3), 'cost': cost_sum/count})

        candidates.sort(key=lambda x: x['cost'])
        used1, used2, used3 = set(), set(), set()
        pid_counter = 0

        # --- 3. Person Triangulation & Draw ---
        for cand in candidates:
            i1, i2, i3 = cand['ids']
            if i1 in used1 or i2 in used2 or i3 in used3: continue
            
            persons = [ppl_lists[0][i1], ppl_lists[1][i2], ppl_lists[2][i3]]
            col = colors[pid_counter % len(colors)]
            kps_3d = np.full((33, 3), np.nan)
            
            # 3D化
            for j in range(33):
                if all(p["conf"][j]>0.5 for p in persons):
                    pts = [p["kps_norm"][j] for p in persons]
                    X, _ = triangulate_DLT([P1, P2, P3], pts)
                    if np.linalg.norm(X) < 50.0:
                        kps_3d[j] = X
                        writer.writerow([i, pid_counter, MP_JOINTS[j], X[0], X[1], X[2]])

            # 描画
            for cam_i in range(3):
                frame = frames[cam_i]
                p_data = persons[cam_i]
                
                # A. YOLO 2D (Yellow Dots)
                if "yolo_kps" in p_data:
                    for ykp in p_data["yolo_kps"]:
                        yx, yy = int(ykp[0]), int(ykp[1])
                        if 0 <= yx < W and 0 <= yy < H:
                            cv2.circle(frame, (yx, yy), 3, (0, 255, 255), -1)

                # B. MediaPipe Skeleton (Lines)
                kps_raw = p_data["kps_raw"]
                for u, v in mp_conn:
                    if u < len(kps_raw) and v < len(kps_raw):
                        pt1 = (int(kps_raw[u][0]), int(kps_raw[u][1]))
                        pt2 = (int(kps_raw[v][0]), int(kps_raw[v][1]))
                        cv2.line(frame, pt1, pt2, col, 2)
                
                # C. Fingers (Cyan)
                for tip in [17, 18, 19, 20, 21, 22]:
                    pt = (int(kps_raw[tip][0]), int(kps_raw[tip][1]))
                    cv2.circle(frame, pt, 4, (255, 255, 0), -1)

            used1.add(i1); used2.add(i2); used3.add(i3)
            pid_counter += 1

        # --- 4. Ball ---
        b1, b2, b3 = [bl[0] if bl else None for bl in ball_lists]
        if b1 and b2 and b3:
            pts_b = [b1["center_norm"], b2["center_norm"], b3["center_norm"]]
            X_b, err_b = triangulate_DLT([P1, P2, P3], pts_b)
            if np.linalg.norm(X_b) < 50.0 and err_b < 5.0:
                writer.writerow([i, -1, "ball", X_b[0], X_b[1], X_b[2]])
                
                for cam_i, bl in enumerate([b1, b2, b3]):
                    box = bl["box"]
                    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    cv2.circle(frames[cam_i], (cx, cy), 8, (0, 165, 255), -1)
                    cv2.putText(frames[cam_i], "BALL", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        vw.write(np.hstack(frames))

    f_csv.close()
    pose_estimator.close()
    vw.release()
    for c in caps: c.release()

    if os.path.exists(temp_out):
        subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT_VIDEO], check=False)
        os.remove(temp_out)
        print(f"Saved: {OUT_VIDEO}")

if __name__ == "__main__":
    main()