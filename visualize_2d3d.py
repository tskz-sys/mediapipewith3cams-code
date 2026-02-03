#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# VISUALIZE: 3D CSV Reprojection + YOLO 2D Detection
#   - 入力: 動画(Prematch), 3D CSV(MediaPipe)
#   - 処理: 
#       1. YOLOでその場の2D骨格を検出して描画 (Yellow Dots)
#       2. CSVの3D座標を読み込んで再投影して描画 (Green/Red Lines)
# ==========================================================

import os
import csv
import cv2
import numpy as np
import subprocess
import mediapipe as mp
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# ★ ユーザー設定
# ==========================================
VIDEO_LEFT   = "./movie/Prematch1.mp4"
VIDEO_CENTER = "./movie/Prematch2.mp4"
VIDEO_RIGHT  = "./movie/Prematch3.mp4"

CALIB_NPZ = "./npz/11253cams_fixedd.npz"
INPUT_CSV = "./csv/hybrid_output.csv"

OUT_VIDEO = "./movie/prematch_hybrid_vis_with_2d.mp4"

# モデル設定 (2D検出用)
DET_MODEL = "yolo11x-pose.pt" # 2D検出用にYOLOを使用
CONF_THRES = 0.25

# MediaPipe関節定義 (CSVの読み込み用)
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
J_MAP_MP = {n: i for i, n in enumerate(MP_JOINTS)}

# ==========================================
# 関数群 (Pattern BR)
# ==========================================

def get_inverse_transform(R, T):
    R_inv = R.T
    T_inv = -R.T @ T
    return R_inv, T_inv

def scale_camera_matrix(K, dist, target_w, target_h):
    orig_w, orig_h = K[0, 2] * 2, K[1, 2] * 2
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05: return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx; K_new[1, 1] *= sy
    K_new[0, 2] *= sx; K_new[1, 2] *= sy
    return K_new, dist

def load_params_BR(npz_path, v1, v2, v3):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
        
    d = np.load(npz_path, allow_pickle=True)
    def get_k(k_new, k_old): return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1","cam_matrix1"), get_k("dist1","dist_coeffs1")
    K2, D2 = get_k("K2","cam_matrix2"), get_k("dist2","dist_coeffs2")
    K3, D3 = get_k("K3","cam_matrix3"), get_k("dist3","dist_coeffs3")

    def get_wh(v):
        c = cv2.VideoCapture(v)
        if not c.isOpened(): return 1920, 1080 # Fallback
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        return w, h
    
    w1, h1 = get_wh(v1); K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    w2, h2 = get_wh(v2); K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    w3, h3 = get_wh(v3); K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    R1_raw = d["R1"]; t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3,1))

    R2 = np.eye(3); t2 = np.zeros((3, 1))

    R3_raw = d["R3"]; t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3,1))

    return [(K1, D1), (K2, D2), (K3, D3)], [(R1, t1), (R2, t2), (R3, t3)]

def load_csv_data(csv_path):
    print(f"Loading CSV: {csv_path} ...")
    if not os.path.exists(csv_path): raise FileNotFoundError("CSV file not found")

    data = {} 
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row['frame'])
            pid = int(row['person_id'])
            joint = row['joint']
            try: x, y, z = float(row['X']), float(row['Y']), float(row['Z'])
            except: continue
            
            if fi not in data: data[fi] = {'people': {}, 'ball': None}
            
            if joint == 'ball':
                data[fi]['ball'] = np.array([x, y, z])
            else:
                if pid not in data[fi]['people']:
                    data[fi]['people'][pid] = np.full((33, 3), np.nan)
                if joint in J_MAP_MP:
                    data[fi]['people'][pid][J_MAP_MP[joint]] = [x, y, z]
    return data

# ==========================================
# メイン
# ==========================================
def main():
    if not os.path.exists(CALIB_NPZ):
        print(f"NPZ not found: {CALIB_NPZ}")
        return

    # 1. パラメータロード
    print("Loading Parameters (Pattern BR)...")
    cam_params, extrinsics = load_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    
    # 2. CSVロード (3D結果)
    history_data = load_csv_data(INPUT_CSV)
    
    # 3. YOLOロード (2D可視化用)
    print("Loading YOLO for 2D visualization...")
    model = YOLO(DET_MODEL)

    # 4. 動画準備
    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_out = OUT_VIDEO + ".temp.mp4"
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    
    # 描画設定
    mp_conn = mp.solutions.pose.POSE_CONNECTIONS
    colors_3d = [(0,255,0), (0,0,255)] # 3D骨格: 緑, 赤
    LIMIT = 5000

    print(f"Processing {total_frames} frames...")
    
    for i in tqdm(range(total_frames)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break
        
        # --- A. YOLO 2D Detection & Draw (Yellow Dots) ---
        # その場で検出して描画
        for cam_i in range(3):
            res = model.predict(frames[cam_i], conf=CONF_THRES, verbose=False)[0]
            if res.keypoints is not None and res.keypoints.xy is not None:
                kps_batch = res.keypoints.xy.cpu().numpy() # [N, 17, 2]
                for kps in kps_batch:
                    for (x, y) in kps:
                        if x > 0 and y > 0:
                            # 黄色い点で描画 (Raw 2D Keypoints)
                            cv2.circle(frames[cam_i], (int(x), int(y)), 4, (0, 255, 255), -1)

        # --- B. 3D CSV Reprojection & Draw (Green Lines) ---
        if i in history_data:
            frame_info = history_data[i]
            
            # People
            for pid, kps3d in frame_info['people'].items():
                col = colors_3d[pid % len(colors_3d)]
                
                for cam_i in range(3):
                    K, D = cam_params[cam_i]
                    R, t = extrinsics[cam_i]
                    rvec, _ = cv2.Rodrigues(R)
                    
                    mask = ~np.isnan(kps3d).any(axis=1)
                    if np.any(mask):
                        img_pts, _ = cv2.projectPoints(kps3d[mask], rvec, t, K, D)
                        img_pts = img_pts.reshape(-1, 2)
                        
                        kps2d = {}
                        cnt = 0
                        for j in range(33):
                            if mask[j]: kps2d[j] = img_pts[cnt]; cnt+=1
                        
                        # Skeleton Lines
                        for u, v in mp_conn:
                            if u in kps2d and v in kps2d:
                                pt1 = (int(np.clip(kps2d[u][0], -LIMIT, LIMIT)), int(np.clip(kps2d[u][1], -LIMIT, LIMIT)))
                                pt2 = (int(np.clip(kps2d[v][0], -LIMIT, LIMIT)), int(np.clip(kps2d[v][1], -LIMIT, LIMIT)))
                                
                                # 画面内判定
                                if (0 <= pt1[0] < W and 0 <= pt1[1] < H) or (0 <= pt2[0] < W and 0 <= pt2[1] < H):
                                    cv2.line(frames[cam_i], pt1, pt2, col, 2)
                        
                        # 手先を目立たせる (Pinky, Index, Thumb)
                        for tip in [17, 18, 19, 20, 21, 22]:
                            if tip in kps2d:
                                pt = (int(np.clip(kps2d[tip][0], -LIMIT, LIMIT)), int(np.clip(kps2d[tip][1], -LIMIT, LIMIT)))
                                if 0 <= pt[0] < W and 0 <= pt[1] < H:
                                    cv2.circle(frames[cam_i], pt, 4, (255, 0, 255), -1) # Magenta

            # Ball
            if frame_info['ball'] is not None:
                b3d = frame_info['ball']
                for cam_i in range(3):
                    K, D = cam_params[cam_i]
                    R, t = extrinsics[cam_i]
                    rvec, _ = cv2.Rodrigues(R)
                    pt, _ = cv2.projectPoints(b3d.reshape(1,1,3), rvec, t, K, D)
                    bx = int(np.clip(pt[0][0][0], -LIMIT, LIMIT))
                    by = int(np.clip(pt[0][0][1], -LIMIT, LIMIT))
                    
                    if 0 <= bx < W and 0 <= by < H:
                        cv2.circle(frames[cam_i], (bx, by), 10, (0, 165, 255), 2)
                        cv2.putText(frames[cam_i], "BALL (3D)", (bx+10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        vw.write(np.hstack(frames))

    vw.release()
    for c in caps: c.release()

    # FFmpeg conversion
    if os.path.exists(temp_out):
        print("Converting to H.264...")
        subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT_VIDEO], check=False)
        os.remove(temp_out)
        print(f"Done! Saved: {OUT_VIDEO}")

if __name__ == "__main__":
    main()