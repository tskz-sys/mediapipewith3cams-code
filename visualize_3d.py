#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Test5 CSV Reprojection [BR Fixed & Distortion Aware]
#   - 入力: test5.csv (hybrid_test2.pyで作成されたもの)
#   - BR問題: get_inverse_transform を適用して座標系を合わせる
#   - 歪み: cv2.projectPoints でレンズ歪みを再現
# ==========================================================

import os
import cv2
import csv
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm

# ==========================================
# ★ ユーザー設定
# ==========================================
VIDEO_LEFT   = "../ffmpeg/output/1match1_2.mp4"
VIDEO_CENTER = "../ffmpeg/output/2match1_2.mp4"
VIDEO_RIGHT  = "../ffmpeg/output/3match1_2.mp4"

CALIB_NPZ = "./npz/11253cams_fixedd.npz"
INPUT_CSV = "./csv/test5.csv"

# 出力ファイル
TEMP_VIDEO = "./movie/temp5.mp4"
OUTPUT_VIDEO = "./movie/test5_reprojection.mp4"

# 描画から除外する関節（顔など）
IGNORE_JOINTS = {
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right'
}

# 骨格の接続定義
SKELETON_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_ankle', 'left_heel'), ('left_ankle', 'left_foot_index'),
    ('right_ankle', 'right_heel'), ('right_ankle', 'right_foot_index'),
    ('left_wrist', 'left_pinky'), ('left_wrist', 'left_index'), ('left_wrist', 'left_thumb'),
    ('right_wrist', 'right_pinky'), ('right_wrist', 'right_index'), ('right_wrist', 'right_thumb')
]

# IDごとの色 (B, G, R)
COLORS = {
    0: (0, 255, 0),    # Green
    1: (0, 0, 255),    # Red
    -1: (0, 255, 255)  # Yellow (Ball/Unknown)
}

# ==========================================
# ★重要: キャリブレーション (BR問題対応)
# ==========================================
def get_inverse_transform(R, T):
    """
    BR問題対応: カメラの位置(Pose)から、射影用の外部パラメータ(Extrinsics)へ変換
    R_new = R.T
    T_new = -R.T @ T
    """
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
    """ 
    hybrid_test2.py と全く同じロジックでパラメータを読み込む 
    これにより CSV の座標系(Cam2原点)と一致させる
    """
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

    # ★重要: ここで逆変換を入れることで、test5.csvの座標系と一致させる
    R1_raw = d["R1"]; t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3,1))

    # Cam2を原点とする
    R2 = np.eye(3); t2 = np.zeros((3, 1))

    R3_raw = d["R3"]; t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3,1))

    # K, D, R, t のセットを返す
    return [(K1, D1, R1, t1), (K2, D2, R2, t2), (K3, D3, R3, t3)]

# ==========================================
# 幾何計算 & データ処理
# ==========================================
def project_point_distorted(X_3d, K, D, R, t):
    """ 
    レンズ歪みを考慮して3D点を2Dに投影 (cv2.projectPoints)
    """
    # 3D座標の配列化 (1, 3)
    object_points = np.array([X_3d], dtype=np.float64) 
    
    # Rodrigues変換
    rvec, _ = cv2.Rodrigues(R)
    
    # 投影
    img_points, _ = cv2.projectPoints(object_points, rvec, t, K, D)
    
    u, v = img_points[0][0]
    return (int(u), int(v))

def load_csv_data(csv_path):
    print(f"Loading CSV: {csv_path} ...")
    if not os.path.exists(csv_path):
        print("CSV file not found.")
        return {}

    df = pd.read_csv(csv_path)
    data = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        f = int(row['frame'])
        pid = int(row['person_id'])
        jname = row['joint']
        
        if jname in IGNORE_JOINTS: continue
            
        x, y, z = row['X'], row['Y'], row['Z']
        
        if pd.isna(x) or pd.isna(y) or pd.isna(z): continue
            
        if f not in data: data[f] = {}
        if pid not in data[f]: data[f][pid] = {}
        data[f][pid][jname] = np.array([x, y, z])
        
    return data

def draw_skeleton(img, proj_2d, color):
    for j1, j2 in SKELETON_CONNECTIONS:
        if j1 in proj_2d and j2 in proj_2d:
            cv2.line(img, proj_2d[j1], proj_2d[j2], color, 2)
    
    for jname, pt in proj_2d.items():
        if jname == 'ball':
            cv2.circle(img, pt, 8, (0, 165, 255), -1)
            cv2.putText(img, "Ball", (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.circle(img, pt, 4, color, -1)

# ==========================================
# Main
# ==========================================
def main():
    print("=== Test5 Reprojection (BR Fixed) ===")
    
    if not os.path.exists(CALIB_NPZ):
        print("Calibration file missing.")
        return

    # 1. パラメータ取得 (BR問題対応済み)
    cams = load_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    
    # 2. 動画準備
    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. CSVデータ読み込み
    csv_data = load_csv_data(INPUT_CSV)
    
    # 4. 書き出し準備
    vw = cv2.VideoWriter(TEMP_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    
    print(f"Processing {total_frames} frames...")
    
    for frame_idx in tqdm(range(total_frames)):
        frames = []
        for c in caps: 
            ret, f = c.read()
            if not ret: f = np.zeros((H, W, 3), dtype=np.uint8)
            frames.append(f)
            
        if len(frames) < 3: break

        if frame_idx in csv_data:
            persons = csv_data[frame_idx]
            
            for cam_i in range(3):
                K, D, R, t = cams[cam_i]
                img = frames[cam_i]
                
                for pid, joints in persons.items():
                    col = COLORS.get(pid, COLORS[-1])
                    proj_2d = {} 
                    
                    for jname, pos_3d in joints.items():
                        # レンズ歪み + BR対応済みExtrinsics で再投影
                        uv = project_point_distorted(pos_3d, K, D, R, t)
                        
                        # 画面内チェック
                        if -100 <= uv[0] < W+100 and -100 <= uv[1] < H+100:
                            proj_2d[jname] = uv
                    
                    draw_skeleton(img, proj_2d, col)
                    
                    # ID表示
                    center_pt = None
                    if 'left_shoulder' in proj_2d and 'right_shoulder' in proj_2d:
                        center_pt = (
                            (proj_2d['left_shoulder'][0] + proj_2d['right_shoulder'][0]) // 2,
                            (proj_2d['left_shoulder'][1] + proj_2d['right_shoulder'][1]) // 2
                        )
                    elif len(proj_2d) > 0:
                        center_pt = list(proj_2d.values())[0]
                    
                    if center_pt and pid != -1:
                        cv2.putText(img, f"ID:{pid}", (center_pt[0], center_pt[1]-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        vw.write(np.hstack(frames))

    vw.release()
    for c in caps: c.release()

    # FFmpeg変換
    print("Converting video with FFmpeg...")
    if os.path.exists(TEMP_VIDEO):
        cmd = [
            "ffmpeg", "-y",
            "-i", str(TEMP_VIDEO),
            "-vf", "scale=2880:540,fps=30",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.0",
            "-an",
            str(OUTPUT_VIDEO),
        ]
        subprocess.run(cmd, check=True)
        os.remove(TEMP_VIDEO)
        print(f"Done. Output: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()