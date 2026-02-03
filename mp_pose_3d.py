#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# MediaPipe 3-View 3D Pose [Single Person]
#   - YOLOの代わりにMediaPipe Pose(BlazePose)を使用
#   - Pattern BR (左右反転) キャリブレーション適用済み
#   - 手先(Index, Thumb, Pinky)も取得可能
# ==========================================================

import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from tqdm import tqdm

# ==========================================
# ★ ユーザー設定
# ==========================================
VIDEO_LEFT   = "./movie/Alone1.mp4"
VIDEO_CENTER = "./movie/Alone2.mp4"
VIDEO_RIGHT  = "./movie/Alone3.mp4"

CALIB_NPZ = "./npz/11253cams_fixedd.npz"

OUT_CSV   = "./csv/mp_3d_output.csv"
OUT_VIDEO = "./movie/mp_3d_debug.mp4"

# MediaPipe設定
# model_complexity: 0=Lite, 1=Full, 2=Heavy (2が最も高精度だが遅い)
MODEL_COMPLEXITY = 2 
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF  = 0.5

# MediaPipeの関節定義 (33点)
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

# ==========================================
# キャリブレーション補正関数 (Pattern BR)
# ==========================================

def get_inverse_transform(R, T):
    R_inv = R.T
    T_inv = -R.T @ T
    return R_inv, T_inv

def scale_camera_matrix(K, dist, target_w, target_h):
    orig_w = K[0, 2] * 2
    orig_h = K[1, 2] * 2
    sx = target_w / orig_w
    sy = target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05:
        return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx; K_new[1, 1] *= sy
    K_new[0, 2] *= sx; K_new[1, 2] *= sy
    return K_new, dist

def load_final_params_BR(npz_path, v1, v2, v3):
    d = np.load(npz_path, allow_pickle=True)
    def get_k(k_new, k_old): return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1","cam_matrix1"), get_k("dist1","dist_coeffs1")
    K2, D2 = get_k("K2","cam_matrix2"), get_k("dist2","dist_coeffs2")
    K3, D3 = get_k("K3","cam_matrix3"), get_k("dist3","dist_coeffs3")

    def get_wh(v):
        c = cv2.VideoCapture(v)
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        return w, h
    
    w1, h1 = get_wh(v1); K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    w2, h2 = get_wh(v2); K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    w3, h3 = get_wh(v3); K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    R1_raw = d["R1"]; t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3,1)) # BR Pattern

    R2 = np.eye(3); t2 = np.zeros((3, 1))

    R3_raw = d["R3"]; t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3,1)) # BR Pattern

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
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]
    return X

# ==========================================
# MediaPipe 処理ロジック
# ==========================================

def get_mp_keypoints(image, pose_model, W, H):
    """
    画像からMediaPipeでキーポイントを取得 (Pixel座標に変換)
    """
    # MediaPipeはRGB入力を期待
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)
    
    kps = []
    confs = []
    
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            # Normalized (0.0-1.0) -> Pixel Coordinates
            px = lm.x * W
            py = lm.y * H
            kps.append([px, py])
            confs.append(lm.visibility) # visibilityを信頼度として使う
    else:
        # 検出されなかった場合
        return None, None
        
    return np.array(kps), np.array(confs)

# ==========================================
# メイン
# ==========================================
def main():
    print(f"=== Starting MediaPipe 3D Pose (Complexity: {MODEL_COMPLEXITY}) ===")
    
    if not os.path.exists(CALIB_NPZ):
        print("Error: Calibration file not found.")
        return

    # パラメータロード
    cam_params_full, extrinsics = load_final_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    P1, P2, P3 = [p[2] for p in cam_params_full]

    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MediaPipe Pose 初期化
    mp_pose = mp.solutions.pose
    # 3台分それぞれのインスタンスを作る（状態保持のため）
    poses = [
        mp_pose.Pose(static_image_mode=False, model_complexity=MODEL_COMPLEXITY, 
                     min_detection_confidence=MIN_DETECTION_CONF, min_tracking_confidence=MIN_TRACKING_CONF)
        for _ in range(3)
    ]

    f_csv = open(OUT_CSV, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z", "conf"])

    history_3d = {}

    print(f"Processing {total_frames} frames...")
    for i in tqdm(range(total_frames)):
        frames = []
        for c in caps:
            ret, f = c.read()
            frames.append(f if ret else None)
        if any(f is None for f in frames): break

        # 各カメラで検出
        kps_raw_list = []
        confs_list = []
        
        for cam_i in range(3):
            kps, conf = get_mp_keypoints(frames[cam_i], poses[cam_i], W, H)
            kps_raw_list.append(kps)
            confs_list.append(conf)

        # 3視点とも検出できた場合のみ3D化
        if all(k is not None for k in kps_raw_list):
            
            # 歪み補正
            ns_list = []
            for cam_i in range(3):
                K, D, _ = cam_params_full[cam_i]
                ns = undistort_points(kps_raw_list[cam_i], K, D)
                ns_list.append(ns)
            
            frame_kps_3d = np.full((33, 3), np.nan)
            
            # 全33関節についてループ
            for j in range(33):
                # 信頼度チェック (全視点で閾値超え)
                c1, c2, c3 = confs_list[0][j], confs_list[1][j], confs_list[2][j]
                if c1 > 0.5 and c2 > 0.5 and c3 > 0.5:
                    pts = [ns_list[0][j], ns_list[1][j], ns_list[2][j]]
                    Ps  = [P1, P2, P3]
                    
                    X = triangulate_DLT(Ps, pts)
                    
                    if np.linalg.norm(X) < 50.0:
                        frame_kps_3d[j] = X
                        min_conf = min(c1, c2, c3)
                        writer.writerow([i, 0, MP_JOINTS[j], X[0], X[1], X[2], min_conf])
            
            history_3d[i] = frame_kps_3d

    f_csv.close()
    for c in caps: c.release()
    for p in poses: p.close()

    # ==========================================
    # 確認動画生成
    # ==========================================
    print("Generating confirmation video...")
    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    
    temp_out = OUT_VIDEO + ".temp.mp4"
    out_w_total = W * 3
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w_total, H))
    
    # 描画用の接続定義 (MediaPipe Style)
    MP_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
    LIMIT = 5000

    for i in tqdm(range(total_frames)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3: break

        if i in history_3d:
            kps3d = history_3d[i]
            for cam_i in range(3):
                K, D, _ = cam_params_full[cam_i]
                R, t = extrinsics[cam_i]
                rvec, _ = cv2.Rodrigues(R)
                
                mask = ~np.isnan(kps3d).any(axis=1)
                if np.any(mask):
                    valid_3d = kps3d[mask]
                    img_pts, _ = cv2.projectPoints(valid_3d, rvec, t, K, D)
                    img_pts = img_pts.reshape(-1, 2)
                    
                    kps2d = {}
                    cnt = 0
                    for j in range(33):
                        if mask[j]: kps2d[j] = img_pts[cnt]; cnt += 1
                        
                    # 描画 (Cyan Color for MediaPipe)
                    for conn in MP_CONNECTIONS:
                        u, v = conn
                        if u in kps2d and v in kps2d:
                            pt1 = (int(np.clip(kps2d[u][0], -LIMIT, LIMIT)), int(np.clip(kps2d[u][1], -LIMIT, LIMIT)))
                            pt2 = (int(np.clip(kps2d[v][0], -LIMIT, LIMIT)), int(np.clip(kps2d[v][1], -LIMIT, LIMIT)))
                            cv2.line(frames[cam_i], pt1, pt2, (255, 255, 0), 2)
                    
                    # 手先を目立たせる (Pinky, Index, Thumb)
                    for tip_idx in [17, 18, 19, 20, 21, 22]:
                        if tip_idx in kps2d:
                            pt = (int(np.clip(kps2d[tip_idx][0], -LIMIT, LIMIT)), int(np.clip(kps2d[tip_idx][1], -LIMIT, LIMIT)))
                            cv2.circle(frames[cam_i], pt, 5, (0, 0, 255), -1)

        combined = np.hstack(frames)
        vw.write(combined)

    vw.release()
    for c in caps: c.release()

    if os.path.exists(temp_out):
        cmd = ["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT_VIDEO]
        subprocess.run(cmd, check=False)
        if os.path.exists(OUT_VIDEO):
            os.remove(temp_out)
            print(f"Done! Saved: {OUT_VIDEO}")

if __name__ == "__main__":
    main()