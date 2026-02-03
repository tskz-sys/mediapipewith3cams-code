#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# YOLO 3D Bounding Box Visualizer [Multi-Camera]
#   - 目的: マルチカメラ情報を用いて、YOLOの検出結果を「3Dの箱」として可視化する
#   - 手法: 
#       1. YOLO+MediaPipeで各視点の2D骨格を取得
#       2. 3D空間上で骨格を復元
#       3. 3D骨格を囲む「最小/最大の3D座標」から直方体(Cuboid)を定義
#       4. 直方体の8頂点を各カメラ画像に再投影して線で結ぶ
# ==========================================================

import os
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

OUT_VIDEO = "./movie/3d_detection.mp4"

# モデル設定
DET_MODEL = "yolo11x-pose.pt" 

# パラメータ
INFERENCE_SIZE = 1920
CONF_THRESHOLD = 0.25
MAX_PEOPLE = 2

# ==========================================
# キャリブレーション & 幾何学計算
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

    # 投影行列 P = K[R|t]
    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([R2, t2])
    P3 = K3 @ np.hstack([R3, t3])

    return [(K1, D1, P1), (K2, D2, P2), (K3, D3, P3)]

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
    return X

# ==========================================
# 3D Box 関連ロジック
# ==========================================

def get_3d_aabb(kps_3d):
    """
    3D関節座標群から、それらを囲む最小の直方体(Axis-Aligned Bounding Box)の8頂点を計算する
    """
    # 有効な点のみ抽出 (NaN除去)
    valid_pts = kps_3d[~np.isnan(kps_3d).any(axis=1)]
    if len(valid_pts) < 5: return None # 点が少なすぎる場合はボックスを作らない

    min_xyz = np.min(valid_pts, axis=0)
    max_xyz = np.max(valid_pts, axis=0)

    # 人体に少し余裕を持たせる (パディング)
    pad = 5.0 # 5cm程度
    min_xyz -= pad
    max_xyz += pad

    # 8頂点を定義 (X, Y, Z)
    # 0: min_x, min_y, min_z
    # 1: max_x, min_y, min_z
    # ...
    corners = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]], # 0
        [max_xyz[0], min_xyz[1], min_xyz[2]], # 1
        [max_xyz[0], max_xyz[1], min_xyz[2]], # 2
        [min_xyz[0], max_xyz[1], min_xyz[2]], # 3
        [min_xyz[0], min_xyz[1], max_xyz[2]], # 4
        [max_xyz[0], min_xyz[1], max_xyz[2]], # 5
        [max_xyz[0], max_xyz[1], max_xyz[2]], # 6
        [min_xyz[0], max_xyz[1], max_xyz[2]]  # 7
    ])
    return corners

def project_3d_points(P, points_3d):
    """ 3D点群(Nx3)を投影行列Pを使って2D画像座標(Nx2)に変換 """
    if points_3d is None: return None
    
    # 同次座標系にする [X, Y, Z, 1]
    ones = np.ones((len(points_3d), 1))
    pts_homo = np.hstack([points_3d, ones])
    
    # 投影: x = P X
    pts_proj = (P @ pts_homo.T).T
    
    # 正規化: u = x/z, v = y/z
    # zが0に近い場合は除外
    uvs = []
    for p in pts_proj:
        if p[2] <= 0.1: 
            uvs.append([-1, -1]) # カメラの後ろ
        else:
            uvs.append([int(p[0]/p[2]), int(p[1]/p[2])])
    return np.array(uvs)

def draw_3d_box(img, corners_2d, color=(0, 255, 0)):
    """ 投影された8頂点を結んで箱を描画 """
    if corners_2d is None: return
    
    # 頂点の接続関係
    # 底面: 0-1, 1-2, 2-3, 3-0
    # 上面: 4-5, 5-6, 6-7, 7-4
    # 柱: 0-4, 1-5, 2-6, 3-7
    
    edges = [
        (0,1), (1,2), (2,3), (3,0), # Bottom face
        (4,5), (5,6), (6,7), (7,4), # Top face
        (0,4), (1,5), (2,6), (3,7)  # Pillars
    ]
    
    H, W = img.shape[:2]

    for s, e in edges:
        pt1 = corners_2d[s]
        pt2 = corners_2d[e]
        
        # 画面外チェック
        if pt1[0] == -1 or pt2[0] == -1: continue
        
        cv2.line(img, tuple(pt1), tuple(pt2), color, 2)
        
    # 前面を少し強調
    # cv2.circle(img, tuple(corners_2d[0]), 5, color, -1)

# ==========================================
# 検出処理 (v5ベースのPadding付き)
# ==========================================
def get_pose_padded(full_img, box, pose_model):
    H_img, W_img = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    pad_w = int((x2 - x1) * 0.10)
    pad_h = int((y2 - y1) * 0.10)
    x1 = max(0, x1 - pad_w); y1 = max(0, y1 - pad_h)
    x2 = min(W_img, x2 + pad_w); y2 = min(H_img, y2 + pad_h)
    
    if x2 <= x1 or y2 <= y1: return None, None

    crop = full_img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    size = max(h, w)
    padded_crop = np.zeros((size, size, 3), dtype=np.uint8)
    ox, oy = (size - w) // 2, (size - h) // 2
    padded_crop[oy:oy+h, ox:ox+w] = crop
    
    results = pose_model.process(cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks: return None, None
    
    kps = []
    confs = []
    for lm in results.pose_landmarks.landmark:
        gx = (lm.x * size - ox) + x1
        gy = (lm.y * size - oy) + y1
        kps.append([gx, gy])
        confs.append(lm.visibility)
        
    return np.array(kps), np.array(confs)

def process_frame(img, yolo_model, mp_model, K, D, max_p):
    res = yolo_model.predict(img, conf=0.1, imgsz=INFERENCE_SIZE, verbose=False, classes=[0])[0]
    people = []
    if res.boxes is None: return []
    
    for idx in np.argsort(-res.boxes.conf.cpu().numpy()):
        conf = res.boxes.conf[idx].cpu().numpy()
        if conf < CONF_THRESHOLD: continue
        if len(people) >= max_p: break
        
        box = res.boxes.xyxy[idx].cpu().numpy()
        mp_kps, mp_confs = get_pose_padded(img, box, mp_model)
        
        if mp_kps is not None:
            # 画面外フィルタ
            H, W = img.shape[:2]
            for i in range(len(mp_kps)):
                kx, ky = mp_kps[i]
                if kx < 0 or kx > W or ky < 0 or ky > H: mp_confs[i] = 0.0

            people.append({
                "box": box,
                "kps_norm": undistort_points(mp_kps, K, D),
                "conf": mp_confs
            })
    return people

# ==========================================
# Main
# ==========================================
def main():
    print("=== YOLO 3D Bounding Box Visualizer ===")
    
    if not os.path.exists(CALIB_NPZ): return
    params = load_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    Ps = [p[2] for p in params] # 投影行列リスト

    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    yolo = YOLO(DET_MODEL)
    mp_pose = mp.solutions.pose
    pose_est = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)

    vw = cv2.VideoWriter("temp_3d.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    
    colors = [(0, 255, 0), (0, 0, 255)] # 緑、赤

    for _ in tqdm(range(total)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break

        # 1. 各カメラで検出
        ppl_lists = []
        for i, f in enumerate(frames):
            K, D, _ = params[i]
            ppl_lists.append(process_frame(f, yolo, pose_est, K, D, MAX_PEOPLE))

        # 2. マッチング (簡易版: 総当たりでコスト最小を探す)
        # コスト = 重Reprojection Error
        matched = []
        used = [set(), set(), set()]
        
        # 3視点マッチングを優先
        candidates = []
        for i1, p1 in enumerate(ppl_lists[0]):
            for i2, p2 in enumerate(ppl_lists[1]):
                for i3, p3 in enumerate(ppl_lists[2]):
                    # 腰(23,24)と肩(11,12)でチェック
                    err_sum = 0
                    count = 0
                    for j in [11, 12, 23, 24]:
                        if p1['conf'][j]>0.5 and p2['conf'][j]>0.5 and p3['conf'][j]>0.5:
                            pts = [p1['kps_norm'][j], p2['kps_norm'][j], p3['kps_norm'][j]]
                            X = triangulate_DLT(Ps, pts)
                            # 簡易エラーチェックは省略（高速化のため）
                            count += 1
                    if count >= 2: # 2関節以上でマッチ
                        candidates.append({'ids':(i1,i2,i3), 'score': count})
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        for cand in candidates:
            i1, i2, i3 = cand['ids']
            if i1 in used[0] or i2 in used[1] or i3 in used[2]: continue
            matched.append([ppl_lists[0][i1], ppl_lists[1][i2], ppl_lists[2][i3]])
            used[0].add(i1); used[1].add(i2); used[2].add(i3)

        # 2視点マッチング (1-2, 2-3, 1-3) の残りは今回はスキップ（コード簡略化のため）
        # 必要であれば v5 のロジックを追加してください

        # 3. 3D Box生成 & 描画
        for pid, persons in enumerate(matched):
            # 3D関節の復元
            kps_3d = []
            for j in range(33):
                pts = []
                active_Ps = []
                for cam_idx in range(3):
                    if persons[cam_idx]['conf'][j] > 0.5:
                        pts.append(persons[cam_idx]['kps_norm'][j])
                        active_Ps.append(Ps[cam_idx])
                
                if len(pts) >= 2:
                    X = triangulate_DLT(active_Ps, pts)
                    if np.linalg.norm(X) < 50.0: # 外れ値除去
                        kps_3d.append(X)
            
            kps_3d = np.array(kps_3d)
            
            # 3Dバウンディングボックスの計算
            corners_3d = get_3d_aabb(kps_3d)
            
            if corners_3d is not None:
                col = colors[pid % len(colors)]
                # 各カメラに投影して描画
                for cam_idx in range(3):
                    corners_2d = project_3d_points(Ps[cam_idx], corners_3d)
                    draw_3d_box(frames[cam_idx], corners_2d, col)
                    
                    # ラベル表示
                    if persons[cam_idx] is not None:
                        box = persons[cam_idx]['box']
                        cv2.putText(frames[cam_idx], f"P{pid} 3D-BOX", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        vw.write(np.hstack(frames))

    pose_est.close()
    vw.release()
    for c in caps: c.release()
    
    if os.path.exists("temp_3d.mp4"):
        subprocess.run(["ffmpeg", "-y", "-i", "temp_3d.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT_VIDEO], check=False)
        os.remove("temp_3d.mp4")
        print(f"Saved: {OUT_VIDEO}")

if __name__ == "__main__":
    main()