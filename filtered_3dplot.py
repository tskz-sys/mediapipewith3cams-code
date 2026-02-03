#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Hybrid 3D Tracking v9 (Fixed CSV Output & Skeleton Draw)
#   - YOLO + MediaPipe
#   - 3D再構成 + 50cm移動制限フィルタ
#   - ★修正: CSVへのデータ書き出しを実装
#   - ★修正: 骨格線の描画ロジックを安定化
#   - FFmpeg自動変換
# ==========================================================

import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# ★ ユーザー設定
# ==========================================
VIDEO_LEFT   = "../ffmpeg/output/1match1_2.mp4"
VIDEO_CENTER = "../ffmpeg/output/2match1_2.mp4"
VIDEO_RIGHT  = "../ffmpeg/output/3match1_2.mp4"

CALIB_NPZ = "./npz/11253cams_fixedd.npz"

# 出力ファイル設定
OUT_CSV_PATH    = "./csv/test8.csv"          # CSV出力先
TEMP_VIDEO_PATH = "./movie/test8.mp4"   # 一時動画
FINAL_OUTPUT_PATH = "./movie/testtt8.mp4"      # 最終動画

# モデル設定
DET_MODEL = "yolo11x.pt" 
MAX_PEOPLE = 2
MAX_MOVE_METER = 0.40  # 50cmフィルタ

MP_COMPLEXITY = 1
MIN_MP_CONF   = 0.3

# 関節名リスト (CSVヘッダー用)
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

# 骨格接続定義 (描画用)
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), 
    (12, 14), (14, 16),           
    (11, 23), (12, 24),           
    (23, 24),                     
    (23, 25), (25, 27), (27, 29), (29, 31), 
    (24, 26), (26, 28), (28, 30), (30, 32)  
]

# ==========================================
# クラス: 3Dフィルタリング & トラッカー
# ==========================================
class PersonTracker3D:
    def __init__(self, pid):
        self.pid = pid
        self.center_3d = None
        # 前回の有効な3D座標 {joint_idx: np.array([x, y, z])}
        self.last_valid_joints_3d = {}
        # 今回描画・記録する3D座標 {joint_idx: np.array([x, y, z])}
        self.current_filtered_joints = {}

    def update_center(self, kps_3d):
        hips = []
        if not np.isnan(kps_3d[23][0]): hips.append(kps_3d[23])
        if not np.isnan(kps_3d[24][0]): hips.append(kps_3d[24])
        
        if len(hips) > 0:
            self.center_3d = np.mean(hips, axis=0)
        else:
            valid = kps_3d[~np.isnan(kps_3d).any(axis=1)]
            if len(valid) > 0:
                self.center_3d = np.mean(valid, axis=0)

    def filter_and_update(self, raw_kps_3d):
        """ 50cm移動制限フィルタを適用し、結果を保存する """
        self.current_filtered_joints = {} # リセット
        
        filtered_array = np.full((33, 3), np.nan)

        for j_idx in range(33):
            curr_pos = raw_kps_3d[j_idx]
            
            # 1. 検出なし (NaN)
            if np.isnan(curr_pos[0]):
                # もし前回値があれば、それを維持して描画させるか？
                # ここでは「ロスト」としてNaNのままにする
                continue 

            # 2. 初回検出
            if j_idx not in self.last_valid_joints_3d:
                self.last_valid_joints_3d[j_idx] = curr_pos
                self.current_filtered_joints[j_idx] = curr_pos
                filtered_array[j_idx] = curr_pos
                continue

            # 3. 距離チェック
            prev_pos = self.last_valid_joints_3d[j_idx]
            dist = np.linalg.norm(curr_pos - prev_pos)

            if dist > MAX_MOVE_METER:
                # 50cm以上動いた -> 誤検出とみなす
                # 「前回の位置」を採用することで、CSVには安定した値を書き、動画も飛ばないようにする
                filtered_array[j_idx] = prev_pos
                self.current_filtered_joints[j_idx] = prev_pos
                # last_valid_joints_3d は更新しない（誤検出に引っ張られないため）
            else:
                # 正常 -> 更新
                self.last_valid_joints_3d[j_idx] = curr_pos
                self.current_filtered_joints[j_idx] = curr_pos
                filtered_array[j_idx] = curr_pos
        
        return filtered_array

# ==========================================
# 関数群
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
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    X = X[:3] / X[3]
    return X

def get_pose_padded(full_img, box, pose_model):
    H_img, W_img = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    pad_w = int((x2 - x1) * 0.2)
    pad_h = int((y2 - y1) * 0.2)
    x1 = max(0, x1 - pad_w); y1 = max(0, y1 - pad_h)
    x2 = min(W_img, x2 + pad_w); y2 = min(H_img, y2 + pad_h)
    
    if x2 <= x1 or y2 <= y1: return None, None

    crop = full_img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    ox, oy = (size - w)//2, (size - h)//2
    padded[oy:oy+h, ox:ox+w] = crop
    
    res = pose_model.process(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks: return None, None
    
    kps, confs = [], []
    for lm in res.pose_landmarks.landmark:
        gx = (lm.x * size - ox) + x1
        gy = (lm.y * size - oy) + y1
        kps.append([gx, gy])
        confs.append(lm.visibility)
    return np.array(kps), np.array(confs)

def find_best_matches(ppl_lists, Ps):
    matched = []
    used = [set(), set(), set()]
    
    # 3-View Match
    for i1, p1 in enumerate(ppl_lists[0]):
        for i2, p2 in enumerate(ppl_lists[1]):
            for i3, p3 in enumerate(ppl_lists[2]):
                pts = [p1['norm'][24], p2['norm'][24], p3['norm'][24]]
                matched.append({'ids':(i1,i2,i3), 'persons':[p1,p2,p3], 'cams':[0,1,2]})
                used[0].add(i1); used[1].add(i2); used[2].add(i3)
                if len(matched) >= MAX_PEOPLE: break
        if len(matched) >= MAX_PEOPLE: break

    # 2-View Match
    pairs = [(0,1), (1,2), (0,2)]
    for c1, c2 in pairs:
        if len(matched) >= MAX_PEOPLE: break
        for i1, p1 in enumerate(ppl_lists[c1]):
            if i1 in used[c1]: continue
            for i2, p2 in enumerate(ppl_lists[c2]):
                if i2 in used[c2]: continue
                p_res = [None, None, None]
                p_res[c1] = p1
                p_res[c2] = p2
                matched.append({'ids':(i1, i2), 'persons': p_res, 'cams':[c1, c2]})
                used[c1].add(i1); used[c2].add(i2)
                if len(matched) >= MAX_PEOPLE: break
                
    return matched

def solve_3d_candidate(match, Ps):
    persons = match['persons']
    cams = match['cams']
    kps_3d = np.full((33, 3), np.nan)
    
    for j in range(33):
        pts = []
        active_Ps = []
        for c_idx in cams:
            p = persons[c_idx]
            if p and p['conf'][j] > 0.3:
                pts.append(p['norm'][j])
                active_Ps.append(Ps[c_idx])
        
        if len(pts) >= 2:
            X = triangulate_DLT(active_Ps, pts)
            if 0.1 < np.linalg.norm(X) < 50.0:
                kps_3d[j] = X
    return kps_3d

def reproject_3d_to_2d(joints_3d_dict, P):
    """ 3D座標(dict)を2D投影する """
    projected = {}
    for j_idx, pos in joints_3d_dict.items():
        if np.isnan(pos[0]): continue
        X = np.hstack([pos, 1.0])
        x_proj = P @ X
        if x_proj[2] > 0.1:
            u = int(x_proj[0] / x_proj[2])
            v = int(x_proj[1] / x_proj[2])
            projected[j_idx] = (u, v)
    return projected

def draw_skeleton_reprojected(img, proj_2d, color):
    for u, v in SKELETON_CONNECTIONS:
        if u in proj_2d and v in proj_2d:
            cv2.line(img, proj_2d[u], proj_2d[v], color, 2)
    for idx, pt in proj_2d.items():
        cv2.circle(img, pt, 3, color, -1)

# ==========================================
# Main
# ==========================================
def main():
    print("=== Hybrid 3D Tracking v9 (CSV + Skeleton Draw) ===")
    
    if not os.path.exists(CALIB_NPZ):
        print(f"Calibration file not found: {CALIB_NPZ}")
        return

    cam_params, _ = load_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    Ps = [p[2] for p in cam_params]
    Ks = [p[0] for p in cam_params]
    Ds = [p[1] for p in cam_params]

    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    yolo = YOLO(DET_MODEL)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=MP_COMPLEXITY, min_detection_confidence=MIN_MP_CONF)

    trackers = [PersonTracker3D(i) for i in range(MAX_PEOPLE)]

    # 動画書き出し
    vw = cv2.VideoWriter(TEMP_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    
    # ★ CSVファイルオープン
    f_csv = open(OUT_CSV_PATH, 'w', newline='')
    writer = csv.writer(f_csv)
    # ヘッダー書き込み
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z"])

    colors = [(0, 255, 0), (0, 0, 255)] # ID:0=緑, ID:1=赤

    for i in tqdm(range(total)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break

        # 1. 検出
        ppl_lists = []
        for c_idx, f in enumerate(frames):
            res = yolo.predict(f, conf=0.25, verbose=False, classes=[0])[0]
            cam_ppl = []
            if res.boxes:
                for box in res.boxes.xyxy.cpu().numpy():
                    kps, mp_conf = get_pose_padded(f, box, mp_pose)
                    if kps is not None:
                        norm = undistort_points(kps, Ks[c_idx], Ds[c_idx])
                        cam_ppl.append({'norm': norm, 'conf': mp_conf, 'box': box})
            ppl_lists.append(cam_ppl)

        # 2. マッチング
        matches = find_best_matches(ppl_lists, Ps)
        candidates = []
        for m in matches:
            kps_3d = solve_3d_candidate(m, Ps)
            valid_hips = [kps_3d[23], kps_3d[24]]
            valid_hips = [p for p in valid_hips if not np.isnan(p[0])]
            center = np.mean(valid_hips, axis=0) if valid_hips else np.zeros(3)
            candidates.append({'kps_3d': kps_3d, 'center': center, 'match': m})

        # 3. IDトラッキング
        assigned = {pid: None for pid in range(MAX_PEOPLE)}
        used = set()
        for pid in range(MAX_PEOPLE):
            tracker = trackers[pid]
            if tracker.center_3d is not None:
                best_idx = -1
                min_dist = 2.0
                for c_idx, cand in enumerate(candidates):
                    if c_idx in used: continue
                    dist = np.linalg.norm(cand['center'] - tracker.center_3d)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = c_idx
                if best_idx != -1:
                    assigned[pid] = candidates[best_idx]
                    used.add(best_idx)
                    tracker.update_center(candidates[best_idx]['kps_3d'])
        
        for c_idx, cand in enumerate(candidates):
            if c_idx not in used:
                for pid in range(MAX_PEOPLE):
                    if assigned[pid] is None and trackers[pid].center_3d is None:
                        assigned[pid] = cand
                        trackers[pid].update_center(cand['kps_3d'])
                        break

        # 4. フィルタリング & CSV出力 & 描画
        for pid in range(MAX_PEOPLE):
            cand = assigned[pid]
            tracker = trackers[pid]
            col = colors[pid % len(colors)]
            
            if cand is None: 
                # 今回検出なしの場合も、前回値があれば描画はできるが、今回はスキップ
                continue

            # --- 3Dフィルタ適用 (結果は tracker.current_filtered_joints に入る) ---
            filtered_3d_array = tracker.filter_and_update(cand['kps_3d'])
            
            # ★ CSV書き込み
            for j_idx in range(33):
                x, y, z = filtered_3d_array[j_idx]
                if not np.isnan(x):
                    writer.writerow([i, pid, MP_JOINTS[j_idx], x, y, z])

            # --- 描画 (再投影) ---
            persons = cand['match']['persons']
            for cam_i in range(3):
                if persons[cam_i]:
                    bx = persons[cam_i]['box'].astype(int)
                    cv2.rectangle(frames[cam_i], (bx[0], bx[1]), (bx[2], bx[3]), col, 2)
                    cv2.putText(frames[cam_i], f"ID:{pid}", (bx[0], bx[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                    
                    # フィルタ後の3D座標を再投影して描画
                    # (これにより、カメラ2の見た目が狂っていても正しい骨格が表示される)
                    proj_2d = reproject_3d_to_2d(tracker.current_filtered_joints, Ps[cam_i])
                    draw_skeleton_reprojected(frames[cam_i], proj_2d, col)

        vw.write(np.hstack(frames))

    f_csv.close()
    mp_pose.close()
    vw.release()
    for c in caps: c.release()
    
    # FFmpeg変換
    print("Converting video with FFmpeg...")
    INPUT  = Path(TEMP_VIDEO_PATH)
    OUTPUT = Path(FINAL_OUTPUT_PATH)
    
    if INPUT.exists():
        cmd = [
            "ffmpeg", "-y", "-i", str(INPUT),
            "-vf", "scale=2880:540,fps=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level", "4.0", "-an", str(OUTPUT),
        ]
        subprocess.run(cmd, check=True)
        os.remove(INPUT)
        print("Done:", OUTPUT)

if __name__ == "__main__":
    main()