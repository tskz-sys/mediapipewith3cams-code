#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Hybrid 3D Tracking v10 [PARAMETER FIX]
#   - 原因特定: キャリブレーションパラメータの「逆変換」が不要でした。
#   - 修正: npzのR, Tをそのまま使用してP行列を作成します。
#   - これにより、3D再構成の精度が劇的に向上し、CSVの「ぐちゃぐちゃ」が直ります。
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

# 出力先
OUT_CSV_PATH      = "./csv/test9.csv"
TEMP_VIDEO_PATH   = "./movie/temp9.mp4"
FINAL_OUTPUT_PATH = "./movie/test9.mp4"

DET_MODEL = "yolo11x.pt" 
MAX_PEOPLE = 2
MAX_MOVE_METER = 0.50 # フィルタは念のため残すが、パラメータが合えばほぼ不要になるはず

MP_COMPLEXITY = 1
MIN_MP_CONF   = 0.3

# CSVヘッダー用
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

# 描画用接続
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),           
    (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8) # Face
]

# ==========================================
# ★修正されたパラメータ読み込み関数
# ==========================================
def scale_camera_matrix(K, dist, target_w, target_h):
    orig_w, orig_h = K[0, 2] * 2, K[1, 2] * 2
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05: return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx; K_new[1, 1] *= sy
    K_new[0, 2] *= sx; K_new[1, 2] *= sy
    return K_new, dist

def load_params_simple(npz_path, v1, v2, v3):
    """ 
    ★重要: 逆変換(get_inverse_transform)を行わず、
    npzの値を素直に読み込むバージョン 
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

    # そのまま取得 (World -> Camera)
    R1 = d["R1"]; T1 = d["T1"] if "T1" in d else d["t1"]
    R3 = d["R3"]; T3 = d["T3"] if "T3" in d else d["t3"]
    
    # Cam2は原点
    R2 = np.eye(3); T2 = np.zeros((3, 1))
    
    # Tの形状確認 (3,)なら(3,1)に
    if T1.ndim == 1: T1 = T1.reshape(3, 1)
    if T3.ndim == 1: T3 = T3.reshape(3, 1)

    # 射影行列 P = K @ [R|t]
    P1 = K1 @ np.hstack([R1, T1])
    P2 = K2 @ np.hstack([R2, T2])
    P3 = K3 @ np.hstack([R3, T3])

    # 再投影用に個別のパラメータも返す
    params_sep = [(K1, D1, R1, T1), (K2, D2, R2, T2), (K3, D3, R3, T3)]
    
    return [P1, P2, P3], [K1, K2, K3], [D1, D2, D3], params_sep

# ==========================================
# 幾何計算 & トラッカー
# ==========================================
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

class PersonTracker3D:
    def __init__(self, pid):
        self.pid = pid
        self.center_3d = None
        self.last_valid_joints_3d = {}
        self.current_filtered_joints = {}

    def update_center(self, kps_3d):
        hips = []
        if not np.isnan(kps_3d[23][0]): hips.append(kps_3d[23])
        if not np.isnan(kps_3d[24][0]): hips.append(kps_3d[24])
        if len(hips) > 0: self.center_3d = np.mean(hips, axis=0)
        else:
            valid = kps_3d[~np.isnan(kps_3d).any(axis=1)]
            if len(valid) > 0: self.center_3d = np.mean(valid, axis=0)

    def filter_and_update(self, raw_kps_3d):
        self.current_filtered_joints = {}
        filtered_array = np.full((33, 3), np.nan)

        for j_idx in range(33):
            curr_pos = raw_kps_3d[j_idx]
            if np.isnan(curr_pos[0]): continue

            if j_idx not in self.last_valid_joints_3d:
                self.last_valid_joints_3d[j_idx] = curr_pos
                self.current_filtered_joints[j_idx] = curr_pos
                filtered_array[j_idx] = curr_pos
                continue

            prev_pos = self.last_valid_joints_3d[j_idx]
            dist = np.linalg.norm(curr_pos - prev_pos)

            if dist > MAX_MOVE_METER:
                # 誤検出 -> 前回値を維持
                filtered_array[j_idx] = prev_pos
                self.current_filtered_joints[j_idx] = prev_pos
            else:
                self.last_valid_joints_3d[j_idx] = curr_pos
                self.current_filtered_joints[j_idx] = curr_pos
                filtered_array[j_idx] = curr_pos
        return filtered_array

def get_pose_padded(full_img, box, pose_model):
    H_img, W_img = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    pad_w = int((x2 - x1) * 0.2); pad_h = int((y2 - y1) * 0.2)
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
    # 3-View
    for i1, p1 in enumerate(ppl_lists[0]):
        for i2, p2 in enumerate(ppl_lists[1]):
            for i3, p3 in enumerate(ppl_lists[2]):
                matched.append({'ids':(i1,i2,i3), 'persons':[p1,p2,p3], 'cams':[0,1,2]})
                used[0].add(i1); used[1].add(i2); used[2].add(i3)
                if len(matched) >= MAX_PEOPLE: break
        if len(matched) >= MAX_PEOPLE: break
    # 2-View
    pairs = [(0,1), (1,2), (0,2)]
    for c1, c2 in pairs:
        if len(matched) >= MAX_PEOPLE: break
        for i1, p1 in enumerate(ppl_lists[c1]):
            if i1 in used[c1]: continue
            for i2, p2 in enumerate(ppl_lists[c2]):
                if i2 in used[c2]: continue
                p_res = [None, None, None]; p_res[c1] = p1; p_res[c2] = p2
                matched.append({'ids':(i1, i2), 'persons': p_res, 'cams':[c1, c2]})
                used[c1].add(i1); used[c2].add(i2)
                if len(matched) >= MAX_PEOPLE: break
    return matched

def solve_3d_candidate(match, Ps):
    persons, cams = match['persons'], match['cams']
    kps_3d = np.full((33, 3), np.nan)
    for j in range(33):
        pts, active_Ps = [], []
        for c_idx in cams:
            p = persons[c_idx]
            if p and p['conf'][j] > 0.3:
                pts.append(p['norm'][j]); active_Ps.append(Ps[c_idx])
        if len(pts) >= 2:
            X = triangulate_DLT(active_Ps, pts)
            if 0.1 < np.linalg.norm(X) < 50.0: kps_3d[j] = X
    return kps_3d

# ★正しい再投影関数 (Distortion対応)
def reproject_distorted(joints_3d, K, D, R, t):
    proj = {}
    rvec, _ = cv2.Rodrigues(R)
    for j_idx, pos in joints_3d.items():
        if np.isnan(pos[0]): continue
        pts = np.array([pos], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(pts, rvec, t, K, D)
        u, v = img_pts[0][0]
        proj[j_idx] = (int(u), int(v))
    return proj

def draw_skeleton(img, proj, col):
    for u, v in SKELETON_CONNECTIONS:
        if u in proj and v in proj:
            cv2.line(img, proj[u], proj[v], col, 2)
    for idx, pt in proj.items():
        cv2.circle(img, pt, 3, col, -1)

# ==========================================
# Main
# ==========================================
def main():
    print("=== Hybrid 3D Tracking v10 [FIXED PARAMS] ===")
    if not os.path.exists(CALIB_NPZ): return

    # 1. パラメータ読み込み (Simple mode)
    Ps, Ks, Ds, params_sep = load_params_simple(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)

    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    yolo = YOLO(DET_MODEL)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=MP_COMPLEXITY, min_detection_confidence=MIN_MP_CONF)
    trackers = [PersonTracker3D(i) for i in range(MAX_PEOPLE)]

    vw = cv2.VideoWriter(TEMP_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    f_csv = open(OUT_CSV_PATH, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z"])
    
    colors = [(0, 255, 0), (0, 0, 255)]

    for i in tqdm(range(total)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break

        # Detection
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

        # Matching
        matches = find_best_matches(ppl_lists, Ps)
        candidates = []
        for m in matches:
            kps_3d = solve_3d_candidate(m, Ps)
            valid = kps_3d[~np.isnan(kps_3d).any(axis=1)]
            center = np.mean(valid, axis=0) if len(valid)>0 else np.zeros(3)
            candidates.append({'kps_3d': kps_3d, 'center': center, 'match': m})

        # ID Tracking
        assigned = {pid: None for pid in range(MAX_PEOPLE)}
        used = set()
        for pid in range(MAX_PEOPLE):
            tracker = trackers[pid]
            if tracker.center_3d is not None:
                best_idx = -1; min_dist = 2.0
                for c_idx, cand in enumerate(candidates):
                    if c_idx in used: continue
                    dist = np.linalg.norm(cand['center'] - tracker.center_3d)
                    if dist < min_dist: min_dist = dist; best_idx = c_idx
                if best_idx != -1: assigned[pid] = candidates[best_idx]; used.add(best_idx); tracker.update_center(candidates[best_idx]['kps_3d'])
        
        for c_idx, cand in enumerate(candidates):
            if c_idx not in used:
                for pid in range(MAX_PEOPLE):
                    if assigned[pid] is None and trackers[pid].center_3d is None:
                        assigned[pid] = cand; trackers[pid].update_center(cand['kps_3d']); break

        # Output & Draw
        for pid in range(MAX_PEOPLE):
            cand = assigned[pid]
            tracker = trackers[pid]
            col = colors[pid % len(colors)]
            if cand is None: continue

            filtered_3d = tracker.filter_and_update(cand['kps_3d'])
            
            # CSV
            for j in range(33):
                x, y, z = filtered_3d[j]
                if not np.isnan(x): writer.writerow([i, pid, MP_JOINTS[j], x, y, z])

            # Draw (Reprojection)
            persons = cand['match']['persons']
            for cam_i in range(3):
                # カメラパラメータ
                K, D, R, t = params_sep[cam_i]
                
                # BBoxを描画 (そのカメラで検出されていれば)
                if persons[cam_i]:
                    bx = persons[cam_i]['box'].astype(int)
                    cv2.rectangle(frames[cam_i], (bx[0], bx[1]), (bx[2], bx[3]), col, 2)
                    cv2.putText(frames[cam_i], f"ID:{pid}", (bx[0], bx[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                
                # 骨格を再投影 (全カメラに描画)
                proj = reproject_distorted(tracker.current_filtered_joints, K, D, R, t)
                draw_skeleton(frames[cam_i], proj, col)

        vw.write(np.hstack(frames))

    f_csv.close(); mp_pose.close(); vw.release(); [c.release() for c in caps]
    
    print("Converting to MP4...")
    if os.path.exists(TEMP_VIDEO_PATH):
        cmd = ["ffmpeg", "-y", "-i", TEMP_VIDEO_PATH, "-vf", "scale=2880:540,fps=30",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", FINAL_OUTPUT_PATH]
        subprocess.run(cmd, check=True)
        os.remove(TEMP_VIDEO_PATH)
        print("Done:", FINAL_OUTPUT_PATH)

if __name__ == "__main__":
    main()