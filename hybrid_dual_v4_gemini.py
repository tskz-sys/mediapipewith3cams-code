#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Hybrid v4: YOLO+MediaPipe Batch Processing [ACCURACY FOCUSED]
#   - Detection: YOLOv11x (Box) -> MediaPipe (33 Keypoints)
#   - Tracking: 3D Centroid Matching (Fixes ID switching)
#   - Input/Output: Batch processing (Match/Clip loops)
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
# ★ 設定エリア
# ==========================================

# 入力動画ディレクトリ (ffmpeg/output)
MOVIE_DIR = "../ffmpeg/output"

# キャリブレーションファイル
CALIB_NPZ = "./npz/11253cams_fixedd.npz"

# 出力先ディレクトリ
OUTPUT_DIR = "./output/3d_pose_result"

# 処理対象
TARGET_MATCHES = [1]
TARGET_CLIPS   = [1, 2, 3, 4, 5, 6]

# ★Pattern BR (逆変換) スイッチ★
ENABLE_PATTERN_BR = False 

# --- モデル設定 ---
DET_MODEL_NAME = "yolo11x.pt" 

# --- 検出パラメータ ---
INFERENCE_SIZE = 1920  
CONF_LOW_LIMIT = 0.10  
CONF_PERSON    = 0.25  
CONF_BALL      = 0.15  
MAX_PEOPLE     = 2

# MediaPipe設定 (精度優先)
MP_COMPLEXITY = 2 
MIN_MP_CONF   = 0.5

# MediaPipe Joint Names (33 points)
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
# マッチング計算に使う主要関節
USE_JOINTS_IDX = [11, 12, 23, 24, 25, 26] 

# ==========================================
# ファイルパス生成
# ==========================================
def get_video_path(base_dir, cam_id, match_id, clip_id):
    filename = f"{cam_id}match{match_id}_{clip_id}.mp4"
    return os.path.join(base_dir, filename)

# ==========================================
# キャリブレーション補正
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

def load_params_switchable(npz_path, v1_path, v2_path, v3_path):
    d = np.load(npz_path, allow_pickle=True)
    def get_k(k_new, k_old): return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1","cam_matrix1"), get_k("dist1","dist_coeffs1")
    K2, D2 = get_k("K2","cam_matrix2"), get_k("dist2","dist_coeffs2")
    K3, D3 = get_k("K3","cam_matrix3"), get_k("dist3","dist_coeffs3")

    def get_wh(v):
        c = cv2.VideoCapture(v)
        if not c.isOpened(): return 1920, 1080 
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        return w, h
    
    w1, h1 = get_wh(v1_path)
    w2, h2 = get_wh(v2_path)
    w3, h3 = get_wh(v3_path)
    
    K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    R1_raw = d["R1"]; t1_raw = d["T1"] if "T1" in d else d["t1"]
    R3_raw = d["R3"]; t3_raw = d["T3"] if "T3" in d else d["t3"]

    if ENABLE_PATTERN_BR:
        R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3,1))
        R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3,1))
    else:
        R1 = R1_raw
        t1 = t1_raw.reshape(3,1)
        R3 = R3_raw
        t3 = t3_raw.reshape(3,1)

    R2 = np.eye(3); t2 = np.zeros((3, 1))

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
        xh = P @ np.append(X, 1.0)
        if abs(xh[2]) < 1e-9: 
            errs.append(1000.0)
            continue
        xp = xh[0] / xh[2]
        yp = xh[1] / xh[2]
        errs.append(np.sqrt((x - xp)**2 + (y - yp)**2))
    
    return X, np.mean(errs)

# ==========================================
# 3Dトラッキング (ID固定化)
# ==========================================
def track_3d_ids(prev_centroids, current_people_data, dist_thresh=200.0):
    """
    前のフレームの3D重心位置と比較してIDを引き継ぐ
    current_people_data: [(kps3d), ...] ※マッチング済みのリスト
    """
    current_centroids = []
    for idx, kps3d in enumerate(current_people_data):
        valid = ~np.isnan(kps3d).any(axis=1)
        if np.sum(valid) > 5:
            cent = np.mean(kps3d[valid], axis=0)
            current_centroids.append({'idx': idx, 'pos': cent, 'matched': False})
        else:
            current_centroids.append({'idx': idx, 'pos': None, 'matched': False})

    new_results = {} 
    updated_centroids = {}

    # 既存IDとのマッチング
    for pid, prev_pos in prev_centroids.items():
        if prev_pos is None: continue
        
        best_dist = dist_thresh
        best_match_idx = -1
        
        for i, curr in enumerate(current_centroids):
            if curr['pos'] is not None and not curr['matched']:
                dist = np.linalg.norm(curr['pos'] - prev_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_match_idx = i
        
        if best_match_idx != -1:
            current_centroids[best_match_idx]['matched'] = True
            orig_idx = current_centroids[best_match_idx]['idx']
            
            kps3d = current_people_data[orig_idx]
            new_results[pid] = kps3d
            updated_centroids[pid] = current_centroids[best_match_idx]['pos']

    # 新規ID
    next_id = 0
    if prev_centroids: next_id = max(prev_centroids.keys()) + 1
    
    for i, curr in enumerate(current_centroids):
        if curr['pos'] is not None and not curr['matched']:
            orig_idx = curr['idx']
            kps3d = current_people_data[orig_idx]
            
            new_results[next_id] = kps3d
            updated_centroids[next_id] = curr['pos']
            next_id += 1
            
    return new_results, updated_centroids

# ==========================================
# 検出ロジック (YOLO Box -> MediaPipe Pose)
# ==========================================

def get_pose_from_crop(full_img, box, pose_model):
    H, W = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # マージン確保
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

def process_frame_hybrid(img, det_model, mp_model, K, D, max_p):
    det_res = det_model.predict(img, conf=CONF_LOW_LIMIT, imgsz=INFERENCE_SIZE, verbose=False, classes=[0, 32])[0]
    
    people = []
    balls = []

    if det_res.boxes is not None:
        indices = np.argsort(-det_res.boxes.conf.cpu().numpy())
        
        p_count = 0
        for idx in indices:
            cls_id = int(det_res.boxes.cls[idx])
            conf = det_res.boxes.conf[idx].cpu().numpy()
            box = det_res.boxes.xyxy[idx].cpu().numpy()

            if cls_id == 0: # Person
                if conf > CONF_PERSON and p_count < max_p:
                    # ★MediaPipeを使用★
                    mp_kps, mp_confs, padded_box = get_pose_from_crop(img, box, mp_model)
                    
                    if mp_kps is not None:
                        norm_mp_kps = undistort_points(mp_kps, K, D)
                        people.append({
                            "type": "person",
                            "box": box,
                            "kps_raw": mp_kps,       
                            "kps_norm": norm_mp_kps, 
                            "conf": mp_confs
                        })
                        p_count += 1
            
            elif cls_id == 32: # Ball
                if conf > CONF_BALL:
                    if len(balls) == 0:
                        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                        norm_pt = undistort_points([(cx, cy)], K, D)[0]
                        balls.append({
                            "center_raw": np.array([cx, cy]),
                            "center_norm": norm_pt,
                            "box": box,
                            "type": "ball",
                            "conf": conf
                        })

    return people, balls

# ==========================================
# Main
# ==========================================

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"=== Hybrid v4 Batch (YOLO+MediaPipe + Tracking) ===")
    print(f"Pattern BR: {'ENABLED' if ENABLE_PATTERN_BR else 'DISABLED'}")

    # Models
    print(f"Loading Detection Model: {DET_MODEL_NAME} ...")
    det_model = YOLO(DET_MODEL_NAME) 
    
    print("Loading MediaPipe (Complexity=2)...")
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=MP_COMPLEXITY,
        min_detection_confidence=MIN_MP_CONF
    )
    
    mp_conn = mp.solutions.pose.POSE_CONNECTIONS
    colors = [(0,255,0), (0,0,255), (255,255,0)]
    LIMIT = 5000 

    # --- Batch Loop ---
    for match_id in TARGET_MATCHES:
        for clip_id in TARGET_CLIPS:
            print(f"\nProcessing Match {match_id} - Clip {clip_id}...")
            
            v_paths = [get_video_path(MOVIE_DIR, i, match_id, clip_id) for i in [1, 2, 3]]
            
            if not all(os.path.exists(p) for p in v_paths):
                print(f"  [Skip] Missing files for Match{match_id}-{clip_id}")
                continue
            
            try:
                cam_params_full, extrinsics = load_params_switchable(CALIB_NPZ, v_paths[0], v_paths[1], v_paths[2])
            except Exception as e:
                print(f"  [Error] Calibration load failed: {e}")
                continue

            P1, P2, P3 = [p[2] for p in cam_params_full]

            caps = [cv2.VideoCapture(p) for p in v_paths]
            fps = caps[0].get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            
            total_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
            W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Output Files
            base_name = f"hybrid_v4_match{match_id}_clip{clip_id}"
            out_csv_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
            temp_vid_path = os.path.join(OUTPUT_DIR, f"temp_{base_name}.mp4")
            out_vid_path = os.path.join(OUTPUT_DIR, f"{base_name}.mp4")

            f_csv = open(out_csv_path, 'w', newline='')
            writer = csv.writer(f_csv)
            writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z"])

            vw = cv2.VideoWriter(temp_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))

            # Tracking変数 (クリップごとにリセット)
            prev_centroids = {}

            # --- Frame Loop ---
            for i in tqdm(range(total_frames), leave=False):
                frames = []
                for c in caps: _, f = c.read(); frames.append(f if f is not None else np.zeros((H,W,3), np.uint8))

                ppl_lists = []
                ball_lists = []
                
                # 1. Detection & Pose
                for cam_i, f in enumerate(frames):
                    K, D, _ = cam_params_full[cam_i]
                    p, b = process_frame_hybrid(f, det_model, pose_estimator, K, D, MAX_PEOPLE)
                    ppl_lists.append(p)
                    ball_lists.append(b)

                # 2. Matching (Brute Force 3D Error)
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
                
                # 3D化してリスト化 (トラッキング準備)
                current_people_3d = [] # [kps3d, ...]
                valid_candidates = []  # 対応する候補情報

                for cand in candidates:
                    i1, i2, i3 = cand['ids']
                    if i1 in used1 or i2 in used2 or i3 in used3: continue
                    
                    persons = [ppl_lists[0][i1], ppl_lists[1][i2], ppl_lists[2][i3]]
                    kps_3d = np.full((33, 3), np.nan)
                    
                    for j in range(33):
                        if all(p["conf"][j]>0.5 for p in persons):
                            pts = [p["kps_norm"][j] for p in persons]
                            X, _ = triangulate_DLT([P1, P2, P3], pts)
                            if np.linalg.norm(X) < 50.0: kps_3d[j] = X
                    
                    current_people_3d.append(kps_3d)
                    valid_candidates.append(cand)
                    used1.add(i1); used2.add(i2); used3.add(i3)

                # 3. Tracking (ID Assignment)
                tracked_results, new_centroids = track_3d_ids(prev_centroids, current_people_3d)
                prev_centroids = new_centroids

                # 4. Save & Draw
                for pid, kps3d in tracked_results.items():
                    # CSV Save
                    for j in range(33):
                        if not np.isnan(kps3d[j][0]):
                            writer.writerow([i, pid, MP_JOINTS[j], kps3d[j][0], kps3d[j][1], kps3d[j][2]])

                    # Visualization
                    # kps3dに対応する元の2D情報を探して描画
                    # (簡易的に、current_people_3dの中から一致するものを探す)
                    matched_idx = -1
                    for idx, c_kps in enumerate(current_people_3d):
                        if np.array_equal(c_kps, kps3d):
                            matched_idx = idx
                            break
                    
                    if matched_idx != -1:
                        cand = valid_candidates[matched_idx]
                        i1, i2, i3 = cand['ids']
                        persons_2d = [ppl_lists[0][i1], ppl_lists[1][i2], ppl_lists[2][i3]]
                        col = colors[pid % len(colors)]

                        for cam_i in range(3):
                            frame = frames[cam_i]
                            kps_raw = persons_2d[cam_i]["kps_raw"]
                            box = persons_2d[cam_i]["box"]
                            
                            # ID
                            cv2.putText(frame, f"ID:{pid}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

                            # Skeleton
                            for u, v in mp_conn:
                                if u < len(kps_raw) and v < len(kps_raw):
                                    pt1 = (int(np.clip(kps_raw[u][0], -LIMIT, LIMIT)), int(np.clip(kps_raw[u][1], -LIMIT, LIMIT)))
                                    pt2 = (int(np.clip(kps_raw[v][0], -LIMIT, LIMIT)), int(np.clip(kps_raw[v][1], -LIMIT, LIMIT)))
                                    cv2.line(frame, pt1, pt2, col, 2)
                            
                            # Fingers
                            for tip in [17, 18, 19, 20, 21, 22]:
                                pt = (int(np.clip(kps_raw[tip][0], -LIMIT, LIMIT)), int(np.clip(kps_raw[tip][1], -LIMIT, LIMIT)))
                                cv2.circle(frame, pt, 3, (0, 255, 255), -1)

                # 5. Ball
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
            vw.release()
            for c in caps: c.release()

            # Convert
            if os.path.exists(temp_vid_path):
                subprocess.run(["ffmpeg", "-y", "-i", temp_vid_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", out_vid_path], check=False)
                os.remove(temp_vid_path)
                print(f"Saved: {out_vid_path}")

    pose_estimator.close()
    print("Done.")

if __name__ == "__main__":
    main()