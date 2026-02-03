#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp
from tqdm import tqdm

# ==========================================
# ★ 設定エリア
# ==========================================

# 入力動画ディレクトリ (ffmpeg/output)
MOVIE_DIR = "../ffmpeg/output"

# キャリブレーションファイル
CALIB_NPZ = "./11253cams_fixedd.npz"

# 出力先
OUTPUT_DIR = "./output/3d_pose_result"

# 処理対象
# 例: Match1 の Clip 1〜6 を処理する
TARGET_MATCHES = [1]
TARGET_CLIPS   = [1, 2, 3, 4, 5, 6]

# ★重要: Pattern BR (逆変換) のスイッチ★
# False: そのまま読み込む (骨格が飛ぶ場合、まずはこれを試す)
# True:  逆変換をかける (FalseでダメならTrueにする)
ENABLE_PATTERN_BR = False 

# モデル設定
YOLO_MODEL = "yolo11x.pt"      # 検出用
MP_COMPLEXITY = 2              # 2=最高精度

# 検出パラメータ
CONF_YOLO_PERSON = 0.25        # YOLOの人検出閾値
MAX_PEOPLE = 2                 # 1画面あたりの最大人数
RESIZE_SCALE = 0.5             # 確認動画の縮小率

# MediaPipeの関節名定義 (33点)
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
# ファイルパス生成 (命名規則対応)
# ==========================================
def get_video_path(base_dir, cam_id, match_id, clip_id):
    """
    命名規則: [videonumber]match[wholematchnumber]_[matchnumber].mp4
    
    videonumber (cam_id):
      1 -> Left Camera
      2 -> Center Camera
      3 -> Right Camera
    """
    filename = f"{cam_id}match{match_id}_{clip_id}.mp4"
    return os.path.join(base_dir, filename)

# ==========================================
# 数学・キャリブレーション関数
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
    """ 
    設定(ENABLE_PATTERN_BR)に従ってパラメータを読み込む 
    引数は 左(v1), 中(v2), 右(v3) の順で渡されることを想定
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
        
    d = np.load(npz_path, allow_pickle=True)
    def get_k(k_new, k_old): return d[k_new] if k_new in d else d[k_old]

    # NPZ内のキー定義 (作成時のルール)
    # K1: Left, K2: Center, K3: Right と仮定
    K1, D1 = get_k("K1","cam_matrix1"), get_k("dist1","dist_coeffs1")
    K2, D2 = get_k("K2","cam_matrix2"), get_k("dist2","dist_coeffs2")
    K3, D3 = get_k("K3","cam_matrix3"), get_k("dist3","dist_coeffs3")

    def get_wh(v):
        c = cv2.VideoCapture(v)
        if not c.isOpened(): return 1920, 1080 # default
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        return w, h
    
    # 動画サイズに合わせてK行列をスケーリング
    w1, h1 = get_wh(v1_path)
    w2, h2 = get_wh(v2_path)
    w3, h3 = get_wh(v3_path)
    
    K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    # パラメータ取得
    R1_raw = d["R1"]; t1_raw = d["T1"] if "T1" in d else d["t1"]
    R3_raw = d["R3"]; t3_raw = d["T3"] if "T3" in d else d["t3"]

    # === スイッチによる切り替え ===
    if ENABLE_PATTERN_BR:
        print(">> Pattern BR: ON (Applying Inverse Transform)")
        R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3,1))
        R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3,1))
    else:
        print(">> Pattern BR: OFF (Using Raw Parameters)")
        R1 = R1_raw
        t1 = t1_raw.reshape(3,1)
        R3 = R3_raw
        t3 = t3_raw.reshape(3,1)
    # ============================

    R2 = np.eye(3); t2 = np.zeros((3, 1))

    # Projection Matrices [R|t]
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

# ==========================================
# ハイブリッド抽出 & 3D計算クラス
# ==========================================
class Pose3DEstimator:
    def __init__(self):
        print(f"Loading YOLO: {YOLO_MODEL}...")
        self.yolo = YOLO(YOLO_MODEL)
        
        print(f"Loading MediaPipe (Complexity={MP_COMPLEXITY})...")
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True, 
            model_complexity=MP_COMPLEXITY,
            min_detection_confidence=0.5
        )

    def get_2d_pose(self, img, K, D):
        H, W = img.shape[:2]
        results = self.yolo.predict(img, conf=CONF_YOLO_PERSON, verbose=False, classes=[0])
        
        candidates = []
        if results[0].boxes is None: return candidates

        boxes = results[0].boxes.data.cpu().numpy()
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

        for i, box_data in enumerate(boxes):
            if i >= MAX_PEOPLE: break
            x1, y1, x2, y2 = map(int, box_data[:4])

            # Crop with padding
            pad_w = int((x2 - x1) * 0.1)
            pad_h = int((y2 - y1) * 0.1)
            cx1 = max(0, x1 - pad_w); cy1 = max(0, y1 - pad_h)
            cx2 = min(W, x2 + pad_w); cy2 = min(H, y2 + pad_h)
            
            if cx2 <= cx1 or cy2 <= cy1: continue
            
            crop = img[cy1:cy2, cx1:cx2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            mp_res = self.mp_pose.process(crop_rgb)
            
            kps_raw = []
            kps_confs = []
            
            if mp_res.pose_landmarks:
                crop_h, crop_w = crop.shape[:2]
                for lm in mp_res.pose_landmarks.landmark:
                    gx = cx1 + lm.x * crop_w
                    gy = cy1 + lm.y * crop_h
                    kps_raw.append([gx, gy])
                    kps_confs.append(lm.visibility)
                
                kps_norm = undistort_points(kps_raw, K, D)
                
                candidates.append({
                    "kps_raw": np.array(kps_raw),
                    "kps_norm": kps_norm,
                    "conf": np.array(kps_confs),
                    "box": (x1, y1, x2, y2)
                })
        
        return candidates

    def triangulate_people(self, ppl1, ppl2, ppl3, P1, P2, P3):
        combos = []
        for i1 in range(len(ppl1)):
            for i2 in range(len(ppl2)):
                for i3 in range(len(ppl3)):
                    p1, p2, p3 = ppl1[i1], ppl2[i2], ppl3[i3]
                    cost_sum = 0
                    count = 0
                    for j in USE_JOINTS_IDX:
                        if p1["conf"][j]>0.5 and p2["conf"][j]>0.5 and p3["conf"][j]>0.5:
                            pts = [p1["kps_norm"][j], p2["kps_norm"][j], p3["kps_norm"][j]]
                            X = triangulate_DLT([P1, P2, P3], pts)
                            
                            # 異常値(50m以上)は弾く
                            if np.linalg.norm(X) < 50.0:
                                count += 1
                            else:
                                cost_sum += 100.0
                    
                    if count >= 3:
                        combos.append({'ids': (i1, i2, i3), 'score': count, 'dist_sum': cost_sum})
        
        combos.sort(key=lambda x: (-x['score'], x['dist_sum']))
        
        used1, used2, used3 = set(), set(), set()
        final_3d_people = []
        
        for c in combos:
            i1, i2, i3 = c['ids']
            if i1 in used1 or i2 in used2 or i3 in used3: continue
            
            p1, p2, p3 = ppl1[i1], ppl2[i2], ppl3[i3]
            kps_3d = np.full((33, 3), np.nan)
            
            for j in range(33):
                Ps = []; pts = []
                if p1["conf"][j] > 0.5: pts.append(p1["kps_norm"][j]); Ps.append(P1)
                if p2["conf"][j] > 0.5: pts.append(p2["kps_norm"][j]); Ps.append(P2)
                if p3["conf"][j] > 0.5: pts.append(p3["kps_norm"][j]); Ps.append(P3)
                
                if len(Ps) >= 2:
                    X = triangulate_DLT(Ps, pts)
                    if np.linalg.norm(X) < 50.0:
                        kps_3d[j] = X
            
            final_3d_people.append(kps_3d)
            used1.add(i1); used2.add(i2); used3.add(i3)
            
        return final_3d_people

# ==========================================
# 描画関数 (クリッピング付き)
# ==========================================
def draw_3d_projection(frame, kps_3d, K, D, R, t, color=(0, 255, 0)):
    """ 3D座標を2D画像に再投影して描画 (オーバーフロー防止処理付き) """
    if kps_3d is None: return frame
    
    mp_conn = mp.solutions.pose.POSE_CONNECTIONS
    rvec, _ = cv2.Rodrigues(R)
    
    valid_mask = ~np.isnan(kps_3d).any(axis=1)
    if not np.any(valid_mask): return frame

    # 3D -> 2D Projection
    img_pts, _ = cv2.projectPoints(kps_3d[valid_mask], rvec, t, K, D)
    img_pts = img_pts.reshape(-1, 2)
    
    kps2d = {}
    cnt = 0
    H, W = frame.shape[:2]

    # ★安全な描画範囲 (これがないとcv2.lineがクラッシュする)
    SAFE_MIN, SAFE_MAX = -5000, 5000 

    for i in range(33):
        if valid_mask[i]:
            x_raw, y_raw = img_pts[cnt]
            
            # NaN/Inf チェック & クリッピング
            if np.isfinite(x_raw) and np.isfinite(y_raw):
                x = int(np.clip(x_raw, SAFE_MIN, SAFE_MAX))
                y = int(np.clip(y_raw, SAFE_MIN, SAFE_MAX))
                kps2d[i] = (x, y)
            cnt += 1
            
    # 線を描画
    for u, v in mp_conn:
        if u in kps2d and v in kps2d:
            pt1 = kps2d[u]
            pt2 = kps2d[v]
            # 画面内に少なくとも片方が近い場合のみ描画
            if (0 <= pt1[0] < W or 0 <= pt2[0] < W) and (0 <= pt1[1] < H or 0 <= pt2[1] < H):
                try:
                    cv2.line(frame, pt1, pt2, color, 2)
                except Exception:
                    pass 
    
    return frame

# ==========================================
# メイン処理
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"=== 3D Pose Extraction Batch ===")
    print(f"Pattern BR: {'ENABLED' if ENABLE_PATTERN_BR else 'DISABLED'}")
    
    estimator = Pose3DEstimator()
    writer = None
    all_3d_data = []

    # --- Batch Loop ---
    for match_id in TARGET_MATCHES:
        for clip_id in TARGET_CLIPS:
            print(f"\nProcessing Match {match_id} - Clip {clip_id}...")
            
            # 3台のカメラ (1=Left, 2=Center, 3=Right)
            v_paths = [get_video_path(MOVIE_DIR, i, match_id, clip_id) for i in [1, 2, 3]]
            
            if not all(os.path.exists(p) for p in v_paths):
                print(f"  [Skip] Missing files for Match{match_id}-{clip_id}")
                continue
            
            try:
                # パラメータ読み込み (Left, Center, Right の順で渡す)
                cam_params, extrinsics = load_params_switchable(CALIB_NPZ, v_paths[0], v_paths[1], v_paths[2])
            except Exception as e:
                print(f"  [Error] Calibration load failed: {e}")
                continue

            P1, P2, P3 = [cp[2] for cp in cam_params]

            caps = [cv2.VideoCapture(p) for p in v_paths]
            fps = caps[0].get(cv2.CAP_PROP_FPS)
            total_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
            W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if writer is None:
                out_path = os.path.join(OUTPUT_DIR, "3d_verification_summary.mp4")
                out_w = int(W * 3 * RESIZE_SCALE)
                out_h = int(H * RESIZE_SCALE)
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
                print(f"Output Video: {out_path}")

            for f_idx in tqdm(range(total_frames), leave=False):
                frames = []
                for c in caps: 
                    ret, f = c.read()
                    frames.append(f if ret else np.zeros((H, W, 3), dtype=np.uint8))
                
                people_2d_lists = []
                for i, frame in enumerate(frames):
                    K, D, _ = cam_params[i]
                    ppl = estimator.get_2d_pose(frame, K, D)
                    people_2d_lists.append(ppl)
                
                people_3d = estimator.triangulate_people(
                    people_2d_lists[0], people_2d_lists[1], people_2d_lists[2],
                    P1, P2, P3
                )
                
                for pid, kps3d in enumerate(people_3d):
                    for j, (x, y, z) in enumerate(kps3d):
                        if not np.isnan(x):
                            all_3d_data.append({
                                "MatchID": match_id, "ClipID": clip_id, "Frame": f_idx,
                                "PersonID": pid, "Joint": MP_JOINTS[j],
                                "X": x, "Y": y, "Z": z
                            })
                    
                    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]
                    col = colors[pid % len(colors)]
                    
                    for cam_i in range(3):
                        K, D, _ = cam_params[cam_i]
                        R, t = extrinsics[cam_i]
                        draw_3d_projection(frames[cam_i], kps3d, K, D, R, t, color=col)

                resized_frames = [cv2.resize(f, (int(W*RESIZE_SCALE), int(H*RESIZE_SCALE))) for f in frames]
                combined = np.hstack(resized_frames)
                writer.write(combined)

            for c in caps: c.release()

    if writer: writer.release()
    print("\nBatch Processing Complete!")
    
    if all_3d_data:
        print("Saving CSV...")
        df = pd.DataFrame(all_3d_data)
        csv_path = os.path.join(OUTPUT_DIR, "batch_3d_pose_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    else:
        print("No 3D data extracted.")

if __name__ == "__main__":
    main()