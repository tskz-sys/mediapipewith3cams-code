#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# YOLO + MediaPipe Hybrid [v4 DEBUG & ROBUST]
#   - 改善: 2視点だけでも3D化するように条件緩和（検出率向上）
#   - 改善: ボールの2視点3D化対応
#   - 機能: デバッグ用BBox描画（青：人、赤：ボール）
# ==========================================================

import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from ultralytics import YOLO
from tqdm import tqdm
import itertools

# ==========================================
# ★ ユーザー設定
# ==========================================
VIDEO_LEFT   = "../ffmpeg/output/1match1_2.mp4"
VIDEO_CENTER = "../ffmpeg/output/2match1_2.mp4"
VIDEO_RIGHT  = "../ffmpeg/output/3match1_2.mp4"

CALIB_NPZ = "./npz/11253cams_fixedd.npz"

OUT_CSV   = "./csv/test3.csv"
OUT_VIDEO = "./movie/testtt3.mp4"

# --- モデル設定 ---
DET_MODEL = "yolo11x-pose.pt" # Person用
BALL_MODEL = "yolo11x.pt"     # Ball用

# --- 検出パラメータ ---
INFERENCE_SIZE = 1920  # 高解像度維持
CONF_LOW_LIMIT = 0.10  # 足切りライン

CONF_PERSON    = 0.25  # 人採用閾値
CONF_BALL      = 0.15  # ボール採用閾値

MAX_PEOPLE     = 2     # 最大人数

# MediaPipe設定
MP_COMPLEXITY = 1
MIN_MP_CONF   = 0.3    # 少し下げる（YOLOで切り抜いているため）

# Debug描画（TrueにするとYOLOの生BBoxを描画）
DRAW_DEBUG_BBOX = True

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
# マッチングに使う関節（腰、肩、膝など体の中心に近い信頼できる点）
USE_JOINTS_IDX = [11, 12, 23, 24, 25, 26] 

# ==========================================
# キャリブレーション関数群
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
    # DLT法による三角測量
    # Ps: 投影行列のリスト (2つ以上)
    # pts: undistortされた2D座標のリスト (2つ以上)
    A = []
    for P, (x, y) in zip(Ps, pts):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    X = X[:3] / X[3]
    
    # 再投影誤差の計算
    errs = []
    for P, (x, y) in zip(Ps, pts):
        xh = P @ np.hstack([X, 1.0])
        if abs(xh[2]) < 1e-9: 
            errs.append(1000.0)
            continue
        xp, yp = xh[0]/xh[2], xh[1]/xh[2]
        errs.append((xp-x)**2 + (yp-y)**2)
    return X, np.mean(errs)

# ==========================================
# 検出ロジック
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
    res = pose_model.predict(img, conf=CONF_LOW_LIMIT, imgsz=INFERENCE_SIZE, verbose=False, classes=[0])[0]
    
    people = []
    if res.boxes is None or len(res.boxes) == 0: return []

    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    
    p_count = 0
    for idx in indices:
        conf = res.boxes.conf[idx].cpu().numpy()
        
        # デバッグ用に低信頼度でも残すが、解析には使わないフラグを立てる手もある
        # ここでは閾値チェック
        if conf < CONF_PERSON: continue
        if p_count >= max_p + 1: break # 少し余裕を持って取る

        box = res.boxes.xyxy[idx].cpu().numpy()
        yolo_kps = res.keypoints.xy[idx].cpu().numpy()

        mp_kps, mp_confs, padded_box = get_pose_from_crop(img, box, mp_model)
        
        if mp_kps is not None:
            norm_mp_kps = undistort_points(mp_kps, K, D)
            people.append({
                "type": "person",
                "box": box,
                "yolo_kps": yolo_kps,
                "kps_raw": mp_kps,
                "kps_norm": norm_mp_kps,
                "conf": mp_confs,
                "score": conf
            })
            p_count += 1
    return people

def detect_ball_yolo(img, det_model, K, D):
    res = det_model.predict(img, conf=CONF_LOW_LIMIT, imgsz=INFERENCE_SIZE, verbose=False, classes=[32])[0]
    balls = []
    if res.boxes is None or len(res.boxes) == 0: return []

    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    for idx in indices:
        conf = res.boxes.conf[idx].cpu().numpy()
        if conf > CONF_BALL:
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
    return balls

# ==========================================
# マッチング & 3D化ロジック (改良版)
# ==========================================
def find_best_matches(ppl_lists, Ps):
    """
    3カメラ分の検出リストを受け取り、
    1. 3視点マッチング
    2. 残り物で2視点マッチング (1-2, 2-3, 1-3)
    を行い、統合された人物リストを返す
    """
    # インデックスの組み合わせ候補を作成
    # ppl_lists = [list_cam1, list_cam2, list_cam3]
    
    # マッチングコストを計算する関数
    def calc_cost(p_list, p_indices, cam_indices):
        # p_list: 選択されたPersonデータのリスト
        # p_indices: 各カメラでのインデックス
        # cam_indices: カメラIDのリスト (0,1,2 のうちどれか)
        
        total_err = 0
        valid_joints = 0
        
        # 共通して信頼度が高い関節でReprojection Errorを見る
        for j in USE_JOINTS_IDX:
            pts = []
            proj_matrices = []
            
            is_good = True
            for i, p_idx in enumerate(p_indices):
                if p_list[i]["conf"][j] < 0.5:
                    is_good = False
                    break
                pts.append(p_list[i]["kps_norm"][j])
                proj_matrices.append(Ps[cam_indices[i]])
            
            if is_good:
                _, err = triangulate_DLT(proj_matrices, pts)
                total_err += err
                valid_joints += 1
        
        if valid_joints == 0: return 1000.0 # マッチしない
        return total_err / valid_joints

    matched_people = []
    used_indices = [set() for _ in range(3)]

    # --- 1. 3視点マッチング ---
    candidates3 = []
    for i1, p1 in enumerate(ppl_lists[0]):
        for i2, p2 in enumerate(ppl_lists[1]):
            for i3, p3 in enumerate(ppl_lists[2]):
                cost = calc_cost([p1, p2, p3], [i1, i2, i3], [0, 1, 2])
                if cost < 0.3: # 閾値
                    candidates3.append({'ids': (i1, i2, i3), 'cost': cost, 'cams': (0,1,2)})
    
    candidates3.sort(key=lambda x: x['cost'])
    
    for cand in candidates3:
        i1, i2, i3 = cand['ids']
        if i1 in used_indices[0] or i2 in used_indices[1] or i3 in used_indices[2]: continue
        
        matched_people.append({
            'persons': [ppl_lists[0][i1], ppl_lists[1][i2], ppl_lists[2][i3]],
            'cams': [0, 1, 2]
        })
        used_indices[0].add(i1)
        used_indices[1].add(i2)
        used_indices[2].add(i3)

    # --- 2. 2視点マッチング (Cam1-2, Cam2-3, Cam1-3) ---
    cam_pairs = [(0, 1), (1, 2), (0, 2)]
    
    for c1, c2 in cam_pairs:
        candidates2 = []
        for i1, p1 in enumerate(ppl_lists[c1]):
            if i1 in used_indices[c1]: continue
            for i2, p2 in enumerate(ppl_lists[c2]):
                if i2 in used_indices[c2]: continue
                
                cost = calc_cost([p1, p2], [i1, i2], [c1, c2])
                if cost < 0.2: # 2視点の場合は少し厳しく
                    candidates2.append({'ids': (i1, i2), 'cost': cost, 'cams': (c1, c2)})
        
        candidates2.sort(key=lambda x: x['cost'])
        
        for cand in candidates2:
            idx1, idx2 = cand['ids']
            if idx1 in used_indices[c1] or idx2 in used_indices[c2]: continue
            
            # Personリストの順序を (Cam1, Cam2, Cam3) に合わせるため None を入れる
            p_res = [None, None, None]
            p_res[c1] = ppl_lists[c1][idx1]
            p_res[c2] = ppl_lists[c2][idx2]
            
            matched_people.append({
                'persons': p_res,
                'cams': [c1, c2]
            })
            used_indices[c1].add(idx1)
            used_indices[c2].add(idx2)

    return matched_people

def solve_3d_joints(persons, cams, Ps):
    """ マッチングした人物データから3D関節を計算する（柔軟なカメラ選択） """
    kps_3d = np.full((33, 3), np.nan)
    
    for j in range(33):
        valid_pts = []
        valid_Ps = []
        
        # 登録されているカメラ(cams)の中で、信頼度が高いものを集める
        for cam_idx in cams:
            p = persons[cam_idx]
            if p is not None and p["conf"][j] > 0.5:
                valid_pts.append(p["kps_norm"][j])
                valid_Ps.append(Ps[cam_idx])
        
        # 2視点以上あれば3D化
        if len(valid_pts) >= 2:
            X, _ = triangulate_DLT(valid_Ps, valid_pts)
            if np.linalg.norm(X) < 50.0: # 異常値除外
                kps_3d[j] = X
                
    return kps_3d

# ==========================================
# Main
# ==========================================

def main():
    print(f"=== Hybrid v4 Debug & Robust: Dual Pose + Ball ===")
    
    if not os.path.exists(CALIB_NPZ): return

    cam_params_full, extrinsics = load_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    P1, P2, P3 = [p[2] for p in cam_params_full]
    Ps = [P1, P2, P3]

    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Loading YOLO Pose...")
    yolo_pose = YOLO(DET_MODEL) 
    print("Loading YOLO Detect...")
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
    colors = [(0,255,0), (0,0,255), (255,255,0)] # 緑、赤、水色

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

            # --- DEBUG: 生のBBox描画 ---
            if DRAW_DEBUG_BBOX:
                # 人 (青)
                for per in p:
                    bx = list(map(int, per["box"]))
                    cv2.rectangle(f, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 2)
                    cv2.putText(f, f"{per['score']:.2f}", (bx[0], bx[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                # ボール (赤)
                for bal in b:
                    bx = list(map(int, bal["box"]))
                    cv2.rectangle(f, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)

        # --- 2. Person Matching (Robust) ---
        matched_results = find_best_matches(ppl_lists, Ps)
        
        # --- 3. Person 3D Reconstruction & Draw ---
        for pid, match in enumerate(matched_results):
            persons = match['persons'] # [p_cam1, p_cam2, p_cam3] (None含む)
            active_cams = match['cams']
            col = colors[pid % len(colors)]
            
            # 3D計算 (柔軟なカメラ選択)
            kps_3d = solve_3d_joints(persons, active_cams, Ps)
            
            # CSV出力
            for j in range(33):
                if not np.isnan(kps_3d[j][0]):
                    writer.writerow([i, pid, MP_JOINTS[j], kps_3d[j][0], kps_3d[j][1], kps_3d[j][2]])

            # 描画 (骨格)
            for cam_idx in range(3):
                p_data = persons[cam_idx]
                if p_data is None: continue # このカメラには映っていない
                
                frame = frames[cam_idx]
                
                # YOLO Keypoints (Yellow Dots)
                if "yolo_kps" in p_data:
                    for ykp in p_data["yolo_kps"]:
                        yx, yy = int(ykp[0]), int(ykp[1])
                        if 0 <= yx < W and 0 <= yy < H:
                            cv2.circle(frame, (yx, yy), 3, (0, 255, 255), -1)

                # MediaPipe Skeleton (Lines)
                kps_raw = p_data["kps_raw"]
                for u, v in mp_conn:
                    if u < len(kps_raw) and v < len(kps_raw):
                        pt1 = (int(kps_raw[u][0]), int(kps_raw[u][1]))
                        pt2 = (int(kps_raw[v][0]), int(kps_raw[v][1]))
                        cv2.line(frame, pt1, pt2, col, 2)
                        
                # ID表示
                box = p_data["box"]
                cv2.putText(frame, f"ID:{pid}", (int(box[0]), int(box[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

        # --- 4. Ball Matching (2視点以上対応) ---
        # ボールは1フレーム1個と仮定するが、複数候補がある場合は最も信頼度が高い組み合わせを探す
        # ここでは簡単のため、各カメラのTop1ボールを使って、2視点以上あれば採用する
        
        valid_balls = []
        valid_ball_Ps = []
        ball_centers_for_draw = [None, None, None]

        for cam_idx in range(3):
            if len(ball_lists[cam_idx]) > 0:
                # 一番信頼度の高いボールを採用
                b = ball_lists[cam_idx][0] 
                valid_balls.append(b["center_norm"])
                valid_ball_Ps.append(Ps[cam_idx])
                ball_centers_for_draw[cam_idx] = b["box"]

        if len(valid_balls) >= 2:
            X_b, err_b = triangulate_DLT(valid_ball_Ps, valid_balls)
            
            # エラー閾値チェック (ボールは小さいので厳しめにしたいが、動きが速いので甘めに)
            if np.linalg.norm(X_b) < 50.0 and err_b < 10.0:
                writer.writerow([i, -1, "ball", X_b[0], X_b[1], X_b[2]])
                
                # 描画
                for cam_idx in range(3):
                    box = ball_centers_for_draw[cam_idx]
                    if box is not None:
                        cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                        cv2.circle(frames[cam_idx], (cx, cy), 10, (0, 165, 255), -1)
                        cv2.putText(frames[cam_idx], "BALL 3D", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

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