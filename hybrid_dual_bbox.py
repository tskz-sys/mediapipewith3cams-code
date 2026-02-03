#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# YOLO + MediaPipe Hybrid [v5 PADDING & EDGE FILTER]
#   - 修正1: 画面端で切れた画像を無理やり伸ばさず、黒帯(パディング)を入れてアスペクト比を維持
#            → これにより「潰れた姿勢」や「太った姿勢」になるのを防ぐ
#   - 修正2: 画面外(0未満やW以上)にあると推定された関節の信頼度を強制的に0にする
#            → これにより、見えていない足を無理やり3D化して精度が落ちるのを防ぐ
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
OUT_BBOX_CSV = "./csv/bbox_log.csv"  # ★追加: BBox保存先
OUT_VIDEO = "./movie/kakunin.mp4"

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
MIN_MP_CONF   = 0.3

# Debug描画（TrueにするとYOLOの生BBoxとパディング状況を確認可能）
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
    A = []
    for P, (x, y) in zip(Ps, pts):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    X = X[:3] / X[3]
    
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
# 検出ロジック (パディング追加版)
# ==========================================

def get_pose_from_crop_padded(full_img, box, pose_model):
    """
    アスペクト比を維持したまま正方形にパディングし、座標を正確に元画像へ戻す関数
    """
    H_img, W_img = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # 1. 枠を少し広げる（マージン）: 体が見切れるのを防ぐため重要
    margin_ratio = 0.2  # 20%広げる
    w_box = x2 - x1
    h_box = y2 - y1
    pad_w = int(w_box * margin_ratio)
    pad_h = int(h_box * margin_ratio)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(W_img, x2 + pad_w)
    y2 = min(H_img, y2 + pad_h)
    
    if x2 <= x1 or y2 <= y1: return None, None, (x1, y1, x2, y2)

    crop = full_img[y1:y2, x1:x2]
    h_crop, w_crop = crop.shape[:2]
    
    # 2. 正方形の黒画像（キャンバス）を作成
    # MediaPipeは正方形入力が最も精度が出ます
    size = max(h_crop, w_crop)
    padded_img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # 3. 画像を中央に配置するためのオフセット計算
    ox = (size - w_crop) // 2
    oy = (size - h_crop) // 2
    padded_img[oy:oy+h_crop, ox:ox+w_crop] = crop
    
    # 4. 推論実行
    crop_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(crop_rgb)
    
    if not results.pose_landmarks: return None, None, (x1, y1, x2, y2)
    
    kps = []
    confs = []
    
    # 5. 座標変換 (ここが最も重要)
    # MediaPipeの正規化座標(0~1) -> パディング画像座標 -> 元のクロップ座標 -> 全体画像座標
    for lm in results.pose_landmarks.landmark:
        # パディング画像上のピクセル座標 (0 ~ size)
        px = lm.x * size
        py = lm.y * size
        
        # パディング(ox, oy)を除去してクロップ画像上の座標へ
        cx = px - ox
        cy = py - oy
        
        # 全体画像の座標へ (x1, y1 を足す)
        global_x = cx + x1
        global_y = cy + y1
        
        kps.append([global_x, global_y])
        confs.append(lm.visibility)
        
    return np.array(kps), np.array(confs), (x1, y1, x2, y2)

def process_person_yolo(img, pose_model, mp_model, K, D, max_p):
    H, W = img.shape[:2]
    res = pose_model.predict(img, conf=CONF_LOW_LIMIT, imgsz=INFERENCE_SIZE, verbose=False, classes=[0])[0]
    
    people = []
    if res.boxes is None or len(res.boxes) == 0: return []

    indices = np.argsort(-res.boxes.conf.cpu().numpy())
    
    p_count = 0
    for idx in indices:
        conf = res.boxes.conf[idx].cpu().numpy()
        if conf < CONF_PERSON: continue
        if p_count >= max_p + 1: break

        box = res.boxes.xyxy[idx].cpu().numpy()
        yolo_kps = res.keypoints.xy[idx].cpu().numpy()

        # パディング付き推論を実行
        mp_kps, mp_confs, _ = get_pose_from_crop_padded(img, box, mp_model)
        
        if mp_kps is not None:
            # --- 画面外判定フィルタ ---
            # 関節座標が画像の端(0やW)を超えている場合、信頼度を強制的に0にする
            # これにより「切れている足」を「そこにある」と誤認するのを防ぐ
            for i in range(len(mp_kps)):
                kx, ky = mp_kps[i]
                margin = 5 # 5ピクセル余裕を見る
                if kx < margin or kx > W - margin or ky < margin or ky > H - margin:
                    mp_confs[i] = 0.0

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
# マッチング & 3D化
# ==========================================
def find_best_matches(ppl_lists, Ps):
    # 2視点マッチングを許容するロジック
    def calc_cost(p_list, p_indices, cam_indices):
        total_err = 0
        valid_joints = 0
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
        
        if valid_joints == 0: return 1000.0
        return total_err / valid_joints

    matched_people = []
    used_indices = [set() for _ in range(3)]

    # 1. 3視点マッチング
    candidates3 = []
    for i1, p1 in enumerate(ppl_lists[0]):
        for i2, p2 in enumerate(ppl_lists[1]):
            for i3, p3 in enumerate(ppl_lists[2]):
                cost = calc_cost([p1, p2, p3], [i1, i2, i3], [0, 1, 2])
                if cost < 0.3:
                    candidates3.append({'ids': (i1, i2, i3), 'cost': cost, 'cams': (0,1,2)})
    
    candidates3.sort(key=lambda x: x['cost'])
    
    for cand in candidates3:
        i1, i2, i3 = cand['ids']
        if i1 in used_indices[0] or i2 in used_indices[1] or i3 in used_indices[2]: continue
        matched_people.append({
            'persons': [ppl_lists[0][i1], ppl_lists[1][i2], ppl_lists[2][i3]],
            'cams': [0, 1, 2]
        })
        used_indices[0].add(i1); used_indices[1].add(i2); used_indices[2].add(i3)

    # 2. 2視点マッチング
    cam_pairs = [(0, 1), (1, 2), (0, 2)]
    for c1, c2 in cam_pairs:
        candidates2 = []
        for i1, p1 in enumerate(ppl_lists[c1]):
            if i1 in used_indices[c1]: continue
            for i2, p2 in enumerate(ppl_lists[c2]):
                if i2 in used_indices[c2]: continue
                cost = calc_cost([p1, p2], [i1, i2], [c1, c2])
                if cost < 0.2:
                    candidates2.append({'ids': (i1, i2), 'cost': cost, 'cams': (c1, c2)})
        
        candidates2.sort(key=lambda x: x['cost'])
        
        for cand in candidates2:
            idx1, idx2 = cand['ids']
            if idx1 in used_indices[c1] or idx2 in used_indices[c2]: continue
            p_res = [None, None, None]
            p_res[c1] = ppl_lists[c1][idx1]
            p_res[c2] = ppl_lists[c2][idx2]
            matched_people.append({'persons': p_res, 'cams': [c1, c2]})
            used_indices[c1].add(idx1); used_indices[c2].add(idx2)

    return matched_people

def solve_3d_joints(persons, cams, Ps):
    kps_3d = np.full((33, 3), np.nan)
    for j in range(33):
        valid_pts = []
        valid_Ps = []
        for cam_idx in cams:
            p = persons[cam_idx]
            # 信頼度チェック（画面外フィルタ済みの値）
            if p is not None and p["conf"][j] > 0.5:
                valid_pts.append(p["kps_norm"][j])
                valid_Ps.append(Ps[cam_idx])
        if len(valid_pts) >= 2:
            X, _ = triangulate_DLT(valid_Ps, valid_pts)
            if np.linalg.norm(X) < 50.0:
                kps_3d[j] = X
    return kps_3d

# ==========================================
# Main
# ==========================================

def main():
    print(f"=== Hybrid v5 Final + BBox Export ===")
    
    if not os.path.exists(CALIB_NPZ): return

    cam_params_full, _ = load_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    P1, P2, P3 = [p[2] for p in cam_params_full]
    Ps = [P1, P2, P3]

    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Loading Models...")
    yolo_pose = YOLO(DET_MODEL) 
    yolo_det = YOLO(BALL_MODEL) 
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(static_image_mode=True, model_complexity=MP_COMPLEXITY, min_detection_confidence=MIN_MP_CONF)

    f_csv = open(OUT_CSV, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z"])


    # ★追加: BBox CSV Open
    f_bbox = open(OUT_BBOX_CSV, 'w', newline='')
    writer_bbox = csv.writer(f_bbox)
    writer_bbox.writerow(["frame", "person_id", "cam_idx", "x1", "y1", "x2", "y2"])
    temp_out = OUT_VIDEO + ".temp.mp4"
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    
    mp_conn = mp.solutions.pose.POSE_CONNECTIONS
    colors = [(0,255,0), (0,0,255), (255,255,0)]

    print(f"Processing {total_frames} frames...")

    prev_centers = {}

    for i in tqdm(range(total_frames)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break

        ppl_lists = []
        ball_lists = []
        
        for cam_i, f in enumerate(frames):
            K, D, _ = cam_params_full[cam_i]
            p = process_person_yolo(f, yolo_pose, pose_estimator, K, D, MAX_PEOPLE)
            ppl_lists.append(p)
            b = detect_ball_yolo(f, yolo_det, K, D)
            ball_lists.append(b)

            if DRAW_DEBUG_BBOX:
                for per in p:
                    bx = list(map(int, per["box"]))
                    cv2.rectangle(f, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 2)
                for bal in b:
                    bx = list(map(int, bal["box"]))
                    cv2.rectangle(f, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)

        matched_results = find_best_matches(ppl_lists, Ps)

        final_persons = [None] * MAX_PEOPLE # ID順に格納するリスト
        
        # 現在のフレームで検出された候補リストを作成
        candidates = []
        for match in matched_results:
            persons = match['persons']
            active_cams = match['cams']
            # 仮の3D座標を計算して重心を取得
            kps_3d = solve_3d_joints(persons, active_cams, Ps)
            
            # 有効な関節の平均座標(重心)を計算
            valid_pts = kps_3d[~np.isnan(kps_3d).any(axis=1)]
            if len(valid_pts) > 0:
                center = np.mean(valid_pts, axis=0)
            else:
                center = np.array([0,0,0]) # 検出なし
            
            candidates.append({
                'match_data': match,
                'center': center,
                'kps_3d': kps_3d
            })

        # 最初のフレーム、または誰もいなかった後は、そのままリスト順で登録
        if i == 0 or len(prev_centers) == 0:
            for idx, cand in enumerate(candidates):
                if idx < MAX_PEOPLE:
                    final_persons[idx] = cand
                    prev_centers[idx] = cand['center']
        else:
            # 2フレーム目以降: 前回の位置に近い順に割り当て
            used_candidates = set()
            
            for pid in range(MAX_PEOPLE):
                if pid not in prev_centers: continue
                last_pos = prev_centers[pid]
                
                best_dist = 200.0 # 閾値 (2m以上飛んだら別人扱い)
                best_idx = -1
                
                for c_idx, cand in enumerate(candidates):
                    if c_idx in used_candidates: continue
                    
                    dist = np.linalg.norm(cand['center'] - last_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = c_idx
                
                if best_idx != -1:
                    final_persons[pid] = candidates[best_idx]
                    prev_centers[pid] = candidates[best_idx]['center']
                    used_candidates.add(best_idx)
            
            # 余った候補があれば、空いているIDに入れる (新規出現など)
            for c_idx, cand in enumerate(candidates):
                if c_idx not in used_candidates:
                    for pid in range(MAX_PEOPLE):
                        if final_persons[pid] is None:
                            final_persons[pid] = cand
                            prev_centers[pid] = cand['center']
                            break

        # ---------------------------------------------------------
        # CSV書き込み & 描画 (final_persons を使用)
        # ---------------------------------------------------------
        for pid in range(MAX_PEOPLE):
            cand = final_persons[pid]
            if cand is None: continue # このIDの人は今回いない

            kps_3d = cand['kps_3d']
            match = cand['match_data']
            persons = match['persons']
            col = colors[pid % len(colors)]

            # CSV出力
            for j in range(33):
                if not np.isnan(kps_3d[j][0]):
                    writer.writerow([i, pid, MP_JOINTS[j], kps_3d[j][0], kps_3d[j][1], kps_3d[j][2]])


            # ★追加: BBox書き込み（そのフレーム・その人の各カメラBBoxを保存）
            for cam_idx in range(3):
                p_data = persons[cam_idx]
                if p_data is None: 
                    continue
                box = p_data["box"]  # [x1, y1, x2, y2]
                writer_bbox.writerow([i, pid, cam_idx, box[0], box[1], box[2], box[3]])

            # 描画
            # (描画コードは元の matched_results ループの中身と同じですが、
            #  pid が固定されるため色が安定します)
            for cam_idx in range(3):
                p_data = persons[cam_idx]
                if p_data is None: continue
                
                frame = frames[cam_idx]
        
        # for pid, match in enumerate(matched_results):
        #     persons = match['persons']
        #     active_cams = match['cams']
        #     col = colors[pid % len(colors)]
        #     kps_3d = solve_3d_joints(persons, active_cams, Ps)
            
        #     for j in range(33):
        #         if not np.isnan(kps_3d[j][0]):
        #             writer.writerow([i, pid, MP_JOINTS[j], kps_3d[j][0], kps_3d[j][1], kps_3d[j][2]])

            for cam_idx in range(3):
                p_data = persons[cam_idx]
                if p_data is None: continue
                
                frame = frames[cam_idx]
                kps_raw = p_data["kps_raw"]
                
                # YOLO点
                if "yolo_kps" in p_data:
                    for ykp in p_data["yolo_kps"]:
                        cv2.circle(frame, (int(ykp[0]), int(ykp[1])), 3, (0, 255, 255), -1)

                # MediaPipe線 (信頼度0.5以上のみ描画)
                for u, v in mp_conn:
                    if u < len(kps_raw) and v < len(kps_raw):
                        if p_data["conf"][u] > 0.3 and p_data["conf"][v] > 0.3:
                            pt1 = (int(kps_raw[u][0]), int(kps_raw[u][1]))
                            pt2 = (int(kps_raw[v][0]), int(kps_raw[v][1]))
                            cv2.line(frame, pt1, pt2, col, 2)
                
                box = p_data["box"]
                cv2.putText(frame, f"ID:{pid}", (int(box[0]), int(box[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

        # Ball
        valid_balls = []
        valid_ball_Ps = []
        ball_centers_for_draw = [None]*3

        for cam_idx in range(3):
            if len(ball_lists[cam_idx]) > 0:
                b = ball_lists[cam_idx][0] 
                valid_balls.append(b["center_norm"])
                valid_ball_Ps.append(Ps[cam_idx])
                ball_centers_for_draw[cam_idx] = b["box"]

        if len(valid_balls) >= 2:
            X_b, err_b = triangulate_DLT(valid_ball_Ps, valid_balls)
            if np.linalg.norm(X_b) < 50.0 and err_b < 20.0: # ボール誤差許容範囲
                writer.writerow([i, -1, "ball", X_b[0], X_b[1], X_b[2]])
                for cam_idx in range(3):
                    box = ball_centers_for_draw[cam_idx]
                    if box is not None:
                        cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                        cv2.circle(frames[cam_idx], (cx, cy), 10, (0, 165, 255), -1)
                        cv2.putText(frames[cam_idx], "BALL 3D", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        vw.write(np.hstack(frames))

    f_csv.close()
    f_bbox.close()
    pose_estimator.close()
    vw.release()
    for c in caps: c.release()

    if os.path.exists(temp_out):
        subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT_VIDEO], check=False)
        os.remove(temp_out)
        print(f"Saved: {OUT_VIDEO}")

if __name__ == "__main__":
    main()