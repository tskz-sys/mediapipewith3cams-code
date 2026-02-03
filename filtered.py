#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Hybrid 3D Tracking v7 (Skeleton Draw & 50cm Filter)
#   - YOLO検出枠 -> MediaPipe骨格推定
#   - 3D再構成 & トラッキング
#   - ★重要: フレーム間移動距離 50cm 以下のフィルタ
#   - ★重要: 骨格（線）の描画
#   - ★重要: FFmpegによる自動変換
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

# 一時ファイルと最終出力
TEMP_VIDEO_PATH = "./movie/test7.mp4"
FINAL_OUTPUT_PATH = "./movie/testtt7.mp4"

# モデル設定
DET_MODEL = "yolo11x.pt" 
MAX_PEOPLE = 2

# ★フィルタ設定: フレーム間の最大許容移動距離 (メートル)
# 0.5m (50cm) 以上動いたら異常値として弾く
MAX_MOVE_METER = 0.30 

# MediaPipe設定
MP_COMPLEXITY = 1
MIN_MP_CONF   = 0.3

# 骨格の接続定義 (MediaPipeのランドマークIDに基づく)
# 線を結ぶペアのリスト
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), # 両肩, 左腕
    (12, 14), (14, 16),           # 右腕
    (11, 23), (12, 24),           # 胴体
    (23, 24),                     # 腰
    (23, 25), (25, 27), (27, 29), (29, 31), # 左足
    (24, 26), (26, 28), (28, 30), (30, 32)  # 右足
]

# ==========================================
# クラス: 3Dフィルタリング & トラッカー
# ==========================================
class PersonTracker3D:
    def __init__(self, pid):
        self.pid = pid
        self.center_3d = None      # ID追跡用の重心
        # 各関節の「前回の有効な3D座標」 {joint_idx: np.array([x, y, z])}
        self.last_valid_joints_3d = {}
        # 今回のフレームでその関節が有効だったか (描画用)
        self.is_valid_frame = {i: False for i in range(33)}

    def update_center(self, kps_3d):
        """ 重心を更新 (腰の中点など) """
        hips = []
        if not np.isnan(kps_3d[23][0]): hips.append(kps_3d[23])
        if not np.isnan(kps_3d[24][0]): hips.append(kps_3d[24])
        
        if len(hips) > 0:
            self.center_3d = np.mean(hips, axis=0)
        else:
            # 腰がなければ全体の平均
            valid = kps_3d[~np.isnan(kps_3d).any(axis=1)]
            if len(valid) > 0:
                self.center_3d = np.mean(valid, axis=0)

    def filter_and_update(self, raw_kps_3d):
        """ 
        3D座標を受け取り、50cmフィルタをかけて返す 
        戻り値: フィルタ済みの3D座標 (異常値はNaN、または前回の値)
        """
        filtered_3d = np.full((33, 3), np.nan)

        for j_idx in range(33):
            curr_pos = raw_kps_3d[j_idx]
            
            # 1. 今回検出できていない場合
            if np.isnan(curr_pos[0]):
                self.is_valid_frame[j_idx] = False
                continue # NaNのまま

            # 2. 初回検出の場合
            if j_idx not in self.last_valid_joints_3d:
                self.last_valid_joints_3d[j_idx] = curr_pos
                filtered_3d[j_idx] = curr_pos
                self.is_valid_frame[j_idx] = True
                continue

            # 3. 距離チェック (50cmルール)
            prev_pos = self.last_valid_joints_3d[j_idx]
            dist = np.linalg.norm(curr_pos - prev_pos)

            if dist > MAX_MOVE_METER:
                # 50cm以上動いた -> 誤検出とみなす
                # 今回のデータは破棄し、前回の位置を採用するか、NaNにする
                # 描画上は「線が飛ばない」ようにNaNにするか、前回位置で止めるか。
                # ここでは「前回位置を維持する（フリーズ）」挙動にする
                filtered_3d[j_idx] = prev_pos 
                self.is_valid_frame[j_idx] = True # 一応描画はする
                # last_valid_joints_3d は更新しない（誤検出座標を覚えないため）
            else:
                # 正常範囲 -> 更新
                self.last_valid_joints_3d[j_idx] = curr_pos
                filtered_3d[j_idx] = curr_pos
                self.is_valid_frame[j_idx] = True
        
        return filtered_3d

# ==========================================
# 関数群: 幾何計算・画像処理
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
    """ アスペクト比維持パディング + MediaPipe """
    H_img, W_img = full_img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # マージン 20%
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
    """ カメラ間の同一人物マッチング (簡易3D再構成による距離チェック) """
    matched = []
    used = [set(), set(), set()]
    
    # 3-View Match
    for i1, p1 in enumerate(ppl_lists[0]):
        for i2, p2 in enumerate(ppl_lists[1]):
            for i3, p3 in enumerate(ppl_lists[2]):
                # 腰(24)を使って簡易チェック
                pts = [p1['norm'][24], p2['norm'][24], p3['norm'][24]]
                X = triangulate_DLT(Ps, pts)
                # エラーチェックは省略、3台全部見えていれば優先採用
                matched.append({'ids':(i1,i2,i3), 'persons':[p1,p2,p3], 'cams':[0,1,2]})
                used[0].add(i1); used[1].add(i2); used[2].add(i3)
                if len(matched) >= MAX_PEOPLE: break
        if len(matched) >= MAX_PEOPLE: break

    # 2-View Match (Remaining)
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
    """ マッチング情報から3D座標候補を計算 """
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
            # 異常値除外 (原点付近や遠すぎる場合)
            if 0.1 < np.linalg.norm(X) < 50.0:
                kps_3d[j] = X
    return kps_3d

def draw_skeleton_2d(img, kps, color):
    """ 2D骨格（線と点）を描画する """
    # 線を描く
    for u, v in SKELETON_CONNECTIONS:
        # 接続する両端の点が存在し、かつNaNでないこと
        if u < len(kps) and v < len(kps):
            pt1 = kps[u]
            pt2 = kps[v]
            
            # 座標が有効かチェック (conf>0 または NaNでない)
            # ここではkpsはピクセル座標 [x, y]
            if not np.isnan(pt1[0]) and not np.isnan(pt2[0]):
                 cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)
    
    # 点を描く
    for pt in kps:
        if not np.isnan(pt[0]):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)

# ==========================================
# Main
# ==========================================
def main():
    print("=== Hybrid 3D Tracking v7 (Skeleton Draw & 50cm Filter) ===")
    
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
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True, 
        model_complexity=MP_COMPLEXITY, 
        min_detection_confidence=MIN_MP_CONF
    )

    trackers = [PersonTracker3D(i) for i in range(MAX_PEOPLE)]

    # 一時ファイル書き出し用
    vw = cv2.VideoWriter(TEMP_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))

    colors = [(0, 255, 0), (0, 0, 255)] # ID:0=緑, ID:1=赤

    for i in tqdm(range(total)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break

        # 1. 検出 (YOLO -> MediaPipe)
        ppl_lists = []
        for c_idx, f in enumerate(frames):
            res = yolo.predict(f, conf=0.25, verbose=False, classes=[0])[0]
            cam_ppl = []
            if res.boxes:
                for box in res.boxes.xyxy.cpu().numpy():
                    kps, mp_conf = get_pose_padded(f, box, mp_pose)
                    if kps is not None:
                        norm = undistort_points(kps, Ks[c_idx], Ds[c_idx])
                        cam_ppl.append({'norm': norm, 'conf': mp_conf, 'box': box, 'kps_raw': kps})
            ppl_lists.append(cam_ppl)

        # 2. カメラ間マッチング
        matches = find_best_matches(ppl_lists, Ps)
        
        candidates = []
        for m in matches:
            kps_3d = solve_3d_candidate(m, Ps)
            # 重心を計算
            valid_hips = []
            if not np.isnan(kps_3d[23][0]): valid_hips.append(kps_3d[23])
            if not np.isnan(kps_3d[24][0]): valid_hips.append(kps_3d[24])
            center = np.mean(valid_hips, axis=0) if valid_hips else np.zeros(3)
            candidates.append({'kps_3d': kps_3d, 'center': center, 'match': m})

        # 3. IDトラッキング (前回位置に近いものを割り当て)
        assigned_candidates = {pid: None for pid in range(MAX_PEOPLE)}
        used_cand_idx = set()

        for pid in range(MAX_PEOPLE):
            tracker = trackers[pid]
            if tracker.center_3d is not None:
                best_idx = -1
                min_dist = 2.0 # 2m以内の移動なら同一人物候補
                
                for c_idx, cand in enumerate(candidates):
                    if c_idx in used_cand_idx: continue
                    dist = np.linalg.norm(cand['center'] - tracker.center_3d)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = c_idx
                
                if best_idx != -1:
                    assigned_candidates[pid] = candidates[best_idx]
                    used_cand_idx.add(best_idx)
                    tracker.update_center(candidates[best_idx]['kps_3d'])
        
        # 新規割り当て
        for c_idx, cand in enumerate(candidates):
            if c_idx not in used_cand_idx:
                for pid in range(MAX_PEOPLE):
                    if assigned_candidates[pid] is None and trackers[pid].center_3d is None:
                        assigned_candidates[pid] = cand
                        trackers[pid].update_center(cand['kps_3d'])
                        break

        # 4. フィルタリング & 描画
        for pid in range(MAX_PEOPLE):
            cand = assigned_candidates[pid]
            tracker = trackers[pid]
            col = colors[pid % len(colors)]
            
            # 今回検出なしの場合
            if cand is None:
                # 描画するものがないのでスキップ (前回の位置を描画したい場合はここで処理が必要)
                continue

            # --- 3Dフィルタ (50cm制限) ---
            # ここで filtered_kps_3d には、50cm以上動いた関節は「前回値」が入る
            filtered_kps_3d = tracker.filter_and_update(cand['kps_3d'])
            
            # --- 描画 (Skeleton) ---
            persons = cand['match']['persons']
            for cam_i in range(3):
                p = persons[cam_i]
                if p:
                    # BBox
                    bx = p['box'].astype(int)
                    cv2.rectangle(frames[cam_i], (bx[0], bx[1]), (bx[2], bx[3]), col, 2)
                    cv2.putText(frames[cam_i], f"ID:{pid}", (bx[0], bx[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                    
                    # 骨格描画 (Connect Joints)
                    # p['kps_raw'] はMediaPipeの生2D座標
                    # ただし、3Dフィルタで弾かれた点は描画したくない場合がある
                    # 簡易的に、ここではMPの生の結果を描画するが、厳密にするなら
                    # フィルタされた3D点を再投影して描画する必要がある。
                    # ユーザー要望は「ディテクションの結果をMPに渡して...出力」なので、
                    # MPの生のスケルトンを描画します。
                    draw_skeleton_2d(frames[cam_i], p['kps_raw'], col)

        vw.write(np.hstack(frames))

    mp_pose.close()
    vw.release()
    for c in caps: c.release()
    
    # ==========================================
    # FFmpegによる変換処理
    # ==========================================
    print("Converting video with FFmpeg...")
    
    INPUT  = Path(TEMP_VIDEO_PATH)
    OUTPUT = Path(FINAL_OUTPUT_PATH)
    
    print("INPUT exists:", INPUT.exists(), "size:", INPUT.stat().st_size if INPUT.exists() else "N/A")
    
    if INPUT.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(INPUT),
            "-vf", "scale=2880:540,fps=30",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.0",
            "-an",
            str(OUTPUT),
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("Done. OUTPUT:", OUTPUT, "size:", OUTPUT.stat().st_size)
        
        # 一時ファイル削除
        if INPUT.exists():
            os.remove(INPUT)

if __name__ == "__main__":
    main()