#!/usr/bin/env python3
import csv
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
import os

# === 設定 ===
VIDEO_LEFT   = "./movie/Prematch1.mp4"
VIDEO_CENTER = "./movie/Prematch2.mp4"
VIDEO_RIGHT  = "./movie/Prematch3.mp4"
CALIB_NPZ    = "./npz/11253cams_fixedd.npz"

CSV_PERSON = "./csv/hybrid_v3_output.csv" # Step1で作った人のCSV
CSV_BALL   = "./csv/ball_output.csv"     # Step2で作ったボールのCSV
OUT_VIDEO  = "./movie/merged_final.mp4"

# 関節定義
JOINTS = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
J_MAP = {n: i for i, n in enumerate(JOINTS)}
# ============

# ... (get_inverse_transform, scale_camera_matrix, load_params_BR は先ほどと同じなので省略可能ですが、
#      単体で動くようにここにコピーしておくのが安全です。下記に最小限記述します) ...

def get_inverse_transform(R, T): return R.T, -R.T @ T
def scale_camera_matrix(K, dist, target_w, target_h):
    orig_w, orig_h = K[0, 2] * 2, K[1, 2] * 2
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05: return K, dist
    K_new = K.copy(); K_new[0, 0] *= sx; K_new[1, 1] *= sy; K_new[0, 2] *= sx; K_new[1, 2] *= sy
    return K_new, dist
def load_params_BR(npz_path, w, h):
    d = np.load(npz_path, allow_pickle=True)
    def get_k(k_new, k_old): return d[k_new] if k_new in d else d[k_old]
    K1, D1 = get_k("K1","cam_matrix1"), get_k("dist1","dist_coeffs1")
    K2, D2 = get_k("K2","cam_matrix2"), get_k("dist2","dist_coeffs2")
    K3, D3 = get_k("K3","cam_matrix3"), get_k("dist3","dist_coeffs3")
    w1, h1 = w, h; w2, h2 = w, h; w3, h3 = w, h
    K1, D1 = scale_camera_matrix(K1, D1, w1, h1); K2, D2 = scale_camera_matrix(K2, D2, w2, h2); K3, D3 = scale_camera_matrix(K3, D3, w3, h3)
    R1_raw = d["R1"]; t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3,1))
    R2 = np.eye(3); t2 = np.zeros((3, 1))
    R3_raw = d["R3"]; t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3,1))
    P1_ext = np.hstack([R1, t1]); P2_ext = np.hstack([R2, t2]); P3_ext = np.hstack([R3, t3])
    return [(K1, D1, P1_ext), (K2, D2, P2_ext), (K3, D3, P3_ext)], [(R1, t1), (R2, t2), (R3, t3)]

def load_data(csv_path):
    data = {}
    if not os.path.exists(csv_path): return data
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row['frame'])
            pid = int(row['person_id'])
            jn = row['joint']
            x, y, z = float(row['X']), float(row['Y']), float(row['Z'])
            if fi not in data: data[fi] = {'people': {}, 'ball': None}
            
            if jn == 'ball':
                data[fi]['ball'] = np.array([x, y, z])
            else:
                if pid not in data[fi]['people']: data[fi]['people'][pid] = np.full((17, 3), np.nan)
                if jn in J_MAP: data[fi]['people'][pid][J_MAP[jn]] = [x, y, z]
    return data

def main():
    print("=== Merge & Visualize ===")
    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    
    cam_params, extrinsics = load_params_BR(CALIB_NPZ, W, H)
    
    # Load both CSVs
    data_person = load_data(CSV_PERSON)
    data_ball   = load_data(CSV_BALL)
    
    temp_out = "temp_merge.mp4"
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*3, H))
    
    edges = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    colors = [(0,255,0), (0,0,255), (255,0,0)]
    LIMIT = 5000

    for i in tqdm(range(total)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3: break
        
        # Draw Person
        if i in data_person:
            for pid, kps3d in data_person[i]['people'].items():
                col = colors[pid % 3]
                for cam_i in range(3):
                    K, D, _ = cam_params[cam_i]
                    R, t = extrinsics[cam_i]
                    rvec, _ = cv2.Rodrigues(R)
                    mask = ~np.isnan(kps3d).any(axis=1)
                    if np.any(mask):
                        img_pts, _ = cv2.projectPoints(kps3d[mask], rvec, t, K, D)
                        img_pts = img_pts.reshape(-1, 2)
                        kps2d = {}
                        cnt = 0
                        for j in range(17):
                            if mask[j]: kps2d[j] = img_pts[cnt]; cnt+=1
                        for u, v in edges:
                            if u in kps2d and v in kps2d:
                                pt1 = (int(np.clip(kps2d[u][0], -LIMIT, LIMIT)), int(np.clip(kps2d[u][1], -LIMIT, LIMIT)))
                                pt2 = (int(np.clip(kps2d[v][0], -LIMIT, LIMIT)), int(np.clip(kps2d[v][1], -LIMIT, LIMIT)))
                                cv2.line(frames[cam_i], pt1, pt2, col, 2)

        # Draw Ball
        if i in data_ball and data_ball[i]['ball'] is not None:
            b3d = data_ball[i]['ball']
            for cam_i in range(3):
                K, D, _ = cam_params[cam_i]
                R, t = extrinsics[cam_i]
                rvec, _ = cv2.Rodrigues(R)
                pt, _ = cv2.projectPoints(b3d.reshape(1,1,3), rvec, t, K, D)
                bx, by = int(np.clip(pt[0][0][0], -LIMIT, LIMIT)), int(np.clip(pt[0][0][1], -LIMIT, LIMIT))
                if 0<=bx<W and 0<=by<H:
                    cv2.circle(frames[cam_i], (bx, by), 10, (0, 165, 255), 2)
                    cv2.putText(frames[cam_i], "BALL", (bx+10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        vw.write(np.hstack(frames))

    vw.release()
    for c in caps: c.release()
    
    if os.path.exists(temp_out):
        subprocess.run(["ffmpeg", "-y", "-i", temp_out, "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT_VIDEO], check=False)
        os.remove(temp_out)
        print(f"Saved: {OUT_VIDEO}")

if __name__ == "__main__":
    main()