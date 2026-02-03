#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# === 設定 ===
VIDEO_LEFT   = "./movie/Prematch1.mp4"
VIDEO_CENTER = "./movie/Prematch2.mp4"
VIDEO_RIGHT  = "./movie/Prematch3.mp4"
CALIB_NPZ    = "./npz/11253cams_fixedd.npz"
OUT_CSV      = "./csv/ball_output.csv" # ボール専用CSV

CONF_BALL      = 0.25
INFERENCE_SIZE = 1280
# ============

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
    
    c = cv2.VideoCapture(v1); w, h = int(c.get(3)), int(c.get(4)); c.release()
    w1, h1 = w, h; w2, h2 = w, h; w3, h3 = w, h # Same resolution assumed
    
    K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

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
        A.append(x * P[2] - P[0]); A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]; X = X[:3] / X[3]
    errs = []
    for P, (x, y) in zip(Ps, pts):
        xh = P @ np.hstack([X, 1.0])
        if abs(xh[2]) < 1e-9: errs.append(1000.0); continue
        xp, yp = xh[0]/xh[2], xh[1]/xh[2]
        errs.append((xp-x)**2 + (yp-y)**2)
    return X, np.mean(errs)

def main():
    print("=== Ball Detection Only ===")
    cam_params_full, extrinsics = load_params_BR(CALIB_NPZ, VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT)
    P1, P2, P3 = [p[2] for p in cam_params_full]
    
    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    
    model = YOLO("yolo11x.pt") # Detection Model
    
    f_csv = open(OUT_CSV, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(["frame", "person_id", "joint", "X", "Y", "Z"])
    
    print(f"Processing {total_frames} frames for BALL...")
    for i in tqdm(range(total_frames)):
        frames = []
        for c in caps: _, f = c.read(); frames.append(f)
        if len(frames)<3 or frames[0] is None: break
        
        ball_lists = []
        for cam_i, f in enumerate(frames):
            K, D, _ = cam_params_full[cam_i]
            res = model.predict(f, conf=CONF_BALL, imgsz=INFERENCE_SIZE, verbose=False, classes=[32])[0]
            balls = []
            if res.boxes is not None and len(res.boxes) > 0:
                best_idx = np.argmax(res.boxes.conf.cpu().numpy())
                box = res.boxes.xyxy[best_idx].cpu().numpy()
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                norm_pt = undistort_points([(cx, cy)], K, D)[0]
                balls.append({"center_norm": norm_pt})
            ball_lists.append(balls)
            
        b1, b2, b3 = [bl[0] if bl else None for bl in ball_lists]
        if b1 and b2 and b3:
            pts = [b1["center_norm"], b2["center_norm"], b3["center_norm"]]
            X_b, err_b = triangulate_DLT([P1, P2, P3], pts)
            if np.linalg.norm(X_b) < 50.0 and err_b < 5.0:
                writer.writerow([i, -1, "ball", X_b[0], X_b[1], X_b[2]])
                
    f_csv.close()
    print(f"Ball CSV saved: {OUT_CSV}")

if __name__ == "__main__":
    main()