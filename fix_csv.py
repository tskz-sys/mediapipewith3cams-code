import pandas as pd
import cv2
import numpy as np
import os
import sys

# ==========================================
# 1. ユーザー設定 (パスが正しいか必ず確認してください)
# ==========================================
CSV_PATH = "./csv/kakunin.csv"
NPZ_PATH = "./npz/11253cams_fixedd.npz"
VIDEO_PATH = "../ffmpeg/output/2match1_2.mp4" 
CAM_INDEX = 1  # 0:Cam1, 1:Cam2, 2:Cam3

SAVE_PATH = "./csv/kakunin_fixed.csv"

# ==========================================
# 2. 高精度投影クラス
# ==========================================
class ProtopProjector:
    def __init__(self, npz_path, video_path, cam_idx):
        if not os.path.exists(npz_path):
            print(f"エラー: NPZファイルが見つかりません: {npz_path}")
            sys.exit()
            
        d = np.load(npz_path, allow_pickle=True)
        print(f"NPZを読み込みました。利用可能なキー: {list(d.keys())}")
        
        def get_data(prefix_new, prefix_old, idx):
            k_new = f"{prefix_new}{idx}"
            k_old = f"{prefix_old}{idx}"
            if k_new in d: return d[k_new]
            if k_old in d: return d[k_old]
            # dist_coeffs対策
            k_extra = f"dist_coeffs{idx}"
            if k_extra in d: return d[k_extra]
            return None

        self.K = get_data("K", "cam_matrix", cam_idx + 1)
        self.dist = get_data("D", "dist", cam_idx + 1)
        
        if self.K is None or self.dist is None:
            print(f"エラー: カメラ {cam_idx+1} のパラメータが見つかりません。")
            sys.exit()

        if cam_idx == 0:
            R_raw = d["R1"]; t_raw = d["T1"] if "T1" in d else d["t1"]
            self.R = R_raw.T; self.t = -R_raw.T @ t_raw.reshape(3,1)
        elif cam_idx == 1:
            self.R = np.eye(3); self.t = np.zeros((3,1))
        else:
            R_raw = d["R3"]; t_raw = d["T3"] if "T3" in d else d["t3"]
            self.R = R_raw.T; self.t = -R_raw.T @ t_raw.reshape(3,1)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"エラー: 動画が開けません: {video_path}")
            sys.exit()
        self.vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        orig_w, orig_h = self.K[0, 2] * 2, self.K[1, 2] * 2
        sx, sy = self.vw / orig_w, self.vh / orig_h
        self.K[0, 0] *= sx; self.K[1, 1] *= sy
        self.K[0, 2] *= sx; self.K[1, 2] *= sy
        print(f"動画解像度: {self.vw}x{self.vh} に合わせてKをスケーリングしました。")

    def project(self, x, y, z):
        pts_3d = np.array([[x, y, z]], dtype=np.float32)
        pts_2d, _ = cv2.projectPoints(pts_3d, self.R, self.t, self.K, self.dist)
        p = pts_2d.ravel()
        return int(p[0]), int(p[1])

# ==========================================
# 3. メインループ
# ==========================================
if not os.path.exists(CSV_PATH):
    print(f"エラー: CSVファイルが見つかりません: {CSV_PATH}")
    sys.exit()

df = pd.read_csv(CSV_PATH)
projector = ProtopProjector(NPZ_PATH, VIDEO_PATH, CAM_INDEX)
cap = cv2.VideoCapture(VIDEO_PATH)

current_frame = 0
cv2.namedWindow("Fixer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fixer", 1280, 720) # ウィンドウサイズを見やすく固定

print("\n--- 操作方法 ---")
print("D キー: 次のフレーム")
print("A キー: 前のフレーム")
print("S キー: 現在移行の人物ID(0/1)を入れ替え")
print("W キー: CSVを保存")
print("ESC キー: 終了")

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    
    if not ret:
        print(f"フレーム {current_frame} の読み込みに失敗しました。終了します。")
        break

    # 描画処理
    f_data = df[df['frame'] == current_frame]
    for _, row in f_data.iterrows():
        px, py = projector.project(row['X'], row['Y'], row['Z'])
        
        if row['person_id'] == -1:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0) if row['person_id'] == 0 else (0, 0, 255)

        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
            cv2.circle(frame, (px, py), 5, color, -1)

    cv2.putText(frame, f"Frame: {current_frame}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Fixer", frame)

    # キー待機
    key = cv2.waitKey(0) & 0xFF
    
    if key == 27: # ESC
        break
    elif key == ord('d'):
        current_frame += 1
    elif key == ord('a'):
        current_frame = max(0, current_frame - 1)
    elif key == ord('w'):
        df.to_csv(SAVE_PATH, index=False)
        print(f"保存しました: {SAVE_PATH}")
    elif key == ord('s'):
        mask = (df['frame'] >= current_frame) & (df['person_id'].isin([0, 1]))
        df.loc[mask, 'person_id'] = df.loc[mask, 'person_id'].map({0: 1, 1: 0})
        print(f"フレーム {current_frame} 以降のIDを入れ替えました。")

cap.release()
cv2.destroyAllWindows()