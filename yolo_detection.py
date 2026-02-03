#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# YOLOv11 Detection Check (Person & Ball)
#   - YOLOが正しく検出できているか確認するためのスクリプト
#   - 3D計算は行わず、2Dの検出ボックス(BBox)のみを描画
# ==========================================================

import os
import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# ★ 設定
# ==========================================
VIDEO_LEFT   = "./movie/Prematch1.mp4"
VIDEO_CENTER = "./movie/Prematch2.mp4"
VIDEO_RIGHT  = "./movie/Prematch3.mp4"

OUT_VIDEO = "./movie/yolo_detection_check.mp4"

# モデル設定
MODEL_PATH = "yolo11x.pt"
CONF_THRES = 0.25        # 検出の閾値 (低すぎるとゴミを拾い、高すぎると見逃す)
INFERENCE_SIZE = 1280    # 推論サイズ (大きいほど遠くの小物体に強い)

# ==========================================
# メイン処理
# ==========================================
def main():
    print("=== YOLOv11 Detection Check (Person & Ball) ===")

    # 動画オープン
    caps = [cv2.VideoCapture(v) for v in [VIDEO_LEFT, VIDEO_CENTER, VIDEO_RIGHT]]
    for i, c in enumerate(caps):
        if not c.isOpened():
            print(f"Error: Cannot open video {i}")
            return

    # 動画情報取得
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # モデルロード
    print(f"Loading Model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # 出力設定
    temp_out = OUT_VIDEO + ".temp.mp4"
    out_w_total = W * 3
    vw = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w_total, H))

    print(f"Processing {total_frames} frames...")

    for _ in tqdm(range(total_frames)):
        frames = []
        for c in caps:
            ret, f = c.read()
            frames.append(f if ret else None)
        
        if any(f is None for f in frames): break

        # 各カメラで推論 & 描画
        annotated_frames = []
        for f in frames:
            # classes=[0, 32] -> Person(0) と Sports ball(32) のみ検出
            results = model.predict(
                f, 
                conf=CONF_THRES, 
                imgsz=INFERENCE_SIZE, 
                classes=[0, 32], 
                verbose=False
            )
            
            # YOLO標準の描画機能を使ってBBoxを描き込む
            # line_width=3: 線の太さ
            # font_size=1.0: 文字サイズ
            res_plotted = results[0].plot(line_width=3, font_size=1.0)
            annotated_frames.append(res_plotted)

        # 3画面連結
        combined = np.hstack(annotated_frames)
        vw.write(combined)

    vw.release()
    for c in caps: c.release()

    # FFmpeg変換 (H.264)
    if os.path.exists(temp_out):
        print("Converting to H.264...")
        cmd = [
            "ffmpeg", "-y", "-i", temp_out,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            OUT_VIDEO
        ]
        subprocess.run(cmd, check=False)
        if os.path.exists(OUT_VIDEO):
            os.remove(temp_out)
            print(f"Done! Saved: {OUT_VIDEO}")

if __name__ == "__main__":
    main()