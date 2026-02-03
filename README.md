# mediapipewith3cams-code

3台カメラの動画から人物（＋ボール）を検出し、3D姿勢を三角測量してCSV化・後処理・可視化するためのスクリプト集です。
研究・実験用途の複数パイプラインが共存しており、`run_pipeline_hfs.py` と `run_pipeline_hybrid_hfs.py` が代表的な入口です。

**Features**
- 3視点（左/中央/右 or BR構成）での3D姿勢推定
- YOLO + MediaPipe のハイブリッド推定
- ボール検出の3D化（任意）
- CSV後処理（re-ID、ジャンプ除去、平滑化、QA、可視化）

**Requirements**
- Python 3.9+
- `mediapipe`, `ultralytics`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `tqdm`
- GPUは任意（YOLO推論が高速化されます）

**Install**
```
# uv を使う場合
uv sync

# venv + pip を使う場合
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Quick Start (batch3 パイプライン)**
1. 入力動画を1つのディレクトリに用意
```
1match{game}_{match}.mp4
2match{game}_{match}.mp4
3match{game}_{match}.mp4
```
2. キャリブレーションNPZを用意
3. 実行
```
python run_pipeline_hfs.py \
  --input_dir /path/to/sync1 \
  --calib_npz /path/to/calib.npz \
  --out_dir /path/to/output \
  --game 1 --matches 1-3 \
  --postprocess external \
  --compare_3d
```

**Alternative: Hybrid パイプライン**
`hybrid_dual_final_patched.py` を使う流れです。入力は同じ命名規則のディレクトリ、または3本動画を直接指定できます。
```
python run_pipeline_hybrid_hfs.py \
  --input_dir /path/to/sync1 \
  --calib_npz /path/to/calib.npz \
  --out_dir /path/to/output
```

**Inputs**
- 動画命名: `1match{game}_{match}.mp4` / `2match{game}_{match}.mp4` / `3match{game}_{match}.mp4`
- キャリブレーションNPZの期待キー: `K1/K2/K3`, `dist1/dist2/dist3`, `R1/t1`, `R3/t3`
- キャリブレーションNPZの互換キー（例）: `cam_matrix1`, `dist_coeffs1` など
- `run_pipeline_hfs.py` は不足キーがあれば互換NPZを自動生成します
- WSLのUNCパスは一部スクリプトで自動変換されます

**Outputs (代表)**
- `match{g}_{m}_raw.csv`: 生の3D推定
- `match{g}_{m}_fixed.csv`: re-ID / 追跡補正後
- `match{g}_{m}_smoothed.csv`: 平滑化後
- `match{g}_{m}_bbox.csv`: 2D bboxログ（出力するパイプラインのみ）
- `*_viz.mp4` / `*_compare.mp4`: 可視化動画

**CSV Format**
| column | description |
| --- | --- |
| frame | フレーム番号 |
| person_id | 人物ID（ボールは `-1`） |
| joint | 関節名（ボールは `"ball"`） |
| X, Y, Z | 3D座標 |

**Postprocess Utilities**
- `fix_pose_csv_adaptive.py`: 追跡の跳ねを抑制し、見失い時にリセット
- `smooth_csv.py`: 欠損補間 + 平滑化
- `postprocess_3d_pose_pipeline.py`: re-ID、骨長修復、QA、可視化まで一括

**Key Scripts**
- `batch3_3dposeestimation.py`: 3D推定の中核（YOLO + MediaPipe + 三角測量）
- `run_pipeline_hfs.py`: まとめ実行のラッパー（batch3系）
- `run_pipeline_hybrid_hfs.py`: hybrid系パイプライン
- `mp_pose_3d.py`: MediaPipeのみの3D推定サンプル
- `detect_ball_only/py`: ボール検出のみのサンプル

**Notes**
- YOLOの重いモデル（例: `yolo11x-pose.pt`）は初回に自動ダウンロードされます。
- 大きい動画ではVRAM/CPU負荷が高いので、`INFERENCE_SIZE` や `CONF_*` を調整してください。
- CSV後処理は `run_pipeline_hfs.py --postprocess external` が最も簡易です。

**Development**
- 変更が大きい場合は小さなスクリプト単位で検証してください。
- ブランチ運用ルールは `AGENTS.md` を参照してください。
