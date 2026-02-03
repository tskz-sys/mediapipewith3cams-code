import pandas as pd
import numpy as np

# CSVの読み込み
file_path = './csv/kakunin.csv'  # アップロードされたファイル名
df = pd.read_csv(file_path)

# 60fpsと仮定して6秒付近 (340~380フレーム) を抽出
target_frame_start = 340
target_frame_end = 380
target_joint = "right_ankle" # ズレが目立ちやすい足首などを指定

print(f"--- Frame {target_frame_start} to {target_frame_end} Analysis ({target_joint}) ---")

subset = df[(df['frame'] >= target_frame_start) & (df['frame'] <= target_frame_end) & (df['joint'] == target_joint)]

if subset.empty:
    print("指定範囲にデータがありません。フレームレートが30fpsなら 170~190 を試してください。")
else:
    # 前後のフレームとの差分（移動量）を計算して、急激なジャンプがないか確認
    subset = subset.sort_values('frame')
    diffs = np.sqrt(np.diff(subset['X'])**2 + np.diff(subset['Y'])**2 + np.diff(subset['Z'])**2)
    
    # 閾値 (例: 1フレームで0.2m以上動いたら異常)
    jump_threshold = 0.2
    
    print(f"{'Frame':<6} | {'X':<8} | {'Y':<8} | {'Z':<8} | {'Move(m)':<8}")
    print("-" * 50)
    
    prev_row = None
    for idx, row in subset.iterrows():
        move = 0.0
        if prev_row is not None:
            move = np.sqrt((row['X']-prev_row['X'])**2 + (row['Y']-prev_row['Y'])**2 + (row['Z']-prev_row['Z'])**2)
        
        mark = "!!" if move > jump_threshold else ""
        print(f"{int(row['frame']):<6} | {row['X']:.4f} | {row['Y']:.4f} | {row['Z']:.4f} | {move:.4f} {mark}")
        prev_row = row

    print("\n【診断】")
    print("1. 'Move' に '!!' が多い場合 -> マッチングミス（別人を同一人物と誤認）")
    print("2. 'Move' が安定しているのに映像でズレる場合 -> カメラ3のキャリブレーション誤差（カメラ1・2のみで計算されている）")