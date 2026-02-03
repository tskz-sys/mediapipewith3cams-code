import pandas as pd
import numpy as np
import argparse
import math
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output CSV")
    
    # パラメータ設定
    parser.add_argument("--base_anchor_thr", type=float, default=0.5, help="通常の移動許容範囲(m)")
    parser.add_argument("--force_reset_frames", type=int, default=10, help="何フレーム見失ったら強制リセットするか")
    
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    # 必須カラムチェック
    required_cols = ['frame', 'person_id', 'joint', 'X', 'Y', 'Z']
    if not all(c in df.columns for c in required_cols):
        print("Error: Missing required columns.")
        return

    # ソート
    df = df.sort_values(['frame', 'person_id']).reset_index(drop=True)
    
    frames = sorted(df['frame'].unique())
    pids = sorted(df['person_id'].unique())
    joints = df['joint'].unique()
    
    # 出力用リスト
    fixed_rows = []

    # --- Personごとに処理 ---
    for pid in pids:
        print(f"Processing PID {pid}...")
        
        # この人の全データを抽出
        df_p = df[df['person_id'] == pid].copy()
        
        # フレームごとの辞書作成 {frame: {joint: (x,y,z)}}
        data_map = {}
        for _, row in df_p.iterrows():
            f = int(row['frame'])
            j = row['joint']
            if not np.isnan(row['X']):
                if f not in data_map: data_map[f] = {}
                data_map[f][j] = np.array([row['X'], row['Y'], row['Z']])

        # 追跡用ステート
        # アンカー: {joint: np.array([x,y,z])}
        anchors = {} 
        missing_counts = {} # ジョイントごとの連続欠損数

        # 最初の有効なフレームを探してアンカーを初期化
        start_frame_idx = 0
        for i, f in enumerate(frames):
            if f in data_map:
                anchors = data_map[f].copy()
                # 欠損カウント初期化
                for j in joints: missing_counts[j] = 0
                start_frame_idx = i
                break
        
        # データがないIDはスキップ
        if not anchors:
            continue

        # --- 時系列スキャン ---
        for i in range(len(frames)):
            f = frames[i]
            
            # まだ開始フレーム以前ならスキップ（ただし出力には含めない、またはNaNで埋める）
            if i < start_frame_idx:
                continue

            current_frame_data = data_map.get(f, {})
            
            for j in joints:
                # 入力データがあるか？
                has_input = (j in current_frame_data)
                valid_val = False
                out_val = [np.nan, np.nan, np.nan]

                if has_input:
                    raw_pos = current_frame_data[j]
                    
                    # アンカーがある場合、距離チェック
                    if j in anchors:
                        prev_pos = anchors[j]
                        dist = np.linalg.norm(raw_pos - prev_pos)
                        
                        # 条件1: 距離が閾値以内なら採用
                        if dist < args.base_anchor_thr:
                            valid_val = True
                        
                        # 条件2: 【重要】長時間見失っていた場合 (Force Reset)
                        # ずっとNaNが続いていて、やっとデータが来たなら、距離が遠くても採用する（テレポート許容）
                        elif missing_counts[j] > args.force_reset_frames:
                            valid_val = True
                            # print(f"  [Recover] PID{pid} {j} at frame {f} (Gap: {dist:.2f}m)")

                    else:
                        # アンカー未定義なら（初回など）無条件採用
                        valid_val = True

                    if valid_val:
                        out_val = raw_pos
                        anchors[j] = raw_pos # アンカー更新
                        missing_counts[j] = 0
                    else:
                        # 閾値外で、かつ最近まで見えていた -> ノイズ判定して捨てる
                        missing_counts[j] += 1
                else:
                    # 入力データ自体がない
                    missing_counts[j] += 1
                
                # 結果格納
                if not np.isnan(out_val[0]):
                    fixed_rows.append([f, pid, j, out_val[0], out_val[1], out_val[2]])

    # 書き出し
    print("Writing output csv...")
    out_df = pd.DataFrame(fixed_rows, columns=['frame', 'person_id', 'joint', 'X', 'Y', 'Z'])
    out_df.to_csv(args.output, index=False)
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()