import pandas as pd
import numpy as np
import argparse

def suppress_joint_jumps(g, jump_thresh, neighbor_thresh, max_gap, median_window):
    if jump_thresh <= 0:
        return g, 0
    coords_df = g[['X', 'Y', 'Z']]
    coords = coords_df.to_numpy()
    frames = g.index.to_numpy()
    valid = ~np.isnan(coords).any(axis=1)
    if valid.sum() < 3:
        return g, 0

    to_drop = np.zeros(len(g), dtype=bool)
    if median_window and median_window >= 3:
        med = coords_df.rolling(window=median_window, center=True, min_periods=1).median()
        diff = coords - med.to_numpy()
        dist = np.linalg.norm(diff, axis=1)
        to_drop |= (dist > jump_thresh) & valid

    prev_idx = np.full(len(g), -1, dtype=np.int32)
    last = -1
    for i in range(len(g)):
        if valid[i]:
            last = i
        prev_idx[i] = last

    next_idx = np.full(len(g), -1, dtype=np.int32)
    last = -1
    for i in range(len(g) - 1, -1, -1):
        if valid[i]:
            last = i
        next_idx[i] = last

    for i in range(len(g)):
        if not valid[i]:
            continue
        prev_i = prev_idx[i]
        next_i = next_idx[i]
        if prev_i <= -1 or next_i <= -1:
            continue
        if prev_i == i or next_i == i:
            continue
        if (frames[i] - frames[prev_i] > max_gap) or (frames[next_i] - frames[i] > max_gap):
            continue
        d_prev = np.linalg.norm(coords[i] - coords[prev_i])
        d_next = np.linalg.norm(coords[i] - coords[next_i])
        d_neighbor = np.linalg.norm(coords[next_i] - coords[prev_i])
        if d_prev > jump_thresh and d_next > jump_thresh and d_neighbor < neighbor_thresh:
            to_drop[i] = True

    if to_drop.any():
        g.loc[to_drop, ['X', 'Y', 'Z']] = np.nan
    return g, int(to_drop.sum())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV (fixed)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output CSV (smoothed)")
    parser.add_argument("--window", "-w", type=int, default=5, help="Smoothing window size (frames)")
    parser.add_argument("--jump_thresh", type=float, default=0.0, help="Suppress joint outliers over this distance (m); 0=off")
    parser.add_argument("--neighbor_thresh", type=float, default=0.25, help="Neighbor distance threshold for jump suppression (m)")
    parser.add_argument("--max_gap", type=int, default=2, help="Max frame gap for jump suppression")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input)

    # ソート（念のため）
    df = df.sort_values(['person_id', 'joint', 'frame'])

    # 結果を格納するリスト
    smoothed_dfs = []

    # 人物・関節ごとに処理
    # (groupbyで一括処理するとメモリ効率が良い)
    print("Applying smoothing filter...")
    
    # グループ化: [person_id, joint]
    # 欠損値(NaN)があっても、rolling().mean() は計算可能（min_periodsで調整）
    
    # 1. まずインデックスを整える（フレーム抜けがある場合、飛ばして平均するとおかしくなるため）
    #    全フレームのリストを作成
    all_frames = sorted(df['frame'].unique())
    full_idx = pd.Index(all_frames, name='frame')
    total_dropped = 0

    for (pid, joint), group in df.groupby(['person_id', 'joint']):
        # フレームをインデックスに設定
        g = group.set_index('frame')
        
        # 全フレームを持つようにリインデックス（欠損行を作る）
        g = g.reindex(full_idx)
        g, dropped = suppress_joint_jumps(
            g, args.jump_thresh, args.neighbor_thresh, args.max_gap, args.window
        )
        total_dropped += dropped
        
        # 補間 (Interpolate): 小さな隙間（3フレーム以内）を埋める
        # limit=3: 3フレーム連続欠損までなら埋める。それ以上は埋めない（捏造防止）
        g[['X', 'Y', 'Z']] = g[['X', 'Y', 'Z']].interpolate(method='linear', limit=3)

        # スムージング (Rolling Mean): 前後数フレームの平均をとって滑らかにする
        # center=True: ズレを防ぐために中心をとる
        # min_periods=1: 端っこも計算する
        g[['X', 'Y', 'Z']] = g[['X', 'Y', 'Z']].rolling(window=args.window, center=True, min_periods=1).mean()

        # データ復元
        g['person_id'] = pid
        g['joint'] = joint
        g = g.reset_index() # frameを列に戻す
        
        # 元々データがなかった場所（リインデックスで増えた場所）のうち、
        # 補間で埋まらなかった（NaNのままの）行は削除する
        g = g.dropna(subset=['X', 'Y', 'Z'])
        
        smoothed_dfs.append(g)

    # 結合
    final_df = pd.concat(smoothed_dfs, ignore_index=True)
    
    # 元の順序っぽくソート
    final_df = final_df.sort_values(['frame', 'person_id', 'joint'])
    
    # 保存
    final_df.to_csv(args.output, index=False)
    if total_dropped > 0:
        print(f"Suppressed jump frames: {total_dropped}")
    print(f"Done. Smoothed data saved to {args.output}")

if __name__ == "__main__":
    main()
