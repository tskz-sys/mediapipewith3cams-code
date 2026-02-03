import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

# 重心計算用
CORE_JOINTS = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
# 手先（ボールを持っているか、チェックにいっているか判定用）
HAND_JOINTS = ['left_wrist', 'right_wrist', 'left_index', 'right_index']

def get_hand_pos(df_person):
    """ 指定された人の手先座標リストを取得 """
    subset = df_person[df_person['joint'].isin(HAND_JOINTS)]
    return subset[['X', 'Y', 'Z']].values.astype(float)

def calculate_centroid(df_person, joints):
    """ 重心計算 """
    subset = df_person[df_person['joint'].isin(joints)]
    if len(subset) == 0: return None
    return subset[['X', 'Y', 'Z']].values.astype(float).mean(axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input CSV")
    parser.add_argument("--output_img", "-o", default="metrics_result.png", help="Output Image")
    parser.add_argument("--hold_thresh", type=float, default=0.4, help="Ball hold threshold (m)")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    # 数値型変換（エラー回避）
    for col in ['X', 'Y', 'Z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ボールと人
    df_ball = df[(df['joint'] == 'ball') | (df['person_id'] == -1)].copy()
    df_person = df[(df['joint'] != 'ball') & (df['person_id'] != -1)].copy()
    
    ball_frames = sorted(df_ball['frame'].unique())
    if not ball_frames:
        print("Error: No ball found.")
        return

    # 全フレーム範囲
    start_frame = ball_frames[0]
    end_frame = ball_frames[-1]
    frames = range(start_frame, end_frame + 1)
    
    # ------------------------------------------------
    # 時系列データの収集
    # ------------------------------------------------
    # data_log[frame] = {
    #    'dist_p0': float, 'dist_p1': float, 
    #    'spacing': float, 'ball_z': float
    # }
    data_log = {f: {'dist_p0': np.nan, 'dist_p1': np.nan, 'spacing': np.nan} for f in frames}

    pids = sorted(df_person['person_id'].unique()) # [0, 1]想定
    
    print("Analyzing frames...")
    for f in frames:
        # ボール位置
        row_b = df_ball[df_ball['frame'] == f]
        if row_b.empty: continue
        b_pos = row_b.iloc[0][['X', 'Y', 'Z']].values.astype(float)
        
        # --- 1. ボールと各人の手の距離 ---
        for pid in pids:
            row_p = df_person[(df_person['frame'] == f) & (df_person['person_id'] == pid)]
            h_pos = get_hand_pos(row_p)
            if len(h_pos) > 0:
                # 一番近い手との距離
                d = np.min(np.linalg.norm(h_pos - b_pos, axis=1))
                key = f'dist_p{pid}'
                data_log[f][key] = d

        # --- 2. スペーシング (重心間距離) ---
        if len(pids) >= 2:
            p0_data = df_person[(df_person['frame'] == f) & (df_person['person_id'] == pids[0])]
            p1_data = df_person[(df_person['frame'] == f) & (df_person['person_id'] == pids[1])]
            c0 = calculate_centroid(p0_data, CORE_JOINTS)
            c1 = calculate_centroid(p1_data, CORE_JOINTS)
            if c0 is not None and c1 is not None:
                spacing = np.linalg.norm(c0 - c1)
                data_log[f]['spacing'] = spacing

    # DataFrame化して扱いやすくする
    log_df = pd.DataFrame.from_dict(data_log, orient='index').sort_index()
    log_df['frame'] = log_df.index

    # ------------------------------------------------
    # 判定ロジック (前回と同じ)
    # ------------------------------------------------
    # 後半(50%以降)で、平均してボールに近い方をシューターとする
    half_idx = len(log_df) // 2
    late_df = log_df.iloc[half_idx:]
    
    avg_d0 = late_df['dist_p0'].mean()
    avg_d1 = late_df['dist_p1'].mean()
    
    # どちらもNaNならエラー
    if np.isnan(avg_d0) and np.isnan(avg_d1):
        print("Error: Could not determine shooter.")
        return

    # NaNの場合は無限大扱いして比較
    val0 = avg_d0 if not np.isnan(avg_d0) else 999
    val1 = avg_d1 if not np.isnan(avg_d1) else 999
    
    shooter_id = pids[0] if val0 < val1 else pids[1]
    defender_id = pids[1] if shooter_id == pids[0] else pids[0]
    
    print(f"Shooter: ID {shooter_id}, Defender: ID {defender_id}")

    # リリース判定: 後ろから見て「シューターの手とボールの距離」が閾値を下回った直後
    shooter_col = f'dist_p{shooter_id}'
    defender_col = f'dist_p{defender_id}'
    
    release_frame = frames[-1]
    found = False
    
    # 逆順探索
    for i in range(len(log_df) - 2, -1, -1):
        d_curr = log_df.iloc[i][shooter_col]
        d_next = log_df.iloc[i+1][shooter_col]
        
        # 「今は持ってる(<=thresh)」かつ「次は離れてる(>thresh)」
        # あるいは単純に「持っている状態」を見つけたら、その次のフレームをリリースとする簡易ロジック
        if d_curr <= args.hold_thresh:
            release_frame = log_df.iloc[i+1]['frame']
            found = True
            break
            
    print(f"Release Frame: {release_frame}")
    
    # 結果取得
    try:
        release_row = log_df[log_df['frame'] == release_frame].iloc[0]
        contest_dist = release_row[defender_col]
        spacing_at_release = release_row['spacing']
    except:
        contest_dist = np.nan
        spacing_at_release = np.nan

    # 平均スペーシング (開始〜リリース)
    until_release = log_df[log_df['frame'] <= release_frame]
    avg_spacing = until_release['spacing'].mean()

    print(f"Contest Dist: {contest_dist:.4f} m")
    print(f"Avg Spacing: {avg_spacing:.4f} m")

    # ------------------------------------------------
    # 可視化 (Matplotlib)
    # ------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. ボールとの距離グラフ
    ax1.plot(log_df['frame'], log_df[f'dist_p{pids[0]}'], label=f'ID {pids[0]} Hand-Ball', color='blue', alpha=0.7)
    if len(pids) > 1:
        ax1.plot(log_df['frame'], log_df[f'dist_p{pids[1]}'], label=f'ID {pids[1]} Hand-Ball', color='green', alpha=0.7)
    
    # 閾値ライン
    ax1.axhline(y=args.hold_thresh, color='gray', linestyle='--', alpha=0.5, label='Hold Threshold')
    
    # リリースポイント
    ax1.axvline(x=release_frame, color='red', linestyle='-', linewidth=2, label='Release')
    
    # シューターとディフェンダーの注釈
    if not np.isnan(contest_dist):
        ax1.scatter(release_frame, contest_dist, color='red', s=100, zorder=5)
        ax1.annotate(f'Contest: {contest_dist:.2f}m\n(ID {defender_id})', 
                     (release_frame, contest_dist), xytext=(10, 20), textcoords='offset points',
                     arrowprops=dict(facecolor='red', shrink=0.05))

    ax1.set_title('Distance between Hands and Ball')
    ax1.set_ylabel('Distance (m)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. スペーシンググラフ
    ax2.plot(log_df['frame'], log_df['spacing'], label='Player Spacing', color='orange', linewidth=2)
    ax2.axvline(x=release_frame, color='red', linestyle='-', linewidth=2)
    
    # 平均スペーシングのライン
    ax2.hlines(avg_spacing, log_df['frame'].min(), release_frame, colors='purple', linestyles='--', label=f'Avg: {avg_spacing:.2f}m')
    
    ax2.set_title('Spacing between Players')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Distance (m)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_img)
    print(f"Graph saved to {args.output_img}")

if __name__ == "__main__":
    main()