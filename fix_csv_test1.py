import pandas as pd
import numpy as np

# ==========================================
# 1. 設定パラメータ
# ==========================================
INPUT_CSV = "./csv/kakunin.csv"
OUTPUT_CSV = "./csv/kakunin_cleaned.csv"

THRESHOLD_DIST = 0.4  # ジャンプ判定しきい値 (40cm)
INVALID_SKELETON_RATIO = 0.3  # 全身の30%以上が異常なら破棄
MAX_PEOPLE = 2

# アンカー計算に使用する関節名 (MediaPipe名)
ANCHOR_JOINTS = [
    'left_shoulder', 'right_shoulder', 
    'left_hip', 'right_hip'
]

# ==========================================
# 2. 補助関数の定義
# ==========================================

def get_anchor_4pts(person_df):
    """両肩・両尻の4点平均位置を計算する"""
    pts = []
    for j_name in ANCHOR_JOINTS:
        row = person_df[person_df['joint'] == j_name]
        if not row.empty:
            pt = row[['X', 'Y', 'Z']].values[0]
            if not np.isnan(pt).any():
                pts.append(pt)
    
    if len(pts) > 0:
        return np.mean(pts, axis=0)
    return None

def calc_dist(p1, p2):
    if p1 is None or p2 is None or np.isnan(p1).any() or np.isnan(p2).any():
        return 999.0
    return np.linalg.norm(p1 - p2)

# ==========================================
# 3. メイン処理
# ==========================================
df = pd.read_csv(INPUT_CSV)
frames = sorted(df['frame'].unique())
joints = df['joint'].unique()

final_rows = []
# 前フレームのtrack情報を保持
tracks = {i: {'anchor': None, 'joints': {}} for i in range(MAX_PEOPLE)}

for f in frames:
    f_df = df[df['frame'] == f]
    
    # 検出データの整理
    detections = []
    pids = [p for p in f_df['person_id'].unique() if p != -1]
    for pid in pids:
        p_df = f_df[f_df['person_id'] == pid]
        detections.append({
            'original_id': pid,
            'df': p_df,
            'anchor': get_anchor_4pts(p_df) # 4点平均アンカー
        })
    
    # ID対応付け (コスト行列)
    assignment = {i: None for i in range(MAX_PEOPLE)}
    if detections:
        costs = []
        for t_id in range(MAX_PEOPLE):
            for d_idx, det in enumerate(detections):
                d = calc_dist(tracks[t_id]['anchor'], det['anchor'])
                costs.append((t_id, d_idx, d))
        
        costs.sort(key=lambda x: x[2])
        used_det, used_track = set(), set()
        for t_id, d_idx, d in costs:
            if t_id not in used_track and d_idx not in used_det:
                # 前回の位置から40cm以内、または初回検出なら割り当て
                if d < THRESHOLD_DIST or tracks[t_id]['anchor'] is None:
                    assignment[t_id] = d_idx
                    used_det.add(d_idx)
                    used_track.add(t_id)

    # 関節判定と記録
    for t_id in range(MAX_PEOPLE):
        det_idx = assignment[t_id]
        if det_idx is not None:
            det = detections[det_idx]
            valid_count = 0
            temp_rows = []
            
            for j_name in joints:
                row = det['df'][det['df']['joint'] == j_name]
                if not row.empty:
                    curr_pos = row[['X', 'Y', 'Z']].values[0].astype(float)
                    prev_pos = tracks[t_id]['joints'].get(j_name)
                    
                    # 関節単体ジャンプ判定
                    if prev_pos is not None and calc_dist(curr_pos, prev_pos) > THRESHOLD_DIST:
                        # 異常値はNaN
                        temp_rows.append([f, t_id, j_name, np.nan, np.nan, np.nan])
                    else:
                        temp_rows.append([f, t_id, j_name, curr_pos[0], curr_pos[1], curr_pos[2]])
                        tracks[t_id]['joints'][j_name] = curr_pos
                        valid_count += 1
                else:
                    temp_rows.append([f, t_id, j_name, np.nan, np.nan, np.nan])
            
            # 全身壊れ判定
            if valid_count / len(joints) > (1 - INVALID_SKELETON_RATIO):
                final_rows.extend(temp_rows)
                tracks[t_id]['anchor'] = det['anchor']
            else:
                final_rows.extend([[f, t_id, j, np.nan, np.nan, np.nan] for j in joints])
        else:
            final_rows.extend([[f, t_id, j, np.nan, np.nan, np.nan] for j in joints])

    # ボール保存
    for _, row in f_df[f_df['person_id'] == -1].iterrows():
        final_rows.append([f, -1, row['joint'], row['X'], row['Y'], row['Z']])

# 補間処理
result_df = pd.DataFrame(final_rows, columns=['frame', 'person_id', 'joint', 'X', 'Y', 'Z'])
def interpolate_group(g):
    if g['person_id'].iloc[0] == -1: return g
    g[['X', 'Y', 'Z']] = g[['X', 'Y', 'Z']].interpolate(method='linear', limit=10, limit_direction='both')
    return g

result_df = result_df.groupby(['person_id', 'joint'], group_keys=False).apply(interpolate_group)
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"クレンジング完了: {OUTPUT_CSV}")