import pandas as pd
import numpy as np
import argparse
import math
import json

# 関節名の定義
# 重心計算用
CORE_JOINTS = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
# 手先（ボールを持っているか、チェックにいっているか判定用）
HAND_JOINTS = ['left_wrist', 'right_wrist', 'left_index', 'right_index']

def calculate_centroid(df_person, joints):
    """Calculate centroid for the given joints, skipping invalid rows."""
    subset = df_person[df_person['joint'].isin(joints)][['X', 'Y', 'Z']]
    arr = subset.values.astype(float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        return None
    return arr.mean(axis=0)

def get_hand_pos(df_person):
    """Return hand joint positions as a filtered ndarray."""
    subset = df_person[df_person['joint'].isin(HAND_JOINTS)][['X', 'Y', 'Z']]
    arr = subset.values.astype(float)
    return arr[np.isfinite(arr).all(axis=1)]

def estimate_goal_position(df_ball, release_frame=None, tail_frames=10, min_frames=3):
    """Estimate goal position from ball trajectory near the end."""
    if df_ball is None or df_ball.empty:
        return None, "Error: No ball data for goal detection."

    dfb = df_ball.copy()
    dfb = dfb[['frame', 'X', 'Y', 'Z']]
    dfb = dfb[np.isfinite(dfb[['X', 'Y', 'Z']].values).all(axis=1)]
    if dfb.empty:
        return None, "Error: No valid ball positions for goal detection."

    use_df = dfb
    if release_frame is not None:
        df_after = dfb[dfb['frame'] >= int(release_frame)]
        if df_after['frame'].nunique() >= min_frames:
            use_df = df_after

    frames = sorted(use_df['frame'].unique())
    if not frames:
        return None, "Error: No ball frames for goal detection."

    tail = frames[-min(tail_frames, len(frames)):]
    pts = use_df[use_df['frame'].isin(tail)][['X', 'Y', 'Z']].to_numpy(dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < min_frames:
        return None, "Error: Not enough ball frames for goal detection."

    goal_pos = np.median(pts, axis=0)
    return goal_pos, ""

def load_goal_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        return None, f"Error: Failed to read goal json: {exc}"

    goal_data = data.get('goal_pos', data) if isinstance(data, dict) else data
    if isinstance(goal_data, dict):
        keys = {k.lower(): v for k, v in goal_data.items()}
        x = keys.get('x')
        y = keys.get('y')
        z = keys.get('z')
        if x is None or y is None or z is None:
            return None, "Error: goal json missing X/Y/Z."
        arr = np.array([x, y, z], dtype=float)
    elif isinstance(goal_data, (list, tuple)) and len(goal_data) == 3:
        arr = np.array(goal_data, dtype=float)
    else:
        return None, "Error: Unsupported goal json format."

    if not np.isfinite(arr).all():
        return None, "Error: goal json contains non-finite values."
    return arr, ""

def select_primary_pids(df_person, limit=2):
    counts = df_person.groupby('person_id')['frame'].nunique()
    sorted_pids = sorted(counts.index.tolist(), key=lambda pid: (-counts.loc[pid], pid))
    return sorted_pids[:limit]

def compute_metrics(
    df,
    hold_thresh=0.4,
    late_ratio=0.5,
    release_tail=0,
    ball_gap_allow=15,
    goal_pos=None,
    goal_tail=10,
    goal_min_frames=3,
):
    metrics = {
        'avg_spacing': np.nan,
        'contest_dist': None,
        'contest_error': '',
        'release_frame': None,
        'shooter_id': None,
        'defender_id': None,
        'goal_pos': None,
        'goal_error': '',
        'offense_goal_dist': None,
        'spacing_frames': 0,
        'frame_count': 0,
        'start_frame': None,
        'errors': [],
        'warnings': [],
    }

    if df is None or df.empty:
        metrics['errors'].append("Error: Empty CSV data.")
        return metrics

    df = df.copy()
    for col in ['X', 'Y', 'Z', 'frame', 'person_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['frame', 'person_id'])
    df['frame'] = df['frame'].astype(int)
    df['person_id'] = df['person_id'].astype(int)

    df_ball = df[(df['joint'] == 'ball') | (df['person_id'] == -1)].copy()
    df_person = df[(df['joint'] != 'ball') & (df['person_id'] != -1)].copy()

    if df_person.empty:
        metrics['errors'].append("Error: No person data found.")
        return metrics

    pids = select_primary_pids(df_person, limit=2)
    if len(pids) < 2:
        metrics['warnings'].append("Warning: Less than 2 people detected. Spacing metric may be empty.")

    ball_frames = sorted(df_ball['frame'].unique())
    if not ball_frames:
        metrics['errors'].append("Error: No ball detected in the CSV.")
        return metrics

    frame_dists = []

    for f in ball_frames:
        row_b = df_ball[df_ball['frame'] == f]
        if row_b.empty:
            continue
        b_pos = row_b.iloc[0][['X', 'Y', 'Z']].values.astype(float)
        if not np.isfinite(b_pos).all():
            continue

        dist_by_pid = {}
        for pid in pids:
            row_p = df_person[(df_person['frame'] == f) & (df_person['person_id'] == pid)]
            h_positions = get_hand_pos(row_p)
            if len(h_positions) > 0:
                d = np.min(np.linalg.norm(h_positions - b_pos, axis=1))
            else:
                d = np.inf
            dist_by_pid[pid] = d

        frame_dists.append((f, dist_by_pid))

    if not frame_dists:
        metrics['errors'].append("Error: No valid ball frames for release detection.")
        return metrics

    start_search_idx = int(len(frame_dists) * late_ratio)
    if start_search_idx >= len(frame_dists):
        start_search_idx = max(len(frame_dists) - 1, 0)
    shooter_frames = frame_dists[start_search_idx:] if frame_dists else []

    min_dists = {pid: [] for pid in pids}
    for _, dist_by_pid in shooter_frames:
        for pid in pids:
            min_dists[pid].append(dist_by_pid.get(pid, np.inf))

    avg_dist = {}
    for pid in pids:
        finite = [d for d in min_dists[pid] if np.isfinite(d)]
        avg_dist[pid] = float(np.mean(finite)) if finite else np.inf

    if not avg_dist or all(np.isinf(v) for v in avg_dist.values()):
        metrics['errors'].append("Error: Could not determine shooter (no valid hand data).")
        return metrics

    shooter_id = min(avg_dist, key=avg_dist.get)
    defender_id = None
    if len(pids) > 1:
        defender_id = [p for p in pids if p != shooter_id][0]

    metrics['shooter_id'] = shooter_id
    metrics['defender_id'] = defender_id

    release_start_frame = None
    all_frames = sorted(df['frame'].unique())
    if all_frames:
        release_start_frame = all_frames[int(len(all_frames) * late_ratio)]

    if release_tail is None or release_tail <= 0:
        if release_start_frame is None:
            release_frames = frame_dists
        else:
            release_frames = [(f, dist_by_pid) for f, dist_by_pid in frame_dists if f >= release_start_frame]
            if not release_frames:
                release_frames = frame_dists
                metrics['warnings'].append("Warning: No ball frames after mid-point; using all frames.")
    else:
        release_frames = frame_dists[-min(release_tail, len(frame_dists)):]

    release_frame = None
    release_via_gap = False
    gap_allow = int(ball_gap_allow) if ball_gap_allow is not None else 0
    if gap_allow < 0:
        gap_allow = 0

    hold_frames = [f for f, dist_by_pid in release_frames if dist_by_pid.get(shooter_id, np.inf) <= hold_thresh]
    if hold_frames:
        ball_index = {f: i for i, f in enumerate(ball_frames)}
        frame_max = int(df['frame'].max())
        for f_hold in sorted(hold_frames, reverse=True):
            idx = ball_index.get(f_hold)
            if idx is None:
                continue
            next_ball = ball_frames[idx + 1] if idx + 1 < len(ball_frames) else None
            if next_ball is None:
                gap_len = frame_max - f_hold
                if gap_len >= 1:
                    release_frame = f_hold + 1
                    release_via_gap = True
                    break
            else:
                gap_len = next_ball - f_hold - 1
                if gap_len <= 0:
                    continue
                if gap_len > gap_allow:
                    release_frame = f_hold + 1
                    release_via_gap = True
                    break

    shooter_series = [(f, dist_by_pid.get(shooter_id, np.inf)) for f, dist_by_pid in release_frames]

    if release_frame is None:
        for i in range(len(shooter_series) - 1, 0, -1):
            d_prev = shooter_series[i - 1][1]
            d_curr = shooter_series[i][1]
            if d_prev <= hold_thresh and d_curr > hold_thresh:
                release_frame = shooter_series[i][0]
                break

    if release_frame is None:
        last_hold_idx = None
        for i in range(len(shooter_series) - 1, -1, -1):
            if shooter_series[i][1] <= hold_thresh:
                last_hold_idx = i
                break
        if last_hold_idx is not None and last_hold_idx < len(shooter_series) - 1:
            release_frame = shooter_series[last_hold_idx + 1][0]
        else:
            release_frame = shooter_series[-1][0]
            metrics['warnings'].append("Warning: Release not found; using last ball frame.")
    elif release_via_gap:
        metrics['warnings'].append("Warning: Release set to first missing ball frame after hold.")

    metrics['release_frame'] = int(release_frame)

    unique_frames = sorted(df_person['frame'].unique())
    metrics['start_frame'] = int(unique_frames[0])

    if metrics['release_frame'] is not None:
        target_frames = [f for f in unique_frames if f <= metrics['release_frame']]
    else:
        target_frames = unique_frames
        metrics['warnings'].append("Warning: Release frame missing; spacing uses all frames.")
    metrics['frame_count'] = len(target_frames)

    dist_list = []
    if len(pids) >= 2:
        for f in target_frames:
            persons_in_frame = df_person[df_person['frame'] == f]
            current_pids = set(persons_in_frame['person_id'].unique())
            if not all(pid in current_pids for pid in pids):
                continue

            p1_data = persons_in_frame[persons_in_frame['person_id'] == pids[0]]
            p2_data = persons_in_frame[persons_in_frame['person_id'] == pids[1]]

            c1 = calculate_centroid(p1_data, CORE_JOINTS)
            c2 = calculate_centroid(p2_data, CORE_JOINTS)

            if c1 is not None and c2 is not None:
                dist = np.linalg.norm(c1 - c2)
                dist_list.append(abs(float(dist)))

    if dist_list:
        metrics['avg_spacing'] = float(np.mean(dist_list))
        metrics['spacing_frames'] = len(dist_list)
    else:
        metrics['warnings'].append("Warning: No frames found with both players visible.")

    contest_dist = None
    contest_error = ""

    row_ball_rel = df_ball[df_ball['frame'] == metrics['release_frame']]
    if row_ball_rel.empty:
        contest_error = "Error: Ball not detected at release frame."
    elif defender_id is None:
        contest_error = "Error: Defender not detected."
    else:
        ball_pos = row_ball_rel.iloc[0][['X', 'Y', 'Z']].values.astype(float)
        if not np.isfinite(ball_pos).all():
            contest_error = "Error: Ball position invalid at release frame."
        else:
            row_def_rel = df_person[(df_person['frame'] == metrics['release_frame']) & (df_person['person_id'] == defender_id)]
            if row_def_rel.empty:
                contest_error = "Error: Defender not visible at release frame."
            else:
                def_hand_pos = get_hand_pos(row_def_rel)
                if len(def_hand_pos) == 0:
                    contest_error = "Error: Defender hands not detected at release frame."
                else:
                    dists = np.linalg.norm(def_hand_pos - ball_pos, axis=1)
                    contest_dist = float(np.min(dists))

    metrics['contest_dist'] = contest_dist
    metrics['contest_error'] = contest_error

    if goal_pos is not None:
        goal_arr = np.array(goal_pos, dtype=float)
        if not np.isfinite(goal_arr).all():
            metrics['goal_error'] = "Error: Goal position invalid."
            metrics['goal_pos'] = None
        else:
            metrics['goal_pos'] = goal_arr
    else:
        goal_pos, goal_error = estimate_goal_position(
            df_ball,
            release_frame=metrics['release_frame'],
            tail_frames=goal_tail,
            min_frames=goal_min_frames,
        )
        metrics['goal_pos'] = goal_pos
        metrics['goal_error'] = goal_error

    if goal_pos is not None and metrics['shooter_id'] is not None and metrics['release_frame'] is not None:
        row_shooter = df_person[
            (df_person['frame'] == metrics['release_frame']) &
            (df_person['person_id'] == metrics['shooter_id'])
        ]
        if row_shooter.empty:
            metrics['goal_error'] = metrics['goal_error'] or "Error: Shooter not visible at release frame."
        else:
            shooter_center = calculate_centroid(row_shooter, CORE_JOINTS)
            if shooter_center is None:
                metrics['goal_error'] = metrics['goal_error'] or "Error: Shooter center not available at release frame."
            else:
                metrics['offense_goal_dist'] = float(np.linalg.norm(shooter_center - goal_pos))

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Calculate Basketball Metrics from Pose CSV")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV file")
    # シュート判定の閾値（メートル）。手がボールからこれ以上離れたら「離れた」とみなす
    parser.add_argument("--hold_thresh", type=float, default=0.4, help="Distance threshold for holding ball (m)")
    parser.add_argument("--release_tail", type=int, default=0, help="Search release within the last N ball frames (0=use latter half)")
    parser.add_argument("--goal_tail", type=int, default=10, help="Use last N ball frames to estimate goal position")
    parser.add_argument("--goal_min_frames", type=int, default=3, help="Minimum ball frames required for goal detection")
    parser.add_argument("--goal_json", type=str, default=None, help="Goal position json (manual override)")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    goal_pos = None
    goal_err = ""
    if args.goal_json:
        goal_pos, goal_err = load_goal_json(args.goal_json)

    metrics = compute_metrics(
        df,
        hold_thresh=args.hold_thresh,
        release_tail=args.release_tail,
        goal_pos=goal_pos,
        goal_tail=args.goal_tail,
        goal_min_frames=args.goal_min_frames,
    )
    if goal_err:
        metrics['errors'].append(goal_err)

    for err in metrics['errors']:
        print(err)
    for warn in metrics['warnings']:
        print(warn)

    if metrics['release_frame'] is None:
        return

    if metrics['shooter_id'] is not None:
        print(f"Detected Shooter: ID {metrics['shooter_id']}")
    if metrics['defender_id'] is not None:
        print(f"Detected Defender: ID {metrics['defender_id']}")
    print(f"Shoot Release Frame: {metrics['release_frame']}")

    # ---------------------------------------------------------
    # 結果出力
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("      BASKETBALL METRICS REPORT")
    print("="*40)
    print(f"1. Match Duration (frames) : {metrics['frame_count']} (Start -> Frame {metrics['release_frame']})")
    if not np.isnan(metrics['avg_spacing']):
        print(f"2. Average Spacing         : {metrics['avg_spacing']:.4f} m")
    else:
        print("2. Average Spacing         : [N/A]")

    print(f"3. Shot Contest Distance   : ", end="")
    if metrics['contest_dist'] is not None:
        print(f"{metrics['contest_dist']:.4f} m")
    else:
        print(f"[N/A]\n   -> {metrics['contest_error']}")
    print(f"4. Offense-Goal Distance   : ", end="")
    if metrics['offense_goal_dist'] is not None:
        print(f"{metrics['offense_goal_dist']:.4f} m")
    else:
        print(f"[N/A]\n   -> {metrics['goal_error']}")
    if metrics['goal_pos'] is not None:
        gx, gy, gz = metrics['goal_pos']
        print(f"   Goal Pos (X,Y,Z)        : {gx:.3f}, {gy:.3f}, {gz:.3f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
