import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from calc_metrics import compute_metrics, load_goal_json

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (17, 19), (18, 20), (27, 29), (27, 31), (28, 30), (28, 32)
]

JOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

def apply_view_transform(df, y_max):
    df = df.copy()
    df['Xv'] = df['X']
    df['Yv'] = df['Z']
    df['Zv'] = y_max - df['Y']
    return df

def format_metric_value(value, digits=3):
    if value is None:
        return "N/A"
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        pass
    return f"{value:.{digits}f}"

def build_metrics_text(metrics):
    release_frame = metrics.get('release_frame')
    avg_spacing = metrics.get('avg_spacing')
    contest_dist = metrics.get('contest_dist')
    contest_error = metrics.get('contest_error')
    offense_goal_dist = metrics.get('offense_goal_dist')
    goal_error = metrics.get('goal_error')

    lines = [
        f"Release Frame: {release_frame if release_frame is not None else 'N/A'}",
        f"Avg Spacing: {format_metric_value(avg_spacing)} m",
        f"Contest Dist: {format_metric_value(contest_dist)} m",
        f"Offense-Goal Dist: {format_metric_value(offense_goal_dist)} m",
    ]
    if contest_error:
        lines.append(f"Contest Error: {contest_error}")
    if goal_error:
        lines.append(f"Goal Error: {goal_error}")
    return "\n".join(lines)

def select_frame(frames, target):
    if target in frames:
        return target
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f - target))

def main():
    parser = argparse.ArgumentParser(description="Create a 3D snapshot from CSV")
    parser.add_argument("--input", "-i", required=True, help="Input CSV")
    parser.add_argument("--output", "-o", default="3d_snapshot.png", help="Output PNG")
    parser.add_argument("--frame", type=int, default=None, help="Target frame (default: release frame)")
    parser.add_argument("--elev", type=int, default=20, help="Camera elevation angle")
    parser.add_argument("--azim", type=int, default=45, help="Camera azimuth angle")
    parser.add_argument("--show_metrics", action="store_true", help="Overlay metrics on the snapshot")
    parser.add_argument("--show_goal", action="store_true", help="Plot estimated goal position")
    parser.add_argument("--goal_json", type=str, default=None, help="Goal position json (manual override)")
    parser.add_argument("--hold_thresh", type=float, default=0.4, help="Distance threshold for holding ball (m)")
    parser.add_argument("--release_tail", type=int, default=0, help="Search release within the last N ball frames (0=use latter half)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    for col in ['X', 'Y', 'Z', 'frame', 'person_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['frame'])
    df['frame'] = df['frame'].astype(int)

    if df.empty:
        raise ValueError("Empty CSV data.")

    metrics = None
    metrics_text = None
    if args.show_metrics or args.show_goal:
        goal_pos = None
        if args.goal_json:
            goal_pos, goal_err = load_goal_json(args.goal_json)
            if goal_err:
                print(goal_err)
        metrics = compute_metrics(
            df,
            hold_thresh=args.hold_thresh,
            release_tail=args.release_tail,
            goal_pos=goal_pos,
        )
        if args.show_metrics:
            metrics_text = build_metrics_text(metrics)

    target_frame = args.frame
    if target_frame is None:
        if metrics and metrics.get('release_frame') is not None:
            target_frame = int(metrics['release_frame'])
        else:
            target_frame = int(df['frame'].max())

    frames = sorted(df['frame'].unique())
    target_frame = select_frame(frames, target_frame)
    if target_frame is None:
        raise ValueError("No frames available in CSV.")

    y_max_raw = df['Y'].max()

    goal_v = None
    if metrics and metrics.get('goal_pos') is not None:
        gx, gy, gz = metrics['goal_pos']
        goal_v = (gx, gz, y_max_raw - gy)

    df = apply_view_transform(df, y_max_raw)

    x_min, x_max = df['Xv'].min(), df['Xv'].max()
    y_min, y_max = df['Yv'].min(), df['Yv'].max()
    z_min, z_max = df['Zv'].min(), df['Zv'].max()
    margin = 0.5
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)
    z_lim = (0, z_max + margin)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_xlabel('X (Side)')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Height)')
    if metrics and metrics.get('release_frame') == target_frame:
        ax.set_title(f"Frame: {target_frame} (Release)")
    else:
        ax.set_title(f"Frame: {target_frame}")
    ax.view_init(elev=args.elev, azim=args.azim)

    current_df = df[df['frame'] == target_frame]
    ball_df = current_df[(current_df['joint'] == 'ball') | (current_df['person_id'] == -1)]
    people_df = current_df[(current_df['joint'] != 'ball') & (current_df['person_id'] != -1)]

    colors = ['blue', 'red', 'green', 'orange', 'cyan', 'brown']

    for pid in sorted(people_df['person_id'].unique()):
        p_data = people_df[people_df['person_id'] == pid]
        pts = {}
        for _, row in p_data.iterrows():
            pts[str(row['joint'])] = (row['Xv'], row['Yv'], row['Zv'])

        if not pts:
            continue

        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        zs = [p[2] for p in pts.values()]
        c = colors[int(pid) % len(colors)]
        ax.scatter(xs, ys, zs, c=c, s=20, label=f"ID:{int(pid)}")

        for u_idx, v_idx in POSE_CONNECTIONS:
            u_name = JOINT_NAMES[u_idx] if u_idx < len(JOINT_NAMES) else ""
            v_name = JOINT_NAMES[v_idx] if v_idx < len(JOINT_NAMES) else ""
            if u_name in pts and v_name in pts:
                p1 = pts[u_name]
                p2 = pts[v_name]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=c, linewidth=2)

    if not ball_df.empty:
        bx = ball_df['Xv'].to_numpy()
        by = ball_df['Yv'].to_numpy()
        bz = ball_df['Zv'].to_numpy()
        ax.scatter(bx, by, bz, c='orange', s=60, marker='o', label='ball')

    if args.show_goal and goal_v is not None:
        ax.scatter([goal_v[0]], [goal_v[1]], [goal_v[2]], c='black', s=80, marker='X', label='goal')

    ax.legend(loc='upper right')

    if metrics_text:
        ax.text2D(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
