import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from calc_metrics import compute_metrics


POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (17, 19), (18, 20), (27, 29), (27, 31), (28, 30), (28, 32),
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
    'left_foot_index', 'right_foot_index',
]


def apply_view_transform(df, y_max):
    df = df.copy()
    df['Xv'] = df['X']
    df['Yv'] = df['Z']
    df['Zv'] = y_max - df['Y']
    return df


def select_nearest_frame(frames, target):
    if target in frames:
        return target
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f - target))


def load_pose_csv(path):
    df = pd.read_csv(path)
    for col in ['X', 'Y', 'Z', 'frame', 'person_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['frame'])
    df['frame'] = df['frame'].astype(int)
    if df.empty:
        raise ValueError(f"Empty CSV data: {path}")
    return df


def build_dataset(path, release_tail):
    df = load_pose_csv(path)
    frames_all = sorted(df['frame'].unique())

    metrics = compute_metrics(df, release_tail=release_tail)
    for err in metrics.get('errors', []):
        print(err)
    for warn in metrics.get('warnings', []):
        print(warn)

    release_frame = metrics.get('release_frame')
    release_missing = any("Release not found" in w for w in metrics.get('warnings', []))
    if release_missing:
        release_frame = frames_all[-1]
    elif release_frame is None:
        release_frame = frames_all[-1]

    release_frame = select_nearest_frame(frames_all, int(release_frame))
    frames = [f for f in frames_all if f <= release_frame]
    if not frames:
        frames = [release_frame]
    if frames[-1] != release_frame:
        frames.append(release_frame)

    y_max = df['Y'].max()
    df = apply_view_transform(df, y_max)

    label = os.path.basename(path)
    label = label.replace('_smoothed.csv', '').replace('.csv', '')
    return {
        'label': label,
        'df': df,
        'frames': frames,
        'release_frame': release_frame,
    }


def main():
    parser = argparse.ArgumentParser(description="Create a 3D grid animation from multiple CSVs")
    parser.add_argument("--inputs", nargs='+', required=True, help="Input CSV files")
    parser.add_argument("--output", "-o", required=True, help="Output MP4")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--elev", type=int, default=20, help="Camera elevation angle")
    parser.add_argument("--azim", type=int, default=45, help="Camera azimuth angle")
    parser.add_argument("--release_tail", type=int, default=0, help="Search release within the last N ball frames (0=use latter half)")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid")
    args = parser.parse_args()

    datasets = []
    for path in args.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input CSV not found: {path}")
        datasets.append(build_dataset(path, args.release_tail))

    if not datasets:
        raise ValueError("No datasets provided.")

    max_len = max(len(d['frames']) for d in datasets)
    cols = max(1, args.cols)
    rows = int(math.ceil(len(datasets) / cols))

    x_min = min(d['df']['Xv'].min() for d in datasets)
    x_max = max(d['df']['Xv'].max() for d in datasets)
    y_min = min(d['df']['Yv'].min() for d in datasets)
    y_max = max(d['df']['Yv'].max() for d in datasets)
    z_min = min(d['df']['Zv'].min() for d in datasets)
    z_max = max(d['df']['Zv'].max() for d in datasets)
    margin = 0.5
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)
    z_lim = (0, z_max + margin)

    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    axes = []
    for i in range(len(datasets)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        axes.append(ax)

    colors = ['blue', 'red', 'green', 'orange']
    writer = animation.FFMpegWriter(fps=args.fps, metadata=dict(artist='Me'), bitrate=3000)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Generating grid animation to {args.output} ...")

    with writer.saving(fig, args.output, dpi=100):
        for t in tqdm(range(max_len)):
            for ax, data in zip(axes, datasets):
                ax.cla()
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_zlim(z_lim)
                ax.set_xlabel('X (Side)')
                ax.set_ylabel('Z (Depth)')
                ax.set_zlabel('Y (Height)')
                ax.view_init(elev=args.elev, azim=args.azim)

                frames = data['frames']
                frame = frames[t] if t < len(frames) else frames[-1]
                title = f"{data['label']} Frame: {frame}"
                if frame == data['release_frame']:
                    title += " (Release)"
                ax.set_title(title)

                current_df = data['df'][data['df']['frame'] == frame]
                ball_df = current_df[(current_df['joint'] == 'ball') | (current_df['person_id'] == -1)]
                people_df = current_df[(current_df['joint'] != 'ball') & (current_df['person_id'] != -1)]

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
                    ax.scatter(xs, ys, zs, c=c, s=16)

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
                    ax.scatter(bx, by, bz, c='orange', s=40, marker='o')

            writer.grab_frame()

    print("Done!")


if __name__ == "__main__":
    main()
