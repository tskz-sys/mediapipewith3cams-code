import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
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

def sanitize_df(df):
    df = df.copy()
    for col in ['X', 'Y', 'Z', 'frame', 'person_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['frame', 'person_id'])
    df['frame'] = df['frame'].astype(int)
    df['person_id'] = df['person_id'].astype(int)
    return df

def get_label(path):
    base = os.path.basename(path)
    if base.endswith('_smoothed.csv'):
        return base.replace('_smoothed.csv', '')
    return os.path.splitext(base)[0]

def main():
    parser = argparse.ArgumentParser(description="Create 3D comparison video from multiple CSVs")
    parser.add_argument("--inputs", "-i", nargs='+', required=True, help="Input CSV files")
    parser.add_argument("--output", "-o", type=str, default="comparison_3d.mp4", help="Output MP4")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--elev", type=int, default=20, help="Camera elevation angle")
    parser.add_argument("--azim", type=int, default=45, help="Camera azimuth angle")
    parser.add_argument("--hold_thresh", type=float, default=0.4, help="Distance threshold for holding ball (m)")
    parser.add_argument("--release_tail", type=int, default=0, help="Search release within the last N ball frames (0=use latter half)")
    args = parser.parse_args()

    print("Loading CSVs...")
    dfs_raw = [sanitize_df(pd.read_csv(path)) for path in args.inputs]
    if not dfs_raw:
        print("Error: No input data.")
        return

    y_max_global = max(df['Y'].max() for df in dfs_raw if not df.empty)
    dfs = [apply_view_transform(df, y_max_global) for df in dfs_raw]

    metrics_list = []
    frame_lists = []
    release_frames = []
    labels = [get_label(path) for path in args.inputs]

    for df_raw in dfs_raw:
        metrics = compute_metrics(
            df_raw,
            hold_thresh=args.hold_thresh,
            release_tail=args.release_tail,
        )
        for err in metrics.get('errors', []):
            print(err)
        for warn in metrics.get('warnings', []):
            print(warn)
        release_frame = metrics.get('release_frame')
        metrics_list.append(metrics)
        release_frames.append(release_frame)

    for df, release_frame in zip(dfs, release_frames):
        frames_all = sorted(df['frame'].unique())
        if release_frame is None:
            frame_list = frames_all
        else:
            frame_list = [f for f in frames_all if f <= release_frame]
            if not frame_list:
                frame_list = frames_all
        frame_lists.append(frame_list)

    total_frames = max(len(frames) for frames in frame_lists if frames)

    x_min = min(df['Xv'].min() for df in dfs if not df.empty)
    x_max = max(df['Xv'].max() for df in dfs if not df.empty)
    y_min = min(df['Yv'].min() for df in dfs if not df.empty)
    y_max = max(df['Yv'].max() for df in dfs if not df.empty)
    z_min = min(df['Zv'].min() for df in dfs if not df.empty)
    z_max = max(df['Zv'].max() for df in dfs if not df.empty)

    margin = 0.5
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)
    z_lim = (0, z_max + margin)

    cols = 3
    rows = int(np.ceil(len(dfs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), subplot_kw={'projection': '3d'})
    axes = np.array(axes).reshape(-1)

    colors = ['blue', 'red', 'green', 'orange']
    writer = animation.FFMpegWriter(fps=args.fps, metadata=dict(artist='Me'), bitrate=3000)

    print(f"Generating comparison video to {args.output} ...")
    with writer.saving(fig, args.output, dpi=100):
        for t in tqdm(range(total_frames)):
            for idx, ax in enumerate(axes):
                ax.cla()
                if idx >= len(dfs):
                    ax.axis('off')
                    continue

                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_zlim(z_lim)
                ax.set_xlabel('X (Side)')
                ax.set_ylabel('Z (Depth)')
                ax.set_zlabel('Y (Height)')
                ax.view_init(elev=args.elev, azim=args.azim)

                frames = frame_lists[idx]
                if not frames:
                    ax.set_title(f"{labels[idx]} (No Frames)")
                    continue

                frame_idx = frames[min(t, len(frames) - 1)]
                release_frame = release_frames[idx]
                if release_frame is not None and frame_idx == release_frame:
                    ax.set_title(f"{labels[idx]} Frame: {frame_idx} (Release)")
                else:
                    ax.set_title(f"{labels[idx]} Frame: {frame_idx}")

                current_df = dfs[idx][dfs[idx]['frame'] == frame_idx]
                pids = current_df['person_id'].unique()

                for pid in pids:
                    p_data = current_df[current_df['person_id'] == pid]
                    pts = {}
                    for _, row in p_data.iterrows():
                        pts[row['joint']] = (row['Xv'], row['Yv'], row['Zv'])

                    xs = [p[0] for p in pts.values()]
                    ys = [p[1] for p in pts.values()]
                    zs = [p[2] for p in pts.values()]
                    c = colors[int(pid) % len(colors)]
                    ax.scatter(xs, ys, zs, c=c, s=12)

                    for u_idx, v_idx in POSE_CONNECTIONS:
                        u_name = JOINT_NAMES[u_idx] if u_idx < len(JOINT_NAMES) else ""
                        v_name = JOINT_NAMES[v_idx] if v_idx < len(JOINT_NAMES) else ""
                        if u_name in pts and v_name in pts:
                            p1 = pts[u_name]
                            p2 = pts[v_name]
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=c, linewidth=1.5)

            writer.grab_frame()

    print("Done!")

if __name__ == "__main__":
    main()
