import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from tqdm import tqdm

IGNORE_JOINTS = {
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
}

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    required = {"frame", "X", "Y", "Z"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"CSV missing columns: {missing}")

    for col in ["frame", "X", "Y", "Z", "person_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame", "X", "Y", "Z"])
    df["frame"] = df["frame"].astype(int)
    if "person_id" in df.columns:
        df["person_id"] = df["person_id"].fillna(0).astype(int)
    else:
        df["person_id"] = 0
    return df


def apply_view_transform(df, y_max):
    df = df.copy()
    df["Xv"] = df["X"]
    df["Yv"] = df["Z"]
    df["Zv"] = y_max - df["Y"]
    return df


def build_joint_map(df):
    if "joint" not in df.columns:
        return {}
    joint_map = {}
    for joint, x, y, z in df[["joint", "Xv", "Yv", "Zv"]].itertuples(index=False, name=None):
        if not isinstance(joint, str):
            continue
        if joint in IGNORE_JOINTS:
            continue
        if not np.isfinite([x, y, z]).all():
            continue
        joint_map[joint] = (x, y, z)
    return joint_map


def draw_skeleton(ax, joint_map, color):
    if not joint_map:
        return
    for u_idx, v_idx in POSE_CONNECTIONS:
        u_name = JOINT_NAMES[u_idx]
        v_name = JOINT_NAMES[v_idx]
        p1 = joint_map.get(u_name)
        p2 = joint_map.get(v_name)
        if p1 is None or p2 is None:
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color, linewidth=1.5)


def get_label(path):
    base = os.path.basename(path)
    if base.endswith("_smoothed.csv"):
        return base.replace("_smoothed.csv", "")
    return os.path.splitext(base)[0]


def collect_frames(dfs):
    all_frames = set()
    for df in dfs:
        all_frames.update(df["frame"].unique().tolist())
    return sorted(all_frames)


def configure_writer(fps):
    try:
        import imageio_ffmpeg

        rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return animation.FFMpegWriter(fps=fps, metadata={"artist": "codex"}, bitrate=3000)


def apply_zoom(lim, zoom):
    if zoom <= 0:
        return lim
    span = lim[1] - lim[0]
    if span <= 0:
        return lim
    center = (lim[0] + lim[1]) / 2.0
    half = span / 2.0 / zoom
    return (center - half, center + half)


def draw_axes(ax, x_lim, y_lim, z_lim, elev, azim, title):
    ax.cla()
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_xlabel("X (Side)")
    ax.set_ylabel("Y (Depth)")
    ax.set_zlabel("Z (Height)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    try:
        ax.set_box_aspect(
            (
                x_lim[1] - x_lim[0],
                y_lim[1] - y_lim[0],
                z_lim[1] - z_lim[0],
            )
        )
    except Exception:
        pass


def plot_frame(ax, df, frame, label, colors):
    frame_df = df[df["frame"] == frame]
    if frame_df.empty:
        ax.text2D(0.05, 0.9, "No data", transform=ax.transAxes)
        return

    ball_mask = (frame_df["person_id"] == -1) | (frame_df.get("joint") == "ball")
    people_df = frame_df[~ball_mask]
    ball_df = frame_df[ball_mask]

    for pid in sorted(people_df["person_id"].unique()):
        p_df = people_df[people_df["person_id"] == pid]
        if p_df.empty:
            continue
        color = colors[int(pid) % len(colors)]
        draw_skeleton(ax, build_joint_map(p_df), color)
        ax.scatter(p_df["Xv"], p_df["Yv"], p_df["Zv"], c=color, s=18)

    if not ball_df.empty:
        ax.scatter(ball_df["Xv"], ball_df["Yv"], ball_df["Zv"], c="orange", s=40, marker="o")


def main():
    parser = argparse.ArgumentParser(description="Create a simple 3D comparison video from CSVs")
    parser.add_argument("--inputs", "-i", nargs="+", required=True, help="Input CSV files")
    parser.add_argument("--output", "-o", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--elev", type=int, default=20, help="Camera elevation angle")
    parser.add_argument("--azim", type=int, default=45, help="Camera azimuth angle")
    parser.add_argument("--cols", type=int, default=3, help="Columns in the grid")
    parser.add_argument("--zoom", type=float, default=1.4, help="Zoom factor (>1 zooms in)")
    args = parser.parse_args()

    dfs_raw = [load_csv(path) for path in args.inputs]
    labels = [get_label(path) for path in args.inputs]

    frames = collect_frames(dfs_raw)
    if not frames:
        raise ValueError("No frames found in inputs.")

    y_max_global = max(df["Y"].max() for df in dfs_raw)
    dfs = [apply_view_transform(df, y_max_global) for df in dfs_raw]

    x_min = min(df["Xv"].min() for df in dfs)
    x_max = max(df["Xv"].max() for df in dfs)
    y_min = min(df["Yv"].min() for df in dfs)
    y_max = max(df["Yv"].max() for df in dfs)
    z_min = min(df["Zv"].min() for df in dfs)
    z_max = max(df["Zv"].max() for df in dfs)

    margin = 0.2
    x_lim = apply_zoom((x_min - margin, x_max + margin), args.zoom)
    y_lim = apply_zoom((y_min - margin, y_max + margin), args.zoom)
    z_lim = apply_zoom((z_min - margin, z_max + margin), args.zoom)

    cols = max(1, args.cols)
    rows = int(np.ceil(len(dfs) / cols))
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    axes = []
    for i in range(len(dfs)):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        axes.append(ax)

    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown"]
    writer = configure_writer(args.fps)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Generating video: {args.output}")
    with writer.saving(fig, args.output, dpi=110):
        for frame in tqdm(frames):
            for idx, ax in enumerate(axes):
                label = labels[idx]
                draw_axes(
                    ax,
                    x_lim=x_lim,
                    y_lim=y_lim,
                    z_lim=z_lim,
                    elev=args.elev,
                    azim=args.azim,
                    title=f"{label} Frame: {frame}",
                )
                plot_frame(ax, dfs[idx], frame, label, colors)
            writer.grab_frame()

    print("Done.")


if __name__ == "__main__":
    main()
