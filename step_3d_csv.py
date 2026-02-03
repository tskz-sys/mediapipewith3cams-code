#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step through 3D CSV frames with key controls and OpenCV display.
"""

from __future__ import annotations

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["X", "Y", "Z", "frame", "person_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["X", "Y", "Z", "frame"])
    df["frame"] = df["frame"].astype(int)
    return df


def apply_view_transform(df: pd.DataFrame) -> pd.DataFrame:
    y_max = float(df["Y"].max()) if not df["Y"].empty else 0.0
    out = df.copy()
    out["Xv"] = out["X"]
    out["Yv"] = out["Z"]
    out["Zv"] = y_max - out["Y"]
    return out


class FrameRenderer:
    def __init__(self, df: pd.DataFrame, elev: int, azim: int, fig_size: tuple[float, float]):
        self.df = df
        self.elev = elev
        self.azim = azim
        self.fig = plt.figure(figsize=fig_size)
        self.ax = self.fig.add_subplot(111, projection="3d")

        x_min, x_max = df["Xv"].min(), df["Xv"].max()
        y_min, y_max = df["Yv"].min(), df["Yv"].max()
        z_min, z_max = df["Zv"].min(), df["Zv"].max()
        margin = 0.5
        self.x_lim = (x_min - margin, x_max + margin)
        self.y_lim = (y_min - margin, y_max + margin)
        self.z_lim = (0, z_max + margin)

    def render(self, frame: int) -> np.ndarray:
        self.ax.cla()
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_zlim(self.z_lim)
        self.ax.set_xlabel("X (Side)")
        self.ax.set_ylabel("Z (Depth)")
        self.ax.set_zlabel("Y (Height)")
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.ax.set_title(f"Frame: {frame}")

        current_df = self.df[self.df["frame"] == frame]
        ball_df = current_df[(current_df["joint"] == "ball") | (current_df["person_id"] == -1)]
        people_df = current_df[(current_df["joint"] != "ball") & (current_df["person_id"] != -1)]

        colors = ["blue", "red", "green", "orange", "cyan", "brown"]

        for pid in sorted(people_df["person_id"].unique()):
            p_data = people_df[people_df["person_id"] == pid]
            pts = {}
            for _, row in p_data.iterrows():
                pts[str(row["joint"])] = (row["Xv"], row["Yv"], row["Zv"])

            if not pts:
                continue

            xs = [p[0] for p in pts.values()]
            ys = [p[1] for p in pts.values()]
            zs = [p[2] for p in pts.values()]
            c = colors[int(pid) % len(colors)]
            self.ax.scatter(xs, ys, zs, c=c, s=20, label=f"ID:{int(pid)}")

            for u_idx, v_idx in POSE_CONNECTIONS:
                u_name = JOINT_NAMES[u_idx] if u_idx < len(JOINT_NAMES) else ""
                v_name = JOINT_NAMES[v_idx] if v_idx < len(JOINT_NAMES) else ""
                if u_name in pts and v_name in pts:
                    p1 = pts[u_name]
                    p2 = pts[v_name]
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=c, linewidth=2)

        if not ball_df.empty:
            bx = ball_df["Xv"].to_numpy()
            by = ball_df["Yv"].to_numpy()
            bz = ball_df["Zv"].to_numpy()
            self.ax.scatter(bx, by, bz, c="orange", s=60, marker="o", label="ball")

        self.ax.legend(loc="upper right")
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Step through 3D CSV frames with OpenCV")
    parser.add_argument("--input", "-i", required=True, help="Input CSV")
    parser.add_argument("--start", type=int, default=None, help="Start frame (default: first)")
    parser.add_argument("--elev", type=int, default=20, help="View elevation")
    parser.add_argument("--azim", type=int, default=45, help="View azimuth")
    parser.add_argument("--width", type=float, default=7.0, help="Figure width (inches)")
    parser.add_argument("--height", type=float, default=6.0, help="Figure height (inches)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"CSV not found: {args.input}")

    df = load_csv(args.input)
    if df.empty:
        raise ValueError("CSV has no valid rows.")

    df = apply_view_transform(df)
    frames = sorted(df["frame"].unique())
    if not frames:
        raise ValueError("No frames available in CSV.")

    if args.start is None:
        idx = 0
    else:
        target = int(args.start)
        idx = int(np.argmin([abs(f - target) for f in frames]))

    renderer = FrameRenderer(df, args.elev, args.azim, (args.width, args.height))

    cv2.namedWindow("CSV 3D Viewer", cv2.WINDOW_NORMAL)
    while True:
        frame = int(frames[idx])
        rgb = renderer.render(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("CSV 3D Viewer", bgr)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("d"), ord("l")) or key == 83:
            idx = min(idx + 1, len(frames) - 1)
        elif key in (ord("a"), ord("j")) or key == 81:
            idx = max(idx - 1, 0)
        elif key == ord("n"):
            idx = min(idx + 10, len(frames) - 1)
        elif key == ord("p"):
            idx = max(idx - 10, 0)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
