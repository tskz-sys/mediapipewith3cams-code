import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

backend_set = False
try:
    import PyQt6  # noqa: F401
    matplotlib.use("QtAgg")
    backend_set = True
except Exception:
    pass

if not backend_set:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

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


class ManualIdEditor:
    def __init__(self, df, id_a, id_b, output_path, elev, azim):
        self.df = df
        self.id_a = id_a
        self.id_b = id_b
        self.output_path = output_path
        self.elev = elev
        self.azim = azim
        self.frames = sorted(df['frame'].unique())
        self.frame_idx = 0
        self.range_start = None
        self.edited_frames = set()
        self._updating_slider = False

        y_max = df['Y'].max()
        self.df = apply_view_transform(self.df, y_max)

        self.x_lim, self.y_lim, self.z_lim = self._compute_limits()

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.slider_ax = self.fig.add_axes([0.12, 0.04, 0.76, 0.03])
        self.slider = Slider(
            self.slider_ax,
            'Frame Index',
            0,
            max(len(self.frames) - 1, 0),
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self.on_slider)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self._redraw()

    def _compute_limits(self):
        x_min, x_max = self.df['Xv'].min(), self.df['Xv'].max()
        y_min, y_max = self.df['Yv'].min(), self.df['Yv'].max()
        z_min, z_max = self.df['Zv'].min(), self.df['Zv'].max()
        margin = 0.5
        return (
            (x_min - margin, x_max + margin),
            (y_min - margin, y_max + margin),
            (0, z_max + margin),
        )

    def _current_frame(self):
        if not self.frames:
            return None
        return self.frames[self.frame_idx]

    def _swap_mask(self, mask):
        mask_a = mask & (self.df['person_id'] == self.id_a)
        mask_b = mask & (self.df['person_id'] == self.id_b)
        if not mask_a.any() and not mask_b.any():
            print("No rows to swap for the selected range.")
            return
        self.df.loc[mask_a, 'person_id'] = self.id_b
        self.df.loc[mask_b, 'person_id'] = self.id_a

    def swap_current_frame(self):
        frame = self._current_frame()
        if frame is None:
            return
        mask = self.df['frame'] == frame
        self._swap_mask(mask)
        self.edited_frames.add(frame)
        print(f"Swapped IDs at frame {frame}.")
        self._redraw()

    def swap_range(self, start_frame, end_frame):
        if start_frame is None or end_frame is None:
            print("Range start/end is not set.")
            return
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        mask = (self.df['frame'] >= start_frame) & (self.df['frame'] <= end_frame)
        self._swap_mask(mask)
        for f in self.frames:
            if start_frame <= f <= end_frame:
                self.edited_frames.add(f)
        print(f"Swapped IDs in range {start_frame} - {end_frame}.")
        self.range_start = None
        self._redraw()

    def save(self):
        out_df = self.df.drop(columns=['Xv', 'Yv', 'Zv'], errors='ignore')
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
        print(f"Saved: {self.output_path}")

    def on_slider(self, val):
        if self._updating_slider:
            return
        idx = int(val)
        idx = max(0, min(idx, len(self.frames) - 1))
        self.frame_idx = idx
        self._redraw()

    def on_key(self, event):
        if event.key in ['right', 'd', 'l']:
            self.set_frame_idx(self.frame_idx + 1)
        elif event.key in ['left', 'a', 'j']:
            self.set_frame_idx(self.frame_idx - 1)
        elif event.key in ['n', 'pageup']:
            self.set_frame_idx(self.frame_idx + 10)
        elif event.key in ['p', 'pagedown']:
            self.set_frame_idx(self.frame_idx - 10)
        elif event.key == 's':
            self.swap_current_frame()
        elif event.key == 'm':
            self.range_start = self._current_frame()
            print(f"Range start set: {self.range_start}")
            self._redraw()
        elif event.key == 'r':
            self.swap_range(self.range_start, self._current_frame())
        elif event.key == 'w':
            self.save()
        elif event.key == 'q':
            plt.close(self.fig)

    def set_frame_idx(self, idx):
        if not self.frames:
            return
        idx = max(0, min(idx, len(self.frames) - 1))
        if idx == self.frame_idx:
            return
        self.frame_idx = idx
        self._updating_slider = True
        self.slider.set_val(idx)
        self._updating_slider = False
        self._redraw()

    def _redraw(self):
        frame = self._current_frame()
        self.ax.cla()
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_zlim(self.z_lim)
        self.ax.set_xlabel('X (Side)')
        self.ax.set_ylabel('Z (Depth)')
        self.ax.set_zlabel('Y (Height)')
        self.ax.view_init(elev=self.elev, azim=self.azim)

        status = f"Frame: {frame}"
        if frame in self.edited_frames:
            status += " [edited]"
        if self.range_start is not None:
            status += f" | range start: {self.range_start}"
        status += f" | swap {self.id_a} <-> {self.id_b}"
        self.ax.set_title(status)

        if frame is None:
            self.fig.canvas.draw_idle()
            return

        current_df = self.df[self.df['frame'] == frame]
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
            self.ax.scatter(xs, ys, zs, c=c, s=20, label=f"ID:{int(pid)}")

            for u_idx, v_idx in POSE_CONNECTIONS:
                u_name = JOINT_NAMES[u_idx] if u_idx < len(JOINT_NAMES) else ""
                v_name = JOINT_NAMES[v_idx] if v_idx < len(JOINT_NAMES) else ""
                if u_name in pts and v_name in pts:
                    p1 = pts[u_name]
                    p2 = pts[v_name]
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=c, linewidth=2)

        if not ball_df.empty:
            bx = ball_df['Xv'].to_numpy()
            by = ball_df['Yv'].to_numpy()
            bz = ball_df['Zv'].to_numpy()
            self.ax.scatter(bx, by, bz, c='orange', s=60, marker='o', label='ball')

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw_idle()


def parse_args():
    parser = argparse.ArgumentParser(description="Manual ID editor for 3D pose CSV")
    parser.add_argument("--input", "-i", required=True, help="Input CSV")
    parser.add_argument("--output", "-o", required=True, help="Output CSV")
    parser.add_argument("--id_a", type=int, default=0, help="First person_id to swap")
    parser.add_argument("--id_b", type=int, default=1, help="Second person_id to swap")
    parser.add_argument("--elev", type=int, default=20, help="Camera elevation angle")
    parser.add_argument("--azim", type=int, default=45, help="Camera azimuth angle")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    for col in ['X', 'Y', 'Z', 'frame', 'person_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['frame', 'person_id'])
    df['frame'] = df['frame'].astype(int)
    df['person_id'] = df['person_id'].astype(int)

    editor = ManualIdEditor(
        df=df,
        id_a=args.id_a,
        id_b=args.id_b,
        output_path=args.output,
        elev=args.elev,
        azim=args.azim,
    )
    print("Controls: left/right (or a/d) move frame, n/p jump 10 frames.")
    print("Controls: s swap current frame, m mark range start, r swap range.")
    print("Controls: w write CSV, q quit.")
    if "agg" in plt.get_backend().lower():
        print("Warning: Non-interactive backend detected. Set DISPLAY/WAYLAND_DISPLAY and rerun.")
    plt.show()
