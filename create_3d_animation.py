import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from tqdm import tqdm
import argparse
from calc_metrics import compute_metrics, load_goal_json

# MediaPipeの接続定義
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
JOINT_MAP = {name: i for i, name in enumerate(JOINT_NAMES)}

def apply_view_transform(df, y_max, raw_axes):
    df = df.copy()
    if raw_axes:
        df['Xv'] = df['X']
        df['Yv'] = df['Y']
        df['Zv'] = df['Z']
        return df
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

def main():
    parser = argparse.ArgumentParser(description="Create 3D Animation from CSV")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV")
    parser.add_argument("--output", "-o", type=str, default="3d_animation.mp4", help="Output MP4")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--elev", type=int, default=20, help="Camera elevation angle")
    parser.add_argument("--azim", type=int, default=45, help="Camera azimuth angle")
    parser.add_argument("--show_metrics", action="store_true", help="Overlay metrics on the animation")
    parser.add_argument("--show_goal", action="store_true", help="Plot estimated goal position")
    parser.add_argument("--goal_json", type=str, default=None, help="Goal position json (manual override)")
    parser.add_argument("--hold_thresh", type=float, default=0.4, help="Distance threshold for holding ball (m)")
    parser.add_argument("--release_tail", type=int, default=0, help="Search release within the last N ball frames (0=use latter half)")
    parser.add_argument("--raw_axes", action="store_true", help="Use raw X/Y/Z axes without view transform")
    parser.add_argument("--stop_at_release", action="store_true", help="Stop animation at release frame")
    parser.add_argument("--pad_to_frames", type=int, default=None, help="Pad to this frame count by holding last frame")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    # フレーム一覧
    frames = sorted(df['frame'].unique())

    y_max_raw = df['Y'].max()

    metrics = None
    metrics_text = None
    release_frame = None
    if args.show_metrics or args.show_goal or args.stop_at_release or args.pad_to_frames:
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
        for err in metrics.get('errors', []):
            print(err)
        for warn in metrics.get('warnings', []):
            print(warn)
        if args.show_metrics:
            metrics_text = build_metrics_text(metrics)
        if metrics.get('release_frame') is not None:
            release_frame = int(metrics['release_frame'])

    goal_v = None
    if metrics and metrics.get('goal_pos') is not None:
        gx, gy, gz = metrics['goal_pos']
        if args.raw_axes:
            goal_v = (gx, gy, gz)
        else:
            goal_v = (gx, gz, y_max_raw - gy)

    df = apply_view_transform(df, y_max_raw, args.raw_axes)

    if args.stop_at_release and release_frame is not None:
        frames = [f for f in frames if f <= release_frame]
        print(f"Stop at release frame: {release_frame} (frames={len(frames)})")

    if not frames:
        print("No frames to render.")
        return

    if args.pad_to_frames is not None and args.pad_to_frames > len(frames):
        frames = frames + [frames[-1]] * (args.pad_to_frames - len(frames))
        print(f"Padded frames: {len(frames)}")

    print(f"Total frames: {len(frames)}")

    # 軸の範囲を全データの最大・最小から決定（動画がガタつかないように固定）
    x_min, x_max = df['Xv'].min(), df['Xv'].max()
    y_min, y_max = df['Yv'].min(), df['Yv'].max()
    z_min, z_max = df['Zv'].min(), df['Zv'].max()
    
    # 少しマージンを追加
    margin = 0.5
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)
    z_lim = (z_min - margin, z_max + margin) if args.raw_axes else (0, z_max + margin)

    # 描画設定
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 色設定 (IDごと)
    colors = ['blue', 'red', 'green', 'orange']

    # 動画書き出し設定 (FFmpeg)
    writer = animation.FFMpegWriter(fps=args.fps, metadata=dict(artist='Me'), bitrate=3000)

    print(f"Generating animation to {args.output} ...")
    
    with writer.saving(fig, args.output, dpi=100):
        for f in tqdm(frames):
            ax.cla() # 前のフレームを消去
            
            # 軸の設定（毎回リセットされるため再設定が必要）
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            if args.raw_axes:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            else:
                ax.set_xlabel('X (Side)')
                ax.set_ylabel('Z (Depth)')
                ax.set_zlabel('Y (Height)')
            if metrics and metrics.get('release_frame') == f:
                ax.set_title(f"Frame: {f} (Release)")
            else:
                ax.set_title(f"Frame: {f}")
            
            # カメラアングル固定
            ax.view_init(elev=args.elev, azim=args.azim)

            # 現在のフレームのデータを抽出
            current_df = df[df['frame'] == f]
            pids = current_df['person_id'].unique()

            for pid in pids:
                p_data = current_df[current_df['person_id'] == pid]
                
                # 座標辞書作成
                pts = {}
                for _, row in p_data.iterrows():
                    pts[row['joint']] = (row['Xv'], row['Yv'], row['Zv'])
                
                # 点の描画
                xs = [p[0] for p in pts.values()]
                ys = [p[1] for p in pts.values()]
                zs = [p[2] for p in pts.values()]
                c = colors[int(pid) % len(colors)]
                
                ax.scatter(xs, ys, zs, c=c, s=20, label=f"ID:{pid}")

                # 骨格線の描画
                for u_idx, v_idx in POSE_CONNECTIONS:
                    u_name = JOINT_NAMES[u_idx] if u_idx < len(JOINT_NAMES) else ""
                    v_name = JOINT_NAMES[v_idx] if v_idx < len(JOINT_NAMES) else ""
                    
                    if u_name in pts and v_name in pts:
                        p1 = pts[u_name]
                        p2 = pts[v_name]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=c, linewidth=2)
            
            if args.show_goal and goal_v is not None:
                ax.scatter([goal_v[0]], [goal_v[1]], [goal_v[2]], c='black', s=80, marker='X', label='goal')

            # 凡例
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

            # フレーム書き込み
            writer.grab_frame()

    print("Done!")

if __name__ == "__main__":
    main()
