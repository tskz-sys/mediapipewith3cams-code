import argparse
import os
import subprocess
import sys

import pandas as pd

from batch3_3dposeestimation import fix_pose_csv_adaptive_like, smooth_csv_like
from filter_csv_by_court import filter_df_by_court


def derive_stem(input_csv: str) -> str:
    base = os.path.splitext(os.path.basename(input_csv))[0]
    if base.endswith("_raw"):
        return base[:-4]
    return base


def main():
    parser = argparse.ArgumentParser(description="Filter 3D CSV by court and run post-processing.")
    parser.add_argument("--input_csv", required=True, help="Input raw 3D CSV")
    parser.add_argument("--court_json", required=True, help="Court bounds JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--suffix", default="court_union", help="Suffix for output files")
    parser.add_argument(
        "--court_mode",
        choices=("union", "intersection"),
        default="union",
        help="How to combine multiple polygons in the court JSON.",
    )
    parser.add_argument(
        "--max_outside_dist",
        type=float,
        default=0.5,
        help="Allow points outside polygon within this distance (meters in court plane).",
    )
    parser.add_argument("--height_min", type=float, default=None, help="Optional height min (plane-normal axis)")
    parser.add_argument("--height_max", type=float, default=None, help="Optional height max (plane-normal axis)")
    parser.add_argument("--base_anchor_thr", type=float, default=0.5, help="fix step: normal move tolerance (m)")
    parser.add_argument("--force_reset_frames", type=int, default=10, help="fix step: force reset after missing frames")
    parser.add_argument("--smooth_window", type=int, default=5, help="smooth step: rolling mean window")
    parser.add_argument("--interpolate_limit", type=int, default=3, help="smooth step: linear interpolate limit")
    parser.add_argument("--foot_ratio_max", type=float, default=2.2, help="smooth step: drop foot/ankle by ratio")
    parser.add_argument("--foot_abs_max", type=float, default=None, help="smooth step: drop foot/ankle by abs length")
    parser.add_argument("--smooth_ema_alpha", type=float, default=0.35, help="smooth step: bidirectional EMA alpha")
    parser.add_argument("--fps", type=int, default=60, help="Output video FPS")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df_filtered, _ = filter_df_by_court(
        df,
        args.court_json,
        max_outside_dist=args.max_outside_dist,
        height_min=args.height_min,
        height_max=args.height_max,
        court_mode=args.court_mode,
    )

    stem = derive_stem(args.input_csv)
    suffix = args.suffix
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_video_dir = os.path.join(out_dir, "video")
    os.makedirs(out_video_dir, exist_ok=True)

    out_raw = os.path.join(out_dir, f"{stem}_raw_{suffix}.csv")
    out_fixed = os.path.join(out_dir, f"{stem}_fixed_{suffix}.csv")
    out_smoothed = os.path.join(out_dir, f"{stem}_smoothed_{suffix}.csv")
    out_video = os.path.join(out_video_dir, f"{stem}_3d_{suffix}.mp4")

    df_filtered.to_csv(out_raw, index=False)

    fix_pose_csv_adaptive_like(
        out_raw,
        out_fixed,
        base_anchor_thr=args.base_anchor_thr,
        force_reset_frames=args.force_reset_frames,
    )
    smooth_csv_like(
        out_fixed,
        out_smoothed,
        window=args.smooth_window,
        interpolate_limit=args.interpolate_limit,
        foot_ratio_max=args.foot_ratio_max,
        foot_abs_max=args.foot_abs_max,
        ema_alpha=args.smooth_ema_alpha,
    )

    plot_script = os.path.join(os.path.dirname(__file__), "plot_3d_simple_compare.py")
    cmd = [
        sys.executable,
        plot_script,
        "-i",
        out_smoothed,
        "-o",
        out_video,
        "--cols",
        "1",
        "--fps",
        str(args.fps),
    ]
    subprocess.run(cmd, check=False)

    if not os.path.exists(out_video):
        raise RuntimeError(f"3D plot video not created: {out_video}")

    print(f"Filtered raw : {out_raw}")
    print(f"Fixed CSV    : {out_fixed}")
    print(f"Smoothed CSV : {out_smoothed}")
    print(f"3D video     : {out_video}")


if __name__ == "__main__":
    main()
