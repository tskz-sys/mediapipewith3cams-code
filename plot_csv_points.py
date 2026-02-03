#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot CSV 3D points as a simple scatter (no skeleton connections).
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["X", "Y", "Z", "frame", "person_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["X", "Y", "Z"])
    return df


def get_axes_data(df: pd.DataFrame, raw_axes: bool):
    if raw_axes:
        return df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy(), ("X", "Y", "Z")
    y_max = float(df["Y"].max()) if not df["Y"].empty else 0.0
    return (
        df["X"].to_numpy(),
        df["Z"].to_numpy(),
        (y_max - df["Y"]).to_numpy(),
        ("X (Side)", "Z (Depth)", "Y (Height)"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot CSV points in 3D")
    parser.add_argument("--input", "-i", required=True, help="Input CSV")
    parser.add_argument("--output", "-o", required=True, help="Output PNG path")
    parser.add_argument("--frame", type=int, default=None, help="Frame index (default: all frames)")
    parser.add_argument("--raw_axes", action="store_true", help="Use raw X/Y/Z axes")
    parser.add_argument("--no_ball", action="store_true", help="Exclude ball points")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"CSV not found: {args.input}")

    df = load_csv(args.input)
    if args.frame is not None:
        df = df[df["frame"] == int(args.frame)]

    if df.empty:
        raise ValueError("No points to plot.")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ball_mask = (df["joint"] == "ball") | (df["person_id"] == -1)
    if args.no_ball:
        ball_mask = pd.Series([False] * len(df))

    people_df = df[~ball_mask]
    ball_df = df[ball_mask]

    colors = ["blue", "red", "green", "orange", "purple", "cyan"]

    for pid in sorted(people_df["person_id"].unique()):
        p_df = people_df[people_df["person_id"] == pid]
        x, y, z, labels = get_axes_data(p_df, args.raw_axes)
        ax.scatter(x, y, z, s=6, alpha=0.5, c=colors[int(pid) % len(colors)], label=f"ID:{int(pid)}")

    if not ball_df.empty:
        bx, by, bz, labels = get_axes_data(ball_df, args.raw_axes)
        ax.scatter(bx, by, bz, s=14, alpha=0.8, c="orange", label="ball")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.legend(loc="upper right")
    ax.set_title(os.path.basename(args.input))

    fig.tight_layout()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
