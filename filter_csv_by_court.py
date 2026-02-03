import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_court(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "plane" not in data or "polygon_uv" not in data:
        raise ValueError("Court JSON missing plane/polygon_uv.")
    plane = data["plane"]
    polygon = np.array(data["polygon_uv"], dtype=float)
    if polygon.shape[0] < 3:
        raise ValueError("Court polygon needs at least 3 points.")
    polygons = []
    if isinstance(data.get("polygons"), list):
        for entry in data["polygons"]:
            if not isinstance(entry, dict):
                continue
            if "plane" not in entry or "polygon_uv" not in entry:
                continue
            poly = np.array(entry["polygon_uv"], dtype=float)
            if poly.shape[0] < 3:
                continue
            polygons.append(
                {
                    "name": entry.get("name", "poly"),
                    "plane": entry["plane"],
                    "polygon_uv": poly,
                }
            )
    return data, plane, polygon, polygons


def to_uvh(points: np.ndarray, plane: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.array(plane["origin"], dtype=float)
    u_axis = np.array(plane["u"], dtype=float)
    v_axis = np.array(plane["v"], dtype=float)
    n_axis = np.array(plane["n"], dtype=float)
    rel = points - origin
    u = rel @ u_axis
    v = rel @ v_axis
    h = rel @ n_axis
    return u, v, h


def points_in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros_like(x, dtype=bool)
    n = len(poly)
    px = poly[:, 0]
    py = poly[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= intersect
        j = i
    return inside


def _distance_points_to_segment(
    px: np.ndarray,
    py: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom < 1e-12:
        return np.sqrt(wx * wx + wy * wy)
    t = (wx * vx + wy * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    dx = px - proj_x
    dy = py - proj_y
    return np.sqrt(dx * dx + dy * dy)


def min_dist_to_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    if len(poly) < 2:
        return np.full_like(px, np.inf, dtype=float)
    min_dist = np.full_like(px, np.inf, dtype=float)
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        dist = _distance_points_to_segment(px, py, poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1])
        min_dist = np.minimum(min_dist, dist)
    return min_dist


def filter_df_by_court(
    df: pd.DataFrame,
    court_json: str,
    max_outside_dist: float = None,
    height_min: float = None,
    height_max: float = None,
    court_mode: str = "union",
):
    _, plane, polygon, polygons = load_court(court_json)

    for col in ("X", "Y", "Z"):
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    coords = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    valid = np.isfinite(coords).all(axis=1)
    coords_valid = coords[valid]
    if len(coords_valid) == 0:
        raise ValueError("No valid 3D points in CSV.")

    def compute_keep(poly_uv: np.ndarray, poly_plane: dict):
        u, v, h = to_uvh(coords_valid, poly_plane)
        inside = points_in_polygon(u, v, poly_uv)
        kept_local = inside.copy()
        outside = ~inside
        outside_kept = 0
        outside_dropped = 0
        if max_outside_dist is not None and outside.any():
            dist_out = min_dist_to_polygon(u[outside], v[outside], poly_uv)
            keep_outside = dist_out <= max_outside_dist
            idx_outside = np.where(outside)[0]
            kept_local[idx_outside] = keep_outside
            outside_kept = int(keep_outside.sum())
            outside_dropped = int((~keep_outside).sum())

        if height_min is not None:
            kept_local &= h >= height_min
        if height_max is not None:
            kept_local &= h <= height_max

        stats = {"outside_kept": outside_kept, "outside_drop": outside_dropped}
        return kept_local, stats

    stats = {"polygons": []}
    if polygons:
        keep_list = []
        for entry in polygons:
            kept_local, st = compute_keep(entry["polygon_uv"], entry["plane"])
            keep_list.append(kept_local)
            stats["polygons"].append(
                {"name": entry.get("name", "poly"), "outside_kept": st["outside_kept"], "outside_drop": st["outside_drop"]}
            )
        if court_mode == "intersection":
            kept = np.logical_and.reduce(keep_list)
        else:
            kept = np.logical_or.reduce(keep_list)
    else:
        kept, st = compute_keep(polygon, plane)
        stats["polygons"].append(
            {"name": "poly", "outside_kept": st["outside_kept"], "outside_drop": st["outside_drop"]}
        )

    keep_mask = np.zeros(len(df), dtype=bool)
    keep_mask[np.where(valid)[0]] = kept
    df_out = df[keep_mask].copy()
    return df_out, stats


def main():
    parser = argparse.ArgumentParser(description="Filter 3D CSV by court polygon.")
    parser.add_argument("--input_csv", required=True, help="Input 3D CSV")
    parser.add_argument("--court_json", required=True, help="Court bounds JSON from pick_court_bounds.py")
    parser.add_argument("--output_csv", required=True, help="Output filtered CSV")
    parser.add_argument(
        "--max_outside_dist",
        type=float,
        default=None,
        help="Allow points outside polygon within this distance (meters in court plane).",
    )
    parser.add_argument("--height_min", type=float, default=None, help="Optional height min (plane-normal axis)")
    parser.add_argument("--height_max", type=float, default=None, help="Optional height max (plane-normal axis)")
    parser.add_argument(
        "--court_mode",
        choices=("union", "intersection"),
        default="union",
        help="How to combine multiple polygons in the court JSON.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df_out, stats = filter_df_by_court(
        df,
        args.court_json,
        max_outside_dist=args.max_outside_dist,
        height_min=args.height_min,
        height_max=args.height_max,
        court_mode=args.court_mode,
    )

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)

    print(f"Input rows : {len(df)}")
    print(f"Kept rows  : {len(df_out)}")
    for entry in stats["polygons"]:
        print(f"[{entry['name']}] Outside kept: {entry['outside_kept']}")
        print(f"[{entry['name']}] Outside drop: {entry['outside_drop']}")
    print(f"Output CSV : {args.output_csv}")


if __name__ == "__main__":
    main()
