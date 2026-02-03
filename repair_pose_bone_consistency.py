import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32),
]

ANCHOR_JOINTS = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]


def _robust_stats(values: np.ndarray, eps: float = 1e-6, refine: bool = True) -> Tuple[float, float]:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = max(mad * 1.4826, eps)
    if refine and len(vals) >= 10:
        z = np.abs(vals - med) / sigma
        inliers = vals[z <= 2.5]
        if len(inliers) >= 5:
            med = float(np.median(inliers))
            mad = float(np.median(np.abs(inliers - med)))
            sigma = max(mad * 1.4826, eps)
    return med, sigma


def _mask_to_intervals(frames: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int]]:
    intervals = []
    start = None
    for i, is_bad in enumerate(mask):
        if is_bad and start is None:
            start = i
        elif not is_bad and start is not None:
            intervals.append((int(frames[start]), int(frames[i - 1])))
            start = None
    if start is not None:
        intervals.append((int(frames[start]), int(frames[-1])))
    return intervals


def _bidirectional_ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or alpha >= 1:
        return arr
    out = arr.copy()
    for t in range(1, len(out)):
        prev = out[t - 1]
        curr = out[t]
        mask = np.isfinite(curr).all(axis=-1) & np.isfinite(prev).all(axis=-1)
        out[t, mask] = alpha * prev[mask] + (1.0 - alpha) * curr[mask]
    for t in range(len(out) - 2, -1, -1):
        nxt = out[t + 1]
        curr = out[t]
        mask = np.isfinite(curr).all(axis=-1) & np.isfinite(nxt).all(axis=-1)
        out[t, mask] = alpha * nxt[mask] + (1.0 - alpha) * curr[mask]
    return out

def _bidirectional_ema_segmented(arr: np.ndarray, alpha: float, segments: List[Tuple[int, int]]) -> np.ndarray:
    if alpha <= 0 or alpha >= 1:
        return arr
    out = arr.copy()
    for start, end in segments:
        if end <= start:
            continue
        out[start:end + 1] = _bidirectional_ema(out[start:end + 1], alpha)
    return out


def _detect_edges(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    z_len: float,
    z_delta: float,
    max_ratio: float,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    t_len, _, dims = positions.shape
    bad_edges = np.zeros((t_len, len(edges)), dtype=bool)
    ref_lengths = {}
    for e_idx, (a, b) in enumerate(edges):
        pa = positions[:, a]
        pb = positions[:, b]
        valid = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
        if valid.sum() < 5:
            continue
        diff = pb - pa
        lens = np.linalg.norm(diff, axis=1)
        med, sigma = _robust_stats(lens)
        if not np.isfinite(med):
            continue
        ref_lengths[(a, b)] = med
        z = np.zeros_like(lens)
        z[valid] = np.abs(lens[valid] - med) / sigma
        delta = np.zeros_like(lens)
        delta[1:] = np.abs(lens[1:] - lens[:-1])
        d_med, d_sigma = _robust_stats(delta[valid])
        d_sigma = max(d_sigma, 1e-6)
        dz = np.zeros_like(lens)
        dz[valid] = np.abs(delta[valid] - d_med) / d_sigma
        bad = np.zeros_like(lens, dtype=bool)
        bad[valid] = (z[valid] > z_len) | (dz[valid] > z_delta)
        bad[valid] |= lens[valid] > med * max_ratio
        bad_edges[:, e_idx] = bad
    return bad_edges, ref_lengths


def _compute_root(positions: np.ndarray, name_to_idx: Dict[str, int]) -> np.ndarray:
    anchors = [name_to_idx[n] for n in ANCHOR_JOINTS if n in name_to_idx]
    root = np.full((positions.shape[0], positions.shape[2]), np.nan, dtype=float)
    if not anchors:
        return root
    for t in range(positions.shape[0]):
        pts = positions[t, anchors]
        mask = np.isfinite(pts).all(axis=1)
        if mask.any():
            root[t] = pts[mask].mean(axis=0)
    return root

def _detect_root_breaks(root: np.ndarray, z_thresh: float, abs_thresh: float) -> np.ndarray:
    n_frames = root.shape[0]
    breaks = np.zeros(n_frames, dtype=bool)
    if n_frames < 2:
        return breaks
    valid = np.isfinite(root).all(axis=1)
    dists = np.full(n_frames, np.nan, dtype=float)
    for i in range(1, n_frames):
        if valid[i] and valid[i - 1]:
            dists[i] = float(np.linalg.norm(root[i] - root[i - 1]))
    valid_d = np.isfinite(dists)
    if valid_d.sum() < 5:
        if abs_thresh is not None and abs_thresh > 0:
            breaks[valid_d] = dists[valid_d] > abs_thresh
        return breaks
    med, sigma = _robust_stats(dists[valid_d])
    thresh = med + z_thresh * sigma if np.isfinite(med) else float("inf")
    if abs_thresh is not None and abs_thresh > 0:
        thresh = max(thresh, abs_thresh)
    breaks[valid_d] = dists[valid_d] > thresh
    return breaks

def _build_segments(n_frames: int, breaks: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    if n_frames <= 0:
        return segments
    start = 0
    for i in range(1, n_frames):
        if breaks[i]:
            segments.append((start, i - 1))
            start = i
    segments.append((start, n_frames - 1))
    return segments


def _detect_bad_joints(
    positions: np.ndarray,
    bad_edges: np.ndarray,
    edges: List[Tuple[int, int]],
    name_to_idx: Dict[str, int],
    conf: Optional[np.ndarray],
    conf_thr: float,
    root_z: float,
    root_break_mask: Optional[np.ndarray],
) -> np.ndarray:
    t_len, j_len, _ = positions.shape
    bad = np.zeros((t_len, j_len), dtype=bool)
    bad |= ~np.isfinite(positions).all(axis=2)

    edge_map = [[] for _ in range(j_len)]
    for e_idx, (a, b) in enumerate(edges):
        edge_map[a].append(e_idx)
        edge_map[b].append(e_idx)
    for j in range(j_len):
        if edge_map[j]:
            bad[:, j] |= bad_edges[:, edge_map[j]].any(axis=1)

    if conf is not None:
        bad |= conf < conf_thr

    root = _compute_root(positions, name_to_idx)
    if root_break_mask is None:
        root_break_mask = np.zeros(t_len, dtype=bool)
    for j in range(j_len):
        joint = positions[:, j]
        valid = np.isfinite(joint).all(axis=1) & np.isfinite(root).all(axis=1)
        if valid.sum() < 10:
            continue
        dist = np.linalg.norm(joint - root, axis=1)
        med, sigma = _robust_stats(dist[valid])
        if not np.isfinite(med):
            continue
        z = np.zeros_like(dist)
        z[valid] = np.abs(dist[valid] - med) / sigma
        valid_root = valid & (~root_break_mask)
        bad[valid_root, j] |= z[valid_root] > root_z
    return bad


def _interpolate_joint(
    frames: np.ndarray,
    values: np.ndarray,
    bad_mask: np.ndarray,
    root: np.ndarray,
    offset_med: Optional[np.ndarray],
    segments: List[Tuple[int, int]],
) -> np.ndarray:
    out = values.copy()
    good_idx = np.where(~bad_mask & np.isfinite(values).all(axis=1))[0]
    if len(good_idx) == 0:
        return out
    for seg_start, seg_end in segments:
        if seg_end < seg_start:
            continue
        seg_frames = frames[seg_start:seg_end + 1]
        seg_bad = bad_mask[seg_start:seg_end + 1]
        if not seg_bad.any():
            continue
        seg_good = good_idx[(good_idx >= seg_start) & (good_idx <= seg_end)]
        for start, end in _mask_to_intervals(seg_frames, seg_bad):
            s = seg_start + int(np.where(seg_frames == start)[0][0])
            e = seg_start + int(np.where(seg_frames == end)[0][0])
            prev = seg_good[seg_good < s]
            nxt = seg_good[seg_good > e]
            i0 = prev[-1] if len(prev) else None
            i1 = nxt[0] if len(nxt) else None
            if i0 is not None and i1 is not None:
                for t in range(s, e + 1):
                    alpha = (t - i0) / max(i1 - i0, 1)
                    out[t] = (1 - alpha) * out[i0] + alpha * out[i1]
            else:
                for t in range(s, e + 1):
                    if offset_med is not None and np.isfinite(root[t]).all():
                        out[t] = root[t] + offset_med
                    elif i0 is not None:
                        out[t] = out[i0]
                    elif i1 is not None:
                        out[t] = out[i1]
    return out


def _enforce_bone_lengths(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    ref_lengths: Dict[Tuple[int, int], float],
    anchors: List[int],
    ref_positions: np.ndarray,
    max_step_ratio: float,
    n_iter: int,
    pull_alpha: float,
) -> np.ndarray:
    out = positions.copy()
    anchor_set = set(anchors)
    for _ in range(n_iter):
        for (a, b) in edges:
            L_ref = ref_lengths.get((a, b))
            if L_ref is None or not np.isfinite(L_ref):
                continue
            pa = out[a]
            pb = out[b]
            if not np.isfinite(pa).all() or not np.isfinite(pb).all():
                continue
            v = pb - pa
            L = float(np.linalg.norm(v))
            if L < 1e-9:
                continue
            diff = L - L_ref
            max_step = max_step_ratio * L_ref
            step = float(np.clip(diff, -max_step, max_step))
            u = v / L
            if a in anchor_set and b in anchor_set:
                continue
            if a in anchor_set:
                out[b] = pb - step * u
            elif b in anchor_set:
                out[a] = pa + step * u
            else:
                out[a] = pa + 0.5 * step * u
                out[b] = pb - 0.5 * step * u

        if pull_alpha < 1.0:
            mask = np.isfinite(ref_positions).all(axis=1)
            for j in range(out.shape[0]):
                if j in anchor_set:
                    continue
                if mask[j] and np.isfinite(out[j]).all():
                    out[j] = pull_alpha * out[j] + (1.0 - pull_alpha) * ref_positions[j]
    return out


def _process_person(
    df_person: pd.DataFrame,
    name_to_idx: Dict[str, int],
    edges: List[Tuple[int, int]],
    dims: int,
    conf_col: Optional[str],
    conf_thr: float,
    z_len: float,
    z_delta: float,
    root_z: float,
    max_ratio: float,
    max_step_ratio: float,
    n_iter: int,
    pull_alpha: float,
    ema_alpha: float,
    max_passes: int,
    root_jump_z: float,
    root_jump_abs: float,
):
    frames = np.array(sorted(df_person["frame"].unique()))
    f_idx = {f: i for i, f in enumerate(frames)}
    n_frames = len(frames)
    n_joints = len(name_to_idx)
    positions = np.full((n_frames, n_joints, dims), np.nan, dtype=float)
    conf = None
    if conf_col:
        conf = np.full((n_frames, n_joints), np.nan, dtype=float)

    for _, row in df_person.iterrows():
        jname = row["joint"]
        if not isinstance(jname, str) or jname not in name_to_idx:
            continue
        i = f_idx[int(row["frame"])]
        j = name_to_idx[jname]
        if dims == 2:
            positions[i, j, 0] = float(row["X"])
            positions[i, j, 1] = float(row["Y"])
        else:
            positions[i, j, 0] = float(row["X"])
            positions[i, j, 1] = float(row["Y"])
            positions[i, j, 2] = float(row["Z"])
        if conf_col:
            conf[i, j] = float(row[conf_col])

    root_initial = _compute_root(positions, name_to_idx)
    if root_jump_z is not None and root_jump_z > 0:
        break_mask = _detect_root_breaks(root_initial, root_jump_z, root_jump_abs)
    elif root_jump_abs is not None and root_jump_abs > 0:
        break_mask = _detect_root_breaks(root_initial, 0.0, root_jump_abs)
    else:
        break_mask = np.zeros(n_frames, dtype=bool)
    segments = _build_segments(n_frames, break_mask)

    logs = []
    for pass_idx in range(max_passes):
        bad_edges, ref_lengths = _detect_edges(positions, edges, z_len, z_delta, max_ratio)
        bad_joints = _detect_bad_joints(
            positions,
            bad_edges,
            edges,
            name_to_idx,
            conf,
            conf_thr,
            root_z,
            break_mask,
        )
        total_bad = int(bad_joints.any(axis=1).sum())
        if total_bad == 0:
            logs.append({"pass": pass_idx, "bad_frames": 0, "intervals": []})
            break

        intervals = []
        for j_name, j_idx in name_to_idx.items():
            mask = bad_joints[:, j_idx]
            if mask.any():
                for start, end in _mask_to_intervals(frames, mask):
                    intervals.append({"joint": j_name, "start": start, "end": end})

        logs.append({"pass": pass_idx, "bad_frames": total_bad, "intervals": intervals})

        root = _compute_root(positions, name_to_idx)
        ref_positions = positions.copy()
        for j_name, j_idx in name_to_idx.items():
            bad_mask = bad_joints[:, j_idx]
            if not bad_mask.any():
                continue
            offset_med = None
            if np.isfinite(root).all(axis=1).any():
                valid = np.isfinite(positions[:, j_idx]).all(axis=1) & np.isfinite(root).all(axis=1) & (~bad_mask)
                if valid.sum() >= 5:
                    offset_med = np.median(positions[valid, j_idx] - root[valid], axis=0)
            positions[:, j_idx] = _interpolate_joint(
                frames,
                positions[:, j_idx],
                bad_mask,
                root,
                offset_med,
                segments,
            )

        anchors = [name_to_idx[n] for n in ANCHOR_JOINTS if n in name_to_idx]
        for t in range(n_frames):
            positions[t] = _enforce_bone_lengths(
                positions[t],
                edges,
                ref_lengths,
                anchors,
                ref_positions[t],
                max_step_ratio,
                n_iter,
                pull_alpha,
            )

        positions = _bidirectional_ema_segmented(positions, ema_alpha, segments)

    return frames, positions, logs, break_mask, segments


def _select_conf_col(df: pd.DataFrame) -> Optional[str]:
    for name in ("confidence", "conf", "score", "visibility", "presence"):
        if name in df.columns:
            return name
    return None


def _build_name_to_idx(df: pd.DataFrame) -> Dict[str, int]:
    if "joint" not in df.columns:
        raise ValueError("CSV missing 'joint' column.")
    return {name: idx for idx, name in enumerate(JOINT_NAMES)}


def _edges_for_names(name_to_idx: Dict[str, int]) -> List[Tuple[int, int]]:
    max_idx = max(name_to_idx.values())
    edges = []
    for a, b in POSE_CONNECTIONS:
        if a <= max_idx and b <= max_idx:
            edges.append((a, b))
    return edges


def _write_debug_images(
    out_dir: str,
    frames: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    edges: List[Tuple[int, int]],
    pick_frames: List[int],
):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    for f in pick_frames:
        if f not in frames:
            continue
        idx = int(np.where(frames == f)[0][0])
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        for color, data in (("red", before[idx]), ("green", after[idx])):
            for a, b in edges:
                pa = data[a]
                pb = data[b]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], c=color, linewidth=1.5)
            pts = data[np.isfinite(data).all(axis=1)]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=10)
        ax.set_title(f"Frame {f} (red=before, green=after)")
        out_path = os.path.join(out_dir, f"frame_{f}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Repair skeleton jumps by bone-length consistency.")
    parser.add_argument("--input_csv", required=True, help="Input CSV")
    parser.add_argument("--output_csv", required=True, help="Output CSV")
    parser.add_argument("--log_json", required=True, help="Output log JSON")
    parser.add_argument("--conf_thr", type=float, default=0.3, help="Confidence threshold (if available)")
    parser.add_argument("--z_len", type=float, default=3.5, help="Bone length z-score threshold")
    parser.add_argument("--z_delta", type=float, default=4.0, help="Bone length delta z-score threshold")
    parser.add_argument("--root_z", type=float, default=4.0, help="Root-distance z-score threshold")
    parser.add_argument("--max_ratio", type=float, default=2.5, help="Max ratio to median length")
    parser.add_argument("--max_step_ratio", type=float, default=0.4, help="Max correction step ratio")
    parser.add_argument("--iter", type=int, default=3, help="Bone length correction iterations per frame")
    parser.add_argument("--pull_alpha", type=float, default=0.9, help="Pull-back to interpolated positions")
    parser.add_argument("--ema_alpha", type=float, default=0.2, help="Temporal smoothing alpha (0 disables)")
    parser.add_argument("--passes", type=int, default=2, help="Max detect->repair passes")
    parser.add_argument("--root_jump_z", type=float, default=4.0, help="Root jump z-score threshold (segment break)")
    parser.add_argument("--root_jump_abs", type=float, default=1.0, help="Root jump absolute threshold (segment break)")
    parser.add_argument("--debug_dir", default=None, help="Optional debug image output dir")
    parser.add_argument("--debug_frames", default=None, help="Comma list of frames to visualize")
    parser.add_argument("--output_video", default=None, help="Optional 3D video output mp4")
    parser.add_argument("--fps", type=int, default=60, help="Video FPS (if output_video set)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError("Empty CSV.")

    dims = 3 if "Z" in df.columns and df["Z"].notna().any() else 2
    conf_col = _select_conf_col(df)
    name_to_idx = _build_name_to_idx(df)
    edges = _edges_for_names(name_to_idx)

    people = sorted(df["person_id"].dropna().unique().tolist()) if "person_id" in df.columns else [0]
    out_rows = []
    log = {"input_csv": args.input_csv, "people": [], "params": vars(args)}

    debug_frames = []
    if args.debug_frames:
        for tok in args.debug_frames.split(","):
            tok = tok.strip()
            if tok:
                debug_frames.append(int(tok))

    for pid in people:
        df_p = df[df["person_id"] == pid] if "person_id" in df.columns else df
        df_pose = df_p[df_p["joint"] != "ball"].copy()
        if df_pose.empty:
            continue
        frames, repaired, logs, break_mask, segments = _process_person(
            df_pose,
            name_to_idx,
            edges,
            dims,
            conf_col,
            args.conf_thr,
            args.z_len,
            args.z_delta,
            args.root_z,
            args.max_ratio,
            args.max_step_ratio,
            args.iter,
            args.pull_alpha,
            args.ema_alpha,
            args.passes,
            args.root_jump_z,
            args.root_jump_abs,
        )

        if args.debug_dir and dims == 3 and debug_frames:
            before = np.full_like(repaired, np.nan)
            for _, row in df_pose.iterrows():
                jname = row["joint"]
                if jname not in name_to_idx:
                    continue
                i = int(np.where(frames == int(row["frame"]))[0][0])
                j = name_to_idx[jname]
                before[i, j, 0] = float(row["X"])
                before[i, j, 1] = float(row["Y"])
                before[i, j, 2] = float(row["Z"])
            _write_debug_images(args.debug_dir, frames, before, repaired, edges, debug_frames)

        for i, f in enumerate(frames):
            for jname, jidx in name_to_idx.items():
                pos = repaired[i, jidx]
                if not np.isfinite(pos).all():
                    continue
                row = {
                    "frame": int(f),
                    "person_id": int(pid),
                    "joint": jname,
                    "X": float(pos[0]),
                    "Y": float(pos[1]),
                }
                if dims == 3:
                    row["Z"] = float(pos[2])
                out_rows.append(row)

        break_frames = [int(frames[i]) for i, val in enumerate(break_mask) if val]
        seg_ranges = [(int(frames[s]), int(frames[e])) for s, e in segments]
        log["people"].append(
            {
                "person_id": int(pid),
                "break_frames": break_frames,
                "segments": seg_ranges,
                "passes": logs,
            }
        )

    df_out = pd.DataFrame(out_rows)
    df_out = df_out.sort_values(["frame", "person_id", "joint"])
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)

    log_dir = os.path.dirname(args.log_json)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(args.log_json, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=True, indent=2)

    if args.output_video:
        plot_script = os.path.join(os.path.dirname(__file__), "plot_3d_simple_compare.py")
        cmd = [
            sys.executable,
            plot_script,
            "-i",
            args.output_csv,
            "-o",
            args.output_video,
            "--cols",
            "1",
            "--fps",
            str(args.fps),
        ]
        subprocess.run(cmd, check=False)
        if not os.path.exists(args.output_video):
            raise RuntimeError(f"3D plot video not created: {args.output_video}")

    print(f"Output CSV: {args.output_csv}")
    print(f"Log JSON  : {args.log_json}")
    if args.output_video:
        print(f"3D video  : {args.output_video}")


if __name__ == "__main__":
    main()
