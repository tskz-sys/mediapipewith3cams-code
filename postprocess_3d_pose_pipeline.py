#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess 3D pose CSV: re-ID, jump filtering, bone-length repair, smoothing, QA, and 3D viz.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
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

IGNORE_JOINTS = {
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
}

ANCHOR_JOINTS = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
HAND_JOINTS = ["left_wrist", "right_wrist", "left_index", "right_index", "left_thumb", "right_thumb"]


@dataclass
class TrackState:
    track_id: int
    pos: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None
    last_frame: Optional[int] = None
    missing: int = 0


@dataclass
class TemplateConfig:
    enabled: bool
    init_frames: int
    min_valid_joints: int
    max_bone_z: float
    dev_thresh: float
    allow_jump: bool
    long_gap_strategy: str
    allow_full_person_fill: bool


@dataclass
class TemplatePose:
    offsets: np.ndarray
    bone_median: Dict[Tuple[int, int], float]
    bone_mad: Dict[Tuple[int, int], float]
    frames_used: np.ndarray


def _looks_like_wsl_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    return path.startswith("\\\\wsl.localhost\\") or path.startswith("//wsl.localhost/")


def _convert_wsl_path(path: str) -> str:
    if not isinstance(path, str):
        return path
    if path.startswith("\\\\wsl.localhost\\"):
        m = re.match(r"^\\\\wsl\.localhost\\[^\\]+\\(.+)$", path)
        if m:
            return "/" + m.group(1).replace("\\", "/").lstrip("/")
    if path.startswith("//wsl.localhost/"):
        m = re.match(r"^//wsl\.localhost/[^/]+/(.+)$", path)
        if m:
            return "/" + m.group(1).lstrip("/")
    return path


def _normalize_path(path: Optional[str], must_exist: bool, label: str) -> Optional[str]:
    if path is None:
        return None
    fixed = _convert_wsl_path(path) if os.name == "posix" else path
    if must_exist and not os.path.exists(fixed):
        if _looks_like_wsl_path(path):
            example = _convert_wsl_path(path)
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "If you are running on WSL, pass a /home/... path.\n"
                f"Example: {example}"
            )
        raise FileNotFoundError(f"{label} not found: {fixed}")
    return fixed


def _normalize_joint_name(name: str) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text.startswith("l_") and not text.startswith("left_"):
        text = "left_" + text[2:]
    if text.startswith("r_") and not text.startswith("right_"):
        text = "right_" + text[2:]
    for side in ("left", "right"):
        if text.startswith(side) and not text.startswith(side + "_") and len(text) > len(side):
            text = side + "_" + text[len(side):]
    bases = [
        "shoulder", "elbow", "wrist", "pinky", "index", "thumb",
        "hip", "knee", "ankle", "heel", "foot_index",
        "ear", "eye", "eye_inner", "eye_outer", "mouth",
    ]
    for base in bases:
        if text == f"{base}_left":
            return f"left_{base}"
        if text == f"{base}_right":
            return f"right_{base}"
    if "footindex" in text:
        text = text.replace("footindex", "foot_index")
    return text


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


def _build_template_pose(
    frames: np.ndarray,
    positions: np.ndarray,
    name_to_idx: Dict[str, int],
    edges: List[Tuple[int, int]],
    jump_mask: Optional[np.ndarray],
    bone_z_max: Optional[np.ndarray],
    config: TemplateConfig,
) -> Optional[TemplatePose]:
    if not config.enabled:
        return None
    n_frames = positions.shape[0]
    if n_frames == 0:
        return None
    if config.init_frames <= 0:
        return None

    max_frames = min(config.init_frames, n_frames)
    valid_joint_count = np.isfinite(positions).all(axis=2).sum(axis=1)
    root = _compute_root(positions, name_to_idx)

    candidate_mask = np.zeros(n_frames, dtype=bool)
    candidate_mask[:max_frames] = True
    candidate_mask &= valid_joint_count >= config.min_valid_joints
    if jump_mask is not None and not config.allow_jump:
        candidate_mask &= ~jump_mask
    if bone_z_max is not None and np.isfinite(config.max_bone_z):
        bone_ok = np.isfinite(bone_z_max) & (bone_z_max <= config.max_bone_z)
        candidate_mask &= bone_ok

    if candidate_mask.sum() == 0:
        candidate_mask = np.zeros(n_frames, dtype=bool)
        candidate_mask[:max_frames] = True
        candidate_mask &= valid_joint_count >= config.min_valid_joints
        if jump_mask is not None and not config.allow_jump:
            candidate_mask &= ~jump_mask

    if candidate_mask.sum() == 0:
        candidate_mask = np.zeros(n_frames, dtype=bool)
        candidate_mask[:max_frames] = True
        candidate_mask &= np.isfinite(root).all(axis=1)

    if candidate_mask.sum() == 0:
        return None

    n_joints = positions.shape[1]
    dims = positions.shape[2]
    offsets = np.full((n_joints, dims), np.nan, dtype=float)
    for j in range(n_joints):
        valid = candidate_mask
        valid &= np.isfinite(root).all(axis=1)
        valid &= np.isfinite(positions[:, j]).all(axis=1)
        if valid.sum() >= 3:
            offsets[j] = np.median(positions[valid, j] - root[valid], axis=0)

    bone_median: Dict[Tuple[int, int], float] = {}
    bone_mad: Dict[Tuple[int, int], float] = {}
    for a, b in edges:
        valid = candidate_mask
        valid &= np.isfinite(positions[:, a]).all(axis=1)
        valid &= np.isfinite(positions[:, b]).all(axis=1)
        if valid.sum() < 3:
            continue
        lengths = np.linalg.norm(positions[valid, b] - positions[valid, a], axis=1)
        med = float(np.median(lengths))
        mad = float(np.median(np.abs(lengths - med)))
        bone_median[(a, b)] = med
        bone_mad[(a, b)] = mad

    return TemplatePose(
        offsets=offsets,
        bone_median=bone_median,
        bone_mad=bone_mad,
        frames_used=frames[candidate_mask],
    )


def _template_deviation_mask(
    positions: np.ndarray,
    root: np.ndarray,
    template_offsets: Optional[np.ndarray],
    dev_thresh: float,
) -> Optional[np.ndarray]:
    if template_offsets is None or dev_thresh <= 0:
        return None
    valid_root = np.isfinite(root).all(axis=1)
    valid_offsets = np.isfinite(template_offsets).all(axis=1)
    if not valid_root.any() or not valid_offsets.any():
        return None
    target = root[:, None, :] + template_offsets[None, :, :]
    dist = np.linalg.norm(positions - target, axis=2)
    valid = np.isfinite(positions).all(axis=2) & valid_root[:, None] & valid_offsets[None, :]
    return valid & (dist > dev_thresh)


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


def _detect_edges(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    z_len: float,
    z_delta: float,
    max_ratio: float,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    t_len, _, _ = positions.shape
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
    max_gap: int,
    long_gap_strategy: str,
    template_offset: Optional[np.ndarray] = None,
    template_long_gap_strategy: str = "nan",
    force_template_frames: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out = values.copy()
    interp_mask = np.zeros(len(values), dtype=bool)
    long_gap_mask = np.zeros(len(values), dtype=bool)
    template_fill_mask = np.zeros(len(values), dtype=bool)
    if force_template_frames is None:
        force_template_frames = np.zeros(len(values), dtype=bool)
    good_idx = np.where(~bad_mask & np.isfinite(values).all(axis=1))[0]
    if len(good_idx) == 0:
        return out, interp_mask, long_gap_mask, template_fill_mask
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
            gap_len = e - s + 1
            if max_gap > 0 and gap_len > max_gap:
                long_gap_mask[s:e + 1] = True
                for t in range(s, e + 1):
                    if (
                        template_offset is not None
                        and force_template_frames[t]
                        and np.isfinite(root[t]).all()
                    ):
                        out[t] = root[t] + template_offset
                        interp_mask[t] = True
                        template_fill_mask[t] = True
                if template_offset is not None and template_long_gap_strategy == "template":
                    for t in range(s, e + 1):
                        if template_fill_mask[t]:
                            continue
                        if np.isfinite(root[t]).all():
                            out[t] = root[t] + template_offset
                            interp_mask[t] = True
                            template_fill_mask[t] = True
                elif long_gap_strategy == "root" and offset_med is not None:
                    for t in range(s, e + 1):
                        if template_fill_mask[t]:
                            continue
                        if np.isfinite(root[t]).all():
                            out[t] = root[t] + offset_med
                            interp_mask[t] = True
                continue

            prev = seg_good[seg_good < s]
            nxt = seg_good[seg_good > e]
            i0 = prev[-1] if len(prev) else None
            i1 = nxt[0] if len(nxt) else None
            if i0 is not None and i1 is not None:
                for t in range(s, e + 1):
                    if (
                        template_offset is not None
                        and force_template_frames[t]
                        and np.isfinite(root[t]).all()
                    ):
                        out[t] = root[t] + template_offset
                        interp_mask[t] = True
                        template_fill_mask[t] = True
                    else:
                        alpha = (t - i0) / max(i1 - i0, 1)
                        out[t] = (1 - alpha) * out[i0] + alpha * out[i1]
                        interp_mask[t] = True
            else:
                for t in range(s, e + 1):
                    if (
                        template_offset is not None
                        and force_template_frames[t]
                        and np.isfinite(root[t]).all()
                    ):
                        out[t] = root[t] + template_offset
                        interp_mask[t] = True
                        template_fill_mask[t] = True
                    elif offset_med is not None and np.isfinite(root[t]).all():
                        out[t] = root[t] + offset_med
                        interp_mask[t] = True
                    elif template_offset is not None and np.isfinite(root[t]).all():
                        out[t] = root[t] + template_offset
                        interp_mask[t] = True
                        template_fill_mask[t] = True
                    elif i0 is not None:
                        out[t] = out[i0]
                        interp_mask[t] = True
                    elif i1 is not None:
                        out[t] = out[i1]
                        interp_mask[t] = True
    return out, interp_mask, long_gap_mask, template_fill_mask


def _final_fill_series(
    values: np.ndarray,
    max_interp_gap: int,
    edge_strategy: str,
    max_edge_gap: Optional[int],
    allow_frames: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = values.copy()
    interp_mask = np.zeros(len(values), dtype=bool)
    edge_mask = np.zeros(len(values), dtype=bool)

    valid = np.isfinite(out)
    if not valid.any():
        return out, interp_mask, edge_mask

    n = len(out)
    max_interp_gap = int(max_interp_gap) if max_interp_gap is not None else 0
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        start = i
        while i < n and not valid[i]:
            i += 1
        end = i - 1
        prev_idx = start - 1 if start > 0 and valid[start - 1] else None
        next_idx = i if i < n and valid[i] else None
        gap_len = end - start + 1
        if prev_idx is not None and next_idx is not None:
            if max_interp_gap > 0 and gap_len <= max_interp_gap:
                v0 = out[prev_idx]
                v1 = out[next_idx]
                for t in range(start, end + 1):
                    if allow_frames is not None and not allow_frames[t]:
                        continue
                    alpha = (t - prev_idx) / max(next_idx - prev_idx, 1)
                    out[t] = (1 - alpha) * v0 + alpha * v1
                    interp_mask[t] = True
        # edge gaps handled separately

    if edge_strategy == "hold":
        valid_idx = np.where(valid)[0]
        first_valid = int(valid_idx[0])
        last_valid = int(valid_idx[-1])
        lead_len = first_valid
        if lead_len > 0 and (max_edge_gap is None or max_edge_gap <= 0 or lead_len <= max_edge_gap):
            for t in range(0, first_valid):
                if allow_frames is not None and not allow_frames[t]:
                    continue
                out[t] = out[first_valid]
                edge_mask[t] = True
        tail_len = n - 1 - last_valid
        if tail_len > 0 and (max_edge_gap is None or max_edge_gap <= 0 or tail_len <= max_edge_gap):
            for t in range(last_valid + 1, n):
                if allow_frames is not None and not allow_frames[t]:
                    continue
                out[t] = out[last_valid]
                edge_mask[t] = True

    return out, interp_mask, edge_mask


def _final_fill_positions(
    positions: np.ndarray,
    max_interp_gap: int,
    edge_strategy: str,
    max_edge_gap: Optional[int],
    allow_frames: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = positions.copy()
    n_frames, n_joints, dims = out.shape
    interp_mask = np.zeros((n_frames, n_joints), dtype=bool)
    edge_mask = np.zeros((n_frames, n_joints), dtype=bool)
    for j in range(n_joints):
        for d in range(dims):
            filled, interp_f, edge_f = _final_fill_series(
                out[:, j, d],
                max_interp_gap,
                edge_strategy,
                max_edge_gap,
                allow_frames,
            )
            out[:, j, d] = filled
            interp_mask[:, j] |= interp_f
            edge_mask[:, j] |= edge_f
    return out, interp_mask, edge_mask


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


def _select_conf_col(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in ("confidence", "conf", "score", "visibility", "presence"):
        if name in cols_lower:
            return cols_lower[name]
    return None


def _build_name_to_idx() -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(JOINT_NAMES)}


def _edges_for_names(name_to_idx: Dict[str, int]) -> List[Tuple[int, int]]:
    max_idx = max(name_to_idx.values())
    edges = []
    for a, b in POSE_CONNECTIONS:
        if a <= max_idx and b <= max_idx:
            edges.append((a, b))
    return edges


def _compute_bone_zscores(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = positions.shape[0]
    z_scores = np.full((n_frames, len(edges)), np.nan, dtype=float)
    for e_idx, (a, b) in enumerate(edges):
        pa = positions[:, a]
        pb = positions[:, b]
        valid = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
        if valid.sum() < 5:
            continue
        lens = np.linalg.norm(pb - pa, axis=1)
        med, sigma = _robust_stats(lens[valid])
        if not np.isfinite(med):
            continue
        z = np.full(n_frames, np.nan, dtype=float)
        z[valid] = np.abs(lens[valid] - med) / sigma
        z_scores[:, e_idx] = z
    with np.errstate(all="ignore"):
        z_max = np.nanmax(z_scores, axis=1)
        z_mean = np.nanmean(z_scores, axis=1)
    z_max[~np.isfinite(z_max)] = np.nan
    z_mean[~np.isfinite(z_mean)] = np.nan
    return z_max, z_mean


def _compute_root_speed(frames: np.ndarray, root: np.ndarray) -> np.ndarray:
    speeds = np.full(root.shape[0], np.nan, dtype=float)
    for i in range(1, root.shape[0]):
        if not (np.isfinite(root[i]).all() and np.isfinite(root[i - 1]).all()):
            continue
        dt = max(int(frames[i] - frames[i - 1]), 1)
        speeds[i] = float(np.linalg.norm(root[i] - root[i - 1])) / dt
    return speeds


def _detect_root_jumps(
    frames: np.ndarray,
    root: np.ndarray,
    z_thresh: float,
    abs_thresh: float,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    speeds = _compute_root_speed(frames, root)
    jump_mask = np.zeros(root.shape[0], dtype=bool)
    reasons = [""] * root.shape[0]
    valid = np.isfinite(speeds)
    if valid.sum() < 5:
        if abs_thresh is not None and abs_thresh > 0:
            mask = valid & (speeds > abs_thresh)
            jump_mask[mask] = True
            for idx in np.where(mask)[0]:
                reasons[idx] = "abs"
        return jump_mask, reasons, speeds

    med, sigma = _robust_stats(speeds[valid])
    z_thresh_val = med + z_thresh * sigma if np.isfinite(med) else float("inf")
    for i in np.where(valid)[0]:
        triggered = []
        if speeds[i] > z_thresh_val:
            triggered.append("mad")
        if abs_thresh is not None and abs_thresh > 0 and speeds[i] > abs_thresh:
            triggered.append("abs")
        if triggered:
            jump_mask[i] = True
            reasons[i] = "+".join(triggered)
    return jump_mask, reasons, speeds


def _build_lower_map(columns: List[str]) -> Dict[str, str]:
    lower_map: Dict[str, str] = {}
    for col in columns:
        key = col.lower()
        if key not in lower_map:
            lower_map[key] = col
    return lower_map


def _find_column(columns: List[str], name: str) -> Optional[str]:
    lower_map = _build_lower_map(columns)
    return lower_map.get(name)


def _detect_schema(columns: List[str]) -> Tuple[str, Dict[str, set[str]]]:
    lower_map = _build_lower_map(columns)
    has_joint = "joint" in lower_map
    wide_axes: Dict[str, set[str]] = {}
    for col in columns:
        lower = col.lower().strip()
        m = re.match(r"^(.+)[_ ]([xyz])$", lower)
        if not m:
            continue
        base = m.group(1).strip().strip("_")
        if base in ("frame", "person_id", "personid", "id", "conf", "confidence", "score", "visibility", "presence"):
            continue
        wide_axes.setdefault(base, set()).add(m.group(2))
    has_wide = any("x" in axes and "y" in axes for axes in wide_axes.values())
    if has_joint:
        return "long", wide_axes
    if has_wide:
        return "wide", wide_axes
    return "unknown", wide_axes


def _resolve_requested_columns(columns: List[str], requested: List[str], label: str) -> List[str]:
    lower_map = _build_lower_map(columns)
    resolved = []
    missing = []
    for name in requested:
        key = name.lower().strip()
        col = lower_map.get(key)
        if col is None:
            missing.append(name)
        else:
            resolved.append(col)
    if missing:
        raise ValueError(f"{label} columns missing: {', '.join(missing)}")
    return resolved


def _wide_to_long(
    df: pd.DataFrame,
    frame_col: str,
    person_col: Optional[str],
    wide_axes: Dict[str, set[str]],
    ball_columns: Optional[List[str]],
    ball_joint_name: Optional[str],
    ball_person_id: Optional[int],
) -> Tuple[pd.DataFrame, int, List[str]]:
    allowed_joints = set(JOINT_NAMES)
    if ball_joint_name:
        allowed_joints.add(ball_joint_name)

    joint_map: Dict[str, Dict[str, str]] = {}
    unknown = set()
    dims = 2

    ball_bases: set[str] = set()
    if ball_columns:
        for col in ball_columns:
            lower = col.lower().strip()
            m = re.match(r"^(.+)[_ ]([xyz])$", lower)
            if m:
                ball_bases.add(m.group(1).strip().strip("_"))

    for base, axes in wide_axes.items():
        if base in ball_bases:
            continue
        if "x" not in axes or "y" not in axes:
            continue
        norm = _normalize_joint_name(base)
        if not norm:
            continue
        if norm not in allowed_joints:
            unknown.add(norm)
            continue
        joint_map.setdefault(norm, {})
        for axis in ("x", "y", "z"):
            axis_col = _find_column(df.columns.tolist(), f"{base}_{axis}") or _find_column(df.columns.tolist(), f"{base} {axis}")
            if axis_col:
                joint_map[norm][axis] = axis_col
        if "z" in joint_map[norm]:
            dims = 3

    long_dfs = []
    for joint_name, axis_cols in joint_map.items():
        data = {
            "frame": df[frame_col],
            "person_id": df[person_col] if person_col else 0,
            "joint": joint_name,
            "X": df[axis_cols.get("x")],
            "Y": df[axis_cols.get("y")],
        }
        if "z" in axis_cols:
            data["Z"] = df[axis_cols["z"]]
        else:
            data["Z"] = np.nan
        long_dfs.append(pd.DataFrame(data))

    if ball_columns:
        if len(ball_columns) >= 2:
            data = {
                "frame": df[frame_col],
                "person_id": ball_person_id if ball_person_id is not None else -1,
                "joint": ball_joint_name or "ball",
                "X": df[ball_columns[0]],
                "Y": df[ball_columns[1]],
            }
            if len(ball_columns) >= 3:
                data["Z"] = df[ball_columns[2]]
                dims = 3
            else:
                data["Z"] = np.nan
            long_dfs.append(pd.DataFrame(data))

    if not long_dfs:
        raise ValueError("Wide format detected, but no valid joint columns were found.")
    out = pd.concat(long_dfs, ignore_index=True)
    return out, dims, sorted(unknown)


def _load_pose_csv(
    path: str,
    ball_joint_name: Optional[str],
    ball_columns: Optional[str],
    ball_person_id: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Empty CSV.")

    schema, wide_axes = _detect_schema(df.columns.tolist())
    schema_info: Dict[str, object] = {"format": schema, "columns": df.columns.tolist()}

    if schema == "long":
        frame_col = _find_column(df.columns.tolist(), "frame")
        person_col = _find_column(df.columns.tolist(), "person_id")
        joint_col = _find_column(df.columns.tolist(), "joint")
        x_col = _find_column(df.columns.tolist(), "x")
        y_col = _find_column(df.columns.tolist(), "y")
        z_col = _find_column(df.columns.tolist(), "z")
        conf_col = _select_conf_col(df)

        missing = [name for name, col in (("frame", frame_col), ("person_id", person_col), ("joint", joint_col), ("x", x_col), ("y", y_col)) if col is None]
        if missing:
            raise ValueError(
                "CSV schema detected: long. Missing columns: "
                + ", ".join(missing)
                + f". Found columns: {', '.join(df.columns)}"
            )
        cols = [frame_col, person_col, joint_col, x_col, y_col]
        if z_col:
            cols.append(z_col)
        if conf_col:
            cols.append(conf_col)
        df = df[cols].copy()
        df = df.rename(
            columns={
                frame_col: "frame",
                person_col: "person_id",
                joint_col: "joint",
                x_col: "X",
                y_col: "Y",
                z_col: "Z",
            }
        )
        df["joint"] = df["joint"].apply(_normalize_joint_name)
        allowed = set(JOINT_NAMES)
        if ball_joint_name:
            allowed.add(ball_joint_name)
        unknown = sorted(set(j for j in df["joint"].dropna().unique() if j not in allowed))
        if unknown:
            df = df[df["joint"].isin(allowed)]
        schema_info["unknown_joints"] = unknown
        schema_info["dims"] = 3 if "Z" in df.columns and df["Z"].notna().any() else 2
    elif schema == "wide":
        frame_col = _find_column(df.columns.tolist(), "frame")
        person_col = _find_column(df.columns.tolist(), "person_id")
        if frame_col is None:
            raise ValueError(
                "CSV schema detected: wide. Missing required column: frame. "
                f"Found columns: {', '.join(df.columns)}"
            )
        resolved_ball_cols = None
        if ball_columns:
            requested = [c.strip() for c in ball_columns.split(",") if c.strip()]
            resolved_ball_cols = _resolve_requested_columns(df.columns.tolist(), requested, "ball_columns")
        df, dims, unknown = _wide_to_long(
            df,
            frame_col=frame_col,
            person_col=person_col,
            wide_axes=wide_axes,
            ball_columns=resolved_ball_cols,
            ball_joint_name=ball_joint_name,
            ball_person_id=ball_person_id,
        )
        schema_info["unknown_joints"] = unknown
        schema_info["dims"] = dims
    else:
        raise ValueError(
            "CSV schema detection failed. "
            "Expected long format (frame/person_id/joint/x/y[/z]) or wide format (joint_x/joint_y[/joint_z]). "
            f"Found columns: {', '.join(df.columns)}"
        )

    for col in ["frame", "person_id", "X", "Y", "Z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "person_id" not in df.columns:
        df["person_id"] = 0
    df = df.dropna(subset=["frame", "person_id"])
    df["frame"] = df["frame"].astype(int)
    df["person_id"] = df["person_id"].astype(int)
    return df, schema_info


def _build_joint_map(df_rows: pd.DataFrame, dims: int) -> Dict[str, np.ndarray]:
    joints: Dict[str, np.ndarray] = {}
    cols = ["joint", "X", "Y"] if dims == 2 else ["joint", "X", "Y", "Z"]
    for values in df_rows[cols].itertuples(index=False, name=None):
        joint = values[0]
        if not isinstance(joint, str):
            continue
        if dims == 2:
            _, x, y = values
            pos = np.array([x, y], dtype=float)
        else:
            _, x, y, z = values
            pos = np.array([x, y, z], dtype=float)
        if not np.isfinite(pos).all():
            continue
        joints[joint] = pos
    return joints


def _compute_root_from_joints(joints: Dict[str, np.ndarray], dims: int) -> Optional[np.ndarray]:
    pts = [joints[name] for name in ANCHOR_JOINTS if name in joints]
    if not pts:
        if joints:
            vals = np.stack(list(joints.values()), axis=0)
            return np.mean(vals, axis=0)
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def _compute_hand_distance(joints: Dict[str, np.ndarray], ball_pos: np.ndarray) -> Optional[float]:
    dists = []
    for name in HAND_JOINTS:
        pos = joints.get(name)
        if pos is None:
            continue
        if not np.isfinite(pos).all() or not np.isfinite(ball_pos).all():
            continue
        dists.append(float(np.linalg.norm(pos - ball_pos)))
    if not dists:
        return None
    return float(min(dists))


def _predict_track_position(track: TrackState, frame: int, dims: int) -> Optional[np.ndarray]:
    if track.pos is None or track.last_frame is None:
        return None
    if track.vel is None:
        return track.pos
    dt = max(frame - track.last_frame, 1)
    return track.pos + track.vel * dt


def _compute_cost_matrix(
    tracks: List[TrackState],
    detections: List[Dict[str, object]],
    frame: int,
    dims: int,
    max_dist: Optional[float],
) -> np.ndarray:
    costs = np.full((len(tracks), len(detections)), np.inf, dtype=float)
    for t_idx, track in enumerate(tracks):
        pred = _predict_track_position(track, frame, dims)
        for d_idx, det in enumerate(detections):
            root = det["root"]
            if root is None:
                continue
            if pred is None:
                dist = float(max_dist) if max_dist is not None and max_dist > 0 else 1e3
            else:
                dist = float(np.linalg.norm(root - pred))
            if max_dist is not None and max_dist > 0 and dist > max_dist:
                continue
            costs[t_idx, d_idx] = dist
    return costs


def _assign_tracks(
    tracks: List[TrackState],
    detections: List[Dict[str, object]],
    frame: int,
    dims: int,
    max_dist: Optional[float],
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    if not detections:
        return [], np.empty((len(tracks), 0), dtype=float)
    if all(t.pos is None for t in tracks):
        dets_sorted = sorted(detections, key=lambda d: int(d["orig_pid"]))
        assignments = []
        for det, track in zip(dets_sorted, tracks):
            assignments.append((detections.index(det), track.track_id))
        costs = _compute_cost_matrix(tracks, detections, frame, dims, max_dist)
        return assignments, costs

    costs = _compute_cost_matrix(tracks, detections, frame, dims, max_dist)

    if len(detections) == 1:
        d_idx = 0
        t_idx = int(np.nanargmin(costs[:, d_idx]))
        if not np.isfinite(costs[t_idx, d_idx]):
            t_idx = int(np.nanargmin([t.missing for t in tracks]))
        return [(d_idx, tracks[t_idx].track_id)], costs

    if len(tracks) == 1:
        t_idx = 0
        d_idx = int(np.nanargmin(costs[t_idx]))
        return [(d_idx, tracks[t_idx].track_id)], costs

    assign_a = costs[0, 0] + costs[1, 1]
    assign_b = costs[0, 1] + costs[1, 0]
    if np.isfinite(assign_a) and (assign_a <= assign_b or not np.isfinite(assign_b)):
        return [(0, tracks[0].track_id), (1, tracks[1].track_id)], costs
    if np.isfinite(assign_b):
        return [(0, tracks[1].track_id), (1, tracks[0].track_id)], costs

    dets_sorted = sorted(range(len(detections)), key=lambda idx: int(detections[idx]["orig_pid"]))
    return [(dets_sorted[0], tracks[0].track_id), (dets_sorted[1], tracks[1].track_id)], costs


def _update_track(track: TrackState, root: np.ndarray, frame: int) -> None:
    if track.pos is not None and track.last_frame is not None:
        dt = max(frame - track.last_frame, 1)
        track.vel = (root - track.pos) / dt
    else:
        track.vel = np.zeros_like(root)
    track.pos = root
    track.last_frame = frame
    track.missing = 0


def _reid_tracks(
    df_people: pd.DataFrame,
    df_ball: pd.DataFrame,
    frames: np.ndarray,
    dims: int,
    max_people: int,
    use_ball: bool,
    ball_dist_max: float,
    ball_dist_margin: float,
    ball_hold_frames: int,
    assign_dist_max: Optional[float],
) -> Tuple[
    Dict[int, Dict[int, int]],
    Dict[int, Optional[int]],
    Dict[int, bool],
    List[Dict[str, object]],
]:
    tracks = [TrackState(0), TrackState(1)]
    frame_map: Dict[int, Dict[int, int]] = {}
    offense_track_by_frame: Dict[int, Optional[int]] = {}
    role_change_by_frame: Dict[int, bool] = {}
    role_change_events: List[Dict[str, object]] = []

    ball_by_frame: Dict[int, np.ndarray] = {}
    if not df_ball.empty:
        for frame, group in df_ball.groupby("frame"):
            cols = ["X", "Y"] if dims == 2 else ["X", "Y", "Z"]
            vals = group[cols].to_numpy(dtype=float)
            if np.isfinite(vals).all():
                ball_by_frame[int(frame)] = np.mean(vals, axis=0)

    offense_track: Optional[int] = None
    switch_streak = 0

    for frame in frames:
        frame_rows = df_people[df_people["frame"] == frame]
        detections = []
        for pid, group in frame_rows.groupby("person_id"):
            joints = _build_joint_map(group, dims)
            root = _compute_root_from_joints(joints, dims)
            if root is None or not np.isfinite(root).all():
                continue
            valid_joint_count = len(joints)
            hand_dist = None
            ball_pos = ball_by_frame.get(int(frame))
            if use_ball and ball_pos is not None:
                hand_dist = _compute_hand_distance(joints, ball_pos)
            detections.append(
                {
                    "orig_pid": int(pid),
                    "root": root,
                    "valid_joint_count": valid_joint_count,
                    "hand_dist": hand_dist,
                }
            )

        if not detections:
            for track in tracks:
                track.missing += 1
            offense_track_by_frame[frame] = offense_track
            role_change_by_frame[frame] = False
            continue

        detections.sort(key=lambda d: d["valid_joint_count"], reverse=True)
        detections = detections[:max_people]

        assignments, costs = _assign_tracks(tracks, detections, frame, dims, assign_dist_max)
        mapping: Dict[int, int] = {}
        det_by_track: Dict[int, Dict[str, object]] = {}
        for det_idx, track_id in assignments:
            det = detections[det_idx]
            mapping[int(det["orig_pid"])] = track_id
            det_by_track[track_id] = det

        for track in tracks:
            det = det_by_track.get(track.track_id)
            if det is None:
                track.missing += 1
            else:
                _update_track(track, det["root"], frame)

        frame_map[frame] = mapping

        candidate_track = None
        dist_by_track: Dict[int, float] = {}
        if use_ball:
            dist_list = []
            for track_id, det in det_by_track.items():
                hand_dist = det.get("hand_dist")
                if hand_dist is None or not np.isfinite(hand_dist):
                    continue
                dist_list.append((track_id, float(hand_dist)))
                dist_by_track[track_id] = float(hand_dist)
            dist_list.sort(key=lambda x: x[1])
            if dist_list:
                best_track, best_dist = dist_list[0]
                if best_dist <= ball_dist_max:
                    if len(dist_list) == 1 or (dist_list[1][1] - best_dist) >= ball_dist_margin:
                        candidate_track = best_track

        role_changed = False
        old_offense = offense_track
        if candidate_track is not None:
            if offense_track is None:
                offense_track = candidate_track
                switch_streak = 0
                role_changed = True
            elif candidate_track != offense_track:
                switch_streak += 1
                if switch_streak >= max(ball_hold_frames, 1):
                    offense_track = candidate_track
                    switch_streak = 0
                    role_changed = True
            else:
                switch_streak = 0
        else:
            switch_streak = max(switch_streak - 1, 0)

        offense_track_by_frame[frame] = offense_track
        role_change_by_frame[frame] = bool(role_changed and old_offense is not None)
        if role_changed:
            det_orig_pids = [int(detections[i]["orig_pid"]) for i in range(len(detections))]
            cost_matrix = np.full((2, 2), np.nan, dtype=float)
            for t_idx in range(min(2, costs.shape[0])):
                for d_idx in range(min(2, costs.shape[1])):
                    val = costs[t_idx, d_idx]
                    cost_matrix[t_idx, d_idx] = float(val) if np.isfinite(val) else np.nan
            track_to_det = {track_id: int(det["orig_pid"]) for track_id, det in det_by_track.items()}
            other_track = None
            if offense_track is not None:
                other_track = 1 - offense_track if offense_track in (0, 1) else None
            role_change_events.append(
                {
                    "frame": int(frame),
                    "old_offense_track": old_offense,
                    "new_offense_track": offense_track,
                    "candidate_track": candidate_track,
                    "ball_dist_candidate": dist_by_track.get(candidate_track),
                    "ball_dist_other": dist_by_track.get(other_track),
                    "ball_dist_margin": ball_dist_margin,
                    "det_orig_pid_0": det_orig_pids[0] if len(det_orig_pids) > 0 else None,
                    "det_orig_pid_1": det_orig_pids[1] if len(det_orig_pids) > 1 else None,
                    "assign_cost_t0_d0": cost_matrix[0, 0],
                    "assign_cost_t0_d1": cost_matrix[0, 1],
                    "assign_cost_t1_d0": cost_matrix[1, 0],
                    "assign_cost_t1_d1": cost_matrix[1, 1],
                    "assign_track0_orig_pid": track_to_det.get(0),
                    "assign_track1_orig_pid": track_to_det.get(1),
                }
            )

    return frame_map, offense_track_by_frame, role_change_by_frame, role_change_events


def _remap_people_ids(
    df_people: pd.DataFrame,
    frame_map: Dict[int, Dict[int, int]],
    offense_track_by_frame: Dict[int, Optional[int]],
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], str]]:
    rows = []
    role_map: Dict[Tuple[int, int], str] = {}
    for row in df_people.itertuples(index=False):
        frame = int(row.frame)
        orig_pid = int(row.person_id)
        track_id = frame_map.get(frame, {}).get(orig_pid)
        if track_id is None:
            continue
        offense_track = offense_track_by_frame.get(frame)
        role = "unknown"
        new_pid = track_id
        if offense_track is not None:
            if track_id == offense_track:
                new_pid = 0
                role = "offense"
            else:
                new_pid = 1
                role = "defense"
        role_map[(frame, new_pid)] = role
        rows.append(
            {
                "frame": frame,
                "person_id": new_pid,
                "joint": row.joint,
                "X": row.X,
                "Y": row.Y,
                "Z": row.Z,
            }
        )
    return pd.DataFrame(rows), role_map


def _build_positions(
    df_people: pd.DataFrame,
    frames: np.ndarray,
    name_to_idx: Dict[str, int],
    dims: int,
    person_id: int,
    conf_col: Optional[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n_frames = len(frames)
    n_joints = len(name_to_idx)
    positions = np.full((n_frames, n_joints, dims), np.nan, dtype=float)
    conf = None
    if conf_col:
        conf = np.full((n_frames, n_joints), np.nan, dtype=float)

    frame_idx = {f: i for i, f in enumerate(frames)}
    df_p = df_people[df_people["person_id"] == person_id]
    for row in df_p.itertuples(index=False):
        if not isinstance(row.joint, str):
            continue
        j_idx = name_to_idx.get(row.joint)
        if j_idx is None:
            continue
        i = frame_idx.get(int(row.frame))
        if i is None:
            continue
        if dims == 2:
            positions[i, j_idx, 0] = float(row.X)
            positions[i, j_idx, 1] = float(row.Y)
        else:
            positions[i, j_idx, 0] = float(row.X)
            positions[i, j_idx, 1] = float(row.Y)
            positions[i, j_idx, 2] = float(row.Z)
        if conf_col:
            conf[i, j_idx] = float(getattr(row, conf_col))
    return positions, conf


def _process_person(
    frames: np.ndarray,
    positions: np.ndarray,
    name_to_idx: Dict[str, int],
    edges: List[Tuple[int, int]],
    conf: Optional[np.ndarray],
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
    break_mask: Optional[np.ndarray],
    max_interp_gap: int,
    long_gap_strategy: str,
    template: Optional[TemplatePose],
    template_config: Optional[TemplateConfig],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    List[dict],
    List[Tuple[int, int]],
    Dict[Tuple[int, int], float],
]:
    n_frames = positions.shape[0]
    if break_mask is None:
        break_mask = np.zeros(n_frames, dtype=bool)
    segments = _build_segments(n_frames, break_mask)

    logs = []
    bad_initial = None
    bad_union = np.zeros((n_frames, positions.shape[1]), dtype=bool)
    interp_total = np.zeros_like(bad_union)
    long_gap_total = np.zeros_like(bad_union)
    template_fill_total = np.zeros_like(bad_union)
    template_dev_mask = None
    ref_lengths_last: Dict[Tuple[int, int], float] = {}

    template_offsets = None
    template_long_gap_strategy = "nan"
    force_template_frames = None
    if template is not None and template_config is not None and template_config.enabled:
        template_offsets = template.offsets
        template_long_gap_strategy = template_config.long_gap_strategy
        root_for_template = _compute_root(positions, name_to_idx)
        template_dev_mask = _template_deviation_mask(
            positions, root_for_template, template_offsets, template_config.dev_thresh
        )
        if template_config.allow_full_person_fill:
            valid_joint_count = np.isfinite(positions).all(axis=2).sum(axis=1)
            force_template_frames = valid_joint_count < template_config.min_valid_joints

    for pass_idx in range(max_passes):
        bad_edges, ref_lengths = _detect_edges(positions, edges, z_len, z_delta, max_ratio)
        ref_lengths_last = ref_lengths
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
        if template_dev_mask is not None:
            bad_joints |= template_dev_mask
        if bad_initial is None:
            bad_initial = bad_joints.copy()
        bad_union |= bad_joints

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
            template_offset = None
            if template_offsets is not None and np.isfinite(template_offsets[j_idx]).all():
                template_offset = template_offsets[j_idx]
            repaired, interp_mask, long_gap_mask, template_fill_mask = _interpolate_joint(
                frames,
                positions[:, j_idx],
                bad_mask,
                root,
                offset_med,
                segments,
                max_interp_gap,
                long_gap_strategy,
                template_offset,
                template_long_gap_strategy,
                force_template_frames,
            )
            positions[:, j_idx] = repaired
            interp_total[:, j_idx] |= interp_mask
            long_gap_total[:, j_idx] |= long_gap_mask
            template_fill_total[:, j_idx] |= template_fill_mask

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

    if bad_initial is None:
        bad_initial = bad_union
    return (
        positions,
        bad_initial,
        interp_total,
        long_gap_total,
        template_fill_total,
        template_dev_mask,
        logs,
        segments,
        ref_lengths_last,
    )


def _summarize_values(values: np.ndarray) -> Dict[str, float]:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return {"count": 0}
    return {
        "count": int(len(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
    }


def _extract_worst_intervals(
    qa_report: pd.DataFrame,
    min_valid_joints: int,
    z_len: float,
    top_k: int = 5,
) -> List[Dict[str, object]]:
    intervals: List[Dict[str, object]] = []
    for pid in sorted(qa_report["person_id"].unique()):
        df_p = qa_report[qa_report["person_id"] == pid].sort_values("frame")
        frames = df_p["frame"].to_numpy()
        if len(frames) == 0:
            continue
        masks = {
            "jump": df_p["jump_flag"].to_numpy(dtype=bool),
            "low_valid": (df_p["valid_joint_count"] < min_valid_joints).to_numpy(dtype=bool),
            "bone_bad": (df_p["bone_zscore_max"] > z_len).to_numpy(dtype=bool),
        }
        for reason, mask in masks.items():
            if not mask.any():
                continue
            for start, end in _mask_to_intervals(frames, mask):
                frame_mask = (frames >= start) & (frames <= end)
                length = int(np.sum(frame_mask))
                severity = 0.0
                if reason == "jump":
                    vals = df_p.loc[frame_mask, "root_speed_raw"].to_numpy(dtype=float)
                    severity = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0
                elif reason == "low_valid":
                    vals = df_p.loc[frame_mask, "valid_joint_count"].to_numpy(dtype=float)
                    if np.isfinite(vals).any():
                        severity = float(max(min_valid_joints - np.nanmin(vals), 0.0))
                elif reason == "bone_bad":
                    vals = df_p.loc[frame_mask, "bone_zscore_max"].to_numpy(dtype=float)
                    severity = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0
                intervals.append(
                    {
                        "person_id": int(pid),
                        "start": int(start),
                        "end": int(end),
                        "length": length,
                        "reason": reason,
                        "severity": severity,
                    }
                )
    intervals.sort(key=lambda x: (x["length"], x["severity"]), reverse=True)
    return intervals[:top_k]


def _extract_template_intervals(
    frames: np.ndarray,
    template_used_by_person: Dict[int, np.ndarray],
    template_filled_count_by_person: Dict[int, np.ndarray],
    template_reason_by_person: Dict[int, List[str]],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    intervals: List[Dict[str, object]] = []
    for pid in sorted(template_used_by_person):
        used = template_used_by_person[pid]
        if used is None or not used.any():
            continue
        reasons = template_reason_by_person.get(pid, [""] * len(frames))
        filled_counts = template_filled_count_by_person.get(pid, np.zeros(len(frames)))
        for start, end in _mask_to_intervals(frames, used):
            mask = (frames >= start) & (frames <= end)
            length = int(np.sum(mask))
            filled_total = int(np.nansum(filled_counts[mask])) if length > 0 else 0
            filled_max = int(np.nanmax(filled_counts[mask])) if length > 0 else 0
            reason_counts: Dict[str, int] = {}
            for r in np.array(reasons, dtype=object)[mask]:
                if r:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
            if reason_counts:
                reason = max(reason_counts.items(), key=lambda x: x[1])[0]
            else:
                reason = "template"
            intervals.append(
                {
                    "person_id": int(pid),
                    "start": int(start),
                    "end": int(end),
                    "length": length,
                    "reason": reason,
                    "template_filled_joints_total": filled_total,
                    "template_filled_joints_max": filled_max,
                }
            )
    intervals.sort(key=lambda x: (x["length"], x["template_filled_joints_total"]), reverse=True)
    return intervals[:top_k]


def _extract_final_fill_intervals(
    frames: np.ndarray,
    people_ids: List[int],
    final_interp_masks: np.ndarray,
    final_edge_masks: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, object]]:
    intervals: List[Dict[str, object]] = []
    for idx, pid in enumerate(people_ids):
        interp_mask = final_interp_masks[idx].any(axis=1)
        edge_mask = final_edge_masks[idx].any(axis=1)
        interp_counts = final_interp_masks[idx].sum(axis=1)
        edge_counts = final_edge_masks[idx].sum(axis=1)

        if edge_mask.any():
            for start, end in _mask_to_intervals(frames, edge_mask):
                mask = (frames >= start) & (frames <= end)
                count = int(np.nansum(edge_counts[mask])) if mask.any() else 0
                intervals.append(
                    {
                        "person_id": int(pid),
                        "start": int(start),
                        "end": int(end),
                        "length": int(mask.sum()),
                        "reason": "edge",
                        "count": count,
                    }
                )

        interp_only = interp_mask & (~edge_mask)
        if interp_only.any():
            for start, end in _mask_to_intervals(frames, interp_only):
                mask = (frames >= start) & (frames <= end)
                count = int(np.nansum(interp_counts[mask])) if mask.any() else 0
                intervals.append(
                    {
                        "person_id": int(pid),
                        "start": int(start),
                        "end": int(end),
                        "length": int(mask.sum()),
                        "reason": "interp",
                        "count": count,
                    }
                )

    intervals.sort(key=lambda x: (x["length"], x["count"]), reverse=True)
    return intervals[:top_k]


def _write_npz_masks(
    path: str,
    frames: np.ndarray,
    joint_names: List[str],
    person_ids: List[int],
    bad_mask: np.ndarray,
    interp_mask: np.ndarray,
    long_gap_mask: np.ndarray,
    jump_mask: np.ndarray,
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        path,
        frames=frames,
        joint_names=np.array(joint_names),
        person_ids=np.array(person_ids),
        bad_mask=bad_mask,
        interp_mask=interp_mask,
        long_gap_mask=long_gap_mask,
        jump_mask=jump_mask,
    )


def _apply_view_transform(df: pd.DataFrame, y_max: float) -> pd.DataFrame:
    df = df.copy()
    df["Xv"] = df["X"]
    df["Yv"] = df["Z"]
    df["Zv"] = y_max - df["Y"]
    return df


def _build_joint_map_for_frame(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    joints = {}
    for joint, x, y, z in df[["joint", "Xv", "Yv", "Zv"]].itertuples(index=False, name=None):
        if not isinstance(joint, str):
            continue
        if joint in IGNORE_JOINTS:
            continue
        if not np.isfinite([x, y, z]).all():
            continue
        joints[joint] = (float(x), float(y), float(z))
    return joints


def _draw_skeleton(ax, joint_map: Dict[str, Tuple[float, float, float]], color: str) -> None:
    for u_idx, v_idx in POSE_CONNECTIONS:
        u_name = JOINT_NAMES[u_idx]
        v_name = JOINT_NAMES[v_idx]
        p1 = joint_map.get(u_name)
        p2 = joint_map.get(v_name)
        if p1 is None or p2 is None:
            continue
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color, linewidth=1.5)


def _configure_writer(fps: int):
    try:
        import imageio_ffmpeg
        from matplotlib import rcParams

        rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    from matplotlib import animation

    return animation.FFMpegWriter(fps=fps, metadata={"artist": "codex"}, bitrate=3000)


def _apply_zoom(lim: Tuple[float, float], zoom: float) -> Tuple[float, float]:
    if zoom <= 0:
        return lim
    span = lim[1] - lim[0]
    if span <= 0:
        return lim
    center = (lim[0] + lim[1]) / 2.0
    half = span / 2.0 / zoom
    return (center - half, center + half)


def _render_qa_video(
    df: pd.DataFrame,
    qa_df: pd.DataFrame,
    out_path: str,
    fps: int,
    trail: int,
    elev: int,
    azim: int,
    zoom: float,
) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("No data to render.")
    frames = sorted(df["frame"].unique())
    y_max_global = float(df["Y"].max())
    df_v = _apply_view_transform(df, y_max_global)

    x_min = float(df_v["Xv"].min())
    x_max = float(df_v["Xv"].max())
    y_min = float(df_v["Yv"].min())
    y_max = float(df_v["Yv"].max())
    z_min = float(df_v["Zv"].min())
    z_max = float(df_v["Zv"].max())

    margin = 0.2
    x_lim = _apply_zoom((x_min - margin, x_max + margin), zoom)
    y_lim = _apply_zoom((y_min - margin, y_max + margin), zoom)
    z_lim = _apply_zoom((z_min - margin, z_max + margin), zoom)

    df_by_frame = {int(k): v for k, v in df_v.groupby("frame")}
    qa_by_frame = {int(k): v for k, v in qa_df.groupby("frame")}

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    writer = _configure_writer(fps)

    trail_points = {0: [], 1: []}
    colors = {0: "red", 1: "blue"}

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with writer.saving(fig, out_path, dpi=120):
        for frame in frames:
            ax.cla()
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"Frame {frame}")
            try:
                ax.set_box_aspect((x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]))
            except Exception:
                pass

            frame_df = df_by_frame.get(int(frame))
            if frame_df is not None and not frame_df.empty:
                for pid in sorted(frame_df["person_id"].unique()):
                    if int(pid) == -1:
                        continue
                    p_df = frame_df[frame_df["person_id"] == pid]
                    color = colors.get(int(pid), "gray")
                    joint_map = _build_joint_map_for_frame(p_df)
                    _draw_skeleton(ax, joint_map, color)
                    ax.scatter(p_df["Xv"], p_df["Yv"], p_df["Zv"], c=color, s=16)

                    root = None
                    for name in ANCHOR_JOINTS:
                        if name in joint_map:
                            root = np.array(joint_map[name], dtype=float) if root is None else root + joint_map[name]
                    if root is not None:
                        root = root / len([n for n in ANCHOR_JOINTS if n in joint_map])
                        if trail > 0:
                            trail_points[int(pid)].append(root)
                            if len(trail_points[int(pid)]) > trail:
                                trail_points[int(pid)] = trail_points[int(pid)][-trail:]
                            if trail >= 2 and len(trail_points[int(pid)]) >= 2:
                                trail_arr = np.array(trail_points[int(pid)])
                                ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2], c=color, linewidth=1.0)

                ball_df = frame_df[(frame_df["person_id"] == -1) | (frame_df["joint"] == "ball")]
                if not ball_df.empty:
                    ax.scatter(ball_df["Xv"], ball_df["Yv"], ball_df["Zv"], c="orange", s=40, marker="o")

            qa_frame = qa_by_frame.get(int(frame))
            if qa_frame is not None and not qa_frame.empty:
                swap_flag = bool(qa_frame.get("role_change_flag", pd.Series([False])).any())
                lines = [f"Frame {frame}"]
                for pid in [0, 1]:
                    row = qa_frame[qa_frame["person_id"] == pid]
                    if row.empty:
                        continue
                    role = row["role"].iloc[0]
                    jump_flag = bool(row["jump_flag"].iloc[0])
                    valid_joints = int(row["valid_joint_count"].iloc[0])
                    bone_max = row["bone_zscore_max"].iloc[0]
                    bone_txt = f"{bone_max:.2f}" if np.isfinite(bone_max) else "nan"
                    lines.append(f"{role}: jump={jump_flag} valid={valid_joints} bone_max={bone_txt}")
                if swap_flag:
                    lines.append("ROLE SWAP")
                ax.text2D(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top")

            writer.grab_frame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess 3D pose CSV with re-ID, repair, QA, and viz.")
    parser.add_argument("--input_csv", required=True, help="Input 3D CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_csv", default=None, help="Output cleaned CSV (default: <stem>_clean.csv)")
    parser.add_argument("--qa_report", default=None, help="QA report CSV (default: <stem>_qa_report.csv)")
    parser.add_argument("--qa_summary", default=None, help="QA summary JSON (default: <stem>_qa_summary.json)")
    parser.add_argument("--mask_npz", default=None, help="Optional masks npz output")
    parser.add_argument("--viz_output", default=None, help="Optional 3D viz mp4 output")
    parser.add_argument("--viz_fps", type=int, default=30, help="Viz fps")
    parser.add_argument("--viz_trail", type=int, default=60, help="Root trail length")
    parser.add_argument("--viz_elev", type=int, default=20, help="Viz camera elevation")
    parser.add_argument("--viz_azim", type=int, default=45, help="Viz camera azimuth")
    parser.add_argument("--viz_zoom", type=float, default=1.2, help="Viz zoom")
    parser.add_argument("--fps", type=float, default=60.0, help="FPS for rate calculations")

    parser.add_argument("--max_people", type=int, default=2, help="Max people to track")
    parser.add_argument("--no_ball_reid", action="store_true", help="Disable ball-based role assignment")
    parser.add_argument("--ball_dist_max", type=float, default=1.5, help="Max ball-hand distance for offense")
    parser.add_argument("--ball_dist_margin", type=float, default=0.05, help="Min distance margin to accept offense")
    parser.add_argument("--ball_hold_frames", type=int, default=5, help="Hysteresis frames for offense switch")
    parser.add_argument("--assign_dist_max", type=float, default=2.0, help="Max root distance for tracking assignment")
    parser.add_argument("--ball_person_id", type=int, default=-1, help="person_id treated as ball (long format)")
    parser.add_argument("--ball_joint_name", type=str, default="ball", help="Joint name treated as ball (long format)")
    parser.add_argument("--ball_columns", type=str, default=None, help="Comma list of ball columns for wide format")
    parser.add_argument("--compare_raw", action="store_true", help="Include raw vs clean comparison in summary")
    parser.add_argument("--role_change_csv", default=None, help="Role change debug CSV (default: <stem>_role_changes.csv)")
    parser.add_argument("--jump_csv", default=None, help="Jump frames CSV (default: <stem>_jump_frames.csv)")

    parser.add_argument("--conf_col", default=None, help="Override confidence column name")
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
    parser.add_argument("--jump_z", type=float, default=4.0, help="Root jump z-score threshold")
    parser.add_argument("--jump_abs", type=float, default=1.0, help="Root jump absolute threshold")
    parser.add_argument("--max_interp_gap", type=int, default=10, help="Max gap (frames) to interpolate")
    parser.add_argument("--long_gap_strategy", choices=["nan", "root"], default="nan", help="Long gap fill strategy")
    parser.add_argument("--use_template_fill", action="store_true", help="Enable template-based fill")
    parser.add_argument("--template_init_frames", type=int, default=60, help="Initial frames to build template")
    parser.add_argument("--template_min_valid_joints", type=int, default=10, help="Min valid joints for template frames")
    parser.add_argument("--template_max_bone_z", type=float, default=3.5, help="Max bone zscore for template frames")
    parser.add_argument("--template_dev_thresh_m", type=float, default=0.25, help="Template deviation threshold")
    parser.add_argument(
        "--template_long_gap_strategy",
        choices=["nan", "template"],
        default="nan",
        help="Long gap fill strategy for template mode",
    )
    parser.add_argument("--template_allow_full_person_fill", action="store_true", help="Allow template fill when person is missing")
    parser.add_argument("--template_allow_jump", action="store_true", help="Allow jump frames when building template")
    parser.add_argument("--final_fill", action="store_true", help="Enable final fill pass")
    parser.add_argument("--final_fill_people", choices=["all", "roles"], default="all", help="Final fill target")
    parser.add_argument("--final_max_interp_gap", type=int, default=30, help="Final fill max interpolation gap")
    parser.add_argument("--final_edge_strategy", choices=["nan", "hold"], default="nan", help="Final fill edge strategy")
    parser.add_argument("--final_max_edge_hold_gap", type=int, default=60, help="Final fill max edge hold gap")
    parser.add_argument("--min_valid_joints", type=int, default=12, help="QA threshold for low joint count")
    args = parser.parse_args()

    args.input_csv = _normalize_path(args.input_csv, must_exist=True, label="input_csv")
    args.output_dir = _normalize_path(args.output_dir, must_exist=False, label="output_dir")
    args.output_csv = _normalize_path(args.output_csv, must_exist=False, label="output_csv")
    args.qa_report = _normalize_path(args.qa_report, must_exist=False, label="qa_report")
    args.qa_summary = _normalize_path(args.qa_summary, must_exist=False, label="qa_summary")
    args.mask_npz = _normalize_path(args.mask_npz, must_exist=False, label="mask_npz")
    args.viz_output = _normalize_path(args.viz_output, must_exist=False, label="viz_output")
    args.role_change_csv = _normalize_path(args.role_change_csv, must_exist=False, label="role_change_csv")
    args.jump_csv = _normalize_path(args.jump_csv, must_exist=False, label="jump_csv")

    ball_joint_name = _normalize_joint_name(args.ball_joint_name) if args.ball_joint_name else None
    df, schema_info = _load_pose_csv(
        args.input_csv,
        ball_joint_name=ball_joint_name,
        ball_columns=args.ball_columns,
        ball_person_id=args.ball_person_id,
    )
    print(f"[schema] format={schema_info.get('format')} dims={schema_info.get('dims')}")
    if schema_info.get("unknown_joints"):
        print(f"[schema] dropped unknown joints: {schema_info['unknown_joints']}")
    dims = int(schema_info.get("dims") or (3 if "Z" in df.columns and df["Z"].notna().any() else 2))
    if dims != 3 and args.viz_output:
        raise ValueError("viz_output requires 3D input with a valid Z column.")
    conf_col = args.conf_col or _select_conf_col(df)
    name_to_idx = _build_name_to_idx()
    edges = _edges_for_names(name_to_idx)

    ball_mask = np.zeros(len(df), dtype=bool)
    if ball_joint_name:
        ball_mask |= df["joint"] == ball_joint_name
    if args.ball_person_id is not None:
        ball_mask |= df["person_id"] == args.ball_person_id
    df_ball = df[ball_mask].copy()
    df_people = df[~ball_mask].copy()

    frames = np.array(sorted(df_people["frame"].unique()))
    if len(frames) == 0:
        raise ValueError("No person frames found.")

    frame_map, offense_track_by_frame, role_change_by_frame, role_change_events = _reid_tracks(
        df_people=df_people,
        df_ball=df_ball,
        frames=frames,
        dims=dims,
        max_people=args.max_people,
        use_ball=not args.no_ball_reid,
        ball_dist_max=args.ball_dist_max,
        ball_dist_margin=args.ball_dist_margin,
        ball_hold_frames=args.ball_hold_frames,
        assign_dist_max=args.assign_dist_max,
    )
    df_reid, role_map = _remap_people_ids(df_people, frame_map, offense_track_by_frame)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    out_csv = args.output_csv or os.path.join(out_dir, f"{stem}_clean.csv")
    out_report = args.qa_report or os.path.join(out_dir, f"{stem}_qa_report.csv")
    out_summary = args.qa_summary or os.path.join(out_dir, f"{stem}_qa_summary.json")

    people_ids = sorted([pid for pid in df_reid["person_id"].unique() if pid >= 0])
    people_ids = people_ids[: args.max_people]
    if not people_ids:
        raise ValueError("No people found after re-ID.")

    n_frames = len(frames)
    n_joints = len(name_to_idx)
    bad_masks = np.zeros((len(people_ids), n_frames, n_joints), dtype=bool)
    interp_masks = np.zeros_like(bad_masks)
    long_gap_masks = np.zeros_like(bad_masks)
    template_fill_masks = np.zeros_like(bad_masks)
    template_dev_masks = np.zeros_like(bad_masks)
    final_interp_masks = np.zeros_like(bad_masks)
    final_edge_masks = np.zeros_like(bad_masks)
    jump_masks = np.zeros((len(people_ids), n_frames), dtype=bool)
    jump_reasons: Dict[int, List[str]] = {}
    root_speed_raw: Dict[int, np.ndarray] = {}
    root_speed_clean: Dict[int, np.ndarray] = {}
    bone_z_max: Dict[int, np.ndarray] = {}
    bone_z_mean: Dict[int, np.ndarray] = {}
    repaired_positions_by_person: Dict[int, np.ndarray] = {}
    raw_positions_by_person: Dict[int, np.ndarray] = {}
    conf_by_person: Dict[int, Optional[np.ndarray]] = {}
    template_used_by_person: Dict[int, np.ndarray] = {}
    template_reason_by_person: Dict[int, List[str]] = {}
    template_filled_count_by_person: Dict[int, np.ndarray] = {}

    template_config = TemplateConfig(
        enabled=args.use_template_fill,
        init_frames=args.template_init_frames,
        min_valid_joints=args.template_min_valid_joints,
        max_bone_z=args.template_max_bone_z,
        dev_thresh=args.template_dev_thresh_m,
        allow_jump=args.template_allow_jump,
        long_gap_strategy=args.template_long_gap_strategy,
        allow_full_person_fill=args.template_allow_full_person_fill,
    )

    for idx, pid in enumerate(people_ids):
        positions_raw, conf = _build_positions(df_reid, frames, name_to_idx, dims, pid, conf_col)
        raw_positions_by_person[pid] = positions_raw.copy()
        conf_by_person[pid] = conf
        root_raw = _compute_root(positions_raw, name_to_idx)
        jump_mask, reasons, speed_raw = _detect_root_jumps(frames, root_raw, args.jump_z, args.jump_abs)
        jump_masks[idx] = jump_mask
        jump_reasons[pid] = reasons
        root_speed_raw[pid] = speed_raw

        positions_jump = positions_raw.copy()
        positions_jump[jump_mask] = np.nan
        z_max, z_mean = _compute_bone_zscores(positions_jump, edges)
        bone_z_max[pid] = z_max
        bone_z_mean[pid] = z_mean

        template = _build_template_pose(
            frames,
            positions_raw,
            name_to_idx,
            edges,
            jump_mask,
            z_max,
            template_config,
        )

        (
            repaired,
            bad_initial,
            interp_mask,
            long_gap_mask,
            template_fill_mask,
            template_dev_mask,
            logs,
            segments,
            _,
        ) = _process_person(
            frames,
            positions_jump,
            name_to_idx,
            edges,
            conf,
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
            jump_mask,
            args.max_interp_gap,
            args.long_gap_strategy,
            template,
            template_config if template_config.enabled else None,
        )
        bad_masks[idx] = bad_initial
        interp_masks[idx] = interp_mask
        long_gap_masks[idx] = long_gap_mask
        template_fill_masks[idx] = template_fill_mask
        if template_dev_mask is not None:
            template_dev_masks[idx] = template_dev_mask

        if template_config.enabled and template_fill_mask.any():
            valid_raw = np.isfinite(positions_raw).all(axis=2).sum(axis=1)
            valid_clean = np.isfinite(repaired).all(axis=2).sum(axis=1)
            z_clean_max, _ = _compute_bone_zscores(repaired, edges)
            z_raw_max, _ = _compute_bone_zscores(positions_raw, edges)
            for t in range(n_frames):
                if jump_mask[t]:
                    continue
                if valid_clean[t] < valid_raw[t]:
                    revert = True
                elif np.isfinite(z_raw_max[t]) and np.isfinite(z_clean_max[t]) and z_clean_max[t] > z_raw_max[t]:
                    revert = True
                else:
                    revert = False
                if not revert:
                    continue
                raw_frame = positions_raw[t]
                clean_frame = repaired[t]
                revert_mask = template_fill_mask[t] & np.isfinite(raw_frame).all(axis=1)
                if not revert_mask.any():
                    continue
                clean_frame[revert_mask] = raw_frame[revert_mask]
                repaired[t] = clean_frame
                template_fill_mask[t, revert_mask] = False
                template_dev_masks[idx, t, revert_mask] = False
            template_fill_masks[idx] = template_fill_mask

        template_used = template_fill_mask.any(axis=1)
        template_used_by_person[pid] = template_used
        template_filled_count_by_person[pid] = template_fill_mask.sum(axis=1)
        template_reasons = [""] * n_frames
        if template_config.enabled and template_used.any():
            low_valid = np.isfinite(positions_raw).all(axis=2).sum(axis=1) < template_config.min_valid_joints
            template_dev_count = template_dev_masks[idx].sum(axis=1)
            template_long_gap_count = (template_fill_mask & long_gap_mask).sum(axis=1)
            for t in range(n_frames):
                if not template_used[t]:
                    continue
                if template_long_gap_count[t] > 0:
                    template_reasons[t] = "long_gap"
                elif low_valid[t]:
                    template_reasons[t] = "low_valid"
                elif template_dev_count[t] > 0:
                    template_reasons[t] = "deviation"
                else:
                    template_reasons[t] = "template"
        template_reason_by_person[pid] = template_reasons

        final_interp_mask = np.zeros((n_frames, n_joints), dtype=bool)
        final_edge_mask = np.zeros((n_frames, n_joints), dtype=bool)
        if args.final_fill:
            allow_frames = None
            if args.final_fill_people == "roles":
                allow_frames = np.array(
                    [
                        role_map.get((int(frame), int(pid)), "unknown") in ("offense", "defense")
                        for frame in frames
                    ],
                    dtype=bool,
                )
            repaired, final_interp_mask, final_edge_mask = _final_fill_positions(
                repaired,
                args.final_max_interp_gap,
                args.final_edge_strategy,
                args.final_max_edge_hold_gap,
                allow_frames,
            )
        final_interp_masks[idx] = final_interp_mask
        final_edge_masks[idx] = final_edge_mask

        repaired_positions_by_person[pid] = repaired
        root_clean = _compute_root(repaired, name_to_idx)
        root_speed_clean[pid] = _compute_root_speed(frames, root_clean)

    out_rows = []
    for pid in people_ids:
        repaired = repaired_positions_by_person[pid]
        bad_initial = bad_masks[people_ids.index(pid)]
        conf = conf_by_person.get(pid)
        for i, frame in enumerate(frames):
            for jname, jidx in name_to_idx.items():
                pos = repaired[i, jidx]
                if not np.isfinite(pos).all():
                    continue
                row = {
                    "frame": int(frame),
                    "person_id": int(pid),
                    "joint": jname,
                    "X": float(pos[0]),
                    "Y": float(pos[1]),
                }
                if dims == 3:
                    row["Z"] = float(pos[2])
                if conf_col:
                    conf_val = np.nan
                    if not bad_initial[i, jidx]:
                        raw_pos = raw_positions_by_person[pid][i, jidx]
                        if np.isfinite(raw_pos).all():
                            conf_val = float(conf[i, jidx]) if conf is not None else np.nan
                    row[conf_col] = conf_val
                out_rows.append(row)

    if not df_ball.empty:
        for row in df_ball.itertuples(index=False):
            ball_row = {
                "frame": int(row.frame),
                "person_id": int(row.person_id),
                "joint": row.joint,
                "X": float(row.X),
                "Y": float(row.Y),
            }
            if dims == 3:
                ball_row["Z"] = float(row.Z)
            out_rows.append(ball_row)

    df_out = pd.DataFrame(out_rows)
    df_out = df_out.sort_values(["frame", "person_id", "joint"])
    df_out.to_csv(out_csv, index=False)

    qa_rows = []
    for idx, pid in enumerate(people_ids):
        repaired = repaired_positions_by_person[pid]
        raw = raw_positions_by_person[pid]
        bad_initial = bad_masks[idx]
        interp_mask = interp_masks[idx]
        long_gap_mask = long_gap_masks[idx]
        template_fill_mask = template_fill_masks[idx]
        template_dev_mask = template_dev_masks[idx]
        final_interp_mask = final_interp_masks[idx]
        final_edge_mask = final_edge_masks[idx]
        speed_raw = root_speed_raw[pid]
        speed_clean = root_speed_clean[pid]
        z_max = bone_z_max[pid]
        z_mean = bone_z_mean[pid]
        for i, frame in enumerate(frames):
            valid_raw = int(np.isfinite(raw[i]).all(axis=1).sum())
            valid_clean = int(np.isfinite(repaired[i]).all(axis=1).sum())
            repaired_count = int((bad_initial[i] & np.isfinite(repaired[i]).all(axis=1)).sum())
            interp_count = int(interp_mask[i].sum())
            long_gap_count = int(long_gap_mask[i].sum())
            template_fill_count = int(template_fill_mask[i].sum())
            template_trigger_count = int((template_fill_mask[i] | template_dev_mask[i]).sum())
            final_fill_mask = final_interp_mask[i] | final_edge_mask[i]
            final_fill_count = int(final_fill_mask.sum())
            final_edge_count = int(final_edge_mask[i].sum())
            role = role_map.get((int(frame), int(pid)), "unknown")
            qa_rows.append(
                {
                    "frame": int(frame),
                    "person_id": int(pid),
                    "role": role,
                    "valid_joint_count_raw": valid_raw,
                    "valid_joint_count": valid_clean,
                    "root_speed_raw": float(speed_raw[i]) if np.isfinite(speed_raw[i]) else np.nan,
                    "root_speed": float(speed_clean[i]) if np.isfinite(speed_clean[i]) else np.nan,
                    "jump_flag": bool(jump_masks[idx, i]),
                    "jump_reason": jump_reasons[pid][i],
                    "bone_zscore_max": float(z_max[i]) if np.isfinite(z_max[i]) else np.nan,
                    "bone_zscore_mean": float(z_mean[i]) if np.isfinite(z_mean[i]) else np.nan,
                    "bad_joint_count": int(bad_initial[i].sum()),
                    "repaired_joint_count": repaired_count,
                    "interpolated_joint_count": interp_count,
                    "long_gap_joint_count": long_gap_count,
                    "template_used_flag": int(template_fill_count > 0),
                    "template_filled_joint_count": template_fill_count,
                    "template_frame_count": template_trigger_count,
                    "final_filled_joint_count": final_fill_count,
                    "final_filled_flag": int(final_fill_count > 0),
                    "final_edge_filled_joint_count": final_edge_count,
                    "role_change_flag": bool(role_change_by_frame.get(int(frame), False)),
                }
            )

    qa_report = pd.DataFrame(qa_rows)
    qa_report.to_csv(out_report, index=False)

    role_change_csv = args.role_change_csv or os.path.join(out_dir, f"{stem}_role_changes.csv")
    if role_change_events:
        pd.DataFrame(role_change_events).to_csv(role_change_csv, index=False)
    else:
        pd.DataFrame(columns=["frame", "old_offense_track", "new_offense_track"]).to_csv(role_change_csv, index=False)

    jump_csv = args.jump_csv or os.path.join(out_dir, f"{stem}_jump_frames.csv")
    jump_rows = []
    for idx, pid in enumerate(people_ids):
        reasons = jump_reasons.get(pid, [])
        for i, frame in enumerate(frames):
            if jump_masks[idx, i]:
                jump_rows.append(
                    {
                        "frame": int(frame),
                        "person_id": int(pid),
                        "reason": reasons[i] if i < len(reasons) else "",
                        "root_speed_raw": float(root_speed_raw[pid][i]) if np.isfinite(root_speed_raw[pid][i]) else np.nan,
                    }
                )
    pd.DataFrame(jump_rows).to_csv(jump_csv, index=False)

    switch_frames = sorted(
        [
            int(ev["frame"])
            for ev in role_change_events
            if ev.get("old_offense_track") is not None
        ]
    )
    switch_per_min = None
    if len(frames) > 1:
        duration_frames = frames[-1] - frames[0] + 1
        duration_seconds = duration_frames / max(args.fps, 1e-6)
        switch_per_min = float(len(switch_frames)) / max(duration_seconds, 1e-6) * 60.0

    people_summary = []
    for idx, pid in enumerate(people_ids):
        valid_raw = qa_report[qa_report["person_id"] == pid]["valid_joint_count_raw"].to_numpy()
        valid_clean = qa_report[qa_report["person_id"] == pid]["valid_joint_count"].to_numpy()
        jump_count = int(jump_masks[idx].sum())
        low_valid = int((valid_clean < args.min_valid_joints).sum())
        z_max = qa_report[qa_report["person_id"] == pid]["bone_zscore_max"].to_numpy()
        people_summary.append(
            {
                "person_id": int(pid),
                "role": role_map.get((int(frames[0]), int(pid)), "unknown"),
                "valid_joint_median_raw": float(np.median(valid_raw)) if len(valid_raw) else None,
                "valid_joint_median": float(np.median(valid_clean)) if len(valid_clean) else None,
                "jump_frames": jump_count,
                "jump_ratio": float(jump_count / len(frames)),
                "low_valid_frames": low_valid,
                "low_valid_ratio": float(low_valid / len(frames)),
                "bone_bad_frames": int(np.nansum(z_max > args.z_len)),
            }
        )

    worst_intervals = _extract_worst_intervals(
        qa_report,
        min_valid_joints=args.min_valid_joints,
        z_len=args.z_len,
        top_k=5,
    )
    template_intervals = _extract_template_intervals(
        frames,
        template_used_by_person,
        template_filled_count_by_person,
        template_reason_by_person,
        top_k=5,
    )
    final_fill_total_joints = int(qa_report["final_filled_joint_count"].sum()) if not qa_report.empty else 0
    if qa_report.empty:
        final_fill_frames = 0
    else:
        final_fill_frames = int(qa_report.groupby("frame")["final_filled_flag"].max().sum())
    final_fill_intervals = _extract_final_fill_intervals(
        frames,
        people_ids,
        final_interp_masks,
        final_edge_masks,
        top_k=5,
    )

    summary = {
        "input_csv": os.path.abspath(args.input_csv),
        "output_csv": os.path.abspath(out_csv),
        "frames": int(len(frames)),
        "fps": float(args.fps),
        "switch_frames": switch_frames,
        "switch_count": int(len(switch_frames)),
        "switch_per_min": switch_per_min,
        "root_speed_raw": _summarize_values(np.concatenate([root_speed_raw[pid] for pid in people_ids])),
        "root_speed_clean": _summarize_values(np.concatenate([root_speed_clean[pid] for pid in people_ids])),
        "people": people_summary,
        "worst_intervals": worst_intervals,
        "template_intervals": template_intervals,
        "final_fill_total_joints": final_fill_total_joints,
        "final_fill_frames": final_fill_frames,
        "final_fill_intervals_top5": final_fill_intervals,
        "params": vars(args),
    }

    if args.compare_raw:
        raw_valid = qa_report["valid_joint_count_raw"].to_numpy()
        clean_valid = qa_report["valid_joint_count"].to_numpy()
        compare_people = []
        for pid in people_ids:
            p_rows = qa_report[qa_report["person_id"] == pid]
            compare_people.append(
                {
                    "person_id": int(pid),
                    "valid_joint_median_raw": float(np.median(p_rows["valid_joint_count_raw"])),
                    "valid_joint_median_clean": float(np.median(p_rows["valid_joint_count"])),
                    "root_speed_raw": _summarize_values(root_speed_raw[pid]),
                    "root_speed_clean": _summarize_values(root_speed_clean[pid]),
                }
            )
        summary["compare"] = {
            "root_speed": {
                "raw": _summarize_values(np.concatenate([root_speed_raw[pid] for pid in people_ids])),
                "clean": _summarize_values(np.concatenate([root_speed_clean[pid] for pid in people_ids])),
            },
            "valid_joint_count": {
                "raw_median": float(np.median(raw_valid)) if len(raw_valid) else None,
                "clean_median": float(np.median(clean_valid)) if len(clean_valid) else None,
            },
            "switch_count": {
                "raw": int(len(switch_frames)),
                "clean": int(len(switch_frames)),
            },
            "people": compare_people,
        }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    if args.mask_npz:
        _write_npz_masks(
            args.mask_npz,
            frames,
            JOINT_NAMES,
            people_ids,
            bad_masks,
            interp_masks,
            long_gap_masks,
            jump_masks,
        )

    if args.viz_output:
        qa_for_viz = qa_report[qa_report["person_id"].isin(people_ids)]
        _render_qa_video(
            df_out,
            qa_for_viz,
            args.viz_output,
            args.viz_fps,
            args.viz_trail,
            args.viz_elev,
            args.viz_azim,
            args.viz_zoom,
        )

    print(f"Cleaned CSV : {out_csv}")
    print(f"QA report   : {out_report}")
    print(f"QA summary  : {out_summary}")
    print(f"Role change : {role_change_csv}")
    print(f"Jump frames : {jump_csv}")
    if args.mask_npz:
        print(f"Masks npz   : {args.mask_npz}")
    if args.viz_output:
        print(f"Viz video   : {args.viz_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
