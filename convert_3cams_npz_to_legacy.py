#!/usr/bin/env python3
"""convert_3cams_npz_to_legacy.py

目的:
  3台キャリブの npz (例: 1208_3cams_center_cam2.npz) は
  R_w2c1 / t_w2c1 のように "world(cam2) -> cam" (w2c) の外部パラを持つ。
  一方で batch3_3dposeestimation.py などは R1 / t1 (および R3 / t3)
  を必須キーとして読み込むため KeyError になる。

このスクリプトは、w2c外部パラから "互換 npz" を生成する。

互換 npz に入れるもの:
  - cam_matrix{1,2,3}, dist_coeffs{1,2,3} （既存キーを尊重）
  - K{1,2,3}, dist{1,2,3} （別名）
  - R1,t1,R2,t2,R3,t3 : c2w 形式（多くの古いスクリプトが直後に inverse する前提）
  - P1,P2,P3 : 投影行列 K@[R_w2c|t_w2c] (w2c)（batch3 の choose_extrinsics が向きを自動判定できる）
  - image_size1/2/3 : 推定 (2*cx, 2*cy) で埋める
  - 元npzのメタ情報は 가능한限り引き継ぐ

使い方:
  uv run python convert_3cams_npz_to_legacy.py \
    --in_npz  ../calibrationwith3cams/output/1208_3cams_center_cam2.npz \
    --out_npz ../calibrationwith3cams/output/1208_3cams_center_cam2_legacy.npz
"""

from __future__ import annotations

import argparse
import numpy as np


def _pick(d: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray:
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of the keys exist in npz: {keys}")


def _as_mat33(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(3, 3)
    return x


def _as_vec3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(3, 1)


def _infer_image_size_from_K(K: np.ndarray) -> tuple[int, int]:
    # Many calibrations use (cx,cy) ~ (W/2, H/2). Round to nearest integer.
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    w = int(round(cx * 2.0))
    h = int(round(cy * 2.0))
    # Fallback sanity
    if w <= 0 or h <= 0:
        w, h = 1920, 1080
    return w, h


def _w2c_to_c2w(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given w2c: X_cam = R X_w + t, return c2w transform."""
    R = _as_mat33(R)
    t = _as_vec3(t)
    R_c2w = R.T
    t_c2w = -R.T @ t
    return R_c2w, t_c2w


def _build_P(K: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R_w2c = _as_mat33(R_w2c)
    t_w2c = _as_vec3(t_w2c)
    return K @ np.hstack([R_w2c, t_w2c])


def convert(in_npz: str, out_npz: str) -> None:
    d = np.load(in_npz, allow_pickle=True)

    # Intrinsics
    K1 = _pick(d, "cam_matrix1", "K1")
    K2 = _pick(d, "cam_matrix2", "K2")
    K3 = _pick(d, "cam_matrix3", "K3")

    dist1 = _pick(d, "dist_coeffs1", "dist1")
    dist2 = _pick(d, "dist_coeffs2", "dist2")
    dist3 = _pick(d, "dist_coeffs3", "dist3")

    # Extrinsics (preferred)
    if "R_w2c1" in d and "t_w2c1" in d:
        R1_w2c = d["R_w2c1"]
        t1_w2c = d["t_w2c1"]
    elif "Tw2c1" in d:
        T = np.asarray(d["Tw2c1"], dtype=np.float64).reshape(4, 4)
        R1_w2c = T[:3, :3]
        t1_w2c = T[:3, 3:4]
    else:
        raise KeyError("Cannot find w2c extrinsics for cam1 (need R_w2c1/t_w2c1 or Tw2c1)")

    if "R_w2c2" in d and "t_w2c2" in d:
        R2_w2c = d["R_w2c2"]
        t2_w2c = d["t_w2c2"]
    elif "Tw2c2" in d:
        T = np.asarray(d["Tw2c2"], dtype=np.float64).reshape(4, 4)
        R2_w2c = T[:3, :3]
        t2_w2c = T[:3, 3:4]
    else:
        # center cam2 assumed world origin
        R2_w2c = np.eye(3)
        t2_w2c = np.zeros((3, 1))

    if "R_w2c3" in d and "t_w2c3" in d:
        R3_w2c = d["R_w2c3"]
        t3_w2c = d["t_w2c3"]
    elif "Tw2c3" in d:
        T = np.asarray(d["Tw2c3"], dtype=np.float64).reshape(4, 4)
        R3_w2c = T[:3, :3]
        t3_w2c = T[:3, 3:4]
    else:
        raise KeyError("Cannot find w2c extrinsics for cam3 (need R_w2c3/t_w2c3 or Tw2c3)")

    # Legacy convention used by some scripts: store c2w, then they invert
    R1_c2w, t1_c2w = _w2c_to_c2w(R1_w2c, t1_w2c)
    R2_c2w, t2_c2w = _w2c_to_c2w(R2_w2c, t2_w2c)
    R3_c2w, t3_c2w = _w2c_to_c2w(R3_w2c, t3_w2c)

    # Projection matrices (w2c) for batch3 choose_extrinsics
    P1 = _build_P(K1, R1_w2c, t1_w2c)
    P2 = _build_P(K2, R2_w2c, t2_w2c)
    P3 = _build_P(K3, R3_w2c, t3_w2c)

    # Infer image sizes
    w1, h1 = _infer_image_size_from_K(K1)
    w2, h2 = _infer_image_size_from_K(K2)
    w3, h3 = _infer_image_size_from_K(K3)

    payload: dict[str, object] = {
        # Intrinsics (both naming schemes)
        "cam_matrix1": K1,
        "cam_matrix2": K2,
        "cam_matrix3": K3,
        "dist_coeffs1": dist1,
        "dist_coeffs2": dist2,
        "dist_coeffs3": dist3,
        "K1": K1,
        "K2": K2,
        "K3": K3,
        "dist1": dist1,
        "dist2": dist2,
        "dist3": dist3,
        # Legacy extrinsics (c2w)
        "R1": R1_c2w,
        "t1": t1_c2w,
        "R2": R2_c2w,
        "t2": t2_c2w,
        "R3": R3_c2w,
        "t3": t3_c2w,
        # Projection matrices (w2c)
        "P1": P1,
        "P2": P2,
        "P3": P3,
        # Image size hints
        "image_size1": np.array([w1, h1], dtype=np.int32),
        "image_size2": np.array([w2, h2], dtype=np.int32),
        "image_size3": np.array([w3, h3], dtype=np.int32),
        "image_size": np.array([w2, h2], dtype=np.int32),
    }

    # Copy over other useful keys if present (safe keep)
    passthrough_keys = [
        "R_w2c1",
        "t_w2c1",
        "R_w2c2",
        "t_w2c2",
        "R_w2c3",
        "t_w2c3",
        "Tw2c1",
        "Tw2c2",
        "Tw2c3",
        "C1",
        "C2",
        "C3",
        "coord_system",
        "units",
        "cam2_intrinsics_policy",
        "squares_x",
        "squares_y",
        "square_length",
        "marker_length",
        "R12",
        "T12",
        "R21",
        "T21",
        "R23",
        "T23",
        "R32",
        "T32",
        "R13",
        "T13",
        "R31",
        "T31",
    ]
    for k in passthrough_keys:
        if k in d and k not in payload:
            payload[k] = d[k]

    np.savez(out_npz, **payload)

    # Quick sanity prints
    print("[OK] wrote:", out_npz)
    print(" keys:", sorted(payload.keys()))
    print(" image_size1/2/3:", payload["image_size1"], payload["image_size2"], payload["image_size3"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()
    convert(args.in_npz, args.out_npz)


if __name__ == "__main__":
    main()
