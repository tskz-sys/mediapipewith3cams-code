import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def get_inverse_transform(R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return R.T, -R.T @ T


def projection_error(P_ref: np.ndarray, K: np.ndarray, R: np.ndarray, T: np.ndarray) -> float:
    P_est = K @ np.hstack([R, T.reshape(3, 1)])
    denom = float(np.linalg.norm(P_ref))
    if denom < 1e-9:
        return float("inf")
    return float(np.linalg.norm(P_est - P_ref) / denom)


def choose_extrinsics(
    data: np.lib.npyio.NpzFile,
    cam_idx: int,
    K_raw: np.ndarray,
    R_raw: np.ndarray,
    t_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str, Optional[float], Optional[float]]:
    t_raw = t_raw.reshape(3, 1)
    P_key = f"P{cam_idx}"
    if P_key in data:
        err_direct = projection_error(data[P_key], K_raw, R_raw, t_raw)
        R_inv, t_inv = get_inverse_transform(R_raw, t_raw)
        err_inv = projection_error(data[P_key], K_raw, R_inv, t_inv)
        if err_direct <= err_inv:
            return R_raw, t_raw, "direct", err_direct, err_inv
        return R_inv, t_inv, "inverse", err_direct, err_inv
    R_inv, t_inv = get_inverse_transform(R_raw, t_raw)
    return R_inv, t_inv, "inverse", None, None


def scale_camera_matrix(
    K: np.ndarray,
    dist: np.ndarray,
    target_w: int,
    target_h: int,
    orig_size: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    if orig_size is None:
        return K, dist
    orig_w, orig_h = orig_size
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05:
        return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx
    K_new[1, 1] *= sy
    K_new[0, 2] *= sx
    K_new[1, 2] *= sy
    return K_new, dist


def load_params_BR(
    npz_path: str,
    v1: str,
    v2: str,
    v3: str,
    invert_both: bool = False,
    invert_cam1: bool = False,
    invert_cam3: bool = False,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    d = np.load(npz_path, allow_pickle=True)

    def get_k(k_new, k_old):
        return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1", "cam_matrix1"), get_k("dist1", "dist_coeffs1")
    K2, D2 = get_k("K2", "cam_matrix2"), get_k("dist2", "dist_coeffs2")
    K3, D3 = get_k("K3", "cam_matrix3"), get_k("dist3", "dist_coeffs3")

    K1_raw = K1.copy()
    K3_raw = K3.copy()

    def get_calib_size(cam_idx: int) -> Optional[Tuple[int, int]]:
        for key in (f"image_size{cam_idx}", "image_size"):
            if key in d:
                size = d[key]
                if size is None:
                    continue
                w, h = int(size[0]), int(size[1])
                if w > 0 and h > 0:
                    return (w, h)
        return None

    def get_wh(v):
        c = cv2.VideoCapture(v)
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        return w, h

    w1, h1 = get_wh(v1)
    K1, D1 = scale_camera_matrix(K1, D1, w1, h1, get_calib_size(1))
    w2, h2 = get_wh(v2)
    K2, D2 = scale_camera_matrix(K2, D2, w2, h2, get_calib_size(2))
    w3, h3 = get_wh(v3)
    K3, D3 = scale_camera_matrix(K3, D3, w3, h3, get_calib_size(3))

    R1_raw = d["R1"]
    t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1, mode1, err_d1, err_i1 = choose_extrinsics(d, 1, K1_raw, R1_raw, t1_raw)

    R2 = np.eye(3)
    t2 = np.zeros((3, 1))

    R3_raw = d["R3"]
    t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3, mode3, err_d3, err_i3 = choose_extrinsics(d, 3, K3_raw, R3_raw, t3_raw)

    if err_d1 is not None and err_i1 is not None:
        print(f"[calib] cam1 extrinsics: {mode1} (direct={err_d1:.3g}, inv={err_i1:.3g})")
    if err_d3 is not None and err_i3 is not None:
        print(f"[calib] cam3 extrinsics: {mode3} (direct={err_d3:.3g}, inv={err_i3:.3g})")

    if invert_both:
        invert_cam1 = True
        invert_cam3 = True
        print("[calib] invert_both applied")

    if invert_cam1:
        R1, t1 = get_inverse_transform(R1, t1)
        print("[calib] invert_cam1 applied")
    if invert_cam3:
        R3, t3 = get_inverse_transform(R3, t3)
        print("[calib] invert_cam3 applied")

    P1_ext = np.hstack([R1, t1])
    P2_ext = np.hstack([R2, t2])
    P3_ext = np.hstack([R3, t3])

    return [(K1, D1, P1_ext), (K2, D2, P2_ext), (K3, D3, P3_ext)], [(R1, t1), (R2, t2), (R3, t3)]


def undistort_points(kps: List[Tuple[float, float]], K: np.ndarray, dist: np.ndarray) -> List[np.ndarray]:
    if len(kps) == 0:
        return []
    pts = np.array(kps, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=None).reshape(-1, 2).tolist()


def triangulate_DLT(Ps: List[np.ndarray], pts: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    A = []
    for P, (x, y) in zip(Ps, pts):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    X = X[:3] / X[3]

    errs = []
    for P, (x, y) in zip(Ps, pts):
        xh = P @ np.hstack([X, 1.0])
        if abs(xh[2]) < 1e-9:
            errs.append(1000.0)
            continue
        xp, yp = xh[0] / xh[2], xh[1] / xh[2]
        errs.append((xp - x) ** 2 + (yp - y) ** 2)
    return X, float(np.mean(errs))


def read_frame(video_path: str, frame_idx: Optional[int]):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Failed to read total frames: {video_path}")

    if frame_idx is None:
        frame_idx = total - 1
    elif frame_idx < 0:
        frame_idx = max(total + frame_idx, 0)

    frame_idx = min(max(frame_idx, 0), total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame, total, frame_idx


def _draw_points(view, points: List[Tuple[int, int]]):
    for idx, pt in enumerate(points):
        cv2.circle(view, pt, 5, (0, 0, 255), -1)
        cv2.putText(
            view,
            str(idx + 1),
            (pt[0] + 6, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    if len(points) >= 2:
        cv2.polylines(view, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2)


def pick_poly_points(window_name: str, frame: np.ndarray, min_points: int = 3) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        view = frame.copy()
        msg = f"Click boundary points. ENTER=ok  u=undo  r=reset  q=quit  ({len(points)}/{min_points}+)"
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _draw_points(view, points)
        cv2.imshow(window_name, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if len(points) >= min_points:
                break
        elif key == ord("u"):
            if points:
                points.pop()
        elif key == ord("r"):
            points = []
        elif key == ord("q"):
            raise KeyboardInterrupt

    cv2.destroyWindow(window_name)
    return points


def fit_plane(points: np.ndarray) -> Dict[str, List[float]]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    u_axis = vt[0]
    v_axis = np.cross(normal, u_axis)
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)
    normal = normal / np.linalg.norm(normal)
    return {
        "origin": center.tolist(),
        "u": u_axis.tolist(),
        "v": v_axis.tolist(),
        "n": normal.tolist(),
    }


def to_uv(points: np.ndarray, plane: Dict[str, List[float]]) -> np.ndarray:
    origin = np.array(plane["origin"], dtype=float)
    u_axis = np.array(plane["u"], dtype=float)
    v_axis = np.array(plane["v"], dtype=float)
    rel = points - origin
    u = rel @ u_axis
    v = rel @ v_axis
    return np.stack([u, v], axis=1)


def convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 2:
        return points
    pts = np.array(sorted(points.tolist()))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull)


def parse_pairs(text: str) -> List[Tuple[int, int]]:
    pairs = []
    for token in text.split(","):
        token = token.strip()
        if len(token) != 2:
            continue
        a, b = int(token[0]), int(token[1])
        if a == b:
            continue
        pairs.append((a, b))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Pick court boundary points and build union polygon.")
    parser.add_argument("--input_dir", type=str, required=True, help="video dir")
    parser.add_argument("--calib_npz", type=str, required=True, help="calibration npz")
    parser.add_argument("--game", type=int, required=True, help="game number")
    parser.add_argument("--match", type=int, required=True, help="match number")
    parser.add_argument("--frame", type=int, default=None, help="frame index (default: last frame)")
    parser.add_argument("--pairs", type=str, default="12,23,13", help="camera pairs, e.g. 12,23,13")
    parser.add_argument("--min_points", type=int, default=4, help="minimum points per pair")
    parser.add_argument("--invert_both", action="store_true", help="invert cam1+cam3 extrinsics")
    parser.add_argument("--invert_cam1", action="store_true", help="invert cam1 extrinsics")
    parser.add_argument("--invert_cam3", action="store_true", help="invert cam3 extrinsics")
    parser.add_argument("--out", type=str, required=True, help="output json")
    args = parser.parse_args()

    stem = f"match{args.game}_{args.match}"
    v1 = os.path.join(args.input_dir, f"1{stem}.mp4")
    v2 = os.path.join(args.input_dir, f"2{stem}.mp4")
    v3 = os.path.join(args.input_dir, f"3{stem}.mp4")
    videos = {1: v1, 2: v2, 3: v3}

    for v in (v1, v2, v3):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Video not found: {v}")

    pairs = parse_pairs(args.pairs)
    if not pairs:
        raise ValueError("No valid camera pairs provided.")

    frames = {}
    totals = []
    for cam_id, v in ((1, v1), (2, v2), (3, v3)):
        frame, total, use_idx = read_frame(v, args.frame)
        frames[cam_id] = (frame, use_idx)
        totals.append(total)

    min_total = min(totals)
    if args.frame is None:
        use_idx = min_total - 1
    else:
        use_idx = args.frame
        if use_idx < 0:
            use_idx = max(min_total + use_idx, 0)
        if use_idx >= min_total:
            use_idx = min_total - 1

    if use_idx != frames[1][1]:
        for cam_id, v in ((1, v1), (2, v2), (3, v3)):
            frame, _, _ = read_frame(v, use_idx)
            frames[cam_id] = (frame, use_idx)

    print(f"Frame index: {use_idx}")

    points_by_pair: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
    try:
        for cam_a, cam_b in pairs:
            name = f"cam{cam_a}{cam_b}"
            while True:
                frame_a, _ = frames[cam_a]
                frame_b, _ = frames[cam_b]
                pts_a = pick_poly_points(f"{name}: Cam{cam_a}", frame_a, min_points=args.min_points)
                pts_b = pick_poly_points(f"{name}: Cam{cam_b}", frame_b, min_points=args.min_points)
                if len(pts_a) != len(pts_b):
                    print(f"[{name}] point count mismatch ({len(pts_a)} vs {len(pts_b)}). redo.")
                    continue
                points_by_pair[name] = {cam_a: pts_a, cam_b: pts_b}
                break
    finally:
        cv2.destroyAllWindows()

    cam_params, _ = load_params_BR(
        args.calib_npz,
        v1,
        v2,
        v3,
        invert_both=args.invert_both,
        invert_cam1=args.invert_cam1,
        invert_cam3=args.invert_cam3,
    )

    all_points_3d = []
    pair_entries = []
    for name, pts in points_by_pair.items():
        cams = sorted(pts.keys())
        P_list = []
        pts_list = []
        for cam_id in cams:
            K, D, P = cam_params[cam_id - 1]
            P_list.append(P)
            und = undistort_points(pts[cam_id], K, D)
            pts_list.append(und)
        per_pair_3d = []
        per_pair_err = []
        for idx in range(len(pts_list[0])):
            used_Ps = []
            used_pts = []
            for cam_idx in range(len(cams)):
                used_Ps.append(P_list[cam_idx])
                used_pts.append(pts_list[cam_idx][idx])
            X, err = triangulate_DLT(used_Ps, used_pts)
            per_pair_3d.append(X)
            per_pair_err.append(err)
        per_pair_3d = np.array(per_pair_3d)
        all_points_3d.append(per_pair_3d)
        pair_entries.append(
            {
                "name": name,
                "cams": cams,
                "points_3d": per_pair_3d.tolist(),
                "reproj_error": [float(e) for e in per_pair_err],
            }
        )
        print(f"[{name}] points: {len(per_pair_3d)}  mean reproj: {float(np.mean(per_pair_err)):.6f}")

    all_points_3d_arr = np.vstack(all_points_3d)
    plane = fit_plane(all_points_3d_arr)

    polygons = []
    for entry in pair_entries:
        pts = np.array(entry["points_3d"], dtype=float)
        uv = to_uv(pts, plane)
        polygons.append(
            {
                "name": entry["name"],
                "cams": entry["cams"],
                "plane": plane,
                "polygon_uv": uv.tolist(),
                "reproj_error": entry["reproj_error"],
            }
        )

    uv_all = to_uv(all_points_3d_arr, plane)
    hull = convex_hull(uv_all)
    polygon_uv = hull.tolist()

    out_data = {
        "frame": int(use_idx),
        "plane": plane,
        "polygon_uv": polygon_uv,
        "polygons": polygons,
        "points_3d": all_points_3d_arr.tolist(),
        "videos": {"1": os.path.basename(v1), "2": os.path.basename(v2), "3": os.path.basename(v3)},
        "calib_npz": args.calib_npz,
        "pairs": [f"{a}{b}" for a, b in pairs],
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=True, indent=2)

    print(f"Saved court json: {args.out}")


if __name__ == "__main__":
    main()
