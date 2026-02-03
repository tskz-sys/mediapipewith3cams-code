import argparse
import json
import os

import cv2
import numpy as np


def get_inverse_transform(R: np.ndarray, T: np.ndarray):
    return R.T, -R.T @ T


def scale_camera_matrix(K: np.ndarray, dist: np.ndarray, target_w: int, target_h: int):
    orig_w, orig_h = K[0, 2] * 2, K[1, 2] * 2
    sx, sy = target_w / orig_w, target_h / orig_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05:
        return K, dist
    K_new = K.copy()
    K_new[0, 0] *= sx
    K_new[1, 1] *= sy
    K_new[0, 2] *= sx
    K_new[1, 2] *= sy
    return K_new, dist


def load_params_BR(npz_path: str, v1: str, v2: str, v3: str):
    d = np.load(npz_path, allow_pickle=True)

    def get_k(k_new, k_old):
        return d[k_new] if k_new in d else d[k_old]

    K1, D1 = get_k("K1", "cam_matrix1"), get_k("dist1", "dist_coeffs1")
    K2, D2 = get_k("K2", "cam_matrix2"), get_k("dist2", "dist_coeffs2")
    K3, D3 = get_k("K3", "cam_matrix3"), get_k("dist3", "dist_coeffs3")

    def get_wh(v):
        c = cv2.VideoCapture(v)
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        return w, h

    w1, h1 = get_wh(v1)
    K1, D1 = scale_camera_matrix(K1, D1, w1, h1)
    w2, h2 = get_wh(v2)
    K2, D2 = scale_camera_matrix(K2, D2, w2, h2)
    w3, h3 = get_wh(v3)
    K3, D3 = scale_camera_matrix(K3, D3, w3, h3)

    R1_raw = d["R1"]
    t1_raw = d["T1"] if "T1" in d else d["t1"]
    R1, t1 = get_inverse_transform(R1_raw, t1_raw.reshape(3, 1))

    R2 = np.eye(3)
    t2 = np.zeros((3, 1))

    R3_raw = d["R3"]
    t3_raw = d["T3"] if "T3" in d else d["t3"]
    R3, t3 = get_inverse_transform(R3_raw, t3_raw.reshape(3, 1))

    P1_ext = np.hstack([R1, t1])
    P2_ext = np.hstack([R2, t2])
    P3_ext = np.hstack([R3, t3])

    return [(K1, D1, P1_ext), (K2, D2, P2_ext), (K3, D3, P3_ext)]


def undistort_points(kps, K: np.ndarray, dist: np.ndarray):
    if len(kps) == 0:
        return []
    pts = np.array(kps, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=None).reshape(-1, 2)


def triangulate_DLT(Ps, pts):
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


def read_frame(video_path, frame_idx):
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


def pick_point(window_name, frame):
    clicked = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (int(x), int(y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        view = frame.copy()
        msg = "Click goal. ENTER=ok  r=reset  s=skip  q=quit"
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if clicked["pt"] is not None:
            cv2.drawMarker(view, clicked["pt"], (0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)

        cv2.imshow(window_name, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if clicked["pt"] is not None:
                break
        elif key == ord("r"):
            clicked["pt"] = None
        elif key == ord("s"):
            clicked["pt"] = None
            break
        elif key == ord("q"):
            raise KeyboardInterrupt

    cv2.destroyWindow(window_name)
    return clicked["pt"]


def main():
    parser = argparse.ArgumentParser(description="Pick goal position from videos and triangulate to 3D")
    parser.add_argument("--input_dir", type=str, default="../ffmpeg/output", help="video dir")
    parser.add_argument("--calib_npz", type=str, required=True, help="calibration npz")
    parser.add_argument("--game", type=int, required=True, help="game number")
    parser.add_argument("--match", type=int, required=True, help="match number")
    parser.add_argument("--frame", type=int, default=None, help="frame index (default: last frame)")
    parser.add_argument("--out", type=str, default="output/3dposeestimation/goal_position.json", help="output json")
    args = parser.parse_args()

    stem = f"match{args.game}_{args.match}"
    v1 = os.path.join(args.input_dir, f"1{stem}.mp4")
    v2 = os.path.join(args.input_dir, f"2{stem}.mp4")
    v3 = os.path.join(args.input_dir, f"3{stem}.mp4")

    for v in (v1, v2, v3):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Video not found: {v}")

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
    points_2d = {}
    try:
        for cam_id in (1, 2, 3):
            frame, _ = frames[cam_id]
            window = f"Cam{cam_id}: click goal"
            pt = pick_point(window, frame)
            if pt is not None:
                points_2d[cam_id] = pt
    finally:
        cv2.destroyAllWindows()

    if len(points_2d) < 2:
        raise RuntimeError("Need at least 2 camera clicks for triangulation.")

    cam_params = load_params_BR(args.calib_npz, v1, v2, v3)
    Ps = [p[2] for p in cam_params]

    used_Ps = []
    used_pts = []
    cams_used = []
    for cam_id, pt in sorted(points_2d.items()):
        K, D, P = cam_params[cam_id - 1]
        norm_pt = undistort_points([pt], K, D)[0]
        used_Ps.append(P)
        used_pts.append(norm_pt)
        cams_used.append(cam_id)

    X, err = triangulate_DLT(used_Ps, used_pts)
    goal = {"X": float(X[0]), "Y": float(X[1]), "Z": float(X[2])}

    out_data = {
        "goal_pos": goal,
        "frame": int(use_idx),
        "cameras": cams_used,
        "points_2d": {str(k): [float(v[0]), float(v[1])] for k, v in points_2d.items()},
        "reproj_error": float(err),
        "videos": {"1": os.path.basename(v1), "2": os.path.basename(v2), "3": os.path.basename(v3)},
        "calib_npz": args.calib_npz,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=True, indent=2)

    print(f"Saved goal json: {args.out}")
    print(f"Goal 3D: X={goal['X']:.3f} Y={goal['Y']:.3f} Z={goal['Z']:.3f}")
    print(f"Reproj error: {err:.6f}")


if __name__ == "__main__":
    main()
