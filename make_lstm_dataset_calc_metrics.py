#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# -------------------------
#  calc_metrics のロード
# -------------------------
def load_calc_metrics(module_path: str):
    """
    calc_metrics.py をパス指定で読み込む。
    例: --calc_metrics_py /path/to/calc_metrics.py
    """
    module_path = os.path.abspath(module_path)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"calc_metrics.py not found: {module_path}")

    spec = importlib.util.spec_from_file_location("calc_metrics", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import calc_metrics from: {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# -------------------------
#  長さ揃え
# -------------------------
def _fill_nan_for_resample(X: np.ndarray) -> np.ndarray:
    # 列ごとに補間→端はffill/bfill→残りNaNは0
    df = pd.DataFrame(X)
    df = df.interpolate(limit_direction="both")
    df = df.ffill().bfill()
    X2 = df.to_numpy(dtype=np.float32)
    return np.nan_to_num(X2, nan=0.0)

def resample_sequence(X: np.ndarray, target_len: int) -> np.ndarray:
    """
    (T, F) を target_len に時間方向だけ線形補間でリサンプル。
    """
    T, F = X.shape
    if T == target_len:
        return X.astype(np.float32, copy=False)

    X2 = _fill_nan_for_resample(X)

    t_old = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)

    out = np.empty((target_len, F), dtype=np.float32)
    for j in range(F):
        out[:, j] = np.interp(t_new, t_old, X2[:, j]).astype(np.float32)
    return out

def pad_sequence(X: np.ndarray, target_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    (T, F) を末尾0埋めで (target_len, F) にする。maskも返す。
    """
    T, F = X.shape
    if T >= target_len:
        Xc = X[:target_len].astype(np.float32, copy=False)
        mask = np.ones((target_len,), dtype=np.float32)
        return Xc, mask

    Xp = np.zeros((target_len, F), dtype=np.float32)
    Xp[:T] = X.astype(np.float32, copy=False)
    mask = np.zeros((target_len,), dtype=np.float32)
    mask[:T] = 1.0
    return Xp, mask


# -------------------------
#  プレーデータ
# -------------------------
@dataclass
class PlayRow:
    play_id: str
    start_frame: int
    end_frame: int
    label: int


def read_plays_csv(path: str) -> List[PlayRow]:
    """
    plays.csv 必須:
      - play_id
      - start_frame
      - end_frame
      - label (0/1)

    ※「1サンプル = プレー開始〜リリース」なので end_frame は
      “プレー区間の終端（だいたいでOK）” を入れておけば、releaseはcalc_metrics側が検出します。
    """
    df = pd.read_csv(path)
    req = ["play_id", "start_frame", "end_frame", "label"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"plays_csv missing column: {c}")

    out: List[PlayRow] = []
    for _, r in df.iterrows():
        out.append(
            PlayRow(
                play_id=str(r["play_id"]),
                start_frame=int(r["start_frame"]),
                end_frame=int(r["end_frame"]),
                label=int(r["label"]),
            )
        )
    return out


# -------------------------
#  smooth CSV 前処理
# -------------------------
def load_smooth_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["frame", "person_id", "joint", "X", "Y", "Z"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"smooth_csv missing column: {c}")

    # compute_metrics と同じ前提で数値化
    for c in ["X", "Y", "Z", "frame", "person_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["frame", "person_id"])
    df["frame"] = df["frame"].astype(int)
    df["person_id"] = df["person_id"].astype(int)
    return df


def split_ball_person(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # calc_metrics と同じ定義（joint=='ball' or person_id==-1）で分割
    df_ball = df[(df["joint"] == "ball") | (df["person_id"] == -1)].copy()
    df_person = df[(df["joint"] != "ball") & (df["person_id"] != -1)].copy()
    return df_ball, df_person


def get_ball_pos_at(df_ball: pd.DataFrame, frame: int) -> Optional[np.ndarray]:
    row = df_ball[df_ball["frame"] == frame]
    if row.empty:
        return None
    # 先頭の1点を採用（calc_metrics と同様の扱い）
    xyz = row.iloc[0][["X", "Y", "Z"]].to_numpy(dtype=float)
    if not np.isfinite(xyz).all():
        return None
    return xyz


def min_hand_ball_dist(calc, df_person: pd.DataFrame, pid: int, frame: int, ball_pos: Optional[np.ndarray]) -> Tuple[float, int]:
    """
    HAND_JOINTSの点群とball_posの最短距離。
    返り値: (distance, visible_flag)
    visible_flag=1 は「手が取れてて距離計算できた」
    """
    if ball_pos is None:
        return np.nan, 0
    row_p = df_person[(df_person["frame"] == frame) & (df_person["person_id"] == pid)]
    if row_p.empty:
        return np.nan, 0
    hands = calc.get_hand_pos(row_p)  # HAND_JOINTSにフィルタ済み
    if hands is None or len(hands) == 0:
        return np.nan, 0
    d = np.min(np.linalg.norm(hands - ball_pos, axis=1))
    return float(d), 1


def centroid_at(calc, df_person: pd.DataFrame, pid: int, frame: int) -> Tuple[Optional[np.ndarray], int]:
    """
    CORE_JOINTSの重心。返り値: (centroid, visible_flag)
    """
    row_p = df_person[(df_person["frame"] == frame) & (df_person["person_id"] == pid)]
    if row_p.empty:
        return None, 0
    c = calc.calculate_centroid(row_p, calc.CORE_JOINTS)
    if c is None or not np.isfinite(c).all():
        return None, 0
    return np.array(c, dtype=float), 1


# -------------------------
#  1プレー -> (T,F)
# -------------------------
def build_timeseries_features(calc, play_df: pd.DataFrame,
                              hold_thresh: float,
                              release_tail: int,
                              ball_gap_allow: int,
                              goal_json: Optional[str],
                              goal_tail: int,
                              goal_min_frames: int,
                              use_given_start_frame: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """
    calc_metrics.compute_metrics を実行して shooter/defender/release/goal を決め、
    その定義に従った時系列特徴 (T,F) を作る。

    meta には compute_metrics のスカラー出力も入れる。
    """
    goal_pos = None
    goal_err = ""
    if goal_json:
        goal_pos, goal_err = calc.load_goal_json(goal_json)

    metrics = calc.compute_metrics(
        play_df,
        hold_thresh=hold_thresh,
        release_tail=release_tail,
        ball_gap_allow=ball_gap_allow,
        goal_pos=goal_pos,
        goal_tail=goal_tail,
        goal_min_frames=goal_min_frames,
    )

    # エラーが致命的ならスキップ
    if metrics.get("release_frame") is None:
        return np.empty((0, 0), dtype=np.float32), {"errors": metrics.get("errors", []) + [goal_err]}

    shooter_id = metrics.get("shooter_id")
    defender_id = metrics.get("defender_id")
    release_frame = int(metrics["release_frame"])
    goal_pos = metrics.get("goal_pos", None)

    df_ball, df_person = split_ball_person(play_df)

    # 2人のID（avg_spacingと同じ「主に映ってる2人」を採用）
    pids = calc.select_primary_pids(df_person, limit=2)

    # サンプル開始フレーム：plays.csv の start_frame を優先したい場合は use_given_start_frame を使う
    if use_given_start_frame is not None:
        start_frame = int(use_given_start_frame)
    else:
        start_frame = int(metrics.get("start_frame", int(play_df["frame"].min())))

    # 使うフレーム列（開始〜リリース）
    if release_frame < start_frame:
        return np.empty((0, 0), dtype=np.float32), {"errors": ["Error: release_frame < start_frame"]}

    frames = list(range(start_frame, release_frame + 1))

    # 特徴量（calc_metricsに合わせた距離 + 可視性フラグ）
    feat_names = [
        "spacing_m",                 # 2人の重心距離（CORE_JOINTS）
        "shooter_hand_ball_m",       # shooter手-ボール最短距離（HAND_JOINTS）
        "defender_hand_ball_m",      # defender手-ボール最短距離（HAND_JOINTS）
        "shooter_goal_m",            # shooter重心-ゴール距離
        "ball_visible",              # 0/1
        "shooter_hands_visible",     # 0/1
        "defender_hands_visible",    # 0/1
        "both_centroids_visible",    # 0/1（2人の重心が取れた）
    ]

    X = np.full((len(frames), len(feat_names)), np.nan, dtype=np.float32)

    for i, f in enumerate(frames):
        ball_pos = get_ball_pos_at(df_ball, f)
        ball_visible = 1.0 if ball_pos is not None else 0.0

        # spacing
        spacing = np.nan
        both_centroids_visible = 0.0
        if len(pids) >= 2:
            c1, v1 = centroid_at(calc, df_person, pids[0], f)
            c2, v2 = centroid_at(calc, df_person, pids[1], f)
            if c1 is not None and c2 is not None and v1 == 1 and v2 == 1:
                spacing = float(np.linalg.norm(c1 - c2))
                both_centroids_visible = 1.0

        # shooter hand-ball
        shooter_d = np.nan
        shooter_vis = 0.0
        if shooter_id is not None:
            d, v = min_hand_ball_dist(calc, df_person, int(shooter_id), f, ball_pos)
            shooter_d = d
            shooter_vis = float(v)

        # defender hand-ball
        defender_d = np.nan
        defender_vis = 0.0
        if defender_id is not None:
            d, v = min_hand_ball_dist(calc, df_person, int(defender_id), f, ball_pos)
            defender_d = d
            defender_vis = float(v)

        # shooter-goal
        shooter_goal = np.nan
        if goal_pos is not None and shooter_id is not None:
            cS, vS = centroid_at(calc, df_person, int(shooter_id), f)
            if cS is not None and vS == 1:
                shooter_goal = float(np.linalg.norm(cS - np.array(goal_pos, dtype=float)))

        X[i, :] = np.array(
            [spacing, shooter_d, defender_d, shooter_goal,
             ball_visible, shooter_vis, defender_vis, both_centroids_visible],
            dtype=np.float32
        )

    meta = {
        "feature_names": feat_names,
        "start_frame": start_frame,
        "release_frame": release_frame,
        "shooter_id": shooter_id,
        "defender_id": defender_id,
        "goal_pos": goal_pos,
        "avg_spacing": metrics.get("avg_spacing", np.nan),
        "contest_dist": metrics.get("contest_dist", np.nan),
        "offense_goal_dist": metrics.get("offense_goal_dist", np.nan),
        "errors": metrics.get("errors", []) + ([goal_err] if goal_err else []),
        "warnings": metrics.get("warnings", []),
    }
    return X, meta


# -------------------------
#  CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth_csv", required=True, help="smooth CSV（人2人＋ボール3D）")
    ap.add_argument("--calc_metrics_py", required=True, help="calc_metrics.py のパス（今回のcalc_matrics）")

    # 単一プレー運用
    ap.add_argument("--label", type=int, default=None, help="plays.csvを使わない場合のラベル(0/1)")
    ap.add_argument("--play_id", default="play0", help="plays.csvを使わない場合のplay_id")

    # 複数プレー運用（推奨）
    ap.add_argument("--plays_csv", default="", help="plays.csv（play_id,start_frame,end_frame,label）")

    # calc_metrics のパラメータ
    ap.add_argument("--hold_thresh", type=float, default=0.4, help="保持判定距離(m)")
    ap.add_argument("--release_tail", type=int, default=0, help="0=後半から探索, >0=最後Nボールフレームだけ")
    ap.add_argument("--ball_gap_allow", type=int, default=15, help="ボール欠損ギャップ許容(frame)")
    ap.add_argument("--goal_json", default="", help="ゴール位置json（手動指定）")
    ap.add_argument("--goal_tail", type=int, default=10, help="ゴール推定に使う終盤ボールフレーム数")
    ap.add_argument("--goal_min_frames", type=int, default=3, help="ゴール推定に必要な最小フレーム数")

    # サンプル開始を plays.csv の start_frame に合わせたいか（基本True推奨）
    ap.add_argument("--use_play_start", action="store_true",
                    help="plays.csvのstart_frameをサンプル開始に強制する（推奨）")

    # 長さ揃え
    ap.add_argument("--mode", choices=["resample", "pad"], default="resample",
                    help="長さ統一: resample=補間, pad=0埋め+mask")
    ap.add_argument("--seq_len", type=int, default=120, help="固定長T")
    ap.add_argument("--out_npz", required=True, help="出力npz")
    ap.add_argument("--debug_csv", default="", help="デバッグ用：整形後先頭数サンプルをCSV出力")
    ap.add_argument("--debug_n", type=int, default=3, help="デバッグ出力サンプル数")
    return ap.parse_args()


def main():
    args = parse_args()

    calc = load_calc_metrics(args.calc_metrics_py)
    df_all = load_smooth_csv(args.smooth_csv)

    # plays の解釈
    plays: List[PlayRow] = []
    single_mode = False

    if args.plays_csv:
        plays = read_plays_csv(args.plays_csv)
    else:
        # 単一プレーとして扱う（label必須）
        if args.label is None:
            raise ValueError("plays.csv を使わない場合は --label 0/1 を指定してください。")
        plays = [PlayRow(args.play_id, int(df_all["frame"].min()), int(df_all["frame"].max()), int(args.label))]
        single_mode = True

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    mask_list: List[np.ndarray] = []
    play_ids: List[str] = []

    # メタ（スカラー）
    meta_release: List[int] = []
    meta_start: List[int] = []
    meta_shooter: List[int] = []
    meta_defender: List[int] = []
    meta_avg_spacing: List[float] = []
    meta_contest: List[float] = []
    meta_off_goal: List[float] = []
    meta_goal_pos: List[List[float]] = []
    meta_errors: List[str] = []
    meta_warnings: List[str] = []

    feature_names: Optional[List[str]] = None
    debug_rows = []

    for pr in plays:
        # plays.csv の区間でまず切る（releaseはcalc_metrics側が検出）
        seg = df_all[(df_all["frame"] >= pr.start_frame) & (df_all["frame"] <= pr.end_frame)].copy()
        seg = seg.sort_values("frame")

        if seg.empty:
            continue

        use_start = pr.start_frame if (args.use_play_start or single_mode) else None

        X_raw, meta = build_timeseries_features(
            calc, seg,
            hold_thresh=args.hold_thresh,
            release_tail=args.release_tail,
            ball_gap_allow=args.ball_gap_allow,
            goal_json=(args.goal_json if args.goal_json else None),
            goal_tail=args.goal_tail,
            goal_min_frames=args.goal_min_frames,
            use_given_start_frame=use_start,
        )

        # 失敗はスキップ
        if X_raw.size == 0:
            meta_errors.append(f"{pr.play_id}: " + "; ".join(meta.get("errors", [])))
            continue

        # feature_names の整合
        if feature_names is None:
            feature_names = meta["feature_names"]
        else:
            if feature_names != meta["feature_names"]:
                raise ValueError("feature_names mismatch across plays.")

        # 長さ統一
        if args.mode == "resample":
            X_fix = resample_sequence(X_raw, args.seq_len)
            mask = np.ones((args.seq_len,), dtype=np.float32)
        else:
            X_fix, mask = pad_sequence(X_raw, args.seq_len)

        X_list.append(X_fix)
        y_list.append(pr.label)
        mask_list.append(mask)
        play_ids.append(pr.play_id)

        meta_release.append(int(meta["release_frame"]))
        meta_start.append(int(meta["start_frame"]))
        meta_shooter.append(int(meta["shooter_id"]) if meta["shooter_id"] is not None else -1)
        meta_defender.append(int(meta["defender_id"]) if meta["defender_id"] is not None else -1)

        meta_avg_spacing.append(float(meta.get("avg_spacing", np.nan)))
        meta_contest.append(float(meta.get("contest_dist", np.nan)) if meta.get("contest_dist") is not None else np.nan)
        meta_off_goal.append(float(meta.get("offense_goal_dist", np.nan)) if meta.get("offense_goal_dist") is not None else np.nan)

        gp = meta.get("goal_pos", None)
        if gp is None:
            meta_goal_pos.append([np.nan, np.nan, np.nan])
        else:
            meta_goal_pos.append([float(gp[0]), float(gp[1]), float(gp[2])])

        meta_warnings.append(f"{pr.play_id}: " + "; ".join(meta.get("warnings", [])))
        # meta_errors は build_timeseries_features が空返しのときのみ追加してる

        # デバッグ
        if args.debug_csv and len(debug_rows) < args.debug_n:
            # 各プレー先頭5ステップだけ出す
            for t in range(min(5, args.seq_len)):
                row = {"play_id": pr.play_id, "t": t, "label": pr.label}
                for j, name in enumerate(feature_names):
                    row[name] = float(X_fix[t, j])
                debug_rows.append(row)

    if len(X_list) == 0:
        raise RuntimeError("No valid plays produced. plays.csv区間・ボール検出・人物検出を確認してください。")

    X = np.stack(X_list, axis=0)  # (N,T,F)
    y = np.array(y_list, dtype=np.int64)
    mask = np.stack(mask_list, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        X=X,
        y=y,
        mask=mask,
        feature_names=np.array(feature_names, dtype=object),
        play_ids=np.array(play_ids, dtype=object),
        seq_len=np.int64(args.seq_len),
        mode=np.array(args.mode, dtype=object),

        meta_start=np.array(meta_start, dtype=np.int64),
        meta_release=np.array(meta_release, dtype=np.int64),
        meta_shooter=np.array(meta_shooter, dtype=np.int64),
        meta_defender=np.array(meta_defender, dtype=np.int64),
        meta_goal_pos=np.array(meta_goal_pos, dtype=np.float32),

        meta_avg_spacing=np.array(meta_avg_spacing, dtype=np.float32),
        meta_contest_dist=np.array(meta_contest, dtype=np.float32),
        meta_offense_goal_dist=np.array(meta_off_goal, dtype=np.float32),

        meta_errors=np.array(meta_errors, dtype=object),
        meta_warnings=np.array(meta_warnings, dtype=object),
    )

    if args.debug_csv:
        pd.DataFrame(debug_rows).to_csv(args.debug_csv, index=False)

    print("Saved:", args.out_npz)
    print("X:", X.shape, "y:", y.shape, "mask:", mask.shape)
    print("Features:", len(feature_names))
    if meta_errors:
        print("Some plays failed:")
        for e in meta_errors[:10]:
            print(" -", e)


if __name__ == "__main__":
    main()
