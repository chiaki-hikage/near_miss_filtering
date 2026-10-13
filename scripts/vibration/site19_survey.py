"""地点 19 (一般道の交差点) の各通過について、交差点の入口・出口と走行状態を整理する。

映像の路面の帯 (自車線の中央付近) で、車線を横切る白線 (停止線・横断歩道の線) が
近づいてくる様子を行と時刻の図にし、線ごとに「カメラの真下に来る時刻」を当てはめる。

  線の見かけの行 y と、カメラから線までの距離 d の関係   y − y_h = K / d   (K = カメラの高さ × 焦点距離)
  車速 v で近づくので   d(t) = ∫_t^{t_c} v dt

y_h (地平線の行) と K は全走行・全線で共通として当てはめる。前輪は
カメラより camera_behind_front_axle_m だけ前にあるので、その分だけ早く線を通過する。

映像からは線の位置 (時刻) だけを取る。路面の凹凸の高さは推定しない。

    uv run python scripts/vibration/site19_survey.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.signal import find_peaks

import find_roll_bumps as F
from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent


def rowtime(seg_dir: Path, rows, cols) -> np.ndarray:
    """フレームごとに、指定した列範囲の行ごとの平均輝度を返す (フレーム数, 行数)。"""
    h = rows[1] - rows[0]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-r", "20", "-i", str(seg_dir / "video.hevc"),
                          "-vf", f"crop=1164:{h}:0:{rows[0]},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, 1164)
    return fr[:, :, cols[0]:cols[1]].astype(float).mean(axis=2)


def line_points(rt: np.ndarray, ft: np.ndarray, rows, t_lo: float, t_hi: float) -> np.ndarray:
    """白線の候補点 (時刻, 行)。行方向の局所的な明るさのピークを拾う。"""
    pts = []
    m = np.flatnonzero((ft >= t_lo) & (ft <= t_hi))
    for i in m:
        prof = rt[i]
        base = np.convolve(prof, np.ones(31) / 31, mode="same")
        resid = prof - base
        pk, _ = find_peaks(resid, height=12, distance=4)
        for p in pk:
            y = rows[0] + p
            if rows[0] + 70 < y < rows[1] - 25:    # 地平線付近とボンネット際は除く
                pts.append((ft[i], y))
    return np.array(pts)


def tc_of(pts: np.ndarray, yh: float, K: float, dist_to: callable) -> np.ndarray:
    """各点について、その線がカメラの真下に来る時刻を求める。"""
    d = K / np.maximum(pts[:, 1] - yh, 1.0)
    return np.array([dist_to(t, dd) for (t, _), dd in zip(pts, d)])


def make_dist_to(t_v: np.ndarray, v: np.ndarray):
    """時刻 t から距離 d を進み終える時刻を返す関数 (車速の積分)。"""
    s = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t_v))])

    def f(t, d):
        target = np.interp(t, t_v, s) + d
        return float(np.interp(target, s, t_v)) if target <= s[-1] else np.nan   # データの外は求めない
    return f


def cluster(tc: np.ndarray, gap: float = 0.08) -> list[np.ndarray]:
    tc = np.where(np.isfinite(tc), tc, np.inf)
    order = np.argsort(tc)
    order = order[np.isfinite(tc[order])]
    if len(order) == 0:
        return []
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if tc[b] - tc[a] > gap:
            groups.append(np.array(cur))
            cur = []
        cur.append(b)
    groups.append(np.array(cur))
    return [g for g in groups if len(g) >= 8]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_pavement.yaml")
    ap.add_argument("--vib-config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--data-root", type=Path, default=REPO / "raw_data/Chunk_1")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/pavement")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    vib = yaml.safe_load(args.vib_config.read_text())
    cam = cfg["camera"]
    site = cfg["site19"]
    args.out.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    configure_japanese_font()
    import matplotlib.pyplot as plt

    refs = find_segments(args.data_root)
    lat0, lon0 = site["center_latlon"]
    passes = []
    for drive, seg in site["passes"]:
        ref = next(r for r in refs if r.drive_id.endswith(drive) and r.index == seg)
        t0 = np.load(ref.path / "processed_log/IMU/accelerometer/t").ravel()[0]
        ft = np.load(ref.path / "global_pose/frame_times") - t0
        pos = np.load(ref.path / "global_pose/frame_positions")
        ll = np.array([F.ecef_to_llh(*p) for p in pos])
        x = (ll[:, 1] - lon0) * 111320 * np.cos(np.deg2rad(lat0))
        y = (ll[:, 0] - lat0) * 110540
        k = int(np.argmin(np.hypot(x, y)))
        tc_, vc = F._load(ref.path / "processed_log/CAN/speed")
        t_v = tc_ - t0
        v = vc.ravel()
        rt = rowtime(ref.path, cam["rows_used"], cam["cols_used"])[:len(ft)]
        pts = line_points(rt, ft, cam["rows_used"], ft[k] - 6, ft[k] + 4)
        passes.append(dict(drive=drive, seg=seg, ref=ref, ft=ft, t_closest=ft[k], x=x, y=y, t_v=t_v, v=v, rt=rt,
                           pts=pts, dist_to=make_dist_to(t_v, v)))

    # 地平線の行 y_h と K を全走行で共通に当てはめる: 同じ線に属する点の t_c のばらつきを最小にする
    # 同じ線の点の t_c がそろうほど良い。データの外に出た点は罰則を与え、
    # 地平線の行は 400〜470、カメラ高さは 1.0〜1.6 m に制限する (退化した解を避ける)
    def spread(theta):
        yh, K = theta
        if not (400 <= yh <= 470 and 1.0 * cam["focal_px"] <= K <= 1.6 * cam["focal_px"]):
            return 1e9
        tot, n = 0.0, 0
        for P in passes:
            if len(P["pts"]) < 10:
                continue
            tc = tc_of(P["pts"], yh, K, P["dist_to"])
            tot += 0.05 * np.sum(~np.isfinite(tc))
            for g in cluster(tc):
                tot += np.sum((tc[g] - np.median(tc[g])) ** 2)
                n += len(g)
        return tot / max(n, 1)

    best = None
    for yh0 in (420.0, 440.0, 460.0):
        for k0 in (1.15, 1.3, 1.45):
            r_ = minimize(spread, x0=[yh0, k0 * cam["focal_px"]], method="Nelder-Mead",
                          options=dict(xatol=0.5, fatol=1e-8, maxiter=300))
            if best is None or r_.fun < best.fun:
                best = r_
    res = best
    yh, K = res.x
    cam_h = K / cam["focal_px"]

    rows = []
    off = cam["camera_behind_front_axle_m"]
    for P in passes:
        tc = tc_of(P["pts"], yh, K, P["dist_to"]) if len(P["pts"]) else np.array([])
        groups = cluster(tc) if len(tc) else []
        lines = sorted(float(np.median(tc[g])) for g in groups)
        # 前輪の通過時刻 = カメラ真下の時刻 − (カメラが前輪より後ろにある距離) / 車速
        vat = lambda t: float(np.interp(t, P["t_v"], P["v"]))
        axle = [t - off / max(vat(t), 0.5) for t in lines]   # 停止中は前輪とカメラの差を 0.5 m/s で近似
        P["lines_axle"] = axle
        # 横断歩道の 2 本線は 1 つの群にまとまる。1 つ目の群を入口 (手前の横断歩道)、2 つ目を出口 (向こうの横断歩道) とする。
        # 3 つ目以降は交差点の先の別の標示
        entry = axle[0] if axle else np.nan
        exit_ = axle[1] if len(axle) >= 2 else np.nan
        P["entry"], P["exit"] = entry, exit_
        s = np.concatenate([[0.0], np.cumsum((P["v"][1:] + P["v"][:-1]) / 2 * np.diff(P["t_v"]))])
        length = float(np.interp(exit_, P["t_v"], s) - np.interp(entry, P["t_v"], s)) if np.isfinite(exit_) else np.nan
        # 自車の左右位置: 交差点の中央付近での GPS の東西方向の位置 (進行方向がほぼ北なので)
        mid = (entry + exit_) / 2 if np.isfinite(exit_) else P["t_closest"]
        j = int(np.clip(np.searchsorted(P["ft"], mid), 0, len(P["ft"]) - 1))
        rows.append(dict(
            drive=P["drive"], segment=P["seg"], gyro_hz=round(F.gyro_rate_hz(P["ref"].path)),
            lines_detected=len(lines), lines_axle_s=";".join(f"{t:.2f}" for t in axle),
            entry_s=round(entry, 2), exit_s=round(exit_, 2) if np.isfinite(exit_) else np.nan,
            crossing_length_m=round(length, 1) if np.isfinite(length) else np.nan,
            speed_kmh=round(vat(mid) * 3.6, 1),
            gps_east_m=round(float(P["x"][j]), 2), gps_north_m=round(float(P["y"][j]), 2),
            seg_end_s=round(float(P["ft"][-1]), 2),
        ))
    df = pd.DataFrame(rows)
    df["gps_east_rel_m"] = (df.gps_east_m - df.gps_east_m.median()).round(2)
    df.to_csv(args.out / "site19_passes.csv", index=False)
    meta = dict(horizon_row=round(float(yh), 1), K=round(float(K), 1), camera_height_m=round(float(cam_h), 3),
                camera_behind_front_axle_m=off, fit_success=bool(res.success))
    (args.out / "site19_camera_fit.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True))
    print(meta)
    print(df.to_string(index=False))

    # 図: 行と時刻の輝度図に、当てはめた線の軌跡と前輪通過時刻を重ねる
    fig, axs = plt.subplots(len(passes), 1, figsize=(12, 3.2 * len(passes)))
    r0, r1 = cam["rows_used"]
    for ax, P in zip(axs, passes):
        m = (P["ft"] >= P["t_closest"] - 6) & (P["ft"] <= P["t_closest"] + 4)
        ax.imshow(P["rt"][m].T, aspect="auto", cmap="gray", extent=[P["ft"][m][0], P["ft"][m][-1], r1, r0])
        for t in P["lines_axle"]:
            ax.axvline(t, color="tab:cyan", lw=0.8)
        if np.isfinite(P["entry"]):
            ax.axvline(P["entry"], color="lime", lw=1.6, label="交差点に入る (前輪)")
        if np.isfinite(P["exit"]):
            ax.axvline(P["exit"], color="red", lw=1.6, label="交差点を出る (前輪)")
        ax.set_title(f"{P['drive']} seg{P['seg']}", fontsize=10, loc="left")
        ax.set_ylabel("画像の行")
        ax.legend(fontsize=8, loc="lower left")
    axs[-1].set_xlabel("セグメント内経過時刻 [s] (IMU 先頭 = 0)   水色 = 白線を前輪が通過する推定時刻")
    fig.suptitle(f"地点 19: 車線を横切る白線の検出 (地平線の行 {yh:.0f}、カメラ高さ {cam_h:.2f} m を全走行共通で当てはめ)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "site19_landmarks.png", dpi=90)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
