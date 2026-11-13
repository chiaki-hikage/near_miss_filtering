#!/usr/bin/env python3
"""Ego Motion で拾った横方向の動きの候補を、前方カメラの映像と同じ時間軸で並べて確かめる。

  uv run --with pyarrow --with scipy --extra viz python scripts/physical_ai_av/video_event_review.py \
      --out out/physical_ai_av/video_review

入力:
  data/physical_ai_av/labels/egomotion/egomotion.chunk_0000.zip               (Ego Motion)
  data/physical_ai_av/camera/camera_front_wide_120fov/chunk_0000/             (fetch_clip_members.py で取ったもの)
  out/physical_ai_av/egomotion_check/events.csv                               (egomotion_lateral_check.py の候補)

やること:
  1. 時刻の対応の確認: 映像の画面の横の流れ (隣り合うフレームの位相相関) とヨーレートの相互相関から、
     映像の時刻と Ego Motion の時刻のずれを測る
  2. 候補 5 件 (下の EVENTS) について、映像・車速・ヨーレート (0.3〜3 Hz)・横加速度・軌跡・候補の区間を
     同じ時間軸で並べた動画 (mp4) と一覧の図 (png) を作る

出力 (<out>/): alignment.csv, events.csv, <event>.mp4, <event>_sheet.png
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import egomotion_lateral_check as E  # noqa: E402

CAM = Path("data/physical_ai_av/camera/camera_front_wide_120fov/chunk_0000")
SMALL = (320, 180)            # 時刻の対応を測るときの縮小サイズ
VIEW = (960, 540)             # 動画に載せる映像の大きさ
FPS_OUT = 10                  # 出力する動画のフレームレート (映像の 3 枚に 1 枚)
PAD = 4.0                     # 候補の区間の前後に足す時間 [s]

# 候補 (Phase 2 の events.csv から選んだもの。路面入力は横加速度の 5〜20 Hz 成分の山から決める)
EVENTS = [
    dict(name="E1_lane_change", clip="db2c5fbe", kind="車線変更と判断された事象", src="events.csv 上位 1"),
    dict(name="E2_sway", clip="0931a850", kind="左右に振れた事象", src="events.csv の左右の振れ (唯一の 1 件)"),
    dict(name="E3_single", clip="5e9a00f1", kind="単発のヨーレート変動", src="events.csv 上位 4"),
    dict(name="E4_road_input", clip="5e9a00f1", kind="路面入力が疑われる事象", src="横加速度の 5〜20 Hz 成分の山", road=True),
    dict(name="E5_straight", clip="5c8a7587", kind="通常の直線走行 (比較)", src="events.csv で振れが小さく横移動 < 1 m の窓"),
]


def frames(mp4: Path, size: tuple[int, int], gray: bool) -> np.ndarray:
    """ffmpeg で全フレームを縮小して読む。"""
    pix, ch = ("gray", 1) if gray else ("rgb24", 3)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4), "-vf", f"scale={size[0]}:{size[1]}",
                          "-f", "rawvideo", "-pix_fmt", pix, "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size[1], size[0], ch).squeeze()


def image_yaw_shift(gray: np.ndarray) -> np.ndarray:
    """隣り合うフレームの横方向のずれ [px] (位相相関)。画面の上半分 (遠くの景色) だけを使う。"""
    h, w = gray.shape[1:]
    band = gray[:, int(h * 0.25):int(h * 0.55), :].astype(float)
    win = np.hanning(w)[None, :]
    out = [0.0]
    for a, b in zip(band[:-1], band[1:]):
        A = np.fft.fft(a * win, axis=1); B = np.fft.fft(b * win, axis=1)
        X = (A * np.conj(B)).sum(axis=0)
        r = np.fft.ifft(X / (np.abs(X) + 1e-9)).real
        k = int(np.argmax(r)); k = k - w if k > w // 2 else k
        # 放物線でサブピクセルに
        y0, y1, y2 = r[(k - 1) % w], r[k % w], r[(k + 1) % w]
        out.append(k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12))
    return np.array(out)


def alignment(c: dict, ts: np.ndarray, shift: np.ndarray) -> dict:
    """画面の横の流れの速さ と ヨーレート の相互相関で、時刻のずれを測る。"""
    from scipy.signal import butter, filtfilt
    tt = c["t"] + c["t0"]
    ok = np.r_[False, np.diff(ts) > 0]
    rate = np.zeros_like(shift); rate[ok] = shift[ok] / np.diff(ts, prepend=ts[0])[ok]
    b, a = butter(2, 3 / 15)
    img = filtfilt(b, a, rate)
    lags = np.arange(-300, 301, 5)
    cc = np.array([np.corrcoef(img[5:-5], np.interp(ts + l / 1e3, tt, c["r"])[5:-5])[0, 1] for l in lags])
    i = int(np.argmax(np.abs(cc)))
    near = lags[np.abs(cc) >= np.abs(cc[i]) - 0.005]          # 最大から 0.005 以内の幅 (ずれの推定の幅)
    yr0 = np.interp(ts, tt, c["r"])
    k = np.polyfit(yr0[5:-5], img[5:-5], 1)[0]
    sd = float(np.degrees(yr0.std()))
    ok = abs(cc[i]) >= 0.9 and sd >= 2.0 and 0 < i < len(lags) - 1
    return dict(lag_ms=int(lags[i]), lag_range_ms=f"{near.min()}〜{near.max()}", corr_at_lag=float(cc[i]),
                corr_at_0=float(cc[lags == 0][0]), px_per_rad=float(k), yaw_rate_sd_deg=sd,
                判定="確認できた" if ok else "判定不能 (向きの変化が小さく相関が弱い)")


def road_input_window(c: dict) -> tuple[float, float]:
    """横加速度の 5〜20 Hz 成分の包絡線 (0.5 s 移動 RMS) が最大になる時刻の前後 1 s。"""
    hf = E.bp(c["ab"][:, 1], 5, 20)
    env = np.sqrt(pd.Series(hf ** 2).rolling(50, center=True, min_periods=1).mean().to_numpy())
    i = int(np.argmax(env)); t = c["t"][i] + c["t0"]
    return t - 1.0, t + 1.0


def render(ev: dict, c: dict, rgb: np.ndarray, ts: np.ndarray, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    tt = c["t"] + c["t0"]
    t0, t1 = ev["t_start"] - PAD, ev["t_end"] + PAD
    t0, t1 = max(t0, tt[0], ts[0]), min(t1, tt[-1], ts[-1])
    w = (tt >= t0) & (tt <= t1)
    we = (tt >= ev["t_start"]) & (tt <= ev["t_end"])
    yaw_bp = np.degrees(E.bp(c["r"], 0.3, 3))
    mid = int(np.argmin(np.abs(tt - (ev["t_start"] + ev["t_end"]) / 2)))
    ang = c["yaw"][mid]
    p = c["pos"][:, :2] - c["pos"][mid, :2]
    lon = np.cos(ang) * p[:, 0] + np.sin(ang) * p[:, 1]; lat = -np.sin(ang) * p[:, 0] + np.cos(ang) * p[:, 1]
    series = [(c["sp"] * 3.6, "車速 [km/h]"), (yaw_bp, "ヨーレート 0.3–3 Hz [deg/s]"), (c["ab"][:, 1], "横加速度 (車体) [m/s²]")]
    fi = [i for i in range(len(ts)) if t0 <= ts[i] <= t1][::30 // FPS_OUT]

    def draw(i_frame: int, fig):
        fig.clf()
        gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.18)
        axv = fig.add_subplot(gs[:3, 0]); axv.imshow(rgb[i_frame]); axv.axis("off")
        tc = ts[i_frame]
        inside = ev["t_start"] <= tc <= ev["t_end"]
        axv.set_title(f"{ev['kind']}  映像の時刻 {tc:6.2f} s" + ("  [候補の区間]" if inside else ""), fontsize=10,
                      color="#B03A2E" if inside else "#222222")
        for k, (y, yl) in enumerate(series):
            ax = fig.add_subplot(gs[k, 1])
            ax.plot(tt[w], y[w], color="#2F6FB0", lw=1)
            ax.axvspan(ev["t_start"], ev["t_end"], color="#F6DADA", zorder=0)
            ax.axvline(tc, color="#222222", lw=1)
            ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=7); ax.set_xlim(t0, t1); ax.grid(alpha=0.3)
        ax = fig.add_subplot(gs[3, 1])
        ax.plot(lon[w], lat[w], color="#9CA3AF", lw=1)
        ax.plot(lon[we], lat[we], color="#B03A2E", lw=2)
        j = int(np.argmin(np.abs(tt - tc)))
        ax.plot(lon[j], lat[j], "o", color="#222222", ms=4)
        ax.set_xlabel("前後 [m] (候補の中央の向き)", fontsize=7); ax.set_ylabel("横位置 [m]", fontsize=7)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        axs = fig.add_subplot(gs[3, 0]); axs.axis("off")
        axs.text(0, 0.9, f"{ev['name']}  clip {c['clip'][:8]}\n候補の区間 {ev['t_start']:.2f}〜{ev['t_end']:.2f} s (赤)\n"
                         f"出典: {ev['src']}\n縦線 = 映像の時刻 (Ego Motion の時刻と同じ時計。ずれは alignment.csv)",
                 fontsize=8, va="top")

    fig = plt.figure(figsize=(13.0, 7.5), dpi=96)            # 1248×720 (H.264 は偶数の画素数が要る)
    canvas = FigureCanvasAgg(fig)
    W, H = canvas.get_width_height()
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                            "-r", str(FPS_OUT), "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                            str(out / f"{ev['name']}.mp4")], stdin=subprocess.PIPE)
    for i in fi:
        draw(i, fig); canvas.draw()
        enc.stdin.write(np.asarray(canvas.buffer_rgba()).tobytes())
    enc.stdin.close(); enc.wait()
    plt.close(fig)

    # 一覧の図: 候補の区間の前・中・後の 6 枚と時系列
    picks = np.linspace(0, len(fi) - 1, 6).astype(int)
    fig, axs = plt.subplots(2, 3, figsize=(13.33, 5.2))
    for ax, k in zip(axs.flat, picks):
        i = fi[k]; ax.imshow(rgb[i]); ax.axis("off")
        inside = ev["t_start"] <= ts[i] <= ev["t_end"]
        ax.set_title(f"{ts[i]:.2f} s" + (" [候補]" if inside else ""), fontsize=9, color="#B03A2E" if inside else "#222222")
    fig.suptitle(f"{ev['name']}  {ev['kind']}  clip {c['clip'][:8]}  候補 {ev['t_start']:.2f}〜{ev['t_end']:.2f} s", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"{ev['name']}_sheet.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(4, 1, figsize=(10, 6.5), sharex=False)
    for ax, (y, yl) in zip(axs[:3], series):
        ax.plot(tt[w], y[w], color="#2F6FB0", lw=1); ax.axvspan(ev["t_start"], ev["t_end"], color="#F6DADA", zorder=0)
        for k in picks:
            ax.axvline(ts[fi[k]], color="#9CA3AF", lw=0.8, ls=":")
        ax.set_ylabel(yl, fontsize=8); ax.set_xlim(t0, t1); ax.grid(alpha=0.3)
    axs[2].set_xlabel("クリップ開始からの時間 [s] (点線 = 上の 6 枚の時刻)")
    axs[3].plot(lon[w], lat[w], color="#9CA3AF"); axs[3].plot(lon[we], lat[we], color="#B03A2E", lw=2)
    axs[3].set_xlabel("前後 [m]"); axs[3].set_ylabel("横位置 [m]", fontsize=8); axs[3].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"{ev['name']}_signals.png", dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out/physical_ai_av/video_review"))
    ap.add_argument("--events", type=Path, default=Path("out/physical_ai_av/egomotion_check/events.csv"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(E.EGO)
    names = {n.split(".")[0]: n for n in zf.namelist()}
    cand = pd.read_csv(args.events)
    clips, rows, align = {}, [], []
    for ev in EVENTS:
        full = next(k for k in names if k.startswith(ev["clip"]))
        if full not in clips:
            c = E.load_clip(zf, names[full])
            ts = pd.read_parquet(next(CAM.glob(f"{full}*.timestamps.parquet")))["timestamp"].to_numpy() / 1e6
            mp4 = next(CAM.glob(f"{full}*.mp4"))
            gray = frames(mp4, SMALL, gray=True)
            n = min(len(gray), len(ts))
            a = alignment(c, ts[:n], image_yaw_shift(gray[:n]))
            align.append(dict(clip=full, frames_video=len(gray), frames_timestamps=len(ts), **a))
            clips[full] = (c, ts[:n], mp4)
        c, ts, mp4 = clips[full]
        if ev.get("road"):
            t_s, t_e = road_input_window(c)
        else:
            g = cand[cand["clip"] == full]
            if ev["name"] == "E5_straight":
                g = g[(g.lat_ctx < 1.0) & (g.flips == 0)].sort_values("r_bp")
            elif ev["name"] == "E2_sway":
                g = g[g.flips >= 2]
            r = g.sort_values("r_bp", ascending=False).iloc[0] if ev["name"] != "E5_straight" else g.iloc[0]
            t_s, t_e = float(r.t_start), float(r.t_end)
        e = dict(ev, clip=full, t_start=round(t_s, 2), t_end=round(t_e, 2))
        rows.append({k: v for k, v in e.items() if k != "road"})
        if not args.no_video:
            rgb = frames(mp4, VIEW, gray=False)[: len(ts)]
            render(e, c, rgb, ts, args.out)
        print(e["name"], full[:8], e["t_start"], e["t_end"])
    pd.DataFrame(align).round(4).to_csv(args.out / "alignment.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out / "events.csv", index=False)
    print(pd.DataFrame(align).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
