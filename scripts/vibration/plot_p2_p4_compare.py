#!/usr/bin/env python3
"""1 イベントについて、P2 (前輪 → 後輪の時間差 ≈ L/v) と P4 (3〜8 Hz の波形の一致) が
図から読み取れる比較図を描く。プレゼン用。

判定と同じ関数 (validate_sites_with_fullcar.evaluate_event・metrics・band_corr と同じ計算) を使うので、
図に書き込む相関・時間差は event_physics_validation.csv の値と一致する。

  uv run python scripts/vibration/plot_p2_p4_compare.py --event E0474 --roots raw_data/Chunk_1 \
      --out out/vibration/near_miss/report/site007_E0474_p2_p4.png

段:
  1. 上下加速度 3〜8 Hz: 実測とモデル。モデルは相関が最大になる時間ずれ (±lag_search) だけずらし、
     上下の 3〜8 Hz の RMS を実測に合わせる共通の倍率を掛ける (形の比較。高さの推定ではない)
  2. ロール角速度 3〜8 Hz: 同じ倍率・同じずらし方
  3. 強い側の前輪・後輪の輪速の揺れ: 包絡線の山 (判定と同じ求め方) に印を付け、実測の時間差と L/v を示す
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/vibration"))
sys.path.insert(0, str(REPO / "src"))
import validate_sites_with_fullcar as V  # noqa: E402

CF = V.CF
SIDE_JA = {"R": "右", "L": "左"}


def best_lag(meas: np.ndarray, pred: np.ndarray, band, fs: float, w: np.ndarray, max_lag_s: float) -> tuple[int, float]:
    """band_corr と同じ計算で、相関が最大になるずれ (標本数) とその相関。正 = モデルを遅らせる。"""
    a, b = CF.bp(meas, *band, fs), CF.bp(pred, *band, fs)
    idx = np.flatnonzero(w)
    L = int(max_lag_s * fs)
    cc = {l: float(np.corrcoef(a[idx + l], b[idx])[0, 1]) for l in range(-L, L + 1)
          if idx[0] + l >= 0 and idx[-1] + l < len(a)}
    l = max(cc, key=cc.get)
    return l, cc[l]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", required=True, help="candidates.csv の event_id")
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--shape", default=None, help="モデルの形 (省略時は event_physics_validation.csv の model_shape_used)")
    ap.add_argument("--side", default=None, help="モデルの側 R / L (省略時は同 model_side_used)")
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_physics_validation.yaml")
    ap.add_argument("--validation-csv", type=Path,
                    default=REPO / "out/vibration/physics_validation_near_miss/event_physics_validation.csv")
    ap.add_argument("--xlim", type=float, nargs=2, default=(-0.5, 1.25))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V.configure_japanese_font()

    pv = yaml.safe_load(args.config.read_text())
    cfg = yaml.safe_load((REPO / pv["base_config"]).read_text())
    p = V.FC.build_params(V.FC.load_config(REPO / pv["fullcar_config"]))
    cand = pd.read_csv(REPO / pv["locate_dir"] / "candidates.csv")
    e = cand[cand.event_id == args.event].iloc[0]
    ref_row = pd.read_csv(args.validation_csv).set_index("event_id").loc[args.event]
    shape = args.shape or ref_row.model_shape_used
    side = args.side or ref_row.model_side_used

    ev, preds = V.evaluate_event(e, V.MeasuredLoader(args.roots, cfg), p, cfg, pv)
    fs = cfg["grid_hz"]
    t, meas = ev["t"], ev["meas_sig"]
    sim = preds[(shape, side)]
    v = ev["v"]
    lag_exp = p.wheelbase / v
    ws = pv["window_s"]
    road = pv["bands_hz"]["road"]
    w = (t >= -ws["metric_pre"]) & (t <= lag_exp + ws["metric_post"])

    # P4: 3〜8 Hz。倍率は上下で決めて、ロールにも同じだけ掛ける (比を保つ)
    mb = {k: CF.bp(meas[k], *road, fs) for k in ("heave", "roll")}
    sb = {k: CF.bp(sim[k], *road, fs) for k in ("heave", "roll")}
    k_shape = V._rms(mb["heave"][w]) / max(V._rms(sb["heave"][w]), 1e-12)
    corr, shift = {}, {}
    for k in ("heave", "roll"):
        shift[k], corr[k] = best_lag(meas[k], sim[k], road, fs, w, ws["lag_search"])

    # P2: metrics と同じ求め方 (|輪速の揺れ| の 10 点移動平均の山、窓は [−0.3, L/v + 0.7])
    ww = (t >= -0.3) & (t <= lag_exp + 0.7)
    sd = "R" if ev["meas"]["wheel_right"] >= ev["meas"]["wheel_left"] else "L"
    tw = t[ww]
    env = {k: np.convolve(np.abs(meas[k][ww]), np.ones(10) / 10, "same") for k in ("F" + sd, "R" + sd)}
    t_f, t_r = tw[np.argmax(env["F" + sd])], tw[np.argmax(env["R" + sd])]
    lag_meas = t_r - t_f
    assert abs(lag_meas - ev["meas"]["front_rear_lag_s"]) < 1e-9

    th = pv["judge"] if "judge" in pv else {}
    pw = (t >= args.xlim[0]) & (t <= args.xlim[1])
    ORANGE, BLACK, SHADE = "#e8740c", "#222222", "#e9eef5"
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)

    rows = (("heave", "上下加速度\n[m/s²]", 0.5), ("roll", "ロール角速度\n[deg/s]", 0.4))
    for ax, (k, yl, thr) in zip(axs[:2], rows):
        ax.axvspan(t[w][0], t[w][-1], color=SHADE, zorder=0)
        ax.plot(t[pw], mb[k][pw], color=BLACK, lw=1.4, label="実測 (3〜8 Hz)")
        sim_shift = np.roll(sb[k], shift[k]) * k_shape
        ax.plot(t[pw], sim_shift[pw], color=ORANGE, lw=1.6, label=f"モデル ({SIDE_JA.get(side, side)}側だけの入力, 3〜8 Hz, 振幅は実測に合わせて表示)")
        ax.set_ylabel(yl)
        ax.text(0.99, 0.95, f"波形の相関 r = {corr[k]:.2f}  (P4 の基準 {thr} 以上)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))
    axs[0].legend(fontsize=8, loc="lower right", frameon=False)
    for a in axs[:2]:
        a.set_ylim(a.get_ylim()[0], a.get_ylim()[1] * 1.25)
    axs[0].text(t[w][0] + 0.01, axs[0].get_ylim()[1] * 0.97, "相関を計算した区間", fontsize=7.5, color="0.35", va="top")

    ax = axs[2]
    fk, rk = "F" + sd, "R" + sd
    # 判定は揺れの包絡線 (|揺れ| の 10 点移動平均) の山の時刻差で見ている。包絡線を薄く塗って示す
    for k, c in ((fk, "#1f6fb4"), (rk, "#c0392b")):
        ax.fill_between(tw, 0, env[k] * 3.6, color=c, alpha=0.18, lw=0)
    ax.plot(t[pw], meas[fk][pw] * 3.6, color="#1f6fb4", lw=1.1, label=f"実測 {SIDE_JA[sd]}前輪")
    ax.plot(t[pw], meas[rk][pw] * 3.6, color="#c0392b", lw=1.1, label=f"実測 {SIDE_JA[sd]}後輪")
    ax.fill_between([], [], [], color="0.5", alpha=0.25, label="揺れの包絡線 (山の時刻を比べる)")
    ymax = np.nanmax(np.abs(np.r_[meas[fk][pw], meas[rk][pw]])) * 3.6
    ax.set_ylim(-1.75 * ymax, 1.8 * ymax)
    for tt, c in ((t_f, "#1f6fb4"), (t_r, "#c0392b")):
        ax.axvline(tt, color=c, lw=0.9, ls=":")
    y1 = 1.25 * ymax
    ax.annotate("", xy=(t_r, y1), xytext=(t_f, y1), arrowprops=dict(arrowstyle="<->", color=BLACK, lw=1.2))
    ax.text((t_f + t_r) / 2, y1 * 1.04, f"実測の時間差 {lag_meas:.2f} s", ha="center", va="bottom", fontsize=9)
    y2 = -1.45 * ymax
    ax.annotate("", xy=(t_f + lag_exp, y2), xytext=(t_f, y2), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2))
    ax.text(t_f + lag_exp + 0.02, y2, f"ホイールベース ÷ 車速 = {p.wheelbase:.2f} m ÷ {v:.1f} m/s = {lag_exp:.2f} s",
            ha="left", va="center", fontsize=8.5, color=ORANGE)
    ax.set_ylabel("輪速の揺れ\n[km/h]")
    ax.legend(fontsize=7.5, loc="upper right", frameon=False, ncol=1)
    ax.set_xlabel("前輪が当たった時刻からの時間 [s]")
    for a in axs:
        a.axhline(0, color="0.85", lw=0.8, zorder=1)
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(*args.xlim)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"{args.event}: r_heave={corr['heave']:.3f} (ずらし {shift['heave'] / fs:+.2f} s), "
          f"r_roll={corr['roll']:.3f} (ずらし {shift['roll'] / fs:+.2f} s), 倍率 ×{k_shape:.1f}, "
          f"前→後 {lag_meas:.2f} s, L/v {lag_exp:.3f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
