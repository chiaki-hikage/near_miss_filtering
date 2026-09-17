#!/usr/bin/env python3
"""スリップ事象の説明用プロット。**発表資料に貼る用途**で、解析用ではない。

scripts/plot_segment.py が「全部の特徴量を確認する」ための道具なのに対し、
こちらは 1 事象を 5 枚のパネルで説明することに絞ってある。
主要値を図中に注記するので、図だけ見て話ができる。

パネル
  1. 車速 + 縦加速度          急減速と速度低下
  2. 舵角 + ヨーレート        大操舵・切り返しとヨーの急変
  3. 横加速度                 横方向の車両運動 (CAN センサと v*yaw の両方)
  4. 4 輪の輪速 + 輪速差 + 各輪の瞬間減速度   タイヤの異常挙動
  5. 制御フラグ               ABS / VSC / 滑り警告灯 / PCS

  uv run python scripts/plot_slip_presentation.py --platform TOYOTA_COROLLA_TSS2 \\
      --events out/sideslip_TOYOTA_COROLLA_TSS2_v3/direct_slip.csv \\
      --filter vsc --out out/slip_plots

`--filter` は事象の選び方。vsc = VSC 作動を含む、abs = ABS 作動を含む、
multi = multi_evidence、all = 全部。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss import plotting

plotting.use_headless_backend()
plotting.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from near_miss.config import (  # noqa: E402
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features  # noqa: E402
from near_miss.io.canonical import concat_segments  # noqa: E402
from near_miss.signals import to_grid  # noqa: E402
from near_miss.sources import car_segments_source, comma2k19_source  # noqa: E402

MPS_TO_KMH = 3.6
PAD_S = 4.0                      # 事象の前後に描く秒数
WHEELS = (("ws_fl_mps", "左前"), ("ws_fr_mps", "右前"),
          ("ws_rl_mps", "左後"), ("ws_rr_mps", "右後"))
WHEEL_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
FLAGS = (("abs_active_flag", "ABS 作動", "tab:red"),
         ("vsc_active_flag", "VSC 作動", "tab:purple"),
         ("slip_warn", "滑り警告灯", "tab:orange"),
         ("precollision_active", "PCS 作動", "tab:brown"))

NOTE = dict(fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec="0.6", alpha=0.92))


def series(df: pd.DataFrame, name: str) -> np.ndarray | None:
    return df[name].to_numpy() if name in df.columns else None


def _mark(ax, t, y, i, color="k"):
    """1 点を丸で示すだけ。値は凡例枠にまとめる (吹き出しが重なるため)。"""
    if i is None or not np.isfinite(y[i]):
        return
    ax.plot(t[i], y[i], "o", ms=7, mfc="none", mec=color, mew=2.0, zorder=6)


def _keybox(ax, lines: list[tuple[str, str]], loc="upper right") -> None:
    """主要値を軸の隅に積む。曲線の上に吹き出しを置くと必ず重なるため。"""
    if not lines:
        return
    x, ha = (0.992, "right") if "right" in loc else (0.008, "left")
    y0, va, step = (0.93, "top", -0.20) if "upper" in loc else (0.07, "bottom", 0.20)
    ax._nkeys = max(getattr(ax, "_nkeys", 0), len(lines))
    for k, (text, color) in enumerate(lines):
        ax.text(x, y0 + step * k, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=10, color=color, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color,
                          alpha=0.94, lw=1.0))


def _argnan(f, x: np.ndarray, lo: int, hi: int) -> int | None:
    """区間 [lo, hi) の中の最大/最小の位置。全部 NaN なら None。"""
    seg = x[lo:hi]
    if not np.isfinite(seg).any():
        return None
    return lo + int(f(np.where(np.isfinite(seg), seg, np.nan)))


def panel_speed(ax, t, df, sel) -> list[str]:
    v = series(df, "v_mps")
    axl = series(df, "ax_mps2")
    notes, keys = [], []
    if v is not None:
        kmh = v * MPS_TO_KMH
        ax.plot(t, kmh, lw=2.4, color="tab:blue", label="車速")
        hi = _argnan(np.nanargmax, kmh, *sel)
        lo = _argnan(np.nanargmin, kmh, *sel)
        _mark(ax, t, kmh, hi, "tab:blue")
        _mark(ax, t, kmh, lo, "tab:blue")
        if hi is not None and lo is not None:
            keys.append((f"車速 {kmh[hi]:.0f} → {kmh[lo]:.0f} km/h "
                         f"(−{kmh[hi]-kmh[lo]:.0f})", "tab:blue"))
            notes.append(keys[-1][0])
        ax.set_ylim(0, max(np.nanmax(kmh) * 1.35, 10))
    ax.set_ylabel("車速 [km/h]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    if axl is not None:
        a = ax.twinx()
        ax._twin = a
        a.plot(t, axl, lw=1.6, color="tab:red", alpha=0.9)
        a.axhline(0, color="gray", lw=0.5)
        i = _argnan(np.nanargmin, axl, *sel)
        _mark(a, t, axl, i, "tab:red")
        a.set_ylabel("縦加速度 [m/s²]", color="tab:red")
        a.tick_params(axis="y", labelcolor="tab:red")
        if i is not None:
            keys.append((f"縦加速度 最小 {axl[i]:.1f} m/s²", "tab:red"))
            notes.append(keys[-1][0])
    _keybox(ax, keys, "upper left")
    return notes


def panel_steer(ax, t, df, sel) -> list[str]:
    st = series(df, "steer_deg")
    yr = series(df, "yaw_rate_dps")
    notes, keys = [], []
    ax.axhline(0, color="gray", lw=0.5)
    if st is not None:
        ax.plot(t, st, lw=2.4, color="tab:blue", label="舵角")
        lo = _argnan(np.nanargmin, st, *sel)
        hi = _argnan(np.nanargmax, st, *sel)
        _mark(ax, t, st, lo, "tab:blue")
        _mark(ax, t, st, hi, "tab:blue")
        if lo is not None and hi is not None:
            a, b = (lo, hi) if lo < hi else (hi, lo)
            keys.append((f"舵角 {st[a]:+.0f}° → {st[b]:+.0f}° "
                         f"(切り返し {abs(st[hi]-st[lo]):.0f}°)", "tab:blue"))
            notes.append(keys[-1][0])
    ax.set_ylabel("舵角 [deg]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    if yr is not None:
        a = ax.twinx()
        ax._twin = a
        a.plot(t, yr, lw=1.8, color="tab:purple", alpha=0.9)
        lo = _argnan(np.nanargmin, yr, *sel)
        hi = _argnan(np.nanargmax, yr, *sel)
        _mark(a, t, yr, lo, "tab:purple")
        _mark(a, t, yr, hi, "tab:purple")
        a.set_ylabel("ヨーレート [deg/s]", color="tab:purple")
        a.tick_params(axis="y", labelcolor="tab:purple")
        if lo is not None and hi is not None:
            p, q = (lo, hi) if lo < hi else (hi, lo)
            keys.append((f"ヨーレート {yr[p]:+.0f} → {yr[q]:+.0f} deg/s", "tab:purple"))
            notes.append(keys[-1][0])
    _keybox(ax, keys, "upper left")
    return notes


def panel_lateral(ax, t, df, sel) -> list[str]:
    notes, keys = [], []
    ax.axhline(0, color="gray", lw=0.5)
    can = series(df, "ay_can_mps2")
    kin = series(df, "ay_kin_mps2")
    if kin is not None:
        ax.plot(t, kin, lw=1.3, color="0.55", alpha=0.95,
                label="v × ヨーレート (運動学)")
    if can is not None:
        ax.plot(t, can, lw=2.4, color="tab:green", label="CAN センサ")
        i = _argnan(lambda x: np.nanargmax(np.abs(x)), can, *sel)
        _mark(ax, t, can, i, "tab:green")
        if i is not None:
            g = abs(can[i]) / 9.80665
            keys.append((f"|ay| 最大 {abs(can[i]):.2f} m/s²  ≒ {g:.2f} G", "tab:green"))
            notes.append(keys[-1][0])
    ax.set_ylabel("横加速度 [m/s²]")
    ax.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.9)
    _keybox(ax, keys, "upper left")
    return notes


def panel_wheels(ax, t, df, sel) -> list[str]:
    notes, keys = [], []
    have = [(c, lab, col) for (c, lab), col in zip(WHEELS, WHEEL_COLORS)
            if c in df.columns]
    for c, lab, col in have:
        ax.plot(t, df[c].to_numpy() * MPS_TO_KMH, lw=1.5, color=col, label=lab)
    ax.set_ylabel("輪速 [km/h]")

    spread = series(df, "ws_spread_mps")
    if spread is not None:
        i = _argnan(np.nanargmax, spread, *sel)
        if i is not None:
            ax.axvline(t[i], color="k", ls=":", lw=1.2)
            keys.append((f"輪速差 最大 {spread[i]:.2f} m/s", "k"))
            notes.append(keys[-1][0])

    dec = series(df, "wheel_decel_mps2")
    if dec is not None:
        a = ax.twinx()
        ax._twin = a
        a.plot(t, dec, lw=1.5, color="tab:brown", alpha=0.85)
        a.axhline(0, color="gray", lw=0.5)
        i = _argnan(np.nanargmin, dec, *sel)
        _mark(a, t, dec, i, "tab:brown")
        a.set_ylabel("輪の瞬間減速度 [m/s²]", color="tab:brown")
        a.tick_params(axis="y", labelcolor="tab:brown")
        if i is not None:
            keys.append((f"輪の瞬間減速度 最深 {dec[i]:.1f} m/s²", "tab:brown"))
            notes.append(keys[-1][0])
    if have:
        ax.legend(loc="lower left", fontsize=9, ncol=4, framealpha=0.9)
    _keybox(ax, keys, "upper left")
    return notes


def panel_flags(ax, t, df, sel) -> list[str]:
    """フラグを段にして並べる。0/1 を重ねると読めないため。"""
    notes = []
    rows = [(c, lab, col) for c, lab, col in FLAGS if c in df.columns]
    for k, (c, lab, col) in enumerate(rows):
        x = df[c].to_numpy()
        # slip_warn は多値。3 = 始動時ランプ試験、4 = 実作動。実作動だけ塗る。
        on = (x > 3.5) & (x <= 4.5) if c == "slip_warn" else (x > 0.5)
        on = np.nan_to_num(on).astype(bool)
        base = len(rows) - 1 - k
        ax.fill_between(t, base + 0.12, base + 0.84, where=on,
                        color=col, alpha=0.9, step="post", lw=0)
        ax.axhline(base, color="0.85", lw=0.6)
        n = int(on[sel[0]:sel[1]].sum())
        if n:
            notes.append(lab)
        # x は軸座標で置く。データ座標だと表示範囲の外に出てしまう。
        ax.text(0.008, (base + 0.48) / len(rows), lab + ("" if n else "  (作動なし)"),
                transform=ax.transAxes, va="center", fontsize=10.5,
                color=col if n else "0.55", fontweight="bold" if n else "normal",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=0.8), zorder=6)
    ax.set_ylim(0, len(rows))
    ax.set_yticks([])
    ax.set_ylabel("制御フラグ")
    return notes


PANELS = (("車速・縦加速度", panel_speed),
          ("舵角・ヨーレート", panel_steer),
          ("横加速度", panel_lateral),
          ("輪速", panel_wheels),
          ("制御フラグ", panel_flags))


def make_figure(gs, ev: dict, title: str, out: Path) -> list[str]:
    t = gs.t - float(gs.t[0])
    df = gs.df
    t0 = float(ev["t_in_segment_s"])
    t1 = t0 + float(ev["duration_s"])
    lo, hi = max(0.0, t0 - PAD_S), min(float(t[-1]), t1 + PAD_S)
    sel = (int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1)))

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(14, 12.5), sharex=True)
    all_notes: list[str] = []
    for ax, (label, fn) in zip(axes, PANELS):
        ax.axvspan(t0, t1, color="tab:red", alpha=0.10, lw=0, zorder=0)
        ax.grid(alpha=0.25, lw=0.6)
        all_notes.extend(fn(ax, t, df, sel))
    axes[-1].set_xlabel("セグメント内の経過時間 [s]")
    axes[0].set_xlim(lo, hi)

    # 左上の注記枠のぶん、上に余白を空ける。曲線と重ねない。
    # フラグのパネル (最後) は段で描いているので触らない。
    for ax in axes[:-1]:
        for a in [ax] + ([ax._twin] if getattr(ax, "_twin", None) is not None else []):
            y0, y1 = a.get_ylim()
            a.set_ylim(y0, y1 + (0.15 * getattr(ax, "_nkeys", 1) + 0.05) * (y1 - y0))

    fig.suptitle(title + "\n赤帯 = 検出されたスリップ区間", fontsize=13, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return all_notes


def pick(d: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "vsc":
        return d[d.evidence.str.contains("vsc_active", na=False)]
    if how == "abs":
        return d[d.evidence.str.contains("abs_active", na=False)]
    if how == "multi":
        return d[d.grade == "multi_evidence"]
    if how == "safety":
        return d[d.evidence.str.contains("vsc_active|abs_active", na=False)]
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=Path, required=True, help="direct_slip.csv")
    p.add_argument("--platform", default=None, help="commaCarSegments の車種キー")
    p.add_argument("--data-root", type=Path, default=Path("raw_data/comma_car_segments"))
    p.add_argument("--dataset", choices=("comma_car_segments", "comma2k19"),
                   default="comma_car_segments")
    p.add_argument("--filter", choices=("vsc", "abs", "multi", "safety", "all"),
                   default="safety")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prefix", default=None, help="出力名の先頭 (既定は車種名の小文字)")
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/slip_plots"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    vcs = load_vehicle_configs(args.vehicles)
    src = (comma2k19_source(args.data_root, vcs) if args.dataset == "comma2k19"
           else car_segments_source(args.data_root, args.platform, vcs))
    by_key = {(r.drive_id, r.index): r for r in src.refs}

    d = pd.read_csv(args.events)
    d = pick(d, args.filter).sort_values(
        ["grade", "ws_spread_peak_mps"], ascending=[True, False])
    if args.limit:
        d = d.head(args.limit)
    print(f"{len(d)} 件を描きます (--filter {args.filter})")

    prefix = args.prefix or (args.platform or "c2k19").replace("TOYOTA_", "").lower()
    for n, ev in enumerate(d.to_dict("records"), 1):
        ref = by_key.get((ev["drive_id"], int(ev["segment"])))
        if ref is None:
            print(f"  [{n}] セグメントが見つかりません: {ev['drive_id']}/{ev['segment']}")
            continue
        veh = src.vehicle_for(ref)
        seg = concat_segments([src.load(ref, veh, True)])
        gs = compute_features(to_grid(seg, cfg), cfg, radar=seg.radar, vehicle=veh)

        short = ev["drive_id"].split("|")[0][:8]
        title = (f"{ev['grade']}  —  {args.platform or 'comma2k19'}  "
                 f"{short} / seg {int(ev['segment'])}  "
                 f"(t={ev['t_in_segment_s']:.1f}〜{ev['t_in_segment_s']+ev['duration_s']:.1f} s)\n"
                 f"証拠: {ev['evidence'].replace('|', ' + ')}")
        out = args.out / f"{prefix}_{ev['grade'].split('_')[0]}_{short}_seg{int(ev['segment'])}_pres.png"
        notes = make_figure(gs, ev, title, out)
        print(f"  [{n}] {out}")
        for x in notes:
            print(f"        {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
