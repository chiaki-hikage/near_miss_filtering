#!/usr/bin/env python3
"""comma2k19 と commaCarSegments の信号・単位・周期の整合性を確認する。

3 つの確認を行う。

A. 復号経路の検証 (comma2k19)
   同じ raw_can から復号し直した車速・輪速・舵角が、データセットが配る
   processed_log と一致するかを見る。commaCarSegments には processed_log が
   無く主系列を生 CAN から作るので、ここが合っていることが前提になる。

B. 物理整合の検証 (commaCarSegments)
   突き合わせる相手が無いので、信号どうしの物理関係で符号とスケールを確かめる。
     車速     ≈ 4 輪の輪速平均
     ax       ≈ 車速の微分 ≈ GVC (車両が報告する前後 G)
     ay       ≈ 車速 × ヨーレート   (ay = v·r)
     舵角     と ヨーレート が同符号
     舵角の微分 ≈ 車両が報告する舵角レート

C. 諸元の比較
   両データセットのチャネルごとに、受信周期・値域・欠測率を並べる。

使い方:
  python scripts/check_signal_parity.py \\
      --comma2k19 raw_data/Chunk_1 --limit-drives 2 \\
      --car-segments raw_data/comma_car_segments --platform TOYOTA_RAV4_TSS2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features, derivative
from near_miss.io import comma2k19, comma_car_segments
from near_miss.signals import to_grid

# A で突き合わせる組。(processed_log 由来, raw_can 由来, 期待する傾き, 許容 rmse, 単位)
# 車速だけは一致しない。processed_log の speed は 4 輪の輪速平均そのもので、
# 0x0B4 の SPEED は表示車速 (2.13% 高い) だった。両方載せて差が想定どおりかを見る。
PARITY_PAIRS = (
    ("speed_mps", "ws_mean_mps", 1.0, 0.01, "m/s"),
    ("speed_mps", "speed_can_mps", 1.0213, 0.60, "m/s"),
    ("ws_fl_mps", "ws_fl_can_mps", 1.0, 0.05, "m/s"),
    ("ws_fr_mps", "ws_fr_can_mps", 1.0, 0.05, "m/s"),
    ("ws_rl_mps", "ws_rl_can_mps", 1.0, 0.05, "m/s"),
    ("ws_rr_mps", "ws_rr_can_mps", 1.0, 0.05, "m/s"),
    ("steer_deg", "steer_can_deg", 1.0, 0.20, "deg"),
)
WS_CAN = ("ws_fl_can_mps", "ws_fr_can_mps", "ws_rl_can_mps", "ws_rr_can_mps")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comma2k19", type=Path, default=Path("raw_data/Chunk_1"))
    p.add_argument("--car-segments", type=Path, default=comma_car_segments.DEFAULT_CACHE)
    p.add_argument("--platform", default="TOYOTA_RAV4_TSS2")
    p.add_argument("--limit-drives", type=int, default=2, help="comma2k19 側で使うドライブ数")
    p.add_argument("--limit-segments", type=int, default=None, help="commaCarSegments 側の上限")
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=None, help="比較表の保存先 CSV")
    return p.parse_args()


# ---------------------------------------------------------------------------
def channel_stats(segments) -> pd.DataFrame:
    """チャネルごとの受信周期と値域をまとめる。"""
    rows: dict[str, dict] = {}
    for seg in segments:
        for name, ch in seg.channels.items():
            r = rows.setdefault(name, {"channel": name, "unit": ch.unit, "kind": ch.kind,
                                       "n": 0, "hz": [], "lo": np.inf, "hi": -np.inf, "segs": 0})
            r["segs"] += 1
            r["n"] += int(ch.t.size)
            if ch.t.size > 1:
                dur = float(ch.t[-1] - ch.t[0])
                if dur > 0:
                    r["hz"].append(ch.t.size / dur)
            if ch.v.size:
                r["lo"] = min(r["lo"], float(np.nanmin(ch.v)))
                r["hi"] = max(r["hi"], float(np.nanmax(ch.v)))
    out = []
    for r in rows.values():
        out.append({
            "channel": r["channel"], "unit": r["unit"], "kind": r["kind"],
            "segments": r["segs"], "samples": r["n"],
            "rate_hz": round(float(np.median(r["hz"])), 1) if r["hz"] else np.nan,
            "min": round(r["lo"], 3), "max": round(r["hi"], 3),
        })
    return pd.DataFrame(out).sort_values("channel").reset_index(drop=True)


def _agree(a: np.ndarray, b: np.ndarray) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return {"n": int(ok.sum()), "r": np.nan, "slope": np.nan, "rmse": np.nan, "max_abs": np.nan}
    x, y = a[ok], b[ok]
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 1e-9 else np.nan
    return {
        "n": int(ok.sum()),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "slope": slope,
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "max_abs": float(np.max(np.abs(x - y))),
    }


# ---------------------------------------------------------------------------
def part_a(root: Path, cfg, vehicles, limit_drives: int) -> tuple[pd.DataFrame, list]:
    refs = comma2k19.find_segments(root)
    drives = comma2k19.group_by_drive(refs)
    picked = [r for _, rs in list(drives.items())[:limit_drives] for r in rs]
    if not picked:
        return pd.DataFrame(), []

    vehicle = find_vehicle_config(picked[0].dongle_id, vehicles)
    segments = [comma2k19.load_segment(r, vehicle, with_raw_can=True) for r in picked]

    acc: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for seg in segments:
        gs = to_grid(seg, cfg)
        if gs.df.empty:
            continue
        if all(c in gs.df for c in WS_CAN):
            gs.df["ws_mean_mps"] = gs.df[list(WS_CAN)].mean(axis=1)
        for ref_col, can_col, _slope, _tol, _u in PARITY_PAIRS:
            if ref_col in gs.df and can_col in gs.df:
                acc.setdefault((ref_col, can_col), []).append(
                    (gs.df[ref_col].to_numpy(), gs.df[can_col].to_numpy())
                )

    rows = []
    for ref_col, can_col, expect, tol, unit in PARITY_PAIRS:
        key = (ref_col, can_col)
        if key not in acc:
            rows.append({"processed_log": ref_col, "raw_can": can_col, "判定": "欠測"})
            continue
        a = np.concatenate([p[0] for p in acc[key]])
        b = np.concatenate([p[1] for p in acc[key]])
        m = _agree(a, b)
        rows.append({
            "processed_log": ref_col, "raw_can": can_col, "unit": unit,
            "n": m["n"], "r": round(m["r"], 6), "slope": round(m["slope"], 6),
            "期待slope": expect,
            "rmse": round(m["rmse"], 4), "max_abs": round(m["max_abs"], 4),
            "許容rmse": tol,
            "判定": "想定どおり" if (m["rmse"] <= tol and abs(m["slope"] - expect) < 0.01) else "要確認",
        })
    return pd.DataFrame(rows), segments


def part_b(cache: Path, platform: str, cfg, vehicles, limit: int | None):
    vehicle = find_vehicle_config_for_platform(platform, vehicles)
    if vehicle is None:
        raise SystemExit(f"車種設定が見つかりません: {platform}")
    refs = comma_car_segments.find_segments(cache, platform)
    if limit:
        refs = refs[:limit]
    if not refs:
        raise SystemExit(f"セグメントがありません: {cache}")

    segments = [comma_car_segments.load_segment(r, vehicle) for r in refs]
    frames = []
    for seg in segments:
        gs = to_grid(seg, cfg)
        if gs.df.empty:
            continue
        gs = compute_features(gs, cfg, radar=seg.radar, vehicle=vehicle)
        frames.append(gs.df.assign(_rate=gs.rate_hz))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, segments, vehicle


# 期待する傾きは comma2k19 (RAV4 2017, 220,181 点) での実測値。
# 1.0 ではないものは、信号そのものの性質でそうなる。値は docs/comma_car_segments.md 参照。
def physics_checks(df: pd.DataFrame, rate_hz: float, min_r: float = 0.8) -> pd.DataFrame:
    """物理関係で符号とスケールを確かめる。

    見るのは「符号が合っているか」と「傾きが comma2k19 と同じか」の 2 点。
    傾きが 1.0 でないこと自体は誤りではない (加速度計は勾配や路面カントを拾う)。
    データセット間で傾きが揃っていれば、同じ閾値をそのまま持ち込める。
    """
    rows = []

    def add(name, a, b, expect, tol, note):
        m = _agree(a, b)
        ok = (
            np.isfinite(m["slope"])
            and (np.isnan(expect) or abs(m["slope"] - expect) <= tol)
            and (m["r"] or 0) > min_r
            and (np.isnan(expect) or m["slope"] * expect > 0)
        )
        rows.append({
            "確認": name, "n": m["n"], "r": round(m["r"], 4) if np.isfinite(m["r"]) else np.nan,
            "slope": round(m["slope"], 4) if np.isfinite(m["slope"]) else np.nan,
            "期待slope": expect, "rmse": round(m["rmse"], 4) if np.isfinite(m["rmse"]) else np.nan,
            "判定": "OK" if ok else "要確認", "備考": note,
        })

    ws = [c for c in ("ws_fl_mps", "ws_fr_mps", "ws_rl_mps", "ws_rr_mps") if c in df]
    if "speed_mps" in df and len(ws) == 4:
        add("車速 vs 輪速平均", df["speed_mps"].to_numpy(),
            df[ws].to_numpy().mean(axis=1), 1.0, 0.05, "同じ車輪速から出ているので一致するはず")

    if "ax_mps2" in df and "gvc_mps2" in df:
        add("ax (車速微分) vs GVC", df["ax_mps2"].to_numpy(), df["gvc_mps2"].to_numpy(),
            0.94, 0.10, "GVC は車両が報告する前後 G。comma2k19 実測 0.943")
    if "ax_mps2" in df and "ax_can_mps2" in df:
        add("ax (車速微分) vs ACCEL_X", df["ax_mps2"].to_numpy(), df["ax_can_mps2"].to_numpy(),
            1.0, 0.30, "加速度計は勾配を拾う。comma2k19 実測 0.996")

    if "ay_kin_mps2" in df and "ay_can_mps2" in df:
        add("ay = v x yaw vs ACCEL_Y", df["ay_kin_mps2"].to_numpy(), df["ay_can_mps2"].to_numpy(),
            0.70, 0.15, "ACCEL_Y は一貫して小さく出る。comma2k19 実測 0.703")

    if "steer_deg" in df and "yaw_rate_dps" in df and "v_mps" in df:
        fast = df["v_mps"].to_numpy() > 8.0
        # 傾きは速度で変わるので大きさは見ない。相関が正であることだけを確かめる。
        m = _agree(df["steer_deg"].to_numpy()[fast], df["yaw_rate_dps"].to_numpy()[fast])
        rows.append({
            "確認": "舵角 vs ヨーレート (符号)", "n": m["n"], "r": round(m["r"], 4),
            "slope": round(m["slope"], 4), "期待slope": np.nan, "rmse": np.nan,
            "判定": "OK" if m["r"] > 0.8 else "要確認",
            "備考": "相関が正なら左舵で左旋回。傾きは速度で変わるので見ない",
        })

    if "steer_deg" in df and "steer_rate_can_dps" in df:
        add("舵角の微分 vs STEER_RATE", derivative(df["steer_deg"].to_numpy(), 1.0 / rate_hz),
            df["steer_rate_can_dps"].to_numpy(), 1.0, 0.40,
            "20 Hz へ落として微分するぶん 1 未満に出る。符号と概ねの大きさだけ見る")

    return pd.DataFrame(rows)


def show(title: str, df: pd.DataFrame) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print("該当なし" if df.empty else df.to_string(index=False))


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    vehicles = load_vehicle_configs(args.vehicles)

    a_df, c2k_segments = part_a(args.comma2k19, cfg, vehicles, args.limit_drives)
    show(f"A. 復号経路の検証  comma2k19 processed_log vs raw_can  ({len(c2k_segments)} セグメント)", a_df)

    b_df, ccs_segments, vehicle = part_b(
        args.car_segments, args.platform, cfg, vehicles, args.limit_segments
    )
    rate = float(cfg["resample"]["rate_hz"])
    show(f"B. 物理整合の検証  commaCarSegments / {args.platform}  ({len(ccs_segments)} セグメント)",
         physics_checks(b_df, rate))

    s_c2k = channel_stats(c2k_segments)
    s_ccs = channel_stats(ccs_segments)
    merged = s_c2k.merge(s_ccs, on="channel", how="outer", suffixes=("_c2k19", "_ccs"))
    cols = ["channel", "unit_c2k19", "unit_ccs", "rate_hz_c2k19", "rate_hz_ccs",
            "min_c2k19", "max_c2k19", "min_ccs", "max_ccs"]
    show("C. チャネル諸元の比較  (左 comma2k19 / 右 commaCarSegments)",
         merged[[c for c in cols if c in merged.columns]])

    if not b_df.empty:
        print()
        print("commaCarSegments 側の走行条件:")
        for col, label in (("v_mps", "車速 [m/s]"), ("ax_mps2", "前後加速度 [m/s^2]"),
                           ("yaw_rate_dps", "ヨーレート [deg/s]"), ("lead_distance_m", "先行車距離 [m]")):
            if col in b_df:
                x = b_df[col].to_numpy()
                x = x[np.isfinite(x)]
                if x.size:
                    print(f"  {label:<22} 中央 {np.median(x):8.2f}   5% {np.percentile(x,5):8.2f}   95% {np.percentile(x,95):8.2f}")
        if "op_engaged" in b_df:
            print(f"  openpilot 介入率        {np.nanmean(b_df['op_engaged']):.1%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out, index=False)
        print(f"\n比較表を書き出しました: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
