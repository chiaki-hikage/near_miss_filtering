#!/usr/bin/env python3
"""車種設定に書いた符号・前提を実データで検証する。

configs/vehicles/*.yaml の `sign` は実測で決めた値なので、別のチャンクや
別の車両に広げるときは必ずこれを回して、前提が崩れていないか確かめる。

使い方:
  python scripts/validate_signals.py <セグメントまたはその親ディレクトリ> [--max-segments 5]

検証項目:
  A. ACCEL_X の符号   基準: 一様グリッド上での車速微分
  B. YAW_RATE の符号   基準: 舵角
  C. ACCEL_Y の符号    基準: YAW_RATE
  D. radar 相対速度の符号  基準: 前方距離の時間変化率
  E. 受信時刻のジッタ
  F. 診断通信 (0x7xx) の有無
  G. openpilot 送信フレームの割合
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from near_miss.config import DEFAULT_VEHICLE_DIR, find_vehicle_config, load_vehicle_configs, load_yaml
from near_miss.config import DEFAULT_DETECTION
from near_miss.features import derivative
from near_miss.io.comma2k19 import find_segments, load_segment
from near_miss.signals import moving_average, to_grid, window_samples

# 判定に使う相関の下限。これを下回る区間は「励起不足で判定不能」とする。
MIN_ABS_CORR = 0.5


def _verdict(r: float, expect_positive: bool) -> str:
    if not np.isfinite(r) or abs(r) < MIN_ABS_CORR:
        return "判定不能 (励起不足)"
    ok = (r > 0) if expect_positive else (r < 0)
    return "一致" if ok else "不一致 → 設定の sign を見直すこと"


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    if np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def check_signs(seg, cfg) -> list[tuple[str, float, str]]:
    """設定の sign を適用した後の系列どうしで整合を見る。

    sign が正しければ、ここでの相関はすべて正になるはず。
    """
    gs = to_grid(seg, cfg)
    df = gs.df
    if df.empty:
        return []

    rows: list[tuple[str, float, str]] = []
    sm = cfg["smoothing"]
    w_speed = window_samples(sm["speed_window_s"], gs.rate_hz)
    w_acc = window_samples(sm["accel_window_s"], gs.rate_hz)

    if "speed_mps" in df and "accel_x" in df:
        v = moving_average(df["speed_mps"].to_numpy(), w_speed)
        a_ref = moving_average(derivative(v, gs.dt), w_acc)
        ax = moving_average(df["accel_x"].to_numpy(), w_acc)
        r = _corr(a_ref, ax)
        rows.append(("A. ACCEL_X vs 車速微分", r, _verdict(r, expect_positive=True)))

    if "yaw_rate" in df and "steer_deg" in df:
        r = _corr(df["yaw_rate"].to_numpy(), df["steer_deg"].to_numpy())
        rows.append(("B. YAW_RATE vs 舵角", r, _verdict(r, expect_positive=True)))

    if "accel_y" in df and "yaw_rate" in df:
        r = _corr(df["accel_y"].to_numpy(), df["yaw_rate"].to_numpy())
        rows.append(("C. ACCEL_Y vs YAW_RATE", r, _verdict(r, expect_positive=True)))

    return rows


def check_radar_sign(seg) -> tuple[float, int]:
    """トラックごとに 前方距離の変化率 と 相対速度 を突き合わせる。

    相対速度が負で接近なら、両者は正の相関になる。
    """
    r = seg.radar
    if r is None or r.t.size == 0:
        return np.nan, 0
    slopes, means = [], []
    for tid in np.unique(r.track_id):
        m = r.track_id == tid
        seg_no = np.cumsum(r.new_track[m])
        for s in np.unique(seg_no):
            k = seg_no == s
            tt, dd, vv = r.t[m][k], r.distance_m[m][k], r.vrel_mps[m][k]
            ok = np.isfinite(dd) & np.isfinite(vv)
            if ok.sum() < 30 or (tt[ok][-1] - tt[ok][0]) < 2.0:
                continue
            slopes.append(np.polyfit(tt[ok], dd[ok], 1)[0])
            means.append(float(np.mean(vv[ok])))
    if len(slopes) < 5:
        return np.nan, len(slopes)
    return float(np.corrcoef(slopes, means)[0, 1]), len(slopes)


def check_timestamps(seg) -> list[tuple[str, float, float, float]]:
    rows = []
    for name in ("speed_mps", "steer_deg"):
        ch = seg.channels.get(name)
        if ch is None or ch.t.size < 3:
            continue
        dt = np.diff(ch.t)
        rows.append((name, float(dt.min()), float(np.median(dt)), float(dt.max())))
    return rows


def check_raw_can(seg_dir: Path, vehicle) -> dict[str, object]:
    base = seg_dir / "processed_log" / "CAN" / "raw_can"
    if not base.is_dir():
        return {}
    address = np.load(base / "address", allow_pickle=True).ravel().astype(np.int64)
    src = np.load(base / "src", allow_pickle=True).ravel().astype(np.int64)
    diag = np.unique(address[address >= 0x700])
    is_tx = (src & vehicle.src_tx_flag) != 0
    ctrl = np.isin(address, np.asarray(vehicle.control_addresses, dtype=np.int64))
    return {
        "n_addresses": int(np.unique(address).size),
        "diagnostic_addresses": [hex(int(a)) for a in diag],
        "tx_ratio": float(is_tx.mean()),
        "control_tx_count": int((is_tx & ctrl).sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--max-segments", type=int, default=5)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    vehicles = load_vehicle_configs(args.vehicles)
    refs = find_segments(args.path)[: args.max_segments]
    if not refs:
        print(f"セグメントが見つかりません: {args.path}")
        return 1

    for ref in refs:
        vehicle = find_vehicle_config(ref.dongle_id, vehicles)
        print("=" * 78)
        print(f"{ref.segment_id}   車種設定: {vehicle.name if vehicle else '該当なし'}")
        print("=" * 78)
        if vehicle is None:
            print("  この dongle に対応する車種設定がありません。検証を飛ばします。\n")
            continue

        seg = load_segment(ref, vehicle, with_raw_can=True)
        print(f"  バイト順: {seg.byte_order}   読み出しの注記: {seg.notes or 'なし'}")

        print("\n  [符号の整合] 設定の sign 適用後。すべて正の相関になるのが正しい")
        rows = check_signs(seg, cfg)
        if not rows:
            print("    判定に必要な信号が揃っていません")
        for label, r, verdict in rows:
            print(f"    {label:28s} r = {r:+.3f}   {verdict}")

        r_radar, n_tracks = check_radar_sign(seg)
        verdict = _verdict(r_radar, expect_positive=True)
        print(f"    {'D. radar v_rel vs 距離変化率':28s} r = {r_radar:+.3f}   {verdict}  (n={n_tracks})")

        print("\n  [受信時刻] 生の時刻で微分してはいけないことの確認")
        for name, lo, med, hi in check_timestamps(seg):
            print(f"    {name:12s} dt min/med/max = {lo:.6f} / {med:.6f} / {hi:.6f} s")

        info = check_raw_can(ref.path, vehicle)
        if info:
            print("\n  [raw_can]")
            print(f"    address 種類数        : {info['n_addresses']}")
            print(f"    診断レンジ (0x7xx)    : {info['diagnostic_addresses'] or 'なし'}")
            print(f"    送信フレーム比率      : {info['tx_ratio']:.1%}")
            print(f"    openpilot 制御フレーム: {info['control_tx_count']} 件")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
