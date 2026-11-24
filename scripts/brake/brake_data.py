"""comma2k19 のセグメントから、制動の評価に使う信号を 20 Hz の表にする (データセット固有の層)。

- CAN の復号は src/near_miss/io/can_decode.py の関数をそのまま使う (DBC の Motorola 表記)
- RAV4 の信号は configs/vehicles/toyota_rav4.yaml、Civic の信号は configs/brake/brake_performance.yaml から読む
- 車速は processed_log/CAN/speed (4 輪速の平均。docs/comma2k19_data.md)
- 勾配は global_pose の ECEF 速度から (brake_model.grade_from_ecef)
- 値は補完しない。古すぎる値 (stale_s より前) は欠損にする
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

from near_miss.config import SignalSpec
from near_miss.io.can_decode import extract_bits, frame_mask, payload_to_u64

import brake_model as BM


@dataclass(frozen=True)
class Sig:
    spec: SignalSpec
    include_tx: bool = False


def car_signals(car_cfg: dict) -> dict[str, Sig]:
    """PoC で使う名前 -> 信号の定義。車種設定にあるものはそこから、無いものは extra_signals から。"""
    sigs: dict[str, Sig] = {}
    if car_cfg.get("vehicle_config"):
        vc = yaml.safe_load(open(car_cfg["vehicle_config"]))
        for name, d in vc["signals"].items():
            sigs[name] = Sig(SignalSpec.from_dict(name, d))
    for name, d in (car_cfg.get("extra_signals") or {}).items():
        sigs[name] = Sig(SignalSpec.from_dict(name, d), bool(d.get("include_tx", False)))
    return sigs


def decode(t, address, src, payload, sig: Sig) -> tuple[np.ndarray, np.ndarray]:
    s = sig.spec
    m = frame_mask(address, src, s.can_id, s.bus, include_tx=sig.include_tx)
    if sig.include_tx:
        m &= (src & 0x80) != 0     # 送信フレームだけ
    if not m.any():
        return np.empty(0), np.empty(0)
    raw = extract_bits(payload[m], s.start_bit, s.length, s.signed)
    v = (raw * s.factor + s.offset) * s.sign
    tt = t[m]
    o = np.argsort(tt, kind="stable")
    return tt[o], v[o].astype(float)


def hold(tt: np.ndarray, vv: np.ndarray, tg: np.ndarray, stale_s: float) -> np.ndarray:
    """直前の値を持つ (サンプル・アンド・ホールド)。stale_s より古ければ欠損。"""
    if tt.size == 0:
        return np.full(tg.size, np.nan)
    ix = np.searchsorted(tt, tg, side="right") - 1
    out = vv[np.clip(ix, 0, None)].astype(float)
    age = tg - tt[np.clip(ix, 0, None)]
    out[(ix < 0) | (age > stale_s)] = np.nan
    return out


def find_segments(car_cfg: dict) -> list[Path]:
    segs = []
    for r in car_cfg["roots"]:
        segs += [p for p in Path(r).glob(car_cfg["seg_glob"]) if (p / "processed_log/CAN/raw_can").is_dir()]
    return sorted(segs, key=lambda p: (p.parent.name, int(p.name)))


def load_segment(seg: Path, car_cfg: dict, sigs: dict[str, Sig], cfg: dict) -> pd.DataFrame | None:
    can = seg / "processed_log/CAN"
    t = np.load(can / "raw_can/t")
    address = np.load(can / "raw_can/address").astype(np.int64)
    src = np.load(can / "raw_can/src").astype(np.int64)
    payload = payload_to_u64(np.load(can / "raw_can/data"))
    dt = cfg["grid_dt_s"]
    w = max(1, int(round(cfg["speed_smooth_s"] / dt)))
    tg = np.arange(t.min() + 1.0, t.max() - 1.0, dt)
    if tg.size < 4 * w:
        return None
    vt = np.load(can / "speed/t")
    vv = np.load(can / "speed/value").ravel()
    v = np.interp(tg, vt, vv)
    vs = np.convolve(v, np.ones(w) / w, mode="same")
    a = np.gradient(vs, dt)
    df = pd.DataFrame(dict(t=tg, v=vs, a=a))
    names = dict(car_cfg["channels"])
    names.update({k: k for k in (car_cfg.get("extra_signals") or {})})
    for key, name in names.items():
        tt, val = decode(t, address, src, payload, sigs[name])
        df[key] = hold(tt, val, tg, cfg["stale_s"])
    # 値は車速と同じ幅でならす (減速度と同じ時間の分解能にそろえる)
    for key in ("level", "level_alt"):
        if key in df:
            x = df[key].to_numpy()
            df[key + "_s"] = pd.Series(x).rolling(w, center=True, min_periods=w).mean().to_numpy()
    gp = seg / "global_pose"
    if gp.is_dir():
        ft = np.load(gp / "frame_times")
        pos = np.load(gp / "frame_positions")
        vel = np.load(gp / "frame_velocities")
        th, lat, lon, hd = BM.grade_from_ecef(pos, vel, cfg["grade"]["min_speed_mps"])
        ok = np.isfinite(th)
        ws = max(1, int(round(cfg["grade"]["smooth_s"] / np.median(np.diff(ft)))))
        ths = pd.Series(th).rolling(ws, center=True, min_periods=max(1, ws // 2)).mean().to_numpy()
        inside = (tg >= ft[0]) & (tg <= ft[-1])
        df["grade_rad"] = np.where(inside, np.interp(tg, ft[ok], ths[ok]), np.nan) if ok.sum() > 10 else np.nan
        df["lat"] = np.where(inside, np.interp(tg, ft, lat), np.nan)
        df["lon"] = np.where(inside, np.interp(tg, ft, lon), np.nan)
        okh = np.isfinite(hd)
        df["heading_deg"] = np.where(inside, np.degrees(np.interp(tg, ft[okh], np.unwrap(np.radians(hd[okh])))) % 360, np.nan) if okh.sum() > 10 else np.nan
    else:
        df["grade_rad"] = np.nan
    edge = np.zeros(len(df), bool)
    edge[:w] = edge[-w:] = True
    df["edge"] = edge
    df["drive"] = seg.parent.name
    df["seg"] = int(seg.name)
    return df


def load_car(car: str, cfg: dict, limit: int | None = None) -> pd.DataFrame:
    car_cfg = cfg["cars"][car]
    sigs = car_signals(car_cfg)
    segs = find_segments(car_cfg)[:limit]
    parts = [d for s in segs if (d := load_segment(s, car_cfg, sigs, cfg)) is not None]
    D = pd.concat(parts, ignore_index=True)
    D = D.sort_values(["drive", "t"]).reset_index(drop=True)
    # 連続する時刻の塊 (セグメントの継ぎ目で切れない、欠番で切れる)
    # 各セグメントは前後 1 s を切って読むので、続いたセグメントの間には約 2 s の隙間がある。これは続きとみなす
    gap = D.groupby("drive").t.diff().fillna(1e9) > cfg["block_gap_s"]
    D["block"] = gap.cumsum()
    return D
