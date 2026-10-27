"""振動イベントに ISO 2631-1 の評価量 (MTVV・VDV・波高率・快適性の目安) を付ける。

ISO 2631-1 は乗り心地・健康の評価で、ヒヤリハットの基準ではない。
ここで出すのは「ISO 2631-1 で見るとどれくらい強い衝撃だったか」という客観値である。

IMU はフロントガラスの取付部にあり、座面ではない。座席の伝達 (4〜6 Hz で増幅、高い周波数で減衰) は
入っていないので、値は「IMU の位置で、座面と同じ重み付けをかけたもの」の近似である。

    uv run python scripts/vibration/iso2631_events.py raw_data/Chunk_1 raw_data/Chunk_2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

import find_roll_bumps as F
import iso2631 as ISO
import validate_sites_with_fullcar as V

REPO = _bootstrap.SRC.parent
G = F.G
AXES = (("x", "Wd"), ("y", "Wd"), ("z", "Wk"), ("rx", "We"), ("ry", "We"))


def raw_axes(ref, rot: np.ndarray, fs: float) -> tuple[np.ndarray, dict]:
    """1 セグメントの生の比力 (前・右・上、重力を除く上下) と角加速度 (ロール・ピッチ) を fs の格子で返す。

    find_roll_bumps.load_seg と同じ回転・同じ時刻の原点 (加速度計の先頭) を使う。
    """
    ta, a = F._load(ref.path / "processed_log/IMU/accelerometer")
    tg, g = F._load(ref.path / "processed_log/IMU/gyro")
    t0 = ta[0]
    grid = np.arange(max(ta[0], tg[0]), min(ta[-1], tg[-1]), 1 / fs)
    a = np.einsum("ij,nj->ni", rot, a)
    g = np.einsum("ij,nj->ni", rot, g)
    ai = np.c_[[np.interp(grid, ta, a[:, k]) for k in range(3)]].T
    gi = np.c_[[np.interp(grid, tg, g[:, k]) for k in range(3)]].T
    out = dict(x=ai[:, 0], y=ai[:, 1], z=-ai[:, 2] - G,
               rx=np.gradient(gi[:, 0], 1 / fs), ry=np.gradient(gi[:, 1], 1 / fs))   # [rad/s²]
    return grid - t0, out


class RawLoader:
    """イベントの前後を、隣のセグメントもつないで切り出す (つなぐのは重み付けの前)。"""

    def __init__(self, loader: V.MeasuredLoader, fs: float):
        self.ld, self.fs = loader, fs

    def load(self, drive: str, segment: int, t_hit: float, span: tuple[float, float]) -> tuple[np.ndarray, dict]:
        rs = sorted(self.ld.refs[drive], key=lambda r: r.index)
        if drive not in self.ld.rot:
            self.ld.rot[drive] = F.drive_rotation([r.path for r in rs])
        rot = self.ld.rot[drive]
        by = {r.index: r for r in rs}
        t_base = float(np.load(by[segment].path / "processed_log/IMU/accelerometer/t").ravel()[0])
        ts, xs = [], []
        for k in (segment - 1, segment, segment + 1):
            if k not in by:
                continue
            t, x = raw_axes(by[k], rot, self.fs)
            t0 = float(np.load(by[k].path / "processed_log/IMU/accelerometer/t").ravel()[0])
            ts.append(t + t0 - t_base)
            xs.append(x)
        t = np.concatenate(ts)
        order = np.argsort(t)
        t = t[order]
        keep = np.r_[True, np.diff(t) > 0]
        grid = np.arange(t_hit + span[0], t_hit + span[1], 1 / self.fs)
        grid = grid[(grid >= t[keep][0]) & (grid <= t[keep][-1])]
        out = {k: np.interp(grid, t[keep], np.concatenate([x[k] for x in xs])[order][keep]) for k in xs[0]}
        return grid - t_hit, out


def iso_metrics(t: np.ndarray, x: dict, fs: float, near_s: float, vdv_span: tuple[float, float]) -> dict:
    """前輪の当たり (t = 0) の前後で、軸ごとと座面の合成の評価量。"""
    aw = {k: ISO.weight(x[k], fs, w) for k, w in AXES}
    rr = {k: ISO.running_rms(aw[k], fs) for k in aw}
    near = (t >= -near_s) & (t <= near_s + 1.0)          # 1 秒の移動 RMS は過去 1 秒なので、後ろに 1 秒足す
    win = (t >= vdv_span[0]) & (t <= vdv_span[1])
    tot = np.sqrt(sum((ISO.K_SEAT[k] * rr[k]) ** 2 for k in rr))
    tot_tr = np.sqrt(sum(rr[k] ** 2 for k in ("x", "y", "z")))
    out = {f"mtvv_{k}": float(np.nanmax(rr[k][near])) for k in rr}
    out["mtvv_total"] = float(np.nanmax(tot[near]))
    out["mtvv_translational"] = float(np.nanmax(tot_tr[near]))
    out["vdv_z"] = ISO.vdv(aw["z"][win], fs)
    out["vdv_total"] = float(sum((ISO.K_SEAT[k] ** 4) * ISO.vdv(aw[k][win], fs) ** 4 for k in aw) ** 0.25)
    out["aw_rms_z"] = float(np.sqrt(np.mean(aw["z"][win] ** 2)))
    out["crest_z"] = float(np.max(np.abs(aw["z"][win])) / max(out["aw_rms_z"], 1e-9))
    out["mtvv_over_aw_z"] = out["mtvv_z"] / max(out["aw_rms_z"], 1e-9)
    # 合成の最大のときに一番効いている軸
    i = int(np.nanargmax(np.where(near, tot, -1)))
    out["dominant_axis"] = max(rr, key=lambda k: (ISO.K_SEAT[k] * rr[k][i]) if np.isfinite(rr[k][i]) else -1)
    out["comfort"] = ISO.comfort_label(out["mtvv_total"])
    # 感度: 上下の 8 Hz 以上 (ばね下の共振 11 Hz と、IMU 取付部の共振の可能性がある帯域) を落として同じ計算をする。
    # 座席は 8 Hz 以上を減衰させるので、座面の値はこちらに近い可能性がある
    z_lp = ISO.weight(F._lowpass(x["z"] - np.mean(x["z"]), 8.0, fs), fs, "Wk")
    rr_lp = ISO.running_rms(z_lp, fs)
    tot_lp = np.sqrt(sum((ISO.K_SEAT[k] * rr[k]) ** 2 for k in rr if k != "z") + rr_lp ** 2)
    out["mtvv_z_lp8"] = float(np.nanmax(rr_lp[near]))
    out["mtvv_total_lp8"] = float(np.nanmax(tot_lp[near]))
    out["comfort_lp8"] = ISO.comfort_label(out["mtvv_total_lp8"])
    return out


def baseline(loader: V.MeasuredLoader, raw: RawLoader, cfg: dict, fs: float, near_s: float, n_seg: int, seed: int) -> pd.DataFrame:
    """普段の走行: 無作為なセグメントの、イベントと同じ長さの窓で同じ評価量を取る。"""
    rng = np.random.default_rng(seed)
    refs = [r for rs in loader.refs.values() for r in rs if F.gyro_rate_hz(r.path) >= cfg["min_gyro_hz"]]
    rows = []
    for k in rng.choice(len(refs), size=min(n_seg, len(refs)), replace=False):
        r = refs[k]
        for tc in np.arange(5.0, 55.0, 5.0):
            t, x = raw.load(r.drive_id, r.index, tc, (-4.0, 4.0))
            if len(t) < 7 * fs:
                continue
            tv, v = F._load(r.path / "processed_log/CAN/speed")
            t0 = float(np.load(r.path / "processed_log/IMU/accelerometer/t").ravel()[0])
            if np.interp(tc + t0, tv, v.ravel()) < cfg["min_speed_mps"]:
                continue
            rows.append(iso_metrics(t, x, fs, near_s, (-2.5, 2.5)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--near-miss", type=Path, default=REPO / "out/vibration/near_miss")
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--baseline-segments", type=int, default=120)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    fs = cfg["grid_hz"]
    near_s = 1.0
    ev = pd.read_csv(args.near_miss / "near_miss_vibration_events.csv")
    loader = V.MeasuredLoader(args.roots, cfg)
    raw = RawLoader(loader, fs)

    rows = []
    for _, e in ev.iterrows():
        t, x = raw.load(e.drive, int(e.segment), float(e.event_time_s), (-4.0, 4.0))
        rows.append(dict(event_id=e.event_id, **iso_metrics(t, x, fs, near_s, (-2.5, 2.5))))
    iso = pd.DataFrame(rows)
    base = baseline(loader, raw, cfg, fs, near_s, args.baseline_segments, 0)
    for k in ("mtvv_total", "mtvv_z", "vdv_z", "mtvv_total_lp8"):
        b = np.sort(base[k].to_numpy())
        iso[f"{k}_normal_pct"] = np.round(100 * np.searchsorted(b, iso[k], side="right") / len(b), 1)
    iso.round(4).to_csv(args.near_miss / "iso2631_events.csv", index=False)
    base.round(4).to_csv(args.near_miss / "iso2631_normal_windows.csv", index=False)

    q = lambda s: {f"p{int(p * 100)}": round(float(s.quantile(p)), 3) for p in (0.5, 0.9, 0.99)}
    meta = dict(events=len(iso), normal_windows=len(base),
                normal=dict(mtvv_total=q(base.mtvv_total), mtvv_z=q(base.mtvv_z), vdv_z=q(base.vdv_z),
                            mtvv_total_lp8=q(base.mtvv_total_lp8)),
                events_q=dict(mtvv_total=q(iso.mtvv_total), mtvv_z=q(iso.mtvv_z), vdv_z=q(iso.vdv_z),
                              mtvv_total_lp8=q(iso.mtvv_total_lp8)),
                comfort_lp8_events=iso.comfort_lp8.value_counts().to_dict(),
                comfort_lp8_normal=base.comfort_lp8.value_counts().to_dict(),
                comfort_events=iso.comfort.value_counts().to_dict(), comfort_normal=base.comfort.value_counts().to_dict(),
                dominant_axis_events=iso.dominant_axis.value_counts().to_dict())
    (args.near_miss / "iso2631_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
