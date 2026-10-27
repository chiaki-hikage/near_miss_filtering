#!/usr/bin/env python3
"""縦躍度 (jerk) が comma のデータで指標として使えるかを、信号源と平滑化の帯域ごとに調べる。

信号源 (縦加速度の出どころ)
  speed   車速 (4 輪速の平均、0.01 km/h 刻み) を 2 回微分する。現行の jerk_mps3 と同じ出どころ
  accel_x CAN 0x024 ACCEL_X (0.0359 m/s² 刻み) を 1 回微分する
  gvc     CAN 0x320 GVC (0.04 m/s² 刻み、約 21 Hz) を 1 回微分する
  imu     IMU の前向き比力 (comma2k19 のみ、約 104 Hz) を 1 回微分する
  current 現行パイプラインそのもの (20 Hz、車速 0.25 s・加速度 0.25 s の移動平均)

各信号源を 100 Hz の一様グリッドに載せ、零位相の 2 次バターワース低域通過 (カットオフ fc) を
かけてから微分する。見る量は 3 つ。

  ノイズ床   一定速の巡航中 (ブレーキ無し、車速 15 m/s 超、縦加速度 |ax| < 0.2) の |jerk| の p99
  制動の信号 ブレーキを踏み始めてから 1.5 s 以内の jerk の最小値 (減速の立ち上がり)
  一致度     制動の前後 (−1〜+2 s) で、独立な信号源どうしの jerk の相関

独立なセンサどうしで jerk の波形が一致すれば、その帯域では物理的な躍度を捉えていると読む。

    uv run python scripts/validate/jerk_feasibility.py --comma2k19 raw_data/Chunk_1 \\
        --car-segments raw_data/comma_car_segments --ccs-limit 300
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

import _bootstrap  # noqa: F401

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features
from near_miss.io import comma2k19, comma_car_segments
from near_miss.signals import resample_continuous, resample_flag, to_grid

FS = 100.0
FCS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
G = 9.80665
CRUISE = dict(min_speed=15.0, max_abs_ax=0.2)
BRAKE_WIN_S = (-1.0, 2.0)
ONSET_SEARCH_S = 1.5
MIN_EVENT_DECEL = 1.0         # 制動イベントとして数える最小の減速度 [m/s²]
HARD_DECEL = 2.5
THRESHOLDS = (2.0, 3.0, 4.0, 6.0)   # 制動の立ち上がりとみなす jerk [m/s³] (負の向き)
MIN_RUN_S = 0.2                     # 誤検出として数える最短の継続


def lowpass(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x)
    if ok.sum() < 30 or fc >= fs / 2:
        return out
    # NaN を跨がない連続区間ごとに通す
    idx = np.flatnonzero(ok)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    for seg in np.split(idx, breaks + 1):
        if seg.size < 30:
            continue
        sos = butter(2, fc, fs=fs, output="sos")
        out[seg] = sosfiltfilt(sos, x[seg])
    return out


def deriv(x: np.ndarray, fs: float) -> np.ndarray:
    d = np.full_like(x, np.nan)
    d[1:-1] = (x[2:] - x[:-2]) * fs / 2
    return d


def jerk_from(source: str, raw: np.ndarray, fc: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(縦加速度, 躍度)。speed は 2 回、それ以外は 1 回微分する。"""
    if source == "speed":
        v = lowpass(raw, fc, fs)
        ax = deriv(v, fs)
    else:
        ax = lowpass(raw, fc, fs)
    return ax, deriv(ax, fs)


def imu_forward(seg_dir: Path, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    p = seg_dir / "processed_log/IMU/accelerometer"
    if not (p / "t").exists():
        return None
    t = np.load(p / "t").ravel().astype(float)
    a = np.load(p / "value").astype(float)
    t, i = np.unique(t, return_index=True)
    a = np.einsum("ij,nj->ni", rot, a[i])
    if (t.size - 1) / (t[-1] - t[0]) < 50:      # 初期ドライブの 10 Hz 相当は使わない
        return None
    return t, a[:, 0]


def rotation_to_down(mean_acc: np.ndarray) -> np.ndarray:
    a = mean_acc / np.linalg.norm(mean_acc)
    b = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, b)
    c = float(a @ b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def grid_segment(seg, extra: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame | None:
    ch = seg.channels
    if "speed_mps" not in ch or "brake_pressed" not in ch:
        return None
    t0 = max(ch["speed_mps"].t[0], ch["brake_pressed"].t[0]) + 0.5
    t1 = min(ch["speed_mps"].t[-1], ch["brake_pressed"].t[-1]) - 0.5
    grid = np.arange(t0, t1, 1 / FS)
    if grid.size < 1000:
        return None
    df = {"t": grid,
          "speed": resample_continuous(ch["speed_mps"].t, ch["speed_mps"].v, grid, 0.5),
          "brake": resample_flag(ch["brake_pressed"].t, ch["brake_pressed"].v, grid, 0.5)}
    for name, key in (("accel_x", "accel_x"), ("gvc", "gvc_mps2"), ("brake_pressure", "brake_pressure")):
        if key in ch and ch[key].t.size > 100:
            df[name] = resample_continuous(ch[key].t, ch[key].v, grid, 0.5)
    for name, (t, v) in extra.items():
        df[name] = resample_continuous(t, v, grid, 0.2)
    return pd.DataFrame(df)


def current_pipeline_jerk(seg, cfg) -> pd.DataFrame | None:
    gs = compute_features(to_grid(seg, cfg), cfg)
    if "jerk_mps3" not in gs.df:
        return None
    return gs.df[["t", "jerk_mps3", "ax_mps2"]]


def runs_at_least(mask: np.ndarray, n: int) -> int:
    """True が n サンプル以上続く区間の数。"""
    if not mask.any():
        return 0
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int(((ends - starts) >= n).sum())


def analyse(df: pd.DataFrame, sources: list[str], cur: pd.DataFrame | None, seg_id: str,
            all_sources: list[str]) -> tuple[list, list, dict, list]:
    """1 セグメントぶんの、巡航中の |jerk| と制動イベントの jerk を集める。"""
    ref_ax = deriv(lowpass(df["speed"].to_numpy(), 0.5, FS), FS)      # 状態判定用の縦加速度 (0.5 Hz)
    speed = df["speed"].to_numpy()
    brake = df["brake"].to_numpy()
    cruise = (speed > CRUISE["min_speed"]) & (brake < 0.5) & (np.abs(ref_ax) < CRUISE["max_abs_ax"])
    jerks = {}
    for s in sources:
        raw = df[s].to_numpy()
        for fc in FCS:
            if s == "gvc" and fc > 8:
                continue
            ax, j = jerk_from(s, raw, fc, FS)
            jerks[(s, fc)] = (ax, j)
    if cur is not None:
        jc = np.interp(df["t"], cur["t"], cur["jerk_mps3"], left=np.nan, right=np.nan)
        jerks[("current", 0.0)] = (None, jc)

    noise_rows = []
    for (s, fc), (_, j) in jerks.items():
        v = np.abs(j[cruise & np.isfinite(j)])
        if v.size:
            noise_rows.append(dict(segment=seg_id, source=s, fc=fc, n=v.size, abs_jerk=v))

    # ブレーキを踏んでいない走行中に、しきい値を下回る状態が MIN_RUN_S 以上続いた回数 (誤検出)
    # ブレーキの前後 2 s は除く (踏む直前のアクセルオフも制動の一部とみなす)
    near_brake = np.convolve(brake >= 0.5, np.ones(int(4 * FS)), "same") > 0
    free = (speed > 5) & ~near_brake
    fa_rows = []
    for (s, fc), (_, j) in jerks.items():
        ok = free & np.isfinite(j)
        hours = ok.sum() / FS / 3600
        for thr in THRESHOLDS:
            fa_rows.append(dict(source=s, fc=fc, thr=thr, hours=hours,
                                n_fa=runs_at_least(ok & (j < -thr), int(MIN_RUN_S * FS))))

    # 制動イベント: ブレーキの立ち上がり
    on = np.flatnonzero((brake[1:] >= 0.5) & (brake[:-1] < 0.5)) + 1
    ev_rows = []
    pairs = {}
    for k in on:
        a, b = k + int(BRAKE_WIN_S[0] * FS), k + int(BRAKE_WIN_S[1] * FS)
        if a < 0 or b >= len(df) or speed[k] < 5:
            continue
        peak = float(np.nanmin(ref_ax[k:k + int(ONSET_SEARCH_S * FS)]))
        if not np.isfinite(peak) or -peak < MIN_EVENT_DECEL:
            continue
        row = dict(segment=seg_id, t=float(df["t"].iloc[k]), speed=float(speed[k]), peak_decel=-peak)
        win = slice(a, b)
        nan_win = np.full(b - a, np.nan)
        for s in all_sources + ["current"]:
            for fc in (FCS if s != "current" else (0.0,)):
                if (s, fc) in jerks:
                    j = jerks[(s, fc)][1]
                    jj = j[k:k + int(ONSET_SEARCH_S * FS)]
                    row[f"{s}@{fc}"] = float(np.nanmin(jj)) if np.isfinite(jj).any() else np.nan
                    pairs.setdefault((s, fc), []).append(j[win])
                else:
                    pairs.setdefault((s, fc), []).append(nan_win)
        ev_rows.append(row)
    return noise_rows, ev_rows, pairs, fa_rows


def run_dataset(name: str, segs_iter, sources: list[str], cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict, list]:
    noise, events, pairs, psd, fa = [], [], {}, [], []
    for seg, extra in segs_iter:
        df = grid_segment(seg, extra)
        if df is None:
            continue
        srcs = [s for s in sources if s in df and np.isfinite(df[s]).mean() > 0.9]
        cur = current_pipeline_jerk(seg, cfg)
        n, e, p, f = analyse(df, srcs, cur, seg.ref.segment_id, sources)
        noise += n
        events += e
        fa += f
        for k, v in p.items():
            pairs.setdefault(k, []).extend(v)
        # 縦加速度の PSD (巡航中 / 制動前後) は accel_x か imu で
        for s in ("imu", "accel_x"):
            if s in srcs:
                x = df[s].to_numpy()
                if np.isfinite(x).all():
                    f, P = welch(x - np.mean(x), fs=FS, nperseg=512)
                    psd.append(dict(source=s, f=f, P=P, braking=bool((df["brake"] > 0.5).mean() > 0.05)))
                break
    print(f"[{name}] 制動イベント {len(events)} 件")
    return pd.DataFrame(noise), pd.DataFrame(events), pairs, psd, pd.DataFrame(fa)


def comma2k19_iter(root: Path, vehicles, limit: int | None):
    refs = comma2k19.find_segments(root)
    drives = comma2k19.group_by_drive(refs)
    n = 0
    for _, rs in drives.items():
        rot = None
        accs = []
        for r in rs:
            p = r.path / "processed_log/IMU/accelerometer/value"
            if p.exists():
                accs.append(np.load(p).astype(float).mean(axis=0))
        if accs:
            rot = rotation_to_down(np.mean(accs, axis=0))
        veh = find_vehicle_config(rs[0].dongle_id, vehicles)
        for r in rs:
            seg = comma2k19.load_segment(r, veh, with_raw_can=True)
            extra = {}
            if rot is not None:
                imu = imu_forward(r.path, rot)
                if imu is not None:
                    extra["imu"] = imu
            yield seg, extra
            n += 1
            if limit and n >= limit:
                return


def ccs_iter(cache: Path, platform: str, vehicles, limit: int, seed: int):
    veh = find_vehicle_config_for_platform(platform, vehicles)
    refs = comma_car_segments.find_segments(cache, platform)
    random.Random(seed).shuffle(refs)
    for r in refs[:limit]:
        try:
            seg = comma_car_segments.load_segment(r, veh, with_raw_can=True)
        except Exception as e:  # noqa: BLE001 壊れたログは飛ばして件数に残す
            print("skip", r.segment_id, type(e).__name__)
            continue
        yield seg, {}


def summarise(noise: pd.DataFrame, events: pd.DataFrame, pairs: dict, ref_pair: tuple[str, str]) -> pd.DataFrame:
    rows = []
    for (s, fc), g in noise.groupby(["source", "fc"]):
        v = np.concatenate(g["abs_jerk"].to_list())
        col = f"{s}@{fc}"
        e = events[col].dropna() if col in events else pd.Series(dtype=float)
        hard = events.loc[events.peak_decel >= HARD_DECEL, col].dropna() if col in events else pd.Series(dtype=float)
        p99 = float(np.percentile(v, 99))
        # 独立な信号源との一致度 (制動前後の波形相関)
        corr = {}
        for other in ref_pair:
            if other == s or (other, fc) not in pairs or (s, fc) not in pairs:
                continue
            a = np.concatenate(pairs[(s, fc)])
            b = np.concatenate(pairs[(other, fc)])
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 100:
                corr[other] = float(np.corrcoef(a[ok], b[ok])[0, 1])
        rows.append(dict(
            source=s, fc_hz=fc, cruise_n=v.size,
            cruise_p50=float(np.median(v)), cruise_p99=p99,
            onset_median=float(e.median()) if e.size else np.nan,
            onset_hard_median=float(hard.median()) if hard.size else np.nan,
            n_events=int(e.size), n_hard=int(hard.size),
            snr_hard=float(-hard.median() / p99) if hard.size else np.nan,
            **{f"corr_{k}": round(v_, 3) for k, v_ in corr.items()},
        ))
    return pd.DataFrame(rows).sort_values(["source", "fc_hz"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comma2k19", type=Path, nargs="*", default=[Path("raw_data/Chunk_1")])
    ap.add_argument("--car-segments", type=Path, default=comma_car_segments.DEFAULT_CACHE)
    ap.add_argument("--platform", default="TOYOTA_RAV4_TSS2")
    ap.add_argument("--ccs-limit", type=int, default=300)
    ap.add_argument("--k19-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    ap.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    ap.add_argument("--out", type=Path, default=Path("out/jerk_feasibility"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)
    vehicles = load_vehicle_configs(args.vehicles)
    results = {}

    def k19():
        for root in args.comma2k19:
            yield from comma2k19_iter(root, vehicles, args.k19_limit)

    for name, it, sources, ref_pair in (
        ("comma2k19", k19(), ["speed", "accel_x", "gvc", "imu"], ("imu", "accel_x", "gvc")),
        ("ccs", ccs_iter(args.car_segments, args.platform, vehicles, args.ccs_limit, args.seed),
         ["speed", "accel_x", "gvc"], ("accel_x", "gvc")),
    ):
        noise, events, pairs, psd, fa = run_dataset(name, it, sources, cfg)
        if noise.empty:
            continue
        summ = summarise(noise, events, pairs, ref_pair)
        # しきい値ごとの誤検出 (件/時) と、強い制動での検出率
        far = fa.groupby(["source", "fc", "thr"]).agg(hours=("hours", "sum"), n_fa=("n_fa", "sum")).reset_index()
        far["fa_per_hour"] = far.n_fa / far.hours
        hard = events[events.peak_decel >= HARD_DECEL]
        far["hit_hard"] = [float((hard[f"{r.source}@{r.fc}"] < -r.thr).mean()) if f"{r.source}@{r.fc}" in hard and len(hard)
                           else np.nan for r in far.itertuples()]
        far["hit_all"] = [float((events[f"{r.source}@{r.fc}"] < -r.thr).mean()) if f"{r.source}@{r.fc}" in events and len(events)
                          else np.nan for r in far.itertuples()]
        far.to_csv(args.out / f"{name}_false_alarm.csv", index=False)
        summ.to_csv(args.out / f"{name}_summary.csv", index=False)
        events.to_csv(args.out / f"{name}_events.csv", index=False)
        np.save(args.out / f"{name}_psd.npy", np.array(psd, dtype=object), allow_pickle=True)
        # 波形の例を描くため、制動前後の窓を保存する
        np.save(args.out / f"{name}_pairs.npy", np.array({f"{k[0]}@{k[1]}": v[:400] for k, v in pairs.items()},
                                                          dtype=object), allow_pickle=True)
        results[name] = summ
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 30)
        print(summ.round(3).to_string(index=False))
    (args.out / "meta.json").write_text(json.dumps(dict(
        fcs=FCS, fs=FS, cruise=CRUISE, brake_win_s=BRAKE_WIN_S, onset_search_s=ONSET_SEARCH_S,
        min_event_decel=MIN_EVENT_DECEL, hard_decel=HARD_DECEL, ccs_limit=args.ccs_limit, seed=args.seed,
        comma2k19=[str(p) for p in args.comma2k19]), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
