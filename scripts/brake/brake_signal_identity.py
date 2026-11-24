"""RAV4 2017 の 0x224 byte4〜5 (BRAKE_LEVEL) が、実測油圧・踏力・踏み込み量・制動要求値のどれに近いかを確かめる。

比べる相手: commaCarSegments の RAV4 TSS2 (別の世代の車) の 0x226 `BRAKE_PRESSURE` (DBC の注記 "seems prop to pedal force")
と `BRAKE_POSITION` ("seems proportional to pedal displacement")。同じ物差しで振る舞いを測る:
  1. ペダルのスイッチ (BRAKE_PRESSED) が立つ時刻と、値が上がり始める時刻の前後 (踏み込み量なら遊びの分だけ先に動く)
  2. スイッチが立った瞬間の値 (p99 との比)
  3. 減速度との関係の 0 点のずれ (減速度 = b + k x の x 切片) と曲がり (2 次の項)
  4. 停止中に踏み続けたときの値 (p99 との比)
  5. 自動ブレーキ (openpilot の減速の指令) のときに上がるか
あわせて、commaCarSegments の RAV4 2017 世代 (TOYOTA_RAV4、別の車) と RAV4 Hybrid で 0x224 byte4〜5 があるかを見る。

    uv run python scripts/brake/brake_signal_identity.py --run-id <新しい ID> [--tss2-segments 150]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

from near_miss.config import SignalSpec
from near_miss.io import comma_car_segments as CCS
from near_miss.io.can_decode import extract_bits, frame_mask
from near_miss.io.rlog import read_rlog

import brake_data as BD

OUT = Path("out/brake_performance")
DT = 0.05


def spec(name, can_id, start, length, factor=1.0, signed=False, bus=0):
    return BD.Sig(SignalSpec.from_dict(name, dict(can_id=can_id, start_bit=start, length=length, factor=factor, signed=signed, bus=bus)))


def grid_from_frames(t, address, src, payload, sigs: dict, speed_keys: list[str]) -> pd.DataFrame | None:
    tg = np.arange(t.min() + 1, t.max() - 1, DT)
    if tg.size < 400:
        return None
    df = pd.DataFrame(dict(t=tg))
    for k, s in sigs.items():
        tt, vv = BD.decode(t, address, src, payload, s)
        df[k] = BD.hold(tt, vv, tg, 0.5)
    v = df[speed_keys].mean(axis=1).to_numpy()
    if np.isnan(v).mean() > 0.5:
        return None
    v = pd.Series(v).interpolate(limit=4).to_numpy()
    w = int(0.5 / DT)
    vs = np.convolve(np.nan_to_num(v), np.ones(w) / w, mode="same")
    df["v"] = vs
    df["a"] = np.gradient(vs, DT)
    df = df.iloc[w:-w].reset_index(drop=True)
    return df


def behaviour(D: pd.DataFrame, x: str, pressed: str = "pressed", auto: str | None = None) -> dict:
    X = D[x].to_numpy()
    P = D[pressed].to_numpy() > 0.5
    v = D.v.to_numpy()
    a = D.a.to_numpy()
    mv = v > 3
    p99 = float(np.nanpercentile(X[P & mv], 99))
    base = float(np.nanmedian(X[~P & mv]))
    thr = base + 0.03 * (p99 - base)
    seg = D.seg.to_numpy()
    on_lags, off_lags, at_on = [], [], []
    for i in np.where((P[1:] & ~P[:-1]) & (seg[1:] == seg[:-1]) & mv[1:])[0] + 1:
        lo, hi = max(0, i - 20), min(len(X), i + 21)
        if (seg[lo:hi] != seg[i]).any():
            continue
        w = X[lo:hi] > thr
        if X[lo] > thr or not w.any():
            continue
        on_lags.append((np.argmax(w) + lo - i) * DT)
        at_on.append((X[i] - base) / (p99 - base))
    for i in np.where((~P[1:] & P[:-1]) & (seg[1:] == seg[:-1]))[0] + 1:
        lo, hi = max(0, i - 20), min(len(X), i + 21)
        if (seg[lo:hi] != seg[i]).any():
            continue
        w = X[lo:hi] <= thr
        if X[lo] <= thr or not w.any():
            continue
        off_lags.append((np.argmax(w) + lo - i) * DT)
    m = P & mv & np.isfinite(X)
    xs = (X[m] - base) / (p99 - base)
    ys = -a[m]
    sel = xs > 0.05
    k1 = np.polyfit(xs[sel], ys[sel], 1)
    k2 = np.polyfit(xs[sel], ys[sel], 2)
    st = P & (v < 0.1)
    out = dict(n_press=len(on_lags), p99=p99, base=base,
               onset_lag_s_median=float(np.median(on_lags)) if on_lags else None,
               onset_lag_s_iqr=[float(np.percentile(on_lags, 25)), float(np.percentile(on_lags, 75))] if on_lags else None,
               frac_onsets_before_switch=float(np.mean(np.array(on_lags) < 0)) if on_lags else None,
               release_lag_s_median=float(np.median(off_lags)) if off_lags else None,
               value_at_switch_frac_p99=float(np.median(at_on)) if at_on else None,
               x_intercept_frac_p99=float(-k1[1] / k1[0]),
               linear_slope_per_p99=float(k1[0]),
               quad_term_rel=float(k2[0] / k1[0]),
               r_sample=float(np.corrcoef(xs, ys)[0, 1]),
               standstill_pressed_frac_p99=float(np.nanmedian((X[st] - base) / (p99 - base))) if st.any() else None,
               above_thr_when_not_pressed_moving=float(np.mean(X[~P & mv] > thr)))
    if auto and auto in D:
        A = D[auto].to_numpy() < -0.5
        mm = A & ~P & mv
        out["auto_brake_n"] = int(mm.sum())
        out["auto_brake_above_thr"] = float(np.mean(X[mm] > thr)) if mm.any() else None
        out["auto_brake_decel_median"] = float(np.median(-a[mm])) if mm.any() else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/brake/brake_performance.yaml")
    ap.add_argument("--ccs", default="raw_data/comma_car_segments")
    ap.add_argument("--tss2-segments", type=int, default=150)
    ap.add_argument("--rav4-cache", type=Path, default=None, help="brake_performance_poc の --cache と同じ表 (あれば読み直さない)")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    # RAV4 2017 (comma2k19)
    if a.rav4_cache and a.rav4_cache.exists():
        D = pd.read_pickle(a.rav4_cache)
    else:
        D = BD.load_car("rav4", cfg)
    D = D[~D.edge].reset_index(drop=True)   # セグメントの端は平滑化の縁で減速度が振れるので除く
    D = D.assign(seg=D.drive + "/" + D.seg.astype(str), x=D.level)
    R["rav4_2017_comma2k19_BRAKE_LEVEL"] = behaviour(D, "x", auto="ACC_ACCEL_CMD")

    # RAV4 TSS2 (commaCarSegments)
    refs = CCS.find_segments(a.ccs, "TOYOTA_RAV4_TSS2")
    rng = np.random.default_rng(20261118)
    by_dongle: dict[str, list] = {}
    for r in refs:
        by_dongle.setdefault(r.dongle_id, []).append(r)
    pick = []
    dongles = list(by_dongle)
    rng.shuffle(dongles)
    while len(pick) < a.tss2_segments and dongles:
        for d in list(dongles):
            if by_dongle[d]:
                pick.append(by_dongle[d].pop(rng.integers(len(by_dongle[d]))))
            else:
                dongles.remove(d)
            if len(pick) >= a.tss2_segments:
                break
    tss2 = dict(pressed=spec("BRAKE_PRESSED", 550, 37, 1), pressure=spec("BRAKE_PRESSURE", 550, 0, 9), position=spec("BRAKE_POSITION", 550, 16, 9),
                ws_fl=spec("FL", 170, 6, 15, 0.01), ws_fr=spec("FR", 170, 22, 15, 0.01), ws_rl=spec("RL", 170, 38, 15, 0.01), ws_rr=spec("RR", 170, 54, 15, 0.01),
                acc_cmd=BD.Sig(SignalSpec.from_dict("ACC", dict(can_id=835, start_bit=7, length=16, signed=True, factor=0.001, bus=0)), include_tx=True))
    parts, used = [], []
    for r in pick:
        try:
            d = read_rlog(r.path)
        except Exception as e:  # noqa: BLE001
            continue
        c = d.can
        g = grid_from_frames(c.t, c.address, c.src, c.payload_u64, tss2, ["ws_fl", "ws_fr", "ws_rl", "ws_rr"])
        if g is None or g.pressure.isna().mean() > 0.5:
            continue
        # 輪速は km/h (オフセット -67.67、_toyota_2017.dbc) -> m/s
        for k in ("v", "a"):
            g[k] = (g[k] - (67.67 if k == "v" else 0)) / 3.6
        g["seg"] = f"{r.drive_id}/{r.index}"
        parts.append(g)
        used.append(f"{r.drive_id}/{r.index}")
    T = pd.concat(parts, ignore_index=True)
    R["tss2_segments"] = len(used)
    R["tss2_dongles"] = len({u.split("|")[0] for u in used})
    R["tss2_acc_cmd_present_frac"] = float(T.acc_cmd.notna().mean())
    for x in ("pressure", "position"):
        R[f"rav4_tss2_ccs_{x.upper()}"] = behaviour(T, x, auto="acc_cmd")
    # 踏み込み量と踏力 (油圧) の関係の形
    m = (T.pressed > 0.5) & (T.position > 5)
    if m.sum() > 100:
        q = pd.qcut(T.position[m], 10, duplicates="drop")
        R["tss2_pressure_by_position_decile"] = T.pressure[m].groupby(q, observed=True).median().round(1).astype(float).tolist()
        R["tss2_position_decile_centers"] = T.position[m].groupby(q, observed=True).median().round(1).astype(float).tolist()
    (out / "tss2_segments.txt").write_text("\n".join(used))

    # RAV4 2017 世代の別の車・RAV4 Hybrid (commaCarSegments、各 1 本)
    other = {}
    for plat in ("TOYOTA_RAV4", "TOYOTA_RAV4H"):
        for r in CCS.find_segments(a.ccs, plat):
            d = read_rlog(r.path)
            c = d.can
            row = {}
            for addr in (548, 550):
                for bus in (0, 1, 2):
                    m = frame_mask(c.address, c.src, addr, bus)
                    row[f"0x{addr:03X}_bus{bus}_n"] = int(m.sum())
            sig = dict(pressed=spec("P", 548, 5, 1), level=spec("L", 548, 32, 9), ws_fl=spec("FL", 170, 6, 15, 0.01), ws_fr=spec("FR", 170, 22, 15, 0.01),
                       ws_rl=spec("RL", 170, 38, 15, 0.01), ws_rr=spec("RR", 170, 54, 15, 0.01))
            g = grid_from_frames(c.t, c.address, c.src, c.payload_u64, sig, ["ws_fl", "ws_fr", "ws_rl", "ws_rr"])
            if g is not None and g.level.notna().any():
                g["v"] = (g.v - 67.67) / 3.6
                g["a"] = g.a / 3.6
                P = g.pressed > 0.5
                mv = g.v > 3
                row.update(pressed_frac=float(P.mean()), level_max=float(g.level.max()),
                           level_off_moving_p99=float(g.level[~P & mv].quantile(0.99)) if (~P & mv).any() else None,
                           r_level_decel_pressed_moving=float(np.corrcoef(g.level[P & mv], -g.a[P & mv])[0, 1]) if (P & mv).sum() > 50 else None,
                           fingerprint=str(d.car_params.get("carFingerprint", "")))
            other[f"{plat}:{r.drive_id}/{r.index}"] = row
    R["ccs_rav4_2017_generation"] = other
    (out / "summary.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(run_id=a.run_id, tss2_segments=a.tss2_segments, ccs=a.ccs), ensure_ascii=False, indent=1))
    print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
