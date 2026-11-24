"""TartanDrive 2.0 で、強い振動を IMU だけで取り出し、サスペンションのストロークと運転操作で確かめる PoC。

- 評価量は comma2k19 と同じ (ISO 2631-1 の重み付き 1 秒移動 RMS を座面の係数で合成した MTVV_total。
  scripts/vibration/iso2631.py をそのまま使い、軸は x, y, z, ロール・ピッチの角加速度)
- 強い振動のイベント: 走行中の窓の MTVV_total の上位 1% (TartanDrive 2.0 の中で決める) を超えるピーク
- ストローク (後輪の左右 2 本) は正解ラベルにはしない。補助の確かめとして、イベントの時刻と無作為の時刻で同じ手順で比べる
- racepak (ストローク・車輪の回転・スロットル・ブレーキ) の時刻は IMU に対して 0.4〜1.9 s 遅れ、遅れが動くので、
  幅のある窓で見る。遅れの推定は lags.csv に出す
- 「操作の変化」「ストロークの端」は危険の手がかり (D/E) の探索用であって、実際に危険があったことは言わない

    uv run python scripts/offroad/td2_vibration_poc.py --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (scripts/ の各サブフォルダを import できるようにする)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, find_peaks, sosfiltfilt

import iso2631 as ISO   # scripts/vibration/iso2631.py (読むだけ)

AXES = (("x", "Wd"), ("y", "Wd"), ("z", "Wk"), ("rx", "We"), ("ry", "We"))   # iso2631_events.py と同じ


# ---------------------------------------------------------------- 読み込み
def load_topic(run_dir: Path, sub: str) -> tuple[np.ndarray, np.ndarray] | None:
    d = run_dir / sub
    f = [p for p in d.glob("*.npy")] if d.exists() else []
    if not f or not (d / "timestamps.txt").exists():
        return None
    a = np.load(f[0], allow_pickle=False)
    t = np.loadtxt(d / "timestamps.txt")
    t, ix = np.unique(t, return_index=True)      # 同じ時刻の行は最初の 1 つ
    a = a[ix]
    return t, a.reshape(len(a), -1)


def pedals(c: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """racepak の controls の行列から (ブレーキとみられる値, アクセルとみられる値) を返す。飛び値は NaN。

    列の割り当ては設定の reaction.controls_cols (2026-11-18 に、型の定義の名前と中身が逆とわかった)。
    設定に無いときは、それまでの扱い (0 列目 = スロットル、1 列目 = ブレーキ) にする。
    """
    rc = cfg["reaction"]
    cols = rc.get("controls_cols", {"throttle": 0, "brake": 1})
    inv = rc.get("controls_invalid_above", rc["brake_invalid_above"])
    a = np.where(c > inv, np.nan, c.astype(float))
    return a[:, cols["brake"]], a[:, cols["throttle"]]


def bp(x, lo, hi, fs):
    return sosfiltfilt(butter(2, [lo, hi], btype="band", fs=fs, output="sos"), x)


def lp(x, hi, fs):
    return sosfiltfilt(butter(2, hi, btype="low", fs=fs, output="sos"), x)


def load_run(run_dir: Path, cfg: dict) -> dict | None:
    fs = cfg["grid_hz"]
    imu = load_topic(run_dir, cfg["imu"]["topic"])
    if imu is None or imu[0][-1] - imu[0][0] < cfg["min_run_s"]:
        return None
    ti, im = imu
    c = cfg["imu"]["cols"]
    g = np.arange(ti[0], ti[-1], 1 / fs)
    X = {k: np.interp(g, ti, im[:, j]) for k, j in c.items()}
    # 時刻の抜け: 格子の各点の、直前と直後の IMU サンプルの間隔
    k = np.clip(np.searchsorted(ti, g), 1, len(ti) - 1)
    gap = (ti[k] - ti[k - 1]) > cfg["max_imu_gap_s"]
    od = load_topic(run_dir, cfg["speed"]["topic"])
    speed = np.interp(g, od[0], np.linalg.norm(od[1][:, cfg["speed"]["cols"]], axis=1))
    R = dict(name=run_dir.name, g=g, fs=fs, gap=gap, speed=speed, X=X, odom=od,
             shock=load_topic(run_dir, cfg["shock"]["topic"]),
             controls=load_topic(run_dir, cfg["reaction"]["controls_topic"]),
             steer=load_topic(run_dir, cfg["reaction"]["steering_topic"]))
    return R


# ---------------------------------------------------------------- ISO 2631-1 (comma と同じ合成)
def iso_series(R: dict, lp8_hz: float) -> dict:
    fs, X = R["fs"], R["X"]
    ax = dict(x=X["ax"], y=X["ay"], z=X["az"] - np.mean(X["az"]),
              rx=np.gradient(X["wx"], 1 / fs), ry=np.gradient(X["wy"], 1 / fs))
    aw = {k: ISO.weight(ax[k], fs, w) for k, w in AXES}
    rr = {k: ISO.running_rms(aw[k], fs) for k in aw}
    tot = np.sqrt(sum((ISO.K_SEAT[k] * rr[k]) ** 2 for k in rr))
    z_lp = ISO.weight(lp(ax["z"], lp8_hz, fs), fs, "Wk")
    rr_lp = ISO.running_rms(z_lp, fs)
    tot_lp8 = np.sqrt(sum((ISO.K_SEAT[k] * rr[k]) ** 2 for k in rr if k != "z") + rr_lp ** 2)
    return dict(aw=aw, rr=rr, tot=tot, tot_lp8=tot_lp8, raw=ax)


def window_rows(R: dict, S: dict, cfg: dict) -> list[dict]:
    g, fs = R["g"], R["fs"]
    lo, hi = cfg["windows"]["near_s"]
    rows = []
    for tc in np.arange(g[0] + 5.0, g[-1] - 5.0, cfg["windows"]["step_s"]):
        m = (g >= tc + lo) & (g <= tc + hi)
        if R["gap"][(g >= tc - 4) & (g <= tc + 4)].any() or np.isnan(S["tot"][m]).all():
            continue
        i = np.nanargmax(np.where(m, S["tot"], -1))
        rows.append(dict(run=R["name"], t=round(tc - g[0], 2), speed=float(np.median(R["speed"][m])),
                         mtvv_total=float(S["tot"][i]), mtvv_total_lp8=float(np.nanmax(S["tot_lp8"][m])),
                         mtvv_z=float(np.nanmax(S["rr"]["z"][m])),
                         dominant_axis=max(S["rr"], key=lambda k: ISO.K_SEAT[k] * S["rr"][k][i])))
    return rows


# ---------------------------------------------------------------- racepak の信号
def shock_series(R: dict, cfg: dict) -> dict | None:
    if R["shock"] is None:
        return None
    ts, a = R["shock"]
    fs = R["fs"]
    gs = np.arange(ts[0], ts[-1], 1 / fs)
    sc = cfg["shock"]
    out = dict(g=gs)
    for side, j in sc["use_cols"].items():
        s = np.interp(gs, ts, a[:, j])
        v = bp(np.gradient(s, 1 / fs), *sc["band_hz"], fs)
        out[side] = s
        out[side + "_act"] = ISO.running_rms(v, fs)          # ストローク速度の 1 秒移動 RMS [inch/s とされる]
        out[side + "_vel"] = v
    out["act"] = np.fmax(out["RL_act"], out["RR_act"])
    return out


def max_in(t: np.ndarray, x: np.ndarray, lo: float, hi: float) -> float:
    m = (t >= lo) & (t <= hi)
    return float(np.nanmax(x[m])) if m.any() and np.isfinite(x[m]).any() else np.nan


def min_in(t, x, lo, hi):
    m = (t >= lo) & (t <= hi)
    return float(np.nanmin(x[m])) if m.any() and np.isfinite(x[m]).any() else np.nan


def mean_in(t, x, lo, hi):
    m = (t >= lo) & (t <= hi)
    return float(np.nanmean(x[m])) if m.any() else np.nan


def estimate_lags(R: dict, Sh: dict, cfg: dict) -> list[dict]:
    """包絡線 (2〜10 Hz の |ストローク速度| と |上下加速度|) の相互相関で、racepak の遅れを block ごとに出す。"""
    ls = cfg["shock"]["lag_search"]
    fs, g = R["fs"], R["g"]
    s = np.interp(g, Sh["g"], Sh["RL"] + Sh["RR"])
    es = bp(np.abs(bp(np.gradient(s, 1 / fs), *ls["env_band_hz"], fs)), *ls["env_smooth_hz"], fs)
    ea = bp(np.abs(bp(R["X"]["az"] - R["X"]["az"].mean(), *ls["env_band_hz"], fs)), *ls["env_smooth_hz"], fs)
    L = np.arange(int(ls["range_s"][0] * fs), int(ls["range_s"][1] * fs) + 1, 2)
    n, rows = int(ls["block_s"] * fs), []
    edge = int(3 * fs)
    for i in range(0, len(g) - n + 1, n):
        a, b = es[i:i + n], ea[i:i + n]
        c = np.array([np.corrcoef(np.roll(a, l)[edge:-edge], b[edge:-edge])[0, 1] for l in L])
        j = int(np.nanargmax(c))
        # np.roll(a, l) が b に合う → shock の時刻 t - l/fs の値が IMU の時刻 t に当たる。遅れ = -l/fs
        rows.append(dict(run=R["name"], block_start_s=round(i / fs, 1), lag_s=round(-L[j] / fs, 2), corr=round(float(c[j]), 3),
                         speed_med=round(float(np.median(R["speed"][i:i + n])), 2)))
    return rows


def reaction_metrics(R: dict, te: float, cfg: dict) -> dict:
    rc, (lo, hi) = cfg["reaction"], cfg["shock"]["lag_range_s"]
    out = {}
    if R["controls"] is not None:
        tc, c = R["controls"]
        thr, brk = c[:, 0], np.where(c[:, 1] > rc["brake_invalid_above"], np.nan, c[:, 1])
        pre_t = mean_in(tc, thr, te + lo - rc["pre_s"], te + lo)
        out["throttle_drop"] = pre_t - min_in(tc, thr, te + lo, te + hi + rc["post_s"])
        pre_b = mean_in(tc, brk, te + lo - rc["pre_s"], te + lo)
        out["brake_rise"] = max_in(tc, brk, te + lo, te + hi + rc["post_s"]) - pre_b
    if R["steer"] is not None:
        tsr, s = R["steer"]
        rate = np.r_[0.0, np.diff(s[:, 0]) / np.maximum(np.diff(tsr), 1e-3)]
        out["steer_rate_max"] = max_in(tsr, np.abs(rate), te - 0.5, te + rc["post_s"])
    return out


# ---------------------------------------------------------------- 確かめ (軸・姿勢・ストロークの向き)
def quat_roll_pitch(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, w = q.T     # gps_odom 3〜6 列目。並びは x, y, z, w とみて、角速度との相関で確かめる
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def run_checks(R: dict, Sh: dict | None, lag_med: float) -> dict:
    fs, g = R["fs"], R["g"]
    to, o = R["odom"]
    roll, pitch = quat_roll_pitch(o[:, 3:7])
    rg, pg = np.interp(g, to, np.unwrap(roll)), np.interp(g, to, pitch)
    c = dict(run=R["name"],
             corr_droll_wx=float(np.corrcoef(lp(np.gradient(rg, 1 / fs), 2, fs), lp(R["X"]["wx"], 2, fs))[0, 1]),
             corr_dpitch_wy=float(np.corrcoef(lp(np.gradient(pg, 1 / fs), 2, fs), lp(R["X"]["wy"], 2, fs))[0, 1]))
    if Sh is not None and np.isfinite(lag_med):
        # ストロークの向き: 0.2 Hz 以下で、後輪の平均と前後加速度、左右差と左右加速度 (遅れの中央値でずらす)
        rm = np.interp(g + lag_med, Sh["g"], (Sh["RL"] + Sh["RR"]) / 2)
        rd = np.interp(g + lag_med, Sh["g"], Sh["RL"] - Sh["RR"])
        mv = R["speed"] >= 1.0
        if mv.sum() > 30 * fs:
            c["corr_rear_mean_vs_ax"] = float(np.corrcoef(lp(rm, 0.2, fs)[mv], lp(R["X"]["ax"], 0.2, fs)[mv])[0, 1])
            c["corr_rear_diff_vs_ay"] = float(np.corrcoef(lp(rd, 0.2, fs)[mv], lp(R["X"]["ay"], 0.2, fs)[mv])[0, 1])
    R["roll_g"], R["pitch_g"] = rg, pg
    return c


# ---------------------------------------------------------------- 本体
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    runs, wins, lag_rows, checks = {}, [], [], []
    for d in sorted(Path(cfg["data_dir"]).iterdir()):
        R = load_run(d, cfg)
        if R is None:
            continue
        S = iso_series(R, cfg["lp8_hz"])
        Sh = shock_series(R, cfg)
        lr = estimate_lags(R, Sh, cfg) if Sh is not None else []
        lag_rows += lr
        good = [x["lag_s"] for x in lr if x["speed_med"] >= cfg["moving_min_speed_mps"]]
        checks.append(run_checks(R, Sh, float(np.median(good)) if good else np.nan))
        wins += window_rows(R, S, cfg)
        runs[R["name"]] = (R, S, Sh)
        print(f"{R['name'][:45]:45s} {R['g'][-1] - R['g'][0]:6.0f} s", flush=True)
    W = pd.DataFrame(wins)
    mov = W.speed >= cfg["moving_min_speed_mps"]
    thr = float(np.percentile(W.mtvv_total[mov], cfg["events"]["threshold_percentile"]))

    # ---- イベント (IMU だけ)
    sc, (lo, hi) = cfg["shock"], cfg["shock"]["lag_range_s"]
    ev = []
    for name, (R, S, Sh) in runs.items():
        g, fs = R["g"], R["fs"]
        x = np.where((R["speed"] >= cfg["moving_min_speed_mps"]) & ~R["gap"], np.nan_to_num(S["tot"]), 0)
        pk, _ = find_peaks(x, height=thr, distance=int(cfg["events"]["min_separation_s"] * fs))
        for i in pk:
            te = g[i]
            w = (g >= te - 2.5) & (g <= te + 2.5)
            awz = S["aw"]["z"][w]
            row = dict(run=name, t=round(te - g[0], 2), t_abs=round(te, 3), speed=float(R["speed"][i]),
                       mtvv_total=float(S["tot"][i]), mtvv_total_lp8=float(np.nanmax(S["tot_lp8"][(g >= te - 1) & (g <= te + 1)])),
                       **{f"mtvv_{k}": float(S["rr"][k][i]) for k in S["rr"]},
                       dominant_axis=max(S["rr"], key=lambda k: ISO.K_SEAT[k] * S["rr"][k][i]),
                       crest_z=float(np.max(np.abs(awz)) / max(np.sqrt(np.mean(awz ** 2)), 1e-9)),
                       az_peak_abs=float(np.max(np.abs(S["raw"]["z"][(g >= te - 1) & (g <= te)]))),
                       roll_rate_max_dps=float(np.degrees(np.max(np.abs(R["X"]["wx"][(g >= te - 1) & (g <= te)])))),
                       roll_abs_max_deg=float(np.degrees(np.max(np.abs(R["roll_g"][w] - np.median(R["roll_g"]))))),
                       pitch_abs_max_deg=float(np.degrees(np.max(np.abs(R["pitch_g"][w])))),
                       comfort=ISO.comfort_label(float(S["tot"][i])))
            if Sh is not None:
                row["shock_act"] = max_in(Sh["g"], Sh["act"], te + lo, te + hi)
                for side in sc["use_cols"]:
                    row[f"stroke_{side}_min"] = min_in(Sh["g"], Sh[side], te + lo - 1, te + hi)
                    row[f"stroke_{side}_max"] = max_in(Sh["g"], Sh[side], te + lo - 1, te + hi)
            row.update(reaction_metrics(R, te, cfg))
            ev.append(row)
    E = pd.DataFrame(ev).sort_values("mtvv_total", ascending=False).reset_index(drop=True)
    E.insert(0, "event_id", [f"T{i:03d}" for i in range(len(E))])

    # ---- 無作為の時刻 (走行中) で同じ手順
    null = []
    names = list(runs)
    wts = np.array([(runs[n][0]["speed"] >= cfg["moving_min_speed_mps"]).sum() for n in names], float)
    while len(null) < sc["null_samples"]:
        n = names[rng.choice(len(names), p=wts / wts.sum())]
        R, S, Sh = runs[n]
        cand = np.flatnonzero((R["speed"] >= cfg["moving_min_speed_mps"]) & ~R["gap"])
        i = cand[rng.integers(len(cand))]
        te = R["g"][i]
        et = E.t_abs[E.run == n].to_numpy()
        if len(et) and np.min(np.abs(et - te)) < sc["null_exclude_s"]:
            continue
        if Sh is None or np.isnan(S["tot"][i]):
            continue
        r = dict(run=n, t_abs=te, mtvv_total=float(S["tot"][i]), speed=float(R["speed"][i]),
                 shock_act=max_in(Sh["g"], Sh["act"], te + lo, te + hi))
        for side in sc["use_cols"]:
            r[f"stroke_{side}_min"] = min_in(Sh["g"], Sh[side], te + lo - 1, te + hi)
            r[f"stroke_{side}_max"] = max_in(Sh["g"], Sh[side], te + lo - 1, te + hi)
        r.update(reaction_metrics(R, te, cfg))
        null.append(r)
    N = pd.DataFrame(null)

    # ---- ストロークの端 (全走行・走行中のサンプル)
    ext = {}
    for side in sc["use_cols"]:
        v = np.concatenate([np.interp(R["g"], Sh["g"], Sh[side])[R["speed"] >= cfg["moving_min_speed_mps"]]
                            for R, S, Sh in runs.values() if Sh is not None])
        ext[side] = [float(x) for x in np.percentile(v, sc["extreme_percentiles"])]

    def flags(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        # 値が大きいほど縮み (後輪の平均が前後加速度と正、左右差が左右加速度と負の相関。checks.csv) とみる。
        # 止まりの位置は分からないので、全走行の端に入ったかどうかだけを見る
        f["stroke_compress_extreme"] = np.zeros(len(df), bool)
        f["stroke_extend_extreme"] = np.zeros(len(df), bool)
        for side, (pl, ph) in ext.items():
            f["stroke_compress_extreme"] |= df[f"stroke_{side}_max"] >= ph
            f["stroke_extend_extreme"] |= df[f"stroke_{side}_min"] <= pl
        f["stroke_extreme"] = f.stroke_compress_extreme | f.stroke_extend_extreme
        for k in ("throttle_drop", "brake_rise", "steer_rate_max"):
            if k in df:
                f[f"{k}_flag"] = df[k] >= np.nanpercentile(N[k], cfg["reaction"]["flag_percentile"])
        return f

    pct = lambda col, s: 100 * np.searchsorted(np.sort(N[col].dropna()), s, side="right") / N[col].notna().sum()
    E["shock_act_null_pct"] = np.round(pct("shock_act", E.shock_act), 1)
    # 速度をそろえた比較: イベントの速度 ±max(1 m/s, 25%) の無作為の時刻だけで百分位を取る
    def pct_speed(r):
        w = max(cfg["shock"]["speed_match_abs_mps"], cfg["shock"]["speed_match_rel"] * r.speed)
        v = np.sort(N.shock_act[(N.speed - r.speed).abs() <= w].dropna())
        return (100 * np.searchsorted(v, r.shock_act, side="right") / len(v), len(v)) if len(v) >= 30 else (np.nan, len(v))
    sm = E.apply(pct_speed, axis=1, result_type="expand")
    E["shock_act_null_pct_speed"], E["n_null_speed"] = np.round(sm[0], 1), sm[1].astype(int)
    E = pd.concat([E, flags(E)], axis=1)
    E.insert(2, "site", E.run.str.replace(r"\d{4}(-\d{2}){5}", "", regex=True).str.strip("_"))
    Nf = flags(N)
    E["reasons"] = E.apply(lambda r: ";".join(
        [f"MTVV_total {r.mtvv_total:.2f} >= TD2 走行中の p{cfg['events']['threshold_percentile']:.0f} {thr:.2f}"]
        + ([f"ストローク速度 無作為の p{r.shock_act_null_pct:.0f}"] if np.isfinite(r.shock_act_null_pct) else [])
        + (["ストロークが全走行の端"] if r.stroke_extreme else [])
        + (["スロットルを戻す"] if r.get("throttle_drop_flag", False) else [])
        + (["ブレーキ"] if r.get("brake_rise_flag", False) else [])
        + (["速い操舵"] if r.get("steer_rate_max_flag", False) else [])), axis=1)

    # ---- 逆向き: ストロークの大きいピークのうち、IMU でも強い振動のもの
    act_all = np.concatenate([Sh["act"][np.interp(Sh["g"], R["g"], R["speed"]) >= cfg["moving_min_speed_mps"]]
                              for R, S, Sh in runs.values() if Sh is not None])
    act_thr = float(np.nanpercentile(act_all, 99))
    tot_p90 = float(np.percentile(W.mtvv_total[mov], 90))
    rec = []
    for name, (R, S, Sh) in runs.items():
        if Sh is None:
            continue
        sp = np.interp(Sh["g"], R["g"], R["speed"])
        x = np.where(sp >= cfg["moving_min_speed_mps"], np.nan_to_num(Sh["act"]), 0)
        for i in find_peaks(x, height=act_thr, distance=int(cfg["events"]["min_separation_s"] * R["fs"]))[0]:
            ts = Sh["g"][i]
            rec.append(dict(run=name, t_abs=ts, imu_max=max_in(R["g"], S["tot"], ts - hi, ts - lo)))
    Rc = pd.DataFrame(rec)
    Nimu = np.array([max_in(runs[r][0]["g"], runs[r][1]["tot"], t - hi, t - lo) for r, t in zip(N.run, N.t_abs)])

    # ---- comma2k19 (読むだけ)
    cb = pd.read_csv(cfg["comma_reference"]["normal_windows"])
    ce = pd.read_csv(cfg["comma_reference"]["events"])
    q = lambda s: {f"p{p}": round(float(np.nanpercentile(s, p)), 3) for p in (50, 90, 99)}
    sets = {"td2_moving": W.mtvv_total[mov], "td2_ge5mps": W.mtvv_total[W.speed >= cfg["comma_like_min_speed_mps"]],
            "td2_standstill": W.mtvv_total[W.speed < cfg["standstill_max_speed_mps"]],
            "comma_normal": cb.mtvv_total, "comma_events": ce.mtvv_total}
    sets_lp8 = {"td2_moving": W.mtvv_total_lp8[mov], "td2_standstill": W.mtvv_total_lp8[W.speed < cfg["standstill_max_speed_mps"]],
                "comma_normal": cb.mtvv_total_lp8, "comma_events": ce.mtvv_total_lp8}
    Lg = pd.DataFrame(lag_rows)
    Lm = Lg[Lg.speed_med >= cfg["moving_min_speed_mps"]]
    C = pd.DataFrame(checks)
    summ = dict(
        runs=len(runs), hours=round(sum(R["g"][-1] - R["g"][0] for R, _, _ in runs.values()) / 3600, 3),
        windows=dict(all=len(W), moving=int(mov.sum())),
        mtvv_total={k: dict(n=int(v.notna().sum()), **q(v)) for k, v in sets.items()},
        mtvv_total_lp8={k: dict(n=int(v.notna().sum()), **q(v)) for k, v in sets_lp8.items()},
        frac_moving_windows_ge={str(x): round(float((W.mtvv_total[mov] >= x).mean()), 3) for x in cfg["events"]["iso_levels"]},
        frac_comma_normal_ge={str(x): round(float((cb.mtvv_total >= x).mean()), 3) for x in cfg["events"]["iso_levels"]},
        dominant_axis_moving=W.dominant_axis[mov].value_counts().to_dict(),
        event_threshold=round(thr, 3), events=len(E), events_dominant_axis=E.dominant_axis.value_counts().to_dict(),
        events_comfort=E.comfort.value_counts().to_dict(), events_site=E.site.value_counts().to_dict(),
        events_speed=dict(p10=float(E.speed.quantile(0.1)), p50=float(E.speed.median()), p90=float(E.speed.quantile(0.9))),
        lag=dict(blocks=len(Lm), p10=float(Lm.lag_s.quantile(0.1)), p50=float(Lm.lag_s.median()), p90=float(Lm.lag_s.quantile(0.9)),
                 corr_p50=float(Lm["corr"].median()),
                 within_run_range_p50=float(Lm.groupby("run").lag_s.agg(lambda s: s.max() - s.min())[Lm.groupby("run").size() >= 3].median())),
        shock_validation=dict(
            window_s=[lo, hi],
            events_shock_null_pct_p50=float(E.shock_act_null_pct.median()),
            events_frac_shock_ge_null_p90=float((E.shock_act_null_pct >= 90).mean()),
            events_shock_null_pct_speed_p50=float(E.shock_act_null_pct_speed.median()),
            events_frac_shock_ge_null_speed_p90=float((E.shock_act_null_pct_speed >= 90).mean()),
            events_n_speed_matched=int(E.shock_act_null_pct_speed.notna().sum()),
            null_frac_ge_p90=0.1,
            # 順位の AUC: イベントの shock_act が無作為の時刻より大きい確率
            auc_events_vs_null=float(np.mean([np.mean(N.shock_act.dropna() < s) for s in E.shock_act.dropna()])),
            corr_spearman_null_mtvv_vs_shock=float(N[["mtvv_total", "shock_act"]].corr("spearman").iloc[0, 1]),
            shock_peaks=len(Rc), shock_peak_threshold=act_thr,
            shock_peaks_frac_imu_ge_event_thr=float((Rc.imu_max >= thr).mean()) if len(Rc) else None,
            shock_peaks_frac_imu_ge_p90=float((Rc.imu_max >= tot_p90).mean()) if len(Rc) else None,
            null_frac_imu_ge_event_thr=float(np.nanmean(Nimu >= thr)), null_frac_imu_ge_p90=float(np.nanmean(Nimu >= tot_p90))),
        stroke_extremes=ext,
        flags_events={k: round(float(E[k].mean()), 3) for k in Nf.columns},
        flags_null={k: round(float(Nf[k].mean()), 3) for k in Nf.columns},
        checks=dict(corr_droll_wx_p50=float(C.corr_droll_wx.median()), corr_dpitch_wy_p50=float(C.corr_dpitch_wy.median()),
                    corr_rear_mean_vs_ax_p50=float(C.corr_rear_mean_vs_ax.median()),
                    corr_rear_mean_vs_ax_frac_neg=float((C.corr_rear_mean_vs_ax < 0).mean()),
                    corr_rear_diff_vs_ay_p50=float(C.corr_rear_diff_vs_ay.median()),
                    corr_rear_diff_vs_ay_frac_neg=float((C.corr_rear_diff_vs_ay < 0).mean())),
    )
    W.round(4).to_csv(out / "windows.csv", index=False)
    E.round(4).to_csv(out / "events.csv", index=False)
    N.round(4).to_csv(out / "null_samples.csv", index=False)
    Lg.to_csv(out / "lags.csv", index=False)
    C.round(4).to_csv(out / "checks.csv", index=False)
    Rc.round(3).to_csv(out / "shock_peaks.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    (out / "meta.json").write_text(json.dumps(dict(config=cfg, config_path=a.config, run_id=a.run_id), ensure_ascii=False, indent=2))
    plots(out, sets, E, N, runs, cfg, thr, Lm)
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    return 0


# ---------------------------------------------------------------- 図
def plots(out: Path, sets: dict, E: pd.DataFrame, N: pd.DataFrame, runs: dict, cfg: dict, thr: float, Lm: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    lab = {"td2_moving": "TD2 走行中 (≥1 m/s)", "td2_ge5mps": "TD2 ≥5 m/s", "td2_standstill": "TD2 停車中",
           "comma_normal": "comma2k19 普段の窓", "comma_events": "comma2k19 振動イベント"}
    col = {"td2_moving": "#1f4e9c", "td2_ge5mps": "#5b8bd6", "td2_standstill": "#9aa5b1", "comma_normal": "#c2410c", "comma_events": "#f59e0b"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, v in sets.items():
        v = np.sort(v.dropna().to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=2, color=col[k], label=f"{lab[k]} (n={len(v)})")
    for x, s in ((cfg["comma_reference"]["candidate_floor"], "comma 候補の下限 0.9"), (2.0, "ISO 極めて不快 2.0"), (thr, f"TD2 イベント閾値 {thr:.2f}")):
        ax.axvline(x, color="#6b7280", lw=1, ls="--")
        ax.text(x, 0.02, " " + s, rotation=90, fontsize=8, color="#374151", va="bottom")
    ax.set_xlabel("MTVV_total [m/s²] (ISO 2631-1 の重み付き 1 秒移動 RMS の最大、座面の係数で合成)")
    ax.set_ylabel("累積割合")
    ax.set_xlim(0, 7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "mtvv_cdf_td2_vs_comma.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    b = np.linspace(0, 100, 21)
    ax[0].hist(E.shock_act_null_pct.dropna(), bins=b, color="#1f4e9c")
    ax[0].axhline(len(E) / 20, color="#c2410c", ls="--", lw=1, label="無作為なら各 5%")
    ax[0].set_xlabel("イベントのストローク速度 (無作為の時刻での百分位)")
    ax[0].set_ylabel("イベント数")
    ax[0].legend(fontsize=8)
    ax[1].hist(Lm.lag_s, bins=np.arange(-2.5, 1.6, 0.1), color="#5b8bd6")
    ax[1].set_xlabel("racepak の遅れ [s] (60 s ごとの推定、走行中)")
    ax[1].set_ylabel("block 数")
    fig.tight_layout()
    fig.savefig(out / "shock_validation.png", dpi=150)
    plt.close(fig)

    pd_dir = out / "event_panels"
    pd_dir.mkdir(exist_ok=True)
    lo, hi = cfg["shock"]["lag_range_s"]
    for _, e in E.head(cfg["report"]["top_n_panels"]).iterrows():
        R, S, Sh = runs[e.run]
        g, te = R["g"], e.t_abs
        m = (g >= te - 6) & (g <= te + 6)
        fig, axs = plt.subplots(5, 1, figsize=(9, 9), sharex=True)
        axs[0].plot(g[m] - te, S["tot"][m], color="#1f4e9c", label="MTVV_total (1 s 移動)")
        axs[0].axhline(thr, color="#6b7280", ls="--", lw=1)
        axs[0].set_ylabel("m/s²")
        axs[0].legend(fontsize=8, loc="upper left")
        axs[1].plot(g[m] - te, S["raw"]["z"][m], lw=0.8, color="#374151", label="上下加速度 (重力を除く)")
        axs[1].set_ylabel("m/s²")
        axs[1].legend(fontsize=8, loc="upper left")
        axs[2].plot(g[m] - te, np.degrees(R["X"]["wx"][m]), lw=0.8, color="#7c3aed", label="ロール角速度")
        axs[2].set_ylabel("deg/s")
        axs[2].legend(fontsize=8, loc="upper left")
        if Sh is not None:
            ms = (Sh["g"] >= te - 6) & (Sh["g"] <= te + 6)
            for side, c in (("RL", "#c2410c"), ("RR", "#0f766e")):
                axs[3].plot(Sh["g"][ms] - te, Sh[side][ms], color=c, label=f"ストローク {side} (racepak の時刻のまま)")
            axs[3].axvspan(lo, hi, color="#fde68a", alpha=0.4, label="照合に使う窓")
            axs[3].set_ylabel("inch とされる")
            axs[3].legend(fontsize=7, loc="upper left")
        axs[4].plot(g[m] - te, R["speed"][m], color="#111827", label="速度 [m/s]")
        if R["controls"] is not None:
            tc, c = R["controls"]
            mc = (tc >= te - 6) & (tc <= te + 6)
            a2 = axs[4].twinx()
            a2.plot(tc[mc] - te, c[mc, 0], color="#16a34a", lw=1, label="スロットル (racepak、単位未確認)")
            a2.legend(fontsize=7, loc="upper right")
        axs[4].legend(fontsize=7, loc="upper left")
        axs[4].set_xlabel("イベントの時刻からの秒 (IMU の時刻)")
        fig.suptitle(f"{e.event_id} {e.run}  t={e.t:.1f} s  MTVV_total {e.mtvv_total:.2f}  主な軸 {e.dominant_axis}", fontsize=9)
        fig.tight_layout()
        fig.savefig(pd_dir / f"{e.event_id}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
