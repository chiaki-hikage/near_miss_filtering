"""片輪が路面の凹凸を踏んだ可能性のある区間を、comma2k19 の IMU から探す。

上下加速度だけでは両輪同時の継ぎ目と区別できないので、上下とロールが
同時に普段より大きい区間を「片輪段差候補 (roll_heave)」として順位付けする。
ピッチだけが大きい区間は「両輪同時候補 (pitch_only)」として別に残す。

普段の大きさは車速で変わるため、各信号の移動 RMS を車速ビンごとの中央値で
割った比 (普段の何倍か) で比べる。IMU とは別系統の裏付けとして、
左右の輪速 (CAN) の細かい揺れの大きさも並べて出す。

    uv run python scripts/vibration/find_roll_bumps.py raw_data/Chunk_1 \
        --out out/vibration/search_roll --clips 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, find_peaks, sosfiltfilt

from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
G = 9.80665
EDGE_TRIM_S = 1.0  # filtfilt の端の影響を避ける
# 端を NaN にした区間に輪速の窓がかかると出る。該当区間は NaN のまま扱う
warnings.filterwarnings("ignore", "All-NaN slice")


@dataclass
class Seg:
    drive_id: str
    index: int
    path: Path
    t0: float            # IMU 先頭の boot time。セグメント内経過時刻の 0 秒
    t: np.ndarray        # 経過時刻 [s]
    v: np.ndarray
    steer: np.ndarray
    yaw_dps: np.ndarray  # 左旋回正
    ay: np.ndarray       # v × yaw
    raw: dict            # 帯域通過前の上下・ロール・ピッチ
    bp: dict             # 帯域通過後
    env: dict            # 移動 RMS


def _load(d: Path) -> tuple[np.ndarray, np.ndarray]:
    t = np.load(d / "t", allow_pickle=True).ravel().astype(float)
    v = np.load(d / "value", allow_pickle=True).astype(float)
    t, i = np.unique(t, return_index=True)
    return t, v[i]


def _rotation_to_down(mean_acc: np.ndarray) -> np.ndarray:
    """平均の比力が [0, 0, -g] を向くように、取付の傾きを戻す回転。"""
    a = mean_acc / np.linalg.norm(mean_acc)
    b = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, b)
    c = float(a @ b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _bandpass(x: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _lowpass(x: np.ndarray, hi: float, fs: float) -> np.ndarray:
    return sosfiltfilt(butter(2, hi, btype="low", fs=fs, output="sos"), x)


def _mrms(x: np.ndarray, n: int) -> np.ndarray:
    return np.sqrt(np.convolve(x * x, np.ones(n) / n, mode="same"))


def gyro_rate_hz(seg_dir: Path) -> float:
    t, _ = _load(seg_dir / "processed_log/IMU/gyro")
    return (len(t) - 1) / (t[-1] - t[0])


def drive_rotation(paths: list[Path]) -> np.ndarray:
    acc = np.vstack([_load(p / "processed_log/IMU/accelerometer")[1] for p in paths])
    return _rotation_to_down(acc.mean(axis=0))


def load_seg(ref, rot: np.ndarray, cfg: dict) -> Seg:
    p = ref.path
    fs = cfg["grid_hz"]
    ta, a = _load(p / "processed_log/IMU/accelerometer")
    tg, g = _load(p / "processed_log/IMU/gyro")
    tc, v = _load(p / "processed_log/CAN/speed")
    ts, st = _load(p / "processed_log/CAN/steering_angle")
    tw, w = _load(p / "processed_log/CAN/wheel_speed")
    t0 = ta[0]
    lo = max(ta[0], tg[0], tc[0], tw[0])
    hi = min(ta[-1], tg[-1], tc[-1], tw[-1])
    grid = np.arange(lo, hi, 1 / fs)
    a = np.einsum("ij,nj->ni", rot, a)
    g = np.einsum("ij,nj->ni", rot, g)
    ai = np.c_[[np.interp(grid, ta, a[:, k]) for k in range(3)]].T
    gi = np.c_[[np.interp(grid, tg, g[:, k]) for k in range(3)]].T
    vv = np.interp(grid, tc, v.ravel())
    raw = {
        "heave": -ai[:, 2] - G,
        "roll": np.rad2deg(gi[:, 0]),   # 正 = 右側が下がる向き
        "pitch": np.rad2deg(gi[:, 1]),  # 正 = 機首上げ
    }
    for k, name in enumerate(("FL", "FR", "RL", "RR")):
        raw[name] = np.interp(grid, tw, w[:, k])
    yaw = _lowpass(-np.rad2deg(gi[:, 2]), 2.0, fs)
    bp, env = {}, {}
    n = max(1, int(round(cfg["envelope_s"] * fs)))
    trim = int(EDGE_TRIM_S * fs)
    for k, x in raw.items():
        band = cfg["band_hz"]["wheel" if k in ("FL", "FR", "RL", "RR") else k]
        bp[k] = _bandpass(x, band[0], band[1], fs)
        e = _mrms(bp[k], n)
        e[:trim] = np.nan
        e[-trim:] = np.nan
        env[k] = e
    return Seg(
        drive_id=ref.drive_id, index=ref.index, path=p, t0=t0, t=grid - t0, v=vv,
        steer=np.interp(grid, ts, st.ravel()), yaw_dps=yaw,
        ay=_lowpass(vv, 2.0, fs) * np.deg2rad(yaw), raw=raw, bp=bp, env=env,
    )


def baselines(segs: list[Seg], cfg: dict) -> pd.DataFrame:
    """車速ビンごとの包絡線の中央値。「普段の大きさ」の基準。"""
    rows = []
    for s in segs:
        b = np.floor(s.v / cfg["speed_bin_mps"]).astype(int)
        ok = (s.v >= cfg["min_speed_mps"]) & np.isfinite(s.env["heave"])
        rows.append(pd.DataFrame({"bin": b[ok], **{k: s.env[k][ok] for k in s.env}}))
    df = pd.concat(rows)
    out = df.groupby("bin").median()
    out["n"] = df.groupby("bin").size()
    return out


def ratios(s: Seg, base: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    b = np.floor(s.v / cfg["speed_bin_mps"]).astype(int)
    full = base.reindex(range(base.index.min(), base.index.max() + 1)).interpolate()
    b = np.clip(b, full.index.min(), full.index.max())
    r = {k: s.env[k] / full.loc[b, k].to_numpy() for k in s.env}
    low = s.v < cfg["min_speed_mps"]
    for k in r:
        r[k][low] = np.nan
    return r


def _span(score: np.ndarray, i: int, frac: float) -> tuple[int, int]:
    th = score[i] * frac
    a = i
    while a > 0 and np.nan_to_num(score[a - 1]) > th:
        a -= 1
    b = i
    while b < len(score) - 1 and np.nan_to_num(score[b + 1]) > th:
        b += 1
    return a, b


def rebound(x: np.ndarray, frac: float) -> tuple[int, float]:
    """符号が交互に入れ替わる極値の数 (揺れ戻しの回数) と、最大の極値に続く逆符号の極値の比。"""
    m = np.max(np.abs(x))
    if m == 0:
        return 0, np.nan
    pk, _ = find_peaks(np.abs(x), height=frac * m)
    if len(pk) == 0:
        return 0, np.nan
    sgn = np.sign(x[pk])
    keep = [0]
    for j in range(1, len(pk)):
        if sgn[j] != sgn[keep[-1]]:
            keep.append(j)
        elif abs(x[pk[j]]) > abs(x[pk[keep[-1]]]):
            keep[-1] = j
    ext = x[pk[keep]]
    k = int(np.argmax(np.abs(ext)))
    decay = abs(ext[k + 1]) / abs(ext[k]) if k + 1 < len(ext) else np.nan
    return len(ext) - 1, decay


def dominant_hz(x: np.ndarray, fs: float) -> float:
    n = 1 << int(np.ceil(np.log2(max(len(x), 8) * 4)))
    f = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), n))
    return float(f[1:][np.argmax(P[1:])])


def ecef_to_llh(x: float, y: float, z: float) -> tuple[float, float]:
    a, e2 = 6378137.0, 6.69437999014e-3
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    return float(np.rad2deg(lat)), float(np.rad2deg(lon))


def pose_at(s: Seg, t_rel: float) -> tuple[float, float, float]:
    ft = np.load(s.path / "global_pose/frame_times") - s.t0
    pos = np.load(s.path / "global_pose/frame_positions")
    vel = np.load(s.path / "global_pose/frame_velocities")
    i = int(np.clip(np.searchsorted(ft, t_rel), 0, len(ft) - 1))
    lat, lon = ecef_to_llh(*pos[i])
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    east = -np.sin(lo) * vel[i, 0] + np.cos(lo) * vel[i, 1]
    north = (-np.sin(la) * np.cos(lo) * vel[i, 0] - np.sin(la) * np.sin(lo) * vel[i, 1]
             + np.cos(la) * vel[i, 2])
    return lat, lon, float(np.rad2deg(np.arctan2(east, north)) % 360)


def find_events(s: Seg, r: dict, cfg: dict, wheelbase: float) -> list[dict]:
    fs = cfg["grid_hz"]
    ev_cfg = cfg["event"]
    rh_c, po_c = ev_cfg["roll_heave"], ev_cfg["pitch_only"]
    score = np.sqrt(np.nan_to_num(r["heave"]) * np.nan_to_num(r["roll"]))
    dist = int(ev_cfg["min_separation_s"] * fs)
    out = []
    seen: list[int] = []
    for group, sig in (("roll_heave", score), ("pitch_only", np.nan_to_num(r["pitch"]))):
        pk, _ = find_peaks(sig, distance=dist)
        for i in pk:
            rh, rr, rp = r["heave"][i], r["roll"][i], r["pitch"][i]
            if not np.isfinite(rh + rr + rp):
                continue
            if group == "roll_heave":
                ok = (rr >= rh_c["roll_ratio_min"] and rh >= rh_c["heave_ratio_min"]
                      and rr / rp >= rh_c["roll_over_pitch_min"])
            else:
                ok = rp >= po_c["pitch_ratio_min"] and rr < po_c["roll_ratio_max"]
                if any(abs(i - j) < dist for j in seen):
                    ok = False
            if not ok:
                continue
            seen.append(i)
            out.append(_describe(s, r, i, group, sig, cfg, wheelbase))
    return out


def _describe(s: Seg, r: dict, i: int, group: str, sig: np.ndarray, cfg: dict, wheelbase: float) -> dict:
    fs = cfg["grid_hz"]
    a, b = _span(sig, i, cfg["event"]["half_width_frac"])
    pad = int(0.2 * fs)
    w = slice(max(0, a - pad), min(len(s.t), b + pad + 1))
    pk = {k: float(np.max(np.abs(s.bp[k][w]))) for k in ("heave", "roll", "pitch")}

    rw = cfg["rebound"]["window_s"]
    rs = slice(max(0, i + int(rw[0] * fs)), min(len(s.t), i + int(rw[1] * fs)))
    n_reb, decay = rebound(s.bp["roll"][rs], cfg["rebound"]["extremum_frac"])
    after = r["roll"][i:min(len(s.t), i + int(3 * fs))]
    settle = np.argmax(np.nan_to_num(after, nan=0) < 1.5)
    first = s.bp["roll"][rs]
    j0 = int(np.argmax(np.abs(first) >= 0.5 * np.max(np.abs(first))))

    m = cfg["maneuver"]
    mw = slice(max(0, i - int(m["window_s"] * fs)), min(len(s.t), i + int(m["window_s"] * fs)))
    steer_rng = float(np.ptp(s.steer[mw]))
    yaw_rng = float(np.ptp(s.yaw_dps[mw]))
    ay_rng = float(np.ptp(s.ay[mw]))
    maneuver = (steer_rng > m["steer_range_deg"] or yaw_rng > m["yaw_range_dps"]
                or ay_rng > m["ay_range_mps2"])

    nmax = cfg["spike"]["max_samples_above_half"]
    above = {k: int((np.abs(s.bp[k][w]) > 0.5 * pk[k]).sum()) for k in ("heave", "roll")}
    spike = above["heave"] <= nmax and above["roll"] <= nmax

    ww = cfg["wheel"]["window_s"]
    v_pk = float(s.v[i])
    lag_exp = wheelbase / max(v_pk, 0.1)
    wsl = slice(max(0, i + int(ww[0] * fs)), min(len(s.t), i + int((ww[1] + lag_exp) * fs)))
    wr = {k: float(np.nanmax(r[k][wsl])) for k in ("FL", "FR", "RL", "RR")}
    left, right = max(wr["FL"], wr["RL"]), max(wr["FR"], wr["RR"])
    front = np.nanmax(np.c_[r["FL"][wsl], r["FR"][wsl]], axis=1)
    rear = np.nanmax(np.c_[r["RL"][wsl], r["RR"][wsl]], axis=1)
    lag = (np.nanargmax(rear) - np.nanargmax(front)) / fs

    hw = s.bp["heave"][w]
    lat, lon, hdg = pose_at(s, float(s.t[i]))
    return dict(
        group=group, drive_id=s.drive_id, segment=s.index,
        t_peak_s=round(float(s.t[i]), 2), t_start_s=round(float(s.t[a]), 2), t_end_s=round(float(s.t[b]), 2),
        speed_mps=round(v_pk, 2), speed_kmh=round(v_pk * 3.6, 1),
        heave_peak_mps2=round(pk["heave"], 2), roll_rate_peak_dps=round(pk["roll"], 2),
        pitch_rate_peak_dps=round(pk["pitch"], 2),
        heave_ratio=round(float(r["heave"][i]), 2), roll_ratio=round(float(r["roll"][i]), 2),
        pitch_ratio=round(float(r["pitch"][i]), 2),
        score=round(float(sig[i]), 2),
        roll_over_pitch=round(float(r["roll"][i] / r["pitch"][i]), 2),
        roll_first_dir="右下がり" if first[j0] > 0 else "左下がり",
        roll_rebounds=n_reb, roll_decay=round(float(decay), 2) if np.isfinite(decay) else np.nan,
        roll_settle_s=round(float(settle) / fs, 2) if np.any(np.nan_to_num(after) < 1.5) else np.nan,
        heave_dom_hz=round(dominant_hz(hw, fs), 1),
        steer_range_deg=round(steer_rng, 1), yaw_range_dps=round(yaw_rng, 2), ay_range_mps2=round(ay_rng, 2),
        maneuver=maneuver, spike_like=spike,
        wheel_ratio_FL=round(wr["FL"], 1), wheel_ratio_FR=round(wr["FR"], 1),
        wheel_ratio_RL=round(wr["RL"], 1), wheel_ratio_RR=round(wr["RR"], 1),
        wheel_lr_log2=round(float(np.log2(left / right)), 2),
        wheel_front_rear_lag_s=round(float(lag), 2), wheel_lag_expected_s=round(lag_exp, 2),
        lat=round(lat, 6), lon=round(lon, 6), heading_deg=round(hdg, 1),
    )


def repeats(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """同じ地点・同じ向きで、別ドライブにも候補があるか。"""
    rc = cfg["repeat"]
    pool = ev.nlargest(rc["pool"], "score")
    lat0 = np.deg2rad(pool.lat.mean())
    xy = np.c_[pool.lon.to_numpy() * 111320 * np.cos(lat0), pool.lat.to_numpy() * 110540]
    hits, drives = [], []
    for _, e in ev.iterrows():
        p = np.array([e.lon * 111320 * np.cos(lat0), e.lat * 110540])
        d = np.hypot(*(xy - p).T)
        dh = np.abs((pool.heading_deg.to_numpy() - e.heading_deg + 180) % 360 - 180)
        m = (d <= rc["radius_m"]) & (dh <= rc["heading_deg"]) & (pool.drive_id.to_numpy() != e.drive_id)
        other = sorted(set(pool.drive_id[m]))
        hits.append(len(other))
        drives.append(";".join(o.split("|")[1] for o in other))
    ev = ev.copy()
    ev["repeat_drives"] = hits
    ev["repeat_drive_ids"] = drives
    return ev


def reason(e: pd.Series) -> str:
    parts = [f"上下 {e.heave_ratio:.1f} 倍・ロール {e.roll_ratio:.1f} 倍が同時 (普段比)"]
    parts.append(f"ロール/ピッチ比 {e.roll_over_pitch:.2f}")
    if e.roll_rebounds >= 2:
        parts.append(f"揺れ戻し {int(e.roll_rebounds)} 回")
    if abs(e.wheel_lr_log2) >= 0.5:
        parts.append(f"輪速の揺れが{'左' if e.wheel_lr_log2 > 0 else '右'}側で {2 ** abs(e.wheel_lr_log2):.1f} 倍")
    if e.repeat_drives > 0:
        parts.append(f"同地点で別 {int(e.repeat_drives)} ドライブにも候補")
    return "。".join(parts)


def drive_tag(drive_id: str) -> str:
    stamp = drive_id.split("|")[1]          # 2018-08-02--08-34-47
    date, time = stamp.split("--")
    return f"{date[5:]}--{time[:5]}"        # 08-02--08-34


def join_segs(parts: list[Seg], base: Seg, cfg: dict) -> Seg:
    """連番セグメントを base の時間軸でつなぐ。帯域通過はつないだ後にかけ直す。"""
    if len(parts) == 1:
        return base
    cat = lambda f: np.concatenate([f(p) for p in parts])
    raw = {k: cat(lambda p: p.raw[k]) for k in base.raw}
    fs = cfg["grid_hz"]
    bp = {}
    for k, x in raw.items():
        band = cfg["band_hz"]["wheel" if k in ("FL", "FR", "RL", "RR") else k]
        bp[k] = _bandpass(x, band[0], band[1], fs)
    return Seg(
        drive_id=base.drive_id, index=base.index, path=base.path, t0=base.t0,
        t=cat(lambda p: p.t + p.t0 - base.t0), v=cat(lambda p: p.v), steer=cat(lambda p: p.steer),
        yaw_dps=cat(lambda p: p.yaw_dps), ay=cat(lambda p: p.ay), raw=raw, bp=bp, env={},
    )


def clip_video(parts: list[Seg], base: Seg, a: float, b: float, out: Path) -> dict:
    """[a, b] に入るフレームを、必要なら隣のセグメントの映像もつないで切り出す。"""
    inputs, chains, frames, times = [], [], [], []
    for p in parts:
        ft = np.load(p.path / "global_pose/frame_times") - base.t0
        idx = np.flatnonzero((ft >= a) & (ft <= b))
        if len(idx) == 0:
            continue
        k = len(inputs) // 4
        inputs += ["-r", "20", "-i", str(p.path / "video.hevc")]
        chains.append(f"[{k}:v]select='between(n\\,{idx[0]}\\,{idx[-1]})',setpts=N/(20*TB)[v{k}]")
        frames.append(f"seg{p.index}:{idx[0]}-{idx[-1]}")
        times += [ft[idx[0]], ft[idx[-1]]]
    n = len(chains)
    graph = ";".join(chains) + ";" + "".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + inputs
           + ["-filter_complex", graph, "-map", "[out]", "-fps_mode", "passthrough",
              "-c:v", "libx264", "-preset", "slow", "-crf", "16",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)])
    subprocess.run(cmd, check=True)
    return dict(frames=";".join(frames), t_first=round(float(times[0]), 3), t_last=round(float(times[-1]), 3),
                cmd=" ".join(c if " " not in c and "|" not in c else f'"{c}"' for c in cmd))


def plot_event(s: Seg, e: pd.Series, a: float, b: float, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    m = (s.t >= a) & (s.t <= b)
    t = s.t[m]
    fig, axs = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    spec = [
        ("heave", "上下加速度 [m/s²]\n(上向き正)", "tab:blue"),
        ("roll", "ロール角速度 [deg/s]\n(正=右下がり)", "tab:green"),
        ("pitch", "ピッチ角速度 [deg/s]\n(正=機首上げ)", "tab:purple"),
    ]
    for ax, (k, lab, c) in zip(axs, spec):
        raw = s.raw[k][m] - (np.mean(s.raw[k][m]) if k == "heave" else 0)
        ax.plot(t, raw, lw=0.5, color="0.65", label="生値")
        ax.plot(t, s.bp[k][m], lw=0.8, color=c, label="帯域通過 (0.7 Hz〜)")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel(lab, fontsize=9)
    axs[0].legend(loc="upper right", fontsize=8, ncol=2)
    axs[3].plot(t, s.v[m] * 3.6, color="tab:red", lw=0.9)
    axs[3].set_ylabel("車速 [km/h]", fontsize=9)
    ax4 = axs[4]
    ax4.plot(t, s.steer[m], color="tab:brown", lw=0.9, label="ハンドル角 [deg]")
    ax4.set_ylabel("ハンドル角 [deg]", fontsize=9)
    tw = ax4.twinx()
    tw.plot(t, s.yaw_dps[m], color="tab:cyan", lw=0.9, label="ヨーレート [deg/s]")
    tw.set_ylabel("ヨーレート [deg/s]", fontsize=9)
    ax4.legend(loc="upper left", fontsize=8)
    tw.legend(loc="upper right", fontsize=8)
    for k, c in (("FL", "tab:blue"), ("RL", "tab:cyan"), ("FR", "tab:red"), ("RR", "tab:orange")):
        axs[5].plot(t, s.bp[k][m], lw=0.7, color=c, label=k)
    axs[5].set_ylabel("輪速の揺れ [m/s]\n(2〜15 Hz)", fontsize=9)
    axs[5].legend(loc="upper right", fontsize=8, ncol=4)
    for ax in axs:
        ax.axvspan(e.t_start_s, e.t_end_s, color="orange", alpha=0.25)
        ax.grid(alpha=0.3)
    axs[-1].set_xlim(a, b)
    axs[-1].set_xlabel("セグメント内経過時刻 [s] (IMU 先頭 = 0 s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--vehicle", type=Path, default=REPO / "configs/vehicles/toyota_rav4.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/search_roll")
    ap.add_argument("--clips", type=int, default=0, help="上位この件数の動画とグラフを作る")
    ap.add_argument("--video-dir", type=Path, default=REPO / "out/vibration/videos")
    ap.add_argument("--plot-dir", type=Path, default=REPO / "out/vibration/plots")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    wheelbase = yaml.safe_load(args.vehicle.read_text())["geometry"]["wheelbase_m"]
    args.out.mkdir(parents=True, exist_ok=True)

    refs = find_segments(args.data_root)
    rates = {r.segment_id: gyro_rate_hz(r.path) for r in refs}
    use = [r for r in refs if rates[r.segment_id] >= cfg["min_gyro_hz"]]
    print(f"セグメント {len(refs)} 本のうち、ジャイロ {cfg['min_gyro_hz']:.0f} Hz 以上の {len(use)} 本を使う")

    by_drive: dict[str, list] = {}
    for r in use:
        by_drive.setdefault(r.drive_id, []).append(r)
    segs: list[Seg] = []
    for d, rs in by_drive.items():
        rot = drive_rotation([r.path for r in rs])
        segs += [load_seg(r, rot, cfg) for r in rs]

    base = baselines(segs, cfg)
    base.to_csv(args.out / "baseline_by_speed.csv")

    rows: list[dict] = []
    keep: dict[tuple, Seg] = {}
    for s in segs:
        r = ratios(s, base, cfg)
        for e in find_events(s, r, cfg, wheelbase):
            rows.append(e)
            keep[(s.drive_id, s.index)] = s
    ev = pd.DataFrame(rows)
    ev = repeats(ev, cfg)
    ev["reason"] = ev.apply(reason, axis=1)
    ev["excluded"] = np.where(ev.spike_like, "飛び値", np.where(ev.maneuver, "操舵・旋回中", ""))
    ev = ev.sort_values("score", ascending=False)

    rh = ev[ev.group == "roll_heave"]
    top = rh[rh.excluded == ""].head(cfg["output"]["top_n"]).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top.to_csv(args.out / "candidates_top20.csv", index=False)
    rh.head(cfg["output"]["csv_pool"]).to_csv(args.out / "roll_heave_all.csv", index=False)
    ev[ev.group == "pitch_only"].sort_values("pitch_ratio", ascending=False).to_csv(
        args.out / "pitch_only.csv", index=False)

    clips = []
    if args.clips:
        matplotlib.use("Agg")
        configure_japanese_font()
        args.video_dir.mkdir(parents=True, exist_ok=True)
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        loaded = {(s.drive_id, s.index): s for s in segs}
        for _, e in top.head(args.clips).iterrows():
            s = keep[(e.drive_id, e.segment)]
            a, b = e.t_start_s - 5.0, e.t_end_s + 5.0
            parts = [x for x in (loaded.get((e.drive_id, e.segment - 1)) if a < 0 else None, s,
                                 loaded.get((e.drive_id, e.segment + 1)) if b > s.t[-1] else None) if x]
            j = join_segs(parts, s, cfg)
            a, b = max(a, float(j.t[0])), min(b, float(j.t[-1]))
            stem = f"rank{int(e['rank']):02d}_{drive_tag(e.drive_id)}_seg{e.segment}_{a:.1f}-{b:.1f}"
            info = clip_video(parts, s, a, b, args.video_dir / f"{stem}.mp4")
            s = j
            title = (f"#{int(e['rank'])} {e.drive_id} seg{e.segment}  {e.speed_kmh:.0f} km/h  "
                     f"上下 {e.heave_ratio:.1f}倍 / ロール {e.roll_ratio:.1f}倍 / ピッチ {e.pitch_ratio:.1f}倍  "
                     f"(帯 = 候補区間 {e.t_start_s:.2f}–{e.t_end_s:.2f} s)")
            plot_event(s, e, a, b, args.plot_dir / f"{stem}_vibration.png", title)
            clips.append(dict(rank=int(e["rank"]), stem=stem, clip_start_s=round(a, 2),
                              clip_end_s=round(b, 2), clamped=(a > e.t_start_s - 5.0) or (b < e.t_end_s + 5.0),
                              **info))
        pd.DataFrame(clips).to_csv(args.out / "clips.csv", index=False)

    meta = dict(
        config=str(args.config.relative_to(REPO)),
        config_sha1=hashlib.sha1(args.config.read_bytes()).hexdigest()[:10],
        segments_total=len(refs), segments_used=len(use), drives_used=len(by_drive),
        events_roll_heave=int((ev.group == "roll_heave").sum()),
        events_pitch_only=int((ev.group == "pitch_only").sum()),
        excluded=ev[ev.group == "roll_heave"].excluded.value_counts().to_dict(),
    )
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
