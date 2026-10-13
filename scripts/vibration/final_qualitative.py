"""最終確認: 代表イベントで「左右対称・右側・左側」の 3 仮説を定性的に比べる。

モデルは既存の 7 自由度 Full-car (fullcar.py) と、単発の 1−cos 凹凸 (compare_fullcar.fit_single) をそのまま使う。
当てはめるのは上下加速度の振幅と時刻だけ。ロール・ピッチは予測として、符号・時刻・波形の特徴を比べる。
新しいモデル・新しいパラメータ調整は行わない。路面の高さは評価に使わない (参考値として出すだけ)。

映像の図は、平坦な路面と直進を仮定してカメラから車輪の通り道を投影したもの。
カメラの横位置は車両中心と仮定した (記録は無い)。路面の高さは映像から推定しない。

    uv run python scripts/vibration/final_qualitative.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml

import compare_fullcar as CF
import find_roll_bumps as F
import fullcar as FC
from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent

# (ドライブ, セグメント, 振動開始, 振動終了, 説明, 地点, 映像で見るフレームの何秒前)
EVENTS = [
    ("2018-08-15--09-01-03", 21, 29.15, 29.57, "高速道路 117 km/h・舗装の境目が右寄り", 0, 0.30),
    ("2018-08-17--14-55-39", 9, 54.04, 54.59, "地点19 縁石側の車線・右寄りに補修跡", 19, 0.30),
    ("2018-08-17--12-07-08", 41, 34.70, 35.67, "地点19 内側の車線 (対照)", 19, 0.40),
]
HYP = {"左右対称": "単発・両", "右側入力": "単発・右", "左側入力": "単発・左"}
COLORS = {"左右対称": "0.45", "右側入力": "tab:red", "左側入力": "tab:blue"}
FIRST_WIN_S = (-0.15, 0.6)     # 最初の極値を探す範囲 (当てはめた当たり時刻から)
FIRST_FRAC = 0.5               # 最大振幅のこの割合を最初に超えた極値を「最初の向き」とする
ENV_FRAC = 0.3                 # 8〜15 Hz の包絡線の継続時間のしきい値
IMPACT_WIN_S = (-0.15, 0.6)    # 衝撃の前後で波形を比べる範囲
MAX_LAG_S = 0.15               # 時刻のずれを探す範囲
PRE_WIN_S = (-0.4, -0.03)      # 前ぶれ (当たる前の揺れ) を見る範囲


def first_sign(x: np.ndarray, t: np.ndarray, t_hit: float) -> tuple[int, float]:
    """当たり時刻の近くで、最大振幅の半分を最初に超えた点の符号と時刻。"""
    w = (t >= t_hit + FIRST_WIN_S[0]) & (t <= t_hit + FIRST_WIN_S[1])
    xs, ts = x[w], t[w]
    if np.max(np.abs(xs)) < 1e-6:
        return 0, np.nan
    j = int(np.argmax(np.abs(xs) >= FIRST_FRAC * np.max(np.abs(xs))))
    return int(np.sign(xs[j])), float(ts[j] - t_hit)


def corr0(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def impact_corr(meas: np.ndarray, pred: np.ndarray, t: np.ndarray, t_hit: float, fs: float) -> tuple[float, float, float]:
    """衝撃の前後 (IMPACT_WIN_S) での相関。時間ずれ 0 の値と、±MAX_LAG_S で最大になる値・ずれ [ms]。"""
    w = np.flatnonzero((t >= t_hit + IMPACT_WIN_S[0]) & (t <= t_hit + IMPACT_WIN_S[1]))
    if np.std(pred[w]) < 1e-9:
        return np.nan, np.nan, np.nan
    L = int(MAX_LAG_S * fs)
    cc = {l: np.corrcoef(meas[w + l], pred[w])[0, 1] for l in range(-L, L + 1)
          if w[0] + l >= 0 and w[-1] + l < len(meas)}
    best = max(cc, key=cc.get)
    return float(cc[0]), float(cc[best]), best / fs * 1000


def precursor_ratio(x: np.ndarray, t: np.ndarray, t_hit: float) -> float:
    """当たり時刻より前 (PRE_WIN_S) の最大振幅が、全体の最大振幅の何割か。"""
    pre = (t >= t_hit + PRE_WIN_S[0]) & (t <= t_hit + PRE_WIN_S[1])
    return float(np.max(np.abs(x[pre])) / np.max(np.abs(x)))


def env_duration(x: np.ndarray, fs: float) -> float:
    a, b = CF.envelope_span(x, fs, ENV_FRAC)
    return b - a


def wheel_evidence(seg, E, fs: float, wheelbase: float) -> dict:
    """CAN 輪速の 2〜15 Hz の揺れ: 左右の大きさの比と、前輪 → 後輪の時間差。"""
    w = (seg.t >= E.ev[2] - 0.3) & (seg.t <= E.ev[3] + wheelbase / E.v + 0.3)
    r = {k: float(np.sqrt(np.mean(seg.bp[k][w] ** 2))) for k in ("FL", "FR", "RL", "RR")}
    side = "R" if r["FR"] + r["RR"] > r["FL"] + r["RL"] else "L"
    env = {k: np.convolve(np.abs(seg.bp[k][w]), np.ones(10) / 10, "same") for k in ("F" + side, "R" + side)}
    lag = (np.argmax(env["R" + side]) - np.argmax(env["F" + side])) / fs
    return dict(wheel_rms_right_over_left=round((r["FR"] + r["RR"]) / (r["FL"] + r["RL"]), 2),
                wheel_side="右" if side == "R" else "左",
                wheel_front_rear_lag_s=round(lag, 3), wheel_lag_expected_s=round(wheelbase / E.v, 3))


def evaluate(E, seg, p) -> tuple[list[dict], dict]:
    fs = E.fs
    rows, preds = [], {}
    meas = E.meas
    for name, key in HYP.items():
        d = CF.fit_single(E, p, CF.SINGLE[key], "unilateral")
        preds[name] = d
        pr = d["pred"]
        t_hit = d["t_hit_s"]
        row = dict(drive=E.ev[0], segment=E.ev[1], event=E.ev[4], speed_kmh=round(E.v * 3.6),
                   hypothesis=name, t_hit_s=t_hit, height_cm_ref=d["height_cm"], lift_off_ms=d["lift_off_ms"],
                   heave_r2=round(CF.r2(meas["heave"], pr["heave"]), 2))
        for k in ("roll", "pitch"):
            sm, tm = first_sign(meas[k], E.t, t_hit)
            sp, tp = first_sign(pr[k], E.t, t_hit)
            row[f"{k}_first_sign_meas"] = sm
            row[f"{k}_first_sign_pred"] = sp
            row[f"{k}_first_time_meas_s"] = round(tm, 3)
            row[f"{k}_first_time_pred_s"] = round(tp, 3) if np.isfinite(tp) else np.nan
            row[f"{k}_corr0"] = round(corr0(meas[k], pr[k]), 2)
            c0, cb, lag = impact_corr(meas[k], pr[k], E.t, t_hit, fs)
            row[f"{k}_impact_corr0"] = round(c0, 2)
            row[f"{k}_impact_corr_best"] = round(cb, 2)
            row[f"{k}_impact_lag_ms"] = round(lag) if np.isfinite(lag) else np.nan
            row[f"{k}_precursor_meas"] = round(precursor_ratio(meas[k], E.t, t_hit), 2)
            lo = CF.bp(meas[k], 0.5, 3.0, fs)
            row[f"{k}_corr0_0.5-3"] = round(corr0(lo, CF.bp(pr[k], 0.5, 3.0, fs)), 2)
            row[f"{k}_rms_meas_over_pred"] = (round(CF.rms(meas[k]) / CF.rms(pr[k]), 2)
                                             if CF.rms(pr[k]) > 1e-6 else np.inf)
        row["roll_over_heave_meas"] = round(CF.rms(meas["roll"]) / CF.rms(meas["heave"]), 2)
        row["roll_over_heave_pred"] = round(CF.rms(pr["roll"]) / CF.rms(pr["heave"]), 2)
        row["heave_8-15_dur_meas_s"] = round(env_duration(meas["heave"], fs), 2)
        row["heave_8-15_dur_pred_s"] = round(env_duration(pr["heave"], fs), 2)
        rows.append(row)
    return rows, preds


def plot_waveforms(items, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, len(items), figsize=(5.2 * len(items), 8.2), sharex="col")
    labels = ("上下加速度 [m/s²]\n(振幅と時刻を当てはめ)", "ロール角速度 [deg/s]\n(予測、正 = 右下がり)",
              "ピッチ角速度 [deg/s]\n(予測、正 = 機首上げ)")
    for c, (E, preds) in enumerate(items):
        t_hit = preds["右側入力"]["t_hit_s"]
        t = E.t - t_hit
        for r, k in enumerate(("heave", "roll", "pitch")):
            ax = axs[r, c]
            ax.plot(t, E.meas[k], color="k", lw=1.6, label="実測 (IMU)")
            for name, d in preds.items():
                ax.plot(t, d["pred"][k], color=COLORS[name], lw=1.0, alpha=0.85, label=name)
            ax.axvline(0, color="orange", lw=0.8, ls="--")
            ax.set_xlim(-0.5, 1.5)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=9)
        axs[0, c].set_title(f"{E.ev[0][5:10]} seg{E.ev[1]}  {E.v * 3.6:.0f} km/h\n{E.ev[4]}", fontsize=9)
        axs[2, c].set_xlabel("前輪が凹凸に当たった時刻 (当てはめ値) からの時間 [s]", fontsize=8)
    axs[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("3 仮説の比較 (単発の突起・長さ 0.5 m。高さは上下加速度で当てはめた参考値で、評価には使わない)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_matrix(df: pd.DataFrame, out: Path) -> None:
    """定性的な整合性を、値と色 (緑 合う / 黄 弱い / 赤 合わない) の表にする。色分けの境は表示のための目安。"""
    import matplotlib.pyplot as plt

    G, Y, R, W = "#cfe8cf", "#f6e7b8", "#f3c6c6", "white"
    dirs = {1: "右下", -1: "左下", 0: "0"}

    def ratio(x):
        if not np.isfinite(x) or x < 1e-3:
            return "0", R
        return f"{x:.1f}", G if 0.5 <= x <= 2 else Y if 0.25 <= x <= 4 else R

    def corr(x, lag=None):
        if not np.isfinite(x):
            return "— (予測 0)", W
        s = f"{x:+.2f}" + (f" ({lag:+.0f} ms)" if lag is not None and np.isfinite(lag) else "")
        return s, G if x >= 0.3 else Y if x >= 0.1 else R

    checks = [
        ("ロール/上下\n予測÷実測", lambda r: ratio(r.roll_over_heave_pred / r.roll_over_heave_meas)),
        ("ロールの相関\n(衝撃前後)", lambda r: corr(r.roll_impact_corr0)),
        ("ロールの最初の向き\n実測 / 予測", lambda r: (f"{dirs[r.roll_first_sign_meas]} / {dirs[r.roll_first_sign_pred]}",
                                             G if r.roll_first_sign_meas == r.roll_first_sign_pred else R)),
        ("ピッチの相関\n(衝撃前後)", lambda r: corr(r.pitch_impact_corr0)),
        ("8〜15 Hz 継続\n予測÷実測", lambda r: ratio(r["heave_8-15_dur_pred_s"] / r["heave_8-15_dur_meas_s"])),
    ]
    rows = [f"{r.drive[5:10]} seg{r.segment}  {r.hypothesis}" for r in df.itertuples()]
    cells = [[f(r) for _, f in checks] for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.3))
    ax.axis("off")
    tb = ax.table(cellText=[[c[0] for c in row] for row in cells], rowLabels=rows,
                  colLabels=[c for c, _ in checks], cellLoc="center", loc="center",
                  cellColours=[[c[1] for c in row] for row in cells])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.7)
    ax.set_title("定性的な整合性 (緑 合う / 黄 弱い / 赤 合わない。比は 0.5〜2 を緑、相関は 0.3 以上を緑)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_model(p, out: Path) -> None:
    """7 自由度モデルの模式図 (正面・側面) と、3 仮説で期待される IMU の符号。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])

    def spring(ax, x, y0, y1, n=6, w=0.06, **kw):
        ys = np.linspace(y0, y1, 2 * n + 1)
        xs = x + np.r_[0, np.tile([w, -w], n)][: len(ys)]
        xs[-1] = x
        ax.plot(xs, ys, **kw)

    for k, (title, half, labels, mom) in enumerate((
            ("正面 (後ろから見る)", float(p.track[0]) / 2, ("左輪", "右輪"), "ロール φ\n(正 = 右下がり)"),
            ("側面 (右から見る)", p.wheelbase / 2, ("後輪", "前輪"), "ピッチ θ\n(正 = 機首上げ)"))):
        ax = fig.add_subplot(gs[k])
        ax.add_patch(FancyBboxPatch((-half - 0.25, 1.15), 2 * half + 0.5, 0.45, boxstyle="round,pad=0.02",
                                    fc="#dbe6f4", ec="k"))
        ax.text(-0.1, 1.38, "ばね上 (車体)\nm_s, I_xx, I_yy", ha="center", va="center", fontsize=8)
        ax.plot(0, 1.38 - 0.0, "k+", ms=0)
        for sx, lab in zip((-half, half), labels):
            spring(ax, sx - 0.1, 0.62, 1.15, color="k", lw=1)
            ax.plot([sx + 0.1] * 2, [0.62, 0.85], "k", lw=1)
            ax.plot([sx + 0.05, sx + 0.15], [0.85, 0.85], "k", lw=1)
            ax.plot([sx + 0.03, sx + 0.17, sx + 0.17, sx + 0.03], [0.9, 0.9, 0.8, 0.8], "k", lw=1)
            ax.plot([sx + 0.1] * 2, [0.9, 1.15], "k", lw=1)
            ax.add_patch(FancyBboxPatch((sx - 0.2, 0.45), 0.4, 0.17, boxstyle="round,pad=0.01", fc="#eee", ec="k"))
            ax.text(sx, 0.535, "m_u", ha="center", va="center", fontsize=7)
            spring(ax, sx, 0.1, 0.45, n=4, color="0.3", lw=1)
            ax.text(sx + 0.12, 0.27, "タイヤ k_t\n(浮きを扱う)", fontsize=6.5, va="center")
            ax.annotate("", (sx, 0.1), (sx, -0.08), arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5))
            ax.text(sx, -0.15, f"{lab}\n路面 z_r", ha="center", va="top", fontsize=7.5, color="tab:red")
            ax.text(sx - 0.17, 0.9, "k, c", fontsize=6.5, ha="right")
        if k == 0:
            ax.plot([-half, half], [1.08, 1.08], color="tab:green", lw=1.5)
            ax.text(0, 1.03, "スタビライザー (左右差にだけ効く)", ha="center", va="top", fontsize=7, color="tab:green")
        # ロールは右が下がる向き、ピッチ (右が前) は右が上がる向きを正として矢印を描く
        a, b = ((-0.35, 0.35) if k == 0 else (0.35, -0.35))
        ax.annotate("", (b, 1.72), (a, 1.72), arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={-0.4 if k == 0 else 0.4}"))
        ax.text(0, 1.86, mom, ha="center", fontsize=8)
        ax.plot(half - 0.05, 1.5, "s", color="tab:orange", ms=6)
        ax.text(half - 0.05, 1.66, "IMU", fontsize=7, color="tab:orange", ha="center")
        ax.set_xlim(-half - 0.6, half + 0.6)
        ax.set_ylim(-0.45, 2.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9)

    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    cells = [["0", "出る", "左右の輪速が\n同程度"],
             ["突起: 左下がり\n凹み: 右下がり", "出る", "右側の輪速が\n大きく揺れる"],
             ["突起: 右下がり\n凹み: 左下がり", "出る", "左側の輪速が\n大きく揺れる"]]
    tb = ax.table(cellText=cells, rowLabels=list(HYP), colLabels=["ロールの最初の向き", "ピッチ・上下", "CAN 輪速 (独立)"],
                  cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 3.0)
    ax.set_title("3 仮説で期待される符号 (前輪が先に当たり、L/v 後に後輪)", fontsize=9)
    fig.suptitle("線形 7 自由度 Full-car モデル (上下・ピッチ・ロール + 4 輪のばね下)。車両パラメータは固定、路面だけを変える", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def frame_at(seg_dir: Path, t0: float, t_rel: float) -> np.ndarray:
    ft = np.load(seg_dir / "global_pose/frame_times") - t0
    i = int(np.argmin(np.abs(ft - t_rel)))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-r", "20", "-i", str(seg_dir / "video.hevc"),
                          "-vf", f"select=eq(n\\,{i})", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(874, 1164, 3), float(ft[i])


def plot_tracks(items, cam: dict, track: float, out: Path) -> None:
    """振動開始の少し前のフレームに、左右の車輪の通り道と、前輪が振動開始時に通る位置を重ねる。"""
    import matplotlib.pyplot as plt

    f, yh, K = cam["focal_px"], cam["horizon_row"], cam["K"]
    behind = cam["camera_behind_front_axle_m"]
    fig, axs = plt.subplots(len(items), 1, figsize=(9, 3.9 * len(items)))
    for ax, (E, seg_dir, t0, frame_dt, wheel) in zip(np.atleast_1d(axs), items):
        img, t_frame = frame_at(seg_dir, t0, E.ev[2] - frame_dt)
        ax.imshow(img)
        d_on = E.v * (E.ev[2] - t_frame) + behind
        d = np.linspace(3.5, 40, 200)
        for y_lat, c, lab in ((-track / 2, "tab:blue", "左輪の通り道"), (track / 2, "tab:red", "右輪の通り道")):
            ax.plot(582 + f * y_lat / d, yh + K / d, color=c, lw=1.4, ls="--", label=lab)
            ax.plot(582 + f * y_lat / d_on, yh + K / d_on, "o", ms=11, mfc="none", mec=c, mew=2.2)
        ax.axhline(yh + K / d_on, color="yellow", lw=0.6, alpha=0.7)
        ax.set_xlim(250, 1100)
        ax.set_ylim(700, 430)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{E.ev[0][5:10]} seg{E.ev[1]}  振動開始の {E.ev[2] - t_frame:.2f} s 前  "
                     f"(○ = 前輪が振動開始時に通る位置、約 {d_on:.1f} m 先)\n"
                     f"輪速の揺れ 右/左 = {wheel['wheel_rms_right_over_left']}、"
                     f"{wheel['wheel_side']}前輪→後輪 {wheel['wheel_front_rear_lag_s']:.2f} s "
                     f"(L/v = {wheel['wheel_lag_expected_s']:.2f} s)", fontsize=9)
    np.atleast_1d(axs)[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("映像上の車輪の通り道 (平坦・直進・カメラは車両中心と仮定。位置の誤差は横 ±0.3 m 程度を見込む)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=105)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO / "raw_data/Chunk_1")
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_fullcar.yaml")
    ap.add_argument("--vib-config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--pavement-config", type=Path, default=REPO / "configs/vibration_pavement.yaml")
    ap.add_argument("--camera-fit", type=Path, default=REPO / "out/vibration/pavement/site19_camera_fit.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/final")
    args = ap.parse_args()

    matplotlib.use("Agg")
    configure_japanese_font()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg_vib = yaml.safe_load(args.vib_config.read_text())
    p = FC.build_params(FC.load_config(args.config))
    cam = dict(yaml.safe_load(args.camera_fit.read_text()))
    cam["focal_px"] = yaml.safe_load(args.pavement_config.read_text())["camera"]["focal_px"]
    refs = find_segments(args.data_root)

    rows, wheel_rows, wave_items, track_items = [], [], [], []
    for ev in EVENTS:
        drive_refs = [r for r in refs if r.drive_id.endswith(ev[0])]
        rot = F.drive_rotation([r.path for r in drive_refs])
        ref = next(r for r in drive_refs if r.index == ev[1])
        seg = F.load_seg(ref, rot, cfg_vib)
        E = CF.Event(seg, ev[:6], cfg_vib)
        r, preds = evaluate(E, seg, p)
        wh = wheel_evidence(seg, E, E.fs, p.wheelbase)
        rows += r
        wheel_rows.append(dict(drive=ev[0], segment=ev[1], **wh))
        wave_items.append((E, preds))
        track_items.append((E, ref.path, seg.t0, ev[6], wh))

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "three_hypotheses.csv", index=False)
    pd.DataFrame(wheel_rows).to_csv(args.out / "wheel_evidence.csv", index=False)
    plot_model(p, args.out / "model_schematic.png")
    plot_waveforms(wave_items, args.out / "three_hypotheses_waveforms.png")
    plot_matrix(df, args.out / "three_hypotheses_matrix.png")
    plot_tracks(track_items, cam, float(p.track[0]), args.out / "wheel_tracks_video.png")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df.drop(columns=["drive", "event"]).to_string(index=False))
    print(pd.DataFrame(wheel_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
