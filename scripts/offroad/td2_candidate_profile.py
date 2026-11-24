"""TartanDrive 2.0 の振動の候補 (td2_adaptive_detector.py の 124 件) を、信号で特徴づけて型に分け、
「大きな路面振動だけ」と「車両の挙動・運転操作に影響が出た可能性」をどこまで分けられるかを見る。

時刻: IMU の時刻 te (1 秒移動 MTVV の最大) を基準にする。racepak (ストローク・車輪の回転・スロットル・ブレーキ) は
  td2_sync_correct.py の 2 状態で補正し、さらに計測系の一定の遅れ (racepak_latency_s、r10 の当てはめの切片) を引く。
  舵角 (/ros_talon/steering_angle) はヘッダの無い型なので、KITTI の時刻の出どころが分からない。補正せずに使う (参考)。

指標 (どれも、同じ走行の無作為の走行中の時刻で同じ手順の値を取り、その上位 p 百分位を超えたら「旗」を立てる):
  入力: 主な軸 (上下 / ロール・左右 / ピッチ・前後)、続く長さ、波高率
  車両の応答: ヨー角速度の急な変化、ロール角の振れ、ストロークの伸び側・縮み側の端、後輪の回転の左右差の急変、速度の落ち
  運転操作: ブレーキとみられる信号の上がり、スロットルとみられる信号の戻し、舵角の速さ
「旗」は探索用の手がかりで、実際に危険があったことは言わない。

    uv run python scripts/offroad/td2_candidate_profile.py --detections out/offroad/tartandrive2/20261117_r11/detections.csv \
        --sync out/offroad/tartandrive2/20261117_r10 --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import iso2631 as ISO
import td2_sync_correct as SC
import td2_vibration_poc as P


def corrected_racepak(R: dict, St: pd.DataFrame, latency: float) -> dict:
    """racepak の話題を、補正した時刻 (ヘッダ − 状態 × 1 s − 一定の遅れ) で並べ直す。"""
    out = {}
    for key in ("shock", "controls"):
        if R[key] is None:
            continue
        t, a = R[key]
        tc, keep = SC.correct_times(t, St, latency)      # 状態の継ぎ目の重なりは捨てる
        out[key] = (tc[keep], a[keep])
    wr = R["run_dir"] / "wheel_rpm"
    if wr.exists():
        t, a = P.load_topic(R["run_dir"], "wheel_rpm")
        tc, keep = SC.correct_times(t, St, latency)
        out["wheel"] = (tc[keep], a[keep])
    return out


def seg(t, x, lo, hi):
    m = (t >= lo) & (t <= hi)
    return x[m]


def features(R: dict, S: dict, C: dict, te: float, cfg: dict) -> dict:
    pc = cfg["profile"]
    g, fs = R["g"], R["fs"]
    i = int(np.clip(np.searchsorted(g, te), 0, len(g) - 1))
    f = {}
    # ---- 入力 (IMU)
    comp = {k: ISO.K_SEAT[k] * S["rr"][k][i] for k in S["rr"]}
    tot = S["tot"][i]
    f["share_vertical"] = comp["z"] ** 2 / tot ** 2
    f["share_roll_lat"] = (comp["y"] ** 2 + comp["rx"] ** 2) / tot ** 2
    f["share_pitch_long"] = (comp["x"] ** 2 + comp["ry"] ** 2) / tot ** 2
    f["input_axis"] = max((("上下", f["share_vertical"]), ("ロール・左右", f["share_roll_lat"]), ("ピッチ・前後", f["share_pitch_long"])), key=lambda x: x[1])[0]
    above = S["tot"] >= 0.5 * tot
    j0, j1 = i, i
    while j0 > 0 and above[j0 - 1]:
        j0 -= 1
    while j1 < len(g) - 1 and above[j1 + 1]:
        j1 += 1
    f["duration_s"] = (j1 - j0) / fs
    w = (g >= te - 2.5) & (g <= te + 2.5)
    awz = S["aw"]["z"][w]
    f["crest_z"] = float(np.max(np.abs(awz)) / max(np.sqrt(np.mean(awz ** 2)), 1e-9))
    imp = (g >= te - 1.0) & (g <= te + 0.3)          # 衝撃は 1 秒移動 RMS の最大より前にある
    f["az_peak"] = float(np.max(np.abs(S["raw"]["z"][imp])))
    f["roll_rate_max_dps"] = float(np.degrees(np.max(np.abs(R["X"]["wx"][imp]))))
    f["pitch_rate_max_dps"] = float(np.degrees(np.max(np.abs(R["X"]["wy"][imp]))))
    # ---- 車両の応答
    ctx = (g >= te - pc["context_s"]) & (g <= te + pc["context_s"])
    resp = (g >= te - 1.0) & (g <= te + pc["response_s"])
    f["roll_dev_max_deg"] = float(np.degrees(np.max(np.abs(R["roll_g"][resp] - np.median(R["roll_g"][ctx])))))
    f["pitch_dev_max_deg"] = float(np.degrees(np.max(np.abs(R["pitch_g"][resp] - np.median(R["pitch_g"][ctx])))))
    f["yaw_dist_dps"] = float(np.degrees(np.max(np.abs(R["wz_hp"][resp]))))
    pre = (g >= te - 2.0) & (g <= te - 1.0)
    f["speed_pre"] = float(np.mean(R["speed"][pre])) if pre.any() else np.nan
    f["speed_drop"] = f["speed_pre"] - float(np.min(R["speed"][resp]))
    if "shock" in C:
        ts, a = C["shock"]
        cols = cfg["shock"]["use_cols"]
        lo, hi = te - 1.0, te + pc["response_s"]
        for side, j in cols.items():
            v = seg(ts, a[:, j], lo, hi)
            f[f"stroke_{side}_min"] = float(v.min()) if len(v) else np.nan
            f[f"stroke_{side}_max"] = float(v.max()) if len(v) else np.nan
        dl, dr = (f["stroke_RL_max"] - f["stroke_RL_min"]), (f["stroke_RR_max"] - f["stroke_RR_min"])
        f["stroke_asym"] = abs(dl - dr) / max(dl + dr, 1e-6)          # 0 = 左右同じ、1 = 片側だけ
        f["stroke_max"] = max(f["stroke_RL_max"], f["stroke_RR_max"])       # 縮み側 (値が大きいほど縮み、r3 の checks)
        f["neg_stroke_min"] = -min(f["stroke_RL_min"], f["stroke_RR_min"])  # 伸び側 (符号を変えて大きいほど伸び)
    if "wheel" in C:
        tw, wv = C["wheel"]
        # 1 サンプルだけの飛び値 (A047 で見つけた) で左右差が跳ねないよう、中央値フィルタをかける
        from scipy.signal import medfilt
        k = pc["wheel_median_n"]
        rl, rr = medfilt(wv[:, 2].astype(float), k), medfilt(wv[:, 3].astype(float), k)
        ok = (rl > pc["wheel_min_rpm"]) & (rr > pc["wheel_min_rpm"])
        ratio = np.where(ok, np.log(np.maximum(rl, 1) / np.maximum(rr, 1)), np.nan)
        base = seg(tw, ratio, te - pc["context_s"], te + pc["context_s"])
        cur = seg(tw, ratio, te - 1.0, te + pc["response_s"])
        f["wheel_lr_jump"] = float(np.nanmax(np.abs(cur - np.nanmedian(base)))) if np.isfinite(cur).any() and np.isfinite(base).any() else np.nan
    # ---- 運転操作 (補正した racepak の時刻)
    if "controls" in C:
        tc, c = C["controls"]
        brk, thr = P.pedals(c, cfg)                      # 列の割り当ては設定 (reaction.controls_cols)
        pre_lo, pre_hi = te - 3.0, te - 1.0
        post_lo, post_hi = te - 1.0, te + pc["response_s"]
        f["throttle_drop"] = float(np.nanmean(seg(tc, thr, pre_lo, pre_hi)) - np.nanmin(seg(tc, thr, post_lo, post_hi))) if np.isfinite(seg(tc, thr, post_lo, post_hi)).any() else np.nan
        b_pre = np.nanmean(seg(tc, brk, pre_lo, pre_hi)) if len(seg(tc, brk, pre_lo, pre_hi)) else np.nan
        b_post = seg(tc, brk, post_lo, post_hi)
        f["brake_rise"] = float(np.nanmax(b_post) - b_pre) if len(b_post) and np.isfinite(b_post).any() else np.nan
        # ブレーキの立ち上がりの時刻 (前の平均 + しきい値を最初に超えた所、te からの秒)
        bb = seg(tc, brk, te - 3.0, post_hi)
        tb = seg(tc, tc, te - 3.0, post_hi)
        k = np.flatnonzero(bb - b_pre >= pc["brake_onset_rise"]) if np.isfinite(b_pre) else []
        f["brake_onset_s"] = float(tb[k[0]] - te) if len(k) else np.nan
    if R["steer"] is not None:
        tsr, s = R["steer"]
        rate = np.r_[0.0, np.diff(s[:, 0]) / np.maximum(np.diff(tsr), 1e-3)]
        r_ = seg(tsr, np.abs(rate), te - 1.0, te + pc["response_s"])
        f["steer_rate_max"] = float(r_.max()) if len(r_) else np.nan
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--sync", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    pc = cfg["profile"]
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    D = pd.read_csv(a.detections)
    D = D[D.kind == "adaptive"].reset_index(drop=True)
    D.insert(0, "cand_id", [f"A{i:03d}" for i in range(len(D))])
    ST = pd.read_csv(a.sync / "sync_states.csv")
    rows, nrows = [], []
    for run, grp in D.groupby("run"):
        R = P.load_run(Path(cfg["data_dir"]) / run, cfg)
        R["run_dir"] = Path(cfg["data_dir"]) / run
        S = P.iso_series(R, cfg["lp8_hz"])
        Sh = P.shock_series(R, cfg)
        P.run_checks(R, Sh, np.nan)                       # roll_g, pitch_g を作る
        R["wz_hp"] = P.bp(R["X"]["wz"], *pc["yaw_band_hz"], R["fs"])
        C = corrected_racepak(R, ST[ST.run == run], pc["racepak_latency_s"])
        for _, d in grp.iterrows():
            f = features(R, S, C, d.t_abs, cfg)
            f["near_run_edge"] = bool(d.t_abs - R["g"][0] < pc["edge_s"] or R["g"][-1] - d.t_abs < pc["edge_s"])
            rows.append(dict(d.to_dict(), **f))
        # 無作為の時刻 (同じ走行の走行中、候補から 5 s 以上離れた所)
        mv = np.flatnonzero((R["speed"] >= cfg["moving_min_speed_mps"]) & ~R["gap"])
        mv = mv[(R["g"][mv] - R["g"][0] > pc["edge_s"]) & (R["g"][-1] - R["g"][mv] > pc["edge_s"])]
        k, tries = 0, 0
        while k < pc["null_per_candidate"] * len(grp) and len(mv) and tries < 50 * pc["null_per_candidate"] * len(grp):
            tries += 1                                     # 短い走行で候補が詰まっていると、離れた時刻が無いことがある
            i = mv[rng.integers(len(mv))]
            te = R["g"][i]
            if np.min(np.abs(grp.t_abs.to_numpy() - te)) < 5:
                continue
            nrows.append(dict(run=run, t_abs=te, speed=float(R["speed"][i]), **features(R, S, C, te, cfg)))
            k += 1
        print(run[:40], len(grp), flush=True)
    F = pd.DataFrame(rows)
    N = pd.DataFrame(nrows)

    # ---- 旗 (無作為の時刻の上位 p 百分位)
    flag_def = pc["flags"]
    thr = {}
    for name, d in flag_def.items():
        col = d["col"]
        thr[name] = float(np.nanpercentile(N[col], d["pct"]))
        F[f"flag_{name}"] = F[col] >= thr[name]
        N[f"flag_{name}"] = N[col] >= thr[name]
    veh = [n for n, d in flag_def.items() if d["group"] == "vehicle"]
    drv = [n for n, d in flag_def.items() if d["group"] == "driver"]
    for df in (F, N):
        df["n_vehicle_flags"] = df[[f"flag_{n}" for n in veh]].sum(axis=1)
        df["n_driver_flags"] = df[[f"flag_{n}" for n in drv]].sum(axis=1)
    F["temporal"] = np.where((F.duration_s <= pc["single_max_duration_s"]) & (F.crest_z >= pc["single_min_crest"]), "単発", "続く揺れ")
    F["shock_support"] = np.where(F.local_shock_pct >= 90, "あり", np.where(F.local_shock_pct >= 50, "弱い", "なし"))

    def tier(r):
        if r.n_vehicle_flags >= 2 or (r.n_vehicle_flags >= 1 and r.n_driver_flags >= 1):
            return "C: 挙動の変化 + (操作 or 別の挙動)"
        if r.n_vehicle_flags >= 1 or r.n_driver_flags >= 1:
            return "B: 挙動か操作の変化が 1 つ"
        return "A: 振動だけ"
    F["tier"] = F.apply(tier, axis=1)
    N["tier"] = N.apply(tier, axis=1)
    F["reasons"] = F.apply(lambda r: ";".join(
        [f"前後±30sの比 {r.contrast:.1f}", f"入力 {r.input_axis}・{r.temporal}", f"ストローク {r.shock_support}"]
        + [flag_def[n]["label"] for n in flag_def if r[f"flag_{n}"]]), axis=1)

    # ---- まとめ
    rate = pd.DataFrame([dict(flag=n, label=flag_def[n]["label"], group=flag_def[n]["group"], threshold=thr[n],
                              candidates=float(F[f"flag_{n}"].mean()), random=float(N[f"flag_{n}"].mean()),
                              ratio=float(F[f"flag_{n}"].mean() / max(N[f"flag_{n}"].mean(), 1e-9))) for n in flag_def])
    sp = lambda x, y: float(F[[x, y]].corr("spearman").iloc[0, 1])
    summ = dict(
        candidates=len(F), random=len(N), near_run_edge=int(F.near_run_edge.sum()),
        input_axis=F.input_axis.value_counts().to_dict(), temporal=F.temporal.value_counts().to_dict(),
        shock_support=F.shock_support.value_counts().to_dict(),
        tier_candidates=F.tier.value_counts().to_dict(), tier_random=N.tier.value_counts(normalize=True).round(3).to_dict(),
        flag_rates=rate.round(3).to_dict(orient="records"),
        crosstab_axis_tier=pd.crosstab(F.input_axis, F.tier).to_dict(),
        crosstab_temporal_tier=pd.crosstab(F.temporal, F.tier).to_dict(),
        spearman_mtvv_vs_n_vehicle=sp("mtvv_total", "n_vehicle_flags"), spearman_contrast_vs_n_vehicle=sp("contrast", "n_vehicle_flags"),
        spearman_mtvv_vs_n_driver=sp("mtvv_total", "n_driver_flags"), spearman_speed_vs_n_vehicle=sp("speed", "n_vehicle_flags"),
        brake_onset_s=dict(n=int(F.brake_onset_s[F.flag_brake].notna().sum()),
                           before_impact=int((F.brake_onset_s[F.flag_brake] < -1.0).sum()),
                           around_or_after=int((F.brake_onset_s[F.flag_brake] >= -1.0).sum()),
                           values=[round(x, 2) for x in F.brake_onset_s[F.flag_brake].dropna().sort_values()]),
        thresholds=thr,
    )
    F.round(4).to_csv(out / "candidates.csv", index=False)
    N.round(4).to_csv(out / "random_times.csv", index=False)
    rate.round(4).to_csv(out / "flag_rates.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    (out / "meta.json").write_text(json.dumps(dict(profile=pc, detections=str(a.detections), sync=str(a.sync), run_id=a.run_id), ensure_ascii=False, indent=2))
    # 人が付ける欄を空けた確認表
    rv = F[["cand_id", "run", "t", "speed", "mtvv_total", "contrast", "input_axis", "temporal", "shock_support", "tier", "reasons"]].copy()
    for c in ("human_cause_visible", "human_cause", "human_abnormal", "human_safety_relevant", "human_note"):
        rv[c] = ""
    rv.round(3).to_csv(out / "review_sheet.csv", index=False)

    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(rate))
    ax[0].bar(x - 0.2, rate.candidates, 0.4, color="#1f4e9c", label=f"候補 (n={len(F)})")
    ax[0].bar(x + 0.2, rate.random, 0.4, color="#9aa5b1", label=f"無作為の時刻 (n={len(N)})")
    ax[0].set_xticks(x, rate.label, rotation=40, ha="right", fontsize=8)
    ax[0].set_ylabel("旗が立った割合")
    ax[0].legend(fontsize=8)
    ax[0].set_title("挙動 (左) と操作 (右) の手がかり", fontsize=10)
    tiers = sorted(F.tier.unique())
    ct = pd.crosstab(F.input_axis, F.tier).reindex(columns=tiers, fill_value=0)
    bottom = np.zeros(len(ct))
    colors = ["#9aa5b1", "#5b8bd6", "#c2410c"]
    for c, col in zip(tiers, colors):
        ax[1].bar(ct.index, ct[c], bottom=bottom, color=col, label=c)
        bottom += ct[c].to_numpy()
    ax[1].set_ylabel("候補の数")
    ax[1].legend(fontsize=8)
    ax[1].set_title("入力の型と、挙動・操作の変化の段", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "profile_overview.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
