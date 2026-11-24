"""実走行 CAN から制動性能 (利き) をどこまで正確に評価できるかの PoC (段階 1)。

    uv run python scripts/brake/brake_performance_poc.py --car rav4 --run-id <新しい ID>
    uv run python scripts/brake/brake_performance_poc.py --car civic --run-id <新しい ID>
    uv run python scripts/brake/brake_performance_poc.py --car civic --level level_alt --run-id <新しい ID>

出力 (out/brake_performance/<run-id>/):
  summary.json        集計 (相関と R² の計算の区別、勾配の確かめ、模型の比較、利きのばらつき、熱負荷)
  events.csv          1 回の制動ごとの値 (定常部の値・減速度・予測・eta・直前の熱の指標・分割の番号)
  model_cv.csv        模型 × 勾配の扱い × 遅れの、学習に使っていない走行での誤差
  grade_checks.csv    勾配の確かめ (global_pose・ASLP・加速度センサ由来)
  grade_repeat.csv    同じ場所・同じ向きを通った走行どうしの global_pose の勾配の差
  heat_by_drive.csv   走行ごとの熱の指標の最大
  *.png               図
  meta.json           設定の写し・入力
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import brake_data as BD
import brake_model as BM
from near_miss.plotting import configure_japanese_font

OUT = Path("out/brake_performance")


def robust_cv(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    return float((np.percentile(x, 75) - np.percentile(x, 25)) / 1.349 / np.median(x))


def main() -> int:
    np.seterr(divide="ignore", over="ignore", invalid="ignore")   # macOS の BLAS が NaN を含む行列積で出す警告を止める
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/brake/brake_performance.yaml")
    ap.add_argument("--car", required=True, choices=["rav4", "civic"])
    ap.add_argument("--level", default="level", help="制動の値の列 (civic の USER_BRAKE は level_alt)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=None, help="試しに先頭のセグメントだけ")
    ap.add_argument("--cache", type=Path, default=None, help="読み込んだ表の置き場 (再実行を速くする。out/ の外)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    car_cfg = cfg["cars"][a.car]
    dt, g = cfg["grid_dt_s"], cfg["g"]

    if a.cache and a.cache.exists():
        D = pd.read_pickle(a.cache)
    else:
        D = BD.load_car(a.car, cfg, a.limit)
        if a.cache:
            D.to_pickle(a.cache)
    D["block"] = (D.groupby("drive").t.diff().fillna(1e9) > cfg["block_gap_s"]).cumsum()   # キャッシュが古い区切りでも作り直す
    print(f"loaded {len(D)} rows, {D.drive.nunique()} drives, {time.time() - t0:.0f} s", flush=True)

    lev = a.level + "_s"
    pressed = D.pressed.to_numpy() > 0.5
    ab = car_cfg["auto_brake"]
    sig = D[ab["signal"]].to_numpy()
    auto = (sig < ab["below"]) if ab.get("below") is not None else (sig > ab["above"])
    auto = np.nan_to_num(auto).astype(bool)
    moving = D.v.to_numpy() > cfg["events"]["min_speed_mps"]
    # 0 点: RAV4 は 0、Civic は走行ごとに、踏んでおらず自動ブレーキも無い走行中の中央値
    if car_cfg.get("level_zero") is not None and a.level == "level":
        zero = np.full(len(D), float(car_cfg["level_zero"]))
    else:
        base = pd.Series(np.where(~pressed & ~auto & moving, D[a.level], np.nan)).groupby(D.drive).transform("median")
        zero = base.to_numpy()
    L = D[lev].to_numpy() - zero
    Lraw = D[a.level].to_numpy() - zero
    # 制動中: RAV4 と Civic の USER_BRAKE はペダル、Civic の BRAKE_PRESSURE1 は自動ブレーキも含む
    uses_auto = (a.car == "civic" and a.level == "level")
    active = pressed | auto if uses_auto else pressed
    l_p99 = float(np.nanpercentile(L[pressed & moving], 99))
    S: dict = dict(car=a.car, level=a.level, n_rows=len(D), hours=round(len(D) * dt / 3600, 2), drives=int(D.drive.nunique()),
                   segments=int(D.groupby(["drive", "seg"]).ngroups), pressed_frac=round(float(pressed.mean()), 4),
                   auto_frac_moving=round(float(auto[moving].mean()), 4), level_p99_pressed=l_p99)

    # ---------------- 2. 相関と R² の区別 -----------------------------------
    m = pressed & moving & np.isfinite(Lraw) & ~D.edge.to_numpy()
    x, y = Lraw[m], -D.a.to_numpy()[m]
    r = float(np.corrcoef(x, y)[0, 1])
    k1 = np.polyfit(x, y, 1)
    q = pd.qcut(pd.Series(x), 10, duplicates="drop")
    xb = pd.Series(x).groupby(q, observed=True).mean().to_numpy()
    yb = pd.Series(y).groupby(q, observed=True).mean().to_numpy()
    S["q2_correlation"] = dict(
        target="走行中 (3 m/s 超) にペダルを踏んでいる 20 Hz の時刻すべて。値は生の値 (ならしなし)、減速度は輪速の 0.5 s 平均の微分。勾配の補正なし",
        n_samples=int(m.sum()),
        r_sample=round(r, 4), r2_sample_single_regression=round(r * r, 4),
        r2_sample_from_fit=round(BM.r2(y, np.polyval(k1, x)), 4),
        r2_decile_means=round(BM.r2(yb, np.polyval(np.polyfit(xb, yb, 1), xb)), 4),
        note="r^2 と 10 分位の平均どうしの R^2 は別の量。後者は各階級の平均をとってから当てはめるので、時刻ごとのばらつきが消える")

    # ---------------- 3. 勾配の確かめ ---------------------------------------
    th = D.grade_rad.to_numpy()
    gc = {}
    mc = moving & np.isfinite(th) & ~D.edge.to_numpy()
    cruise = mc & ~pressed & ~auto & (np.abs(D.a.to_numpy()) < 0.3)
    brk = mc & pressed & (D[lev].to_numpy() - zero > 0.1 * l_p99)
    rows = []
    if "ASLP" in D:
        asl = D.ASLP.to_numpy()
        for name, mm in [("cruise", cruise), ("braking", brk)]:
            ok = mm & np.isfinite(asl)
            kk = np.polyfit(np.degrees(th[ok]), asl[ok], 1)
            rows.append(dict(check="ASLP_vs_gp_grade", subset=name, n=int(ok.sum()), r=float(np.corrcoef(np.degrees(th[ok]), asl[ok])[0, 1]),
                             slope=float(kk[0]), intercept=float(kk[1])))
        # 制動中に ASLP が減速度でずれるか (車体のピッチが入るか)
        ok = brk & np.isfinite(asl)
        kc = np.polyfit(np.degrees(th[cruise & np.isfinite(asl)]), asl[cruise & np.isfinite(asl)], 1)
        res = asl[ok] - np.polyval(kc, np.degrees(th[ok]))
        kr = np.polyfit(-D.a.to_numpy()[ok], res, 1)
        rows.append(dict(check="ASLP_residual_vs_decel", subset="braking", n=int(ok.sum()), r=float(np.corrcoef(-D.a.to_numpy()[ok], res)[0, 1]),
                         slope=float(kr[0]), intercept=float(kr[1]), unit="ASLP の値 / (m/s^2)"))
    # 車両の加速度センサ由来の「勾配 + ピッチ」: (加速度 - 車速の微分) / g
    acc_key = "accel_x" if "accel_x" in D else ("long_accel" if "long_accel" in D else None)
    for key in [k for k in (acc_key, "gvc") if k and k in D]:
        ax = D[key].to_numpy()
        ok = cruise & np.isfinite(ax)
        # 加速度センサの値 = s (dv/dt + g sin(theta + pitch)) + b を巡航で当てはめ、s と b を出す
        X = np.stack([D.a.to_numpy()[ok], g * np.sin(th[ok]), np.ones(ok.sum())], -1)
        cs = BM.lstsq(X, ax[ok])
        rows.append(dict(check=f"{key}_model_cruise", subset="cruise", n=int(ok.sum()), coef_dvdt=float(cs[0]), coef_gsin_grade=float(cs[1]), intercept=float(cs[2]),
                         note="coef_gsin_grade が 1 に近ければ重力 (勾配) を含む加速度センサ、0 に近ければ車速から作った値"))
        if abs(cs[1]) > 0.5:
            okb = brk & np.isfinite(ax)
            # 勾配 + ピッチの推定 - 勾配 = ピッチ (rad)。cs[0] で割って目盛りをそろえる
            pitch = np.arcsin(np.clip(((ax[okb] - cs[2]) / cs[0] - D.a.to_numpy()[okb]) / g, -1, 1)) - th[okb]
            pitch0 = np.arcsin(np.clip(((ax[ok] - cs[2]) / cs[0] - D.a.to_numpy()[ok]) / g, -1, 1)) - th[ok]
            kp = np.polyfit(-D.a.to_numpy()[okb], np.degrees(pitch - np.median(pitch0)), 1)
            rows.append(dict(check=f"{key}_pitch_vs_decel", subset="braking", n=int(okb.sum()), slope=float(kp[0]), intercept=float(kp[1]),
                             r=float(np.corrcoef(-D.a.to_numpy()[okb], pitch)[0, 1]), unit="deg / (m/s^2)",
                             note="加速度センサで勾配を出すと、制動中はこの分だけ車体の前のめりが勾配に混ざる"))
    G = pd.DataFrame(rows)
    G.to_csv(out / "grade_checks.csv", index=False)
    gc["checks"] = G.round(4).to_dict("records")
    # 同じ場所・同じ向きの再現性 (global_pose の勾配)
    if "lat" in D:
        lat0, lon0 = np.nanmedian(D.lat), np.nanmedian(D.lon)
        ex = (D.lon - lon0).to_numpy() * np.cos(np.radians(lat0)) * 111320.0
        ny = (D.lat - lat0).to_numpy() * 110540.0
        cell = cfg["grade"]["repeat_cell_m"]
        okr = mc & np.isfinite(ex) & np.isfinite(D.heading_deg.to_numpy())
        R = pd.DataFrame(dict(cx=np.floor(ex[okr] / cell), cy=np.floor(ny[okr] / cell), hd=np.floor(D.heading_deg.to_numpy()[okr] / 45),
                              drive=D.drive.to_numpy()[okr], grade=np.degrees(th[okr])))
        per = R.groupby(["cx", "cy", "hd", "drive"]).grade.mean().reset_index()
        cnt = per.groupby(["cx", "cy", "hd"]).drive.nunique()
        keep = cnt[cnt >= cfg["grade"]["repeat_min_drives"]].index
        per = per.set_index(["cx", "cy", "hd"]).loc[keep].reset_index()
        dev = per.grade - per.groupby(["cx", "cy", "hd"]).grade.transform("mean")
        nn = per.groupby(["cx", "cy", "hd"]).grade.transform("count")
        dev_unbiased = dev * np.sqrt(nn / (nn - 1))
        per["dev_deg"] = dev_unbiased
        per.to_csv(out / "grade_repeat.csv", index=False)
        gc["repeat"] = dict(cells=int(len(keep)), samples=int(len(per)), sd_between_drives_deg=round(float(np.sqrt(np.mean(dev_unbiased**2))), 4),
                            p95_abs_dev_deg=round(float(np.percentile(np.abs(dev_unbiased), 95)), 4),
                            grade_sd_deg_all=round(float(np.nanstd(np.degrees(th[mc]))), 3),
                            accel_equiv_sd_mps2=round(float(g * np.radians(np.sqrt(np.mean(dev_unbiased**2)))), 4))
    S["q3_grade"] = gc

    # ---------------- 4. 正常制動の模型と交差検証 ---------------------------
    folds = BM.drive_folds(D.drive.to_numpy(), cfg["model"]["folds"], cfg["model"]["seed"])
    fold = D.drive.map(folds).to_numpy()
    gas = D.gas.to_numpy()
    edge = D.edge.to_numpy()
    grades = {"none": np.zeros(len(D)), "global_pose": th}
    if "ASLP" in D:
        okc = cruise & np.isfinite(D.ASLP.to_numpy())
        kc = np.polyfit(np.degrees(th[okc]), D.ASLP.to_numpy()[okc], 1)
        grades["ASLP"] = np.radians((D.ASLP.to_numpy() - kc[1]) / kc[0])   # global_pose の勾配に合わせて目盛りを直す
    lag_steps = [int(round(x / dt)) for x in cfg["model"]["lags_s"]]
    forms = cfg["model"]["forms"]
    blk = D.block
    v = D.v.to_numpy()
    acc = D.a.to_numpy()
    E = BM.find_events(D, active & ~edge, L, cfg, l_p99, dt)
    S["events_found"] = int(len(E))
    ev_auto = np.array([auto[r.i0:r.i1 + 1].any() for r in E.itertuples()]) if len(E) else np.array([], bool)
    use_ev = ~ev_auto if (cfg["events"]["exclude_auto_brake"] and not uses_auto) else np.ones(len(E), bool)
    E = E[use_ev].reset_index(drop=True)
    S["events_used"] = int(len(E))
    S["events_excluded_auto_overlap"] = int((~use_ev).sum())
    cv_rows = []
    ev_pred = {}
    sample_pred = {}
    std_coast = bool(cfg["coast"].get("use_in_standard", False))
    for gname, thg in grades.items():
        for use_coast in ([std_coast, not std_coast] if gname == "global_pose" else [std_coast]):
            pre = np.zeros(len(D))
            dvals = np.full(len(D), np.nan)
            for f in range(cfg["model"]["folds"]):
                tr = fold != f
                cm = tr & (v > cfg["coast"]["min_speed_mps"]) & (np.nan_to_num(gas) <= cfg["coast"]["gas_max"]) & ~pressed & ~auto & np.isfinite(thg) & ~edge
                if uses_auto is False and a.car == "civic":
                    cm &= ~auto
                resid = -acc - g * np.sin(np.nan_to_num(thg))
                if use_coast:
                    cc = BM.fit_coast(v[cm], resid[cm], cfg["coast"]["form"])
                    dc = BM.coast(v, cc, cfg["coast"]["form"])
                else:
                    dc = 0.0
                te = fold == f
                dvals[te] = (resid - dc)[te]
                if f == 0 and use_coast and gname == "global_pose":  # 比較用の惰行の模型の係数
                    S.setdefault("coast_coef_fold0", dict(zip(cfg["coast"]["form"], np.round(cc, 5).tolist())))
            dvals[~np.isfinite(thg)] = np.nan
            dser = pd.Series(dvals)
            for fname, cols in forms.items():
                for ls in lag_steps:
                    dl = BM.shift_by_lag(dser, blk, ls).to_numpy()
                    ms = active & ~edge & moving & np.isfinite(L) & np.isfinite(dl) & (L >= cfg["model"]["sample_min_frac"] * l_p99)
                    if not uses_auto:
                        ms &= ~auto
                    pred = np.full(len(D), np.nan)
                    coefs = []
                    for f in range(cfg["model"]["folds"]):
                        tr, te = ms & (fold != f), ms & (fold == f)
                        X = BM.design(cols, L[tr], v[tr])
                        c = BM.lstsq(X, dl[tr])
                        coefs.append(c)
                        allte = (fold == f) & np.isfinite(L)
                        pred[allte] = BM.design(cols, L[allte], v[allte]) @ c
                    yy, pp = dl[ms], pred[ms]
                    key = (gname, use_coast, fname, ls)
                    # 制動ごと (学習に使っていない走行での予測)
                    obs = np.array([np.nanmean(dl[r.steady]) for r in E.itertuples()]) if len(E) else np.array([])
                    prd = np.array([np.nanmean(pred[r.steady]) for r in E.itertuples()]) if len(E) else np.array([])
                    eta = obs / prd
                    okv = np.isfinite(eta) & (prd > 0.3)
                    cv_rows.append(dict(grade=gname, coast=use_coast, form=fname, lag_s=ls * dt, n_samples=int(ms.sum()),
                                        rmse_sample=float(np.sqrt(np.mean((yy - pp) ** 2))), r2_sample=BM.r2(yy, pp),
                                        n_events=int(okv.sum()),
                                        event_rmse=float(np.sqrt(np.nanmean((obs[okv] - prd[okv]) ** 2))) if okv.any() else np.nan,
                                        eta_median=float(np.median(eta[okv])) if okv.any() else np.nan,
                                        eta_robust_cv=robust_cv(eta[okv]) if okv.sum() > 5 else np.nan,
                                        eta_p05=float(np.percentile(eta[okv], 5)) if okv.sum() > 5 else np.nan,
                                        eta_p95=float(np.percentile(eta[okv], 95)) if okv.sum() > 5 else np.nan,
                                        coef_mean=np.round(np.mean(coefs, 0), 6).tolist(), coef_sd_folds=np.round(np.std(coefs, 0), 6).tolist()))
                    ev_pred[key] = (obs, prd)
                    sample_pred[key] = (ms, dl, pred)
    CV = pd.DataFrame(cv_rows)
    CV.to_csv(out / "model_cv.csv", index=False)
    # 標準: 勾配 global_pose・惰行の模型あり・模型と遅れは学習に使っていない走行の誤差 (制動ごと) が一番小さいもの
    base_cv = CV[(CV.grade == "global_pose") & (CV.coast == std_coast)]
    std = base_cv.sort_values("event_rmse").iloc[0]
    key = ("global_pose", std_coast, std.form, int(round(std.lag_s / dt)))
    S["q4_model"] = dict(selected=dict(form=std.form, lag_s=float(std.lag_s), terms=forms[std.form]),
                         by_form=CV[(CV.grade == "global_pose") & (CV.coast == std_coast) & (CV.lag_s == std.lag_s)][["form", "rmse_sample", "r2_sample", "event_rmse", "eta_robust_cv", "eta_p05", "eta_p95", "coef_mean", "coef_sd_folds"]].round(4).to_dict("records"),
                         by_grade=CV[(CV.form == std.form) & (CV.lag_s == std.lag_s)][["grade", "coast", "rmse_sample", "r2_sample", "event_rmse", "eta_median", "eta_robust_cv", "eta_p05", "eta_p95"]].round(4).to_dict("records"),
                         by_lag=CV[(CV.grade == "global_pose") & (CV.coast == std_coast) & (CV.form == std.form)][["lag_s", "rmse_sample", "r2_sample", "event_rmse"]].round(4).to_dict("records"))
    obs, prd = ev_pred[key]
    ms, dl, pred = sample_pred[key]

    # 制動ごとの表
    blk_np = blk.to_numpy()
    # ブレーキで熱になった仕事率 (W/kg): ペダルでも自動ブレーキでも、ブレーキが作動している間の
    #   max(0, -dv/dt - g sin(theta) - 抵抗) · v。抵抗は模型 M4 の (1, v^2) の項 (制動中の空気・転がり・エンジンブレーキ)
    cm4 = np.array(CV[(CV.grade == "global_pose") & (CV.coast == std_coast) & (CV.form == "M4") & (CV.lag_s == 0)].coef_mean.iloc[0])
    resist = cm4[0] + cm4[1] * v * v
    d_heat = -acc - g * np.sin(np.nan_to_num(th)) - resist
    brake_on = pressed | auto
    power = np.where(brake_on & ~edge, np.clip(d_heat, 0, None) * v, 0.0)
    S["heat_resistance_terms"] = dict(c0=round(float(cm4[0]), 4), c_v2=round(float(cm4[1]), 6))
    H = BM.heat_indices(power, blk_np, dt, cfg["heat"]["taus_s"], cfg["heat"]["windows_s"])
    rows = []
    for j, r in enumerate(E.itertuples()):
        st = r.steady
        rows.append(dict(drive=D.drive.iat[r.i0], seg=int(D.seg.iat[r.i0]), t0=float(D.t.iat[r.i0]), dur_s=(r.i1 - r.i0 + 1) * dt,
                         fold=int(fold[r.i0]), v0=float(v[r.i0]), v_steady=float(np.mean(v[st])), level_steady=float(np.nanmean(L[st])),
                         peak=r.peak, grade_deg=float(np.degrees(np.nanmean(th[st]))), decel_wheel=float(-np.mean(acc[st])),
                         decel_brake_obs=float(obs[j]), decel_brake_pred=float(prd[j]), eta=float(obs[j] / prd[j]) if prd[j] > 0.3 else np.nan,
                         auto_overlap=bool(auto[r.i0:r.i1 + 1].any()),
                         **{f"{k}_before": float(H[k][max(r.i0 - 1, 0)]) for k in H}))
    EV = pd.DataFrame(rows)
    EV.to_csv(out / "events.csv", index=False)
    S["events_eta_nonpositive"] = int((EV.eta <= 0).sum())   # 観測の減速度が 0 以下 (勾配・抵抗の見積もりの誤りが大きい制動)
    ok = EV.eta.notna() & (EV.eta > 0)
    eta = EV.eta[ok].to_numpy()
    per_drive = EV[ok].groupby("drive").eta.agg(["median", "count"])
    big = per_drive[per_drive["count"] >= 3]
    # 走行内と走行間のばらつき (対数で分ける)
    le = np.log(EV.eta[ok])
    within = float(np.sqrt(np.mean((le - le.groupby(EV.drive[ok]).transform("mean")) ** 2)))
    S["q4_events"] = dict(n_events=int(ok.sum()), n_drives=int(EV[ok].drive.nunique()),
                          eta_median=round(float(np.median(eta)), 4), eta_robust_cv=round(robust_cv(eta), 4),
                          eta_p01_p05_p95_p99=np.round(np.percentile(eta, [1, 5, 95, 99]), 4).tolist(),
                          drives_with_3plus=int(len(big)), drive_median_eta_sd=round(float(big["median"].std()), 4) if len(big) > 2 else None,
                          log_eta_sd_total=round(float(le.std()), 4), log_eta_sd_within_drive=round(within, 4),
                          detect_single_event_drop_at_5pct_false_alarm=round(float(1 - np.percentile(eta, 5) / np.median(eta)), 3),
                          by_speed=EV[ok].groupby(pd.cut(EV.v_steady[ok], [0, 10, 20, 30, 50]), observed=True).eta.agg(["count", "median", robust_cv]).round(3).reset_index().astype(str).to_dict("records"),
                          by_level=EV[ok].groupby(pd.qcut(EV.level_steady[ok], 4, duplicates="drop"), observed=True).eta.agg(["count", "median", robust_cv]).round(3).reset_index().astype(str).to_dict("records"),
                          naive_gain_robust_cv=round(robust_cv((EV.decel_wheel / EV.level_steady)[ok & (EV.decel_wheel > 0.3)].to_numpy()), 4))
    # n 回の制動の平均で見分けられる低下 (同じ走行の中で n 回をくじ引き)
    rng = np.random.default_rng(cfg["model"]["seed"])
    det = {}
    for n in (1, 3, 5, 10):
        means = []
        for dname, gdf in EV[ok].groupby("drive"):
            x = np.log(gdf.eta.to_numpy())
            if len(x) >= n:
                for _ in range(50):
                    means.append(np.mean(rng.choice(x, n, replace=False)))
        if len(means) > 20:
            det[n] = round(float(1 - np.exp(np.percentile(means, 5) - np.median(means))), 3)
    S["q4_events"]["detectable_drop_mean_of_n_same_drive_5pct"] = det

    # ---------------- 5. 熱負荷 --------------------------------------------
    ref = 0.5 * (cfg["heat"]["ref_stop_kmh"] / 3.6) ** 2
    hb = pd.DataFrame({k: pd.Series(H[k]).groupby(D.drive).max() for k in H})
    hb["drive_min"] = D.groupby("drive").size() * dt / 60
    hb["brake_events_per_h"] = EV.groupby("drive").size().reindex(hb.index).fillna(0) / hb.drive_min * 60
    hb["energy_total_Jkg"] = pd.Series(power * dt).groupby(D.drive).sum()
    on = np.r_[False, brake_on[1:] & ~brake_on[:-1]] & moving
    n300 = BM.heat_indices(on.astype(float) / dt, blk_np, dt, [], [300])["E300"]
    hb["brake_onsets_in_300s_max"] = pd.Series(n300).groupby(D.drive).max()
    hb.to_csv(out / "heat_by_drive.csv")
    S["q5_heat"] = dict(ref_one_stop_Jkg=round(ref, 1),
                        power_definition="ブレーキ作動中 (ペダルか自動ブレーキ) の max(0, -dv/dt - g sin(勾配) - 抵抗)·v。抵抗は模型 M4 の (1, v^2) の項",
                        max_over_drives={k: round(float(hb[k].max()), 1) for k in H},
                        p90_over_drives={k: round(float(hb[k].quantile(0.9)), 1) for k in H},
                        E300_max_in_ref_stops=round(float(hb["E300"].max() / ref), 2),
                        E900_max_in_ref_stops=round(float(hb["E900"].max() / ref), 2),
                        brake_onsets_in_300s_max=int(round(hb["brake_onsets_in_300s_max"].max())),
                        brake_onsets_in_300s_p90_over_drives=round(float(hb["brake_onsets_in_300s_max"].quantile(0.9)), 1),
                        top_drives_E300=hb.sort_values("E300", ascending=False).head(5)[["E300", "E900", "H300", "drive_min"]].round(1).reset_index().to_dict("records"))
    # 熱の指標と利きの関係 (学習に使っていない走行での eta)
    if ok.sum() > 20:
        for k in ("H300", "E300"):
            rr = pd.Series(EV[k + "_before"][ok]).rank().corr(pd.Series(eta).rank())
            S["q5_heat"][f"spearman_eta_vs_{k}_before"] = round(float(rr), 4)
        S["q5_heat"]["eta_median_top10pct_H300_vs_rest"] = [round(float(EV.eta[ok][EV.H300_before[ok] >= EV.H300_before[ok].quantile(0.9)].median()), 4),
                                                            round(float(EV.eta[ok][EV.H300_before[ok] < EV.H300_before[ok].quantile(0.9)].median()), 4)]
    S["decel_p001_wheel"] = round(float(np.nanpercentile(acc[moving & ~edge], 0.1)), 3)
    S["elapsed_s"] = round(time.time() - t0, 1)

    # ---------------- 図 ---------------------------------------------------
    configure_japanese_font()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    sel = ms & np.isfinite(pred)
    ii = np.where(sel)[0][:: max(1, sel.sum() // 6000)]
    ax[0].scatter(L[ii], dl[ii], s=3, alpha=0.25, color="#4a6fa5", label="時刻ごと (学習に使っていない走行)")
    xs = np.linspace(0, np.nanpercentile(L[sel], 99.5), 50)
    c = np.array(CV[(CV.grade == "global_pose") & (CV.coast == std_coast) & (CV.form == std.form) & (CV.lag_s == std.lag_s)].coef_mean.iloc[0])
    ax[0].plot(xs, BM.design(forms[std.form], xs, np.full_like(xs, np.nanmedian(v[sel]))) @ c, color="#c0392b", lw=2, label=f"模型 {std.form} (速度は中央値)")
    ax[0].scatter(EV.level_steady, EV.decel_brake_obs, s=14, color="#e67e22", label="1 回の制動の定常部")
    ax[0].set_xlabel("制動の値 (0 点を引いたもの、単位不明)")
    ax[0].set_ylabel("ブレーキによる減速度 [m/s²]")
    ax[0].legend(fontsize=8)
    ax[1].hist(eta, bins=40, color="#4a6fa5")
    ax[1].axvline(1, color="k", lw=1)
    for p_ in (5, 95):
        ax[1].axvline(np.percentile(eta, p_), color="#c0392b", ls="--", lw=1)
    ax[1].set_xlabel("利きの比 eta = 観測 / 予測 (学習に使っていない走行)")
    ax[1].set_ylabel("制動の回数")
    ax[1].set_title(f"n={ok.sum()}  変動係数 {robust_cv(eta):.2f}  p5={np.percentile(eta, 5):.2f}")
    ax[2].scatter(EV.H300_before[ok], eta, s=12, color="#4a6fa5")
    ax[2].axhline(1, color="k", lw=1)
    ax[2].set_xscale("symlog", linthresh=50)
    ax[2].set_xlabel("直前の熱の指標 H300 [J/kg] (時定数 300 s)")
    ax[2].set_ylabel("eta")
    ax[2].axvline(ref, color="#c0392b", ls="--", lw=1)
    ax[2].text(ref, ax[2].get_ylim()[1], " 100 km/h から 1 回の停止", fontsize=8, va="top", color="#c0392b")
    for x_ in ax:
        x_.grid(alpha=0.3)
    fig.suptitle(f"{a.car} / {a.level}: 正常制動の模型と利きのばらつき")
    fig.tight_layout()
    fig.savefig(out / "model_eta_heat.png", dpi=120)
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    (out / "meta.json").write_text(json.dumps(dict(config=cfg, car=a.car, level=a.level, run_id=a.run_id, limit=a.limit), ensure_ascii=False, indent=1))
    print(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
