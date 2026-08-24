#!/usr/bin/env python3
"""KIT Multi-Surface Driving Maneuvers を物差しに、横滑り関係の特徴量を検証する。

comma 系のデータでは横滑り角 beta が推定器のノイズ床に埋もれて検証できなかった。
このデータセットは光学式センサで beta を 1000 Hz 実測しており、
乾燥アスファルト (mu=1.1) と敷石 (mu=0.7) の閉鎖路で横方向の限界まで走っている。

確認するのは 4 つ。

  A. 符号の取り決め   ISO 8855 と本プロジェクトの「左が正」が一致しているか
  B. ay_kin の妥当性  v * yaw_rate を横加速度として使ってよい範囲
  C. 単軌道モデル     舵角から期待されるヨーレートが、どの beta まで通用するか
  D. beta の推定      CAN で取れる量 (v, yaw_rate, a_y) だけから beta を出せるか

D が通れば commaCarSegments の 3,148 時間に横滑りの指標を持ち込める。

使い方:
  python scripts/validate_sideslip_kit.py
  python scripts/validate_sideslip_kit.py --out out/kit_msdm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.io import kit_msdm as kit

MIN_SPEED = 3.0          # これ未満は beta が定義できない
BETA_BINS = [0, 2, 4, 6, 8, 10, 90]


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, n // 2, mode="edge"), k, mode="same")[n // 2 : n // 2 + x.size]


def collect(paths: list[Path], params: kit.VehicleParams) -> pd.DataFrame:
    """dynamic run をまとめて 1 枚の表にする。"""
    L, l_f, l_r = params.wheelbase_m, params.l_f, params.l_r
    m, c_r = params.mass_kg, params.c_r
    kus = params.understeer_gradient()

    frames = []
    for p in paths:
        run = kit.read_run(p)
        need = ("v_x_cor_mps", "v_y_cor_mps", "w_z_cor_radps", "a_y_ra_mps2", "delta_stm_rad")
        if any(c not in run for c in need):
            continue
        v_x = run["v_x_cor_mps"]
        v_y = run["v_y_cor_mps"]
        yaw = run["w_z_cor_radps"]                       # rad/s
        delta = run["delta_stm_rad"]                     # タイヤ切れ角 [rad]
        a_y = run["a_y_ra_mps2"]                         # 後軸位置。微分を挟まない

        # Correvit は重心より約 2.4 m 後ろにある。横滑り角は計測点で大きく変わるので、
        # 実測値もモデルも重心位置に揃える。
        r_cog = np.asarray(params.trvec("cor_ra"), float) + np.asarray(params.trvec("ra_cog"), float)
        vx_cog, vy_cog = kit.translate_velocity(v_x, v_y, yaw, r_cog)

        beta_cor = np.degrees(np.arctan2(v_y, v_x))      # Correvit 位置 (生の計測)
        beta = np.degrees(np.arctan2(vy_cog, vx_cog))    # 重心位置
        v = np.hypot(v_x, v_y)

        ay_kin = v * yaw                                 # 本プロジェクトが使っている作り方
        yaw_exp = v * delta / (L + kus * v**2)           # 線形単軌道モデルの期待ヨーレート
        # 定常の単軌道モデルから出す重心位置の beta (CAN で取れる量だけを使う)
        beta_model = np.degrees(l_r * yaw / np.maximum(v, 1e-3) - m * l_f * a_y / (c_r * L))

        frames.append(pd.DataFrame({
            "run": run.name, "surface": run.surface, "t": run.t,
            "v": v, "v_x": v_x, "v_y": v_y, "yaw": yaw,
            "delta": delta, "a_y": a_y,
            "beta": beta, "beta_cor": beta_cor,
            "ay_kin": ay_kin, "yaw_exp": yaw_exp, "beta_model": beta_model,
        }))
    df = pd.concat(frames, ignore_index=True)
    return df[df.v_x > MIN_SPEED].reset_index(drop=True)


def section_a(df: pd.DataFrame) -> None:
    print("=== A. 符号の取り決め ===")
    print("ISO 8855 は x 前 / y 左 / z 上。本プロジェクトも「左旋回・左向きが正」で揃えている。")
    r_dy = np.corrcoef(df.delta, df.yaw)[0, 1]
    r_ay = np.corrcoef(df.yaw, df.ay_kin)[0, 1]
    r_bd = np.corrcoef(df.delta, df.beta)[0, 1]
    print(f"  舵角 delta と ヨーレート          r = {r_dy:+.4f}  (左に切れば左に回る → 正で一致)")
    print(f"  ヨーレート と ay_kin(=v*yaw)      r = {r_ay:+.4f}  (定義どおり)")
    print(f"  舵角 delta と 実測 beta (重心)    r = {r_bd:+.4f}")
    print("    低速では幾何の項が支配的で舵角と同符号になる。速度が上がるとタイヤの")
    print("    横滑りの項が勝って逆符号へ移る。このデータは 50 km/h 以下なので同符号。")
    if r_dy < 0:
        print("  !! 舵角とヨーレートが逆符号。符号規約の見直しが要る")


def section_b(df: pd.DataFrame) -> None:
    print("\n=== B. ay_kin = v * yaw_rate をどこまで信じてよいか ===")
    err = df.ay_kin - df.a_y
    print("  比較先は後軸位置の実測 a_y。微分を挟まないので、比較そのものにノイズが乗らない。")
    print(f"  差: 中央値 {err.median():+.3f}  標準偏差 {err.std():.3f}  |最大| {err.abs().max():.2f} m/s^2")
    print(f"  相関 r = {np.corrcoef(df.ay_kin, df.a_y)[0, 1]:.4f}")
    g = df.assign(bin=pd.cut(df.beta.abs(), BETA_BINS)).groupby("bin", observed=True)
    tab = g.apply(lambda s: pd.Series({
        "n": len(s),
        "秒": round(len(s) / kit.RATE_HZ, 1),
        "|a_y|中央値": round(float(s.a_y.abs().median()), 2),
        "誤差中央値": round(float((s.ay_kin - s.a_y).median()), 3),
        "誤差σ": round(float((s.ay_kin - s.a_y).std()), 3),
        "誤差|最大|": round(float((s.ay_kin - s.a_y).abs().max()), 2),
    }), include_groups=False)
    print(tab.to_string())


def section_c(df: pd.DataFrame, params: kit.VehicleParams) -> None:
    print("\n=== C. 線形単軌道モデルはどの beta まで通用するか ===")
    print(f"  Kus = {params.understeer_gradient():+.6f} rad*s^2/m  "
          f"(l_f={params.l_f} > l_r={params.l_r} で重心が後ろ寄り → オーバーステア傾向)")
    res = np.degrees(df.yaw - df.yaw_exp)
    print(f"  残差 (実測 - 期待) [deg/s]: 中央値 {np.median(res):+.2f}  σ {np.std(res):.2f}  "
          f"|最大| {np.max(np.abs(res)):.1f}")
    g = df.assign(bin=pd.cut(df.beta.abs(), BETA_BINS), res=res).groupby("bin", observed=True)
    tab = g.apply(lambda s: pd.Series({
        "秒": round(len(s) / kit.RATE_HZ, 1),
        "残差中央値": round(float(s.res.median()), 2),
        "残差σ": round(float(s.res.std()), 2),
        "|残差|p99": round(float(s.res.abs().quantile(0.99)), 2),
        "|残差|/|実測yaw|": round(float((s.res.abs() / np.degrees(s.yaw).abs().clip(lower=1)).median()), 3),
    }), include_groups=False)
    print(tab.to_string())


def section_c2(df: pd.DataFrame) -> None:
    """ヨー残差が何に反応しているのかを見る。"""
    print("\n  -- 残差は何に反応しているか --")
    res = np.abs(np.degrees(df.yaw - df.yaw_exp))
    # 舵角は量子化されているので、1 kHz のまま微分すると 0 とスパイクしか出ない
    rate = np.concatenate([
        np.degrees(np.gradient(smooth(g.delta.to_numpy(), 51), 1.0 / kit.RATE_HZ))
        for _, g in df.groupby("run", sort=False)
    ])
    for nm, x in (("|beta|", df.beta.abs().to_numpy()),
                  ("|舵角レート| [deg/s]", np.abs(rate)),
                  ("|ヨーレート| [deg/s]", np.degrees(df.yaw.abs().to_numpy())),
                  ("速度 [m/s]", df.v.to_numpy())):
        print(f"    |残差| と {nm:22s} の相関 r = {np.corrcoef(res, x)[0, 1]:+.4f}")
    print("    -> 残差は横滑りではなく速度と過渡に反応する。sideslip の検出器にはならない。")


def section_e(df: pd.DataFrame, params: kit.VehicleParams) -> None:
    """beta 推定が車両諸元の誤差にどれだけ耐えるか。

    commaCarSegments では C_r (コーナリング剛性) が分からない。
    そこがどれだけ効くのかを決めておかないと、他車種へ持ち込めない。
    """
    print("\n=== E. beta 推定の車両諸元に対する感度 ===")
    L, l_f, l_r = params.wheelbase_m, params.l_f, params.l_r
    m, c_r = params.mass_kg, params.c_r
    k_nom = m * l_f / (c_r * L)
    print(f"  第2項の係数 k = m*l_f/(C_r*L) = {k_nom:.6f} s^2/m")

    v, yaw, a_y, beta = df.v.to_numpy(), df.yaw.to_numpy(), df.a_y.to_numpy(), df.beta.to_numpy()

    def estimate(k: float, lr: float) -> np.ndarray:
        return np.degrees(lr * yaw / np.maximum(v, 1e-3) - k * a_y)

    rows = []
    for nm, k, lr in (("公称", k_nom, l_r),
                      ("C_r を 1.5 倍", k_nom / 1.5, l_r),
                      ("C_r を 0.5 倍", k_nom / 0.5, l_r),
                      ("第2項を落とす (幾何のみ)", 0.0, l_r),
                      ("l_r を 0.9 倍", k_nom, l_r * 0.9),
                      ("l_r を 1.1 倍", k_nom, l_r * 1.1)):
        est = estimate(k, lr)
        err = est - beta
        hit = (np.abs(est) >= 5) & (np.abs(beta) >= 5)
        rows.append({
            "条件": nm,
            "相関": round(float(np.corrcoef(beta, est)[0, 1]), 4),
            "傾き": round(float(np.polyfit(beta, est, 1)[0]), 3),
            "誤差σ": round(float(err.std()), 2),
            "誤差|最大|": round(float(np.abs(err).max()), 1),
            "再現率(5deg)": round(float(hit.sum() / max((np.abs(beta) >= 5).sum(), 1)), 3),
            "適合率(5deg)": round(float(hit.sum() / max((np.abs(est) >= 5).sum(), 1)), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def section_d(df: pd.DataFrame, params: kit.VehicleParams) -> None:
    print("\n=== D. CAN で取れる量だけから beta を出せるか ===")
    print("  定常単軌道モデル:  beta = l_r*yaw/v - m*l_f*a_y/(C_r*L)")
    print("  使うのは v / yaw_rate / a_y の 3 つだけ。すべて commaCarSegments で復号済み。")
    print("  実測・モデルとも重心位置に揃えてある (Correvit 位置のままだと符号すら合わない)。")

    # 計測点の当てはめ: 実測 beta がどの位置でモデルと一致するかを探し、
    # 公称の Correvit->重心 距離と合うかどうかで幾何の理解を裏取りする。
    r_nom = float(np.asarray(params.trvec("cor_ra"))[0] + np.asarray(params.trvec("ra_cog"))[0])
    best = None
    for dx in np.arange(0.0, 4.01, 0.05):
        b = np.degrees(np.arctan2(df.v_y + df.yaw * dx, df.v_x + df.yaw * 0.159))
        rms = float(np.sqrt(np.mean((b - df.beta_model) ** 2)))
        if best is None or rms < best[1]:
            best = (dx, rms)
    print(f"  計測点の当てはめ: RMS 最小は Correvit から前方 {best[0]:.2f} m "
          f"(公称の重心位置 {r_nom:.3f} m)  そこでの RMS {best[1]:.2f} deg")
    ok = np.isfinite(df.beta_model) & np.isfinite(df.beta)
    b, bm = df.beta[ok], df.beta_model[ok]
    err = bm - b
    slope, intercept = np.polyfit(b, bm, 1)
    print(f"  相関 r = {np.corrcoef(b, bm)[0, 1]:.4f}   回帰 傾き {slope:.3f} 切片 {intercept:+.2f}")
    print(f"  誤差: 中央値 {err.median():+.2f}  σ {err.std():.2f}  |最大| {err.abs().max():.1f} deg")
    g = df[ok].assign(bin=pd.cut(b.abs(), BETA_BINS), err=err).groupby("bin", observed=True)
    tab = g.apply(lambda s: pd.Series({
        "秒": round(len(s) / kit.RATE_HZ, 1),
        "実測|beta|中央値": round(float(s.beta.abs().median()), 2),
        "推定|beta|中央値": round(float(s.beta_model.abs().median()), 2),
        "誤差中央値": round(float(s.err.median()), 2),
        "誤差σ": round(float(s.err.std()), 2),
    }), include_groups=False)
    print(tab.to_string())
    for th in (3.0, 5.0, 8.0):
        tp = ((bm.abs() >= th) & (b.abs() >= th)).sum()
        fp = ((bm.abs() >= th) & (b.abs() < th)).sum()
        fn = ((bm.abs() < th) & (b.abs() >= th)).sum()
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  |beta| >= {th:.0f} deg の判定: 適合率 {prec:.3f}  再現率 {rec:.3f}  (該当 {tp + fn} サンプル)")


def surface_table(df: pd.DataFrame) -> None:
    print("\n=== 路面別 (mu: asphalt_a 1.1 / cobblestone 0.7) ===")
    g = df.groupby("surface")
    tab = g.apply(lambda s: pd.Series({
        "秒": round(len(s) / kit.RATE_HZ, 1),
        "|beta|中央値": round(float(s.beta.abs().median()), 2),
        "|beta|p99": round(float(s.beta.abs().quantile(0.99)), 2),
        "|beta|最大": round(float(s.beta.abs().max()), 2),
        "|a_y|最大": round(float(s.a_y.abs().max()), 2),
        "v最大[km/h]": round(float(s.v.max()) * 3.6, 1),
    }), include_groups=False)
    print(tab.to_string())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=kit.DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=Path("out/kit_msdm"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    params = kit.load_parameters(args.root / "parameter.m")
    paths = kit.find_runs(args.root, kind="dynamic")
    print(f"dynamic run {len(paths)} 本\n")
    df = collect(paths, params)
    print(f"解析対象 {len(df):,} サンプル = {len(df) / kit.RATE_HZ / 60:.1f} 分 (v_x > {MIN_SPEED} m/s)\n")

    section_a(df)
    section_b(df)
    section_c(df, params)
    section_c2(df)
    section_d(df, params)
    section_e(df, params)
    surface_table(df)

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["run", "surface", "t", "v", "yaw", "delta", "a_y",
            "beta", "beta_cor", "ay_kin", "yaw_exp", "beta_model"]
    df[cols].to_csv(args.out / "samples.csv.gz", index=False, compression="gzip")
    print(f"\n{args.out / 'samples.csv.gz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
