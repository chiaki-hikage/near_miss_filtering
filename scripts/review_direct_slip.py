#!/usr/bin/env python3
"""直接証拠で拾った事象を 1 件ずつ読み直し、**制御介入と車両挙動が整合するか**を測る。

screen_sideslip.py の direct_slip.csv はピーク時刻の 1 点しか持たない。
「ABS が立ったのに減速していない」「slip_warn が立ったのに何も動いていない」
といった食い違いはピーク 1 点では分からないので、窓の中の波形を読み直して数える。

判定は**証拠の種類ごとに期待される挙動**を、測れる量だけで確かめる。
閾値は対照窓 (走行中の無作為な時刻 10.3 万件、4 車種) の分位から取る。

  ABS      -> 減速しているか (ax か ブレーキスイッチ) /
              輪が滑ったか (輪単体の瞬間減速度 <= -8 m/s^2 が主。輪速の開きは従)
  slip_warn -> 走行中か / 加速側なら輪速が開いているか
  VSC      -> 横加速度が対照 p99 を超えるか
  共通     -> ay の 2 系統 (CAN の横 G と 車速 x ヨー) が整合するか
              物理量のピークがフラグの前後 1 秒以内にあるか

  uv run python scripts/review_direct_slip.py \\
      --dir out/sideslip_rav4_tss2_full=TOYOTA_RAV4_TSS2 \\
      --dir out/sideslip_TOYOTA_COROLLA_TSS2_full=TOYOTA_COROLLA_TSS2 \\
      --grade multi_evidence --grade high_evidence -w 8 --out out/review_direct_slip

出力
  reviewed.csv  1 行 1 事象。窓の中の測定値と、判定ごとの合否と総合判定
  summary.json  車種別・判定別の件数
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.features import compute_features
from near_miss.io import comma_car_segments as ccs
from near_miss.signals import to_grid

# 対照窓の分位から取った閾値 (scripts/rank_slip.py と同じ出どころ)
AY_CTRL_P99 = 3.9          # |ay| の対照 p99 [m/s^2]
DECEL_MIN = 1.5            # ABS に伴う減速とみなす下限 [m/s^2]
WS_ABS_MIN = 1.0           # ABS に伴う輪速の開きの下限 [m/s]
WHEEL_DECEL_MIN = -8.0     # ABS に伴う輪単体の減速の下限 [m/s^2]。対照窓の 2% しか届かない
WS_SPIN_MIN = 2.0          # 空転とみなす輪速の開きの下限 [m/s]
AY_AGREE_MAX = 1.5         # ay の 2 系統の差の上限 (中央値) [m/s^2]
LAG_MAX_S = 1.0            # 物理量のピークとフラグのずれの上限 [s]
PAD_S = 2.0                # 窓の前後に足す余白 [s]

VERDICT_OK = "整合"
VERDICT_CHECK = "要確認"
VERDICT_BAD = "不整合"

_S: dict = {}


def init(platform: str, vdir: Path, cfg: dict, cache: Path) -> None:
    _S["vehicle"] = find_vehicle_config_for_platform(platform, load_vehicle_configs(vdir))
    _S["cfg"] = cfg
    # 1 事象ごとにキャッシュを走査すると遅いので、所在の索引を一度だけ作る。
    _S["index"] = {(r.drive_id, r.index): r for r in ccs.find_segments(cache, platform)}


def _absmax(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.abs(x).max()) if x.size else float("nan")


def _peak_lag(t: np.ndarray, x: np.ndarray, t0: float) -> float:
    """|x| が最大になった時刻 - フラグの立ち上がり。負ならフラグより先に出た。"""
    x = np.abs(x)
    ok = np.isfinite(x)
    if not ok.any():
        return float("nan")
    return float(t[ok][int(np.argmax(x[ok]))] - t0)


def measure(ev: dict) -> dict:
    """1 事象を読み直して窓の中を測る。読めなければ error に理由を残す。"""
    out = dict(ev)
    veh, cfg = _S["vehicle"], _S["cfg"]
    ref = _S["index"].get((ev["drive_id"], int(ev["segment"])))
    if ref is None:
        out["error"] = "セグメントが手元にない"
        return out
    try:
        sd = ccs.load_segment(ref, veh, with_raw_can=True, with_radar=False)
        gs = to_grid(sd, cfg)
        gs = compute_features(gs, cfg, radar=None, vehicle=veh)
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    df = gs.df
    if df.empty:
        out["error"] = "グリッドが空"
        return out

    t = df["t"].to_numpy()
    # 時刻はルート先頭からの通し秒。direct_slip.csv の t_in_segment_s は
    # セグメント先頭からの相対秒なので、セグメントの開始時刻を足して合わせる。
    t0 = float(sd.t_span[0]) + float(ev["t_in_segment_s"])
    t1 = t0 + float(ev["duration_s"])
    m = (t >= t0 - PAD_S) & (t <= t1 + PAD_S)
    if not m.any():
        out["error"] = "窓が範囲外"
        return out
    w = df[m]
    out["window_truncated"] = bool(t0 - PAD_S < t[0] or t1 + PAD_S > t[-1])
    out["n_samples"] = int(m.sum())

    def col(c) -> np.ndarray:
        return w[c].to_numpy() if c in w else np.array([])

    v, ax = col("v_mps"), col("ax_mps2")
    ay_kin, ay_can = col("ay_kin_mps2"), col("ay_can_mps2")
    yaw, steer = col("yaw_rate_dps"), col("steer_deg_s")
    ws = col("ws_spread_mps")
    out["v_min_mps"] = float(np.nanmin(v)) if v.size else np.nan
    # フラグが立った時点の車速。窓の最小値だと、停止まで踏み切った ABS が
    # 「停車中に立った」ことになってしまう。
    tw = w["t"].to_numpy()
    out["v_at_flag_mps"] = float(v[int(np.argmin(np.abs(tw - t0)))]) if v.size else np.nan
    out["v_max_win_mps"] = float(np.nanmax(v)) if v.size else np.nan
    out["ax_min_mps2"] = float(np.nanmin(ax)) if ax.size else np.nan
    out["ax_max_mps2"] = float(np.nanmax(ax)) if ax.size else np.nan
    out["ay_kin_absmax"] = _absmax(ay_kin)
    out["ay_can_absmax"] = _absmax(ay_can)
    out["yaw_absmax_dps"] = _absmax(yaw)
    out["steer_absmax_deg"] = _absmax(steer)
    out["steer_rate_absmax_dps"] = _absmax(col("steer_rate_dps"))
    out["ws_spread_max_mps"] = float(np.nanmax(ws)) if ws.size else np.nan
    out["brake_pressed_any"] = bool(np.nanmax(col("brake_pressed")) > 0) if "brake_pressed" in w else None
    # 運転者ではなく別の制御 (ACC / openpilot) がブレーキを掛けている区間かどうか。
    # ABS が立っても輪速が開かない事象の説明になりうるので、文脈として残す。
    for c in ("op_engaged", "cruise_active", "acc_braking"):
        out[f"{c}_any"] = bool(np.nanmax(col(c)) > 0) if c in w and col(c).size else None
    # ay の 2 系統の差。片方の符号や倍率がおかしければここが開く。
    if ay_kin.size and ay_can.size:
        d = np.abs(ay_can - ay_kin)
        out["ay_diff_med_mps2"] = float(np.nanmedian(d))
    else:
        out["ay_diff_med_mps2"] = np.nan
    # 逆操舵: 舵とヨーの符号が逆 (4 車種で相関 0.94〜0.96 なので通常は同符号)
    if yaw.size and steer.size:
        big = np.isfinite(yaw) & np.isfinite(steer) & (np.abs(yaw) > 10.0) & (np.abs(steer) > 30.0)
        out["counter_steer_frac"] = float((yaw[big] * steer[big] < 0).mean()) if big.any() else 0.0
    else:
        out["counter_steer_frac"] = np.nan
    # 4 輪が同時にロック・解放を繰り返す ABS の作動は、輪と輪の差 (ws_spread) には
    # ほとんど出ない。輪単体の減速が車体の減速より深いかを見る。
    wheels = [col(c) for c in ("ws_fl_mps", "ws_fr_mps", "ws_rl_mps", "ws_rr_mps")]
    wheels = [x for x in wheels if x.size]
    if wheels and tw.size > 1:
        dt = float(np.median(np.diff(tw)))
        dec = [np.nanmin(np.gradient(x, dt)) for x in wheels]
        out["wheel_decel_min_mps2"] = float(np.nanmin(dec))
        out["wheel_vs_body_decel"] = (out["wheel_decel_min_mps2"] / out["ax_min_mps2"]
                                      if np.isfinite(out["ax_min_mps2"]) and out["ax_min_mps2"] < -0.1
                                      else np.nan)
    else:
        out["wheel_decel_min_mps2"] = out["wheel_vs_body_decel"] = np.nan
    out["ws_peak_lag_s"] = _peak_lag(w["t"].to_numpy(), ws, t0) if ws.size else np.nan
    out["ay_peak_lag_s"] = _peak_lag(w["t"].to_numpy(), ay_kin, t0) if ay_kin.size else np.nan

    out.update(judge(out))
    return out


def judge(r: dict) -> dict:
    """証拠の種類ごとに期待される挙動を確かめる。該当しない判定は None。"""
    checks: dict[str, bool | None] = {}
    decel = (r["ax_min_mps2"] <= -DECEL_MIN) if np.isfinite(r["ax_min_mps2"]) else None
    braking = r.get("brake_pressed_any")

    if r.get("abs_active"):
        checks["ABS_減速"] = bool(decel) or bool(braking)
        # ABS の裏付けは**輪単体の減速が主**。4 輪が同時にロックと解放を
        # 繰り返すと輪と輪の差には出ないので、輪速の開きだけを課すと
        # 実測 66 件中 35 件が不合格になってしまう。
        deep = (r["wheel_decel_min_mps2"] <= WHEEL_DECEL_MIN
                if np.isfinite(r.get("wheel_decel_min_mps2", np.nan)) else None)
        spread = (r["ws_spread_max_mps"] >= WS_ABS_MIN
                  if np.isfinite(r["ws_spread_max_mps"]) else None)
        checks["ABS_輪の挙動"] = (None if deep is None and spread is None
                                   else bool(deep) or bool(spread))
    if r.get("slip_warn"):
        checks["作動_走行中"] = (r["v_at_flag_mps"] >= 2.0
                                 if np.isfinite(r["v_at_flag_mps"]) else None)
        if np.isfinite(r["ax_max_mps2"]) and r["ax_max_mps2"] > 1.0 and not r.get("abs_active"):
            # 加速側で立ったなら駆動輪の空転のはず。輪速が開いていなければ食い違い。
            checks["空転_輪速"] = (r["ws_spread_max_mps"] >= WS_SPIN_MIN
                                   if np.isfinite(r["ws_spread_max_mps"]) else None)
    if r.get("vsc_active"):
        checks["VSC_横G"] = (r["ay_kin_absmax"] >= AY_CTRL_P99
                             if np.isfinite(r["ay_kin_absmax"]) else None)
    if np.isfinite(r.get("ay_diff_med_mps2", np.nan)):
        checks["ay_2系統"] = r["ay_diff_med_mps2"] <= AY_AGREE_MAX
    lag = r.get("ws_peak_lag_s", np.nan)
    if r.get("wheel_spread") and np.isfinite(lag):
        # ピークが作動区間の中 (前後 1 秒の余裕つき) にあるか。事象が長いときに
        # 「立ち上がりから 1 秒以内」を課すと、区間の後半のピークを落としてしまう。
        checks["輪速_時間整合"] = -LAG_MAX_S <= lag <= float(r["duration_s"]) + LAG_MAX_S

    ng = [k for k, v in checks.items() if v is False]
    verdict = VERDICT_OK if not ng else (VERDICT_CHECK if len(ng) == 1 else VERDICT_BAD)
    return {**{f"chk_{k}": v for k, v in checks.items()},
            "n_checks": len([v for v in checks.values() if v is not None]),
            "n_failed": len(ng), "failed": "|".join(ng), "verdict": verdict}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", action="append", required=True, metavar="DIR=PLATFORM",
                   help="screen_sideslip.py の出力と車種キー。複数指定できる")
    p.add_argument("--grade", action="append", default=None,
                   help="対象の等級 (既定 multi_evidence + high_evidence)。複数指定できる。all で全件")
    p.add_argument("-w", "--workers", type=int, default=4)
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/review_direct_slip"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    grades = args.grade or ["multi_evidence", "high_evidence"]

    by_platform: dict[str, list[dict]] = {}
    for spec in args.dir:
        d, _, platform = spec.partition("=")
        if not platform:
            raise SystemExit(f"--dir は DIR=PLATFORM の形で指定してください: {spec}")
        f = pd.read_csv(Path(d) / "direct_slip.csv")
        if "all" not in grades:
            f = f[f.grade.isin(grades)]
        f = f.assign(platform=platform, source=d)
        by_platform.setdefault(platform, []).extend(f.to_dict("records"))

    rows: list[dict] = []
    for platform, events in by_platform.items():
        print(f"{platform}: {len(events)} 件を読み直します", flush=True)
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init,
                                     initargs=(platform, args.vehicles, cfg, args.cache)) as ex:
                rows.extend(ex.map(measure, events, chunksize=1))
        else:
            init(platform, args.vehicles, cfg, args.cache)
            rows.extend(measure(e) for e in events)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "reviewed.csv", index=False)

    pd.set_option("display.width", 250)
    err = df[df.get("error").notna()] if "error" in df else pd.DataFrame()
    if len(err):
        print(f"\n読めなかった事象 {len(err)} 件")
        print(err[["platform", "drive_id", "segment", "error"]].to_string(index=False))
    if "verdict" not in df.columns:
        raise SystemExit("判定できた事象がありません (上の理由を確認してください)")
    ok = df[df.verdict.notna()]
    print("\n■ 総合判定 x 車種")
    print(pd.crosstab(ok.verdict, ok.platform, margins=True).to_string())
    print("\n■ 判定ごとの合否 (True / False / 該当なし)")
    for c in sorted(c for c in ok.columns if c.startswith("chk_")):
        s = ok[c]
        print(f"  {c[4:]:<14}合 {int((s == True).sum()):>4}   否 {int((s == False).sum()):>4}"  # noqa: E712
              f"   該当なし {int(s.isna().sum()):>4}")
    if (ok.n_failed > 0).any():
        print("\n■ 不合格を含む事象")
        cols = ["platform", "grade", "evidence", "drive_id", "segment", "t_in_segment_s",
                "duration_s", "failed", "verdict", "v_min_mps", "ax_min_mps2",
                "ay_kin_absmax", "ws_spread_max_mps", "brake_pressed_any"]
        print(ok[ok.n_failed > 0][[c for c in cols if c in ok.columns]]
              .round(2).to_string(index=False))

    summary = {
        "grades": grades,
        "n_events": int(len(df)),
        "n_error": int(len(err)),
        "verdict": ok.verdict.value_counts().to_dict() if "verdict" in ok else {},
        "by_platform": {p: ok[ok.platform == p].verdict.value_counts().to_dict()
                        for p in ok.platform.unique()} if "verdict" in ok else {},
        "criteria": {"AY_CTRL_P99": AY_CTRL_P99, "DECEL_MIN": DECEL_MIN,
                     "WS_ABS_MIN": WS_ABS_MIN, "WS_SPIN_MIN": WS_SPIN_MIN,
                     "WHEEL_DECEL_MIN": WHEEL_DECEL_MIN,
                     "AY_AGREE_MAX": AY_AGREE_MAX, "LAG_MAX_S": LAG_MAX_S,
                     "PAD_S": PAD_S},
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
