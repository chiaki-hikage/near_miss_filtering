#!/usr/bin/env python3
"""validate_slip_flags.py の結果を車種ごとに並べ、借用したビットの意味が再現するかを判定する。

先頭の車種 (既定 TOYOTA_RAV4_TSS2) を基準にして、他の車種で同じ振る舞いが出るかを見る。
**件数が多いか少ないかは判定に使わない。** 見るのは値ごとの「出方」と、
物理量との整合だけ。

判定の基準 (CRITERIA) は RAV4 TSS2 の実測に合わせて置いたもので、検出の閾値ではない。
基準車種自身がこれを満たさなければ、基準の置き方の方が間違っている。

  uv run python scripts/report/report_slip_flag_validation.py \\
      TOYOTA_RAV4_TSS2 TOYOTA_COROLLA_TSS2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

# 判定の基準。RAV4 TSS2 の 189.8 時間で観測した振る舞いを言葉にしたもの。
CRITERIA = {
    # slip_warn = 3 は始動時のランプ試験: ルート先頭かつ停車中に出る
    "sw3_route_start_min": 0.80,   # ルート先頭 (segment 0, t < 1 s) の割合の下限
    "sw3_stopped_min": 0.90,       # 開始時の車速 < 1 m/s の割合の下限
    # slip_warn = 4 は走行中の作動: 動いていて、短く、滑りの物理量を伴う
    "sw4_moving_min": 0.90,        # 開始時の車速 > 2 m/s の割合の下限
    "sw4_dur_p95_max": 2.5,        # 持続時間の 95% 点の上限 [s]
    "sw4_physics_min": 0.50,       # 輪速差か |ax| か |ay| が対照の 95% 点を超える割合の下限
    # ABS 作動は制動中に出る
    "abs_braking_min": 0.80,       # 窓内でブレーキスイッチが立っている割合の下限
    "abs_decel_min": 0.50,         # ax 最小が対照の 5% 点を下回る割合の下限
}


def load(root: Path, plat: str) -> dict:
    d = root / plat
    return {
        "runs": pd.read_csv(d / "runs.csv") if (d / "runs.csv").stat().st_size > 1 else pd.DataFrame(),
        "ctrl": pd.read_csv(d / "control.csv"),
        "values": pd.read_csv(d / "values.csv"),
        "meta": json.loads((d / "sanity.json").read_text(encoding="utf-8")),
    }


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0%}"


def num(x, f="{:.2f}") -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


def physics_exceed(runs: pd.DataFrame, ctrl: pd.DataFrame) -> pd.Series:
    """輪速差・|ax|・|ay| のどれかが、対照窓の 95% 点を超えたか。"""
    ws95 = ctrl.ws_spread_max.quantile(0.95)
    axn5 = ctrl.ax_min.quantile(0.05)
    axp95 = ctrl.ax_max.quantile(0.95)
    ay95 = ctrl.ay_absmax.quantile(0.95)
    return ((runs.ws_spread_max > ws95) | (runs.ax_min < axn5)
            | (runs.ax_max > axp95) | (runs.ay_absmax > ay95))


def describe(data: dict) -> dict:
    """1 車種ぶんの要約。値ごとの出方と、物理量との整合。"""
    r, c = data["runs"], data["ctrl"]
    out: dict = {"hours": data["meta"]["hours"], "segments": data["meta"]["segments"]}
    v = data["values"]
    sw = v[v.flag == "slip_warn"].copy()
    sw = sw[sw.value != "欠落"]
    sw["value"] = sw.value.astype(float)
    tot = sw.samples.sum()
    out["sw_values"] = {int(k): int(n) for k, n in sw.groupby("value").samples.sum().items()}
    out["sw_samples"] = int(tot)
    out["ctrl"] = {"n": len(c), "ws95": c.ws_spread_max.quantile(0.95),
                   "ax5": c.ax_min.quantile(0.05), "ay95": c.ay_absmax.quantile(0.95),
                   "yaw95": c.yaw_absmax.quantile(0.95)}
    if r.empty:
        return out

    def group(mask, name):
        g = r[mask]
        if g.empty:
            out[name] = {"n": 0}
            return
        ex = physics_exceed(g, c)
        out[name] = {
            "n": len(g),
            "route_start": g.at_route_start.mean(),
            "stopped": (g.v_at_start < 1.0).mean(),
            "moving": (g.v_at_start > 2.0).mean(),
            "v_med": g.v_at_start.median(),
            "dur_med": g.duration_s.median(),
            "dur_p95": g.duration_s.quantile(0.95),
            "dur_max": g.duration_s.max(),
            "with_tc": (g.tc_disabled_max > 0).mean(),
            "with_abs": (g.abs_active_flag_max > 0).mean(),
            "with_vsc": (g.vsc_active_flag_max > 0).mean(),
            "with_sw4": (g.slip_warn_max == 4).mean(),
            "braking": (g.brake_pressed_any > 0).mean(),
            "ws_med": g.ws_spread_max.median(),
            "ws_over": (g.ws_spread_max > out["ctrl"]["ws95"]).mean(),
            "ax_min_med": g.ax_min.median(),
            "decel_over": (g.ax_min < out["ctrl"]["ax5"]).mean(),
            "ay_med": g.ay_absmax.median(),
            "yaw_med": g.yaw_absmax.median(),
            "physics": ex.mean(),
            "ws_lag_med": g.ws_spread_lag_s.median(),
            "ax_lag_med": g.ax_min_lag_s.median(),
            "by_wmi": g.vin_wmi.value_counts().to_dict(),
        }

    for val in sorted(r[r.flag == "slip_warn"].value.unique()):
        group((r.flag == "slip_warn") & (r.value == val), f"sw{int(val)}")
    group(r.flag == "abs_active_flag", "abs")
    group(r.flag == "vsc_active_flag", "vsc")
    group(r.flag == "tc_disabled", "tc")
    return out


def verdict(s: dict) -> list[tuple[str, bool | None, str]]:
    C = CRITERIA
    res = []
    sw3, sw4, ab = s.get("sw3", {"n": 0}), s.get("sw4", {"n": 0}), s.get("abs", {"n": 0})
    if sw3["n"]:
        res.append(("slip_warn=3 はルート先頭に出る", sw3["route_start"] >= C["sw3_route_start_min"],
                    f"{pct(sw3['route_start'])} (基準 {pct(C['sw3_route_start_min'])} 以上)"))
        res.append(("slip_warn=3 は停車中に出る", sw3["stopped"] >= C["sw3_stopped_min"],
                    f"{pct(sw3['stopped'])} (基準 {pct(C['sw3_stopped_min'])} 以上)"))
    else:
        res.append(("slip_warn=3 が出る", None, "0 件"))
    if sw4["n"]:
        res.append(("slip_warn=4 は走行中に出る", sw4["moving"] >= C["sw4_moving_min"],
                    f"{pct(sw4['moving'])} (基準 {pct(C['sw4_moving_min'])} 以上)"))
        res.append(("slip_warn=4 は短い", sw4["dur_p95"] <= C["sw4_dur_p95_max"],
                    f"p95 {num(sw4['dur_p95'])} 秒 (基準 {C['sw4_dur_p95_max']} 秒以下)"))
        res.append(("slip_warn=4 は滑りの物理量を伴う", sw4["physics"] >= C["sw4_physics_min"],
                    f"{pct(sw4['physics'])} (基準 {pct(C['sw4_physics_min'])} 以上)"))
    else:
        res.append(("slip_warn=4 が出る", None, "0 件"))
    if ab["n"]:
        res.append(("ABS は制動中に出る", ab["braking"] >= C["abs_braking_min"],
                    f"{pct(ab['braking'])} (基準 {pct(C['abs_braking_min'])} 以上)"))
        res.append(("ABS は強い減速を伴う", ab["decel_over"] >= C["abs_decel_min"],
                    f"{pct(ab['decel_over'])} (基準 {pct(C['abs_decel_min'])} 以上)"))
    else:
        res.append(("ABS が出る", None, "0 件"))
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("platforms", nargs="+", help="先頭が基準")
    p.add_argument("--root", type=Path, default=Path("out/validate_slip_flags"))
    p.add_argument("--out", type=Path, default=None, help="要約 JSON の書き出し先")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    S = {p: describe(load(args.root, p)) for p in args.platforms}
    plats = args.platforms
    w = 22

    def row(label, f):
        print(f"  {label:<34}" + "".join(f"{f(S[p]):>{w}}" for p in plats))

    print(" " * 36 + "".join(f"{p[-w + 1:]:>{w}}" for p in plats))
    row("収録 [h] / セグメント", lambda s: f"{s['hours']} / {s['segments']}")

    print("\n■ slip_warn の値の分布 (サンプル数)")
    vals = sorted({k for s in S.values() for k in s["sw_values"]})
    for v in vals:
        row(f"値 {v}", lambda s, v=v: f"{s['sw_values'].get(v, 0):,} "
            f"({s['sw_values'].get(v, 0) / max(s['sw_samples'], 1):.4%})")

    def block(title, key, fields):
        print(f"\n■ {title}")
        for label, f in fields:
            row(label, lambda s, f=f: f(s[key]) if s.get(key, {}).get("n") else "-")

    common = [
        ("区間数", lambda g: f"{g['n']}"),
        ("ルート先頭 (seg0, t<1s)", lambda g: pct(g["route_start"])),
        ("開始時 停車 (v<1)", lambda g: pct(g["stopped"])),
        ("開始時 走行 (v>2)", lambda g: pct(g["moving"])),
        ("開始時車速 中央 [m/s]", lambda g: num(g["v_med"], "{:.1f}")),
        ("持続 中央 / p95 / 最大 [s]",
         lambda g: f"{g['dur_med']:.2f}/{g['dur_p95']:.2f}/{g['dur_max']:.1f}"),
        ("tc_disabled と同時", lambda g: pct(g["with_tc"])),
        ("ABS と同時 (±ctx)", lambda g: pct(g["with_abs"])),
        ("VSC と同時 (±ctx)", lambda g: pct(g["with_vsc"])),
        ("slip_warn=4 と同時", lambda g: pct(g["with_sw4"])),
        ("ブレーキスイッチ", lambda g: pct(g["braking"])),
        ("輪速差 中央 [m/s]", lambda g: num(g["ws_med"])),
        ("輪速差 > 対照 p95", lambda g: pct(g["ws_over"])),
        ("ax 最小 中央 [m/s²]", lambda g: num(g["ax_min_med"])),
        ("ax 最小 < 対照 p5", lambda g: pct(g["decel_over"])),
        ("|ay| 中央 / |yaw| 中央", lambda g: f"{g['ay_med']:.2f} / {g['yaw_med']:.1f}"),
        ("物理量のどれかが対照外", lambda g: pct(g["physics"])),
        ("輪速差ピークの遅れ 中央 [s]", lambda g: num(g["ws_lag_med"])),
        ("ax 最小の遅れ 中央 [s]", lambda g: num(g["ax_lag_med"])),
    ]
    for key in sorted({k for s in S.values() for k in s if k.startswith("sw") and k[2:].isdigit()},
                      key=lambda k: int(k[2:])):
        block(f"slip_warn = {key[2:]}", key, common)
    block("ABS 作動 (abs_active_flag)", "abs", common)
    block("VSC 作動 (vsc_active_flag)", "vsc", common)
    block("TC 解除 (tc_disabled)", "tc", common)

    print("\n■ 対照窓 (走行中の無作為な時刻、±ctx)")
    row("窓数", lambda s: f"{s['ctrl']['n']:,}")
    row("輪速差 p95 [m/s]", lambda s: num(s["ctrl"]["ws95"]))
    row("ax 最小 p5 [m/s²]", lambda s: num(s["ctrl"]["ax5"]))
    row("|ay| p95 / |yaw| p95", lambda s: f"{s['ctrl']['ay95']:.2f} / {s['ctrl']['yaw95']:.1f}")

    print("\n■ 基本信号の整合 (原点を通る傾き / 相関)")
    names = {"speed_ratio": "表示車速 vs 輪速平均 (0x0B4/0x0AA)",
             "ax_dvdt": "ax 1秒平均 vs Δv (0x024)",
             "ay_vyaw": "ay vs v·yaw (0x024)",
             "yaw_steer": "yaw vs 舵角 (符号の向き)"}
    for k, label in names.items():
        def f(s, k=k):
            m = load(args.root, [p for p in plats if S[p] is s][0])["meta"]["sanity"].get(k, {})
            return f"{m.get('slope', float('nan')):.3f} / {m.get('corr', float('nan')):.3f}" if "slope" in m else "-"
        row(label, f)

    print("\n■ 判定 (RAV4 TSS2 の振る舞いが再現するか)")
    for p in plats:
        print(f"\n  {p}")
        for label, ok, detail in verdict(S[p]):
            mark = "—" if ok is None else ("○" if ok else "×")
            print(f"    {mark} {label:<30} {detail}")

    for p in plats[1:]:
        for key in ("sw3", "sw4", "abs"):
            g = S[p].get(key, {})
            if g.get("n"):
                print(f"\n  {p} {key} の VIN 製造者コード別の区間数: {g['by_wmi']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
