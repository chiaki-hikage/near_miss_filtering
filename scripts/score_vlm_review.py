#!/usr/bin/env python3
"""VLM の判定を人手ラベルと突き合わせて採点する (Phase 1)。

GPU は要らない。EC2 で推論した結果 (JSONL) を持ち帰って Mac で回せる。

Phase 1 は**探索的 PoC** であり、性能の確定はしない。
  - 指標は必ず件数を併記する。百分率だけを出さない
  - 信頼区間は参考値として出す。結論の根拠にしない
  - negative は clip / episode 単位を主指標にする。約 900 時刻を
    独立標本として扱わない (同一クリップ内は強く相関する)
  - positive 8 件は集約値だけでなくイベント別の個票を残す
  - 時間評価の基準は人手の t_onset_human。CAN 由来の t_start ではない

使い方:
  uv run python scripts/score_vlm_review.py --results out/chunk1/vlm/results_qwen2_5_vl_7b.jsonl
  uv run python scripts/score_vlm_review.py --results ... --out out/chunk1/vlm/score
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from near_miss.config import load_yaml
from near_miss.vlm.schema import validate
from near_miss.vlm.scoring import (
    Series,
    alarm_runs,
    alarm_time,
    bootstrap_clip,
    cohen_kappa,
    duration_ratio,
    fmt_count,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, nargs="+", required=True,
                   help="推論結果の JSONL。複数渡すとモデル間で対応比較する")
    p.add_argument("--labels", type=Path, default=Path("out/chunk1/labels.csv"))
    p.add_argument("--onset", type=Path, default=Path("out/chunk1/vlm/labels_onset.csv"))
    p.add_argument("--config", type=Path, default=Path("configs/vlm.yaml"))
    p.add_argument("--out", type=Path, default=Path("out/chunk1/vlm/score"))
    p.add_argument("--brief", action="store_true",
                   help="貼り付けられる短い要約だけを出す (結果を持ち出せない環境向け)")
    p.add_argument("--examples", type=int, default=0,
                   help="失敗事例の説明文を N 件ずつ出す。**持ち出さず現地で読む**")
    return p.parse_args()


def load_results(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in Path(p).open(encoding="utf-8"):
            r = json.loads(line)
            resp = r.get("response") or {}
            rows.append({
                "model": r.get("model", Path(p).stem),
                "request_id": r["request_id"], "event_id": r["event_id"],
                "mode": r["mode"], "condition": r.get("condition", "C"),
                "rep": int(r.get("rep", 0)),
                "t_eval": r.get("t_eval"), "t_rel": r.get("t_rel"),
                "n_frames": r.get("n_frames"), "partial_window": bool(r.get("partial_window", False)),
                "state": resp.get("state"), "risky": resp.get("risky"),
                "risk_level": resp.get("risk_level"), "hazard_type": resp.get("hazard_type"),
                "evidence": resp.get("evidence"),
                "insufficient": resp.get("insufficient_information"),
                "confidence": resp.get("confidence"),
                "n_schema_errors": len(r.get("schema_errors")
                                       or validate(resp, "clip" if r["mode"] == "clip" else "online")),
                "ok": bool(resp),
                "config_hash": r.get("config_hash"),
                "prompt_version": r.get("prompt_version"),
                "guard_s": r.get("guard_s"),
                # 説明文。持ち出せない環境では EC2 側で読む (--examples)
                "scene": resp.get("scene"), "ego_behavior": resp.get("ego_behavior"),
                "difference": resp.get("difference_from_normal"),
                "evidence_detail": resp.get("evidence_detail"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def score_mode_a(res: pd.DataFrame, truth: dict[str, bool], out: Path) -> pd.DataFrame:
    """一括判定。A/B/C を主比較、D は別表 (既存フィルタのヒントを与えた条件)。"""
    a = res[res["mode"] == "clip"].copy()
    if a.empty:
        return pd.DataFrame()
    a["truth"] = a["event_id"].map(truth)

    rows = []
    for (model, cond), g in a.groupby(["model", "condition"]):
        g0 = g[g["rep"] == g["rep"].min()]           # 反復 0 を代表にする
        g0 = g0.dropna(subset=["risky", "truth"])
        if g0.empty:
            continue
        pred = g0["risky"].astype(bool).to_numpy()
        tru = g0["truth"].astype(bool).to_numpy()
        tp = int((pred & tru).sum()); fn = int((~pred & tru).sum())
        fp = int((pred & ~tru).sum()); tn = int((~pred & ~tru).sum())
        k, se = cohen_kappa(tru, pred)
        # 反復間の一致 (temperature=0 なら再現性確認であって自己一致率ではない)
        rep = g.pivot_table(index="event_id", columns="rep", values="risky", aggfunc="first")
        stable = int((rep.nunique(axis=1, dropna=True) <= 1).sum()) if rep.shape[1] > 1 else len(rep)
        rows.append({
            "model": model, "condition": cond, "n": len(g0),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "recall_k": tp, "recall_n": tp + fn,
            "falsealarm_k": fp, "falsealarm_n": fp + tn,
            "kappa": round(k, 3), "kappa_se": round(se, 3),
            "schema_err": int(g0["n_schema_errors"].sum()),
            "rep_stable_k": stable, "rep_stable_n": len(rep),
            "evidence_video": int((g0["evidence"] == "video").sum()),
            "evidence_both": int((g0["evidence"] == "both").sum()),
            "evidence_can": int((g0["evidence"] == "can").sum()),
            "insufficient": int(g0["insufficient"].fillna(False).astype(bool).sum()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "mode_a.csv", index=False)

    print("\n" + "=" * 74)
    print("モード A (一括判定)  ※ 主比較は A/B/C。D は別表")
    print("=" * 74)
    for model, g in df.groupby("model"):
        print(f"\n[{model}]")
        main = g[g.condition.isin(["A", "B", "C"])].sort_values("condition")
        for _, r in main.iterrows():
            print(f"  条件 {r.condition}  再現率 {fmt_count(r.recall_k, r.recall_n)}"
                  f"   誤警報 {fmt_count(r.falsealarm_k, r.falsealarm_n)}"
                  f"   κ {r.kappa:+.3f} [参考 SE {r.kappa_se:.3f}]")
        if len(main) == 3:
            c = main[main.condition == "C"].iloc[0]
            b = main[main.condition == "B"].iloc[0]
            aa = main[main.condition == "A"].iloc[0]
            print(f"\n  CAN 追加の効果 (C - B): 再現率 {c.recall_k - b.recall_k:+d} 件 / "
                  f"誤警報 {c.falsealarm_k - b.falsealarm_k:+d} 件 / κ {c.kappa - b.kappa:+.3f}")
            print(f"  映像追加の効果 (C - A): 再現率 {c.recall_k - aa.recall_k:+d} 件 / "
                  f"誤警報 {c.falsealarm_k - aa.falsealarm_k:+d} 件 / κ {c.kappa - aa.kappa:+.3f}")
        d = g[g.condition == "D"]
        if not d.empty:
            r = d.iloc[0]
            print(f"\n  [別表] 条件 D (既存フィルタのヒントあり。独立性能ではない)")
            print(f"    再現率 {fmt_count(r.recall_k, r.recall_n)}"
                  f"   誤警報 {fmt_count(r.falsealarm_k, r.falsealarm_n)}")
        r = main.iloc[-1] if len(main) else g.iloc[0]
        print(f"\n  根拠の内訳 (条件 C): 映像 {r.evidence_video} / 両方 {r.evidence_both}"
              f" / CAN {r.evidence_can} / 判断不能 {r.insufficient}")
        print(f"  schema 違反 {int(g.schema_err.sum())} 件"
              f" / 反復間で判定が一致 {fmt_count(int(r.rep_stable_k), int(r.rep_stable_n), ci=False)}"
              "  ※ temperature=0 なら再現性確認")
    return df


# ---------------------------------------------------------------------------
def build_series(res: pd.DataFrame, truth: dict[str, bool]) -> dict[tuple[str, str], Series]:
    out: dict[tuple[str, str], Series] = {}
    b = res[(res["mode"] == "online") & res["state"].notna()]
    for (model, eid), g in b.groupby(["model", "event_id"]):
        s = Series(event_id=eid, risky=bool(truth.get(eid, False)))
        for _, r in g.iterrows():
            s.t.append(float(r["t_eval"]))
            s.state.append(str(r["state"]))
            s.partial.append(bool(r["partial_window"]))
        out[(model, eid)] = s.sort()
    return out


def score_negative(series, cfg, out: Path) -> pd.DataFrame:
    """negative は clip / episode 単位が主指標。時刻単位は補助。"""
    states = cfg["alarm"]["states"]
    deb = int(cfg["alarm"]["debounce_steps"])
    stride = float(cfg["timeline"]["stride_s"])

    rows = []
    for (model, eid), s in series.items():
        if s.risky:
            continue
        runs = alarm_runs(s, states, deb)
        ratio = duration_ratio(s, states)
        rows.append({
            "model": model, "event_id": eid, "n_points": s.n,
            "span_s": round(s.span_s + stride, 2),
            # 「全時間 normal を維持」は文字どおり全時刻。デバウンスをかけない。
            # かけると単発のちらつきを持つクリップまで無警報に数えてしまう。
            "clean": ratio == 0.0,
            # エピソード数のほうはデバウンスをかける。1 時刻のちらつきを
            # 警報 1 件と数えると、判定の揺れがそのまま件数になる。
            "n_episodes": len(runs),
            # ゲートはエピソード基準で見る。運用上、単発 0.5 秒のちらつきは
            # 警報として扱われない。clean (全時刻 normal) は記述指標として残す。
            "episode_clean": len(runs) == 0,
            "flicker_only": ratio > 0.0 and len(runs) == 0,
            "duration_ratio": round(duration_ratio(s, states), 4),
            "first_alarm_t": alarm_time(s, states, deb),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(out / "mode_b_negative.csv", index=False)

    print("\n" + "=" * 74)
    print("モード B / negative  ※ 主指標は clip / episode 単位")
    print("=" * 74)
    for model, g in df.groupby("model"):
        minutes = g["span_s"].sum() / 60.0
        clean = int(g["clean"].sum())
        ep = int(g["n_episodes"].sum())
        ratios = g["duration_ratio"].to_numpy()
        lo, hi = bootstrap_clip(ratios)
        print(f"\n[{model}]  negative {len(g)} クリップ / 計 {minutes:.1f} 分")
        flick = int(g["flicker_only"].sum())
        print(f"  全時間 normal を維持       : {fmt_count(clean, len(g), ci=False)}"
              "  ※ 全時刻が normal。デバウンス無し")
        print(f"  誤警報が発生したクリップ   : {fmt_count(len(g) - clean, len(g), ci=False)}")
        print(f"    うち単発のちらつきのみ   : {flick} 件"
              f"  (デバウンス {deb} を満たさずエピソードに数えない)")
        print(f"  エピソードが無いクリップ   : "
              f"{fmt_count(int(g['episode_clean'].sum()), len(g), ci=False)}"
              "  ← ゲート 4 はこちらを見る")
        print(f"  誤警報エピソード           : {ep} 件"
              f"  (毎分 {ep / minutes:.2f} ※分母 {minutes:.1f} 分なので "
              f"{1 / minutes:.2f} 刻みでしか動かない)")
        print(f"  誤警報の時間比             : 中央 {np.median(ratios):.1%} / "
              f"最大 {ratios.max():.1%}  [参考 平均の CI {lo:.1%}-{hi:.1%}]")
        pts = int((g["duration_ratio"] * g["n_points"]).sum())
        print(f"  [補助] 時刻単位の誤警報率  : {pts}/{int(g['n_points'].sum())} "
              f"({pts / g['n_points'].sum():.1%}) ※独立標本ではない")
        worst = g.sort_values(["n_episodes", "duration_ratio"], ascending=False).head(3)
        if not worst.empty and worst["duration_ratio"].iloc[0] > 0:
            print("  誤警報の目立つクリップ:")
            for _, r in worst.iterrows():
                if r.duration_ratio > 0:
                    tag = " (ちらつきのみ)" if r.flicker_only else ""
                    print(f"    {r.event_id}  エピソード {r.n_episodes} 件"
                          f" / 時間比 {r.duration_ratio:.1%}{tag}")
    return df


def score_positive(series, onset: pd.DataFrame, cfg, out: Path) -> pd.DataFrame:
    """positive はイベント別の個票を残す。n=8 で平均を語らない。"""
    states = cfg["alarm"]["states"]
    deb = int(cfg["alarm"]["debounce_steps"])
    o = onset.set_index("event_id")

    rows = []
    for (model, eid), s in series.items():
        if not s.risky or eid not in o.index:
            continue
        r = o.loc[eid]
        t_alarm = alarm_time(s, states, deb)
        t_on = float(r["t_onset_human"]) if pd.notna(r.get("t_onset_human")) else np.nan
        t_ap = float(r["t_apparent_human"]) if pd.notna(r.get("t_apparent_human")) else np.nan
        floor = float(r.get("latency_floor_s", 0.0) or 0.0)
        rows.append({
            "model": model, "event_id": eid, "onset_cue": r.get("onset_cue"),
            "note": r.get("note"),
            "detected": t_alarm is not None,
            "t_alarm": t_alarm,
            "delta_onset_s": round(t_alarm - t_on, 2) if t_alarm is not None else np.nan,
            "delta_apparent_s": round(t_alarm - t_ap, 2) if t_alarm is not None else np.nan,
            "latency_floor_s": floor,
            "filter_lag_s": r.get("filter_lag_s"),
            "n_partial": int(sum(s.partial)),
            "n_points": s.n,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(out / "mode_b_positive.csv", index=False)

    print("\n" + "=" * 74)
    print("モード B / positive  ※ 基準は人手の t_onset_human。個票を残す")
    print("=" * 74)
    for model, g in df.groupby("model"):
        print(f"\n[{model}]")
        cols = ["event_id", "onset_cue", "detected", "delta_onset_s",
                "delta_apparent_s", "latency_floor_s", "filter_lag_s", "n_partial"]
        print(g[cols].to_string(index=False))
        det = g[g.detected]
        print(f"\n  検出できた: {fmt_count(int(g.detected.sum()), len(g), ci=False)}")
        if not det.empty:
            early = det[det.delta_onset_s <= det.latency_floor_s + 1e-9]
            print(f"  人手のオンセット以前に警報: {fmt_count(len(early), len(det), ci=False)}"
                  "  ※ 床に達しているものを含む")
            print(f"  delta_onset の分布 [秒]: "
                  + " / ".join(f"{v:+.1f}" for v in sorted(det.delta_onset_s)))
    return df


# ---------------------------------------------------------------------------
def gates(a: pd.DataFrame, neg: pd.DataFrame, pos: pd.DataFrame, cfg) -> None:
    print("\n" + "=" * 74)
    print("Phase 1 のゲート条件")
    print("=" * 74)
    print("  これは**探索的な最低条件**であって性能目標ではない。")
    print("  positive 8 件 / negative 24 件では性能を確定できない。")
    print("  満たせば Phase 2 (未ラベル 39 件と hard negative の人手ラベル) へ進む、")
    print("  という判断にのみ使う。数値を目標として最適化しないこと。")
    for model in sorted(set(a.model) | set(neg.model) | set(pos.model)):
        ga = a[a.model == model]; gn = neg[neg.model == model]; gp = pos[pos.model == model]
        res = []
        err = int(ga.schema_err.sum()) if not ga.empty else 0
        n_a = int(ga.n.sum()) if not ga.empty else 0
        res.append(("1 schema 適合 >95%", err == 0 or (n_a and 1 - err / n_a > 0.95),
                    f"違反 {err}/{n_a}"))
        if not ga.empty:
            c = ga[ga.condition == "C"]
            degen = bool(len(c) and (c.iloc[0].TP + c.iloc[0].FP == 0
                                     or c.iloc[0].FN + c.iloc[0].TN == 0))
            res.append(("2 明白な退化なし", not degen, "全件同一判定" if degen else "なし"))
            st = int(ga.rep_stable_k.iloc[0]); sn = int(ga.rep_stable_n.iloc[0])
            res.append(("3 反復で一致", sn == 0 or st / sn >= 0.8, f"{st}/{sn}"))
        if not gn.empty:
            ep_clean = int(gn.episode_clean.sum())
            strict = int(gn.clean.sum())
            med = float(np.median(gn.duration_ratio))
            # 基準は n_episodes == 0。運用上、単発 0.5 秒のちらつきは警報として
            # 成立しないため。strict clean と時間比は併記して判断材料にする。
            res.append(("4 誤警報エピソード無しが >=12/24 かつ 時間比中央 <20%",
                        ep_clean >= 12 and med < 0.20,
                        f"エピソード無し {ep_clean}/{len(gn)} / 時間比中央 {med:.1%} "
                        f"/ 全時刻 normal {strict}/{len(gn)}"))
        if not gp.empty:
            # 映像にしか根拠が無い類型。CAN 側では short_thw / hard_brake としか
            # 見えないので、ここで先行できるかが VLM の付加価値そのもの。
            cut = gp[gp.onset_cue.isin(["cut_in", "crossing"])]
            hit = cut[cut.detected & (cut.delta_onset_s <= cut.latency_floor_s + 1.0)]
            res.append(("5 先行/同時検知の兆候 (映像由来 2 件以上)",
                        len(hit) >= 2, f"{len(hit)}/{len(cut)} 件"))
        print(f"\n[{model}]")
        for name, ok, detail in res:
            print(f"  {'OK  ' if ok else '未達'} {name}  ({detail})")


def brief(res: pd.DataFrame, a: pd.DataFrame, neg: pd.DataFrame,
          pos: pd.DataFrame, cfg, out: Path) -> None:
    """判断に要る数字だけを 40 行程度にまとめる。

    結果ファイルを持ち出せない環境では、この出力だけを手で運ぶ。
    数値は件数で持つ (割合だけでは元に戻せない)。
    """
    L: list[str] = []
    ch = sorted(map(str, res.config_hash.dropna().unique())) if "config_hash" in res else []
    pv = sorted(map(str, res.prompt_version.dropna().unique())) if "prompt_version" in res else []
    L.append("=== VLM PoC Phase1 digest ===")
    L.append(f"config_hash={','.join(ch) or '-'} prompt={','.join(pv) or cfg['prompt_version']} "
             f"guard={cfg['context']['guard_s']} stride={cfg['timeline']['stride_s']} "
             f"debounce={cfg['alarm']['debounce_steps']}")
    L.append(f"n_results={len(res)} models={','.join(sorted(res.model.unique()))}")

    if not a.empty:
        L.append("")
        L.append("[A] model cond TP FN FP TN kappa schema_err rep_stable")
        for _, r in a.sort_values(["model", "condition"]).iterrows():
            L.append(f"  {r.model} {r.condition} {r.TP} {r.FN} {r.FP} {r.TN} "
                     f"{r.kappa:+.3f} {r.schema_err} {r.rep_stable_k}/{r.rep_stable_n}")
        for model, g in a.groupby("model"):
            m = g[g.condition.isin(["A", "B", "C"])].set_index("condition")
            if len(m) == 3:
                c, b, aa = m.loc["C"], m.loc["B"], m.loc["A"]
                L.append(f"  {model} delta C-B: TP{c.TP - b.TP:+d} FP{c.FP - b.FP:+d} "
                         f"kappa{c.kappa - b.kappa:+.3f}")
                L.append(f"  {model} delta C-A: TP{c.TP - aa.TP:+d} FP{c.FP - aa.FP:+d} "
                         f"kappa{c.kappa - aa.kappa:+.3f}")
            if "C" in m.index:
                c = m.loc["C"]
                L.append(f"  {model} evidence(C) video/both/can/insuf="
                         f"{c.evidence_video}/{c.evidence_both}/{c.evidence_can}/{c.insufficient}")

    if not neg.empty:
        L.append("")
        L.append("[B-neg] model clean ep_clean episodes ratio_med ratio_max minutes")
        for model, g in neg.groupby("model"):
            L.append(f"  {model} {int(g.clean.sum())}/{len(g)} "
                     f"{int(g.episode_clean.sum())}/{len(g)} {int(g.n_episodes.sum())} "
                     f"{np.median(g.duration_ratio):.3f} {g.duration_ratio.max():.3f} "
                     f"{g.span_s.sum() / 60:.1f}")

    if not pos.empty:
        L.append("")
        L.append("[B-pos] model event cue det d_onset d_apparent floor partial")
        for _, r in pos.sort_values(["model", "event_id"]).iterrows():
            f = lambda v: "-" if pd.isna(v) else f"{v:+.2f}"
            L.append(f"  {r.model} {r.event_id} {r.onset_cue} "
                     f"{'Y' if r.detected else 'N'} {f(r.delta_onset_s)} "
                     f"{f(r.delta_apparent_s)} {r.latency_floor_s:.2f} {r.n_partial}")

    text = "\n".join(L)
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "digest.txt").write_text(text + "\n", encoding="utf-8")


def show_examples(res: pd.DataFrame, truth: dict[str, bool], n: int) -> None:
    """外した事例の説明文を出す。**持ち出さず、この画面で読む。**

    数値は digest で運べるが、説明の質と幻覚の有無は読まないと分からない。
    読んだ結果は件数だけを記録して運ぶ (下の記入欄)。
    """
    a = res[(res["mode"] == "clip") & (res["condition"] == "C")].copy()
    a["truth"] = a["event_id"].map(truth)
    a = a[a["rep"] == a["rep"].min()].dropna(subset=["risky", "truth"])

    print("\n" + "=" * 74)
    print("失敗事例の説明文  ※ ここで読む。ファイルは持ち出さない")
    print("=" * 74)
    for model, g in a.groupby("model"):
        fn = g[(~g.risky.astype(bool)) & (g.truth.astype(bool))].head(n)
        fp = g[(g.risky.astype(bool)) & (~g.truth.astype(bool))].head(n)
        for tag, sub in (("見落とし (人手 risky / VLM normal)", fn),
                         ("誤検出 (人手 normal / VLM risky)", fp)):
            print(f"\n[{model}] {tag}: {len(sub)} 件")
            for _, r in sub.iterrows():
                print(f"  --- {r.event_id}  state={r.state} hazard={r.hazard_type} "
                      f"evidence={r.evidence} conf={r.confidence}")
                for k, label in (("scene", "情景"), ("ego_behavior", "自車"),
                                 ("difference", "通常との違い"),
                                 ("evidence_detail", "根拠")):
                    v = r.get(k)
                    if isinstance(v, str) and v.strip():
                        print(f"      {label}: {v}")

    print("\n" + "-" * 74)
    print("読んだ結果はこの形で記録して運ぶ (数値だけなので短い):")
    print("  説明の妥当性: 妥当 _ / 部分的 _ / 誤り _   (対象 32 件)")
    print("  幻覚 (映像に無い対象への言及): _ 件")
    print("  所見: (1 行)")
    print("-" * 74)


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    labels = pd.read_csv(args.labels)
    labels["risky"] = labels["risky"].astype(str) == "True"

    from near_miss.vlm.windows import episodes_from_labels
    truth = {e.event_id: e.risky for e in episodes_from_labels(labels, cfg)}
    onset = pd.read_csv(args.onset) if args.onset.is_file() else pd.DataFrame()

    res = load_results(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"結果 {len(res)} 件 / モデル {sorted(res.model.unique())}")
    bad = int((~res.ok).sum())
    if bad:
        print(f"  応答が空: {bad} 件")

    import contextlib, io
    sink = io.StringIO() if args.brief else None
    with contextlib.redirect_stdout(sink) if args.brief else contextlib.nullcontext():
        a = score_mode_a(res, truth, args.out)
        series = build_series(res, truth)
        neg = score_negative(series, cfg, args.out)
        pos = (score_positive(series, onset, cfg, args.out)
               if not onset.empty else pd.DataFrame())

    if args.examples:
        show_examples(res, truth, args.examples)
    if args.brief:
        brief(res, a, neg, pos, cfg, args.out)
    if not a.empty or not neg.empty:
        gates(a, neg, pos, cfg)
    print(f"\n出力: {args.out}"
          + ("  (digest.txt を貼り付けてください)" if args.brief else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
