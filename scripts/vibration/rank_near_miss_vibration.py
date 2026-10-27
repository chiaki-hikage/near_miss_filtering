"""既存の振動イベントを「短時間に強い上下・ロール応答」の観点で並べ直し、既存の順位と比べる。

イベントの検出はやり直さない (locate_candidates.py の candidates.csv をそのまま使う)。
足すのは、乗員が感じる強さに近い量だけである。

  絶対値     上下加速度 [m/s²]・ロール角速度 [deg/s] の最大値 (既存の A は普段比)
  同時性     上下とロールの最大の時刻の差
  集中度     上下の最大の前後 1.5 s のエネルギーのうち、前後 0.5 s に入る割合 (長く揺れる路面は小さい)
  立ち上がり 0.1 s の包絡線が最大の 25% から最大に達するまでの時間

結果は「路面凹凸により短時間に強い上下・ロール応答が発生した異常振動ヒヤリハット候補」であり、
ヒヤリハットだったとは判定しない。

    uv run python scripts/vibration/rank_near_miss_vibration.py raw_data/Chunk_1 raw_data/Chunk_2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

import find_roll_bumps as F
import locate_candidates as LC
import validate_sites_with_fullcar as V

REPO = _bootstrap.SRC.parent


def response_features(m: dict, fs: float, nm: dict) -> dict:
    """当たり付近の上下・ロール・ピッチの最大値、同時性、集中度、立ち上がり。m["t"] は候補の当たり時刻が 0。"""
    t = m["t"]
    near = (t >= -0.5) & (t <= 1.0)
    i_h = np.flatnonzero(near)[int(np.argmax(np.abs(m["heave"][near])))]
    tp = t[i_h]
    W, C = nm["window_s"], nm["core_half_s"]
    win = (t >= tp - W) & (t <= tp + W)
    core = (t >= tp - C) & (t <= tp + C)

    def conc(x):
        e = np.sum(x[win] ** 2)
        return float(np.sum(x[core] ** 2) / e) if e > 0 else np.nan

    i_r = np.flatnonzero(win)[int(np.argmax(np.abs(m["roll"][win])))]
    env = F._mrms(m["heave"], max(1, int(round(nm["envelope_s"] * fs))))
    j = i_h
    while j > 0 and env[j] > 0.25 * env[i_h] and t[j] > tp - W:
        j -= 1
    return dict(
        heave_peak_abs=float(np.abs(m["heave"][i_h])), roll_peak_abs=float(np.abs(m["roll"][i_r])),
        pitch_peak_abs=float(np.max(np.abs(m["pitch"][win]))),
        heave_roll_dt_s=float(t[i_r] - tp),
        concentration_heave=conc(m["heave"]), concentration_roll=conc(m["roll"]),
        rise_s=float(tp - t[j]),
    )


def baseline_exceedance(loader, cfg: dict, heave_th: float, roll_th: float, n_seg: int, seed: int) -> dict:
    """普段の走行 (無作為に選んだセグメント) を 1 s の窓に区切り、上下・ロールの最大値がしきい値を超える割合。

    しきい値が振動イベントの中の相対値 (分位点) なので、走行全体の中でどれくらい珍しい値かを示す。
    ジャイロが実質 104 Hz のセグメントだけ使い、5 m/s 未満の窓は除く。
    """
    rng = np.random.default_rng(seed)
    refs = [r for rs in loader.refs.values() for r in rs if F.gyro_rate_hz(r.path) >= cfg["min_gyro_hz"]]
    pick = rng.choice(len(refs), size=min(n_seg, len(refs)), replace=False)
    fs = cfg["grid_hz"]
    n = h = r_ = both = 0
    for k in pick:
        ref = refs[k]
        drive = [x for x in loader.refs[ref.drive_id]]
        if ref.drive_id not in loader.rot:
            loader.rot[ref.drive_id] = F.drive_rotation([x.path for x in drive])
        sg = F.load_seg(ref, loader.rot[ref.drive_id], cfg)
        w = int(fs)
        trim = int(F.EDGE_TRIM_S * fs)
        for a in range(trim, len(sg.t) - trim - w, w):
            if np.nanmean(sg.v[a:a + w]) < cfg["min_speed_mps"]:
                continue
            hv = np.max(np.abs(sg.bp["heave"][a:a + w])) >= heave_th
            rl = np.max(np.abs(sg.bp["roll"][a:a + w])) >= roll_th
            n += 1
            h += hv
            r_ += rl
            both += hv and rl
    hours = n / 3600
    return dict(segments=int(len(pick)), windows_1s=int(n), hours=round(hours, 2),
                heave_exceed_share=round(h / max(n, 1), 4), roll_exceed_share=round(r_ / max(n, 1), 4),
                both_exceed_share=round(both / max(n, 1), 5), both_exceed_per_hour=round(both / max(hours, 1e-9), 1))


def plot_wave(m: dict, e, out: Path, title: str) -> None:
    """代表イベントの前後の上下・ロール・ピッチ・4 輪の輪速 (帯域通過後)。"""
    import matplotlib.pyplot as plt

    t = m["t"]
    w = (t >= -2.0) & (t <= 2.5)
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(t[w], m["heave"][w], color="tab:blue", lw=0.9)
    axs[0].set_ylabel("上下加速度 [m/s²]\n(0.7〜20 Hz)")
    axs[1].plot(t[w], m["roll"][w], color="tab:green", lw=0.9)
    axs[1].set_ylabel("ロール角速度 [deg/s]\n(正 = 右下がり)")
    axs[2].plot(t[w], m["pitch"][w], color="tab:purple", lw=0.9)
    axs[2].set_ylabel("ピッチ角速度 [deg/s]\n(正 = 機首上げ)")
    for k, c, ls in (("FL", "tab:blue", "--"), ("RL", "tab:blue", "-"), ("FR", "tab:red", "--"), ("RR", "tab:red", "-")):
        axs[3].plot(t[w], m[k][w] * 3.6, color=c, ls=ls, lw=0.8, label=k)
    axs[3].set_ylabel("輪速の揺れ [km/h]\n(2〜15 Hz)")
    axs[3].legend(fontsize=8, ncol=4, loc="upper right")
    for ax in axs:
        ax.axvline(0, color="0.4", lw=0.8)
        ax.axvspan(-0.5, 0.5, color="gold", alpha=0.12)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("前輪が当たった推定時刻からの時間 [s]  (黄 = 集中度を測る前後 0.5 s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    plt.close(fig)


def export_media(top: pd.DataFrame, ev: pd.DataFrame, loader, nm: dict, out: Path) -> pd.DataFrame:
    """上位地点ごとに、代表イベントの前後の動画・波形と、前方映像のフレームを出す。

    フレームは locate_candidates.plot_frames (前輪が当たる地点が 12 m・6 m 先に見えるもの)。
    代表が夜間なら、同じ地点の昼間のイベントのフレームも出す (locate_candidates.daytime_event)。
    """
    import shutil
    from types import SimpleNamespace

    import matplotlib

    import score_longtail as L
    from near_miss.plotting import configure_japanese_font

    matplotlib.use("Agg")
    configure_japanese_font()
    loc = yaml.safe_load((REPO / "configs/vibration_locate.yaml").read_text())
    pav = yaml.safe_load((REPO / loc["pavement_config"]).read_text())
    cam = dict(yaml.safe_load((REPO / loc["camera_fit"]).read_text()))
    cam["focal_px"] = pav["camera"]["focal_px"]
    track = yaml.safe_load((REPO / "configs/vibration_fullcar.yaml").read_text())["vehicle"]["track_front_m"]["value"]
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    T = nm["clip_s"]

    def abs_hit(e):
        refs = sorted(loader.refs[e.drive], key=lambda r: r.index)
        ref = next(r for r in refs if r.index == e.segment)
        t0 = float(np.load(ref.path / "processed_log/IMU/accelerometer/t").ravel()[0])
        return refs, t0 + float(e.event_time_s)

    rows = []
    for s_ in top.itertuples():
        g = ev[ev.site_id == s_.site_id].copy()
        rep = g[g.event_id == s_.event_id].iloc[0]
        g["event_unix"] = pd.to_datetime(g.event_utc).astype("int64") / 1e9
        day = LC.daytime_event(g, loc, rep)
        stem = f"site{s_.site_id:03d}"
        refs, t_hit = abs_hit(rep)
        run = SimpleNamespace(t=np.array([t_hit]), refs=refs)
        L.clip_case(run, t_hit - T, t_hit + T, out / f"{stem}_clip_{rep.event_id}.mp4")
        LC.plot_frames(run, pd.Series(dict(i_hit=0, speed_kmh=rep.speed_kmh)), cam, track, loc,
                       out / f"{stem}_frames_{rep.event_id}.png")
        day_id = ""
        if day.event_id != rep.event_id:
            refs_d, t_d = abs_hit(day)
            run_d = SimpleNamespace(t=np.array([t_d]), refs=refs_d)
            LC.plot_frames(run_d, pd.Series(dict(i_hit=0, speed_kmh=day.speed_kmh)), cam, track, loc,
                           out / f"{stem}_frames_{day.event_id}_day.png")
            L.clip_case(run_d, t_d - T, t_d + T, out / f"{stem}_clip_{day.event_id}_day.mp4")
            day_id = day.event_id
        m = loader.load(rep.drive, int(rep.segment), float(rep.event_time_s), (-2.0, 2.5))
        plot_wave(m, rep, out / f"{stem}_wave_{rep.event_id}.png",
                  f"地点 {s_.site_id}  {rep.event_id}  {rep.drive.split('|')[1]} seg{rep.segment} {rep.event_time_s:.2f} s  "
                  f"{rep.speed_kmh:.0f} km/h  上下 {rep.heave_peak_abs:.1f} m/s²・ロール {rep.roll_peak_abs:.1f} deg/s  "
                  f"集中度 {rep.concentration:.2f}  点 {rep.near_miss_candidate_score:.1f}")
        rows.append(dict(site_id=s_.site_id, rep_event=rep.event_id, day_event=day_id,
                         local_time=(pd.to_datetime(rep.event_utc) + pd.Timedelta(hours=loc["output"]["local_utc_offset_h"])).strftime("%Y-%m-%d %H:%M")))
    return pd.DataFrame(rows)


def write_review_page(top: pd.DataFrame, media: pd.DataFrame, out: Path, rel: str) -> None:
    yes = lambda b: "○" if bool(b) else "—"
    md = media.set_index("site_id")
    # ISO 2631-1 の評価量 (iso2631_events.py の出力があれば)
    iso_path = out.parent / "iso2631_events.csv"
    iso = pd.read_csv(iso_path).set_index("event_id") if iso_path.exists() else None
    iso_txt = (lambda e: f"{iso.loc[e].mtvv_total:.2f} ({iso.loc[e].comfort}・普段の走行の {iso.loc[e].mtvv_total_normal_pct:.1f} パーセンタイル)"
               if iso is not None and e in iso.index else "—")
    L_ = ["# 異常振動ヒヤリハット候補 上位 20 地点 (確認用)", "",
          "「路面凹凸により短時間に強い上下・ロール応答が発生した異常振動ヒヤリハット候補」であり、ヒヤリハットとは判定していない。"
          "映像の判定は暫定 (2 枚の静止画)。方法と表の見方は docs/vibration_near_miss_candidates.md。", "",
          f"上位 {len(top)} 地点 = 点数の上位 20 地点 + 5 条件をすべて満たすイベントのある地点。5 条件の列は地点のイベントのうち満たす数。", "",
          "| 順位 | 地点 | 5 条件 | 代表 | 現地時刻 | 車速 | 上下 [m/s²] | ロール [deg/s] | 集中度 | 点 | 強上下 | 強ロール | 同時 | 急峻 | 路面 | 繰り返し | 映像 (暫定) | ISO 2631-1 MTVV [m/s²] | 動画 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in top.itertuples():
        m = md.loc[s.site_id]
        clip = f"[{s.event_id}]({rel}/site{s.site_id:03d}_clip_{s.event_id}.mp4)"
        if m.day_event:
            clip += f" / [昼 {m.day_event}]({rel}/site{s.site_id:03d}_clip_{m.day_event}_day.mp4)"
        L_.append(f"| {s.site_rank} | {s.site_id} | {s.site_events_all_five}/{s.site_events} | {s.event_id} | {m.local_time} | {s.speed_kmh:.0f} | {s.heave_peak_abs:.2f} | "
                  f"{s.roll_peak_abs:.2f} | {s.concentration:.2f} | {s.near_miss_candidate_score:.1f} | {yes(s.strong_vertical)} | "
                  f"{yes(s.strong_roll)} | {yes(s.simultaneous_vertical_roll)} | {yes(s.impulsive_response)} | "
                  f"{yes(s.road_related_evidence)} | {s.repeatability} | {s.video_road_feature} | {iso_txt(s.event_id)} | {clip} |")
    L_.append("")
    for s in top.itertuples():
        m = md.loc[s.site_id]
        stem = f"{rel}/site{s.site_id:03d}"
        L_ += [f"## {s.site_rank}. 地点 {s.site_id}  ({s.lat:.6f}, {s.lon:.6f})", "",
               f"- 代表 {s.event_id}: {s.drive.split('|')[1]} seg{s.segment} {s.event_time_s:.2f} s (現地 {m.local_time})、{s.speed_kmh:.0f} km/h",
               f"- 上下 {s.heave_peak_abs:.2f} m/s²・ロール {s.roll_peak_abs:.2f} deg/s・ピッチ {s.pitch_peak_abs:.2f} deg/s、"
               f"時刻差 {s.heave_roll_dt_s:+.2f} s、集中度 {s.concentration:.2f}、立ち上がり {s.rise_s:.2f} s",
               f"- 地点のイベント {s.site_events} 件 (5 条件を満たすもの {s.site_events_all_five} 件)、{s.repeatability}",
               f"- 映像 (暫定): {s.video_road_feature}",
               f"- ISO 2631-1 (IMU 位置、座面の重み付け): MTVV {iso_txt(s.event_id)}"
               + (f"、VDV_z {iso.loc[s.event_id].vdv_z:.2f} m/s^1.75、波高率 {iso.loc[s.event_id].crest_z:.1f}、"
                  f"8 Hz 以上を除くと {iso.loc[s.event_id].mtvv_total_lp8:.2f} ({iso.loc[s.event_id].comfort_lp8})"
                  if iso is not None and s.event_id in iso.index else ""),
               f"- [Google Maps]({s.google_maps})",
               f"- 動画 (前後 3 秒): [{s.event_id}]({stem}_clip_{s.event_id}.mp4)"
               + (f" / [昼間の通過 {m.day_event}]({stem}_clip_{m.day_event}_day.mp4)" if m.day_event else ""), "",
               f"![波形]({stem}_wave_{s.event_id}.png)", "",
               f"![映像 {s.event_id}]({stem}_frames_{s.event_id}.png)", ""]
        if m.day_event:
            L_ += [f"昼間の通過 {m.day_event}:", "", f"![映像 {m.day_event}]({stem}_frames_{m.day_event}_day.png)", ""]
    out.write_text("\n".join(L_))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_near_miss.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/near_miss")
    ap.add_argument("--media", action="store_true", help="上位地点の動画・波形・前方映像のフレームと確認用の一覧 (top20.md) を出す")
    args = ap.parse_args()
    nm = yaml.safe_load(args.config.read_text())
    cfg = yaml.safe_load((REPO / nm["base_config"]).read_text())
    fs = cfg["grid_hz"]
    args.out.mkdir(parents=True, exist_ok=True)

    ld = REPO / nm["locate_dir"]
    cand = pd.read_csv(ld / "candidates.csv")
    sites = pd.read_csv(ld / "sites.csv").set_index("site_id")
    top_sites = pd.read_csv(ld / "top20.csv").set_index("site_id")
    side = pd.read_csv(REPO / nm["physics_dir"] / "all_events_side_index.csv").set_index("event_id")
    loader = V.MeasuredLoader(args.roots, cfg)

    rows = []
    for _, e in cand.iterrows():
        try:
            m = loader.load(e.drive, int(e.segment), float(e.event_time_s), (-2.0, 2.5))
        except (KeyError, ValueError, IndexError):
            continue
        rows.append(dict(event_id=e.event_id, **response_features(m, fs, nm)))
    ev = cand.merge(pd.DataFrame(rows), on="event_id")

    # 判定 (列名はご依頼のとおり)
    pct = lambda s: s.rank(pct=True)
    ev["heave_pct"], ev["roll_pct"] = pct(ev.heave_peak_abs), pct(ev.roll_peak_abs)
    sp = nm["strong_pct"]
    ev["vibration_event"] = True                      # 既存の振動イベント (A ≥ 3) であること
    ev["strong_vertical"] = ev.heave_pct >= sp
    ev["strong_roll"] = ev.roll_pct >= sp
    si = nm["simultaneous"]
    ev["simultaneous_vertical_roll"] = ((ev.heave_pct >= si["min_pct"]) & (ev.roll_pct >= si["min_pct"])
                                        & (ev.heave_roll_dt_s.abs() <= si["max_dt_s"]))
    im = nm["impulsive"]
    ev["concentration"] = ev[["concentration_heave", "concentration_roll"]].min(axis=1)
    ev["impulsive_response"] = (ev.concentration >= im["min_concentration"]) & (ev.rise_s <= im["max_rise_s"])
    ev["eleven_hz_only"] = ev.wheelhop_only
    ev["road_related_evidence"] = ((ev.wheel_metric >= nm["road"]["wheel_ratio_min"]) & ~ev.wheelhop_only & ~ev.spike_like)
    ev["side_index"] = ev.event_id.map(side.side_index)
    s = sites.reindex(ev.site_id)
    ev["repeatability"] = np.where(s.flag_C.to_numpy(), "再現 (偶然なら 1% 未満)",
                                   np.where(s.n_event_drives.to_numpy() >= 2, "2 ドライブ以上でイベント", "1 回のみ"))
    # 映像の目視確認は位置と向きで地点に対応付ける (locate_candidates.attach_review と同じ。番号は設定で振り直されるため)
    loc = yaml.safe_load((REPO / "configs/vibration_locate.yaml").read_text())
    frame = LC.LocalFrame(*loc["position"]["ref_latlon"])
    rv = LC.attach_review(sites.reset_index(), REPO / loc["output"]["review_csv"], frame, loc).set_index("site_id")
    ev["video_road_feature"] = ev.site_id.map(rv.d_video).replace("", np.nan).fillna("未確認")
    ev["site_rank_locate"] = ev.site_id.map(sites["rank"])
    ev["feature_type"] = ev.site_id.map(sites.feature_type)
    core = np.sqrt(ev.heave_pct * ev.roll_pct)
    factor = (np.where(ev.eleven_hz_only, nm["wheelhop_factor"], 1.0)
              * np.where(ev.wheel_metric >= nm["road"]["wheel_ratio_min"], 1.0, nm["no_wheel_factor"]))
    ev["near_miss_candidate_score"] = (100 * core * ev.concentration.clip(0, 1) * factor).round(1)
    ev["A_rank"] = ev.A.rank(ascending=False, method="first").astype(int)
    ev = ev.sort_values("near_miss_candidate_score", ascending=False).reset_index(drop=True)
    ev.insert(0, "rank", range(1, len(ev) + 1))
    ev["label"] = "路面凹凸により短時間に強い上下・ロール応答が発生した異常振動ヒヤリハット候補"

    cols = ["rank", "event_id", "label", "drive", "segment", "event_time_s", "event_utc", "lat", "lon", "speed_kmh",
            "site_id", "site_rank_locate", "feature_type",
            "vibration_event", "strong_vertical", "strong_roll", "simultaneous_vertical_roll", "impulsive_response",
            "road_related_evidence", "repeatability", "video_road_feature", "near_miss_candidate_score",
            "heave_peak_abs", "roll_peak_abs", "pitch_peak_abs", "heave_roll_dt_s", "concentration", "rise_s",
            "A", "A_rank", "heave_ratio", "roll_metric", "wheel_metric", "wheel_lr_log2", "side_index",
            "eleven_hz_only", "spike_like", "brake_any", "google_maps"]
    ev["google_maps"] = [f"https://www.google.com/maps/search/?api=1&query={a:.6f},{b:.6f}" for a, b in zip(ev.lat, ev.lon)]
    ev[cols].round(3).to_csv(args.out / "near_miss_vibration_events.csv", index=False)
    ev[cols].head(nm["top_n"]).round(3).to_csv(args.out / "near_miss_vibration_top20.csv", index=False)

    # 地点ごと (同じ地点の複数イベントで一覧が埋まらないように、各地点の最高点のイベントを代表にする)
    g = ev.groupby("site_id")
    st = ev.drop_duplicates("site_id").copy()      # ev は点数の降順なので先頭が各地点の最高点
    st["site_events"] = st.site_id.map(g.size())
    st["site_events_all_five"] = st.site_id.map(
        g.apply(lambda d: int((d.strong_vertical & d.strong_roll & d.simultaneous_vertical_roll
                               & d.impulsive_response & d.road_related_evidence).sum())))
    st["site_score_median"] = st.site_id.map(g.near_miss_candidate_score.median()).round(1)
    st.insert(0, "site_rank", range(1, len(st) + 1))
    scols = ["site_rank", "site_id", "site_rank_locate", "site_events", "site_events_all_five", "site_score_median"] + \
        [c for c in cols if c not in ("rank", "site_id", "site_rank_locate")]
    st[scols].round(3).to_csv(args.out / "near_miss_vibration_sites.csv", index=False)
    st[scols].head(nm["top_n"]).round(3).to_csv(args.out / "near_miss_vibration_top20_sites.csv", index=False)

    if args.media:
        # 上位 20 地点に加え、5 条件をすべて満たすイベントのある地点 (異常振動ヒヤリハット候補) も出す
        show = st[(st.site_rank <= nm["top_n"]) | (st.site_events_all_five > 0)]
        media = export_media(show, ev, loader, nm, args.out / "media")
        write_review_page(show, media, args.out / "top20.md", "media")

    # 既存の順位との比較
    cmp = {}
    for n in (20, 50, 100):
        a = set(ev.nsmallest(n, "A_rank").event_id)
        b = set(ev.head(n).event_id)
        cmp[f"top{n}_overlap_events"] = len(a & b)
    cmp["spearman_A_vs_score"] = round(float(spearmanr(ev.A, ev.near_miss_candidate_score).correlation), 3)
    cmp["spearman_A_vs_heave_abs"] = round(float(spearmanr(ev.A, ev.heave_peak_abs).correlation), 3)
    cmp["spearman_A_vs_roll_abs"] = round(float(spearmanr(ev.A, ev.roll_peak_abs).correlation), 3)
    top_new_sites = list(dict.fromkeys(ev.site_id))[:20]
    cmp["top20_sites_overlap_with_locate_top20"] = len(set(top_new_sites) & set(top_sites.index))
    a_sites = list(dict.fromkeys(ev.sort_values("A", ascending=False).site_id))[:20]
    cmp["top20_sites_overlap_with_A_top20_sites"] = len(set(top_new_sites) & set(a_sites))
    cmp["flags_share"] = {k: round(float(ev[k].mean()), 3) for k in
                          ("strong_vertical", "strong_roll", "simultaneous_vertical_roll", "impulsive_response",
                           "road_related_evidence", "eleven_hz_only")}
    allf = ev.strong_vertical & ev.strong_roll & ev.simultaneous_vertical_roll & ev.impulsive_response & ev.road_related_evidence
    cmp["all_five_flags_events"] = int(allf.sum())
    cmp["all_five_flags_sites"] = int(ev[allf].site_id.nunique())
    cmp["thresholds"] = dict(strong_vertical_mps2=round(float(ev.heave_peak_abs.quantile(nm["strong_pct"])), 2),
                             strong_roll_dps=round(float(ev.roll_peak_abs.quantile(nm["strong_pct"])), 2))
    cmp["candidate_funnel"] = {
        "振動イベント (A ≥ 3)": len(ev),
        "強い上下": int(ev.strong_vertical.sum()),
        "+ 強いロール": int((ev.strong_vertical & ev.strong_roll).sum()),
        "+ 同時": int((ev.strong_vertical & ev.strong_roll & ev.simultaneous_vertical_roll).sum()),
        "+ 急峻": int((ev.strong_vertical & ev.strong_roll & ev.simultaneous_vertical_roll & ev.impulsive_response).sum()),
        "+ 輪速の裏付け (= 候補)": int(allf.sum()),
    }
    be = nm["baseline"]
    cmp["baseline_exceedance"] = baseline_exceedance(loader, cfg, cmp["thresholds"]["strong_vertical_mps2"],
                                                     cmp["thresholds"]["strong_roll_dps"], be["segments"], be["seed"])
    meta = dict(config=str(args.config.relative_to(REPO)),
                config_sha1=hashlib.sha1(args.config.read_bytes()).hexdigest()[:10],
                events=len(ev), comparison=cmp)
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=int))
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
