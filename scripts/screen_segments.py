#!/usr/bin/env python3
"""commaCarSegments を数百〜数千セグメント規模でスクリーニングする。

取得と処理を重ねて流す。1 セグメントの処理は 0.35 秒程度 (実測) で、
所要時間はほぼ回線速度で決まる。ディスクを増やしたくない場合は
--discard-cache を付けると、処理の済んだ rlog をその場で消す。

途中で止まっても、既に処理したセグメントは segments.csv に記録されているので
--resume で続きから流せる。

  # まず見積りだけ (取得はしない)
  python scripts/screen_segments.py TOYOTA_RAV4_TSS2 --limit 500 --dry-run

  # 実行
  python scripts/screen_segments.py TOYOTA_RAV4_TSS2 --limit 500 --out out/screen_rav4_tss2

  # 中断後の再開
  python scripts/screen_segments.py TOYOTA_RAV4_TSS2 --limit 500 --out out/screen_rav4_tss2 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    config_hash,
    find_vehicle_config_for_platform,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.io import comma_car_segments as ccs
from near_miss.pipeline import _annotate_segment, process_block, split_contiguous
from near_miss.scoring import candidates_to_frame, events_to_frame
from near_miss.sources import SegmentSource

log = logging.getLogger(__name__)

TYPICAL_SEGMENT_MB = 1.38
SEGMENT_SECONDS = 60.0

# スクリーニングで注目するイベント。ここに無いものも全部数えるが、
# この並びで先に出して見落とさないようにする。
FOCUS_EVENTS = (
    "abs_active",
    "yaw_instability",
    "s_evasion",
    "brake_and_steer",
    "brake_after_evasion",
    "panic_brake_with_lead",
    "risky_lane_change",
    "closing_fast",
    "low_ttc",
    "hard_brake",
    "cut_in_candidate",
)

# 分布を出す連続量。イベントが 0 件だったときに
# 「信号が無い」のか「その挙動が無い」のかを切り分けるために使う。
CONTINUOUS = (
    ("v_mps", "車速 [m/s]"),
    ("ax_mps2", "前後加速度 [m/s^2]"),
    ("gvc_mps2", "前後 G (車両報告) [m/s^2]"),
    ("ay_kin_mps2", "横加速度 v*yaw [m/s^2]"),
    ("ay_can_mps2", "横加速度 (センサ) [m/s^2]"),
    ("yaw_rate_dps", "ヨーレート [deg/s]"),
    ("steer_rate_dps", "舵角レート [deg/s]"),
    ("net_heading_win_deg", "正味方位変化 [deg]"),
    ("yaw_residual_dps", "ヨー残差 [deg/s]"),
    ("yaw_residual_sigma", "ヨー残差 [sigma]"),
    ("beta_model_deg", "横滑り角 (単軌道モデル) [deg]"),
    ("ws_spread_mps", "輪速ばらつき (生) [m/s]"),
    ("ws_spread_smooth_mps", "輪速ばらつき (0.3s 平滑) [m/s]"),
    ("ws_spread_excess_mps", "輪速ばらつき 超過 [m/s]"),
    ("thw_s", "車間時間 [s]"),
    ("ttc_s", "衝突余裕時間 [s]"),
    ("lead_distance_m", "先行車距離 [m]"),
    ("gas_pedal_pct", "アクセル開度 [%]"),
    ("brake_position", "ブレーキ踏み込み量 [-]"),
    ("brake_pressure", "ブレーキ圧 [-]"),
    ("steer_torque_driver", "操舵トルク (運転者) [-]"),
    ("precollision_force_n", "PCS 制動要求 [N]"),
    ("brake_mc_mpa", "マスタシリンダ圧 [MPa]"),
)

# 分布ではなく「立った割合」を出すフラグ。
# 全セグメントで 0 のままなら、その条件は検証できていないと結果に残す。
FLAGS = (
    ("abs_active_flag", "ABS 作動 (0x226 ABSACT)"),
    ("vsc_active_flag", "VSC 作動 (0x226 VSCACT)"),
    ("slip_warn", "滑り警告灯"),
    ("abs_fault", "ABS 故障 (0x320 FABS)"),
    ("precollision_active", "純正 AEB (PCS) 作動"),

    ("brake_pressed", "ブレーキスイッチ"),
    ("tc_disabled", "TC 無効"),
    ("brake_hold_active", "ブレーキホールド"),
    ("acc_braking", "ACC による制動"),
    ("cruise_active", "ACC 作動"),
    ("op_engaged", "openpilot 介入"),
    ("counter_steer_active", "逆操舵 (特徴量)"),
    ("ws_anomaly_active", "輪速異常 (特徴量)"),
    ("panic_brake_pedal_active", "アクセルOFF→制動 (特徴量)"),
    ("s_evasion_active", "S 字回避 (特徴量)"),
    ("cut_in_active", "割り込み (特徴量)"),
)
AVAILABILITY = {k: v for k, v in FLAGS}


class SignalStats:
    """列ごとの値を間引いて溜め、最後に分位点にする。

    セグメントを跨いだ分位点は、セグメントごとの分位点を平均しても出ない。
    値そのものを持つ必要がある。20 Hz を stride で間引けば
    2,000 セグメントでも 1 列あたり数十万点に収まる。
    """

    def __init__(self, stride: int = 4):
        self.stride = max(1, stride)
        self.values: dict[str, list[np.ndarray]] = {}
        self.present: dict[str, int] = {}
        self.n_segments = 0

    def add(self, df: pd.DataFrame, n_segments: int = 1) -> None:
        """ブロック (連結した連続セグメント) 1 つぶんを足す。

        復号率の分母はセグメント数にする。ブロック数だと連結の仕方で変わってしまう。
        """
        self.n_segments += n_segments
        for col, _ in CONTINUOUS + FLAGS:
            if col not in df.columns:
                continue
            x = df[col].to_numpy()[:: self.stride]
            x = x[np.isfinite(x)]
            if not x.size:
                continue
            self.present[col] = self.present.get(col, 0) + n_segments
            self.values.setdefault(col, []).append(x)

    def _pool(self, col: str) -> np.ndarray:
        parts = self.values.get(col)
        return np.concatenate(parts) if parts else np.empty(0)

    # long-tail を見るための分位点。両裾を出す。
    # 危険側が負になる量 (前後加速度など) は下側、正になる量は上側に出る。
    QUANTILES = (0.01, 0.1, 1.0, 50.0, 99.0, 99.9, 99.99)

    def continuous_table(self) -> pd.DataFrame:
        rows = []
        for col, label in CONTINUOUS:
            x = self._pool(col)
            seg = self.present.get(col, 0)
            row = {"信号": label, "列": col,
                   "復号率": f"{seg / self.n_segments:.1%}" if self.n_segments else "-",
                   "n": int(x.size)}
            if x.size:
                row["min"] = round(float(x.min()), 3)
                for q in self.QUANTILES:
                    row[f"p{q:g}"] = round(float(np.percentile(x, q)), 3)
                row["max"] = round(float(x.max()), 3)
            rows.append(row)
        return pd.DataFrame(rows)

    def flag_table(self) -> pd.DataFrame:
        rows = []
        for col, label in FLAGS:
            x = self._pool(col)
            seg = self.present.get(col, 0)
            rows.append({
                "フラグ": label, "列": col,
                "復号率": f"{seg / self.n_segments:.1%}" if self.n_segments else "-",
                "立った割合": f"{float(np.mean(x > 0.5)):.4%}" if x.size else "-",
                "最大値": round(float(x.max()), 3) if x.size else np.nan,
                "n": int(x.size),
            })
        return pd.DataFrame(rows)


# 単独では優先度を上げないイベント。車間や車線変更そのものは正常な運転にも起きる。
LOW_ALONE = {"cut_in_candidate", "short_thw", "lane_change_candidate"}

# 実データ上に存在するか確かめたい、まれで緊急度の高いイベント。
# 1 つでも含めば高優先度にする。
HIGH_URGENCY = {
    "abs_active", "aeb_active", "s_evasion", "wheel_speed_anomaly",
    "yaw_instability", "brake_and_steer", "brake_after_evasion",
    "panic_brake_with_lead",
}

# 短時間に重なったときに意味を持つ組。events.csv 上で時間の重なりを数える。
PAIRS = (
    ("counter_steer", "hard_brake"),
    ("counter_steer", "yaw_instability"),
    ("s_evasion", "hard_brake"),
    ("yaw_instability", "wheel_speed_anomaly"),
    ("hard_brake", "hard_steer"),
    ("hard_brake", "cut_in_candidate"),
    ("hard_brake", "short_thw"),
    ("counter_steer", "wheel_speed_anomaly"),
    ("panic_brake_pedal", "short_thw"),
    ("yaw_instability", "hard_steer"),
    ("weaving", "hard_brake"),
    ("abs_active", "hard_brake"),
    ("aeb_active", "hard_brake"),
)
PAIR_WINDOW_S = 2.0


def _types(row) -> set[str]:
    v = row.get("event_types")
    return set(str(v).split("|")) if isinstance(v, str) and v else set()


def rank_candidates(cand: pd.DataFrame) -> pd.DataFrame:
    """候補を優先度で並べ直す。

    件数の多い割り込み・車間・車線変更が単独で上位を占めると、
    まれなイベントが埋もれる。種類の重なりを先に見る。
    """
    if cand.empty:
        return cand
    out = cand.copy()
    types = [_types(r) for _, r in out.iterrows()]
    out["n_event_types"] = [len(t) for t in types]

    def priority(t: set[str]) -> str:
        if t & HIGH_URGENCY:
            return "1_高"
        if t and t <= LOW_ALONE:
            return "3_低"
        if len(t) >= 2:
            return "2_中"
        return "3_低"

    out["priority"] = [priority(t) for t in types]
    out["pairs"] = [
        "|".join(f"{a}+{b}" for a, b in PAIRS if a in t and b in t) for t in types
    ]
    return out.sort_values(
        ["priority", "n_event_types", "severity"], ascending=[True, False, False]
    ).reset_index(drop=True)


def pair_counts(events: pd.DataFrame, window_s: float = PAIR_WINDOW_S) -> pd.DataFrame:
    """2 種類のイベントが window_s 以内に重なった回数を数える。

    候補区間の単位 (前後 2 秒を付けて 2 秒以内を統合) より厳しく、
    イベントの区間どうしが直接近いものだけを数える。
    """
    rows = []
    if events.empty:
        return pd.DataFrame(rows)
    for a, b in PAIRS:
        ea = events[events.event_type == a]
        eb = events[events.event_type == b]
        n = 0
        for _, x in ea.iterrows():
            same = eb[eb.drive_id == x.drive_id]
            if same.empty:
                continue
            near = (same.t_start - window_s < x.t_end) & (x.t_start - window_s < same.t_end)
            n += int(near.any())
        rows.append({"組み合わせ": f"{a} + {b}", "件数": n,
                     f"{a}": int(len(ea)), f"{b}": int(len(eb))})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("platform", help="車種キー (例 TOYOTA_RAV4_TSS2)")
    p.add_argument("--limit", type=int, default=200, help="スクリーニングするセグメント数")
    p.add_argument("--sample", choices=("route", "random", "head"), default="route",
                   help="route = ルート単位で連続して選ぶ (既定), random = 無作為, head = 先頭から")
    p.add_argument("--per-route", type=int, default=10, help="--sample route のときの 1 ルートあたり数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", type=Path, default=ccs.DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=Path("out/screen"))
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--workers", type=int, default=6, help="並列ダウンロード数")
    p.add_argument("--discard-cache", action="store_true", help="処理の済んだ rlog を消す")
    p.add_argument("--resume", action="store_true", help="既に処理したセグメントを飛ばす")
    p.add_argument("--dry-run", action="store_true", help="取得量の見積りだけ出す")
    p.add_argument("--stats-stride", type=int, default=4,
                   help="分布を取るときの間引き。20 Hz を 1/N にする")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def choose_names(args) -> list[str]:
    if args.sample == "route":
        # ルート数は決め打ちにしない。連番が長く続くルートから順に、
        # 1 ルートあたり per_route 本まで取り、limit に届くまでルートを足していく。
        # ルートによっては連番が数本しか続かないので、
        # limit / per_route 本のルートで打ち切ると目標数に届かない。
        return ccs.select_segments(args.platform, limit=args.limit, routes=None,
                                   per_route=args.per_route, cache_dir=args.cache)
    names = ccs.platform_segments(args.platform, args.cache)
    if args.sample == "random":
        rng = random.Random(args.seed)
        names = rng.sample(names, min(args.limit, len(names)))
        # 同一ルートを連結できるように並べ直す
        names.sort(key=lambda n: (ccs.SegmentName.parse(n).drive_id, ccs.SegmentName.parse(n).index))
        return names
    return names[: args.limit]


def prefetch(names: list[str], cache: Path, workers: int, q: "queue.Queue", stop: threading.Event) -> None:
    """取得を別スレッドで先回りさせる。失敗しても後続を止めない。"""
    from concurrent.futures import ThreadPoolExecutor

    def one(n):
        if stop.is_set():
            return n, None
        try:
            return n, ccs.ensure_segment(n, cache)
        except Exception as exc:
            return n, exc

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, result in ex.map(one, names):
            q.put((name, result))
    q.put(None)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    ccs.fetch_database(args.cache)
    cfg = load_yaml(args.config)
    vehicles = load_vehicle_configs(args.vehicles)
    vehicle = find_vehicle_config_for_platform(args.platform, vehicles)
    if vehicle is None:
        raise SystemExit(f"車種設定がありません: {args.platform}")
    cfg_hash = config_hash(cfg, [v.raw for v in vehicles])

    names = choose_names(args)
    done: set[str] = set()
    seg_path = args.out / "segments.csv"
    if args.resume and seg_path.is_file():
        prev = pd.read_csv(seg_path)
        done = set(prev.get("segment_name", pd.Series(dtype=str)).dropna())
    todo = [n for n in names if n not in done]
    have = [n for n in todo if ccs.local_path(n, args.cache).is_file()]
    fetch_n = len(todo) - len(have)

    print(f"車種           : {args.platform}")
    print(f"選択           : {len(names)} セグメント  ({len(names) * SEGMENT_SECONDS / 3600:.1f} 時間相当)")
    print(f"処理済みを除く : {len(todo)}")
    print(f"取得済み       : {len(have)}")
    print(f"新規取得       : {fetch_n}  概算 {fetch_n * TYPICAL_SEGMENT_MB / 1024:.2f} GB")
    print(f"処理時間の目安 : {len(todo) * 0.35 / 60:.1f} 分 (取得を除く)")
    print(f"キャッシュ     : {args.cache / 'segments'}" + ("  (処理後に削除)" if args.discard_cache else ""))
    print(f"出力           : {args.out}")
    if args.dry_run:
        print("\n--dry-run のため取得も処理もしません。")
        return 0
    if not todo:
        print("\n処理するものがありません。")
        return 0

    # 読み出し関数だけを渡す軽い供給元。セグメント一覧は流れてくるものを使うので持たない。
    source = SegmentSource(
        name=ccs.DATASET,
        refs=[],
        load=lambda ref, veh, raw: ccs.load_segment(ref, veh, with_raw_can=True),
        vehicle_for=lambda ref: vehicle,
        video_fps=None,
        supports_stage1=False,
        meta={"platform": args.platform},
    )

    args.out.mkdir(parents=True, exist_ok=True)
    q: "queue.Queue" = queue.Queue(maxsize=args.workers * 4)
    stop = threading.Event()
    t = threading.Thread(target=prefetch, args=(todo, args.cache, args.workers, q, stop), daemon=True)
    t.start()

    # ルート単位で溜めてから処理する。60 秒境界を跨ぐイベントを取り逃がさないため。
    pending: dict[str, list[str]] = {}
    seg_rows: list[dict] = []
    ev_frames: list[pd.DataFrame] = []
    cand_frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    n_seen = 0
    stats = SignalStats(args.stats_stride)
    t_start = time.perf_counter()

    def flush(drive_id: str) -> None:
        segs = pending.pop(drive_id, [])
        if not segs:
            return
        refs = [ccs.segment_ref(n, args.platform, args.cache) for n in segs]
        for block in split_contiguous(refs):
            res = process_block(
                block, vehicle, cfg, with_raw_can=True, max_stage=2, source=source
            )
            if res is None:
                for n in segs:
                    seg_rows.append({"segment_name": n, "drive_id": drive_id, "status": "skipped:empty"})
                continue
            stats.add(res.gs.df, len(res.segment_spans))
            ev = _annotate_segment(events_to_frame(res.gs, res.events, cfg_hash),
                                   res.segment_spans, None)
            cd = _annotate_segment(candidates_to_frame(res.candidates, res.gs, cfg_hash),
                                   res.segment_spans, None)
            for f, acc in ((ev, ev_frames), (cd, cand_frames)):
                if not f.empty:
                    f.insert(0, "platform", args.platform)
                    acc.append(f)
            for idx, t0, t1 in res.segment_spans:
                name = next((n for n in segs if ccs.SegmentName.parse(n).index == idx), None)
                seg_rows.append({
                    "segment_name": name, "drive_id": drive_id, "segment": idx,
                    "duration_s": round(t1 - t0, 2),
                    "n_events": sum(1 for e in res.events if t0 <= e.t_start <= t1),
                    "n_candidates": sum(1 for c in res.candidates if t0 <= c.t_start <= t1),
                    "status": "ok",
                })
            for name in segs:
                _availability_row(seg_rows, name, res.gs.df)
        if args.discard_cache:
            for n in segs:
                ccs.local_path(n, args.cache).unlink(missing_ok=True)

    def _availability_row(rows, name, df):
        row = next((r for r in rows if r.get("segment_name") == name and r.get("status") == "ok"), None)
        if row is None:
            return
        for col in AVAILABILITY:
            if col in df:
                x = df[col].to_numpy()
                x = x[np.isfinite(x)]
                row[f"has_{col}"] = bool(x.size)
                row[f"{col}_max"] = float(np.max(x)) if x.size else np.nan
            else:
                row[f"has_{col}"] = False

    try:
        while True:
            item = q.get()
            if item is None:
                break
            name, result = item
            n_seen += 1
            if isinstance(result, Exception) or result is None:
                failures.append({"segment_name": name, "error": str(result)})
                seg_rows.append({"segment_name": name, "status": "failed:download"})
            else:
                drive = ccs.SegmentName.parse(name).drive_id
                pending.setdefault(drive, []).append(name)
            # 別ルートに移ったら前のルートを処理する
            for d in [d for d in list(pending) if d != ccs.SegmentName.parse(name).drive_id]:
                flush(d)
            if n_seen % 25 == 0:
                el = time.perf_counter() - t_start
                print(f"  {n_seen}/{len(todo)}  経過 {el/60:.1f} 分  候補 {sum(len(f) for f in cand_frames)} 件")
        for d in list(pending):
            flush(d)
    except KeyboardInterrupt:
        stop.set()
        print("\n中断しました。ここまでの結果を書き出します。")
        for d in list(pending):
            flush(d)

    segs_df = pd.DataFrame(seg_rows)
    ev_df = pd.concat(ev_frames, ignore_index=True) if ev_frames else pd.DataFrame()
    cd_df = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    if not cd_df.empty:
        cd_df = cd_df.sort_values("severity", ascending=False).reset_index(drop=True)

    if args.resume and seg_path.is_file():
        segs_df = pd.concat([pd.read_csv(seg_path), segs_df], ignore_index=True)
        for f, name in ((ev_df, "events.csv"), (cd_df, "candidates.csv")):
            old = args.out / name
            if old.is_file() and not f.empty:
                merged = pd.concat([pd.read_csv(old), f], ignore_index=True)
                if name == "candidates.csv":
                    merged = merged.sort_values("severity", ascending=False).reset_index(drop=True)
                f = merged
            (f if not f.empty else pd.DataFrame()).to_csv(old, index=False)
    else:
        ev_df.to_csv(args.out / "events.csv", index=False)
        cd_df.to_csv(args.out / "candidates.csv", index=False)
    segs_df.to_csv(seg_path, index=False)

    elapsed = time.perf_counter() - t_start
    ok = segs_df[segs_df.get("status", pd.Series(dtype=str)) == "ok"] if not segs_df.empty else pd.DataFrame()
    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform, "config_hash": cfg_hash,
        "n_selected": len(names), "n_processed": int(len(ok)), "n_failed": len(failures),
        "elapsed_min": round(elapsed / 60, 2),
        "n_events": int(len(ev_df)), "n_candidates": int(len(cd_df)),
    }
    (args.out / "screen_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"処理 {len(ok)} セグメント / 失敗 {len(failures)} 件 / {elapsed/60:.1f} 分")
    print(f"候補 {len(cd_df)} 件 / イベント {len(ev_df)} 件")
    hours = float(segs_df.get("duration_s", pd.Series(dtype=float)).sum()) / 3600.0
    if not ev_df.empty:
        print(f"\nイベント種別 (走査 {hours:.1f} 時間):")
        vc = ev_df.event_type.value_counts()
        print(f"  {'イベント':<24}{'件数':>6}{'件/時':>9}")
        for name in list(FOCUS_EVENTS) + [x for x in vc.index if x not in FOCUS_EVENTS]:
            n = int(vc.get(name, 0))
            rate = f"{n / hours:.2f}" if hours > 0 else "-"
            print(f"  {name:<24}{n:>6}{rate:>9}")

        pairs = pair_counts(ev_df)
        if not pairs.empty:
            pairs.to_csv(args.out / "pair_counts.csv", index=False)
            print(f"\n複数イベントの重なり ({PAIR_WINDOW_S:.0f} 秒以内):")
            print(pairs[["組み合わせ", "件数"]].to_string(index=False))

    if not cd_df.empty:
        ranked = rank_candidates(cd_df)
        ranked.to_csv(args.out / "ranked_candidates.csv", index=False)
        print("\n優先度別の候補数:")
        print(ranked.priority.value_counts().sort_index().to_string())
        cols = [c for c in ("priority", "pairs", "drive_id", "segment", "t_in_segment_s",
                            "severity", "event_types") if c in ranked.columns]
        top = ranked[ranked.priority < "3_低"].head(20)
        if not top.empty:
            print(f"\n高・中優先度の上位 {len(top)} 件:")
            print(top[cols].round(2).to_string(index=False))
    if stats.n_segments:
        cont = stats.continuous_table()
        flags = stats.flag_table()
        cont.to_csv(args.out / "signal_stats.csv", index=False)
        flags.to_csv(args.out / "flag_stats.csv", index=False)
        print("\n連続量の分布 (間引き 1/%d、%d セグメント):" % (stats.stride, stats.n_segments))
        print(cont.to_string(index=False))
        print("\nフラグ:")
        print(flags.to_string(index=False))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
