#!/usr/bin/env python3
"""路面凹凸の確認タスクのリクエストを組み立てる (VLM 路面 手順 1)。GPU は要らない。

    振動イベント (near_miss_vibration_events.csv) + 対照の時刻
        -> 目標地点が前方 20・15・11・8・5 m に見えるフレームを選ぶ (global_pose/frame_times)
        -> セグメントごとに 1 度だけデコードし、路面の部分を切り出して JPEG に
        -> 条件 R (印なし) / O (車輪の通り道と推定位置の印あり) のリクエスト JSONL
        -> manifest.csv (採点用。イベントか対照か・候補か・以前の目視判定など。**モデルには渡さない**)

目標の ID は T0001… の通し番号で、イベントと対照を混ぜて乱数で並べてある
(ID や順番からイベントか対照かが分からないように)。

  uv run python scripts/vlm/build_road_inputs.py --dry-run          # 何を作るか見るだけ
  uv run python scripts/vlm/build_road_inputs.py                    # 全振動イベント + 対照 300
  uv run python scripts/vlm/build_road_inputs.py --targets candidates --control-n 40
  uv run python scripts/vlm/build_road_inputs.py --hwaccel          # EC2 (NVDEC でデコード)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

from near_miss.config import config_hash
from near_miss.io.comma2k19 import find_segments
from near_miss.vlm import road

W, H = 1164, 874


@dataclass
class Target:
    kind: str                 # event / control
    source_id: str            # E0359 など。対照は C0001…
    drive: str
    segment: int              # 時刻の原点を取ったセグメント
    t_rel: float              # そのセグメントの IMU 先頭からの秒
    t_abs: float              # boot time
    v: float                  # m/s
    meta: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)   # [(segment, frame, t_frame, d_m)]
    dropped: list = field(default_factory=list)  # 取れなかった距離


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/vlm_road.yaml"))
    p.add_argument("--roots", type=Path, nargs="+", default=[Path("raw_data/Chunk_1"), Path("raw_data/Chunk_2")])
    p.add_argument("--out", type=Path, default=Path("out/vibration/vlm_road"))
    p.add_argument("--targets", choices=("all", "candidates"), default=None,
                   help="既定は設定の road.targets")
    p.add_argument("--control-n", type=int, default=None, help="対照の数 (既定は設定の road.control.n)")
    p.add_argument("--limit", type=int, default=None, help="振動イベントを先頭 N 件だけ (経路の確認用)")
    p.add_argument("--hwaccel", action="store_true", help="NVDEC を使う (EC2)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def portable(p: Path) -> str:
    """リクエストに書くパス。リポジトリの中ならそこからの相対パスにする (EC2 にそのまま運べるように)。"""
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def imu_t0(seg_dir: Path) -> float:
    return float(np.load(seg_dir / "processed_log/IMU/accelerometer/t").ravel()[0])


def frame_times(seg_dir: Path) -> np.ndarray:
    return np.load(seg_dir / "global_pose/frame_times").ravel().astype(float)


def load_events(rc: dict, which: str, limit: int | None) -> pd.DataFrame:
    ev = pd.read_csv(rc["events"])
    ev["is_candidate"] = ev[rc["candidate_columns"]].fillna(False).astype(bool).all(axis=1)
    if which == "candidates":
        ev = ev[ev.is_candidate]
    if limit:
        ev = ev.head(limit)
    return ev.reset_index(drop=True)


def daytime(utc: pd.Series, rc: dict) -> pd.Series:
    h = (pd.to_datetime(utc) + pd.Timedelta(hours=rc["local_utc_offset_h"])).dt.hour
    lo, hi = rc["daylight_local_h"]
    return (h >= lo) & (h < hi)


def event_targets(ev: pd.DataFrame, segs: dict) -> tuple[list[Target], list[str]]:
    out, skipped = [], []
    for e in ev.itertuples():
        key = (e.drive, int(e.segment))
        if key not in segs:
            skipped.append(f"{e.event_id}: セグメントが無い {key}")
            continue
        t0 = imu_t0(segs[key])
        out.append(Target(
            kind="event", source_id=e.event_id, drive=e.drive, segment=int(e.segment),
            t_rel=float(e.event_time_s), t_abs=t0 + float(e.event_time_s), v=float(e.speed_kmh) / 3.6,
            meta=dict(event_id=e.event_id, is_candidate=bool(e.is_candidate), site_id=e.site_id,
                      rank=e.rank, A=e.A, near_miss_candidate_score=e.near_miss_candidate_score,
                      video_road_feature=e.video_road_feature, wheel_lr_log2=e.wheel_lr_log2,
                      side_index=e.side_index, event_utc=e.event_utc),
        ))
    return out, skipped


def control_targets(ev_all: pd.DataFrame, segs: dict, rc: dict, n: int) -> list[Target]:
    """振動イベントのあるセグメントから、どのイベントからも離れた時刻を無作為に選ぶ。

    距離は全振動イベント (対象を絞っていても 1,037 件すべて) から測る。
    """
    cc = rc["control"]
    rng = np.random.default_rng(int(cc["seed"]))
    keys = sorted({(d, int(s)) for d, s in zip(ev_all.drive, ev_all.segment) if (d, int(s)) in segs})
    ev_abs: dict[str, list[float]] = {}
    unix_off: dict[str, list[float]] = {}     # UTC [s] − boot time。同じドライブのイベントから取る
    for e in ev_all.itertuples():
        k = (e.drive, int(e.segment))
        if k in segs:
            ta = imu_t0(segs[k]) + float(e.event_time_s)
            ev_abs.setdefault(e.drive, []).append(ta)
            unix_off.setdefault(e.drive, []).append(pd.Timestamp(e.event_utc).timestamp() - ta)
    out: list[Target] = []
    tries = 0
    while len(out) < n and tries < 50 * max(n, 1):
        tries += 1
        drive, seg = keys[int(rng.integers(len(keys)))]
        t_rel = float(rng.uniform(*cc["t_range_s"]))
        d = segs[(drive, seg)]
        t0 = imu_t0(d)
        t_abs = t0 + t_rel
        if np.min(np.abs(np.asarray(ev_abs[drive]) - t_abs)) < cc["min_gap_s"]:
            continue
        if any(abs(t.t_abs - t_abs) < cc["min_gap_s"] for t in out if t.drive == drive):
            continue
        tv = np.load(d / "processed_log/CAN/speed/t").ravel().astype(float)
        vv = np.load(d / "processed_log/CAN/speed/value").astype(float).reshape(len(tv), -1)[:, 0]
        v = float(np.interp(t_abs, tv, vv))
        if v < cc["min_speed_mps"]:
            continue
        # UTC は昼夜の層別にしか使わない (event_utc は秒単位なので ±1 秒の精度)
        utc = str(pd.to_datetime(t_abs + float(np.median(unix_off[drive])), unit="s").floor("s"))
        out.append(Target(kind="control", source_id=f"C{len(out) + 1:04d}", drive=drive, segment=seg,
                          t_rel=round(t_rel, 2), t_abs=t_abs, v=v, meta=dict(event_utc=utc)))
    return out


def pick_frames(t: Target, segs: dict, rc: dict, behind: float, ft_cache: dict) -> None:
    """各距離で、狙いの時刻に最も近いフレームを前後のセグメントも含めて探す。"""
    for ahead in rc["ahead_m"]:
        want = road.frame_time_for_distance(t.t_abs, t.v, float(ahead), behind)
        best = None
        for s in (t.segment - 1, t.segment, t.segment + 1):
            if (t.drive, s) not in segs:
                continue
            if (t.drive, s) not in ft_cache:
                ft_cache[(t.drive, s)] = frame_times(segs[(t.drive, s)])
            ft = ft_cache[(t.drive, s)]
            i = int(np.argmin(np.abs(ft - want)))
            if best is None or abs(ft[i] - want) < abs(best[2] - want):
                best = (s, i, float(ft[i]))
        if best is None or abs(best[2] - want) > rc["max_frame_gap_s"]:
            t.dropped.append(float(ahead))
            continue
        s, i, tf = best
        t.frames.append((s, i, tf, road.distance_at(t.t_abs, tf, t.v, behind)))


def decode_needed(video: Path, needed: set[int], hwaccel: bool):
    """セグメントの映像を先頭から読み、要るフレームだけ (番号, 画像) で返す。

    video.hevc はタイムスタンプを持たないので -r 20 で読み、出てきた順の番号を
    フレーム番号とする (final_qualitative.frame_at と同じ数え方)。
    """
    cmd = ["ffmpeg", "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-r", "20", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    last = max(needed)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        n = 0
        while n <= last:
            buf = p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            if n in needed:
                yield n, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            n += 1
        p.kill()


def crop(img: np.ndarray, rc: dict) -> Image.Image:
    (c0, c1), (r0, r1) = rc["crop"]["col"], rc["crop"]["row"]
    return Image.fromarray(img[r0:r1, c0:c1])


def draw_overlay(im: Image.Image, d_m: float, cam: dict, rc: dict) -> Image.Image:
    """左右の車輪の通り道 (破線) と、目標地点の推定位置 (○) を描く。"""
    im = im.copy()
    dr = ImageDraw.Draw(im)
    c0, r0 = rc["crop"]["col"][0], rc["crop"]["row"][0]
    colors = {"left": (40, 110, 230), "right": (230, 50, 50)}
    for side, pts in road.wheel_paths_px(cam, rc["track_m"], tuple(rc["marker_d_range_m"])).items():
        xy = [(x - c0, y - r0) for x, y in pts]
        for k in range(0, len(xy) - 1, 2):      # 破線
            dr.line([xy[k], xy[k + 1]], fill=colors[side], width=2)
    for side, sgn in (("left", -1.0), ("right", 1.0)):
        x, y = road.ground_px(d_m, sgn * rc["track_m"] / 2, cam)
        r = max(6.0, 160.0 / d_m)
        dr.ellipse([x - c0 - r, y - r0 - r, x - c0 + r, y - r0 + r], outline=colors[side], width=3)
    return im


def main() -> int:
    args = parse_args()
    cfg = road.load_config(args.config)
    rc = cfg["road"]
    which = args.targets or rc["targets"]
    n_ctrl = rc["control"]["n"] if args.control_n is None else args.control_n
    fit = yaml.safe_load(Path(rc["camera"]["fit"]).read_text())
    cam = dict(focal_px=rc["camera"]["focal_px"], center_col=rc["camera"]["center_col"],
               horizon_row=fit["horizon_row"], K=fit["K"])
    behind = float(fit["camera_behind_front_axle_m"])

    segs = {(r.drive_id, r.index): r.path for root in args.roots for r in find_segments(root)}
    ev_all = load_events(rc, "all", None)
    ev = load_events(rc, which, args.limit)
    targets, skipped = event_targets(ev, segs)
    targets += control_targets(ev_all, segs, rc, n_ctrl)
    ft_cache: dict = {}
    for t in targets:
        pick_frames(t, segs, rc, behind, ft_cache)
    usable = [t for t in targets if t.frames]
    needed: dict[tuple[str, int], set[int]] = {}
    for t in usable:
        for s, i, _, _ in t.frames:
            needed.setdefault((t.drive, s), set()).add(i)

    print(f"振動イベント {sum(t.kind == 'event' for t in targets)} 件 ({which})"
          f" / 対照 {sum(t.kind == 'control' for t in targets)} 件")
    print(f"フレームが 1 枚も取れない目標: {len(targets) - len(usable)} 件 / "
          f"一部の距離が欠ける目標: {sum(bool(t.dropped) and bool(t.frames) for t in targets)} 件")
    print(f"デコードするセグメント: {len(needed)} 本 / フレーム {sum(map(len, needed.values()))} 枚")
    for s in skipped[:5]:
        print("  飛ばした:", s)
    if args.dry_run:
        return 0

    out = args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    raw_path = lambda drive, s, i: out / "frames" / drive.split("|")[-1] / f"{s:03d}" / f"f{i:05d}.jpg"
    q = int(rc["jpeg_quality"])
    for k, (key, idx) in enumerate(sorted(needed.items())):
        todo = {i for i in idx if not raw_path(key[0], key[1], i).is_file()}
        if todo:
            raw_path(key[0], key[1], 0).parent.mkdir(parents=True, exist_ok=True)
            got = set()
            for i, img in decode_needed(segs[key] / "video.hevc", todo, args.hwaccel):
                crop(img, rc).save(raw_path(key[0], key[1], i), quality=q)
                got.add(i)
            if todo - got:
                print(f"  {key}: 映像に無いフレーム {sorted(todo - got)[:5]}")
        print(f"  デコード {k + 1}/{len(needed)}", end="\r", flush=True)
    print()

    # 目標の ID はイベントと対照を混ぜて乱数で並べる
    rng = np.random.default_rng(int(rc["control"]["seed"]) + 1)
    order = rng.permutation(len(usable))
    chash = config_hash(cfg)
    reqs, man = [], []
    for n, j in enumerate(order, 1):
        t = usable[j]
        tid = f"T{n:04d}"
        ok = [(s, i, tf, d) for s, i, tf, d in t.frames if raw_path(t.drive, s, i).is_file()]
        if not ok:
            continue
        info = [{"d_m": round(d, 1), "dt_s": round(t.t_abs - tf, 3)} for _, _, tf, d in ok]
        for cond, spec in rc["conditions"].items():
            if spec["overlay"]:
                paths = []
                for k, (s, i, tf, d) in enumerate(ok):
                    p = out / "overlay" / f"{tid}_{k + 1}.jpg"
                    draw_overlay(Image.open(raw_path(t.drive, s, i)), d, cam, rc).save(p, quality=q)
                    paths.append(portable(p))
            else:
                paths = [portable(raw_path(t.drive, s, i)) for s, i, _, _ in ok]
            reqs.append(road.make_request(tid, cond, paths, info, chash))
        man.append(dict(target_id=tid, kind=t.kind, source_id=t.source_id, drive=t.drive, segment=t.segment,
                        t_rel_s=round(t.t_rel, 2), speed_kmh=round(t.v * 3.6, 1), n_frames=len(ok),
                        dropped_ahead_m=";".join(f"{d:g}" for d in t.dropped),
                        d_m=";".join(f"{x['d_m']:g}" for x in info), **t.meta))
    reqs.sort(key=lambda r: r["request_id"])
    with (out / "requests_mode_road.jsonl").open("w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mf = pd.DataFrame(man).sort_values("target_id")
    mf["daytime"] = daytime(mf.event_utc, rc).where(mf.event_utc.notna())
    mf.to_csv(out / "manifest.csv", index=False)
    segs_used = sorted(needed)
    (out / "needed_segments.txt").write_text(
        "\n".join(str(segs[k]) for k in segs_used) + "\n", encoding="utf-8")
    (out / "build_meta.json").write_text(json.dumps(dict(
        config=str(args.config), config_hash=chash, targets=which, events=int((mf.kind == "event").sum()),
        controls=int((mf.kind == "control").sum()), requests=len(reqs), segments=len(segs_used),
        conditions=list(rc["conditions"]), ahead_m=rc["ahead_m"], skipped=skipped), ensure_ascii=False, indent=2))
    print(f"リクエスト {len(reqs)} 件 -> {out / 'requests_mode_road.jsonl'}")
    print(f"manifest {len(mf)} 件 -> {out / 'manifest.csv'} (採点用。モデルには渡さない)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
