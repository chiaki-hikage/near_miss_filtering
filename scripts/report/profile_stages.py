#!/usr/bin/env python3
"""抽出処理の段ごとの所要時間を測る。**判定の中身は変えない。**

3 段構成 (L1 除去 / L2 特徴量 / L3 判定) を入れる前に、
**いま時間がどこで消えているか**を確かめるための道具。
取得と復号が支配的なら、L1 で棄却しても総時間は縮まない。

  uv run python scripts/report/profile_stages.py --platform TOYOTA_RAV4_TSS2 --limit 100
  uv run python scripts/report/profile_stages.py --comma2k19 raw_data/Chunk_1 --limit 60

**逐次で測る。** 並列にすると 1 段あたりの時間が測れない。
"""

from __future__ import annotations

import argparse
import json
import platform as _pf
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.detectors import detect_all
from near_miss.features import compute_features
from near_miss.io.canonical import concat_segments
from near_miss.pipeline import split_contiguous
from near_miss.scoring import build_candidates
from near_miss.signals import to_grid

# 段の並び。表の順序をここで決める。
STAGES = ("read", "decode", "radar", "load_other", "grid", "features", "detect", "candidates")

T: dict[str, float] = defaultdict(float)
N: dict[str, int] = defaultdict(int)


@contextmanager
def timed(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        T[name] += time.perf_counter() - t0
        N[name] += 1


# 読み出しの内訳。**入れ子になる関数を同時に測ると二重に数える。**
# comma2k19 の _load_raw_can は内部で decode_channels を呼ぶので、外側だけを測る。
PATCH = {
    "ccs": (("read_rlog", "read"), ("decode_channels", "decode"),
            ("radar_from_can", "radar")),
    "c2k19": (("_load_processed_can", "read"), ("_load_raw_can", "decode")),
}


def patch_loader(module, which: str) -> None:
    """読み出しの内訳を測れるように、呼び先を計時付きに差し替える。

    **`from X import Y` で束縛済みなので、利用側モジュールの属性を差し替える。**
    元の関数はそのまま呼ぶので、処理内容は変わらない。
    """
    for attr, label in PATCH[which]:
        fn = getattr(module, attr, None)
        if fn is None or getattr(fn, "_profiled", False):
            continue

        def make(f, lab):
            def wrapper(*a, **kw):
                with timed(lab):
                    return f(*a, **kw)
            wrapper._profiled = True
            return wrapper

        setattr(module, attr, make(fn, label))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--platform", help="commaCarSegments の車種キー")
    g.add_argument("--comma2k19", type=Path, help="comma2k19 のチャンク")
    p.add_argument("--limit", type=int, default=100, help="測るセグメント数 (既定 100)")
    p.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=Path("out/profile_stages.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.detection)
    vehicles = load_vehicle_configs(args.vehicles)

    if args.platform:
        from near_miss.io import comma_car_segments as ccs
        patch_loader(ccs, "ccs")
        refs = ccs.find_segments(platform=args.platform)
        load = lambda ref, veh: ccs.load_segment(ref, veh, with_raw_can=True)  # noqa: E731
        target = f"commaCarSegments {args.platform}"
    else:
        from near_miss.io import comma2k19 as c2k
        patch_loader(c2k, "c2k19")
        refs = c2k.find_segments(args.comma2k19)
        load = lambda ref, veh: c2k.load_segment(ref, veh, with_raw_can=True)  # noqa: E731
        target = f"comma2k19 {args.comma2k19}"

    refs = sorted(refs, key=lambda r: (r.drive_id, r.index))[: args.limit]
    if not refs:
        raise SystemExit("セグメントが見つかりません")

    by_drive: dict[str, list] = defaultdict(list)
    for r in refs:
        by_drive[r.drive_id].append(r)

    print("=" * 72)
    print(f"対象   : {target}  {len(refs)} セグメント / {len(by_drive)} ドライブ")
    print(f"環境   : {_pf.system()} {_pf.machine()} / Python {_pf.python_version()}")
    print("=" * 72)

    n_seg = n_events = n_cands = 0
    duration_s = 0.0
    wall0 = time.perf_counter()

    for drive_id, drefs in by_drive.items():
        vehicle = find_vehicle_config(drefs[0].dongle_id, vehicles) if not args.platform \
            else next((v for v in vehicles if args.platform in v.platforms), None)
        if vehicle is None:
            print(f"  車両設定なし: {drive_id}")
            continue
        for block in split_contiguous(drefs):
            segs = []
            for ref in block:
                with timed("load_other"):
                    pass                     # 呼び出し側の枠。実体は patch 側で測る
                try:
                    sd = load(ref, vehicle)
                except Exception as exc:
                    print(f"  読み出し失敗 {ref.segment_id}: {exc}")
                    continue
                a, b = sd.t_span
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                segs.append(sd)
            if not segs:
                continue
            n_seg += len(segs)

            merged = concat_segments(segs)
            with timed("grid"):
                gs = to_grid(merged, cfg)
            if gs.df.empty:
                continue
            duration_s += float(gs.t[-1] - gs.t[0])
            with timed("features"):
                gs = compute_features(gs, cfg, radar=merged.radar, vehicle=vehicle)
            with timed("detect"):
                events = detect_all(gs, cfg, max_stage=2)
            with timed("candidates"):
                cands = build_candidates(gs, events, cfg)
            n_events += len(events)
            n_cands += len(cands)

    wall = time.perf_counter() - wall0
    total = sum(T[s] for s in STAGES)

    print(f"\n処理: {n_seg} セグメント / {duration_s / 3600:.2f} 時間 "
          f"-> イベント {n_events} / 候補 {n_cands}")
    print(f"壁時計 {wall:.1f} 秒  (計測できた合計 {total:.1f} 秒 = {100 * total / wall:.0f}%)\n")

    print(f"{'段':<12}{'合計[秒]':>10}{'割合':>8}{'1セグメント[ms]':>16}{'呼出':>8}")
    print("-" * 56)
    label = {"read": "取得・展開", "decode": "CAN 復号", "radar": "レーダ復号",
             "load_other": "読み出しその他", "grid": "L2 グリッド",
             "features": "L2 特徴量", "detect": "L3 判定", "candidates": "L3 候補化"}
    for s in STAGES:
        if T[s] <= 0:
            continue
        print(f"{label[s]:<12}{T[s]:>10.2f}{100 * T[s] / total:>7.1f}%"
              f"{1000 * T[s] / max(n_seg, 1):>16.1f}{N[s]:>8}")
    print("-" * 56)
    load_s = T["read"] + T["decode"] + T["radar"] + T["load_other"]
    l2 = T["grid"] + T["features"]
    l3 = T["detect"] + T["candidates"]
    print(f"{'読み出し計':<12}{load_s:>10.2f}{100 * load_s / total:>7.1f}%")
    print(f"{'L2 計':<12}{l2:>10.2f}{100 * l2 / total:>7.1f}%")
    print(f"{'L3 計':<12}{l3:>10.2f}{100 * l3 / total:>7.1f}%")
    print("=" * 56)
    print(f"\nL1 で棄却しても縮まないのは読み出し {100 * load_s / total:.0f}%。"
          f"\nL1 が効く上限は L2+L3 = {100 * (l2 + l3) / total:.0f}% "
          f"({1000 * (l2 + l3) / max(n_seg, 1):.1f} ms/セグメント)。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "target": target, "n_segments": n_seg, "duration_h": duration_s / 3600,
        "n_events": n_events, "n_candidates": n_cands,
        "wall_s": wall, "measured_s": total,
        "stages_s": {s: T[s] for s in STAGES if T[s] > 0},
        "per_segment_ms": {s: 1000 * T[s] / max(n_seg, 1) for s in STAGES if T[s] > 0},
        "system": _pf.system(), "machine": _pf.machine(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
