#!/usr/bin/env python3
"""ドライブ単位の並列実行を worker 数ごとに測り、結果が同じことを確かめる。

worker=1 の出力を基準にして、並列時の出力と**バイト単位で**突き合わせる。
速くなっても結果が変わっていれば意味がないので、必ず両方を見る。

  # キャッシュにあるもの全部 (2,000 セグメント。1 回あたり数分〜十数分)
  uv run python scripts/benchmark_workers.py --platform TOYOTA_RAV4_TSS2 -w 1 2 4 8

  # 短く試す
  uv run python scripts/benchmark_workers.py --platform TOYOTA_RAV4_TSS2 --limit 200 -w 1 4

Mac でも EC2 でも同じコマンドで動く。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

REPO = Path(__file__).resolve().parents[1]

# 実行ごとに必ず変わる項目。突き合わせから外す。
VOLATILE = ("run_at", "elapsed_min", "workers")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def fingerprint(out: Path) -> dict[str, str]:
    """出力の指紋。gzip は mtime が入るので中身を展開してから取る。"""
    fp: dict[str, str] = {}
    cand = out / "candidates.csv"
    if cand.is_file():
        fp["candidates.csv"] = sha256_bytes(cand.read_bytes())
    dump = out / "stage1_samples.csv.gz"
    if dump.is_file():
        with gzip.open(dump, "rb") as f:
            fp["stage1_samples.csv"] = sha256_bytes(f.read())
    counts = out / "counts.json"
    if counts.is_file():
        meta = json.loads(counts.read_text(encoding="utf-8"))
        for k in VOLATILE:
            meta.pop(k, None)
        fp["counts.json"] = sha256_bytes(
            json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
    return fp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--platform", help="commaCarSegments の車種キー")
    g.add_argument("--comma2k19", type=Path, help="comma2k19 のチャンク")
    p.add_argument("-w", "--workers", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="試す worker 数 (既定 1 2 4 8)")
    p.add_argument("--limit", type=int, default=None, help="セグメント数の上限")
    p.add_argument("--select", choices=("cache", "catalog"), default="cache")
    p.add_argument("--dump-stage1", action="store_true",
                   help="1 次通過サンプルの明細も出して突き合わせる")
    p.add_argument("--out", type=Path, default=Path("out/benchmark_workers"))
    p.add_argument("--repeat", type=int, default=1, help="各 worker 数の繰り返し回数")
    p.add_argument("--keep", action="store_true", help="各実行の出力を残す")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = [sys.executable, str(REPO / "scripts" / "screen_sideslip.py")]
    if args.platform:
        base += ["--platform", args.platform, "--select", args.select]
        target = args.platform
    else:
        base += ["--comma2k19", str(args.comma2k19)]
        target = str(args.comma2k19)
    if args.limit is not None:
        base += ["--limit", str(args.limit)]
    if args.dump_stage1:
        base += ["--dump-stage1"]

    import platform as _pf

    print("=" * 72)
    print(f"対象     : {target}"
          + (f"  (先頭 {args.limit} セグメント)" if args.limit else "  (キャッシュ全部)"))
    print(f"環境     : {_pf.system()} {_pf.machine()} / Python {_pf.python_version()}")
    try:
        import os
        print(f"CPU 数   : {os.cpu_count()}")
    except Exception:
        pass
    print(f"worker   : {args.workers}  (各 {args.repeat} 回)")
    print("=" * 72)

    rows: list[dict] = []
    reference: dict[str, str] | None = None
    ref_workers: int | None = None

    for w in args.workers:
        for rep in range(args.repeat):
            run_out = args.out / f"w{w}" if args.repeat == 1 else args.out / f"w{w}_r{rep}"
            if run_out.exists():
                shutil.rmtree(run_out)
            cmd = base + ["--workers", str(w), "--out", str(run_out)]
            print(f"\n-- worker={w} ({rep + 1}/{args.repeat}) 実行中 ...", flush=True)
            t0 = time.perf_counter()
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8")
            elapsed = time.perf_counter() - t0
            if r.returncode != 0:
                print(r.stdout[-2000:])
                print(r.stderr[-2000:], file=sys.stderr)
                raise SystemExit(f"worker={w} の実行が失敗しました")

            fp = fingerprint(run_out)
            if reference is None:
                reference, ref_workers = fp, w
                same = True
            else:
                same = fp == reference
            meta = json.loads((run_out / "counts.json").read_text(encoding="utf-8"))
            rows.append({
                "workers": w, "rep": rep, "秒": elapsed,
                "セグメント": meta["n_segments"],
                "候補": meta["counts"]["n_candidates"],
                "一致": same, "fp": fp,
            })
            print(f"   {elapsed:6.1f} 秒  候補 {meta['counts']['n_candidates']} 件  "
                  f"一致 {'OK' if same else '**不一致**'}")
            if not args.keep and rep > 0:
                shutil.rmtree(run_out, ignore_errors=True)

    # --- 表 ---
    best = {}
    for row in rows:
        best.setdefault(row["workers"], []).append(row["秒"])
    baseline = min(best[args.workers[0]])

    print("\n" + "=" * 72)
    print(f"{'worker':>7}{'最短 [秒]':>12}{'中央 [秒]':>12}{'速度比':>9}{'効率':>8}"
          f"{'候補':>7}{'結果の一致':>12}")
    for w in args.workers:
        ts = sorted(best[w])
        med = ts[len(ts) // 2]
        agree = all(r["一致"] for r in rows if r["workers"] == w)
        n_c = next(r["候補"] for r in rows if r["workers"] == w)
        speed = baseline / ts[0]
        print(f"{w:>7}{ts[0]:>12.1f}{med:>12.1f}{speed:>8.2f}x{speed / w:>8.0%}"
              f"{n_c:>7}{('OK' if agree else '不一致'):>12}")
    print("=" * 72)

    bad = [r for r in rows if not r["一致"]]
    if bad:
        print(f"\n**worker={ref_workers} と結果が違う実行があります**")
        for r in bad:
            print(f"  worker={r['workers']} rep={r['rep']}")
            for k in sorted(set(reference) | set(r["fp"])):
                a, b = reference.get(k, "-"), r["fp"].get(k, "-")
                if a != b:
                    print(f"    {k}: 基準 {a} != {b}")
        return 1

    print(f"\nworker={ref_workers} を基準に、すべての出力が一致しました。")
    print("  突き合わせた指紋:", ", ".join(sorted(reference)))
    summary = {
        "target": target, "limit": args.limit,
        "system": _pf.system(), "machine": _pf.machine(),
        "python": _pf.python_version(),
        "reference_workers": ref_workers, "fingerprint": reference,
        "runs": [{k: v for k, v in r.items() if k != "fp"} for r in rows],
    }
    (args.out / "benchmark.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力: {args.out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
