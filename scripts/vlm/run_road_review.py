#!/usr/bin/env python3
"""路面凹凸の確認タスクのリクエストを VLM に流して結果 JSONL を書く (VLM 路面 手順 2)。

**GPU が要る。** --model echo なら GPU 無しで経路だけ通せる。
モデルの一覧・復号の設定は configs/vlm.yaml (ヒヤリハット判定と共通) を
configs/vlm_road.yaml の base_config から読む。既存の run_vlm_review.py には触らない。

  # 経路の確認 (GPU 不要)
  uv run python scripts/vlm/run_road_review.py --model echo --limit 8

  # EC2
  uv run python scripts/vlm/run_road_review.py --model qwen2_5_vl_7b
  uv run python scripts/vlm/run_road_review.py --model cosmos_reason1_7b
  uv run python scripts/vlm/run_road_review.py --model qwen3_vl_8b --conditions O

途中で落ちても、書き終えた分は飛ばして続きから流せる。
時間と資源は既定で計測し (--no-perf で止める)、<dir>/perf_<model>_mode_road.json に書く。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.vlm import road
from near_miss.vlm.perf import Meter
from near_miss.vlm.road_runner import RoadRunner, make_road_adapter
from near_miss.vlm.runner import done_ids, load_requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="configs/vlm.yaml の models のキー。または echo")
    p.add_argument("--conditions", default=None, help="条件を絞る (R / O / R,O)")
    p.add_argument("--dir", type=Path, default=Path("out/vibration/vlm_road"))
    p.add_argument("--out", type=Path, default=None, help="既定 <dir>/results_<model>_mode_road.jsonl")
    p.add_argument("--config", type=Path, default=Path("configs/vlm_road.yaml"))
    p.add_argument("--limit", type=int, default=None, help="先頭 N 件だけ流す")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="指定しなければ configs/vlm.yaml の models.<key>.max_model_len")
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--no-resume", action="store_true", help="既存の結果を消して最初から")
    p.add_argument("--no-perf", action="store_true")
    p.add_argument("--perf-interval", type=float, default=0.2)
    p.add_argument("--perf-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = road.load_config(args.config)
    spec: dict = {}
    if args.model == "echo":
        adapter, model_key = make_road_adapter("echo", "echo", cfg), "echo"
    else:
        spec = cfg["models"].get(args.model)
        if spec is None:
            raise SystemExit(f"models に {args.model} がありません: {sorted(cfg['models'])}")
        adapter, model_key = make_road_adapter(spec["adapter"], spec["model_id"], cfg, spec), args.model

    src = args.dir / "requests_mode_road.jsonl"
    if not src.is_file():
        raise SystemExit(f"リクエストがありません: {src}\n  先に scripts/vlm/build_road_inputs.py を実行してください")
    reqs = load_requests(src)
    if args.conditions:
        want = {c.strip() for c in args.conditions.split(",")}
        reqs = [r for r in reqs if r["condition"] in want]

    out = args.out or (args.dir / f"results_{model_key}_mode_road.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume and out.exists():
        out.unlink()
    already = done_ids(out)
    reps = int(cfg["repeats"]["mode_road"])
    max_len = args.max_model_len or spec.get("max_model_len")
    print(f"モデル   : {model_key} ({adapter.model_id}) / adapter {adapter.name}")
    print(f"入力     : {src}  {len(reqs)} 件 / 反復 {reps} 回")
    print(f"出力     : {out}" + (f"  (済 {len(already)} 件を飛ばす)" if already else ""))

    meter = None
    if not args.no_perf:
        meter = Meter(model_key=model_key, model_id=adapter.model_id, mode="road",
                      conditions=args.conditions or "", interval=args.perf_interval)
        rc = cfg["road"]
        meter.settings = {
            "task": road.TASK, "batch": args.batch, "gpu_util": args.gpu_util,
            "max_num_seqs": args.max_num_seqs, "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": max_len, "reps": reps, "media_kind": cfg.get("media_kind"),
            "ahead_m": rc["ahead_m"], "crop": rc["crop"],
            "temperature": float(cfg["decode"]["temperature"]), "max_tokens": cfg["decode"]["max_tokens"],
            "prompt_version": cfg["prompt_version"],
        }
        meter.begin()

    runner = RoadRunner(model_key, cfg, adapter, max_model_len=max_len, gpu_memory_utilization=args.gpu_util,
                        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                        meter=meter)
    total = 0
    try:
        for rep in range(reps):
            todo = [r for r in reqs if f"{r['request_id']}|{rep}" not in already]
            if args.limit:
                todo = todo[: args.limit]
            if not todo:
                print(f"\n反復 {rep}: 済")
                continue
            print(f"\n反復 {rep}: {len(todo)} 件")
            total += runner.run(todo, rep, out, batch=args.batch)
        print(f"\n書き出し {total} 件 -> {out}")
    finally:
        if meter is not None:
            meter.sampler.stop()
            dest = args.perf_out or (out.parent / f"perf_{model_key}_mode_road.json")
            print("\n" + meter.report())
            print(f"計測の明細: {meter.write(dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
