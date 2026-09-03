#!/usr/bin/env python3
"""リクエスト JSONL を VLM に流して結果 JSONL を書く (Phase 1 段階 2 以降)。

**GPU が要る (EC2)。** ただし --backend echo なら GPU 無しで全経路を通せるので、
Mac でプロンプト組み立てと入出力の確認ができる。

モデルを差し替えても入力生成・prompt・schema・評価は変わらない。
変わるのは重み・chat template・視覚トークン化だけ (adapters.py が吸収)。

  # 経路の確認 (GPU 不要)
  uv run python scripts/run_vlm_review.py --model echo --mode a --limit 8

  # EC2: 基準モデル
  uv run python scripts/run_vlm_review.py --model qwen2_5_vl_7b --mode a
  uv run python scripts/run_vlm_review.py --model qwen2_5_vl_7b --mode b

  # 同一条件で差し替え
  uv run python scripts/run_vlm_review.py --model cosmos_reason1_7b --mode b
  uv run python scripts/run_vlm_review.py --model qwen3_vl_8b --mode b

途中で落ちても、書き終えた分は飛ばして続きから流せる (--resume は既定)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.config import load_yaml
from near_miss.vlm.adapters import make_adapter
from near_miss.vlm.runner import Runner, done_ids, load_requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="configs/vlm.yaml の models のキー。または echo")
    p.add_argument("--mode", choices=("a", "b"), required=True,
                   help="a = 一括判定 / b = オンライン判定")
    p.add_argument("--conditions", default=None,
                   help="モード A の条件を絞る (例 C または A,B,C)")
    p.add_argument("--dir", type=Path, default=Path("out/chunk1/vlm"))
    p.add_argument("--config", type=Path, default=Path("configs/vlm.yaml"))
    p.add_argument("--limit", type=int, default=None, help="先頭 N 件だけ流す")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--no-resume", action="store_true", help="既存の結果を無視して最初から")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.model == "echo":
        adapter, model_key = make_adapter("echo", "echo", cfg), "echo"
    else:
        spec = cfg["models"].get(args.model)
        if spec is None:
            raise SystemExit(f"models に {args.model} がありません: "
                             f"{sorted(cfg['models'])}")
        adapter = make_adapter(spec["adapter"], spec["model_id"], cfg, spec)
        model_key = args.model

    src = args.dir / f"requests_mode_{args.mode}.jsonl"
    if not src.is_file():
        raise SystemExit(f"リクエストがありません: {src}\n"
                         "  先に scripts/build_vlm_inputs.py を実行してください")
    reqs = load_requests(src)
    if args.conditions and args.mode == "a":
        want = {c.strip() for c in args.conditions.split(",")}
        reqs = [r for r in reqs if r["condition"] in want]

    out = args.dir / f"results_{model_key}_mode_{args.mode}.jsonl"
    if args.no_resume and out.exists():
        out.unlink()
    already = done_ids(out)

    reps = int(cfg["repeats"]["mode_a" if args.mode == "a" else "mode_b"])
    temp = float(cfg["decode"]["temperature"])
    print(f"モデル   : {model_key} ({adapter.model_id}) / adapter {adapter.name}")
    print(f"入力     : {src.name}  {len(reqs)} 件")
    print(f"反復     : {reps} 回"
          + ("  ※ temperature=0 なので再現性確認であって自己一致率ではない"
             if temp == 0 else "  ※ temperature>0 なので自己一致率を主指標にできる"))
    print(f"出力     : {out}"
          + (f"  (済 {len(already)} 件を飛ばす)" if already else ""))

    runner = Runner(model_key, cfg, adapter,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_util)
    total = 0
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
