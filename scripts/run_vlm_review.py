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

**時間と資源は既定で計測する** (--no-perf で止める)。判定の中身には触らない。
実行の最後に要約を出し、<dir>/perf_<model>_mode_<x>.json に明細を書く。

  モデルロード時間 / 1 件あたりの推論時間 / 総処理時間 / 映像 1 分あたりの処理時間
  GPU メモリ (起動前・ロード後・推論中ピーク) と GPU 使用率
  CPU 使用率と RAM (機械全体、および自プロセス + 子プロセスの RSS)
  GPU 名 / モデル / dtype / max_model_len / 入力フレーム数などの測定条件

  # 短く測るだけ (結果を汚さないよう別ディレクトリに出す)
  uv run python scripts/run_vlm_review.py --model qwen3_vl_8b --mode b \
      --limit 64 --dir out/chunk1/vlm --perf-out out/perf_qwen3_8b.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from near_miss.config import load_yaml
from near_miss.vlm.adapters import make_adapter
from near_miss.vlm.perf import Meter
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
    p.add_argument("--out", type=Path, default=None,
                   help="結果 JSONL の書き出し先 "
                        "(既定 <dir>/results_<model>_mode_<x>.jsonl)。"
                        "流し終えた結果を残したまま測り直すときに別ファイルを指定する")
    p.add_argument("--config", type=Path, default=Path("configs/vlm.yaml"))
    p.add_argument("--limit", type=int, default=None, help="先頭 N 件だけ流す")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="指定しなければ configs/vlm.yaml の models.<key>.max_model_len")
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--no-resume", action="store_true", help="既存の結果を無視して最初から")
    p.add_argument("--no-perf", action="store_true",
                   help="時間と GPU メモリの計測をしない (既定は計測する)")
    p.add_argument("--perf-interval", type=float, default=0.2,
                   help="GPU メモリの採取間隔 [秒] (既定 0.2)")
    p.add_argument("--perf-out", type=Path, default=None,
                   help="計測結果の書き出し先 (既定 <dir>/perf_<model>_mode_<x>.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    spec: dict = {}
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

    out = args.out or (args.dir / f"results_{model_key}_mode_{args.mode}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
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

    # モデル固有の上限。Qwen3-VL-30B-A3B は既定の 262144 では KV キャッシュが
    # 96 GB に収まらない。CLI の指定があればそちらを優先する。
    max_len = args.max_model_len or spec.get("max_model_len")
    if max_len:
        print(f"max_model_len: {max_len}"
              + ("  (コマンドラインの指定)" if args.max_model_len else "  (設定から)"))
    meter = None
    if not args.no_perf:
        meter = Meter(model_key=model_key, model_id=adapter.model_id,
                      mode=args.mode, conditions=args.conditions or "",
                      interval=args.perf_interval)
        # 測定条件。後から「何を測ったのか」が分かるように全部残す。
        inp = cfg["input"]
        meter.settings = {
            "batch": args.batch, "gpu_util": args.gpu_util,
            "max_model_len": max_len, "reps": reps,
            "video_long_edge": inp["video_long_edge"], "video_fps": inp["video_fps"],
            "window_video_s": inp["window_video_s"], "window_can_s": inp["window_can_s"],
            "temperature": temp, "max_tokens": cfg["decode"]["max_tokens"],
            "prompt_version": cfg["prompt_version"],
        }
        # モデルを読む前の使用量を「通常時」の基準にする。
        meter.begin()
        gpus = ", ".join(d["name"] for d in meter.gpu.devices if "name" in d)
        print(f"計測     : {meter.gpu.source or '時間のみ (NVML/nvidia-smi 無し)'}"
              + (f"  {gpus}" if gpus else ""))

    runner = Runner(model_key, cfg, adapter,
                    max_model_len=max_len,
                    gpu_memory_utilization=args.gpu_util,
                    meter=meter)
    total = 0
    try:
        for rep in range(reps):
            todo = [r for r in reqs if f"{r['request_id']}|{rep}" not in already]
            if args.limit:
                todo = todo[: args.limit]
            if not todo:
                print(f"\n反復 {rep}: 済 "
                      f"({out.name} に結果があるので飛ばします)")
                continue
            print(f"\n反復 {rep}: {len(todo)} 件")
            total += runner.run(todo, rep, out, batch=args.batch)
        print(f"\n書き出し {total} 件 -> {out}")
        if total == 0 and meter is not None:
            # 流し終えた後に計測だけしたい場合。既存の結果は消さない。
            print("\n  計測する件がありません。結果を残したまま測り直すなら:\n"
                  f"    --limit 64 --out {args.dir}/perf_run/"
                  f"results_{model_key}_mode_{args.mode}.jsonl")
    finally:
        # 途中で落ちても、そこまでの計測は残す。
        # OOM のときこそ「どこでピークに達したか」が要る。
        if meter is not None:
            meter.sampler.stop()
            # 結果の隣に置く。--out で別に出したときも一緒に移る。
            dest = args.perf_out or (out.parent
                                     / f"perf_{model_key}_mode_{args.mode}.json")
            print("\n" + meter.report())
            print(f"計測の明細: {meter.write(dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
