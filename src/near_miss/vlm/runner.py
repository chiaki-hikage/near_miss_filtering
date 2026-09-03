"""推論の実行。**このモジュールだけが vllm に依存する。**

前処理・採点・試験は vllm 無しで動く。vllm は torch の CUDA ビルドを
引き連れており、GPU 世代に合うものを EC2 の上で選ぶ必要があるため、
pyproject では管理していない (docs/environment.md)。

リクエスト JSONL を読んで結果 JSONL を書くだけに閉じている。
入力生成にもモデルにも依存しない形にしてあるので、モデルを足すときは
adapters.py と configs/vlm.yaml だけを触ればよい。

**再開できる。** 途中で落ちても、書き終えた request_id は飛ばして続きから流す。
1,574 件を流し直すのは無駄なので。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .prompt import build as build_prompt
from .schema import build_schema, validate


def done_ids(path: Path) -> set[str]:
    """既に結果が書かれている request_id。再開に使う。"""
    if not path.is_file():
        return set()
    out = set()
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # 途中で落ちた最終行。読み飛ばす
        key = f"{r['request_id']}|{r.get('rep', 0)}"
        out.add(key)
    return out


def load_requests(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def parse_response(text: str) -> tuple[dict[str, Any] | None, str]:
    """モデルの出力から JSON を取り出す。

    構造化出力 (structured_outputs) が効いていれば素直に読めるはず。
    効いていない場合に備えて
    最初の { から最後の } までを試すが、**それを常態にしない**。
    schema 適合率はゲート条件 1 なので、直せているかを必ず見る。
    """
    t = (text or "").strip()
    try:
        return json.loads(t), ""
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1]), "括弧を切り出して復旧"
        except json.JSONDecodeError as exc:
            return None, f"JSON として読めない: {exc}"
    return None, "JSON が見つからない"


class Runner:
    """vLLM を 1 度だけ起動して、リクエストをまとめて流す。"""

    def __init__(self, model_key: str, cfg: dict[str, Any], adapter,
                 max_model_len: int | None = None,
                 gpu_memory_utilization: float = 0.85) -> None:
        self.cfg = cfg
        self.model_key = model_key
        self.adapter = adapter
        self._llm = None
        self._max_model_len = max_model_len
        self._gpu_util = gpu_memory_utilization

    def _ensure(self):
        if self._llm is not None or self.adapter.name == "echo":
            return
        from vllm import LLM
        kw: dict[str, Any] = {
            "model": self.adapter.model_id,
            "gpu_memory_utilization": self._gpu_util,
            "trust_remote_code": True,
        }
        if self._max_model_len:
            kw["max_model_len"] = self._max_model_len
        limits = self.adapter.limits()
        if limits:
            kw["limit_mm_per_prompt"] = limits
        self._llm = LLM(**kw)

    def _sampling(self, mode: str):
        d = self.cfg["decode"]
        if self.adapter.name == "echo":
            return None
        from vllm import SamplingParams
        # vLLM 0.28 で名前が変わった。
        #   GuidedDecodingParams -> StructuredOutputsParams
        #   SamplingParams(guided_decoding=...) -> structured_outputs=...
        # 渡す JSON Schema も、それが効いているかの確認方法 (schema_errors) も
        # 変わらない。
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(
            temperature=float(d["temperature"]), top_p=float(d["top_p"]),
            max_tokens=int(d["max_tokens"]), seed=int(d["seed"]),
            structured_outputs=StructuredOutputsParams(json=build_schema(mode)),
        )

    def run(self, requests: Iterable[dict[str, Any]], rep: int,
            out: Path, batch: int = 32) -> int:
        reqs = list(requests)
        if not reqs:
            return 0
        self._ensure()
        mode = "clip" if reqs[0]["mode"] == "clip" else "online"
        sp = self._sampling(mode)
        n = 0
        with out.open("a", encoding="utf-8") as f:
            for i in range(0, len(reqs), batch):
                chunk = reqs[i:i + batch]
                t0 = time.perf_counter()
                texts = self._generate(chunk, sp)
                dt = (time.perf_counter() - t0) / max(len(chunk), 1)
                for req, raw in zip(chunk, texts):
                    resp, note = parse_response(raw)
                    errs = validate(resp, mode) if resp else ["応答が空"]
                    f.write(json.dumps({
                        "request_id": req["request_id"], "event_id": req["event_id"],
                        "mode": req["mode"], "condition": req.get("condition", "C"),
                        "rep": rep, "t_eval": req.get("t_eval"), "t_rel": req.get("t_rel"),
                        "model": self.model_key,
                        "n_frames": req.get("n_frames", len(req.get("frames", []))),
                        "partial_window": bool(req.get("partial_window", False)),
                        "guard_s": req.get("guard_s"),
                        "prompt_version": self.cfg["prompt_version"],
                        "config_hash": req.get("config_hash"),
                        "schema_errors": errs, "parse_note": note,
                        "latency_s": round(dt, 3),
                        "response": resp, "raw": raw if errs else "",
                    }, ensure_ascii=False) + "\n")
                    n += 1
                f.flush()
                print(f"  {min(i + batch, len(reqs))}/{len(reqs)}  "
                      f"{dt:.2f} 秒/件", flush=True)
        return n

    def _generate(self, chunk: list[dict[str, Any]], sp) -> list[str]:
        if self.adapter.name == "echo":
            return [_echo(r, self.cfg) for r in chunk]
        inputs = []
        for r in chunk:
            prompt = build_prompt(r, self.cfg)
            frames = r.get("frames", [])
            # chat template と multi_modal_data は同じフレーム列から作る。
            # 別々に判断するとモダリティがずれて processor が失敗する。
            item: dict[str, Any] = {"prompt": self.adapter.chat_text(prompt, frames)}
            mm = self.adapter.multi_modal(frames)
            if mm:
                item["multi_modal_data"] = mm
            inputs.append(item)
        outs = self._llm.generate(inputs, sp)
        return [o.outputs[0].text for o in outs]


def _echo(req: dict[str, Any], cfg: dict[str, Any]) -> str:
    """schema に適合する固定応答。経路の確認だけに使う。"""
    d = {
        "scene": "(echo)", "ego_behavior": "(echo)", "other_agents": [],
        "state": "normal", "hazard_type": "none", "risk_level": 0, "risky": False,
        "difference_from_normal": "(echo)",
        "evidence": ("both" if req.get("frames") and req.get("can_text")
                     else "video" if req.get("frames") else "can"),
        "insufficient_information": False, "confidence": 0.5,
    }
    if req["mode"] != "clip":
        d |= {"t_eval_s": req.get("t_rel", 0.0),
              "change_from_previous": "(echo)", "expected_next": "(echo)"}
    return json.dumps(d, ensure_ascii=False)
