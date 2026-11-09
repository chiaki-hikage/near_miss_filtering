"""路面凹凸の確認タスクの推論。**vllm に依存するのはこのモジュール (と runner.py) だけ。**

既存のヒヤリハット判定には触らない。runner.Runner と adapters.QwenVLAdapter を
継承し、タスクに固有の 3 点だけを差し替える。

    schema   road.build_schema()      (ヒヤリハット判定の build_schema(mode) の代わり)
    prompt   road.build_prompt()      (prompt.build の代わり)
    画像     見出し付きで枚数分の枠を置く (QwenVLAdapter.chat_text は枠を 1 つしか置かない)

vLLM の起動・計測・再開の仕組みは Runner のものをそのまま使う。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import road
from .adapters import EchoAdapter, QwenVLAdapter, make_adapter
from .perf import sync
from .runner import Runner, _nullphase, parse_response


@dataclass
class RoadQwenAdapter(QwenVLAdapter):
    """画像を見出しと交互に並べる。multi_modal (全枚数を image で渡す) は親のまま。"""

    def chat_text(self, prompt: str, frames: list[str], captions: list[str] | None = None) -> str:
        if captions is not None and len(captions) != len(frames):
            raise ValueError(f"見出し {len(captions)} 件と画像 {len(frames)} 枚が合わない")
        content: list[dict[str, Any]] = []
        for i in range(len(frames)):
            if captions:
                content.append({"type": "text", "text": captions[i]})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        return self.processor().apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_road_adapter(name: str, model_id: str, cfg: dict[str, Any], spec: dict[str, Any] | None = None):
    """configs/vlm.yaml の models.<key> から adapter を作る。画像の列で渡す前提。"""
    if cfg.get("media_kind") != "images":
        raise ValueError("路面のタスクは media_kind: images (configs/vlm_road.yaml) を前提にしている")
    base = make_adapter(name, model_id, cfg, spec)
    if isinstance(base, EchoAdapter):
        return base
    return RoadQwenAdapter(model_id=base.model_id, input_kind="images", video_metadata=base.video_metadata,
                           video_fps=base.video_fps, min_video_frames=base.min_video_frames)


class RoadRunner(Runner):
    """Runner の起動と計測を使い、schema・prompt・echo だけを路面のものにする。"""

    def _sampling(self, mode: str):
        if self.adapter.name == "echo":
            return None
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        d = self.cfg["decode"]
        return SamplingParams(
            temperature=float(d["temperature"]), top_p=float(d["top_p"]),
            max_tokens=int(d["max_tokens"]), seed=int(d["seed"]),
            structured_outputs=StructuredOutputsParams(json=road.build_schema()),
        )

    def run(self, requests: Iterable[dict[str, Any]], rep: int, out: Path, batch: int = 32) -> int:
        reqs = list(requests)
        if not reqs:
            return 0
        bad = [r["request_id"] for r in reqs if r.get("task") != road.TASK]
        if bad:
            raise ValueError(f"路面のタスクでないリクエストが {len(bad)} 件ある (例 {bad[0]})")
        if self.meter is not None:
            with self.meter.phase("load"):
                self._ensure()
        else:
            self._ensure()
        sp = self._sampling("road")
        n = 0
        infer = self.meter.phase("infer") if self.meter is not None else _nullphase()
        with out.open("a", encoding="utf-8") as f, infer:
            for i in range(0, len(reqs), batch):
                chunk = reqs[i:i + batch]
                sync()
                t0 = time.perf_counter()
                texts = self._generate(chunk, sp)
                sync()
                batch_s = time.perf_counter() - t0
                dt = batch_s / max(len(chunk), 1)
                if self.meter is not None:
                    self.meter.add_batch(len(chunk), batch_s, sum(float(r.get("span_s") or 0.0) for r in chunk),
                                         [int(r.get("n_frames", len(r.get("frames", [])))) for r in chunk])
                for req, raw in zip(chunk, texts):
                    resp, note = parse_response(raw)
                    errs = road.validate(resp) if resp else ["応答が空"]
                    f.write(json.dumps({
                        "request_id": req["request_id"], "event_id": req["event_id"], "task": road.TASK,
                        "mode": req["mode"], "condition": req["condition"], "rep": rep,
                        "model": self.model_key, "n_frames": req.get("n_frames", len(req.get("frames", []))),
                        "prompt_version": self.cfg["prompt_version"], "config_hash": req.get("config_hash"),
                        "schema_errors": errs, "parse_note": note, "latency_s": round(dt, 3),
                        "response": resp, "raw": raw if errs else "",
                    }, ensure_ascii=False) + "\n")
                    n += 1
                f.flush()
                print(f"  {min(i + batch, len(reqs))}/{len(reqs)}  {dt:.2f} 秒/件", flush=True)
        return n

    def _generate(self, chunk: list[dict[str, Any]], sp) -> list[str]:
        if self.adapter.name == "echo":
            return [road.echo(r) for r in chunk]
        inputs = []
        for r in chunk:
            frames = r["frames"]
            item: dict[str, Any] = {"prompt": self.adapter.chat_text(road.build_prompt(r, self.cfg), frames,
                                                                     road.captions(r))}
            mm = self.adapter.multi_modal(frames)
            if mm:
                item["multi_modal_data"] = mm
            inputs.append(item)
        outs = self._llm.generate(inputs, sp)
        return [o.outputs[0].text for o in outs]
