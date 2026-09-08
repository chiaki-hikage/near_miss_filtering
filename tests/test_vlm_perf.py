"""計測 (near_miss.vlm.perf) の試験。GPU も vllm も要らない。

GPU の読み出しは差し替え可能な形にしてあるので、偽の GPU を挿して
「ピークをきちんと拾えるか」「区間の切り出しが合っているか」を確かめる。
"""

from __future__ import annotations

import json
import time

import pytest

from near_miss.vlm import perf
from near_miss.vlm.runner import Runner


class FakeGpu:
    """使用量が呼ぶたびに変わる GPU。ピーク検出の確認に使う。"""

    source = "fake"

    def __init__(self, series: list[float]) -> None:
        self.devices = [{"index": 0, "name": "FakeGPU", "total_mb": 98304.0}]
        self.series = list(series)
        self.i = 0

    @property
    def ok(self) -> bool:
        return True

    def used_mb(self) -> list[float]:
        v = self.series[min(self.i, len(self.series) - 1)]
        self.i += 1
        return [v]

    def util_pct(self) -> list[float]:
        return [50.0]


class FakeHost:
    """CPU 使用率と RAM が呼ぶたびに変わるホスト。"""

    source = "fake"
    n_cpu = 8

    def __init__(self, cpu: list[float], ram: list[float], rss: list[float]) -> None:
        self.cpu, self.ram_used, self.rss = list(cpu), list(ram), list(rss)
        self.i = 0

    @property
    def ok(self) -> bool:
        return True

    def _take(self, seq):
        return seq[min(self.i, len(seq) - 1)]

    def cpu_pct(self) -> float:
        return self._take(self.cpu)

    def ram(self) -> dict:
        return {"used_mb": self._take(self.ram_used), "total_mb": 64000.0, "pct": 50.0}

    def rss_mb(self) -> float:
        v = self._take(self.rss)
        self.i += 1          # 1 標本 = 1 回の読み出しとして進める
        return v


CFG = {
    "timeline": {"stride_s": 0.5},
    "decode": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 0},
    "prompt_version": "v1",
}


def test_video_s_online_is_stride():
    """オンライン判定は 1 時刻で stride_s ぶんの映像が進む。"""
    assert perf.request_video_s({"mode": "online"}, CFG) == 0.5


def test_video_s_clip_is_span():
    """一括判定は区間そのもの。span_s が無ければ 0 (時間だけ測る)。"""
    assert perf.request_video_s({"mode": "clip", "span_s": 29.2}, CFG) == 29.2
    assert perf.request_video_s({"mode": "clip"}, CFG) == 0.0


def test_sampler_picks_peak_in_window():
    gpu = FakeGpu([1000, 2000, 9000, 3000, 1500])
    s = perf.Sampler(gpu, interval=0.01)
    t0 = time.perf_counter()
    for _ in range(5):
        s.sample_once()
    t1 = time.perf_counter()
    w = s.window(t0, t1)
    assert w["peak_used_mb"] == [9000.0]
    assert w["n_samples"] == 5
    assert w["max_util_pct"] == 50.0


def test_sampler_window_outside_range_falls_back_to_nearest():
    """区間が採取間隔より短いと標本が 1 つも入らない。空で返さない。"""
    gpu = FakeGpu([4242])
    s = perf.Sampler(gpu, interval=0.01)
    s.sample_once()
    w = s.window(time.perf_counter() + 10, time.perf_counter() + 11)
    assert w["peak_used_mb"] == [4242.0]


def test_meter_records_load_and_throughput(tmp_path):
    # 採取間隔を長くして背景スレッドが値を消費しないようにする。
    # 読み出し順: begin / sampler 起動 / after_load / 区間内 2 回 / stop
    m = perf.Meter(model_key="m", model_id="id", mode="b", interval=60.0,
                   gpu=FakeGpu([1000, 1000, 70000, 71000, 60000, 55000]))
    m.begin()
    m.after_load(12.5, llm=None)
    with m.phase("infer"):
        m.sampler.sample_once()
        m.add_batch(4, 8.0, 4 * 0.5, [8, 8, 8, 8])
        m.sampler.sample_once()
    m.sampler.stop()

    s = m.summary()
    t = s["時間"]
    assert t["model_load_s"] == 12.5
    assert t["n_requests"] == 4 and t["n_batches"] == 1
    assert t["per_request_mean_s"] == 2.0
    assert t["video_s"] == 2.0
    # 8 秒で映像 2 秒 -> 映像 1 分あたり 240 秒
    assert t["s_per_video_min"] == 240.0
    assert t["realtime_factor"] == 0.25

    g = s["GPUメモリ"]
    assert g["baseline_used_mb"] == [1000.0]
    assert g["after_load_used_mb"] == [70000.0]   # 重み + KV キャッシュ確保後
    assert g["infer_peak_used_mb"] == [71000.0]
    assert s["条件"]["n_frames"]["total"] == 32
    assert s["条件"]["gpu"][0]["name"] == "FakeGPU"

    out = m.write(tmp_path / "perf.json")
    assert json.loads(out.read_text(encoding="utf-8"))["時間"]["video_s"] == 2.0
    assert "FakeGPU" in m.report()


def test_meter_report_without_gpu():
    """GPU が無い機械 (Mac) でも落ちず、時間だけ出す。"""
    m = perf.Meter(model_key="echo", model_id="echo", mode="a", gpu=_NoGpu())
    m.begin()
    m.add_batch(1, 0.5, 10.0, [16])
    r = m.report()
    assert "未計測" in r
    assert m.summary()["時間"]["n_requests"] == 1


class _NoGpu:
    source = ""
    devices: list = []

    @property
    def ok(self) -> bool:
        return False


def test_runner_keeps_concurrency_separate_from_batch():
    """--batch と max_num_seqs は別物。Runner が両方を保持する。"""
    from near_miss.vlm.adapters import make_adapter
    r = Runner("echo", CFG, make_adapter("echo", "echo", CFG),
               max_num_seqs=512, max_num_batched_tokens=16384)
    assert r._max_num_seqs == 512
    assert r._max_num_batched_tokens == 16384
    # 既定は vLLM 任せ (None なら LLM() に渡さない)
    assert Runner("echo", CFG, make_adapter("echo", "echo", CFG))._max_num_seqs is None


def test_runner_meter_counts_every_request(tmp_path):
    """echo 経路で Runner に挿しても、判定の出力は変わらず計測だけ増える。"""
    cfg = dict(CFG) | {"repeats": {"mode_b": 1}}
    reqs = [{"request_id": f"E{i}|B|0.0", "event_id": f"E{i}", "mode": "online",
             "t_rel": 0.0, "frames": [], "n_frames": 8, "can_text": "x"}
            for i in range(5)]

    from near_miss.vlm.adapters import make_adapter
    adapter = make_adapter("echo", "echo", cfg)

    plain_out = tmp_path / "plain.jsonl"
    Runner("echo", cfg, adapter).run(list(reqs), 0, plain_out, batch=2)

    m = perf.Meter(model_key="echo", model_id="echo", mode="b", gpu=_NoGpu())
    metered_out = tmp_path / "metered.jsonl"
    Runner("echo", cfg, adapter, meter=m).run(list(reqs), 0, metered_out, batch=2)

    # 計測を入れても結果は 1 バイトも変わらない
    assert plain_out.read_bytes() == metered_out.read_bytes()
    assert m.n_requests == 5 and m.n_batches == 3
    assert m.video_s == pytest.approx(2.5)
    assert m.n_frames == [8] * 5


def test_sampler_tracks_cpu_ram_and_rss():
    host = FakeHost(cpu=[10, 90, 30], ram=[8000, 20000, 9000],
                    rss=[500, 40000, 600])
    s = perf.Sampler(_NoGpu(), interval=60.0, host=host)   # GPU 無しの機械
    t0 = time.perf_counter()
    for _ in range(3):
        s.sample_once()
    w = s.window(t0, time.perf_counter())
    assert w["max_cpu"] == 90.0
    assert w["mean_cpu"] == pytest.approx(43.3, abs=0.1)
    assert w["max_ram_used_mb"] == 20000.0
    assert w["max_rss_mb"] == 40000.0
    assert w["ram_total_mb"] == 64000.0


def test_meter_reports_cpu_and_ram():
    m = perf.Meter(model_key="m", model_id="id", mode="b", interval=60.0,
                   gpu=_NoGpu())
    m.host = FakeHost(cpu=[5, 5, 95], ram=[8000, 8000, 30000],
                      rss=[400, 400, 45000])
    m.sampler = perf.Sampler(m.gpu, 60.0, m.host)
    m.begin()
    with m.phase("infer"):
        m.sampler.sample_once()
        m.add_batch(1, 2.0, 0.5, [8])
    m.sampler.stop()

    h = m.summary()["CPU_RAM"]
    assert h["baseline_cpu_pct"] == 5.0
    assert h["baseline_ram_used_mb"] == 8000.0
    assert h["infer_max_cpu_pct"] == 95.0
    assert h["infer_peak_ram_used_mb"] == 30000.0
    assert h["infer_peak_rss_mb"] == 45000.0
    assert "CPU 使用率" in m.report() and "RSS" in m.report()


def test_host_reads_real_machine_or_reports_nothing():
    """本物のホスト。Linux では値が採れ、採れない機械では ok=False になる。

    どちらでも落ちないことだけを担保する (値そのものは機械依存)。
    """
    h = perf.Host()
    if not h.ok:
        pytest.skip("psutil も /proc も無い機械")
    cpu = h.cpu_pct()
    assert cpu is None or 0.0 <= cpu <= 100.0 * (h.n_cpu or 1)
    ram = h.ram()
    assert ram["total_mb"] > 0 and 0 <= ram["used_mb"] <= ram["total_mb"]


def test_peak_rss_is_positive():
    pk = perf.peak_rss_mb()
    if pk:
        assert pk["self_mb"] > 0


def test_first_batch_excluded_mean():
    """立ち上がりの遅い 1 バッチ目を除いた平均も出す。"""
    m = perf.Meter(model_key="m", model_id="id", mode="b", gpu=_NoGpu())
    m.add_batch(2, 20.0, 1.0, [8, 8])     # 1 バッチ目: 10 秒/件
    m.add_batch(2, 4.0, 1.0, [8, 8])      # 以降:        2 秒/件
    m.add_batch(2, 4.0, 1.0, [8, 8])
    t = m.summary()["時間"]
    assert t["per_request_mean_s"] == pytest.approx(4.667, abs=0.001)
    assert t["per_request_mean_excl_first_s"] == 2.0
    assert t["first_batch_s"] == 20.0


def test_single_batch_has_no_excl_first():
    m = perf.Meter(model_key="m", model_id="id", mode="b", gpu=_NoGpu())
    m.add_batch(2, 20.0, 1.0, [8, 8])
    assert m.summary()["時間"]["per_request_mean_excl_first_s"] is None


def _write(p: Path, n: int) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * n)


def test_model_disk_local_directory(tmp_path):
    """ローカルのパスを渡した場合。重みファイルだけの合計も出す。"""
    _write(tmp_path / "model-00001.safetensors", 3 * 1024 * 1024)
    _write(tmp_path / "model-00002.safetensors", 1 * 1024 * 1024)
    _write(tmp_path / "tokenizer.json", 512 * 1024)
    d = perf.model_disk(str(tmp_path))
    assert d["weights_mb"] == 4.0
    assert d["total_mb"] == 4.5
    assert d["n_weight_files"] == 2 and d["n_files"] == 3


def test_model_disk_hf_cache_counts_blob_once(tmp_path, monkeypatch):
    """HF のキャッシュは snapshots が blobs への symlink。二重に数えない。"""
    repo = tmp_path / "models--Qwen--Qwen3-VL-8B-Instruct"
    blob = repo / "blobs" / "abc123"
    _write(blob, 6 * 1024 * 1024)
    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    d = perf.model_disk("Qwen/Qwen3-VL-8B-Instruct")
    assert d["total_mb"] == 6.0          # 12.0 になっていたら重複して数えている
    assert d["weights_mb"] == 6.0
    assert d["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_model_disk_missing_returns_empty(tmp_path, monkeypatch):
    """見つからないときは空。推測で埋めない。"""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nope"))
    assert perf.model_disk("Qwen/does-not-exist") == {}


def test_weights_appear_in_report(tmp_path, monkeypatch):
    _write(tmp_path / "model.safetensors", 17 * 1024 * 1024)
    m = perf.Meter(model_key="m", model_id=str(tmp_path), mode="b", gpu=_NoGpu())
    m.after_load(1.0, llm=None)
    assert m.summary()["条件"]["weights"]["weights_mb"] == 17.0
    assert "重み" in m.report()
