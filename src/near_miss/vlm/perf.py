"""計測だけを行う。**判定そのものには触らない。**

推論の中身 (prompt / schema / 復号 / adapter) は一切変えず、時間と GPU メモリを
外から測る。測る対象は 3 つ。

    モデルロード時間      vLLM の LLM() 構築 (重みの読み込み + KV キャッシュ確保)
    推論時間              1 バッチの壁時計時間から 1 件あたりを出す
    GPU メモリ            起動前 / ロード後の常時 / 推論中のピーク

**GPU メモリは NVML (実使用量) を主に見る。** vLLM は既定で engine を別プロセスに
置くため、こちら側の ``torch.cuda.max_memory_allocated`` は 0 に近い値しか返さない。
同じプロセスで動かしたい場合は ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` を渡す。
PyTorch 側の値も取れるだけ取って併記するが、**主はデバイス実使用量**とする。

**vLLM は gpu_memory_utilization の割合を起動時に確保する。**
既定 0.85 なら 96 GB のうち約 82 GB を先に取るので、推論中のピークは
「そのモデルが要る量」ではなく「確保した量」を見ていることになる。
必要量を知りたいときは ``ロード後 - 起動前`` ではなく、
出力に併記する ``kv_cache`` の項と ``--gpu-util`` を下げた実行を突き合わせること。
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MB = 1024.0 * 1024.0


# --------------------------------------------------------------------------
# NVML / nvidia-smi
# --------------------------------------------------------------------------

def _nvml():
    """NVML のハンドル。取れなければ None。

    vllm は nvidia-ml-py に依存しているので、EC2 側では大抵入っている。
    無ければ nvidia-smi の呼び出しに落とす。
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    return pynvml


def _smi(query: str) -> list[list[str]] | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")]
            for line in r.stdout.strip().splitlines() if line.strip()]


class Gpu:
    """GPU の情報と使用量。NVML があればそれを、無ければ nvidia-smi を使う。

    どちらも無い機械 (Mac) では ``ok`` が False になり、時間の計測だけが残る。
    """

    def __init__(self) -> None:
        self.nvml = _nvml()
        self.source = "nvml" if self.nvml else ("nvidia-smi" if _smi("name") else "")
        self.devices: list[dict[str, Any]] = self._static()

    @property
    def ok(self) -> bool:
        return bool(self.devices)

    def _static(self) -> list[dict[str, Any]]:
        if self.nvml:
            n = self.nvml.nvmlDeviceGetCount()
            out = []
            for i in range(n):
                h = self.nvml.nvmlDeviceGetHandleByIndex(i)
                name = self.nvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = self.nvml.nvmlDeviceGetMemoryInfo(h)
                out.append({"index": i, "name": name,
                            "total_mb": round(mem.total / MB, 1)})
            try:
                drv = self.nvml.nvmlSystemGetDriverVersion()
                if isinstance(drv, bytes):
                    drv = drv.decode()
                for d in out:
                    d["driver"] = drv
            except Exception:
                pass
            return out
        rows = _smi("index,name,memory.total,driver_version")
        if not rows:
            return []
        return [{"index": int(r[0]), "name": r[1],
                 "total_mb": float(r[2]), "driver": r[3]} for r in rows]

    def used_mb(self) -> list[float]:
        """各 GPU の実使用量。**プロセスを問わないデバイス全体の値。**"""
        if self.nvml:
            out = []
            for i in range(len(self.devices)):
                h = self.nvml.nvmlDeviceGetHandleByIndex(i)
                out.append(self.nvml.nvmlDeviceGetMemoryInfo(h).used / MB)
            return out
        rows = _smi("memory.used")
        return [float(r[0]) for r in rows] if rows else []

    def util_pct(self) -> list[float]:
        if self.nvml:
            out = []
            for i in range(len(self.devices)):
                h = self.nvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    out.append(float(self.nvml.nvmlDeviceGetUtilizationRates(h).gpu))
                except Exception:
                    out.append(float("nan"))
            return out
        rows = _smi("utilization.gpu")
        return [float(r[0]) for r in rows] if rows else []


# --------------------------------------------------------------------------
# CPU と RAM
# --------------------------------------------------------------------------

class Host:
    """CPU 使用率と RAM 使用量。

    ``psutil`` があればそれを使う (vllm が依存しているので EC2 側では入っている)。
    無ければ Linux の ``/proc`` を直接読む。**top を 0.2 秒ごとに起動はしない。**
    top が出す値と同じもの (/proc/stat の差分、/proc/meminfo) を直接読んでいる。

    CPU 使用率は**機械全体**の値で、前回呼んだときからの差分。
    vLLM は engine を別プロセスに置くので、自プロセスだけを見ても実態が分からない。
    """

    def __init__(self) -> None:
        try:
            import psutil
            self.psutil = psutil
            self.source = "psutil"
        except ImportError:
            self.psutil = None
            self.source = "proc" if Path("/proc/stat").is_file() else ""
        self._prev: tuple[float, float] | None = None
        self.n_cpu = os.cpu_count()
        if self.psutil is not None:
            self.psutil.cpu_percent(None)      # 1 回目は 0 を返すので捨てる
        else:
            self.cpu_pct()

    @property
    def ok(self) -> bool:
        return bool(self.source)

    def cpu_pct(self) -> float | None:
        """機械全体の CPU 使用率 [%]。前回の呼び出しからの差分。"""
        if self.psutil is not None:
            return float(self.psutil.cpu_percent(None))
        if self.source != "proc":
            return None
        try:
            line = Path("/proc/stat").read_text().splitlines()[0]
        except OSError:
            return None
        v = [float(x) for x in line.split()[1:]]
        total, idle = sum(v), v[3] + (v[4] if len(v) > 4 else 0.0)
        prev, self._prev = self._prev, (total, idle)
        if prev is None or total <= prev[0]:
            return None
        return round(100.0 * (1.0 - (idle - prev[1]) / (total - prev[0])), 1)

    def ram(self) -> dict[str, float]:
        """機械全体の RAM [MB]。"""
        if self.psutil is not None:
            m = self.psutil.virtual_memory()
            return {"used_mb": round((m.total - m.available) / MB, 1),
                    "total_mb": round(m.total / MB, 1), "pct": float(m.percent)}
        if self.source != "proc":
            return {}
        try:
            kv = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, rest = line.partition(":")
                kv[k] = float(rest.split()[0]) / 1024.0     # kB -> MB
        except OSError:
            return {}
        total = kv.get("MemTotal", 0.0)
        avail = kv.get("MemAvailable", kv.get("MemFree", 0.0))
        return {"used_mb": round(total - avail, 1), "total_mb": round(total, 1),
                "pct": round(100.0 * (total - avail) / total, 1) if total else 0.0}

    def rss_mb(self) -> float | None:
        """この Python プロセスと子プロセスの常駐メモリ合計 [MB]。

        vLLM の engine は子プロセスなので、自分だけ見ても足りない。
        """
        if self.psutil is None:
            return None
        try:
            me = self.psutil.Process()
            total = me.memory_info().rss
            for c in me.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except Exception:
                    pass
            return round(total / MB, 1)
        except Exception:
            return None


def peak_rss_mb() -> dict[str, float]:
    """OS が数えたピーク常駐メモリ [MB]。自プロセスと子プロセスで別々に出る。

    ru_maxrss の単位は Linux が kB、macOS が byte。
    """
    try:
        import resource
    except ImportError:
        return {}
    unit = 1.0 if platform.system() == "Darwin" else 1024.0
    out = {}
    for key, who in (("self_mb", "RUSAGE_SELF"), ("children_mb", "RUSAGE_CHILDREN")):
        try:
            out[key] = round(resource.getrusage(getattr(resource, who)).ru_maxrss
                             * unit / MB, 1)
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------
# PyTorch 側
# --------------------------------------------------------------------------

def sync() -> None:
    """CUDA の実行完了を待つ。

    GPU の呼び出しは非同期なので、待たずに時間を測ると転送や起動の時間しか
    測れない。**vLLM が engine を別プロセスに置いている場合はここでは待てない**
    (そのときは ``llm.generate`` の戻り自体が同期点になる) が、
    同一プロセス実行 (VLLM_ENABLE_V1_MULTIPROCESSING=0) では意味がある。
    """
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def torch_reset_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def torch_mem() -> dict[str, float] | None:
    """このプロセスの PyTorch アロケータの値 [MB]。

    別プロセス engine ではほぼ 0 になる。0 だからメモリを使っていない、
    ではないので、出力では NVML の値と必ず並べて出す。
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / MB, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / MB, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / MB, 1),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / MB, 1),
    }


def torch_env() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    d: dict[str, Any] = {"torch": torch.__version__,
                         "available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        d["cuda"] = torch.version.cuda
        d["device_name"] = torch.cuda.get_device_name(0)
        d["capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    return d


def engine_info(llm: Any) -> dict[str, Any]:
    """vLLM から dtype などを引く。版で置き場所が変わるので候補を順に試す。

    取れなくても計測は続ける。**推測で埋めない** (取れなければ項目を出さない)。
    """
    if llm is None:
        return {}
    mc = None
    for path in (("llm_engine", "vllm_config", "model_config"),
                 ("llm_engine", "model_config"),
                 ("vllm_config", "model_config")):
        obj: Any = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            mc = obj
            break
    out: dict[str, Any] = {}
    if mc is not None:
        for key, attr in (("dtype", "dtype"), ("max_model_len", "max_model_len"),
                          ("quantization", "quantization"),
                          ("served_model", "model")):
            v = getattr(mc, attr, None)
            if v is not None:
                out[key] = str(v)
    # 実際に効いた同時実行数。こちらの指定が通ったかを後から確かめられるように。
    for path in (("llm_engine", "vllm_config", "scheduler_config"),
                 ("llm_engine", "scheduler_config")):
        obj: Any = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            for k in ("max_num_seqs", "max_num_batched_tokens"):
                v = getattr(obj, k, None)
                if v is not None:
                    out[k] = v
            break
    for path, key in ((("llm_engine", "vllm_config", "cache_config"), "cache"),
                      (("llm_engine", "cache_config"), "cache")):
        obj: Any = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            for k, attr in (("kv_cache_dtype", "cache_dtype"),
                            ("gpu_memory_utilization", "gpu_memory_utilization"),
                            ("num_gpu_blocks", "num_gpu_blocks"),
                            ("block_size", "block_size")):
                v = getattr(obj, attr, None)
                if v is not None:
                    out[k] = v if isinstance(v, (int, float)) else str(v)
            break
    return out


def model_disk(model_id: str) -> dict[str, Any]:
    """重みのディスク上の大きさ [MB]。

    ローカルのパスならそのディレクトリを、Hugging Face のリポジトリ名なら
    HF のキャッシュを見る。**取れなければ空を返す** (推測で埋めない)。

    重みファイル (safetensors / bin) だけの合計も別に出す。リポジトリには
    tokenizer や設定、元の .bin と .safetensors の両方が入っていることがあり、
    総量は「モデルの大きさ」より大きく出るため。

    bf16 で読む場合、この重みファイルの合計が VRAM 上の常駐量にほぼ一致する
    (KV キャッシュと活性化は別)。
    """
    WEIGHT = (".safetensors", ".bin", ".pt", ".pth", ".gguf")

    def walk(root: Path) -> dict[str, Any]:
        seen: set[tuple[int, int]] = set()
        total = weights = 0
        n_files = n_weight = 0
        for f in root.rglob("*"):
            try:
                stx = f.stat()          # symlink は実体を見る (HF は blobs への link)
            except OSError:
                continue
            if not f.is_file():
                continue
            key = (stx.st_dev, stx.st_ino)
            if key in seen:             # 同じ blob への複数の link を二重に数えない
                continue
            seen.add(key)
            total += stx.st_size
            n_files += 1
            if f.suffix in WEIGHT:
                weights += stx.st_size
                n_weight += 1
        if not n_files:
            return {}
        return {"path": str(root), "total_mb": round(total / MB, 1),
                "weights_mb": round(weights / MB, 1),
                "n_files": n_files, "n_weight_files": n_weight}

    local = Path(model_id)
    if local.is_dir():
        return walk(local)

    # HF のキャッシュ。環境変数の指定を順に見る。
    roots = [os.environ.get("HF_HUB_CACHE"),
             (os.environ.get("HF_HOME") or "") + "/hub" if os.environ.get("HF_HOME") else None,
             str(Path.home() / ".cache" / "huggingface" / "hub")]
    name = "models--" + model_id.replace("/", "--")
    for r in roots:
        if not r:
            continue
        repo = Path(r) / name
        if repo.is_dir():
            d = walk(repo)
            if d:
                d["model_id"] = model_id
                return d
    return {}


# --------------------------------------------------------------------------
# 標本採取
# --------------------------------------------------------------------------

@dataclass
class Phase:
    name: str
    t0: float
    t1: float = float("nan")


class Sampler:
    """別スレッドで GPU / CPU / RAM を定期的に採る。

    区間ごとのピークを後から出せるように、時刻付きで全部持っておく。
    0.2 秒間隔なら 1 時間で 18,000 点。持っていても差し支えない量。
    """

    def __init__(self, gpu: Gpu, interval: float = 0.2,
                 host: Host | None = None) -> None:
        self.gpu = gpu
        self.host = host if host is not None else Host()
        self.interval = float(interval)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    @property
    def ok(self) -> bool:
        return bool(self.gpu.ok or self.host.ok)

    def sample_once(self) -> None:
        if not self.ok:
            return
        row: dict[str, Any] = {"t": time.perf_counter()}
        try:
            if self.gpu.ok:
                row["gpu_used_mb"] = self.gpu.used_mb()
                row["gpu_util"] = self.gpu.util_pct()
            if self.host.ok:
                cpu = self.host.cpu_pct()
                if cpu is not None:
                    row["cpu_pct"] = cpu
                ram = self.host.ram()
                if ram:
                    row["ram_used_mb"] = ram["used_mb"]
                    row["ram_total_mb"] = ram["total_mb"]
                rss = self.host.rss_mb()
                if rss is not None:
                    row["rss_mb"] = rss
        except Exception:
            return
        self.samples.append(row)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample_once()

    def start(self) -> None:
        if not self.ok or self._th is not None:
            return
        self.sample_once()
        self._th = threading.Thread(target=self._loop, daemon=True,
                                    name="vlm-perf-sampler")
        self._th.start()

    def stop(self) -> None:
        if self._th is None:
            return
        self._stop.set()
        self._th.join(timeout=2.0)
        self._th = None
        self.sample_once()

    def window(self, t0: float, t1: float) -> dict[str, Any]:
        """[t0, t1] の区間の統計。標本が 1 つも無ければ空。"""
        rows = [r for r in self.samples if t0 <= r["t"] <= t1]
        if not rows and self.samples:
            # 区間が採取間隔より短い。直近の 1 点で代表させる。
            rows = [min(self.samples, key=lambda r: abs(r["t"] - t1))]
        if not rows:
            return {}
        d: dict[str, Any] = {"n_samples": len(rows)}

        gpu_rows = [r["gpu_used_mb"] for r in rows if "gpu_used_mb" in r]
        if gpu_rows:
            n = max(len(g) for g in gpu_rows)
            d["peak_used_mb"] = [round(max((g[i] for g in gpu_rows if i < len(g)),
                                           default=0.0), 1) for i in range(n)]
        util = [u for r in rows for u in r.get("gpu_util", []) if u == u]
        if util:
            d["mean_util_pct"] = round(sum(util) / len(util), 1)
            d["max_util_pct"] = round(max(util), 1)
        d |= _stat(rows, "cpu_pct", "cpu")
        d |= _stat(rows, "ram_used_mb", "ram_used_mb")
        d |= _stat(rows, "rss_mb", "rss_mb")
        tot = [r["ram_total_mb"] for r in rows if "ram_total_mb" in r]
        if tot:
            d["ram_total_mb"] = tot[-1]
        return d


def _stat(rows: list[dict[str, Any]], key: str, out: str) -> dict[str, float]:
    """区間内の平均と最大。値が無い項目は出さない (0 で埋めない)。"""
    v = [r[key] for r in rows if key in r]
    if not v:
        return {}
    return {f"mean_{out}": round(sum(v) / len(v), 1), f"max_{out}": round(max(v), 1)}


# --------------------------------------------------------------------------
# 集計
# --------------------------------------------------------------------------

@dataclass
class Meter:
    """1 回の実行ぶんの計測。

    ``video_s`` は「判定が覆った映像の長さ」。オンライン判定では 1 時刻ごとに
    stride_s だけ進むので stride_s、一括判定では区間の長さ span_s を足す。
    これで「映像 1 分あたり何秒かかるか」= 実時間比が出る。
    """

    model_key: str
    model_id: str
    mode: str
    conditions: str = ""
    gpu: Gpu = field(default_factory=Gpu)
    interval: float = 0.2
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def __post_init__(self) -> None:
        self.host = Host()
        self.sampler = Sampler(self.gpu, self.interval, self.host)
        self.phases: list[Phase] = []
        self.load_s: float | None = None
        self.n_requests = 0
        self.n_batches = 0
        self.infer_s = 0.0
        self.video_s = 0.0
        self.per_request_s: list[float] = []
        self.batches: list[dict[str, float]] = []
        self.n_frames: list[int] = []
        self.baseline: dict[str, Any] = {}
        self.loaded: dict[str, Any] = {}
        self.engine: dict[str, Any] = {}
        self.weights: dict[str, Any] = {}
        self.settings: dict[str, Any] = {}
        self.t_start = time.perf_counter()

    # -- 進行 ---------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        """今この瞬間の GPU メモリ / CPU / RAM。"""
        d: dict[str, Any] = {}
        if self.gpu.ok:
            d["gpu_used_mb"] = [round(v, 1) for v in self.gpu.used_mb()]
        if self.host.ok:
            cpu = self.host.cpu_pct()
            if cpu is not None:
                d["cpu_pct"] = cpu
            d |= self.host.ram()
            rss = self.host.rss_mb()
            if rss is not None:
                d["rss_mb"] = rss
        return d

    def begin(self) -> None:
        """モデルを読む前。ここが「通常時」の基準になる。"""
        self.baseline = self._snapshot()
        self.sampler.start()

    @contextmanager
    def phase(self, name: str):
        ph = Phase(name, time.perf_counter())
        self.phases.append(ph)
        try:
            yield ph
        finally:
            ph.t1 = time.perf_counter()

    def after_load(self, seconds: float, llm: Any = None) -> None:
        """モデルロード直後。推論を始める前の常時使用量をここで採る。"""
        self.load_s = float(seconds)
        self.loaded = self._snapshot()
        self.engine = engine_info(llm)
        self.weights = model_disk(self.model_id)
        torch_reset_peak()

    def add_batch(self, n: int, seconds: float, video_s: float,
                  frames: list[int]) -> None:
        self.n_batches += 1
        self.n_requests += n
        self.infer_s += float(seconds)
        self.video_s += float(video_s)
        self.per_request_s.extend([seconds / max(n, 1)] * n)
        self.batches.append({"n": n, "s": float(seconds), "video_s": float(video_s)})
        self.n_frames.extend(frames)

    # -- 出力 ---------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        wall = time.perf_counter() - self.t_start
        per = sorted(self.per_request_s)
        infer_phase = [p for p in self.phases if p.name == "infer"]
        win = {}
        if infer_phase:
            win = self.sampler.window(infer_phase[0].t0, infer_phase[-1].t1)
        d: dict[str, Any] = {
            "条件": {
                "model_key": self.model_key,
                "model_id": self.model_id,
                "mode": self.mode,
                "conditions": self.conditions,
                "started_at": self.started_at,
                "host": platform.node(),
                "python": platform.python_version(),
                "gpu": self.devices_note(),
                "torch": torch_env(),
                "engine": self.engine,
                # 重みのディスク上の大きさ。bf16 なら VRAM 常駐量にほぼ一致する
                "weights": self.weights,
                "settings": self.settings,
                "n_frames": self._frames_note(),
            },
            "時間": {
                "model_load_s": round(self.load_s, 2) if self.load_s is not None else None,
                "infer_total_s": round(self.infer_s, 2),
                "wall_total_s": round(wall, 2),
                "n_requests": self.n_requests,
                "n_batches": self.n_batches,
                "per_request_mean_s": round(self.infer_s / self.n_requests, 3) if self.n_requests else None,
                "per_request_median_s": round(per[len(per) // 2], 3) if per else None,
                "per_request_min_s": round(per[0], 3) if per else None,
                "per_request_max_s": round(per[-1], 3) if per else None,
                # 最初のバッチは CUDA graph の取り込みや割り当てで遅くなる。
                # 定常の速度を見たいときは「1 バッチ目を除く」ほうを使う。
                "first_batch_s": (round(self.batches[0]["s"], 2)
                                  if self.batches else None),
                "first_batch_n": int(self.batches[0]["n"]) if self.batches else None,
                "per_request_mean_excl_first_s": self._mean_excl_first(),
                "video_s": round(self.video_s, 1),
                "s_per_video_min": (round(self.infer_s / (self.video_s / 60.0), 1)
                                    if self.video_s > 0 else None),
                "realtime_factor": (round(self.video_s / self.infer_s, 3)
                                    if self.infer_s > 0 else None),
            },
            "GPUメモリ": {
                "source": self.gpu.source or "(取得できず)",
                "baseline_used_mb": self.baseline.get("gpu_used_mb", []),
                "after_load_used_mb": self.loaded.get("gpu_used_mb", []),
                "infer_peak_used_mb": win.get("peak_used_mb"),
                "infer_mean_util_pct": win.get("mean_util_pct"),
                "infer_max_util_pct": win.get("max_util_pct"),
                "n_samples": win.get("n_samples"),
                "sample_interval_s": self.interval,
                "torch_process": torch_mem(),
                "注記": ("vLLM は gpu_memory_utilization の割合を起動時に確保するため、"
                         "ピークは所要量ではなく確保量に近い。"
                         "torch_process はこのプロセスの値で、engine が別プロセスなら 0 に近い "
                         "(VLLM_ENABLE_V1_MULTIPROCESSING=0 で同一プロセスにできる)"),
            },
            "CPU_RAM": {
                "source": self.host.source,
                "n_cpu": self.host.n_cpu,
                # CPU 使用率は機械全体の値。vLLM の engine は別プロセスなので
                # 自プロセスだけを見ても実態が分からない。
                "baseline_cpu_pct": self.baseline.get("cpu_pct"),
                "infer_mean_cpu_pct": win.get("mean_cpu"),
                "infer_max_cpu_pct": win.get("max_cpu"),
                "baseline_ram_used_mb": self.baseline.get("used_mb"),
                "after_load_ram_used_mb": self.loaded.get("used_mb"),
                "infer_mean_ram_used_mb": win.get("mean_ram_used_mb"),
                "infer_peak_ram_used_mb": win.get("max_ram_used_mb"),
                "ram_total_mb": win.get("ram_total_mb") or self.baseline.get("total_mb"),
                # 自分と子プロセス (vLLM engine) の常駐メモリ合計
                "baseline_rss_mb": self.baseline.get("rss_mb"),
                "infer_peak_rss_mb": win.get("max_rss_mb"),
                "os_peak_rss_mb": peak_rss_mb(),
            },
            # 閉じていない区間 (t1 が NaN のまま) は出さない
            "phases": [{"name": p.name, "s": round(p.t1 - p.t0, 3)}
                       for p in self.phases if not math.isnan(p.t1)],
        }
        return d

    def _mean_excl_first(self) -> float | None:
        """1 バッチ目を除いた 1 件あたりの時間。バッチが 1 つだけなら出さない。"""
        rest = self.batches[1:]
        n = sum(b["n"] for b in rest)
        return round(sum(b["s"] for b in rest) / n, 3) if n else None

    def devices_note(self) -> list[dict[str, Any]]:
        """GPU の一覧。CUDA_VISIBLE_DEVICES で絞っている場合はそれも残す。

        NVML は絞りを無視して機械にある全 GPU を返すため、
        どれを使ったのかが後から分からなくなる。
        """
        out = list(self.gpu.devices)
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            out.append({"CUDA_VISIBLE_DEVICES": vis})
        return out

    def _frames_note(self) -> dict[str, Any]:
        if not self.n_frames:
            return {}
        f = sorted(self.n_frames)
        return {"mean": round(sum(f) / len(f), 2), "min": f[0], "max": f[-1],
                "total": sum(f)}

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def report(self) -> str:
        s = self.summary()
        c, t, m, h = s["条件"], s["時間"], s["GPUメモリ"], s["CPU_RAM"]
        gpus = ", ".join(f"{d.get('name')} ({d.get('total_mb', 0) / 1024:.0f} GB)"
                         for d in c["gpu"] if "name" in d) or "(GPU 情報なし)"
        L: list[str] = []
        L.append("=" * 68)
        L.append(f"計測  {c['model_key']}  モード {c['mode'].upper()}"
                 + (f"  条件 {c['conditions']}" if c["conditions"] else ""))
        L.append("=" * 68)
        L.append(f"  GPU        : {gpus}")
        L.append(f"  モデル     : {c['model_id']}")
        eng = c.get("engine") or {}
        if eng.get("max_num_seqs"):
            L.append(f"  同時実行   : max_num_seqs {eng['max_num_seqs']}"
                     + (f"  max_num_batched_tokens {eng['max_num_batched_tokens']}"
                        if eng.get("max_num_batched_tokens") else "")
                     + (f"  KV ブロック {eng['num_gpu_blocks']:,}"
                        if eng.get("num_gpu_blocks") else ""))
        if eng:
            L.append("  dtype      : " + str(eng.get("dtype", "?"))
                     + f"  max_model_len {eng.get('max_model_len', '?')}"
                     + (f"  量子化 {eng['quantization']}" if eng.get("quantization") else "")
                     + (f"  KV {eng['kv_cache_dtype']}" if eng.get("kv_cache_dtype") else ""))
        w = c.get("weights") or {}
        if w:
            L.append(f"  重み       : {_g(w['weights_mb'])} "
                     f"({w['n_weight_files']} ファイル)"
                     f"   リポジトリ全体 {_g(w['total_mb'])}")
        fr = c.get("n_frames") or {}
        if fr:
            L.append(f"  入力フレーム: 平均 {fr['mean']} 枚 (最小 {fr['min']} / 最大 {fr['max']})"
                     f"  計 {fr['total']:,} 枚")
        st = c.get("settings") or {}
        if st:
            L.append("  設定       : " + "  ".join(f"{k} {v}" for k, v in st.items()))
        L.append("-" * 68)
        L.append(f"  モデルロード     : {_fmt(t['model_load_s'])} 秒")
        L.append(f"  推論 合計        : {_fmt(t['infer_total_s'])} 秒 "
                 f"({t['n_requests']} 件 / {t['n_batches']} バッチ)")
        L.append(f"  1 件あたり       : 平均 {_fmt(t['per_request_mean_s'])} 秒  "
                 f"中央 {_fmt(t['per_request_median_s'])}  "
                 f"最小 {_fmt(t['per_request_min_s'])}  最大 {_fmt(t['per_request_max_s'])}")
        if t.get("per_request_mean_excl_first_s") is not None:
            L.append(f"    1 バッチ目を除く: 平均 {_fmt(t['per_request_mean_excl_first_s'])} 秒"
                     f"   (1 バッチ目 {t['first_batch_n']} 件で "
                     f"{_fmt(t['first_batch_s'])} 秒。取り込みと割り当てで遅い)")
        L.append(f"  対象映像の長さ   : {_fmt(t['video_s'])} 秒")
        L.append(f"  映像 1 分あたり  : {_fmt(t['s_per_video_min'])} 秒"
                 + (f"   (実時間比 {t['realtime_factor']}x)"
                    if t.get("realtime_factor") else ""))
        L.append(f"  全体 (壁時計)    : {_fmt(t['wall_total_s'])} 秒")
        L.append("-" * 68)
        if m["baseline_used_mb"] or m["infer_peak_used_mb"]:
            L.append(f"  GPU メモリ ({m['source']})")
            L.append(f"    起動前       : {_mb(m['baseline_used_mb'])}")
            L.append(f"    ロード後常時 : {_mb(m['after_load_used_mb'])}")
            L.append(f"    推論中ピーク : {_mb(m['infer_peak_used_mb'])}"
                     + (f"   ({m['n_samples']} 標本 / {m['sample_interval_s']} 秒間隔)"
                        if m.get("n_samples") else ""))
            if m.get("infer_mean_util_pct") is not None:
                L.append(f"    GPU 使用率   : 平均 {m['infer_mean_util_pct']}% "
                         f"/ 最大 {m['infer_max_util_pct']}%")
        else:
            L.append("  GPU メモリ: NVML も nvidia-smi も使えないため未計測")
        tp = m.get("torch_process")
        if tp:
            L.append(f"    PyTorch (このプロセス) ピーク: "
                     f"allocated {tp['max_allocated_mb']:.0f} MB / "
                     f"reserved {tp['max_reserved_mb']:.0f} MB")
        L.append("-" * 68)
        if h["source"]:
            L.append(f"  CPU / RAM ({h['source']}, {h['n_cpu']} コア)")
            if h.get("infer_mean_cpu_pct") is not None:
                L.append(f"    CPU 使用率   : 推論中 平均 {h['infer_mean_cpu_pct']}% "
                         f"/ 最大 {h['infer_max_cpu_pct']}%"
                         + (f"   (起動前 {h['baseline_cpu_pct']}%)"
                            if h.get("baseline_cpu_pct") is not None else "")
                         + "  ※機械全体")
            if h.get("infer_peak_ram_used_mb") is not None:
                tot = h.get("ram_total_mb")
                L.append(f"    RAM (機械)   : 起動前 {_g(h.get('baseline_ram_used_mb'))} "
                         f"-> ロード後 {_g(h.get('after_load_ram_used_mb'))} "
                         f"-> 推論中ピーク {_g(h['infer_peak_ram_used_mb'])}"
                         + (f"  / 全体 {_g(tot)}" if tot else ""))
            if h.get("infer_peak_rss_mb") is not None:
                L.append(f"    RSS (自+子)  : 起動前 {_g(h.get('baseline_rss_mb'))} "
                         f"-> 推論中ピーク {_g(h['infer_peak_rss_mb'])}")
        else:
            L.append("  CPU / RAM: psutil も /proc も無いため機械全体の値は未計測"
                     " (pip install psutil で採れる)")
        pk = h.get("os_peak_rss_mb") or {}
        if pk:
            # getrusage は psutil が無くても使える。自プロセスと子で別々に出る。
            L.append("    OS 計測ピーク RSS: "
                     + "  ".join(f"{k} {_g(v)}" for k, v in pk.items()))
        L.append("=" * 68)
        return "\n".join(L)


def _fmt(v: Any) -> str:
    """小さい値を 0.00 に丸めてしまわないよう、桁を値に合わせる。"""
    if v is None:
        return "-"
    if not isinstance(v, float):
        return str(v)
    return f"{v:,.3f}" if abs(v) < 1 else f"{v:,.2f}"


def _g(v: float | None) -> str:
    """MB を「x,xxx MB (y.y GB)」で書く。"""
    return "-" if v is None else f"{v:,.0f} MB ({v / 1024:.1f} GB)"


def _mb(v: list[float] | None) -> str:
    if not v:
        return "-"
    return "  ".join(f"{x:,.0f} MB ({x / 1024:.1f} GB)" for x in v)


def request_video_s(req: dict[str, Any], cfg: dict[str, Any]) -> float:
    """このリクエスト 1 件が覆う映像の長さ [秒]。

    オンライン判定は 1 時刻進むごとに stride_s だけ映像が流れるので stride_s。
    一括判定は区間そのものを見るので span_s。
    これを足し上げたものを分母にして「映像 1 分あたりの処理時間」を出す。
    """
    if req.get("mode") == "online":
        return float(cfg["timeline"]["stride_s"])
    return float(req.get("span_s") or 0.0)
