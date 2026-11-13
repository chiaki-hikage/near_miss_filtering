#!/usr/bin/env python3
"""ダウンロードした ZOD (Sequences / Drives) の車両信号と OXTS について、横風の判定の前に
「何が入っているか・単位・周期・時刻の範囲・欠損」を調べる。値の解釈 (符号・座標系) はここでは決めない。

前提 (docs/zod.md):
  - ZOD のダウンロードはユーザーが別の端末で行う (docs/zod_download.md)。この処理は Dropbox にも
    ネットワークにも触れない。読むのはローカルのファイルだけ
  - 各シーケンスのファイルの場所は、公式 SDK (zod 0.8.0) の infos (trainval-<subset>-<version>.json) から取る
  - HDF5 の中身は SDK のクラスを通さずに h5py で全部たどる (SDK が読まない列も確かめるため)

  uv run --offline --extra viz --with "zod==0.8.0" --with "dataclass-wizard==0.39.1" \\
      python scripts/zod/inspect_signals.py --root raw_data/zod --subset sequences --version mini --n 0

  (プロジェクトの環境に zod を上乗せする。pyproject.toml・uv.lock は変わらない。--offline で通信しない)

出力 (<out>/、既定 out/zod/signal_inventory/<subset>_<version>/):
  fields.csv     HDF5 の dataset ごとの形・型・属性・NaN の数 (シーケンスをまたいで集計)
  timing.csv     シーケンス × ストリームごとの時刻の範囲・周期・間隔の最大・逆行の数
  metadata.csv   シーケンスのメタデータ (開始・終了の時刻、国、車両、緯度経度)
  summary.json   ストリームごとの周期の中央値などのまとめ
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# OXTS の timestamp は GPS 時刻 (1980-01-06 起点の秒)。SDK (zod/data_classes/ego_motion.py) は
# この起点 + timestamp + leapSeconds で UTC の epoch 秒にしている。同じ式を使う
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc).timestamp()
GAP_FACTOR = 2.0          # 間隔の中央値のこの倍を超える間隔を「抜け」として数える


def walk_h5(path: Path) -> list[dict]:
    """HDF5 の dataset を全部たどり、形・型・属性・NaN の数を返す。"""
    import h5py
    rows = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        d = dict(path=name, shape=str(obj.shape), dtype=str(obj.dtype),
                 attrs=json.dumps({k: str(v) for k, v in obj.attrs.items()}, ensure_ascii=False))
        if obj.dtype.kind == "f" and obj.size <= 50_000_000:
            a = obj[()]
            d.update(n=int(a.size), n_nan=int(np.isnan(a).sum()),
                     min=float(np.nanmin(a)) if np.isfinite(a).any() else np.nan,
                     max=float(np.nanmax(a)) if np.isfinite(a).any() else np.nan)
        else:
            d.update(n=int(obj.size))
        rows.append(d)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return rows


def timing(t: np.ndarray) -> dict:
    """時刻の列 (秒、UTC epoch) の周期と抜け。"""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return dict(n=int(t.size))
    dt = np.diff(t)
    med = float(np.median(dt))
    return dict(n=int(t.size), t0=float(t[0]), t1=float(t[-1]), duration_s=float(t[-1] - t[0]),
                dt_median_ms=med * 1e3, rate_hz=1.0 / med if med > 0 else np.nan,
                dt_max_ms=float(dt.max() * 1e3), n_gaps=int((dt > GAP_FACTOR * med).sum()),
                n_backward=int((dt <= 0).sum()))


def oxts_times(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return GPS_EPOCH + f["timestamp"][()] + f["leapSeconds"][()]


def vehicle_times(path: Path) -> dict[str, np.ndarray]:
    """車両データの各グループ (ego_vehicle_data など) の時刻 [ns] を秒にする。
    時刻の dataset は名前に 'timestamp' を含むものを、グループの直下から探す (SDK と同じ場所)。"""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for g in f:
            if not isinstance(f[g], h5py.Group):
                continue
            for cand in ("timestamp/nanoseconds/value",):
                if cand in f[g]:
                    out[g] = f[g][cand][()].astype(np.float64) / 1e9
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("raw_data/zod"), help="zod download の --output-dir")
    ap.add_argument("--subset", choices=["sequences", "drives"], default="sequences")
    ap.add_argument("--version", choices=["full", "mini"], default="full")
    ap.add_argument("--n", type=int, default=20, help="調べるシーケンスの数 (先頭から)。0 で全部")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or Path(f"out/zod/signal_inventory/{args.subset}_{args.version}")

    from zod import ZodDrives, ZodSequences
    cls = ZodSequences if args.subset == "sequences" else ZodDrives
    ds = cls(str(args.root), args.version, mp=False)
    ids = sorted(ds.get_all_ids())
    if args.n:
        ids = ids[: args.n]
    print(f"{args.subset}-{args.version}: {len(ds.get_all_ids())} 件のうち {len(ids)} 件を調べる")

    fields, times, metas = [], [], []
    for sid in ids:
        info = ds[sid].info
        meta_path = Path(info.metadata_path)
        if meta_path.exists():
            metas.append(dict(id=sid, **json.loads(meta_path.read_text())))
        files = dict(oxts=Path(info.oxts_path),
                     vehicle_data=Path(info.vehicle_data_path) if info.vehicle_data_path else None)
        for kind, p in files.items():
            if p is None or not p.exists():
                times.append(dict(id=sid, stream=kind, missing=True))
                continue
            fields += [dict(id=sid, file=kind, **r) for r in walk_h5(p)]
        if files["oxts"].exists():
            times.append(dict(id=sid, stream="oxts", **timing(oxts_times(files["oxts"]))))
        if files["vehicle_data"] is not None and files["vehicle_data"].exists():
            for g, t in vehicle_times(files["vehicle_data"]).items():
                times.append(dict(id=sid, stream=f"vehicle_data/{g}", **timing(t)))
        for cam, frames in info.camera_frames.items():
            t = np.array([fr.time.timestamp() for fr in frames])
            n_exist = sum(Path(fr.filepath).exists() for fr in frames)
            times.append(dict(id=sid, stream=f"camera/{cam}", files_present=n_exist, **timing(t)))

    out.mkdir(parents=True, exist_ok=True)
    fd = pd.DataFrame(fields)
    tm = pd.DataFrame(times)
    if not fd.empty:
        agg = {"shape": lambda s: ";".join(sorted(set(s))[:3]), "dtype": "first", "attrs": "first",
               "id": "nunique", "n": "sum"}
        for c in ("n_nan", "min", "max"):
            if c in fd:
                agg[c] = {"n_nan": "sum", "min": "min", "max": "max"}[c]
        fd.groupby(["file", "path"]).agg(agg).rename(columns={"id": "n_sequences"}).to_csv(out / "fields.csv")
    tm.to_csv(out / "timing.csv", index=False)
    pd.DataFrame(metas).to_csv(out / "metadata.csv", index=False)

    summary = dict(root=str(args.root), subset=args.subset, version=args.version, n_inspected=len(ids))
    if "rate_hz" in tm:
        oxts_t0 = tm[tm.stream == "oxts"].set_index("id")["t0"]
        tm["t0_minus_oxts_s"] = tm.t0 - tm.id.map(oxts_t0)
        summary["streams"] = (tm.groupby("stream")
                              .agg(n=("id", "size"), rate_hz_median=("rate_hz", "median"),
                                   dt_max_ms_max=("dt_max_ms", "max"), gaps_total=("n_gaps", "sum"),
                                   backward_total=("n_backward", "sum"), duration_s_median=("duration_s", "median"),
                                   t0_minus_oxts_s_median=("t0_minus_oxts_s", "median"))
                              .round(4).reset_index().to_dict("records"))
        tm.to_csv(out / "timing.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
