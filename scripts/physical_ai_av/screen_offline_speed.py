#!/usr/bin/env python3
"""offline 版の Ego Motion (10 Hz、位置と姿勢だけ、1 チャンク 約 1.6 MB) から、クリップごとの
高速・直進の秒数を求め、高速走行のクリップを選ぶ目安にする。

メタデータには道路の種別も速度も無いので (docs/physical_ai_av.md)、軽い offline 版で速度を測って選ぶ。
国はチャンクを層別に抜き出すためだけに使う (風の有無の推測には使わない)。

  # チャンク 0 で、100 Hz の Ego Motion から求めた値と比べる (取得なし)
  uv run --with pyarrow --with scipy python scripts/physical_ai_av/screen_offline_speed.py --validate
  # 国で層別に抜き出したチャンクの offline 版を取得して集計する
  uv run --with pyarrow --with scipy python scripts/physical_ai_av/screen_offline_speed.py --sample 24 --max-mb 60

出力 (<out>/): offline_clips.csv (クリップごと)、offline_chunks.csv (チャンクごと)、validate.csv、estimate.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_clip_members as F  # noqa: E402

DATA = Path("data/physical_ai_av")
OFF_DIR = DATA / "labels/egomotion.offline"
HI_SPEED = 15.0            # m/s (egomotion_lateral_check.py と同じ)
STRAIGHT_R = 3.0           # deg/s。±1.5 s の |ヨーレート| (1 s 平均) の最大がこれ未満なら直進
MIN_SECONDS = (4, 10)      # 高速・直進がこれ以上あるクリップを数える
EGO_MB_PER_CLIP = 0.41     # 100 Hz の Ego Motion 1 クリップの大きさ (128.6 GB / 306,151)
EGO_MB_PER_CHUNK = 41.4    # 1 チャンクの中央値


def clip_features(d: pd.DataFrame) -> dict:
    d = d.sort_values("timestamp")
    t = d.timestamp.to_numpy() / 1e6
    p = d[["x", "y"]].to_numpy()
    dt = np.diff(t)
    sp = np.r_[np.hypot(*np.diff(p, axis=0).T) / dt, np.nan]
    q = R.from_quat(d[["qx", "qy", "qz", "qw"]].to_numpy())
    r = np.r_[np.degrees((q[:-1].inv() * q[1:]).as_rotvec()[:, 2] / dt), np.nan]
    rs = pd.Series(r).rolling(10, center=True, min_periods=1).mean().abs()
    rmax = rs.rolling(31, center=True, min_periods=1).max().to_numpy()
    hi = sp > HI_SPEED
    st = hi & (rmax < STRAIGHT_R)
    step = float(np.median(dt))
    return dict(hi_speed_s=float(np.nansum(hi) * step), straight_hi_s=float(np.nansum(st) * step),
                v_median=float(np.nanmedian(sp)), v_max=float(np.nanmax(sp)))


def read_chunk(path: Path) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    rows = []
    for n in z.namelist():
        f = clip_features(pd.read_parquet(io.BytesIO(z.read(n))))
        rows.append(dict(clip=n.split(".")[0], **f))
    return pd.DataFrame(rows)


def sample_chunks(n: int, seed: int) -> list[int]:
    """国で層別に抜き出す。米国 (2 種の車両) とドイツから多めに、ほかの国から 1 つずつ。"""
    ci = pd.read_parquet(DATA / "clip_index.parquet").join(pd.read_parquet(DATA / "metadata/data_collection.parquet"))
    g = ci.groupby("chunk").agg(top=("country", lambda s: s.value_counts().index[0]),
                                share=("country", lambda s: s.value_counts(normalize=True).iloc[0]),
                                platform=("platform_class", lambda s: s.value_counts().index[0]))
    g = g[(g.share >= 0.9) & (g.index != 0)]
    rng = np.random.default_rng(seed)
    pick = []
    strata = [("United States", "hyperion_8", 4), ("United States", "hyperion_8.1", 4), ("Germany", None, 4)]
    for country, plat, k in strata:
        cand = g[(g.top == country) & ((g.platform == plat) if plat else True)].index.to_numpy()
        pick += list(rng.choice(cand, size=min(k, len(cand)), replace=False))
    others = g[~g.top.isin(["United States", "Germany"])].groupby("top").size().sort_values(ascending=False).index
    for country in others[: max(n - len(pick), 0)]:
        pick.append(int(rng.choice(g[g.top == country].index.to_numpy())))
    return sorted(int(c) for c in pick[:n])


def download(chunks: list[int], max_mb: float) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download
    tok = F.token("physical_ai_av")
    info = HfApi(token=tok).dataset_info(F.REPO_ID, revision=F.REVISION, files_metadata=True)
    size = {s.rfilename: s.size for s in info.siblings}
    rels = [f"labels/egomotion.offline/egomotion.offline.chunk_{c:04d}.zip" for c in chunks]
    need = [r for r in rels if not (DATA / r).exists()]
    mb = sum(size[r] for r in need) / 1e6
    print(f"取得: {len(need)} ファイル、{mb:.1f} MB (上限 {max_mb} MB)")
    if mb > max_mb:
        sys.exit("上限を超えるので取得しない")
    for r in need:
        hf_hub_download(F.REPO_ID, r, repo_type="dataset", revision=F.REVISION, token=tok, local_dir=str(DATA))
    return [DATA / r for r in rels]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out/physical_ai_av/selection"))
    ap.add_argument("--validate", action="store_true", help="チャンク 0 で 100 Hz 版と比べる")
    ap.add_argument("--sample", type=int, default=0, help="層別に抜き出すチャンクの数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-mb", type=float, default=60.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.validate:
        off = read_chunk(OFF_DIR / "egomotion.offline.chunk_0000.zip")
        ref = pd.read_csv("out/physical_ai_av/egomotion_check/clips.csv")[["clip", "hi_speed_s", "straight_hi_s"]]
        v = off.merge(ref, on="clip", suffixes=("_offline", "_100hz"))
        v.round(2).to_csv(args.out / "validate.csv", index=False)
        for k in ("hi_speed_s", "straight_hi_s"):
            a, b = v[f"{k}_offline"], v[f"{k}_100hz"]
            print(f"{k}: 相関 {np.corrcoef(a, b)[0, 1]:.3f}  差の中央値 {np.median(a - b):+.2f} s  |差| の 90% {np.percentile(np.abs(a - b), 90):.2f} s")
        for th in MIN_SECONDS:
            a, b = v.straight_hi_s_offline >= th, v.straight_hi_s_100hz >= th
            print(f"高速・直進 ≥ {th} s: offline {a.sum()} / 100 Hz {b.sum()} / 両方 {(a & b).sum()}")
    if args.sample:
        chunks = sample_chunks(args.sample, args.seed)
        paths = download(chunks, args.max_mb)
        meta = pd.read_parquet(DATA / "metadata/data_collection.parquet")
        allc = pd.concat([read_chunk(p).assign(chunk=c) for c, p in zip(chunks, paths)]).join(meta, on="clip")
        allc.round(2).to_csv(args.out / "offline_clips.csv", index=False)
        agg = allc.groupby("chunk").agg(country=("country", lambda s: s.value_counts().index[0]),
                                        platform=("platform_class", lambda s: s.value_counts().index[0]),
                                        clips=("clip", "size"), hi_speed_h=("hi_speed_s", lambda s: s.sum() / 3600),
                                        **{f"clips_straight_ge{t}s": ("straight_hi_s", lambda s, t=t: int((s >= t).sum())) for t in MIN_SECONDS})
        agg.round(3).to_csv(args.out / "offline_chunks.csv")
        print(agg.round(3).to_string())
        n_total = len(pd.read_parquet(DATA / "clip_index.parquet"))
        est = dict(sampled_chunks=chunks, sampled_clips=len(allc),
                   **{f"frac_clips_straight_ge{t}s": round(float((allc.straight_hi_s >= t).mean()), 4) for t in MIN_SECONDS},
                   straight_hi_hours_per_clip_mean=round(float(allc.straight_hi_s.mean() / 3600), 5),
                   note="抜き出しは国で層別していて、全体の国の割合と同じではない。全体の見込みは割合を掛けただけの粗い値")
        for t in MIN_SECONDS:
            f = est[f"frac_clips_straight_ge{t}s"]
            est[f"est_clips_straight_ge{t}s"] = int(f * n_total)
            est[f"est_ego_GB_per_clip_fetch_ge{t}s"] = round(f * n_total * EGO_MB_PER_CLIP / 1e3, 1)
        est["offline_all_GB"] = 4.94
        est["ego_all_GB"] = 128.6
        (args.out / "estimate.json").write_text(json.dumps(est, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(est, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
