"""候補の前後の前方映像と時系列を、時刻をそろえて 1 本の動画と 1 枚の図にする (人が確認・評価するため)。

画像は KITTI 形式の image_left_color.tar から、td2_tar_index.py で作った位置の一覧を使い、必要なフレーム (PNG) だけを範囲指定で取る。
フレーム番号と時刻の対応は、tar の中の timestamps.txt (番号の順に 1 行ずつ) を使う。
--check で rosbag から取った画像 (ヘッダの時刻が分かっているもの) と照らして確かめる。

時系列 (どれも IMU の時刻にそろえる):
  MTVV_total (1 秒移動) と前後 ±30 s の中央値 / 上下加速度 / ロール角速度・ヨー角速度の急な変化 /
  後輪のストローク (racepak の時刻を 2 状態で補正し、一定の遅れを引いたもの) / 速度・ブレーキ・スロットルとみられる信号 (同じ補正) /
  舵角 (時刻は未補正)

    uv run python scripts/offroad/td2_event_media.py --candidates out/offroad/tartandrive2/<id>/candidates.csv \
        --sync out/offroad/tartandrive2/20261117_r10 --ids A037,A047 --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import urllib.parse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

import rosbag_range as RB
import td2_candidate_profile as CP
import td2_vibration_poc as P
from td2_event_frames import bag_start  # noqa: F401  (log の読み方を合わせるため)


def kitti_start(log: Path) -> float:
    import re
    m = re.search(r"starting time ([\d.]+)", log.read_text())
    return float(m.group(1))


class Frames:
    def __init__(self, src: dict, root: Path, run: str, tar: str):
        idx = pd.read_csv(root / "tar_index" / f"{run}__{tar}.csv")
        idx = idx[idx.name.str.endswith(".png")]
        idx["k"] = idx.name.str.extract(r"(\d+)\.png").astype(int)
        self.url = f"{src['source']['endpoint']}/{src['source']['bucket']}/{urllib.parse.quote(f'kitti/all_topics/{run}/{tar}')}"
        self.used = 0
        # tar の中の timestamps.txt (フレーム番号の順に 1 行ずつ。ヘッダの時刻とみられる)
        all_ = pd.read_csv(root / "tar_index" / f"{run}__{tar}.csv")
        ts = all_[all_.name.str.endswith("timestamps.txt")].iloc[0]
        if not RB.cached(self.url, int(ts.offset), int(ts.offset + ts["size"] - 1)):
            self.used += int(ts["size"])
        txt = RB.http_range(self.url, int(ts.offset), int(ts.offset + ts["size"] - 1)).decode()
        self.stamps = np.array([float(x) for x in txt.split()])
        self.idx = idx.set_index("k")
        self.t0 = kitti_start(root / "kitti/all_topics" / run / "data_extraction.log")

    def k_at(self, t: float) -> int:
        return int(np.argmin(np.abs(self.stamps - t)))

    def t_of(self, k: int) -> float:
        return float(self.stamps[k])

    def get(self, k: int) -> Image.Image | None:
        if k not in self.idx.index:
            return None
        r = self.idx.loc[k]
        if not RB.cached(self.url, int(r.offset), int(r.offset + r["size"] - 1)):
            self.used += int(r["size"])
        b = RB.http_range(self.url, int(r.offset), int(r.offset + r["size"] - 1))
        return Image.open(io.BytesIO(b)).convert("RGB")


def signals(run: str, cfg: dict, ST: pd.DataFrame) -> dict:
    pc = cfg["profile"]
    R = P.load_run(Path(cfg["data_dir"]) / run, cfg)
    R["run_dir"] = Path(cfg["data_dir"]) / run
    S = P.iso_series(R, cfg["lp8_hz"])
    R["wz_hp"] = P.bp(R["X"]["wz"], *pc["yaw_band_hz"], R["fs"])
    C = CP.corrected_racepak(R, ST[ST.run == run], pc["racepak_latency_s"])
    return dict(R=R, S=S, C=C)


def draw_panel(sig: dict, te: float, span: tuple[float, float], c: pd.Series, cfg: dict):
    R, S, C = sig["R"], sig["S"], sig["C"]
    g = R["g"]
    lo, hi = te + span[0] - 1, te + span[1] + 1
    m = (g >= lo) & (g <= hi)
    x = g[m] - te
    fig, ax = plt.subplots(5, 1, figsize=(10.24, 7.0), sharex=True, dpi=100)
    ax[0].plot(x, S["tot"][m], color="#1f4e9c", label="MTVV_total (1 秒移動) [m/s²]")
    ax[0].axhline(c.mtvv_total / c.contrast, color="#6b7280", ls="--", lw=1, label="前後 ±30 s の中央値")
    ax[1].plot(x, S["raw"]["z"][m], color="#374151", lw=0.7, label="上下加速度 [m/s²]")
    ax[2].plot(x, np.degrees(R["X"]["wx"][m]), color="#7c3aed", lw=0.8, label="ロール角速度 [deg/s]")
    ax[2].plot(x, np.degrees(R["wz_hp"][m]), color="#c2410c", lw=1.0, label="ヨー角速度の急な変化 [deg/s]")
    if "shock" in C:
        ts, a = C["shock"]
        ms = (ts >= lo) & (ts <= hi)
        for side, j, col in (("RL", 2, "#c2410c"), ("RR", 3, "#0f766e")):
            ax[3].plot(ts[ms] - te, a[ms, j], color=col, label=f"ストローク {side} (補正した時刻、inch とされる。大きいほど縮み)")
    ax[4].plot(x, R["speed"][m], color="#111827", label="速度 [m/s]")
    a2 = ax[4].twinx()
    if "controls" in C:
        tc, cc = C["controls"]
        mc = (tc >= lo) & (tc <= hi)
        brk = np.where(cc[:, 1] > cfg["reaction"]["brake_invalid_above"], np.nan, cc[:, 1])
        a2.plot(tc[mc] - te, brk[mc], color="#dc2626", lw=1, label="ブレーキとみられる信号 (単位未確認)")
        a2.plot(tc[mc] - te, cc[mc, 0] * 10, color="#16a34a", lw=1, label="スロットルとみられる信号 ×10 (単位未確認)")
    if R["steer"] is not None:
        tsr, s = R["steer"]
        mst = (tsr >= lo) & (tsr <= hi)
        a2.plot(tsr[mst] - te, s[mst, 0], color="#a16207", lw=1, ls=":", label="舵角 (単位未確認、時刻は未補正)")
    a2.legend(fontsize=6, loc="upper right")
    for k, axx in enumerate(ax):
        axx.legend(fontsize=7, loc="upper left")
        axx.grid(alpha=0.3)
    ax[4].set_xlabel("候補の時刻 te からの秒 (te = 1 秒移動 MTVV の最大。衝撃はその 0〜1 s 前)")
    ax[0].set_xlim(span[0] - 1, span[1] + 1)
    fig.suptitle(f"{c.cand_id}  {c.run}  t={c.t:.1f} s  {c.input_axis}・{c.tier}\n{c.reasons}", fontsize=8)
    fig.tight_layout()
    cur = [axx.axvline(0, color="#f59e0b", lw=2) for axx in ax]
    return fig, cur


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--source", default="configs/offroad/tartandrive2.yaml")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--sync", type=Path, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--check", nargs=3, metavar=("RUN", "BAG_JPG", "HEADER_T"), help="フレームの時刻の対応の確かめ")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mc = cfg["media"]
    src = yaml.safe_load(open(a.source))
    root = Path(src["data_dir"])
    RB.CACHE_DIR = root / "bag_cache"
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    log = []

    if a.check:
        run, jpg, ht = a.check[0], a.check[1], float(a.check[2])
        fr = Frames(src, root, run, mc["tar"])
        ref = np.asarray(Image.open(jpg).convert("L").resize((128, 68)), float)
        k0 = fr.k_at(ht)
        res = []
        for dk in range(-mc["check_search"], mc["check_search"] + 1):
            im = fr.get(k0 + dk)
            if im is None:
                continue
            v = np.asarray(im.convert("L").resize((128, 68)), float)
            res.append(dict(dk=dk, mad=float(np.mean(np.abs(v - ref)))))
        C = pd.DataFrame(res)
        C.to_csv(out / "frame_time_check.csv", index=False)
        log.append(dict(check=run, bag_jpg=jpg, header_t=ht, best_dk=int(C.loc[C.mad.idxmin(), "dk"]), mad=C.mad.round(2).tolist(), bytes=fr.used))
        print(C.to_string(), flush=True)

    Fc = pd.read_csv(a.candidates).set_index("cand_id")
    ST = pd.read_csv(a.sync / "sync_states.csv")
    used_total = sum(x.get("bytes", 0) for x in log)
    cache = {}
    for cid in [x for x in a.ids.split(",") if x]:
        c = Fc.loc[cid]
        c["cand_id"] = cid
        if c.run not in cache:
            cache[c.run] = (signals(c.run, cfg, ST), Frames(src, root, c.run, mc["tar"]))
        sig, fr = cache[c.run]
        te = float(c.t_abs)
        k0, k1 = fr.k_at(te + mc["span_s"][0]), fr.k_at(te + mc["span_s"][1])
        ks = list(range(k0, k1 + 1, mc["frame_step"]))
        est = sum(int(fr.idx.loc[k, "size"]) for k in ks if k in fr.idx.index)
        if used_total + est > mc["max_download_mb"] * 1e6:
            log.append(dict(cand=cid, skipped=f"上限 {mc['max_download_mb']} MB を超えるので取らない (見積もり {est / 1e6:.0f} MB)"))
            print(log[-1], flush=True)
            continue
        fig, cur = draw_panel(sig, te, tuple(mc["span_s"]), c, cfg)
        fdir = out / f"{cid}_frames"
        fdir.mkdir(exist_ok=True)
        thumbs = []
        before = fr.used
        for n, k in enumerate(ks):
            im = fr.get(k)
            if im is None:
                continue
            tk = fr.t_of(k) - te
            for l in cur:
                l.set_xdata([tk, tk])
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            panel = Image.open(buf).convert("RGB")
            frame = Image.new("RGB", (1024, im.height + panel.height), "white")
            frame.paste(im.resize((1024, int(im.height * 1024 / im.width))), (0, 0))
            frame.paste(panel.resize((1024, int(panel.height * 1024 / panel.width))), (0, im.height))
            frame.save(fdir / f"{n:03d}.png")
            thumbs.append((tk, im))
        plt.close(fig)
        used_total += fr.used - before
        # 動画 (実時間の速さ。1 フレーム = frame_step × 0.1 s)
        fps = 10 / mc["frame_step"]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps}", "-i", str(fdir / "%03d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                        str(out / f"{cid}.mp4")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps / mc['slow_factor']}", "-i", str(fdir / "%03d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                        str(out / f"{cid}_slow{mc['slow_factor']}x.mp4")], check=True)
        # 1 枚の図: 時系列 + 画像の並び
        fig, cur = draw_panel(sig, te, tuple(mc["span_s"]), c, cfg)
        for l in cur:
            l.remove()
        sel = [thumbs[i] for i in np.linspace(0, len(thumbs) - 1, min(mc["strip_n"], len(thumbs))).round().astype(int)]
        for ax in fig.axes[:5]:
            for tk, _ in sel:
                ax.axvline(tk, color="#f59e0b", lw=0.8, alpha=0.6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        panel = Image.open(buf).convert("RGB")
        tw = 1024 // len(sel)
        th = int(544 * tw / 1024)
        strip = Image.new("RGB", (1024, th + 18), "white")
        from PIL import ImageDraw
        dr = ImageDraw.Draw(strip)
        for n, (tk, im) in enumerate(sel):
            strip.paste(im.resize((tw, th)), (n * tw, 18))
            dr.text((n * tw + 4, 2), f"{tk:+.1f} s", fill="black")
        sheet = Image.new("RGB", (1024, strip.height + panel.height), "white")
        sheet.paste(strip, (0, 0))
        sheet.paste(panel.resize((1024, int(panel.height * 1024 / panel.width))), (0, strip.height))
        sheet.save(out / f"{cid}_sheet.png")
        log.append(dict(cand=cid, run=c.run, frames=len(thumbs), bytes=fr.used - before, first_k=ks[0], last_k=ks[-1]))
        print(log[-1], f"累計 {used_total / 1e6:.1f} MB", flush=True)
    (out / "media_log.json").write_text(json.dumps(dict(log=log, total_mb=used_total / 1e6, config=mc), ensure_ascii=False, indent=2, default=int))
    print(f"取得 {used_total / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
