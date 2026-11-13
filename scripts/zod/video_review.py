#!/usr/bin/env python3
"""ZOD の横風候補を前方映像で確かめるための MP4 を作る。ネットワークは使わない (画像は取得済みのものだけ)。

  uv run --offline --extra viz python scripts/zod/video_review.py --run-id <id>

設定: configs/zod/video_review.yaml。出力: out/zod/video_review/<run_id>/ (説明は out/zod/video_review/README.md)

シーケンスごとに作るもの (画像が無いシーケンスは図と一覧だけ):
  <seq>_review.mp4   カメラ (幅 1600) の下に時系列 (速さ・舵角とトルク・ヨー・ヨー残差) を並べ、今のフレームの時刻に縦線。
                     左上にフレームの UTC とカメラの開始・候補の始まりからの秒、右上に「検出区間の中か外か」
  <seq>_camera.mp4   カメラだけ (幅 1920)。道路・周りの車を細かく見る用
  plots/<seq>_timeseries.png  時系列の図 (動画と同じもの、縦線なし)
どちらの MP4 も、info.json の各フレームの UTC の間隔をそのまま使う (可変フレームレート、約 10 Hz、実時間で再生される)。

時刻の注意: カメラと OXTS・車両データの時刻の対応は、mini の調べで ±1 s までしか確かめられていない (docs/zod_signals.md)。
候補の検出区間 (Phase 2) がカメラの区間と重ならないときは、映像を事象の映像として扱わない (動画の上にもそう出す)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_time as ET  # noqa: E402

HYP_COLOR = {"持続": "#c0392b", "ヨーの変化": "#8e44ad", "車線変更": "#f39c12", "操舵が先": "#16a085", "センサー": "#7f8c8d", "道路の曲がり": "#2e86c1"}


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S.%f")[:-4]


def hyp_color(h):
    for k, c in HYP_COLOR.items():
        if k in str(h):
            return c
    return "#555555"


def r3(x, n=2):
    try:
        return None if x is None or not np.isfinite(float(x)) else round(float(x), n)
    except (TypeError, ValueError):
        return None


def slow20(dr, blk, n=100):
    s = np.full(len(dr), np.nan)
    for k in np.unique(blk):
        m = np.where(blk == k)[0]
        s[m] = pd.Series(dr[m].astype(float)).rolling(n, center=True, min_periods=1).mean().to_numpy()
    return s


# ---------------------------------------------------------------- 入力
def load_inputs(cfg):
    inv, scr, wr = Path(cfg["inventory_run"]), Path(cfg["screening_run"]), Path(cfg["weather_run"])
    seqs = pd.read_csv(inv / "sequences.csv", dtype={"id": str}).set_index("id")
    cand = pd.read_csv(scr / "candidates_unique.csv", dtype={"in_camera_sequences": str, "sequence_ids": str})
    w = pd.read_csv(wr / "candidates_weather.csv")
    cand = cand.merge(w[["event_id", "verdict", "verdict_reason", "idw_w_left", "nearest_w_left", "w_left_sigma", "idw_speed", "nearest_speed"]],
                      on="event_id", how="left")
    seg = pd.read_csv(wr / "segments_weather.csv")
    calib = pd.read_csv(scr / "offset_calib_by_drive.csv").set_index("drive_id")
    thr = pd.read_csv(scr / "thresholds_by_fold.csv")
    return seqs, cand, seg, calib, thr


def sequence_events(seq_row, cand, c0, c1, cfg, focus=()):
    """カメラの前後 (search_events_s) に閾値超過区間がかかる候補と、注目の候補 (focus。カメラの外でも残す)。
    検出区間 (det_t0, det_t1) = 閾値超過区間 (event_time)。推定開始 + 継続時間は使わない。"""
    a, b = cfg["plot"]["search_events_s"]
    d = cand[cand.drive_id == seq_row.drive_id].copy()
    iv = [ET.event_interval(r) for _, r in d.iterrows()]
    d["det_t0"] = [x[0] for x in iv]
    d["det_t1"] = [x[1] for x in iv]
    near = [ET.overlap(x, y, c0 + a, c1 + b) > 0 for x, y in iv]
    ev = d[np.array(near, bool) | d.event_id.isin(list(focus))].copy()
    ev["onset_rel_cam_s"] = ev.det_t0 - c0
    ev["end_rel_cam_s"] = ev.det_t1 - c0
    ev["rise_rel_cam_s"] = ev.t_rise - c0
    ev["peak_rel_cam_s"] = ev.t_peak - c0
    ev["overlap_s"] = [ET.overlap(x, y, c0, c1) for x, y in zip(ev.det_t0, ev.det_t1)]
    ev["camera_relation"] = [ET.relation_to_window(x, y, c0, c1) for x, y in zip(ev.det_t0, ev.det_t1)]
    return ev.sort_values("det_t0")


def seq_wind(seg, drive_id, c0, c1):
    """シーケンスのカメラの区間に近い比べる区間 (60 s) の推定の横風 (Phase 3)。無ければ None。"""
    g = seg[(seg.drive_id == drive_id)]
    if not len(g):
        return None
    mid = 0.5 * (c0 + c1)
    i = (g.t - mid).abs().idxmin()
    r = g.loc[i]
    if abs(r.t - mid) > 60 or pd.isna(r.w_left_est):
        return None
    return dict(w_left=r.w_left_est, sigma=r.w_left_sigma, speed=r.speed_est, dt_s=r.t - mid)


# ---------------------------------------------------------------- 図
def make_plot(z, c0, c1, ev, thr_slow, thr_fast, title, width_px, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    pre, post = cfg["plot"]["pre_s"], cfg["plot"]["post_s"]
    t = z["t"] - c0
    m = (t >= -pre) & (t <= (c1 - c0) + post)
    if m.sum() < 10:
        return None, None
    tt = t[m]
    slow = slow20(z["dr_cor"], z["blk"])[m]
    fig, ax = plt.subplots(4, 1, figsize=(width_px / 100, 5.4), dpi=100, sharex=True)
    ax[0].plot(tt, z["v"][m] * 3.6, color="#333"); ax[0].set_ylabel("速さ\nkm/h", fontsize=8)
    st = z["steer_deg"][m].astype(float)
    ax[1].plot(tt, st - np.nanmedian(st), color="#2c3e50", label="舵角 − 中央値 [deg]")
    a1 = ax[1].twinx(); a1.plot(tt, z["torque"][m], color="#d35400", lw=0.8); a1.set_ylabel("トルク\nN·m", fontsize=8, color="#d35400")
    on = z["turn"][m] != 0
    if on.any():
        ax[1].fill_between(tt, 0, 1, where=on, transform=ax[1].get_xaxis_transform(), color="#f1c40f", alpha=0.35, label="方向指示器")
    ax[1].set_ylabel("舵角\ndeg", fontsize=8); ax[1].legend(fontsize=7, loc="upper left")
    ax[2].plot(tt, z["r_deg_s"][m], color="#1f4e9c", label="r (OXTS)")
    ax[2].plot(tt, z["r_pred_cor"][m], color="#c0392b", lw=1, label="r_pred (補正後)")
    ax[2].set_ylabel("ヨー\ndeg/s", fontsize=8); ax[2].legend(fontsize=7, loc="upper left")
    ax[3].plot(tt, z["dr_cor"][m], color="#1f4e9c", lw=1, label="Δr 補正後")
    ax[3].plot(tt, z["dr_unc"][m], color="#7f8c8d", lw=0.8, ls="--", label="Δr 補正前")
    ax[3].plot(tt, slow, color="#c0392b", lw=1.6, label="持続成分 (5 s 平均)")
    if np.isfinite(thr_slow):
        for s in (1, -1):
            ax[3].axhline(s * thr_slow, color="#c0392b", lw=0.6, ls=":")
    ax[3].set_ylabel("ヨー残差\ndeg/s", fontsize=8); ax[3].legend(fontsize=7, loc="upper left", ncol=3)
    ax[3].set_xlabel("カメラの開始からの秒 (負 = カメラより前、OXTS だけ)", fontsize=8)
    for a in ax:
        a.axvspan(-pre, 0, color="#bbbbbb", alpha=0.25, lw=0)
        a.axvspan(c1 - c0, (c1 - c0) + post, color="#bbbbbb", alpha=0.25, lw=0)
        for _, e in ev.iterrows():
            a.axvspan(e.det_t0 - c0, e.det_t1 - c0, color=hyp_color(e.hypothesis), alpha=0.18, lw=0)   # 閾値超過区間
            a.axvline(e.t_rise - c0, color=hyp_color(e.hypothesis), lw=0.9, ls="--")                   # 推定開始
            a.axvline(e.t_peak - c0, color=hyp_color(e.hypothesis), lw=0.9, ls=":")                    # ピーク
        a.grid(alpha=0.3); a.tick_params(labelsize=7)
    for _, e in ev.iterrows():
        ax[0].text(max(e.det_t0 - c0, -pre), ax[0].get_ylim()[1], f" {e.event_id}", fontsize=7, va="top", color=hyp_color(e.hypothesis))
    ax[3].text(0.995, 0.02, "帯: 閾値超過区間  破線: 推定開始  点線: ピーク", transform=ax[3].transAxes, ha="right", va="bottom", fontsize=7)
    ax[0].set_title(title, fontsize=9)
    ax[0].set_xlim(-pre, (c1 - c0) + post)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    H = img.shape[0]
    x0 = ax[0].transData.transform((-pre, 0))[0]
    x1 = ax[0].transData.transform(((c1 - c0) + post, 0))[0]
    ytop = H - ax[0].get_window_extent().y1
    ybot = H - ax[3].get_window_extent().y0
    mapping = dict(x0=x0, x1=x1, t0=-pre, t1=(c1 - c0) + post, ytop=ytop, ybot=ybot)
    return fig, (img, mapping)


# ---------------------------------------------------------------- 動画
def encode(list_file: Path, out: Path, crf, preset, vf=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-fps_mode", "vfr"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def concat_list(paths, times, f: Path):
    dt = np.diff(times)
    last = float(np.median(dt)) if len(dt) else 0.1
    with f.open("w") as fh:
        for p, d in zip(paths, list(dt) + [last]):
            fh.write(f"file '{p}'\nduration {d:.6f}\n")
        fh.write(f"file '{paths[-1]}'\n")


def probe(path: Path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=width,height,nb_read_frames:format=duration", "-of", "json", str(path)], capture_output=True, text=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    return dict(width=s["width"], height=s["height"], frames=int(s["nb_read_frames"]), duration_s=round(float(j["format"]["duration"]), 2))


def render_review(frames, times, c0, plot_pack, ev, focus, header_lines, out_mp4, cfg, tmp: Path):
    from PIL import Image, ImageDraw, ImageFont
    vc = cfg["video"]
    W = vc["camera_width"]
    font = ImageFont.truetype(vc["font"], 26)
    small = ImageFont.truetype(vc["font"], 20)
    pimg, mp = plot_pack
    ph = pimg.shape[0]
    fev = ev[ev.event_id.isin(focus)] if len(focus) else ev
    paths = []
    for k, (fp, t) in enumerate(zip(frames, times)):
        im = Image.open(fp)
        im.draft("RGB", (im.width // 2, im.height // 2))
        im = im.convert("RGB")
        h = int(round(W * im.height / im.width / 2) * 2)
        im = im.resize((W, h), Image.BILINEAR)
        canvas = Image.new("RGB", (W, h + ph + (ph % 2)), "white")
        canvas.paste(im, (0, 0))
        pl = Image.fromarray(pimg)
        d = ImageDraw.Draw(pl)
        x = mp["x0"] + (t - c0 - mp["t0"]) / (mp["t1"] - mp["t0"]) * (mp["x1"] - mp["x0"])
        d.line([(x, mp["ytop"]), (x, mp["ybot"])], fill=(220, 20, 60), width=3)
        canvas.paste(pl, (0, h))
        dc = ImageDraw.Draw(canvas, "RGBA")
        lines = list(header_lines) + [f"フレーム {utc(t)} UTC   カメラの開始から {t - c0:+.2f} s"]
        for _, e in fev.iterrows():
            lines.append(f"{e.event_id}: 閾値超過の始まりから {t - e.det_t0:+.2f} s (推定開始から {t - e.t_rise:+.2f} s)")
        box_h = 12 + 32 * len(lines)
        dc.rectangle([0, 0, 900, box_h], fill=(0, 0, 0, 150))
        for i, s in enumerate(lines):
            dc.text((12, 8 + 32 * i), s, font=font if i < len(header_lines) else small, fill=(255, 255, 255))
        inside = [e for _, e in ev.iterrows() if e.det_t0 <= t <= e.det_t1]
        if inside:
            msg = "検出区間の中: " + ", ".join(f"{e.event_id} ({str(e.hypothesis)[:6]}…)" for e in inside)
            col = (192, 57, 43, 220)
        else:
            msg = "候補の検出区間の外"
            col = (90, 90, 90, 200)
        tw = dc.textlength(msg, font=font)
        dc.rectangle([W - tw - 30, 0, W, 44], fill=col)
        dc.text((W - tw - 15, 7), msg, font=font, fill=(255, 255, 255))
        p = tmp / f"{k:04d}.jpg"
        canvas.save(p, quality=90)
        paths.append(p)
    lf = tmp / "review.txt"
    concat_list(paths, times, lf)
    encode(lf, out_mp4, vc["crf"], vc["preset"])


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zod/video_review.yaml")
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%dT%H%M%S"))
    ap.add_argument("--only", nargs="*", default=None, help="このシーケンスだけ")
    ap.add_argument("--tmp", default=None, help="合成したフレームを一時的に置く場所 (既定: OS の一時フォルダ)")
    ap.add_argument("--screening-run", default=None, help="設定の screening_run を上書き")
    ap.add_argument("--weather-run", default=None, help="設定の weather_run を上書き")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.screening_run:
        cfg["screening_run"] = a.screening_run
    if a.weather_run:
        cfg["weather_run"] = a.weather_run
    out = Path(cfg["out_dir"]) / a.run_id
    if out.exists():
        print(f"{out} がすでにある。上書きしない", file=sys.stderr)
        return 1
    (out / "plots").mkdir(parents=True)
    seqs, cand, seg, calib, thr = load_inputs(cfg)
    raw = Path(cfg["raw_root"])
    scr = Path(cfg["screening_run"])
    rows = []
    for tg in cfg["targets"]:
        sid = tg["sequence"]
        if a.only and sid not in a.only:
            continue
        s = seqs.loc[sid]
        info = json.loads((raw / "sequences" / sid / "info.json").read_text())
        cf = info["camera_frames"]["front_blur"]
        times = np.array([ts(f["time"]) for f in cf])
        frames = [raw / f["filepath"] for f in cf]
        have = sum(p.exists() for p in frames)
        c0, c1 = times[0], times[-1]
        ev = sequence_events(s, cand, c0, c1, cfg, tg.get("focus", []))
        focus = [e for e in tg.get("focus", []) if e in set(ev.event_id)]
        fold = calib.fold.get(s.drive_id)
        car = s.car
        tsel = thr[(thr.car == car) & (thr.fold == fold) & (thr.speed_kmh == 80) & (thr.residual == "cor")]
        thr_slow = float(tsel[tsel.kind == "slow"].threshold_deg_s.iloc[0]) if len(tsel) else np.nan
        thr_fast = float(tsel[tsel.kind == "fast"].threshold_deg_s.iloc[0]) if len(tsel) else np.nan
        z = dict(np.load(scr / "residuals" / f"{s.drive_id}.npz"))
        sw = seq_wind(seg, s.drive_id, c0, c1)
        fev = ev[ev.event_id.isin(focus)]
        wtxt = ""
        if len(fev):
            e = fev.iloc[0]
            wl = e.idw_w_left if pd.notna(e.idw_w_left) else e.nearest_w_left
            if pd.notna(wl):
                wtxt = f"推定の横風 (10 m) {wl:+.1f} ± {e.w_left_sigma:.1f} m/s (左から正)、判定: {e.verdict}"
        elif sw:
            wtxt = f"推定の横風 (10 m、近くの 60 s 区間) {sw['w_left']:+.1f} ± {sw['sigma']:.1f} m/s (左から正)"
        title = f"{sid} {car} {s.drive_id}  カメラ {utc(c0)}〜{utc(c1)} UTC  [{tg['role']}]  {wtxt}"
        fig, pack = make_plot(z, c0, c1, ev, thr_slow, thr_fast, title, cfg["video"]["camera_width"], cfg)
        plot_png = out / "plots" / f"{sid}_timeseries.png"
        if fig is not None:
            fig.savefig(plot_png, dpi=100)
            import matplotlib.pyplot as plt
            plt.close(fig)
        review = camera = None
        meta = {}
        if have == len(frames) and pack is not None:
            header = [f"{sid}  {car}  [{tg['role']}]  {datetime.fromtimestamp(c0, timezone.utc):%Y-%m-%d} UTC"]
            for _, e in fev.iterrows():
                if e.overlap_s <= 0:
                    where = "前" if e.det_t1 <= c0 else "後"
                    header.append(f"注意: {e.event_id} の閾値超過区間はカメラの{where} (カメラ開始から {e.onset_rel_cam_s:+.1f}〜{e.end_rel_cam_s:+.1f} s)。この映像は事象そのものではない")
                elif e.camera_relation != "窓の中":
                    header.append(f"注意: {e.event_id} の閾値超過区間は一部だけカメラに入る ({e.camera_relation}、重なり {e.overlap_s:.1f} s)")
            if wtxt:
                header.append(wtxt)
            with tempfile.TemporaryDirectory(dir=a.tmp) as td:
                tmp = Path(td)
                review = out / f"{sid}_review.mp4"
                render_review(frames, times, c0, pack, ev, focus, header, review, cfg, tmp)
                lf = tmp / "camera.txt"
                concat_list([p.resolve() for p in frames], times, lf)
                camera = out / f"{sid}_camera.mp4"
                encode(lf, camera, cfg["video"]["crf"], cfg["video"]["preset"], vf=f"scale={cfg['video']['camera_only_width']}:-2")
            meta = probe(review)
            meta_c = probe(camera)
            print(sid, "review", meta, "camera", meta_c)
        status = "作成" if review else ("画像なし (未取得)" if have == 0 else f"画像の一部だけ ({have}/{len(frames)})")
        base = dict(priority=tg.get("priority"), role=tg["role"], sequence_id=sid, car=car, drive_id=s.drive_id,
                    camera_start_utc=cf[0]["time"], camera_end_utc=cf[-1]["time"], n_frames=len(frames), frame_dt_median_s=round(float(np.median(np.diff(times))), 4),
                    images_present=have, status=status, review_video=review.name if review else "", camera_video=camera.name if camera else "",
                    review_duration_s=meta.get("duration_s"), plot_png=f"plots/{plot_png.name}" if plot_png.exists() else "",
                    seq_w_left_est=r3(sw["w_left"]) if sw else None, seq_w_left_sigma=r3(sw["sigma"]) if sw else None, note=tg.get("note", ""))
        if len(ev):
            for _, e in ev.iterrows():
                rows.append(dict(**base, event_id=e.event_id, is_focus=e.event_id in focus, hypothesis=e.hypothesis, kind=e.kind, verdict=e.verdict if pd.notna(e.verdict) else "",
                                 rise_utc=e.t_rise_utc, rise_rel_camera_s=round(float(e.rise_rel_cam_s), 2), rise_censored=bool(e.rise_censored),
                                 exceed_start_utc=e.t_exceed_start_utc, exceed_end_utc=e.t_exceed_end_utc,
                                 detect_rel_camera_s=f"{e.onset_rel_cam_s:+.2f}..{e.end_rel_cam_s:+.2f}", exceed_duration_s=e.exceed_duration_s,
                                 exceed_start_censored=bool(e.exceed_start_censored), exceed_end_censored=bool(e.exceed_end_censored),
                                 peak_utc=e.t_peak_utc, peak_rel_camera_s=round(float(e.peak_rel_cam_s), 2),
                                 overlap_with_camera_s=round(float(e.overlap_s), 2), camera_relation=e.camera_relation,
                                 in_camera=bool(e.overlap_s > 0), event_w_left=r3(e.idw_w_left if pd.notna(e.idw_w_left) else e.nearest_w_left), event_w_left_sigma=r3(e.w_left_sigma)))
        else:
            rows.append(dict(**base, event_id="", is_focus=False, hypothesis="", kind="", verdict="", detect_rel_camera_s="", overlap_with_camera_s=None,
                             in_camera=False, event_w_left=None, event_w_left_sigma=None))
    idx = pd.DataFrame(rows).sort_values(["priority", "sequence_id", "exceed_start_utc"], na_position="first")
    idx.to_csv(out / "index.csv", index=False)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    man = dict(run_id=a.run_id, created_local=datetime.now().isoformat(timespec="seconds"), script_sha256=sha(__file__), config=cfg,
               config_sha256=sha(a.config), ffmpeg=subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2))
    print(idx[["sequence_id", "role", "status", "event_id", "in_camera", "overlap_with_camera_s"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
