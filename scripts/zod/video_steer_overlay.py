#!/usr/bin/env python3
"""前方映像のレビュー動画に、ハンドルと「上から見た向き」のパネルを足す。ネットワークは使わない。

  uv run --offline --extra viz python scripts/zod/video_steer_overlay.py --config configs/zod/front_video_steer.yaml --run-id <id>

対象・入力・動画の作り方は scripts/zod/video_review.py と同じ (その関数を使う)。違いは、カメラの右に置くパネルだけ:

  ハンドル (運転者から見た向き)
    実線のハンドル = 実際の舵角 δ − c (c = 車両のゼロ点 c_car + 走行の補正 Δc、Phase 2 r7)
    青い丸 = 実際のヨー r を舵角のモデルで出すのに要る舵角 δ_need − c  (δ_need = c + (a/v + b·v)·r)
    橙の弧 = 2 つの差 (= −Δr·(a + b·v²)/v、当て舵の量)
    外側の矢印 = 操舵トルクの、走行の中央値からの差の向き
  上から見た図 (上 = 車の向き)
    青 = 実際の進行方向 (今の r で horizon_s 秒回ったときの向き)
    赤 = 舵角から予測した進行方向 (r_pred で同じ)
    緑 = 推定の風 (Phase 3。地面に対する風、10 分平均・10 m。吹いていく向き。角度は誇張しない)
    橙 = 残差から推定した外力の向きと大きさ (横加速度に換算、A_steer = −v·Δr。右向きの矢印 = 右向きの力)

角度 (ハンドルの回転・青と赤の矢印) は angle_gain 倍に誇張する。実際の値は数字で出す。
外力の向きは「空力の作用点が中立操舵点より前」の仮定に依存する (Phase 3 の解釈と同じ)。
時刻の注意 (カメラと車両信号の対応は約 ±1 s) は video_review.py と同じ。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_screening as F  # noqa: E402
import video_review as VR  # noqa: E402

C_ACT = (31, 78, 156)       # 実際 (図の r と同じ色)
C_PRED = (192, 57, 43)      # 舵角から (図の r_pred と同じ色)
C_WIND = (39, 154, 84)
C_FORCE = (230, 126, 34)
C_WHEEL = (52, 60, 70)
C_TEXT = (30, 30, 30)
C_MUTED = (110, 110, 110)


def smooth(x, blk, n):
    out = np.full(len(x), np.nan)
    for k in np.unique(blk):
        m = np.where(blk == k)[0]
        out[m] = pd.Series(x[m].astype(float)).rolling(n, center=True, min_periods=1).mean().to_numpy()
    return out


def seq_signals(z, sid, drive_id, calib, models, oc):
    """パネルに使う 20 Hz の値 (表示用に smooth_s の移動平均)。"""
    cr = calib.loc[drive_id]
    md = models[(models.car == cr.car) & (models.fold == cr.fold)].iloc[0]
    dc = cr.delta_c_deg if bool(cr.applied) and np.isfinite(cr.delta_c_deg) else 0.0
    c = float(cr.c_car_deg + dc)
    n = max(1, int(round(oc["smooth_s"] * 20)))
    blk = z["blk"]
    v = smooth(z["v"], blk, n)
    steer = smooth(z["steer_deg"], blk, n)
    r = smooth(z["r_deg_s"], blk, n)
    rp = smooth(z["r_pred_cor"], blk, n)
    dr = smooth(z["dr_cor"], blk, n)
    tq = smooth(z["torque"], blk, n)
    base = z["mask80"].astype(bool) | z["mask70"].astype(bool)
    tq0 = float(np.nanmedian(z["torque"][base]))
    with np.errstate(all="ignore"):
        need = c + np.degrees((md.a / v + md.b * v) * np.radians(r))
    # 方位 (OXTS、右回りが正) を 20 Hz の時刻に合わせる
    ox = F.oxts_union([sid])
    hd = np.degrees(np.unwrap(np.radians(ox.heading.to_numpy())))
    H = np.interp(z["t"], ox.t.to_numpy(), hd, left=np.nan, right=np.nan)
    return dict(t=z["t"], v=v, steer=steer, need=need, r=r, rp=rp, dr=dr, tq=tq - tq0, H=H, c=c, tq0=tq0,
                a=float(md.a), b=float(md.b), fold=int(cr.fold), delta_c=float(dc), c_car=float(cr.c_car_deg))


def event_wind(w_row):
    """Phase 3 の候補の推定の風 (地面に対する、吹いていく向きの東・北成分)。IDW が無ければ最寄りの観測所。"""
    if pd.notna(w_row.get("idw_u")):
        u, v = float(w_row.idw_u), float(w_row.idw_v)
        src = f"IDW ({int(w_row.n_stations)} 観測所)"
    elif pd.notna(w_row.get("nearest_speed")):
        s, d = float(w_row.nearest_speed), float(w_row.nearest_dir)
        u, v = -s * math.sin(math.radians(d)), -s * math.cos(math.radians(d))
        src = f"最寄りの観測所 ({w_row.nearest_km:.0f} km)"
    else:
        return None
    frm = (math.degrees(math.atan2(-u, -v)) + 360) % 360
    return dict(u=u, v=v, speed=math.hypot(u, v), from_deg=frm, src=src, sigma=float(w_row.w_left_sigma), event_id=w_row.event_id,
                verdict=w_row.verdict)


def compass(d):
    names = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東", "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
    return names[int((d % 360) / 22.5 + 0.5) % 16]


def arrow(d, p0, p1, col, width, head):
    d.line([p0, p1], fill=col, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (1, -1):
        a = ang + math.pi - s * 0.45
        d.line([p1, (p1[0] + head * math.cos(a), p1[1] + head * math.sin(a))], fill=col, width=width)
    tri = [p1, (p1[0] + head * math.cos(ang + math.pi - 0.42), p1[1] + head * math.sin(ang + math.pi - 0.42)),
           (p1[0] + head * math.cos(ang + math.pi + 0.42), p1[1] + head * math.sin(ang + math.pi + 0.42))]
    d.polygon(tri, fill=col)


def onrim(cx, cy, R, phi_deg):
    """画面の上を 0、右回りを正とする角度 φ の、円周上の点。"""
    p = math.radians(phi_deg)
    return cx + R * math.sin(p), cy - R * math.cos(p)


def draw_panel(sig, i, wind, oc, width, height, fonts):
    from PIL import Image, ImageDraw
    S = oc["supersample"]
    W, Hh = width * S, height * S
    im = Image.new("RGB", (W, Hh), (248, 248, 246))
    d = ImageDraw.Draw(im)
    f_t, f_m, f_s = fonts
    g, cap = oc["angle_gain"], oc["max_display_deg"]
    clip = lambda x: float(np.clip(x, -cap, cap))  # noqa: E731
    st, need, c = sig["steer"][i], sig["need"][i], sig["c"]
    r, rp, v, dr, tq, H = sig["r"][i], sig["rp"][i], sig["v"][i], sig["dr"][i], sig["tq"][i], sig["H"][i]
    ok = all(np.isfinite(x) for x in (st, need, r, rp, v, dr))

    # ---------------- ハンドル
    d.text((20 * S, 12 * S), f"ハンドル (運転者から見た向き、回転は ×{g} に誇張)", font=f_t, fill=C_TEXT)
    cx, cy, R = W * 0.36, Hh * 0.255, Hh * 0.15
    if ok:
        th = clip(g * (st - c))           # 左 (正) = 反時計回り
        thn = clip(g * (need - c))
        # 当て舵の弧 (実舵角とヨーに要る舵角の差)
        a0, a1 = sorted([-th, -thn])
        if a1 - a0 > 0.5:
            d.arc([cx - R * 1.0, cy - R * 1.0, cx + R * 1.0, cy + R * 1.0], a0 - 90, a1 - 90, fill=C_FORCE, width=int(16 * S))
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=C_WHEEL, width=int(11 * S))
        d.ellipse([cx - R * 0.2, cy - R * 0.2, cx + R * 0.2, cy + R * 0.2], fill=C_WHEEL)
        for sp in (90, 180, 270):
            d.line([onrim(cx, cy, R * 0.18, sp - th), onrim(cx, cy, R * 0.95, sp - th)], fill=C_WHEEL, width=int(9 * S))
        p = onrim(cx, cy, R, -th)
        d.ellipse([p[0] - 11 * S, p[1] - 11 * S, p[0] + 11 * S, p[1] + 11 * S], fill=C_PRED)
        q = onrim(cx, cy, R, -thn)
        d.ellipse([q[0] - 11 * S, q[1] - 11 * S, q[0] + 11 * S, q[1] + 11 * S], outline=C_ACT, width=int(4 * S))
        d.line([(cx, cy - R - 26 * S), (cx, cy - R - 6 * S)], fill=C_MUTED, width=int(2 * S))   # まっすぐの位置 (ゼロ点)
        # トルク (走行の中央値からの差)。正 = 左 (トルクと v·r の相関の符号、Phase 3 と同じ)
        if np.isfinite(tq) and abs(tq) > 0.2:
            span = float(np.clip(abs(tq) * 20, 8, 70))
            R2 = R * 1.22
            if tq > 0:
                b0, b1 = -10 - span, -10
            else:
                b0, b1 = 10, 10 + span
            d.arc([cx - R2, cy - R2, cx + R2, cy + R2], b0 - 90, b1 - 90, fill=C_MUTED, width=int(6 * S))
            tip_phi = b0 if tq > 0 else b1
            tip = onrim(cx, cy, R2, tip_phi)
            prev = onrim(cx, cy, R2, tip_phi + (4 if tq > 0 else -4))
            arrow(d, prev, tip, C_MUTED, int(6 * S), 16 * S)
    tx = W * 0.62
    lines = [
        (f"舵角 (ゼロ点 {c:.2f}° から)", f"{st - c:+.2f}°", C_PRED),
        ("実際のヨーに要る舵角", f"{need - c:+.2f}°", C_ACT),
        ("差 = 当て舵の量", f"{st - need:+.2f}°", C_FORCE),
        ("トルク (走行の中央値から)", f"{tq:+.2f} N·m", C_MUTED),
    ] if ok else [("値なし (OXTS・車両データの外)", "", C_MUTED)]
    for k, (lab, val, col) in enumerate(lines):
        y = Hh * 0.10 + k * 62 * S
        d.text((tx, y), lab, font=f_s, fill=C_MUTED)
        d.text((tx, y + 24 * S), val, font=f_m, fill=col)
    d.text((20 * S, Hh * 0.445), "赤い点 = 実際の舵角、青い丸 = 実際のヨーに要る舵角、灰の矢印 = トルクの向き",
           font=f_s, fill=C_MUTED)

    # ---------------- 上から見た図
    y0 = Hh * 0.49
    d.line([(10 * S, y0), (W - 10 * S, y0)], fill=(210, 210, 210), width=int(2 * S))
    d.text((20 * S, y0 + 10 * S), f"上から見た図 (上 = 車の向き。青・赤は {oc['horizon_s']:.0f} s 後の向き、×{g})", font=f_t, fill=C_TEXT)
    ccx, ccy = W * 0.30, Hh * 0.80
    cw, ch = 34 * S, 74 * S
    d.rounded_rectangle([ccx - cw, ccy - ch, ccx + cw, ccy + ch], radius=14 * S, fill=(225, 228, 232), outline=C_WHEEL, width=int(3 * S))
    d.polygon([(ccx - cw * 0.7, ccy - ch * 0.55), (ccx + cw * 0.7, ccy - ch * 0.55), (ccx + cw * 0.55, ccy - ch * 0.2), (ccx - cw * 0.55, ccy - ch * 0.2)],
              fill=(180, 190, 200))
    nose = (ccx, ccy - ch)
    L = Hh * 0.13
    if ok:
        for val, col in ((rp, C_PRED), (r, C_ACT)):
            phi = -clip(g * val * oc["horizon_s"])
            arrow(d, nose, onrim(nose[0], nose[1], L, phi), col, int(7 * S), 26 * S)
        # 外力 (横加速度に換算)。A_steer = −v·Δr > 0 → 右向きの力
        a_st = -v * math.radians(dr)
        fl = float(np.clip(a_st * oc["force_px_per_ms2"] * S, -W * 0.28, W * 0.28))
        if abs(fl) > 4 * S:
            arrow(d, (ccx + math.copysign(cw, fl), ccy), (ccx + math.copysign(cw, fl) + fl, ccy), C_FORCE, int(9 * S), 26 * S)
    if wind is not None and np.isfinite(H):
        to_bearing = math.degrees(math.atan2(wind["u"], wind["v"]))
        phi = to_bearing - H
        lw = wind["speed"] * oc["wind_px_per_ms"] * S
        dx, dy = math.sin(math.radians(phi)), -math.cos(math.radians(phi))
        tip = (ccx - dx * (cw + 30 * S), ccy - dy * (cw + 30 * S))
        arrow(d, (tip[0] - dx * lw, tip[1] - dy * lw), tip, C_WIND, int(9 * S), 28 * S)
    tx = W * 0.60
    rows = []
    if ok:
        a_st = -v * math.radians(dr)
        rows += [("実際の進行方向 (OXTS の r)", f"{r:+.2f} deg/s", C_ACT),
                 ("舵角から予測した向き (r_pred)", f"{rp:+.2f} deg/s", C_PRED),
                 ("残差から推定した外力 (横加速度)", f"{abs(a_st):.2f} m/s² {'右' if a_st > 0 else '左'}向き", C_FORCE)]
    if wind is not None and np.isfinite(H):
        wl = wind["u"] * math.cos(math.radians(H)) - wind["v"] * math.sin(math.radians(H))
        rows += [(f"推定の風 ({wind['src']})", f"{wind['speed']:.1f} m/s {compass(wind['from_deg'])}から", C_WIND),
                 ("  そのうち横風 (左から正)", f"{wl:+.1f} ± {wind['sigma']:.1f} m/s", C_WIND)]
    for k, (lab, val, col) in enumerate(rows):
        y = y0 + 60 * S + k * 62 * S
        d.text((tx, y), lab, font=f_s, fill=C_MUTED)
        d.text((tx, y + 24 * S), val, font=f_m, fill=col)
    notes = [f"角度は ×{g} に誇張 (実際の値は数字)。表示は {oc['smooth_s']} s の移動平均",
             "風: Phase 3 の推定 (観測所の 10 分平均・10 m)。この 20 s は一定と置いた",
             "外力: 残差 Δr から推定。作用点が中立操舵点より前という仮定による"]
    for k, s in enumerate(notes):
        d.text((20 * S, Hh - (3 - k) * 26 * S - 8 * S), s, font=f_s, fill=C_MUTED)
    return im.resize((width, height), Image.LANCZOS)


def render(frames, times, c0, plot_pack, ev, focus, header_lines, sig, wind, out_mp4, cfg, oc, tmp: Path):
    from PIL import Image, ImageDraw, ImageFont
    vc = cfg["video"]
    Wc, Wp = vc["camera_width"], oc["panel_width"]
    font = ImageFont.truetype(vc["font"], 26)
    small = ImageFont.truetype(vc["font"], 20)
    S = oc["supersample"]
    fonts = tuple(ImageFont.truetype(vc["font"], s * S) for s in (21, 24, 15))
    pimg, mp = plot_pack
    ph = pimg.shape[0]
    fev = ev[ev.event_id.isin(focus)] if len(focus) else ev
    paths = []
    for k, (fp, t) in enumerate(zip(frames, times)):
        im = Image.open(fp)
        im.draft("RGB", (im.width // 2, im.height // 2))
        im = im.convert("RGB")
        h = int(round(Wc * im.height / im.width / 2) * 2)
        im = im.resize((Wc, h), Image.BILINEAR)
        canvas = Image.new("RGB", (Wc + Wp, h + ph + (ph % 2)), "white")
        canvas.paste(im, (0, 0))
        i = int(np.argmin(np.abs(sig["t"] - t)))
        if abs(sig["t"][i] - t) < 0.1:
            canvas.paste(draw_panel(sig, i, wind, oc, Wp, h, fonts), (Wc, 0))
        pl = Image.fromarray(pimg)
        d = ImageDraw.Draw(pl)
        x = mp["x0"] + (t - c0 - mp["t0"]) / (mp["t1"] - mp["t0"]) * (mp["x1"] - mp["x0"])
        d.line([(x, mp["ytop"]), (x, mp["ybot"])], fill=(220, 20, 60), width=3)
        canvas.paste(pl, (0, h))
        dc = ImageDraw.Draw(canvas, "RGBA")
        lines = list(header_lines) + [f"フレーム {VR.utc(t)} UTC   カメラの開始から {t - c0:+.2f} s"]
        for _, e in fev.iterrows():
            lines.append(f"{e.event_id}: 閾値超過の始まりから {t - e.det_t0:+.2f} s (推定開始から {t - e.t_rise:+.2f} s)")
        box_h = 12 + 32 * len(lines)
        dc.rectangle([0, 0, 900, box_h], fill=(0, 0, 0, 150))
        for j, s in enumerate(lines):
            dc.text((12, 8 + 32 * j), s, font=font if j < len(header_lines) else small, fill=(255, 255, 255))
        inside = [e for _, e in ev.iterrows() if e.det_t0 <= t <= e.det_t1]
        if inside:
            msg = "検出区間の中: " + ", ".join(f"{e.event_id} ({str(e.hypothesis)[:6]}…)" for e in inside)
            col = (192, 57, 43, 220)
        else:
            msg = "候補の検出区間の外"
            col = (90, 90, 90, 200)
        tw = dc.textlength(msg, font=font)
        dc.rectangle([Wc - tw - 30, 0, Wc, 44], fill=col)
        dc.text((Wc - tw - 15, 7), msg, font=font, fill=(255, 255, 255))
        p = tmp / f"{k:04d}.jpg"
        canvas.save(p, quality=90)
        paths.append(p)
    lf = tmp / "review.txt"
    VR.concat_list(paths, times, lf)
    VR.encode(lf, out_mp4, vc["crf"], vc["preset"])
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zod/front_video_steer.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args()
    ocfg = yaml.safe_load(Path(a.config).read_text())
    cfg = yaml.safe_load(Path(ocfg["base_config"]).read_text())
    oc = ocfg["overlay"]
    out = Path(cfg["out_dir"]) / a.run_id
    if out.exists():
        print(f"{out} がすでにある。上書きしない", file=sys.stderr)
        return 1
    (out / "plots").mkdir(parents=True)
    (out / "stills").mkdir()
    seqs, cand, seg, calib, thr = VR.load_inputs(cfg)
    scr, wr, raw = Path(cfg["screening_run"]), Path(cfg["weather_run"]), Path(cfg["raw_root"])
    models = pd.read_csv(scr / "model_by_fold.csv")
    wdf = pd.read_csv(wr / "candidates_weather.csv")
    tq_sign = json.loads((wr / "summary.json").read_text())["log"]["torque_sign_used"]
    if tq_sign != 1.0:
        raise RuntimeError("トルクの符号が Phase 3 と違う (この表示は正 = 左を前提にしている)")
    F.ROOT = raw
    rows = []
    for tg in cfg["targets"]:
        sid = tg["sequence"]
        if a.only and sid not in a.only:
            continue
        s = seqs.loc[sid]
        info = json.loads((raw / "sequences" / sid / "info.json").read_text())
        cf = info["camera_frames"]["front_blur"]
        times = np.array([VR.ts(f["time"]) for f in cf])
        frames = [raw / f["filepath"] for f in cf]
        if not all(p.exists() for p in frames):
            print(sid, "画像が無い。飛ばす")
            continue
        c0, c1 = times[0], times[-1]
        ev = VR.sequence_events(s, cand, c0, c1, cfg, tg.get("focus", []))
        focus = [e for e in tg.get("focus", []) if e in set(ev.event_id)]
        fold = calib.fold.get(s.drive_id)
        tsel = thr[(thr.car == s.car) & (thr.fold == fold) & (thr.speed_kmh == 80) & (thr.residual == "cor")]
        thr_slow = float(tsel[tsel.kind == "slow"].threshold_deg_s.iloc[0]) if len(tsel) else np.nan
        thr_fast = float(tsel[tsel.kind == "fast"].threshold_deg_s.iloc[0]) if len(tsel) else np.nan
        z = dict(np.load(scr / "residuals" / f"{s.drive_id}.npz"))
        sig = seq_signals(z, sid, s.drive_id, calib, models, oc)
        wrow = wdf[wdf.event_id.isin(focus)]
        wind = event_wind(wrow.iloc[0]) if len(wrow) else None
        fev = ev[ev.event_id.isin(focus)]
        wtxt = ""
        if len(fev):
            e = fev.iloc[0]
            wl = e.idw_w_left if pd.notna(e.idw_w_left) else e.nearest_w_left
            if pd.notna(wl):
                wtxt = f"推定の横風 (10 m) {wl:+.1f} ± {e.w_left_sigma:.1f} m/s (左から正)、判定: {e.verdict}"
        title = f"{sid} {s.car} {s.drive_id}  カメラ {VR.utc(c0)}〜{VR.utc(c1)} UTC  [{tg['role']}]  {wtxt}"
        fig, pack = VR.make_plot(z, c0, c1, ev, thr_slow, thr_fast, title, cfg["video"]["camera_width"] + oc["panel_width"], cfg)
        import matplotlib.pyplot as plt
        fig.savefig(out / "plots" / f"{sid}_timeseries.png", dpi=100)
        plt.close(fig)
        header = [f"{sid}  {s.car}  [{tg['role']}]  {datetime.fromtimestamp(c0, timezone.utc):%Y-%m-%d} UTC"]
        for _, e in fev.iterrows():
            if e.overlap_s <= 0:
                where = "前" if e.det_t1 <= c0 else "後"
                header.append(f"注意: {e.event_id} の閾値超過区間はカメラの{where} (カメラ開始から {e.onset_rel_cam_s:+.1f}〜{e.end_rel_cam_s:+.1f} s)。この映像は事象そのものではない")
            elif e.camera_relation != "窓の中":
                header.append(f"注意: {e.event_id} の閾値超過区間は一部だけカメラに入る ({e.camera_relation}、重なり {e.overlap_s:.1f} s)")
        if wtxt:
            header.append(wtxt)
        mp4 = out / f"{sid}_review_steer.mp4"
        with tempfile.TemporaryDirectory(dir=a.tmp) as td:
            paths = render(frames, times, c0, pack, ev, focus, header, sig, wind, mp4, cfg, oc, Path(td))
            # 静止画: 注目の候補のピーク (カメラの中なら) と、カメラの中央
            picks = {"mid": 0.5 * (c0 + c1)}
            for _, e in fev.iterrows():
                if c0 <= e.t_peak <= c1:
                    picks[f"{e.event_id}_peak"] = e.t_peak
            from PIL import Image
            for name, tt in picks.items():
                j = int(np.argmin(np.abs(times - tt)))
                Image.open(paths[j]).save(out / "stills" / f"{sid}_{name}.jpg", quality=88)
        meta = VR.probe(mp4)
        print(sid, meta)
        # 確かめ: 注目の候補の閾値超過区間で、当て舵の量 (st − need) の平均
        chk = {}
        for _, e in fev.iterrows():
            m = (sig["t"] >= e.det_t0) & (sig["t"] <= max(e.det_t1, e.det_t0 + 2))
            chk[e.event_id] = round(float(np.nanmean(sig["steer"][m] - sig["need"][m])), 3)
        rows.append(dict(sequence_id=sid, car=s.car, drive_id=s.drive_id, role=tg["role"], focus=" ".join(focus), video=mp4.name, **meta,
                         zero_point_c_deg=round(sig["c"], 3), c_car_deg=sig["c_car"], delta_c_deg=sig["delta_c"], model_fold=sig["fold"],
                         model_a=sig["a"], model_b=sig["b"], torque_drive_median_nm=round(sig["tq0"], 3),
                         wind_source=wind["src"] if wind else "", wind_speed_ms=round(wind["speed"], 2) if wind else None,
                         wind_from_deg=round(wind["from_deg"], 0) if wind else None, wind_event=wind["event_id"] if wind else "",
                         countersteer_mean_in_exceedance_deg=json.dumps(chk)))
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "index.csv", index=False)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    man = dict(run_id=a.run_id, created_local=datetime.now().isoformat(timespec="seconds"),
               scripts={str(p): sha(p) for p in [Path(__file__), Path(VR.__file__)]},
               config=ocfg, config_sha256=sha(a.config), base_config=cfg, base_config_sha256=sha(ocfg["base_config"]),
               ffmpeg=subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2, default=str))
    print(idx[["sequence_id", "video", "duration_s", "countersteer_mean_in_exceedance_deg"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
