"""人のレビュー用の動画・図・閲覧ページを作る (伏せた番号 V01〜 で、群や旗は見せない)。

1 フレーム = 前方の画像 (上) + 時系列 (左下) + GPS の軌跡と今の位置 (右下)。時系列は IMU の時刻にそろえ、今の画像の時刻に縦線を引く:
  1. MTVV_total (1 秒移動) と前後 ±30 s の中央値
  2. 上下加速度
  3. ヨー角速度 (そのまま = 旋回、と 0.5〜3 Hz = 急な変化) と舵角 (時刻は未補正)
  4. ロール角速度
  5. ロール角・ピッチ角 (前後 ±10 s の中央値からの差。gps_odom の姿勢)
  6. 後輪のストローク (racepak の時刻を補正) と後輪の回転の左右差
  7. 速度・ブレーキとみられる信号・スロットルとみられる信号 (補正)
画像は KITTI 形式の tar から取る (td2_tar_index.py の一覧と tar の中の timestamps.txt)。取ったものはキャッシュに残る。
閲覧ページ index.html は、2 本を並べて同時に再生・コマ送りでき、レビューの入力をブラウザに保存して CSV に書き出せる。

    uv run python scripts/offroad/td2_review_media.py --review out/offroad/tartandrive2/<review_set の ID> \
        --profile out/offroad/tartandrive2/20261117_r20 --sync out/offroad/tartandrive2/20261117_r15 --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

import rosbag_range as RB
import td2_candidate_profile as CP
import td2_event_media as EM
import td2_vibration_poc as P


def signals(run: str, cfg: dict, ST: pd.DataFrame) -> dict:
    sig = EM.signals(run, cfg, ST)
    R = sig["R"]
    Sh = P.shock_series(R, cfg)
    P.run_checks(R, Sh, np.nan)
    R["wz_lp"] = P.lp(R["X"]["wz"], cfg["review"]["yaw_lp_hz"], R["fs"])
    return sig


def draw(sig: dict, te: float, c: pd.Series, cfg: dict, quality: str):
    R, S, C = sig["R"], sig["S"], sig["C"]
    rc = cfg["review"]
    span = rc["media_span_s"]
    g = R["g"]
    lo, hi = te + span[0] - 1, te + span[1] + 1
    m = (g >= lo) & (g <= hi)
    x = g[m] - te
    fig = plt.figure(figsize=(10.24, 8.6), dpi=100)
    gs = fig.add_gridspec(7, 3, width_ratios=[1, 1, 0.75], hspace=0.12, wspace=0.25, left=0.07, right=0.98, top=0.95, bottom=0.06)
    ax = [fig.add_subplot(gs[i, :2]) for i in range(7)]
    for a_ in ax[1:]:
        a_.sharex(ax[0])
    ax[0].plot(x, S["tot"][m], color="#1f4e9c", label="MTVV_total [m/s²]")
    ax[0].axhline(c.mtvv_total / c.contrast, color="#6b7280", ls="--", lw=1, label="前後 ±30 s の中央値")
    ax[1].plot(x, S["raw"]["z"][m], color="#374151", lw=0.7, label="上下加速度 [m/s²]")
    ax[2].plot(x, np.degrees(R["X"]["wz"][m]), color="#0f766e", lw=0.8, label="ヨー角速度 [deg/s]")
    ax[2].plot(x, np.degrees(R["wz_lp"][m]), color="#0f766e", lw=2, alpha=0.5, label="同 0.5 Hz 以下 (旋回)")
    ax[2].plot(x, np.degrees(R["wz_hp"][m]), color="#c2410c", lw=1.0, label="同 0.5〜3 Hz (急な変化)")
    if R["steer"] is not None:
        tsr, s = R["steer"]
        mst = (tsr >= lo) & (tsr <= hi)
        a2 = ax[2].twinx()
        a2.plot(tsr[mst] - te, s[mst, 0], color="#a16207", lw=1, ls=":", label="舵角 (単位未確認・時刻未補正)")
        a2.legend(fontsize=6, loc="upper right")
        a2.tick_params(labelsize=6)
    ax[3].plot(x, np.degrees(R["X"]["wx"][m]), color="#7c3aed", lw=0.8, label="ロール角速度 [deg/s]")
    ctx = (g >= te - 10) & (g <= te + 10)
    ax[4].plot(x, np.degrees(R["roll_g"][m] - np.median(R["roll_g"][ctx])), color="#7c3aed", label="ロール角 (前後 ±10 s の中央値から) [deg]")
    ax[4].plot(x, np.degrees(R["pitch_g"][m] - np.median(R["pitch_g"][ctx])), color="#16a34a", label="ピッチ角 (同) [deg]")
    if "shock" in C:
        ts, a = C["shock"]
        ms = (ts >= lo) & (ts <= hi)
        ax[5].plot(ts[ms] - te, a[ms, 2], color="#c2410c", label="ストローク RL (大きいほど縮み)")
        ax[5].plot(ts[ms] - te, a[ms, 3], color="#0f766e", label="ストローク RR")
    if "wheel" in C:
        from scipy.signal import medfilt
        tw, wv = C["wheel"]
        mw = (tw >= lo) & (tw <= hi)
        rl, rr = medfilt(wv[mw, 2].astype(float), 5), medfilt(wv[mw, 3].astype(float), 5)
        a5 = ax[5].twinx()
        a5.plot(tw[mw] - te, rl - rr, color="#6b7280", lw=1, ls="--", label="後輪の回転 左 − 右 [rpm]")
        a5.legend(fontsize=6, loc="upper right")
        a5.tick_params(labelsize=6)
    ax[6].plot(x, R["speed"][m], color="#111827", label="速度 [m/s]")
    if "controls" in C:
        tc, cc = C["controls"]
        mc = (tc >= lo) & (tc <= hi)
        brk, thr = P.pedals(cc, cfg)          # 列の割り当ては設定 (reaction.controls_cols。2026-11-18 に名前と中身が逆とわかった)
        a6 = ax[6].twinx()
        if cfg["reaction"].get("controls_cols"):
            a6.plot(tc[mc] - te, brk[mc] * 10, color="#dc2626", lw=1, label="ブレーキとみられる信号 ×10")
            a6.plot(tc[mc] - te, thr[mc], color="#16a34a", lw=1, label="アクセルとみられる信号")
        else:
            a6.plot(tc[mc] - te, brk[mc], color="#dc2626", lw=1, label="ブレーキとみられる信号")
            a6.plot(tc[mc] - te, thr[mc] * 10, color="#16a34a", lw=1, label="スロットルとみられる信号 ×10")
        a6.legend(fontsize=6, loc="upper right")
        a6.tick_params(labelsize=6)
    for a_ in ax:
        a_.legend(fontsize=6, loc="upper left")
        a_.grid(alpha=0.3)
        a_.tick_params(labelsize=7)
    for a_ in ax[:-1]:
        plt.setp(a_.get_xticklabels(), visible=False)
    ax[0].set_xlim(span[0] - 1, span[1] + 1)
    ax[6].set_xlabel("te からの秒 (te = 1 秒移動 MTVV の最大。衝撃は 0〜1 s 前)", fontsize=8)
    # 軌跡
    am = fig.add_subplot(gs[:4, 2])
    to, o = R["odom"]
    mm = (to >= te - rc["map_span_s"]) & (to <= te + rc["map_span_s"])
    p = o[mm, :2] - o[np.argmin(np.abs(to - te)), :2]
    am.plot(p[:, 1], p[:, 0], color="#9aa5b1", lw=2)
    mw_ = (to[mm] >= te + span[0]) & (to[mm] <= te + span[1])
    am.plot(p[mw_, 1], p[mw_, 0], color="#1f4e9c", lw=3, label="動画の範囲")
    am.plot(p[0, 1], p[0, 0], "o", color="#9aa5b1", ms=4)
    cur = am.plot([0], [0], "o", color="#f59e0b", ms=8)[0]
    am.set_aspect("equal", adjustable="datalim")
    am.set_title(f"GPS の軌跡 (te ±{rc['map_span_s']:.0f} s、灰の点が始まり)", fontsize=7)
    am.tick_params(labelsize=6)
    am.legend(fontsize=6, loc="lower right")
    # 品質の注意 (文字)
    at = fig.add_subplot(gs[4:, 2])
    at.axis("off")
    import textwrap
    at.text(0, 1, "信号の品質の注意:\n" + "\n".join(textwrap.fill(q, 22) for q in quality.split(";")), fontsize=7, va="top")
    fig.suptitle(f"{c.review_id}   {c.run}   t = {c.t:.1f} s   速度 {c.speed:.1f} m/s", fontsize=9)
    cursors = [a_.axvline(0, color="#f59e0b", lw=2) for a_ in ax]
    ptimes = to[mm]
    return fig, cursors, cur, (ptimes, p)


HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>TartanDrive 2.0 振動の候補のレビュー</title>
<style>
:root{--bg:#f7f7f5;--fg:#1f2328;--mut:#59636e;--line:#d0d7de;--acc:#1f4e9c}
@media (prefers-color-scheme: dark){:root{--bg:#16181b;--fg:#e6e6e6;--mut:#9aa3ad;--line:#30363d;--acc:#7aa7ff}}
body{margin:0;font-family:-apple-system,"Hiragino Sans","Yu Gothic UI",sans-serif;background:var(--bg);color:var(--fg);font-size:14px}
header{padding:10px 16px;border-bottom:1px solid var(--line)} h1{font-size:17px;margin:0 0 4px}
.wrap{display:grid;grid-template-columns:300px 1fr;gap:0;height:calc(100vh - 62px)}
#list{overflow:auto;border-right:1px solid var(--line)} #list table{border-collapse:collapse;width:100%;font-size:12px}
#list td,#list th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left} #list tr{cursor:pointer}
#list tr.sel{background:rgba(31,78,156,.15)} #list tr.done td:first-child::after{content:" ✓";color:#16a34a}
main{overflow:auto;padding:10px 14px} .players{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.pl{border:1px solid var(--line);border-radius:6px;padding:6px} .pl h2{font-size:14px;margin:2px 0 6px}
video{width:100%;background:#000;border-radius:4px} .ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0}
button,select{font:inherit;padding:3px 8px;border:1px solid var(--line);border-radius:4px;background:transparent;color:var(--fg);cursor:pointer}
.info{font-size:12px;color:var(--mut);white-space:pre-wrap;margin:4px 0}
form{display:grid;grid-template-columns:170px 1fr;gap:4px 8px;font-size:12px;margin-top:6px} form label{color:var(--mut)}
textarea{font:inherit;width:100%;min-height:40px;background:transparent;color:var(--fg);border:1px solid var(--line)}
.opts span{display:inline-block;margin-right:8px;white-space:nowrap}
</style></head><body>
<header><h1>TartanDrive 2.0 振動の候補のレビュー (__N__ 件、伏せた番号)</h1>
<div class="info">左の一覧で選ぶと左の枠に、Shift を押しながら選ぶと右の枠に入ります。「同時に再生」で 2 本をそろえて再生します。入力はこのブラウザに保存されます。終わったら「CSV に書き出す」。映像だけで判断できない項目は「不明」を選んでください。</div></header>
<div class="wrap"><div id="list"><table><thead><tr><th>番号</th><th>速度</th><th>状況 (自動)</th></tr></thead><tbody></tbody></table></div>
<main><div class="ctl"><button id="both">同時に再生 / 停止</button><button id="back">◀ 0.3 s</button><button id="fwd">0.3 s ▶</button>
<label>速さ <select id="rate"><option>0.25</option><option>0.5</option><option selected>1</option></select></label>
<label>動画 <select id="kind"><option value="">実時間</option><option value="_slow3x">3 倍遅い</option></select></label>
<button id="export">CSV に書き出す</button><label>CSV を読み込む <input type="file" id="imp" accept=".csv"></label></div>
<div class="players"><div class="pl" id="L"></div><div class="pl" id="R"></div></div></main></div>
<script>
const CASES=__CASES__; const FORM=__FORM__; const KEY="td2_review___RUNID__";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){st={}}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}; mark()};
const tb=document.querySelector("#list tbody");
CASES.forEach(c=>{const tr=document.createElement("tr");tr.dataset.id=c.review_id;tr.innerHTML=`<td>${c.review_id}</td><td>${c.speed.toFixed(1)}</td><td>${c.auto_context}</td>`;
 tr.onclick=e=>show(e.shiftKey?"R":"L",c.review_id);tb.appendChild(tr)});
function mark(){tb.querySelectorAll("tr").forEach(tr=>{const s=st[tr.dataset.id]||{};tr.classList.toggle("done",FORM.filter(f=>f.opts).every(f=>s[f.col]))})}
const cur={L:null,R:null};
function show(side,id){cur[side]=id;const c=CASES.find(x=>x.review_id===id);const el=document.getElementById(side);const k=document.getElementById("kind").value;
 const s=st[id]||{};
 el.innerHTML=`<h2>${id}</h2><video src="${id}${k}.mp4" controls preload="auto"></video>
 <div class="info">図: <a href="${id}_sheet.png" target="_blank">${id}_sheet.png</a>\n状況 (自動、参考): ${c.auto_context}  ヨー角速度 (0.5 Hz 以下) ${c.yaw_rate_lp_dps.toFixed(1)} deg/s、向きの変化 ${c.heading_change_deg.toFixed(0)}°、ロール角 ${c.roll_base_deg.toFixed(1)}°、ピッチ角 ${c.pitch_base_deg.toFixed(1)}°\n信号の品質の注意: ${c.quality_notes}</div>
 <form>${FORM.map(f=>`<label>${f.label}</label><div class="opts">${f.opts?f.opts.map(o=>`<span><input type="radio" name="${side}_${f.col}" value="${o}" ${s[f.col]===o?"checked":""}> ${o}</span>`).join(""):`<textarea name="${side}_${f.col}">${s[f.col]||""}</textarea>`}</div>`).join("")}</form>`;
 el.querySelectorAll("input,textarea").forEach(inp=>inp.addEventListener("change",()=>{const col=inp.name.slice(2);st[id]=st[id]||{};st[id][col]=inp.value;save()}));
 el.querySelector("video").playbackRate=+document.getElementById("rate").value;
 tb.querySelectorAll("tr").forEach(tr=>tr.classList.toggle("sel",tr.dataset.id===cur.L||tr.dataset.id===cur.R))}
const vids=()=>[...document.querySelectorAll("video")];
document.getElementById("both").onclick=()=>{const v=vids();if(!v.length)return;if(v.some(x=>!x.paused)){v.forEach(x=>x.pause())}else{v.forEach(x=>{x.currentTime=0;x.play()})}};
document.getElementById("back").onclick=()=>vids().forEach(v=>{v.pause();v.currentTime=Math.max(0,v.currentTime-0.3)});
document.getElementById("fwd").onclick=()=>vids().forEach(v=>{v.pause();v.currentTime+=0.3});
document.getElementById("rate").onchange=e=>vids().forEach(v=>v.playbackRate=+e.target.value);
document.getElementById("kind").onchange=()=>{["L","R"].forEach(s=>cur[s]&&show(s,cur[s]))};
document.getElementById("export").onclick=()=>{const cols=["review_id",...FORM.map(f=>f.col)];const q=v=>'"'+String(v??"").replace(/"/g,'""')+'"';
 const lines=[cols.join(",")].concat(CASES.map(c=>[c.review_id,...FORM.map(f=>(st[c.review_id]||{})[f.col]||"")].map(q).join(",")));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["﻿"+lines.join("\n")],{type:"text/csv"}));a.download="review_result.csv";a.click()};
document.getElementById("imp").onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(t=>{const rows=t.replace(/^﻿/,"").trim().split(/\n/).map(r=>r.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map(x=>x.replace(/,$/,"").replace(/^"|"$/g,"").replace(/""/g,'"')));
 const h=rows[0];rows.slice(1).forEach(r=>{const id=r[0];st[id]={};h.slice(1).forEach((c,i)=>{if(r[i+1])st[id][c]=r[i+1]})});save()})};
mark(); if(CASES.length){show("L",CASES[0].review_id)}
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--source", default="configs/offroad/tartandrive2.yaml")
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--sync", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--only", default="", help="この番号だけ作る (試し用)")
    ap.add_argument("--out-root", default="out/offroad/tartandrive2")
    ap.add_argument("--span-s", type=float, nargs=2, default=None, help="動画の範囲 (te から、設定の media_span_s を上書き)")
    ap.add_argument("--frame-step", type=int, default=None, help="10 Hz の何枚に 1 枚を使うか (設定の media_frame_step を上書き)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    rc = cfg["review"]
    if a.span_s:
        rc["media_span_s"] = list(a.span_s)
    if a.frame_step:
        rc["media_frame_step"] = a.frame_step
    src = yaml.safe_load(open(a.source))
    root = Path(src["data_dir"])
    RB.CACHE_DIR = root / "bag_cache"
    out = Path(a.out_root) / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    L = pd.read_csv(a.review / "review_list.csv")
    F = pd.read_csv(a.profile / "candidates.csv")
    key = pd.read_csv(a.review / "key.csv")
    L = L.merge(key[["review_id", "cand_id"]], on="review_id").merge(F[["cand_id", "mtvv_total", "contrast"]], on="cand_id")
    ST = pd.read_csv(a.sync / "sync_states.csv")
    if a.only:
        L = L[L.review_id.isin(a.only.split(","))]
    cache, log, used = {}, [], 0
    for _, c in L.sort_values(["run", "t"]).iterrows():
        if c.run not in cache:
            cache = {c.run: (signals(c.run, cfg, ST), EM.Frames(src, root, c.run, cfg["media"]["tar"]))}  # 走行ごとに入れ替える (メモリ)
        sig, fr = cache[c.run]
        te = float(c.t_abs)
        span = rc["media_span_s"]
        ks = list(range(fr.k_at(te + span[0]), fr.k_at(te + span[1]) + 1, rc["media_frame_step"]))
        fig, cursors, cur, (pt, pp) = draw(sig, te, c, cfg, c.quality_notes)
        fdir = out / f"{c.review_id}_frames"
        fdir.mkdir(exist_ok=True)
        before = fr.used
        thumbs = []
        for n, k in enumerate(ks):
            im = fr.get(k)
            if im is None:
                continue
            tk = fr.t_of(k) - te
            for l in cursors:
                l.set_xdata([tk, tk])
            j = int(np.argmin(np.abs(pt - (te + tk))))
            cur.set_data([pp[j, 1]], [pp[j, 0]])
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            panel = Image.open(buf).convert("RGB")
            frame = Image.new("RGB", (1024, im.height + panel.height), "white")
            frame.paste(im.resize((1024, int(im.height * 1024 / im.width))), (0, 0))
            frame.paste(panel, (0, im.height))
            ImageDraw.Draw(frame).text((8, 6), f"{c.review_id}  {tk:+.1f} s", fill="yellow")
            frame.save(fdir / f"{n:03d}.png")
            thumbs.append((tk, im))
        plt.close(fig)
        used += fr.used - before
        fps = 10 / rc["media_frame_step"]
        for suffix, rate in (("", fps), ("_slow3x", fps / 3)):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{rate}", "-i", str(fdir / "%03d.png"),
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                            "-movflags", "+faststart", str(out / f"{c.review_id}{suffix}.mp4")], check=True)
        # 図 (画像 6 枚の並び + 最後のフレームの時系列)
        sel = [thumbs[i] for i in np.linspace(0, len(thumbs) - 1, min(6, len(thumbs))).round().astype(int)]
        tw = 1024 // len(sel)
        th = int(544 * tw / 1024)
        strip = Image.new("RGB", (1024, th + 16), "white")
        dr = ImageDraw.Draw(strip)
        for n, (tk, im) in enumerate(sel):
            strip.paste(im.resize((tw, th)), (n * tw, 16))
            dr.text((n * tw + 4, 2), f"{tk:+.1f} s", fill="black")
        last = Image.open(sorted(fdir.glob("*.png"))[-1])
        sheet = Image.new("RGB", (1024, strip.height + last.height - 544), "white")
        sheet.paste(strip, (0, 0))
        sheet.paste(last.crop((0, 544, 1024, last.height)), (0, strip.height))
        sheet.save(out / f"{c.review_id}_sheet.png")
        shutil.rmtree(fdir)                      # 中間の画像は残さない (動画と図があれば足りる)
        log.append(dict(review_id=c.review_id, frames=len(thumbs), bytes=fr.used - before))
        print(log[-1], f"累計 {used / 1e6:.1f} MB", flush=True)
    form = [dict(col="cause_visible", label="路面の原因が映像で見えるか", opts=["見える", "見えない", "不明"]),
            dict(col="road_cause", label="原因 (見える場合)", opts=["凹凸・わだち", "段差・縁", "溝・くぼみ", "岩", "斜面の乗り越え", "草・茂みで隠れる", "その他", "不明"]),
            dict(col="vehicle_behavior", label="車両の挙動", opts=["振動だけ", "旋回に伴う", "斜面に沿う", "姿勢・向きが乱れる", "不明"]),
            dict(col="expected_or_unexpected", label="運転から予想できる動きか", opts=["予想どおり", "予想外", "不明"]),
            dict(col="artifact_suspected", label="信号の偽の異常の疑い", opts=["なし", "あり", "不明"]),
            dict(col="safety_relevant", label="安全に関わりそうか", opts=["関わらない", "関わりそう", "不明"]),
            dict(col="confidence", label="判断の確かさ", opts=["高", "中", "低"]),
            dict(col="note", label="メモ", opts=None)]
    cases = L.sort_values("review_id")[["review_id", "speed", "auto_context", "quality_notes", "yaw_rate_lp_dps", "heading_change_deg",
                                        "roll_base_deg", "pitch_base_deg"]].round(2).to_dict(orient="records")
    html = HTML.replace("__CASES__", json.dumps(cases, ensure_ascii=False)).replace("__FORM__", json.dumps(form, ensure_ascii=False)) \
               .replace("__N__", str(len(cases))).replace("__RUNID__", a.run_id)
    (out / "index.html").write_text(html)
    (out / "media_log.json").write_text(json.dumps(dict(log=log, total_mb=used / 1e6, review=str(a.review), form=form,
                                                        span_s=rc["media_span_s"], frame_step=rc["media_frame_step"], only=a.only),
                                                   ensure_ascii=False, indent=2))
    print(f"取得 {used / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
