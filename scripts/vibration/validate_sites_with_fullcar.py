"""映像で路面特徴が見えた振動地点を、既存の 7 自由度 Full-car モデルで説明できるかを確かめる。

路面の高さは当てはめない。代表の路面形状 (凹み・突起・継ぎ目・段差・正弦波パルス・荒れ) を
「右だけ」「左だけ」「左右同時」に与え、実測と次を比べる。

  符号と順序   ロールとピッチの最初の向き、上下のピーク時刻
  時間差       前輪 → 後輪 (輪速。モデルはタイヤ荷重の変動を輪速の代わりに使う) と L/v
  比           ロール/上下、ピッチ/上下 (高さによらない)
  周波数       3〜8 Hz の波形相関、主周波数、8〜15 Hz の継続時間
  左右         輪速の揺れの右/左

モデルは線形なので、比・符号・時刻・周波数は高さによらない。高さ (1〜3 cm) は振幅の幅を
示すためだけに振り、タイヤが浮く高さを記録する。8〜15 Hz の振幅は現行モデルが実車と合わない
ことが分かっているので、判定には使わない (主周波数と時刻だけ使う)。

    uv run python scripts/vibration/validate_sites_with_fullcar.py raw_data/Chunk_1 raw_data/Chunk_2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import pandas as pd
import yaml

import compare_fullcar as CF
import final_qualitative as FQ
import find_roll_bumps as F
import fullcar as FC
import site19_spectra as SP
from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
DT = 0.001
LEAD_S = 1.5
UNIT_H = 0.01          # 線形モデルはこの高さで 1 回だけ解き、他の高さは倍率で得る
WHEELS = FC.CORNERS
SIDE_LABEL = {"R": "右だけ", "L": "左だけ", "B": "左右同時",
              "BsR": "左右・右が遅れる斜め", "BsL": "左右・左が遅れる斜め",
              "BaR": "左右・右が深い", "BaL": "左右・左が深い"}
JOINT_VARIANTS = ("BsR", "BsL", "BaR", "BaL")


# ---------------------------------------------------------------------------
# 映像判定の読み取り
# ---------------------------------------------------------------------------

def parse_video(label: str, vmap: dict) -> dict:
    """映像判定の文言から、路面特徴の種類と、踏んでいるように見える側を読む。"""
    label = str(label or "")
    status = label.split(":")[0].strip() if label else "未確認"
    feature, side = "unknown", "?"
    for key, m in vmap.items():   # 設定の順に照合する (「亀裂の多い」を「亀裂」より先に)
        if key in label:
            feature = m["feature"]
            side = m["side"]
            break
    if "右輪" in label:
        side = "R"
    elif "左輪" in label:
        side = "L"
    elif "両輪" in label:
        side = "B"
    conf = {"R": "右輪らしい (中: 通り道は平坦・直進の仮定で描いた推定)",
            "L": "左輪らしい (中: 同上)",
            "B": "左右同時らしい (中: 横方向に延びる特徴)"}.get(side, "—")
    return dict(video_status=status, road_feature=feature, video_side=side if status == "一致" else "?",
                video_side_confidence=conf if status == "一致" else "—")


# ---------------------------------------------------------------------------
# 実測
# ---------------------------------------------------------------------------

class MeasuredLoader:
    """ドライブごとに取付の傾きを求め、イベントの前後をセグメントの境目をまたいで切り出す。"""

    def __init__(self, roots: list[Path], cfg: dict):
        self.cfg = cfg
        self.refs: dict[str, list] = {}
        for root in roots:
            for r in find_segments(root):
                self.refs.setdefault(r.drive_id, []).append(r)
        self.rot: dict[str, np.ndarray] = {}

    def load(self, drive: str, segment: int, t_hit: float, span: tuple[float, float]) -> dict:
        rs = sorted(self.refs[drive], key=lambda r: r.index)
        if drive not in self.rot:
            self.rot[drive] = F.drive_rotation([r.path for r in rs])
        rot = self.rot[drive]
        by_idx = {r.index: r for r in rs}
        base = F.load_seg(by_idx[segment], rot, self.cfg)
        parts = [base]
        if t_hit + span[0] - 1.0 < base.t[0] + F.EDGE_TRIM_S and segment - 1 in by_idx:
            parts.insert(0, F.load_seg(by_idx[segment - 1], rot, self.cfg))
        if t_hit + span[1] + 1.0 > base.t[-1] - F.EDGE_TRIM_S and segment + 1 in by_idx:
            parts.append(F.load_seg(by_idx[segment + 1], rot, self.cfg))
        seg = F.join_segs(parts, base, self.cfg)
        w = (seg.t >= t_hit + span[0] - 1.0) & (seg.t <= t_hit + span[1] + 1.0)
        out = {k: seg.bp[k][w] for k in ("heave", "roll", "pitch", *WHEELS)}
        out["t"] = seg.t[w] - t_hit
        out["v"] = float(np.mean(seg.v[w]))
        return out


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def shape_fn(spec: dict, h: float, s0: float):
    L, sign = spec["length_m"], spec["sign"]
    if spec["kind"] == "cosine":
        return lambda s: FC.cosine_bump(s, s0, L, sign * h)
    if spec["kind"] == "step":
        return lambda s: FC.ramp_step(s, s0, L, sign * h)
    if spec["kind"] == "sine":
        def f(s):
            u = (s - s0) / L
            out = np.zeros_like(s)
            m = (u >= 0) & (u <= 1)
            out[m] = sign * h * np.sin(2 * np.pi * u[m])
            return out
        return f
    raise ValueError(spec["kind"])


def run_model(p, v: float, left, right, dur: float, cfg: dict, wb: dict, angle_deg: float = 0.0) -> dict:
    """路面入力から、実測と同じ処理 (100 Hz・同じ帯域通過) をした応答を返す。

    輪速はモデルに無いので、各輪のタイヤ荷重の変動 [N] を輪速の帯域で通したものを代わりに使う。
    """
    fs = cfg["grid_hz"]
    t = np.arange(0, dur, DT)
    zr = FC.wheel_inputs(FC.RoadCase("X", "", left, right, joint_angle_deg=angle_deg), p, v, t)
    r = FC.simulate(p, t, zr, contact="linear")
    grid = np.arange(0, dur, 1 / fs)
    out = CF.processed(r, grid, cfg["band_hz"], fs)
    dev = r["tire_load"] - p.static_load
    for i, w in enumerate(WHEELS):
        out[w] = CF.bp(FC.downsample(t, dev[:, i], grid), wb[0], wb[1], fs)
    out["t"] = grid
    out["tire_dev_min"] = dev.min(axis=0)      # 各輪の荷重の最小の変化 (負で抜ける向き)
    return out


def detect_hit(heave: np.ndarray, t: np.ndarray, region: tuple[float, float], search: tuple[float, float], frac: float) -> float:
    """locate_candidates.hit_index と同じ定義: 上下の絶対値が最大の点から search の範囲で、最大値の frac を最初に超えた時刻。"""
    m = (t >= region[0]) & (t <= region[1])
    idx = np.flatnonzero(m)
    x = np.abs(heave)
    k = idx[int(np.argmax(x[idx]))]
    th = frac * x[k]
    w = np.flatnonzero((t >= t[k] + search[0]) & (t <= t[k] + search[1]) & (x >= th))
    return float(t[w[0]]) if w.size else float(t[k])


def align(model: dict, t_rel: np.ndarray, v: float, wheelbase: float, js: dict) -> dict:
    """モデルの当たり時刻 (実測と同じ検出) を 0 にそろえ、実測の時間軸に載せる。"""
    contact = LEAD_S
    t_hit = detect_hit(model["heave"], model["t"], (contact - 0.2, contact + wheelbase / v + 0.8),
                       tuple(js["hit_search"]), js["hit_frac"])
    out = {k: np.interp(t_rel, model["t"] - t_hit, model[k], left=0.0, right=0.0)
           for k in ("heave", "roll", "pitch", *WHEELS)}
    out["contact_offset_s"] = contact - t_hit   # 前輪が凹凸に差しかかった時刻 − 検出した当たり時刻
    out["tire_dev_min"] = model["tire_dev_min"]
    return out


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------

def _rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan


def metrics(sig: dict, t: np.ndarray, v: float, wheelbase: float, cfg: dict, pv: dict) -> dict:
    """時間領域・周波数領域の指標。sig は当たり時刻 0 の時間軸 t の上にある。"""
    fs = cfg["grid_hz"]
    ws = pv["window_s"]
    lag = wheelbase / v
    w = (t >= -ws["metric_pre"]) & (t <= lag + ws["metric_post"])
    road, hop, wb = pv["bands_hz"]["road"], pv["bands_hz"]["wheelhop"], pv["bands_hz"]["wheel"]
    hv = sig["heave"]
    out = dict(
        heave_peak=float(np.max(np.abs(hv[w]))),
        heave_peak_t=float(t[w][np.argmax(np.abs(hv[w]))]),
        heave_rms=_rms(hv[w]), roll_rms=_rms(sig["roll"][w]), pitch_rms=_rms(sig["pitch"][w]),
        roll_angle_pp=float(np.ptp(np.cumsum(sig["roll"][w]) / fs)),
        pitch_angle_pp=float(np.ptp(np.cumsum(sig["pitch"][w]) / fs)),
        heave_road_rms=_rms(CF.bp(hv, *road, fs)[w]), heave_hop_rms=_rms(CF.bp(hv, *hop, fs)[w]),
        roll_road_rms=_rms(CF.bp(sig["roll"], *road, fs)[w]),
        dom_hz=SP.peak_hz(hv[w], fs, 6.0, 20.0),
        hop_duration_s=float(FQ.env_duration(hv[(t >= -0.5) & (t <= lag + 1.5)], fs)),
    )
    out["roll_over_heave"] = out["roll_rms"] / max(out["heave_rms"], 1e-9)
    out["pitch_over_heave"] = out["pitch_rms"] / max(out["heave_rms"], 1e-9)
    # 判定には 3〜8 Hz の比と向きを使う (8〜15 Hz は現行モデルの振幅・減衰が実車と合わないため)
    rd = {k: CF.bp(sig[k], *road, fs) for k in ("heave", "roll", "pitch")}
    out["pitch_road_rms"] = _rms(rd["pitch"][w])
    # 0.5〜3 Hz (車体の上下・ピッチの固有振動 1.3〜1.4 Hz を含む) の上下。3〜8 Hz に対する比で、短い凹凸では出ない長い波長の入力を見る
    out["heave_low_rms"] = _rms(CF.bp(sig["heave"], 0.5, 3.0, fs)[w])
    out["low_over_road"] = out["heave_low_rms"] / max(out["heave_road_rms"], 1e-9)
    # 車体の固有振動の周波数 (モデルは上下・ピッチ 1.3〜1.4 Hz、ロール 1.7 Hz)。分解能のため長めの窓で見る
    wl_ = (t >= -0.5) & (t <= lag + 2.0)
    out["roll_body_hz"] = SP.peak_hz(sig["roll"][wl_], fs, 0.7, 6.0)
    out["heave_body_hz"] = SP.peak_hz(sig["heave"][wl_], fs, 0.7, 4.0)
    out["roll_over_heave_road"] = out["roll_road_rms"] / max(out["heave_road_rms"], 1e-9)
    out["pitch_over_heave_road"] = out["pitch_road_rms"] / max(out["heave_road_rms"], 1e-9)
    out["roll_first_sign_road"], _ = FQ.first_sign(rd["roll"], t, 0.0)
    out["pitch_first_sign_road"], _ = FQ.first_sign(rd["pitch"], t, 0.0)
    out["heave_first_sign_road"], _ = FQ.first_sign(rd["heave"], t, 0.0)
    # 側の指標: ロール角速度と上下速度 (上下加速度の積分) の 3〜8 Hz の相関。
    # モデルでは右だけの入力で −0.73〜−0.95、左だけで同じ大きさの正、左右同時で 0 になり、
    # 凹み/突起・形状・車速 (30〜110 km/h)・高さによらない (docs/vibration_physics_validation.md 2 章)
    hv = CF.bp(np.cumsum(sig["heave"]) / fs, *road, fs)
    out["side_index"] = float(np.corrcoef(hv[w], rd["roll"][w])[0, 1]) if np.std(rd["roll"][w]) > 1e-12 else 0.0
    out["roll_first_sign"], out["roll_first_t"] = FQ.first_sign(sig["roll"], t, 0.0)
    out["pitch_first_sign"], out["pitch_first_t"] = FQ.first_sign(sig["pitch"], t, 0.0)
    out["heave_first_sign"], _ = FQ.first_sign(hv, t, 0.0)
    ww = (t >= -0.3) & (t <= lag + 0.7)
    r = {k: _rms(sig[k][ww]) for k in WHEELS}
    out.update({f"wheel_{k}": r[k] for k in WHEELS})
    out["wheel_left"], out["wheel_right"] = r["FL"] + r["RL"], r["FR"] + r["RR"]
    out["wheel_lr_log2"] = float(np.log2(max(out["wheel_left"], 1e-12) / max(out["wheel_right"], 1e-12)))
    side = "R" if out["wheel_right"] >= out["wheel_left"] else "L"
    env = {k: np.convolve(np.abs(sig[k][ww]), np.ones(10) / 10, "same") for k in ("F" + side, "R" + side)}
    out["front_rear_lag_s"] = float((np.argmax(env["R" + side]) - np.argmax(env["F" + side])) / fs)
    out["lag_expected_s"] = lag
    return out


def band_corr(meas: np.ndarray, pred: np.ndarray, t: np.ndarray, band, fs: float, w: np.ndarray, max_lag_s: float) -> tuple[float, float]:
    """帯域通過した波形の相関。時間ずれ 0 と、±max_lag_s で最大の値。"""
    a, b = CF.bp(meas, *band, fs), CF.bp(pred, *band, fs)
    idx = np.flatnonzero(w)
    if np.std(b[idx]) < 1e-12:
        return np.nan, np.nan
    L = int(max_lag_s * fs)
    cc = {l: float(np.corrcoef(a[idx + l], b[idx])[0, 1]) for l in range(-L, L + 1)
          if idx[0] + l >= 0 and idx[-1] + l < len(a)}
    return cc.get(0, np.nan), max(cc.values())


# ---------------------------------------------------------------------------
# 1 イベントの評価
# ---------------------------------------------------------------------------

def evaluate_event(e: pd.Series, loader: MeasuredLoader, p, cfg: dict, pv: dict) -> tuple[dict, dict]:
    fs = cfg["grid_hz"]
    ws = pv["window_s"]
    span = (ws["plot"][0] - 0.5, ws["plot"][1] + 0.5)
    m = loader.load(e.drive, int(e.segment), float(e.event_time_s), span)
    v = max(m["v"], 3.0)
    L = p.wheelbase
    # 当たり時刻は、モデルと同じ検出をかけ直して 0 にそろえる
    t0 = detect_hit(m["heave"], m["t"], (-0.5, 0.6), tuple(ws["hit_search"]), ws["hit_frac"])
    t = m["t"] - t0
    keep = (t >= span[0]) & (t <= span[1])
    t = t[keep]
    meas = {k: m[k][keep] for k in ("heave", "roll", "pitch", *WHEELS)}
    mm = metrics(meas, t, v, L, cfg, pv)

    dur = LEAD_S + L / v + 3.5
    s0 = v * LEAD_S
    zero = lambda s: np.zeros_like(s)
    w = (t >= -ws["metric_pre"]) & (t <= L / v + ws["metric_post"])
    road = pv["bands_hz"]["road"]
    hyps, preds = [], {}
    for shape, spec in pv["shapes"].items():
        f = shape_fn(spec, UNIT_H, s0)
        jv = pv["joint_variants"]
        for side in (*pv["sides"], *JOINT_VARIANTS):
            # 継ぎ目の変形: 斜め (左右の当たる時刻がずれる) と、左右で深さが違うもの。比は固定で当てはめない
            angle = {"BsR": jv["skew_deg"], "BsL": -jv["skew_deg"]}.get(side, 0.0)
            kl = jv["asym_ratio"] if side == "BaR" else 1.0
            kr = jv["asym_ratio"] if side == "BaL" else 1.0
            left = (lambda s_, k=kl: k * f(s_)) if side in ("L", "B", *JOINT_VARIANTS) else zero
            right = (lambda s_, k=kr: k * f(s_)) if side in ("R", "B", *JOINT_VARIANTS) else zero
            sim = align(run_model(p, v, left, right, dur, cfg, pv["bands_hz"]["wheel"], angle), t, v, L, ws)
            hm = metrics(sim, t, v, L, cfg, pv)
            row = dict(shape=shape, side=side, **{f"model_{k}": val for k, val in hm.items()},
                       contact_offset_s=sim["contact_offset_s"])
            for k in ("heave", "roll", "pitch"):
                c0, cb = band_corr(meas[k], sim[k], t, road, fs, w, ws["lag_search"])
                row[f"{k}_corr0"], row[f"{k}_corr"] = c0, cb
            # タイヤが浮く高さ: 静的荷重 + (h / UNIT_H) × 最小の変化 < 0
            lift = [h for h in pv["heights_m"] if np.any(p.static_load + h / UNIT_H * sim["tire_dev_min"] < 0)]
            row["lift_off_heights_cm"] = ";".join(f"{100 * h:g}" for h in lift)
            hyps.append(row)
            preds[(shape, side)] = sim
    # 荒れた区間 (左右独立)
    rc = pv["rough"]
    Lr = float(np.clip(mm["hop_duration_s"] * v, *rc["length_m"]))
    gd = FC.ISO8608_GD0[rc["iso_class"]]
    rough_rows = []
    for seed in range(rc["seeds"]):
        left = lambda s, sd=seed: FC.rough_patch(s, s0, Lr, gd, sd)
        right = lambda s, sd=seed: FC.rough_patch(s, s0, Lr, gd, sd + 1000)
        sim = align(run_model(p, v, left, right, LEAD_S + (Lr + L) / v + 3.0, cfg, pv["bands_hz"]["wheel"]), t, v, L, ws)
        rough_rows.append(metrics(sim, t, v, L, cfg, pv))
        if seed == 0:
            preds[("rough", "B")] = sim
    rough = pd.DataFrame(rough_rows).median(numeric_only=True).to_dict()
    rough["rough_length_m"] = Lr
    return dict(meas=mm, hyps=pd.DataFrame(hyps), rough=rough, v=v, t=t, meas_sig=meas), preds


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------

def tri(x) -> str:
    return {True: "true", False: "false"}.get(x, "uncertain")


def best_shape(h: pd.DataFrame, side: str) -> pd.Series:
    """側を決めたうえで、上下の 3〜8 Hz 波形相関 (振幅によらない) が最大の形状。上下は左右で同じなので、形は側によらず決まる。"""
    g = h[h.side == side]
    return g.loc[g.heave_corr.idxmax()]


def judge_event(ev: dict, video: dict, candidate: pd.Series, pv: dict) -> dict:
    j = pv["judge"]
    mm, h = ev["meas"], ev["hyps"]
    both = best_shape(h, "B")
    # 片側入力のロール/上下 (3〜8 Hz) の基準: 形状ごとの中央値。左右でこの比は同じ
    one_ref = float(h[h.side == "R"].model_roll_over_heave_road.median())
    frac = mm["roll_over_heave_road"] / max(one_ref, 1e-9)
    # 左右の強さは、車輪ごとに普段の大きさで割った値 (locate_candidates.py と同じ) で決める。
    # 生の RMS は車輪ごとの普段の大きさの違いを含むので参考にとどめる
    wl = float(candidate.wheel_lr_log2)           # 正 = 左が強い
    best_any = h.heave_corr.max()
    lag_err = abs(mm["front_rear_lag_s"] - mm["lag_expected_s"])

    # 物理モデルから見た入力の種類 (映像を使わない)
    if bool(candidate.wheelhop_only) and best_any < j["corr_true"]:
        # 3〜8 Hz が普段並みで、3〜8 Hz の波形も路面入力のモデルと合わない
        inferred = "11 Hz 固有応答だけで説明される"
    elif best_any < j["corr_unc"]:
        inferred = "現行モデルでは説明困難"
    else:
        si = mm["side_index"]
        by_roll = "R" if si <= -j["side_index_min"] else ("L" if si >= j["side_index_min"] else None)
        by_wheel = "R" if wl <= -j["wheel_side_log2"] else ("L" if wl >= j["wheel_side_log2"] else None)
        sides = {x for x in (by_roll, by_wheel) if x}
        if len(sides) == 2:
            inferred = "ロールと輪速で側が食い違う"
        elif frac >= j["one_sided_roll_frac"] and sides:
            inferred = {"R": "右片輪入力らしい", "L": "左片輪入力らしい"}[sides.pop()]
        elif frac < j["both_roll_frac"] and not sides:
            inferred = "左右同時入力らしい"
        else:
            inferred = "片側と左右同時の中間 (混在または判別不能)"

    vs, feat = video["video_side"], video["road_feature"]
    out = dict(inferred_input=inferred, roll_frac_of_one_sided=frac, one_sided_roll_ref=one_ref,
               wheel_lr_log2_normalized=wl, best_heave_corr_any=best_any)
    if vs in ("R", "L"):
        # 輪速: 映像の側が普段比で 2 倍以上強ければ +1、逆側が強ければ −1
        sgn = -1 if vs == "R" else 1
        w_score = 1 if sgn * wl >= j["wheel_side_log2"] else (-1 if -sgn * wl >= j["wheel_side_log2"] else 0)
        # ロール: 側の指標 (ロール角速度と上下速度の相関)。右だけの入力なら負、左だけなら正
        si = mm["side_index"]
        r_score = 1 if sgn * si >= j["side_index_min"] else (-1 if -sgn * si >= j["side_index_min"] else 0)
        sh = best_shape(h, vs)
        sc = w_score + r_score
        out["P1"] = tri(True if sc >= 2 else (False if sc <= -1 else None))
        out["P1_wheel_score"], out["P1_roll_score"] = w_score, r_score
        model_ref = sh
        type_ok = frac >= j["one_sided_roll_frac"]
        type_bad = frac < j["both_roll_frac"]
    elif vs == "B":
        model_ref = both
        if feat == "rough":
            model_ref = pd.Series({f"model_{k}": v for k, v in ev["rough"].items()})
        # 左右同時: 輪速の左右差が小さく、側の指標も 0 に近いこと
        a, si = abs(wl), abs(mm["side_index"])
        n_ok = int(a < j["wheel_side_log2"]) + int(si < j["side_index_min"])
        n_bad = int(a >= 1.5 * j["wheel_side_log2"]) + int(si >= 2 * j["side_index_min"])
        out["P1"] = tri(True if n_ok == 2 else (False if n_bad == 2 else None))
        if feat == "rough":
            # 左右独立の荒れはロールも出る。モデルの荒れのロール/上下と比べる
            r = mm["roll_over_heave_road"] / max(ev["rough"]["roll_over_heave_road"], 1e-9)
            type_ok, type_bad = 0.5 <= r <= 2.0, r > 3.0 or r < 1 / 3
        else:
            type_ok = frac < j["both_roll_frac"]
            # 斜めの継ぎ目・左右で深さが違う継ぎ目 (比は固定) のロール/上下と比べる
            var = h[h.side.isin(JOINT_VARIANTS) & (h["shape"] == both["shape"])]
            q = mm["roll_over_heave_road"] / var.model_roll_over_heave_road.clip(lower=1e-9)
            ok_var = var.side[(q >= 1 / 1.5) & (q <= 1.5)].tolist()
            out["joint_variants_explaining"] = ";".join(SIDE_LABEL[x] for x in ok_var)
            type_bad = (not type_ok) and not ok_var
    else:
        model_ref = best_shape(h, {"右片輪入力らしい": "R", "左片輪入力らしい": "L"}.get(inferred, "B"))
        out["P1"] = "—"
        type_ok = type_bad = None
    out["model_side_used"] = vs if vs in ("R", "L", "B") else "physics"
    out["model_shape_used"] = "rough" if (vs == "B" and feat == "rough") else model_ref.get("shape", "rough")

    out["P2"] = tri(True if lag_err <= j["lag_tol_s"] else (None if lag_err <= j["lag_unc_s"] else False))
    pr = mm["pitch_over_heave_road"] / max(model_ref["model_pitch_over_heave_road"], 1e-9)
    if type_ok is None:
        out["P3"] = "—"
    else:
        pitch_ok = 0.5 <= pr <= 2.0
        out["P3"] = tri(True if (type_ok and pitch_ok) else (False if type_bad else None))
    out["pitch_ratio_meas_over_model"] = pr

    if out["model_shape_used"] == "rough":
        q = (mm["heave_road_rms"] / max(mm["heave_hop_rms"], 1e-9)) / max(
            ev["rough"]["heave_road_rms"] / max(ev["rough"]["heave_hop_rms"], 1e-9), 1e-9)
        out["P4"] = tri(True if 1 / 1.5 <= q <= 1.5 else (None if 1 / 3 <= q <= 3 else False))
        out["band_ratio_meas_over_model"] = q
    else:
        aux = max(model_ref.get("roll_corr", np.nan) if out["model_side_used"] != "B" else -1,
                  model_ref.get("pitch_corr", np.nan))
        hc = model_ref.get("heave_corr", np.nan)
        out["P4"] = tri(True if (hc >= j["corr_true"] and aux >= j["aux_corr_true"])
                        else (False if hc < j["corr_unc"] else None))
        out["heave_corr_3_8"], out["aux_corr_3_8"] = hc, aux
    lo, hi = pv["bands_hz"]["wheelhop_peak"]
    meas_in = lo <= mm["dom_hz"] <= hi
    model_in = lo <= model_ref["model_dom_hz"] <= hi
    out["P5"] = tri(True if (meas_in and model_in) else (False if (meas_in and not model_in) else None))
    out["_model_ref"] = model_ref
    return out


def agg(values: list[str]) -> str:
    v = [x for x in values if x in ("true", "false", "uncertain")]
    if not v:
        return "—"
    s = np.mean([{"true": 1.0, "uncertain": 0.5, "false": 0.0}[x] for x in v])
    return "true" if s >= 2 / 3 else ("false" if s <= 1 / 3 else "uncertain")


def overall(P: dict) -> str:
    vals = [P[f"P{i}"] for i in range(1, 7)]
    n_true = sum(x == "true" for x in vals)
    n_false = sum(x == "false" for x in vals[:4])
    if P["P1"] == "true" and P["P3"] == "true" and n_true >= 4 and n_false == 0:
        return "Strong physical consistency"
    if (P["P1"] == "false" and P["P3"] == "false") or n_true <= 1:
        return "Not explained"
    return "Partial consistency"


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def plot_site(site: pd.Series, ev: dict, preds: dict, judge: dict, pv: dict, cfg: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fs = cfg["grid_hz"]
    t, meas = ev["t"], ev["meas_sig"]
    v = ev["v"]
    lag = ev["meas"]["lag_expected_s"]
    pw = (t >= pv["window_s"]["plot"][0]) & (t <= pv["window_s"]["plot"][1])
    side = judge["model_side_used"] if judge["model_side_used"] in ("R", "L", "B") else "R"
    shape = judge["model_shape_used"]
    main = preds[(shape, side if shape != "rough" else "B")]
    alt_side = {"R": "L", "L": "R", "B": "R"}[side]
    alt_shape = shape if shape != "rough" else best_shape(ev["hyps"], "R")["shape"]
    alt = preds[(alt_shape, alt_side)]
    hs = np.array(pv["heights_m"]) / UNIT_H
    ref = pv["ref_height_m"] / UNIT_H
    # 形の比較用: 上下の 3〜8 Hz の RMS が実測と同じになる共通の倍率を、上下・ロール・ピッチに同じだけ掛ける。
    # 比 (ロール/上下など) は変わらない。路面の高さの推定ではない
    wm = (t >= -pv["window_s"]["metric_pre"]) & (t <= lag + pv["window_s"]["metric_post"])
    road = pv["bands_hz"]["road"]
    k_shape = _rms(CF.bp(meas["heave"], *road, fs)[wm]) / max(_rms(CF.bp(main["heave"], *road, fs)[wm]), 1e-12)
    col = {"R": "tab:red", "L": "tab:blue", "B": "tab:purple"}
    fig, axs = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    rows = (("heave", "上下加速度 [m/s²]\n(0.7〜20 Hz)"), ("roll", "ロール角速度 [deg/s]\n(正 = 右下がり)"),
            ("pitch", "ピッチ角速度 [deg/s]\n(正 = 機首上げ)"))
    for ax, (k, yl) in zip(axs, rows):
        ax.plot(t[pw], meas[k][pw], color="k", lw=1.1, label="実測")
        y = main[k][pw]
        ax.fill_between(t[pw], hs.min() * y, hs.max() * y, color=col[side], alpha=0.15,
                        label=f"モデル {SIDE_LABEL.get(side, side)}・{shape} ({100 * pv['heights_m'][0]:g}〜{100 * pv['heights_m'][-1]:g} cm)")
        ax.plot(t[pw], ref * y, color=col[side], lw=1.0, label=f"同 {100 * pv['ref_height_m']:g} cm")
        ax.plot(t[pw], k_shape * y, color=col[side], lw=1.0, ls=":",
                label=f"同 形の比較用 (上下の 3〜8 Hz を実測に合わせた共通倍率 ×{k_shape / ref:.1f}、高さの推定ではない)")
        ax.plot(t[pw], ref * alt[k][pw], color=col[alt_side], lw=0.8, ls="--",
                label=f"対照 {SIDE_LABEL[alt_side]}・{alt_shape} ({100 * pv['ref_height_m']:g} cm)")
        ax.set_ylabel(yl)
    for k, c, ls in (("FL", "tab:blue", "--"), ("RL", "tab:blue", "-"), ("FR", "tab:red", "--"), ("RR", "tab:red", "-")):
        axs[3].plot(t[pw], meas[k][pw] * 3.6, color=c, ls=ls, lw=0.9, label=f"実測 {k}")
    tw = axs[3].twinx()
    for k, c, ls in (("FL", "tab:cyan", "--"), ("RL", "tab:cyan", "-"), ("FR", "tab:orange", "--"), ("RR", "tab:orange", "-")):
        tw.plot(t[pw], ref * main[k][pw] / 1000, color=c, ls=ls, lw=0.8, alpha=0.8, label=f"モデル {k}")
    axs[3].set_ylabel("実測 輪速の揺れ [km/h]\n(2〜15 Hz)")
    tw.set_ylabel("モデル タイヤ荷重の変動 [kN]\n(輪速の代わり、2〜15 Hz)")
    axs[3].legend(fontsize=7, ncol=4, loc="upper left")
    tw.legend(fontsize=7, ncol=4, loc="lower left")
    for ax in axs:
        ax.axvline(0, color="0.4", lw=0.8)
        ax.axvline(lag, color="0.4", lw=0.8, ls="--")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, loc="upper right")
    axs[-1].set_xlabel(f"前輪が当たった時刻からの時間 [s]  (破線 = 後輪 L/v = {lag:.2f} s)")
    fig.suptitle(f"地点 {site.site_id}  {site.event_id}  {v * 3.6:.0f} km/h  映像: {site.video_label}\n"
                 f"P1–P6 = {' '.join(judge[f'P{i}'] for i in range(1, 7))}  → {judge['physical_consistency']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    plt.close(fig)


def plot_spectrum(site: pd.Series, ev: dict, preds: dict, judge: dict, pv: dict, cfg: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fs = cfg["grid_hz"]
    t, meas = ev["t"], ev["meas_sig"]
    lag = ev["meas"]["lag_expected_s"]
    w = (t >= -0.5) & (t <= lag + 1.5)
    side = judge["model_side_used"] if judge["model_side_used"] in ("R", "L", "B") else "R"
    shape = judge["model_shape_used"]
    main = preds[(shape, side if shape != "rough" else "B")]
    ref = pv["ref_height_m"] / UNIT_H
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, k, yl in ((axs[0], "heave", "上下加速度の振幅スペクトル"), (axs[1], "roll", "ロール角速度の振幅スペクトル")):
        for x, lab, c in ((meas[k][w], "実測", "k"), (ref * main[k][w], f"モデル {SIDE_LABEL.get(side, side)}・{shape} {100 * pv['ref_height_m']:g} cm", "tab:red")):
            n = 1 << int(np.ceil(np.log2(len(x) * 4)))
            f = np.fft.rfftfreq(n, 1 / fs)
            A = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), n))
            m_ = (f >= 0.5) & (f <= 20)
            ax.plot(f, A / A[m_].max(), color=c, label=lab)
        for b, c in ((pv["bands_hz"]["road"], "tab:green"), (pv["bands_hz"]["wheelhop"], "tab:gray")):
            ax.axvspan(*b, color=c, alpha=0.1)
        ax.set_xlim(0.5, 20)
        ax.set_xlabel("周波数 [Hz]  (緑 = 3〜8 Hz、灰 = 8〜15 Hz)")
        ax.set_ylabel(yl + "\n(0.5〜20 Hz の最大値で正規化)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"地点 {site.site_id}  {site.event_id}  主周波数 実測 {ev['meas']['dom_hz']:.1f} Hz / "
                 f"モデル {judge['_model_ref']['model_dom_hz']:.1f} Hz", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    plt.close(fig)


def side_index_all(cand: pd.DataFrame, loader: MeasuredLoader, cfg: dict, pv: dict, wheelbase: float) -> pd.DataFrame:
    """全イベントで、IMU だけから求めた側の指標と、輪速 (普段比) の左右を並べる。モデルは使わない。"""
    ws = pv["window_s"]
    rows = []
    for _, e in cand.iterrows():
        try:
            m = loader.load(e.drive, int(e.segment), float(e.event_time_s), (-1.0, 2.0))
        except (KeyError, ValueError, IndexError):
            continue
        v = max(m["v"], 3.0)
        t0 = detect_hit(m["heave"], m["t"], (-0.5, 0.6), tuple(ws["hit_search"]), ws["hit_frac"])
        mm = metrics({k: m[k] for k in ("heave", "roll", "pitch", *WHEELS)}, m["t"] - t0, v, wheelbase, cfg, pv)
        rows.append(dict(event_id=e.event_id, site_id=e.site_id, speed_kmh=round(v * 3.6, 1),
                         wheel_lr_log2_normalized=e.wheel_lr_log2, wheel_side=e.wheel_side,
                         side_index=round(mm["side_index"], 3), roll_over_heave_3_8=round(mm["roll_over_heave_road"], 3)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_physics_validation.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/physics_validation")
    ap.add_argument("--skip-all-events", action="store_true", help="全イベントの側の指標を計算しない")
    ap.add_argument("--sites", type=str, default="",
                    help="評価する地点の番号 (カンマ区切り)。省略時は位置特定の上位 20 地点 (top20.csv)")
    args = ap.parse_args()
    pv = yaml.safe_load(args.config.read_text())
    cfg = yaml.safe_load((REPO / pv["base_config"]).read_text())
    p = FC.build_params(FC.load_config(REPO / pv["fullcar_config"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    matplotlib.use("Agg")
    configure_japanese_font()

    ld = REPO / pv["locate_dir"]
    if args.sites:
        # 任意の地点: sites.csv から取り、映像の目視は位置で対応付ける (locate_candidates.attach_review)
        import locate_candidates as LC

        loc = yaml.safe_load((REPO / "configs/vibration_locate.yaml").read_text())
        ids = [int(x) for x in args.sites.split(",") if x.strip()]
        allsites = pd.read_csv(ld / "sites.csv")
        top = LC.attach_review(allsites[allsites.site_id.isin(ids)], REPO / loc["output"]["review_csv"],
                               LC.LocalFrame(*loc["position"]["ref_latlon"]), loc)
        top = top.set_index("site_id").loc[ids].reset_index().fillna({"d_video": ""})
    else:
        top = pd.read_csv(ld / "top20.csv").fillna({"d_video": ""})
    cand = pd.read_csv(ld / "candidates.csv")
    loader = MeasuredLoader(args.roots, cfg)

    ev_rows, site_rows = [], []
    for s in top.itertuples():
        video = parse_video(s.d_video, pv["video_map"])
        evs = cand[cand.site_id == s.site_id]
        judged = []
        for _, e in evs.iterrows():
            ev, preds = evaluate_event(e, loader, p, cfg, pv)
            jd = judge_event(ev, video, e, pv)
            judged.append((e, ev, preds, jd))
            mm = ev["meas"]
            ref = jd["_model_ref"]
            ev_rows.append(dict(
                site_id=s.site_id, event_id=e.event_id, drive=e.drive, segment=e.segment, event_time_s=e.event_time_s,
                speed_kmh=round(ev["v"] * 3.6, 1), **video,
                **{f"meas_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in mm.items()},
                model_side_index=round(float(ref.get("model_side_index", np.nan)), 3),
                **{f"model_{k}": (round(float(ref.get(f"model_{k}")), 4) if pd.notna(ref.get(f"model_{k}", np.nan)) else np.nan)
                   for k in ("roll_over_heave", "pitch_over_heave", "roll_over_heave_road", "pitch_over_heave_road",
                             "roll_first_sign_road", "pitch_first_sign_road", "heave_road_rms",
                             "heave_hop_rms", "dom_hz", "hop_duration_s", "wheel_left", "wheel_right", "front_rear_lag_s",
                             "low_over_road", "roll_body_hz", "heave_body_hz")},
                lift_off_heights_cm=ref.get("lift_off_heights_cm", ""),
                **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in jd.items() if not k.startswith("_")},
            ))
        # 地点: 代表 (A 最大) と、全イベントの多数決
        # 代表 (図と表の値): 上下の 3〜8 Hz の相関が地点のイベントの中央値にあたるもの。最良も最悪も選ばない
        cs = np.array([x[1]["hyps"].heave_corr.max() for x in judged])
        rep = judged[int(np.argsort(cs)[len(cs) // 2])]
        e, ev, preds, jd = rep
        P = {f"P{i}": agg([x[3][f"P{i}"] for x in judged]) for i in range(1, 6)}
        sides = [np.sign(x[3]["wheel_lr_log2_normalized"]) for x in judged]
        rolls = [np.sign(x[1]["meas"]["side_index"]) for x in judged]     # IMU から見た側 (輪速と独立)
        if len(judged) < 2:
            P["P6"] = "uncertain"
        else:
            same_side = max(sides.count(1.0), sides.count(-1.0)) / len(sides)
            same_roll = max(rolls.count(1.0), rolls.count(-1.0)) / len(rolls)
            P["P6"] = "true" if min(same_side, same_roll) >= pv["judge"]["repeat_frac"] else "false"
        P["physical_consistency"] = overall(P)
        inferred = pd.Series([x[3]["inferred_input"] for x in judged]).value_counts()
        mm = ev["meas"]
        ref = jd["_model_ref"]
        ref_h = pv["ref_height_m"] / UNIT_H
        notes = []
        lo_ratio = mm["low_over_road"] / max(ref["model_low_over_road"], 1e-9)
        if lo_ratio > 2:
            notes.append(f"0.5〜3 Hz の上下が 3〜8 Hz に比べてモデルの {lo_ratio:.1f} 倍 (短い凹凸では出ない長い波長の入力か、車体の応答が大きい)")
        if mm["dom_hz"] and abs(mm["hop_duration_s"] - ref["model_hop_duration_s"]) > 0.2:
            notes.append(f"8〜15 Hz の継続 実測 {mm['hop_duration_s']:.2f} s / モデル {ref['model_hop_duration_s']:.2f} s")
        notes.append(f"前→後輪 実測 {mm['front_rear_lag_s']:.2f} s / L/v {mm['lag_expected_s']:.2f} s")
        notes.append(f"ロール/上下は片側入力モデルの {jd['roll_frac_of_one_sided']:.2f} 倍")
        expl = [x[3].get("joint_variants_explaining", "") for x in judged]
        if any(expl):
            cnt = pd.Series([y for x in expl for y in x.split(";") if y]).value_counts()
            notes.append("ロール/上下を説明できる継ぎ目の変形: " + "、".join(f"{k} {v}/{len(judged)}" for k, v in cnt.items()))
        if ref.get("lift_off_heights_cm"):
            notes.append(f"モデルは {ref['lift_off_heights_cm']} cm でタイヤが浮く (線形の比較には影響しない)")
        site_rows.append(dict(
            site_id=s.site_id, rank=s.rank, video_label=s.d_video, video_status=video["video_status"],
            road_feature=video["road_feature"], video_side=video["video_side"],
            video_side_confidence=video["video_side_confidence"],
            inferred_side=inferred.index[0], inferred_votes="; ".join(f"{k} {v}" for k, v in inferred.items()),
            speed=round(ev["v"] * 3.6, 1), passes=f"{s.n_pass} (イベント {len(judged)}、再現 A {s.repeat_A_k}/{s.repeat_A_n}・上下 {s.repeat_heave_k}/{s.repeat_heave_n})",
            representative_event=e.event_id,
            heave_metric=round(mm["heave_peak"], 2), roll_metric=round(mm["roll_rms"], 2),
            roll_angle_pp_deg=round(mm["roll_angle_pp"], 3),
            pitch_metric=round(mm["pitch_rms"], 2), pitch_angle_pp_deg=round(mm["pitch_angle_pp"], 3),
            roll_over_heave_3_8_meas=round(mm["roll_over_heave_road"], 2),
            roll_over_heave_3_8_model=round(ref["model_roll_over_heave_road"], 2),
            roll_over_heave_3_8_one_sided_ref=round(jd["one_sided_roll_ref"], 2),
            pitch_over_heave_3_8_meas=round(mm["pitch_over_heave_road"], 2),
            pitch_over_heave_3_8_model=round(ref["model_pitch_over_heave_road"], 2),
            roll_first_sign_3_8_meas=mm["roll_first_sign_road"], roll_first_sign_3_8_model=ref.get("model_roll_first_sign_road", np.nan),
            wheel_lr_log2_normalized=round(jd["wheel_lr_log2_normalized"], 2),
            side_index_meas=round(mm["side_index"], 2), side_index_model=round(float(ref.get("model_side_index", np.nan)), 2),
            wheel_left_metric=round(mm["wheel_left"] * 3.6, 3), wheel_right_metric=round(mm["wheel_right"] * 3.6, 3),
            wheel_right_over_left_meas=round(mm["wheel_right"] / max(mm["wheel_left"], 1e-9), 2),
            wheel_right_over_left_model=round(ref["model_wheel_right"] / max(ref["model_wheel_left"], 1e-9), 2),
            front_rear_lag_meas=round(mm["front_rear_lag_s"], 3), front_rear_lag_expected=round(mm["lag_expected_s"], 3),
            band_3_8_measured=round(mm["heave_road_rms"], 3), band_3_8_model=round(ref_h * ref["model_heave_road_rms"], 3),
            band_8_15_measured=round(mm["heave_hop_rms"], 3), band_8_15_model=round(ref_h * ref["model_heave_hop_rms"], 3),
            dominant_freq_measured=round(mm["dom_hz"], 1), dominant_freq_model=round(ref["model_dom_hz"], 1),
            hop_duration_measured=round(mm["hop_duration_s"], 2), hop_duration_model=round(ref["model_hop_duration_s"], 2),
            roll_body_hz_measured=round(mm["roll_body_hz"], 2), roll_body_hz_model=round(ref["model_roll_body_hz"], 2),
            heave_body_hz_measured=round(mm["heave_body_hz"], 2), heave_body_hz_model=round(ref["model_heave_body_hz"], 2),
            low_over_road_measured=round(mm["low_over_road"], 2), low_over_road_model=round(ref["model_low_over_road"], 2),
            apparent_scale_3_8=round(mm["heave_road_rms"] / max(ref["model_heave_road_rms"], 1e-12) * UNIT_H / pv["ref_height_m"], 2),
            model_side_used=jd["model_side_used"], model_shape_used=jd["model_shape_used"],
            **P, notes="。".join(notes),
        ))
        stem = args.out / "figures" / f"site{s.site_id:03d}"
        site_row = pd.Series(dict(site_id=s.site_id, event_id=e.event_id, video_label=s.d_video or "未確認"))
        jd_plot = dict(jd, **P)
        plot_site(site_row, ev, preds, jd_plot, pv, cfg, Path(f"{stem}_compare.png"))
        plot_spectrum(site_row, ev, preds, jd_plot, pv, cfg, Path(f"{stem}_spectrum.png"))
        print(f"地点 {s.site_id:3d} {video['video_status']:4s} {video['road_feature']:7s} 映像 {video['video_side']} "
              f"物理 {inferred.index[0]}  {' '.join(P[f'P{i}'][0] for i in range(1, 7))}  {P['physical_consistency']}")

    sites = pd.DataFrame(site_rows)
    sites.to_csv(args.out / "site_physics_validation.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(args.out / "event_physics_validation.csv", index=False)
    meta = dict(config=str(args.config.relative_to(REPO)),
                config_sha1=hashlib.sha1(args.config.read_bytes()).hexdigest()[:10],
                fullcar_config_sha1=hashlib.sha1((REPO / pv["fullcar_config"]).read_bytes()).hexdigest()[:10],
                sites=len(sites), events=len(ev_rows),
                modes=[dict(freq_hz=round(m["freq_hz"], 2), damping=round(m["damping"], 2), dominant=m["dominant"]) for m in FC.modes(p)])
    if not args.skip_all_events:
        side_index_all(cand, loader, cfg, pv, p.wheelbase).to_csv(args.out / "all_events_side_index.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
