"""NCST の高速道路の走行 (GPS が切れた後の Cajon Pass の下り) を、運転席から見た前方の CG と信号で並べた動画にする。

実際の映像ではない。
  - 道路の中心線: OpenStreetMap の I-15 南行き (ODbL、ncst_route_fetch.py で取得済み)
  - 道路の高さ: USGS 3DEP の標高 (公有) を 300 m でならしたもの (ncst_route_grade.py の elev_s)
  - 道幅: OSM の lanes × 車線の幅 + 路肩 (設定。この区間で測った値ではない)
  - トラックの位置: ncst_route_grade.py の推定 (目印で区切った区間ごとに一定の速さ。2 本目の下りは数 km ずれうる)
地形: USGS 3DEP の 1 秒角 (約 30 m) の DEM (公有、ncst_dem_crop.py で切り出し)。陰影は北西からの光の模式。
描かないもの: 他の車、草木・建物、天気、トラックの車線 (分からないので車道の中央に置く)。

画面: 左上に前方の CG (右上にフェードリスクの札、順位の色の枠。左下に勾配・峠からの高度差・下りの進み具合・ブレーキの回数)、
      左下に標高の断面 (いまの位置と、地図の長い下り坂 2 本) とブレーキ圧・ブレーキ温度のゲージ、
      右に 5 段の時系列 (標高・勾配・ブレーキ圧・ブレーキ温度・フェードリスク。地図の下り坂の時刻を水色で塗る)。

    uv run python scripts/brake/ncst_forward_view.py --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import PolyCollection
from scipy.interpolate import CubicSpline

from near_miss.plotting import configure_japanese_font

from ncst_scene_illustration import TCMAP, TNORM

OUT = Path("out/brake_fade")
CFG = Path("configs/brake/ncst_forward_view.yaml")
RANK_COL = {"High": "#C0392B", "Medium": "#E08A2E", "Low": "#8A9BB0"}
EARTH_R = 6371000.0


def enu(lat, lon, lat0, lon0):
    """緯度経度 → 局所の東・北 [m] (数十 km の範囲なので平面で近似)。"""
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))) % 360


def lanes_along(R: pd.DataFrame, osm: dict, rc: dict) -> tuple[np.ndarray, float]:
    """経路の点ごとの車線数: 近くの OSM の way (同じ向き) の lanes。取れない所は既定値。ならしてから返す。"""
    nodes = []
    for w in osm["elements"]:
        if w.get("type") != "way" or "geometry" not in w:
            continue
        ln = w.get("tags", {}).get("lanes")
        if ln is None or not str(ln).isdigit():
            continue
        g = w["geometry"]
        for a, b in zip(g[:-1], g[1:]):
            nodes.append((a["lat"], a["lon"], bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"]), int(ln)))
    N = np.array(nodes)
    hd = bearing_deg(R.lat.to_numpy()[:-1], R.lon.to_numpy()[:-1], R.lat.to_numpy()[1:], R.lon.to_numpy()[1:])
    hd = np.append(hd, hd[-1])
    lat0, lon0 = float(R.lat.iat[0]), float(R.lon.iat[0])
    nx, ny = enu(N[:, 0], N[:, 1], lat0, lon0)
    rx, ry = enu(R.lat.to_numpy(), R.lon.to_numpy(), lat0, lon0)
    out = np.full(len(R), np.nan)
    for i in range(len(R)):
        d = np.hypot(nx - rx[i], ny - ry[i])
        same = np.abs((N[:, 2] - hd[i] + 180) % 360 - 180) < 45
        d[~same] = np.inf
        j = int(np.argmin(d))
        if d[j] <= rc["lanes_match_max_m"]:
            out[i] = N[j, 3]
    step = float(R.s_m.diff().median())
    s = pd.Series(out).fillna(rc["lanes_default"])
    share = float(np.mean(np.isfinite(out)))   # OSM の lanes が取れた点の割合
    return s.rolling(max(1, int(round(rc["lanes_smooth_m"] / step))) | 1, center=True, min_periods=1).median().round().to_numpy(), share


class Road:
    """経路の 3 次元の中心線 (s → x, y, z) と道幅。"""

    def __init__(self, R: pd.DataFrame, lanes: np.ndarray, rc: dict):
        lat0, lon0 = float(R.lat.iat[0]), float(R.lon.iat[0])
        x, y = enu(R.lat.to_numpy(), R.lon.to_numpy(), lat0, lon0)
        s = R.s_m.to_numpy()
        z = R.elev_s.interpolate(limit_direction="both").to_numpy()
        self.sx, self.sy, self.sz = CubicSpline(s, x), CubicSpline(s, y), CubicSpline(s, z)
        self.s_max = float(s[-1])
        self.s_pts, self.lanes = s, lanes
        self.rc = rc

    def at(self, s):
        s = np.clip(s, 0, self.s_max)
        return self.sx(s), self.sy(s), self.sz(s)

    def tangent(self, s):
        dx, dy = self.sx(s, 1), self.sy(s, 1)
        n = np.hypot(dx, dy)
        return dx / n, dy / n

    def width(self, s):
        ln = np.interp(s, self.s_pts, self.lanes)
        return ln, ln * self.rc["lane_width_m"]


def project(P, cam, yaw_vec, pitch, F):
    """世界の点 P (N×3) をカメラの画面 (x 右・y 上、焦点距離 F) へ。f はカメラの前方の距離。"""
    fx, fy = yaw_vec
    rx, ry = fy, -fx                       # 進行方向の右 (水平)
    d = P - cam
    f0 = d[:, 0] * fx + d[:, 1] * fy
    r = d[:, 0] * rx + d[:, 1] * ry
    u0 = d[:, 2]
    cp, sp = np.cos(pitch), np.sin(pitch)  # pitch: 下向きが負 (下りで負)
    f = f0 * cp + u0 * sp
    u = -f0 * sp + u0 * cp
    with np.errstate(divide="ignore", invalid="ignore"):
        return F * r / f, F * u / f, f


class Terrain:
    """DEM (ncst_dem_crop.py の npz) を局所の東・北 [m] の格子として持ち、標高と陰影を返す。

    道路の脇 (中心線から corridor_m 以内) は、路面より corridor_drop_m 低くする。
    30 m の格子では切り通しの壁や路肩の盛り土が路面にかぶって見えるため (描画のための処理で、地形の値の補正ではない)。"""

    def __init__(self, npz: Path, road: "Road", lat0: float, lon0: float, tc: dict):
        from scipy.interpolate import RegularGridInterpolator
        from scipy.spatial import cKDTree
        D = np.load(npz)
        lat, lon, Z = D["lat"][::-1], D["lon"], D["elev"][::-1].astype(float)   # 南から北へ
        gx, _ = enu(np.full(len(lon), lat0), lon, lat0, lon0)
        _, gy = enu(lat, np.full(len(lat), lon0), lat0, lon0)
        ss = np.arange(0, road.s_max, 10.0)
        rx, ry, rz = road.at(ss)
        XX, YY = np.meshgrid(gx, gy)
        dist, idx = cKDTree(np.column_stack([rx, ry])).query(np.column_stack([XX.ravel(), YY.ravel()]), distance_upper_bound=tc["corridor_m"])
        near = np.isfinite(dist)
        Zf = Z.ravel().copy()
        Zf[near] = np.minimum(Zf[near], rz[idx[near]] - tc["corridor_drop_m"])
        Z = Zf.reshape(Z.shape)
        dzdx, dzdy = np.gradient(Z, gx, gy, axis=(1, 0))
        az, alt = np.radians(tc["light_azimuth_deg"]), np.radians(tc["light_altitude_deg"])
        L = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])   # 東・北・上
        n = np.stack([-dzdx, -dzdy, np.ones_like(Z)])
        shade = np.clip(np.tensordot(L, n, axes=1) / np.linalg.norm(n, axis=0), 0, 1)
        self.fz = RegularGridInterpolator((gy, gx), Z, bounds_error=False, fill_value=np.nan)
        self.fs = RegularGridInterpolator((gy, gx), shade, bounds_error=False, fill_value=np.nan)
        self.n_corridor = int(near.sum())


def draw_view(ax, road: Road, s0: float, cc: dict, rc: dict, aspect: float, terrain: "Terrain | None" = None, tc: dict | None = None):
    """s0 [m] の位置の運転席から見た前方の CG を ax に描く (ax の範囲は x ±aspect、y ±1)。

    地形・路面・白線の 4 角形をすべて前方の距離で並べ、遠くから近くの順に重ねる (画家のアルゴリズム)。"""
    F = 1 / np.tan(np.radians(cc["hfov_deg"] / 2)) * aspect
    tx, ty = road.tangent(s0)
    x0, y0, z0 = road.at(s0)
    _, _, z1 = road.at(s0 + cc["pitch_window_m"])
    pitch = float(np.arctan2(z1 - z0, cc["pitch_window_m"]))
    cam = np.array([x0 + ty * cc["lateral_m"], y0 - tx * cc["lateral_m"], z0 + cc["height_m"]])
    sky = np.array([0.81, 0.89, 0.95])
    # 空と、地形の外 (遠くの地面): 水平線 (世界の水平) の画面の高さ。視線が下を向く (下り、pitch < 0) と水平線は画面の上へ動く
    hy = -F * np.tan(pitch)
    ax.add_patch(plt.Rectangle((-aspect, hy), 2 * aspect, 1 - hy + 1, color=tuple(sky), zorder=0))
    ax.add_patch(plt.Rectangle((-aspect, -1.5), 2 * aspect, hy + 1.5, color="#c9c3b3", zorder=0))
    polys, cols, depth = [], [], []

    def quads(xa, ya, xb, yb, ok, dd, col_fn):
        for i in range(len(dd) - 1, 0, -1):
            if ok[i] and ok[i - 1]:
                polys.append([(xa[i], ya[i]), (xb[i], yb[i]), (xb[i - 1], yb[i - 1]), (xa[i - 1], ya[i - 1])])
                cols.append(col_fn(i))
                depth.append(dd[i])

    # ---- 地形: 運転席から扇形に標本を取る (方位 × 距離)
    if terrain is not None:
        half = np.radians(cc["hfov_deg"] / 2 + tc["azimuth_margin_deg"])
        yaw = np.arctan2(tx, ty)                             # 北から時計回り
        azs = yaw + np.linspace(-half, half, tc["n_azimuth"])
        rr = np.geomspace(tc["near_m"], tc["far_m"], tc["n_range"])
        A, Rg = np.meshgrid(azs, rr)                         # 行 = 距離、列 = 方位
        PX, PY = cam[0] + Rg * np.sin(A), cam[1] + Rg * np.cos(A)
        PZ = terrain.fz(np.column_stack([PY.ravel(), PX.ravel()])).reshape(Rg.shape)
        SH = terrain.fs(np.column_stack([PY.ravel(), PX.ravel()])).reshape(Rg.shape)
        sx, sy, sf = project(np.column_stack([PX.ravel(), PY.ravel(), PZ.ravel()]), cam, (tx, ty), pitch, F)
        sx, sy, sf = sx.reshape(Rg.shape), sy.reshape(Rg.shape), sf.reshape(Rg.shape)
        ok = np.isfinite(PZ) & (sf > 1.0)
        base_lo, base_hi = np.array(tc["color_dark"]), np.array(tc["color_light"])
        for i in range(len(rr) - 1, 0, -1):
            q = (rr[i] / tc["far_m"]) ** 0.7 * tc["haze"]
            for j in range(len(azs) - 1):
                if ok[i, j] and ok[i, j + 1] and ok[i - 1, j] and ok[i - 1, j + 1]:
                    polys.append([(sx[i, j], sy[i, j]), (sx[i, j + 1], sy[i, j + 1]), (sx[i - 1, j + 1], sy[i - 1, j + 1]), (sx[i - 1, j], sy[i - 1, j])])
                    sh = np.nanmean([SH[i, j], SH[i - 1, j]])
                    c = base_lo + (base_hi - base_lo) * sh
                    cols.append(tuple(c * (1 - q) + sky * q))
                    depth.append(rr[i])
    # ---- 路面
    d = np.geomspace(2.0, cc["view_m"], cc["n_samples"])
    ss = np.clip(s0 + d, 0, road.s_max)
    X, Y, Z = road.at(ss)
    TX, TY = road.tangent(ss)
    NX, NY = TY, -TX                       # 右向きの法線
    ln, w = road.width(ss)
    L, Rr = -w / 2 - rc["shoulder_left_m"], w / 2 + rc["shoulder_right_m"]

    def pt(off):
        P = np.column_stack([X + NX * off, Y + NY * off, Z])
        return project(P, cam, (tx, ty), pitch, F)

    xl, yl, fl = pt(L)
    xr, yr, fr = pt(Rr)
    asphalt = np.array([0.42, 0.435, 0.46])
    quads(xl, yl, xr, yr, (fl > 0.5) & (fr > 0.5), d,
          lambda i: tuple(asphalt * (1 - 0.55 * min(1.0, d[i] / cc["view_m"]) ** 0.6) + sky * 0.55 * min(1.0, d[i] / cc["view_m"]) ** 0.6))
    # 車道の端の実線と、車線の破線 (路面の少し手前の深さにして、同じ距離の路面より後に描く)
    dl = d - 0.01
    for off in (-w / 2, w / 2):
        xa, ya, fa = pt(off - 0.1)
        xb, yb, fb = pt(off + 0.1)
        quads(xa, ya, xb, yb, (fa > 0.5) & (fb > 0.5) & (d <= cc["lines_m"]), dl, lambda i: "white")
    period = rc["dash_len_m"] + rc["dash_gap_m"]
    for k in range(1, int(np.nanmax(ln))):
        off = -w / 2 + k * rc["lane_width_m"]
        xa, ya, fa = pt(off - 0.08)
        xb, yb, _ = pt(off + 0.08)
        m = (k < ln) & ((ss % period) < rc["dash_len_m"]) & (fa > 0.5) & (d <= cc["lines_m"])
        quads(xa, ya, xb, yb, m, dl, lambda i: "white")
    order = np.argsort(-np.asarray(depth), kind="stable")
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=[cols[i] for i in order],
                                     edgecolors=[cols[i] for i in order], lw=0.25, zorder=2))
    # ボンネットの縁 (運転席から見た手前の目安)
    ax.add_patch(plt.Rectangle((-aspect, -1.0), 2 * aspect, 0.12, color="#2c3e50", zorder=4))
    ax.set_xlim(-aspect, aspect)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    return pitch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--start-min", type=float, default=None)
    ap.add_argument("--end-min", type=float, default=None)
    ap.add_argument("--step-s", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--stills-min", type=float, nargs="*", default=[140.0, 144.0, 150.0, 156.0])
    a = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    V = dict(cfg["video"], **{k: v for k, v in (("start_min", a.start_min), ("end_min", a.end_min), ("step_s", a.step_s), ("fps", a.fps)) if v is not None})
    out = OUT / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    G = pd.read_csv(cfg["inputs"]["timeline"])
    R = pd.read_csv(cfg["inputs"]["route_profile"])
    osm = json.loads(Path(cfg["inputs"]["osm"]).read_text())
    rc, cc = cfg["road"], cfg["camera"]
    lanes, lanes_share = lanes_along(R, osm, rc)
    road = Road(R, lanes, rc)
    tc = cfg.get("terrain")
    terrain = Terrain(Path(tc["dem_npz"]), road, float(R.lat.iat[0]), float(R.lon.iat[0]), tc) if tc else None
    pd.DataFrame(dict(s_m=R.s_m, lat=R.lat, lon=R.lon, elev_s=R.elev_s, grade=R.grade, lanes=lanes)).to_csv(out / "road_model.csv", index=False)

    configure_japanese_font()
    W = G[(G["min"] >= V["start_min"]) & (G["min"] <= V["end_min"])].reset_index(drop=True)
    if V.get("trim_to_estimate"):   # 位置の推定がある時刻だけにする (GPS が切れる前と、最後の下りの後を除く)
        fin = np.flatnonzero(np.isfinite(W.s_est_m.to_numpy()))
        W = W.iloc[fin[0]:fin[-1] + 1].reset_index(drop=True)
        V["start_min"], V["end_min"] = float(W["min"].iat[0]), float(W["min"].iat[-1])
    dc = cfg["display"]
    RS = json.loads(Path(cfg["inputs"]["route_summary"]).read_text())
    summit_km, summit_elev = RS["summit"]["km"], RS["summit"]["elev_m"]
    desc = [dict(n=i + 1, km=d["km"], drop=d["drop_m"], mean=d["mean_grade_pct"], elev=d["elev_m"]) for i, d in enumerate(RS["descents"])]
    m = W["min"].to_numpy()
    s_km = W.s_est_m.to_numpy() / 1000
    for d in desc:   # 下りの区間の時刻 (推定位置がその km の範囲にある時刻)
        w = (s_km >= d["km"][0]) & (s_km <= d["km"][1])
        d["min"] = [float(m[w].min()), float(m[w].max())]
    # ブレーキを踏んでいる (リスク指標の連続制動と同じ判定) ・踏み始めの回数・直前 5 分のブレーキ時間の割合
    on = (W.psi > dc["brake_on_psi"]).to_numpy()
    start = on & ~np.r_[False, on[:-1]]
    ks_on = np.flatnonzero(start)
    keep = np.r_[True, np.diff(ks_on) > dc["brake_merge_s"]] if len(ks_on) else np.array([], bool)
    n_cum = np.zeros(len(W), int)
    n_cum[ks_on[keep]] = 1
    n_cum = np.cumsum(n_cum)
    duty = pd.Series(on.astype(float)).rolling(int(dc["duty_window_s"]), min_periods=1).mean().to_numpy()
    tmin, tmax = dc["temp_range_C"]

    fig = plt.figure(figsize=(14.4, 8.1), dpi=100)
    aspect = dc["view_aspect"]
    axc = fig.add_axes([0.008, 0.355, 0.515, 0.585])
    axprof = fig.add_axes([0.045, 0.06, 0.255, 0.235])
    axpg = fig.add_axes([0.315, 0.03, 0.105, 0.29])
    axtg = fig.add_axes([0.43, 0.03, 0.09, 0.29])
    gsr = fig.add_gridspec(5, 1, left=0.585, right=0.975, top=0.94, bottom=0.07, hspace=0.18)
    axp = [fig.add_subplot(gsr[i]) for i in range(5)]
    BLUE_BAND = "#d6eaf8"
    panels = [("標高 [m]", W.elev_s, "#2e86c1"), ("勾配 [%]", W.grade * 100, "#2e86c1"),
              ("ブレーキ圧 [psi]", W.psi, "#c0392b"), ("ブレーキ温度 [°C]", W["T"], "#922b21")]
    for ax, (lab, y, c) in zip(axp, panels):
        ax.plot(m, y, color=c, lw=1)
        ax.set_ylabel(lab, fontsize=8)
    axp[1].axhline(0, color="#888", lw=0.6)
    ar = axp[4]
    rr = yaml.safe_load(Path("configs/brake/fade_risk_index.yaml").read_text())["rank"]
    ar.axhspan(rr["high"], 100, color="#FADBD8", lw=0, zorder=0.2)
    ar.axhspan(rr["medium"], rr["high"], color="#FDEBD0", lw=0, zorder=0.2)
    ar.plot(m, W.risk_score, color="#1F2A44", lw=1.3)
    ar.set_ylim(0, 100)
    ar.set_ylabel("フェードリスク", fontsize=8)
    ar.text(0.01, 0.80, "High ≥ 60 / Medium ≥ 30 (確率ではない)", transform=ar.transAxes, fontsize=7)
    cursors = []
    for ax in axp:
        for d in desc:   # 地図の下り坂の時刻。リスクの段は High・Medium の帯と混ざらないよう、下端の細い帯にする
            if ax is ar:
                ax.axvspan(*d["min"], ymin=0, ymax=0.07, color="#5dade2", lw=0, zorder=0.5)
            else:
                ax.axvspan(*d["min"], color=BLUE_BAND, lw=0, zorder=0.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlim(V["start_min"], V["end_min"])
        cursors.append(ax.axvline(V["start_min"], color="#000", lw=1.2))
    for d in desc:
        axp[0].text(np.mean(d["min"]), 0.06, f"下り {d['n']}  {-d['drop']:.0f} m", transform=axp[0].get_xaxis_transform(), ha="center",
                    fontsize=8, color="#1b4f72", fontweight="bold")
    for ax in axp[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axp[-1].set_xlabel("走り始めからの時間 [分]   (水色 = 地図の長い下り坂。リスクの段は下端の帯)", fontsize=8)
    fig.suptitle("NCST 大型トラック I-15 Cajon Pass 南行き: 前方は地図から描いた CG (実際の映像ではない。位置は推定、地形は DEM 約 30 m、他の車は描いていない)", fontsize=9.5, y=0.985)
    # 標高の断面 (経路全体。縦を強調) の静的な部分
    km = R.s_m.to_numpy() / 1000
    el = R.elev_s.to_numpy()
    axprof.fill_between(km, el.min() - 40, el, color="#e5e8e8")
    axprof.plot(km, el, color="#7f8c8d", lw=1)
    for d in desc:
        w = (km >= d["km"][0]) & (km <= d["km"][1])
        axprof.fill_between(km[w], el.min() - 40, el[w], color="#aed6f1")
        axprof.text(np.mean(d["km"]), el.min() - 25, f"下り {d['n']}\n{d['km'][1] - d['km'][0]:.1f} km で {-d['drop']:.0f} m\n(平均 {d['mean']:.1f}%)",
                    ha="center", va="bottom", fontsize=6.8, color="#1b4f72")
    axprof.annotate(f"峠 {summit_elev:.0f} m", (summit_km, summit_elev), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#333")
    kmax = float(np.nanmax(s_km)) + 1.0
    axprof.set_xlim(0, kmax)
    axprof.set_ylim(el.min() - 40, el[km <= kmax].max() + 60)
    axprof.set_xlabel("I-15 南行きの距離 [km] (推定の経路)", fontsize=7)
    axprof.set_ylabel("標高 [m]", fontsize=7)
    axprof.tick_params(labelsize=6.5)
    axprof.grid(alpha=0.3)
    prof_dyn = []

    def gauge_pressure(ax, psi, braking, du):
        ax.cla()
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.25, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        pmax = dc["psi_max"]
        th = np.linspace(np.pi, 0, 100)
        ax.plot(np.cos(th), np.sin(th), color="#d5d8dc", lw=9, solid_capstyle="butt")
        f = min(max(psi, 0) / pmax, 1)
        th2 = np.linspace(np.pi, np.pi * (1 - f), 60)
        ax.plot(np.cos(th2), np.sin(th2), color="#c0392b" if braking else "#e6b0aa", lw=9, solid_capstyle="butt")
        for v in (0, 10, 20):
            t = np.pi * (1 - v / pmax)
            ax.text(1.22 * np.cos(t), 1.22 * np.sin(t), f"{v}", ha="center", va="center", fontsize=6.5, color="#555")
        t = np.pi * (1 - f)
        ax.plot([0, 0.8 * np.cos(t)], [0, 0.8 * np.sin(t)], color="#1F2A44", lw=2.2)
        ax.add_patch(plt.Circle((0, 0), 0.07, color="#1F2A44"))
        ax.text(0, 1.17, "ブレーキ圧", ha="center", fontsize=8.5, fontweight="bold", color="#1F2A44", va="bottom")
        ax.text(0, -0.25, f"{psi:.0f} psi", ha="center", va="center", fontsize=9, color="#1F2A44")
        ax.text(0, -0.58, "踏んでいる" if braking else "踏んでいない", ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white", bbox=dict(facecolor="#c0392b" if braking else "#aab7b8", lw=0, boxstyle="round,pad=0.25"))
        ax.add_patch(plt.Rectangle((-1, -1.12), 2, 0.17, color="#eaeded"))
        ax.add_patch(plt.Rectangle((-1, -1.12), 2 * du, 0.17, color="#c0392b"))
        ax.text(0, -1.22, f"直前 5 分のブレーキ時間 {du * 100:.0f}%", ha="center", va="top", fontsize=6.8, color="#333")

    def gauge_temp(ax, T):
        """縦の温度計: 下から今の温度まで色 (TCMAP) で塗る。上の空きは薄い灰。"""
        ax.cla()
        ax.set_xlim(0, 1)
        ax.set_ylim(tmin - 26, tmax + 18)
        ax.axis("off")
        Tc = float(np.clip(T, tmin, tmax))
        x0, x1 = 0.30, 0.56
        ax.add_patch(plt.Rectangle((x0, tmin), x1 - x0, tmax - tmin, color="#eaeded", lw=0))
        yf = np.linspace(tmin, Tc, 80)
        ax.imshow(yf[:, None], extent=(x0, x1, tmin, Tc), origin="lower", aspect="auto", cmap=TCMAP, norm=TNORM, zorder=2)
        ax.add_patch(plt.Rectangle((x0, tmin), x1 - x0, tmax - tmin, fill=False, ec="#555", lw=0.8, zorder=3))
        for v in range(int(np.ceil(tmin / 25) * 25), int(tmax) + 1, 25):
            ax.plot([x1, x1 + 0.06], [v, v], color="#555", lw=0.6)
            ax.text(x1 + 0.09, v, f"{v}", fontsize=6.5, va="center", color="#555")
        ax.plot([x0 - 0.06, x1 + 0.06], [Tc, Tc], color="#1F2A44", lw=1.6, zorder=4)
        ax.text((x0 + x1) / 2, tmax + 4, "ブレーキ温度", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1F2A44")
        bright = TNORM(Tc) > 0.6   # 明るい色 (黄) の上は濃い文字
        ax.text((x0 + x1) / 2, tmin - 6, f"{T:.0f} °C", ha="center", va="top", fontsize=11, color="#1F2A44" if bright else "white", fontweight="bold",
                bbox=dict(facecolor=TCMAP(TNORM(Tc)), lw=0, boxstyle="round,pad=0.25"))

    def slope_sign(ax, k, r):
        """CG の左下: 勾配 (傾きを誇張した坂の絵)・峠からの高度差・いまの下りの進み具合・ブレーキの回数。"""
        g = float(r.grade) if np.isfinite(r.grade) else np.nan
        s = float(r.s_est_m) / 1000
        cur = next((d for d in desc if d["km"][0] <= s <= d["km"][1]), None)
        x0, y0, w, h = -aspect + 0.03, -0.86, 1.62, 0.52
        bg = "#1b4f72" if cur else ("#7b241c" if np.isfinite(g) and g > 0.01 else "#566573")
        ax.add_patch(plt.Rectangle((x0, y0), w, h, color=bg, alpha=0.88, zorder=12))
        # 坂の絵 (傾きを 5 倍に誇張)
        ex = dc["slope_icon_exaggeration"]
        cx, cy, L = x0 + 0.20, y0 + 0.30, 0.15
        ang = np.arctan(g * ex) if np.isfinite(g) else 0.0
        ax.plot([cx - L * np.cos(ang), cx + L * np.cos(ang)], [cy - L * np.sin(ang), cy + L * np.sin(ang)], color="white", lw=4, zorder=13, solid_capstyle="round")
        ax.add_patch(plt.Polygon([[cx - L * np.cos(ang), cy - L * np.sin(ang)], [cx + L * np.cos(ang), cy + L * np.sin(ang)], [cx + L * np.cos(ang), cy - L * abs(np.sin(ang)) - 0.03],
                                  [cx - L * np.cos(ang), cy - L * abs(np.sin(ang)) - 0.03]], color="white", alpha=0.25, zorder=13))
        ax.text(cx, y0 + 0.04, f"傾き ×{ex:g}", ha="center", va="bottom", fontsize=6, color="#ddd", zorder=13)
        word = "下り坂" if np.isfinite(g) and g < -0.01 else ("上り坂" if np.isfinite(g) and g > 0.01 else "ほぼ平ら")
        ax.text(x0 + 0.42, y0 + h - 0.04, f"{word}  {g * 100:+.1f}%" if np.isfinite(g) else "勾配 推定なし", ha="left", va="top", fontsize=17, fontweight="bold", color="white", zorder=13)
        if s >= summit_km:
            hd = f"峠から {r.elev_s - summit_elev:+.0f} m (標高 {r.elev_s:.0f} m)"
        else:
            hd = f"峠まで {summit_km - s:.1f} km・あと {summit_elev - r.elev_s:+.0f} m (標高 {r.elev_s:.0f} m)"
        ax.text(x0 + 0.42, y0 + h - 0.24, hd, ha="left", va="top", fontsize=10, color="white", zorder=13)
        if cur:
            i0 = int(np.argmin(np.abs(m - cur["min"][0])))
            done_km, done_m = s - cur["km"][0], cur["elev"][0] - r.elev_s
            nb = int(n_cum[k] - (n_cum[i0 - 1] if i0 > 0 else 0))
            ax.text(x0 + 0.42, y0 + h - 0.38, f"下り {cur['n']}: {done_km:.1f}/{cur['km'][1] - cur['km'][0]:.1f} km で {-done_m:.0f} m  ブレーキ {nb} 回",
                    ha="left", va="top", fontsize=9.5, color="#fdebd0", fontweight="bold", zorder=13)

    def render(k: int, title: str):
        r = W.iloc[k]
        axc.cla()
        axc.axis("off")
        draw_view(axc, road, float(r.s_est_m), cc, rc, aspect, terrain, tc)
        axc.text(-aspect + 0.02, 0.98, f"{title}  {r['min']:6.2f} 分   I-15 南行き {r.s_est_m / 1000:.1f} km (推定)",
                 va="top", fontsize=9.5, color="white", bbox=dict(facecolor="black", alpha=0.5, lw=0), zorder=10)
        rk = r.risk_rank if isinstance(r.risk_rank, str) else None
        col = RANK_COL.get(rk, "#BBBBBB")
        txt = f"フェードリスク {r.risk_score:.0f}  {rk}" if rk else "フェードリスク 算出なし"
        axc.text(aspect - 0.02, 0.98, txt, va="top", ha="right", fontsize=15 if rk else 10, color="white", fontweight="bold",
                 bbox=dict(facecolor=col, alpha=0.92, lw=0, boxstyle="round,pad=0.4"), zorder=10)
        axc.add_patch(plt.Rectangle((-aspect, -1), 2 * aspect, 2, fill=False, lw=6, ec=col, zorder=11, clip_on=False))
        slope_sign(axc, k, r)
        axc.text(aspect - 0.02, -0.97, "CG (OSM の道路・USGS 3DEP の標高と DEM)。視点 2.5 m・車道の中央 (仮定)", ha="right", va="bottom", fontsize=7, color="#ddd", zorder=10)
        gauge_pressure(axpg, float(r.psi), bool(on[k]), float(duty[k]))
        gauge_temp(axtg, float(r["T"]))
        for a_ in prof_dyn:
            a_.remove()
        prof_dyn.clear()
        sk = float(r.s_est_m) / 1000
        w = km <= sk
        prof_dyn.extend(axprof.plot(km[w], el[w], color="#1F2A44", lw=2.2))
        prof_dyn.extend(axprof.plot([sk], [r.elev_s], "o", ms=9, color=TCMAP(TNORM(r["T"])), mec="#111", zorder=5))
        for c in cursors:
            c.set_xdata([r["min"], r["min"]])

    for i, mm in enumerate(a.stills_min):
        k = int(np.argmin(np.abs(W["min"] - mm)))
        render(k, f"場面 {i + 1}")
        fig.savefig(out / f"still_{i + 1}_{mm:g}min.png", dpi=100)
    ks = np.arange(0, len(W), V["step_s"])
    W2, H2 = fig.canvas.get_width_height()
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W2}x{H2}", "-r", str(V["fps"]),
                             "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "23", str(out / "scene.mp4")],
                            stdin=subprocess.PIPE)
    speed = V["step_s"] * V["fps"]
    for k in ks:
        render(int(k), f"{speed:g} 倍速")
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    brake_in_desc = {f"下り{d['n']}": int(n_cum[np.flatnonzero(m <= d["min"][1])[-1]] - n_cum[max(0, np.flatnonzero(m >= d["min"][0])[0] - 1)]) for d in desc}
    S = dict(frames=int(len(ks)), speedup=speed, window_min=[V["start_min"], V["end_min"]],
             lanes_from_osm_share=round(lanes_share, 3), lanes_range=[int(lanes.min()), int(lanes.max())],
             dem_cells_lowered_near_road=None if terrain is None else terrain.n_corridor,
             descents=desc, brake_applications_in_descent=brake_in_desc)
    (out / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))
    (out / "meta.json").write_text(json.dumps(dict(vars(a), config=CFG.as_posix(), cfg=cfg, video=V,
                                                   sources=dict(osm="OpenStreetMap contributors (ODbL)", elevation="USGS 3DEP (public domain)", dem="USGS 3DEP 1 arc-second n35w118 (public domain)")),
                                              ensure_ascii=False, indent=1, default=str))
    print(json.dumps(S, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
