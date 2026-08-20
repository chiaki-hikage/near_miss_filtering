#!/usr/bin/env python3
"""1 セグメントの時系列をプロットする。

中身を目で見るための道具。抽出パイプラインには影響しない。
検出に使っている特徴量はすべてパネルとして出せる。各パネルには
設定ファイルの閾値と、検出された区間、ゲートで判定対象外になった区間を重ねる。

IMU は特徴量には使っていないが、CAN 側の値を裏取りする目的でこのスクリプトでは
読み込む。comma2k19 の IMU 軸は [forward, right, down]。
車載機の取付角のぶんだけ重力成分が forward / right に漏れるため、
縦加速度として重ねるときは直流成分を抜いてある (抜いた量は凡例に出す)。

使い方:
  # comma2k19。review.md や candidates.csv の表記をそのまま貼る
  python scripts/plot_segment.py raw_data/Chunk_1 -s "b0c9d2329ad1606b|2018-08-17--14-55-39/1"

  # commaCarSegments
  python scripts/plot_segment.py raw_data/comma_car_segments \
      --dataset comma_car_segments -s "63e74ebb84173067|0000009c--84e520ce41/2"

  # 前後のセグメントも繋いで描く (候補が 60 秒境界を跨いでいるとき)
  python scripts/plot_segment.py ... -s ... --context 1

  # 日付だけでも部分一致で引ける
  python scripts/plot_segment.py raw_data/Chunk_1 -s 2018-08-17--14-55-39/1

  # パネルを絞る
  python scripts/plot_segment.py raw_data/Chunk_1 -s ... --panels speed,accel_x,jerk,brake
  python scripts/plot_segment.py raw_data/Chunk_1 --list-panels

指定が複数のセグメントに当たった場合は、候補を並べて終了する。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

from near_miss.config import (
    DEFAULT_DETECTION,
    DEFAULT_VEHICLE_DIR,
    find_vehicle_config,
    load_vehicle_configs,
    load_yaml,
)
from near_miss.detectors import build_gate, detect_all
from near_miss.features import compute_features
from near_miss.io.canonical import concat_segments
from near_miss.scoring import build_candidates
from near_miss.sources import car_segments_source, comma2k19_source
from near_miss.signals import moving_average, to_grid, window_samples

plt.rcParams["font.family"] = ["Hiragino Sans", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

WHEELS = (("ws_fl_mps", "前左"), ("ws_fr_mps", "前右"), ("ws_rl_mps", "後左"), ("ws_rr_mps", "後右"))


@dataclass
class PlotContext:
    """パネル描画に必要なものをまとめて持つ。"""

    t: np.ndarray                 # セグメント内の経過時間 [s]
    df: Any                       # 特徴量の DataFrame
    imu: dict[str, np.ndarray]
    cfg: dict[str, Any]
    events: list = field(default_factory=list)
    candidate_spans: list = field(default_factory=list)

    def has(self, *cols: str) -> bool:
        return all(c in self.df.columns for c in cols)

    def col(self, name: str) -> np.ndarray:
        return self.df[name].to_numpy()

    def spec(self, event: str) -> dict[str, Any] | None:
        return self.cfg["events"].get(event)

    def event_spans(self, *names: str) -> list[tuple[float, float]]:
        t0 = self.t[0] - 0.0
        return [
            (e.t_start - self._origin, e.t_end - self._origin)
            for e in self.events
            if e.event_type in names
        ]

    _origin: float = 0.0


# ---------------------------------------------------------------------------
# 描画の共通部品
# ---------------------------------------------------------------------------
def _shade(ax, spans, color, alpha=0.15, label=None):
    for i, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, label=label if i == 0 else None)


def _mask_spans(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """True が連続する区間を時刻の対にする。"""
    if mask is None or not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    st = list(np.flatnonzero(d == 1) + 1)
    en = list(np.flatnonzero(d == -1))
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(mask.size - 1)
    return [(t[a], t[b]) for a, b in zip(st, en)]


def _decorate(ax, ctx: PlotContext, *events: str) -> None:
    """イベント定義の閾値・検出区間・ゲート対象外を重ねる。"""
    drawn_gates: set[str] = set()
    for name in events:
        spec = ctx.spec(name)
        if spec is None:
            continue
        if spec["op"] in ("lt", "gt"):
            ax.axhline(spec["threshold"], color="tab:red", ls="--", lw=0.8)
        elif spec["op"] == "abs_gt":
            ax.axhline(spec["threshold"], color="tab:red", ls="--", lw=0.8)
            ax.axhline(-spec["threshold"], color="tab:red", ls="--", lw=0.8)
        # ゲートで判定対象外になっている区間
        gate, label = build_gate(ctx.df, spec)
        if gate is not None and label not in drawn_gates:
            # 同じ gate を共有するイベントを並べると凡例が重複する
            drawn_gates.add(label)
            _shade(ax, _mask_spans(ctx.t, ~gate), "gray", 0.12, f"判定対象外 ({label})")
        # 実際に発火した区間
        _shade(ax, ctx.event_spans(name), "tab:orange", 0.35, f"{name} 検出")


def _legend(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(loc=kw.pop("loc", "upper right"), fontsize=8, **kw)


# ---------------------------------------------------------------------------
# パネル
# ---------------------------------------------------------------------------
def panel_speed(ax, ctx: PlotContext) -> None:
    for col, name in WHEELS:
        if ctx.has(col):
            ax.plot(ctx.t, ctx.col(col) * 3.6, lw=0.6, alpha=0.5, label=f"輪速 {name}")
    if ctx.has("v_mps"):
        ax.plot(ctx.t, ctx.col("v_mps") * 3.6, lw=1.6, color="k", label="車速 (平滑後)")
    ax.set_ylabel("車速 [km/h]")
    _legend(ax, ncol=3)


def panel_accel_x(ax, ctx: PlotContext) -> None:
    ax.axhline(0, color="gray", lw=0.5)
    _decorate(ax, ctx, "hard_brake", "hard_accel")
    if ctx.has("ax_mps2"):
        ax.plot(ctx.t, ctx.col("ax_mps2"), lw=1.5, color="k", label="縦加速度 (車速微分・主系列)")
    if ctx.has("ax_can_mps2"):
        ax.plot(ctx.t, ctx.col("ax_can_mps2"), lw=0.8, color="tab:orange", alpha=0.8,
                label="ACCEL_X (CAN・スケール未確定)")
    if "acc_forward" in ctx.imu:
        dc = float(np.nanmean(ctx.imu["acc_forward"]))
        ax.plot(ctx.t, ctx.imu["acc_forward"] - dc, lw=0.8, color="tab:green", alpha=0.8,
                label=f"IMU forward (直流 {dc:+.2f} を除去)")
    ax.set_ylabel("縦加速度 [m/s²]")
    _legend(ax, loc="lower right", ncol=2)


def panel_jerk(ax, ctx: PlotContext) -> None:
    ax.axhline(0, color="gray", lw=0.5)
    _decorate(ax, ctx, "brake_jerk")
    if ctx.has("jerk_mps3"):
        ax.plot(ctx.t, ctx.col("jerk_mps3"), lw=1.0, color="tab:purple", label="縦躍度 jerk")
    ax.set_ylabel("縦躍度 [m/s³]")
    _legend(ax, loc="lower right")


def panel_accel_y(ax, ctx: PlotContext) -> None:
    ax.axhline(0, color="gray", lw=0.5)
    _decorate(ax, ctx, "lateral_accel")
    if ctx.has("ay_kin_mps2"):
        ax.plot(ctx.t, ctx.col("ay_kin_mps2"), lw=1.4, color="k",
                label="横加速度 ay_kin = 車速 × ヨーレート (主系列)")
    if ctx.has("ay_can_mps2"):
        ax.plot(ctx.t, ctx.col("ay_can_mps2"), lw=0.8, color="tab:orange", alpha=0.8,
                label="ACCEL_Y (CAN・スケール未確定)")
    if "acc_right" in ctx.imu:
        dc = float(np.nanmean(ctx.imu["acc_right"]))
        ax.plot(ctx.t, ctx.imu["acc_right"] - dc, lw=0.8, color="tab:green", alpha=0.7,
                label=f"IMU right (直流 {dc:+.2f} を除去)")
    ax.set_ylabel("横加速度 [m/s²]")
    _legend(ax, loc="lower right", ncol=2)


def panel_lat_jerk(ax, ctx: PlotContext) -> None:
    ax.axhline(0, color="gray", lw=0.5)
    _decorate(ax, ctx, "hard_steer")
    if ctx.has("lat_jerk_mps3"):
        ax.plot(ctx.t, ctx.col("lat_jerk_mps3"), lw=1.0, color="tab:blue", label="横躍度 (急操舵の判定量)")
    ax.set_ylabel("横躍度 [m/s³]")
    _legend(ax, loc="lower right")


def panel_steer(ax, ctx: PlotContext) -> None:
    """舵角そのもの。判定には使っていないが、蛇行の中身を見るために出す。"""
    ax.axhline(0, color="gray", lw=0.5)
    if ctx.has("steer_deg_s"):
        ax.plot(ctx.t, ctx.col("steer_deg_s"), lw=1.2, color="k", label="舵角 (平滑後)")
    if ctx.has("steer_detrended_deg"):
        ax.plot(ctx.t, ctx.col("steer_detrended_deg"), lw=1.0, color="tab:red", alpha=0.8,
                label="舵角 (5秒移動平均を除去・蛇行判定の入力)")
        amp = ctx.cfg["weaving"]["amplitude_deg"]
        for sign in (1, -1):
            ax.axhline(sign * amp, color="tab:red", ls=":", lw=0.8)
    ax.set_ylabel("舵角 [deg]")
    _legend(ax, ncol=2)
    a2 = ax.twinx()
    a2.grid(False)
    if ctx.has("steer_rate_dps"):
        a2.plot(ctx.t, ctx.col("steer_rate_dps"), lw=0.7, color="tab:green", alpha=0.6,
                label="舵角レート (判定には未使用)")
    a2.set_ylabel("舵角レート [deg/s]")
    _legend(a2, loc="lower right")


def panel_weaving(ax, ctx: PlotContext) -> None:
    _decorate(ax, ctx, "weaving")
    if ctx.has("steer_reversals"):
        wv = ctx.cfg["weaving"]
        ax.step(ctx.t, ctx.col("steer_reversals"), where="post", lw=1.2, color="tab:brown",
                label=f"舵角の符号反転回数 (±{wv['amplitude_deg']}deg・直近{wv['count_window_s']:.0f}秒)")
    ax.set_ylabel("反転回数")
    _legend(ax, loc="upper left")


def panel_brake(ax, ctx: PlotContext) -> None:
    """制動系のフラグ。

    ABS の作動は 0x226 の ABSACT で見る。0x320 の FABS は故障フラグで、
    ABS が作動している区間でも 0 のままだった (docs/comma_car_segments.md 11 章)。
    """
    _decorate(ax, ctx, "abs_active", "vsc_active", "aeb_active")
    for col, name, color in (("brake_pressed", "ブレーキスイッチ", "tab:red"),
                             ("abs_active_flag", "ABS 作動 (ABSACT)", "tab:brown"),
                             ("vsc_active_flag", "VSC 作動 (VSCACT)", "tab:purple"),
                             ("precollision_active", "純正 AEB (PCS)", "tab:orange"),
                             ("abs_fault", "ABS 故障 (FABS)", "gray"),
                             ("fabs", "FABS", "gray")):
        if ctx.has(col):
            ax.step(ctx.t, ctx.col(col), where="post", lw=1.3, color=color, label=name)
    if ctx.has("slip_warn"):
        ax.step(ctx.t, ctx.col("slip_warn") / 4.0, where="post", lw=1.0, color="tab:pink",
                ls=":", label="滑り警告灯 (1/4 倍)")
    ax.set_ylabel("制動フラグ")
    ax.set_ylim(-0.15, 1.35)
    _legend(ax, ncol=3)


def panel_pedal(ax, ctx: PlotContext) -> None:
    """運転者の操作量。アクセルを戻してから制動までの流れを見る。"""
    _decorate(ax, ctx, "panic_brake", "panic_brake_pedal")
    if ctx.has("gas_pedal_pct") and np.isfinite(ctx.col("gas_pedal_pct")).any():
        ax.plot(ctx.t, ctx.col("gas_pedal_pct"), lw=1.2, color="tab:green", label="アクセル開度 [%]")
        ax.set_ylim(-3, 103)
        ax.set_ylabel("アクセル [%]")
    else:
        # 0x2C1 が流れていない車両がある (TSS2 で復号率 40%)
        ax.plot([], [], lw=1.2, color="tab:green", label="アクセル開度 — この車両では復号できない")
        ax.set_yticks([])
        ax.set_ylabel("アクセル")
    _legend(ax, loc="upper left")
    a2 = ax.twinx()
    a2.grid(False)
    if ctx.has("brake_mc_mpa"):
        a2.plot(ctx.t, ctx.col("brake_mc_mpa"), lw=1.4, color="tab:red",
                label="マスタシリンダ圧 [MPa]")
        a2.set_ylabel("ブレーキ圧 [MPa]")
    elif ctx.has("brake_position"):
        a2.plot(ctx.t, ctx.col("brake_position"), lw=1.2, color="tab:red",
                label="ブレーキ踏み込み量 [-]")
        a2.set_ylabel("ブレーキ [-]")
    _legend(a2, loc="upper right")


def panel_physics(ax, ctx: PlotContext) -> None:
    """自転車モデルの残差と逆操舵。車両が舵に従っているかを見る。"""
    _decorate(ax, ctx, "yaw_instability", "counter_steer")
    ax.axhline(0, color="gray", lw=0.5)
    if ctx.has("yaw_residual_sigma"):
        ax.plot(ctx.t, ctx.col("yaw_residual_sigma"), lw=1.2, color="tab:red",
                label="ヨー残差 [σ]")
        thr = ctx.cfg["events"]["yaw_instability"]["threshold"]
        for sign in (1, -1):
            ax.axhline(sign * thr, color="tab:red", ls="--", lw=0.8)
        ax.set_ylim(-max(8.0, thr * 1.6), max(8.0, thr * 1.6))
    ax.set_ylabel("ヨー残差 [σ]")
    _legend(ax, loc="upper left")
    a2 = ax.twinx()
    a2.grid(False)
    if ctx.has("counter_steer_active"):
        a2.step(ctx.t, ctx.col("counter_steer_active"), where="post", lw=1.1,
                color="tab:purple", label="逆操舵")
    if ctx.has("net_heading_win_deg"):
        a2.plot(ctx.t, ctx.col("net_heading_win_deg") / 15.0, lw=0.9, color="tab:gray",
                alpha=0.8, label="正味方位変化 / 15deg (gate)")
        a2.axhline(1.0, color="tab:gray", ls=":", lw=0.8)
    a2.set_ylim(-0.15, 2.2)
    a2.set_ylabel("逆操舵 / gate")
    _legend(a2, loc="upper right", ncol=2)


def panel_lead(ax, ctx: PlotContext) -> None:
    _decorate(ax, ctx, "low_ttc", "short_thw")
    if ctx.has("lead_distance_m"):
        ax.plot(ctx.t, ctx.col("lead_distance_m"), lw=1.2, color="tab:blue", label="先行車までの距離")
    ax.set_ylabel("車間距離 [m]")
    _legend(ax, loc="upper left")
    a2 = ax.twinx()
    a2.grid(False)
    if ctx.has("ttc_s"):
        a2.plot(ctx.t, ctx.col("ttc_s"), lw=1.2, color="tab:red", label="TTC")
        a2.axhline(ctx.cfg["events"]["low_ttc"]["threshold"], color="tab:red", ls="--", lw=0.8)
    if ctx.has("thw_s"):
        a2.plot(ctx.t, ctx.col("thw_s"), lw=1.0, color="tab:green", alpha=0.8, label="車間時間")
        a2.axhline(ctx.cfg["events"]["short_thw"]["threshold"], color="tab:green", ls="--", lw=0.8)
    if ctx.has("lead_vrel_mps"):
        a2.plot(ctx.t, ctx.col("lead_vrel_mps"), lw=0.7, color="gray", alpha=0.7, label="相対速度 [m/s]")
    a2.set_ylim(-12, 12)
    a2.set_ylabel("TTC / 車間時間 [s]")
    _legend(a2, loc="lower right", ncol=2)


def panel_wheel_spread(ax, ctx: PlotContext) -> None:
    """4 輪速のばらつき。旋回で説明できるぶんを引いた超過を判定に使う。"""
    _decorate(ax, ctx, "wheel_speed_anomaly")
    if ctx.has("ws_spread_mps"):
        ax.plot(ctx.t, ctx.col("ws_spread_mps"), lw=0.8, color="lightgray",
                label="最大−最小 (生)")
    if ctx.has("ws_spread_smooth_mps"):
        ax.plot(ctx.t, ctx.col("ws_spread_smooth_mps"), lw=1.0, color="tab:olive",
                label="最大−最小 (0.3s 平滑)")
    if ctx.has("ws_spread_excess_mps"):
        ax.plot(ctx.t, ctx.col("ws_spread_excess_mps"), lw=1.4, color="tab:red",
                label="旋回で説明できる量からの超過")
        ax.axhline(ctx.cfg["wheel_speed"]["min_excess_mps"], color="tab:red", ls="--", lw=0.8)
    ax.set_ylabel("輪速差 [m/s]")
    _legend(ax, loc="upper left", ncol=3)


def panel_imu_accel(ax, ctx: PlotContext) -> None:
    for axis, color in (("forward", "tab:blue"), ("right", "tab:orange"), ("down", "tab:green")):
        k = f"acc_{axis}"
        if k in ctx.imu:
            ax.plot(ctx.t, ctx.imu[k], lw=0.9, color=color, label=f"IMU 加速度 {axis}")
    ax.set_ylabel("IMU 加速度 [m/s²]")
    _legend(ax, loc="center right", ncol=3)


def panel_yaw(ax, ctx: PlotContext) -> None:
    ax.axhline(0, color="gray", lw=0.5)
    if ctx.has("yaw_rate_dps"):
        ax.plot(ctx.t, ctx.col("yaw_rate_dps"), lw=1.4, color="k", label="YAW_RATE (CAN)")
    if "gyro_down" in ctx.imu:
        # down 軸まわりなので、左旋回を正とする CAN と符号が逆になる
        ax.plot(ctx.t, -np.rad2deg(ctx.imu["gyro_down"]), lw=0.9, color="tab:green", alpha=0.8,
                label="IMU gyro down (符号反転)")
    ax.set_ylabel("ヨーレート [deg/s]")
    _legend(ax)


def panel_op(ax, ctx: PlotContext) -> None:
    """openpilot と ACC の作動。

    commaCarSegments では panda が controlsAllowed を直接報告する (op_engaged)。
    comma2k19 には無く、制御フレームの送信有無 (op_tx) しか使えない。
    こちらは全区間 1 に張り付くため作動判別には使えていない。
    """
    for col, name, color in (("op_engaged", "openpilot 介入", "tab:cyan"),
                             ("cruise_active", "ACC 作動", "tab:blue"),
                             ("acc_braking", "ACC 制動", "tab:orange"),
                             ("op_tx", "制御フレーム送信 (判別には未使用)", "gray")):
        if ctx.has(col):
            ax.step(ctx.t, ctx.col(col), where="post", lw=1.2, color=color, label=name)
    ax.set_ylabel("介入")
    ax.set_ylim(-0.15, 1.35)
    _legend(ax, loc="lower right")


PANELS: dict[str, tuple[str, Callable]] = {
    "speed": ("車速・輪速", panel_speed),
    "accel_x": ("縦加速度 (急ブレーキ・急加速)", panel_accel_x),
    "jerk": ("縦躍度 (制動ジャーク)", panel_jerk),
    "accel_y": ("横加速度 (横加速度過大)", panel_accel_y),
    "lat_jerk": ("横躍度 (急操舵)", panel_lat_jerk),
    "steer": ("舵角・舵角レート", panel_steer),
    "weaving": ("舵角の符号反転回数 (蛇行)", panel_weaving),
    "brake": ("制動フラグ (ABS / VSC / AEB)", panel_brake),
    "pedal": ("アクセル開度・ブレーキ圧", panel_pedal),
    "physics": ("ヨー残差・逆操舵", panel_physics),
    "lead": ("車間距離・TTC・車間時間", panel_lead),
    "wheel_spread": ("4輪速のばらつき", panel_wheel_spread),
    "imu_accel": ("IMU 加速度 3軸", panel_imu_accel),
    "yaw": ("ヨーレート (CAN と IMU)", panel_yaw),
    "op": ("openpilot 送信", panel_op),
}
DEFAULT_PANELS = tuple(PANELS)


def select_segments(refs, segment=None, drive=None, index=None):
    """セグメントの指定を解釈する。

    review.md や candidates.csv に出る "<ドライブ>/<番号>" をそのまま渡せる。
    ドライブ名は部分一致なので、日付だけでも引ける。
    """
    if segment:
        text = segment.strip().rstrip("/")
        drive_part, _, index_part = text.rpartition("/")
        if index_part.isdigit() and drive_part:
            drive, index = drive_part, int(index_part)
        elif index_part.isdigit() and not drive_part:
            index = int(index_part)
        else:
            drive = text
    if drive:
        refs = [r for r in refs if drive in r.drive_id]
    if index is not None:
        refs = [r for r in refs if r.index == index]
    return refs


def load_imu(seg_dir: Path, grid: np.ndarray, rate_hz: float, smooth_s: float) -> dict[str, np.ndarray]:
    """IMU を共通グリッドへ載せる。列は [forward, right, down]。"""
    out: dict[str, np.ndarray] = {}
    base = seg_dir / "processed_log" / "IMU"
    win = window_samples(smooth_s, rate_hz)
    for name, prefix in (("accelerometer", "acc"), ("gyro", "gyro")):
        d = base / name
        if not d.is_dir():
            continue
        t = np.load(d / "t", allow_pickle=True).ravel().astype(float)
        v = np.load(d / "value", allow_pickle=True).astype(float)
        for i, axis in enumerate(("forward", "right", "down")):
            out[f"{prefix}_{axis}"] = moving_average(np.interp(grid, t, v[:, i]), win)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root", type=Path, nargs="?")
    p.add_argument("--dataset", default="comma2k19", choices=("comma2k19", "comma_car_segments"))
    p.add_argument("--platform", default="TOYOTA_RAV4_TSS2", help="commaCarSegments の車種キー")
    p.add_argument("--context", type=int, default=0,
                   help="前後この数だけセグメントを繋いで描く (60 秒境界を跨ぐ候補用)")
    p.add_argument("--focus", default=None,
                   help='拡大する位置。"<セグメント番号>@<セグメント内の秒>" (例 2@23.4)')
    p.add_argument("--span", type=float, default=24.0, help="--focus で切り出す幅 [s]")
    p.add_argument("-s", "--segment", type=str, default=None,
                   help='セグメント指定。"<ドライブ>/<番号>" 形式、または部分一致する文字列')
    p.add_argument("--drive", type=str, default=None, help="ドライブ名の一部")
    p.add_argument("--index", type=int, default=None, help="セグメント番号")
    p.add_argument("--panels", type=str, default=None, help="描くパネルをカンマ区切りで指定")
    p.add_argument("--list-panels", action="store_true", help="パネル名を一覧する")
    p.add_argument("--config", type=Path, default=DEFAULT_DETECTION)
    p.add_argument("--vehicles", type=Path, default=DEFAULT_VEHICLE_DIR)
    p.add_argument("--out", type=Path, default=None, help="PNG の出力先")
    p.add_argument("--csv", type=Path, default=None, help="グリッド化した時系列の出力先")
    p.add_argument("--no-imu", action="store_true", help="IMU を読まない")
    args = p.parse_args()

    if args.list_panels:
        print("使えるパネル名 (--panels でカンマ区切り指定):")
        for name, (label, _) in PANELS.items():
            print(f"  {name:14s} {label}")
        return 0
    if args.data_root is None:
        p.error("データルートを指定してください")

    names = [n.strip() for n in args.panels.split(",")] if args.panels else list(DEFAULT_PANELS)
    unknown = [n for n in names if n not in PANELS]
    if unknown:
        print(f"未知のパネル名: {unknown}\n--list-panels で一覧できます")
        return 1

    cfg = load_yaml(args.config)
    vehicle_configs = load_vehicle_configs(args.vehicles)
    if args.dataset == "comma2k19":
        source = comma2k19_source(args.data_root, vehicle_configs)
    else:
        source = car_segments_source(args.data_root, args.platform, vehicle_configs)
    all_refs = source.refs
    refs = select_segments(all_refs, args.segment, args.drive, args.index)
    if not refs:
        print("該当するセグメントがありません")
        return 1
    if len(refs) > 1:
        print(f"指定が {len(refs)} 件のセグメントに当たりました。絞り込んでください:")
        for r in refs[:20]:
            print(f"  {r.segment_id}")
        if len(refs) > 20:
            print(f"  ... 他 {len(refs) - 20} 件")
        return 1
    ref = refs[0]

    vehicle = source.vehicle_for(ref)

    # 候補は連続セグメントを繋いだブロック単位で作られている。
    # 60 秒境界を跨ぐ候補を見るときは --context で前後も繋ぐ。
    block = [ref]
    if args.context > 0:
        want = {ref.index + d for d in range(-args.context, args.context + 1)}
        block = sorted(
            (r for r in all_refs if r.drive_id == ref.drive_id and r.index in want),
            key=lambda r: r.index,
        )
    segs = [source.load(r, vehicle, True) for r in block]
    # 各セグメントがブロック内のどこから始まるか。--focus の基準に使う。
    seg_starts = {r.index: float(s.t_span[0]) for r, s in zip(block, segs)}
    seg = concat_segments(segs)
    gs = compute_features(to_grid(seg, cfg), cfg, radar=seg.radar, vehicle=vehicle)
    t0 = float(gs.t[0])

    events = detect_all(gs, cfg, max_stage=2 if gs.raw_can_loaded else 1)
    cands = build_candidates(gs, events, cfg)
    imu = {}
    if not args.no_imu and source.name == "comma2k19" and len(block) == 1:
        imu = load_imu(ref.path, gs.t, gs.rate_hz, cfg["smoothing"]["accel_window_s"])

    ctx = PlotContext(
        t=gs.t - t0,
        df=gs.df,
        imu=imu,
        cfg=cfg,
        events=events,
        candidate_spans=[(c.t_start - t0, c.t_end - t0) for c in cands],
    )
    ctx._origin = t0

    fig, axes = plt.subplots(len(names), 1, figsize=(15, 1.95 * len(names)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    span = f"{block[0].index}-{block[-1].index}" if len(block) > 1 else str(ref.index)
    fig.suptitle(f"{ref.drive_id}/{span}   ({gs.vehicle} / {source.name})", fontsize=11, y=0.999)

    for ax, name in zip(axes, names):
        # 候補区間は全パネル共通の背景として先に敷く
        _shade(ax, ctx.candidate_spans, "tab:red", 0.10)
        PANELS[name][1](ax, ctx)

    axes[-1].set_xlabel("セグメント内の経過時間 [s]" if len(block) == 1 else "ブロック内の経過時間 [s]")
    lo, hi = ctx.t[0], ctx.t[-1]
    if args.focus:
        idx_s, _, t_in = args.focus.partition("@")
        try:
            center = seg_starts[int(idx_s)] - t0 + float(t_in)
        except (KeyError, ValueError):
            p.error(f'--focus を解釈できません: {args.focus!r} (例: "2@23.4")')
        lo = max(lo, center - args.span / 2)
        hi = min(hi, center + args.span / 2)
        for ax in axes:
            ax.axvline(center, color="k", ls=":", lw=1.0, alpha=0.6)
    axes[-1].set_xlim(lo, hi)
    fig.tight_layout()

    stem = f"{ref.drive_id.split('|')[-1]}_seg{ref.index}"
    out = args.out or Path("out") / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"  {out}")

    csv_path = args.csv or (out.with_suffix(".csv") if args.out is None else None)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        dump = gs.df.copy()
        dump.insert(1, "t_in_segment_s", ctx.t)
        for k, v in imu.items():
            dump[f"imu_{k}"] = v
        dump.to_csv(csv_path, index=False)
        print(f"  {csv_path}")

    print(f"\n検出: イベント {len(events)} 件 / 候補 {len(cands)} 件")
    for e in events:
        gate = f"  gate: {e.gate}" if e.gate else ""
        print(f"  {e.event_type:14s} {e.t_start - t0:6.2f}〜{e.t_end - t0:6.2f}s  "
              f"peak={e.peak_value:+8.2f}  ({e.feature} {e.op} {e.threshold}){gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
