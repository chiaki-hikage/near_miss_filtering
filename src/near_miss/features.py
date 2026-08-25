"""特徴量の算出 (L2)。

ここから下はデータセットにも車種にも依存しない。入力は一様グリッド上の
物理量の時系列だけ。列が無い場合はその特徴量を作らず、下流で欠測として扱う。

縦加速度の主系列は車速の微分にしてある。CAN の ACCEL_X はスケールの妥当性が
確認できていないため (docs/signals_rav4.md)、突き合わせ用の副系列に留める。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .signals import GriddedSignals, moving_average, window_samples

WHEEL_SPEED_COLUMNS = ("ws_fl_mps", "ws_fr_mps", "ws_rl_mps", "ws_rr_mps")


# ---------------------------------------------------------------------------
# 基本の演算
# ---------------------------------------------------------------------------
def derivative(x: np.ndarray, dt: float) -> np.ndarray:
    """一様グリッド前提の中心差分。NaN は前後へ 1 サンプル伝播する。"""
    out = np.full(x.shape, np.nan)
    if x.size < 3:
        return out
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    return out


def rolling(x: np.ndarray, window: int, how: str) -> np.ndarray:
    """中心合わせの移動窓集約。

    躍度のように微分で持ち上がったノイズは、1 サンプルのピークを見ると
    値が暴れる。短い窓で集約した値を補助特徴量として持つ。
    """
    if window <= 1:
        return x.astype(float).copy()
    if window % 2 == 0:
        window += 1
    ser = pd.Series(x).rolling(window, center=True, min_periods=max(2, window // 2))
    if how == "min":
        return ser.min().to_numpy()
    if how == "max":
        return ser.max().to_numpy()
    if how == "mean":
        return ser.mean().to_numpy()
    if how == "median":
        out = ser.median().to_numpy()
        half = window // 2
        out[:half] = np.nan
        out[-half:] = np.nan
        return out
    if how == "absmax":
        return pd.Series(np.abs(x)).rolling(
            window, center=True, min_periods=max(2, window // 2)
        ).max().to_numpy()
    raise ValueError(f"未知の集約方法: {how}")


def count_reversals(t: np.ndarray, x: np.ndarray, amplitude: float, window_s: float) -> np.ndarray:
    """振幅 amplitude を超える符号反転の、直近 window_s 秒間の回数。

    小さな揺れを数えないよう、±amplitude を交互に跨いだ時だけ 1 回と数える。
    """
    counts = np.zeros(t.shape)
    idx = np.flatnonzero(np.isfinite(x) & (np.abs(x) > amplitude))
    if idx.size < 2:
        return counts
    sign = np.sign(x[idx])
    changed = np.flatnonzero(np.diff(sign) != 0)
    if changed.size == 0:
        return counts
    rev_t = t[idx[changed + 1]]
    return (
        np.searchsorted(rev_t, t, side="right")
        - np.searchsorted(rev_t, t - window_s, side="right")
    ).astype(float)



# ---------------------------------------------------------------------------
# 車線変更の候補
# ---------------------------------------------------------------------------
@dataclass
class LaneChange:
    """S 字状の横運動を 1 件分まとめたもの。"""

    i_start: int
    i_end: int
    heading_amp_deg: float     # 2 つのローブの絶対値の平均。振りの大きさ
    net_heading_deg: float     # 対全体の方位角変化。0 に近いほど元の向きに戻っている
    offset_m: float            # 推定した横変位。正が左
    duration_s: float
    speed_mps: float


def _lobes(t: np.ndarray, x: np.ndarray, enter: float, merge_gap_s: float) -> list[tuple[int, int, float]]:
    """符号がそろって振れている区間を取り出す。

    戻り値は (開始, 終了, 積分値)。積分値は方位角の変化 [deg] にあたる。
    ゼロ近傍で細切れになるのを避けるため、同符号なら merge_gap_s まで繋ぐ。
    """
    sig = np.where(np.isfinite(x) & (np.abs(x) > enter), np.sign(x), 0.0)
    lobes: list[tuple[int, int, float]] = []
    i = 0
    n = len(sig)
    while i < n:
        if sig[i] == 0:
            i += 1
            continue
        s0, sign = i, sig[i]
        j = i
        while j + 1 < n:
            if sig[j + 1] == sign:
                j += 1
            elif sig[j + 1] == 0:
                # 同符号が merge_gap_s 以内に再開するなら繋ぐ
                k = j + 1
                while k < n and sig[k] == 0 and (t[k] - t[j]) <= merge_gap_s:
                    k += 1
                if k < n and sig[k] == sign:
                    j = k
                else:
                    break
            else:
                break
        area = float(np.trapezoid(np.nan_to_num(x[s0 : j + 1]), t[s0 : j + 1]))
        lobes.append((s0, j, area))
        i = j + 1
    return lobes


def find_lane_changes(
    t: np.ndarray,
    v_mps: np.ndarray,
    yaw_rate_dps: np.ndarray,
    rate_hz: float,
    cfg: dict[str, Any],
) -> list[LaneChange]:
    """ヨーレートの S 字パターンから車線変更の候補を拾う。

    comma2k19 には車線マーカの正解が無いため、車線変更と断定はできない。
    「1 車線ぶんに相当する横変位を伴い、方位角が元へ戻る左右一組の振り」
    という運動の形だけで候補とする。

    道路の曲率はヨーレートに定常成分として乗るので、長い窓で推定して引く。
    残った偏差が運転者による車線内・車線間の動きにあたる。

    曲率の推定に移動平均は使えない。ヨーレートが段状に変わるカーブの出入りで、
    平均が前後になまされて偏差に左右一組の振れが出る。これは車線変更の S 字と
    区別できず、カーブ 1 つにつき出入りで 2 件の誤検出になる。
    中央値フィルタは段差をそのまま保つのでこの振れが出ない。
    """
    lc = cfg["lane_change"]
    win = window_samples(lc["curvature_window_s"], rate_hz)
    yaw_dev = yaw_rate_dps - rolling(yaw_rate_dps, win, "median")

    lobes = _lobes(t, yaw_dev, lc["enter_dps"], lc["lobe_merge_gap_s"])
    out: list[LaneChange] = []
    for (a0, a1, area_a), (b0, b1, area_b) in zip(lobes, lobes[1:]):
        if np.sign(area_a) == np.sign(area_b):
            continue
        if (t[b0] - t[a1]) > lc["pair_max_gap_s"]:
            continue
        if min(abs(area_a), abs(area_b)) < lc["min_lobe_heading_deg"]:
            continue
        duration = float(t[b1] - t[a0])
        if not (lc["min_duration_s"] <= duration <= lc["max_duration_s"]):
            continue
        if abs(area_a + area_b) > lc["max_net_heading_deg"]:
            continue

        sl = slice(a0, b1 + 1)
        speed = float(np.nanmean(v_mps[sl]))
        if not np.isfinite(speed) or speed < lc["min_speed_mps"]:
            continue
        # 相対方位角を積分し、横変位 = ∫ 車速 × sin(方位角) dt を求める
        psi = np.concatenate(
            [[0.0], np.cumsum(np.deg2rad(np.nan_to_num(yaw_dev[sl][:-1])) * np.diff(t[sl]))]
        )
        offset = float(np.trapezoid(np.nan_to_num(v_mps[sl]) * np.sin(psi), t[sl]))
        if not (lc["min_offset_m"] <= abs(offset) <= lc["max_offset_m"]):
            continue

        out.append(
            LaneChange(
                i_start=a0,
                i_end=b1,
                heading_amp_deg=float((abs(area_a) + abs(area_b)) / 2.0),
                net_heading_deg=float(area_a + area_b),
                offset_m=offset,
                duration_s=duration,
                speed_mps=speed,
            )
        )
    return out




# ---------------------------------------------------------------------------
# 車両物理モデル
# ---------------------------------------------------------------------------
def bicycle_yaw_rate(
    v_mps: np.ndarray,
    steer_deg: np.ndarray,
    wheelbase_m: float,
    steer_ratio: float,
    understeer_gradient: float = 0.0,
    steer_offset_deg: float = 0.0,
    min_speed_mps: float = 3.0,
) -> np.ndarray:
    """線形単軌道モデルで、舵角から期待されるヨーレート [deg/s] を出す。

        r = v · δ / (SR · (L + Kus · v²))

    Kus はアンダーステア勾配。速度が上がるほど同じヨーレートに大きな舵角が要る。
    係数は Chunk_1 の定常旋回への当てはめで求めた (R² = 0.973)。

    定常状態のモデルなので、操舵の立ち上がりでは実測が遅れて残差が出る。
    残差を異常の指標に使うときは、この過渡ぶんを踏まえて判断する。
    """
    out = np.full(v_mps.shape, np.nan)
    ok = np.isfinite(v_mps) & np.isfinite(steer_deg) & (v_mps >= min_speed_mps)
    delta = np.deg2rad(steer_deg[ok] - steer_offset_deg)
    v = v_mps[ok]
    out[ok] = np.rad2deg(v * delta / (steer_ratio * (wheelbase_m + understeer_gradient * v**2)))
    return out


def sideslip_model_deg(
    v_mps: np.ndarray,
    yaw_rate_dps: np.ndarray,
    ay_mps2: np.ndarray,
    l_r: float,
    k: float,
    min_speed_mps: float,
) -> np.ndarray:
    """定常の線形単軌道モデルから重心位置の横滑り角 beta [deg] を出す。

        beta = l_r * yaw_rate / v  -  k * a_y          (k = m*l_f/(C_r*L))

    使うのは車速・ヨーレート・横加速度の 3 つだけで、いずれも CAN から取れる。

    a_y には **車体固定の横加速度センサの値** を入れること。v x yaw_rate を
    入れてはいけない。入れると式が yaw_rate と v だけの関数に潰れ、
    横滑りの情報が消える。
    路面のカントがある区間ではセンサは重力の分だけ小さく読むが、
    タイヤの横力もその分だけ小さいので、モデルの入力としてはセンサ値が正しい。

    KIT Multi-Surface Driving Maneuvers の光学式センサによる実測と照合した結果
    (docs/kit_msdm.md):
        相関 0.9865 / 回帰の傾き 1.048 / 誤差の標準偏差 0.99 deg
        誤差は beta の大きさによらずほぼ一定 (0.74〜1.24 deg、実測 ±19 deg の範囲)

    速度が min_speed_mps 未満では第 1 項が発散するので NaN にする。埋めない。
    """
    v = np.asarray(v_mps, dtype=float)
    out = np.degrees(
        l_r * np.deg2rad(yaw_rate_dps) / np.where(v > 0, v, np.nan) - k * np.asarray(ay_mps2, float)
    )
    out[~(v >= min_speed_mps)] = np.nan
    return out


def sideslip_expected_deg(
    v_mps: np.ndarray,
    yaw_expected_dps: np.ndarray,
    l_r: float,
    k: float,
    min_speed_mps: float,
) -> np.ndarray:
    """舵角だけから期待される重心横滑り角 [deg]。

    定常の線形単軌道モデルでは、舵で決まるヨーレート r_exp に対して
    横加速度は a_y = v * r_exp になるので、beta の式に代入して

        beta_exp = r_exp * (l_r / v  -  k * v)

    となる。実測のヨーレート・横加速度から出す sideslip_model_deg との差
    (beta_excess) が、「舵で説明できない横滑り」にあたる。

    括弧の中は v = sqrt(l_r / k) でゼロになる。この速度では舵が変わっても
    beta_exp が動かないので、差 beta_excess の感度も落ちる。
    ヨー応答の乖離 (yaw_residual_sigma) と併せて見ること。
    """
    v = np.asarray(v_mps, dtype=float)
    vv = np.where(v > 0, v, np.nan)
    out = np.degrees(np.deg2rad(yaw_expected_dps) * (l_r / vv - k * v))
    out[~(v >= min_speed_mps)] = np.nan
    return out


def sideslip_noise_deg(
    v_mps: np.ndarray,
    l_r: float,
    k: float,
    yaw_noise_dps: float,
    ay_noise_mps2: float,
) -> np.ndarray:
    """beta の推定に載るセンサ雑音の標準偏差 [deg]。

    beta = l_r * r / v - k * a_y なので、r と a_y の雑音が独立なら

        sigma_beta(v) = sqrt( (l_r * sigma_r / v)^2 + (k * sigma_ay)^2 )

    第 1 項が 1/v で効くため、低速では同じ beta でも意味が変わる。
    閾値を deg で固定すると低速側だけが雑音で埋まるので、
    この式で割った sigma 単位の量 (beta_sigma) を判定に使う。

    sigma_r / sigma_ay は車種設定の geometry に置く。straight 走行から
    実測して求める (scripts/calibrate_beta_noise.py)。

    ここに入るのはサンプルごとのばらつきだけで、モデル自体の誤差
    (KIT 実測との照合で 0.99 deg) は含まない。絶対値が本物の横滑りに
    当たるかどうかは、別にモデル誤差の水準で判断すること。
    """
    v = np.asarray(v_mps, dtype=float)
    vv = np.where(v > 0, v, np.nan)
    return np.degrees(np.hypot(l_r * np.deg2rad(yaw_noise_dps) / vv, k * ay_noise_mps2))


def counter_steer(
    yaw_rate_dps: np.ndarray,
    steer_rate_dps: np.ndarray,
    min_yaw_dps: float,
    min_steer_rate_dps: float,
) -> np.ndarray:
    """旋回中に逆向きへ操舵している区間を 1 とする。

    ヨーレートが立っている最中に、それを打ち消す向きへ舵を切る操作。
    滑りの立て直しや、旋回中の急な回避で現れる。

    ただしこれだけでは通常の操作と区別できない。カーブや車線変更の
    終わりでは、必ず舵を戻すので同じ形になる。イベントとして使うときは
    ヨーレートが十分大きいことと、他の指標との共起を併せて見る。
    """
    out = np.zeros(yaw_rate_dps.shape)
    ok = (
        np.isfinite(yaw_rate_dps)
        & np.isfinite(steer_rate_dps)
        & (np.abs(yaw_rate_dps) >= min_yaw_dps)
        & (np.abs(steer_rate_dps) >= min_steer_rate_dps)
        & (np.sign(yaw_rate_dps) != np.sign(steer_rate_dps))
    )
    out[ok] = 1.0
    return out


def find_s_evasions(
    t: np.ndarray,
    v_mps: np.ndarray,
    yaw_rate_dps: np.ndarray,
    ay_kin_mps2: np.ndarray,
    rate_hz: float,
    cfg: dict[str, Any],
) -> list[tuple[int, int, float, float]]:
    """出て戻る S 字 (回避操作) を拾う。

    車線変更は横に移って**そこへ留まる**。回避は横に出て**元へ戻る**。
    横変位の軌跡を追い、途中で十分ふくらみ、最後に元へ戻るものを回避とみなす。
    蛇行との違いは、1 往復で完結し、横加速度が大きく、短いこと。

    戻り値は (開始, 終了, 最大横変位[m], 最終横変位[m])。
    """
    ev = cfg["s_evasion"]
    win = window_samples(ev["curvature_window_s"], rate_hz)
    yaw_dev = yaw_rate_dps - rolling(yaw_rate_dps, win, "median")
    lobes = _lobes(t, yaw_dev, ev["enter_dps"], ev["lobe_merge_gap_s"])

    out: list[tuple[int, int, float, float]] = []
    i = 0
    while i < len(lobes):
        j = i
        while (
            j + 1 < len(lobes)
            and np.sign(lobes[j + 1][2]) != np.sign(lobes[j][2])
            and (t[lobes[j + 1][0]] - t[lobes[j][1]]) <= ev["lobe_gap_s"]
        ):
            j += 1
        if j - i + 1 >= ev["min_lobes"]:
            a, b = lobes[i][0], lobes[j][1]
            duration = float(t[b] - t[a])
            sl = slice(a, b + 1)
            speed = float(np.nanmean(v_mps[sl]))
            peak_ay = float(np.nanmax(np.abs(ay_kin_mps2[sl])))
            if (
                ev["min_duration_s"] <= duration <= ev["max_duration_s"]
                and speed >= ev["min_speed_mps"]
                and peak_ay >= ev["min_lateral_accel_mps2"]
            ):
                # 横変位の軌跡
                psi = np.concatenate(
                    [[0.0], np.cumsum(np.deg2rad(np.nan_to_num(yaw_dev[sl][:-1])) * np.diff(t[sl]))]
                )
                lat = np.concatenate(
                    [[0.0], np.cumsum(np.nan_to_num(v_mps[sl][:-1]) * np.sin(psi[:-1]) * np.diff(t[sl]))]
                )
                excursion = float(np.max(np.abs(lat)))
                final = float(lat[-1])
                if excursion >= ev["min_excursion_m"] and abs(final) <= ev["max_return_m"]:
                    out.append((a, b, excursion, final))
        i = j + 1
    return out


# ---------------------------------------------------------------------------
# 割り込み (カットイン) の候補
# ---------------------------------------------------------------------------
@dataclass
class CutIn:
    """先行車が別の車両に入れ替わり、車間が詰まった事象。"""

    i_switch: int
    distance_before_m: float
    distance_after_m: float
    thw_after_s: float
    target_speed_mps: float
    persisted_s: float


def find_cut_ins(
    t: np.ndarray,
    lead_id: np.ndarray,
    lead_distance_m: np.ndarray,
    thw_s: np.ndarray,
    lead_target_speed_mps: np.ndarray,
    v_mps: np.ndarray,
    rate_hz: float,
    cfg: dict[str, Any],
) -> list[CutIn]:
    """先行車が別の車両に入れ替わり、車間が詰まる事象を拾う。

    目視判定でリスクとされた車間逼迫は、いずれも他車が前方に割り込んだ場面だった。
    車間時間の絶対値より、割り込みで急に詰まったことが危険の中身にあたる。

    主判定は**距離の不連続**にしてある。実在の車両は 1 サンプル (0.05 秒) で
    数十 m も動けないので、距離が跳ぶことは別の物体に替わったことを意味する。

    トラック ID の変化は補助にとどめる。レーダはトラックの枠を使い回しており、
    同じ車両を追っていても ID が頻繁に入れ替わる (実測で 14 秒間に 28〜92 回)。
    ID が一定であることを求めると、実際の割り込みが 1 件も取れなかった。
    """
    ci = cfg["cut_in"]
    persist = window_samples(ci["persist_s"], rate_hz)
    n = len(t)
    valid = np.isfinite(lead_distance_m)

    # 1 サンプルでの距離の跳び
    jump = np.full(n, np.nan)
    jump[1:] = lead_distance_m[1:] - lead_distance_m[:-1]

    ids = np.nan_to_num(lead_id, nan=-1.0)
    id_changed = np.zeros(n, dtype=bool)
    id_changed[1:] = ids[1:] != ids[:-1]

    appeared = np.zeros(n, dtype=bool)
    appeared[1:] = valid[1:] & (~valid[:-1])

    cand = np.flatnonzero(
        valid & ((id_changed & (jump <= -ci["min_jump_m"])) | appeared)
    )

    out: list[CutIn] = []
    last_t = -np.inf
    for i in cand:
        if (t[i] - last_t) < ci["window_s"]:
            continue  # 同じ割り込みを重複して数えない
        after = slice(i, min(i + persist, n))
        if (t[after.stop - 1] - t[i]) < ci["persist_s"] * 0.8:
            continue

        if not np.isfinite(lead_distance_m[after]).any() or not np.isfinite(thw_s[after]).any():
            continue
        d_after = float(np.nanmedian(lead_distance_m[after]))
        thw_after = float(np.nanmedian(thw_s[after]))
        speed = float(np.nanmean(v_mps[after]))
        target = float(np.nanmedian(lead_target_speed_mps[after]))
        if not np.isfinite(d_after) or not np.isfinite(speed) or speed < ci["min_speed_mps"]:
            continue
        if not np.isfinite(thw_after) or thw_after > ci["max_thw_after_s"]:
            continue

        before = slice(max(i - persist, 0), i)
        d_before = (
            float(np.nanmedian(lead_distance_m[before]))
            if before.stop > before.start and np.isfinite(lead_distance_m[before]).any()
            else np.nan
        )

        if np.isfinite(d_before) and not appeared[i]:
            # 先行車がいた状態からの入れ替わり。前より近くなったことを求める
            if (d_before - d_after) < ci["min_distance_drop_m"]:
                continue
        else:
            # 先行車がいなかったところに現れた。近い位置に出たことを求める
            if d_after > ci["max_appear_distance_m"]:
                continue

        out.append(
            CutIn(
                i_switch=int(i),
                distance_before_m=d_before,
                distance_after_m=d_after,
                thw_after_s=thw_after,
                target_speed_mps=target,
                persisted_s=float(t[after.stop - 1] - t[i]),
            )
        )
        last_t = t[i]
    return out


# ---------------------------------------------------------------------------
# 蛇行
# ---------------------------------------------------------------------------
def find_weaving(
    t: np.ndarray,
    v_mps: np.ndarray,
    yaw_rate_dps: np.ndarray,
    ay_kin_mps2: np.ndarray,
    steer_rate_dps: np.ndarray,
    rate_hz: float,
    cfg: dict[str, Any],
) -> list[tuple[int, int, int, float]]:
    """短い時間に急な左右操作が交互に繰り返される区間を拾う。

    舵角の符号反転を数えるだけの定義では、通常の複数車線変更を蛇行として
    拾っていた (目視判定 6 件がすべて通常運転)。次の 3 点を加える。

      - 1 回ごとの振れが、横加速度と舵角レートの両方で裏付けられること
        (小さなふらつきや、操作していないのに出る揺れを除く)
      - 3 回以上が交互に、短い時間内に起きること
      - 正味の方位角変化が小さいこと
        (同じ向きへ続けて車線変更しただけのものを除く)

    戻り値は (開始, 終了, 振れの回数, 正味方位角[deg])。
    """
    wv = cfg["weaving"]
    yaw_dev = yaw_rate_dps - rolling(
        yaw_rate_dps, window_samples(wv["curvature_window_s"], rate_hz), "median"
    )
    lobes = _lobes(t, yaw_dev, wv["enter_dps"], wv["lobe_merge_gap_s"])

    # 横加速度と舵角レートの裏付けがある振れだけを残す
    strong: list[tuple[int, int, float]] = []
    for a, b, area in lobes:
        sl = slice(a, b + 1)
        if np.nanmax(np.abs(ay_kin_mps2[sl])) < wv["min_lateral_accel_mps2"]:
            continue
        if np.nanmax(np.abs(steer_rate_dps[sl])) < wv["min_steer_rate_dps"]:
            continue
        strong.append((a, b, area))

    out: list[tuple[int, int, int, float]] = []
    i = 0
    while i < len(strong):
        j = i
        while (
            j + 1 < len(strong)
            and np.sign(strong[j + 1][2]) != np.sign(strong[j][2])
            and (t[strong[j + 1][0]] - t[strong[j][1]]) <= wv["lobe_gap_s"]
        ):
            j += 1
        n = j - i + 1
        if n >= wv["min_reversals"]:
            a, b = strong[i][0], strong[j][1]
            duration = float(t[b] - t[a])
            net = float(sum(x[2] for x in strong[i : j + 1]))
            speed = float(np.nanmean(v_mps[a : b + 1]))
            if (
                wv["min_duration_s"] <= duration <= wv["max_duration_s"]
                and abs(net) <= wv["max_net_heading_deg"]
                and speed >= wv["min_speed_mps"]
            ):
                out.append((a, b, n, net))
        i = j + 1
    return out


def rolling_net_heading(t: np.ndarray, yaw_rate_dps: np.ndarray, half_window_s: float) -> np.ndarray:
    """各時刻を中心とした ±half_window_s の正味方位角変化 [deg] の絶対値。

    左折やランプ、長いカーブは向きが変わり続けるので大きな値になる。
    回避操作や車線変更は元の向きへ戻るので小さい。旋回を除くために使う。
    """
    if t.size < 3:
        return np.full(t.shape, np.nan)
    psi = np.concatenate([[0.0], np.cumsum(np.nan_to_num(yaw_rate_dps[:-1]) * np.diff(t))])
    lo = np.searchsorted(t, t - half_window_s, side="left")
    hi = np.searchsorted(t, t + half_window_s, side="right") - 1
    return np.abs(psi[np.clip(hi, 0, len(t) - 1)] - psi[np.clip(lo, 0, len(t) - 1)])


# ---------------------------------------------------------------------------
# 輪速の異常
# ---------------------------------------------------------------------------
def dilate_mask(mask: np.ndarray, samples: int) -> np.ndarray:
    """マスクを前後 samples 分だけ広げる。センサ間の時間ずれを許容するため。"""
    if samples <= 0 or not mask.any():
        return mask
    return np.convolve(mask.astype(float), np.ones(2 * samples + 1), mode="same") > 0


def wheel_speed_excess(
    ws_spread_mps: np.ndarray, yaw_rate_dps: np.ndarray, track_width_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """旋回で説明できるぶんを差し引いた輪速のばらつきを返す。

    定常旋回では外輪が内輪より速く回る。その差はトレッド幅で決まり、
        左右差 = ヨーレート [rad/s] × トレッド幅
    になる。実測の当てはめでも RAV4 2017 で 1.636 m、RAV4 TSS2 で 1.637 m と、
    ほぼ同じ値が出た (残差の標準偏差 0.03 m/s)。

    さらに、区間ごとの中央値を引く。タイヤの摩耗や空気圧の差で
    輪速には車両ごとの常時ずれがあり、実測でも中央値が
    0.066 m/s (2017) と 0.262 m/s (TSS2) で 4 倍違った。
    絶対値で閾値を置くと車両ごとの較正差を拾ってしまう。

    戻り値は (超過分, 旋回で説明できるぶん)。
    """
    expected = np.abs(np.deg2rad(yaw_rate_dps)) * track_width_m
    excess = ws_spread_mps - expected
    finite = np.isfinite(excess)
    base = float(np.median(excess[finite])) if finite.any() else 0.0
    return excess - base, expected


def find_wheel_speed_anomalies(
    t: np.ndarray,
    excess: np.ndarray,
    v_mps: np.ndarray,
    rate_hz: float,
    cfg: dict[str, Any],
    ax_mps2: np.ndarray | None = None,
    yaw_residual_sigma: np.ndarray | None = None,
    abs_flag: np.ndarray | None = None,
) -> np.ndarray:
    """旋回で説明できない輪速のばらつきを、裏付けを取ったうえで拾う。

    輪速のばらつきは単独ではスリップと判定できない。加減速では駆動輪・制動輪が
    正常にも滑る。そこで「旋回で説明できる量を超えていること」に加えて、
    次のいずれかが同じころに立っていることを求める。

      加減速が大きい          駆動・制動によるスリップ
      ヨーが舵に従っていない   車両が滑っている
      ABS が作動している      車輪ロックの直接の証拠

    どれも無ければ、センサの欠損やドロップアウトと区別できないので拾わない。
    """
    ws = cfg["wheel_speed"]
    win = window_samples(ws["tolerance_s"], rate_hz)
    over = np.isfinite(excess) & (excess > ws["min_excess_mps"])
    over &= np.isfinite(v_mps) & (v_mps > ws["min_speed_mps"])
    if not over.any():
        return np.zeros(len(t))

    support = np.zeros(len(t), dtype=bool)
    if ax_mps2 is not None:
        support |= np.isfinite(ax_mps2) & (np.abs(ax_mps2) > ws["corroborate_ax_mps2"])
    if yaw_residual_sigma is not None:
        support |= np.isfinite(yaw_residual_sigma) & (
            np.abs(yaw_residual_sigma) > ws["corroborate_yaw_sigma"]
        )
    if abs_flag is not None:
        support |= np.isfinite(abs_flag) & (abs_flag > 0.5)

    active = over & dilate_mask(support, win)
    return active.astype(float)


# ---------------------------------------------------------------------------
# アクセル急 OFF からの制動 (ペダル信号を使う版)
# ---------------------------------------------------------------------------
def find_pedal_panic_brakes(
    t: np.ndarray,
    gas_pct: np.ndarray,
    ax_mps2: np.ndarray,
    rate_hz: float,
    cfg: dict[str, Any],
    brake_level: np.ndarray | None = None,
    brake_pressed: np.ndarray | None = None,
) -> np.ndarray:
    """アクセルを急に戻し、短時間でブレーキと強い減速が続く形を拾う。

    縦加速度だけで代用していた版は、前方に何も無いところでの遅れた制動も拾った
    (Chunk_1 の目視判定で確認)。アクセル開度が読めるなら、
    「踏んでいた → 離した → 制動」という運転者の動作そのものを見られる。

    ブレーキ量は単位が確定していない (DBC の注記も "seems prop to pedal force")
    ので、絶対値ではなくアクセルを離した時点からの立ち上がり量で見る。
    """
    pb = cfg["panic_brake"]["pedal"]
    n = len(t)
    out = np.zeros(n)
    if n == 0:
        return out

    rel_w = window_samples(pb["release_window_s"], rate_hz)
    react_w = window_samples(pb["reaction_window_s"], rate_hz)

    # アクセルを離した瞬間: 直前 release_window の間に踏んでいて、今は離している
    was_pressed = rolling(gas_pct, rel_w, "max")
    released = (
        np.isfinite(gas_pct)
        & (gas_pct < pb["released_pct"])
        & np.isfinite(was_pressed)
        & (was_pressed >= pb["min_gas_pct"])
    )
    # 立ち上がりだけを取る (離している間ずっとではなく、離した瞬間)
    edges = np.flatnonzero(released & ~np.r_[False, released[:-1]])

    for i in edges:
        j = min(i + react_w, n - 1)
        if j <= i:
            continue
        ax_min = np.nanmin(ax_mps2[i : j + 1]) if np.isfinite(ax_mps2[i : j + 1]).any() else np.nan
        if not np.isfinite(ax_min) or ax_min > pb["brake_threshold"]:
            continue

        braked = False
        if brake_level is not None and np.isfinite(brake_level[i]):
            seg = brake_level[i : j + 1]
            if np.isfinite(seg).any():
                braked = (np.nanmax(seg) - brake_level[i]) >= pb["min_brake_rise"]
        if not braked and brake_pressed is not None:
            seg = brake_pressed[i : j + 1]
            braked = bool(np.isfinite(seg).any() and np.nanmax(seg) > 0.5)
        if braked:
            out[i : j + 1] = 1.0
    return out


# ---------------------------------------------------------------------------
# 先行車の抽出と車間指標
# ---------------------------------------------------------------------------
def path_lateral_offset(
    distance_m: np.ndarray,
    ego_v: np.ndarray,
    ego_yaw_dps: np.ndarray,
    min_speed: float,
) -> np.ndarray:
    """自車の予測進路が、前方 distance_m の地点で横に何 m ずれるかを返す。

    レーダは車両固定座標で報告するので、正面に取った帯は直進を仮定している。
    旋回中はその帯が路側を掃き、標識やガードレールを先行車として拾う。
    自車のヨーレートから曲率 κ = ヨーレート / 車速 を出し、
    円弧の近似 y = κ x² / 2 で進路のずれを求めて帯を曲げる。

    低速では曲率が発散するので補正しない。
    """
    out = np.zeros(distance_m.shape)
    ok = np.isfinite(ego_v) & np.isfinite(ego_yaw_dps) & (ego_v >= min_speed)
    kappa = np.zeros(distance_m.shape)
    kappa[ok] = np.deg2rad(ego_yaw_dps[ok]) / ego_v[ok]
    out[ok] = 0.5 * kappa[ok] * distance_m[ok] ** 2
    return out


def lead_from_tracks(
    grid: np.ndarray,
    track_t: np.ndarray,
    distance_m: np.ndarray,
    lateral_m: np.ndarray,
    vrel_mps: np.ndarray,
    cfg: dict[str, Any],
    ego_v: np.ndarray | None = None,
    ego_yaw_dps: np.ndarray | None = None,
    track_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """自車レーン内で最も近い「動いている」トラックを先行車とする。

    戻り値は (距離, 相対速度, 先行車の絶対速度, トラック ID)。
    トラック ID は割り込みの検出に使う。先行車が別の車両に入れ替わったことは、
    ID の変化でしか分からない。

    静止物を外すのは、路側の標識やガードレールが先行車として拾われるため。
    静止物は相対速度が -自車速 になるので TTC = 距離 / 自車速 となり、
    実測では low_ttc の検出 25 件すべてがこれだった。先行車の絶対速度
    (自車速 + 相対速度) で足切りすると、実在の先行車 17 件は全数残る。
    """
    lead_d = np.full(grid.shape, np.nan)
    lead_v = np.full(grid.shape, np.nan)
    lead_target = np.full(grid.shape, np.nan)
    lead_id = np.full(grid.shape, np.nan)
    if grid.size == 0 or track_t.size == 0:
        return lead_d, lead_v, lead_target, lead_id
    ids = np.full(track_t.shape, np.nan) if track_id is None else track_id.astype(float)

    # 自車の状態をトラックの観測時刻へ移す
    if ego_v is not None:
        v_at = np.interp(track_t, grid, np.nan_to_num(ego_v, nan=0.0))
    else:
        v_at = np.full(track_t.shape, np.nan)
    if ego_yaw_dps is not None:
        yaw_at = np.interp(track_t, grid, np.nan_to_num(ego_yaw_dps, nan=0.0))
    else:
        yaw_at = np.full(track_t.shape, np.nan)

    # 進路の曲がりぶんだけ帯をずらす
    if cfg.get("curvature_compensation", False):
        y_path = path_lateral_offset(
            distance_m, v_at, yaw_at, float(cfg.get("curvature_min_speed_mps", 3.0))
        )
    else:
        y_path = np.zeros(distance_m.shape)

    target_speed = v_at + vrel_mps

    ok = (
        np.isfinite(distance_m)
        & np.isfinite(lateral_m)
        & np.isfinite(vrel_mps)
        & (np.abs(lateral_m - y_path) <= cfg["lane_half_width_m"])
        & (distance_m >= cfg["min_distance_m"])
        & (distance_m <= cfg["max_distance_m"])
    )
    floor = np.maximum(
        float(cfg.get("min_target_speed_mps", 0.0)),
        float(cfg.get("min_target_speed_ratio", 0.0)) * v_at,
    )
    ok &= np.isfinite(target_speed) & (target_speed >= floor)
    if not ok.any():
        return lead_d, lead_v, lead_target, lead_id

    step = grid[1] - grid[0] if grid.size > 1 else 1.0
    bins = np.floor((track_t[ok] - grid[0]) / step).astype(np.int64)
    d = distance_m[ok]
    v = vrel_mps[ok]
    ts = target_speed[ok]
    tid = ids[ok]
    inside = (bins >= 0) & (bins < grid.size)
    bins, d, v, ts, tid = bins[inside], d[inside], v[inside], ts[inside], tid[inside]
    if bins.size == 0:
        return lead_d, lead_v, lead_target, lead_id

    # 各ビンで最小距離のトラックを残す
    order = np.lexsort((d, bins))
    b_sorted = bins[order]
    first = np.r_[True, b_sorted[1:] != b_sorted[:-1]]
    lead_d[b_sorted[first]] = d[order][first]
    lead_v[b_sorted[first]] = v[order][first]
    lead_target[b_sorted[first]] = ts[order][first]
    lead_id[b_sorted[first]] = tid[order][first]

    # 観測が飛んだ区間は直近値を max_hold_s まで保持する
    hold = window_samples(cfg["max_hold_s"], 1.0 / step)
    return (
        _hold_forward(lead_d, hold),
        _hold_forward(lead_v, hold),
        _hold_forward(lead_target, hold),
        _hold_forward(lead_id, hold),
    )


def _hold_forward(x: np.ndarray, max_hold: int) -> np.ndarray:
    """直近の有効値を最大 max_hold サンプルまで保持する。"""
    out = x.copy()
    last_val, age = np.nan, max_hold + 1
    for i, xi in enumerate(x):
        if np.isfinite(xi):
            last_val, age = xi, 0
        else:
            age += 1
            out[i] = last_val if age <= max_hold else np.nan
    return out


def time_headway(distance_m: np.ndarray, ego_v: np.ndarray, min_ego_speed: float) -> np.ndarray:
    """車間時間 [s]。低速では意味を持たないので閾値未満は算出しない。"""
    out = np.full(distance_m.shape, np.nan)
    ok = np.isfinite(distance_m) & np.isfinite(ego_v) & (ego_v >= min_ego_speed)
    out[ok] = distance_m[ok] / ego_v[ok]
    return out


def time_to_collision(
    distance_m: np.ndarray,
    vrel_mps: np.ndarray,
    ego_v: np.ndarray | None = None,
    min_ego_speed: float = 0.0,
) -> np.ndarray:
    """衝突余裕時間 [s]。相対速度が負 (接近) のときだけ定義する。

    離れていく先行車に TTC は無いので NaN のままにする。閾値判定では
    NaN が発火しないため、これで「接近していない」ことが表現できる。

    自車が止まっている間も除く。停止中に相対速度が負になるのは
    「相手がこちらへ近づいている」状態であって、自車側のヒヤリハットではない。
    車間時間と同じ下限速度を使い、2 つの指標で扱いを揃える。
    """
    out = np.full(distance_m.shape, np.nan)
    ok = np.isfinite(distance_m) & np.isfinite(vrel_mps) & (vrel_mps < 0)
    if ego_v is not None and min_ego_speed > 0:
        ok &= np.isfinite(ego_v) & (ego_v >= min_ego_speed)
    out[ok] = distance_m[ok] / (-vrel_mps[ok])
    return out


# ---------------------------------------------------------------------------
# 特徴量の組み立て
# ---------------------------------------------------------------------------
def compute_features(
    gs: GriddedSignals,
    cfg: dict[str, Any],
    radar: Any | None = None,
    vehicle: Any | None = None,
) -> GriddedSignals:
    """グリッド上の信号から特徴量列を足した GriddedSignals を返す。

    元の列は消さない。どの生信号から出た特徴量か追えるようにしておく。
    """
    df = gs.df.copy()
    t = df["t"].to_numpy()
    dt = gs.dt
    sm = cfg["smoothing"]

    # --- 縦方向 -------------------------------------------------------------
    if "speed_mps" in df:
        v = moving_average(df["speed_mps"].to_numpy(), window_samples(sm["speed_window_s"], gs.rate_hz))
        df["v_mps"] = v
        ax = derivative(v, dt)
        df["ax_mps2"] = moving_average(ax, window_samples(sm["accel_window_s"], gs.rate_hz))
        # 縦躍度は微分 2 回でノイズが乗る (Chunk_1 実測で 1 サンプルごとの符号反転率 23.9%)。
        # 単独のイベント判定には使わず、移動窓で均した補助特徴量として持つ。
        df["jerk_mps3"] = derivative(df["ax_mps2"].to_numpy(), dt)
        aux_win = window_samples(cfg["auxiliary"]["window_s"], gs.rate_hz)
        df["jerk_win_min"] = rolling(df["jerk_mps3"].to_numpy(), aux_win, "min")
        df["jerk_win_mean"] = rolling(df["jerk_mps3"].to_numpy(), aux_win, "mean")

    # CAN の縦加速度は突き合わせ用。主系列との差を残して整合を見る。
    if "accel_x" in df:
        df["ax_can_mps2"] = df["accel_x"]
        if "ax_mps2" in df:
            df["ax_residual_mps2"] = df["ax_can_mps2"] - df["ax_mps2"]

    # --- 横方向 -------------------------------------------------------------
    if "steer_deg" in df:
        steer = moving_average(df["steer_deg"].to_numpy(), window_samples(sm["steer_window_s"], gs.rate_hz))
        df["steer_deg_s"] = steer
        df["steer_rate_dps"] = derivative(steer, dt)

        wv = cfg["weaving"]
        detrended = steer - moving_average(steer, window_samples(wv["detrend_window_s"], gs.rate_hz))
        df["steer_detrended_deg"] = detrended
        df["steer_reversals"] = count_reversals(t, detrended, wv["amplitude_deg"], wv["count_window_s"])

    if "yaw_rate" in df:
        df["yaw_rate_dps"] = df["yaw_rate"]
        if "v_mps" in df:
            # 運動学から出る横加速度。YAW_RATE のスケールは global_pose を基準に
            # 検証済み (回帰係数 0.988) なので、これを横方向の主系列にする。
            # 車速との積なので、同じ舵角レートでも低速では小さく出る。
            ay_kin = df["v_mps"].to_numpy() * np.deg2rad(df["yaw_rate"].to_numpy())
            df["ay_kin_mps2"] = moving_average(
                ay_kin, window_samples(sm["accel_window_s"], gs.rate_hz)
            )
            # 横躍度。縦躍度と同じ理由で単独判定には使わず、補助特徴量として持つ。
            df["lat_jerk_mps3"] = derivative(df["ay_kin_mps2"].to_numpy(), dt)
            aux_win = window_samples(cfg["auxiliary"]["window_s"], gs.rate_hz)
            df["lat_jerk_win_absmax"] = rolling(df["lat_jerk_mps3"].to_numpy(), aux_win, "absmax")
    if "accel_y" in df:
        df["ay_can_mps2"] = df["accel_y"]

    # --- 車両物理モデル -----------------------------------------------------
    # 舵角から期待されるヨーレートと実測の差を見る。滑りや制御介入があれば
    # 差が開く。諸元が未検証の車種では列を作らない。
    if vehicle is not None and all(c in df for c in ("v_mps", "steer_deg_s", "yaw_rate_dps")):
        sr = vehicle.geometry_value("steer_ratio")
        wb = vehicle.geometry_value("wheelbase_m")
        if sr is not None and wb is not None:
            expected = bicycle_yaw_rate(
                df["v_mps"].to_numpy(), df["steer_deg_s"].to_numpy(), wb, sr,
                vehicle.geometry_value("understeer_gradient", 0.0) or 0.0,
                vehicle.geometry_value("steer_offset_deg", 0.0) or 0.0,
                cfg["physics"]["min_speed_mps"],
            )
            df["yaw_expected_dps"] = expected
            df["yaw_residual_dps"] = df["yaw_rate_dps"].to_numpy() - expected
            std = vehicle.geometry_value("yaw_residual_std_dps") or 1.0
            # 標準偏差で割った値。閾値をσ単位で置けるようにする
            df["yaw_residual_sigma"] = df["yaw_residual_dps"] / std
            df["counter_steer_active"] = counter_steer(
                df["yaw_rate_dps"].to_numpy(), df["steer_rate_dps"].to_numpy(),
                cfg["physics"]["counter_steer_min_yaw_dps"],
                cfg["physics"]["counter_steer_min_rate_dps"],
            )
    # 定常単軌道モデルから出す横滑り角。横加速度センサが要る。
    if vehicle is not None and all(c in df for c in ("v_mps", "yaw_rate_dps", "ay_can_mps2")):
        k = vehicle.sideslip_ay_coeff()
        l_r = vehicle.center_to_rear_m()
        if k is not None and l_r is not None:
            df["beta_model_deg"] = sideslip_model_deg(
                df["v_mps"].to_numpy(), df["yaw_rate_dps"].to_numpy(),
                df["ay_can_mps2"].to_numpy(), l_r, k, cfg["physics"]["min_speed_mps"],
            )
            # 舵で期待される横滑り角と、その差。差が「舵で説明できない横滑り」。
            if "yaw_expected_dps" in df:
                df["beta_expected_deg"] = sideslip_expected_deg(
                    df["v_mps"].to_numpy(), df["yaw_expected_dps"].to_numpy(),
                    l_r, k, cfg["physics"]["min_speed_mps"],
                )
                df["beta_excess_deg"] = df["beta_model_deg"] - df["beta_expected_deg"]
            # beta の変化率。急に横滑りが立ち上がる過渡を捉える。
            # beta は適用範囲の外で NaN なので、そこは NaN のまま残る。
            df["beta_rate_dps"] = moving_average(
                derivative(df["beta_model_deg"].to_numpy(), dt),
                window_samples(cfg["auxiliary"]["window_s"], gs.rate_hz),
            )
            # センサ雑音で割った値。低速ほど beta の雑音が大きいので、
            # deg で固定した閾値では速度域ごとに厳しさが変わってしまう。
            s_yaw = vehicle.geometry_value("yaw_rate_noise_dps")
            s_ay = vehicle.geometry_value("accel_y_noise_mps2")
            if s_yaw is not None and s_ay is not None:
                noise = sideslip_noise_deg(df["v_mps"].to_numpy(), l_r, k, s_yaw, s_ay)
                df["beta_noise_deg"] = noise
                df["beta_sigma"] = df["beta_model_deg"] / noise

    # 独立した横加速度センサとの差。片方だけがおかしくなれば開く
    if "ay_can_mps2" in df and "ay_kin_mps2" in df:
        df["ay_residual_mps2"] = df["ay_can_mps2"] - df["ay_kin_mps2"]

    # --- 旋回の除外に使う正味方位角 -----------------------------------------
    if "yaw_rate_dps" in df:
        df["net_heading_win_deg"] = rolling_net_heading(
            t, df["yaw_rate_dps"].to_numpy(), cfg["turn"]["half_window_s"]
        )

    # --- 蛇行 ---------------------------------------------------------------
    need = ("yaw_rate_dps", "ay_kin_mps2", "steer_rate_dps", "v_mps")
    if all(c in df for c in need):
        weaves = find_weaving(
            t, df["v_mps"].to_numpy(), df["yaw_rate_dps"].to_numpy(),
            df["ay_kin_mps2"].to_numpy(), df["steer_rate_dps"].to_numpy(), gs.rate_hz, cfg,
        )
        active = np.zeros(len(df))
        count = np.full(len(df), np.nan)
        for a, b, n, _net in weaves:
            active[a : b + 1] = 1.0
            count[a : b + 1] = n
        df["weave_active"] = active
        df["weave_reversals"] = count
        gs.meta["weaves"] = len(weaves)

    # --- 回避 (出て戻る S 字) -----------------------------------------------
    if all(c in df for c in ("yaw_rate_dps", "ay_kin_mps2", "v_mps")):
        evasions = find_s_evasions(
            t, df["v_mps"].to_numpy(), df["yaw_rate_dps"].to_numpy(),
            df["ay_kin_mps2"].to_numpy(), gs.rate_hz, cfg,
        )
        active = np.zeros(len(df))
        exc = np.full(len(df), np.nan)
        for a, b, excursion, _final in evasions:
            active[a : b + 1] = 1.0
            exc[a : b + 1] = excursion
        df["s_evasion_active"] = active
        df["s_evasion_excursion_m"] = exc
        gs.meta["s_evasions"] = len(evasions)

    # --- アクセル急 OFF からの強い制動 ---------------------------------------
    # 「踏んでいた足を離して即座に強く踏む」形。アクセル開度は復号できていないので、
    # 縦加速度が正から強い負へ短時間で切り替わることで代用する。
    if "ax_mps2" in df:
        pb = cfg["panic_brake"]
        ax = df["ax_mps2"].to_numpy()
        look = window_samples(pb["lookback_s"], gs.rate_hz)
        was_accel = rolling(ax, look * 2 + 1, "max")
        # rolling は中心合わせなので、後ろ半分だけを見るためにずらす
        was_accel = np.concatenate([np.full(look, np.nan), was_accel[:-look]]) if look else was_accel
        active = np.where(
            np.isfinite(ax) & np.isfinite(was_accel)
            & (ax <= pb["brake_threshold"]) & (was_accel >= pb["accel_threshold"]),
            1.0, 0.0,
        )
        df["panic_brake_active"] = active

    # --- 車線変更の候補 -----------------------------------------------------
    if "yaw_rate_dps" in df and "v_mps" in df:
        changes = find_lane_changes(
            t, df["v_mps"].to_numpy(), df["yaw_rate_dps"].to_numpy(), gs.rate_hz, cfg
        )
        active = np.zeros(len(df))
        offset = np.full(len(df), np.nan)
        amp = np.full(len(df), np.nan)
        for c in changes:
            sl = slice(c.i_start, c.i_end + 1)
            active[sl] = 1.0
            offset[sl] = c.offset_m
            amp[sl] = c.heading_amp_deg
        df["lc_active"] = active
        df["lc_offset_m"] = offset
        df["lc_heading_amp_deg"] = amp
        gs.meta["lane_changes"] = len(changes)

    # --- 輪速のばらつき -----------------------------------------------------
    # 加減速時には駆動・制動で正常にも開く。これ単独ではスリップと判定できない。
    # 縦加速度・ヨー/操舵・ABS 等との整合を見る前提の補助特徴量として持つ。
    have_ws = [c for c in WHEEL_SPEED_COLUMNS if c in df]
    if len(have_ws) == len(WHEEL_SPEED_COLUMNS):
        ws = df[list(WHEEL_SPEED_COLUMNS)].to_numpy()
        df["ws_spread_mps"] = np.nanmax(ws, axis=1) - np.nanmin(ws, axis=1)

        # 生のばらつきは 1 サンプルごとの高周波ノイズが支配的で、
        # 符号反転率が 62〜67% (白色雑音なら 50%) ある。平滑化しないまま閾値を置くと
        # ノイズの尖りを拾う。実測では 0.3 秒平滑化で超過の最大が
        # 1.31 -> 0.27 m/s に落ちた (Chunk_1 全体)。滑りは 1 サンプルでは終わらない。
        ws_win = window_samples(sm.get("wheel_speed_window_s", 0.3), gs.rate_hz)
        ws_s = np.stack(
            [moving_average(df[c].to_numpy(), ws_win) for c in WHEEL_SPEED_COLUMNS], axis=1
        )
        # 端は平滑化で全て NaN になる行があるので、そこは NaN のままにする
        all_nan = ~np.isfinite(ws_s).any(axis=1)
        spread_s = np.full(len(df), np.nan)
        if (~all_nan).any():
            good = ws_s[~all_nan]
            spread_s[~all_nan] = np.nanmax(good, axis=1) - np.nanmin(good, axis=1)
        df["ws_spread_smooth_mps"] = spread_s

        # 旋回で説明できるぶんを差し引く。トレッド幅は車種設定の実測値。
        track = vehicle.geometry_value("track_width_m") if vehicle is not None else None
        if track and "yaw_rate_dps" in df:
            excess, expected = wheel_speed_excess(
                df["ws_spread_smooth_mps"].to_numpy(), df["yaw_rate_dps"].to_numpy(), float(track)
            )
            df["ws_spread_expected_mps"] = expected
            df["ws_spread_excess_mps"] = excess
            if "v_mps" in df:
                df["ws_anomaly_active"] = find_wheel_speed_anomalies(
                    t, excess, df["v_mps"].to_numpy(), gs.rate_hz, cfg,
                    ax_mps2=df["ax_mps2"].to_numpy() if "ax_mps2" in df else None,
                    yaw_residual_sigma=(
                        df["yaw_residual_sigma"].to_numpy() if "yaw_residual_sigma" in df else None
                    ),
                    abs_flag=(
                        df["abs_active_flag"].to_numpy() if "abs_active_flag" in df else None
                    ),
                )

    # --- アクセル急 OFF からの制動 (ペダル信号が読める場合) -----------------
    if "gas_pedal_pct" in df and "ax_mps2" in df:
        level = None
        for col in ("brake_position", "brake_pressure"):
            if col in df:
                level = df[col].to_numpy()
                break
        df["panic_brake_pedal_active"] = find_pedal_panic_brakes(
            t, df["gas_pedal_pct"].to_numpy(), df["ax_mps2"].to_numpy(), gs.rate_hz, cfg,
            brake_level=level,
            brake_pressed=df["brake_pressed"].to_numpy() if "brake_pressed" in df else None,
        )

    # --- 先行車 -------------------------------------------------------------
    if radar is not None:
        ego = df["v_mps"].to_numpy() if "v_mps" in df else None
        yaw = df["yaw_rate_dps"].to_numpy() if "yaw_rate_dps" in df else None
        lead_d, lead_v, lead_target, lead_id = lead_from_tracks(
            t, radar.t, radar.distance_m, radar.lateral_m, radar.vrel_mps, cfg["lead"],
            ego_v=ego, ego_yaw_dps=yaw, track_id=radar.track_id,
        )
        df["lead_distance_m"] = lead_d
        df["lead_vrel_mps"] = lead_v
        df["lead_target_speed_mps"] = lead_target
        df["lead_track_id"] = lead_id
        ego = df["v_mps"].to_numpy() if "v_mps" in df else np.full(t.shape, np.nan)
        df["thw_s"] = time_headway(lead_d, ego, cfg["lead"]["min_ego_speed_mps"])
        df["ttc_s"] = time_to_collision(
            lead_d, lead_v, ego, cfg["lead"]["min_ego_speed_mps"]
        )

        # --- 同一の先行車への急接近 (割り込みではない詰まり方) ----------------
        thw = df["thw_s"].to_numpy()
        d_thw = derivative(thw, dt)
        same_lead = np.zeros(len(df), dtype=bool)
        ids = np.nan_to_num(lead_id, nan=-1.0)
        same_lead[1:] = ids[1:] == ids[:-1]
        cl = cfg["closing"]
        df["thw_rate_s_per_s"] = d_thw
        df["closing_fast_active"] = np.where(
            np.isfinite(d_thw) & same_lead
            & (d_thw <= cl["max_thw_rate"]) & (thw <= cl["max_thw_s"]),
            1.0, 0.0,
        )

        # --- 割り込み -------------------------------------------------------
        if ego is not None:
            cut_ins = find_cut_ins(
                t, lead_id, lead_d, df["thw_s"].to_numpy(), lead_target,
                ego, gs.rate_hz, cfg,
            )
            active = np.zeros(len(df))
            drop = np.full(len(df), np.nan)
            thw_after = np.full(len(df), np.nan)
            span = window_samples(cfg["cut_in"]["window_s"], gs.rate_hz)
            for c in cut_ins:
                sl = slice(c.i_switch, min(c.i_switch + span, len(df)))
                active[sl] = 1.0
                if np.isfinite(c.distance_before_m):
                    drop[sl] = c.distance_before_m - c.distance_after_m
                thw_after[sl] = c.thw_after_s
            df["cut_in_active"] = active
            df["cut_in_distance_drop_m"] = drop
            df["cut_in_thw_after_s"] = thw_after
            gs.meta["cut_ins"] = len(cut_ins)

    return GriddedSignals(
        df=df,
        rate_hz=gs.rate_hz,
        segment_id=gs.segment_id,
        drive_id=gs.drive_id,
        vehicle=gs.vehicle,
        raw_can_loaded=gs.raw_can_loaded,
        meta=gs.meta,
    )


def feature_summary(gs: GriddedSignals) -> pd.DataFrame:
    """各特徴量の有効サンプル率と範囲。中間結果の点検用。"""
    rows = []
    for col in gs.df.columns:
        if col == "t":
            continue
        x = gs.df[col].to_numpy()
        finite = np.isfinite(x)
        rows.append(
            {
                "feature": col,
                "coverage": float(finite.mean()),
                "min": float(np.nanmin(x)) if finite.any() else np.nan,
                "max": float(np.nanmax(x)) if finite.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
