"""横滑り候補の 2 段抽出 (L3)。

大量の CAN データから横滑り候補を絞り込むための 2 段構成。

  1 次フィルタ (粗ふるい)
      サンプル単位の安い判定だけで通常走行の大半を落とす。
      「横滑りかどうか」は判定しない。取りこぼさないことだけを見る。

  2 次フィルタ (物理的妥当性)
      1 次を通った区間について、横滑りとして物理的に筋が通るかを見る。
      通常の旋回・幾何で説明できるもの、信号がおかしいものをここで落とす。

  最終候補
      2 次を通った区間を前後に広げてまとめ、確認する単位にする。

判定に使う量はすべて features.py が作った列で、閾値はすべて設定から渡る。
どの条件で通り、どの条件で落ちたかを候補ごとに残す。

なぜ 2 段にするか
-----------------
beta_model_deg は「線形タイヤを仮定したときの定常横滑り角」であって、
横滑りの実測ではない。値が大きいことは、それだけでは異常を意味しない。

    beta = l_r * r / v - k * a_y

第 1 項は旋回半径だけで決まる幾何的な量で、低速の小回りでは正常に数度になる
(実測で 14〜36 km/h の交差点右左折が 4〜15 deg)。第 2 項はタイヤの横力による
コンプライアンスで、こちらは通常走行では 1 deg 未満にしかならない。
つまり「大きい beta」の大半は正常な幾何であって、危険挙動ではない。

そこで
  * 1 次は beta の大きさ (センサ雑音で正規化した量) だけで粗く拾い、
  * 2 次で「幾何と舵で説明できるぶん」を差し引いて残るかどうかを見る、
という分け方にしている。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd

from .detectors import _merge_runs, _runs, dilate
from .signals import GriddedSignals, window_samples

log = logging.getLogger(__name__)

# 1 次フィルタの通過理由
REASON_BETA = "beta"          # beta の大きさ
REASON_LATDYN = "lat_dynamics"  # 横速度の変化率 (beta が効かない速度帯の保険)
REASON_FLAG = "flag"          # VSC / ABS の作動そのもの

# 2 次フィルタの必須条件。どれか 1 つでも欠けたら候補にしない。
#   ここに置くのは「測定として意味があるか」を見るものだけ。
#   「横滑りらしいか」の判断は信頼度 (CONFIDENCE) の側で行う。
REQUIRED = (
    ("duration", "持続時間"),
    ("signal_sane", "横加速度 2 系統の整合"),
)

# 信頼度の項目。満たした項目の重みを足し、min_score 以上なら通す。
#
# 必須の AND にしなかった理由 (KIT MSDM の実測 β との突き合わせによる。
# docs/sideslip_filter.md §4):
#   lateral_force  実測の横滑りのうち 9.3% が閾値に届かない。一方 Toyota の
#                  1 次通過サンプルでは 100% が満たしており、絞り込みに効いていない。
#                  再現率だけを削る条件になっていた。
#   unexplained    定常のドリフト中は舵と車両の応答がつり合ってしまい、
#                  実測の横滑りでも beta_excess が 0 付近に落ちる。
#                  過渡では効く (|dβ/dt| が 25〜50 deg/s の帯で 97.5%) が、
#                  定常では 80.5% までしか出ない。
CONFIDENCE = (
    ("strong_beta", "beta が大きい"),
    ("unexplained", "舵で説明できない"),
    ("lateral_force", "横力の裏付け"),
    ("transient", "急な立ち上がり"),
    ("corroborated", "別系統の裏付け"),
)

# 表示用。必須と信頼度をまとめて引ける。
CHECKS = REQUIRED + CONFIDENCE


@dataclass
class SideslipCandidate:
    """確認する単位。2 次を通った区間を前後に広げてまとめたもの。"""

    t_start: float
    t_end: float
    duration_s: float
    t_peak: float
    beta_peak_deg: float
    beta_sigma_peak: float
    beta_excess_peak_deg: float
    yaw_sigma_peak: float
    ay_peak_mps2: float
    beta_rate_peak_dps: float
    ax_at_peak_mps2: float
    v_at_peak_mps: float
    n_runs: int
    confidence: float
    confidence_items: str
    grade: str
    reasons: str
    corroboration: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilterCounts:
    """各段で何サンプル残ったか。絞り込みの効き方を追えるようにする。"""

    n_samples: int = 0
    n_hours: float = 0.0
    n_beta_valid: int = 0          # beta が計算できたサンプル
    n_in_range: int = 0            # 適用範囲 (速度) に入ったサンプル
    n_stage1: int = 0
    n_stage2: int = 0
    n_runs_stage1: int = 0
    n_runs_stage2: int = 0
    n_candidates: int = 0
    stage1_by_reason: dict[str, int] = field(default_factory=dict)
    stage2_reject: dict[str, int] = field(default_factory=dict)

    def add(self, other: "FilterCounts") -> None:
        for k, v in asdict(other).items():
            cur = getattr(self, k)
            if isinstance(cur, dict):
                for kk, vv in v.items():
                    cur[kk] = cur.get(kk, 0) + vv
            else:
                setattr(self, k, cur + v)


def _peak(x: np.ndarray) -> float:
    """絶対値が最大の値を符号付きで返す。空・全欠測なら NaN。"""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return float("nan")
    idx = np.flatnonzero(ok)
    return float(x[idx[np.argmax(np.abs(x[idx]))]])


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """欠けている列は全 NaN として扱う。列の有無で分岐を散らかさない。"""
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def _abs_ge(x: np.ndarray, th: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(x), np.abs(x) >= th, False)


# ---------------------------------------------------------------------------
# 1 次フィルタ
# ---------------------------------------------------------------------------
def stage1_mask(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """通常走行の大半を落とす粗ふるい。

    適用範囲 (速度) に入っていることを前提に、次のどれかが立てば通す。

      beta         : beta がセンサ雑音に対して十分大きい
      lat_dynamics : 横加速度が v*yaw から離れている (= 横速度が変化している)
      flag         : VSC / ABS が作動している (beta によらない直接の証拠)

    beta の判定は deg の下限と sigma の下限の両方を課す。
    sigma だけだと高速域で 0.1 deg 程度でも通ってしまい、
    deg だけだと低速域が雑音で埋まる。

    lat_dynamics を別枝に置いてあるのは、beta が構造的に効かない速度帯が
    あるため。beta = r*(l_r/v - k*v) - k*(a_y - v*r) の第 1 項の係数は
    v = sqrt(l_r/k) でゼロになり、その付近では**どれだけ横滑りしていても**
    beta が 0 付近になる。TSS2 (v_c = 15.1 m/s) では、ヨーレート 20 deg/s の
    滑りが 43〜70 km/h の帯で |beta| < 1 deg に埋もれる。
    残る手掛かりは a_y - v*r (= 横速度の変化率) だけなので、これを直接見る。

    この枝の効果は解析から導いたもので、実測では確かめられていない。
    臨界速度で横滑りしているデータが手元に無いため (KIT の限界走行は
    最大 15.6 m/s で、この車両の v_c = 16.9 m/s に届かない)。
    公道側の代償だけは測ってあり、6.6 時間で 2.6 秒だった。
    """
    s = cfg["sideslip"]
    s1 = s["stage1"]
    v = _col(df, "v_mps")
    beta = _col(df, "beta_model_deg")

    in_range = np.isfinite(beta) & np.isfinite(v) & (v >= float(s["min_speed_mps"]))

    reasons: dict[str, np.ndarray] = {}
    hit_beta = _abs_ge(beta, float(s1["min_beta_deg"]))
    if "beta_sigma" in df.columns:
        hit_beta &= _abs_ge(_col(df, "beta_sigma"), float(s1["min_beta_sigma"]))
    reasons[REASON_BETA] = in_range & hit_beta

    th = s1.get("min_ay_residual_mps2")
    if th is not None:
        reasons[REASON_LATDYN] = in_range & _abs_ge(_col(df, "ay_residual_mps2"), float(th))

    if s1.get("include_direct_flags", True):
        flag = np.zeros(len(df), dtype=bool)
        for name in s1.get("direct_flags", ()) or ():
            flag |= _col(df, name) > 0.5
        reasons[REASON_FLAG] = in_range & flag

    mask = np.zeros(len(df), dtype=bool)
    for m in reasons.values():
        mask |= m
    return mask, reasons


# ---------------------------------------------------------------------------
# 2 次フィルタ
# ---------------------------------------------------------------------------
def _signal_sane(df: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """横加速度の 2 系統が整合しているサンプルを True にする。

    beta の入力は YAW_RATE と ACCEL_Y の 2 つ。ACCEL_Y が壊れていれば
    beta も壊れるが、beta 単体では気づけない。独立に作れる
    ay_kin (= v * yaw_rate) と符号・比を突き合わせる。

    KIT の実測では、重心横滑り角が 15 deg ある定常ドリフト中でも
    a_y と v*yaw_rate はほぼ重なった (docs/kit_msdm.md)。
    つまりこの検査は本物の横滑りを落とさない。

    横加速度が小さい区間では比が発散するので、その場合は検査しない
    (整合しているものとして扱う)。
    """
    sn = cfg["sideslip"]["stage2"]["signal_sanity"]
    ay_can = _col(df, "ay_can_mps2")
    ay_kin = _col(df, "ay_kin_mps2")
    small = ~(np.abs(ay_kin) >= float(sn["min_ay_mps2"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = ay_can / ay_kin
        ok = (
            np.isfinite(ratio)
            & (ay_can * ay_kin > 0)
            & (ratio >= float(sn["ratio_min"]))
            & (ratio <= float(sn["ratio_max"]))
        )
    # 片方でも欠測なら判定できない。判定できないものは通さない。
    known = np.isfinite(ay_can) & np.isfinite(ay_kin)
    return np.where(known, small | ok, False)


def confidence_masks(
    df: pd.DataFrame, cfg: dict[str, Any], rate_hz: float
) -> dict[str, np.ndarray]:
    """信頼度の各項目を、サンプル単位のマスクにして返す。

    別系統の信号は捉える時刻がずれるので、tolerance_s ぶん前後へ広げてから
    区間と突き合わせる。広げるのは各項目の側だけで、合計の取り方は緩めない。
    """
    s2 = cfg["sideslip"]["stage2"]
    c = s2["confidence"]
    tol = window_samples(float(s2["tolerance_s"]), rate_hz)

    out = {
        "strong_beta": _abs_ge(_col(df, "beta_model_deg"), float(c["strong_beta"]["threshold"])),
        "unexplained": _abs_ge(_col(df, "beta_excess_deg"), float(c["unexplained"]["threshold"])),
        "lateral_force": _abs_ge(_col(df, "ay_kin_mps2"), float(c["lateral_force"]["threshold"])),
        "transient": _abs_ge(_col(df, "beta_rate_dps"), float(c["transient"]["threshold"])),
    }
    flag = np.zeros(len(df), dtype=bool)
    for name in c["corroborated"].get("flags", ()) or ():
        flag |= _col(df, name) > 0.5
    out["corroborated"] = flag
    return {k: dilate(v, tol) for k, v in out.items()}


def stage2_runs(
    t: np.ndarray,
    df: pd.DataFrame,
    mask1: np.ndarray,
    cfg: dict[str, Any],
    rate_hz: float,
) -> tuple[list[tuple[int, int, dict[str, Any]]], dict[str, int]]:
    """1 次を通った区間ごとに、横滑りとして妥当かを検査する。

    2 段構えになっている。

      必須条件 (REQUIRED)
          「測定として意味があるか」だけを見る。欠けたら候補にしない。
      信頼度 (CONFIDENCE)
          「横滑りらしいか」を項目ごとに数え、重みの合計が min_score 以上なら通す。

    どれか 1 つの物理量を必須にすると、その量が効かない条件で本物を落とす。
    実測 beta のある KIT MSDM で測ったところ、

        横力の裏付け   : 実測の横滑りの 9.3% が閾値に届かない。低速・小半径の
                        滑りでは横加速度そのものが小さい。しかも Toyota 側では
                        1 次通過サンプルの 100% が満たしており、絞り込みに効かない
        舵で説明できない: 定常のドリフト中は舵と車両の応答がつり合い、実測の
                        横滑りでも beta_excess が 0 付近に落ちる (定常 80.5% /
                        過渡 97.5%)。一方 Toyota 側では 5.4% しか満たさず、
                        絞り込みには最も効く

    という非対称があったため、両方を信頼度側へ移した。
    戻り値は (通った区間, 落ちた理由の集計)。区間には得点と内訳を添える。
    """
    s2 = cfg["sideslip"]["stage2"]
    conf_cfg = s2["confidence"]
    min_score = float(conf_cfg["min_score"])
    min_dur = float(s2["min_duration_s"])
    min_sane = float(s2["signal_sanity"]["min_ratio"])

    m_sane = _signal_sane(df, cfg)
    conf = confidence_masks(df, cfg, rate_hz)

    runs = _merge_runs(_runs(mask1), t, float(cfg["sideslip"]["stage1"]["merge_gap_s"]))
    passed: list[tuple[int, int, dict[str, Any]]] = []
    reject: dict[str, int] = {}
    for a, b in runs:
        sl = slice(a, b + 1)
        required = {
            "duration": bool(t[b] - t[a] >= min_dur),
            # 整合は「区間の大半で成り立つ」ことを求める。1 サンプルの
            # 突発的なずれで本物を落とさないため。
            "signal_sane": bool(np.mean(m_sane[sl]) >= min_sane),
        }
        if not all(required.values()):
            for name, ok in required.items():
                if not ok:
                    reject[name] = reject.get(name, 0) + 1
            continue
        items = [name for name, _ in CONFIDENCE if conf[name][sl].any()]
        score = sum(float(conf_cfg[name]["weight"]) for name in items)
        if score >= min_score:
            passed.append((a, b, {"score": score, "items": items}))
        else:
            reject["confidence"] = reject.get("confidence", 0) + 1
    return passed, reject


# ---------------------------------------------------------------------------
# 最終候補
# ---------------------------------------------------------------------------
def _grade(beta_peak: float, cfg: dict[str, Any]) -> str:
    f = cfg["sideslip"]["final"]
    a = abs(beta_peak)
    if a >= float(f["grade_high_deg"]):
        return "A_大横滑り"
    if a >= float(f["grade_review_deg"]):
        return "B_要確認"
    return "C_弱い候補"


def _corroboration(df: pd.DataFrame, sl: slice, cfg: dict[str, Any]) -> str:
    """同じ区間に立っている、別系統の裏付けを並べる。判定には使わない。"""
    out = []
    for col in cfg["sideslip"]["final"].get("corroborate", ()) or ():
        x = _col(df, col)[sl]
        if np.isfinite(x).any() and np.nanmax(x) > 0.5:
            out.append(col)
    return "|".join(out)


def find_sideslip_candidates(
    gs: GriddedSignals, cfg: dict[str, Any]
) -> tuple[list[SideslipCandidate], FilterCounts]:
    """1 次 -> 2 次 -> 最終候補まで通す。"""
    df = gs.df
    t = df["t"].to_numpy()
    counts = FilterCounts(n_samples=len(df), n_hours=float(len(df)) / gs.rate_hz / 3600.0)
    if len(df) == 0 or "beta_model_deg" not in df.columns:
        return [], counts

    beta = _col(df, "beta_model_deg")
    v = _col(df, "v_mps")
    counts.n_beta_valid = int(np.isfinite(beta).sum())
    counts.n_in_range = int(
        (np.isfinite(beta) & (v >= float(cfg["sideslip"]["min_speed_mps"]))).sum()
    )

    mask1, reasons = stage1_mask(df, cfg)
    counts.stage1_by_reason = {k: int(m.sum()) for k, m in reasons.items()}
    # 細切れをつないでから数える。以降の段はこの区間を単位に判定するので、
    # ここで数えておかないと「2 次が 1 次より多い」という表になってしまう。
    runs1 = _merge_runs(_runs(mask1), t, float(cfg["sideslip"]["stage1"]["merge_gap_s"]))
    for a, b in runs1:
        mask1[a : b + 1] = True
    counts.n_stage1 = int(mask1.sum())
    counts.n_runs_stage1 = len(runs1)
    if not runs1:
        return [], counts

    passed, reject = stage2_runs(t, df, mask1, cfg, gs.rate_hz)
    counts.stage2_reject = reject
    counts.n_runs_stage2 = len(passed)
    counts.n_stage2 = int(sum(b - a + 1 for a, b, _ in passed))
    if not passed:
        return [], counts

    # 前後に余白を付けてから、近いものをひとつの確認単位にまとめる。
    fin = cfg["sideslip"]["final"]
    pad = window_samples(float(fin["window_pad_s"]), gs.rate_hz)
    padded = np.zeros(len(df), dtype=bool)
    for a, b, _ in passed:
        padded[max(0, a - pad) : min(len(df), b + pad + 1)] = True
    windows = _merge_runs(_runs(padded), t, float(fin["merge_gap_s"]))

    cands: list[SideslipCandidate] = []
    for a, b in windows:
        sl = slice(a, b + 1)
        inner = [(x, y, c) for x, y, c in passed if x <= b and y >= a]
        # 山の位置は 2 次を通ったサンプルの中から選ぶ。余白の側で
        # たまたま大きい値があっても、それは判定の根拠ではない。
        core = np.zeros(len(df), dtype=bool)
        for x, y, _ in inner:
            core[x : y + 1] = True
        idx = np.flatnonzero(core)
        pk = idx[np.nanargmax(np.abs(np.where(np.isfinite(beta[idx]), beta[idx], 0.0)))]
        beta_pk = float(beta[pk])
        cands.append(
            SideslipCandidate(
                t_start=float(t[a]),
                t_end=float(t[b]),
                duration_s=float(t[b] - t[a]),
                t_peak=float(t[pk]),
                beta_peak_deg=round(beta_pk, 3),
                beta_sigma_peak=round(float(_col(df, "beta_sigma")[pk]), 2),
                beta_excess_peak_deg=round(_peak(_col(df, "beta_excess_deg")[core]), 3),
                yaw_sigma_peak=round(_peak(_col(df, "yaw_residual_sigma")[core]), 2),
                ay_peak_mps2=round(_peak(_col(df, "ay_kin_mps2")[sl]), 3),
                beta_rate_peak_dps=round(_peak(_col(df, "beta_rate_dps")[core]), 2),
                ax_at_peak_mps2=round(float(_col(df, "ax_mps2")[pk]), 3),
                v_at_peak_mps=round(float(v[pk]), 2),
                n_runs=len(inner),
                # 信頼度は区間ごとに出るので、まとめた窓では最大のものを代表にする。
                confidence=max(float(c["score"]) for _x, _y, c in inner),
                confidence_items="|".join(
                    sorted({i for _x, _y, c in inner for i in c["items"]},
                           key=lambda k: [n for n, _ in CONFIDENCE].index(k))
                ),
                grade=_grade(beta_pk, cfg),
                reasons="|".join(k for k, m in reasons.items() if m[sl].any()),
                corroboration=_corroboration(df, sl, cfg),
            )
        )
    counts.n_candidates = len(cands)
    return cands, counts


def candidates_to_frame(
    cands: list[SideslipCandidate], config_hash: str = ""
) -> pd.DataFrame:
    if not cands:
        return pd.DataFrame()
    df = pd.DataFrame([c.as_dict() for c in cands])
    if config_hash:
        df["config_hash"] = config_hash
    return df.sort_values("beta_peak_deg", key=lambda s: s.abs(), ascending=False).reset_index(
        drop=True
    )
