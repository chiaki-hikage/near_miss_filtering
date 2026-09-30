"""スリップの直接証拠 (L3)。sideslip.py とは独立に動く。

sideslip.py はモデル推定の `beta_model_deg` を使う。こちらは**モデルを
一切通さず**、車両自身が「滑った」と判断した瞬間だけを拾う。

なぜ別の枝にするか
------------------
beta を使う限り取れない滑りがある。TSS2 の 33.3 時間で実測した内訳:

  1. **低速の駆動輪空転 (TRC 介入)**
     旋回加速中の空転は 4〜9 m/s で起きる。sideslip の適用下限
     `min_speed_mps: 10.0` は beta = l_r*r/v - k*a_y の第 1 項が 1/v で
     発散するために要るもので、**フラグには何の関係もない**。
     実測 2 件のうち 1 件は 1 次フィルタに一度も入らなかった。

  2. **直進制動での車輪ロック**
     車体の姿勢が乱れないので beta に出ない。減速度 -1.7 m/s^2 で ABS が
     作動した (= 低ミュー路が疑われる) 区間でも beta は 1.8 deg しかなかった。

証拠の扱い
----------
種類ごとに分けて保持する。まとめて 1 本のフラグにしない。

  制御作動 (DIRECT)   車両自身の判断。閾値を決める余地がない
      slip_warn     滑り警告灯。値 4 が実作動。3 はランプ試験、7 は継続点灯
      abs_active    ABS 作動
      vsc_active    VSC 作動
  物理挙動 (SUPPORT)  人が閾値を決めるので、**単独では確定にしない**
      wheel_spread  4 輪速の最大 - 最小
      wheel_decel   輪 1 本の瞬間減速度 (4 輪のうち最も深いもの)
      lateral       横加速度 (車速 x ヨーレート)
      yaw_residual  舵で説明できないヨー (諸元が要る。RAV4 TSS2 のみ)
  参考 (CONTEXT)      証拠に数えない。候補行に添えるだけ
      tc_disabled   TC 解除。「作動した」ではない

等級
----
**制御が作動しただけでは等級を上げない。作動に対応する物理挙動が同じ窓に
出ているかまで見る。** 作動の数だけで決めると、軽い介入 (ax 中央 -1.6 m/s^2、
輪速差 0.90 m/s = 対照窓 p95 相当) が上位に並ぶ。RAV4 では slip_warn と ABS が
90% 同時に立つため、作動の数は独立な証拠の数にならない。

  multi_evidence     **異なる物理メカニズム**の裏付けが 2 系統以上
                     ABS と slip_warn を同じ wheel_decel で裏付けても 1 系統。
                     輪速から出る量 (輪速差・輪単体の減速) と、ヨーと舵から出る量
                     (横加速度・舵で説明できないヨー) を別の系統として数える。
  high_evidence      裏付けは取れたが、系統は 1 つ
  activation_only    制御は作動したが、対応する物理挙動が確認できない
  physics_only       制御作動なし。物理挙動だけ

どの作動に何が対応するかは configs/detection.yaml の corroboration で決める。

  ABS       -> 輪単体の深い減速 (主) / 輪速の開き
               4 輪が同時にロックと解放を繰り返すので輪と輪の差には出にくい。
               実測でも ABS 事象 66 件のうち 35 件が輪速差 1.0 m/s 未満、
               そのうち 18 件は輪単体では -8 m/s^2 以下に落ちていた。
  slip_warn -> 輪速の開き (空転) / 輪単体の深い減速 (ロック) / 横加速度 /
               舵で説明できないヨー
  VSC       -> 横加速度 / 舵で説明できないヨー

候補行には各証拠の有無を個別の列で残すので、重なり方を後から数えられる。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .detectors import _merge_runs, _runs
from .signals import GriddedSignals, window_samples

# 制御作動。設定 direct_slip.flags のキーと対応する。
DIRECT = ("slip_warn", "abs_active", "vsc_active")
# 物理挙動。単独では確定にしない。
SUPPORT = ("wheel_spread", "wheel_decel", "lateral", "yaw_residual")
# 参考。証拠には数えない。
CONTEXT = ("tc_disabled",)

ALL_EVIDENCE = DIRECT + SUPPORT

# 確認単位を**作る**証拠。裏付けのうち wheel_decel / lateral / yaw_residual は入れない。
# 閾値が対照窓の p98 / p99 にあるので、単独で拾わせると通常走行が大量に並ぶ
# (300 セグメントの試しで wheel_decel 単独の事象が数十件出た)。
# これらは「作動があったときに、それが本物かを確かめる」ためだけに使う。
TRIGGER = DIRECT + ("wheel_spread",)

GRADE_MULTI = "multi_evidence"
GRADE_HIGH = "high_evidence"
GRADE_ACTIVATION = "activation_only"
GRADE_PHYSICS = "physics_only"

GRADE_ORDER = (GRADE_MULTI, GRADE_HIGH, GRADE_ACTIVATION, GRADE_PHYSICS)
GRADE_NOTE = {
    GRADE_MULTI: "異なる物理メカニズムの裏付けが 2 系統以上",
    GRADE_HIGH: "裏付けは取れたが、系統は 1 つ",
    GRADE_ACTIVATION: "制御は作動したが対応する物理挙動が確認できない",
    GRADE_PHYSICS: "制御作動なし。物理挙動だけ。目視の確認が要る",
}


@dataclass
class DirectSlipEvent:
    """確認する単位。証拠の有無を種類ごとに列で持つ。"""

    t_start: float
    t_end: float
    duration_s: float
    grade: str
    n_direct: int                 # 制御作動が何系統立ったか
    n_corroborated: int           # うち、対応する物理挙動まで揃ったのは何系統か
    n_mechanisms: int             # 裏付けが**異なる物理メカニズム**で何系統あるか
    evidence: str                 # 立った証拠を "|" でつないだもの
    corroboration: str            # "abs_active:wheel_decel" のように作動と裏付けの対応
    mechanisms: str               # 裏付けの系統を "|" でつないだもの
    # --- 証拠ごとの有無と強さ。まとめない ---
    # bool 側が判定の結果。*_peak は確認単位 (余白込み) の中の最大値で、
    # **条件を通ったかどうかとは別物**。たとえば輪速の開きが一瞬 3 m/s に
    # 達しても 0.3 秒続かなければ wheel_spread は False のまま、
    # ws_spread_peak_mps は 3 になる。判定は bool を見ること。
    slip_warn: bool
    slip_warn_peak: float
    abs_active: bool
    vsc_active: bool
    wheel_spread: bool
    ws_spread_peak_mps: float
    wheel_decel: bool
    wheel_decel_peak_mps2: float  # 4 輪のうち最も深い瞬間減速度
    lateral: bool
    yaw_residual: bool
    tc_disabled: bool             # 参考。等級には効かない
    # --- 状況 ---
    v_at_peak_mps: float
    v_max_mps: float
    ax_at_peak_mps2: float
    ay_at_peak_mps2: float
    yaw_rate_at_peak_dps: float
    steer_at_peak_deg: float
    beta_model_peak_deg: float    # sideslip との突き合わせ用。判定には使わない

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DirectSlipCounts:
    """証拠ごとの内訳。どの枝が効いているかを追えるようにする。"""

    n_samples: int = 0
    n_hours: float = 0.0
    n_events: int = 0
    # 信号そのものが有るか。無いデータセットと「有るが立たない」を区別する。
    present: dict[str, int] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)
    # 証拠ごとの区間数 (まとめる前)
    runs_by_evidence: dict[str, int] = field(default_factory=dict)
    # 等級別の件数
    by_grade: dict[str, int] = field(default_factory=dict)
    # 証拠の組み合わせ別の件数。重なり方をそのまま残す
    by_combination: dict[str, int] = field(default_factory=dict)

    def add(self, other: "DirectSlipCounts") -> None:
        for k, v in asdict(other).items():
            cur = getattr(self, k)
            if isinstance(cur, dict):
                for kk, vv in v.items():
                    cur[kk] = cur.get(kk, 0) + vv
            else:
                setattr(self, k, cur + v)


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def _peak(x: np.ndarray) -> float:
    """絶対値が最大の値を符号付きで返す。空・全欠測なら NaN。"""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return float("nan")
    idx = np.flatnonzero(ok)
    return float(x[idx[np.argmax(np.abs(x[idx]))]])


def evidence_masks(
    df: pd.DataFrame, cfg: dict[str, Any], rate_hz: float
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """証拠ごとのサンプル単位のマスクと、信号が存在するかを返す。

    「信号が無い」と「有るが立たない」を混同しない。comma2k19 の RAV4 には
    そもそも 0x226 も 0x3B7 も無いので、後者と区別できないと
    「このデータには滑りが無い」と読み違える。
    """
    ds = cfg["direct_slip"]
    n = len(df)
    masks: dict[str, np.ndarray] = {}
    present: dict[str, bool] = {}

    for group in ("flags", "context"):
        for name, spec in (ds.get(group) or {}).items():
            feat = spec["feature"]
            present[name] = feat in df.columns and bool(np.isfinite(_col(df, feat)).any())
            x = _col(df, feat)
            m = np.where(np.isfinite(x), x > float(spec["threshold"]), False)
            # 上限は任意。多値の信号で、作動を表す値の**上**に別の状態がある
            # ときに使う (slip_warn は 3 = ランプ試験 / 4 = 作動 / 7 = 継続点灯)。
            hi = spec.get("max")
            if hi is not None:
                m &= np.where(np.isfinite(x), x <= float(hi), False)
            masks[name] = m

    # 物理挙動。フラグと違って閾値を人が決めるので、持続時間の条件も課す。
    v = _col(df, "v_mps")
    for name, default_op in (("wheel_spread", "ge"), ("wheel_decel", "le"),
                             ("lateral", "abs_ge"), ("yaw_residual", "abs_ge")):
        spec = ds.get(name)
        if not spec:
            masks[name] = np.zeros(n, dtype=bool)
            present[name] = False
            continue
        feat = spec["feature"]
        present[name] = feat in df.columns and bool(np.isfinite(_col(df, feat)).any())
        x = _col(df, feat)
        thr = float(spec["threshold"])
        op = spec.get("op", default_op)
        if op == "le":
            hit = x <= thr
        elif op == "abs_ge":
            hit = np.abs(x) >= thr
        else:
            hit = x >= thr
        raw = np.where(np.isfinite(x), hit, False)
        vmin = spec.get("min_speed_mps")
        if vmin is not None:
            raw &= np.where(np.isfinite(v), v >= float(vmin), False)
        min_n = max(1, window_samples(float(spec.get("min_duration_s", 0.0)), rate_hz))
        keep = np.zeros(n, dtype=bool)
        for a, b in _runs(raw):
            if b - a + 1 >= min_n:
                keep[a : b + 1] = True
        masks[name] = keep
    return masks, present


def find_direct_slip(
    gs: GriddedSignals, cfg: dict[str, Any]
) -> tuple[list[DirectSlipEvent], DirectSlipCounts]:
    """証拠を集めて確認単位にまとめる。"""
    df = gs.df
    counts = DirectSlipCounts(
        n_samples=len(df), n_hours=float(len(df)) / gs.rate_hz / 3600.0
    )
    if len(df) == 0:
        return [], counts

    ds = cfg["direct_slip"]
    t = df["t"].to_numpy()
    masks, present = evidence_masks(df, cfg, gs.rate_hz)
    for name, ok in present.items():
        (counts.present if ok else counts.missing)[name] = 1

    gap = float(ds["merge_gap_s"])
    # 証拠ごとに細切れをつないでから数える。件数の意味を証拠間でそろえる。
    for name in (*ALL_EVIDENCE, *CONTEXT):
        m = masks[name]
        runs = _merge_runs(_runs(m), t, gap)
        for a, b in runs:
            m[a : b + 1] = True
        if name in ALL_EVIDENCE:
            counts.runs_by_evidence[name] = len(runs)

    # 制御作動か輪速の開きが立っていれば確認単位にする (TRIGGER)。
    # 参考 (tc_disabled) だけでは作らない。ランプチェックを拾ってしまう。
    any_ev = np.zeros(len(df), dtype=bool)
    for name in TRIGGER:
        any_ev |= masks[name]
    if not any_ev.any():
        return [], counts

    pad = window_samples(float(ds["window_pad_s"]), gs.rate_hz)
    padded = np.zeros(len(df), dtype=bool)
    for a, b in _runs(any_ev):
        padded[max(0, a - pad) : min(len(df), b + pad + 1)] = True
    windows = _merge_runs(_runs(padded), t, float(ds["final_merge_gap_s"]))

    v = _col(df, "v_mps")
    # どの作動に何が対応するか。設定で変えられるようにしておく。
    corr_cfg = ds.get("corroboration") or {}
    # 裏付けの信号 -> 系統。同じ物理量から出た裏付けを別々に数えないため。
    mech_of: dict[str, str] = {}
    for mech, members in (ds.get("mechanisms") or {}).items():
        for m in members:
            mech_of[m] = mech
    min_multi = int(ds["grade"]["min_mechanisms_for_multi"])
    out: list[DirectSlipEvent] = []
    for a, b in windows:
        sl = slice(a, b + 1)
        hit = {name: bool(masks[name][sl].any()) for name in (*ALL_EVIDENCE, *CONTEXT)}
        n_direct = sum(hit[name] for name in DIRECT)

        # 作動ごとに、対応する物理挙動が同じ窓に出ているかを見る。
        pairs: list[str] = []
        mechs: list[str] = []
        for name in DIRECT:
            if not hit[name]:
                continue
            backing = [s for s in (corr_cfg.get(name) or []) if hit.get(s)]
            if backing:
                pairs.append(f"{name}:{'+'.join(backing)}")
                for s in backing:
                    m = mech_of.get(s)
                    if m and m not in mechs:
                        mechs.append(m)
        n_corr = len(pairs)
        # **系統の数で決める。** 作動の数ではない。ABS と slip_warn が同じ
        # wheel_decel だけで裏付けられていても、分かった事実は 1 つなので 1 系統。
        n_mech = len(mechs)
        if n_mech >= min_multi:
            grade = GRADE_MULTI
        elif n_corr >= 1:
            grade = GRADE_HIGH
        elif n_direct >= 1:
            grade = GRADE_ACTIVATION
        else:
            grade = GRADE_PHYSICS

        # 山の位置は証拠が立っているサンプルの中から選ぶ。余白の側は見ない。
        core = np.zeros(len(df), dtype=bool)
        for name in TRIGGER:
            core[sl] |= masks[name][sl]
        idx = np.flatnonzero(core)
        # 速度が読めるところを優先し、読めなければ区間の先頭を代表にする。
        pk = int(idx[0]) if idx.size else a
        if idx.size:
            vv = v[idx]
            if np.isfinite(vv).any():
                pk = int(idx[np.nanargmax(np.where(np.isfinite(vv), vv, -np.inf))])

        ev = "|".join(name for name in ALL_EVIDENCE if hit[name])
        out.append(
            DirectSlipEvent(
                t_start=float(t[a]),
                t_end=float(t[b]),
                duration_s=round(float(t[b] - t[a]), 2),
                grade=grade,
                n_direct=n_direct,
                n_corroborated=n_corr,
                n_mechanisms=n_mech,
                evidence=ev,
                corroboration="|".join(pairs),
                mechanisms="|".join(mechs),
                slip_warn=hit["slip_warn"],
                slip_warn_peak=round(float(np.nanmax(_col(df, "slip_warn")[sl]))
                                     if np.isfinite(_col(df, "slip_warn")[sl]).any()
                                     else float("nan"), 1),
                abs_active=hit["abs_active"],
                vsc_active=hit["vsc_active"],
                wheel_spread=hit["wheel_spread"],
                ws_spread_peak_mps=round(
                    float(np.nanmax(_col(df, ds["wheel_spread"]["feature"])[sl]))
                    if np.isfinite(_col(df, ds["wheel_spread"]["feature"])[sl]).any()
                    else float("nan"), 3),
                wheel_decel=hit["wheel_decel"],
                yaw_residual=hit["yaw_residual"],
                wheel_decel_peak_mps2=round(
                    float(np.nanmin(_col(df, "wheel_decel_mps2")[sl]))
                    if np.isfinite(_col(df, "wheel_decel_mps2")[sl]).any()
                    else float("nan"), 2),
                lateral=hit["lateral"],
                tc_disabled=hit["tc_disabled"],
                v_at_peak_mps=round(float(v[pk]), 2),
                v_max_mps=round(float(np.nanmax(v[sl])) if np.isfinite(v[sl]).any()
                                else float("nan"), 2),
                ax_at_peak_mps2=round(float(_col(df, "ax_mps2")[pk]), 3),
                ay_at_peak_mps2=round(_peak(_col(df, "ay_kin_mps2")[sl]), 3),
                yaw_rate_at_peak_dps=round(_peak(_col(df, "yaw_rate_dps")[sl]), 2),
                steer_at_peak_deg=round(_peak(_col(df, "steer_deg")[sl]), 1),
                beta_model_peak_deg=round(_peak(_col(df, "beta_model_deg")[sl]), 3),
            )
        )
        counts.by_grade[grade] = counts.by_grade.get(grade, 0) + 1
        key = ev or "(なし)"
        counts.by_combination[key] = counts.by_combination.get(key, 0) + 1

    counts.n_events = len(out)
    return out, counts


def events_to_frame(
    events: list[DirectSlipEvent], config_hash: str = ""
) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame([e.as_dict() for e in events])
    if config_hash:
        df["config_hash"] = config_hash
    # 等級の強い順、次に裏付けの取れた作動の数、次に速度。
    order = {g: i for i, g in enumerate(GRADE_ORDER)}
    df["_o"] = df["grade"].map(order).fillna(9)
    return (df.sort_values(["_o", "n_mechanisms", "n_corroborated", "v_max_mps"],
                           ascending=[True, False, False, False])
              .drop(columns="_o").reset_index(drop=True))
