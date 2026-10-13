"""7 自由度 Full-car モデル (線形、静的つり合いまわりの微小変位)。

座標 (車両固定。IMU の [前, 右, 下] と回転の向きを合わせてある)

    q = [z, θ, φ, zu_FL, zu_FR, zu_RL, zu_RR]
    z   ばね上重心の上下変位 [m]       上向き正
    θ   ピッチ角 [rad]                機首上げ正 (= IMU gyro right の正)
    φ   ロール角 [rad]                右下がり正 (= IMU gyro forward の正)
    zu  ばね下の上下変位 [m]          上向き正

各輪の位置は重心から (x_i 前正, y_i 右正)。ばね上側の取付点の上下変位は

    zs_i = z + x_i θ − y_i φ

運動方程式は M q̈ + C q̇ + K q = Kr zr + Cr żr + E [ax, ay]。導出と記号は
docs/vibration_fullcar_model.md 2 章。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import StateSpace, butter, lsim, sosfiltfilt

G = 9.80665
CORNERS = ("FL", "FR", "RL", "RR")


def _v(cfg: dict, sec: str, key: str, override: dict | None = None):
    if override and key in override:
        return override[key]
    return cfg[sec][key]["value"]


@dataclass
class Params:
    m_s: float
    m_u: np.ndarray        # (4,)
    I_xx: float
    I_yy: float
    x: np.ndarray          # (4,) 前正
    y: np.ndarray          # (4,) 右正
    k: np.ndarray          # (4,) ばね
    c: np.ndarray          # (4,) ダンパー
    k_arb: np.ndarray      # (2,) スタビライザーの等価ばね [N/m] (前, 後)
    k_t: float
    c_t: float
    h_cg: float
    h_roll: float          # 重心からロール軸までの高さ
    h_pitch: float
    imu: np.ndarray        # (3,) [前, 右, 上]
    envelope_m: float
    static_load: np.ndarray  # (4,) 各輪の静的接地荷重 [N]
    wheelbase: float
    track: np.ndarray      # (2,) 前, 後


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_params(cfg: dict, override: dict | None = None) -> Params:
    """設定から物理パラメータを組む。override はキー名で値を差し替える (感度分析用)。"""
    o = override or {}
    m_tot = _v(cfg, "vehicle", "mass_total_kg", o)
    L = _v(cfg, "vehicle", "wheelbase_m", o)
    tf = _v(cfg, "vehicle", "track_front_m", o)
    tr = _v(cfg, "vehicle", "track_rear_m", o)
    a = _v(cfg, "vehicle", "cg_to_front_m", o)
    b = L - a
    muf = _v(cfg, "unsprung", "mass_front_kg", o)
    mur = _v(cfg, "unsprung", "mass_rear_kg", o)
    m_u = np.array([muf, muf, mur, mur])
    m_s = m_tot - m_u.sum()
    h_cg = _v(cfg, "vehicle", "cg_height_m", o)
    h_roll = h_cg - _v(cfg, "vehicle", "roll_axis_height_m", o)
    h_pitch = h_cg - _v(cfg, "vehicle", "pitch_axis_height_m", o)

    # ばね上を前後の軸に配分した 1 輪あたりの質量
    ms_corner = np.array([m_s * b / L / 2] * 2 + [m_s * a / L / 2] * 2)
    fr = np.array([_v(cfg, "suspension", "ride_freq_front_hz", o)] * 2
                  + [_v(cfg, "suspension", "ride_freq_rear_hz", o)] * 2)
    k = ms_corner * (2 * np.pi * fr) ** 2
    zeta = _v(cfg, "suspension", "damping_ratio", o)
    c = 2 * zeta * np.sqrt(k * ms_corner)

    # 目標のロール勾配からロール剛性の総量を決め、ばねで足りない分をスタビライザーに持たせる
    #   φ / ay = m_s h_r / (Kφ − m_s g h_r)
    grad = np.deg2rad(_v(cfg, "suspension", "roll_gradient_deg_per_g", o)) / G
    K_phi = m_s * h_roll / grad + m_s * G * h_roll
    share = _v(cfg, "suspension", "roll_stiffness_front_share", o)
    K_spring = np.array([k[0] * tf**2 / 2, k[2] * tr**2 / 2])
    K_arb = np.array([share, 1 - share]) * K_phi - K_spring
    K_arb = np.maximum(K_arb, 0.0)
    k_arb = K_arb / np.array([tf, tr]) ** 2

    m_axle = np.array([m_tot * b / L, m_tot * a / L])
    static = np.repeat(m_axle * G / 2, 2)

    return Params(
        m_s=m_s, m_u=m_u,
        I_xx=_v(cfg, "vehicle", "inertia_roll_kgm2", o), I_yy=_v(cfg, "vehicle", "inertia_pitch_kgm2", o),
        x=np.array([a, a, -b, -b]), y=np.array([-tf / 2, tf / 2, -tr / 2, tr / 2]),
        k=k, c=c, k_arb=k_arb,
        k_t=_v(cfg, "tire", "stiffness_n_per_m", o), c_t=_v(cfg, "tire", "damping_ns_per_m", o),
        h_cg=h_cg, h_roll=h_roll, h_pitch=h_pitch,
        imu=np.asarray(_v(cfg, "imu", "pos_from_cg_m", o), float),
        envelope_m=_v(cfg, "tire", "envelope_length_m", o),
        static_load=static, wheelbase=L, track=np.array([tf, tr]),
    )


def matrices(p: Params) -> dict[str, np.ndarray]:
    """M, C, K と入力行列。δ = S q がサスペンションの伸び (ばね上 − ばね下)。"""
    T = np.c_[np.ones(4), p.x, -p.y]                 # zs = T [z, θ, φ]
    S = np.c_[T, -np.eye(4)]                          # δ = S q
    Kd = np.diag(p.k).astype(float)
    for ax, (i, j) in enumerate(((0, 1), (2, 3))):    # スタビライザー: 左右の伸びの差に比例
        kb = p.k_arb[ax]
        Kd[np.ix_([i, j], [i, j])] += kb * np.array([[1, -1], [-1, 1]])
    Cd = np.diag(p.c)
    M = np.diag([p.m_s, p.I_yy, p.I_xx, *p.m_u])
    K = S.T @ Kd @ S
    C = S.T @ Cd @ S
    K[3:, 3:] += p.k_t * np.eye(4)
    C[3:, 3:] += p.c_t * np.eye(4)
    # 重力による転倒側のモーメント (傾くと重心が横・前後にずれる)
    K[2, 2] -= p.m_s * G * p.h_roll
    K[1, 1] -= p.m_s * G * p.h_pitch
    Kr = np.zeros((7, 4))
    Kr[3:, :] = p.k_t * np.eye(4)
    Cr = np.zeros((7, 4))
    Cr[3:, :] = p.c_t * np.eye(4)
    # 慣性力: 減速 (ax < 0) で機首下げ、左旋回 (ay > 0) で右下がり
    E = np.zeros((7, 2))
    E[1, 0] = p.m_s * p.h_pitch
    E[2, 1] = p.m_s * p.h_roll
    return dict(M=M, C=C, K=K, Kr=Kr, Cr=Cr, E=E, S=S, Kd=Kd, Cd=Cd)


def state_space(p: Params) -> tuple[StateSpace, dict]:
    """x = [q, q̇] (14)、u = [zr (4), żr (4), ax, ay]。"""
    m = matrices(p)
    Minv = np.linalg.inv(m["M"])
    A = np.block([[np.zeros((7, 7)), np.eye(7)], [-Minv @ m["K"], -Minv @ m["C"]]])
    Bq = np.c_[m["Kr"], m["Cr"], m["E"]]
    B = np.r_[np.zeros((7, 10)), Minv @ Bq]
    return StateSpace(A, B, np.eye(14), np.zeros((14, 10))), m


def modes(p: Params) -> list[dict]:
    """固有振動数・減衰比と、どの自由度が主かを返す。"""
    sys, _ = state_space(p)
    with np.errstate(all="ignore"):
        w, V = np.linalg.eig(sys.A)
    out = []
    names = ("上下", "ピッチ", "ロール", "ばね下FL", "ばね下FR", "ばね下RL", "ばね下RR")
    scale = np.array([1, p.wheelbase / 2, p.track.mean() / 2, 1, 1, 1, 1])
    for lam, vec in zip(w, V.T):
        if lam.imag <= 0:
            continue
        shape = np.abs(vec[:7]) * scale
        shape /= shape.max()
        main = [names[i] for i in np.argsort(-shape) if shape[i] > 0.3]
        out.append(dict(freq_hz=abs(lam) / (2 * np.pi), damping=-lam.real / abs(lam),
                        dominant="+".join(main)))
    return sorted(out, key=lambda d: d["freq_hz"])


def downsample(t_src: np.ndarray, x: np.ndarray, t_out: np.ndarray, cutoff_hz: float = 40.0) -> np.ndarray:
    """計算結果を IMU のレートへ間引く。間引く前に低域通過をかけて折り返しを防ぐ。"""
    fs = 1.0 / (t_src[1] - t_src[0])
    sos = butter(4, cutoff_hz, btype="low", fs=fs, output="sos")
    return np.interp(t_out, t_src, sosfiltfilt(sos, x))


# ---------------------------------------------------------------------------
# 路面
# ---------------------------------------------------------------------------

def cosine_bump(s: np.ndarray, s0: float, length: float, height: float) -> np.ndarray:
    """s0 から length の 1−cos 形状。height > 0 で突起、< 0 で凹み。"""
    u = (s - s0) / length
    out = np.zeros_like(s)
    m = (u >= 0) & (u <= 1)
    out[m] = height / 2 * (1 - np.cos(2 * np.pi * u[m]))
    return out


def ramp_step(s: np.ndarray, s0: float, length: float, height: float) -> np.ndarray:
    """s0 から length かけて height だけ高さが変わり、その後も保つ段差。"""
    return height * np.clip((s - s0) / length, 0, 1)


ISO8608_GD0 = {"A": 16e-6, "B": 64e-6, "C": 256e-6, "D": 1024e-6, "E": 4096e-6}  # n0 = 0.1 c/m の幾何平均 [m³]


def rough_patch(s: np.ndarray, s0: float, length: float, gd0: float, seed: int,
                n_band: tuple[float, float] = (0.05, 10.0), taper_m: float = 0.5) -> np.ndarray:
    """ISO 8608 型の荒れた区間。変位 PSD Gd(n) = gd0 (n / 0.1)^-2 を位相乱数で合成し、
    s0 から length の区間だけに置く (両端は taper_m の余弦窓)。"""
    rng = np.random.default_rng(seed)
    n = np.linspace(n_band[0], n_band[1], 800)
    dn = n[1] - n[0]
    amp = np.sqrt(2 * gd0 * (n / 0.1) ** -2 * dn)
    phase = rng.uniform(0, 2 * np.pi, len(n))
    u = s - s0
    inside = (u >= 0) & (u <= length)
    h = np.zeros_like(s)
    h[inside] = (amp[None, :] * np.cos(2 * np.pi * n[None, :] * u[inside, None] + phase[None, :])).sum(axis=1)
    a = np.clip(u / taper_m, 0, 1)
    b = np.clip((length - u) / taper_m, 0, 1)
    w = 0.5 * (1 - np.cos(np.pi * np.minimum(a, b)))
    return h * w * inside


def envelope(profile: np.ndarray, ds: float, length: float) -> np.ndarray:
    """タイヤの接地長で路面を平均する (包み込みの近似)。"""
    n = max(1, int(round(length / ds)))
    pad = np.r_[np.full(n, profile[0]), profile, np.full(n, profile[-1])]
    sm = np.convolve(pad, np.ones(n) / n, mode="same")
    return sm[n:-n]


@dataclass
class RoadCase:
    key: str
    title: str
    left: callable     # 距離 s [m] → 高さ [m]。左輪の通り道
    right: callable
    rear_follows: bool = True   # 後輪も同じ通り道を通るか
    joint_angle_deg: float = 0.0  # 斜めの継ぎ目。右輪が遅れて当たる角度


def road_cases(height: float = 0.03, length: float = 0.5, s0: float = 5.0) -> dict[str, RoadCase]:
    """依頼の 6 ケース。高さ・長さは感度分析で振る。s0 は前輪が路面形状に差しかかる距離 [m]。"""
    zero = lambda s: np.zeros_like(s)
    return {
        "A": RoadCase("A", f"右前輪のみ凹み (深さ {height * 100:.0f} cm・長さ {length:.1f} m)", zero, lambda s: cosine_bump(s, s0, length, -height), rear_follows=False),
        "B": RoadCase("B", f"右前輪のみ突起 (高さ {height * 100:.0f} cm・長さ {length:.1f} m)", zero, lambda s: cosine_bump(s, s0, length, height), rear_follows=False),
        "C": RoadCase("C", f"右前輪→右後輪が順に凹みを通過 (深さ {height * 100:.0f} cm・長さ {length:.1f} m)", zero, lambda s: cosine_bump(s, s0, length, -height)),
        "D": RoadCase("D", f"左右同時の舗装段差 (上り {height * 200 / 3:.0f} cm)", lambda s: ramp_step(s, s0, 0.05, height * 2 / 3),
                      lambda s: ramp_step(s, s0, 0.05, height * 2 / 3)),
        "E": RoadCase("E", f"斜めの舗装継ぎ目 (30°、突起 {height * 50:.1f} cm・幅 0.2 m)", lambda s: cosine_bump(s, s0, 0.2, height / 2),
                      lambda s: cosine_bump(s, s0, 0.2, height / 2), joint_angle_deg=30.0),
        "F": RoadCase("F", f"右側だけ {height * 200 / 3:.0f} cm 低い路面へ移る", zero, lambda s: ramp_step(s, s0, length, -height * 2 / 3)),
    }


def wheel_inputs(case: RoadCase, p: Params, v: float, t: np.ndarray) -> np.ndarray:
    """各輪の路面高さ (時刻 t) を返す。(len(t), 4)。後輪はホイールベースぶん遅れる。"""
    ds = 0.005
    s_max = v * t[-1] + 20
    s = np.arange(-20, s_max, ds)
    prof = {
        "L": envelope(case.left(s), ds, p.envelope_m),
        "R": envelope(case.right(s), ds, p.envelope_m),
    }
    zr = np.zeros((len(t), 4))
    lag_joint = {"L": 0.0, "R": 0.0}
    if case.joint_angle_deg:
        # 左輪から当たる向きの斜め継ぎ目: 右輪は track × tanα だけ遠い
        lag_joint["R"] = p.track[0] * np.tan(np.deg2rad(case.joint_angle_deg))
    for i, name in enumerate(CORNERS):
        side = name[1]
        rear = name[0] == "R"
        if rear and not case.rear_follows and side == "R":
            continue
        pos = v * t - (p.wheelbase if rear else 0.0) - lag_joint["L" if side == "L" else "R"]
        zr[:, i] = np.interp(pos, s, prof["L" if side == "L" else "R"])
    return zr


# ---------------------------------------------------------------------------
# 応答
# ---------------------------------------------------------------------------

def simulate(p: Params, t: np.ndarray, zr: np.ndarray, ax: np.ndarray | None = None,
             ay: np.ndarray | None = None, contact: str = "unilateral") -> dict[str, np.ndarray]:
    """路面入力と前後・横の加速度から応答を出す。

    contact="unilateral" (既定) ではタイヤは路面を引っ張れない (接地荷重 ≥ 0)。
    線形解で接地荷重が負になったときだけ RK4 で非線形に解き直す。
    contact="linear" は修正前と同じ線形タイヤ。

    返す量の区別
      IMU で直接観測できる量の予測  imu_heave / imu_fx / imu_fy / roll_rate / pitch_rate
      モデルでしか出ない量         roll / pitch / susp_defl / susp_force / arb_force / tire_load
    """
    # macOS の numpy (Accelerate) は matmul で誤った浮動小数点警告を出す。
    # 警告は抑え、代わりに結果が有限であることを最後に確かめる
    with np.errstate(all="ignore"):
        out = _simulate(p, t, zr, ax, ay, contact)
    bad = [k for k, v in out.items() if v.dtype.kind == "f" and not np.all(np.isfinite(v))]
    if bad:
        raise FloatingPointError(f"有限でない値: {bad}")
    return out


def _tire_dev(p: Params, zr, zr_dot, zu, zu_dot, unilateral: bool) -> np.ndarray:
    """タイヤ力の静的つり合いからの変化分。片側拘束では接地荷重が 0 を下回らない (引っ張れない)。"""
    f = p.k_t * (zr - zu) + p.c_t * (zr_dot - zu_dot)
    if unilateral:
        f = np.maximum(f, -p.static_load)
    return f


def _integrate_unilateral(p: Params, m: dict, t: np.ndarray, zr, zr_dot, a_in) -> tuple[np.ndarray, np.ndarray]:
    """タイヤが浮く場合の非線形応答を RK4 で積分する。サスペンション側は線形のまま。"""
    Ks = m["K"].copy()
    Cs = m["C"].copy()
    Ks[3:, 3:] -= p.k_t * np.eye(4)
    Cs[3:, 3:] -= p.c_t * np.eye(4)
    Minv = np.linalg.inv(m["M"])
    E = m["E"]

    def f(x, zr_, zrd_, a_):
        q, qd = x[:7], x[7:]
        ft = _tire_dev(p, zr_, zrd_, q[3:], qd[3:], True)
        rhs = -Ks @ q - Cs @ qd + E @ a_
        rhs[3:] += ft
        return np.r_[qd, Minv @ rhs]

    x = np.zeros((len(t), 14))
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        zm, zdm, am = (zr[i] + zr[i + 1]) / 2, (zr_dot[i] + zr_dot[i + 1]) / 2, (a_in[i] + a_in[i + 1]) / 2
        k1 = f(x[i], zr[i], zr_dot[i], a_in[i])
        k2 = f(x[i] + h / 2 * k1, zm, zdm, am)
        k3 = f(x[i] + h / 2 * k2, zm, zdm, am)
        k4 = f(x[i] + h * k3, zr[i + 1], zr_dot[i + 1], a_in[i + 1])
        x[i + 1] = x[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x[:, :7], x[:, 7:]


def _simulate(p: Params, t: np.ndarray, zr: np.ndarray, ax, ay, contact: str = "unilateral") -> dict[str, np.ndarray]:
    sys, m = state_space(p)
    dt = t[1] - t[0]
    zr_dot = np.gradient(zr, dt, axis=0)
    ax = np.zeros(len(t)) if ax is None else ax
    ay = np.zeros(len(t)) if ay is None else ay
    u = np.c_[zr, zr_dot, ax, ay]
    _, _, x = lsim(sys, u, t)
    q, qd = x[:, :7], x[:, 7:]
    linear_tire = p.static_load + p.k_t * (zr - q[:, 3:]) + p.c_t * (zr_dot - qd[:, 3:])
    lifted = bool((linear_tire < 0).any())
    unilateral = contact == "unilateral"
    if unilateral and lifted:
        # 線形解で接地荷重が負になったときだけ非線形で解き直す。浮かなければ両者は同じ
        q, qd = _integrate_unilateral(p, m, t, zr, zr_dot, np.c_[ax, ay])
    ft = _tire_dev(p, zr, zr_dot, q[:, 3:], qd[:, 3:], unilateral)
    Minv = np.linalg.inv(m["M"])
    Ks = m["K"].copy()
    Cs = m["C"].copy()
    Ks[3:, 3:] -= p.k_t * np.eye(4)
    Cs[3:, 3:] -= p.c_t * np.eye(4)
    rhs = -(Ks @ q.T) - (Cs @ qd.T) + m["E"] @ np.c_[ax, ay].T
    rhs[3:] += ft.T
    qdd = (Minv @ rhs).T
    z, th, ph = q[:, 0], q[:, 1], q[:, 2]
    xm, ym, zm = p.imu
    delta = (m["S"] @ q.T).T
    delta_d = (m["S"] @ qd.T).T
    spring_damper = -(delta * p.k + delta_d * p.c)            # ばね上に働く上向きの力
    arb = -((m["Kd"] - np.diag(p.k)) @ delta.T).T
    tire = p.static_load + ft
    return dict(
        contact=np.array(contact), lifted_linear=np.array(lifted),
        lift_off=tire <= 1e-9,
        t=t, zr=zr,
        imu_heave=qdd[:, 0] + xm * qdd[:, 1] - ym * qdd[:, 2],   # 上向きの加速度 (重力を除く)
        imu_fx=-zm * qdd[:, 1] + G * th,                          # 前向きの比力 (重力の傾き成分を含む)
        imu_fy=zm * qdd[:, 2] - G * ph,                           # 右向きの比力
        roll_rate=np.rad2deg(qd[:, 2]), pitch_rate=np.rad2deg(qd[:, 1]),
        roll=np.rad2deg(ph), pitch=np.rad2deg(th), heave=z,
        susp_defl=delta, susp_force=spring_damper, arb_force=arb,
        tire_load=tire, unsprung=q[:, 3:],
    )
