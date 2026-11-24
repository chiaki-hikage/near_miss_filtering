"""人のレビューの対象 (段 C 全件 + 段 A から無作為) を選び、伏せた番号を振り、件ごとの状況と信号の品質を点検する。

伏せ方: 対象の順番を乱数で混ぜて V01〜 の番号を振る。どちらの群か・どの旗が立ったかは key.csv にだけ書き、
レビュー用の一覧 (review_list.csv) には書かない。レビューのあとで key.csv と突き合わせる。

状況 (旋回・斜面との区別の参考。自動の値で、判定ではない):
  ヨー角速度の低域 (0.5 Hz 以下) の平均 = 定常の旋回、GPS の軌跡の曲がり具合、舵角の変化 (時刻は未補正)、
  ロール角・ピッチ角の前後 ±10 s の中央値 (斜面)、ピッチ角の変化 (登り・下りの切り替わり)
信号の品質 (偽の異常の疑い):
  racepak の補正の状態の継ぎ目が近い / 状態が相関で決まらなかった窓 / racepak の抜け / ペダルの信号の飛び値 (2026-11-18 からはアクセルとみられる列) /
  後輪の回転の 1 点の飛び値 / IMU の抜け / 走行の先頭・末尾に近い / GPS の速度の飛び / 鏡写しの振れ (r18)

    uv run python scripts/offroad/td2_review_set.py --profile out/offroad/tartandrive2/20261117_r20 \
        --sync out/offroad/tartandrive2/20261117_r15 --artifacts out/offroad/tartandrive2/20261117_r18 --run-id <新しい ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yaml

import td2_sync_correct as SC
import td2_vibration_poc as P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offroad/td2_vibration_poc.yaml")
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--sync", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    rc = cfg["review"]
    lat = cfg["profile"]["racepak_latency_s"]
    out = Path("out/offroad/tartandrive2") / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} はすでにある。新しい --run-id を使う")
    out.mkdir(parents=True, exist_ok=True)
    F = pd.read_csv(a.profile / "candidates.csv")
    ST = pd.read_csv(a.sync / "sync_states.csv")
    MB = pd.read_csv(a.artifacts / "mirror_bursts.csv") if (a.artifacts / "mirror_bursts.csv").stat().st_size > 1 else pd.DataFrame()
    C = F[F.tier.str.startswith("C")].assign(group="C")
    A = F[F.tier.str.startswith("A")].sample(rc["n_random_A"], random_state=rc["seed"]).assign(group="A")
    S = pd.concat([C, A]).sample(frac=1, random_state=rc["seed"] + 1).reset_index(drop=True)
    S.insert(0, "review_id", [f"V{i + 1:02d}" for i in range(len(S))])
    lo, hi = rc["check_window_s"]
    rows = []
    for run, grp in S.groupby("run"):
        R = P.load_run(Path(cfg["data_dir"]) / run, cfg)
        Sh = P.shock_series(R, cfg)
        P.run_checks(R, Sh, np.nan)
        g, fs = R["g"], R["fs"]
        wz_lp = P.lp(R["X"]["wz"], rc["yaw_lp_hz"], fs)
        to, o = R["odom"]
        St = ST[ST.run == run]
        sh = P.load_topic(Path(cfg["data_dir"]) / run, "shock_pos")
        tc, keep = SC.correct_times(sh[0], St, lat)
        tsh = tc[keep]
        ct = P.load_topic(Path(cfg["data_dir"]) / run, "controls")
        t2, k2 = SC.correct_times(ct[0], St, lat)
        c2 = ct[1][k2]
        t2 = t2[k2]
        wr = P.load_topic(Path(cfg["data_dir"]) / run, "wheel_rpm")
        tw, kw = SC.correct_times(wr[0], St, lat)
        wv = wr[1][kw][:, 2:4].astype(float)
        tw = tw[kw]
        for _, c in grp.iterrows():
            te = c.t_abs
            w = (g >= te - 2) & (g <= te + 1)
            ctx = (g >= te - 10) & (g <= te + 10)
            # GPS の軌跡: ±3 s の向きの変化 [deg] と、進んだ距離
            mo = (to >= te - 3) & (to <= te + 3)
            p = o[mo, :2]
            d = np.linalg.norm(np.diff(p, axis=0), axis=1)
            hd = np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0]))[d > 0.05]) if (d > 0.05).sum() > 3 else np.array([0.0])
            steer = R["steer"]
            ms = (steer[0] >= te - 3) & (steer[0] <= te + 2) if steer is not None else None
            sts = St.t_center.to_numpy()
            near = St[(sts >= te + lo) & (sts <= te + hi)]
            sw = (tsh >= te + lo) & (tsh <= te + hi)
            gaps = np.diff(tsh[sw]) if sw.sum() > 1 else np.array([0.0])
            bw = (t2 >= te + lo) & (t2 <= te + hi)
            ww = (tw >= te + lo) & (tw <= te + hi)
            from scipy.signal import medfilt
            spikes = int((np.abs(wv[ww] - np.c_[medfilt(wv[ww, 0], 5), medfilt(wv[ww, 1], 5)]) > rc["wheel_spike_rpm"]).sum()) if ww.sum() > 5 else 0
            sp = R["speed"][(g >= te + lo) & (g <= te + hi)]
            mb = (len(MB) and ((MB.run == run) & (MB.t_end >= te + lo) & (MB.t_start <= te + hi)).any())
            q = dict(
                q_state_switch=bool(near.state.nunique() > 1),
                q_state_undecided=bool((near.state_raw == -1).any()),
                q_racepak_gap_s=float(gaps.max()),
                q_brake_invalid=int(np.isnan(np.c_[P.pedals(c2[bw], cfg)]).any(axis=1).sum()) if bw.any() else 0,   # 列は q_brake_invalid のまま (前の版と合わせる)
                q_wheel_spikes=spikes,
                q_imu_gap=bool(R["gap"][(g >= te + lo) & (g <= te + hi)].any()),
                q_run_edge=bool(te - g[0] < rc["edge_s"] or g[-1] - te < rc["edge_s"]),
                q_gps_speed_jump=float(np.max(np.abs(np.diff(sp))) * fs) if len(sp) > 1 else 0.0,
                q_mirror_burst=bool(mb),
            )
            notes = []
            if q["q_state_switch"]:
                notes.append("racepak の補正の状態が近くで切り替わる (継ぎ目)")
            if q["q_state_undecided"]:
                notes.append("racepak の補正の状態が相関で決まらなかった窓がある")
            if q["q_racepak_gap_s"] > rc["racepak_gap_s"]:
                notes.append(f"racepak の抜け {q['q_racepak_gap_s']:.2f} s")
            if q["q_brake_invalid"]:
                notes.append(f"{'アクセル' if cfg['reaction'].get('controls_cols') else 'ブレーキ'}とみられる信号の飛び値 {q['q_brake_invalid']} 点")
            if q["q_wheel_spikes"]:
                notes.append(f"後輪の回転の 1 点の飛び値 {q['q_wheel_spikes']} 点")
            if q["q_imu_gap"]:
                notes.append("IMU の抜け")
            if q["q_run_edge"]:
                notes.append("走行の先頭・末尾に近い")
            if q["q_gps_speed_jump"] > rc["gps_jump_mps2"]:
                notes.append(f"GPS の速度の飛び ({q['q_gps_speed_jump']:.0f} m/s²)")
            if q["q_mirror_burst"]:
                notes.append("ストロークの鏡写しの振れ")
            ctxv = dict(
                yaw_rate_lp_dps=float(np.degrees(np.mean(wz_lp[w]))),
                heading_change_deg=float(np.degrees(hd[-1] - hd[0])),
                path_m=float(d.sum()),
                steer_range=float(np.ptp(steer[1][ms, 0])) if ms is not None and ms.any() else np.nan,
                roll_base_deg=float(np.degrees(np.median(R["roll_g"][ctx] - np.median(R["roll_g"])))),
                pitch_base_deg=float(np.degrees(np.median(R["pitch_g"][ctx]))),
                pitch_change_deg=float(np.degrees(np.ptp(P.lp(R["pitch_g"], 0.5, fs)[(g >= te - 3) & (g <= te + 2)]))),
            )
            auto = []
            if abs(ctxv["yaw_rate_lp_dps"]) >= rc["turn_yaw_dps"]:
                auto.append("旋回中")
            if abs(ctxv["roll_base_deg"]) >= rc["slope_deg"] or abs(ctxv["pitch_base_deg"]) >= rc["slope_deg"]:
                auto.append("斜面")
            if ctxv["pitch_change_deg"] >= rc["crest_pitch_change_deg"]:
                auto.append("登り下りの切り替わり")
            rows.append(dict(review_id=c.review_id, **ctxv, auto_context=";".join(auto) or "直進・平地", **q,
                             quality_notes=";".join(notes) or "なし"))
        print(run[:40], len(grp), flush=True)
    Q = pd.DataFrame(rows)
    S = S.merge(Q, on="review_id").sort_values("review_id").reset_index(drop=True)
    att = ["flag_yaw", "flag_roll", "flag_wheel_lr"]
    S["n_attitude_flags"] = S[att].sum(axis=1)
    key = S[["review_id", "cand_id", "group", "tier", "reasons", "n_attitude_flags", "n_vehicle_flags", "n_driver_flags"]
            + [c for c in S.columns if c.startswith("flag_")]
            + ["yaw_dist_dps", "roll_dev_max_deg", "wheel_lr_jump", "brake_onset_s", "local_shock_pct", "contrast", "mtvv_total", "input_axis", "temporal"]]
    key.round(4).to_csv(out / "key.csv", index=False)
    lst = S[["review_id", "run", "t", "t_abs", "speed", "auto_context", "quality_notes", "yaw_rate_lp_dps", "heading_change_deg", "path_m",
             "steer_range", "roll_base_deg", "pitch_base_deg", "pitch_change_deg"] + [c for c in S.columns if c.startswith("q_")]]
    lst.round(3).to_csv(out / "review_list.csv", index=False)
    # レビューで付ける欄 (空欄)。選択肢は README と viewer に同じものを書く
    form = S[["review_id"]].copy()
    for col in rc["form_columns"]:
        form[col] = ""
    form.to_csv(out / "review_form_blank.csv", index=False)
    summ = dict(n=len(S), groups=S.group.value_counts().to_dict(), runs=int(S.run.nunique()),
                auto_context_by_group=pd.crosstab(S.group, S.auto_context).to_dict(),
                quality_any_by_group=S.assign(qa=S.quality_notes != "なし").groupby("group").qa.mean().round(3).to_dict(),
                quality_counts={c: int(S[c].astype(float).gt(0).sum()) for c in S.columns if c.startswith("q_")},
                seed=rc["seed"])
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    (out / "meta.json").write_text(json.dumps(dict(review=rc, profile=str(a.profile), sync=str(a.sync), run_id=a.run_id), ensure_ascii=False, indent=2))
    print(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
