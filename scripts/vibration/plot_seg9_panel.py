"""08-17 seg9 の 3 仮説の波形 (左) と、車輪の通り道を重ねた路面画像 (右) を 1 枚に並べる。

    uv run python scripts/vibration/plot_seg9_panel.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
import numpy as np
import yaml

import compare_fullcar as CF
import final_qualitative as Q
import find_roll_bumps as F
import fullcar as FC
from near_miss.io.comma2k19 import find_segments
from near_miss.plotting import configure_japanese_font

REPO = _bootstrap.SRC.parent
EVENT = next(e for e in Q.EVENTS if e[0] == "2018-08-17--14-55-39" and e[1] == 9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO / "raw_data/Chunk_1")
    ap.add_argument("--config", type=Path, default=REPO / "configs/vibration_fullcar.yaml")
    ap.add_argument("--vib-config", type=Path, default=REPO / "configs/vibration.yaml")
    ap.add_argument("--pavement-config", type=Path, default=REPO / "configs/vibration_pavement.yaml")
    ap.add_argument("--camera-fit", type=Path, default=REPO / "out/vibration/pavement/site19_camera_fit.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "out/vibration/final/seg9_waveform_road.png")
    args = ap.parse_args()

    matplotlib.use("Agg")
    configure_japanese_font()
    import matplotlib.pyplot as plt

    cfg_vib = yaml.safe_load(args.vib_config.read_text())
    p = FC.build_params(FC.load_config(args.config))
    cam = dict(yaml.safe_load(args.camera_fit.read_text()))
    f = yaml.safe_load(args.pavement_config.read_text())["camera"]["focal_px"]
    refs = [r for r in find_segments(args.data_root) if r.drive_id.endswith(EVENT[0])]
    ref = next(r for r in refs if r.index == EVENT[1])
    seg = F.load_seg(ref, F.drive_rotation([r.path for r in refs]), cfg_vib)
    E = CF.Event(seg, EVENT[:6], cfg_vib)
    preds = {name: CF.fit_single(E, p, CF.SINGLE[key], "unilateral") for name, key in Q.HYP.items()}

    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], wspace=0.12, hspace=0.12)
    t = E.t - preds["右側入力"]["t_hit_s"]
    ylabels = ("上下加速度 [m/s²]", "ロール角速度 [deg/s]", "ピッチ角速度 [deg/s]")
    for r, (k, yl) in enumerate(zip(("heave", "roll", "pitch"), ylabels)):
        ax = fig.add_subplot(gs[r, 0])
        ax.plot(t, E.meas[k], color="k", lw=1.6, label="実測")
        for name, d in preds.items():
            ax.plot(t, d["pred"][k], color=Q.COLORS[name], lw=1.0, alpha=0.85, label=name)
        ax.axvline(0, color="orange", lw=0.8, ls="--")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(yl, fontsize=9)
        ax.grid(alpha=0.3)
        if r < 2:
            ax.tick_params(labelbottom=False)
        if r == 0:
            ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_xlabel("時間 [s]")

    ax = fig.add_subplot(gs[:, 1])
    img, t_frame = Q.frame_at(ref.path, seg.t0, EVENT[2] - EVENT[6])
    ax.imshow(img)
    yh, K = cam["horizon_row"], cam["K"]
    track = float(p.track[0])
    d_on = E.v * (EVENT[2] - t_frame) + cam["camera_behind_front_axle_m"]
    d = np.linspace(3.5, 40, 200)
    for y_lat, c in ((-track / 2, "tab:blue"), (track / 2, "tab:red")):
        ax.plot(582 + f * y_lat / d, yh + K / d, color=c, lw=1.4, ls="--")
        ax.plot(582 + f * y_lat / d_on, yh + K / d_on, "o", ms=11, mfc="none", mec=c, mew=2.2)
    ax.set_xlim(420, 980)
    ax.set_ylim(690, 440)
    ax.axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
