"""Animated GIF of one running stride: reconstructed skeleton + synchronised GRF.

Usage: python scripts/make_animation.py --subject RBDS008 --speed 3.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rungait.events import force_events  # noqa: E402
from rungait.io import find_trials, load_c3d, load_participant_info  # noqa: E402
from rungait.kinematics import calibrate, track  # noqa: E402
from rungait.plotting import COLORS, style  # noqa: E402
from rungait.pipeline import MARKER_CUTOFF_HZ  # noqa: E402
from rungait.qc import preprocess_markers  # noqa: E402
from rungait.signal import lowpass  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="RBDS008")
    ap.add_argument("--speed", type=float, default=3.5)
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "stride_animation.gif"))
    a = ap.parse_args()
    style()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)

    T = find_trials(ROOT / "data" / "raw")
    t = T[T.subject == a.subject]
    static = load_c3d(t[t.kind == "static"].path.iloc[0], with_forces=False)
    run = load_c3d(t[t.speed == a.speed].path.iloc[0])
    mass = load_participant_info(ROOT / "data" / "raw" / "RBDSinfo.txt").loc[a.subject, "Mass"]
    cal = calibrate(static)
    mk, _, _ = preprocess_markers(run, cal, fc=MARKER_CUTOFF_HZ)
    _, virt = track(mk, cal)
    fh = {s: (mk[f"{s}.Heel.Bottom"][:, 1] + mk[f"{s}.MT1"][:, 1]) / 2 for s in "RL"}
    ev = force_events(run.grf[:, 1], run.analog_rate, run.rate, fh)
    r_ic = ev[ev.side == "R"].ic_frame.to_numpy()
    f0, f1 = r_ic[10], r_ic[11]                      # one right-foot stride
    frames = np.arange(f0, f1 + 1, 2)
    fy = lowpass(run.grf[:, 1], run.analog_rate, 50) / (mass * 9.81)
    tg = np.arange(len(fy)) / run.analog_rate

    def chain(s, f):
        pts = [virt[f"{s}.HJC"][f], virt[f"{s}.KJC"][f], virt[f"{s}.AJC"][f], mk[f"{s}.Heel.Bottom"][f],
               virt[f"{s}.midMT"][f], virt[f"{s}.AJC"][f]]
        return np.array(pts)

    origin = virt["pelvis_centre"][frames].mean(0)
    fig, (ax, axg) = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.25})
    lines = {s: ax.plot([], [], "-o", color=COLORS[s], lw=1.6, ms=2.5, label={"R": "Right", "L": "Left"}[s])[0] for s in "RL"}
    pel = ax.plot([], [], "s", color="k", ms=3.5)[0]
    ax.axhline(np.percentile(mk["R.Heel.Bottom"][:, 1], 2) - origin[1], color="k", lw=0.6)
    ax.set_xlim(-650, 650)
    ax.set_ylim(-1100, 250)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"(a) Sagittal view, {a.subject}, {a.speed} m/s", fontsize=8, loc="left")
    m = (tg >= f0 / run.rate) & (tg <= f1 / run.rate)
    axg.plot(tg[m] - f0 / run.rate, fy[m], color="k", lw=0.9)
    cursor = axg.axvline(0, color=COLORS["L"], lw=0.8)
    axg.set_xlabel("Time (s)")
    axg.set_ylabel("Vertical GRF (BW)")
    axg.set_xlim(0, (f1 - f0) / run.rate)
    axg.minorticks_on()
    axg.set_title("(b) Vertical ground reaction force", fontsize=8, loc="left")
    txt = axg.text(0.97, 0.95, "", transform=axg.transAxes, ha="right", va="top", fontsize=7)

    def update(f):
        for s in "RL":
            c = chain(s, f) - origin
            lines[s].set_data(c[:, 0], c[:, 1])
        p = virt["pelvis_centre"][f] - origin
        pel.set_data([p[0]], [p[1]])
        tt = (f - f0) / run.rate
        cursor.set_xdata([tt, tt])
        on = ev[(ev.ic_frame <= f) & (ev.to_frame >= f)]
        txt.set_text(f"Stance: {'right' if (on.side == 'R').any() else 'left' if len(on) else 'flight'}")
        return [*lines.values(), pel, cursor, txt]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(a.out, writer=PillowWriter(fps=20), dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
