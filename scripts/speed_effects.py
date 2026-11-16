"""Group analysis: how does running speed change the key metrics? (28 runners, 3 speeds)

Repeated-measures design -> Friedman test per metric, Kendall's W as effect size.
Usage: python scripts/speed_effects.py   (after scripts/run_all.py)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rungait.metrics import KEY_METRICS  # noqa: E402
from rungait.plotting import COLORS, DOUBLE_COL, panel_label, save, style, tex  # noqa: E402

RES = ROOT / "results"
SPEEDS = [2.5, 3.5, 4.5]
PLOT_METRICS = ["step_freq", "step_length", "contact_time", "vertical_osc",
                "peak_vgrf", "loading_rate", "knee_flex_peak", "overstride"]


def main():
    style()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RES / "summary_all.csv")
    wide = s.pivot_table(index=["subject", "metric"], columns="speed", values="mean").reset_index()
    wide = wide.dropna(subset=SPEEDS)

    rows = []
    for m in KEY_METRICS:
        d = wide[wide.metric == m]
        stat, p = friedmanchisquare(*(d[sp] for sp in SPEEDS))
        n, k = len(d), len(SPEEDS)
        rows.append(dict(metric=m, n=n, **{f"{sp}": f"{d[sp].mean():.2f} ± {d[sp].std():.2f}" for sp in SPEEDS},
                         chi2=stat, p=p, kendall_w=stat / (n * (k - 1))))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "speed_effects.csv", index=False, float_format="%.4g")

    lines = ["# Effect of running speed (28 runners)", "",
             "Mean ± SD across participants (each participant = mean of all valid steps, both sides). "
             "Friedman test for repeated measures; Kendall's W = effect size (0 = none, 1 = all runners change in the same order).",
             "", "| Metric | Unit | 2.5 m/s | 3.5 m/s | 4.5 m/s | p | Kendall's W |", "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        name, unit = KEY_METRICS[r.metric]
        p = "< 0.001" if r.p < 0.001 else f"{r.p:.3f}"
        lines.append(f"| {name} | {unit} | {r['2.5']} | {r['3.5']} | {r['4.5']} | {p} | {r.kendall_w:.2f} |")
    lines += ["", "Exploratory analysis of 14 metrics; p-values are not corrected for multiple comparisons."]
    (RES / "SPEED_EFFECTS.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # Figure 1: small multiples, individual runners + group mean
    ylab = {
        "step_freq": r"Step frequency (min$^{-1}$)", "step_length": r"Step length (m)",
        "contact_time": r"Contact time (ms)", "vertical_osc": r"Vertical oscillation (cm)",
        "peak_vgrf": r"Peak vertical GRF (BW)", "loading_rate": r"Loading rate (BW s$^{-1}$)",
        "knee_flex_peak": r"Peak knee flexion ($^\circ$)", "overstride": r"Foot ahead of pelvis (cm)",
    }
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL, 3.7), gridspec_kw={"wspace": 0.55, "hspace": 0.38})
    for k, (ax, m) in enumerate(zip(axes.flat, PLOT_METRICS)):
        d = wide[wide.metric == m]
        for _, r in d.iterrows():
            ax.plot(SPEEDS, r[SPEEDS].values.astype(float), color=COLORS["light"], lw=0.5, zorder=1)
        mu, sd = d[SPEEDS].mean(), d[SPEEDS].std()
        ax.errorbar(SPEEDS, mu, yerr=sd, color="k", lw=1.0, marker="o", ms=3, capsize=2, capthick=0.6,
                    elinewidth=0.6, zorder=3)
        w = t.set_index("metric").loc[m, "kendall_w"]
        ax.set_title(f"$W = {w:.2f}$", loc="right", fontsize=7, pad=3)
        ax.set_ylabel(ylab[m])
        ax.set_xticks(SPEEDS)
        ax.set_xlim(2.2, 4.8)
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        if k >= 4:
            ax.set_xlabel(r"Speed (m s$^{-1}$)")
        panel_label(ax, "abcdefgh"[k], x=-0.02, y=1.03)
    save(fig, RES / "figures" / "fig_speed_effects")

    # Figure 2: group-mean curves per speed
    with open(RES / "cache" / "results.pkl", "rb") as fh:
        results = pickle.load(fh)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 1.9), gridspec_kw={"wspace": 0.35})
    x = np.arange(101)
    ls = [(0, (1, 1.2)), (0, (4, 1.5)), "-"]
    for sp, c, l_ in zip(SPEEDS, COLORS["speeds"], ls):
        for ax, key in [(axes[0], "R.hip.flex"), (axes[1], "R.knee.flex")]:
            cur = np.array([r.curves[key][0] for (sub, s_), r in results.items() if s_ == sp and key in r.curves])
            ax.plot(x, cur.mean(0), color=c, ls=l_, lw=1.1)
        g = np.array([r.vgrf_curves["R"][0] for (sub, s_), r in results.items() if s_ == sp and "R" in r.vgrf_curves])
        axes[2].plot(x, g.mean(0), color=c, ls=l_, lw=1.1, label=f"{sp:.1f}" + r" m s$^{-1}$")
    axes[0].set_ylabel(r"Hip flexion ($^\circ$)")
    axes[1].set_ylabel(r"Knee flexion ($^\circ$)")
    axes[2].set_ylabel("Vertical GRF (BW)")
    for ax in axes[:2]:
        ax.set_xlabel(tex("Gait cycle (%)"))
    axes[2].set_xlabel(tex("Stance (%)"))
    for k, ax in enumerate(axes):
        ax.set_xlim(0, 100)
        ax.minorticks_on()
        panel_label(ax, "abc"[k], x=-0.25)
    axes[2].legend(loc="lower center", fontsize=6.5, handlelength=2.2)
    save(fig, RES / "figures" / "fig_group_curves")


if __name__ == "__main__":
    main()
