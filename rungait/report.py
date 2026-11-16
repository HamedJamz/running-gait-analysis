"""One-page participant report (A4, PDF/PNG), typeset with LaTeX.

Content: key metrics at three speeds, left/right asymmetry, joint-angle and
force curves, position relative to the sample, automatically generated
observations and a data-quality statement.
"""
from __future__ import annotations

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from .metrics import DIFFERENCE_METRICS, KEY_METRICS
from .plotting import COLORS, style

SPEEDS = [2.5, 3.5, 4.5]
REF_SPEED = 3.5
SI_FLAG = 10.0      # % - screening threshold for ratio metrics (illustrative)
DIFF_FLAG = 5.0     # deg or cm - screening threshold for angle/position metrics (illustrative)

TABLE_METRICS = ["step_freq", "step_length", "contact_time", "flight_time", "vertical_osc",
                 "peak_vgrf", "loading_rate", "strike_angle", "knee_flex_ic", "knee_flex_peak",
                 "hip_add_peak", "overstride"]
UNITS_TEX = {"steps/min": r"min$^{-1}$", "m": "m", "ms": "ms", "%": r"\%", "cm": "cm", "BW": "BW",
             "BW/s": r"BW\,s$^{-1}$", "deg": r"$^\circ$"}
CONTEXT_METRICS = ["step_freq", "contact_time", "vertical_osc", "loading_rate", "knee_flex_peak", "overstride"]
SHORT_NAMES = {"step_freq": "Step frequency", "contact_time": "Contact time", "vertical_osc": "Vert. oscillation",
               "loading_rate": "Loading rate", "knee_flex_peak": "Peak knee flexion", "overstride": "Foot ahead of pelvis"}


# --------------------------------------------------------------------------- text helpers
def tx(s: str) -> str:
    """Make plain text safe for LaTeX (no-op without usetex)."""
    s = str(s)
    if not mpl.rcParams["text.usetex"]:
        return s
    rep = [("\\", r"\textbackslash{}"), ("<->", r"$\leftrightarrow$"), ("%", r"\%"), ("&", r"\&"),
           ("_", r"\_"), ("#", r"\#"), ("|", r"$|$"), (">", r"$>$"), ("<", r"$<$"), ("±", r"$\pm$"), ("·", r"$\cdot$"),
           ("−", r"$-$"), ("–", "--"), ("°", r"$^\circ$")]
    for a, b in rep:
        s = s.replace(a, b)
    return s


def _ordinal(x: float) -> str:
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _fmt(v, m):
    if pd.isna(v):
        return "--"
    if m in ("step_freq", "contact_time", "flight_time"):
        s = f"{v:.0f}"
    elif m in ("step_length", "peak_vgrf"):
        s = f"{v:.2f}"
    else:
        s = f"{v:.1f}"
    return f"${s}$" if mpl.rcParams["text.usetex"] else s


def _asym(summary_row, m):
    """(value, LaTeX string, flagged) for the asymmetry column."""
    if m in DIFFERENCE_METRICS:
        v = summary_row["diff"]
        return v, f"${v:+.1f}$", abs(v) > DIFF_FLAG
    v = summary_row["SI"]
    return v, f"${v:+.1f}$" + r"\,\%", abs(v) > SI_FLAG


# --------------------------------------------------------------------------- observations
def observations(subject_results: dict, group: pd.DataFrame, subject: str) -> list[str]:
    obs = []
    ref = subject_results[REF_SPEED]
    strikes = {sp: r.steps[r.steps.valid].strike_type.mode().iloc[0] for sp, r in subject_results.items()}
    if len(set(strikes.values())) == 1:
        obs.append(f"Foot strike: {next(iter(strikes.values())).lower()} pattern at all three speeds "
                   "(foot-strike angle, Altman & Davis 2012).")
    else:
        obs.append("Foot strike changes with speed: " + ", ".join(f"{sp} m/s {s.lower()}" for sp, s in strikes.items()) + ".")
    flagged = []
    for m in TABLE_METRICS:
        _, _, flag = _asym(ref.summary.loc[m], m)
        if flag:
            name, unit = KEY_METRICS[m]
            unit = "°" if unit == "deg" else " " + unit
            r_, l_ = ref.summary.loc[m, "R_mean"], ref.summary.loc[m, "L_mean"]
            flagged.append(f"{name.lower()} (right {r_:.1f}{unit}, left {l_:.1f}{unit})")
    if flagged:
        obs.append(f"Left/right differences above the screening threshold at {REF_SPEED} m/s: " + "; ".join(flagged) + ".")
    else:
        obs.append(f"No left/right difference above the screening thresholds at {REF_SPEED} m/s.")
    lo, hi = subject_results[SPEEDS[0]].summary, subject_results[SPEEDS[-1]].summary
    f_ch = 100 * (hi.loc["step_freq", "mean"] / lo.loc["step_freq", "mean"] - 1)
    l_ch = 100 * (hi.loc["step_length", "mean"] / lo.loc["step_length", "mean"] - 1)
    gfreq = group[group.metric == "step_freq"].pivot(index="subject", columns="speed", values="mean")
    g_ch = 100 * (gfreq[SPEEDS[-1]] / gfreq[SPEEDS[0]] - 1)
    obs.append(f"From {SPEEDS[0]} to {SPEEDS[-1]} m/s, step length increases by {l_ch:.0f}% and step frequency by "
               f"{f_ch:.0f}% (sample median for step frequency: {g_ch.median():.0f}%).")
    lr = group[(group.metric == "loading_rate") & (group.speed == REF_SPEED)].set_index("subject")["mean"]
    obs.append(f"Vertical average loading rate at {REF_SPEED} m/s: {lr[subject]:.0f} BW/s "
               f"({_ordinal((lr < lr[subject]).mean() * 100)} percentile of the {len(lr)} runners in the sample).")
    return obs


# --------------------------------------------------------------------------- figure
def _booktabs_table(ax, header, rows, col_x, col_align, fontsize=7.2, row_h=None):
    ax.axis("off")
    n = len(rows) + 1
    row_h = row_h or 1.0 / (n + 0.6)
    y = 1.0
    ax.plot([0, 1], [y, y], color="k", lw=0.9, transform=ax.transAxes, clip_on=False)
    y -= row_h * 0.8
    for x, h, al in zip(col_x, header, col_align):
        ax.text(x, y, h, transform=ax.transAxes, ha=al, va="center", fontsize=fontsize)
    y -= row_h * 0.7
    ax.plot([0, 1], [y, y], color="k", lw=0.5, transform=ax.transAxes, clip_on=False)
    y -= row_h * 0.6
    for r in rows:
        for x, v, al in zip(col_x, r, col_align):
            ax.text(x, y, v, transform=ax.transAxes, ha=al, va="center", fontsize=fontsize)
        y -= row_h
    y += row_h * 0.45
    ax.plot([0, 1], [y, y], color="k", lw=0.9, transform=ax.transAxes, clip_on=False)
    return y


def build_report(subject: str, subject_results: dict, info: pd.Series, group: pd.DataFrame, out_path):
    style()
    ref = subject_results[REF_SPEED]
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(5, 3, figure=fig, height_ratios=[0.36, 2.25, 1.3, 1.3, 1.45],
                  hspace=0.5, wspace=0.42, left=0.1, right=0.95, top=0.96, bottom=0.035)

    # header
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    T = ax.transAxes
    ax.text(0, 1.0, tx(f"Instrumented running analysis – participant {subject}"), fontsize=14, va="top", transform=T)
    inj = info.get("InjuryLoc", "-")
    inj = "none reported" if inj in ("-", None) or pd.isna(inj) else str(inj).lower()
    ax.text(0, 0.38, tx(f"{'Male' if info['Gender'] == 'M' else 'Female'}, {info['Age']} years, "
                        f"{info['Height']:.0f} cm, {info['Mass']:.1f} kg · {info['Level']} runner, weekly volume "
                        f"{info['Volume']} · previous injury: {inj}"), fontsize=8.5, va="top", transform=T)
    ax.text(0, -0.02, tx("Treadmill running at 2.5, 3.5 and 4.5 m/s · optical motion capture (150 Hz) "
                        "and instrumented treadmill (300 Hz)"), fontsize=8, va="top", color="#444", transform=T)
    ax.plot([0, 1], [-0.42, -0.42], color="k", lw=0.6, transform=ax.transAxes, clip_on=False)

    # table
    ax = fig.add_subplot(gs[1, :2])
    rows = []
    for m in TABLE_METRICS:
        name, unit = KEY_METRICS[m]
        _, a_txt, flag = _asym(ref.summary.loc[m], m)
        rows.append([tx(name), UNITS_TEX[unit]] + [_fmt(subject_results[sp].summary.loc[m, "mean"], m) for sp in SPEEDS]
                    + [a_txt + (r"$^{\dagger}$" if flag else r"$^{\phantom{\dagger}}$")])
    header = ["Variable", "Unit", r"2.5 m\,s$^{-1}$", r"3.5 m\,s$^{-1}$", r"4.5 m\,s$^{-1}$", r"R vs.\ L$^{*}$"]
    yb = _booktabs_table(ax, header, rows, [0.0, 0.47, 0.64, 0.76, 0.88, 1.0],
                         ["left", "left", "right", "right", "right", "right"], fontsize=7.4)
    notes = ["Values are means of all valid steps (both sides).",
             f"* At {REF_SPEED} m/s: symmetry index 100(R−L)/mean(R,L) for ratio-scale variables;",
             "   R−L difference for angles and positions.",
             f"† Above the illustrative screening threshold (|SI| > {SI_FLAG:.0f}%, |R−L| > {DIFF_FLAG:.0f}° or {DIFF_FLAG:.0f} cm)."]
    for k, n_ in enumerate(notes):
        ax.text(0, yb - 0.03 - 0.034 * k, tx(n_), transform=ax.transAxes, fontsize=6.3, va="top", color="#333")
    ax.text(-0.02, 1.04, "(a)", transform=ax.transAxes, fontsize=9)

    # position in sample
    ax = fig.add_subplot(gs[1, 2].subgridspec(1, 2, width_ratios=[0.55, 1], wspace=0)[0, 1])
    gref = group[group.speed == REF_SPEED]
    ys = np.arange(len(CONTEXT_METRICS))[::-1]
    for y, m in zip(ys, CONTEXT_METRICS):
        vals = gref[gref.metric == m].set_index("subject")["mean"]
        z = (vals - vals.mean()) / vals.std()
        ax.plot(z.drop(subject), np.full(len(z) - 1, y), "|", color=COLORS["grey"], ms=6, mew=0.6)
        ax.plot(z[subject], y, "o", color="k", ms=4.5)
    ax.axvline(0, color="k", lw=0.4, ls=(0, (2, 2)))
    ax.set_yticks(ys, [tx(SHORT_NAMES[m]) for m in CONTEXT_METRICS])
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_xlabel(r"$z$-score ($n = 28$)")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-0.7, len(CONTEXT_METRICS) - 0.3)
    ax.text(-1.15, 1.04, "(b)", transform=ax.transAxes, fontsize=9)
    ax.set_title(tx(f"Position in sample, {REF_SPEED} m/s"), fontsize=7.5, pad=4)

    # joint angle curves
    x = np.arange(101)
    to = ref.summary.loc["duty_factor", "mean"]
    for i, (j, lab) in enumerate([("hip", "Hip flexion"), ("knee", "Knee flexion"), ("ankle", "Ankle dorsiflexion")]):
        ax = fig.add_subplot(gs[2, i])
        for s in "RL":
            m_, sd, n = ref.curves[f"{s}.{j}.flex"]
            ax.fill_between(x, m_ - sd, m_ + sd, color=COLORS[s], alpha=0.2, lw=0)
            ax.plot(x, m_, color=COLORS[s], lw=1.0, ls="-" if s == "R" else (0, (4, 1.5)),
                    label={"R": "Right", "L": "Left"}[s])
        ax.axvline(to, color="k", lw=0.4, ls=(0, (1, 1.5)))
        ax.set_xlim(0, 100)
        ax.minorticks_on()
        ax.set_xlabel(tx("Gait cycle (%)"))
        ax.set_ylabel(lab + r" ($^\circ$)")
        ax.text(-0.3, 1.04, f"({'cde'[i]})", transform=ax.transAxes, fontsize=9)

    # GRF, knee by speed, foot-strike angle
    ax = fig.add_subplot(gs[3, 0])
    for s in "RL":
        m_, sd, n = ref.vgrf_curves[s]
        ax.fill_between(x, m_ - sd, m_ + sd, color=COLORS[s], alpha=0.2, lw=0)
        ax.plot(x, m_, color=COLORS[s], lw=1.0, ls="-" if s == "R" else (0, (4, 1.5)),
                label={"R": "Right", "L": "Left"}[s])
    ax.legend(loc="lower center", fontsize=6.5, handlelength=2.2)
    ax.set_xlim(0, 100)
    ax.minorticks_on()
    ax.set_xlabel(tx("Stance (%)"))
    ax.set_ylabel("Vertical GRF (BW)")
    ax.text(-0.3, 1.04, "(f)", transform=ax.transAxes, fontsize=9)

    ax = fig.add_subplot(gs[3, 1])
    for sp, c, l_ in zip(SPEEDS, COLORS["speeds"], [(0, (1, 1.2)), (0, (4, 1.5)), "-"]):
        m_ = np.mean([subject_results[sp].curves[f"{s}.knee.flex"][0] for s in "RL"], axis=0)
        ax.plot(x, m_, color=c, ls=l_, lw=1.0, label=f"{sp}" + r" m\,s$^{-1}$")
    ax.legend(loc="upper left", fontsize=6.3, handlelength=2.2)
    ax.set_xlim(0, 100)
    ax.minorticks_on()
    ax.set_xlabel(tx("Gait cycle (%)"))
    ax.set_ylabel(r"Knee flexion ($^\circ$)")
    ax.text(-0.3, 1.04, "(g)", transform=ax.transAxes, fontsize=9)

    ax = fig.add_subplot(gs[3, 2])
    all_fsa = []
    for k, sp in enumerate(SPEEDS):
        st = subject_results[sp].steps
        st = st[st.valid]
        for s, off in (("R", -0.13), ("L", 0.13)):
            v = st[st.side == s].strike_angle.to_numpy()
            all_fsa.extend(v.tolist())
            ax.plot(np.full(len(v), k + off) + np.random.default_rng(k).uniform(-0.05, 0.05, len(v)), v,
                    "o", ms=1.6, color=COLORS[s], alpha=0.6, mew=0)
    lo_, hi_ = min(np.nanmin(all_fsa) - 3, -6), max(np.nanmax(all_fsa) + 3, 13)
    for yv in (8.0, -1.6):
        ax.axhline(yv, color="k", lw=0.4, ls=(0, (2, 2)))
    ax.text(2.45, 8.4, "RFS", fontsize=6, ha="right", va="bottom")
    ax.text(2.45, 7.6, "MFS", fontsize=6, ha="right", va="top")
    ax.text(2.45, -2.0, "FFS", fontsize=6, ha="right", va="top")
    ax.set_ylim(lo_, hi_)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(range(3), [f"{s}" for s in SPEEDS])
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlabel(r"Speed (m\,s$^{-1}$)")
    ax.set_ylabel(r"Foot-strike angle ($^\circ$)")
    ax.text(-0.3, 1.04, "(h)", transform=ax.transAxes, fontsize=9)

    # observations + QC
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    ax.plot([0, 1], [1.02, 1.02], color="k", lw=0.6, transform=ax.transAxes, clip_on=False)
    ax.text(0, 0.98, "Observations (generated automatically)", fontsize=9, va="top", transform=ax.transAxes)
    y = 0.86
    for o in observations(subject_results, group, subject):
        for k, line in enumerate(textwrap.wrap(o, 140)):
            ax.text(0.015 if k else 0, y, (r"$\bullet$\ " if k == 0 else "") + tx(line), fontsize=7.6, va="top",
                    transform=ax.transAxes)
            y -= 0.075
        y -= 0.02
    qc = []
    for sp in SPEEDS:
        q = subject_results[sp].qc
        filled = q["filled_spline_samples"] + q["filled_rigid_samples"] + q["filled_fallback_samples"]
        extra = []
        if any(s_["corrected"] for s_ in q["label_swaps"]):
            extra.append("label swap corrected (" + ", ".join(s_["swapped"] for s_ in q["label_swaps"] if s_["corrected"]) + ")")
        if q["absent_markers"]:
            extra.append("missing marker reconstructed (" + ", ".join(q["absent_markers"]) + ")")
        qc.append(f"{sp} m/s: {q['n_valid_steps']} valid steps; gap-filled marker samples: {filled}; "
                  f"mean vGRF/body weight {q['mean_vgrf_over_bw']:.2f}" + ("; " + "; ".join(extra) if extra else ""))
    y -= 0.03
    ax.text(0, y, "Data quality", fontsize=8, va="top", transform=ax.transAxes)
    y -= 0.075
    for line in qc:
        for k, w in enumerate(textwrap.wrap(line, 160)):
            ax.text(0.015 if k else 0, y, tx(w), fontsize=6.8, va="top", transform=ax.transAxes, color="#222")
            y -= 0.062
    ax.text(0, -0.02, tx("Educational demonstration on a public dataset (Fukuchi, Fukuchi & Duarte 2017, PeerJ 5:e3298; "
                         "CC BY 4.0).\nScreening thresholds are illustrative; this is not a clinical assessment."),
            fontsize=6.3, va="bottom", color="#555", style="italic", transform=ax.transAxes)

    with mpl.rc_context({"savefig.bbox": "standard"}):   # keep the exact A4 page size
        fig.savefig(out_path, dpi=200)
    plt.close(fig)
