"""Publication-style figures: LaTeX typesetting, journal sizes, colour-blind safe palette.

If a LaTeX installation (latex + dvipng + cm-super) is found, all text is
typeset by LaTeX (Computer Modern). Otherwise matplotlib's built-in Computer
Modern math fonts are used, so the scripts still run without LaTeX.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe & Ito (2008) colour-blind safe palette; speeds in grey levels so that
# the figures remain readable in black-and-white print.
COLORS = {
    "R": "#0072B2",                                # right: blue
    "L": "#D55E00",                                # left: vermillion
    "speeds": ["#a6a6a6", "#5a5a5a", "#000000"],   # 2.5, 3.5, 4.5 m/s
    "grey": "#8c8c8c",
    "light": "#d9d9d9",
}

# Journal widths (inches): single and double column (e.g. J Biomech, Gait & Posture)
SINGLE_COL = 3.54
DOUBLE_COL = 7.25


def latex_available() -> bool:
    return all(shutil.which(x) for x in ("latex", "dvipng"))


def style(usetex: bool | None = None):
    if usetex is None:
        usetex = latex_available()
    rc = {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.top": True,
        "ytick.right": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    }
    if usetex:
        rc.update({
            "text.usetex": True,
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{siunitx}\sisetup{detect-all}",
        })
    else:
        rc.update({"text.usetex": False, "font.serif": ["CMU Serif", "DejaVu Serif"], "mathtext.fontset": "cm"})
    mpl.rcParams.update(rc)
    return usetex


def tex(s: str) -> str:
    """Escape characters that LaTeX treats specially (only when usetex is on)."""
    if not mpl.rcParams["text.usetex"]:
        return s
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("#", r"\#")


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left")


def save(fig, path_without_suffix, formats=("pdf", "png")):
    """Save vector (PDF, for papers) and raster (PNG, for the README) versions."""
    p = Path(path_without_suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(p.with_suffix(f".{f}"))
    plt.close(fig)
