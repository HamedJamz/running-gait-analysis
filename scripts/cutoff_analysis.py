"""Winter's residual analysis over all running trials to choose the marker cut-off frequency.

Usage: python scripts/cutoff_analysis.py  ->  results/residual_analysis.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rungait.io import find_trials, load_c3d  # noqa: E402
from rungait.signal import residual_analysis  # noqa: E402

MARKERS = ["R.Heel.Bottom", "R.MT5", "R.Shank.Top.Lateral", "R.Thigh.Bottom.Lateral", "R.ASIS"]

rows = []
trials = find_trials(ROOT / "data" / "raw")
for _, r in trials[trials.kind == "run"].iterrows():
    t = load_c3d(r.path, with_forces=False)
    for m in MARKERS:
        x = t.markers.get(m)
        if x is None or np.isnan(x).any():      # use complete trajectories only
            continue
        for ax, name in ((0, "anterior"), (1, "vertical")):
            fc, _, _ = residual_analysis(x[:, ax], t.rate)
            rows.append(dict(subject=r.subject, speed=r.speed, marker=m, axis=name, fc=fc))
d = pd.DataFrame(rows)
(ROOT / "results").mkdir(exist_ok=True)
d.to_csv(ROOT / "results" / "residual_analysis.csv", index=False)
print(d.groupby("speed").fc.describe().round(1))
print(f"overall median {d.fc.median():.1f} Hz, IQR {d.fc.quantile(.25):.1f}-{d.fc.quantile(.75):.1f} Hz, n = {len(d)}")
