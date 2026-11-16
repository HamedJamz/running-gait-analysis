"""Gait event detection: initial contact (IC) and toe-off (TO).

Two independent methods are implemented so they can validate each other:

* ``force_events``   - gold standard on an instrumented treadmill: threshold on the
                        vertical ground reaction force. Both feet load the same plate,
                        so each contact is assigned to the foot that is lower at mid-stance.
* ``kinematic_events`` - marker-only method (usable without force plates, e.g. in
                        overground or field settings): a contact starts when the lowest
                        foot marker comes within a threshold of its standing height and
                        ends when the forefoot leaves that zone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signal import _runs, lowpass


def force_events(
    fy: np.ndarray,
    analog_rate: float,
    marker_rate: float,
    foot_height: dict[str, np.ndarray],
    threshold: float = 50.0,
    min_contact_s: float = 0.08,
    max_contact_s: float = 0.6,
) -> pd.DataFrame:
    """Detect contacts from the vertical GRF and assign them to a side.

    fy: vertical GRF [N] at ``analog_rate``. foot_height: side -> (n_frames,)
    vertical position of the foot centre at ``marker_rate``.
    Returns one row per contact with IC/TO in seconds and marker-frame indices.
    """
    fyf = lowpass(fy, analog_rate, 30.0)
    rows = []
    for a, b in _runs(fyf > threshold):
        if a == 0 or b >= len(fyf):          # contact cut by the trial start/end
            continue
        dur = (b - a) / analog_rate
        if not (min_contact_s <= dur <= max_contact_s):
            continue
        t_ic, t_to = a / analog_rate, b / analog_rate
        mid = int(round((t_ic + t_to) / 2 * marker_rate))
        side = min(foot_height, key=lambda s: foot_height[s][mid])
        rows.append(dict(side=side, ic=t_ic, to=t_to))
    ev = pd.DataFrame(rows)
    return _finalise(ev, marker_rate)


def kinematic_events(
    foot_markers: dict[str, dict[str, np.ndarray]],
    marker_rate: float,
    ic_margin: float = 5.0,
    to_margin: float = 10.0,
    forefoot=("MT1", "MT5"),
) -> pd.DataFrame:
    """Marker-only event detection from foot marker heights.

    foot_markers: side -> {marker name -> (n_frames,) vertical position in mm}.
    Each marker's ground level is estimated from the trial itself (2nd
    percentile of its height), so the method does not rely on the standing
    trial floor being at the same height as the treadmill belt.
    IC: first frame in which any foot marker is within ``ic_margin`` of its
    ground level. TO: first frame after IC (and after forefoot loading) in which
    all forefoot markers are more than ``to_margin`` above their ground level.
    """
    rows = []
    for s, mk in foot_markers.items():
        rel = {m: h - np.percentile(h, 2) for m, h in mk.items()}
        low = np.min(np.stack(list(rel.values())), axis=0)
        fore = np.min(np.stack([rel[m] for m in forefoot]), axis=0)
        down = low < ic_margin
        prev_to = -1
        for a, _ in _runs(down):
            if a == 0 or a <= prev_to:
                continue
            # the forefoot must first be on the ground (rearfoot strikers land
            # heel first) and then rise above the margin
            on = np.where(fore[a:] <= to_margin)[0]
            if len(on) == 0:
                continue
            up = np.where(fore[a + on[0]:] > to_margin)[0]
            if len(up) == 0:
                continue
            b = a + on[0] + up[0]
            if 0.08 <= (b - a) / marker_rate <= 0.6:
                rows.append(dict(side=s, ic=a / marker_rate, to=b / marker_rate))
                prev_to = b
    return _finalise(pd.DataFrame(rows), marker_rate)


def _finalise(ev: pd.DataFrame, marker_rate: float) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame(columns=["side", "ic", "to", "ic_frame", "to_frame"])
    ev = ev.sort_values("ic").reset_index(drop=True)
    ev["ic_frame"] = np.round(ev["ic"] * marker_rate).astype(int)
    ev["to_frame"] = np.round(ev["to"] * marker_rate).astype(int)
    return ev


def match_events(ref: pd.DataFrame, test: pd.DataFrame, tol: float = 0.1) -> pd.DataFrame:
    """Pair each reference contact with the nearest test contact of the same side.

    Returns IC and TO differences (test - reference) in milliseconds.
    """
    out = []
    for _, r in ref.iterrows():
        cand = test[test.side == r.side]
        if cand.empty:
            continue
        j = (cand.ic - r.ic).abs().idxmin()
        if abs(cand.ic[j] - r.ic) <= tol:
            out.append(dict(side=r.side, d_ic_ms=1000 * (cand.ic[j] - r.ic), d_to_ms=1000 * (cand.to[j] - r.to)))
    return pd.DataFrame(out)
