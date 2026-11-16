"""Data quality control and marker pre-processing.

Every decision that changes the data (filled samples, filter cut-off) is
recorded in a QC dictionary so that it can be reported next to the results.
"""
from __future__ import annotations

import itertools

import numpy as np

from .kinematics import SegmentCalibration, rigid_fit
from .signal import fill_gaps, gap_lengths, lowpass


def detect_label_swaps(markers: dict, cal: dict[str, SegmentCalibration],
                       threshold_mm: float = 10.0, step: int = 10) -> list[dict]:
    """Detect and correct swapped marker labels within each rigid cluster.

    If the median rigid-body fit residual of a cluster is above ``threshold_mm``,
    every pairwise label swap is tested. When one swap reduces the residual to
    below the threshold (and by > 50 %), the labels are exchanged in
    ``markers`` (in place) and the correction is logged. Label swaps are a
    common error in manual or automatic marker labelling.
    """
    log = []
    for name, c in cal.items():
        B = np.stack([markers[l] for l in c.cluster_labels], axis=1)[::step]
        ok = ~np.isnan(B).any(axis=(1, 2))
        if ok.sum() < 10:
            continue
        base = float(np.median(rigid_fit(c.cluster_static, B[ok])[2]))
        if base < threshold_mm:
            continue
        best = (base, None)
        for i, j in itertools.combinations(range(len(c.cluster_labels)), 2):
            perm = list(range(len(c.cluster_labels)))
            perm[i], perm[j] = j, i
            r = float(np.median(rigid_fit(c.cluster_static, B[ok][:, perm])[2]))
            if r < best[0]:
                best = (r, (i, j))
        entry = {"segment": name, "residual_before_mm": base, "residual_after_mm": best[0], "corrected": False}
        if best[1] and best[0] < threshold_mm and best[0] < 0.5 * base:
            a, b = (c.cluster_labels[k] for k in best[1])
            markers[a], markers[b] = markers[b], markers[a]
            entry.update(corrected=True, swapped=f"{a} <-> {b}")
        log.append(entry)
    return log


def rigid_body_fill(markers: dict, cal: dict[str, SegmentCalibration]) -> int:
    """Reconstruct missing cluster markers from the remaining (>= 3) markers of the cluster.

    For each frame with a missing marker, the rigid transformation between the
    static cluster and the visible markers is estimated, and the missing marker
    is predicted from its static position. Modifies ``markers`` in place and
    returns the number of reconstructed marker samples.
    """
    n_filled = 0
    for c in cal.values():
        B = np.stack([markers[l] for l in c.cluster_labels], axis=1)       # (n, k, 3)
        missing = np.isnan(B).any(axis=2)                                   # (n, k)
        frames = np.where(missing.any(axis=1) & ((~missing).sum(axis=1) >= 3))[0]
        for f in frames:
            vis = ~missing[f]
            R, t, _ = rigid_fit(c.cluster_static[vis], B[f, vis][None])
            for j in np.where(~vis)[0]:
                markers[c.cluster_labels[j]][f] = R[0] @ c.cluster_static[j] + t[0]
                n_filled += 1
    return n_filled


def preprocess_markers(trial, cal, fc: float = 10.0, max_spline_gap: int = 10, max_fallback_gap: int = 20):
    """Gap-fill and low-pass filter all markers needed by the model.

    Steps: (0) detect/correct swapped labels, (1) cubic spline for short gaps (<= max_spline_gap frames),
    (2) rigid-body reconstruction for longer gaps and for markers that are
    absent from the file altogether, (3) spline fallback for gaps up to
    ``max_fallback_gap`` frames where fewer than three cluster markers were
    visible, (4) zero-lag Butterworth
    filter. Frames that could not be reconstructed are returned in
    ``invalid`` and are excluded from stride-level analysis.
    """
    needed = sorted({l for c in cal.values() for l in c.cluster_labels})
    absent = [l for l in needed if l not in trial.markers]
    nan_track = np.full((trial.n_frames, 3), np.nan)
    raw = {l: (trial.markers[l].copy() if l in trial.markers else nan_track.copy()) for l in needed}
    qc = {"markers": {}, "absent_markers": absent}
    for l, v in raw.items():
        miss = np.isnan(v).any(axis=1)
        gaps = gap_lengths(miss)
        qc["markers"][l] = {
            "missing_pct": 100 * miss.mean(),
            "n_gaps": len(gaps),
            "max_gap_frames": max(gaps) if gaps else 0,
        }

    swaps = detect_label_swaps(raw, cal)

    n_spline = 0
    for l in needed:
        raw[l], n = fill_gaps(raw[l], max_spline_gap)
        n_spline += n
    n_rigid = rigid_body_fill(raw, cal)
    # (3) fallback: if too few cluster markers were visible for a rigid
    # reconstruction, bridge gaps up to ``max_fallback_gap`` frames by spline
    n_fallback = 0
    for l in needed:
        raw[l], n = fill_gaps(raw[l], max_fallback_gap)
        n_fallback += n

    invalid = np.zeros(trial.n_frames, dtype=bool)
    out = {}
    for l, v in raw.items():
        miss = np.isnan(v).any(axis=1)
        invalid |= miss
        if miss.any():  # bridge only to allow filtering; these frames are flagged invalid
            idx = np.arange(len(v))
            v = np.column_stack([np.interp(idx, idx[~miss], v[~miss, k]) for k in range(3)])
        out[l] = lowpass(v, trial.rate, fc)

    qc.update(
        label_swaps=swaps,
        filled_spline_samples=n_spline,
        filled_rigid_samples=n_rigid,
        filled_fallback_samples=n_fallback,
        invalid_frames=int(invalid.sum()),
        marker_cutoff_hz=fc,
        worst_marker=max(qc["markers"], key=lambda k: qc["markers"][k]["missing_pct"]),
    )
    qc["max_missing_pct"] = qc["markers"][qc["worst_marker"]]["missing_pct"]
    return out, invalid, qc


def force_qc(grf_vertical: np.ndarray, body_mass: float) -> dict:
    """Plausibility check of the force data.

    Over many complete strides of steady treadmill running the mean vertical
    ground reaction force must equal body weight (no net vertical acceleration).
    A ratio far from 1 indicates a calibration, zeroing or scaling problem.
    """
    bw = body_mass * 9.81
    ratio = float(np.mean(grf_vertical) / bw)
    return {"body_weight_N": bw, "mean_vgrf_over_bw": ratio, "force_check_ok": 0.93 <= ratio <= 1.07}
