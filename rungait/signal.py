"""Signal processing: gap filling, zero-lag Butterworth filtering, residual analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def fill_gaps(x: np.ndarray, max_gap: int = 10) -> tuple[np.ndarray, int]:
    """Fill interior gaps of up to ``max_gap`` frames with a cubic spline.

    Works column-wise on an (n, k) array. Returns the filled array and the
    number of samples that were filled. Longer gaps are left as NaN so that
    they are visible in the quality report instead of being silently invented.
    """
    x = np.asarray(x, dtype=float)
    df = pd.DataFrame(x.reshape(len(x), -1))
    n_missing_before = int(df.isna().any(axis=1).sum())
    if n_missing_before == 0:
        return x, 0
    method = "cubic" if df.notna().sum().min() > 4 else "linear"
    filled = df.interpolate(method=method, limit=max_gap, limit_area="inside")
    # pandas `limit` fills the first `limit` samples of a longer gap; undo that
    for col in df.columns:
        isna = df[col].isna().to_numpy()
        if not isna.any():
            continue
        runs = _runs(isna)
        for start, stop in runs:
            if stop - start > max_gap:
                filled.iloc[start:stop, col] = np.nan
    n_missing_after = int(filled.isna().any(axis=1).sum())
    return filled.to_numpy().reshape(x.shape).copy(), n_missing_before - n_missing_after


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop indices of consecutive True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def gap_lengths(mask: np.ndarray) -> list[int]:
    return [b - a for a, b in _runs(mask)]


def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """Zero-lag (forward-backward) Butterworth low-pass filter along axis 0.

    Note: filtfilt doubles the effective order; the -3 dB point moves slightly
    below ``fc``. For the cut-offs used here this is negligible and is the
    common convention in gait analysis software.
    """
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise ValueError("lowpass() received NaNs - fill gaps first")
    return filtfilt(b, a, x, axis=0)


def residual_analysis(x: np.ndarray, fs: float, cutoffs=None) -> tuple[float, np.ndarray, np.ndarray]:
    """Winter's residual analysis to choose a low-pass cut-off frequency.

    The RMS residual between raw and filtered signal is computed for a range of
    cut-offs. A straight line fitted to the high-frequency (noise-only) part of
    the curve is extrapolated to 0 Hz; its intercept is the noise level, and the
    optimal cut-off is where the residual curve falls to that level.
    Returns (optimal_fc, cutoffs, residuals).
    """
    if cutoffs is None:
        cutoffs = np.arange(1.0, min(fs / 2.0 - 1, 40.0), 0.5)
    x = np.asarray(x, dtype=float)
    res = np.array([np.sqrt(np.mean((x - lowpass(x, fs, fc)) ** 2)) for fc in cutoffs])
    tail = cutoffs >= cutoffs[0] + 0.6 * (cutoffs[-1] - cutoffs[0])
    slope, intercept = np.polyfit(cutoffs[tail], res[tail], 1)
    above = np.where(res <= intercept)[0]
    fc_opt = float(cutoffs[above[0]]) if len(above) else float(cutoffs[-1])
    return fc_opt, cutoffs, res
