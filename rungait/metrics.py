"""Step-level running metrics, time normalisation and symmetry."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

G = 9.81

# Foot-strike angle thresholds (deg) after Altman & Davis (2012), Gait & Posture 35:298-300
FSA_REARFOOT = 8.0
FSA_FOREFOOT = -1.6

# Metrics reported per side and used for symmetry / speed comparisons
KEY_METRICS = {
    "step_freq": ("Step frequency", "steps/min"),
    "step_length": ("Step length", "m"),
    "contact_time": ("Contact time", "ms"),
    "flight_time": ("Flight time", "ms"),
    "duty_factor": ("Duty factor", "%"),
    "vertical_osc": ("Vertical oscillation (pelvis)", "cm"),
    "peak_vgrf": ("Peak vertical GRF", "BW"),
    "loading_rate": ("Vertical average loading rate", "BW/s"),
    "strike_angle": ("Foot-strike angle", "deg"),
    "knee_flex_ic": ("Knee flexion at contact", "deg"),
    "knee_flex_peak": ("Peak knee flexion (stance)", "deg"),
    "hip_add_peak": ("Peak hip adduction (stance)", "deg"),
    "ankle_dorsi_peak": ("Peak ankle dorsiflexion (stance)", "deg"),
    "overstride": ("Foot ahead of pelvis at contact", "cm"),
}


def classify_strike(angle: float) -> str:
    if np.isnan(angle):
        return "n/a"
    if angle > FSA_REARFOOT:
        return "Rearfoot"
    if angle < FSA_FOREFOOT:
        return "Forefoot"
    return "Midfoot"


def symmetry_index(right: float, left: float) -> float:
    """Symmetry index (Robinson et al. 1987): 100 * (R - L) / mean(R, L). 0 = perfect symmetry."""
    m = 0.5 * (right + left)
    return float(100 * (right - left) / m) if m else np.nan


def time_normalise(x: np.ndarray, start: int, stop: int, n: int = 101) -> np.ndarray:
    seg = x[start : stop + 1]
    return np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg)


def loading_rate(stance_f: np.ndarray, fs: float, bw: float):
    """Vertical average loading rate (BW/s) between 20 % and 80 % of the impact peak.

    If there is no distinct impact peak (typical for fore-/midfoot strikers),
    the force at 13 % of stance is used as the reference instead. Returns
    (loading_rate, impact_peak_in_BW_or_nan, impact_peak_present).
    """
    f = stance_f / bw
    n = len(f)
    peaks, _ = find_peaks(f, prominence=0.05)
    active = np.argmax(f)
    early = [p for p in peaks if p < 0.3 * n and p < active and f[p] < 0.95 * f[active]]
    if early:
        ref_i, present = early[0], True
    else:
        ref_i, present = max(int(round(0.13 * n)), 2), False
    ref = f[ref_i]
    rise = f[: ref_i + 1]
    i20 = np.argmax(rise >= 0.2 * ref)
    i80 = np.argmax(rise >= 0.8 * ref)
    if i80 <= i20:
        return np.nan, (ref if present else np.nan), present
    rate = (f[i80] - f[i20]) / ((i80 - i20) / fs)
    return float(rate), (float(ref) if present else np.nan), present


def step_table(
    events: pd.DataFrame,
    angles: dict[str, np.ndarray],
    foot_pitch: dict[str, np.ndarray],
    static_pitch: dict[str, float],
    pelvis_centre: np.ndarray,
    ajc: dict[str, np.ndarray],
    fy: np.ndarray,
    analog_rate: float,
    marker_rate: float,
    body_mass: float,
    speed: float,
    invalid: np.ndarray,
) -> pd.DataFrame:
    """One row per foot contact with temporal, kinetic and kinematic metrics."""
    bw = body_mass * G
    ev = events.reset_index(drop=True)
    rows = []
    for i in range(len(ev) - 1):
        e, nxt = ev.loc[i], ev.loc[i + 1]
        s = e["side"]
        same = ev[(ev.side == s) & (ev.index > i)]
        if same.empty:
            continue
        nxt_same = same.iloc[0]
        # note: item access, because Series.to_frame is a pandas method
        ic, to, n_ic = e["ic"], e["to"], nxt["ic"]
        a, b, c = int(e["ic_frame"]), int(e["to_frame"]), int(nxt_same["ic_frame"])
        valid = (nxt["side"] != s) and not invalid[a : c + 1].any()

        st_t = n_ic - ic
        stance = fy[int(round(ic * analog_rate)) : int(round(to * analog_rate)) + 1]
        lr, impact, has_impact = loading_rate(stance, analog_rate, bw)
        knee = angles[f"{s}.knee.flex"]
        rows.append(
            dict(
                side=s, ic=ic, to=to, ic_frame=a, to_frame=b, next_ic_frame=c, valid=valid,
                contact_time=1000 * (to - ic),
                flight_time=1000 * (n_ic - to),
                step_time=1000 * st_t,
                step_freq=60.0 / st_t,
                step_length=speed * st_t,
                duty_factor=100 * (to - ic) / (nxt_same["ic"] - ic),
                vertical_osc=(pelvis_centre[a : c + 1, 1].max() - pelvis_centre[a : c + 1, 1].min()) / 10,
                peak_vgrf=stance.max() / bw,
                impact_peak=impact,
                impact_peak_present=has_impact,
                loading_rate=lr,
                strike_angle=foot_pitch[s][a] - static_pitch[s],
                knee_flex_ic=knee[a],
                knee_flex_peak=knee[a : b + 1].max(),
                knee_excursion=knee[a : b + 1].max() - knee[a],
                hip_flex_ic=angles[f"{s}.hip.flex"][a],
                hip_add_peak=angles[f"{s}.hip.add"][a : b + 1].max(),
                ankle_dorsi_peak=angles[f"{s}.ankle.flex"][a : b + 1].max(),
                overstride=(ajc[s][a, 0] - pelvis_centre[a, 0]) / 10,
            )
        )
    df = pd.DataFrame(rows)
    if len(df):
        df["strike_type"] = df.strike_angle.map(classify_strike)
    return df


# Metrics on an interval scale (zero is arbitrary or values change sign): a
# ratio-based symmetry index is meaningless there, so the R-L difference is used.
DIFFERENCE_METRICS = {"strike_angle", "knee_flex_ic", "knee_flex_peak", "hip_add_peak",
                      "ankle_dorsi_peak", "overstride"}


def summarise(steps: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD per side for valid steps, plus asymmetry.

    ``SI`` (symmetry index, %) for ratio-scale metrics; ``diff`` (R - L, in the
    metric's unit) for angles and positions.
    """
    v = steps[steps.valid]
    out = []
    for m in KEY_METRICS:
        r, l = v.loc[v.side == "R", m], v.loc[v.side == "L", m]
        out.append(
            dict(
                metric=m, R_mean=r.mean(), R_sd=r.std(), L_mean=l.mean(), L_sd=l.std(),
                mean=v[m].mean(),
                SI=np.nan if m in DIFFERENCE_METRICS else symmetry_index(r.mean(), l.mean()),
                diff=r.mean() - l.mean(),
                n_R=int(r.notna().sum()), n_L=int(l.notna().sum()),
            )
        )
    return pd.DataFrame(out).set_index("metric")


def mean_curves(steps: pd.DataFrame, signals: dict[str, np.ndarray], n: int = 101) -> dict:
    """Gait-cycle (IC to next IC of the same foot) normalised mean and SD curves.

    ``signals`` keys must start with the side, e.g. 'R.knee.flex'.
    Returns {key: (mean(n), sd(n), n_cycles)}.
    """
    out = {}
    v = steps[steps.valid]
    for key, x in signals.items():
        side = key.split(".")[0]
        cyc = [time_normalise(x, a, c, n) for a, c in zip(v[v.side == side].ic_frame, v[v.side == side].next_ic_frame)]
        if cyc:
            arr = np.array(cyc)
            out[key] = (arr.mean(0), arr.std(0), len(arr))
    return out


def stance_vgrf_curves(steps: pd.DataFrame, fy: np.ndarray, analog_rate: float, bw: float, n: int = 101) -> dict:
    out = {}
    v = steps[steps.valid]
    for s in ("R", "L"):
        cyc = [
            time_normalise(fy / bw, int(round(a * analog_rate)), int(round(b * analog_rate)), n)
            for a, b in zip(v[v.side == s].ic, v[v.side == s].to)
        ]
        if cyc:
            arr = np.array(cyc)
            out[s] = (arr.mean(0), arr.std(0), len(arr))
    return out
