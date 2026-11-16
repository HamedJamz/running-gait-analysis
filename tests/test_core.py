"""Unit tests on synthetic data with known answers (run: pytest -q)."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from rungait.events import _finalise, match_events
from rungait.kinematics import joint_angles, rigid_fit
from rungait.metrics import classify_strike, symmetry_index, time_normalise
from rungait.proms import KOOS_SUBSCALES, score_koos
from rungait.signal import fill_gaps, lowpass, residual_analysis


# ---------------------------------------------------------------- kinematics
def test_rigid_fit_recovers_known_transformation():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3)) * 50
    R_true = Rotation.from_euler("ZXY", [30, -10, 45], degrees=True).as_matrix()
    t_true = np.array([100.0, -20.0, 5.0])
    B = (A @ R_true.T + t_true)[None]
    R, t, rms = rigid_fit(A, B)
    assert np.allclose(R[0], R_true, atol=1e-10)
    assert np.allclose(t[0], t_true, atol=1e-8)
    assert rms[0] < 1e-8


def test_rigid_fit_residual_reflects_noise():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 3)) * 60
    B = np.repeat(A[None], 200, axis=0) + rng.normal(scale=1.0, size=(200, 4, 3))
    _, _, rms = rigid_fit(A, B)
    assert 0.5 < np.median(rms) < 1.5     # ~ noise level (mm)


def test_rigid_fit_never_returns_reflection():
    A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    B = (A * np.array([1, 1, -1]))[None]  # mirrored cluster
    R, _, _ = rigid_fit(A, B)
    assert np.isclose(np.linalg.det(R[0]), 1.0)


@pytest.mark.parametrize("side", ["R", "L"])
def test_knee_flexion_sign(side):
    prox = np.eye(3)[None]
    # shank rotated so its anterior axis points downwards = knee flexion
    dist = Rotation.from_euler("z", -40, degrees=True).as_matrix()[None]
    ang = joint_angles(prox, dist, side, "knee")
    assert np.isclose(ang["flex"][0], 40)


def test_hip_adduction_mirrored_between_sides():
    prox = np.eye(3)[None]
    dist = Rotation.from_euler("x", 10, degrees=True).as_matrix()[None]  # thigh tilts to the right
    assert np.isclose(joint_angles(prox, dist, "R", "hip")["add"][0], 10)    # right knee moves medially
    assert np.isclose(joint_angles(prox, dist, "L", "hip")["add"][0], -10)   # left knee moves laterally


# ---------------------------------------------------------------- signal
def test_lowpass_keeps_slow_signal_and_removes_noise():
    fs = 150
    t = np.arange(0, 4, 1 / fs)
    slow = np.sin(2 * np.pi * 2 * t)
    noisy = slow + 0.2 * np.sin(2 * np.pi * 40 * t)
    out = lowpass(noisy, fs, 10)
    assert np.max(np.abs(out - slow)[50:-50]) < 0.02


def test_fill_gaps_respects_max_gap():
    x = np.sin(np.linspace(0, 6, 200))[:, None].repeat(3, axis=1)
    x[50:55] = np.nan    # short gap -> filled
    x[100:140] = np.nan  # long gap -> kept as NaN
    out, n = fill_gaps(x, max_gap=10)
    assert not np.isnan(out[50:55]).any()
    assert np.isnan(out[100:140]).all()
    assert n == 5
    assert np.allclose(out[50:55, 0], np.sin(np.linspace(0, 6, 200))[50:55], atol=1e-3)


def test_residual_analysis_returns_reasonable_cutoff():
    fs = 150
    t = np.arange(0, 6, 1 / fs)
    x = 100 * np.sin(2 * np.pi * 1.5 * t) + np.random.default_rng(2).normal(scale=1.0, size=len(t))
    fc, _, _ = residual_analysis(x, fs)
    assert 2 <= fc <= 15


# ---------------------------------------------------------------- metrics & events
def test_symmetry_index():
    assert symmetry_index(110, 90) == pytest.approx(20.0)
    assert symmetry_index(100, 100) == 0


def test_strike_classification_thresholds():
    assert classify_strike(12) == "Rearfoot"
    assert classify_strike(3) == "Midfoot"
    assert classify_strike(-5) == "Forefoot"


def test_time_normalise_length_and_endpoints():
    x = np.arange(50.0)
    y = time_normalise(x, 10, 30)
    assert len(y) == 101 and y[0] == 10 and y[-1] == 30


def test_match_events():
    ref = _finalise(pd.DataFrame({"side": ["R", "L"], "ic": [1.0, 1.4], "to": [1.2, 1.6]}), 150)
    test = _finalise(pd.DataFrame({"side": ["R", "L"], "ic": [1.01, 1.39], "to": [1.2, 1.62]}), 150)
    m = match_events(ref, test)
    assert np.allclose(sorted(m.d_ic_ms), [-10, 10])


# ---------------------------------------------------------------- PROMs
def test_koos_extremes():
    best = {i: 0 for items in KOOS_SUBSCALES.values() for i in items}
    worst = {i: 4 for i in best}
    assert set(score_koos(best).values()) == {100.0}
    assert set(score_koos(worst).values()) == {0.0}


def test_koos_missing_rule():
    a = {i: 2 for items in KOOS_SUBSCALES.values() for i in items}
    a["P1"] = None
    a["P2"] = None                       # 2 missing -> still scored
    assert score_koos(a)["Pain"] == 50.0
    a["P3"] = None                       # 3 missing -> not scored
    assert score_koos(a)["Pain"] is None


def test_koos_rejects_invalid_values():
    a = {i: 1 for items in KOOS_SUBSCALES.values() for i in items}
    a["Q1"] = 5
    with pytest.raises(ValueError):
        score_koos(a)
