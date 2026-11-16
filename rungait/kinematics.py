"""Segment kinematics from marker clusters (static calibration + rigid-body tracking).

Approach (a calibrated anatomical systems technique, CAST):
1. In the standing (static) trial, anatomical landmarks (epicondyles, malleoli,
   ...) and technical marker clusters are recorded together.
2. In each frame of the running trial, the rigid-body transformation that maps
   the static cluster onto the measured cluster is estimated by least squares
   (SVD solution of Söderkvist & Wedin 1993 / Kabsch). This is the same problem
   as a 3D Helmert transformation without scale in geodesy/photogrammetry.
3. The anatomical frames are carried into the running trial with that
   transformation, and joint angles are computed with a Cardan sequence.

All segment frames use the same axis convention as the lab:
    x = anterior, y = superior (proximal), z = right
Joint angles: Cardan sequence Z-X-Y (flexion/extension, ab/adduction, axial rotation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ("R", "L")

CLUSTERS = {
    "pelvis": ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "R.Iliac.Crest", "L.Iliac.Crest"],
    "thigh": ["Thigh.Top.Lateral", "Thigh.Top.Medial", "Thigh.Bottom.Lateral", "Thigh.Bottom.Medial"],
    "shank": ["Shank.Top.Lateral", "Shank.Top.Medial", "Shank.Bottom.Lateral", "Shank.Bottom.Medial"],
    "foot": ["Heel.Bottom", "Heel.Lateral", "Heel.Top", "MT1", "MT5"],
}


# --------------------------------------------------------------------------- #
# Linear algebra helpers
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def frame_from_axes(y: np.ndarray, z_temp: np.ndarray) -> np.ndarray:
    """Orthonormal frame (columns x, y, z) from a primary y axis and a helper z axis."""
    y = _unit(y)
    x = _unit(np.cross(y, z_temp))
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def frame_from_x(x_temp: np.ndarray, z_temp: np.ndarray) -> np.ndarray:
    """Frame with primary x axis (used for pelvis and foot)."""
    x = _unit(x_temp)
    y = _unit(np.cross(z_temp, x))
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def rigid_fit(A: np.ndarray, B: np.ndarray):
    """Least-squares rigid transformation B ≈ R @ A + t for a time series.

    A: (k, 3) reference cluster (static), B: (n, k, 3) measured cluster.
    Returns R (n, 3, 3), t (n, 3) and RMS fit residual per frame (n,) in mm.
    The residual is a direct quality measure: it grows with marker noise,
    soft-tissue artefact or a mislabelled marker.
    """
    ca = A.mean(axis=0)
    cb = B.mean(axis=1)
    Ac = A - ca
    Bc = B - cb[:, None, :]
    H = np.einsum("ki,nkj->nij", Ac, Bc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nji,nkj->nik", Vt, U)))  # det(V U^T)
    D = np.zeros((len(B), 3, 3))
    D[:, 0, 0] = 1
    D[:, 1, 1] = 1
    D[:, 2, 2] = d
    R = np.einsum("nji,njk,nlk->nil", Vt, D, U)                   # V D U^T
    t = cb - np.einsum("nij,j->ni", R, ca)
    fitted = np.einsum("nij,kj->nki", R, A) + t[:, None, :]
    rms = np.sqrt(np.mean(np.sum((fitted - B) ** 2, axis=-1), axis=-1))
    return R, t, rms


# --------------------------------------------------------------------------- #
# Static calibration
# --------------------------------------------------------------------------- #
def harrington_hjc(rasis, lasis, rpsis, lpsis):
    """Hip joint centres (Harrington et al. 2007) in global coordinates.

    Regression in the pelvis frame (origin mid-ASIS; x anterior, y superior,
    z lateral) using pelvic width PW (inter-ASIS) and depth PD (mid-ASIS to mid-PSIS), mm:
        x = -0.24 PD - 9.9,  y = -0.30 PW - 10.9,  z = ±(0.33 PW + 7.3)
    """
    mid_asis = (rasis + lasis) / 2
    mid_psis = (rpsis + lpsis) / 2
    pw = np.linalg.norm(rasis - lasis)
    pd = np.linalg.norm(mid_asis - mid_psis)
    Rp = frame_from_x(mid_asis - mid_psis, rasis - lasis)
    out = {}
    for side, sgn in (("R", 1.0), ("L", -1.0)):
        local = np.array([-0.24 * pd - 9.9, -0.30 * pw - 10.9, sgn * (0.33 * pw + 7.3)])
        out[side] = mid_asis + Rp @ local
    return out


@dataclass
class SegmentCalibration:
    cluster_labels: list[str]
    cluster_static: np.ndarray       # (k, 3)
    R_static: np.ndarray             # (3, 3) anatomical frame in static trial
    origin_static: np.ndarray        # (3,)
    landmarks_static: dict           # name -> (3,) anatomical points to carry along


def _mean(static, label):
    return np.nanmean(static[label], axis=0)


def calibrate(static) -> dict[str, SegmentCalibration]:
    """Build anatomical frames for pelvis, thighs, shanks and feet from a static trial."""
    m = {lab: _mean(static, lab) for lab in static.markers}
    cal: dict[str, SegmentCalibration] = {}

    hjc = harrington_hjc(m["R.ASIS"], m["L.ASIS"], m["R.PSIS"], m["L.PSIS"])
    mid_asis = (m["R.ASIS"] + m["L.ASIS"]) / 2
    mid_psis = (m["R.PSIS"] + m["L.PSIS"]) / 2
    Rp = frame_from_x(mid_asis - mid_psis, m["R.ASIS"] - m["L.ASIS"])
    cal["pelvis"] = SegmentCalibration(
        CLUSTERS["pelvis"], np.array([m[l] for l in CLUSTERS["pelvis"]]), Rp, mid_asis,
        {"R.HJC": hjc["R"], "L.HJC": hjc["L"], "pelvis_centre": (mid_asis + mid_psis) / 2},
    )

    for s in SIDES:
        sgn = 1.0 if s == "R" else -1.0
        kjc = (m[f"{s}.Knee"] + m[f"{s}.Knee.Medial"]) / 2
        ajc = (m[f"{s}.Ankle"] + m[f"{s}.Ankle.Medial"]) / 2
        z_knee = sgn * (m[f"{s}.Knee"] - m[f"{s}.Knee.Medial"])      # always points right
        z_ankle = sgn * (m[f"{s}.Ankle"] - m[f"{s}.Ankle.Medial"])
        z_foot = sgn * (m[f"{s}.MT5"] - m[f"{s}.MT1"])
        mid_mt = (m[f"{s}.MT1"] + m[f"{s}.MT5"]) / 2

        thigh_lbl = [f"{s}.{l}" for l in CLUSTERS["thigh"]]
        cal[f"{s}.thigh"] = SegmentCalibration(
            thigh_lbl, np.array([m[l] for l in thigh_lbl]),
            frame_from_axes(hjc[s] - kjc, z_knee), kjc, {f"{s}.KJC": kjc},
        )
        shank_lbl = [f"{s}.{l}" for l in CLUSTERS["shank"]]
        cal[f"{s}.shank"] = SegmentCalibration(
            shank_lbl, np.array([m[l] for l in shank_lbl]),
            frame_from_axes(kjc - ajc, z_ankle), ajc, {f"{s}.AJC": ajc},
        )
        foot_lbl = [f"{s}.{l}" for l in CLUSTERS["foot"]]
        cal[f"{s}.foot"] = SegmentCalibration(
            foot_lbl, np.array([m[l] for l in foot_lbl]),
            frame_from_x(mid_mt - m[f"{s}.Heel.Bottom"], z_foot), ajc,
            {f"{s}.midMT": mid_mt},
        )
    return cal


# --------------------------------------------------------------------------- #
# Tracking and joint angles
# --------------------------------------------------------------------------- #
@dataclass
class SegmentPose:
    R: np.ndarray        # (n, 3, 3) anatomical frame (columns = x, y, z) in lab
    origin: np.ndarray   # (n, 3)
    fit_rms: np.ndarray  # (n,) rigid-body fit residual [mm]


def track(markers: dict, cal: dict[str, SegmentCalibration]):
    """Track every calibrated segment through a dynamic trial.

    Returns (poses, virtual_points) where virtual_points contains joint centres
    reconstructed from the static calibration (e.g. 'R.HJC', 'R.KJC').
    """
    poses, virtual = {}, {}
    for name, c in cal.items():
        B = np.stack([markers[l] for l in c.cluster_labels], axis=1)   # (n, k, 3)
        R, t, rms = rigid_fit(c.cluster_static, B)
        poses[name] = SegmentPose(
            R=np.einsum("nij,jk->nik", R, c.R_static),
            origin=np.einsum("nij,j->ni", R, c.origin_static) + t,
            fit_rms=rms,
        )
        for lm, p in c.landmarks_static.items():
            virtual[lm] = np.einsum("nij,j->ni", R, p) + t
    return poses, virtual


def joint_angles(prox: np.ndarray, dist: np.ndarray, side: str, joint: str) -> dict[str, np.ndarray]:
    """Clinical joint angles (deg) from proximal and distal segment frames.

    Sign conventions (both sides): hip flexion +, knee flexion +, ankle dorsiflexion +,
    adduction +, internal rotation +.
    """
    Rj = np.einsum("nji,njk->nik", prox, dist)            # prox^T @ dist
    z, x, y = Rotation.from_matrix(Rj).as_euler("ZXY", degrees=True).T
    flex = -z if joint == "knee" else z
    sgn = 1.0 if side == "R" else -1.0
    return {"flex": flex, "add": sgn * x, "rot": sgn * y}


def all_joint_angles(poses: dict[str, SegmentPose]) -> dict[str, np.ndarray]:
    """Hip, knee and ankle angles for both sides, keyed like 'R.knee.flex'."""
    out = {}
    for s in SIDES:
        pairs = {
            "hip": (poses["pelvis"].R, poses[f"{s}.thigh"].R),
            "knee": (poses[f"{s}.thigh"].R, poses[f"{s}.shank"].R),
            "ankle": (poses[f"{s}.shank"].R, poses[f"{s}.foot"].R),
        }
        for joint, (p, d) in pairs.items():
            for comp, v in joint_angles(p, d, s, joint).items():
                out[f"{s}.{joint}.{comp}"] = v
    return out


def foot_pitch(foot_R: np.ndarray) -> np.ndarray:
    """Sagittal inclination (deg) of the foot's long axis relative to the lab horizontal.

    Positive = toes up. Used for the foot-strike angle.
    """
    x = foot_R[:, :, 0]
    return np.degrees(np.arctan2(x[:, 1], np.hypot(x[:, 0], x[:, 2])))
