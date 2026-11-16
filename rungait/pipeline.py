"""End-to-end processing of one running trial and of one participant."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .events import force_events, kinematic_events, match_events
from .io import Trial, find_trials, load_c3d
from .kinematics import all_joint_angles, calibrate, foot_pitch, track
from .metrics import mean_curves, stance_vgrf_curves, step_table, summarise
from .qc import force_qc, preprocess_markers
from .signal import lowpass

MARKER_CUTOFF_HZ = 18.0   # from residual analysis, see docs/methods.md
FORCE_CUTOFF_HZ = 50.0
FOOT_EVENT_MARKERS = ("Heel.Bottom", "Heel.Lateral", "MT1", "MT5")


@dataclass
class TrialResult:
    subject: str
    speed: float
    qc: dict
    events: pd.DataFrame
    event_check: pd.DataFrame
    steps: pd.DataFrame
    summary: pd.DataFrame
    curves: dict
    vgrf_curves: dict
    fit_rms: dict = field(default_factory=dict)


def process_trial(run: Trial, static: Trial, body_mass: float, speed: float, subject: str = "") -> TrialResult:
    cal = calibrate(static)
    markers, invalid, qc = preprocess_markers(run, cal, fc=MARKER_CUTOFF_HZ)
    poses, virtual = track(markers, cal)
    angles = all_joint_angles(poses)

    # Static-trial reference for the foot-strike angle
    st_poses, _ = track({l: static.markers[l] for l in markers if l in static.markers}, cal)
    static_pitch = {s: float(np.nanmean(foot_pitch(st_poses[f"{s}.foot"].R))) for s in "RL"}
    pitch = {s: foot_pitch(poses[f"{s}.foot"].R) for s in "RL"}

    # Forces
    fy_raw = run.grf[:, 1]
    qc.update(force_qc(fy_raw, body_mass))
    fy = lowpass(fy_raw, run.analog_rate, FORCE_CUTOFF_HZ)

    foot_centre = {s: np.mean([markers[f"{s}.{m}"][:, 1] for m in ("Heel.Bottom", "MT1", "MT5")], axis=0) for s in "RL"}
    events = force_events(fy_raw, run.analog_rate, run.rate, foot_centre)
    kin_events = kinematic_events({s: {m: markers[f"{s}.{m}"][:, 1] for m in FOOT_EVENT_MARKERS} for s in "RL"}, run.rate)
    check = match_events(events, kin_events)

    pelvis_centre = virtual["pelvis_centre"]
    ajc = {s: virtual[f"{s}.AJC"] for s in "RL"}
    steps = step_table(events, angles, pitch, static_pitch, pelvis_centre, ajc, fy,
                       run.analog_rate, run.rate, body_mass, speed, invalid)

    qc["n_contacts"] = len(events)
    qc["n_valid_steps"] = int(steps.valid.sum()) if len(steps) else 0
    qc["side_alternation_ok"] = bool((events.side.values[1:] != events.side.values[:-1]).all())
    qc["cluster_fit_rms_mm"] = {k: float(np.median(p.fit_rms)) for k, p in poses.items()}

    curve_signals = {k: v for k, v in angles.items() if k.split(".")[2] in ("flex", "add", "rot")}
    return TrialResult(
        subject=subject, speed=speed, qc=qc, events=events, event_check=check, steps=steps,
        summary=summarise(steps), curves=mean_curves(steps, curve_signals),
        vgrf_curves=stance_vgrf_curves(steps, fy, run.analog_rate, body_mass * 9.81),
        fit_rms={k: p.fit_rms for k, p in poses.items()},
    )


def process_subject(subject: str, data_dir: str | Path, body_mass: float) -> dict[float, TrialResult]:
    trials = find_trials(data_dir)
    t = trials[trials.subject == subject]
    static = load_c3d(t[t.kind == "static"].path.iloc[0], with_forces=False)
    out = {}
    for _, r in t[t.kind == "run"].sort_values("speed").iterrows():
        out[r.speed] = process_trial(load_c3d(r.path), static, body_mass, r.speed, subject)
    return out
