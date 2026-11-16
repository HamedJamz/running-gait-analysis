"""Loading C3D motion-capture trials and the RBDS participant metadata.

Lab coordinate system of the RBDS treadmill data (checked from the marker data):
    X = anterior (running direction), Y = vertical (up), Z = right (medio-lateral)
Marker coordinates are in millimetres; ground reaction forces in newtons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd

# File names in the public dataset are not fully consistent: most use three
# digits (RBDS001...), some use two (RBDS03...). Some subjects also have an
# extra 3.0 m/s trial (runT30) that is not part of the published protocol.
_TRIAL_RE = re.compile(r"^RBDS0*(?P<num>\d+)(?:run(?P<cond>T\d+)|(?P<static>static))\.c3d$")
PROTOCOL_SPEEDS = {"T25": 2.5, "T35": 3.5, "T45": 4.5}


@dataclass
class Trial:
    """A single C3D trial with markers (mm) and optional force-plate data (N)."""

    path: Path
    markers: dict[str, np.ndarray]          # label -> (n_frames, 3), mm, lab frame
    rate: float                              # marker sampling rate [Hz]
    grf: np.ndarray | None = None            # (n_analog, 3) ground reaction force [N]
    analog_rate: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return next(iter(self.markers.values())).shape[0]

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.rate

    def __getitem__(self, label: str) -> np.ndarray:
        return self.markers[label]


def subject_id(num: int | str) -> str:
    return f"RBDS{int(num):03d}"


def load_c3d(path: str | Path, with_forces: bool = True) -> Trial:
    """Read a C3D file with ezc3d and return markers + resultant platform force."""
    path = Path(path)
    c = ezc3d.c3d(str(path), extract_forceplat_data=with_forces)
    params = c["parameters"]
    labels = params["POINT"]["LABELS"]["value"]
    rate = float(params["POINT"]["RATE"]["value"][0])
    units = params["POINT"].get("UNITS", {}).get("value", ["mm"])[0]
    pts = c["data"]["points"][:3].astype(float)          # (3, n_markers, n_frames)
    if units == "m":
        pts *= 1000.0
    # ezc3d marks invalid samples with NaN; also treat exact zeros as missing
    zero = np.all(pts == 0.0, axis=0)
    pts[:, zero] = np.nan
    markers = {lab: pts[:, i, :].T.copy() for i, lab in enumerate(labels)}

    grf, arate = None, None
    if with_forces and c["data"].get("platform"):
        plates = c["data"]["platform"]
        # Instrumented treadmill: sum all plates into one resultant force
        grf = np.sum([p["force"] for p in plates], axis=0).T  # (n_analog, 3)
        arate = float(params["ANALOG"]["RATE"]["value"][0])
    return Trial(path=path, markers=markers, rate=rate, grf=grf, analog_rate=arate)


def find_trials(data_dir: str | Path) -> pd.DataFrame:
    """Index all C3D files in the dataset folder (one row per file)."""
    rows = []
    for f in sorted(Path(data_dir).glob("*.c3d")):
        m = _TRIAL_RE.match(f.name)
        if not m:
            continue
        kind = "static" if m["static"] else "run"
        cond = m["cond"]
        rows.append(
            dict(
                subject=subject_id(m["num"]),
                kind=kind,
                condition=cond,
                speed=PROTOCOL_SPEEDS.get(cond) if cond else None,
                path=f,
            )
        )
    df = pd.DataFrame(rows)
    # Keep only the published protocol speeds (drop e.g. runT30)
    return df[(df.kind == "static") | df.condition.isin(PROTOCOL_SPEEDS)].reset_index(drop=True)


def load_participant_info(path: str | Path) -> pd.DataFrame:
    """One row per participant from RBDSinfo.txt (tab separated)."""
    info = pd.read_csv(path, sep="\t")
    info = info.drop_duplicates("Subject").copy()
    info["subject"] = info["Subject"].map(subject_id)
    return info.set_index("subject")
