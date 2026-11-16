"""Create a one-page report for every runner (PDF) plus PNG previews.

Usage: python scripts/make_reports.py [--subjects RBDS001 ...] [--png RBDS008]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rungait.io import load_participant_info  # noqa: E402
from rungait.report import build_report  # noqa: E402

RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--png", nargs="*", default=["RBDS001", "RBDS008"], help="also save a PNG preview")
    args = ap.parse_args()

    with open(RES / "cache" / "results.pkl", "rb") as fh:
        results = pickle.load(fh)
    group = pd.read_csv(RES / "summary_all.csv")
    info = load_participant_info(ROOT / "data" / "raw" / "RBDSinfo.txt")
    subjects = args.subjects or sorted({s for s, _ in results})
    out = RES / "reports"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sub in subjects:
        sr = {sp: r for (s, sp), r in results.items() if s == sub}
        jobs.append((sub, sr, info.loc[sub], group, out / f"{sub}_report.pdf"))
        if sub in args.png:
            jobs.append((sub, sr, info.loc[sub], group, out / f"{sub}_report.png"))
    # LaTeX typesetting is slow, so reports are rendered in parallel
    with ProcessPoolExecutor() as ex:
        for path in ex.map(_build, jobs):
            print("report", path.name)


def _build(job):
    build_report(*job)
    return job[-1]


if __name__ == "__main__":
    main()
