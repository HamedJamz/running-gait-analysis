"""Process every participant and speed of the RBDS dataset.

Usage:
    python scripts/run_all.py --data data/raw --out results

Writes:
    results/summary_all.csv       one row per subject x speed x metric
    results/steps_all.csv         one row per foot contact
    results/qc_all.csv            one row per trial (data quality log)
    results/event_validation.csv  kinematic vs force-plate event timing
    results/cache/results.pkl     full results for the report/analysis scripts
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rungait.io import find_trials, load_c3d, load_participant_info  # noqa: E402
from rungait.pipeline import process_trial  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--subjects", nargs="*", help="e.g. RBDS001 RBDS002 (default: all)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    trials = find_trials(args.data)
    info = load_participant_info(Path(args.data) / "RBDSinfo.txt")
    subjects = args.subjects or sorted(trials.subject.unique())

    results, summ, steps, qcs, evc = {}, [], [], [], []
    for sub in subjects:
        t = trials[trials.subject == sub]
        static = load_c3d(t[t.kind == "static"].path.iloc[0], with_forces=False)
        mass = float(info.loc[sub, "Mass"])
        for _, r in t[t.kind == "run"].sort_values("speed").iterrows():
            res = process_trial(load_c3d(r.path), static, mass, r.speed, sub)
            results[(sub, r.speed)] = res
            s = res.summary.reset_index().assign(subject=sub, speed=r.speed)
            summ.append(s)
            steps.append(res.steps.assign(subject=sub, speed=r.speed))
            q = {k: v for k, v in res.qc.items() if k not in ("markers", "cluster_fit_rms_mm", "label_swaps")}
            q.update({f"fit_rms_{k}": v for k, v in res.qc["cluster_fit_rms_mm"].items()})
            q["label_swaps_corrected"] = "; ".join(x["swapped"] for x in res.qc["label_swaps"] if x["corrected"])
            q["absent_markers"] = "; ".join(q["absent_markers"])
            qcs.append(dict(subject=sub, speed=r.speed, file=r.path.name, **q))
            evc.append(res.event_check.assign(subject=sub, speed=r.speed))
            print(f"{sub} {r.speed} m/s: {res.qc['n_valid_steps']} valid steps, "
                  f"force check {res.qc['mean_vgrf_over_bw']:.3f} BW"
                  + (f", fixed {q['label_swaps_corrected']}" if q["label_swaps_corrected"] else ""))

    pd.concat(summ).to_csv(out / "summary_all.csv", index=False, float_format="%.4f")
    pd.concat(steps).to_csv(out / "steps_all.csv", index=False, float_format="%.4f")
    pd.DataFrame(qcs).to_csv(out / "qc_all.csv", index=False, float_format="%.4f")
    pd.concat(evc).to_csv(out / "event_validation.csv", index=False, float_format="%.2f")
    with open(out / "cache" / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Processed {len(results)} trials from {len(subjects)} participants -> {out}")


if __name__ == "__main__":
    main()
