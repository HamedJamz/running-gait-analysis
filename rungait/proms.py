"""Patient-reported outcome measures (PROMs): KOOS scoring.

KOOS (Knee injury and Osteoarthritis Outcome Score, Roos et al. 1998) has 42
items in five subscales, each answered on a 0-4 Likert scale
(0 = no problems ... 4 = extreme problems). Each subscale is transformed to
0-100, where 100 = no problems:

    score = 100 - mean(item answers) / 4 * 100

Missing data rule from the KOOS user's guide: one or two missing items per
subscale are replaced by the mean of the answered items of that subscale; with
more than two missing items the subscale is not scored. Always check the
current scoring manual (koos.nu) before clinical use.

Item IDs follow the KOOS questionnaire: S1-S7, P1-P9, A1-A17, SP1-SP5, Q1-Q4.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

KOOS_SUBSCALES: dict[str, list[str]] = {
    "Symptoms": [f"S{i}" for i in range(1, 8)],
    "Pain": [f"P{i}" for i in range(1, 10)],
    "ADL": [f"A{i}" for i in range(1, 18)],
    "Sport/Rec": [f"SP{i}" for i in range(1, 6)],
    "QOL": [f"Q{i}" for i in range(1, 5)],
}
MAX_MISSING = 2


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def score_koos(answers: Mapping[str, float | int | None]) -> dict[str, float | None]:
    """Score one completed KOOS questionnaire.

    ``answers`` maps item ID -> 0..4 (or None/NaN if not answered).
    Returns subscale -> score (0-100) or None if too many items are missing.
    Raises ValueError for answers outside 0-4 (a data-entry error should be
    fixed, not silently scored).
    """
    out: dict[str, float | None] = {}
    for sub, items in KOOS_SUBSCALES.items():
        vals = [answers.get(i) for i in items]
        given = [float(v) for v in vals if not _is_missing(v)]
        for i, v in zip(items, vals):
            if not _is_missing(v) and not (0 <= float(v) <= 4 and float(v).is_integer()):
                raise ValueError(f"KOOS item {i} has invalid value {v!r} (allowed: 0-4)")
        if len(items) - len(given) > MAX_MISSING or not given:
            out[sub] = None
            continue
        mean = sum(given) / len(given)          # mean imputation == mean of answered items
        out[sub] = round(100 - mean / 4 * 100, 1)
    return out


def score_koos_table(df: pd.DataFrame, id_col: str = "patient_id") -> pd.DataFrame:
    """Score a table with one questionnaire per row (columns = item IDs)."""
    rows = []
    for _, r in df.iterrows():
        res = score_koos(r.to_dict())
        n_missing = {f"missing_{s}": int(sum(_is_missing(r.get(i)) for i in items))
                     for s, items in KOOS_SUBSCALES.items()}
        rows.append({id_col: r.get(id_col), **res, **n_missing})
    return pd.DataFrame(rows)
