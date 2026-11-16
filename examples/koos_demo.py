"""Score KOOS questionnaires from a CSV (one patient per row).

The example file contains three *synthetic* (made-up) questionnaires,
including one with missing answers, to show the missing-data rule.

Usage: python examples/koos_demo.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rungait.proms import score_koos_table  # noqa: E402

df = pd.read_csv(ROOT / "examples" / "koos_example.csv")
print(score_koos_table(df).to_string(index=False))
