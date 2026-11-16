"""Download the public running dataset (Fukuchi et al. 2017) from Figshare.

Only the files used by this project are fetched: C3D trials (static + running),
the Visual3D-processed curves (for validation) and the participant metadata.
About 230 MB.

Usage: python scripts/download_data.py [--out data/raw]
Data licence: CC BY 4.0 - DOI 10.6084/m9.figshare.4543435
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

ARTICLE_ID = 4543435
API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files?page_size=1000"
WANTED = re.compile(r"(\.c3d$|processed\.txt$|^RBDSinfo\.txt$)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API) as r:
        files = json.load(r)
    files = [f for f in files if WANTED.search(f["name"])]
    print(f"{len(files)} files to download -> {out}")
    for i, f in enumerate(files, 1):
        dest = out / f["name"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        print(f"[{i}/{len(files)}] {f['name']}")
        urllib.request.urlretrieve(f["download_url"], dest)
    print("done")


if __name__ == "__main__":
    main()
