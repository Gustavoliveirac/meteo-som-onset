"""
raw_manifest.py - list of the BDMEP files used, with size and SHA-256.

    python data/raw_manifest.py --write    # creates docs/raw_files_manifest.csv from config.RAW_DIR
    python data/raw_manifest.py            # checks config.RAW_DIR against that list

A third party who downloads the data again can see which files differ from
the ones used here.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

MANIFEST = config.ROOT / "docs" / "raw_files_manifest.csv"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scan(raw):
    return pd.DataFrame([dict(arquivo=p.name, bytes=p.stat().st_size, sha256=sha256(p))
                         for p in sorted(Path(raw).glob("*.csv"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    cur = scan(config.RAW_DIR)
    print(f"{len(cur)} files in {config.RAW_DIR}")
    if a.write:
        cur.to_csv(MANIFEST, sep=";", index=False)
        print(f"written {MANIFEST}")
        return
    ref = pd.read_csv(MANIFEST, sep=";")
    m = ref.merge(cur, on="arquivo", how="outer", suffixes=("_ref", "_cur"), indicator=True)
    print(f"missing here {int((m._merge == 'left_only').sum())}, "
          f"extra here {int((m._merge == 'right_only').sum())}, "
          f"different content {int(((m._merge == 'both') & (m.sha256_ref != m.sha256_cur)).sum())}")


if __name__ == "__main__":
    main()
