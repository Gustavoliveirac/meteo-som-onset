"""
train.py - trains SOMs and assigns every hour of the three sets to its unit.

    python som/train.py                              # 20x20, seed 42
    python som/train.py --seeds 43 44 45 46 47       # replicas of the 20x20
    python som/train.py --grids 10 30 40 50 60 70 80 90 100

Runs already finished are skipped and interrupted runs continue from the last
checkpoint. The 20x20 with seed 42 is also exported to models/som_20x20_seed42.npz
and .json (weights and settings, no pickle), together with the input statistics.

Outputs in config.WORK_DIR/som/{rows}x{cols}_seed{seed}
  checkpoint.pkl
  units_{set}.npy   unit of each hour, same order as arrays/meta_{set}.parquet
  metrics.csv       quantization and topographic error per set
Log in config.LOG_DIR/train_som_{grids}_{seeds}.log
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from som.model import SOM  # noqa: E402

SETS = ["train", "val_temporal", "val_spatial"]
ARR = config.WORK_DIR / "arrays"


def topographic_error(b1, b2, cols):
    r1, c1 = b1 // cols, b1 % cols
    r2, c2 = b2 // cols, b2 % cols
    return float(((np.abs(r1 - r2) + np.abs(c1 - c2)) > 1).mean())


def run(grid, seed, iters, log):
    tag = f"{grid}x{grid}_seed{seed}"
    out = config.WORK_DIR / "som" / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.csv").exists():
        log.info(f"{tag} already done")
        export(grid, seed, out)
        return pd.read_csv(out / "metrics.csv", sep=";")
    X = np.load(ARR / "X_train.npy", mmap_mode="r")
    ck = out / "checkpoint.pkl"
    som = SOM.load(ck) if ck.exists() else SOM(grid, grid, X.shape[1], seed=seed)
    log.info(f"{tag} training from iteration {som.iteration:,} of {iters:,}, {len(X):,} training hours")
    som.fit(X, iters, ckpt=ck, log=log)

    rows = []
    for k in SETS:
        Xk = np.load(ARR / f"X_{k}.npy", mmap_mode="r")
        b1, b2, qe = som.assign(Xk)
        np.save(out / f"units_{k}.npy", b1)
        rows.append(dict(grid=f"{grid}x{grid}", seed=seed, set=k, hours=len(Xk),
                         qe=float(qe.mean()), te=topographic_error(b1.astype(int), b2.astype(int), grid),
                         units_used=int(len(np.unique(b1)))))
        log.info(f"  {k:13s} hours {len(Xk):>11,}  QE {rows[-1]['qe']:.4f}  TE {rows[-1]['te']:.4f}")
    m = pd.DataFrame(rows)
    m.to_csv(out / "metrics.csv", sep=";", index=False)
    export(grid, seed, out)
    return m


def export(grid, seed, out):
    if grid != 20 or seed != 42:
        return
    d = config.ROOT / "models"
    d.mkdir(exist_ok=True)
    SOM.load(out / "checkpoint.pkl").export(d / "som_20x20_seed42")
    st = ARR / "station_stats.csv"
    if st.exists():
        shutil.copy(st, d / "station_stats.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--iters", type=int, default=400_000)
    a = ap.parse_args()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    name = f"train_som_{'-'.join(map(str, a.grids))}_{'-'.join(map(str, a.seeds))}.log"
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / name), logging.StreamHandler()])
    log = logging.getLogger()
    log.info(f"numpy {np.__version__}")
    res = [run(g, s, a.iters, log) for g in a.grids for s in a.seeds]
    log.info("summary\n" + pd.concat(res).to_string(index=False))


if __name__ == "__main__":
    main()
