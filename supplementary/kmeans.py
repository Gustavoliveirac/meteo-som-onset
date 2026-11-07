"""
kmeans.py - K-means with the same number of groups as the 20x20 SOM (400), on the
same 38 standardized inputs, as the reference for the comparison in compare.py.

Centres are fitted with Lloyd iterations on a seeded sample of training hours
(k-means++ initialisation) and every hour of the three sets is then assigned to
its nearest centre.

    python supplementary/kmeans.py

Outputs in config.WORK_DIR/som/kmeans400_seed42: units_{set}.npy, metrics.csv,
centres.npz. Log in config.LOG_DIR/kmeans.log
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

K, SEED, N_FIT, N_INIT, MAX_IT, TOL = 400, 42, 2_000_000, 200_000, 100, 1e-5
ARR = config.WORK_DIR / "arrays"
OUT = config.WORK_DIR / "som" / f"kmeans{K}_seed{SEED}"
SETS = ["train", "val_temporal", "val_spatial"]
log = logging.getLogger()


def nearest(X, C, batch=8192):
    c2 = (C * C).sum(axis=1)
    idx = np.empty(len(X), dtype=np.int16)
    d = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), batch):
        x = np.asarray(X[s:s + batch], dtype="float64")
        d2 = (x * x).sum(axis=1)[:, None] - 2.0 * x @ C.T + c2
        i = d2.argmin(axis=1)
        idx[s:s + batch] = i
        d[s:s + batch] = np.sqrt(np.maximum(d2[np.arange(len(i)), i], 0))
    return idx, d


def kmeanspp(X, k, rng):
    C = np.empty((k, X.shape[1]))
    C[0] = X[rng.integers(len(X))]
    d2 = ((X - C[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        C[j] = X[rng.choice(len(X), p=d2 / d2.sum())]
        d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(axis=1))
    return C


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "kmeans.log", mode="w"),
                                  logging.StreamHandler()])
    if (OUT / "metrics.csv").exists():
        log.info("already done")
        return
    rng = np.random.default_rng(SEED)
    Xtr = np.load(ARR / "X_train.npy", mmap_mode="r")
    samp = np.sort(rng.choice(len(Xtr), min(N_FIT, len(Xtr)), replace=False))
    S = np.asarray(Xtr[samp], dtype="float64")
    C = kmeanspp(S[rng.choice(len(S), min(N_INIT, len(S)), replace=False)], K, rng)
    prev = np.inf
    for it in range(MAX_IT):
        lab, d = nearest(S, C)
        inertia = float((d.astype("float64") ** 2).mean())
        li = lab.astype(np.int64)
        cnt = np.bincount(li, minlength=K)
        for c in range(S.shape[1]):
            sums = np.bincount(li, weights=S[:, c], minlength=K)
            C[cnt > 0, c] = sums[cnt > 0] / cnt[cnt > 0]
        empty = np.flatnonzero(cnt == 0)
        if len(empty):
            C[empty] = S[rng.choice(len(S), len(empty), replace=False)]
        log.info(f"  iteration {it + 1} inertia {inertia:.5f}")
        if abs(prev - inertia) / inertia < TOL:
            break
        prev = inertia
    np.savez_compressed(OUT / "centres.npz", centres=C)
    rows = []
    for k in SETS:
        X = np.load(ARR / f"X_{k}.npy", mmap_mode="r")
        lab, d = nearest(X, C)
        np.save(OUT / f"units_{k}.npy", lab)
        rows.append(dict(grid="kmeans400", seed=SEED, set=k, hours=len(X), qe=float(d.mean()), te=np.nan,
                         units_used=int(len(np.unique(lab)))))
        log.info(f"  {k:13s} hours {len(X):>11,}  QE {rows[-1]['qe']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", sep=";", index=False)
    log.info("done")


if __name__ == "__main__":
    main()
