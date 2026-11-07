"""
prepare.py - SOM input arrays from the station parquets.

Each of the inputs in config.SOM_INPUTS is standardized per station with the mean
and standard deviation of that station up to config.TRAIN_CUTOFF. Stations
without hours in that period use the pooled training statistics.

Sets
  train         stations not in the spatial validation list, up to TRAIN_CUTOFF
  val_temporal  same stations, after TRAIN_CUTOFF
  val_spatial   the stations in data/spatial_validation_stations.csv, whole period

    python som/prepare.py
    python som/prepare.py --meta-only    # rewrites only meta_{set}.parquet, keeps X_{set}.npy

Outputs in config.WORK_DIR/arrays
  X_{set}.npy        float32 (hours, inputs), rows ordered by station and time
  meta_{set}.parquet station, time, solar hour and rainfall columns, same order
  station_stats.csv  mean and standard deviation used for each station
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SETS = ["train", "val_temporal", "val_spatial"]
META_COLS = ["codigo_estacao", "datetime", "hora_solar", "prec_acum_1h", "prec_acum_2h",
             "prec_acum_3h", "prec_acum_6h", "prec_acum_12h", "prec_acum_24h"] + [f"chuva_prox_{h}h" for h in config.HORIZONS]
OUT = config.WORK_DIR / "arrays"


def set_of(df, spatial):
    if df["codigo_estacao"].iat[0] in spatial:
        return np.full(len(df), "val_spatial", dtype=object)
    return np.where(df["datetime"] <= config.TRAIN_CUTOFF, "train", "val_temporal")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "prepare.log", mode="w"),
                                  logging.StreamHandler()])
    log = logging.getLogger()
    spatial = set(pd.read_csv(config.SPATIAL_STATIONS, sep=";", dtype=str)["codigo"])
    files = sorted(config.PARQUET_DIR.glob("*.parquet"))
    cols = config.SOM_INPUTS
    leak = [c for c in cols if any(k in c for k in ("prec", "chuva"))]
    if leak:
        raise SystemExit(f"rainfall variables in the SOM input: {leak}")
    log.info(f"{len(files)} stations, {len(cols)} inputs, cutoff {config.TRAIN_CUTOFF}")

    # pass 1, sizes and statistics
    sizes = dict.fromkeys(SETS, 0)
    stats, pooled = {}, [0.0, np.zeros(len(cols)), np.zeros(len(cols))]
    for f in files:
        df = pd.read_parquet(f, columns=["codigo_estacao", "datetime"] + cols)
        s = set_of(df, spatial)
        for k in SETS:
            sizes[k] += int((s == k).sum())
        ref = df[df["datetime"] <= config.TRAIN_CUTOFF][cols].to_numpy("float64")
        if len(ref) > 1:
            stats[f.stem] = (ref.mean(axis=0), np.clip(ref.std(axis=0, ddof=1), 1e-9, None))
        if df["codigo_estacao"].iat[0] not in spatial and len(ref):
            pooled[0] += len(ref); pooled[1] += ref.sum(axis=0); pooled[2] += (ref ** 2).sum(axis=0)
    gmu = pooled[1] / pooled[0]
    gsd = np.sqrt(np.maximum(pooled[2] / pooled[0] - gmu ** 2, 1e-18)) * np.sqrt(pooled[0] / (pooled[0] - 1))
    log.info("hours per set " + ", ".join(f"{k} {v:,}" for k, v in sizes.items()))
    log.info(f"stations without training-period hours (pooled statistics): "
             f"{len(files) - len(stats)}")

    # pass 2, arrays
    meta_only = "--meta-only" in sys.argv
    if meta_only:
        for k in SETS:
            n = np.load(OUT / f"X_{k}.npy", mmap_mode="r").shape[0]
            if n != sizes[k]:
                raise SystemExit(f"X_{k}.npy has {n:,} rows, parquets give {sizes[k]:,}; run without --meta-only")
        X = None
    else:
        X = {k: np.lib.format.open_memmap(OUT / f"X_{k}.npy", mode="w+", dtype="float32",
                                          shape=(sizes[k], len(cols))) for k in SETS}
    metas = {k: [] for k in SETS}
    pos = dict.fromkeys(SETS, 0)
    rows_stats = []
    for f in files:
        df = pd.read_parquet(f, columns=list(dict.fromkeys(META_COLS + cols)))
        mu, sd = stats.get(f.stem, (gmu, gsd))
        rows_stats.append([f.stem, f.stem in stats] + list(mu) + list(sd))
        z = None if meta_only else ((df[cols].to_numpy("float64") - mu) / sd).astype("float32")
        s = set_of(df, spatial)
        for k in SETS:
            m = s == k
            n = int(m.sum())
            if n:
                if X is not None:
                    X[k][pos[k]:pos[k] + n] = z[m]
                metas[k].append(df.loc[m, META_COLS])
                pos[k] += n
    for k in SETS:
        if X is not None:
            X[k].flush()
        (pd.concat(metas[k], ignore_index=True) if metas[k] else pd.DataFrame(columns=META_COLS)).to_parquet(OUT / f"meta_{k}.parquet", index=False)
    pd.DataFrame(rows_stats, columns=["codigo", "proprias"] + [f"mean_{c}" for c in cols]
                 + [f"std_{c}" for c in cols]).to_csv(OUT / "station_stats.csv", sep=";", index=False)
    log.info("done")


if __name__ == "__main__":
    main()
