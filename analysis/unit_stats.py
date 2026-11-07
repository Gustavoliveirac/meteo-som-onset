"""
unit_stats.py - mean state and rainfall of every SOM unit, per set.

For each unit: hours, mean of the raw variables, anomalies and changes, circular
mean of the local solar hour and of the wind direction, and, for each horizon,
the mean rainfall of the next hours and the share of hours with more than 1 mm.

    python analysis/unit_stats.py

Reads config.PARQUET_DIR and config.WORK_DIR/som/20x20_seed42.
Writes config.RESULTS_DIR/unidades_estatisticas.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from som.prepare import SETS, set_of  # noqa: E402

UNITS_DIR = config.WORK_DIR / "som" / "20x20_seed42"
OUT = config.RESULTS_DIR / "unidades_estatisticas.csv"
RAW = ["pressao", "temperatura", "ponto_orvalho", "umidade_relativa", "radiacao",
       "vento_velocidade", "theta_e"]
EXTRA = ["vento_dir_sin", "vento_dir_cos"]
RAIN = [f"chuva_prox_{h}h" for h in config.HORIZONS]


def main():
    spatial = set(pd.read_csv(config.SPATIAL_STATIONS, sep=";", dtype=str)["codigo"])
    units = {k: np.load(UNITS_DIR / f"units_{k}.npy").astype(np.int64) for k in SETS}
    n_units = int(max(u.max() for u in units.values())) + 1
    feats = RAW + [c for c in config.SOM_INPUTS if c not in RAW]
    mean_cols = feats + EXTRA
    acc = {k: dict(n=np.zeros(n_units), s=np.zeros((n_units, len(mean_cols))),
                   c=np.zeros((n_units, len(mean_cols))), hs=np.zeros(n_units), hc=np.zeros(n_units),
                   rs=np.zeros((n_units, len(RAIN))), rn=np.zeros((n_units, len(RAIN))),
                   wet=np.zeros((n_units, len(RAIN)))) for k in SETS}
    pos = dict.fromkeys(SETS, 0)
    files = sorted(config.PARQUET_DIR.glob("*.parquet"))
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["codigo_estacao", "datetime", "hora_solar"] + mean_cols + RAIN)
        s = set_of(df, spatial)
        for k in SETS:
            m = s == k
            n = int(m.sum())
            if not n:
                continue
            u = units[k][pos[k]:pos[k] + n]
            pos[k] += n
            d = df[m]
            a = acc[k]
            a["n"] += np.bincount(u, minlength=n_units)
            x = d[mean_cols].to_numpy("float64")
            ok = np.isfinite(x)
            for j in range(x.shape[1]):
                a["s"][:, j] += np.bincount(u[ok[:, j]], weights=x[ok[:, j], j], minlength=n_units)
                a["c"][:, j] += np.bincount(u[ok[:, j]], minlength=n_units)
            ang = d["hora_solar"].to_numpy("float64") / 24 * 2 * np.pi
            a["hs"] += np.bincount(u, weights=np.sin(ang), minlength=n_units)
            a["hc"] += np.bincount(u, weights=np.cos(ang), minlength=n_units)
            r = d[RAIN].to_numpy("float64")
            okr = np.isfinite(r)
            for j in range(r.shape[1]):
                a["rs"][:, j] += np.bincount(u[okr[:, j]], weights=r[okr[:, j], j], minlength=n_units)
                a["rn"][:, j] += np.bincount(u[okr[:, j]], minlength=n_units)
                a["wet"][:, j] += np.bincount(u[okr[:, j] & (r[:, j] > 1.0)], minlength=n_units)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} stations", flush=True)
    for k in SETS:
        if pos[k] != len(units[k]):
            raise SystemExit(f"{k}: {pos[k]:,} hours read, {len(units[k]):,} units; run som/prepare.py again")

    side = int(round(n_units ** 0.5))
    rows = []
    for k in SETS:
        a = acc[k]
        d = pd.DataFrame(dict(conjunto=k, unidade=np.arange(n_units), linha=np.arange(n_units) // side,
                              coluna=np.arange(n_units) % side, horas=a["n"].astype(int)))
        means = a["s"] / np.where(a["c"] > 0, a["c"], np.nan)
        for j, c in enumerate(mean_cols):
            d[c] = means[:, j]
        d["vento_direcao"] = np.degrees(np.arctan2(d.pop("vento_dir_sin"), d.pop("vento_dir_cos"))) % 360
        d["hora_solar"] = (np.degrees(np.arctan2(a["hs"], a["hc"])) % 360) / 15
        for j, c in enumerate(RAIN):
            h = c.split("_")[-1]
            d[f"chuva_media_{h}"] = a["rs"][:, j] / np.where(a["rn"][:, j] > 0, a["rn"][:, j], np.nan)
            d[f"fracao_acima_1mm_{h}"] = a["wet"][:, j] / np.where(a["rn"][:, j] > 0, a["rn"][:, j], np.nan)
        rows.append(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(OUT, sep=";", index=False)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
