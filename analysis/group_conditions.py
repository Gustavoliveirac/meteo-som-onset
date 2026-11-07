"""
group_conditions.py - group profiles split by the rain now and in the next hour.

Table 4 of the paper gives, for each group, the mean of all its hours. Here the same
means are split by the rain at the hour of the record (now) and in the next hour
(after): general, dry now and after, rain after >= 1 mm, rain now >= 1 mm, dry before
and rain after, rain before and rain after, rain now and dry after, rain now >= 20 mm,
rain after >= 20 mm, and the 20 mm onset split into dry before and rain before. Same pass over the station parquets and the same alignment of units
as analysis/group_profiles.py; the three sets together.

    python analysis/group_conditions.py

Writes config.RESULTS_DIR/grupos_condicional.csv (one row per group and condition)
and config.LOG_DIR/group_conditions.log (tables included). Used in section 4.4 of
the paper (dry hours followed by 20 mm or more against dry hours followed by dry hours).
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from som.prepare import set_of  # noqa: E402

UNITS = config.WORK_DIR / "som" / "20x20_seed42"
SETS = ["train", "val_temporal", "val_spatial"]
PHYS = {"pressao": 1.0, "temperatura": 1.0, "ponto_orvalho": 1.0, "umidade_relativa": 1.0,
        "vento_velocidade": 1.0, "radiacao": 1 / 3.6}
# a = chuva na hora do registro (prec_acum_1h), r = chuva na hora seguinte (chuva_prox_1h)
CONDS = {
    "geral": lambda a, r: np.ones(len(a), bool),
    "seco agora e seco depois": lambda a, r: (a == 0) & (r == 0),
    "chuva depois >= 1": lambda a, r: r >= 1,
    "chuva agora >= 1": lambda a, r: a >= 1,
    "seco antes e chuva depois >= 1": lambda a, r: (a == 0) & (r >= 1),
    "chuva antes e chuva depois >= 1": lambda a, r: (a >= 1) & (r >= 1),
    "chuva agora >= 1 e seco depois": lambda a, r: (a >= 1) & (r == 0),
    "chuva agora >= 20": lambda a, r: a >= 20,
    "chuva depois >= 20": lambda a, r: r >= 20,
    "seco antes e chuva depois >= 20": lambda a, r: (a == 0) & (r >= 20),
    "chuva antes e chuva depois >= 20": lambda a, r: (a >= 1) & (r >= 20),
}
OUT = config.RESULTS_DIR
log = logging.getLogger()


def main():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "group_conditions.log", mode="w"),
                                  logging.StreamHandler()])
    import pyarrow.parquet as pq
    nos = pd.read_csv(config.RESULTS_DIR / "nos_monitorados.csv", sep=";")
    gmap = np.zeros(400, dtype=np.int64)
    gmap[nos.unidade.to_numpy()] = nos.grupo.to_numpy()
    ng = int(nos.grupo.max()) + 1
    nc = len(CONDS)
    units = {k: np.load(UNITS / f"units_{k}.npy").astype(np.int64) for k in SETS}
    pos = {k: 0 for k in SETS}
    spatial = set(pd.read_csv(config.SPATIAL_STATIONS, sep=";", dtype=str)["codigo"])
    cols = [f"anom_fis_{v}" for v in PHYS] + [f"grad1h_{v}" for v in PHYS] + [f"grad3h_{v}" for v in PHYS if v != "radiacao"]
    acc = {c: np.zeros((ng * nc, 2)) for c in cols}
    hours = np.zeros(ng * nc)
    for f in sorted(config.PARQUET_DIR.glob("*.parquet")):
        avail = set(pq.read_schema(f).names)
        want = ["codigo_estacao", "datetime", "prec_acum_1h", "chuva_prox_1h"] + list(PHYS) + \
            [f"{w}_{v}" for v in PHYS for w in ("grad1h", "grad3h")]
        df = pd.read_parquet(f, columns=[c for c in dict.fromkeys(want) if c in avail])
        sets = set_of(df, spatial)
        key = df["datetime"].dt.month.to_numpy() * 24 + df["datetime"].dt.hour.to_numpy()
        tr = np.asarray(sets == "train")
        for v, fac in PHYS.items():
            if v not in df:
                continue
            x = df[v].to_numpy("float64") * fac
            ok = tr & np.isfinite(x)
            cnt = np.bincount(key[ok], minlength=13 * 24)
            tot = np.bincount(key[ok], weights=x[ok], minlength=13 * 24)
            clim = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
            df[f"anom_fis_{v}"] = x - clim[key]
            for w in ("grad1h", "grad3h"):
                c = f"{w}_{v}"
                if c in df and fac != 1.0:
                    df[c] = df[c] * fac
        for k in SETS:
            m = np.asarray(sets == k)
            n = int(m.sum())
            if not n:
                continue
            u = units[k][pos[k]:pos[k] + n]
            pos[k] += n
            g = gmap[u]
            sub = df.loc[m]
            r = sub["chuva_prox_1h"].to_numpy("float64")
            a = sub["prec_acum_1h"].to_numpy("float64")
            masks = [fn(a, r) for fn in CONDS.values()]
            for ci, mk in enumerate(masks):
                idx = g[mk] * nc + ci
                hours += np.bincount(idx, minlength=ng * nc)
                for c in cols:
                    if c not in sub:
                        continue
                    x = sub[c].to_numpy("float64")[mk]
                    ok = np.isfinite(x)
                    acc[c][:, 0] += np.bincount(idx[ok], minlength=ng * nc)
                    acc[c][:, 1] += np.bincount(idx[ok], weights=x[ok], minlength=ng * nc)
    log.info(f"positions used: {pos}")
    rows = []
    for gi in range(ng):
        for ci, cond in enumerate(CONDS.keys()):
            i = gi * nc + ci
            row = dict(grupo="fora dos grupos" if gi == 0 else gi, chuva_proxima_hora=cond, horas=int(hours[i]))
            for c in cols:
                row[c] = acc[c][i, 1] / acc[c][i, 0] if acc[c][i, 0] else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.round(3).to_csv(OUT / "grupos_condicional.csv", sep=";", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    log.info("\nhoras por grupo e condição\n" + out.pivot_table(index="grupo", columns="chuva_proxima_hora",
                                                                  values="horas", aggfunc="first")[list(CONDS)].to_string())
    for c in cols:
        log.info(f"\n## {c}\n" + out.pivot_table(index="grupo", columns="chuva_proxima_hora", values=c,
                                                 aggfunc="first")[list(CONDS)].round(2).to_string())
    log.info(f"\nCSV em {OUT / 'grupos_condicional.csv'}")


if __name__ == "__main__":
    main()
