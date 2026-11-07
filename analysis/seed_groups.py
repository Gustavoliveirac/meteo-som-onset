"""
seed_groups.py - groups of monitored units in each seed replica of the 20x20 map.

For every 20x20 map in config.WORK_DIR/som (seeds 42 to 47) the monitored set is
chosen as in analysis/coverage.py (units ordered by the density of 20 mm events in
training, leading set closest to the share of hours with rain in progress) and its
adjacent units are joined into groups. Each group is described by its share of hours
and events, the hour at which its events peak, the shortest run of hours holding half
of them and the mean 1-h changes of temperature, humidity, radiation, wind and
pressure, so the groups of different seeds can be compared.

    python analysis/seed_groups.py

Writes config.RESULTS_DIR/sementes_grupos.csv and config.LOG_DIR/seed_groups.log
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from som.prepare import set_of  # noqa: E402

SOM_ROOT = config.WORK_DIR / "som"
VARS = ["temperatura", "umidade_relativa", "grad1h_temperatura", "grad1h_umidade_relativa", "grad1h_radiacao",
        "grad1h_vento_velocidade", "grad1h_pressao"]
log = logging.getLogger()


def half_window(e):
    e = e / max(e.sum(), 1)
    best = (25, 0)
    for start in range(24):
        acc = 0.0
        for length in range(1, 25):
            acc += e[(start + length - 1) % 24]
            if acc >= 0.5:
                best = min(best, (length, start))
                break
    return f"{best[1]} a {(best[1] + best[0]) % 24} h"


def main():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "seed_groups.log", mode="w"),
                                  logging.StreamHandler()])
    seeds = sorted(d for d in SOM_ROOT.glob("20x20_seed*") if (d / "units_train.npy").exists())
    units = {d.name: np.load(d / "units_train.npy").astype(np.int64) for d in seeds}
    n = 400
    acc = {k: dict(h=np.zeros(n), e20=np.zeros(n), e40=np.zeros(n), rain=np.zeros(n),
                   eh=np.zeros(n * 24), s=np.zeros((n, len(VARS)))) for k in units}
    spatial = set(pd.read_csv(config.SPATIAL_STATIONS, sep=";", dtype=str)["codigo"])
    pos = 0
    rain_hours = tot_hours = 0
    for f in sorted(config.PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["codigo_estacao", "datetime", "hora_solar", "prec_acum_1h",
                                         "chuva_prox_1h"] + VARS)
        df = df[set_of(df, spatial) == "train"]
        m = len(df)
        if not m:
            continue
        y = np.nan_to_num(df["chuva_prox_1h"].to_numpy())
        rn = df["prec_acum_1h"].to_numpy() > 0
        hr = np.floor(df["hora_solar"].to_numpy()).astype(int) % 24
        x = df[VARS].to_numpy("float64")
        rain_hours += int(rn.sum()); tot_hours += m
        for k, u_all in units.items():
            u = u_all[pos:pos + m]
            a = acc[k]
            a["h"] += np.bincount(u, minlength=n)
            a["e20"] += np.bincount(u[y >= 20], minlength=n)
            a["e40"] += np.bincount(u[y >= 40], minlength=n)
            a["rain"] += np.bincount(u[rn], minlength=n)
            a["eh"] += np.bincount(u[y >= 20] * 24 + hr[y >= 20], minlength=n * 24)
            for j in range(len(VARS)):
                ok = np.isfinite(x[:, j])
                a["s"][:, j] += np.bincount(u[ok], weights=x[ok, j], minlength=n)
        pos += m
    budget = rain_hours / tot_hours
    log.info(f"maps: {', '.join(units)}; budget {budget:.4f}")

    rows = []
    for k, a in acc.items():
        dens = a["e20"] / np.maximum(a["h"], 1)
        order = np.argsort(-dens, kind="stable")
        cum = np.cumsum(a["h"][order]) / a["h"].sum()
        n_sel = int(np.argmin(np.abs(cum - budget))) + 1
        sel = np.zeros(n, bool); sel[order[:n_sel]] = True
        lab, ng = label(sel.reshape(20, 20), structure=np.ones((3, 3)))
        lab = lab.ravel()
        means = a["s"] / np.maximum(a["h"], 1)[:, None]
        recs = []
        for g in range(1, ng + 1):
            m = lab == g
            w = a["h"][m]
            eh = a["eh"].reshape(n, 24)[m].sum(axis=0)
            r = dict(mapa=k, nos=int(m.sum()), linhas=f"{m.reshape(20, 20).nonzero()[0].min() + 1}-{m.reshape(20, 20).nonzero()[0].max() + 1}",
                     colunas=f"{m.reshape(20, 20).nonzero()[1].min() + 1}-{m.reshape(20, 20).nonzero()[1].max() + 1}",
                     fracao_horas=w.sum() / a["h"].sum(), fracao_eventos_20=a["e20"][m].sum() / a["e20"].sum(),
                     fracao_eventos_40=a["e40"][m].sum() / a["e40"].sum(),
                     hora_pico_eventos=int(eh.argmax()), metade_dos_eventos=half_window(eh),
                     horas_com_chuva_em_curso=a["rain"][m].sum() / w.sum())
            for j, v in enumerate(VARS):
                r[v] = (means[m, j] * w).sum() / w.sum()
            r["grad1h_radiacao"] = r["grad1h_radiacao"] / 3.6      # W/m2
            recs.append(r)
        recs.sort(key=lambda r: -r["fracao_eventos_20"])
        for i, r in enumerate(recs, 1):
            r["grupo"] = i
        rows += recs
        log.info(f"{k}: {n_sel} units, {ng} groups, {sum(r['nos'] == 1 for r in recs)} isolated units")
    out = pd.DataFrame(rows)
    cols = ["mapa", "grupo"] + [c for c in out.columns if c not in ("mapa", "grupo")]
    out = out[cols]
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.round(3).to_csv(config.RESULTS_DIR / "sementes_grupos.csv", sep=";", index=False)
    pd.set_option("display.width", 250)
    show = ["mapa", "grupo", "nos", "fracao_horas", "fracao_eventos_20", "hora_pico_eventos", "metade_dos_eventos",
            "horas_com_chuva_em_curso", "grad1h_temperatura", "grad1h_radiacao", "grad1h_vento_velocidade", "grad1h_pressao"]
    log.info("\n" + out[show].round(3).to_string(index=False))
    log.info("\ndone")


if __name__ == "__main__":
    main()
