"""
group_profiles.py - how the groups of monitored units differ, how spread they are,
and where they occur.

For every hour of the three sets (one pass over the station parquets) the group of
its unit is taken from results/nos_monitorados.csv. For each group and for the
unmonitored units it accumulates histograms of the 1-h changes of temperature,
relative humidity, radiation, wind speed and pressure, giving mean, spread and
quantiles, and counts the events of 20 mm or more of each group at each station.

    python analysis/group_profiles.py

Writes to config.RESULTS_DIR
  grupos_variacoes.csv   per group and variable: mean, standard deviation, quantiles
                         (10, 25, 50, 75, 90) and the mean in standard deviations of
                         all hours (for the heat map)
  estacoes_grupos.csv    per station: events of 20 mm or more in each group, the most
                         frequent group among its events and how many groups occur
Log in config.LOG_DIR/group_profiles.log
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
VARS = {  # column, label, factor, histogram range
    "grad1h_temperatura": ("temperatura", 1.0, (-15, 15)),
    "grad1h_umidade_relativa": ("umidade_relativa", 1.0, (-60, 60)),
    "grad1h_radiacao": ("radiacao_wm2", 1 / 3.6, (-1200, 1200)),
    "grad1h_vento_velocidade": ("vento_velocidade", 1.0, (-15, 15)),
    "grad1h_pressao": ("pressao", 1.0, (-6, 6)),
}
NB = 2400
# variables for the summary in physical units: anomaly = value minus the mean of the
# same station, month and hour in the training period; changes over 1, 3 and 6 h
PHYS = {"pressao": 1.0, "temperatura": 1.0, "ponto_orvalho": 1.0, "umidade_relativa": 1.0,
        "vento_velocidade": 1.0, "radiacao": 1 / 3.6}
log = logging.getLogger()

STATES = "--estados" in sys.argv


def F(name):
    """File name in group mode, or its state version with --estados (states from analysis/states.py)."""
    if not STATES:
        return name
    return {"nos_monitorados.csv": "nos_estados.csv", "grupos_monitorados.csv": "estados_monitorados.csv",
            "tabela_grupos.csv": "tabela_estados.csv", "estacoes_grupos.csv": "estacoes_estados.csv",
            "unidades_chuva_atual.csv": "unidades_chuva_atual_estados.csv"}.get(name, name.replace("grupos_", "estados_"))


def main():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / ("group_profiles_estados.log" if STATES else "group_profiles.log"), mode="w"),
                                  logging.StreamHandler()])
    nos = pd.read_csv(config.RESULTS_DIR / F("nos_monitorados.csv"), sep=";")
    gmap = np.zeros(400, dtype=np.int64)
    gmap[nos.unidade.to_numpy()] = nos.grupo.to_numpy()
    groups = sorted(nos.grupo.unique())
    ng = max(groups) + 1                       # 0 = unmonitored units
    units = {k: np.load(UNITS / f"units_{k}.npy").astype(np.int64) for k in SETS}
    pos = {k: 0 for k in SETS}
    spatial = set(pd.read_csv(config.SPATIAL_STATIONS, sep=";", dtype=str)["codigo"])
    hist = {v: np.zeros((ng, NB)) for v in VARS}
    sums = {v: np.zeros((ng, 3)) for v in VARS}      # n, sum, sum of squares
    station_rows = []
    pcols = []
    for v in PHYS:
        pcols += [f"anom_fis_{v}"] + [f"{w}_{v}" for w in ("grad1h", "grad3h", "grad6h")]
    psum = {c: np.zeros((ng, 2)) for c in pcols}           # n, sum
    for f in sorted(config.PARQUET_DIR.glob("*.parquet")):
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(f).names)
        want = ["codigo_estacao", "datetime", "chuva_prox_1h"] + list(VARS) + list(PHYS) + \
            [f"{w}_{v}" for v in PHYS for w in ("grad1h", "grad3h", "grad6h")]
        df = pd.read_parquet(f, columns=[c for c in dict.fromkeys(want) if c in avail])
        sets = set_of(df, spatial)
        key = df["datetime"].dt.month.to_numpy() * 24 + df["datetime"].dt.hour.to_numpy()
        tr = np.asarray(sets == "train")
        for v, fac in PHYS.items():
            if v not in df:
                continue
            x = df[v].to_numpy("float64") * fac
            ok = tr & np.isfinite(x)
            cnt = np.bincount(key[ok], minlength=13 * 24); tot_ = np.bincount(key[ok], weights=x[ok], minlength=13 * 24)
            clim = np.where(cnt > 0, tot_ / np.maximum(cnt, 1), np.nan)
            df[f"anom_fis_{v}"] = x - clim[key]
        for v, fac in PHYS.items():
            for w in ("grad1h", "grad3h", "grad6h"):
                c = f"{w}_{v}"
                if c in df and fac != 1.0:
                    df[c] = df[c] * fac
        ev_by_group = np.zeros(ng)
        n_ev = 0
        for k in SETS:
            m = np.asarray(sets == k)
            n = int(m.sum())
            if not n:
                continue
            u = units[k][pos[k]:pos[k] + n]
            pos[k] += n
            g = gmap[u]
            sub = df.loc[m]
            ev = np.nan_to_num(sub["chuva_prox_1h"].to_numpy()) >= 20
            ev_by_group += np.bincount(g[ev], minlength=ng)
            n_ev += int(ev.sum())
            for c in pcols:
                if c not in sub:
                    continue
                x = sub[c].to_numpy("float64")
                ok = np.isfinite(x)
                psum[c][:, 0] += np.bincount(g[ok], minlength=ng)
                psum[c][:, 1] += np.bincount(g[ok], weights=x[ok], minlength=ng)
            for v, (_, fac, (lo, hi)) in VARS.items():
                x = sub[v].to_numpy("float64") * fac
                ok = np.isfinite(x)
                b = np.clip(((x[ok] - lo) / (hi - lo) * NB).astype(int), 0, NB - 1)
                hist[v] += np.bincount(g[ok] * NB + b, minlength=ng * NB).reshape(ng, NB)
                sums[v][:, 0] += np.bincount(g[ok], minlength=ng)
                sums[v][:, 1] += np.bincount(g[ok], weights=x[ok], minlength=ng)
                sums[v][:, 2] += np.bincount(g[ok], weights=x[ok] ** 2, minlength=ng)
        if n_ev:
            row = dict(codigo=f.stem, eventos_20=n_ev)
            for gi in range(ng):
                row["fora_dos_grupos" if gi == 0 else f"grupo_{gi}"] = int(ev_by_group[gi])
            station_rows.append(row)
    log.info(f"positions used: {pos}")

    rows = []
    for v, (name, fac, (lo, hi)) in VARS.items():
        n_all, s_all, q_all = sums[v].sum(axis=0)
        mean_all = s_all / n_all
        sd_all = np.sqrt(q_all / n_all - mean_all ** 2)
        centres = lo + (np.arange(NB) + 0.5) * (hi - lo) / NB
        for gi in range(ng):
            n, s, q = sums[v][gi]
            if n == 0:
                continue
            mean = s / n
            cum = np.cumsum(hist[v][gi]) / hist[v][gi].sum()
            qs = {f"q{int(p * 100)}": centres[np.searchsorted(cum, p)] for p in (0.10, 0.25, 0.50, 0.75, 0.90)}
            rows.append(dict(grupo="fora dos grupos" if gi == 0 else gi, variavel=name, horas=int(n), media=mean,
                             desvio=np.sqrt(max(q / n - mean ** 2, 0)), media_padronizada=(mean - mean_all) / sd_all, **qs))
    out = pd.DataFrame(rows)
    out.round(3).to_csv(config.RESULTS_DIR / F("grupos_variacoes.csv"), sep=";", index=False)
    fis = pd.DataFrame({c: psum[c][:, 1] / np.maximum(psum[c][:, 0], 1) for c in pcols})
    fis.insert(0, "grupo", ["fora dos grupos"] + list(range(1, ng)))
    fis.round(3).to_csv(config.RESULTS_DIR / F("grupos_fisico.csv"), sep=";", index=False)
    resumo = config.RESULTS_DIR / ("estados_resumo.csv" if STATES else "grupos_resumo.csv")
    if resumo.exists():
        res = pd.read_csv(resumo, sep=";")
        units_ = {"pressao": "hPa", "temperatura": "°C", "ponto_orvalho": "°C", "umidade_relativa": "%",
                  "vento_velocidade": "m/s", "radiacao": "W/m²"}
        names = {"pressao": "Pressão", "temperatura": "Temperatura", "ponto_orvalho": "Ponto de orvalho",
                 "umidade_relativa": "Umidade relativa", "vento_velocidade": "Vento", "radiacao": "Radiação"}
        fi = fis.set_index("grupo")
        rows = []
        for _, r in res.iterrows():
            row = dict(estado=int(r.estado), nos=int(r.nos), fracao_horas=r.fracao_horas,
                       fracao_eventos_20=r.fracao_eventos_20, fracao_eventos_40=r.fracao_eventos_40)
            for key, name in names.items():
                if str(r[name]) == "–":
                    row[name] = "–"; continue
                var = r[name + " (variável)"]
                win = var.split("_")[0]
                col = f"anom_fis_{key}" if win == "anom" else f"{win}_{key}"
                val = fi.loc[int(r.estado), col]
                lab = "" if win == "anom" else f" ({win[4:-1]} h)"
                row[name] = f"{val:+.1f} {units_[key]}{lab}".replace(".", ",")
            rows.append(row)
        tab = pd.DataFrame(rows)
        tab.to_csv(config.RESULTS_DIR / ("estados_tabela.csv" if STATES else "grupos_tabela.csv"), sep=";", index=False)
        log.info("\nstates in physical units (anomaly without window; change over the window in brackets)\n"
                 + tab.to_string(index=False))

    st = pd.DataFrame(station_rows)
    gcols = [c for c in st.columns if c.startswith("grupo_")]
    st["grupo_mais_frequente"] = st[gcols].idxmax(axis=1).str.replace("grupo_", "").astype(int)
    st.loc[st[gcols].sum(axis=1) == 0, "grupo_mais_frequente"] = 0
    st["grupos_presentes"] = (st[gcols] > 0).sum(axis=1)
    meta = pd.read_csv(config.STATIONS_META, sep=";", dtype={"codigo": str})[["codigo", "latitude", "longitude"]]
    st = st.merge(meta, on="codigo", how="left")
    st.to_csv(config.RESULTS_DIR / F("estacoes_grupos.csv"), sep=";", index=False)

    pd.set_option("display.width", 220)
    log.info("\nmean 1-h change per group (physical units)\n" +
             out.pivot_table(index="grupo", columns="variavel", values="media", aggfunc="first").round(2).to_string())
    log.info("\ninterquartile range per group\n" +
             (out.assign(iqr=out.q75 - out.q25).pivot_table(index="grupo", columns="variavel", values="iqr",
                                                             aggfunc="first")).round(2).to_string())
    big = st[st.eventos_20 >= 10]
    log.info(f"\nstations with at least 10 events: {len(big)}")
    for c in gcols:
        log.info(f"  {c}: present in {(big[c] > 0).mean():.2f} of them")
    log.info(f"  number of groups present, median {big.grupos_presentes.median():.0f}, "
             f"all groups in {(big.grupos_presentes == len(gcols)).mean():.2f}")
    log.info("most frequent group among the events of each station:\n" +
             big.grupo_mais_frequente.value_counts().sort_index().to_string())
    log.info("\ndone")


if __name__ == "__main__":
    main()
