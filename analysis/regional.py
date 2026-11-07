"""
regional.py - where the SOM covers more events than rain in progress, and why.

Per station, over the three sets together, for 20, 30 and 40 mm: events, coverage of
the SOM set and of rain in progress, their difference and the SOM coverage of the
events without rain in progress. The rainfall regime of the station is described by
its 20 mm events: share with rain in progress, share after 3 and 24 dry hours, rain
already falling at the event hour, and local solar hour of the events. The covariates
are latitude, longitude, altitude, distance to the coast (Natural Earth 1:10m
coastline) and the attributes of the climate and biome maps (CLIMATE_SHP and
BIOMES_SHP in config_local.py).

    python analysis/regional.py

Writes config.RESULTS_DIR/estacoes_regional.csv, regional_costa.csv (narrow distance
bands), regional_classes.csv and
regional_correlacoes.csv, and config.LOG_DIR/regional.log; the three tables and
the log are also put in config.LOG_DIR/resultado_regional.zip.
"""
import logging
import sys
import warnings
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AN = config.RESULTS_DIR
ARR = config.WORK_DIR / "arrays"
UNITS_DIR = config.WORK_DIR / "som" / "20x20_seed42"
CACHE = config.WORK_DIR / "cache"
COAST_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson"
SETS = ["train", "val_temporal", "val_spatial"]
THRESHOLDS = [20, 30, 40]
MIN_EVENTS = 10
log = logging.getLogger()


def load_events(sel):
    parts = []
    for k in SETS:
        m = pd.read_parquet(ARR / f"meta_{k}.parquet",
                            columns=["codigo_estacao", "hora_solar", "prec_acum_1h", "prec_acum_3h",
                                     "prec_acum_24h", "chuva_prox_1h"])
        m["som"] = np.isin(np.load(UNITS_DIR / f"units_{k}.npy"), sel)
        hours = m.assign(chuvosa=m["prec_acum_1h"] > 0).groupby("codigo_estacao", observed=True).agg(
            horas=("som", "size"), horas_chuva=("chuvosa", "sum"))
        parts.append((m[m["chuva_prox_1h"] >= min(THRESHOLDS)].copy(), hours))
    ev = pd.concat([p[0] for p in parts], ignore_index=True)
    hours = pd.concat([p[1] for p in parts]).groupby(level=0).sum()
    hours["fracao_horas_chuva"] = hours.pop("horas_chuva") / hours["horas"]
    ev["chuva"] = ev["prec_acum_1h"] > 0
    return ev, hours


def station_table(ev, hours):
    rows = []
    for code, g in ev.groupby("codigo_estacao", observed=True):
        r = dict(codigo=code)
        for T in THRESHOLDS:
            e = g[g["chuva_prox_1h"] >= T]
            dry = e[~e["chuva"]]
            r[f"eventos_{T}"] = len(e)
            r[f"cobertura_som_{T}"] = e["som"].mean() if len(e) else np.nan
            r[f"cobertura_chuva_{T}"] = e["chuva"].mean() if len(e) else np.nan
            r[f"diferenca_{T}"] = r[f"cobertura_som_{T}"] - r[f"cobertura_chuva_{T}"]
            r[f"cobertura_som_seco_{T}"] = dry["som"].mean() if len(dry) else np.nan
        e = g[g["chuva_prox_1h"] >= 20]
        if len(e):
            r["regime_com_chuva_em_curso"] = e["chuva"].mean()
            r["regime_seca_3h"] = (e["prec_acum_3h"] == 0).mean()
            r["regime_seca_24h"] = (e["prec_acum_24h"] == 0).mean()
            r["regime_chuva_na_hora_mediana"] = e["prec_acum_1h"].median()
            ang = e["hora_solar"].to_numpy() / 24 * 2 * np.pi
            r["regime_hora_solar_media"] = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 15
            r["regime_fracao_tarde_12_18"] = ((e["hora_solar"] >= 12) & (e["hora_solar"] < 18)).mean()
            r["regime_fracao_madrugada_0_6"] = (e["hora_solar"] < 6).mean()
            r["regime_fracao_noturna_19_7"] = ((e["hora_solar"] >= 19) | (e["hora_solar"] < 7)).mean()
            r["regime_fracao_tarde_12_19"] = ((e["hora_solar"] >= 12) & (e["hora_solar"] < 19)).mean()
        rows.append(r)
    st = pd.DataFrame(rows).set_index("codigo").join(hours, how="left")
    return st.reset_index()


def coast_distance(st):
    try:
        import geopandas as gpd
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / "ne_10m_coastline.geojson"
        if not f.exists():
            req = urllib.request.Request(COAST_URL, headers={"User-Agent": "meteo-som-onset (research code)"})
            f.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        coast = gpd.read_file(f).cx[-80:-25, -40:10].to_crs(5880)
        pts = gpd.GeoSeries(gpd.points_from_xy(st.longitude, st.latitude), crs=4326).to_crs(5880)
        line = coast.geometry.union_all()
        return pts.distance(line).to_numpy() / 1000
    except Exception as e:  # noqa: BLE001
        log.info(f"distance to the coast not computed: {type(e).__name__}: {e}")
        return np.full(len(st), np.nan)


def map_attributes(st, path, prefix):
    if not path or not Path(path).exists():
        return st
    import geopandas as gpd
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(4674)
    g = g.to_crs(4326)
    cols = [c for c in g.columns if c != "geometry"
            and (g[c].dtype == object or pd.api.types.is_string_dtype(g[c])) and g[c].nunique() <= 40]
    marine = g[cols].astype(str).apply(lambda s: s.str.contains("Econ|Massa d|ilha|MARINHO|Oceano|Mar ", case=False)).any(axis=1)
    g = g[~marine & g.geometry.notna()].to_crs(5880)
    pts = gpd.GeoDataFrame(st[["codigo"]], geometry=gpd.points_from_xy(st.longitude, st.latitude), crs=4326).to_crs(5880)
    # stations on the coast can fall outside the 1:5,000,000 land polygons; they take the nearest land class
    j = gpd.sjoin_nearest(pts, g[cols + ["geometry"]], how="left", distance_col="dist_m").drop_duplicates("codigo")
    log.info(f"{prefix}: {int((j.dist_m > 0).sum())} stations outside the land polygons took the nearest class "
             f"(largest distance {j.dist_m.max() / 1000:.1f} km)")
    for c in cols:
        st[f"{prefix}_{c}"] = j.set_index("codigo").loc[st.codigo, c].to_numpy()
    log.info(f"{prefix} attributes: {', '.join(cols)}")
    return st


def pooled(df, by):
    rows = []
    for key, g in df.groupby(by, observed=True, dropna=False):
        r = {"variavel": by, "classe": key, "estacoes": len(g)}
        for T in THRESHOLDS:
            n = g[f"eventos_{T}"]
            r[f"eventos_{T}"] = int(n.sum())
            r[f"cobertura_som_{T}"] = (g[f"cobertura_som_{T}"] * n).sum() / max(n.sum(), 1)
            r[f"cobertura_chuva_{T}"] = (g[f"cobertura_chuva_{T}"] * n).sum() / max(n.sum(), 1)
            r[f"diferenca_{T}"] = r[f"cobertura_som_{T}"] - r[f"cobertura_chuva_{T}"]
        n = g["eventos_20"]
        r["regime_com_chuva_em_curso"] = (g["regime_com_chuva_em_curso"] * n).sum() / max(n.sum(), 1)
        r["regime_fracao_tarde_12_18"] = (g["regime_fracao_tarde_12_18"] * n).sum() / max(n.sum(), 1)
        rows.append(r)
    return rows


def main():
    warnings.filterwarnings("ignore")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "regional.log", mode="w"),
                                  logging.StreamHandler()])
    sel = pd.read_csv(AN / "conjunto_som.csv", sep=";")["unidade"].to_numpy()
    ev, hours = load_events(sel)
    st = station_table(ev, hours)
    meta = pd.read_csv(config.STATIONS_META, sep=";", dtype={"codigo": str})[
        ["codigo", "nome", "latitude", "longitude", "altitude"]]
    st = st.merge(meta, on="codigo", how="left")
    st["distancia_costa_km"] = coast_distance(st)
    wind = []
    for f in sorted(config.PARQUET_DIR.glob("*.parquet")):
        w = pd.read_parquet(f, columns=["vento_dir_sin", "vento_dir_cos", "vento_velocidade"])
        s_, c_ = (w.vento_dir_sin * w.vento_velocidade).mean(), (w.vento_dir_cos * w.vento_velocidade).mean()
        wind.append(dict(codigo=f.stem, vento_direcao_media=np.degrees(np.arctan2(s_, c_)) % 360,
                         vento_constancia=np.hypot(s_, c_) / max(w.vento_velocidade.mean(), 1e-9)))
    st = st.merge(pd.DataFrame(wind), on="codigo", how="left")
    st = map_attributes(st, getattr(config, "CLIMATE_SHP", None), "clima")
    st = map_attributes(st, getattr(config, "BIOMES_SHP", None), "bioma")
    st.to_csv(AN / "estacoes_regional.csv", sep=";", index=False)
    log.info(f"{len(st)} stations, {int((st.eventos_20 >= MIN_EVENTS).sum())} with at least {MIN_EVENTS} events of 20 mm")

    # classes
    st["faixa_distancia_costa"] = pd.cut(st.distancia_costa_km, [-1, 50, 150, 400, 5000],
                                         labels=["ate 50 km", "50 a 150 km", "150 a 400 km", "acima de 400 km"])
    st["faixa_altitude"] = pd.cut(st.altitude, [-100, 200, 500, 900, 3000],
                                  labels=["ate 200 m", "200 a 500 m", "500 a 900 m", "acima de 900 m"])
    st["faixa_chuva_em_curso"] = pd.cut(st.regime_com_chuva_em_curso, [0, 0.4, 0.5, 0.6, 0.7, 1.0],
                                        labels=["ate 40%", "40 a 50%", "50 a 60%", "60 a 70%", "acima de 70%"])
    cats = ["faixa_distancia_costa", "faixa_altitude", "faixa_chuva_em_curso"] + \
        [c for c in st.columns if c.startswith(("clima_", "bioma_"))]
    rows = []
    for c in cats:
        rows += pooled(st, c)
    cl = pd.DataFrame(rows)
    cl.to_csv(AN / "regional_classes.csv", sep=";", index=False)

    # coverage by narrow bands of distance to the coast, events of all partitions pooled
    edges = [-1, 25, 50, 600, 5000]
    st["faixa_costa_fina"] = pd.cut(st.distancia_costa_km, edges)
    rows = []
    for f, d in st.groupby("faixa_costa_fina", observed=True):
        for T in (20, 30, 40):
            n = d[f"eventos_{T}"]
            if n.sum() == 0:
                continue
            rows.append(dict(faixa_km=f"{max(f.left, 0):.0f} a {f.right:.0f}", limiar=T, estacoes=len(d),
                             estacoes_com_5_eventos=int((n >= 5).sum()), eventos=int(n.sum()),
                             fracao_eventos=n.sum() / st[f"eventos_{T}"].sum(),
                             cobertura_som=(d[f"cobertura_som_{T}"] * n).sum() / n.sum(),
                             cobertura_chuva=(d[f"cobertura_chuva_{T}"] * n).sum() / n.sum(),
                             eventos_com_chuva_em_curso=(d.regime_com_chuva_em_curso * d.eventos_20).sum() / d.eventos_20.sum(),
                             eventos_noturnos=(d.regime_fracao_noturna_19_7 * d.eventos_20).sum() / d.eventos_20.sum()))
    cf = pd.DataFrame(rows)
    cf.round(3).to_csv(AN / "regional_costa.csv", sep=";", index=False)
    log.info("\ncoverage by distance to the coast, all partitions\n" + cf[cf.limiar.isin([20, 30])].round(3).to_string(index=False))

    # rank correlations with the station difference
    s = st[st.eventos_20 >= MIN_EVENTS]
    num = ["latitude", "longitude", "altitude", "distancia_costa_km", "fracao_horas_chuva",
           "regime_com_chuva_em_curso", "regime_seca_3h", "regime_seca_24h", "regime_chuva_na_hora_mediana",
           "regime_fracao_tarde_12_18", "regime_fracao_madrugada_0_6"]
    cr = []
    for T in THRESHOLDS:
        for v in num:
            cr.append(dict(limiar=T, variavel=v, spearman=s[f"diferenca_{T}"].corr(s[v], method="spearman"),
                           estacoes=int(s[[f"diferenca_{T}", v]].dropna().shape[0])))
    cr = pd.DataFrame(cr)
    cr.to_csv(AN / "regional_correlacoes.csv", sep=";", index=False)

    pd.set_option("display.width", 220)
    show = ["classe", "estacoes", "eventos_20", "cobertura_som_20", "cobertura_chuva_20", "diferenca_20",
            "diferenca_30", "diferenca_40", "regime_com_chuva_em_curso", "regime_fracao_tarde_12_18"]
    for c in cats:
        t = cl[cl.variavel == c][show].copy()
        for x in show[3:]:
            t[x] = (t[x] * 100).round(1)
        log.info(f"\n{c}\n{t.to_string(index=False)}")
    def circ(x, w=None):
        a = np.deg2rad(x.dropna())
        return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360
    wtab = st.groupby("faixa_distancia_costa", observed=True).agg(
        estacoes=("codigo", "size"), direcao_media=("vento_direcao_media", circ),
        fracao_de_leste_45_135=("vento_direcao_media", lambda x: ((x >= 45) & (x <= 135)).mean()),
        constancia_media=("vento_constancia", "mean")).round(2)
    log.info(f"\nmean wind direction per station (speed-weighted vector mean), by distance to the coast\n{wtab}")
    log.info(f"all stations: direction {circ(st.vento_direcao_media):.0f}, "
             f"share from the east (45 to 135) {((st.vento_direcao_media >= 45) & (st.vento_direcao_media <= 135)).mean():.2f}")
    log.info(f"\nSpearman with SOM minus rain in progress (stations with >= {MIN_EVENTS} events of 20 mm)\n"
             f"{cr.pivot_table(index='variavel', columns='limiar', values='spearman').round(2)}")
    log.info("\ndone")
    with zipfile.ZipFile(config.LOG_DIR / "resultado_regional.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(config.LOG_DIR / "regional.log", "regional.log")
        for f in ["estacoes_regional.csv", "regional_classes.csv", "regional_correlacoes.csv"]:
            z.write(AN / f, f)


if __name__ == "__main__":
    main()
