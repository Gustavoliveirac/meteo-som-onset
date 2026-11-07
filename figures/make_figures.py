"""
make_figures.py - figures of the SOM study, in Portuguese and English.

    python figures/make_figures.py
    python figures/make_figures.py --only fig04_13 fig04_15

Reads the tables in config.RESULTS_DIR (analysis/coverage.py and
analysis/unit_stats.py), config.STATIONS_META and, for the maps, the IBGE
climate shapefile set as CLIMATE_SHP in config_local.py (docs/DATA.md).
Writes config.FIG_DIR/{pt,en}/<name>.png (and .pdf with --pdf). File names carry the
number of the figure in the chapter (fig04_07_..., fig04_18_...).
"""
import argparse
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AN = config.RESULTS_DIR
OUT = config.FIG_DIR
DPI = 300
MIN_HOURS = 10
SET_ORDER = ["train", "val_temporal", "val_spatial"]
SET_COLORS = {"train": "#4d4d4d", "val_temporal": "#2166ac", "val_spatial": "#b2182b"}
CRIT_COLORS = {"chuva_em_curso": "#7f7f7f", "precursor": "#4a8c4a", "precursor_seco": "#a6d96a", "som": "#2b5c8a"}
RAD = 1 / 3.6       # kJ/m2 per hour -> W/m2

T = {
    "pt": dict(
        train="Treinamento", val_temporal="Validação temporal", val_spatial="Validação espacial",
        row="Linha", col="Coluna", solar_hour="Hora solar local média", radiation="Radiação solar",
        pressao="Pressão", temperatura="Temperatura do ar", ponto_orvalho="Ponto de orvalho",
        umidade_relativa="Umidade relativa", radiacao="Radiação solar", vento_velocidade="Velocidade do vento",
        mean="Média", g1="Variação em 1 h", g6="Variação em 6 h", anom="Anomalia", ga1="Variação da anomalia em 1 h",
        direction="Direção", plus_panom="+ anomalia de pressão", plus_tanom="+ anomalia de temperatura",
        wind_dir="Direção do vento", next_h="Próximas {h} h", prob1="Horas com chuva acima de 1 mm (%)",
        horizon="Horizonte (h)", corr="Correlação de Pearson entre nós",
        anom_pressao="Anomalia de pressão", anom_temperatura="Anomalia de temperatura",
        anom_ponto_orvalho="Anomalia do ponto de orvalho", anom_umidade_relativa="Anomalia de umidade relativa",
        anom_vento_velocidade="Anomalia da velocidade do vento", anom_radiacao="Anomalia de radiação",
        grad1h_anom_pressao="Variação em 1 h da anomalia de pressão",
        grad6h_anom_pressao="Variação em 6 h da anomalia de pressão",
        grad1h_anom_temperatura="Variação em 1 h da anomalia de temperatura",
        grad6h_anom_temperatura="Variação em 6 h da anomalia de temperatura",
        events_frac="Fração dos eventos (%)", ongoing="Com chuva em curso", dry_h="Após {h} h sem chuva",
        threshold="Chuva na próxima hora igual ou acima de", no_rain="Sem chuva em curso",
        band="{a} a {b} mm", band_top="Acima de {a} mm", share40="Eventos acima de 40 mm por nó (%)", shareT="Eventos ≥ {t} mm por nó (%)",
        cum_y="Fração acumulada dos eventos (%)", cum_x="Fração dos nós do SOM (%)",
        coverage="Cobertura (%)", all_hours="(A) Todas as horas", dry_events="(B) Eventos sem chuva em curso",
        chuva_em_curso="Chuva em curso", precursor="Precursor individual", precursor_seco="Precursor das horas secas",
        som="SOM", diff="SOM menos chuva em curso (p.p.)", lat="Latitude (°)", lon="Longitude (°)",
        frac_hours="Fração das horas monitoradas", cov_frac="Cobertura dos eventos",
        area="área", zone="Zona climática", role="Papel da estação", role_train="Treinamento e validação temporal",
        role_spatial="Validação espacial", semiarid="Semiárido", semihumid="Semiúmido", humid="Úmido",
        superhumid="Superúmido", mm="mm",
        regime_rain="Eventos ≥ 20 mm com chuva em curso (%)", regime_hour="Hora solar média dos eventos ≥ 20 mm",
        periods=["0 a 6 h", "6 a 12 h", "12 a 18 h", "18 a 24 h"],
        period_en={},
    ),
    "en": dict(
        train="Training", val_temporal="Temporal validation", val_spatial="Spatial validation",
        row="Row", col="Column", solar_hour="Mean local solar hour", radiation="Solar radiation",
        pressao="Pressure", temperatura="Air temperature", ponto_orvalho="Dew point",
        umidade_relativa="Relative humidity", radiacao="Solar radiation", vento_velocidade="Wind speed",
        mean="Mean", g1="1-h change", g6="6-h change", anom="Anomaly", ga1="1-h change of the anomaly",
        direction="Direction", plus_panom="+ pressure anomaly", plus_tanom="+ temperature anomaly",
        wind_dir="Wind direction", next_h="Next {h} h", prob1="Hours with rainfall above 1 mm (%)",
        horizon="Horizon (h)", corr="Pearson correlation across units",
        anom_pressao="Pressure anomaly", anom_temperatura="Temperature anomaly",
        anom_ponto_orvalho="Dew point anomaly", anom_umidade_relativa="Relative humidity anomaly",
        anom_vento_velocidade="Wind speed anomaly", anom_radiacao="Radiation anomaly",
        grad1h_anom_pressao="1-h change of the pressure anomaly",
        grad6h_anom_pressao="6-h change of the pressure anomaly",
        grad1h_anom_temperatura="1-h change of the temperature anomaly",
        grad6h_anom_temperatura="6-h change of the temperature anomaly",
        events_frac="Share of events (%)", ongoing="With rain in progress", dry_h="After {h} h without rain",
        threshold="Next-hour rainfall at or above", no_rain="No rain in progress",
        band="{a} to {b} mm", band_top="Above {a} mm", share40="Events above 40 mm per unit (%)", shareT="Events ≥ {t} mm per unit (%)",
        cum_y="Cumulative share of events (%)", cum_x="Share of SOM units (%)",
        coverage="Coverage (%)", all_hours="(A) All hours", dry_events="(B) Events without rain in progress",
        chuva_em_curso="Rain in progress", precursor="Single precursor", precursor_seco="Dry-hour precursor",
        som="SOM", diff="SOM minus rain in progress (p.p.)", lat="Latitude (°)", lon="Longitude (°)",
        frac_hours="Share of monitored hours", cov_frac="Event coverage",
        area="area", zone="Climate zone", role="Station role", role_train="Training and temporal validation",
        role_spatial="Spatial validation", semiarid="Semi-arid", semihumid="Semi-humid", humid="Humid",
        superhumid="Super-humid", mm="mm",
        regime_rain="Events ≥ 20 mm with rain in progress (%)", regime_hour="Mean solar hour of events ≥ 20 mm",
        periods=["0 to 6 h", "6 to 12 h", "12 to 18 h", "18 to 24 h"],
        period_en={"madrugada": "night (early)", "manhã": "morning", "tarde": "afternoon",
                   "fim da tarde": "late afternoon", "noite": "evening"},
    ),
}
UNITS = dict(pressao="hPa", temperatura="°C", ponto_orvalho="°C", umidade_relativa="%",
             radiacao="W/m²", vento_velocidade="m/s")
STYLES = {
    "a_original": dict(
        vars=dict(pressao="RdYlBu", temperatura="RdYlBu_r", ponto_orvalho="BuGn", umidade_relativa="YlGnBu",
                  radiacao="YlOrRd", vento_velocidade="Blues"),
        div="RdBu_r", rain="YlGnBu", share="YlGnBu", hour="ciclo", diff="RdBu", cum="plasma",
        set_colors=dict(train="#4d4d4d", val_temporal="#2166ac", val_spatial="#b2182b"),
        crit_colors=dict(chuva_em_curso="#7f7f7f", precursor="#4a8c4a", precursor_seco="#a6d96a", som="#2b5c8a"),
        context=("#2b5c8a", "Oranges"), bands="Blues"),
}
S = STYLES["a_original"]
HOUR_CMAP = mcolors.LinearSegmentedColormap.from_list("ciclo", [
    (0.0, "#1b1f3b"), (0.125, "#1d7a8c"), (0.25, "#f2d27a"), (0.375, "#f08a24"), (0.5, "#c41f2e"),
    (0.625, "#d1207a"), (0.75, "#7b3fa8"), (0.875, "#5a6fe0"), (1.0, "#1b1f3b")], N=512)


def cmap_of(name):
    return HOUR_CMAP if name == "ciclo" else plt.get_cmap(name)


def cells(ax, m, cmap, vmin=None, vmax=None, norm=None, ticks=True):
    """Unit map drawn cell by cell with thin white gaps, row 1 at the top."""
    n_r, n_c = m.shape
    mm = np.ma.masked_invalid(m)
    cm = cmap_of(cmap) if isinstance(cmap, str) else cmap
    cm = cm.copy(); cm.set_bad("#e6e6e6")
    im = ax.pcolormesh(np.arange(n_c + 1) - 0.5, np.arange(n_r + 1) - 0.5, mm, cmap=cm, vmin=vmin, vmax=vmax,
                       norm=norm, edgecolors="white", linewidth=0.5)
    ax.set_xlim(-0.5, n_c - 0.5); ax.set_ylim(n_r - 0.5, -0.5); ax.set_aspect("equal")
    if ticks:
        ax.set_xticks([0, 4, 9, 14, 19]); ax.set_xticklabels([1, 5, 10, 15, 20])
        ax.set_yticks([0, 4, 9, 14, 19]); ax.set_yticklabels([1, 5, 10, 15, 20])
    else:
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#7f7f7f"); sp.set_linewidth(0.8)
    return im
# fixed colour of each group in every figure, visible over the blue and red of the diverging scales
GROUP_COLORS = ["#000000", "#1b9e77", "#e6ab02", "#7570b3", "#e7298a", "#a6761d", "#66a61e", "#7f7f7f"]
CMAPS = dict(pressao="RdYlBu", temperatura="RdYlBu_r", ponto_orvalho="BuGn", umidade_relativa="YlGnBu",
             radiacao="YlOrRd", vento_velocidade="Blues")
WIND_CMAP = mcolors.LinearSegmentedColormap.from_list("wind", [
    (0, "#CC0000"), (30 / 360, "#FF2200"), (50 / 360, "#FF7700"), (90 / 360, "#AADD00"), (110 / 360, "#00CC00"),
    (140 / 360, "#00CCAA"), (180 / 360, "#0044DD"), (240 / 360, "#000088"), (270 / 360, "#220055"),
    (330 / 360, "#880022"), (1, "#CC0000")], N=512)

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": True, "axes.spines.right": True,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


# ------------------------------------------------------------------ helpers
FORMATS = ["png"]


def save(fig, name, lang):
    d = OUT / lang
    d.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(d / f"{name}.{ext}", dpi=DPI)
    plt.close(fig)


def unit_stats():
    u = pd.read_csv(AN / "unidades_estatisticas.csv", sep=";")
    u["radiacao"] = u["radiacao"] * RAD
    for c in [c for c in u.columns if "radiacao" in c and c.startswith("grad") and "anom" not in c]:
        u[c] = u[c] * RAD
    return u


def grid(u, col, k="train"):
    d = u[u.conjunto == k]
    side = int(d.linha.max()) + 1
    m = np.full((side, side), np.nan)
    v = np.array(d[col], dtype=float)
    v[d.horas.to_numpy() < MIN_HOURS] = np.nan
    m[d.linha.to_numpy(), d.coluna.to_numpy()] = v
    return m


def smooth(m, sigma=0.8):
    nan = np.isnan(m)
    x = np.where(nan, np.nanmean(m), m)
    s = gaussian_filter(x, sigma)
    s[nan] = np.nan
    return s


def plane(ax, m, cmap, unit="", diverging=False, vmin=None, vmax=None, sm=True):
    x = smooth(m) if sm else m
    if diverging:
        lim = np.nanpercentile(np.abs(x), 97) or 1.0
        im = ax.imshow(x, cmap=S["div"], norm=mcolors.TwoSlopeNorm(0, -lim, lim), aspect="equal")
    else:
        im = ax.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    if unit:
        cb.set_label(unit, fontsize=10)
    return im


def contour(ax, m, color="k"):
    try:
        ax.contour(smooth(m), levels=6, colors=color, linewidths=0.55, alpha=0.6)
    except ValueError:
        pass


def arrows(ax, d):
    r, c = d.shape
    Y, X = np.mgrid[0:r, 0:c]
    th = np.deg2rad(d)
    ax.quiver(X, Y, -np.sin(th) * 0.6, np.cos(th) * 0.6, scale=1, scale_units="xy", angles="xy",
              width=0.004, headwidth=3.5, headlength=4, color="k", alpha=0.75, pivot="mid")


def climate_layer():
    shp = getattr(config, "CLIMATE_SHP", None)
    if not shp or not Path(shp).exists():
        return None
    import geopandas as gpd
    g = gpd.read_file(shp)
    if g.crs is None:
        g = g.set_crs(4674)          # IBGE maps without .prj are in SIRGAS 2000 geographic
    g = g.to_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g["geometry"] = g.geometry.buffer(0)
    col = next((c for c in g.columns if c != "geometry"
                and g[c].astype(str).str.lower().str.contains("mido|rido").any()), None)
    if col is None:
        return g.assign(zona="humid")
    s = g[col].astype(str).str.lower()
    g["zona"] = np.select([s.str.contains("semi") & s.str.contains("rido"), s.str.contains("semi"),
                           s.str.contains("super"), s.str.contains("mido")],
                          ["semiarid", "semihumid", "superhumid", "humid"], "")
    LOG.append("climate classes: " + "; ".join(f"{k} -> {v or 'dropped'}" for k, v in
                                               g.groupby(g[col].astype(str))["zona"].first().items()))
    g = g[g["zona"] != ""]
    sea = g.geometry.bounds["maxx"] > -34.0      # features reaching the ocean islands are the marine strip
    if sea.any():
        LOG.append(f"climate features dropped as marine: {int(sea.sum())}")
    g = g[~sea]
    return g.dissolve("zona").reset_index()


def _climate_or_none():
    try:
        return climate_layer()
    except Exception as e:  # noqa: BLE001
        LOG.append(f"climate map not used: {type(e).__name__}: {e}")
        return None


LOG = []
ZONE_COLORS = dict(semiarid="#d7301f", semihumid="#fc8d59", humid="#9ecae1", superhumid="#3182bd")


# ------------------------------------------------------------------ figures
def fig03_01(L, lang):
    meta = pd.read_csv(config.STATIONS_META, sep=";", dtype={"codigo": str})
    meta = meta[meta.horas_finais.fillna(0) > 0]
    sp = set(pd.read_csv(config.SPATIAL_STATIONS, sep=";", dtype=str)["codigo"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    g = _climate_or_none()
    if g is not None:
        for z in ["semiarid", "semihumid", "humid", "superhumid"]:
            part = g[g.zona == z]
            if len(part):
                part.plot(ax=ax, color=ZONE_COLORS[z], edgecolor="none", label=L[z])
        handles = [plt.Rectangle((0, 0), 1, 1, color=ZONE_COLORS[z]) for z in ZONE_COLORS]
        leg1 = ax.legend(handles, [L[z] for z in ZONE_COLORS], title=L["zone"], loc="lower left", fontsize=10, title_fontsize=11)
        ax.add_artist(leg1)
    tr = meta[~meta.codigo.isin(sp)]
    va = meta[meta.codigo.isin(sp)]
    h1 = ax.scatter(tr.longitude, tr.latitude, s=6, c="k", zorder=3)
    h2 = ax.scatter(va.longitude, va.latitude, s=28, c="gold", edgecolor="k", linewidth=0.6, zorder=4)
    ax.legend([h1, h2], [L["role_train"], L["role_spatial"]], title=L["role"], loc="lower right", fontsize=10, title_fontsize=11)
    ax.set_xlabel(L["lon"]); ax.set_ylabel(L["lat"]); ax.set_aspect("equal")
    save(fig, "fig03_01_estacoes" if lang == "pt" else "fig03_01_stations", lang)


def fig04_01(L, lang, u):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, col, cmap, lab, unit, tag, lim in [
            (axes[0], "hora_solar", S["hour"], L["solar_hour"], "h", "(A)", (0, 24)),
            (axes[1], "radiacao", S["vars"]["radiacao"], L["radiation"], "W/m²", "(B)", (0, None))]:
        im = cells(ax, grid(u, col), cmap, vmin=lim[0], vmax=lim[1])
        cb = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.03); cb.set_label(f"{lab} ({unit})")
        if col == "hora_solar":
            cb.set_ticks([0, 6, 12, 18, 24])
        ax.set_xlabel(L["col"]); ax.set_ylabel(L["row"]); ax.set_title(tag, loc="left", fontweight="bold", x=-0.12)
    fig.tight_layout()
    save(fig, "fig04_01_guia" if lang == "pt" else "fig04_01_guide", lang)


def fig04_02(L, lang, u):
    vars_ = ["pressao", "temperatura", "ponto_orvalho", "umidade_relativa", "radiacao", "vento_velocidade"]
    fig, axes = plt.subplots(len(vars_), 3, figsize=(8, 14))
    for i, v in enumerate(vars_):
        ms = [grid(u, v, k) for k in SET_ORDER]
        vmin = np.nanmin([np.nanpercentile(m, 2) for m in ms]); vmax = np.nanmax([np.nanpercentile(m, 98) for m in ms])
        for j, k in enumerate(SET_ORDER):
            ax = axes[i, j]
            im = cells(ax, ms[j], S["vars"][v], vmin=vmin, vmax=vmax, ticks=False)
            if j == 2:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                cax = inset_axes(ax, width="6%", height="85%", loc="center left",
                                 bbox_to_anchor=(1.04, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                plt.colorbar(im, cax=cax).set_label(UNITS[v], fontsize=10)
            if i == 0:
                ax.set_title(L[k], fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(L[v], fontsize=12)
    fig.tight_layout()
    save(fig, "fig04_02_planos" if lang == "pt" else "fig04_02_component_planes", lang)


def fig04_03(L, lang, u):
    rows = [("radiacao", None), ("temperatura", "radiacao"), ("umidade_relativa", "temperatura")]
    cols = [("", "mean"), ("grad1h_", "g1"), ("anom_", "anom"), ("grad1h_anom_", "ga1")]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5))
    for i, (v, cont) in enumerate(rows):
        for j, (p, lab) in enumerate(cols):
            ax = axes[i, j]
            key = f"{p}{v}"
            m = grid(u, key)
            unit = UNITS[v] if p == "" else (f"{UNITS[v]}/h" if p == "grad1h_" else ("z" if p == "anom_" else "z/h"))
            plane(ax, m, S["vars"][v], unit, diverging=p != "")
            if cont:
                contour(ax, grid(u, f"{p}{cont}"))
            if i == 0:
                ax.set_title(L[lab], fontsize=13)
            if j == 0:
                ax.set_ylabel(L[v], fontsize=13)
    fig.tight_layout()
    save(fig, "fig04_03_radiacao_temperatura_umidade" if lang == "pt" else "fig04_03_radiation_temperature_humidity", lang)


def _decomp(L, lang, u, v, name_pt, name_en):
    cols = [("", "mean", UNITS[v]), ("grad1h_", "g1", f"{UNITS[v]}/h"), ("anom_", "anom", "z"),
            ("grad1h_anom_", "ga1", "z/h"), ("grad6h_", "g6", f"{UNITS[v]}/6 h")]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.8))
    for ax, (p, lab, unit) in zip(axes, cols):
        plane(ax, grid(u, f"{p}{v}"), S["vars"][v], unit, diverging=p != "")
        ax.set_title(L[lab], fontsize=13)
    axes[0].set_ylabel(L[v], fontsize=13)
    fig.tight_layout()
    save(fig, name_pt if lang == "pt" else name_en, lang)


def fig04_04(L, lang, u):
    _decomp(L, lang, u, "pressao", "fig04_04_pressao", "fig04_04_pressure")


def fig04_06(L, lang, u):
    _decomp(L, lang, u, "ponto_orvalho", "fig04_06_ponto_orvalho", "fig04_06_dew_point")


def mark_wind_sectors(ax, u):
    """Units whose mean wind comes from the south (135 to 225 degrees) outlined in black,
    from the north (315 to 45 degrees) in white."""
    t = u[(u.conjunto == "train") & (u.horas >= MIN_HOURS)]
    for _, r in t.iterrows():
        dd = r.vento_direcao
        col = "black" if 135 <= dd <= 225 else ("white" if (dd >= 315 or dd <= 45) else None)
        if col:
            ax.add_patch(plt.Rectangle((r.coluna - 0.5, r.linha - 0.5), 1, 1, fill=False, lw=2.0, edgecolor=col))


def fig04_05(L, lang, u):
    d = grid(u, "vento_direcao")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6))
    for ax, (p, lab, unit) in zip(axes[0], [("", "mean", "m/s"), ("grad1h_", "g1", "m/s/h"), ("anom_", "anom", "z")]):
        plane(ax, grid(u, f"{p}vento_velocidade"), S["vars"]["vento_velocidade"], unit, diverging=p != "")
        arrows(ax, d)
        ax.set_title(L[lab], fontsize=13)
    axes[0, 0].set_ylabel(L["vento_velocidade"], fontsize=13)
    for ax, (cont, lab) in zip(axes[1], [(None, "direction"), ("anom_pressao", "plus_panom"), ("anom_temperatura", "plus_tanom")]):
        plane(ax, d, WIND_CMAP, "°", vmin=0, vmax=360, sm=False)
        if cont:
            contour(ax, grid(u, cont))
        arrows(ax, d)
        ax.set_title(L[lab], fontsize=13)
    axes[1, 0].set_ylabel(L["wind_dir"], fontsize=13)
    fig.tight_layout()
    save(fig, "fig04_05_vento" if lang == "pt" else "fig04_05_wind", lang)


def fig04_07(L, lang, u):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    ms = [grid(u, f"fracao_acima_1mm_{h}h") * 100 for h in range(1, 7)]
    vmax = np.nanpercentile(np.concatenate([m.ravel() for m in ms]), 99)
    for h, (ax, m) in enumerate(zip(axes.ravel(), ms), 1):
        im = cells(ax, m, S["rain"], vmin=0, vmax=vmax, ticks=False)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03).set_label("%", fontsize=10)
        ax.set_title(L["next_h"].format(h=h), fontsize=13)
    fig.suptitle(L["prob1"], fontsize=14)
    fig.tight_layout()
    save(fig, "fig04_09_chuva_1mm_horizontes" if lang == "pt" else "fig04_09_rain_1mm_horizons", lang)


def fig04_08(L, lang, u):
    d = u[(u.conjunto == "train") & (u.horas >= MIN_HOURS)]
    vars_ = ["anom_pressao", "anom_temperatura", "anom_ponto_orvalho", "anom_umidade_relativa",
             "anom_vento_velocidade", "anom_radiacao", "grad1h_anom_pressao", "grad6h_anom_pressao",
             "grad1h_anom_temperatura", "grad6h_anom_temperatura"]
    hs = config.HORIZONS
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cmap = plt.get_cmap("tab10")
    for i, v in enumerate(vars_):
        r = [np.corrcoef(d[v], d[f"chuva_media_{h}h"])[0, 1] for h in hs]
        ax.plot(range(len(hs)), r, marker="o", ms=3, lw=1.3, color=cmap(i), label=L[v],
                ls="-" if v.startswith("anom") else "--")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels(hs)
    ax.set_xlabel(L["horizon"]); ax.set_ylabel(L["corr"]); ax.set_ylim(-1, 1)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, "fig04_08_correlacao_horizontes" if lang == "pt" else "fig04_08_correlation_horizons", lang)


def fig04_09(L, lang):
    c = pd.read_csv(AN / "contexto_eventos.csv", sep=";")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    bars = [("com_chuva_em_curso", L["ongoing"])] + [(f"apos_seca_{h}h", L["dry_h"].format(h=h)) for h in (3, 6, 12, 24)]
    colors = [S["context"][0]] + list(plt.get_cmap(S["context"][1])(np.linspace(0.85, 0.35, 4)))
    w = 0.16
    for ax, k, tag in zip(axes, SET_ORDER, "ABC"):
        d = c[c.conjunto == k].sort_values("limiar")
        x = np.arange(len(d))
        for i, (col, lab) in enumerate(bars):
            ax.bar(x + (i - 2) * w, d[col] * 100, w, color=colors[i], label=lab)
        for xi, n in zip(x, d.eventos):
            ax.text(xi, 102, f"n={n:,}".replace(",", "." if lang == "pt" else ","), ha="center", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([f"{t} mm" for t in d.limiar])
        ax.set_title(f"({tag}) {L[k]}", loc="left", fontsize=13)
        ax.set_ylim(0, 108)
    axes[0].set_ylabel(L["events_frac"])
    axes[1].set_xlabel(L["threshold"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig04_12_seca_antecedente" if lang == "pt" else "fig04_12_antecedent_dry", lang)


def fig04_10(L, lang):
    c = pd.read_csv(AN / "contexto_eventos.csv", sep=";")
    d = c[(c.conjunto == "val_temporal") & c.limiar.isin([20, 30, 40])].sort_values("limiar")
    cols = [("sem_chuva_em_curso", L["no_rain"])]
    for a, b in [(0, 1), (1, 5), (5, 10), (10, 20)]:
        cols.append((f"chuva_em_curso_{a}_{b}", L["band"].format(a=a if a else "0,2" if lang == "pt" else "0.2", b=b)))
    cols.append(("chuva_em_curso_20_inf", L["band_top"].format(a=20)))
    colors = ["#d9d9d9"] + list(plt.get_cmap(S["bands"])(np.linspace(0.3, 0.95, 5)))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bottom = np.zeros(len(d))
    x = np.arange(len(d))
    for (col, lab), color in zip(cols, colors):
        v = d[col].to_numpy() * 100
        ax.bar(x, v, 0.6, bottom=bottom, color=color, label=lab, edgecolor="white", linewidth=0.5)
        for xi, b, vi in zip(x, bottom, v):
            if vi >= 4:
                ax.text(xi, b + vi / 2, f"{vi:.0f}%", ha="center", va="center", fontsize=10)
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([f"{t} mm" for t in d.limiar])
    ax.set_xlabel(L["threshold"]); ax.set_ylabel(L["events_frac"]); ax.set_ylim(0, 100)
    ax.legend(fontsize=10, frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left")
    fig.tight_layout()
    save(fig, "fig04_13_chuva_em_curso" if lang == "pt" else "fig04_13_rain_in_progress", lang)


def _groups():
    f = AN / "nos_monitorados.csv"
    return pd.read_csv(f, sep=";") if f.exists() else None


def outline_groups(ax, numbers=True, lw=3.6):
    """Monitored units outlined with the fixed colour of their group (thick lines), group number at its centre."""
    g = _groups()
    if g is None:
        return
    g = g[~g.grupo.isin(getattr(config, "GROUPS_EXCLUDED", []))]
    for _, r in g.iterrows():
        col = GROUP_COLORS[(int(r.grupo) - 1) % len(GROUP_COLORS)]
        if col.lower() == "#ffffff":                      # white needs a dark halo
            ax.add_patch(plt.Rectangle((r.coluna - 1.5, r.linha - 1.5), 1, 1, fill=False, lw=lw + 1.6, edgecolor="#333333"))
        ax.add_patch(plt.Rectangle((r.coluna - 1.5, r.linha - 1.5), 1, 1, fill=False, lw=lw, edgecolor=col))
    if numbers:
        for k, gg in g.groupby("grupo"):
            col = GROUP_COLORS[(int(k) - 1) % len(GROUP_COLORS)]
            txt = "black" if col.lower() in ("#ffffff", "#e6ab02") else "white"
            for part in _connected(gg):
                ax.text(np.mean([c for _, c in part]) - 1, np.mean([r for r, _ in part]) - 1, str(k), ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt,
                        bbox=dict(boxstyle="circle,pad=0.2", fc=col, ec="#333333", lw=0.8))

def outline_monitored(ax, color="#d7301f", lw=1.4):
    """Monitored units outlined in one colour, without group numbers (groups are introduced later)."""
    g = _groups()
    if g is None:
        return
    for _, r in g.iterrows():
        ax.add_patch(plt.Rectangle((r.coluna - 1.5, r.linha - 1.5), 1, 1, fill=False, lw=lw, edgecolor=color))


def fig04_11(L, lang):
    un = pd.read_csv(AN / "unidades.csv", sep=";")
    d = un[un.conjunto == "train"]
    side = int(d.linha.max()) + 1
    for T in [20]:
        m = np.full((side, side), np.nan)
        v = d[f"eventos_{T}"].to_numpy() / d[f"eventos_{T}"].sum() * 100
        m[d.linha, d.coluna] = np.where(v > 0, v, np.nan)
        fig, ax = plt.subplots(figsize=(6, 5.2))
        im = cells(ax, m, S["share"], vmin=0)
        outline_monitored(ax)
        ax.set_xlabel(L["col"]); ax.set_ylabel(L["row"])
        plt.colorbar(im, ax=ax, shrink=0.85).set_label(L["shareT"].format(t=T))
        fig.tight_layout()
        save(fig, ("fig04_11_participacao_20mm" if lang == "pt" else "fig04_11_share_20mm"), lang)


def fig04_12(L, lang):
    un = pd.read_csv(AN / "unidades.csv", sep=";")
    d = un[un.conjunto == "train"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap(S["cum"])
    for i, t in enumerate([1, 5, 10, 20, 30, 40]):
        e = np.sort(d[f"eventos_{t}"].to_numpy())[::-1]
        y = np.concatenate([[0], np.cumsum(e) / e.sum() * 100])
        x = np.linspace(0, 100, len(y))
        ax.plot(x, y, color=cmap(i / 6), lw=1.6, label=f"≥ {t} mm")
    ax.plot([0, 100], [0, 100], color="k", lw=0.6, ls=":")
    ax.set_xlabel(L["cum_x"]); ax.set_ylabel(L["cum_y"]); ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "fig04_10_concentracao" if lang == "pt" else "fig04_10_concentration", lang)


def fig04_13(L, lang):
    p = pd.read_csv(AN / "pontos.csv", sep=";")
    p = p[(p.conjunto == "val_temporal") & p.limiar.isin([20, 30, 40])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cond, crits, title in [(axes[0], "all", ["chuva_em_curso", "precursor", "som"], L["all_hours"]),
                                   (axes[1], "dry", ["precursor", "precursor_seco", "som"], L["dry_events"])]:
        d = p[p.condicao == cond]
        x = np.arange(3); w = 0.26
        for i, c in enumerate(crits):
            r = d[d.criterio == c].sort_values("limiar")
            y = r.cobertura.to_numpy() * 100
            err = np.vstack([y - r.cobertura_ic_inf * 100, r.cobertura_ic_sup * 100 - y])
            ax.bar(x + (i - 1) * w, y, w, color=S["crit_colors"][c], label=L[c], yerr=err, capsize=2,
                   error_kw=dict(lw=0.7))
            top = r.cobertura_ic_sup.to_numpy() * 100
            for xi, yi, ti in zip(x + (i - 1) * w, y, top):
                ax.text(xi, ti + 1.2, f"{yi:.1f}".replace(".", "," if lang == "pt" else "."), ha="center", fontsize=9.5)
        ax.set_xticks(x); ax.set_xticklabels(["20 mm", "30 mm", "40 mm"])
        ax.set_title(title, loc="left", fontsize=13); ax.set_ylim(0, 90)
        ax.legend(frameon=False, fontsize=11, loc="upper left")
    axes[0].set_ylabel(L["coverage"])
    fig.tight_layout()
    save(fig, "fig04_14_cobertura_criterios" if lang == "pt" else "fig04_14_coverage_criteria", lang)


def _brazil(ax):
    g = _climate_or_none()
    if g is not None:
        g.dissolve().plot(ax=ax, color="#f2f2f2", edgecolor="#9e9e9e", linewidth=0.5)


def fig04_14(L, lang, min_events=5):
    """Per station, coverage of the monitored set minus coverage of rain in progress, events of 20 (A) and 30 mm (B)."""
    s_ = pd.read_csv(AN / "estacoes_regional.csv", sep=";", dtype={"codigo": str})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, T, tag in zip(axes, [20, 30], "AB"):
        d = s_[s_[f"eventos_{T}"] >= min_events]
        _brazil(ax)
        sc = ax.scatter(d.longitude, d.latitude, c=d[f"diferenca_{T}"] * 100, cmap=S["diff"], vmin=-50, vmax=50,
                        s=16, edgecolor="k", linewidth=0.25, zorder=3)
        ax.set_title(f"({tag}) ≥ {T} mm", loc="left", fontsize=13)
        ax.set_xlabel(L["lon"]); ax.set_ylabel(""); ax.set_aspect("equal")
    axes[0].set_ylabel(L["lat"])
    plt.colorbar(sc, ax=axes, shrink=0.8, pad=0.01).set_label(L["diff"])
    save(fig, "fig04_16_diferenca_estacoes" if lang == "pt" else "fig04_16_station_difference", lang)

def fig04_16(L, lang, min_events=10):
    """Per station: share of the >= 20 mm events in the afternoon, 12 to 19 h (A), and at night, 19 to 7 h (B),
    local solar time; colour scales limited to the 2nd and 98th percentiles."""
    s_ = pd.read_csv(AN / "estacoes_regional.csv", sep=";", dtype={"codigo": str})
    s_ = s_[s_.eventos_20 >= min_events]
    panels = [("regime_fracao_tarde_12_19", "YlOrRd", "Eventos ≥ 20 mm entre 12 h e 19 h (%)", "Events ≥ 20 mm between 12 and 19 h (%)"),
              ("regime_fracao_noturna_19_7", "YlGnBu", "Eventos ≥ 20 mm entre 19 h e 7 h (%)", "Events ≥ 20 mm between 19 and 7 h (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, tag, (col, cmap, tpt, ten) in zip(axes, "AB", panels):
        _brazil(ax)
        v = s_[col] * 100
        lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        sc = ax.scatter(s_.longitude, s_.latitude, c=v.clip(lo, hi), cmap=cmap, vmin=lo, vmax=hi,
                        s=20, edgecolor="k", linewidth=0.3, zorder=3)
        plt.colorbar(sc, ax=ax, shrink=0.8, extend="both").set_label("%")
        ax.set_title(f"({tag}) {tpt if lang == 'pt' else ten}", loc="left", fontsize=12)
        ax.set_xlabel(L["lon"]); ax.set_ylabel(L["lat"]); ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "fig04_17_regime_estacoes" if lang == "pt" else "fig04_17_station_regime", lang)

def _node_map(ax, m, cmap, label, vmin=None, vmax=None, diverging=False):
    if diverging:
        lim = np.nanpercentile(np.abs(m), 97) or 1.0
        im = cells(ax, m, S["div"], norm=mcolors.TwoSlopeNorm(0, -lim, lim), ticks=False)
    else:
        im = cells(ax, m, cmap, vmin=vmin, vmax=vmax, ticks=False)
    outline_groups(ax, numbers=True, lw=1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)


def _grid_from(df, col):
    side = int(df.linha.max())
    m = np.full((side, side), np.nan)
    m[df.linha - 1, df.coluna - 1] = df[col]
    return m


def exp_hora_grupos(L, lang, u):
    h = pd.read_csv(AN / "grupos_hora_solar.csv", sep=";")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, col, title in [(axes[0], "fracao_horas", "Horas de cada grupo"),
                           (axes[1], "fracao_eventos", "Eventos ≥ 20 mm de cada grupo")]:
        for g, gg in h.groupby("grupo"):
            if g != "fora dos grupos" and int(g) in getattr(config, "GROUPS_EXCLUDED", []):
                continue
            if g == "fora dos grupos":
                ax.plot(gg.hora_solar, gg[col] * 100, color="#444444", lw=1.6, ls=(0, (5, 3)), label="nós não monitorados")
            else:
                c = GROUP_COLORS[(int(g) - 1) % len(GROUP_COLORS)]
                ax.plot(gg.hora_solar, gg[col] * 100, color=c if c.lower() != "#ffffff" else "#d9d9d9", lw=2.2, marker="o", ms=4,
                        mec="#333333", mew=0.6, label=f"grupo {g}")
        ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("Hora solar local"); ax.set_title(title, fontsize=13)
    axes[0].set_ylabel("% das horas ou dos eventos do grupo")
    h_, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=len(l_), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig04_19_grupos_horas" if lang == "pt" else "fig04_19_group_hours", lang)

def fig04_15(L, lang):
    c = pd.read_csv(AN / "curvas.csv", sep=";")
    a = pd.read_csv(AN / "areas.csv", sep=";")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, t in zip(axes, [20, 30, 40]):
        for k in SET_ORDER:
            d = c[(c.conjunto == k) & (c.limiar == t)]
            ar = a[(a.conjunto == k) & (a.limiar == t)].area.iloc[0]
            lab = f"{L[k]} ({L['area']} {ar:.2f})"
            if lang == "pt":
                lab = lab.replace(".", ",")
            ax.plot(np.r_[0, d.fracao_horas], np.r_[0, d.cobertura], color=S["set_colors"][k], lw=1.4, label=lab)
        ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
        ax.set_title(f"{t} mm", loc="left", fontsize=13)
        ax.set_xlabel(L["frac_hours"]); ax.legend(frameon=False, fontsize=10, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel(L["cov_frac"])
    fig.tight_layout()
    save(fig, "fig04_15_curvas_cobertura" if lang == "pt" else "fig04_15_coverage_curves", lang)


VAR_LABELS = {"temperatura": ("Temperatura", "°C"), "umidade_relativa": ("Umidade relativa", "pontos"),
              "radiacao_wm2": ("Radiação", "W/m²"), "vento_velocidade": ("Vento", "m/s"), "pressao": ("Pressão", "hPa")}


def _states():
    f = AN / "nos_estados.csv"
    return pd.read_csv(f, sep=";") if f.exists() else None


def outline_states(ax, lw=1.6):
    """Monitored units outlined in black, with the number of their state."""
    g = _states()
    if g is None:
        return
    for _, r in g.iterrows():
        ax.add_patch(plt.Rectangle((r.coluna - 1.5, r.linha - 1.5), 1, 1, fill=False, lw=lw, edgecolor="black"))
    for k, gg in g.groupby("grupo"):
        for part in _connected(gg):
            ax.text(np.mean([c for _, c in part]) - 1, np.mean([r for r, _ in part]) - 1, f"E{k}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.6))


def _connected(gg):
    cells_ = set(zip(gg.linha, gg.coluna)); parts = []
    while cells_:
        stack = [cells_.pop()]; part = []
        while stack:
            r, c = stack.pop(); part.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if n in cells_:
                        cells_.remove(n); stack.append(n)
        parts.append(part)
    return parts


STATE_MAPS = [("grad1h_temperatura", "Temperatura, variação em 1 h (°C)"),
              ("grad1h_ponto_orvalho", "Ponto de orvalho, variação em 1 h (°C)"),
              ("grad1h_umidade_relativa", "Umidade relativa, variação em 1 h (pontos)"),
              ("grad1h_pressao", "Pressão, variação em 1 h (hPa)"),
              ("grad1h_vento_velocidade", "Vento, variação em 1 h (m/s)"),
              ("anom_temperatura", "Temperatura, anomalia (desvios padrão)")]


def fig_estados_assinaturas(L, lang, u):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for ax, (col, lab) in zip(axes.ravel(), STATE_MAPS):
        m = grid(u, col)
        lim = np.nanpercentile(np.abs(m), 97) or 1.0
        im = cells(ax, m, S["div"], norm=mcolors.TwoSlopeNorm(0, -lim, lim), ticks=False)
        outline_groups(ax)
        ax.set_title(lab, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save(fig, "fig04_18_grupos_assinaturas" if lang == "pt" else "fig04_18_group_signatures", lang)


def fig_chuva_media(L, lang, u):
    """Mean rainfall in the next 1, 6 and 24 h after the hours of each unit."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, h in zip(axes, (1, 6, 24)):
        m = grid(u, f"chuva_media_{h}h")
        im = cells(ax, m, S["rain"], vmin=0, ticks=False)
        ax.set_title((f"Próximas {h} h" if h > 1 else "Próxima 1 h") if lang == "pt" else f"Next {h} h", fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.8).set_label("mm")
    fig.tight_layout()
    save(fig, "fig04_07_chuva_media" if lang == "pt" else "fig04_07_mean_rainfall", lang)


TAB_LABELS = {
    "pt": dict(
        info=[("metade_dos_eventos", "Metade dos\neventos\n(hora solar)", None, None), ("fracao_horas", "Horas\n(%)", 100, 1), ("fracao_eventos_20", "Eventos de\n20 mm (%)", 100, 1),
              ("eventos_por_mil_horas", "Eventos\npor mil\nhoras", 1, 1), ("horas_com_chuva_em_curso", "Horas com\nchuva em\ncurso (%)", 100, 0),
              ("eventos_partindo_de_hora_seca", "Eventos a\npartir de hora\nseca (%)", 100, 0)],
        var=[("variacao_1h_temperatura", "Temperatura\n(°C)", 1), ("ponto_orvalho", "Ponto de\norvalho (°C)", 1),
             ("variacao_1h_umidade", "Umidade\nrelativa (pontos)", 1), ("variacao_1h_radiacao_wm2", "Radiação\n(W/m²)", 0),
             ("variacao_1h_vento", "Vento\n(m/s)", 1), ("variacao_1h_pressao", "Pressão\n(hPa)", 2)],
        group="Grupo {g}", half="{h}", ta="(A) Horas e eventos", tb="(B) Variação na última hora",
        name="fig04_20_grupos_tabela", dec=","),
    "en": dict(
        info=[("metade_dos_eventos", "Half of the\nevents\n(solar hour)", None, None), ("fracao_horas", "Hours\n(%)", 100, 1), ("fracao_eventos_20", "20 mm\nevents (%)", 100, 1),
              ("eventos_por_mil_horas", "Events per\nthousand\nhours", 1, 1), ("horas_com_chuva_em_curso", "Hours with\nrain in\nprogress (%)", 100, 0),
              ("eventos_partindo_de_hora_seca", "Events\nfrom a dry\nhour (%)", 100, 0)],
        var=[("variacao_1h_temperatura", "Temperature\n(°C)", 1), ("ponto_orvalho", "Dew point\n(°C)", 1),
             ("variacao_1h_umidade", "Relative\nhumidity (points)", 1), ("variacao_1h_radiacao_wm2", "Radiation\n(W/m²)", 0),
             ("variacao_1h_vento", "Wind\n(m/s)", 1), ("variacao_1h_pressao", "Pressure\n(hPa)", 2)],
        group="Group {g}", half="{h}", ta="(A) Hours and events", tb="(B) Change in the last hour",
        name="fig04_20_group_table", dec="."),
}


def fig04_20(L, lang):
    """Table 4 as a figure: groups ordered by the time of their events, each cell coloured within its column."""
    TL = TAB_LABELS[lang]
    t = pd.read_csv(AN / "tabela_grupos.csv", sep=";")
    t = t[~t.grupo.isin(getattr(config, "GROUPS_EXCLUDED", []))].copy()
    f = AN / "grupos_fisico.csv"
    if f.exists():
        fis = pd.read_csv(f, sep=";")
        fis = fis[fis.grupo.astype(str) != "fora dos grupos"]
        fis["grupo"] = fis.grupo.astype(int)
        t = t.merge(fis[["grupo", "grad1h_ponto_orvalho"]].rename(columns={"grad1h_ponto_orvalho": "ponto_orvalho"}),
                    on="grupo", how="left")
    else:
        t["ponto_orvalho"] = np.nan
    start = t.metade_dos_eventos.str.extract(r"(\d+)")[0].astype(int)
    t = t.sort_values("grupo").reset_index(drop=True)
    half = t.metade_dos_eventos.str.replace(" h", "").str.replace(" a ", "–")
    if lang == "en":
        half = half.str.replace("–", "–")

    def fmt(v, nd):
        v = float(np.floor(abs(v) * 10 ** nd + 0.5) / 10 ** nd) * np.sign(v)
        s = f"{v:+.{nd}f}" if v != 0 else f"{0:.{nd}f}"
        return s.replace("-", "−").replace(".", TL["dec"])

    fig = plt.figure(figsize=(16, 0.62 * len(t) + 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(TL["info"]), len(TL["var"])], wspace=0.04)
    rows = [TL["group"].format(g=int(g)) for g in t.grupo]
    for k, (cols, title, diverging) in enumerate([(TL["info"], TL["ta"], False), (TL["var"], TL["tb"], True)]):
        ax = fig.add_subplot(gs[0, k])
        z = np.zeros((len(t), len(cols)))
        for j, c in enumerate(cols):
            if c[2] is None:
                z[:, j] = 0.12
                continue
            v = t[c[0]].to_numpy(float) * (c[2] if not diverging else 1)
            z[:, j] = v / np.nanmax(np.abs(v)) if np.nanmax(np.abs(v)) > 0 else 0
        if diverging:
            ax.imshow(z, cmap=S["div"], vmin=-1, vmax=1, aspect="auto")
        else:
            ax.imshow(z, cmap="Blues", vmin=0, vmax=1.25, aspect="auto")
        for i in range(len(t)):
            for j, c in enumerate(cols):
                if diverging:
                    v = t[c[0]].iloc[i]
                    s = "" if pd.isna(v) else fmt(v, c[2])
                elif c[2] is None:
                    s = half.iloc[i] + " h"
                else:
                    v = t[c[0]].iloc[i] * c[2]
                    v = np.floor(v * 10 ** c[3] + 0.5) / 10 ** c[3]
                    s = f"{v:.{c[3]}f}".replace(".", TL["dec"])
                ax.text(j, i, s, ha="center", va="center", fontsize=12,
                        color="white" if abs(z[i, j]) > (0.62 if diverging else 0.8) else "black")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], fontsize=10.5)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(t))); ax.set_yticklabels(rows if k == 0 else [], fontsize=10.5)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(t)), minor=True)
        ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(title, fontsize=13, pad=62, loc="left")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.68, bottom=0.03)
    save(fig, TL["name"], lang)


# figures in the order of the chapter; file names carry the chapter number
FIGS = dict(fig03_01=(fig03_01, False), fig04_01=(fig04_01, True), fig04_02=(fig04_02, True),
            fig04_03=(fig04_03, True), fig04_04=(fig04_04, True), fig04_05=(fig04_05, True),
            fig04_06=(fig04_06, True), fig04_07=(fig_chuva_media, True), fig04_08=(fig04_08, True),
            fig04_09=(fig04_07, True), fig04_10=(fig04_12, False), fig04_11=(fig04_11, False),
            fig04_12=(fig04_09, False), fig04_13=(fig04_10, False), fig04_14=(fig04_13, False),
            fig04_15=(fig04_15, False), fig04_16=(fig04_14, False), fig04_17=(fig04_16, False),
            fig04_18=(fig_estados_assinaturas, True), fig04_19=(exp_hora_grupos, True),
            fig04_20=(fig04_20, False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--pdf", action="store_true", help="also write vector PDF files (useful for LaTeX)")
    a = ap.parse_args()
    if a.pdf:
        FORMATS.append("pdf")
    names = a.only or list(FIGS)
    if a.only is None and OUT.exists():
        import shutil
        shutil.rmtree(OUT)                     # a full run starts clean, so old file names do not linger
    u = unit_stats()
    for lang in ("pt", "en"):
        for n in names:
            f, needs_u = FIGS[n]
            try:
                f(T[lang], lang, u) if needs_u else f(T[lang], lang)
                LOG.append(f"{lang} {n} ok")
            except Exception:  # noqa: BLE001
                LOG.append(f"{lang} {n} FAILED\n{traceback.format_exc()}")
            print(LOG[-1], flush=True)
    LOG.append(f"figures in {OUT}")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR / "figures.log").write_text("\n".join(LOG) + "\n")
    print(f"figures in {OUT}, log in {config.LOG_DIR / 'figures.log'}")


if __name__ == "__main__":
    main()
