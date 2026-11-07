"""
inmet_features.py - quality control and features for the INMET automatic stations.

The steps are described in docs/PROCESSING.md and their parameters are in config.QC.

Per station
  1. read and merge the BDMEP files (duplicate hours keep the first file in name order)
  2. quality control of each hour (null essentials, physical limits, hourly jumps)
  3. rainfall checks (repeated values, days above the discard limit, neighbours)
  4. climatology per month and hour from the reference period (training period)
  5. anomalies, gradients over 1, 3 and 6 h, future rainfall
  6. keeps hours of active months whose 6 previous hours are contiguous and valid
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

Q = config.QC

# ----------------------------------------------------------------- reading
COLUMN_RULES = [
    ("precipitacao", [["PRECIPITACAO"]]),
    ("pressao", [["PRESSAO", "ESTACAO", "HORARIA"], ["PRESSAO", "ESTACAO"]]),
    ("pressao_nivel_mar", [["PRESSAO", "MAR"], ["PRESSAO", "NIVEL"]]),
    ("pressao_max", [["PRESSAO", "MAX"]]),
    ("pressao_min", [["PRESSAO", "MIN"]]),
    ("radiacao", [["RADIACAO"]]),
    ("temperatura", [["TEMPERATURA", "BULBO"], ["TEMPERATURA", "AR", "HORARIA"]]),
    ("ponto_orvalho", [["TEMPERATURA", "PONTO", "ORVALHO"], ["PONTO", "ORVALHO"],
                       ["ORVALHO", "HORARIA"]]),
    ("umidade_relativa", [["UMIDADE", "RELATIVA", "HORARIA"], ["UMIDADE", "HORARIA"]]),
    ("vento_direcao", [["VENTO", "DIRECAO"]]),
    ("vento_rajada", [["VENTO", "RAJADA"]]),
    ("vento_velocidade", [["VENTO", "VELOCIDADE"]]),
]
RAW_VARS = ["precipitacao", "pressao", "radiacao", "temperatura", "ponto_orvalho",
            "umidade_relativa", "vento_direcao", "vento_velocidade"]
ESSENTIAL = ["temperatura", "ponto_orvalho", "pressao", "umidade_relativa", "precipitacao"]
NULL_TOKENS = {"", "-9999", "-9999.0", "null", "//", "-"}


def _num(s):
    s = (s or "").strip().replace(",", ".")
    if s in NULL_TOKENS:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def parse_header(lines):
    info = dict(codigo="", nome="", latitude=np.nan, longitude=np.nan, altitude=np.nan)
    for line in lines[:15]:
        t = line.strip().rstrip(";")
        i = t.find(":")
        if i <= 0:
            continue
        key, val = t[:i].strip().upper(), t[i + 1:].strip()
        if key == "NOME" or key.startswith("NOME "):
            info["nome"] = val
        elif "CODIGO" in key:
            m = re.search(r"[A-Z]\d{3,4}", val)
            info["codigo"] = m.group(0) if m else val
        elif "LATITUDE" in key:
            info["latitude"] = _num(val)
        elif "LONGITUDE" in key:
            info["longitude"] = _num(val)
        elif "ALTITUDE" in key:
            info["altitude"] = _num(val)
    return info


def _map_columns(header):
    mapa = {}
    cols = header.split(";")
    for c in range(2, len(cols)):
        col = cols[c].upper()
        if not col.strip():
            continue
        best, best_n = None, -1
        for campo, rules in COLUMN_RULES:
            if campo in mapa:
                continue
            for rule in rules:
                if all(f in col for f in rule) and len(rule) > best_n:
                    best_n, best = len(rule), campo
        if best is not None:
            mapa[best] = c
    return mapa


def _parse_dates(dates, hours):
    h = pd.Series(hours, dtype=str).str.replace(":", "", regex=False).str.rjust(4, "0")
    hi = pd.to_numeric(h, errors="coerce")
    hh, mi = hi // 100, hi % 100
    bad = hi.isna() | (hh > 23) | (mi != 0)
    d = pd.Series(dates, dtype=str)
    day = pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%Y/%m/%d", "%d/%m/%Y"):
        miss = day.isna()
        if miss.any():
            day[miss] = pd.to_datetime(d[miss], format=fmt, errors="coerce")
    out = day + pd.to_timedelta(hh.fillna(0), unit="h")
    out[bad] = pd.NaT
    return out


def read_bdmep_file(path):
    lines = _lines(path)
    info = parse_header(lines)
    cab = next((i for i, l in enumerate(lines) if l.startswith("Data Medicao;Hora Medicao")), -1)
    if cab < 0:
        return info, pd.DataFrame(columns=["dt"] + RAW_VARS)
    mapa = _map_columns(lines[cab])
    dates, hours, cols = [], [], {v: [] for v in RAW_VARS}
    for line in lines[cab + 1:]:
        p = line.split(";")
        if len(p) < 3:
            continue
        dates.append(p[0].strip()); hours.append(p[1].strip())
        for v in RAW_VARS:
            i = mapa.get(v)
            cols[v].append(p[i] if i is not None and i < len(p) else "")
    df = pd.DataFrame({"dt": _parse_dates(dates, hours)})
    for v in RAW_VARS:
        x = pd.Series(cols[v], dtype=str).str.strip().str.replace(",", ".", regex=False)
        x[x.isin(NULL_TOKENS)] = ""
        df[v] = pd.to_numeric(x, errors="coerce").astype("float64")
    df = df[df["dt"].notna()]
    return info, df.sort_values("dt", kind="stable").reset_index(drop=True)


def group_files_by_station(raw_dir):
    groups = {}
    for f in sorted(Path(raw_dir).glob("*.csv"), key=lambda p: p.name):
        try:
            code = parse_header(_lines(f)[:15])["codigo"]
        except OSError:
            continue
        if code:
            groups.setdefault(code, []).append(f)
    return groups


def read_station(files):
    """All files of one station on a complete hourly index (missing hours become NaN rows)."""
    info, frames = None, []
    for f in files:
        est, df = read_bdmep_file(f)
        info = info or est
        if est["nome"]:
            info["nome"] = est["nome"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("dt", keep="first")
    df = df.set_index("dt").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full).rename_axis("dt").reset_index()
    return info, df


def daily_rain(df):
    """Daily totals over the hours that have a rainfall value (used for the neighbour check)."""
    p = df.set_index("dt")["precipitacao"]
    g = p.groupby(p.index.normalize())
    return pd.DataFrame({"total": g.sum(min_count=1), "horas": g.count()}).dropna()

# ----------------------------------------------------------------- quality control


def solar_hour(dt, lon):
    return ((dt.dt.hour + dt.dt.minute / 60.0 + lon / 15.0) % 24).to_numpy()


def fix_radiation(df, lon):
    """Night hours (solar time) with negative radiation, or with no value on a day the
    sensor reported during daylight, are set to zero."""
    sh = solar_hour(df["dt"], lon)
    night = (sh < Q["night_end_solar"]) | (sh >= Q["night_start_solar"])
    rad = df["radiacao"].to_numpy(copy=True)
    solar_day = (df["dt"] + pd.to_timedelta(lon / 15.0, unit="h")).dt.normalize()
    day_has = pd.Series(~np.isnan(rad) & ~night).groupby(solar_day.to_numpy()).transform("any").to_numpy()
    filled = night & np.isnan(rad) & day_has
    rad[night & (rad < 0)] = 0.0
    rad[filled] = 0.0
    df["radiacao"] = rad
    return int(filled.sum())


def quality_control(df, altitude):
    """Returns quality (1 good, 2 minor, 3 bad) and the reason codes per hour."""
    n = len(df)
    v = {k: df[k].to_numpy() for k in RAW_VARS}
    p_min = max(1013.25 * np.exp(-(altitude if np.isfinite(altitude) else 0) / 8500.0)
                * Q["p_margin_alt"], 500.0)
    checks = [
        ("NULL_T", np.isnan(v["temperatura"]), 3), ("NULL_TD", np.isnan(v["ponto_orvalho"]), 3),
        ("NULL_P", np.isnan(v["pressao"]), 3), ("NULL_UR", np.isnan(v["umidade_relativa"]), 3),
        ("NULL_PREC", np.isnan(v["precipitacao"]), 3),
        ("LIM_T", (v["temperatura"] < Q["t_min"]) | (v["temperatura"] > Q["t_max"]), 3),
        ("LIM_TD", (v["ponto_orvalho"] < Q["td_min"]) | (v["ponto_orvalho"] > Q["td_max"]), 3),
        ("TD_GT_T", v["ponto_orvalho"] > v["temperatura"] + 0.5, 3),
        ("LIM_P", (v["pressao"] < p_min) | (v["pressao"] > Q["p_max"]), 3),
        ("LIM_UR", (v["umidade_relativa"] < Q["ur_min"]) | (v["umidade_relativa"] > Q["ur_max"]), 3),
        ("LIM_PREC", (v["precipitacao"] < 0) | (v["precipitacao"] > Q["prec_max_hour"]), 3),
        ("LIM_RAD", (v["radiacao"] < Q["rad_min"]) | (v["radiacao"] > Q["rad_max"]), 3),
        ("LIM_VV", (v["vento_velocidade"] < 0) | (v["vento_velocidade"] > Q["vv_max"]), 3),
    ]
    for code, k, lim in [("GRAD_T", "temperatura", Q["jump_t"]), ("GRAD_TD", "ponto_orvalho", Q["jump_td"]),
                         ("GRAD_P", "pressao", Q["jump_p"]), ("GRAD_UR", "umidade_relativa", Q["jump_ur"])]:
        d = np.full(n, np.nan)
        d[1:] = np.abs(np.diff(v[k]))
        checks.append((code, d > lim * Q["jump_bad_factor"], 3))
        checks.append((code, (d > lim) & ~(d > lim * Q["jump_bad_factor"]), 2))
    quality = np.ones(n, dtype=np.int8)
    reasons = [[] for _ in range(n)]
    for code, mask, level in checks:
        for i in np.flatnonzero(mask):
            if code not in reasons[i]:
                reasons[i].append(code)
            quality[i] = max(quality[i], level)
    return quality, reasons


def rain_checks(df, quality, reasons, info, neighbours_daily):
    """Repeated hourly values, implausible days and confirmation by neighbours.
    Returns counts per check."""
    pr = df["precipitacao"].to_numpy()
    cnt = dict(prec_dup=0, dia_desc=0, dia_viz_desc=0, dia_viz_conf=0, dia_viz_sem=0)

    def mark(rows, code):
        for i in rows:
            quality[i] = 3
            if code not in reasons[i]:
                reasons[i].append(code)

    # same value (>= limit) in two consecutive hours
    a = pr[:-1]; b = pr[1:]
    pair = np.flatnonzero((a >= Q["prec_dup_min"]) & (np.abs(b - a) < 0.1))
    dup = np.unique(np.concatenate([pair, pair + 1]))      # both hours: which one is right is unknown
    mark(dup, "PREC_DUP"); cnt["prec_dup"] = len(dup)

    day = df["dt"].dt.normalize()
    tot = pd.Series(pr).groupby(day.to_numpy()).sum(min_count=1)
    has = ~np.isnan(pr)
    for d in tot.index[tot >= Q["prec_day_discard"]]:
        mark(np.flatnonzero(has & (day == d).to_numpy()), "PREC_DIA_DESC"); cnt["dia_desc"] += 1
    for d in tot.index[(tot >= Q["prec_day_check"]) & (tot < Q["prec_day_discard"])]:
        vals = []
        for code, dist, daily in neighbours_daily:
            if d in daily.index:
                vals.append(daily.at[d, "total"])
            if len(vals) >= Q["neigh_n"]:
                break
        if not vals:
            cnt["dia_viz_sem"] += 1
        elif max(vals) >= Q["neigh_confirm_mm"]:
            cnt["dia_viz_conf"] += 1
        else:
            mark(np.flatnonzero(has & (day == d).to_numpy()), "PREC_VIZ"); cnt["dia_viz_desc"] += 1
    return cnt

# ----------------------------------------------------------------- climatology


def reference_mask(df, quality):
    """Hours used for the climatology: good or minor quality, up to the training cutoff.
    Stations with less than a year of such hours use all of their valid hours."""
    ok = quality < 3
    ref = ok & (df["dt"] <= config.TRAIN_CUTOFF).to_numpy()
    if ref.sum() < 365 * 24 * Q["ref_min_fraction"]:
        return ok, False
    return ref, True


def active_months(df, ref):
    pr = df["precipitacao"].where(ref)
    g = pd.DataFrame({"y": df["dt"].dt.year, "m": df["dt"].dt.month, "p": pr})
    by = g.groupby(["m", "y"])["p"].agg(["sum", "count"])
    by = by[by["count"] > Q["month_min_hours"]]
    mean = by.groupby(level="m")["sum"].mean()
    return set(int(m) for m in mean.index[mean >= config.ACTIVE_MONTH_MM])


def climatology(dt, ref, values):
    """Mean and standard deviation per month and hour over the reference hours."""
    x = pd.Series(values).where(ref)
    key = pd.DataFrame({"m": dt.dt.month.to_numpy(), "h": dt.dt.hour.to_numpy(), "x": x.to_numpy()})
    g = key.groupby(["m", "h"])["x"].agg(["mean", "std", "count"])
    allv = x.dropna()
    mu = np.full((13, 24), allv.mean() if len(allv) else 0.0)
    sd = np.full((13, 24), allv.std() if len(allv) > 1 else 1.0)
    good = g[g["count"] >= Q["clim_min_count"]]
    for (m, h), r in good.iterrows():
        mu[m, h] = r["mean"]
        sd[m, h] = r["std"] if r["std"] > 0 else 1.0
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    return mu, sd

# ----------------------------------------------------------------- features


def theta_e(T, Td, P):
    with np.errstate(all="ignore"):
        Tk, Tdk = T + 273.15, Td + 273.15
        esd = 6.112 * np.exp(17.67 * Td / (Td + 243.5))
        w = 0.622 * esd / (P - esd)
        TL = 1.0 / (1.0 / (Tdk - 56.0) + np.log(Tk / Tdk) / 800.0) + 56.0
        th = Tk * np.power(1000.0 / P, 0.2854 * (1.0 - 0.28e-3 * w * 1000.0))
        return th * np.exp((3.376 / TL - 0.00254) * w * 1000.0 * (1.0 + 0.81e-3 * w * 1000.0))


def _change(x, k):
    """x(t) - x(t-k), NaN when either is missing (equal to the sum of the k hourly changes)."""
    out = np.full(len(x), np.nan)
    out[k:] = x[k:] - x[:-k]
    return out


def build_station(info, df, neighbours_daily):
    """Returns (DataFrame for the parquet, summary dict)."""
    s = dict(codigo=info["codigo"], nome=info["nome"], latitude=info["latitude"],
             longitude=info["longitude"], altitude=info["altitude"], horas_brutas=len(df))
    df = df[(df["dt"] >= config.DATA_START) & (df["dt"] <= config.DATA_END)].reset_index(drop=True)
    if len(df) < 24 * 30:
        s["motivo_sem_dados"] = "less than 30 days"
        return None, s
    lon = info["longitude"] if np.isfinite(info["longitude"]) else -45.0
    s["rad_noite_preenchida"] = fix_radiation(df, lon)
    quality, reasons = quality_control(df, info["altitude"])
    s.update(rain_checks(df, quality, reasons, info, neighbours_daily))
    ref, ref_train = reference_mask(df, quality)
    s["climatologia_periodo_treino"] = ref_train
    act = active_months(df, ref)
    s["meses_ativos"] = ",".join(map(str, sorted(act)))
    if not act:
        s["motivo_sem_dados"] = "no active month"
        return None, s

    month = df["dt"].dt.month.to_numpy(); hour = df["dt"].dt.hour.to_numpy()
    in_act = np.isin(month, list(act))
    s["qc_horas_meses_ativos"] = int(in_act.sum())
    cats = {"qc_campo_essencial_ausente": ("NULL_",), "qc_limite_fisico": ("LIM_", "TD_GT_T"),
            "qc_variacao_horaria": ("GRAD_",), "qc_chuva": ("PREC_",)}
    bad = in_act & (quality == 3)
    for name, prefixes in cats.items():
        s[name] = int(sum(1 for i in np.flatnonzero(bad) if any(r.startswith(prefixes) for r in reasons[i])))
    s["qc_removidas_total"] = int(bad.sum())
    valid = quality < 3                          # bad hours become missing
    raw = {k: np.where(valid, df[k].to_numpy(), np.nan) for k in RAW_VARS}
    raw["theta_e"] = theta_e(raw["temperatura"], raw["ponto_orvalho"], raw["pressao"])

    f = {}
    f["pressao"], f["temperatura"] = raw["pressao"], raw["temperatura"]
    f["ponto_orvalho"], f["umidade_relativa"] = raw["ponto_orvalho"], raw["umidade_relativa"]
    f["radiacao"], f["vento_velocidade"], f["theta_e"] = raw["radiacao"], raw["vento_velocidade"], raw["theta_e"]
    wd = np.deg2rad(raw["vento_direcao"])
    f["vento_dir_sin"], f["vento_dir_cos"] = np.sin(wd), np.cos(wd)
    for v in config.ANOM_VARS:
        mu, sd = climatology(df["dt"], ref, raw[v])
        f[f"anom_{v}"] = np.clip((raw[v] - mu[month, hour]) / sd[month, hour], -4.0, 4.0)
    for k in config.GRAD_WINDOWS:
        for v in config.GRAD_VARS[k]:
            f[f"grad{k}h_{v}"] = _change(raw[v], k)
        for v in config.GRAD_ANOM_VARS[k]:
            f[f"grad{k}h_anom_{v}"] = _change(f[f"anom_{v}"], k)
    p = raw["precipitacao"]
    for k in (1, 2, 3, 6, 12, 24):
        f[f"prec_acum_{k}h"] = pd.Series(p).rolling(k, min_periods=k).sum().to_numpy()
    for k in config.HORIZONS:
        f[f"chuva_prox_{k}h"] = pd.Series(p[::-1]).rolling(k, min_periods=k).sum().to_numpy()[::-1]
        f[f"chuva_prox_{k}h"] = np.concatenate([f[f"chuva_prox_{k}h"][1:], [np.nan]])

    # hours kept: active month, valid essentials, complete 6 h history, SOM inputs and 1 h target present
    ess = np.column_stack([raw[k] for k in ESSENTIAL])
    ok_hour = ~np.isnan(ess).any(axis=1)
    hist = pd.Series(ok_hour.astype(int)).rolling(max(config.GRAD_WINDOWS) + 1).sum().to_numpy()
    complete = hist == max(config.GRAD_WINDOWS) + 1
    inputs = np.column_stack([f[c] for c in config.SOM_INPUTS])
    in_month = np.isin(month, list(act))
    keep_steps = dict(
        mes_ativo=in_month,
        hora_valida=ok_hour,
        janela_6h_completa=complete,
        entradas_completas=~np.isnan(inputs).any(axis=1),
        alvo_1h=~np.isnan(f["chuva_prox_1h"]),
    )
    keep = np.ones(len(df), dtype=bool)
    for name, m in keep_steps.items():
        keep &= m
        s[f"apos_{name}"] = int(keep.sum())
    rows = np.flatnonzero(keep)

    out = pd.DataFrame({"codigo_estacao": info["codigo"], "datetime": df["dt"].to_numpy()[rows]})
    out["hora_solar"] = solar_hour(df["dt"].iloc[rows], lon).astype("float32")
    for c, x in f.items():
        out[c] = x[rows].astype("float32")
    out["qualidade"] = quality[rows]
    out["motivo_qualidade"] = ["|".join(reasons[i]) for i in rows]
    s["horas_finais"] = len(out)
    s["data_inicio"] = str(out["datetime"].min().date()) if len(out) else ""
    s["data_fim"] = str(out["datetime"].max().date()) if len(out) else ""
    return out, s
