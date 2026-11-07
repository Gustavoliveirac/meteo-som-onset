"""
groups.py - who and when falls in each group of monitored units.

Groups come from analysis/text_numbers.py (adjacent monitored units, numbered by
their share of 20 mm events). For every group, over the three sets together:
hours and events, the stations, biomes, climate classes, distance to the coast,
altitude and latitude bands, seasons and solar hours of its hours and of its
events, compared with all hours, and the rain already falling at the hour.
Per unit: share of hours with rain in progress and next-hour rainfall with and
without rain in progress.

    python analysis/groups.py

Writes to config.RESULTS_DIR tabela_grupos.csv, grupos_transicoes.csv, grupos_perfil.csv, grupos_atributos.csv,
grupos_chuva_atual.csv, grupos_hora_solar.csv, grupos_estacoes.csv and
unidades_chuva_atual.csv. In grupos_atributos.csv, razao_horas is the share of the
group's hours in a class divided by the share of all hours in that class (above 1,
the class is over-represented in the group);
log in config.LOG_DIR/groups.log
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AN = config.RESULTS_DIR
ARR = config.WORK_DIR / "arrays"
UNITS_DIR = config.WORK_DIR / "som" / "20x20_seed42"
SETS = ["train", "val_temporal", "val_spatial"]
T = 20
RAIN_CLASSES = [(-1, 0, "sem chuva"), (0, 1, "ate 1 mm"), (1, 5, "1 a 5 mm"), (5, 1e9, "acima de 5 mm")]
log = logging.getLogger()

STATES = "--estados" in sys.argv


def F(name):
    """File name in group mode, or its state version with --estados (states from analysis/states.py)."""
    if not STATES:
        return name
    return {"nos_monitorados.csv": "nos_estados.csv", "grupos_monitorados.csv": "estados_monitorados.csv",
            "tabela_grupos.csv": "tabela_estados.csv", "estacoes_grupos.csv": "estacoes_estados.csv",
            "unidades_chuva_atual.csv": "unidades_chuva_atual_estados.csv"}.get(name, name.replace("grupos_", "estados_"))


def load(stat):
    parts = []
    for k in SETS:
        m = pd.read_parquet(ARR / f"meta_{k}.parquet",
                            columns=["codigo_estacao", "datetime", "hora_solar", "prec_acum_1h", "chuva_prox_1h"])
        m["unidade"] = np.load(UNITS_DIR / f"units_{k}.npy").astype(np.int16)
        m["conjunto"] = k
        parts.append(m)
    d = pd.concat(parts, ignore_index=True)
    d["codigo_estacao"] = d["codigo_estacao"].astype("category")
    d["evento"] = d["chuva_prox_1h"].fillna(0) >= T
    d["mes"] = d["datetime"].dt.month.astype(np.int8)
    st = stat.set_index("codigo")
    for c in ["bioma", "zona", "umidade", "costa", "altitude", "latitude_faixa"]:
        d[c] = d["codigo_estacao"].map(st[c]).astype("category")
    return d


def station_attributes():
    s = pd.read_csv(AN / F("estacoes_regional.csv"), sep=";", dtype={"codigo": str})
    out = pd.DataFrame({"codigo": s.codigo})
    out["bioma"] = s.get("bioma_biomas", pd.Series(dtype=str)).fillna("sem classe")
    out["zona"] = s.get("clima_ZONA", pd.Series(dtype=str)).fillna("sem classe")
    out["umidade"] = s.get("clima_TP_UMIDADE", pd.Series(dtype=str)).fillna("sem classe")
    out["costa"] = pd.cut(s.distancia_costa_km, [-1, 50, 150, 400, 5000],
                          labels=["ate 50 km", "50 a 150 km", "150 a 400 km", "acima de 400 km"]).astype(str)
    out["altitude"] = pd.cut(s.altitude, [-100, 200, 500, 900, 3000],
                             labels=["ate 200 m", "200 a 500 m", "500 a 900 m", "acima de 900 m"]).astype(str)
    out["latitude_faixa"] = pd.cut(s.latitude, [-40, -25, -15, -5, 10],
                                   labels=["sul de 25S", "15S a 25S", "5S a 15S", "norte de 5S"]).astype(str)
    return out


def main():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / ("groups_estados.log" if STATES else "groups.log"), mode="w"),
                                  logging.StreamHandler()])
    nos = pd.read_csv(AN / F("nos_monitorados.csv"), sep=";")
    stat = station_attributes()
    d = load(stat)
    gmap = np.zeros(int(d.unidade.max()) + 1, dtype=np.int8)
    gmap[nos.unidade.to_numpy()] = nos.grupo.to_numpy()
    d["grupo"] = gmap[d.unidade.to_numpy()]
    d["estacao_ano"] = d.mes.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                  6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    edges = [p[0] for p in config.SOLAR_PERIODS] + [24.01]
    d["periodo_solar"] = pd.cut(d.hora_solar, edges, right=False,
                                labels=[f"{p[2]} ({p[0]} a {p[1]} h)" for p in config.SOLAR_PERIODS])
    groups = sorted(g for g in d.grupo.unique() if g > 0)
    n_all, e_all = len(d), int(d.evento.sum())
    log.info(f"{n_all:,} hours, {e_all:,} events of {T} mm or more, groups {groups}")

    # profile
    rows = []
    for g in groups + [0]:
        x = d[d.grupo == g]
        ang = x.hora_solar.to_numpy() / 24 * 2 * np.pi
        ev = x[x.evento]
        ang_e = ev.hora_solar.to_numpy() / 24 * 2 * np.pi
        rows.append(dict(
            grupo=g if g else "fora dos grupos", nos=int((nos.grupo == g).sum()) if g else 400 - len(nos),
            horas=len(x), fracao_horas=len(x) / n_all, eventos=len(ev), fracao_eventos=len(ev) / e_all,
            taxa_eventos_por_mil_horas=1000 * len(ev) / max(len(x), 1),
            eventos_40=int((x.chuva_prox_1h >= 40).sum()),
            hora_solar_media_horas=(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 15,
            hora_solar_media_eventos=(np.degrees(np.arctan2(np.sin(ang_e).mean(), np.cos(ang_e).mean())) % 360) / 15 if len(ev) else np.nan,
            estacoes=x.codigo_estacao.nunique(),
            estacoes_metade_horas=int((x.codigo_estacao.value_counts(normalize=True).cumsum() < 0.5).sum() + 1),
            estacoes_metade_eventos=int((ev.codigo_estacao.value_counts(normalize=True).cumsum() < 0.5).sum() + 1) if len(ev) else 0))
    prof = pd.DataFrame(rows)
    prof.to_csv(AN / F("grupos_perfil.csv"), sep=";", index=False)

    # attributes: share of the group's hours and events in each class, against all hours
    rows = []
    for att in ["bioma", "zona", "umidade", "costa", "altitude", "latitude_faixa", "estacao_ano", "periodo_solar"]:
        base = d[att].value_counts(normalize=True)
        base_e = d.loc[d.evento, att].value_counts(normalize=True)
        for g in groups:
            x = d[d.grupo == g]
            h = x[att].value_counts(normalize=True)
            e = x.loc[x.evento, att].value_counts(normalize=True)
            for c in base.index:
                rows.append(dict(grupo=g, atributo=att, classe=c, fracao_horas_todas=base.get(c, 0),
                                 fracao_horas_grupo=h.get(c, 0), razao_horas=h.get(c, 0) / base.get(c, np.nan),
                                 fracao_eventos_todos=base_e.get(c, 0), fracao_eventos_grupo=e.get(c, 0)))
    att = pd.DataFrame(rows)
    att.to_csv(AN / F("grupos_atributos.csv"), sep=";", index=False)

    # rain already falling
    def rain_class(v):
        out = np.full(len(v), "", dtype=object)
        for lo, hi, name in RAIN_CLASSES:
            out[(v > lo) & (v <= hi)] = name
        return out
    d["chuva_atual"] = rain_class(d.prec_acum_1h.to_numpy())
    rows = []
    for g in groups + [0]:
        x = d[d.grupo == g]
        for _, _, name in RAIN_CLASSES:
            m = x.chuva_atual == name
            rows.append(dict(grupo=g if g else "fora dos grupos", chuva_atual=name, fracao_horas=m.mean(),
                             fracao_eventos=(m & x.evento).sum() / max(x.evento.sum(), 1),
                             taxa_eventos_por_mil_horas=1000 * (m & x.evento).sum() / max(m.sum(), 1)))
    pd.DataFrame(rows).to_csv(AN / F("grupos_chuva_atual.csv"), sep=";", index=False)

    # solar hour, hour by hour, of the hours and of the events of each group
    d["hora"] = np.floor(d.hora_solar).astype(int) % 24
    rows = []
    for g in groups + [0]:
        x = d[d.grupo == g]
        h = x.hora.value_counts(normalize=True)
        e = x.loc[x.evento, "hora"].value_counts(normalize=True)
        for k in range(24):
            rows.append(dict(grupo=g if g else "fora dos grupos", hora_solar=k,
                             fracao_horas=h.get(k, 0), fracao_eventos=e.get(k, 0)))
    hs = pd.DataFrame(rows)
    hs.to_csv(AN / F("grupos_hora_solar.csv"), sep=";", index=False)

    # transitions: where the station is 1, 2 and 3 hours later (same station, consecutive hours)
    st = d.codigo_estacao.cat.codes.to_numpy()
    tt = d.datetime.to_numpy()
    gg = d.grupo.to_numpy()
    rows = []
    for lag in (1, 2, 3):
        ok = np.zeros(len(d), bool)
        ok[:-lag] = (st[lag:] == st[:-lag]) & ((tt[lag:] - tt[:-lag]) == np.timedelta64(lag, "h"))
        nxt = np.full(len(d), -1); nxt[:-lag] = gg[lag:]
        for a_ in groups + [0]:
            m = ok & (gg == a_)
            cnt = np.bincount(nxt[m], minlength=max(groups) + 1)
            for b_ in groups + [0]:
                rows.append(dict(defasagem_h=lag, de=a_ if a_ else "fora", para=b_ if b_ else "fora",
                                 fracao=cnt[b_] / max(m.sum(), 1)))
    tr = pd.DataFrame(rows)
    tr.to_csv(AN / F("grupos_transicoes.csv"), sep=";", index=False)

    # summary table: peak hour of the events and the shortest run of hours holding half of them
    grp = pd.read_csv(AN / F("grupos_monitorados.csv"), sep=";").set_index("grupo")
    rows = []
    for g in groups:
        e = hs[hs.grupo == g].set_index("hora_solar").fracao_eventos.reindex(range(24), fill_value=0).to_numpy()
        best = (99, 0)
        for start in range(24):
            for length in range(1, 25):
                if e[[(start + k) % 24 for k in range(length)]].sum() >= 0.5:
                    best = min(best, (length, start))
                    break
        length, start = best
        x = prof.set_index("grupo").loc[g]
        ca = pd.read_csv(AN / F("grupos_chuva_atual.csv"), sep=";")
        ca = ca[ca.grupo.astype(str) == str(g)].set_index("chuva_atual")
        gm = grp.loc[g] if g in grp.index else None
        rows.append(dict(
            grupo=g, nos=int(x.nos), linhas=gm.linhas if gm is not None else "", colunas=gm.colunas if gm is not None else "",
            fracao_horas=x.fracao_horas, fracao_eventos_20=x.fracao_eventos, eventos_por_mil_horas=x.taxa_eventos_por_mil_horas,
            hora_pico_eventos=int(np.argmax(e)), metade_dos_eventos=f"{start} a {(start + length) % 24} h",
            horas_com_chuva_em_curso=1 - ca.loc["sem chuva", "fracao_horas"],
            eventos_partindo_de_hora_seca=ca.loc["sem chuva", "fracao_eventos"],
            temperatura=gm.temperatura if gm is not None else np.nan,
            umidade_relativa=gm.umidade_relativa if gm is not None else np.nan,
            variacao_1h_temperatura=gm.grad1h_temperatura if gm is not None else np.nan,
            variacao_1h_umidade=gm.grad1h_umidade_relativa if gm is not None else np.nan,
            variacao_1h_radiacao_wm2=gm.grad1h_radiacao_wm2 if gm is not None else np.nan,
            variacao_1h_vento=gm.grad1h_vento_velocidade if gm is not None else np.nan,
            variacao_1h_pressao=gm.grad1h_pressao if gm is not None else np.nan))
    pd.DataFrame(rows).round(3).to_csv(AN / F("tabela_grupos.csv"), sep=";", index=False)

    # stations contributing most events to each group
    meta = pd.read_csv(config.STATIONS_META, sep=";", dtype={"codigo": str}).set_index("codigo")
    rows = []
    for g in groups:
        ev = d[(d.grupo == g) & d.evento].codigo_estacao.value_counts()
        hrs = d[d.grupo == g].codigo_estacao.value_counts()
        for code, n in ev.head(10).items():
            rows.append(dict(grupo=g, codigo=code, nome=meta.nome.get(code, ""), eventos=int(n),
                             fracao_eventos_grupo=n / ev.sum(), horas_no_grupo=int(hrs.get(code, 0)),
                             latitude=meta.latitude.get(code), longitude=meta.longitude.get(code)))
    pd.DataFrame(rows).to_csv(AN / F("grupos_estacoes.csv"), sep=";", index=False)

    # per unit, rain in progress and next hour (training)
    t = d[d.conjunto == "train"]
    now = t.prec_acum_1h > 0
    nxt = t.chuva_prox_1h.fillna(0) > 1
    g_ = t.groupby("unidade")
    u = pd.DataFrame(dict(
        horas=g_.size(),
        fracao_chuva_atual=now.groupby(t.unidade).mean(),
        fracao_chuva_atual_acima_5mm=(t.prec_acum_1h > 5).groupby(t.unidade).mean(),
        prob_chuva_1h_sem_chuva_atual=nxt[~now].groupby(t.unidade[~now]).mean(),
        prob_chuva_1h_com_chuva_atual=nxt[now].groupby(t.unidade[now]).mean(),
        eventos=t.evento.groupby(t.unidade).sum(),
        fracao_eventos_com_chuva_atual=(t.evento & now)[t.evento].groupby(t.unidade[t.evento]).mean(),
    )).reset_index()
    side = int(round((int(d.unidade.max()) + 1) ** 0.5))
    u.insert(1, "linha", u.unidade // side + 1)
    u.insert(2, "coluna", u.unidade % side + 1)
    u["grupo"] = gmap[u.unidade.to_numpy()]
    u.to_csv(AN / F("unidades_chuva_atual.csv"), sep=";", index=False)

    pd.set_option("display.width", 220)
    log.info("\nprofile\n" + prof.round(3).to_string(index=False))
    for a in ["bioma", "costa", "latitude_faixa", "altitude", "estacao_ano", "periodo_solar"]:
        t_ = att[att.atributo == a].pivot_table(index="classe", columns="grupo", values="razao_horas").round(2)
        log.info(f"\n{a}: share of the group's hours divided by the share of all hours\n{t_}")
    t1 = tr[tr.defasagem_h == 1].pivot_table(index="de", columns="para", values="fracao").round(3)
    log.info(f"\ntransitions after 1 hour (row: group now, column: group one hour later)\n{t1}")
    for lag in (1, 2, 3):
        x = tr[(tr.defasagem_h == lag) & (tr.de.astype(str) == "2") & (tr.para.astype(str) == "1")].fracao
        if len(x):
            log.info(f"  from group 2 to group 1 after {lag} h: {x.iloc[0]:.3f}")
    log.info("\nrain in progress\n" + pd.read_csv(AN / F("grupos_chuva_atual.csv"), sep=";").round(3).to_string(index=False))
    log.info(f"\nunits: correlation between share of hours with rain in progress and next-hour probability without "
             f"rain in progress: {u.fracao_chuva_atual.corr(u.prob_chuva_1h_sem_chuva_atual):.2f}")
    log.info("\ndone")


if __name__ == "__main__":
    main()
