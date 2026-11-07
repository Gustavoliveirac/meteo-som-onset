"""
refined_groups.py - groups of monitored units used in the text.

Starting point: the groups of adjacent monitored units (results/nos_monitorados.csv,
from analysis/text_numbers.py) and the profile of each unit over the 38 inputs
(results/nos_estados.csv, from analysis/states.py). A group is kept when its units
share one profile; when adjacent units have different profiles the group is split
along them; a single isolated unit joins the group whose profile it shares. Groups
are numbered by their share of 20 mm events.

    python analysis/refined_groups.py

Rewrites results/nos_monitorados.csv (column grupo; the adjacency group stays in
grupo_vizinhanca) and results/grupos_monitorados.csv, and writes grupos_resumo.csv
(for each variable, the anomaly-type input that stands out most in the group) used
by analysis/group_profiles.py for the table in physical units.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AN = config.RESULTS_DIR
VARIABLES = {"pressao": "Pressão", "temperatura": "Temperatura", "ponto_orvalho": "Ponto de orvalho",
             "umidade_relativa": "Umidade relativa", "vento_velocidade": "Vento", "radiacao": "Radiação"}
WINDOW = {"anom": "", "grad1h_anom": " (1 h)", "grad3h_anom": " (3 h)", "grad6h_anom": " (6 h)"}
log = logging.getLogger()


def components(cells):
    cells = set(cells); parts = []
    while cells:
        stack = [cells.pop()]; part = []
        while stack:
            r, c = stack.pop(); part.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if n in cells:
                        cells.remove(n); stack.append(n)
        parts.append(part)
    return parts


def main():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "refined_groups.log", mode="w"), logging.StreamHandler()])
    nos = pd.read_csv(AN / "nos_monitorados.csv", sep=";")
    if "grupo_vizinhanca" in nos.columns:
        nos["grupo"] = nos["grupo_vizinhanca"]
    est = pd.read_csv(AN / "nos_estados.csv", sep=";")[["unidade", "grupo"]].rename(columns={"grupo": "perfil"})
    nos = nos.merge(est, on="unidade")
    key = {(r.linha, r.coluna): r for r in nos.itertuples()}

    # adjacent units with the same profile form a group
    parts = []
    for prof, g in nos.groupby("perfil"):
        parts += [(prof, p) for p in components(zip(g.linha, g.coluna))]
    singles = [x for x in parts if len(x[1]) == 1]
    groups = [x for x in parts if len(x[1]) > 1]
    for prof, p in singles:
        same = [i for i, (pr, _) in enumerate(groups) if pr == prof]
        if same:                                   # isolated unit joins the nearest group with its profile
            r0, c0 = p[0]
            i = min(same, key=lambda i: min(max(abs(r0 - r), abs(c0 - c)) for r, c in groups[i][1]))
            groups[i] = (groups[i][0], groups[i][1] + p)
        else:
            groups.append((prof, p))

    u = pd.read_csv(AN / "unidades.csv", sep=";")
    u = u[u.conjunto == "train"].set_index(["linha", "coluna"])
    share = [sum(u.loc[(r - 1, c - 1), "eventos_20"] for r, c in p) for _, p in groups]
    order = np.argsort(share)[::-1]
    lab = {}
    for new, i in enumerate(order, 1):
        for cell in groups[i][1]:
            lab[cell] = new
    nos["grupo_vizinhanca"] = nos["grupo"]
    nos["grupo"] = [lab[(r, c)] for r, c in zip(nos.linha, nos.coluna)]
    log.info("adjacency group and profile of each refined group\n" +
             nos.groupby("grupo").agg(nos=("unidade", "size"), vizinhanca=("grupo_vizinhanca", lambda x: sorted(set(x))),
                                      perfil=("perfil", lambda x: sorted(set(x)))).to_string())
    nos.drop(columns=["perfil"]).to_csv(AN / "nos_monitorados.csv", sep=";", index=False)

    # mean state of each group (physical units) and the anomaly-type input that stands out per variable
    st = pd.read_csv(AN / "unidades_estatisticas.csv", sep=";")
    st = st[st.conjunto == "train"].set_index("unidade")
    inputs = list(config.SOM_INPUTS)
    w = st.horas / st.horas.sum()
    Z = st[inputs].copy()
    for c in inputs:
        m = (st[c] * w).sum()
        Z[c] = (st[c] - m) / np.sqrt(((st[c] - m) ** 2 * w).sum())
    ev = pd.read_csv(AN / "unidades.csv", sep=";")
    ev = ev[ev.conjunto == "train"].set_index("unidade")
    rows, res = [], []
    for g, gg in nos.groupby("grupo"):
        us = gg.unidade.to_numpy(); hw = st.loc[us, "horas"]
        mean = lambda c: (st.loc[us, c] * hw).sum() / hw.sum()
        rows.append(dict(grupo=g, linhas=f"{gg.linha.min()}-{gg.linha.max()}", colunas=f"{gg.coluna.min()}-{gg.coluna.max()}",
                         temperatura=mean("temperatura"), umidade_relativa=mean("umidade_relativa"),
                         grad1h_temperatura=mean("grad1h_temperatura"), grad1h_umidade_relativa=mean("grad1h_umidade_relativa"),
                         grad1h_radiacao_wm2=mean("grad1h_radiacao") / 3.6, grad1h_vento_velocidade=mean("grad1h_vento_velocidade"),
                         grad1h_pressao=mean("grad1h_pressao")))
        zp = Z.loc[us, inputs].mul(hw, axis=0).sum() / hw.sum()
        raw = st.loc[us, inputs].mul(hw, axis=0).sum() / hw.sum()
        r = dict(estado=g, nos=len(us), fracao_horas=hw.sum() / st.horas.sum(),
                 fracao_eventos_20=ev.loc[us, "eventos_20"].sum() / ev.eventos_20.sum(),
                 fracao_eventos_40=ev.loc[us, "eventos_40"].sum() / ev.eventos_40.sum())
        for k, name in VARIABLES.items():
            cols = [c for c in inputs if c.endswith("_" + k) and "anom" in c]
            v = zp[cols].abs().idxmax()
            r[name] = "–" if abs(zp[v]) < 0.8 else f"{raw[v]:+.1f}{WINDOW[v[:-len(k) - 1]]}"
            r[name + " (variável)"] = v
        res.append(r)
    pd.DataFrame(rows).round(3).to_csv(AN / "grupos_monitorados.csv", sep=";", index=False)
    pd.DataFrame(res).round(3).to_csv(AN / "grupos_resumo.csv", sep=";", index=False)
    log.info("\n" + pd.DataFrame(res)[["estado", "nos", "fracao_horas", "fracao_eventos_20", "fracao_eventos_40"]].round(3).to_string(index=False))
    log.info("\ndone")


if __name__ == "__main__":
    main()
