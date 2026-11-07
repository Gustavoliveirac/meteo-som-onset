"""
states.py - atmospheric states of the monitored units, from all 38 SOM inputs.

The monitored units (results/nos_monitorados.csv) are described by the mean of the
38 inputs over their training hours (results/unidades_estatisticas.csv), each input
standardized across the 400 units. They are grouped by K-means, and the grouping is
number of states is the one with the highest mean silhouette among 2 to 9 states
(config.STATES_K fixes it if set). States are numbered by their share of 20 mm events.

    python analysis/states.py

Writes to config.RESULTS_DIR
  estados_selecao.csv     k, share of variance explained, its gain and mean silhouette
  nos_estados.csv         monitored units with their spatial group and their state
                          (column "grupo" holds the state, so analysis/groups.py and
                          analysis/group_profiles.py run on it with --estados)
  estados_resumo.csv      per state: units, share of hours and events and, for each
                          forcing, the anomaly-type input that stands out most, with its
                          mean in standard deviations of the station climatology
  estados_monitorados.csv per state: mean state and 1-h changes in physical units
  estados_variaveis.csv   share of the variance of each input explained by the states
Log in config.LOG_DIR/states.log
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AN = config.RESULTS_DIR
K = getattr(config, "STATES_K", 5)
FORCINGS = {"pressao": "Pressão", "temperatura": "Temperatura", "ponto_orvalho": "Ponto de orvalho",
            "umidade_relativa": "Umidade relativa", "vento_velocidade": "Vento", "radiacao": "Radiação"}
WINDOW = {"anom": "", "grad1h_anom": " (1 h)", "grad3h_anom": " (3 h)", "grad6h_anom": " (6 h)"}
log = logging.getLogger()


def kmeans(X, k, seed, n_init=50, iters=300):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(n_init):
        C = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else X[rng.integers(len(X))] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def ari(a, b):
    a = pd.factorize(a)[0]; b = pd.factorize(b)[0]
    ct = pd.crosstab(a, b).to_numpy()
    comb = lambda x: x * (x - 1) / 2
    s_ij = comb(ct).sum(); s_a = comb(ct.sum(1)).sum(); s_b = comb(ct.sum(0)).sum(); n = comb(ct.sum())
    exp = s_a * s_b / n
    return (s_ij - exp) / (0.5 * (s_a + s_b) - exp)


def main():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "states.log", mode="w"), logging.StreamHandler()])
    u = pd.read_csv(AN / "unidades_estatisticas.csv", sep=";")
    u = u[u.conjunto == "train"].set_index("unidade")
    nos = pd.read_csv(AN / "nos_monitorados.csv", sep=";")
    ev = pd.read_csv(AN / "unidades.csv", sep=";")
    ev = ev[ev.conjunto == "train"].set_index("unidade")
    inputs = list(config.SOM_INPUTS)
    w = u.horas / u.horas.sum()
    Z = u[inputs].copy()
    for c in inputs:
        m = (u[c] * w).sum()
        Z[c] = (u[c] - m) / np.sqrt(((u[c] - m) ** 2 * w).sum())
    units = nos.unidade.to_numpy()
    X = Z.loc[units, inputs].to_numpy()

    # number of states: K-means for 2 to 9 states; the share of variance explained and the mean
    # silhouette (how much closer each unit is to its own state than to the nearest other state)
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    tot = ((X - X.mean(0)) ** 2).sum()
    rows = []
    for k in range(2, 10):
        lab_k = kmeans(X, k, 0)
        within = sum(((X[lab_k == j] - X[lab_k == j].mean(0)) ** 2).sum() for j in range(k))
        sil = []
        for i in range(len(X)):
            same = lab_k == lab_k[i]
            if same.sum() == 1:
                sil.append(0.0); continue
            a_ = D[i, same].sum() / (same.sum() - 1)
            b_ = min(D[i, lab_k == j].mean() for j in range(k) if j != lab_k[i] and (lab_k == j).any())
            sil.append((b_ - a_) / max(a_, b_))
        rows.append(dict(k=k, variancia_explicada=1 - within / tot, silhueta=float(np.mean(sil))))
    sel = pd.DataFrame(rows)
    sel["ganho"] = sel.variancia_explicada.diff()
    sel.round(3).to_csv(AN / "estados_selecao.csv", sep=";", index=False)
    log.info("number of states\n" + sel.round(3).to_string(index=False))
    k_best = int(sel.loc[sel.silhueta.idxmax(), "k"])
    global K
    if getattr(config, "STATES_K", None) is None:
        K = k_best
    log.info(f"highest silhouette at {k_best} states; using {K}")

    lab = kmeans(X, K, 0)
    share = pd.Series({j: ev.loc[units[lab == j], "eventos_20"].sum() for j in range(K)}).sort_values(ascending=False)
    remap = {old: i for i, old in enumerate(share.index, 1)}
    st = pd.Series([remap[j] for j in lab], index=units)
    out = nos[["unidade", "linha", "coluna"]].copy()
    out["grupo_espacial"] = nos.grupo
    out["grupo"] = st.loc[out.unidade].to_numpy()          # the state, under the column name the other scripts read
    out.to_csv(AN / "nos_estados.csv", sep=";", index=False)
    log.info(f"\n{K} states")
    log.info("state of each spatial group:\n" + pd.crosstab(out.grupo_espacial, out.grupo).to_string())

    tot20, tot40, toth = ev.eventos_20.sum(), ev.eventos_40.sum(), u.horas.sum()
    res, phys = [], []
    for s in range(1, K + 1):
        us = out.unidade[out.grupo == s].to_numpy()
        hw = u.loc[us, "horas"]
        zprof = (Z.loc[us, inputs].mul(hw, axis=0).sum() / hw.sum())
        raw = (u.loc[us, inputs].mul(hw, axis=0).sum() / hw.sum())
        r = dict(estado=s, nos=len(us), fracao_horas=hw.sum() / toth,
                 fracao_eventos_20=ev.loc[us, "eventos_20"].sum() / tot20,
                 fracao_eventos_40=ev.loc[us, "eventos_40"].sum() / tot40)
        for key, name in FORCINGS.items():
            cols = [c for c in inputs if c.endswith("_" + key) and "anom" in c]
            v = zprof[cols].abs().idxmax()
            r[name] = "–" if abs(zprof[v]) < 0.8 else f"{raw[v]:+.1f}{WINDOW[v[:-len(key) - 1]]}"
            r[name + " (variável)"] = v
        res.append(r)
        p = dict(estado=s, linhas=f"{(u.loc[us, 'linha'] + 1).min()}-{(u.loc[us, 'linha'] + 1).max()}",
                 colunas=f"{(u.loc[us, 'coluna'] + 1).min()}-{(u.loc[us, 'coluna'] + 1).max()}")
        for c in ["temperatura", "umidade_relativa", "ponto_orvalho", "grad1h_temperatura", "grad1h_umidade_relativa",
                  "grad1h_ponto_orvalho", "grad1h_radiacao", "grad1h_vento_velocidade", "grad1h_pressao", "grad3h_pressao"]:
            p[c] = (u.loc[us, c] * hw).sum() / hw.sum()
        p["grad1h_radiacao_wm2"] = p.pop("grad1h_radiacao") / 3.6
        phys.append(p)
    resumo = pd.DataFrame(res)
    resumo.round(3).to_csv(AN / "estados_resumo.csv", sep=";", index=False)
    pm = pd.DataFrame(phys).rename(columns={"estado": "grupo"})
    pm.round(3).to_csv(AN / "estados_monitorados.csv", sep=";", index=False)

    labs_s = out.set_index("unidade").grupo.loc[units].to_numpy()
    Xd = pd.DataFrame(X, columns=inputs)
    tot = ((Xd - Xd.mean()) ** 2).sum()
    within = sum(((g - g.mean()) ** 2).sum() for _, g in Xd.groupby(labs_s))
    var = (1 - within / tot).sort_values(ascending=False)
    var.rename("fracao_variancia").round(3).to_csv(AN / "estados_variaveis.csv", sep=";")

    pd.set_option("display.width", 250)
    show = ["estado", "nos", "fracao_horas", "fracao_eventos_20", "fracao_eventos_40"] + list(FORCINGS.values())
    log.info("\nstates (anomaly-type inputs, standard deviations; window in brackets, none for the anomaly itself)\n"
             + resumo[show].round(3).to_string(index=False))
    log.info("\nmean state and 1-h changes in physical units\n" + pm.round(2).to_string(index=False))
    log.info("\ninputs that best separate the states\n" + var.head(12).round(2).to_string())
    log.info("\ndone")


if __name__ == "__main__":
    main()
