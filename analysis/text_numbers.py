"""
text_numbers.py - every number quoted in the text about the SOM state space.

    python analysis/text_numbers.py

Reads the tables of analysis/coverage.py and analysis/unit_stats.py and the
arrays metadata. Writes to config.RESULTS_DIR
  numeros_texto.csv          secao;item;valor
  nos_monitorados.csv        monitored units with rainfall, event counts and mean state in physical units
  grupos_monitorados.csv     the same aggregated by group of adjacent monitored units
  correlacao_horizontes.csv  Pearson r across units between each variable and
                             the mean rainfall of each horizon (training)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AN = config.RESULTS_DIR
ARR = config.WORK_DIR / "arrays"
UNITS_DIR = config.WORK_DIR / "som" / "20x20_seed42"
MIN_HOURS = 10
rows = []


def put(sec, item, val):
    rows.append(dict(secao=sec, item=item, valor=round(float(val), 4) if np.isscalar(val) and not isinstance(val, str) else val))


def r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    u = pd.read_csv(AN / "unidades_estatisticas.csv", sep=";")
    un = pd.read_csv(AN / "unidades.csv", sep=";")
    meta = pd.read_csv(config.STATIONS_META, sep=";", dtype={"codigo": str})

    # data
    put("dados", "estacoes_com_horas", int((meta.horas_finais.fillna(0) > 0).sum()))
    for k in ["train", "val_temporal", "val_spatial"]:
        put("dados", f"horas_{k}", int(un[un.conjunto == k].horas.sum()))
    put("dados", "horas_total", int(un.horas.sum()))
    put("dados", "horas_por_no_treino_media", un[un.conjunto == "train"].horas.mean())

    tr = u[u.conjunto == "train"].set_index("unidade")
    ok = tr.horas >= MIN_HOURS
    t = tr[ok]

    # reproducibility
    for v in ["pressao", "temperatura", "ponto_orvalho", "umidade_relativa", "radiacao", "vento_velocidade",
              "anom_pressao", "anom_temperatura", "anom_ponto_orvalho", "anom_umidade_relativa",
              "anom_radiacao", "anom_vento_velocidade"]:
        for k in ["val_temporal", "val_spatial"]:
            va = u[u.conjunto == k].set_index("unidade")
            m = ok & (va.horas >= MIN_HOURS)
            put("reprodutibilidade", f"r_{v}_treino_x_{k}", r(tr.loc[m, v].to_numpy(), va.loc[m, v].to_numpy()))

    # relations between unit means (training)
    pairs = [("temperatura", "radiacao"), ("umidade_relativa", "temperatura"), ("umidade_relativa", "radiacao"),
             ("vento_velocidade", "radiacao"), ("vento_velocidade", "temperatura"),
             ("vento_velocidade", "umidade_relativa"), ("vento_direcao", "anom_pressao"),
             ("vento_direcao", "anom_temperatura"), ("ponto_orvalho", "radiacao"),
             ("anom_ponto_orvalho", "anom_umidade_relativa"), ("pressao", "radiacao"),
             ("pressao", "temperatura"), ("pressao", "umidade_relativa"), ("hora_solar", "radiacao")]
    for a, b in pairs:
        put("relacoes_entre_nos", f"r_{a}_x_{b}", r(t[a].to_numpy(), t[b].to_numpy()))
    w = t.horas.to_numpy()
    ang = np.deg2rad(t.vento_direcao.to_numpy())
    put("relacoes_entre_nos", "direcao_vento_media_ponderada_graus",
        np.degrees(np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())) % 360)
    put("relacoes_entre_nos", "fracao_nos_vento_de_sul_135_225",
        float(((t.vento_direcao >= 135) & (t.vento_direcao <= 225)).mean()))

    # where radiation and the day cycle sit
    side = int(t.linha.max()) + 1
    for name, col in [("radiacao_max", "radiacao"), ("temperatura_max", "temperatura"),
                      ("umidade_max", "umidade_relativa")]:
        q = t[col] >= t[col].quantile(0.9)
        put("posicoes", f"{name}_linha_media", t.loc[q, "linha"].mean() + 1)
        put("posicoes", f"{name}_coluna_media", t.loc[q, "coluna"].mean() + 1)

    # rain by unit
    for h in [1, 6]:
        p = t[f"fracao_acima_1mm_{h}h"]
        put("chuva_por_no", f"nos_acima_10pct_{h}h", int((p > 0.10).sum()))
        put("chuva_por_no", f"nos_acima_20pct_{h}h", int((p > 0.20).sum()))
    top = t.sort_values("fracao_acima_1mm_1h", ascending=False).head(15)
    for i, (idx, x) in enumerate(top.iterrows(), 1):
        put("chuva_por_no", f"top{i:02d}_linha{int(x.linha) + 1}_coluna{int(x.coluna) + 1}",
            f"1h {x.fracao_acima_1mm_1h:.3f}  6h {x.fracao_acima_1mm_6h:.3f}  "
            f"anomT {x.anom_temperatura:+.2f}  anomUR {x.anom_umidade_relativa:+.2f}  "
            f"anomRad {x.anom_radiacao:+.2f}  g1anomP {x.grad1h_anom_pressao:+.2f}  "
            f"g1anomT {x.grad1h_anom_temperatura:+.2f}  g1T {x.grad1h_temperatura:+.2f}")

    # correlation by horizon
    vars_ = [c for c in t.columns if c in config.SOM_INPUTS or c in
             ["pressao", "temperatura", "ponto_orvalho", "umidade_relativa", "radiacao", "vento_velocidade", "theta_e"]]
    cor = pd.DataFrame({h: {v: r(t[v].to_numpy(), t[f"chuva_media_{h}h"].to_numpy()) for v in vars_}
                        for h in config.HORIZONS})
    cor.columns = [f"{h}h" for h in config.HORIZONS]
    cor.index.name = "variavel"
    cor.round(3).to_csv(AN / "correlacao_horizontes.csv", sep=";")
    for h in ["1h", "2h", "6h", "12h", "24h"]:
        s = cor[h].abs().sort_values(ascending=False).head(4)
        for i, v in enumerate(s.index, 1):
            put("hierarquia", f"{h}_posicao{i}_{v}", cor.loc[v, h])

    # concentration of events
    d = un[un.conjunto == "train"]
    for T in [1, 5, 10, 20, 30, 40]:
        e = np.sort(d[f"eventos_{T}"].to_numpy())[::-1]
        c = np.cumsum(e) / e.sum()
        put("concentracao", f"nos_para_50pct_{T}mm", int((c < 0.5).sum() + 1))
        put("concentracao", f"nos_para_80pct_{T}mm", int((c < 0.8).sum() + 1))
    d40 = d.assign(share=d.eventos_40 / d.eventos_40.sum()).sort_values("share", ascending=False)
    for i, (_, x) in enumerate(d40.head(8).iterrows(), 1):
        put("concentracao", f"40mm_top{i}_linha{int(x.linha) + 1}_coluna{int(x.coluna) + 1}", x.share)

    # monitored units and their groups on the grid, in physical units
    from scipy.ndimage import label
    sel_df = pd.read_csv(AN / "conjunto_som.csv", sep=";")
    side = int(un.linha.max()) + 1
    mask = np.zeros((side, side), dtype=bool)
    mask[sel_df.linha, sel_df.coluna] = True
    lab, n_groups = label(mask, structure=np.ones((3, 3)))
    tt = tr.copy()
    tt["radiacao_wm2"] = tt["radiacao"] / 3.6
    tt["grad1h_radiacao_wm2"] = tt["grad1h_radiacao"] / 3.6
    ev = d.set_index("unidade")
    phys = ["temperatura", "umidade_relativa", "radiacao_wm2", "vento_velocidade", "grad1h_temperatura",
            "grad3h_temperatura", "grad1h_pressao", "grad3h_pressao", "grad1h_umidade_relativa",
            "grad1h_radiacao_wm2", "grad1h_vento_velocidade"]
    rows_u, rows_g = [], []
    for u_ in sel_df.unidade:
        r_, c_ = divmod(int(u_), side)
        x = tt.loc[u_]
        rows_u.append(dict(grupo=int(lab[r_, c_]), unidade=int(u_), linha=r_ + 1, coluna=c_ + 1, horas=int(x.horas),
                           p1h=x.fracao_acima_1mm_1h, p6h=x.fracao_acima_1mm_6h, hora_solar=x.hora_solar,
                           **{f"eventos_{T}": int(ev.loc[u_, f"eventos_{T}"]) for T in [20, 30, 40]},
                           **{c: x[c] for c in phys}))
    nu = pd.DataFrame(rows_u)
    tot = {T: d[f"eventos_{T}"].sum() for T in [20, 30, 40]}
    for g, gg in nu.groupby("grupo"):
        w = gg.horas.to_numpy()
        ang = gg.hora_solar.to_numpy() / 24 * 2 * np.pi
        rec = dict(grupo=g, nos=len(gg), linhas=f"{gg.linha.min()}-{gg.linha.max()}",
                 colunas=f"{gg.coluna.min()}-{gg.coluna.max()}", fracao_horas_treino=w.sum() / un[un.conjunto == "train"].horas.sum(),
                 p1h_min=gg.p1h.min(), p1h_max=gg.p1h.max(), p6h_min=gg.p6h.min(), p6h_max=gg.p6h.max(),
                 hora_solar_media=(np.degrees(np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())) % 360) / 15)
        for T in [20, 30, 40]:
            rec[f"fracao_eventos_{T}"] = gg[f"eventos_{T}"].sum() / tot[T]
        for c in phys:
            rec[c] = (gg[c] * w).sum() / w.sum()
        rows_g.append(rec)
    gr = pd.DataFrame(rows_g).sort_values("fracao_eventos_20", ascending=False)
    new = {old: i for i, old in enumerate(gr.grupo, 1)}      # group 1 holds most 20 mm events
    gr["grupo"] = gr.grupo.map(new)
    nu["grupo"] = nu.grupo.map(new)
    nu.round(3).to_csv(AN / "nos_monitorados.csv", sep=";", index=False)
    gr.round(3).to_csv(AN / "grupos_monitorados.csv", sep=";", index=False)
    put("grupos", "numero_de_grupos", n_groups)

    # rain in progress or SOM, and dry share of the SOM captures
    sel = pd.read_csv(AN / "conjunto_som.csv", sep=";")["unidade"].to_numpy()
    for k in ["val_temporal", "val_spatial"]:
        m = pd.read_parquet(ARR / f"meta_{k}.parquet", columns=["prec_acum_1h", "chuva_prox_1h"])
        unit = np.load(UNITS_DIR / f"units_{k}.npy")
        som = np.isin(unit, sel)
        rain = m.prec_acum_1h.to_numpy() > 0
        y = np.nan_to_num(m.chuva_prox_1h.to_numpy())
        put("uniao", f"{k}_fracao_horas_chuva_ou_som", float((som | rain).mean()))
        for T in [20, 30, 40]:
            ev = y >= T
            put("uniao", f"{k}_{T}mm_cobertura_chuva_ou_som", float((som | rain)[ev].mean()))
            cap = som & ev
            put("uniao", f"{k}_{T}mm_capturas_som_sem_chuva", float((~rain[cap]).mean()))
            put("uniao", f"{k}_{T}mm_capturas_som_sem_chuva_sobre_todos", float((cap & ~rain).sum() / ev.sum()))

    out = pd.DataFrame(rows)
    out.to_csv(AN / "numeros_texto.csv", sep=";", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
