"""
coverage.py - events of the next hour captured by the SOM units and by the references.

Operating point
  budget         share of training hours with rain in progress (prec_acum_1h > 0)
  rain in progress  station on alert when it is raining
  precursor      the SOM input, and tail, with the largest coverage of 20 mm events
                 in training when it flags the budget share of training hours;
                 a second one is chosen the same way on hours without rain
  SOM            units ordered by the density of 20 mm events in training; the
                 leading set whose share of training hours is closest to the budget

Coverage is the share of events (chuva_prox_1h >= T) in flagged hours; amplification
is the event rate in flagged hours divided by the event rate of all hours. Both are
also computed on hours without rain in progress. Confidence intervals come from 1000
resamples of calendar days (Poisson weights), with the SOM set chosen again on the
resampled training days and independent resamples of the validation days.

    python analysis/coverage.py

Reads config.WORK_DIR/arrays and config.WORK_DIR/som. Writes the tables to
config.RESULTS_DIR, the log to config.LOG_DIR/coverage.log and a copy of
both in config.LOG_DIR/resultado_analise.zip.
"""
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

ARR = config.WORK_DIR / "arrays"
SOM_ROOT = config.WORK_DIR / "som"
MAIN = SOM_ROOT / "20x20_seed42"
OUT = config.RESULTS_DIR
SETS = ["train", "val_temporal", "val_spatial"]
VAL = ["val_temporal", "val_spatial"]
T_MAIN = [20, 30, 40]
T_ALL = [1, 5, 10, 20, 30, 40]
T_REF = 20
DRY_H = [3, 6, 12, 24]
RAIN_BANDS = [(0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)]
FRACTIONS = np.arange(1, 101) / 100
N_BOOT = 1000
SEED = 42
CHUNK = 2_000_000
log = logging.getLogger()


# ------------------------------------------------------------------ loading
def load_set(k, units_dir=MAIN):
    cols = ["codigo_estacao", "datetime", "prec_acum_1h", "chuva_prox_1h"]
    m = pd.read_parquet(ARR / f"meta_{k}.parquet", columns=cols)
    s = dict(
        unit=np.load(units_dir / f"units_{k}.npy").astype(np.int64),
        rain_now=m["prec_acum_1h"].to_numpy("float32"),
        y=np.nan_to_num(m["chuva_prox_1h"].to_numpy("float32"), nan=0.0),
        station=m["codigo_estacao"].astype("category"),
    )
    s["dry"] = s["rain_now"] == 0
    s["day"], days = pd.factorize(m["datetime"].dt.floor("D"), sort=True)
    s["n_days"] = len(days)
    s["n"] = len(m)
    return s


def column_pass(k, cols, unit=None, n_units=0):
    """One sequential read of X_{k}: values of the requested columns and, when
    unit is given, the per-unit sums of every input."""
    X = np.load(ARR / f"X_{k}.npy", mmap_mode="r")
    vals = {j: np.empty(len(X), dtype="float32") for j in cols}
    sums = np.zeros((n_units, X.shape[1])) if unit is not None else None
    for s in range(0, len(X), CHUNK):
        x = np.asarray(X[s:s + CHUNK])
        for j in cols:
            vals[j][s:s + CHUNK] = x[:, j]
        if sums is not None:
            u = unit[s:s + CHUNK]
            for c in range(x.shape[1]):
                sums[:, c] += np.bincount(u, weights=x[:, c], minlength=n_units)
    return vals, sums


# ------------------------------------------------------------------ selection
def select_units(cnt, ev, budget):
    dens = np.where(cnt > 0, ev / np.maximum(cnt, 1), -1.0)
    order = np.argsort(-dens, kind="stable")
    cum = np.cumsum(cnt[order]) / cnt.sum()
    n_sel = int(np.argmin(np.abs(cum - budget))) + 1      # leading set whose share of hours is closest to the budget
    sel = np.zeros(len(cnt), dtype=bool)
    sel[order[:n_sel]] = True
    return sel, order


def choose_precursor(tr, budget):
    X = np.load(ARR / "X_train.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(tr["n"], min(3_000_000, tr["n"]), replace=False))
    Xs = np.asarray(X[samp])
    ev = np.flatnonzero(tr["y"] >= T_REF)
    Xe = np.asarray(X[ev])
    dry_s, dry_e = tr["dry"][samp], tr["dry"][ev]
    rows = []
    for j, name in enumerate(config.SOM_INPUTS):
        for sign in (1, -1):
            q = 1 - budget if sign > 0 else budget
            cut = float(np.quantile(Xs[:, j], q))
            cut_d = float(np.quantile(Xs[dry_s, j], q))
            hit = Xe[:, j] >= cut if sign > 0 else Xe[:, j] <= cut
            hit_d = Xe[dry_e, j] >= cut_d if sign > 0 else Xe[dry_e, j] <= cut_d
            rows.append(dict(entrada=name, coluna=j, sinal=sign, corte=cut, cobertura_treino=hit.mean(),
                             corte_seco=cut_d, cobertura_treino_seco=hit_d.mean()))
    r = pd.DataFrame(rows)
    return r, r.loc[r["cobertura_treino"].idxmax()], r.loc[r["cobertura_treino_seco"].idxmax()]


def flag(v, sign, cut):
    return v >= cut if sign > 0 else v <= cut


# ------------------------------------------------------------------ metrics
def point(fl, s, T, cond):
    m = s["dry"] if cond == "dry" else np.ones(s["n"], dtype=bool)
    ev = (s["y"] >= T) & m
    n_ev, n_h = int(ev.sum()), int(m.sum())
    f_h = int((fl & m).sum())
    cap = int((fl & ev).sum())
    cov = cap / max(n_ev, 1)
    amp = (cap / max(f_h, 1)) / (n_ev / max(n_h, 1)) if n_ev else np.nan
    return dict(eventos=n_ev, horas=n_h, fracao_horas=f_h / max(n_h, 1), capturados=cap,
                cobertura=cov, amplificacao=amp)


def coverage_curve(cnt_k, ev_k, order):
    c = np.concatenate([[0.0], np.cumsum(cnt_k[order])])
    e = np.concatenate([[0.0], np.cumsum(ev_k[order])])
    N, E = c[-1], e[-1]
    cov = np.interp(FRACTIONS * N, c, e) / max(E, 1e-9)
    area = np.trapezoid(np.concatenate([[0.0], cov]), np.concatenate([[0.0], FRACTIONS]))
    perfect = 1.0 - (E / N) / 2.0
    return cov, (area - 0.5) / (perfect - 0.5), area


def day_matrix(s, mask, n_units):
    flat = s["day"][mask].astype(np.int64) * n_units + s["unit"][mask]
    return np.bincount(flat, minlength=s["n_days"] * n_units).reshape(s["n_days"], n_units).astype("float32")


# ------------------------------------------------------------------ main
def main():
    global N_BOOT
    if "--boot" in sys.argv:
        N_BOOT = int(sys.argv[sys.argv.index("--boot") + 1])
    OUT.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "coverage.log", mode="w"),
                                  logging.StreamHandler()])
    S = {k: load_set(k) for k in SETS}
    n_units = int(max(s["unit"].max() for s in S.values())) + 1
    tr = S["train"]
    side = int(round(n_units ** 0.5))
    budget = float((tr["rain_now"] > 0).mean())
    log.info(f"units {n_units}, budget (training hours with rain in progress) {budget:.4f}")
    for k in SETS:
        log.info(f"  {k:13s} hours {S[k]['n']:>11,}  days {S[k]['n_days']:>6,}  "
                 f"rain in progress {(S[k]['rain_now'] > 0).mean():.4f}")

    # SOM set
    cnt_tr = np.bincount(tr["unit"], minlength=n_units).astype(float)
    ev_tr = {T: np.bincount(tr["unit"][tr["y"] >= T], minlength=n_units).astype(float) for T in T_ALL}
    sel, order20 = select_units(cnt_tr, ev_tr[T_REF], budget)
    log.info(f"SOM set: {sel.sum()} units, {cnt_tr[sel].sum() / cnt_tr.sum():.4f} of training hours")
    cum = np.cumsum(cnt_tr[order20]) / cnt_tr.sum()
    for n in range(max(int(sel.sum()) - 2, 1), int(sel.sum()) + 3):
        log.info(f"  first {n} units: {cum[n - 1]:.4f} of training hours")
    pd.DataFrame(dict(unidade=np.flatnonzero(sel), linha=np.flatnonzero(sel) // side,
                      coluna=np.flatnonzero(sel) % side, horas_treino=cnt_tr[sel].astype(int),
                      eventos20_treino=ev_tr[T_REF][sel].astype(int))
                 ).to_csv(OUT / "conjunto_som.csv", sep=";", index=False)

    # precursors
    rank, best, best_d = choose_precursor(tr, budget)
    rank.sort_values("cobertura_treino", ascending=False).to_csv(OUT / "precursores.csv", sep=";", index=False)
    log.info(f"precursor (all hours) {best.entrada} sign {best.sinal:+d} cut {best.corte:+.3f} "
             f"training coverage {best.cobertura_treino:.3f}")
    log.info(f"precursor (dry hours) {best_d.entrada} sign {best_d.sinal:+d} cut {best_d.corte_seco:+.3f} "
             f"training coverage {best_d.cobertura_treino_seco:.3f}")
    jb, jd = int(best.coluna), int(best_d.coluna)

    # flags and unit means of the inputs
    unit_sums = None
    for k in SETS:
        s = S[k]
        vals, sums = column_pass(k, sorted({jb, jd}), s["unit"] if k == "train" else None, n_units)
        if sums is not None:
            unit_sums = sums
        s["f_chuva"] = s["rain_now"] > 0
        s["f_prec"] = flag(vals[jb], best.sinal, best.corte)
        s["f_prec_seco"] = flag(vals[jd], best_d.sinal, best_d.corte_seco)
        s["f_som"] = sel[s["unit"]]
    crit = {"chuva_em_curso": "f_chuva", "precursor": "f_prec", "precursor_seco": "f_prec_seco", "som": "f_som"}

    # points
    rows = []
    for k in SETS:
        for cond in ("all", "dry"):
            for T in T_ALL:
                for c, f in crit.items():
                    if c == "precursor_seco" and cond == "all":
                        continue
                    rows.append(dict(conjunto=k, condicao=cond, limiar=T, criterio=c, **point(S[k][f], S[k], T, cond)))
    pts = pd.DataFrame(rows)

    # bootstrap
    log.info(f"bootstrap, {N_BOOT} resamples of days")
    cnt_d_tr = day_matrix(tr, np.ones(tr["n"], bool), n_units)
    ev_d_tr = day_matrix(tr, tr["y"] >= T_REF, n_units)
    agg = {}
    for k in VAL:
        s = S[k]
        a = {}
        for cond in ("all", "dry"):
            m = s["dry"] if cond == "dry" else np.ones(s["n"], bool)
            a[cond, "h"] = np.bincount(s["day"][m], minlength=s["n_days"]).astype(float)
            a[cond, "cnt_u"] = day_matrix(s, m, n_units)
            for c, f in crit.items():
                a[cond, "fh", c] = np.bincount(s["day"][m & s[f]], minlength=s["n_days"]).astype(float)
            for T in T_MAIN:
                e = m & (s["y"] >= T)
                a[cond, "e", T] = np.bincount(s["day"][e], minlength=s["n_days"]).astype(float)
                a[cond, "ev_u", T] = day_matrix(s, e, n_units)
                for c, f in crit.items():
                    a[cond, "ef", T, c] = np.bincount(s["day"][e & s[f]], minlength=s["n_days"]).astype(float)
        agg[k] = a
    rng = np.random.default_rng(SEED)
    boot = []
    for b in range(N_BOOT):
        w = rng.poisson(1.0, tr["n_days"]).astype("float32")
        sel_b, _ = select_units(w @ cnt_d_tr, w @ ev_d_tr, budget)
        for k in VAL:
            a = agg[k]
            wv = rng.poisson(1.0, S[k]["n_days"]).astype("float32")
            for cond in ("all", "dry"):
                H = wv @ a[cond, "h"]
                fh = {c: wv @ a[cond, "fh", c] for c in crit}
                fh["som"] = float((wv @ a[cond, "cnt_u"])[sel_b].sum())
                for T in T_MAIN:
                    E = wv @ a[cond, "e", T]
                    ef = {c: wv @ a[cond, "ef", T, c] for c in crit}
                    ef["som"] = float((wv @ a[cond, "ev_u", T])[sel_b].sum())
                    for c in crit:
                        if c == "precursor_seco" and cond == "all":
                            continue
                        cov = ef[c] / max(E, 1e-9)
                        amp = (ef[c] / max(fh[c], 1e-9)) / (E / H) if E else np.nan
                        boot.append((b, k, cond, T, c, cov, fh[c] / H, amp))
        if (b + 1) % 200 == 0:
            log.info(f"  {b + 1}/{N_BOOT}")
    bt = pd.DataFrame(boot, columns=["b", "conjunto", "condicao", "limiar", "criterio",
                                     "cobertura", "fracao_horas", "amplificacao"])
    q = bt.groupby(["conjunto", "condicao", "limiar", "criterio"])[["cobertura", "fracao_horas", "amplificacao"]] \
        .quantile([0.025, 0.975]).unstack()
    q.columns = [f"{a}_{'ic_inf' if p < 0.5 else 'ic_sup'}" for a, p in q.columns]
    pts = pts.merge(q.reset_index(), on=["conjunto", "condicao", "limiar", "criterio"], how="left")
    pts.to_csv(OUT / "pontos.csv", sep=";", index=False)

    # curves and areas
    rows_c, rows_a = [], []
    for k in SETS:
        s = S[k]
        cnt_k = np.bincount(s["unit"], minlength=n_units).astype(float)
        for T in T_ALL:
            _, order = select_units(cnt_tr, ev_tr[T], budget)
            ev_k = np.bincount(s["unit"][s["y"] >= T], minlength=n_units).astype(float)
            cov, auc, raw = coverage_curve(cnt_k, ev_k, order)
            rows_a.append(dict(conjunto=k, limiar=T, area=raw, area_normalizada=auc))
            rows_c += [dict(conjunto=k, limiar=T, fracao_horas=f, cobertura=c) for f, c in zip(FRACTIONS, cov)]
    pd.DataFrame(rows_c).to_csv(OUT / "curvas.csv", sep=";", index=False)
    areas = pd.DataFrame(rows_a)
    areas.to_csv(OUT / "areas.csv", sep=";", index=False)

    # units
    rows_u = []
    for k in SETS:
        s = S[k]
        cnt_k = np.bincount(s["unit"], minlength=n_units)
        d = dict(conjunto=k, unidade=np.arange(n_units), linha=np.arange(n_units) // side,
                 coluna=np.arange(n_units) % side, horas=cnt_k, monitorada=sel)
        for T in T_ALL:
            d[f"eventos_{T}"] = np.bincount(s["unit"][s["y"] >= T], minlength=n_units)
        rows_u.append(pd.DataFrame(d))
    units = pd.concat(rows_u)
    means = pd.DataFrame(unit_sums / np.maximum(cnt_tr[:, None], 1), columns=config.SOM_INPUTS)
    means.insert(0, "unidade", np.arange(n_units))
    units.to_csv(OUT / "unidades.csv", sep=";", index=False)
    means.to_csv(OUT / "unidades_media_entradas_treino.csv", sep=";", index=False)

    # event context
    rows_e = []
    for k in SETS:
        m = pd.read_parquet(ARR / f"meta_{k}.parquet",
                            columns=["prec_acum_1h"] + [f"prec_acum_{h}h" for h in DRY_H] + ["chuva_prox_1h"])
        y = m["chuva_prox_1h"].to_numpy()
        for T in [10, 20, 30, 40]:
            e = m[y >= T]
            r = dict(conjunto=k, limiar=T, eventos=len(e), horas=len(m), taxa=len(e) / len(m),
                     com_chuva_em_curso=float((e["prec_acum_1h"] > 0).mean()))
            for h in DRY_H:
                known = e[f"prec_acum_{h}h"].notna()
                r[f"apos_seca_{h}h"] = float((e.loc[known, f"prec_acum_{h}h"] == 0).mean())
                r[f"conhecidos_{h}h"] = int(known.sum())
            r["sem_chuva_em_curso"] = float((e["prec_acum_1h"] == 0).mean())
            for lo, hi in RAIN_BANDS:
                r[f"chuva_em_curso_{lo:g}_{hi:g}"] = float(((e["prec_acum_1h"] > lo) & (e["prec_acum_1h"] <= hi)).mean())
            rows_e.append(r)
    ctx = pd.DataFrame(rows_e)
    ctx.to_csv(OUT / "contexto_eventos.csv", sep=";", index=False)

    # stations
    meta = pd.read_csv(config.STATIONS_META, sep=";", dtype={"codigo": str})[["codigo", "nome", "latitude", "longitude"]]
    rows_s = []
    for k in SETS + ["todos"]:
        ks = SETS if k == "todos" else [k]
        parts = []
        for j in ks:
            e = S[j]["y"] >= T_REF
            st = S[j]["station"]
            parts.append(pd.DataFrame(dict(
                codigo=np.asarray(st.cat.categories)[st.cat.codes.to_numpy()[e]],
                som=S[j]["f_som"][e], chuva=S[j]["f_chuva"][e])))
        df = pd.concat(parts, ignore_index=True)
        g = df.groupby("codigo").agg(eventos20=("som", "size"), cobertura_som=("som", "mean"),
                                     cobertura_chuva_em_curso=("chuva", "mean")).reset_index()
        g["diferenca"] = g["cobertura_som"] - g["cobertura_chuva_em_curso"]
        g.insert(0, "conjunto", k)
        rows_s.append(g)
    pd.concat(rows_s).merge(meta, on="codigo", how="left").to_csv(OUT / "estacoes.csv", sep=";", index=False)

    # seeds
    rows_r = []
    for d in sorted(SOM_ROOT.glob("20x20_seed*")):
        if not (d / "metrics.csv").exists():
            continue
        seed = int(d.name.split("seed")[1])
        u = {k: np.load(d / f"units_{k}.npy").astype(np.int64) for k in SETS}
        c_tr = np.bincount(u["train"], minlength=n_units).astype(float)
        e_tr = {T: np.bincount(u["train"][tr["y"] >= T], minlength=n_units).astype(float) for T in T_MAIN}
        sel_r, _ = select_units(c_tr, e_tr[T_REF], budget)
        te = pd.read_csv(d / "metrics.csv", sep=";").set_index("set")["te"]
        for k in VAL:
            s = dict(S[k]); s["unit"] = u[k]
            for T in T_MAIN:
                p = point(sel_r[u[k]], s, T, "all")
                pd_ = point(sel_r[u[k]], s, T, "dry")
                _, order = select_units(c_tr, e_tr[T], budget)
                _, auc, _ = coverage_curve(np.bincount(u[k], minlength=n_units).astype(float),
                                        np.bincount(u[k][s["y"] >= T], minlength=n_units).astype(float), order)
                rows_r.append(dict(semente=seed, conjunto=k, limiar=T, unidades=int(sel_r.sum()),
                                   cobertura=p["cobertura"], amplificacao=p["amplificacao"],
                                   cobertura_seco=pd_["cobertura"], area_normalizada=auc, te=te[k]))
    seeds = pd.DataFrame(rows_r)
    seeds.to_csv(OUT / "sementes.csv", sep=";", index=False)

    # summary in the log
    pd.set_option("display.width", 200)
    v = pts[pts.limiar.isin(T_MAIN)]
    for k in SETS:
        for cond in ("all", "dry"):
            t = v[(v.conjunto == k) & (v.condicao == cond)].pivot_table(
                index="criterio", columns="limiar", values="cobertura").mul(100).round(1)
            log.info(f"\ncoverage (%) {k} {cond}\n{t}")
    t = v[(v.conjunto == "val_temporal") & (v.condicao == "all")].pivot_table(
        index="criterio", columns="limiar", values="amplificacao").round(2)
    log.info(f"\namplification val_temporal all\n{t}")
    ci = v[(v.conjunto.isin(VAL)) & v.cobertura_ic_inf.notna()]
    log.info("\n95% intervals\n" + ci.assign(
        txt=lambda d: (d.cobertura * 100).round(1).astype(str) + " (" + (d.cobertura_ic_inf * 100).round(1).astype(str)
        + " a " + (d.cobertura_ic_sup * 100).round(1).astype(str) + ")"
    ).pivot_table(index=["conjunto", "condicao", "criterio"], columns="limiar", values="txt", aggfunc="first").to_string())
    log.info(f"\narea under the coverage curve (0.5 = random order)\n{areas.pivot_table(index='limiar', columns='conjunto', values='area').round(3)}")
    log.info(f"\nnormalized area (0 = random order, 1 = perfect)\n{areas.pivot_table(index='limiar', columns='conjunto', values='area_normalizada').round(3)}")
    log.info(f"\nevent context\n{ctx.round(3).to_string(index=False)}")
    if len(seeds):
        log.info(f"\nseeds, mean and standard deviation\n"
                 f"{seeds.groupby(['conjunto', 'limiar'])[['cobertura', 'amplificacao', 'cobertura_seco', 'area_normalizada', 'te']].agg(['mean', 'std']).round(3).to_string()}")
    log.info("\ndone")
    with zipfile.ZipFile(config.LOG_DIR / "resultado_analise.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(config.LOG_DIR / "coverage.log", "coverage.log")
        for f in sorted(OUT.glob("*.csv")):
            z.write(f, f.name)


if __name__ == "__main__":
    main()
