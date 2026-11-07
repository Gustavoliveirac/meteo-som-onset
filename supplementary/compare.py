"""
compare.py - comparison of the trained maps (grid sizes, seed replicas) and of the
K-means reference, with the operating point of analysis/coverage.py.

For every model in config.WORK_DIR/som with the three units files: quantization and
topographic error, coverage of the 20, 30 and 40 mm events of the validation sets
at the budget (all hours and hours without rain in progress) and area under the
coverage curve. The 20x20 seed 42 map is compared with the K-means by 1000 paired
resamples of calendar days, choosing the unit set of both again in each resample.

    python supplementary/compare.py

Writes config.RESULTS_DIR/suplementar_modelos.csv and suplementar_som_kmeans.csv,
log in config.LOG_DIR/compare.log
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from analysis.coverage import (SETS, VAL, coverage_curve, day_matrix, load_set,  # noqa: E402
                               point, select_units)

SOM_ROOT = config.WORK_DIR / "som"
REF = "20x20_seed42"
KM = "kmeans400_seed42"
T_MAIN = [20, 30, 40]
N_BOOT = 1000
log = logging.getLogger()


def models():
    return sorted(d.name for d in SOM_ROOT.iterdir()
                  if all((d / f"units_{k}.npy").exists() for k in SETS) and (d / "metrics.csv").exists())


def main():
    global N_BOOT
    if "--boot" in sys.argv:
        N_BOOT = int(sys.argv[sys.argv.index("--boot") + 1])
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "compare.log", mode="w"),
                                  logging.StreamHandler()])
    S = {k: load_set(k) for k in SETS}
    tr = S["train"]
    budget = float((tr["rain_now"] > 0).mean())
    names = models()
    log.info(f"budget {budget:.4f}, models: {', '.join(names)}")

    rows, units = [], {}
    for name in names:
        u = {k: np.load(SOM_ROOT / name / f"units_{k}.npy").astype(np.int64) for k in SETS}
        units[name] = u
        n = int(max(x.max() for x in u.values())) + 1
        cnt = np.bincount(u["train"], minlength=n).astype(float)
        ev = {T: np.bincount(u["train"][tr["y"] >= T], minlength=n).astype(float) for T in T_MAIN}
        sel, _ = select_units(cnt, ev[20], budget)
        met = pd.read_csv(SOM_ROOT / name / "metrics.csv", sep=";").set_index("set")
        for k in VAL:
            s = dict(S[k]); s["unit"] = u[k]
            f = sel[u[k]]
            for T in T_MAIN:
                _, order = select_units(cnt, ev[T], budget)
                _, auc, area = coverage_curve(np.bincount(u[k], minlength=n).astype(float),
                                              np.bincount(u[k][s["y"] >= T], minlength=n).astype(float), order)
                # node-based view: the 10% of units with the highest training density of events
                dens = ev[T] / np.maximum(cnt, 1)
                top = np.zeros(n, bool); top[np.argsort(-dens, kind="stable")[:int(np.ceil(0.1 * n))]] = True
                hv = np.bincount(u[k], minlength=n).astype(float)
                evv = np.bincount(u[k][s["y"] >= T], minlength=n).astype(float)
                rec_n = evv[top].sum() / max(evv.sum(), 1)
                hrs_n = hv[top].sum() / hv.sum()
                cum = np.concatenate([[0], np.cumsum(evv[np.argsort(-dens, kind="stable")]) / max(evv.sum(), 1)])
                area_n = float(((cum[1:] + cum[:-1]) / 2).sum() / n)
                rows.append(dict(modelo=name, unidades=n, conjunto=k, limiar=T, unidades_monitoradas=int(sel.sum()),
                                 fracao_horas_treino=cnt[sel].sum() / cnt.sum(),
                                 cobertura=point(f, s, T, "all")["cobertura"],
                                 cobertura_seco=point(f, s, T, "dry")["cobertura"],
                                 area=area, recall_10pct_nos=rec_n, horas_10pct_nos=hrs_n, lift_10pct_nos=rec_n / hrs_n,
                                 area_nos=area_n, qe=met.loc[k, "qe"], te=met.loc[k, "te"]))
        log.info(f"  {name} done")
    tab = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(config.RESULTS_DIR / "suplementar_modelos.csv", sep=";", index=False)

    if REF in units and KM in units:
        log.info(f"paired bootstrap {REF} x {KM}, {N_BOOT} resamples")
        mats = {}
        for name in (REF, KM):
            u = units[name]
            n = int(max(x.max() for x in u.values())) + 1
            t = dict(tr); t["unit"] = u["train"]
            m = dict(cnt=day_matrix(t, np.ones(t["n"], bool), n), ev=day_matrix(t, t["y"] >= 20, n))
            for k in VAL:
                s = dict(S[k]); s["unit"] = u[k]
                m[k, "hours"] = day_matrix(s, np.ones(s["n"], bool), n)
                for cond in ("all", "dry"):
                    c = s["dry"] if cond == "dry" else np.ones(s["n"], bool)
                    for T in T_MAIN:
                        m[k, cond, T] = day_matrix(s, c & (s["y"] >= T), n)
            mats[name] = m
        rng = np.random.default_rng(42)
        out = []
        for b in range(N_BOOT):
            w = rng.poisson(1.0, tr["n_days"]).astype("float32")
            wv = {k: rng.poisson(1.0, S[k]["n_days"]).astype("float32") for k in VAL}
            cov = {}
            for name, m in mats.items():
                c_, e_ = w @ m["cnt"], w @ m["ev"]
                sel, _ = select_units(c_, e_, budget)
                n = len(c_)
                top = np.zeros(n, bool); top[np.argsort(-(e_ / np.maximum(c_, 1e-9)), kind="stable")[:int(np.ceil(0.1 * n))]] = True
                for k in VAL:
                    hv = wv[k] @ m[k, "hours"]
                    for cond in ("all", "dry"):
                        for T in T_MAIN:
                            e = wv[k] @ m[k, cond, T]
                            cov[name, k, cond, T, "horas"] = e[sel].sum() / max(e.sum(), 1e-9)
                            rec = e[top].sum() / max(e.sum(), 1e-9)
                            cov[name, k, cond, T, "nos"] = rec
                            cov[name, k, cond, T, "lift_nos"] = rec / max(hv[top].sum() / hv.sum(), 1e-9)
            for k in VAL:
                for cond in ("all", "dry"):
                    for T in T_MAIN:
                        for crit in ("horas", "nos", "lift_nos"):
                            out.append((crit, k, cond, T, cov[REF, k, cond, T, crit] - cov[KM, k, cond, T, crit]))
        bt = pd.DataFrame(out, columns=["criterio", "conjunto", "condicao", "limiar", "diferenca"])
        q = bt.groupby(["criterio", "conjunto", "condicao", "limiar"])["diferenca"].agg(
            media="mean", ic_inf=lambda x: x.quantile(0.025), ic_sup=lambda x: x.quantile(0.975)).reset_index()
        q["significativa"] = (q.ic_inf > 0) | (q.ic_sup < 0)
        q.to_csv(config.RESULTS_DIR / "suplementar_som_kmeans.csv", sep=";", index=False)
        log.info("\nSOM 20x20 minus K-means. criterio horas: coverage with the same share of hours (9.5%); "
                 "nos: recall in the 10% of units with the highest density; lift_nos: that recall divided by "
                 f"the share of hours in those units\n{q.round(4).to_string(index=False)}")

    pd.set_option("display.width", 200)
    v = tab[tab.conjunto == "val_temporal"]
    log.info("\nvalidation temporal, coverage at the budget\n" +
             v.pivot_table(index="modelo", columns="limiar", values="cobertura").round(3).to_string())
    log.info("\nvalidation temporal, recall in the 10% of units with the highest density\n" +
             v.pivot_table(index="modelo", columns="limiar", values="recall_10pct_nos").round(3).to_string())
    log.info("\nvalidation temporal, lift in those units\n" +
             v.pivot_table(index="modelo", columns="limiar", values="lift_10pct_nos").round(2).to_string())
    log.info("\nvalidation temporal, area under the coverage curve\n" +
             v.pivot_table(index="modelo", columns="limiar", values="area").round(3).to_string())
    log.info("\nquantization and topographic error, validation temporal\n" +
             v.groupby("modelo")[["unidades", "qe", "te"]].first().round(4).to_string())
    log.info("\ndone")


if __name__ == "__main__":
    main()
