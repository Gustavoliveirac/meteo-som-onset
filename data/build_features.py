"""
build_features.py - BDMEP hourly files -> quality control -> one parquet per station.

    python data/build_features.py              # skips stations already done
    python data/build_features.py --overwrite

Pass 1 reads every station and stores the daily rainfall totals (needed by the
neighbour check). Pass 2 applies the quality control and writes the features.

Inputs   config.RAW_DIR
Outputs  config.PARQUET_DIR/{code}.parquet
         config.STATIONS_META      one line per station, hours kept after each step
         config.LOG_DIR/build_features.log
"""
import argparse
import logging
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data import inmet_features as F  # noqa: E402


def _km(a, b):
    la1, lo1, la2, lo2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def _pass1(code, files):
    info, df = F.read_station(files)
    df = df[(df["dt"] >= config.DATA_START) & (df["dt"] <= config.DATA_END)]
    return code, info, F.daily_rain(df)


def _pass2(code, files, neigh, out_dir):
    info, df = F.read_station(files)
    out, s = F.build_station(info, df, neigh)
    if out is not None and len(out):
        out.to_parquet(out_dir / f"{code}.parquet", compression="zstd", index=False)
    s["arquivos"] = len(files)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    out_dir = config.PARQUET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(config.LOG_DIR / "build_features.log"),
                                  logging.StreamHandler()])
    log = logging.getLogger()
    import numpy, platform, pyarrow
    log.info(f"python {platform.python_version()} numpy {numpy.__version__} "
             f"pandas {pd.__version__} pyarrow {pyarrow.__version__}")
    log.info(f"raw {config.RAW_DIR}\nwork {config.WORK_DIR}")
    groups = F.group_files_by_station(config.RAW_DIR)
    log.info(f"{sum(map(len, groups.values()))} files, {len(groups)} stations")
    if not groups:
        log.error("no BDMEP file found, check RAW_DIR in config_local.py")
        return

    # pass 1, daily totals and coordinates of every station
    cache = config.WORK_DIR / "daily_rain.pkl"
    if cache.exists() and not a.overwrite:
        daily, infos = pickle.loads(cache.read_bytes())
    else:
        daily, infos = {}, {}
        with ProcessPoolExecutor(a.workers) as ex:
            for fut in as_completed([ex.submit(_pass1, k, v) for k, v in groups.items()]):
                code, info, d = fut.result()
                daily[code], infos[code] = d, info
        cache.write_bytes(pickle.dumps((daily, infos)))
    log.info(f"pass 1 done, {len(daily)} stations with daily totals")

    def neighbours(code):
        me = infos[code]
        out = []
        for k, inf in infos.items():
            if k == code or pd.isna(inf["latitude"]) or pd.isna(me["latitude"]):
                continue
            d = _km((me["latitude"], me["longitude"]), (inf["latitude"], inf["longitude"]))
            if d <= config.QC["neigh_radius_km"]:
                out.append((k, d, daily[k]))
        return sorted(out, key=lambda x: x[1])

    todo = {k: v for k, v in groups.items() if a.overwrite or not (out_dir / f"{k}.parquet").exists()}
    log.info(f"pass 2, {len(todo)} stations to process")
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_pass2, k, v, neighbours(k), out_dir): k for k, v in todo.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            k = futs[fut]
            try:
                s = fut.result()
                rows.append(s)
                log.info(f"[{i}/{len(todo)}] {k} {s.get('horas_finais', 0):,} hours")
            except Exception as e:  # noqa: BLE001
                log.error(f"[{i}/{len(todo)}] {k} FAILED {type(e).__name__}: {e}")

    meta = pd.read_csv(config.STATIONS_META, sep=";", dtype={"codigo": str}) \
        if config.STATIONS_META.exists() and not a.overwrite else pd.DataFrame()
    if rows:
        new = pd.DataFrame(rows)
        if len(meta):
            meta = meta[~meta["codigo"].isin(new["codigo"])]
        meta = pd.concat([meta, new], ignore_index=True).sort_values("codigo")
        meta.to_csv(config.STATIONS_META, sep=";", index=False)

    # summary
    m = meta.fillna(0)
    log.info("-" * 60)
    log.info(f"stations with hours kept {int((m['horas_finais'] > 0).sum())} of {len(m)}")
    for c in [c for c in m.columns if c.startswith("apos_")] + ["horas_finais"]:
        log.info(f"  {c:28s} {int(m[c].sum()):>14,}")
    for c in ["rad_noite_preenchida", "prec_dup", "dia_desc", "dia_viz_desc", "dia_viz_conf", "dia_viz_sem"]:
        if c in m:
            log.info(f"  {c:28s} {int(m[c].sum()):>14,}")
    if "qc_horas_meses_ativos" in m:
        tot = m["qc_horas_meses_ativos"].sum()
        log.info("quality control over the hours of active months (an hour can fail more than one test)")
        for c in ["qc_campo_essencial_ausente", "qc_limite_fisico", "qc_variacao_horaria", "qc_chuva", "qc_removidas_total"]:
            log.info(f"  {c:28s} {int(m[c].sum()):>14,}  {100 * m[c].sum() / tot:6.2f}%")
    if "climatologia_periodo_treino" in m:
        log.info(f"  climatology from all years (short training record): "
                 f"{int((m['climatologia_periodo_treino'] == False).sum())} stations")  # noqa: E712
    log.info("done")


if __name__ == "__main__":
    main()
