"""
config.py - paths and parameters shared by the whole pipeline.

Local paths go in config_local.py (not versioned), for example

    RAW_DIR = "/media/disk/som_dados/inmet_bdmep"
    WORK_DIR = "/media/disk/som_dados/work"

Without config_local.py the folders ../meteo-som-onset-dados/inmet_bdmep and
../meteo-som-onset-dados/work, next to the repository, are used. Only the raw data
and the large intermediate files live there; the result tables, the figures and
the trained 20x20 map are written inside the repository (results/, figures/output/,
models/).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / "meteo-som-onset-dados" / "inmet_bdmep"
WORK_DIR = ROOT.parent / "meteo-som-onset-dados" / "work"
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
RAW_DIR, WORK_DIR = Path(RAW_DIR), Path(WORK_DIR)

PARQUET_DIR = WORK_DIR / "parquet"
STATIONS_META = WORK_DIR / "stations_metadata.csv"
RESULTS_DIR = ROOT / "results"          # tables behind the text, versioned
FIG_DIR = ROOT / "figures" / "output"   # figures in Portuguese and English, versioned
LOG_DIR = ROOT / "logs"          # logs and deliveries, not versioned
SPATIAL_STATIONS = ROOT / "data" / "spatial_validation_stations.csv"

# ------------------------------------------------------------ periods
DATA_START = pd.Timestamp("2000-04-01 00:00")
TRAIN_CUTOFF = pd.Timestamp("2022-09-30 23:00")   # training up to here, temporal validation after
DATA_END = pd.Timestamp("2025-03-31 23:00")
ACTIVE_MONTH_MM = 50.0        # months whose mean total is below this are excluded

# ------------------------------------------------------------ quality control
QC = dict(
    t_min=-10.0, t_max=50.0, td_min=-15.0, td_max=35.0,
    p_max=1050.0, p_margin_alt=0.95,            # lower limit = 0.95 x barometric pressure at the altitude
    ur_min=1.0, ur_max=100.0,
    rad_min=-50.0, rad_max=5000.0,              # kJ/m2
    night_start_solar=18.0, night_end_solar=6.0,
    vv_max=60.0, prec_max_hour=150.0,
    jump_t=10.0, jump_p=5.0, jump_ur=40.0, jump_td=8.0, jump_bad_factor=2.0,
    prec_dup_min=10.0,                          # two consecutive hours with the same value >= 10 mm are discarded
    prec_day_discard=600.0,                     # days >= 600 mm are discarded
    prec_day_check=200.0,                       # days between 200 and 600 mm need a neighbour
    neigh_radius_km=150.0, neigh_n=2, neigh_confirm_mm=100.0,
    month_min_hours=10, clim_min_count=5, ref_min_fraction=1.0,
)

# ------------------------------------------------------------ SOM input vector
ANOM_VARS = ["pressao", "temperatura", "ponto_orvalho", "umidade_relativa",
             "vento_velocidade", "radiacao"]
GRAD_WINDOWS = [1, 3, 6]
_G = ["pressao", "temperatura", "ponto_orvalho", "umidade_relativa", "vento_velocidade", "theta_e"]
GRAD_VARS = {1: _G[:5] + ["radiacao", "theta_e"], 3: _G, 6: _G}
_A = ["pressao", "temperatura", "ponto_orvalho", "umidade_relativa"]
GRAD_ANOM_VARS = {1: _A + ["radiacao"], 3: _A, 6: _A}
SOM_INPUTS = ([f"anom_{v}" for v in ANOM_VARS]
              + [f"grad{k}h_{v}" for k in GRAD_WINDOWS for v in GRAD_VARS[k]]
              + [f"grad{k}h_anom_{v}" for k in GRAD_WINDOWS for v in GRAD_ANOM_VARS[k]])
HORIZONS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24]

# number of atmospheric states of the monitored units (analysis/states.py)
STATES_K = None
# groups kept in the monitored set but left out of the group description (too few events)
GROUPS_EXCLUDED = [6]        # None: the number with the highest silhouette

# periods of the day (local solar hour) taken from the hours at which each group of
# monitored units concentrates its events: [start, end) in hours
SOLAR_PERIODS = [(0, 7, "madrugada"), (7, 12, "manhã"), (12, 17, "tarde"), (17, 19, "fim da tarde"), (19, 24, "noite")]
