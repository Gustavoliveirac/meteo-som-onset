# meteo-som-onset

Self-organizing map of the hourly state of the atmosphere at INMET automatic
surface stations in Brazil, and the rainfall of the following hours projected
onto it. The map is trained on anomalies and 1, 3 and 6 h changes of pressure,
temperature, humidity, wind and radiation, without access to rainfall.

## Folders

| Folder | Content |
|---|---|
| `data/` | reading of the BDMEP files, quality control and features |
| `som/` | input arrays, SOM training and assignment of every hour to a unit |
| `analysis/` | coverage of the next-hour events by the SOM units and by the references |
| `figures/` | figures in Portuguese and English |
| `supplementary/` | K-means reference and comparison of grid sizes |
| `docs/` | data source and processing details |

## Running

```
conda env create -f environment.yml
conda activate meteo-som-onset
cp config_local.example.py config_local.py    # set RAW_DIR and WORK_DIR
python data/raw_manifest.py --write
python data/build_features.py
python som/prepare.py
python som/train.py
python som/train.py --seeds 43 44 45 46 47
python analysis/coverage.py
python analysis/unit_stats.py
python analysis/text_numbers.py
python analysis/regional.py
python analysis/states.py
python analysis/refined_groups.py
python analysis/groups.py
python analysis/seed_groups.py
python analysis/group_profiles.py
python analysis/group_conditions.py
python figures/make_figures.py
```

`python figures/make_figures.py --pdf` also writes vector PDF files (for LaTeX). File
names carry the number of the figure in the chapter.

`python run_all.py` runs the same sequence and skips the steps already done.
The supplementary material (grids from 10x10 to 100x100, K-means with 400
groups and their comparison with the 20x20 map) is run separately with
`python run_supplementary.py`, after `run_all.py`.

Raw data and large intermediate files stay outside the repository, in the
folders set in `config_local.py`. The tables behind the text are written to
`results/`, the figures to `figures/output/` (Portuguese and English) and the
trained 20x20 map to `models/`, all versioned, so a new run can be compared
with them file by file.

The map can be used without retraining:

```
from som.model import SOM
som = SOM.from_export("models/som_20x20_seed42")   # .npz weights and .json settings
```

The inputs must be standardized per station with `models/station_stats.csv`. How to obtain the data is in `docs/DATA.md` and the
processing is described in `docs/PROCESSING.md`. Logs go to `logs/`.
