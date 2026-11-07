# Input data

The study uses the hourly records of the INMET automatic surface stations,
obtained from BDMEP (Banco de Dados Meteorológicos para Ensino e Pesquisa),
https://bdmep.inmet.gov.br/. The raw files are not redistributed here.

## Download

1. At https://bdmep.inmet.gov.br/ request the data of the automatic stations
   with hourly periodicity, all variables, CSV with `;` as separator, for
   all stations and the period from 2000-01-01 to 2025-03-31.
2. INMET sends the files by e-mail as a zip. Unzip everything into a single
   folder. Each file is named `dados_<CODE>_H_<start>_<end>.csv` and starts
   with a header block (`Nome:`, `Codigo Estacao:`, `Latitude:`, ...)
   followed by the line `Data Medicao;Hora Medicao;...`.
3. Set that folder as `RAW_DIR` in `config_local.py` (copy of
   `config_local.example.py`). The data never go into the repository.

Times are UTC. Stations with several files are merged; when the same hour
appears in more than one file, the file that comes first in name order is used.

## Processing

`python data/build_features.py` applies the quality control and writes one
parquet per station in `WORK_DIR/parquet/`. The steps are described in
`docs/PROCESSING.md` and implemented in `data/inmet_features.py`.

## Checking a new download

`docs/raw_files_manifest.csv` lists the files used, with size
and SHA-256. `python data/raw_manifest.py` compares the folder in `RAW_DIR`
with that list and reports missing, extra and changed files. INMET may revise
past records, so a new download is not guaranteed to be identical.

## Climate map

The map figures use the IBGE climate map of Brazil (Mapa de Clima do Brasil,
1:5,000,000), available as a shapefile at https://www.ibge.gov.br/geociencias.
Set its path as `CLIMATE_SHP` in `config_local.py`. Without it the maps are drawn
without the climate zones.

## Biomes and coastline

`analysis/regional.py` uses the IBGE biome map (`BIOMES_SHP` in `config_local.py`)
and the Natural Earth 1:10m coastline, downloaded once from
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson
into `WORK_DIR/cache`.
