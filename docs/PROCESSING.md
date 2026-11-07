# Processing of the station records

All parameters are in `config.py` (`QC`, periods and the SOM input list).
Times are UTC; night is defined by local solar time (UTC + longitude/15).

## Quality control (per hour)

| Check | Rule | Result |
|---|---|---|
| essential fields | temperature, dew point, pressure, relative humidity or rainfall missing | hour removed |
| physical limits | T -10 to 50 °C, Td -15 to 35 °C, Td <= T + 0.5, RH 1 to 100 %, rain 0 to 150 mm/h, radiation -50 to 5000 kJ/m², wind 0 to 60 m/s | hour removed |
| pressure | between 0.95 x barometric pressure at the station altitude and 1050 hPa | hour removed |
| hourly jumps | T 10 °C, Td 8 °C, p 5 hPa, RH 40 %; above twice the limit | hour removed (flagged when between 1 and 2 times) |
| night radiation | negative values, or no value on a day with daytime values | set to zero |
| repeated rain | same value >= 10 mm in two consecutive hours | both hours removed |
| daily rain | day >= 600 mm | day removed |
| neighbours | day between 200 and 600 mm; if the two nearest stations within 150 km with data that day both have less than 100 mm | day removed (kept when no neighbour has data) |

## Climatology and months used

Mean and standard deviation per month and hour of each station are computed
from the valid hours up to the end of the training period (2022-09-30).
Stations with less than one year of valid hours in that period use all their
valid hours. Months whose mean total is below 50 mm are excluded.

## SOM input (38 variables)

| Group | Variables |
|---|---|
| anomalies (6) | pressure, temperature, dew point, RH, wind speed, radiation, standardized by the station climatology and limited to ±4 |
| 1 h change (7) | pressure, temperature, dew point, RH, wind speed, radiation, θe (equivalent potential temperature) |
| 3 h and 6 h change (6 each) | pressure, temperature, dew point, RH, wind speed, θe |
| 1 h change of the anomaly (5) | pressure, temperature, dew point, RH, radiation |
| 3 h and 6 h change of the anomaly (4 each) | pressure, temperature, dew point, RH |

The k-hour change is x(t) - x(t-k), equal to the sum of the k hourly changes.
An hour is kept only if the 6 previous hours exist, are valid and all 38
inputs are present.

## Kept for evaluation, not given to the SOM

Raw values, wind direction, local solar hour, current rainfall and rainfall
accumulated over the previous 2, 3, 6, 12 and 24 h, and the rainfall of the next
1, 2, 3, 4, 5, 6, 8, 10, 12, 18 and 24 h (missing when any hour of the
horizon is missing).

`stations_metadata.csv` reports, per station, the hours left after each step.
