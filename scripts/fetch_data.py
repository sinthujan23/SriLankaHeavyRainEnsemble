"""
Download daily weather (2016-01-01 -> 2026-10-31) for 20 Sri Lankan cities
covering all 9 provinces, from the free NASA POWER API (no API key needed).

Source : NASA Langley Research Center POWER Project, https://power.larc.nasa.gov
         (MERRA-2 reanalysis + bias-corrected precipitation, daily, local solar time)
Output : data/sri_lanka_weather_daily.csv

Run    : python scripts/fetch_data.py   (from the project root)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

START, END = "20160101", "20261031"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sri_lanka_weather_daily.csv"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# city, district, province, climatic zone (Dept. of Meteorology SL), latitude, longitude
CITIES = [
    # NASA POWER meteorology uses a ~0.5 x 0.625 degree grid, so towns were chosen
    # so that every one falls in a DIFFERENT grid cell (no duplicated time-series).
    ("Colombo",      "Colombo",      "Western",       "Wet",          6.9271, 79.8612),
    ("Kalutara",     "Kalutara",     "Western",       "Wet",          6.5854, 79.9607),
    ("Kandy",        "Kandy",        "Central",       "Wet",          7.2906, 80.6337),
    ("Nuwara Eliya", "Nuwara Eliya", "Central",       "Wet",          6.9497, 80.7891),
    ("Dambulla",     "Matale",       "Central",       "Dry",          7.8742, 80.6511),
    ("Galle",        "Galle",        "Southern",      "Wet",          6.0535, 80.2210),
    ("Matara",       "Matara",       "Southern",      "Wet",          5.9549, 80.5550),
    ("Hambantota",   "Hambantota",   "Southern",      "Dry",          6.1241, 81.1185),
    ("Ratnapura",    "Ratnapura",    "Sabaragamuwa",  "Wet",          6.6828, 80.3992),
    ("Badulla",      "Badulla",      "Uva",           "Intermediate", 6.9934, 81.0550),
    ("Wellawaya",    "Monaragala",   "Uva",           "Intermediate", 6.7372, 81.1028),
    ("Chilaw",       "Puttalam",     "North Western", "Intermediate", 7.5759, 79.7953),
    ("Puttalam",     "Puttalam",     "North Western", "Dry",          8.0362, 79.8283),
    ("Anuradhapura", "Anuradhapura", "North Central", "Dry",          8.3114, 80.4037),
    ("Polonnaruwa",  "Polonnaruwa",  "North Central", "Dry",          7.9403, 81.0188),
    ("Trincomalee",  "Trincomalee",  "Eastern",       "Dry",          8.5874, 81.2152),
    ("Batticaloa",   "Batticaloa",   "Eastern",       "Dry",          7.7310, 81.6747),
    ("Pottuvil",     "Ampara",       "Eastern",       "Dry",          6.8767, 81.8341),
    ("Jaffna",       "Jaffna",       "Northern",      "Dry",          9.6615, 80.0255),
    ("Vavuniya",     "Vavuniya",     "Northern",      "Dry",          8.7514, 80.4971),
]

# NASA POWER parameter -> readable column name
PARAMS = {
    "T2M": "temp_mean_c", "T2M_MAX": "temp_max_c", "T2M_MIN": "temp_min_c",
    "PRECTOTCORR": "rain_mm", "RH2M": "humidity_pct", "T2MDEW": "dew_point_c",
    "QV2M": "specific_humidity_gkg", "PS": "pressure_kpa",
    "WS10M": "wind_speed_ms", "WS10M_MAX": "wind_max_ms", "WD10M": "wind_dir_deg",
    "ALLSKY_SFC_SW_DWN": "solar_rad_mj", "CLOUD_AMT": "cloud_cover_pct",
    "GWETTOP": "soil_wetness_top", "GWETROOT": "soil_wetness_root",
}


def fetch_city(name, district, province, zone, lat, lon):
    params = {"parameters": ",".join(PARAMS), "community": "AG", "latitude": lat, "longitude": lon,
              "start": START, "end": END, "format": "JSON", "time-standard": "LST"}
    for attempt in range(5):
        r = requests.get(URL, params=params, timeout=180)
        if r.status_code == 200:
            break
        time.sleep(20 * (attempt + 1))
    r.raise_for_status()
    js = r.json()
    d = pd.DataFrame(js["properties"]["parameter"]).rename(columns=PARAMS)
    d = d.replace(-999.0, np.nan)                       # NASA POWER fill value -> NaN
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    d = d.rename_axis("date").reset_index()
    d.insert(1, "city", name); d.insert(2, "district", district)
    d.insert(3, "province", province); d.insert(4, "climate_zone", zone)
    d["latitude"], d["longitude"] = lat, lon
    d["elevation"] = js["geometry"]["coordinates"][2]
    return d


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for c in CITIES:
        print(f"Downloading {c[0]:<13} ...", end=" ", flush=True)
        frames.append(fetch_city(*c))
        print(len(frames[-1]), "days")
        time.sleep(1)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
