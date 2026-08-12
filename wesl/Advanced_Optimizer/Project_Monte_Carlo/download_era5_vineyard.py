

import os
import sys
import time
from datetime import datetime
import cdsapi
import numpy as np
import xarray as xr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # avoid buffered prints interleaving with cdsapi's own logs

# Real center coordinate of the cluster (Vineyard Northeast)
LAT_CENTER = 40.9508654
LON_CENTER = -70.6705419

# [north, west, south, east] bounding box, ~0.3 deg margin around the
# center to cover the ERA5 native 0.25 deg grid around the cluster
AREA = [LAT_CENTER + 0.3, LON_CENTER - 0.3, LAT_CENTER - 0.3, LON_CENTER + 0.3]

YEARS = [str(y) for y in range(2000, 2024)]  # 24 years: 2000-2023

DATA_FORMAT = "grib"  # ERA5's native format -- faster and more stable than netCDF
OUTPUT_DIR = "era5_vineyard"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def file_extension(data_format):
    return ".grib" if data_format == "grib" else ".nc"


def download_year(client, year, area=AREA, output_dir=OUTPUT_DIR, data_format=DATA_FORMAT):
    """Downloads one year. Returns (file_path, duration_seconds_or_None)."""
    ext = file_extension(data_format)
    dest_path = os.path.join(output_dir, f"era5_{year}{ext}")
    if os.path.exists(dest_path):
        print(f"[{year}] already exists ({ext}), skipping.")
        return dest_path, None

    print(f"[{year}] requesting from CDS ({data_format}) at {datetime.now().strftime('%H:%M:%S')}...")
    start = time.time()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
            ],
            "year": [year],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "data_format": data_format,
        },
        dest_path,
    )
    duration = time.time() - start
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[{year}] saved to {dest_path} -- {duration:.1f}s, {size_mb:.2f} MB")
    return dest_path, duration


def download_all_years(years=YEARS):
    """Downloads all years in sequence, printing a running speed/ETA summary."""
    client = cdsapi.Client()
    paths = []
    durations = []
    total_start = time.time()

    for i, year in enumerate(years):
        dest_path, duration = download_year(client, year)
        paths.append(dest_path)

        if duration is not None:
            durations.append(duration)
            avg = sum(durations) / len(durations)
            remaining = len(years) - (i + 1)
            eta_s = avg * remaining
            print(f"    -> avg so far: {avg:.1f}s/year | {remaining} years left | ETA: {eta_s/60:.1f} min")

    total_time = time.time() - total_start
    print(f"\nDone. Run time: {total_time/60:.1f} min "
          f"({len(durations)} years actually downloaded, {len(years) - len(durations)} already existed).")

    return paths


def load_wind_series(file_path, target_lat, target_lon):
    """
    Opens an ERA5 file (.grib or .nc) and returns the hourly speed (m/s)
    and direction (deg, meteorological convention) series at the grid
    point nearest to (target_lat, target_lon).
    """
    if file_path.endswith(".grib"):
        ds = xr.open_dataset(file_path, engine="cfgrib")
    else:
        ds = xr.open_dataset(file_path)

    # dimension names vary by format/version (latitude/longitude vs lat/lon,
    # time vs valid_time) -- inspect with print(ds) if a new format is added
    point = ds.sel(latitude=target_lat, longitude=target_lon, method="nearest")

    u = point["u100"].values
    v = point["v100"].values

    speed = np.sqrt(u**2 + v**2)
    direction = (180 + np.degrees(np.arctan2(u, v))) % 360

    time_arr = point["valid_time"].values if "valid_time" in point.coords else point["time"].values

    return time_arr, speed, direction


if __name__ == "__main__":
    paths = download_all_years()

    time_arr, speed, direction = load_wind_series(paths[0], LAT_CENTER, LON_CENTER)
    print(f"Series loaded: {len(speed)} hours, mean speed = {speed.mean():.2f} m/s")
