"""
Isolated test: downloads ONLY year 2001 in GRIB format and reports the
time spent, to compare directly against year 2000 (netCDF), which took
about 20.5 minutes.

Run this before deciding whether to download the remaining 21 years.
"""

from download_era5_vineyard import download_year
import cdsapi

client = cdsapi.Client()

dest_path, duration = download_year(client, "2001", data_format="grib")

if duration is None:
    print("File already existed -- delete era5_vineyard/era5_2001.grib and rerun to measure the time.")
else:
    print(f"\n--- Test result ---")
    print(f"2001 (GRIB): {duration:.1f}s ({duration/60:.1f} min)")
    print(f"2000 (netCDF, reference): ~1231s (~20.5 min)")
    print(f"Difference: GRIB was {1231/duration:.1f}x faster" if duration < 1231 else "GRIB was not faster this time")
