"""
Builds the 24 PyWake Site objects (one per climate year) from the ERA5
.grib files already downloaded in era5_vineyard/ -- methodology Section
5.1, steps 1-3 (one-time preprocessing).

Pre-requisites:
    - the 24 files era5_vineyard/era5_2000.grib ... era5_2023.grib
    - pip install cfgrib scipy py_wake --break-system-packages
      (cfgrib needs the eccodes system lib -- on anaconda,
      'conda install -c conda-forge eccodes' if missing)
"""

import os

import numpy as np

from download_era5_vineyard import load_wind_series, LAT_CENTER, LON_CENTER

YEARS = [str(y) for y in range(2000, 2024)]  # 24 years: 2000-2023
ERA5_DIR = "era5_vineyard"
N_SECTORS = 12


def fit_sector_weibull(speed, direction, n_sectors=N_SECTORS):
    """
    Splits the hourly series into direction sectors and fits a 2-parameter
    Weibull (A, k) per sector, plus each sector's frequency. Returns
    (freq, A, k) lists, ready for PyWake's UniformWeibullSite.
    """
    from scipy.stats import weibull_min

    width = 360 / n_sectors
    bounds = np.arange(0, 360 + width, width)

    freq, A_list, k_list = [], [], []

    for i in range(n_sectors):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (direction >= lo) & (direction < hi)
        sector_speed = speed[mask]

        freq.append(mask.sum() / len(direction))

        if len(sector_speed) > 10:
            k, _loc, A = weibull_min.fit(sector_speed, floc=0)
        else:
            k, A = 2.0, 0.1  # sparse sector -- shouldn't happen with a full year

        A_list.append(A)
        k_list.append(k)

    return freq, A_list, k_list


def build_site_for_year(grib_path, ti=0.06):
    """
    Reads one year of ERA5 (.grib), fits sector Weibull, and returns a
    ready-to-use PyWake UniformWeibullSite.

    ti: fixed turbulence intensity placeholder (ERA5's hourly fields don't
        give TI directly). 0.06 is a typical offshore value.
    """
    from py_wake.site import UniformWeibullSite

    _time, speed, direction = load_wind_series(grib_path, LAT_CENTER, LON_CENTER)
    freq, A_list, k_list = fit_sector_weibull(speed, direction)

    site = UniformWeibullSite(p_wd=freq, a=A_list, k=k_list, ti=ti)

    summary = {
        "n_hours": len(speed),
        "mean_speed": float(speed.mean()),
        "max_speed": float(speed.max()),
        "freq_sum": float(sum(freq)),
    }
    return site, summary


def build_all_sites(years=YEARS, era5_dir=ERA5_DIR, ti=0.06, verbose=True):
    """
    Returns {year (str): PyWake Site}. Call once at pipeline start --
    not inside the 216-combination loop.
    """
    sites_by_year = {}

    for year in years:
        path = os.path.join(era5_dir, f"era5_{year}.grib")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}. Check that all 24 .grib files are in "
                f"'{era5_dir}' named era5_<year>.grib."
            )

        site, summary = build_site_for_year(path, ti=ti)
        sites_by_year[year] = site

        if verbose:
            print(f"[{year}] Site OK -- {summary['n_hours']} hours, "
                  f"mean speed={summary['mean_speed']:.2f} m/s, "
                  f"max speed={summary['max_speed']:.2f} m/s, "
                  f"freq_sum={summary['freq_sum']:.4f}")

    return sites_by_year


if __name__ == "__main__":
    print(f"Building {len(YEARS)} Site objects from '{ERA5_DIR}/'...\n")
    sites_by_year = build_all_sites()

    print(f"\n{len(sites_by_year)} sites built successfully: {sorted(sites_by_year.keys())}")

    # Sanity check -- 1 isolated turbine in 3 different years (start/middle/end)
    # confirms the sites actually vary year to year
    from py_wake import NOJ
    from py_wake.examples.data.hornsrev1 import V80

    test_years = [YEARS[0], YEARS[len(YEARS) // 2], YEARS[-1]]
    print(f"\nSanity check -- AEP of 1 isolated V80 in {test_years}:")
    for year in test_years:
        wfm_test = NOJ(sites_by_year[year], V80())
        aep = wfm_test([0], [0]).aep().sum().values
        print(f"  {year}: AEP = {aep:.4f} GWh")
