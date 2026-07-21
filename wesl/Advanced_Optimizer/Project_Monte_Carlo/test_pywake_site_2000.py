"""
Builds and tests a PyWake Site from ONE year of ERA5 data already
downloaded locally -- isolated test for year 2000, before scaling to
the full 24 years of Phase 1a.

Covers methodology Section 5.1, steps 2-3:
    2) Fit sector Weibull (12 x 30 deg sectors) for the year
    3) Build the PyWake Site from those parameters

Pre-requisites:
    - era5_vineyard/era5_2000.nc already downloaded (adjust NC_FILE below)
    - run from the same directory as download_era5_vineyard.py
    - pip install scipy py_wake --break-system-packages
"""

import numpy as np

from download_era5_vineyard import load_wind_series

NC_FILE = "/Users/brunoboer/Desktop/Advanced_Optimizer/era5_vineyard/era5_2000.nc"

LAT_CENTER = 40.9508654
LON_CENTER = -70.6705419

N_SECTORS = 12
SECTOR_WIDTH = 360 / N_SECTORS  # 30 deg


def fit_sector_weibull(speed, direction, n_sectors=N_SECTORS):
    """
    Splits the hourly series into direction sectors (0-360, starting at 0)
    and fits a 2-parameter Weibull (A, k) per sector, plus each sector's
    frequency. Returns (freq, A, k) lists ready for PyWake's UniformWeibullSite.
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
            k, _loc, A = weibull_min.fit(sector_speed, floc=0)  # loc fixed at 0, standard convention
        else:
            k, A = 2.0, 0.1  # sparse sector -- shouldn't happen with a full year

        A_list.append(A)
        k_list.append(k)

    return freq, A_list, k_list


if __name__ == "__main__":
    print(f"Opening {NC_FILE} ...")
    time_arr, speed, direction = load_wind_series(NC_FILE, LAT_CENTER, LON_CENTER)

    print(f"Series loaded: {len(speed)} hours")
    print(f"Mean speed: {speed.mean():.2f} m/s")
    print(f"Max speed: {speed.max():.2f} m/s")

    freq, A_list, k_list = fit_sector_weibull(speed, direction)

    print("\nSector parameters (12 x 30 deg):")
    for i, (f, a, k) in enumerate(zip(freq, A_list, k_list)):
        lo = i * 30
        print(f"  Sector {lo:3d}-{lo + 30:3d} deg: freq={f:.3f}  A={a:5.2f}  k={k:.2f}")
    print(f"\nFrequency sum (should be ~1.0): {sum(freq):.4f}")

    from py_wake.site import UniformWeibullSite

    site_2000 = UniformWeibullSite(
        p_wd=freq,
        a=A_list,
        k=k_list,
        ti=0.1,  # placeholder -- can be refined with real ERA5-derived TI later
    )
    print("\nSite created successfully: UniformWeibullSite (year 2000)")

    from py_wake import NOJ
    from py_wake.examples.data.hornsrev1 import V80

    test_wt = V80()
    wfm_test = NOJ(site_2000, test_wt)
    sim_res = wfm_test([0], [0])  # 1 isolated turbine at the origin

    total_aep = sim_res.aep().sum().values
    print(f"\nSanity check -- AEP of 1 isolated V80 turbine on this site: {total_aep:.4f} GWh")
    print("A non-zero value in the few-GWh range confirms the 2000 Site works correctly.")
