"""ERA5 hourly wind -> PyWake Site (per-sector Weibull fit), one year at a time."""
import numpy as np
import xarray as xr
from scipy.stats import weibull_min

from py_wake.site import UniformWeibullSite

LAT_CENTER = 40.9508654
LON_CENTER = -70.6705419

N_SECTORS = 12
TURBULENCE_INTENSITY = 0.06


def _load_wind_series(grib_path, lat=LAT_CENTER, lon=LON_CENTER):
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    u, v = point["u100"].values, point["v100"].values
    speed = np.sqrt(u**2 + v**2)
    direction = (180 + np.degrees(np.arctan2(u, v))) % 360
    return speed, direction


def _fit_sector_weibull(speed, direction, n_sectors=N_SECTORS):
    width = 360 / n_sectors
    bounds = np.arange(0, 360 + width, width)

    freq, A_list, k_list = [], [], []
    for i in range(n_sectors):
        lo, hi = bounds[i], bounds[i + 1]
        sector_speed = speed[(direction >= lo) & (direction < hi)]
        freq.append(len(sector_speed) / len(direction))
        k, _loc, A = weibull_min.fit(sector_speed, floc=0)
        A_list.append(A)
        k_list.append(k)
    return freq, A_list, k_list


def build_site(grib_path, ti=TURBULENCE_INTENSITY):
    speed, direction = _load_wind_series(grib_path)
    freq, A_list, k_list = _fit_sector_weibull(speed, direction)
    site = UniformWeibullSite(p_wd=freq, a=A_list, k=k_list, ti=ti)
    summary = dict(n_hours=len(speed), mean_speed=float(speed.mean()), max_speed=float(speed.max()), freq=freq)
    return site, summary
