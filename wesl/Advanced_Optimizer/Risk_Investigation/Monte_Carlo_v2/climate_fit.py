"""One-time PCA fit over the 24 ERA5 years (2000-2023) -- ported from
Wind_Resource/wind_resource_emulator.ipynb section 3. Run this once (`python
climate_fit.py`, or call `fit_and_cache()` directly) to read all 24 GRIB files and
save the fitted basis to cache/wind_climate_pca_basis.npz. climate.py loads that
cache at import time -- nothing in the Monte Carlo loop itself re-reads a GRIB file
or re-runs an SVD.

Why this is safe to cache: the 24 historical years are a fixed dataset, not something
that changes trial to trial. The fit (mean, std, and the SVD of the standardized
24x36 matrix) is fixed geometry derived from that fixed dataset -- only the *sampling*
step (climate.py's sample_synthetic_site) is random, and it is cheap (a few numpy ops
on an already-loaded basis). Re-run this script only if the underlying ERA5 data
changes (e.g. more years added).
"""
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_MC_DIR = _THIS_DIR.parent.parent / "Project_Monte_Carlo"
sys.path.insert(0, str(_PROJECT_MC_DIR))

from download_era5_vineyard import load_wind_series, LAT_CENTER, LON_CENTER
from build_pywake_sites import fit_sector_weibull, YEARS, N_SECTORS

ERA5_DIR = _PROJECT_MC_DIR / "era5_vineyard"
CACHE_PATH = _THIS_DIR / "cache" / "wind_climate_pca_basis.npz"


def fit_and_cache(era5_dir=ERA5_DIR, years=YEARS, cache_path=CACHE_PATH, variance_target=0.90):
    t0 = time.time()
    rows = []
    for year in years:
        path = era5_dir / f"era5_{year}.grib"
        _t, speed, direction = load_wind_series(str(path), LAT_CENTER, LON_CENTER)
        freq, A, k = fit_sector_weibull(speed, direction)
        rows.append(np.concatenate([freq, A, k]))

    real_matrix = np.array(rows)  # 24 x 36 (12 freq, 12 A, 12 k)

    col_mean = real_matrix.mean(axis=0)
    col_std = real_matrix.std(axis=0)
    std_matrix = (real_matrix - col_mean) / col_std

    U, S, Vt = np.linalg.svd(std_matrix, full_matrices=False)
    explained_var = S ** 2 / np.sum(S ** 2)
    n_modes = int(np.searchsorted(np.cumsum(explained_var), variance_target) + 1)

    cache_path.parent.mkdir(exist_ok=True)
    np.savez(
        cache_path,
        S=S, Vt=Vt, col_mean=col_mean, col_std=col_std,
        n_modes=n_modes, n_years=len(years), n_sectors=N_SECTORS,
        years=np.array(years),
    )
    print(f"Fit {len(years)} years in {time.time()-t0:.1f}s -- {n_modes} modes for "
          f"{variance_target*100:.0f}% variance. Saved: {cache_path}")
    return cache_path


if __name__ == "__main__":
    fit_and_cache()
