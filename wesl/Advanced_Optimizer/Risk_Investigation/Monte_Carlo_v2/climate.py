"""Per-trial climate draw for the Monte Carlo -- ported from
Wind_Resource/wind_resource_emulator.ipynb section 4. Loads the cached PCA basis
(built once by climate_fit.py) at import time; every call after that is just a
sampling operation, no GRIB reads, no SVD.

If cache/wind_climate_pca_basis.npz does not exist yet, run `python climate_fit.py`
once before using this module.
"""
from pathlib import Path

import numpy as np
from py_wake.site import UniformWeibullSite

_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "wind_climate_pca_basis.npz"

if not _CACHE_PATH.exists():
    raise FileNotFoundError(
        f"{_CACHE_PATH} not found -- run `python climate_fit.py` once first "
        "(one-time PCA fit over the 24 ERA5 years, cached so the Monte Carlo "
        "loop never re-reads GRIB files or re-runs the SVD)."
    )

_basis = np.load(_CACHE_PATH)
S = _basis["S"]
Vt = _basis["Vt"]
COL_MEAN = _basis["col_mean"]
COL_STD = _basis["col_std"]
N_MODES = int(_basis["n_modes"])
N_YEARS = int(_basis["n_years"])
TI_DEFAULT = 0.06


def sample_synthetic_row(rng, n_modes=N_MODES, scale=1.0, extreme_modes=None):
    """One point in the 36-dim (freq, A, k) space, sampled from the fitted PCA basis.
    `extreme_modes`, if set, applies `scale` only to the first `extreme_modes` (the
    strongest, most physically real modes) and leaves the rest at their natural size --
    see the emulator notebook section 5 for why (pushing every mode to an extreme at
    once produces climates no real year resembles even loosely)."""
    scale_vec = np.full(n_modes, scale if extreme_modes is None else 1.0)
    if extreme_modes is not None:
        scale_vec[:extreme_modes] = scale

    scores = rng.normal(0, 1, size=n_modes) * scale_vec * S[:n_modes] / np.sqrt(N_YEARS - 1)
    std_row = scores @ Vt[:n_modes]
    row = std_row * COL_STD + COL_MEAN

    freq, A, k = row[:12], row[12:24], row[24:36]
    freq = np.clip(freq, 1e-4, None)
    freq = freq / freq.sum()
    A = np.clip(A, 0.5, None)
    k = np.clip(k, 0.3, None)
    return np.concatenate([freq, A, k])


def sample_synthetic_site(rng, n_modes=N_MODES, scale=1.0, extreme_modes=None, ti=TI_DEFAULT):
    """One Monte Carlo climate draw -- a ready PyWake UniformWeibullSite, the exact
    same constructor used everywhere else in the pipeline. `scale=1.0` stays within
    the spread the 24 real years already show; `scale=3.0, extreme_modes=3` is the
    calibrated "deliberately extreme, still physically anchored" setting from the
    emulator notebook."""
    row = sample_synthetic_row(rng, n_modes=n_modes, scale=scale, extreme_modes=extreme_modes)
    freq, A, k = row[:12], row[12:24], row[24:36]
    return UniformWeibullSite(p_wd=list(freq), a=list(A), k=list(k), ti=ti)


def climate_summary(rng, n_modes=N_MODES, scale=1.0, extreme_modes=None):
    """Same draw as sample_synthetic_site, but returns the raw (freq, A, k) plus a
    couple of summary stats instead of a Site object -- for instrumentation (log what
    climate a trial actually got, not just the Site object PyWake consumes)."""
    from scipy.special import gamma
    row = sample_synthetic_row(rng, n_modes=n_modes, scale=scale, extreme_modes=extreme_modes)
    freq, A, k = row[:12], row[12:24], row[24:36]
    mean_speed_by_sector = A * gamma(1 + 1 / k)
    overall_mean_speed = float(np.sum(freq * mean_speed_by_sector))
    dominant_sector = int(np.argmax(freq))
    return dict(
        overall_mean_speed=overall_mean_speed,
        dominant_sector=dominant_sector,
        dominant_sector_freq=float(freq[dominant_sector]),
        scale=scale, extreme_modes=extreme_modes,
        freq=freq.tolist(), A=A.tolist(), k=k.tolist(),
    )
