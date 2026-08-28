from pathlib import Path

import numpy as np
from scipy.special import i0
from py_wake.site import UniformWeibullSite

_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "wind_climate_basis.npz"

if not _CACHE_PATH.exists():
    raise FileNotFoundError(f"{_CACHE_PATH} not found -- run `python climate_fit.py` once first.")

_basis = np.load(_CACHE_PATH)
VM_WEIGHTS, VM_MUS, VM_KAPPAS = _basis["vm_weights"], _basis["vm_mus"], _basis["vm_kappas"]
SECTOR_C, SECTOR_K = _basis["sector_C"], _basis["sector_K"]
SECTOR_RAD = np.radians(_basis["sector_deg"])
TI_DEFAULT = 0.06

# 3 knobs that move the whole wind rose at once (rotation, speed, spread).
# Ranges come straight from climate_fit.py's bootstrap, capped at the actual
# observed min/max so no draw is stronger than what 24 years of real ERA5
# data ever produced. See PIPELINE.md for the full reasoning.
ROTATION_STD_DEG, ROTATION_MAX_DEG = float(_basis["rotation_deg_std"]), float(_basis["rotation_deg_max"])
SPEED_RATIO_STD = float(_basis["speed_ratio_std"])
SPEED_RATIO_MIN, SPEED_RATIO_MAX = float(_basis["speed_ratio_min"]), float(_basis["speed_ratio_max"])
SPREAD_RATIO_STD = float(_basis["spread_ratio_std"])
SPREAD_RATIO_MIN, SPREAD_RATIO_MAX = float(_basis["spread_ratio_min"]), float(_basis["spread_ratio_max"])


def vonmises_pdf(x, mu, kappa):
    return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))


def sample_scenario(rng):
    # One draw of the 3 climate knobs, applied on top of the base fitted rose:
    #   rotation_deg      -- rotates all wind directions together
    #   speed_multiplier  -- scales mean wind speed (all sectors together)
    #   spread_multiplier -- scales how concentrated the directions are
    # Each is a capped Normal draw around its base value. Weights and the
    # Weibull shape parameter stay fixed -- only these 3 things vary.
    rotation_deg = float(np.clip(rng.normal(0.0, ROTATION_STD_DEG), -ROTATION_MAX_DEG, ROTATION_MAX_DEG))
    speed_multiplier = float(np.clip(rng.normal(1.0, SPEED_RATIO_STD), SPEED_RATIO_MIN, SPEED_RATIO_MAX))
    spread_multiplier = float(np.clip(rng.normal(1.0, SPREAD_RATIO_STD), SPREAD_RATIO_MIN, SPREAD_RATIO_MAX))

    weights = VM_WEIGHTS
    mus = VM_MUS + np.radians(rotation_deg)
    kappas = np.clip(VM_KAPPAS * spread_multiplier, 0.1, 50.0)
    sector_C = np.clip(SECTOR_C * speed_multiplier, 1.0, None)
    sector_K = SECTOR_K

    return dict(weights=weights, mus=mus, kappas=kappas, sector_C=sector_C, sector_K=sector_K,
                rotation_deg=rotation_deg, speed_multiplier=speed_multiplier, spread_multiplier=spread_multiplier)


def scenario_to_site_params(scenario):
    weights, mus, kappas = scenario["weights"], scenario["mus"], scenario["kappas"]
    dens = np.array([w * vonmises_pdf(SECTOR_RAD, mu, kp) for w, mu, kp in zip(weights, mus, kappas)])
    freq = dens.sum(axis=0)
    freq = freq / freq.sum()
    return freq, scenario["sector_C"], scenario["sector_K"]


def sample_site(rng, ti=TI_DEFAULT):
    scenario = sample_scenario(rng)
    freq, A, k = scenario_to_site_params(scenario)
    site = UniformWeibullSite(p_wd=list(freq), a=list(A), k=list(k), ti=ti)
    return site, scenario
