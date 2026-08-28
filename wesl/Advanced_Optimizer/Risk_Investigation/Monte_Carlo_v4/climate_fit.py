import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import i0
from scipy.optimize import linear_sum_assignment

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_MC_DIR = _THIS_DIR.parent.parent / "Project_Monte_Carlo"
sys.path.insert(0, str(_PROJECT_MC_DIR))

from download_era5_vineyard import load_wind_series, LAT_CENTER, LON_CENTER
from build_pywake_sites import YEARS, N_SECTORS, fit_sector_weibull

ERA5_DIR = _PROJECT_MC_DIR / "era5_vineyard"
CACHE_PATH = _THIS_DIR / "cache" / "wind_climate_basis.npz"
K_COMPONENTS = 5
N_BOOT = 40
SECTOR_DEG = np.linspace(0, 360, N_SECTORS, endpoint=False) + 360 / N_SECTORS / 2  # bin centers


def vonmises_pdf(x, mu, kappa):
    return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))


def _banerjee_kappa(r_bar):
    if r_bar < 0.53:
        kappa = 2 * r_bar + r_bar ** 3 + (5 * r_bar ** 5) / 6
    elif r_bar < 0.85:
        kappa = -0.4 + 1.39 * r_bar + 0.43 / (1 - r_bar)
    else:
        kappa = 1 / (r_bar ** 3 - 4 * r_bar ** 2 + 3 * r_bar)
    return np.clip(kappa, 0.1, 50.0)


def fit_von_mises_mixture(theta, k=K_COMPONENTS, max_iter=100, tol=1e-4):
    n = len(theta)
    mu = np.linspace(-np.pi, np.pi, k + 1)[:k]
    kappa = np.full(k, 2.0)
    weights = np.full(k, 1 / k)

    for _ in range(max_iter):
        resp = np.array([weights[j] * vonmises_pdf(theta, mu[j], kappa[j]) for j in range(k)])
        resp /= np.clip(resp.sum(axis=0), 1e-12, None)

        n_k = resp.sum(axis=1)
        weights = n_k / n
        old_mu = mu.copy()

        for j in range(k):
            if n_k[j] < 1e-5:
                continue
            sin_sum = np.sum(resp[j] * np.sin(theta))
            cos_sum = np.sum(resp[j] * np.cos(theta))
            mu[j] = np.arctan2(sin_sum, cos_sum)
            r_bar = np.sqrt(sin_sum ** 2 + cos_sum ** 2) / n_k[j]
            kappa[j] = _banerjee_kappa(r_bar)

        if np.allclose(mu, old_mu, atol=tol):
            break

    return weights, mu, kappa


def _circular_dist(a, b):
    return np.abs(np.angle(np.exp(1j * (a[:, None] - b[None, :]))))


def _align_to_base(mus, base_mus):
    # mixture components are unordered (label-switching) -- match each bootstrap
    # draw's components to the base fit's by nearest circular direction before
    # pooling across draws, otherwise per-mode mean/std would blend different modes
    _, order = linear_sum_assignment(_circular_dist(mus, base_mus))
    return order


def _fit_once(speed, direction_deg, k=K_COMPONENTS):
    theta = np.radians(direction_deg)
    weights, mus, kappas = fit_von_mises_mixture(theta, k=k)
    _freq, sector_C, sector_K = fit_sector_weibull(speed, direction_deg, n_sectors=N_SECTORS)
    return weights, mus, kappas, np.array(sector_C), np.array(sector_K)


def _rose_summary(speed, direction_deg):
    # Whole-rose stats for the 3-knob model: mean direction, mean speed, and one
    # overall concentration value (kappa, treating the whole sample as one mode
    # instead of per-component). Computed from the raw data, not the fitted mixture.
    theta = np.radians(direction_deg)
    cos_sum, sin_sum = np.mean(np.cos(theta)), np.mean(np.sin(theta))
    mean_dir_deg = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
    r_bar = np.clip(np.hypot(cos_sum, sin_sum), 1e-4, 0.999)
    return mean_dir_deg, float(np.mean(speed)), _banerjee_kappa(r_bar)


def _circular_diff_deg(a_deg, b_deg):
    # a - b, wrapped to [-180, 180]
    return (a_deg - b_deg + 180) % 360 - 180


def fit_and_cache(era5_dir=ERA5_DIR, years=YEARS, cache_path=CACHE_PATH, k=K_COMPONENTS, n_boot=N_BOOT):
    t0 = time.time()
    speed_by_year, dir_by_year = {}, {}
    for year in years:
        path = era5_dir / f"era5_{year}.grib"
        _t, speed, direction = load_wind_series(str(path), LAT_CENTER, LON_CENTER)
        speed_by_year[year], dir_by_year[year] = speed, direction

    speed_pooled = np.concatenate([speed_by_year[y] for y in years])
    dir_pooled = np.concatenate([dir_by_year[y] for y in years])

    base_weights, base_mus, base_kappas, base_C, base_K = _fit_once(speed_pooled, dir_pooled, k=k)
    base_mean_dir_deg, base_mean_speed, base_kappa_equiv = _rose_summary(speed_pooled, dir_pooled)

    rng = np.random.default_rng(0)
    boot = dict(weights=[], mus=[], kappas=[], C=[], K=[])
    rotation_deg, speed_ratio, spread_ratio = [], [], []
    for _ in range(n_boot):
        for _attempt in range(3):
            years_sample = rng.choice(years, size=len(years), replace=True)
            speed_s = np.concatenate([speed_by_year[y] for y in years_sample])
            dir_s = np.concatenate([dir_by_year[y] for y in years_sample])
            try:
                w, mu, kp, Cs, Ks = _fit_once(speed_s, dir_s, k=k)
                break
            except (ValueError, RuntimeError):
                continue  # a degenerate component (too few points) -- redraw the sample
        order = _align_to_base(mu, base_mus)
        boot["weights"].append(w[order])
        boot["mus"].append(mu[order])
        boot["kappas"].append(kp[order])
        boot["C"].append(Cs)  # sectors are fixed geographically -- no re-ordering needed
        boot["K"].append(Ks)

        # whole-rose 3-knob aggregate stats -- same resample, computed straight from
        # its raw (speed_s, dir_s), independent of the per-component mixture fit above
        mean_dir_s, mean_speed_s, kappa_equiv_s = _rose_summary(speed_s, dir_s)
        rotation_deg.append(_circular_diff_deg(mean_dir_s, base_mean_dir_deg))
        speed_ratio.append(mean_speed_s / base_mean_speed)
        spread_ratio.append(kappa_equiv_s / base_kappa_equiv)
    boot = {key: np.array(val) for key, val in boot.items()}
    rotation_deg, speed_ratio, spread_ratio = np.array(rotation_deg), np.array(speed_ratio), np.array(spread_ratio)

    cache_path.parent.mkdir(exist_ok=True)
    np.savez(
        cache_path,
        vm_weights=base_weights, vm_mus=base_mus, vm_kappas=base_kappas,
        vm_weights_std=boot["weights"].std(axis=0), vm_mus_std=boot["mus"].std(axis=0),
        vm_kappas_std=boot["kappas"].std(axis=0),
        sector_C=base_C, sector_K=base_K,
        sector_C_std=boot["C"].std(axis=0), sector_K_std=boot["K"].std(axis=0),
        sector_deg=SECTOR_DEG, n_years=len(years), n_boot=n_boot,
        # 3-knob stats measured from the same bootstrap resamples above -- std for
        # the draw distribution, min/max as the hard cap so a draw never goes past
        # what the real ERA5 data actually produced.
        rotation_deg_std=float(rotation_deg.std()), rotation_deg_max=float(np.abs(rotation_deg).max()),
        speed_ratio_std=float(speed_ratio.std()),
        speed_ratio_min=float(speed_ratio.min()), speed_ratio_max=float(speed_ratio.max()),
        spread_ratio_std=float(spread_ratio.std()),
        spread_ratio_min=float(spread_ratio.min()), spread_ratio_max=float(spread_ratio.max()),
    )
    print(f"Fit {len(years)} years + {n_boot}-draw block bootstrap in {time.time()-t0:.1f}s -- "
          f"{k} von Mises modes x 12 sectors (base rose) + 3-knob aggregates "
          f"(rotation std={rotation_deg.std():.2f} deg max={np.abs(rotation_deg).max():.2f} deg, "
          f"speed ratio std={speed_ratio.std():.3f} range=[{speed_ratio.min():.3f},{speed_ratio.max():.3f}], "
          f"spread ratio std={spread_ratio.std():.3f} range=[{spread_ratio.min():.3f},{spread_ratio.max():.3f}]). "
          f"Saved: {cache_path}")
    return cache_path


if __name__ == "__main__":
    fit_and_cache()
