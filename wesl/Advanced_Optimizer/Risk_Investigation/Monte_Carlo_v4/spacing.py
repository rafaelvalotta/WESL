import numpy as np

import settings


def from_density(rng, rating_mw, diameter_m, max_resamples=10):
    # Draws a plausible capacity density first (settings.RHO_TARGET_TRIANGULAR,
    # based on real-project data), then works out the turbine spacing that would
    # produce it: spacing_d = sqrt(rating_MW / (rho_target_MW_km2 * diameter_km^2)).
    # If that spacing falls outside settings.SPACING_D_BOUNDS, redraw the density
    # a few times; if it's still out of range, clip the spacing and flag it instead
    # of throwing the trial away. Returns (spacing_d, rho_target_mw_km2, was_clipped).
    lo, mode, hi = settings.RHO_TARGET_TRIANGULAR
    d_min, d_max = settings.SPACING_D_BOUNDS
    d_km = diameter_m / 1000.0

    rho_target = spacing_d = None
    for _ in range(max_resamples):
        rho_target = float(rng.triangular(lo, mode, hi))
        spacing_d = float(np.sqrt(rating_mw / (rho_target * d_km ** 2)))
        if d_min <= spacing_d <= d_max:
            return spacing_d, rho_target, False

    return float(np.clip(spacing_d, d_min, d_max)), rho_target, True
