import numpy as np

import settings

T = settings.HORIZON_YEARS


def cumulative_share(t, gamma=settings.TIMELINE_GAMMA):
    return (np.exp(gamma * t) - 1) / (np.exp(gamma * T) - 1)


def _inverse(u, gamma=settings.TIMELINE_GAMMA):
    return (1 / gamma) * np.log(1 + u * (np.exp(gamma * T) - 1))


def draw_arrival_years(rng, n, gamma=settings.TIMELINE_GAMMA):
    u = np.sort(rng.uniform(size=n))
    return settings.YEAR_START + _inverse(u, gamma)
