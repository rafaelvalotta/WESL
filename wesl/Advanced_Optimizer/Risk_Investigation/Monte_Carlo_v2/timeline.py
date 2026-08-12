"""The time axis: when new speculative neighbors show up, over self's operating
horizon. See PIPELINE_DESIGN_v2.md §4.
"""


def sample_arrivals(rng, lam, horizon_years):
    """Poisson process of rate `lam` (farms/year), truncated to `horizon_years`. Draws
    exponential inter-arrival gaps and cumulative-sums them; stops once the running sum
    passes the horizon. The resulting arrival COUNT is itself random (mean lam *
    horizon_years) -- this is what makes a slow tier deliver less total growth, with no
    separate MW cap needed (§2.7)."""
    years = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / lam)
        if t > horizon_years:
            break
        years.append(t)
    return years


def build_segments(new_farms, horizon_years):
    """Sorts new_farms by arrival_year and returns one dict per segment: the interval
    during which self's active neighbor set doesn't change. Segment 0 always has zero
    active new farms -- its AEP is identical to real_neighbors_aep, so callers can reuse
    that number instead of paying for another PyWake call."""
    ordered = sorted(new_farms, key=lambda f: f["arrival_year"])
    boundaries = [0.0] + [f["arrival_year"] for f in ordered] + [horizon_years]
    return [
        dict(start_year=boundaries[i], end_year=boundaries[i + 1],
             duration_years=boundaries[i + 1] - boundaries[i], active_farms=ordered[:i])
        for i in range(len(boundaries) - 1)
    ]
