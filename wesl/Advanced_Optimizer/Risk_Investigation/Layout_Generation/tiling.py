"""Parallelogram-tiling turbine placement, per the 6-parameter grid family
in Malisani, Bartement & Bozonnet (2025), "Offshore wind farm layout
optimization with alignment constraints", Wind Energ. Sci. 10, 1611-1623,
https://doi.org/10.5194/wes-10-1611-2025 (patent WO 2024/061627), Eq. (6)-(9).

The paper *optimizes* (r1, r2, theta1, theta2, vx, vy) to find the single
AEP-maximizing aligned grid for a given boundary. Here the same 6-parameter
family is used the other way round -- as a *sampling space* for the Monte
Carlo risk study: not "what is the best grid" but "how much does the turbine
count / AEP vary across grids a real developer could plausibly choose",
given that alignment (theta1/theta2) is fixed by navigational regulation and
only spacing (r1, r2) and phase (vx, vy) are genuinely uncertain.

Supersedes the standalone snippet in `test.py` (rotation only, no
independent theta1/theta2, offset unused) with the full parameterization,
plus an explicit MC sampling wrapper (`sample_aligned_grid`).
"""
import numpy as np
from shapely.geometry import Point, Polygon


def basis_matrix(r1, r2, theta1_deg, theta2_deg):
    """P^{B(theta1,theta2,r1,r2)}_{B0} from Eq. (6) -- grid-basis to Cartesian.
    r1, r2 are the parallelogram side lengths; theta1, theta2 the angles of
    those sides to the x axis. theta2 = theta1 + 90 is the rectangular case
    (every real lease in this cluster, see tiling_alignment.ipynb Sect. 1)."""
    t1, t2 = np.radians(theta1_deg), np.radians(theta2_deg)
    return np.array([
        [r1 * np.cos(t1), r2 * np.cos(t2)],
        [r1 * np.sin(t1), r2 * np.sin(t2)],
    ])


def grid_intersections(polygon, r1, r2, theta1_deg, theta2_deg=None, vx=0.0, vy=0.0):
    """All lattice points of the aligned grid (Eq. 7-9) that fall inside `polygon`.

    p = (k1, k2) + P^-1 (vx, vy), k1, k2 in Z, mapped back to Cartesian by P
    and centered on the polygon so (vx, vy) is a *local* phase offset, not an
    absolute one -- it does not need to track the polygon's absolute position.
    """
    if theta2_deg is None:
        theta2_deg = theta1_deg + 90.0
    P = basis_matrix(r1, r2, theta1_deg, theta2_deg)
    Pinv = np.linalg.inv(P)

    xmin, ymin, xmax, ymax = polygon.bounds
    diag = np.hypot(xmax - xmin, ymax - ymin)
    n1 = int(diag / r1) + 2
    n2 = int(diag / r2) + 2
    cx, cy = polygon.centroid.x, polygon.centroid.y

    k1 = np.arange(-n1, n1 + 1)
    k2 = np.arange(-n2, n2 + 1)
    K1, K2 = np.meshgrid(k1, k2)
    k = np.column_stack([K1.ravel(), K2.ravel()])

    offset_grid_basis = Pinv @ np.array([vx, vy])
    pts_grid_basis = k + offset_grid_basis
    pts = (P @ pts_grid_basis.T).T + [cx, cy]

    mask = [polygon.contains(Point(p)) for p in pts]
    return pts[mask]


def sample_aligned_grid(polygon, rng, r_nominal, r_jitter_frac=0.1,
                         theta1_deg=0.0, theta2_deg=None,
                         drop_prob=0.0, drop_cluster_scale=None):
    """One Monte Carlo draw of a real-world-plausible aligned grid fill.

    - r1, r2 ~ U[(1-jitter)*r_nominal, (1+jitter)*r_nominal], drawn
      independently (spacing uncertainty, e.g. 0.9-1.1 nm around a 1 nm norm).
    - vx, vy ~ U[0, r1) x U[0, r2): sampling a phase offset beyond one grid
      cell is redundant -- the lattice is periodic in (vx, vy) by construction,
      so this is the whole support, not a truncation.
    - theta1 (and theta2) is NOT sampled here: alignment is a navigational
      constraint imposed by regulators, not a random variable, so it stays
      fixed at the regionally observed value unless the caller is
      deliberately building a separate misalignment-risk scenario.
    - drop_prob: per-intersection thinning probability, simulating positions
      that turn out unbuildable for reasons the regional data doesn't show.
      drop_cluster_scale, if given, replaces iid thinning with a smooth
      random field (`_cluster_mask`) so a bad patch removes a contiguous
      group of positions instead of a random scatter -- closer to a real
      localized geotechnical exclusion than iid dropout.
    """
    r1 = rng.uniform((1 - r_jitter_frac) * r_nominal, (1 + r_jitter_frac) * r_nominal)
    r2 = rng.uniform((1 - r_jitter_frac) * r_nominal, (1 + r_jitter_frac) * r_nominal)
    vx = rng.uniform(0, r1)
    vy = rng.uniform(0, r2)
    pts = grid_intersections(polygon, r1, r2, theta1_deg, theta2_deg, vx, vy)
    params = dict(r1=r1, r2=r2, theta1=theta1_deg, vx=vx, vy=vy)

    if len(pts) == 0 or drop_prob <= 0:
        return pts, params

    if drop_cluster_scale is None:
        keep = rng.uniform(size=len(pts)) > drop_prob
    else:
        keep = _cluster_mask(pts, rng, drop_prob, drop_cluster_scale)
    return pts[keep], params


def _cluster_mask(pts, rng, drop_prob, scale):
    """Low-frequency random field over `pts` (sum of a few random-phase plane
    waves + small iid noise), thresholded to drop an expected `drop_prob`
    fraction of points as spatially contiguous blobs rather than iid noise."""
    n_waves = 4
    field = np.zeros(len(pts))
    for _ in range(n_waves):
        k_dir = rng.normal(size=2)
        k_dir /= np.linalg.norm(k_dir)
        k_mag = 2 * np.pi / (scale * rng.uniform(0.7, 1.3))
        phase = rng.uniform(0, 2 * np.pi)
        field += np.cos(pts @ (k_dir * k_mag) + phase)
    field += rng.normal(scale=0.3, size=len(pts))
    threshold = np.quantile(field, drop_prob)
    return field > threshold
