"""
Combined v2 engine: 150km radius, coast-based connection proxy, precomputed
eligibility grid (federal + shallow + near-coast, merged), verify-on-accept
against the live/exact checks, bounded size backoff, full instrumentation.

Depends on v2_harness.py being exec'd first (LEASE_TEMPLATES, is_federal,
is_shallow_enough, bathy_xy, bathy_depth, VINEYARD_CENTER, build_scaled_farm,
sample_exterior, is_far_enough, fragment_budget, REGIONAL_CAPACITY_MW, etc.)
"""
import time
import numpy as np
from scipy.spatial import cKDTree

# ================================================================== radius + bands extended to 150km
DISTANCE_BANDS_KM_150 = [(3, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 75),
                          (75, 90), (90, 110), (110, 130), (130, 150)]
MAX_CONNECTION_DISTANCE_M = 70_000

# ================================================================== coast proxy (replaces substation proxy)
_coast_mask = (bathy_depth >= 0) & (bathy_depth < 5)
_coast_xy = bathy_xy[_coast_mask]
_coast_tree = cKDTree(_coast_xy)


def is_near_coast_live(xs, ys, max_dist_m=MAX_CONNECTION_DISTANCE_M):
    d, _ = _coast_tree.query(np.column_stack([xs, ys]))
    return d < max_dist_m


# ================================================================== precomputed eligibility grid (3 static filters merged)
def build_eligibility_grid_150(resolution_m=300.0, radius_km=150.0, margin_m=5_000.0):
    t0 = time.time()
    R = radius_km * 1000
    x0, x1 = VINEYARD_CENTER[0] - R - margin_m, VINEYARD_CENTER[0] + R + margin_m
    y0, y1 = VINEYARD_CENTER[1] - R - margin_m, VINEYARD_CENTER[1] + R + margin_m
    x0, x1 = min(x0, cluster_all_x.min() - margin_m), max(x1, cluster_all_x.max() + margin_m)
    y0, y1 = min(y0, cluster_all_y.min() - margin_m), max(y1, cluster_all_y.max() + margin_m)

    nx = int((x1 - x0) / resolution_m) + 1
    ny = int((y1 - y0) / resolution_m) + 1
    gx = x0 + np.arange(nx) * resolution_m
    gy = y0 + np.arange(ny) * resolution_m
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    GXf, GYf = GX.ravel(), GY.ravel()

    fed = is_federal(GXf, GYf).reshape(GX.shape)
    shallow = is_shallow_enough(GXf, GYf).reshape(GX.shape)
    coast = is_near_coast_live(GXf, GYf).reshape(GX.shape)
    eligible = fed & shallow & coast

    build_time = time.time() - t0
    return dict(x0=x0, y0=y0, res=resolution_m, nx=nx, ny=ny, eligible=eligible, build_time=build_time)


def make_grid_lookup(grid):
    x0, y0, res, nx, ny, eligible = grid["x0"], grid["y0"], grid["res"], grid["nx"], grid["ny"], grid["eligible"]

    def is_eligible_fast(xs, ys):
        ix = np.clip(((np.asarray(xs) - x0) / res).astype(int), 0, nx - 1)
        iy = np.clip(((np.asarray(ys) - y0) / res).astype(int), 0, ny - 1)
        return eligible[iy, ix]

    return is_eligible_fast


def verify_exact(bx, by, pts):
    """Live, exact re-check of the ONE candidate about to be accepted -- undoes any
    false-accept from the 300m grid discretization, at negligible cost (once per
    accepted farm, not once per attempt)."""
    ok = is_federal(bx, by).all() and is_shallow_enough(bx, by).all() and is_near_coast_live(
        np.array([bx.mean()]), np.array([by.mean()]))[0]
    if not ok:
        return False
    if not (is_federal(pts[:, 0], pts[:, 1]).all() and is_shallow_enough(pts[:, 0], pts[:, 1]).all()):
        return False
    return True


# ================================================================== V2 final: grid + backoff + verify-on-accept + instrumentation
CAPACITY_FRACTIONS = [1.0, 0.75, 0.5, 0.25]
MIN_FARM_MW_FLOOR = 30.0
TRIES_PER_STEP = 40


def placement_from_cell_150(distance_band, angle_deg, rng):
    r = rng.uniform(distance_band[0], distance_band[1]) * 1000
    theta = np.radians(angle_deg + rng.uniform(-15, 15))
    dx, dy = r * np.sin(theta), r * np.cos(theta)
    return VINEYARD_CENTER[0] + dx, VINEYARD_CENTER[1] + dy


def generate_farm_v2f(target_capacity_mw, turbine_spec, spacing_d, distance_band, angle_deg,
                       rng, existing_shapes, is_eligible_fast, tries_per_step=TRIES_PER_STEP):
    template_name = rng.choice(list(LEASE_TEMPLATES.keys()))
    n_tries_total = 0

    for frac in CAPACITY_FRACTIONS:
        cap = target_capacity_mw * frac
        if cap < MIN_FARM_MW_FLOOR:
            break
        for _ in range(tries_per_step):
            n_tries_total += 1
            target_xy = placement_from_cell_150(distance_band, angle_deg, rng)
            rotation_deg = rng.uniform(0, 360)
            shape, pts = build_scaled_farm(cap, turbine_spec, spacing_d, template_name, target_xy, rotation_deg)
            if len(pts) == 0:
                continue
            bx, by = sample_exterior(shape)
            if not is_eligible_fast(bx, by).all():
                continue
            if not is_eligible_fast(pts[:, 0], pts[:, 1]).all():
                continue
            if not is_far_enough(shape, existing_shapes):
                continue
            if not verify_exact(bx, by, pts):        # undo grid false-accepts
                continue
            log = dict(target_mw=target_capacity_mw, delivered_mw=cap, fraction_used=frac,
                       n_tries=n_tries_total, success=True, band=distance_band, angle=int(angle_deg),
                       template=template_name)
            return shape, pts, template_name, log

    log = dict(target_mw=target_capacity_mw, delivered_mw=0.0, fraction_used=0.0,
               n_tries=n_tries_total, success=False, band=distance_band, angle=int(angle_deg),
               template=None)
    return None, None, None, log


def populate_scenario_v2f(level, rng, is_eligible_fast, log_rows=None):
    cluster_x, cluster_y = fill_rest_of_cluster(rng)
    delta = REGIONAL_CAPACITY_MW[level] - REGIONAL_CAPACITY_MW["low"]
    farm_budgets = fragment_budget(delta, rng)
    existing_shapes = list(LEASE_TEMPLATES.values())
    results = []
    for cap in farm_budgets:
        name = rng.choice(list(TURBINES.keys()))
        spacing_d = rng.uniform(8.0, 11.0)
        band = DISTANCE_BANDS_KM_150[rng.integers(len(DISTANCE_BANDS_KM_150))]
        angle = ANGLE_SECTORS_DEG[rng.integers(len(ANGLE_SECTORS_DEG))]
        shape, pts, template, log = generate_farm_v2f(cap, TURBINES[name], spacing_d, band, angle,
                                                        rng, existing_shapes, is_eligible_fast)
        if shape is not None:
            existing_shapes.append(shape)
            results.append(dict(capacity_mw=log["delivered_mw"], turbine=name, shape=shape, pts=pts,
                                 target_mw=cap, fraction_used=log["fraction_used"]))
        if log_rows is not None:
            log_rows.append(log)
    return cluster_x, cluster_y, results


print("v2_final engine loaded.")
