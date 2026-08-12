"""
Three variants of generate_farm / populate_scenario, built on top of v2_harness.py:

  v1_original     -- exact original logic (200 tries, no backoff, no instrumentation)
  v1_instrumented -- identical logic to v1, only adds logging (isolates instrumentation cost)
  v2              -- precomputed eligibility grid + bounded size backoff + instrumentation

Run after v2_harness.py in the same process (exec'd together by run_benchmark.py).
"""
import time
import numpy as np

# ================================================================== V1 (original, faithful copy)
def generate_farm_v1(target_capacity_mw, turbine_spec, spacing_d, distance_band, angle_deg,
                      rng, existing_shapes, max_tries=200):
    template_name = rng.choice(list(LEASE_TEMPLATES.keys()))
    for _ in range(max_tries):
        target_xy = placement_from_cell(distance_band, angle_deg, rng)
        rotation_deg = rng.uniform(0, 360)
        shape, pts = build_scaled_farm(target_capacity_mw, turbine_spec, spacing_d, template_name, target_xy, rotation_deg)
        if len(pts) == 0:
            continue
        bx, by = sample_exterior(shape)
        if not is_federal(bx, by).all():
            continue
        if not is_shallow_enough(pts[:, 0], pts[:, 1]).all():
            continue
        if not is_connection_plausible(np.array([target_xy[0]]), np.array([target_xy[1]]))[0]:
            continue
        if not is_far_enough(shape, existing_shapes):
            continue
        return shape, pts, template_name
    return None, None, None


def populate_scenario_v1(level, rng, log_rows=None):
    """log_rows, if passed, gets one dict appended per farm attempt -- instrumentation
    bolted onto v1 with zero change to the accept/reject logic itself, to isolate its cost."""
    cluster_x, cluster_y = fill_rest_of_cluster(rng)
    delta = REGIONAL_CAPACITY_MW[level] - REGIONAL_CAPACITY_MW["low"]
    farm_budgets = fragment_budget(delta, rng)
    existing_shapes = list(LEASE_TEMPLATES.values())
    results = []
    for cap in farm_budgets:
        name = rng.choice(list(TURBINES.keys()))
        spacing_d = rng.uniform(8.0, 11.0)
        band = DISTANCE_BANDS_KM[rng.integers(len(DISTANCE_BANDS_KM))]
        angle = ANGLE_SECTORS_DEG[rng.integers(len(ANGLE_SECTORS_DEG))]
        shape, pts, template = generate_farm_v1(cap, TURBINES[name], spacing_d, band, angle, rng, existing_shapes, max_tries=200)
        success = shape is not None
        if success:
            existing_shapes.append(shape)
            results.append(dict(capacity_mw=cap, turbine=name, shape=shape, pts=pts))
        if log_rows is not None:
            log_rows.append(dict(
                band=band, angle=int(angle), target_mw=cap, delivered_mw=(cap if success else 0.0),
                fraction_used=(1.0 if success else 0.0), success=success, n_turbines=(len(pts) if success else 0),
            ))
    return cluster_x, cluster_y, results


# ================================================================== Precomputed eligibility grid
def build_eligibility_grid(resolution_m=300.0, margin_m=5_000.0):
    t0 = time.time()
    max_band_m = DISTANCE_BANDS_KM[-1][1] * 1000
    x0, x1 = VINEYARD_CENTER[0] - max_band_m - margin_m, VINEYARD_CENTER[0] + max_band_m + margin_m
    y0, y1 = VINEYARD_CENTER[1] - max_band_m - margin_m, VINEYARD_CENTER[1] + max_band_m + margin_m
    # also cover the existing cluster footprint, in case it extends past the band circle
    x0, x1 = min(x0, cluster_all_x.min() - margin_m), max(x1, cluster_all_x.max() + margin_m)
    y0, y1 = min(y0, cluster_all_y.min() - margin_m), max(y1, cluster_all_y.max() + margin_m)

    nx = int((x1 - x0) / resolution_m) + 1
    ny = int((y1 - y0) / resolution_m) + 1
    gx = x0 + np.arange(nx) * resolution_m
    gy = y0 + np.arange(ny) * resolution_m
    GX, GY = np.meshgrid(gx, gy, indexing="xy")  # shape (ny, nx)

    fed = is_federal(GX.ravel(), GY.ravel()).reshape(GX.shape)
    shallow = is_shallow_enough(GX.ravel(), GY.ravel()).reshape(GX.shape)
    eligible = fed & shallow

    build_time = time.time() - t0
    return dict(x0=x0, y0=y0, res=resolution_m, nx=nx, ny=ny, eligible=eligible, build_time=build_time)


def make_grid_lookup(grid):
    x0, y0, res, nx, ny, eligible = grid["x0"], grid["y0"], grid["res"], grid["nx"], grid["ny"], grid["eligible"]

    def is_eligible_fast(xs, ys):
        ix = np.clip(((np.asarray(xs) - x0) / res).astype(int), 0, nx - 1)
        iy = np.clip(((np.asarray(ys) - y0) / res).astype(int), 0, ny - 1)
        return eligible[iy, ix]

    return is_eligible_fast


# ================================================================== V2 (grid + bounded backoff + instrumentation)
CAPACITY_FRACTIONS = [1.0, 0.75, 0.5, 0.25]
MIN_FARM_MW_FLOOR = 30.0          # smallest real project in REAL_FARM_SIZES_MW
TRIES_PER_STEP = 40               # vs. 200 in one block for v1


def generate_farm_v2(target_capacity_mw, turbine_spec, spacing_d, distance_band, angle_deg,
                      rng, existing_shapes, is_eligible_fast, tries_per_step=TRIES_PER_STEP):
    template_name = rng.choice(list(LEASE_TEMPLATES.keys()))
    n_tries_total = 0
    steps_tried = []

    for frac in CAPACITY_FRACTIONS:
        cap = target_capacity_mw * frac
        if cap < MIN_FARM_MW_FLOOR:
            break  # below floor -- stop stepping down, this farm fails
        for _ in range(tries_per_step):
            n_tries_total += 1
            target_xy = placement_from_cell(distance_band, angle_deg, rng)
            rotation_deg = rng.uniform(0, 360)
            shape, pts = build_scaled_farm(cap, turbine_spec, spacing_d, template_name, target_xy, rotation_deg)
            if len(pts) == 0:
                continue
            bx, by = sample_exterior(shape)
            if not is_eligible_fast(bx, by).all():
                continue
            if not is_eligible_fast(pts[:, 0], pts[:, 1]).all():
                continue
            if not is_connection_plausible(np.array([target_xy[0]]), np.array([target_xy[1]]))[0]:
                continue
            if not is_far_enough(shape, existing_shapes):
                continue
            log = dict(target_mw=target_capacity_mw, delivered_mw=cap, fraction_used=frac,
                       n_tries=n_tries_total, success=True, band=distance_band, angle=int(angle_deg),
                       template=template_name)
            return shape, pts, template_name, log
        steps_tried.append(frac)

    log = dict(target_mw=target_capacity_mw, delivered_mw=0.0, fraction_used=0.0,
               n_tries=n_tries_total, success=False, band=distance_band, angle=int(angle_deg),
               template=None)
    return None, None, None, log


def populate_scenario_v2(level, rng, is_eligible_fast, log_rows=None):
    cluster_x, cluster_y = fill_rest_of_cluster(rng)
    delta = REGIONAL_CAPACITY_MW[level] - REGIONAL_CAPACITY_MW["low"]
    farm_budgets = fragment_budget(delta, rng)
    existing_shapes = list(LEASE_TEMPLATES.values())
    results = []
    for cap in farm_budgets:
        name = rng.choice(list(TURBINES.keys()))
        spacing_d = rng.uniform(8.0, 11.0)
        band = DISTANCE_BANDS_KM[rng.integers(len(DISTANCE_BANDS_KM))]
        angle = ANGLE_SECTORS_DEG[rng.integers(len(ANGLE_SECTORS_DEG))]
        shape, pts, template, log = generate_farm_v2(cap, TURBINES[name], spacing_d, band, angle,
                                                       rng, existing_shapes, is_eligible_fast)
        if shape is not None:
            existing_shapes.append(shape)
            results.append(dict(capacity_mw=log["delivered_mw"], turbine=name, shape=shape, pts=pts,
                                 target_mw=cap, fraction_used=log["fraction_used"]))
        if log_rows is not None:
            log_rows.append(log)
    return cluster_x, cluster_y, results


print("Engines loaded.")
