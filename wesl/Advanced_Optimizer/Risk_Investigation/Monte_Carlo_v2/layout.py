"""Scenario-generation engine, v2 -- same staircase-walk / eligibility-grid machinery as
Monte_Carlo/layout.py (see that file and Layout_Generation/staircase.py for the full
rationale), rebuilt around a time-phased arrival process instead of a one-shot MW budget.

Two structural changes from v1:
  - self (settings.FOCUS_FARM) is pulled out of the random-infill pool and built by its
    own function, build_self_farm -- mechanically identical to an infill lease today, but
    kept separate because the future regret branch will replace THIS ONE call with an
    optimizer's design, nothing else.
  - new farms are generated one at a time as timeline.sample_arrivals fires, drawing
    their own size from settings.REAL_FARM_SIZES_MW, instead of being pre-planned from a
    fixed regional MW delta.
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import yaml
import windIO
from shapely.geometry import Polygon, Point
from shapely import affinity
from scipy.spatial import cKDTree
import xarray as xr
from pyproj import Transformer

import settings
import timeline

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "Layout_Generation"))
import staircase

DATA_DIR = settings.DATA_DIR


def _load_turbine_geom(fname):
    with open(DATA_DIR / fname) as f:
        d = yaml.safe_load(f)
    return dict(diameter=d["rotor_diameter"], rated_mw=d["performance"]["rated_power"] / 1e6)


TURBINES = {name: _load_turbine_geom(fname) for name, fname in settings.TURBINE_YAML_FILES.items()}

BOUNDARY_NAMES = [
    "Vineyard Wind", "Revolution Wind", "SouthCoast Wind", "South Fork Wind",
    "Sunrise Wind South", "Sunrise Wind North", "Revolution Wind North",
    "Vineyard Northeast", "New England Wind 1", "New England Wind 2 South",
    "Beacon Wind", "Bay State Wind South", "Bay State Wind North",
]

site_us = windIO.load_yaml(DATA_DIR / "site_us.yaml")
boundary_polygons = site_us["boundaries"]["polygons"]
wind_farm_us = windIO.load_yaml(DATA_DIR / "wind_farm_us.yaml")

LEASE_TEMPLATES = {name: Polygon(zip(p["x"], p["y"])) for name, p in zip(BOUNDARY_NAMES, boundary_polygons)}

# Vineyard/Revolution/South Fork/Sunrise South already have a real, built, fixed layout
# -- unaffected by self's move from Vineyard to SouthCoast Wind (§2.6): they're just
# ordinary committed neighbors now.
REAL_LAYOUT_FARMS = {"Vineyard Wind", "Revolution Wind", "South Fork Wind", "Sunrise Wind South"}
_REAL_FARM_NODE_BY_NAME = {f["name"]: f for f in wind_farm_us}
REAL_FARM_X = [np.array(f["layouts"][0]["coordinates"]["x"]) for f in wind_farm_us]
REAL_FARM_Y = [np.array(f["layouts"][0]["coordinates"]["y"]) for f in wind_farm_us]

# Placement geometry (distance bands, eligibility grid) is centered on self now, not
# Vineyard -- speculative neighbors matter insofar as they wake self.
SELF_CENTER = np.array(LEASE_TEMPLATES[settings.FOCUS_FARM].centroid.coords[0])


def grid_fill_points(polygon, diameter, spacing_d):
    spacing = spacing_d * diameter
    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    return np.array([(x, y) for x in xs for y in ys if polygon.contains(Point(x, y))])


def fill_committed_cluster(rng):
    """The 12 leases that are not self -- always active from year 0, no arrival timing
    (PIPELINE_DESIGN_v2.md's "committed neighbors"). 4 have a real fixed layout; the
    other 8 are real leases with no layout yet, so they get a turbine drawn fresh each
    trial at the fixed spacing."""
    farms = []
    for name in BOUNDARY_NAMES:
        if name == settings.FOCUS_FARM:
            continue
        if name in REAL_LAYOUT_FARMS:
            node = _REAL_FARM_NODE_BY_NAME[name]
            x = np.array(node["layouts"][0]["coordinates"]["x"])
            y = np.array(node["layouts"][0]["coordinates"]["y"])
            farms.append(dict(name=name, x=x, y=y, n_turbines=len(x), is_synthetic=False,
                               turbine_node=node["turbines"], turbine_name=None))
        else:
            turbine_name = rng.choice(list(TURBINES.keys()))
            pts = grid_fill_points(LEASE_TEMPLATES[name], TURBINES[turbine_name]["diameter"], settings.SPACING_D)
            x, y = (pts[:, 0], pts[:, 1]) if len(pts) else (np.array([]), np.array([]))
            farms.append(dict(name=name, x=x, y=y, n_turbines=len(x), is_synthetic=True,
                               turbine_node=None, turbine_name=turbine_name, spacing_d=settings.SPACING_D))
    return farms


def build_self_farm(rng):
    """Self. Same mechanism as any committed infill lease above -- kept as its own
    function so the regret branch can later swap this one call for an optimizer."""
    turbine_name = rng.choice(list(TURBINES.keys()))
    pts = grid_fill_points(LEASE_TEMPLATES[settings.FOCUS_FARM], TURBINES[turbine_name]["diameter"], settings.SPACING_D)
    x, y = (pts[:, 0], pts[:, 1]) if len(pts) else (np.array([]), np.array([]))
    return dict(name=settings.FOCUS_FARM, x=x, y=y, n_turbines=len(x), is_synthetic=True,
                turbine_node=None, turbine_name=turbine_name, spacing_d=settings.SPACING_D)


# ---------------------------------------------------------------------------
# Bathymetry, federal boundary, coastline-based connection proxy -- unchanged from v1.
# ---------------------------------------------------------------------------
bathy = xr.open_dataset(DATA_DIR / "bathymetry_us.nc")
bathy_xy = np.column_stack([bathy["x"].values, bathy["y"].values])
bathy_depth = bathy["depth"].values
bathy_tree = cKDTree(bathy_xy)


def depth_at(xs, ys):
    _, idx = bathy_tree.query(np.column_stack([xs, ys]))
    return bathy_depth[idx]


def is_shallow_enough(xs, ys):
    return depth_at(xs, ys) < settings.FIXED_BOTTOM_MAX_DEPTH_M


SLA_PATH = DATA_DIR.parent / "WindFarms" / "US_Cluster" / "Data" / "Federal_State_Boundary_SLA.geojson"
sla_raw = json.load(open(SLA_PATH))["features"][0]["geometry"]["coordinates"]
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -73.0, -68.5, 40.0, 43.0
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)

sla_lines_utm = []
for line in sla_raw:
    pts = np.array(line)
    mask = (pts[:, 0] >= LON_MIN) & (pts[:, 0] <= LON_MAX) & (pts[:, 1] >= LAT_MIN) & (pts[:, 1] <= LAT_MAX)
    if mask.sum() < 2:
        continue
    sub = pts[mask]
    x, y = to_utm.transform(sub[:, 0], sub[:, 1])
    sla_lines_utm.append(np.column_stack([x, y]))

sla_all = np.vstack(sla_lines_utm)
tangents = np.zeros_like(sla_all)
_offset = 0
for line in sla_lines_utm:
    n = len(line)
    tangents[_offset:_offset + n] = np.gradient(line, axis=0)
    _offset += n
sla_tree = cKDTree(sla_all)


def sla_side(xs, ys):
    pts = np.column_stack([xs, ys])
    _, idx = sla_tree.query(pts)
    vec = pts - sla_all[idx]
    cross = tangents[idx, 0] * vec[:, 1] - tangents[idx, 1] * vec[:, 0]
    return np.sign(cross)


_all_lease_x = np.concatenate(REAL_FARM_X)
_all_lease_y = np.concatenate(REAL_FARM_Y)
FEDERAL_SIGN = np.sign(sla_side(_all_lease_x, _all_lease_y).mean())


def is_federal(xs, ys):
    return sla_side(xs, ys) == FEDERAL_SIGN


_coast_mask = (bathy_depth >= 0) & (bathy_depth < 5)
_coast_xy = bathy_xy[_coast_mask]
_coast_tree = cKDTree(_coast_xy)


def is_near_coast_live(xs, ys, max_dist_m=settings.MAX_CONNECTION_DISTANCE_M):
    d, _ = _coast_tree.query(np.column_stack([xs, ys]))
    return d < max_dist_m


# ---------------------------------------------------------------------------
# Placement grid, eligibility grid -- centered on self (SELF_CENTER), radius from settings.
# ---------------------------------------------------------------------------
def placement_from_cell(distance_band, angle_deg, rng):
    r = rng.uniform(distance_band[0], distance_band[1]) * 1000
    theta = np.radians(angle_deg + rng.uniform(-15, 15))
    dx, dy = r * np.sin(theta), r * np.cos(theta)
    return SELF_CENTER[0] + dx, SELF_CENTER[1] + dy


def build_eligibility_grid(resolution_m=300.0, radius_km=settings.RADIUS_KM, margin_m=5_000.0):
    t0 = time.time()
    R = radius_km * 1000
    x0, x1 = SELF_CENTER[0] - R - margin_m, SELF_CENTER[0] + R + margin_m
    y0, y1 = SELF_CENTER[1] - R - margin_m, SELF_CENTER[1] + R + margin_m
    x0, x1 = min(x0, _all_lease_x.min() - margin_m), max(x1, _all_lease_x.max() + margin_m)
    y0, y1 = min(y0, _all_lease_y.min() - margin_m), max(y1, _all_lease_y.max() + margin_m)

    nx = int((x1 - x0) / resolution_m) + 1
    ny = int((y1 - y0) / resolution_m) + 1
    gx = x0 + np.arange(nx) * resolution_m
    gy = y0 + np.arange(ny) * resolution_m
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    GXf, GYf = GX.ravel(), GY.ravel()

    eligible = (is_federal(GXf, GYf) & is_shallow_enough(GXf, GYf) & is_near_coast_live(GXf, GYf)).reshape(GX.shape)
    return dict(x0=x0, y0=y0, res=resolution_m, nx=nx, ny=ny, eligible=eligible, build_time=time.time() - t0)


def make_grid_lookup(grid):
    x0, y0, res, nx, ny, eligible = grid["x0"], grid["y0"], grid["res"], grid["nx"], grid["ny"], grid["eligible"]

    def is_eligible_fast(xs, ys):
        ix = np.clip(((np.asarray(xs) - x0) / res).astype(int), 0, nx - 1)
        iy = np.clip(((np.asarray(ys) - y0) / res).astype(int), 0, ny - 1)
        return eligible[iy, ix]

    return is_eligible_fast


def verify_exact(bx, by, pts):
    if not (is_federal(bx, by).all() and is_shallow_enough(bx, by).all()
            and is_near_coast_live(np.array([bx.mean()]), np.array([by.mean()]))[0]):
        return False
    return is_federal(pts[:, 0], pts[:, 1]).all() and is_shallow_enough(pts[:, 0], pts[:, 1]).all()


ELIGIBILITY_GRID = build_eligibility_grid()  # built once at import time, reused every trial
is_eligible_fast = make_grid_lookup(ELIGIBILITY_GRID)


# ---------------------------------------------------------------------------
# New-farm boundary: staircase walk (see staircase.py), not a rescaled real lease.
# ---------------------------------------------------------------------------
def build_staircase_farm(target_capacity_mw, turbine_spec, spacing_d, raw_shape_cells, target_xy, rotation_deg):
    n_needed = int(np.ceil(target_capacity_mw / turbine_spec["rated_mw"]))
    target_area_m2 = n_needed * (spacing_d * turbine_spec["diameter"]) ** 2

    cell_side_m = np.sqrt(staircase.CELL_AREA_KM2) * 1000.0
    raw_m = affinity.scale(raw_shape_cells, xfact=cell_side_m, yfact=cell_side_m, origin=(0, 0))
    scale = np.sqrt(target_area_m2 / raw_m.area)
    shape = affinity.scale(raw_m, xfact=scale, yfact=scale, origin="centroid")
    shape = affinity.rotate(shape, rotation_deg, origin="centroid")
    shape = affinity.translate(shape, xoff=target_xy[0] - shape.centroid.x, yoff=target_xy[1] - shape.centroid.y)

    pts = grid_fill_points(shape, turbine_spec["diameter"], spacing_d)
    return shape, pts


def sample_exterior(shape, spacing=300.0):
    exterior = shape.exterior
    n = max(int(exterior.length // spacing), 12)
    dists = np.linspace(0, exterior.length, n)
    pts = [exterior.interpolate(d) for d in dists]
    return np.array([p.x for p in pts]), np.array([p.y for p in pts])


def is_far_enough(shape, existing_shapes, min_sep=settings.MIN_FARM_SEPARATION_M):
    for other in existing_shapes:
        if shape.intersection(other).area > 0 or shape.distance(other) < min_sep:
            return False
    return True


def generate_farm(target_capacity_mw, turbine_spec, spacing_d, distance_band, angle_deg,
                   rng, existing_shapes, tries_per_step=settings.TRIES_PER_STEP):
    n_corners = int(rng.integers(*staircase.N_CORNERS_RANGE))
    raw_shape = staircase._walk_raw(rng, n_corners)
    n_tries_total = 0

    if raw_shape is None:
        log = dict(target_mw=target_capacity_mw, delivered_mw=0.0, fraction_used=0.0,
                   n_tries=0, success=False, band=distance_band, angle=int(angle_deg),
                   template="staircase_walk_failed")
        return None, None, None, log

    for frac in settings.CAPACITY_FRACTIONS:
        cap = target_capacity_mw * frac
        if cap < settings.MIN_FARM_MW_FLOOR:
            break
        for _ in range(tries_per_step):
            n_tries_total += 1
            target_xy = placement_from_cell(distance_band, angle_deg, rng)
            rotation_deg = rng.uniform(0, 360)
            shape, pts = build_staircase_farm(cap, turbine_spec, spacing_d, raw_shape, target_xy, rotation_deg)
            if len(pts) == 0:
                continue
            bx, by = sample_exterior(shape)
            if not is_eligible_fast(bx, by).all() or not is_eligible_fast(pts[:, 0], pts[:, 1]).all():
                continue
            if not is_far_enough(shape, existing_shapes):
                continue
            if not verify_exact(bx, by, pts):
                continue
            log = dict(target_mw=target_capacity_mw, delivered_mw=cap, fraction_used=frac,
                       n_tries=n_tries_total, success=True, band=distance_band, angle=int(angle_deg),
                       template=f"staircase_{n_corners}corners")
            return shape, pts, log["template"], log

    log = dict(target_mw=target_capacity_mw, delivered_mw=0.0, fraction_used=0.0,
               n_tries=n_tries_total, success=False, band=distance_band, angle=int(angle_deg), template=None)
    return None, None, None, log


def populate_scenario(tier, rng, horizon_years=None, log_rows=None):
    """The v2 Monte Carlo interface. Returns:
      cluster_x, cluster_y  -- flat arrays, the 13-lease cluster incl. self
      cluster_farms         -- same cluster, per-lease breakdown (12 committed + self)
      new_farms              -- speculative neighbors that arrived within the horizon,
                                 each carrying its own arrival_year
    log_rows, if passed, gets one dict per arrival ATTEMPT (not just successes)."""
    horizon_years = horizon_years or settings.HORIZON_YEARS
    cluster_farms = fill_committed_cluster(rng) + [build_self_farm(rng)]
    cluster_x = np.concatenate([f["x"] for f in cluster_farms if f["n_turbines"] > 0])
    cluster_y = np.concatenate([f["y"] for f in cluster_farms if f["n_turbines"] > 0])

    arrival_years = timeline.sample_arrivals(rng, settings.TIER_LAMBDA[tier], horizon_years)
    existing_shapes = list(LEASE_TEMPLATES.values())  # every real lease blocks new placements
    new_farms = []
    for arrival_year in arrival_years:
        cap = float(rng.choice(settings.REAL_FARM_SIZES_MW))
        name = rng.choice(list(TURBINES.keys()))
        band = settings.DISTANCE_BANDS_KM[rng.integers(len(settings.DISTANCE_BANDS_KM))]
        angle = settings.ANGLE_SECTORS_DEG[rng.integers(len(settings.ANGLE_SECTORS_DEG))]
        shape, pts, template, log = generate_farm(cap, TURBINES[name], settings.SPACING_D, band, angle, rng, existing_shapes)
        log["turbine"] = name
        log["arrival_year"] = round(float(arrival_year), 3)
        if shape is not None:
            existing_shapes.append(shape)
            new_farms.append(dict(arrival_year=float(arrival_year), capacity_mw=log["delivered_mw"],
                                   turbine=name, shape=shape, pts=pts, template=template))
        if log_rows is not None:
            log_rows.append(log)
    return cluster_x, cluster_y, cluster_farms, new_farms
