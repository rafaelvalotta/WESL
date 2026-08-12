"""Scenario-generation engine -- ported from Layout_Generation/layout_generation_v2.ipynb
(validated: 150km radius, coastline-based connection plausibility, precomputed eligibility
grid with verify-on-accept, staircase-walk boundaries for new farms, bounded size backoff,
per-attempt instrumentation). See that notebook for the full rationale of each choice.

ONE CHANGE from the notebook, needed for the AEP bridge: `fill_rest_of_cluster` there
returns flat (x, y) arrays with no record of which turbine was used on which of the 9
undeveloped-but-real leases -- fine for plotting, useless for AEP (which needs a turbine
type per farm). `populate_scenario` here returns an additional `cluster_farms` field
(one dict per lease, real or infill, with its turbine info) alongside the flat arrays,
so nothing that already used the flat (x, y) output breaks.
"""
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import windIO
from shapely.geometry import Polygon, Point
from shapely import affinity, prepared
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from scipy import ndimage

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "Layout_Generation"))
import staircase

DATA_DIR = _THIS_DIR.parent.parent / "Data"

COLORS = {"Haliade-X 13MW": "#1B6E86", "Haliade-X 15.5MW": "#2E8CA6", "DTU10MW": "#C17A1F"}
TURBINE_YAML_FILES = {
    "Haliade-X 13MW": "haliadex_13mw.yaml",
    "Haliade-X 15.5MW": "haliadex_15.5.yaml",
    "DTU10MW": "dtu10mw.yaml",
}


def _load_turbine_geom(fname):
    with open(DATA_DIR / fname) as f:
        d = yaml.safe_load(f)
    return dict(diameter=d["rotor_diameter"], rated_mw=d["performance"]["rated_power"] / 1e6)


TURBINES = {name: _load_turbine_geom(fname) for name, fname in TURBINE_YAML_FILES.items()}

BOUNDARY_NAMES = [
    "Vineyard Wind", "Revolution Wind", "SouthCoast Wind", "South Fork Wind",
    "Sunrise Wind South", "Sunrise Wind North", "Revolution Wind North",
    "Vineyard Northeast", "New England Wind 1", "New England Wind 2 South",
    "Beacon Wind", "Bay State Wind South", "Bay State Wind North",
]

site_us = windIO.load_yaml(DATA_DIR / "site_us.yaml")
boundary_polygons = site_us["boundaries"]["polygons"]
wind_farm_us = windIO.load_yaml(DATA_DIR / "wind_farm_us.yaml")

LEASE_TEMPLATES = {
    name: Polygon(zip(p["x"], p["y"]))
    for name, p in zip(BOUNDARY_NAMES, boundary_polygons)
}

REAL_LAYOUT_FARMS = {"Vineyard Wind", "Revolution Wind", "South Fork Wind", "Sunrise Wind South"}
_REAL_FARM_NODE_BY_NAME = {f["name"]: f for f in wind_farm_us}
REAL_FARM_X = [np.array(f["layouts"][0]["coordinates"]["x"]) for f in wind_farm_us]
REAL_FARM_Y = [np.array(f["layouts"][0]["coordinates"]["y"]) for f in wind_farm_us]

vineyard = next(f for f in wind_farm_us if f["name"] == "Vineyard Wind")
VINEYARD_CENTER = np.array([
    np.mean(vineyard["layouts"][0]["coordinates"]["x"]),
    np.mean(vineyard["layouts"][0]["coordinates"]["y"]),
])


def grid_fill_points(polygon, diameter, spacing_d):
    spacing = spacing_d * diameter
    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    return np.array([(x, y) for x in xs for y in ys if polygon.contains(Point(x, y))])


def fill_rest_of_cluster(rng):
    """Per-lease breakdown of the existing 13-lease cluster: the 4 real-layout farms
    never move (own real boundary + own real turbine_node, full power/Ct curve already
    in wind_farm_us.yaml); the other 9 leases keep their own real boundary but get a
    fresh random turbine + spacing fill every call. Returns a list of farm dicts, each
    carrying enough to build a proper PyWake turbine later (turbine_node for real farms,
    turbine_name for infill -- see turbines_bridge.py)."""
    farms = []
    for name in BOUNDARY_NAMES:
        if name in REAL_LAYOUT_FARMS:
            node = _REAL_FARM_NODE_BY_NAME[name]
            x = np.array(node["layouts"][0]["coordinates"]["x"])
            y = np.array(node["layouts"][0]["coordinates"]["y"])
            farms.append(dict(name=name, x=x, y=y, n_turbines=len(x), is_synthetic=False,
                               turbine_node=node["turbines"], turbine_name=None))
        else:
            turbine_name = rng.choice(list(TURBINES.keys()))
            spacing_d = rng.uniform(8.0, 11.0)
            pts = grid_fill_points(LEASE_TEMPLATES[name], TURBINES[turbine_name]["diameter"], spacing_d)
            x = pts[:, 0] if len(pts) else np.array([])
            y = pts[:, 1] if len(pts) else np.array([])
            farms.append(dict(name=name, x=x, y=y, n_turbines=len(x), is_synthetic=True,
                               turbine_node=None, turbine_name=turbine_name, spacing_d=spacing_d))
    return farms


# ---------------------------------------------------------------------------
# Bathymetry, federal boundary, coastline-based connection proxy -- unchanged from v2
# ---------------------------------------------------------------------------
import xarray as xr
from pyproj import Transformer

bathy = xr.open_dataset(DATA_DIR / "bathymetry_us.nc")
bathy_xy = np.column_stack([bathy["x"].values, bathy["y"].values])
bathy_depth = bathy["depth"].values
bathy_tree = cKDTree(bathy_xy)
FIXED_BOTTOM_MAX_DEPTH = 80.0


def depth_at(xs, ys):
    _, idx = bathy_tree.query(np.column_stack([xs, ys]))
    return bathy_depth[idx]


def is_shallow_enough(xs, ys):
    return depth_at(xs, ys) < FIXED_BOTTOM_MAX_DEPTH


SLA_PATH = DATA_DIR.parent / "WindFarms" / "US_Cluster" / "Data" / "Federal_State_Boundary_SLA.geojson"
import json
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
MAX_CONNECTION_DISTANCE_M = 70_000


def is_near_coast_live(xs, ys, max_dist_m=MAX_CONNECTION_DISTANCE_M):
    d, _ = _coast_tree.query(np.column_stack([xs, ys]))
    return d < max_dist_m


# ---------------------------------------------------------------------------
# Placement grid, eligibility grid -- unchanged from v2
# ---------------------------------------------------------------------------
DISTANCE_BANDS_KM = [(3, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 75),
                      (75, 90), (90, 110), (110, 130), (130, 150)]
ANGLE_SECTORS_DEG = np.arange(0, 360, 30)


def placement_from_cell(distance_band, angle_deg, rng):
    r = rng.uniform(distance_band[0], distance_band[1]) * 1000
    theta = np.radians(angle_deg + rng.uniform(-15, 15))
    dx, dy = r * np.sin(theta), r * np.cos(theta)
    return VINEYARD_CENTER[0] + dx, VINEYARD_CENTER[1] + dy


def build_eligibility_grid(resolution_m=300.0, radius_km=150.0, margin_m=5_000.0):
    t0 = time.time()
    R = radius_km * 1000
    x0, x1 = VINEYARD_CENTER[0] - R - margin_m, VINEYARD_CENTER[0] + R + margin_m
    y0, y1 = VINEYARD_CENTER[1] - R - margin_m, VINEYARD_CENTER[1] + R + margin_m
    x0, x1 = min(x0, _all_lease_x.min() - margin_m), max(x1, _all_lease_x.max() + margin_m)
    y0, y1 = min(y0, _all_lease_y.min() - margin_m), max(y1, _all_lease_y.max() + margin_m)

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

    return dict(x0=x0, y0=y0, res=resolution_m, nx=nx, ny=ny, eligible=eligible,
                build_time=time.time() - t0)


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
    if not (is_federal(pts[:, 0], pts[:, 1]).all() and is_shallow_enough(pts[:, 0], pts[:, 1]).all()):
        return False
    return True


# Built once at import time -- reused across every trial of the whole Monte Carlo.
ELIGIBILITY_GRID = build_eligibility_grid()
is_eligible_fast = make_grid_lookup(ELIGIBILITY_GRID)


# ---------------------------------------------------------------------------
# New-farm boundary: staircase walk (staircase.py), not a rescaled real lease -- see
# Layout_Generation/layout_generation_v2.ipynb section 6 for why.
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


MIN_FARM_SEPARATION_M = 800.0


def is_far_enough(shape, existing_shapes, min_sep=MIN_FARM_SEPARATION_M):
    for other in existing_shapes:
        if shape.intersection(other).area > 0:
            return False
        if shape.distance(other) < min_sep:
            return False
    return True


CAPACITY_FRACTIONS = [1.0, 0.75, 0.5, 0.25]
MIN_FARM_MW_FLOOR = 30.0
TRIES_PER_STEP = 40


def generate_farm(target_capacity_mw, turbine_spec, spacing_d, distance_band, angle_deg,
                   rng, existing_shapes, tries_per_step=TRIES_PER_STEP):
    n_corners = int(rng.integers(*staircase.N_CORNERS_RANGE))
    raw_shape = staircase._walk_raw(rng, n_corners)
    n_tries_total = 0

    if raw_shape is None:
        log = dict(target_mw=target_capacity_mw, delivered_mw=0.0, fraction_used=0.0,
                   n_tries=0, success=False, band=distance_band, angle=int(angle_deg),
                   template="staircase_walk_failed")
        return None, None, None, log

    for frac in CAPACITY_FRACTIONS:
        cap = target_capacity_mw * frac
        if cap < MIN_FARM_MW_FLOOR:
            break
        for _ in range(tries_per_step):
            n_tries_total += 1
            target_xy = placement_from_cell(distance_band, angle_deg, rng)
            rotation_deg = rng.uniform(0, 360)
            shape, pts = build_staircase_farm(cap, turbine_spec, spacing_d, raw_shape, target_xy, rotation_deg)
            if len(pts) == 0:
                continue
            bx, by = sample_exterior(shape)
            if not is_eligible_fast(bx, by).all():
                continue
            if not is_eligible_fast(pts[:, 0], pts[:, 1]).all():
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
               n_tries=n_tries_total, success=False, band=distance_band, angle=int(angle_deg),
               template=None)
    return None, None, None, log


REGIONAL_CAPACITY_MW = {"low": 7872.0, "medium": 7872.0 * 2.5, "high": 7872.0 * 4.0}
REAL_FARM_SIZES_MW = [30, 132, 572, 715, 780, 806, 924, 1833, 2080]


def fragment_budget(delta_mw, rng):
    farms, remaining = [], delta_mw
    while remaining > REAL_FARM_SIZES_MW[0] / 2:
        draw = rng.choice(REAL_FARM_SIZES_MW)
        farms.append(min(draw, remaining))
        remaining -= draw
    return farms


def populate_scenario(level, rng, log_rows=None):
    """The Monte Carlo interface. Returns:
      cluster_x, cluster_y  -- flat arrays, existing 13-lease cluster (plotting-compatible)
      cluster_farms         -- same cluster, per-lease breakdown incl. turbine_node/turbine_name
      new_farms             -- list of new speculative farm dicts (shape, pts, turbine, capacity_mw)
    log_rows, if passed, gets one dict appended per new-farm *attempt* (not just success).
    """
    cluster_farms = fill_rest_of_cluster(rng)
    cluster_x = np.concatenate([f["x"] for f in cluster_farms if f["n_turbines"] > 0])
    cluster_y = np.concatenate([f["y"] for f in cluster_farms if f["n_turbines"] > 0])

    delta = REGIONAL_CAPACITY_MW[level] - REGIONAL_CAPACITY_MW["low"]
    farm_budgets = fragment_budget(delta, rng)
    existing_shapes = list(LEASE_TEMPLATES.values())
    new_farms = []
    for cap in farm_budgets:
        name = rng.choice(list(TURBINES.keys()))
        spacing_d = rng.uniform(8.0, 11.0)
        band = DISTANCE_BANDS_KM[rng.integers(len(DISTANCE_BANDS_KM))]
        angle = ANGLE_SECTORS_DEG[rng.integers(len(ANGLE_SECTORS_DEG))]
        shape, pts, template, log = generate_farm(cap, TURBINES[name], spacing_d, band, angle, rng, existing_shapes)
        log["turbine"] = name
        if shape is not None:
            existing_shapes.append(shape)
            new_farms.append(dict(capacity_mw=log["delivered_mw"], turbine=name, shape=shape, pts=pts,
                                   target_mw=cap, fraction_used=log["fraction_used"], template=template))
        if log_rows is not None:
            log_rows.append(log)
    return cluster_x, cluster_y, cluster_farms, new_farms
