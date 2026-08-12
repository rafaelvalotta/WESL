"""
Test harness comparing v1 (original layout_generation.ipynb logic) against v2
(precomputed eligibility grid + bounded size backoff + instrumentation).

Run from Risk_Investigation/Layout_Generation/ so relative Data paths resolve
the same way the notebook does.
"""
import json
import time
from pathlib import Path

import numpy as np
import yaml
import windIO
from shapely.geometry import Polygon, Point
from shapely import affinity
from scipy.spatial import cKDTree

DATA_DIR = Path("..") / ".." / "Data"

# ---------------------------------------------------------------- 1. Setup (identical to v1)
def load_turbine(fname):
    with open(DATA_DIR / fname) as f:
        d = yaml.safe_load(f)
    return dict(diameter=d["rotor_diameter"], rated_mw=d["performance"]["rated_power"] / 1e6)

TURBINES = {
    "Haliade-X 13MW": load_turbine("haliadex_13mw.yaml"),
    "Haliade-X 15.5MW": load_turbine("haliadex_15.5.yaml"),
    "DTU10MW": load_turbine("dtu10mw.yaml"),
}

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
REAL_FARM_X, REAL_FARM_Y = [], []
for farm in wind_farm_us:
    REAL_FARM_X.append(np.array(farm["layouts"][0]["coordinates"]["x"]))
    REAL_FARM_Y.append(np.array(farm["layouts"][0]["coordinates"]["y"]))


def grid_fill_points(polygon, diameter, spacing_d):
    spacing = spacing_d * diameter
    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    return np.array([(x, y) for x in xs for y in ys if polygon.contains(Point(x, y))])


def fill_rest_of_cluster(rng, verbose=False):
    fill_x, fill_y = list(REAL_FARM_X), list(REAL_FARM_Y)
    for name in BOUNDARY_NAMES:
        if name in REAL_LAYOUT_FARMS:
            continue
        turbine_name = rng.choice(list(TURBINES.keys()))
        spacing_d = rng.uniform(8.0, 11.0)
        pts = grid_fill_points(LEASE_TEMPLATES[name], TURBINES[turbine_name]["diameter"], spacing_d)
        if len(pts):
            fill_x.append(pts[:, 0])
            fill_y.append(pts[:, 1])
    return np.concatenate(fill_x), np.concatenate(fill_y)


cluster_all_x, cluster_all_y = fill_rest_of_cluster(np.random.default_rng(42))

# ---------------------------------------------------------------- Bathymetry + SLA (identical to v1)
import xarray as xr
from pyproj import Transformer

bathy = xr.open_dataset(DATA_DIR / "bathymetry_us.nc")
bathy_xy = np.column_stack([bathy["x"].values, bathy["y"].values])
bathy_depth = bathy["depth"].values
bathy_tree = cKDTree(bathy_xy)
BATHY_X_MIN, BATHY_X_MAX = bathy_xy[:, 0].min(), bathy_xy[:, 0].max()
BATHY_Y_MIN, BATHY_Y_MAX = bathy_xy[:, 1].min(), bathy_xy[:, 1].max()
FIXED_BOTTOM_MAX_DEPTH = 80.0


def depth_at(xs, ys):
    _, idx = bathy_tree.query(np.column_stack([xs, ys]))
    return bathy_depth[idx]


def is_shallow_enough(xs, ys):
    return depth_at(xs, ys) < FIXED_BOTTOM_MAX_DEPTH


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
offset = 0
for line in sla_lines_utm:
    n = len(line)
    tangents[offset:offset + n] = np.gradient(line, axis=0)
    offset += n
sla_tree = cKDTree(sla_all)


def sla_side(xs, ys):
    pts = np.column_stack([xs, ys])
    _, idx = sla_tree.query(pts)
    vec = pts - sla_all[idx]
    cross = tangents[idx, 0] * vec[:, 1] - tangents[idx, 1] * vec[:, 0]
    return np.sign(cross)


FEDERAL_SIGN = np.sign(sla_side(cluster_all_x, cluster_all_y).mean())


def is_federal(xs, ys):
    return sla_side(xs, ys) == FEDERAL_SIGN


SUBSTATIONS = []
for farm in wind_farm_us:
    for sub in farm.get("electrical_substations", []):
        c = sub["electrical_substation"]["coordinates"]
        SUBSTATIONS.append((c["x"][0], c["y"][0]))
SUBSTATIONS = np.array(SUBSTATIONS)
substation_tree = cKDTree(SUBSTATIONS)
MAX_CONNECTION_DISTANCE_M = 70_000


def is_connection_plausible(xs, ys):
    d, _ = substation_tree.query(np.column_stack([xs, ys]))
    return d < MAX_CONNECTION_DISTANCE_M


vineyard = next(f for f in wind_farm_us if f["name"] == "Vineyard Wind")
VINEYARD_CENTER = np.array([
    np.mean(vineyard["layouts"][0]["coordinates"]["x"]),
    np.mean(vineyard["layouts"][0]["coordinates"]["y"]),
])

DISTANCE_BANDS_KM = [(3, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
ANGLE_SECTORS_DEG = np.arange(0, 360, 30)


def placement_from_cell(distance_band, angle_deg, rng):
    r = rng.uniform(distance_band[0], distance_band[1]) * 1000
    theta = np.radians(angle_deg + rng.uniform(-15, 15))
    dx, dy = r * np.sin(theta), r * np.cos(theta)
    return VINEYARD_CENTER[0] + dx, VINEYARD_CENTER[1] + dy


def build_scaled_farm(target_capacity_mw, turbine_spec, spacing_d, template_name, target_xy, rotation_deg):
    n_needed = int(np.ceil(target_capacity_mw / turbine_spec["rated_mw"]))
    target_area_m2 = n_needed * (spacing_d * turbine_spec["diameter"]) ** 2
    template = LEASE_TEMPLATES[template_name]
    scale_factor = np.sqrt(target_area_m2 / template.area)
    shape = affinity.scale(template, xfact=scale_factor, yfact=scale_factor, origin="centroid")
    shape = affinity.rotate(shape, rotation_deg, origin="centroid")
    shape = affinity.translate(shape, xoff=target_xy[0] - shape.centroid.x, yoff=target_xy[1] - shape.centroid.y)
    pts = grid_fill_points(shape, turbine_spec["diameter"], spacing_d)
    return shape, pts


MIN_FARM_SEPARATION_M = 800.0


def is_far_enough(shape, existing_shapes, min_sep=MIN_FARM_SEPARATION_M):
    for other in existing_shapes:
        if shape.intersection(other).area > 0:
            return False
        if shape.distance(other) < min_sep:
            return False
    return True


def sample_exterior(shape, spacing=300.0):
    exterior = shape.exterior
    n = max(int(exterior.length // spacing), 12)
    dists = np.linspace(0, exterior.length, n)
    pts = [exterior.interpolate(d) for d in dists]
    return np.array([p.x for p in pts]), np.array([p.y for p in pts])


REGIONAL_CAPACITY_MW = {"low": 7872.0, "medium": 7872.0 * 2.5, "high": 7872.0 * 4.0}
REAL_FARM_SIZES_MW = [30, 132, 572, 715, 780, 806, 924, 1833, 2080]


def fragment_budget(delta_mw, rng):
    farms, remaining = [], delta_mw
    while remaining > REAL_FARM_SIZES_MW[0] / 2:
        draw = rng.choice(REAL_FARM_SIZES_MW)
        farms.append(min(draw, remaining))
        remaining -= draw
    return farms


print("Setup complete.")
print(f"Eligible bathymetry points: {len(bathy_xy)}")
