import json
import sys
from pathlib import Path

import numpy as np
import windIO
import xarray as xr
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.ops import unary_union
from shapely.vectorized import contains as shapely_contains

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "Layout_Generation"))
import mosaic

import settings
import timeline
import turbines
import spacing
import grid_fill
import scenarios
from cluster import LEASE_TEMPLATES

SLA_PATH = settings.DATA_DIR.parent / "WindFarms" / "US_Cluster" / "Data" / "Federal_State_Boundary_SLA.geojson"

RESOLUTION_M = 300.0
RADIUS_KM = 100.0
# How far the grid reaches along the cluster's axis. Not an eligibility limit --
# just needs to cover the real federal/bathymetry boundary so that boundary is
# what actually cuts off the corridor (see _load_base_grid), not the grid itself.
AOI_LENGTH_HALF_KM = 260.0
FIXED_BOTTOM_MAX_DEPTH_M = 60.0
BATHY_SMOOTH_M = 2_000.0
CLUSTER_BUFFER_M = 1_000.0
BUFFER_CELLS = 3
FRAGMENT_AREA_KM2 = 30.0


def _cluster_axis_angle():
    # Orientation (radians from +x/east, CCW) of the 13-lease cluster's own axis,
    # found via PCA on the lease centroids. Used to rotate the AOI so it follows
    # the coastline instead of staying aligned to UTM easting/northing.
    centroids = np.array([p.centroid.coords[0] for p in LEASE_TEMPLATES.values()])
    centered = centroids - centroids.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    principal = eigvecs[:, np.argmax(eigvals)]
    return np.arctan2(principal[1], principal[0])


def _load_base_grid():
    # Scenario-independent part: bathymetry + federal/state split + radius clip.
    # Computed once at import. Which farms count as "occupied" is scenario-specific
    # and handled separately in eligible_for(), not here.

    # Anchor the AOI on self's own lease centroid. Self has no real turbine layout
    # yet, so we use its lease polygon's centroid instead of turbine positions.
    aoi_center = np.array(LEASE_TEMPLATES[scenarios.SELF_FARM].centroid.coords[0])

    bathy = xr.open_dataset(settings.DATA_DIR / "bathymetry_us.nc")
    bathy_xy = np.column_stack([bathy["x"].values, bathy["y"].values])
    bathy_depth = bathy["depth"].values
    bathy_tree = cKDTree(bathy_xy)

    sla_raw = json.load(open(SLA_PATH))["features"][0]["geometry"]["coordinates"]
    lon_min, lon_max, lat_min, lat_max = -73.0, -68.5, 40.0, 43.0
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
    sla_lines_utm = []
    for line in sla_raw:
        pts = np.array(line)
        mask = (pts[:, 0] >= lon_min) & (pts[:, 0] <= lon_max) & (pts[:, 1] >= lat_min) & (pts[:, 1] <= lat_max)
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

    federal_sign = np.sign(sla_side(np.array([aoi_center[0]]), np.array([aoi_center[1]])).mean())

    # AOI shape is a corridor, not a circle: within RADIUS_KM of a line through
    # self (the cluster's own axis), with no cutoff along that line -- the real
    # federal/state boundary and bathymetry are what end the corridor lengthwise
    # (see `within_corridor` below, it only constrains the perpendicular side).
    R = RADIUS_KM * 1000
    U_HALF = AOI_LENGTH_HALF_KM * 1000
    theta = _cluster_axis_angle()
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    margin = 5_000.0
    all_x = np.concatenate([np.array(p.exterior.coords.xy[0]) for p in LEASE_TEMPLATES.values()])
    all_y = np.concatenate([np.array(p.exterior.coords.xy[1]) for p in LEASE_TEMPLATES.values()])
    # bounding box must contain the rotated corridor's 4 corners (long in u,
    # narrow in v), not just center +/- R
    corners_local = np.array([[U_HALF, R], [U_HALF, -R], [-U_HALF, R], [-U_HALF, -R]])
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_world = corners_local @ rot.T + aoi_center
    x0 = min(corners_world[:, 0].min() - margin, all_x.min() - margin)
    x1 = max(corners_world[:, 0].max() + margin, all_x.max() + margin)
    y0 = min(corners_world[:, 1].min() - margin, all_y.min() - margin)
    y1 = max(corners_world[:, 1].max() + margin, all_y.max() + margin)
    nx = int((x1 - x0) / RESOLUTION_M) + 1
    ny = int((y1 - y0) / RESOLUTION_M) + 1
    gx = x0 + np.arange(nx) * RESOLUTION_M
    gy = y0 + np.arange(ny) * RESOLUTION_M
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    GXf, GYf = GX.ravel(), GY.ravel()

    _, bidx = bathy_tree.query(np.column_stack([GXf, GYf]))
    depth_grid = bathy_depth[bidx].reshape(GX.shape)
    depth_grid_smooth = ndimage.gaussian_filter(depth_grid, sigma=BATHY_SMOOTH_M / RESOLUTION_M)
    shallow = depth_grid_smooth < FIXED_BOTTOM_MAX_DEPTH_M
    fed = (sla_side(GXf, GYf) == federal_sign).reshape(GX.shape)
    dx, dy = GX - aoi_center[0], GY - aoi_center[1]
    v = -dx * sin_t + dy * cos_t   # coordinate perpendicular to the cluster axis
    within_corridor = np.abs(v) < R   # NO constraint along the axis -- fed & shallow bound it naturally
    eligible = fed & shallow & within_corridor

    return dict(eligible=eligible, GX=GX, GY=GY, x0=x0, y0=y0, res=RESOLUTION_M,
                aoi_center=aoi_center, aoi_axis_angle=theta, aoi_half_width_m=R)


def _radius_range(lease_templates, min_area_km2=FRAGMENT_AREA_KM2):
    areas = np.array([p.area / 1e6 for p in lease_templates.values()])
    r_eq_km = np.sqrt(areas[areas >= min_area_km2] / np.pi)
    return r_eq_km.min() * 1000.0, r_eq_km.max() * 1000.0


BASE_GRID = _load_base_grid()
R_MIN_M, R_MAX_M = _radius_range(LEASE_TEMPLATES)
_eligibility_cache = {}  # scenario.key -> eligible bool grid, computed once per scenario per process


def eligible_for(scenario):
    # Base eligibility grid minus whatever this scenario already treats as occupied.
    # Memoized per scenario.key so it's computed once, not on every trial.
    if scenario.key not in _eligibility_cache:
        occupied = unary_union([LEASE_TEMPLATES[n] for n in scenario.fixed_farm_names]).buffer(CLUSTER_BUFFER_M)
        not_occupied = ~shapely_contains(occupied, BASE_GRID["GX"], BASE_GRID["GY"])
        _eligibility_cache[scenario.key] = BASE_GRID["eligible"] & not_occupied
    return _eligibility_cache[scenario.key]


def generate_scenario(rng, scenario, fixed_capacity_mw=0.0):
    # Walks the mosaic's candidate parcels in random order, filling each one with
    # its own density-derived spacing, and stops once total capacity (fixed +
    # speculative) reaches settings.SPECULATIVE_TARGET_MW. fixed_capacity_mw lets
    # the caller pass in what's already committed (real-layout farms + self) so
    # the target is the scenario's TOTAL buildout, not speculative capacity added
    # on top -- otherwise scenario_1 and scenario_2 would land at very different
    # totals despite both aiming for the same ~30GW buildout. Defaults to 0.0 for
    # standalone use. A small overshoot past the target from the last parcel is
    # expected and left as-is, not trimmed. See SCENARIOS.md for the full reasoning.
    eligible = eligible_for(scenario)
    out = mosaic.generate_mosaic(eligible, BASE_GRID["x0"], BASE_GRID["y0"],
                                  BASE_GRID["res"], R_MIN_M, R_MAX_M, rng, buffer_cells=BUFFER_CELLS)
    parcel_ids = list(out["polygons"].keys())
    order = rng.permutation(len(parcel_ids))

    farms = []
    cumulative_mw = 0.0
    for idx in order:
        pid = parcel_ids[idx]
        polygon = out["polygons"][pid]

        year = float(timeline.draw_arrival_years(rng, 1)[0])
        turbine = turbines.draw(rng, year)
        diameter_m = turbines.diameter(turbine)
        rating_mw = turbines.rated_mw(turbine)
        spacing_d, rho_target, was_clipped = spacing.from_density(rng, rating_mw, diameter_m)

        pts = grid_fill.fill(polygon, diameter_m, spacing_d)
        if len(pts) == 0:
            continue  # too small/awkward a parcel for this spacing -- skip, try the next candidate

        farms.append(dict(name=f"speculative_{int(pid)}", is_self=False, has_real_layout=False,
                           arrival_year=year, turbine=turbine, spacing_d=spacing_d,
                           rho_target_mw_km2=rho_target, spacing_clipped=was_clipped,
                           polygon=polygon, x=pts[:, 0], y=pts[:, 1]))
        cumulative_mw += len(pts) * rating_mw

        if fixed_capacity_mw + cumulative_mw >= settings.SPECULATIVE_TARGET_MW:
            break

    return farms
