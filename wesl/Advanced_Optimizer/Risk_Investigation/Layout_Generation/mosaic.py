"""Geometry-only mosaic generator for speculative farm parcels.

Pipeline: dart-throwing Poisson-disk sampling -> weighted-Voronoi
labelling -> gap buffer + cleanup -> polygons. 
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.ops import unary_union

DEFAULT_RING_SAMPLES = 10   # points checked on a candidate disk's edge for mask containment
DEFAULT_NEIGHBOR_K = 12     # nearest seeds considered per cell when labelling


def sample_variable_poisson_disk(eligible, x0, y0, res, r_min, r_max, rng,
                                  ring_samples=DEFAULT_RING_SAMPLES,
                                  max_consecutive_failures=2000):
# Samples non-overlapping disks with random radii across the full domain (handles disconnected masks).
# discards points that collide (dist < r_i + r_j) and stops after max consecutive failures.
    ny, nx = eligible.shape
    x1, y1 = x0 + nx * res, y0 + ny * res

    def eligible_at(x, y):
        ix, iy = int((x - x0) / res), int((y - y0) / res)
        if not (0 <= ix < nx and 0 <= iy < ny):
            return False
        return bool(eligible[iy, ix])

    def disk_fits(cx, cy, r):
        if not eligible_at(cx, cy):
            return False
        angles = np.linspace(0, 2 * np.pi, ring_samples, endpoint=False)
        return all(eligible_at(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles)

    points, radii = [], []
    fails = 0
    while fails < max_consecutive_failures:
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        rc = rng.uniform(r_min, r_max)
        if not disk_fits(cx, cy, rc):
            fails += 1
            continue
        if points:
            pts_arr, rad_arr = np.asarray(points), np.asarray(radii)
            if np.any(np.hypot(pts_arr[:, 0] - cx, pts_arr[:, 1] - cy) < (rad_arr + rc)):
                fails += 1
                continue
        points.append((cx, cy))
        radii.append(rc)
        fails = 0

    if not points:
        return np.empty((0, 2)), np.empty((0,))
    return np.asarray(points), np.asarray(radii)


def label_weighted_voronoi(eligible, x0, y0, res, points, radii, neighbor_k=DEFAULT_NEIGHBOR_K,
                            reach_factor=2.5):
# Assigns each cell to the closest disk boundary (min: dist - r_i)
# Returns an int32 grid of seed IDs 
    label = np.full(eligible.shape, -1, dtype=np.int32)
    if len(points) == 0:
        return label

    iy, ix = np.nonzero(eligible)
    cx = x0 + (ix + 0.5) * res
    cy = y0 + (iy + 0.5) * res

    tree = cKDTree(points)
    k = min(neighbor_k, len(points))
    dists, idx = tree.query(np.column_stack([cx, cy]), k=k)
    if k == 1:
        dists, idx = dists[:, None], idx[:, None]

    weighted = dists - radii[idx]
    owner_pos = np.argmin(weighted, axis=1)
    rows = np.arange(len(cx))
    owner = idx[rows, owner_pos]
    within_reach = dists[rows, owner_pos] <= reach_factor * radii[owner]

    label[iy[within_reach], ix[within_reach]] = owner[within_reach]
    return label


def apply_gap_buffer(label, buffer_cells=1):

    if buffer_cells <= 0: #debug hereeee
        return label
    shifted = label + 1  # ndimage wants background == 0
    if shifted.max() <= 0:
        return label

    out = np.full(label.shape, -1, dtype=label.dtype)
    structure = ndimage.generate_binary_structure(2, 1)
    for lab_id, slc in enumerate(ndimage.find_objects(shifted), start=1):
        if slc is None:
            continue
        sub = shifted[slc] == lab_id
        eroded = ndimage.binary_erosion(sub, structure=structure, iterations=buffer_cells, border_value=0)
        out[slc][eroded] = lab_id - 1
    return out


def clean_labels(label, min_cells=4):

    shifted = label + 1
    if shifted.max() <= 0:
        return label

    out = label.copy()
    for lab_id, slc in enumerate(ndimage.find_objects(shifted), start=1):
        if slc is None:
            continue
        sub = shifted[slc] == lab_id
        cc, n_cc = ndimage.label(sub, structure=np.ones((3, 3)))
        if n_cc == 0:
            continue
        if n_cc > 1:
            sizes = ndimage.sum(sub, cc, index=np.arange(1, n_cc + 1))
            keep = np.argmax(sizes) + 1
            out[slc][sub & (cc != keep)] = -1
        if (out[slc] == (lab_id - 1)).sum() < min_cells:
            out[slc][out[slc] == (lab_id - 1)] = -1
    return out


def _row_runs(iy, ix):
    # Vectorized: given per-cell (row, col) arrays, return (row, x_start, x_end)
    # for each maximal run of horizontally-consecutive columns within a row
    order = np.lexsort((ix, iy))
    iy_s, ix_s = iy[order], ix[order]
    new_run = np.empty(len(iy_s), dtype=bool)
    new_run[0] = True
    new_run[1:] = (iy_s[1:] != iy_s[:-1]) | (ix_s[1:] != ix_s[:-1] + 1)
    first_idx = np.flatnonzero(new_run)
    next_idx = np.r_[first_idx[1:], len(ix_s)]
    return iy_s[first_idx], ix_s[first_idx], ix_s[next_idx - 1]


def polygons_from_labels(label, x0, y0, res):
# Converts each label into a polygon by unioning horizontal cell runs instead of individual cells.


    polys = {}
    for lab_id in np.unique(label):
        if lab_id < 0:
            continue
        iy, ix = np.nonzero(label == lab_id)
        rows, x_starts, x_ends = _row_runs(iy, ix)
        boxes = [box(x0 + xs * res, y0 + r * res, x0 + (xe + 1) * res, y0 + (r + 1) * res)
                 for r, xs, xe in zip(rows, x_starts, x_ends)]
        shape = unary_union(boxes)
        if isinstance(shape, MultiPolygon):
            shape = max(shape.geoms, key=lambda g: g.area)
        if shape.interiors:
            shape = Polygon(shape.exterior)
        polys[int(lab_id)] = shape
    return polys

    # polys = {}
    # for lab_id in np.unique(label):
    #     if lab_id > 0: #--- test or opposite..
    #         continue
    #     iy, ix = np.nonzero(label == lab_id) #label_id --"note for me"--debug this 
    #     rows, x_starts, x_ends = _row_runs(iy, ix)
    #     boxes = [box(x0 + xs * res, y0 + r * res, x0 + (xe + 1) * res, y0 + (r + 1) * res)
    #              for r, xs, xe in zip(rows, x_starts, x_ends)]
    #     shape = unary_union(boxes)
    #     if isinstance(shape, MultiPolygon):
    #         shape = max(shape.geoms, key=lambda g: g.area)
    #     if shape.interiors:
    #         shape = Polygon(shape.exterior)
    #     polys[int(lab_id)] = shape
    # return polys


def generate_mosaic(eligible, x0, y0, res, r_min, r_max, rng, neighbor_k=DEFAULT_NEIGHBOR_K,
                     reach_factor=2.5, buffer_cells=1, max_consecutive_failures=2000):
    # Runs the full pipeline once. Returns dict(points, radii, label, polygons).
    points, radii = sample_variable_poisson_disk(eligible, x0, y0, res, r_min, r_max, rng,
                                                   max_consecutive_failures=max_consecutive_failures)
    label = label_weighted_voronoi(eligible, x0, y0, res, points, radii, neighbor_k=neighbor_k,
                                    reach_factor=reach_factor)
    label = apply_gap_buffer(label, buffer_cells=buffer_cells)
    label = clean_labels(label)
    polygons = polygons_from_labels(label, x0, y0, res)
    return dict(points=points, radii=radii, label=label, polygons=polygons)


def order_by_distance(points, ref_xy):
    # Parcel-seed indices ranked by distance to `ref_xy`, nearest first --
    # truncating this list at N is one way to pick a scenario's active set
    d = np.hypot(points[:, 0] - ref_xy[0], points[:, 1] - ref_xy[1])
    return np.argsort(d)
