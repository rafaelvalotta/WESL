"""Rectilinear ('staircase') boundary generator for speculative farms.

Replaces Generator B's old approach -- pick a real lease polygon and
isotropically rescale it (`build_scaled_farm` in layout_generation.ipynb) --
with a shape generated natively on the regional grid instead of distorted
from one. Three methods were built and measured against the real data in
this cluster, in order, each superseded by the next for a specific,
measured reason -- kept here, in order, as the record of why:

1. `grow_eden_boundary` -- a random polyomino grown cell-by-cell (Eden
   growth). Gets the real 0/90 edge alignment and the target area right by
   construction, but "solidity" (area / convex-hull area, 1.0 = no
   concavity) comes out 0.65-0.68 against a real mean of 0.79-0.81 -- Eden
   growth is known to produce dendritic fingering all over the perimeter,
   not the mostly-solid-with-a-jagged-edge look real leases actually have.
   Biasing it to elongate (`elongation`) fixes aspect ratio but makes
   solidity worse (0.648), and does nothing for edge coherence (see 2).

2. `grow_smooth_rasterize_boundary` -- build a smooth, mostly-convex
   outline first (1-3 corner-overlapped rectangles), then rasterize it onto
   the block grid. Fixes solidity (0.853) and aspect ratio (1.32) at once,
   because the body is convex-ish by construction. But with the outline's
   own edges already axis-aligned, rasterizing them creates no zigzag at
   all in many draws (a perfectly clean rectangle, `staircase_run_lengths`
   still reads as "coherent" because a straight edge trivially never
   reverses) -- visually too clean, not what real leases look like.

3. `grow_staircase_boundary` (current) -- generate the boundary as a direct
   walk of K real corners (K ~ the range actually counted on a real BOEM
   map: Attentive Energy ~7, Community Offshore Wind ~8, Atlantic Shores /
   Leading Light ~4), where each wall between two corners is independently
   either PLAIN (one axis direction, bounded length -- never tilted, per
   real data: every edge is exactly 0 or 90 deg) or STAIRCASE (a genuine
   diagonal displacement, drawn as monotonic alternating 0/90 unit steps,
   never backtracking -- this is what actually produces a coherent,
   one-direction-drifting zigzag, not rasterizing an already-straight line).
   Every draw must pass six hard constraints (reject-and-retry, not a
   post-hoc filter -- see `_walk_raw`), each one a direct response to a
   specific real-vs-generated mismatch found by visual review:
     - no single wall may dominate the perimeter (found: an unconstrained
       auto-closing wall would sometimes draw almost the whole shape)
     - bounding-box aspect ratio in the real range (1.00-1.83)
     - at least some minimum share of the perimeter must be staircase
       (found: an all-plain draw looks like Method 2's too-clean problem)
     - `buffer(-1 wall)` must stay one connected piece (found: a deep
       concave turn can nearly pinch the shape into two lobes)
     - `buffer(-1 cell).area / area` must stay above ~0.50 (found: a
       shape can have a locally thin winding corridor even when its
       overall bounding box looks fine -- aspect ratio alone misses this)
     - the walk must actually close
   Measured on the real 10-lease sample: solidity 0.81-0.88 (real
   0.79-0.81), aspect ratio 1.29-1.36 (real 1.28), staircase run-length
   16-23 (real 12-23 for well-behaved leases). All three metrics, plus
   the two new geometric checks, hold at once -- and every check is a
   measured real/generated gap translated into a number, not a look/feel
   tuning pass. See `Layout_Generation/check_walk_boundary.py` in the repo
   history for the full iterative calibration (four rounds of visual
   review against labeled examples).

Why rectilinear at all: measuring the 13 real lease boundaries in this
cluster shows their edges sit at 0/90 deg (mod 90) ~100% of the time,
length-weighted -- they are not "mostly rectilinear", they are exactly
rectilinear, no exceptions. A sample of 11 UK / Irish Sea leases measured
the same way averages ~56% -- organic, diagonal-heavy boundaries. This
"staircase" look is specific to this region (almost certainly inherited
from the BOEM OCS block grid), not a universal offshore-lease trait.

Cell size: no official BOEM block dimension is asserted here (not verified
against a primary source). CELL_AREA_KM2 defaults to (1 nm)^2 = 3.43 km2 --
the same regional reference length already used for turbine spacing
(SYNTHETIC_SPACING_M in config.py) and close to the best-fit "area quantum"
found by minimizing, over candidate block areas b, the residual of
(real_lease_area / b) to the nearest integer across the 10 larger real
leases (best fit ~3.5-3.9 km2, mean residual 0.21 blocks). One shared
number, not a new constant. `grow_staircase_boundary` walks on this cell
grid and is scaled (uniform, from centroid) to the exact requested area
afterwards -- a similarity transform, so it changes nothing about the
validated ratios (solidity, aspect ratio, run counts, corridor clearance
are all scale-invariant).
"""
import numpy as np
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

CELL_AREA_KM2 = 1.852 ** 2  # (1 nm)^2 -- same regional reference as turbine spacing
DEFAULT_ELONGATION = 0.35  # grow_eden_boundary only -- matches real aspect-ratio distribution
N_RECTS_WEIGHTS = (0.5, 0.30, 0.20)  # grow_smooth_rasterize_boundary: P(1 rect), P(2), P(3)
CORNER_OVERLAP_RANGE = (0.05, 0.30)  # grow_smooth_rasterize_boundary: corner-share fraction

# grow_staircase_boundary (walk method) -- calibrated in check_walk_boundary.py, see docstring
N_CORNERS_RANGE = (6, 11)         # counted on a real map: ~4-11; 4 alone was rejected in review
WALL_SEG_RANGE = (2, 8)           # wall length, in cells
STAIR_RATIO_RANGE = (2, 4)        # staircase forward:lateral step ratio
STAIR_PROB = 0.4                  # P(a given wall is staircase, not plain)
MAX_WALL_FRACTION = 0.30          # no wall (incl. the closing one) may exceed this share of perimeter
ASPECT_RANGE = (1.0, 1.9)         # bounding-box aspect ratio -- real: 1.00-1.83
MIN_STAIR_FRACTION = 0.15         # perimeter share that must be staircase -- no all-plain draws
PINCH_CLEARANCE = 0.75            # cell units -- buffer(-x) must stay one piece
CORRIDOR_EROSION = 1.0            # cell units -- buffer(-x).area / area must clear MIN_AREA_RETAINED
MIN_AREA_RETAINED = 0.50          # calibrated midpoint: labeled-bad 0.33-0.49, labeled-good 0.55-0.73

HEADINGS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, N, W, S


def solidity(shape):
    """area / convex-hull area -- 1.0 = fully convex, lower = more concave 'legs'."""
    return shape.area / shape.convex_hull.area


def staircase_run_lengths(shape, min_edge_len=1.0):
    """How many consecutive boundary edges keep drifting in a compatible pair
    of directions (e.g. only E/N, never backtracking to W or S) before a real
    reversal happens -- one run = one coherent stretch of staircase tracing a
    single underlying line. Real leases: long runs (Vineyard mean 12.2,
    SouthCoast 21.6). `min_edge_len` filters degenerate zero-length edges;
    lower it (e.g. 0.5) for shapes still in raw cell-unit coordinates, where
    every edge is exactly length 1 and the 1.0 default would exclude all of
    them.
    """
    x, y = np.array(shape.exterior.coords.xy[0]), np.array(shape.exterior.coords.xy[1])
    dx, dy = np.diff(x), np.diff(y)
    length = np.hypot(dx, dy)
    mask = length > min_edge_len
    dx, dy = dx[mask], dy[mask]
    dirs = [("E" if a > 0 else "W") if abs(a) > abs(b) else ("N" if b > 0 else "S") for a, b in zip(dx, dy)]
    if not dirs:
        return [0]

    opposite = {"E": "W", "W": "E", "N": "S", "S": "N"}
    runs, run_len, active_pair = [], 1, {dirs[0]}
    for d in dirs[1:]:
        if d in active_pair or (len(active_pair) < 2 and d != opposite[next(iter(active_pair))]):
            active_pair.add(d)
            run_len += 1
        else:
            runs.append(run_len)
            run_len, active_pair = 1, {d}
    runs.append(run_len)
    return runs


def bbox_aspect_ratio(shape):
    xmin, ymin, xmax, ymax = shape.bounds
    w, h = xmax - xmin, ymax - ymin
    return max(w, h) / min(w, h)


# ---------------------------------------------------------------------------
# Method 3 (current): direct corner walk, PLAIN/STAIRCASE walls, 6 constraints
# ---------------------------------------------------------------------------

def _draw_wall(pos, kind, heading_idx, length, stair_ratio):
    """Unit-step points for one wall from `pos` (exclusive). PLAIN: `length`
    steps in `heading_idx`'s direction. STAIRCASE: alternates `heading_idx`
    and the direction 90 deg to its left, `stair_ratio`:1, monotonic (never
    backtracks either component -- no tilt, only 0/90 micro-steps)."""
    pts = []
    x, y = pos
    if kind == "plain":
        dx, dy = HEADINGS[heading_idx]
        for _ in range(length):
            x, y = x + dx, y + dy
            pts.append((x, y))
    else:
        dx, dy = HEADINGS[heading_idx]
        lx, ly = HEADINGS[(heading_idx + 1) % 4]
        for _ in range(length):
            for _ in range(stair_ratio):
                x, y = x + dx, y + dy
                pts.append((x, y))
            x, y = x + lx, y + ly
            pts.append((x, y))
    return pts, (x, y)


def _is_single_piece(poly, clearance):
    shrunk = poly.buffer(-clearance)
    if shrunk.is_empty or isinstance(shrunk, MultiPolygon):
        return False
    return True


def _no_thin_corridor(poly, erosion, min_retained):
    """A locally narrow corridor can exist even when the overall bounding-box
    aspect ratio looks fine -- erosion collapses it (loses most of its area)
    even though the rest of the shape survives, so the area ratio before/after
    a moderate shrink is a more direct test than aspect ratio alone."""
    shrunk = poly.buffer(-erosion)
    if shrunk.is_empty:
        return False
    return (shrunk.area / poly.area) >= min_retained


def _walk_raw(rng, n_corners, seg_range=WALL_SEG_RANGE, stair_ratio_range=STAIR_RATIO_RANGE,
              stair_prob=STAIR_PROB, max_wall_fraction=MAX_WALL_FRACTION, aspect_range=ASPECT_RANGE,
              min_stair_fraction=MIN_STAIR_FRACTION, pinch_clearance=PINCH_CLEARANCE,
              corridor_erosion=CORRIDOR_EROSION, min_area_retained=MIN_AREA_RETAINED):
    """K walls with mostly-consistent left turns (a few right turns for a
    concave notch), the last wall forced to close the loop back to the
    start. Returns a polygon in raw integer cell-unit coordinates (not yet
    scaled to a target area), or None if no draw satisfied all 6
    constraints within the retry budget."""
    for _attempt in range(2000):
        heading_idx = 0
        pos = (0, 0)
        pts = [pos]
        wall_kinds, wall_lengths = [], []
        for _ in range(n_corners - 1):
            kind = "staircase" if rng.random() < stair_prob else "plain"
            length = int(rng.integers(*seg_range))
            stair_ratio = int(rng.integers(*stair_ratio_range))
            new_pts, pos = _draw_wall(pos, kind, heading_idx, length, stair_ratio)
            pts.extend(new_pts)
            wall_kinds.append(kind)
            wall_lengths.append(len(new_pts))
            turn = rng.choice([1, 0, -1], p=[0.55, 0.25, 0.20])  # mostly left turn (convex-ish)
            heading_idx = (heading_idx + turn) % 4

        cx, cy = pts[0][0] - pos[0], pts[0][1] - pos[1]
        if cx == 0 and cy == 0:
            continue
        closing_len = abs(cx) + abs(cy)
        total_perimeter = sum(wall_lengths) + closing_len
        if closing_len > max_wall_fraction * total_perimeter:
            continue
        if max(wall_lengths, default=0) > max_wall_fraction * total_perimeter:
            continue

        if cx == 0 or cy == 0:
            steps = abs(cx) + abs(cy)
            dxu, dyu = (np.sign(cx), 0) if cx != 0 else (0, np.sign(cy))
            x, y = pos
            for _ in range(steps):
                x, y = x + dxu, y + dyu
                pts.append((x, y))
            wall_kinds.append("plain")
            wall_lengths.append(steps)
        else:
            steps = max(abs(cx), abs(cy))
            dxu, dyu = np.sign(cx), np.sign(cy)
            x, y = pos
            ex, ey = 0, 0
            for _ in range(steps):
                ex += abs(cx); ey += abs(cy)
                if ex >= ey:
                    x += dxu; ex -= steps
                else:
                    y += dyu; ey -= steps
                pts.append((x, y))
            pts[-1] = (pts[0][0], pts[0][1])
            wall_kinds.append("staircase")
            wall_lengths.append(steps)

        poly = Polygon(pts)
        if not (poly.is_valid and not poly.is_empty and poly.area > 0):
            continue

        if not (aspect_range[0] <= bbox_aspect_ratio(poly) <= aspect_range[1]):
            continue

        stair_len = sum(l for k, l in zip(wall_kinds, wall_lengths) if k == "staircase")
        if stair_len / total_perimeter < min_stair_fraction:
            continue

        if not _is_single_piece(poly, pinch_clearance):
            continue

        if not _no_thin_corridor(poly, corridor_erosion, min_area_retained):
            continue

        return poly

    return None


def grow_staircase_boundary(target_area_km2, rng, cell_area_km2=CELL_AREA_KM2, theta1_deg=0.0,
                             n_corners=None, n_corners_range=N_CORNERS_RANGE):
    """One random rectilinear boundary with area close to `target_area_km2`,
    via the walk method (see module docstring). Returns
    (shape, n_corners, achieved_area_km2), or (None, None, None) if no valid
    draw was found in the retry budget (should be rare -- all 5 test
    configurations accepted 150/150 draws during calibration).

    `theta1_deg` rotates the whole result rigidly to the regionally observed
    alignment (e.g. 0 deg for this cluster) -- applied once at the end, not
    sampled per wall. The walk runs on the abstract cell grid first (so the
    6 constraints are checked at a fixed, calibrated scale), then the whole
    shape is scaled uniformly from its centroid to hit `target_area_km2`
    exactly -- a similarity transform, so it does not change solidity,
    aspect ratio, run-length, or corridor clearance; only absolute size.
    """
    if n_corners is None:
        n_corners = int(rng.integers(*n_corners_range))

    cell_side_m = np.sqrt(cell_area_km2) * 1000.0
    raw = _walk_raw(rng, n_corners)
    if raw is None:
        return None, None, None
    raw_m = affinity.scale(raw, xfact=cell_side_m, yfact=cell_side_m, origin=(0, 0))

    target_area_m2 = target_area_km2 * 1e6
    scale = np.sqrt(target_area_m2 / raw_m.area)
    shape = affinity.scale(raw_m, xfact=scale, yfact=scale, origin="centroid")
    if theta1_deg != 0.0:
        shape = affinity.rotate(shape, theta1_deg, origin="centroid")

    return shape, n_corners, shape.area / 1e6


# ---------------------------------------------------------------------------
# Method 2 (superseded by the walk method above): smooth outline, rasterized
# ---------------------------------------------------------------------------

def _rasterize(poly, cell_side_m, threshold=0.5):
    """Union of grid cells more than `threshold` covered by `poly`."""
    xmin, ymin, xmax, ymax = poly.bounds
    i0, i1 = int(np.floor(xmin / cell_side_m)) - 1, int(np.ceil(xmax / cell_side_m)) + 1
    j0, j1 = int(np.floor(ymin / cell_side_m)) - 1, int(np.ceil(ymax / cell_side_m)) + 1
    cells = []
    for i in range(i0, i1):
        for j in range(j0, j1):
            cell = box(i * cell_side_m, j * cell_side_m, (i + 1) * cell_side_m, (j + 1) * cell_side_m)
            inter = cell.intersection(poly)
            if not inter.is_empty and inter.area / cell.area > threshold:
                cells.append(cell)
    return unary_union(cells) if cells else None


def _sample_smooth_outline(target_area_m2, rng, n_rects_weights=N_RECTS_WEIGHTS,
                            corner_overlap_range=CORNER_OVERLAP_RANGE):
    """1-3 axis-aligned rectangles, the 2nd/3rd corner-anchored onto the piece(s)
    already placed so the union stays connected -- a plain rectangle most of the
    time, an L/T-shaped notch the rest, matching the real solidity split."""
    n_rects = rng.choice([1, 2, 3], p=n_rects_weights)
    aspect_ratio = rng.uniform(1.0, 1.8)
    long_axis_x = rng.integers(2) == 0

    if n_rects == 1:
        fracs = [1.0]
    elif n_rects == 2:
        f = rng.uniform(0.5, 0.7)
        fracs = [f, 1 - f]
    else:
        f1 = rng.uniform(0.4, 0.55)
        f2 = rng.uniform(0.15, 1 - f1 - 0.15)
        fracs = [f1, f2, 1 - f1 - f2]

    def dims(area, ar, axis_x):
        return (np.sqrt(area * ar), np.sqrt(area / ar)) if axis_x else (np.sqrt(area / ar), np.sqrt(area * ar))

    w0, h0 = dims(target_area_m2 * fracs[0], aspect_ratio, long_axis_x)
    rect0 = box(-w0 / 2, -h0 / 2, w0 / 2, h0 / 2)
    rects = [rect0]
    ref_xmin, ref_ymin, ref_xmax, ref_ymax = rect0.bounds

    for frac in fracs[1:]:
        area_k = target_area_m2 * frac
        ar_k = rng.uniform(1.0, 2.0)
        wk, hk = dims(area_k, ar_k, rng.integers(2) == 0)
        overlap_frac = rng.uniform(*corner_overlap_range)

        sign_x, sign_y = rng.choice([-1, 1]), rng.choice([-1, 1])
        corner_x = ref_xmax if sign_x > 0 else ref_xmin
        corner_y = ref_ymax if sign_y > 0 else ref_ymin
        if sign_x > 0:
            xmin_k = corner_x - overlap_frac * wk; xmax_k = xmin_k + wk
        else:
            xmax_k = corner_x + overlap_frac * wk; xmin_k = xmax_k - wk
        if sign_y > 0:
            ymin_k = corner_y - overlap_frac * hk; ymax_k = ymin_k + hk
        else:
            ymax_k = corner_y + overlap_frac * hk; ymin_k = ymax_k - hk

        rects.append(box(xmin_k, ymin_k, xmax_k, ymax_k))
        ref_xmin, ref_ymin = min(ref_xmin, xmin_k), min(ref_ymin, ymin_k)
        ref_xmax, ref_ymax = max(ref_xmax, xmax_k), max(ref_ymax, ymax_k)

    return unary_union(rects), n_rects


def grow_smooth_rasterize_boundary(target_area_km2, rng, cell_area_km2=CELL_AREA_KM2, theta1_deg=0.0):
    """Superseded by `grow_staircase_boundary` above -- kept for comparison.
    Smooth-outline-then-rasterize method: fixes solidity and aspect ratio
    relative to `grow_eden_boundary`, but the outline's own edges are already
    axis-aligned, so rasterizing them often produces no zigzag at all (too
    clean) -- see module docstring, Method 2.
    """
    cell_side_m = np.sqrt(cell_area_km2) * 1000.0
    outline, n_rects = _sample_smooth_outline(target_area_km2 * 1e6, rng)

    shape = _rasterize(outline, cell_side_m)
    if shape.geom_type == "MultiPolygon":
        shape = max(shape.geoms, key=lambda g: g.area)
    if shape.interiors:
        shape = Polygon(shape.exterior)
    if theta1_deg != 0.0:
        shape = affinity.rotate(shape, theta1_deg, origin="centroid")

    n_cells = round(shape.area / (cell_side_m ** 2))
    achieved_area_km2 = shape.area / 1e6
    return shape, n_cells, achieved_area_km2


# ---------------------------------------------------------------------------
# Method 1 (superseded): plain cell-by-cell random growth (Eden growth)
# ---------------------------------------------------------------------------

def grow_eden_boundary(target_area_km2, rng, cell_area_km2=CELL_AREA_KM2,
                        theta1_deg=0.0, elongation=0.0):
    """Superseded by `grow_staircase_boundary` -- kept for comparison. A
    random polyomino grown on the block grid one cell at a time (Eden growth
    / lattice-animal model), optionally biased with `elongation` to favor
    one axis. Reproduces the real 0/90 edge alignment and roughly the real
    area, but not solidity or edge coherence -- see module docstring, Method 1.
    """
    n_cells = max(1, round(target_area_km2 / cell_area_km2))
    cell_side_m = np.sqrt(cell_area_km2) * 1000.0

    occupied = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    axis = rng.integers(2)  # 0 -> long axis is x, 1 -> long axis is y; per-farm coin flip

    while len(occupied) < n_cells and frontier:
        candidates = list(frontier)
        if elongation > 0:
            occ_coord = np.fromiter((c[axis] for c in occupied), dtype=float, count=len(occupied))
            center = occ_coord.mean()
            cand_coord = np.fromiter((c[axis] for c in candidates), dtype=float, count=len(candidates))
            weights = 1.0 + elongation * np.abs(cand_coord - center)
            weights /= weights.sum()
            choice = candidates[rng.choice(len(candidates), p=weights)]
        else:
            choice = candidates[rng.integers(len(candidates))]
        occupied.add(choice)
        frontier.discard(choice)
        cx, cy = choice
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            n = (cx + d[0], cy + d[1])
            if n not in occupied:
                frontier.add(n)

    squares = [box(cx * cell_side_m, cy * cell_side_m,
                    (cx + 1) * cell_side_m, (cy + 1) * cell_side_m)
               for cx, cy in occupied]
    shape = unary_union(squares)
    if shape.interiors:
        shape = Polygon(shape.exterior)
    if theta1_deg != 0.0:
        shape = affinity.rotate(shape, theta1_deg, origin="centroid")

    achieved_area_km2 = len(occupied) * cell_area_km2
    return shape, len(occupied), achieved_area_km2
