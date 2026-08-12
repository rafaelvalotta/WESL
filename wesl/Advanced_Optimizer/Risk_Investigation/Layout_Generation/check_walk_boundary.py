"""Quick debug script (not wired into staircase.py yet) -- direct boundary
walk: K real corners, each wall between two corners is either PLAIN (one
axis direction, bounded length, no tilt -- only 0/90 micro-steps ever) or
STAIRCASE (a true diagonal displacement, drawn as monotonic alternating
unit steps -- no backtracking). K is drawn from a small range matched to
counted real corners (Attentive Energy ~7, Community Offshore ~8, Atlantic
Shores/Leading Light ~4, from a hand-annotated real map) -- not fixed, not
wild either.

Every draw is checked against six constraints before being accepted --
each one traces back to a specific real-vs-generated mismatch found by
visual review in this pipeline, not a tuned-by-eye default:

  1. no single wall (least of all the auto-closing one) may exceed
     `max_wall_fraction` of the total perimeter -- a dominant wall was
     the single most common rejection in the first review pass.
  2. bounding-box aspect ratio must fall in `aspect_range` -- matches the
     9 real leases over 100 km2 in the main cluster (1.00-1.83 measured).
  3. at least `min_stair_fraction` of the perimeter must be staircase --
     an all-plain rectangle was rejected in review (real leases always
     show *some* jagged edge, never a perfectly clean silhouette).
  4. `polygon.buffer(-pinch_clearance)` must stay a single connected piece
     -- catches a concave turn that nearly pinches the shape into two
     lobes, a standard way to test for a too-thin neck.
  5. `polygon.buffer(-corridor_erosion).area / polygon.area` must exceed
     `min_area_retained` -- catches a thin winding corridor that a plain
     bounding-box aspect ratio misses entirely (it can be locally narrow
     while the overall box is not stretched). Calibrated on 9 hand-labeled
     examples from a second review pass: at 1-cell erosion, the 4 rejected
     ones retained 0.33-0.49 of their area, the 5 approved ones 0.55-0.73
     -- a clean gap, threshold set at the midpoint (0.50).
  6. the walk must actually close (already enforced structurally).

Each is a hard reject-and-retry in the same sampling loop used for
closure, not a post-hoc filter -- a draw that fails is discarded and
resampled, so every *returned* shape satisfies all six by construction.
Saves labeled PNGs to figures_check/ for visual review; a stats pass runs
first (no plotting) to confirm the accept rate and the metric
distributions are reasonable before spending time on a gallery.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon

sys.path.insert(0, ".")
from staircase import solidity, bbox_aspect_ratio, staircase_run_lengths

FIGDIR = Path("figures_check")
FIGDIR.mkdir(exist_ok=True)

HEADINGS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, N, W, S

ASPECT_RANGE = (1.0, 1.9)       # measured on the 9 real large leases: 1.00-1.83
MIN_STAIR_FRACTION = 0.15       # perimeter share -- no draw may come out fully plain
PINCH_CLEARANCE = 0.75          # in cell units -- shrink test for near-self-touching notches
CORRIDOR_EROSION = 1.0          # in cell units -- shrink test for a thin winding corridor
MIN_AREA_RETAINED = 0.50        # calibrated midpoint between labeled-bad (0.33-0.49) and labeled-good (0.55-0.73)


def _draw_wall(pos, kind, heading_idx, length, stair_ratio):
    """Return the list of unit-step points for one wall, starting at `pos`
    (exclusive) -- PLAIN: `length` steps in `heading_idx`'s direction.
    STAIRCASE: alternates `heading_idx` and the direction 90 deg to its left,
    in `stair_ratio`:1, monotonic (never backtracks either component)."""
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
    if shrunk.is_empty:
        return False
    if isinstance(shrunk, MultiPolygon):
        return False
    return True


def _no_thin_corridor(poly, erosion, min_retained):
    """A locally narrow corridor can exist even when the overall bounding-box
    aspect ratio looks fine -- erosion collapses it (loses most of its area)
    even though the rest of the shape survives, so the AREA RATIO before/after
    a moderate shrink is a more direct test than the aspect ratio alone."""
    shrunk = poly.buffer(-erosion)
    if shrunk.is_empty:
        return False
    return (shrunk.area / poly.area) >= min_retained


def walk_boundary(rng, n_corners, seg_range=(2, 8), stair_ratio_range=(2, 4), stair_prob=0.4,
                   max_wall_fraction=0.30, aspect_range=ASPECT_RANGE,
                   min_stair_fraction=MIN_STAIR_FRACTION, pinch_clearance=PINCH_CLEARANCE,
                   corridor_erosion=CORRIDOR_EROSION, min_area_retained=MIN_AREA_RETAINED):
    """Build K walls with mostly-consistent left turns (a few right turns for
    a concave notch), force the last wall to close the loop, and accept only
    if all 6 constraints from the module docstring hold -- otherwise retry."""
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
            for i in range(steps):
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

        ar = bbox_aspect_ratio(poly)
        if not (aspect_range[0] <= ar <= aspect_range[1]):
            continue

        stair_len = sum(l for k, l in zip(wall_kinds, wall_lengths) if k == "staircase")
        if stair_len / total_perimeter < min_stair_fraction:
            continue

        if not _is_single_piece(poly, pinch_clearance):
            continue

        if not _no_thin_corridor(poly, corridor_erosion, min_area_retained):
            continue

        return poly, wall_kinds

    return None, None


def plot_example(ax, poly, title):
    if poly is None:
        ax.set_title(f"{title}\n(no valid draw in budget)")
        return
    x, y = poly.exterior.xy
    ax.fill(x, y, color="#1B6E86", alpha=0.6)
    ax.plot(x, y, color="#0d3b47", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\narea={poly.area:.0f} cells, AR={bbox_aspect_ratio(poly):.2f}, "
                 f"solidity={solidity(poly):.2f}", fontsize=9)


def run_stats(configs, n=150):
    print(f"=== Stats pass, n={n} draws per config (no plotting) ===")
    for label, kwargs in configs:
        ars, sols, runs, retained, fails = [], [], [], [], 0
        for seed in range(n):
            rng = np.random.default_rng(5000 + seed)
            poly, kinds = walk_boundary(rng, **kwargs)
            if poly is None:
                fails += 1
                continue
            ars.append(bbox_aspect_ratio(poly))
            sols.append(solidity(poly))
            # min_edge_len=0.5, not the 1.0 default -- here coordinates are raw
            # unit cell-steps (edges are exactly length 1), not meters like the
            # smooth+rasterize method; the 1.0 default would filter every edge out
            runs.append(np.mean(staircase_run_lengths(poly, min_edge_len=0.5)))
            shrunk = poly.buffer(-CORRIDOR_EROSION)
            retained.append(shrunk.area / poly.area if not shrunk.is_empty else 0.0)
        accepted = n - fails
        print(f"{label:28s} accepted={accepted:3d}/{n}  "
              f"AR mean={np.mean(ars):.2f}  solidity mean={np.mean(sols):.2f}  "
              f"run-length mean={np.mean(runs):.2f}  area-retained mean={np.mean(retained):.2f}"
              if accepted else f"{label}: 0 accepted")
    print("real reference: AR mean=1.28 (1.00-1.83), solidity mean=0.79 (0.60-0.98), "
          "run-length mean=12-23 (well-behaved) to 4.9 (Revolution)")


if __name__ == "__main__":
    configs = [
        ("a_6corners_mid_stair", dict(n_corners=6, stair_prob=0.4)),
        ("b_7corners_mid_stair", dict(n_corners=7, stair_prob=0.4)),
        ("c_8corners_mid_stair", dict(n_corners=8, stair_prob=0.4)),
        ("d_8corners_high_stair", dict(n_corners=8, stair_prob=0.7)),
        ("e_10corners_mid_stair", dict(n_corners=10, stair_prob=0.4)),
    ]

    run_stats(configs)

    fig, axes = plt.subplots(len(configs), 6, figsize=(22, 4 * len(configs)))
    for row, (label, kwargs) in enumerate(configs):
        for col in range(6):
            rng = np.random.default_rng(row * 100 + col)
            poly, kinds = walk_boundary(rng, **kwargs)
            plot_example(axes[row, col], poly, f"{label} seed={col}")

    fig.suptitle("check_walk_boundary.py v3 -- 6 hard constraints, reject+retry", fontsize=13)
    fig.tight_layout()
    out_path = FIGDIR / "walk_boundary_grid_v3.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
