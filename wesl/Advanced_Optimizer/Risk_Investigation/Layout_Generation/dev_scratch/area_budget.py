import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
import time

exec(open("v2_harness.py").read())

t0 = time.time()
# Fine grid, big enough to cover out to 200km
RES = 200.0  # meters
MAXR = 200_000.0
x0, x1 = VINEYARD_CENTER[0]-MAXR, VINEYARD_CENTER[0]+MAXR
y0, y1 = VINEYARD_CENTER[1]-MAXR, VINEYARD_CENTER[1]+MAXR
nx = int((x1-x0)/RES)+1
ny = int((y1-y0)/RES)+1
gx = x0 + np.arange(nx)*RES
gy = y0 + np.arange(ny)*RES
GX, GY = np.meshgrid(gx, gy, indexing="xy")
GXf, GYf = GX.ravel(), GY.ravel()

fed = is_federal(GXf, GYf)
shallow = is_shallow_enough(GXf, GYf)
eligible = fed & shallow
dist_km = np.hypot(GXf - VINEYARD_CENTER[0], GYf - VINEYARD_CENTER[1]) / 1000.0
conn = is_connection_plausible(GXf, GYf)
print(f"grid built in {time.time()-t0:.1f}s, {nx}x{ny} = {nx*ny:,} points")

# existing 13 leases union, for "already occupied" test
existing_union = unary_union(list(LEASE_TEMPLATES.values()))

cell_area_km2 = (RES/1000.0)**2

print(f"\n{'raio(km)':>9s} {'area elegivel bruta (km2)':>27s} {'ja ocupada pelas 13 leases (km2)':>34s} {'livre (km2)':>13s} {'livre + conexao<=70km (km2)':>28s} {'MW estimado (~3MW/km2)':>24s}")
for R in [60, 90, 120, 150, 180]:
    within = dist_km <= R
    elig_here = eligible & within
    area_elig = elig_here.sum() * cell_area_km2

    # occupied mask: grid points that fall inside the existing 13-lease union
    pts_elig_idx = np.where(elig_here)[0]
    # vectorized-ish occupied check via STRtree would be better, but do a coarse union.contains via prepared geometry
    from shapely import prepared
    prep_union = prepared.prep(existing_union)
    occ_mask = np.array([prep_union.contains(Point(GXf[i], GYf[i])) for i in pts_elig_idx])
    area_occ = occ_mask.sum() * cell_area_km2
    area_free = area_elig - area_occ

    elig_conn_here = elig_here & conn
    pts2 = np.where(elig_conn_here)[0]
    occ_mask2 = np.array([prep_union.contains(Point(GXf[i], GYf[i])) for i in pts2])
    area_free_conn = elig_conn_here.sum()*cell_area_km2 - occ_mask2.sum()*cell_area_km2

    mw_est = area_free_conn * 3.0
    print(f"{R:9d} {area_elig:27.0f} {area_occ:34.0f} {area_free:13.0f} {area_free_conn:28.0f} {mw_est:24.0f}")

print(f"\nDelta 'high' necessario: {7872*4 - 7872:.0f} MW")
print(f"Total 'high': {7872*4:.0f} MW,  Total 'medium': {7872*2.5:.0f} MW")
