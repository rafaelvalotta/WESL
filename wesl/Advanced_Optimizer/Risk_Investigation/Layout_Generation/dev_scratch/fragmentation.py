import numpy as np
from scipy import ndimage
from shapely.geometry import Point
from shapely import prepared
from shapely.ops import unary_union

exec(open("v2_harness.py").read())
exec(open("v2_final_engines.py").read())

RES = 300.0
grid = build_eligibility_grid_150(resolution_m=RES, radius_km=150.0)
eligible = grid["eligible"]
x0, y0, nx, ny = grid["x0"], grid["y0"], grid["nx"], grid["ny"]
print(f"grid: {nx}x{ny}, built in {grid['build_time']:.1f}s")

# rasterize "already occupied" = union of 13 real leases, buffered by half the min
# new-farm separation (800m/2) since a new farm's edge must clear that gap too
existing_union = unary_union(list(LEASE_TEMPLATES.values())).buffer(MIN_FARM_SEPARATION_M / 2)
prep = prepared.prep(existing_union)

gx = x0 + np.arange(nx) * RES
gy = y0 + np.arange(ny) * RES
GX, GY = np.meshgrid(gx, gy, indexing="xy")

# vectorized-ish occupied mask via STRtree-free bounding box prefilter, then prepared.contains
minx, miny, maxx, maxy = existing_union.bounds
bbox_mask = (GX >= minx) & (GX <= maxx) & (GY >= miny) & (GY <= maxy)
occ = np.zeros_like(eligible, dtype=bool)
idxs = np.where(bbox_mask)
print(f"checking {len(idxs[0]):,} candidate points against existing-cluster union...")
for iy, ix in zip(*idxs):
    if prep.contains(Point(GX[iy, ix], GY[iy, ix])):
        occ[iy, ix] = True

free = eligible & ~occ
cell_km2 = (RES / 1000.0) ** 2
print(f"\nArea elegivel total: {eligible.sum()*cell_km2:.0f} km2")
print(f"Area ja ocupada (13 leases + buffer 400m): {occ.sum()*cell_km2:.0f} km2")
print(f"Area livre: {free.sum()*cell_km2:.0f} km2")

# connected components (4-connectivity) on the free mask
labeled, n_components = ndimage.label(free, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
sizes_cells = ndimage.sum(free, labeled, index=np.arange(1, n_components + 1))
sizes_km2 = sizes_cells * cell_km2
sizes_km2 = np.sort(sizes_km2)[::-1]

print(f"\nNumero de fragmentos contiguos de area livre: {n_components}")
print(f"Top 15 maiores fragmentos (km2): {np.round(sizes_km2[:15], 1)}")
print(f"Fragmento mediano: {np.median(sizes_km2):.2f} km2")

# density used elsewhere in the project (~3 MW/km2, matching config/paper values) to
# translate real farm-size targets into an area a single contiguous patch would need
DENSITY_MW_KM2 = 3.0
print(f"\n--- Quantos fragmentos cabem cada tamanho real de fazenda (a ~{DENSITY_MW_KM2} MW/km2)? ---")
for mw in REAL_FARM_SIZES_MW:
    need_km2 = mw / DENSITY_MW_KM2
    n_fit = (sizes_km2 >= need_km2).sum()
    area_in_fitting_fragments = sizes_km2[sizes_km2 >= need_km2].sum()
    print(f"{mw:5.0f} MW (precisa >= {need_km2:6.1f} km2 num fragmento so): "
          f"{n_fit:3d} fragmentos servem, area total neles = {area_in_fitting_fragments:8.0f} km2")

print(f"\nMeta 'high' precisa de: {(7872*4 - 7872):.0f} MW no total, distribuido em pedacos de ate 2080MW cada")
