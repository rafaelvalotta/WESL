# Offshore wind cluster extraction — reusable pipeline

How we pulled the US (Massachusetts/Rhode Island) and UK (East Scotland) clusters,
written so the same process works for any new region. Two parts: a **browser
step** (needs a live session — a human or an agent driving a real page) and a
**local processing step** (fully scriptable, `build_region_pipeline.py` below).

Source: the map embedded at `sea-impact.com/offshore-wind-map` is actually an
iframe running `viewer.eu.windgis.lautec.com` (Mapbox GL JS), fed by vector
tiles from `api.eu.windgis.lautec.com/api/tile/vector/public/{layerId}/{z}/{x}/{y}`.
Reading the raw tiles means dealing with protobuf + reassembling anything that
spans more than one tile. Easier: let the browser's own Mapbox GL instance parse
the tiles, then read the parsed features back out with `map.querySourceFeatures()`.

Before doing this for a new region, read the LICENSING note at the bottom —
same caveat applies everywhere: this is public-facing embedded data, but that's
not automatically the same as free to redistribute/use commercially. Verify
with Sea Impact / LAUTEC before anything beyond internal use.

## Part 1 — Browser step (produces `raw_extracted.json`)

1. Open `https://sea-impact.com/offshore-wind-map/`, let it load, then find the
   iframe's real URL (`document.querySelector('iframe#shareMapPoints').src` in
   the console, or check Network/Elements tab) and navigate to that URL
   directly — you need to run console JS *inside* that iframe's own page, not
   the parent.
2. Open the Layers panel (the stacked-diamond icon) and toggle on whatever you
   need: **Wind Turbine Generators**, **Floating Wind Turbine Generators**,
   **Substations & Converters**, **Wind Farms** (boundary), and
   **Inter-Array Cables** / **Export Cables & Interconnectors** if you want
   cables (see the cable caveat below — costs more time).
   - The toggle switches sometimes render visually "off" even after a
     successful click — don't trust the switch's look, verify by checking
     `map.getStyle().sources` / `.layers` in the console (see step 4).
3. Paste `extract_windgis_layers.js` (same folder as this file) into the
   DevTools console. It locates the `mapboxgl.Map` instance living inside
   Mapbox GL's React tree and stashes it at `window.__map` — no manual
   digging needed.
4. Position the map over your target region and pull the data:
   ```js
   window.__map.fitBounds([[minLon, minLat], [maxLon, maxLat]], {padding: 10, animate: false});
   ```
   Wait ~3-5s for tiles to settle, then extract:
   ```js
   const sourcesWanted = [
     {source: '31d3523e-a5e7-4cf0-9315-371ffef98ec6', sourceLayer: 'wind_farms_28082023', key: 'leaseAreas'},
     {source: '31d3523e-a5e7-4cf0-9315-371ffef98ec6', sourceLayer: 'b0520f17-270b-4dd3-b57e-99b094d962d0', key: 'turbinesFixed'},
     {source: '31d3523e-a5e7-4cf0-9315-371ffef98ec6', sourceLayer: 'c811b67a-3702-459d-a96e-fce12eafe93b', key: 'turbinesFloating'},
     {source: '31d3523e-a5e7-4cf0-9315-371ffef98ec6', sourceLayer: '9b68c77f-e0a2-4609-902f-7652375b6ab7', key: 'substations'},
   ];
   const extracted = {};
   for (const {source, sourceLayer, key} of sourcesWanted) {
     extracted[key] = window.__map.querySourceFeatures(source, {sourceLayer})
       .map(f => ({type:'Feature', properties: f.properties, geometry: f.geometry}));
   }
   window.__extracted = extracted;
   JSON.stringify(Object.fromEntries(Object.entries(extracted).map(([k,v])=>[k,v.length])));
   ```
   Those `source`/`sourceLayer` UUIDs are **stable across the whole global
   dataset** — same ones work for any region, only the map center/bbox changes.
5. Copy `JSON.stringify(window.__extracted)` out and save as `raw_extracted.json`
   (for a big region this can be several MB — if your tool truncates large
   console output, save to a file via a Blob download instead:
   `const blob = new Blob([JSON.stringify(window.__extracted)], {type:'application/json'}); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'raw_extracted.json'; a.click();`).

### Pitfalls (both hit in practice, both cost real time)

- **Side panel or narrow window shrinking the map**: `querySourceFeatures` only
  returns features from tiles the browser actually loaded — i.e. what's/was on
  screen. A side panel covering part of the browser window shrinks the visible
  map and silently truncates polygons at whatever edge fell outside the loaded
  tiles, with no error. Use a wide window, close overlapping panels, and make
  the `fitBounds` box wider than you think you need.
- **WebGL context exhaustion**: after many navigations/page loads in one
  browser session, Mapbox GL can silently render solid black (WebGL context
  lost/exhausted). Fix: hard-reload the page fresh before extracting.
- **Cables need much higher zoom than boundaries/turbines**: lease-area
  polygons and turbine/substation points load fine in a single wide pass at a
  low zoom (~6-7) covering the whole region. Inter-array/export cable *lines*
  come back empty at that same low zoom — the tileset appears to drop thin
  line features at low zoom (a common vector-tile generalization behavior).
  Getting cables reliably needs zoom ~8-9 **per sub-area** (several passes
  across a large region), which is meaningfully more expensive than the single
  wide pass that covers boundaries+turbines. Decide up front whether cables are
  worth the extra passes for your use case.
- **Tile clipping**: a lease area spanning more than one tile at your capture
  zoom comes back as several fragments sharing identical properties but
  different (partial) geometry. Expected — `build_region_pipeline.py` dissolves
  these with `shapely.ops.unary_union`, grouped by `wf_name`.
- **Cross-farm turbine assignment**: in a dense cluster, a turbine sitting near
  its own lease boundary can be geometrically nearer another farm's boundary.
  Fix used here: first try strict polygon containment against **every** farm
  found in the raw extraction (not just your target list) — this stops points
  that truly belong to a neighboring, out-of-scope farm from being
  misattributed to one of your target farms via a naive nearest-match. Only
  fall back to "nearest target farm, capped at ~1km" for points outside every
  known polygon.
- **Declared turbine count vs. what's actually in the source**: cross-check
  `no_turbine` (declared, in the lease polygon's properties) against the
  turbine point count you extract. They can legitimately differ — sometimes by
  a little (a permit filing not yet updated), sometimes by a lot (~1.6-1.8x
  seen in the Scotland cluster for several fully-commissioned farms, most
  likely extra planned/study positions co-located in the same layer with no
  field distinguishing them). Always report both numbers, don't silently trust
  the point count as ground truth.

## Part 2 — Local processing (`build_region_pipeline.py`)

Fully scriptable once you have `raw_extracted.json`. Edit the `CONFIG` block at
the top of `build_region_pipeline.py` - for a brand-new region, you don't have
to type out every farm name by hand:

```python
CONFIG = {
    "region_name": "East Scotland",                 # for titles/labels only
    "raw_json_path": "/path/to/raw_extracted.json",
    "output_dir": "/path/to/<Region>_Cluster",        # sibling of US_Cluster
    "target_farms": None,   # None = every wf_name found in the lease layer
    "name_map": None,       # None = auto-generate (accent-stripped, no spaces)
    "raw_keys": None,       # None = auto-detect lease/turbines/substations keys
}
```

`resolve_config()` fills in the three `None`s by inspecting `raw_json_path`
itself: it finds which top-level key holds the lease-area polygons (by
property signature: `wf_name` + `capacity_m` on Polygon/MultiPolygon
features), which holds turbines (`turbinemod` on Point features) and which
holds substations (`sub_name` or `item == "Substation"` on Point features) -
these signatures are the actual WindGIS schema and have held across every
region pulled so far. It then lists every `wf_name` it finds and
auto-generates a clean pipeline-style name for each (accents stripped, spaces
removed). It prints what it detected/generated before proceeding, so you can
Ctrl-C and override anything that looks wrong (e.g. two farms colliding on
the same auto-generated clean name - it warns about this explicitly).

Only fall back to typing `target_farms`/`name_map` out by hand if you want to
(a) keep a **subset** of the farms found in the raw file, or (b) override a
specific auto-generated name you don't like. `raw_keys` almost never needs to
be set by hand - the four keys have proven stable across every region so far
(`f9bdb3d0-...` = lease areas, `0fd44220-...` = fixed turbines, etc. - see the
`CONFIG` already filled in for the Irish Sea cluster in this same file for the
literal values, if you'd rather hardcode them than rely on auto-detection).

Then run:
```bash
python3 build_region_pipeline.py
```

It produces, inside `output_dir`:
```
deliverables/by_farm/<slug>/lease_area.geojson
deliverables/by_farm/<slug>/turbines.geojson       (only if any exist)
deliverables/by_farm/<slug>/substations.geojson    (only if any exist)
deliverables/by_farm/_summary.json
deliverables/README.txt                             (method + per-region caveats, edit this)
Data/<CleanName>_polygon.geojson                     (one flat folder, PyWake-ready)
Data/<CleanName>_turbines.geojson
Data/<CleanName>_substations.geojson
Data/manifest.json
Data/cluster_map.html                                (self-contained Leaflet QA viewer)
```

This is the exact same structure as `US_Cluster/` and `UK_Cluster/` — any
downstream code that already reads one of those folders works unchanged
against a new region's folder.

### Multipolygon farms (lease area split across disconnected pieces)

If a farm comes back as a `MultiPolygon` (some real leases genuinely are —
several were found in both the US and UK clusters), decide per-farm whether
to split it into separate boundary files (e.g. `_North` / `_South`) before
handing it to a solver that expects one simple polygon per site. Check with:
```python
from shapely.geometry import shape
geom = shape(json.load(open('Data/X_polygon.geojson'))['features'][0]['geometry'])
print(geom.geom_type, len(geom.geoms) if geom.geom_type == 'MultiPolygon' else 1)
```
`build_region_pipeline.py` does **not** auto-split these — always check by
hand which sub-part actually holds any turbines/substations (point-in-polygon
against each part) before deciding how to name and file the split, the same
way it was done for `RevolutionWind_North/South`, `SunriseWind_North/South`,
`BayStateWind_North/South` and `NewEnglandWind2_North/South` in the US cluster.

## LICENSING

This data is pulled from Sea Impact / LAUTEC's public embedded viewer. Fine
for internal review/validation; verify redistribution/commercial-use terms
with them directly before anything beyond that, for every new region same as
the first.
