MASSACHUSETTS / RHODE ISLAND SOUND OFFSHORE WIND CLUSTER
Geometry extracted directly from the WindGIS vector-tile backend that Sea
Impact's map (sea-impact.com/offshore-wind-map) embeds, NOT traced by hand.

WHAT'S IN by_farm/
------------------
One subfolder per lease area, named after the project (spaces/parens/slashes
replaced with "_"). Each subfolder contains only the files that apply to that
project - a pre-construction project with no installed turbines will not have
a turbines.geojson file, for example.

  lease_area.geojson        Always present. One Polygon/MultiPolygon feature:
                             the full lease-area boundary, already dissolved
                             across vector-tile seams (see METHOD below).
  turbines.geojson          Present only if the farm has installed/placed
                             turbine positions today. Point features,
                             properties.point_type = "turbine".
  substations.geojson       Present only if an offshore substation/converter
                             station exists in that lease area. Point
                             features, properties.point_type = "substation".
                             NEVER mixed into turbines.geojson.
  inter_array_cables.geojson  Present only if inter-array cable routes exist.
                             LineString/MultiLineString.
  export_cables.geojson     Present only if export cable routes exist.
                             LineString/MultiLineString.

_summary.json               One row per farm: turbine/substation/cable counts,
                             the project's declared "no_turbine" attribute
                             (from the source dataset, for cross-checking),
                             and its permitting status.

Farms in this cluster (11): Block Island, South Fork Wind, Revolution Wind,
Sunrise Wind, Vineyard Wind 1, Bay State Wind (OCS-A 0500), New England Wind 1,
New England Wind 2, Beacon Wind, SouthCoast Wind, Vineyard Northeast (OCS-A 0522).
Only Block Island, South Fork Wind, Revolution Wind, Sunrise Wind and
Vineyard Wind 1 have turbines placed today - the rest are still in permitting
(no turbine layout published yet), so they only have a lease_area.geojson.

METHOD (for reproducing on another cluster/region)
---------------------------------------------------
The map is Mapbox GL JS, fed by vector tiles from:
  https://api.eu.windgis.lautec.com/api/tile/vector/public/{layerId}/{z}/{x}/{y}

Reading those tiles raw means dealing with protobuf + reassembling anything
that spans more than one tile. Easier: let the browser's own Mapbox GL
instance parse the tiles for you, then read the parsed features back out with
map.querySourceFeatures(). That's what extract_windgis_layers.js does -
paste it into the DevTools console on the WindGIS viewer iframe (see comments
at the top of that file for the exact steps).

The catch: vector tiles clip polygons/lines exactly at tile borders. A lease
area spanning multiple tiles comes back as several fragments with identical
properties but different (partial) geometry - if you don't rejoin them, you
get exactly the "boundaries overlapping/gaps" problem manual tracing has in
a dense cluster. dissolve_tiles.py fixes this with shapely's unary_union,
grouping fragments by a shared property (wf_name) before merging. Turbine/
substation points get deduped by rounded coordinate instead (a point only
duplicates if it sits exactly on a tile edge).

IMPORTANT for reproducing this on another cluster: map.querySourceFeatures()
only returns features from tiles the browser has actually loaded (i.e. what's
currently on screen). If the map viewport doesn't fully cover a lease area -
including because a side panel (layers list, etc.) is covering part of the
browser window and shrinking the visible map - that polygon comes back
truncated at whatever edge fell outside the loaded tiles, with no error or
warning. Use map.fitBounds([[minLon,minLat],[maxLon,maxLat]]) with a bbox that
generously covers the whole area you want (wider than you think you need),
close any overlapping side panels, and give the map a couple of seconds to
finish loading tiles before extracting. This is exactly what caused
SouthCoast Wind and Vineyard Northeast (OCS-A 0522) to come out clipped on
their eastern edge in an earlier pass over this same cluster.

For THIS cluster, the per-farm split you're looking at also had to resolve a
second problem: several lease areas sit close enough together (Revolution
Wind / South Fork Wind / Sunrise Wind especially) that a turbine sitting
right at its own lease boundary could snap to the wrong neighboring farm.
build_per_farm.py assigns each point to whichever farm polygon strictly
contains it first, and only falls back to "nearest lease boundary" (capped at
~1 km) for points that fall just outside their own polygon. After that fix,
turbine counts matched the source's declared no_turbine exactly for
Revolution Wind (65/65), South Fork Wind (12/12) and Vineyard Wind 1 (62/62).

CAVEAT: export cables can be shared between farms
---------------------------------------------------
Export cable routes run from a lease boundary to a shore landfall, often
bundled together with a neighboring project's cable for part of that run.
When a dissolved export cable sits within snapping distance of more than one
farm's polygon, the SAME feature is written into every matching farm's
export_cables.geojson, with a `shared_with_farms` property listing the other
project(s) it's bundled with. Don't double-count these if you sum cable
length across farms.

Block Island's own export cable wasn't present in the vector-tile layer this
was pulled from (possibly because it's a short nearshore run tagged
differently in the source) - Block_Island/ has no export_cables.geojson.

CAVEAT: Sunrise Wind
--------------------
Sunrise Wind shows 96 turbine point features vs. a declared no_turbine of 84
in the source's own attribute table. This isn't a dedupe/assignment bug on
our side (spot-checked against the lease polygon) - it looks like the
source's "no_turbine" attribute (likely from an older permit filing) hasn't
been updated to match the current as-drawn turbine layout. Treat
turbines.geojson as the ground truth for actual positions, and no_turbine in
_summary.json / lease_area.geojson properties as the developer's officially
stated count - they can legitimately differ for projects under construction.

FILES FOR REUSE ON OTHER CLUSTERS/REGIONS
------------------------------------------
  extract_windgis_layers.js   Paste into DevTools console -> extracts raw
                               GeoJSON fragments for every visible layer.
  dissolve_tiles.py            Generic tile-fragment dissolver (any region).
  build_per_farm.py            The script used for this specific MA/RI
                               cluster - copy and adjust the source/
                               sourceLayer IDs and farm-splitting logic for a
                               different region (those vector-tile IDs are
                               shared across the whole global dataset, so the
                               same source/sourceLayer pairs work everywhere -
                               only the map center/zoom and farm names change).

qa_map.html                  Plain, no-dependency SVG map for visually
                               sanity-checking the by_farm/ files (this is NOT
                               a deliverable for your PyWake pipeline - just a
                               look before you build with it).

LICENSING
---------
Verify redistribution/usage terms with Sea Impact / LAUTEC before using this
commercially - this data is pulled from their public embedded viewer.
