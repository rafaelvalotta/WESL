EAST SCOTLAND OFFSHORE WIND CLUSTER (Moray Firth to Firth of Forth)
Geometry extracted directly from the WindGIS vector-tile backend that Sea
Impact's map (sea-impact.com/offshore-wind-map) embeds, NOT traced by hand.
Same method as the US/MA-RI cluster (see US_Cluster/scripts/README.txt for
the full methodology writeup) - this file only covers what's specific to
this region.

FARMS IN THIS CLUSTER (32)
---------------------------
Matches the reference screenshot: Pentland Floating Offshore Wind Farm,
Stromar Wind, Beatrice, Moray East, Moray West, Sinclair, Broadshore, Buchan
Offshore Wind, MarramWind, Green Volt, Aspen, Harbour Energy 1, Beech,
TotalEnergies E&P UK, Cenos, Salamander Floating Wind, Hywind Scotland,
Muir Mhor, CampionWind, Aberdeen Offshore Wind Farm, Kincardine, Bowdun,
Morven, Ossian, Bellrock, Cedar, Harbour Energy 2, Seagreen 1A, Inch Cape,
Neart na Gaoithe, Berwick Bank Wind Farm, Forthwind Offshore Wind Turbine
Demonstration.

NOT included (adjacent but outside the reference screenshot's cluster,
west-coast/Orkney projects that showed up in the wider raw extraction):
West of Orkney, Havbredey, EMEC, Cluaran Ear-Thuath, Scaraben, Caledonia
Offshore Wind Farm, Malin Sea Wind, MachairWind, Spiorad na Mara, Talisk
Offshore Wind, Seagreen (the base project, distinct from "Seagreen 1A"),
BP Alternative Energy Investments. Ask if you want any of these added -
their lease polygons were still used internally so points genuinely
belonging to them wouldn't get miscounted into the 32 target farms (see
CAVEAT below).

WHAT'S IN by_farm/ (mirrors the US cluster's structure)
--------------------------------------------------------
  lease_area.geojson   Always present. Dissolved across vector-tile seams.
  turbines.geojson     Only if the farm has installed/placed turbine
                         positions in the source today.
  substations.geojson  Only if a substation exists in that lease area.

Not produced for this pass (by explicit request - deprioritized after
inter-array cables turned out to need a much more expensive extraction):
  inter_array_cables.geojson, export_cables.geojson

CAVEAT: turbine counts run ~1.6-1.8x above the declared count for several
FULLY COMMISSIONED farms
---------------------------------------------------------------------------
Beatrice: 144 found vs. 84 declared. Moray East: 174 vs. 100. Moray West:
105 vs. 60. Inch Cape: 124 vs. 72. Neart na Gaoithe: 97 vs. 54. Aberdeen
Offshore Wind Farm: 20 vs. 11. Hywind Scotland: 36 vs. 5. Kincardine: 34
vs. 5.

This was checked, not assumed:
  1. Verified each farm's dissolved lease polygon has normal, expected
     bounds (not accidentally merged with a neighbor).
  2. Verified point counts BEFORE any farm assignment - i.e. counted every
     unique turbine coordinate that falls inside Beatrice's polygon alone:
     still 144.
  3. Verified across the combined Beatrice+Moray East+Moray West bounding
     box (removing any possibility of cross-farm-boundary bleeding): 396
     unique points vs. 244 real/declared total - the excess persists even
     pooled across all three, so it isn't a shared-boundary snapping issue.
  4. Re-ran farm assignment using ALL 348 farm polygons found in the raw
     North Sea-wide extraction (not just these 32) as the containment
     test, so a point actually inside a nearby NON-target farm (e.g.
     "Caledonia Offshore Wind Farm", right next to Beatrice) can match that
     farm instead of falling back to "nearest target farm" - counts didn't
     change.

Conclusion: the source's own turbine-position layer contains more unique,
non-duplicate coordinates in and around these specific lease areas than the
officially declared turbine count for the installed project. Likely
explanation: a mix of as-installed positions with additional
planned/study/repowering positions co-located in the same layer, without a
field in the data to tell them apart. Not something dissolve/dedupe logic
can fix - flagging as a source data quality issue, same spirit as the
Sunrise Wind caveat in the US cluster's README (96 found vs. 84 declared
there - this region's discrepancy is just much larger in relative terms).

RECOMMENDATION: for any layout/optimization pipeline, prefer the declared
"no_turbine" count (in lease_area.geojson properties) as the trusted design
target for these 7-8 farms, and treat turbines.geojson as "candidate
positions the source shows today" rather than ground truth, until this is
cross-checked against another source (e.g. each developer's public FEED/
Section 36 consent documents).

CAVEAT: inter-array and export cables
----------------------------------------
Both cable layers loaded at zoom 6-7 across the whole region return ZERO
features - not a bug, verified by zooming into Beatrice/Moray East directly
at zoom 8-9, where 298 inter-array + 21 export cable fragments DID load.
The vector tileset appears to drop these thin line features at low zoom
(a common tippecanoe simplification behavior for lines), so capturing them
cluster-wide would need a separate zoomed-in pass per sub-area (4-6 passes
across this cluster's geographic spread) rather than the single wide pass
that worked for polygons/points. Skipped for this round per explicit
request to prioritize boundary + turbines; can be added in a follow-up if
needed.

LICENSING
---------
Verify redistribution/usage terms with Sea Impact / LAUTEC before using
this commercially - this data is pulled from their public embedded viewer.
