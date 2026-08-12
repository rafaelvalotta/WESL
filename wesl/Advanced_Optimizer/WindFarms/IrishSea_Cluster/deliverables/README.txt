IRISH SEA / LIVERPOOL BAY OFFSHORE WIND CLUSTER
Geometry extracted directly from the WindGIS vector-tile backend that Sea
Impact's map (sea-impact.com/offshore-wind-map) embeds, NOT traced by hand.
Same method as the US/MA-RI and UK/East Scotland clusters - see
tools/offshore_wind_extraction/TUTORIAL.md for the full methodology.

FARMS IN THIS CLUSTER (19)
---------------------------
Awel Y Mor, Barrow, Burbo Bank, Burbo Bank Extension, Greystones, Gwynt y Mor,
Mona, Mooir Vannin, Morecambe Offshore Windfarm, Morgan, North Hoyle, Ormonde,
Rhyl Flats, Robin Rigg East, Robin Rigg West, Walney Extension, Walney Phase 1,
Walney Phase 2, West of Duddon Sands.

WHAT'S IN by_farm/
-------------------
  lease_area.geojson   Always present. Dissolved across vector-tile seams.
  turbines.geojson     Only if the farm has installed/placed turbine
                         positions in the source today.
  substations.geojson  Only if a substation exists in that lease area.

Not produced for this pass: inter_array_cables.geojson, export_cables.geojson.
The raw extraction DOES include cable data this time (696 inter-array + 39
export line fragments were captured), but build_region_pipeline.py doesn't
process cables yet - only lease/turbines/substations. Add cable dissolve
logic (same unary_union-by-property approach as lease areas) if needed.

NOTE: Walney Extension turbine count (87 found vs. 40 "declared")
---------------------------------------------------------------------------
Checked and NOT a data quality problem - the opposite, actually. Walney
Extension is a real, documented case of an offshore wind farm built with two
different turbine models on the same site: 40x Siemens SWT-7.0-154 (7MW) +
47x MHI Vestas V164-8.25 (8.25MW) = 87 turbines total, ~659MW. Confirmed by
checking `turbinemod` on the extracted points - exactly these two models
appear, splitting close to 40/47. The `no_turbine` property in the lease
polygon (40) looks like it reflects only one phase/model or an outdated
filing, not the actual built count. Extracted turbines.geojson here is more
accurate than the declared count, unlike the Scotland cluster's caveat where
it was the other way around - always spot-check rather than assuming either
number is automatically right.

Everything else matched declared counts exactly: Barrow 30/30, Burbo Bank
25/25, Burbo Bank Extension 32/32, Gwynt y Mor 160/160, North Hoyle 30/30,
Ormonde 30/30, Rhyl Flats 25/25, Robin Rigg West 30/30, Walney Phase 1 51/51,
Walney Phase 2 51/51, West of Duddon Sands 108/108. Robin Rigg East was
close (30 found vs. 28 declared).

LICENSING
---------
Verify redistribution/usage terms with Sea Impact / LAUTEC before using this
commercially - this data is pulled from their public embedded viewer.
