"""
Generic offshore-wind-cluster pipeline: raw_extracted.json (from
extract_windgis_layers.js, see TUTORIAL.md) -> deliverables/by_farm/ +
Data/ (clean, PyWake-ready GeoJSON + manifest + QA map), same structure as
US_Cluster/ and UK_Cluster/.

Edit CONFIG below for a new region, then: python3 build_region_pipeline.py
"""
import json, os, re, shutil, unicodedata
from collections import defaultdict
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union

# This CONFIG reflects the Irish Sea / Liverpool Bay cluster exactly as
# already generated at output_dir below (explicit target_farms/name_map, so
# re-running this file can't silently rename anything already shipped there).
#
# FOR A NEW REGION: start from the template at the bottom of this file
# (NEW_REGION_CONFIG_TEMPLATE) instead of editing this one - leave
# target_farms/name_map/raw_keys as None there and the script fills them in
# automatically from raw_json_path (see resolve_config() below). Only fall
# back to typing them out by hand if you want a SUBSET of farms, or want to
# override specific auto-generated names.

# CONFIG = {
#     "region_name": "Irish Sea / Liverpool Bay",
#     "raw_json_path": "/Users/brunoboer/Downloads/raw_extracted.json",
#     "target_farms": {
#         "Awel Y Mor", "Barrow", "Burbo Bank", "Burbo Bank Extension",
#         "Greystones", "Gwynt y Mor", "Mona", "Mooir Vannin",
#         "Morecambe Offshore Windfarm", "Morgan", "North Hoyle", "Ormonde",
#         "Rhyl Flats", "Robin Rigg East", "Robin Rigg West",
#         "Walney Extension", "Walney Phase 1", "Walney Phase 2",
#         "West of Duddon Sands",
#     },
#     "output_dir": "/Users/brunoboer/Documents/Software/WESL_jul_31/WESL/wesl/Advanced_Optimizer/IrishSea_Cluster",
#     "name_map": {
#         "Awel Y Mor": "AwelYMor",
#         "Barrow": "Barrow",
#         "Burbo Bank": "BurboBank",
#         "Burbo Bank Extension": "BurboBankExtension",
#         "Greystones": "Greystones",
#         "Gwynt y Mor": "GwyntYMor",
#         "Mona": "Mona",
#         "Mooir Vannin": "MooirVannin",
#         "Morecambe Offshore Windfarm": "Morecambe",
#         "Morgan": "Morgan",
#         "North Hoyle": "NorthHoyle",
#         "Ormonde": "Ormonde",
#         "Rhyl Flats": "RhylFlats",
#         "Robin Rigg East": "RobinRiggEast",
#         "Robin Rigg West": "RobinRiggWest",
#         "Walney Extension": "WalneyExtension",
#         "Walney Phase 1": "WalneyPhase1",
#         "Walney Phase 2": "WalneyPhase2",
#         "West of Duddon Sands": "WestOfDuddonSands",
#     },
#     "raw_keys": {
#         "lease": "f9bdb3d0-2e0d-41f2-a1ad-a8080838ea5c",
#         "turbines_fixed": "0fd44220-d56e-4e0a-a611-89dfa92dfa6e",
#         "turbines_floating": "8316c7c7-0ba2-42e6-8c51-ff0c4389f8a4",
#         "substations": "9878eab0-8a0a-4212-87bb-ed0ca561acbf",
#     },
# }

# Copy this block, replace the CONFIG assignment above with it (or just
# swap in these three values), for a brand-new region where you don't want
# to type out every farm name by hand:
NEW_REGION_CONFIG_TEMPLATE = {
    "region_name": "REGION NAME HERE",
    "raw_json_path": "/path/to/raw_extracted.json",
    "output_dir": "/path/to/<Region>_Cluster",
    "target_farms": None,   # None = every wf_name found in the lease layer
    "name_map": None,       # None = auto-generate (accent-stripped, no spaces)
    "raw_keys": None,       # None = auto-detect lease/turbines/substations keys
}

CONFIG = {
    "region_name": "Irish Sea auto-test",
    "raw_json_path": "/Users/brunoboer/Downloads/raw_extracted.json",
    "output_dir": "/Users/brunoboer/Documents/Software/WESL_jul_31/WESL/wesl/Advanced_Optimizer/IrishSea_Cluster_auto_test",
    "target_farms": None,
    "name_map": None,
    "raw_keys": None,
}


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def clean_name(name):
    """Auto-generated pipeline-style name: strip accents, drop everything
    that isn't alphanumeric, keep word boundaries as-is (no case changes) so
    "Gwynt y Mor" -> "GwyntyMor", "Robin Rigg East" -> "RobinRiggEast"."""
    return re.sub(r'[^A-Za-z0-9]+', '', strip_accents(name))


def auto_detect_raw_keys(data):
    """Look at each top-level key's features and guess which one is the
    lease-area layer, the turbine layer(s), and the substation layer, purely
    from geometry type + property signature - so you don't need to know the
    layer-id UUIDs in advance. These signatures come from the actual WindGIS
    schema (wf_name+capacity_m on lease polygons, turbinemod on turbine
    points, sub_name/item=Substation on substation points) and have held
    across every region pulled so far (US, Scotland, Irish Sea)."""
    lease_keys, turbine_keys, substation_keys = [], [], []
    for key, val in data.items():
        feats = val.get('features', val) if isinstance(val, dict) else val
        if not feats:
            continue
        sample = feats[0]
        geom_type = sample.get('geometry', {}).get('type', '')
        props = sample.get('properties', {})
        if geom_type in ('Polygon', 'MultiPolygon') and 'wf_name' in props and 'capacity_m' in props:
            lease_keys.append(key)
        elif geom_type == 'Point' and ('sub_name' in props or props.get('item') == 'Substation'):
            substation_keys.append(key)
        elif geom_type == 'Point' and 'turbinemod' in props:
            turbine_keys.append(key)

    if not lease_keys:
        raise ValueError(
            "Couldn't auto-detect the lease-area layer. Set CONFIG['raw_keys'] "
            "manually - inspect raw_json_path's top-level keys yourself (each "
            "key's features[0]['properties'] tells you what it is)."
        )
    # a symbol/label layer duplicating the fill layer's data is common (same
    # wf_name/capacity_m properties, just rendered as text) - any one works
    # for our purposes since dissolve is idempotent on identical geometry.
    return {
        'lease': lease_keys[0],
        'turbines_fixed': turbine_keys[0] if turbine_keys else '__none__',
        'turbines_floating': turbine_keys[1] if len(turbine_keys) > 1 else '__none__',
        'substations': substation_keys[0] if substation_keys else '__none__',
    }


def resolve_config(cfg):
    """Fill in target_farms/name_map/raw_keys from the raw file when left as
    None. Mutates and returns cfg."""
    data = json.load(open(cfg['raw_json_path']))

    if cfg.get('raw_keys') is None:
        cfg['raw_keys'] = auto_detect_raw_keys(data)
        print('Auto-detected raw_keys:', cfg['raw_keys'])

    lease_feats = get_features(data, cfg['raw_keys']['lease'])
    all_names = sorted({f['properties'].get('wf_name') for f in lease_feats} - {None})

    if cfg.get('target_farms') is None:
        cfg['target_farms'] = set(all_names)
        print(f'Auto-selected all {len(all_names)} farms found in the lease layer:')
        for n in all_names:
            print('  -', n)

    if cfg.get('name_map') is None:
        cfg['name_map'] = {n: clean_name(n) for n in cfg['target_farms']}
        dupes = defaultdict(list)
        for raw, clean in cfg['name_map'].items():
            dupes[clean].append(raw)
        collided = {c: raws for c, raws in dupes.items() if len(raws) > 1}
        if collided:
            print('WARNING: these farms produced the SAME clean name - set name_map manually for these:')
            for c, raws in collided.items():
                print(f'  "{c}" <- {raws}')

    return cfg

PALETTE = ['#3fb6a8', '#4f8fc0', '#7b7fd4', '#b06fc9', '#3f9e6e',
           '#5aa8d8', '#8b6fd0', '#c76f9e', '#4fb08a', '#d68a3d', '#a4413f',
           '#6f9ecb', '#9b7fd0', '#5fb894', '#c98f4a', '#4a9dc9']


def slug(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def get_features(data, key):
    """extractAll() (the browser script) saves each layer as
    {type, meta, features: [...]} - but a hand-built window.__extracted
    (like the one used ad hoc for the US/UK clusters) can also just be a bare
    list of features under that key. Accept either."""
    val = data.get(key)
    if val is None:
        return []
    if isinstance(val, dict):
        return val.get('features', [])
    return val


def dedupe_points(feats, precision=6):
    seen = set()
    out = []
    for f in feats:
        c = tuple(round(x, precision) for x in f['geometry']['coordinates'])
        if c in seen:
            continue
        seen.add(c)
        out.append(f)
    return out


def build_by_farm(cfg):
    data = json.load(open(cfg['raw_json_path']))
    rk = cfg['raw_keys']
    target_farms = cfg['target_farms']
    out_dir = os.path.join(cfg['output_dir'], 'deliverables', 'by_farm')
    os.makedirs(out_dir, exist_ok=True)

    # dissolve EVERY farm found in the raw extraction (not just targets) -
    # needed so turbine/substation assignment can correctly match points that
    # truly belong to a nearby non-target farm instead of misattributing them
    lease_groups = defaultdict(list)
    lease_props = {}
    for f in get_features(data, rk['lease']):
        p = f['properties']
        name = p.get('wf_name')
        lease_groups[name].append(shape(f['geometry']))
        lease_props[name] = p

    all_farms = {}
    for name, geoms in lease_groups.items():
        merged = unary_union(geoms)
        all_farms[name] = {'slug': slug(name), 'polygon': merged, 'props': lease_props[name]}

    farms = {name: info for name, info in all_farms.items() if name in target_farms}
    print(f'All farms dissolved for assignment purposes: {len(all_farms)}')
    print(f'Target farms found in data: {len(farms)} / {len(target_farms)}')
    missing = target_farms - set(farms.keys())
    if missing:
        print('MISSING (not found in raw extraction at all):', missing)
        for m in missing:
            print(f'  -> check exact spelling/accents of "{m}" against wf_name values in the source')

    def assign_farm(pt, max_snap_deg=0.01):
        contains_matches = [n for n, info in all_farms.items() if info['polygon'].contains(pt)]
        if len(contains_matches) == 1:
            return contains_matches[0] if contains_matches[0] in target_farms else None
        if len(contains_matches) > 1:
            best = min(contains_matches, key=lambda n: all_farms[n]['polygon'].area)
            return best if best in target_farms else None
        best_name, best_dist = None, None
        for n, info in farms.items():
            d = info['polygon'].distance(pt)
            if best_dist is None or d < best_dist:
                best_name, best_dist = n, d
        return best_name if (best_dist is not None and best_dist < max_snap_deg) else None

    turbines_by_farm = defaultdict(list)
    unmatched_turbines = 0
    raw_turbines = get_features(data, rk['turbines_fixed']) + get_features(data, rk['turbines_floating'])
    for f in dedupe_points(raw_turbines):
        lon, lat = f['geometry']['coordinates']
        name = assign_farm(Point(lon, lat))
        f['properties'] = {'point_type': 'turbine', **f['properties']}
        if name:
            turbines_by_farm[name].append(f)
        else:
            unmatched_turbines += 1

    substations_by_farm = defaultdict(list)
    unmatched_subs = 0
    for f in dedupe_points(get_features(data, rk['substations'])):
        p = f['properties']
        f['properties'] = {'point_type': 'substation', **p}
        name = p.get('wf_name')
        if name not in farms:
            lon, lat = f['geometry']['coordinates']
            name = assign_farm(Point(lon, lat))
        if name:
            substations_by_farm[name].append(f)
        else:
            unmatched_subs += 1

    summary = []
    for name, info in sorted(farms.items()):
        fslug = info['slug']
        fdir = os.path.join(out_dir, fslug)
        os.makedirs(fdir, exist_ok=True)

        lease_fc = {'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': info['props'], 'geometry': mapping(info['polygon'])}
        ]}
        json.dump(lease_fc, open(f'{fdir}/lease_area.geojson', 'w'))

        n_turb = 0
        if turbines_by_farm.get(name):
            json.dump({'type': 'FeatureCollection', 'features': turbines_by_farm[name]}, open(f'{fdir}/turbines.geojson', 'w'))
            n_turb = len(turbines_by_farm[name])

        n_sub = 0
        if substations_by_farm.get(name):
            json.dump({'type': 'FeatureCollection', 'features': substations_by_farm[name]}, open(f'{fdir}/substations.geojson', 'w'))
            n_sub = len(substations_by_farm[name])

        summary.append({
            'name': name, 'slug': fslug, 'turbines': n_turb, 'substations': n_sub,
            'declared_no_turbine': info['props'].get('no_turbine'),
            'status': info['props'].get('status'),
            'capacity_mw': info['props'].get('capacity_m'),
        })

    print()
    print(f'{"Farm":45s} {"turb":>5s} {"decl":>5s} {"sub":>4s}  status')
    for s in summary:
        flag = '  <-- CHECK: found >> declared' if (s['declared_no_turbine'] and s['turbines'] > 1.3 * s['declared_no_turbine']) else ''
        print(f'{s["name"]:45s} {s["turbines"]:5d} {str(s["declared_no_turbine"] or "-"):>5s} {s["substations"]:4d}  {s["status"]}{flag}')
    print()
    print('Unmatched turbines (outside snap distance of any target farm):', unmatched_turbines)
    print('Unmatched substations:', unmatched_subs)

    json.dump(summary, open(f'{out_dir}/_summary.json', 'w'), indent=2)
    print()
    print('by_farm/ written to', out_dir)
    return out_dir, summary


def build_data_folder(cfg, by_farm_dir):
    data_dir = os.path.join(cfg['output_dir'], 'Data')
    os.makedirs(data_dir, exist_ok=True)
    file_suffix = {'lease_area.geojson': 'polygon', 'turbines.geojson': 'turbines', 'substations.geojson': 'substations'}

    manifest = []
    for raw_name, clean_name in sorted(cfg['name_map'].items(), key=lambda kv: kv[1]):
        src_slug = slug(raw_name)
        src_dir = os.path.join(by_farm_dir, src_slug)
        if not os.path.isdir(src_dir):
            print(f'WARNING: no by_farm folder for "{raw_name}" (slug {src_slug}) - skipping in Data/')
            continue
        entry = {'name': clean_name, 'files': {}}
        for src_file, suffix in file_suffix.items():
            src_path = os.path.join(src_dir, src_file)
            if not os.path.exists(src_path):
                continue
            dest_name = f'{clean_name}_{suffix}.geojson'
            dest_path = os.path.join(data_dir, dest_name)
            shutil.copyfile(src_path, dest_path)
            with open(dest_path) as f:
                fc = json.load(f)
            entry['files'][suffix] = {'file': dest_name, 'count': len(fc['features'])}
        manifest.append(entry)

    json.dump(manifest, open(os.path.join(data_dir, 'manifest.json'), 'w'), indent=2)
    print('Data/ written to', data_dir, f'({len(manifest)} farms)')
    return data_dir, manifest


def build_cluster_map(cfg, data_dir, manifest):
    all_lons, all_lats = [], []
    farms_js = []
    for i, entry in enumerate(manifest):
        name = entry['name']
        color = PALETTE[i % len(PALETTE)]
        files = entry['files']

        def load(suffix):
            if suffix not in files:
                return 'null'
            with open(os.path.join(data_dir, files[suffix]['file'])) as f:
                return json.dumps(json.load(f))

        poly_fc = json.load(open(os.path.join(data_dir, files['polygon']['file'])))
        geom = poly_fc['features'][0]['geometry']
        rings = geom['coordinates'] if geom['type'] == 'Polygon' else [r for poly in geom['coordinates'] for r in poly]
        for ring in rings:
            for lon, lat in ring:
                all_lons.append(lon)
                all_lats.append(lat)

        farms_js.append(f'''{{
    name: {json.dumps(name)},
    color: {json.dumps(color)},
    polygon: {load('polygon')},
    turbines: {load('turbines')},
    substations: {load('substations')}
  }}''')

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    farms_js_str = ',\n  '.join(farms_js)
    region_name = cfg['region_name']

    html = f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{region_name} Offshore Wind Cluster - Data Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin:0; padding:0; height:100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background:#0d1a20; color:#e7f1ee; }}
  #app {{ display:flex; height:100vh; }}
  #map {{ flex:1; height:100%; }}
  #sidebar {{ width:300px; flex-shrink:0; background:#122630; border-left:1px solid rgba(231,241,238,0.1); overflow-y:auto; padding:16px; box-sizing:border-box; }}
  h1 {{ font-size:15px; margin:0 0 4px; }}
  .sub {{ font-size:11.5px; color:#86a0a8; margin-bottom:14px; line-height:1.4; }}
  .toggle-row {{ display:flex; align-items:center; gap:8px; padding:5px 0; font-size:12.5px; border-bottom:1px solid rgba(231,241,238,0.06); }}
  .toggle-row input {{ accent-color: #45b8a8; }}
  .swatch {{ width:10px; height:10px; border-radius:3px; flex-shrink:0; }}
  .farm-label {{ flex:1; }}
  .counts {{ font-size:10.5px; color:#6f868d; font-variant-numeric: tabular-nums; }}
  .section-title {{ font-size:10.5px; text-transform:uppercase; letter-spacing:0.06em; color:#6f868d; margin:16px 0 6px; }}
  .global-toggles {{ display:flex; flex-direction:column; gap:4px; margin-bottom:6px; }}
  .global-toggles label {{ font-size:12px; display:flex; align-items:center; gap:6px; }}
  .leaflet-popup-content {{ font-size:12.5px; line-height:1.5; }}
  .leaflet-popup-content b {{ display:block; margin-bottom:3px; }}
</style>
</head>
<body>
<div id="app">
  <div id="map"></div>
  <div id="sidebar">
    <h1>{region_name} offshore wind cluster</h1>
    <div class="sub">Boundaries, turbines and substations extracted from the WindGIS vector-tile source and dissolved across tile seams. Click any shape for details.</div>
    <div class="section-title">Show</div>
    <div class="global-toggles">
      <label><input type="checkbox" id="toggle-polygons" checked> Lease boundaries</label>
      <label><input type="checkbox" id="toggle-turbines" checked> Turbines</label>
      <label><input type="checkbox" id="toggle-substations" checked> Substations</label>
    </div>
    <div class="section-title">Farms</div>
    <div id="farm-list"></div>
  </div>
</div>
<script>
const FARMS = [
  {farms_js_str}
];
const map = L.map('map', {{ zoomControl: true }}).setView([{center_lat}, {center_lon}], 7);
const DATA_BOUNDS = L.latLngBounds([{min_lat}, {min_lon}], [{max_lat}, {max_lon}]);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
}}).addTo(map);
const groups = {{ polygons: [], turbines: [], substations: [] }};
const farmListEl = document.getElementById('farm-list');
FARMS.forEach(farm => {{
  const layerRefs = {{}};
  if (farm.polygon) {{
    const layer = L.geoJSON(farm.polygon, {{
      style: {{ color: farm.color, weight: 1.6, fillColor: farm.color, fillOpacity: 0.18 }},
      onEachFeature: (feature, lyr) => {{
        const p = feature.properties || {{}};
        lyr.bindPopup(`<b>${{farm.name}}</b>Status: ${{p.status || '-'}}<br>Turbines (declared): ${{p.no_turbine ?? '-'}}<br>Capacity: ${{p.capacity_m ?? '-'}} MW`);
      }}
    }}).addTo(map);
    groups.polygons.push(layer); layerRefs.polygon = layer;
  }}
  if (farm.turbines) {{
    const layer = L.geoJSON(farm.turbines, {{
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{ radius: 2.6, color: '#111', weight: 0.4, fillColor: farm.color, fillOpacity: 0.95 }}),
      onEachFeature: (feature, lyr) => {{
        const p = feature.properties || {{}};
        lyr.bindPopup(`<b>${{farm.name}} - turbine</b>Model: ${{p.turbinemod || '-'}}<br>Foundation: ${{p.foundation || '-'}}`);
      }}
    }}).addTo(map);
    groups.turbines.push(layer); layerRefs.turbines = layer;
  }}
  if (farm.substations) {{
    const layer = L.geoJSON(farm.substations, {{
      pointToLayer: (feature, latlng) => L.marker(latlng, {{ icon: L.divIcon({{ className: '', html: '<div style="width:10px;height:10px;background:#f2b134;border:1px solid #111;transform:rotate(45deg);"></div>', iconSize: [10, 10] }}) }}),
      onEachFeature: (feature, lyr) => {{
        const p = feature.properties || {{}};
        lyr.bindPopup(`<b>${{farm.name}} - substation</b>Name: ${{p.sub_name || '-'}}`);
      }}
    }}).addTo(map);
    groups.substations.push(layer); layerRefs.substations = layer;
  }}
  const row = document.createElement('div');
  row.className = 'toggle-row';
  const counts = [farm.turbines ? farm.turbines.features.length + ' WTG' : null, farm.substations ? farm.substations.features.length + ' sub' : null].filter(Boolean).join(' / ');
  row.innerHTML = `<input type="checkbox" checked><span class="swatch" style="background:${{farm.color}}"></span><span class="farm-label">${{farm.name}}</span><span class="counts">${{counts}}</span>`;
  farmListEl.appendChild(row);
  row.querySelector('input').addEventListener('change', (e) => {{
    Object.values(layerRefs).forEach(lyr => e.target.checked ? map.addLayer(lyr) : map.removeLayer(lyr));
  }});
}});
let didInitialFit = false;
new ResizeObserver(() => {{
  map.invalidateSize();
  if (!didInitialFit) {{ map.fitBounds(DATA_BOUNDS, {{ padding: [20, 20], animate: false }}); didInitialFit = true; }}
}}).observe(document.getElementById('map'));
function wireGlobalToggle(id, key) {{
  document.getElementById(id).addEventListener('change', (e) => groups[key].forEach(l => e.target.checked ? map.addLayer(l) : map.removeLayer(l)));
}}
wireGlobalToggle('toggle-polygons', 'polygons');
wireGlobalToggle('toggle-turbines', 'turbines');
wireGlobalToggle('toggle-substations', 'substations');
</script>
</body>
</html>
'''
    out_path = os.path.join(data_dir, 'cluster_map.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print('cluster_map.html written to', out_path)


if __name__ == '__main__':
    if 'REGION NAME HERE' in CONFIG['region_name']:
        raise SystemExit('Edit CONFIG at the top of this script for your region before running.')
    resolve_config(CONFIG)
    by_farm_dir, summary = build_by_farm(CONFIG)
    data_dir, manifest = build_data_folder(CONFIG, by_farm_dir)
    build_cluster_map(CONFIG, data_dir, manifest)
    print()
    print('Done. Remember to write/update deliverables/README.txt with any')
    print('region-specific caveats found in the summary above (missing farms,')
    print('turbine-count mismatches flagged "<-- CHECK", etc).')
