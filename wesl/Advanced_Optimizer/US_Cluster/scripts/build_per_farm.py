import json, os, re
from collections import defaultdict
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union

SC = '/private/tmp/claude-504/-Users-brunoboer-Desktop-Advanced-Optimizer/80878806-fb39-4b45-8f7e-0735ec84b10e/scratchpad'
OUT = '/Users/brunoboer/Desktop/Advanced_Optimizer/US_Cluster/deliverables/by_farm'
os.makedirs(OUT, exist_ok=True)

TARGET_FARMS = {
    'Block Island', 'South Fork Wind', 'Revolution Wind', 'Sunrise Wind',
    'Vineyard Wind 1', 'Bay State Wind (OCS-A 0500)', 'New England Wind 1',
    'New England Wind 2', 'Beacon Wind', 'SouthCoast Wind',
    'Vineyard Northeast (OCS-A 0522)',
}

data_raw = json.load(open(f'{SC}/ma_cluster_raw3.json'))
data = {
    'leaseAreas': [f for f in data_raw['leaseAreas'] if f['properties'].get('wf_name') in TARGET_FARMS],
    'turbinesFixed': data_raw['turbinesFixed'],
    'substations': data_raw['substations'],
    'interArrayCables': data_raw['interArrayCables'],
    'exportCables': data_raw['exportCables'],
}

def slug(name):
    s = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    return s

# ---------- dissolve lease-area polygon fragments per farm ----------
lease_groups = defaultdict(list)
lease_props = {}
for f in data['leaseAreas']:
    p = f['properties']
    name = p.get('wf_name')
    lease_groups[name].append(shape(f['geometry']))
    lease_props[name] = p  # keep last (all identical across fragments)

farms = {}
for name, geoms in lease_groups.items():
    merged = unary_union(geoms)
    farms[name] = {
        'slug': slug(name),
        'polygon': merged,
        'props': lease_props[name],
    }

print(f'Lease areas found: {len(farms)}')
for name in sorted(farms):
    print(' -', name)

# ---------- dedupe points, then spatial-join to farm ----------
def dedupe_points(feats):
    seen = set()
    out = []
    for f in feats:
        c = tuple(round(x, 6) for x in f['geometry']['coordinates'])
        if c in seen:
            continue
        seen.add(c)
        out.append(f)
    return out

def assign_farm(pt, max_snap_deg=0.01):
    # 1) strict containment wins, always - even if another polygon happens to be closer
    contains_matches = [name for name, info in farms.items() if info['polygon'].contains(pt)]
    if len(contains_matches) == 1:
        return contains_matches[0]
    if len(contains_matches) > 1:
        # ambiguous (shouldn't happen for disjoint lease areas) - pick the smallest containing polygon
        return min(contains_matches, key=lambda n: farms[n]['polygon'].area)
    # 2) no strict containment (point just outside its own lease boundary) - snap to nearest, capped
    best_name, best_dist = None, None
    for name, info in farms.items():
        dist = info['polygon'].distance(pt)
        if best_dist is None or dist < best_dist:
            best_name, best_dist = name, dist
    if best_dist is not None and best_dist < max_snap_deg:
        return best_name
    return None

turbines_by_farm = defaultdict(list)
unmatched_turbines = []
for f in dedupe_points(data['turbinesFixed']):
    lon, lat = f['geometry']['coordinates']
    name = assign_farm(Point(lon, lat))
    f['properties'] = {'point_type': 'turbine', **f['properties']}
    if name:
        turbines_by_farm[name].append(f)
    else:
        unmatched_turbines.append(f)

substations_by_farm = defaultdict(list)
unmatched_subs = []
for f in dedupe_points(data['substations']):
    p = f['properties']
    f['properties'] = {'point_type': 'substation', **p}
    # substations carry wf_name directly in properties - use that first
    name = p.get('wf_name')
    if name not in farms:
        lon, lat = f['geometry']['coordinates']
        name = assign_farm(Point(lon, lat))
    if name:
        substations_by_farm[name].append(f)
    else:
        unmatched_subs.append(f)

# ---------- inter-array cables: has wf_id (not wf_name) - map wf_id -> wf_name ----------

def farms_within(geom, max_dist):
    return sorted(
        [(info['polygon'].distance(geom), name) for name, info in farms.items() if info['polygon'].distance(geom) < max_dist],
        key=lambda t: t[0],
    )

INTER_ARRAY_SNAP_DEG = 0.02  # ~2 km - inter-array cables sit inside a single farm
cable_groups = defaultdict(list)  # (farm_name, wf_id) -> geoms, so fragments with same wf_id dissolve together
skipped_cables = 0
for f in data['interArrayCables']:
    geom = shape(f['geometry'])
    matches = farms_within(geom, INTER_ARRAY_SNAP_DEG)
    if not matches:
        skipped_cables += 1
        continue
    best_name = matches[0][1]  # nearest only - inter-array cables belong to one farm
    key = (best_name, f['properties'].get('wf_id'))
    cable_groups[key].append((geom, f['properties']))

inter_array_by_farm = defaultdict(list)
for (farm_name, wf_id), items in cable_groups.items():
    geoms = [g for g, _ in items]
    merged = unary_union(geoms)
    props = items[0][1]
    inter_array_by_farm[farm_name].append({'type': 'Feature', 'properties': props, 'geometry': mapping(merged)})

# ---------- export cables: dissolve by (item, manufacturer, length_m) ----------

export_groups = defaultdict(list)
for f in data['exportCables']:
    p = f['properties']
    key = (p.get('item'), p.get('manufacturer'), p.get('length_m'))
    export_groups[key].append(shape(f['geometry']))

export_dissolved = []
for key, geoms in export_groups.items():
    merged = unary_union(geoms)
    export_dissolved.append((merged, key))

EXPORT_SNAP_DEG = 0.015  # ~1.6 km - cable must actually touch/originate at the lease boundary
export_by_farm = defaultdict(list)
skipped_export = 0
for merged, key in export_dissolved:
    matches = farms_within(merged, EXPORT_SNAP_DEG)
    if not matches:
        skipped_export += 1
        continue
    shared_with = [n for _, n in matches]
    for _, farm_name in matches:
        export_by_farm[farm_name].append({
            'type': 'Feature',
            'properties': {
                'item': key[0], 'manufacturer': key[1], 'length_m': key[2],
                'shared_with_farms': [n for n in shared_with if n != farm_name] or None,
            },
            'geometry': mapping(merged),
        })

# ---------- write per-farm folders ----------
summary = []
for name, info in sorted(farms.items()):
    fslug = info['slug']
    fdir = os.path.join(OUT, fslug)
    os.makedirs(fdir, exist_ok=True)

    lease_fc = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': info['props'], 'geometry': mapping(info['polygon'])}
    ]}
    json.dump(lease_fc, open(f'{fdir}/lease_area.geojson', 'w'))

    n_turb = 0
    if turbines_by_farm.get(name):
        turb_fc = {'type': 'FeatureCollection', 'features': turbines_by_farm[name]}
        json.dump(turb_fc, open(f'{fdir}/turbines.geojson', 'w'))
        n_turb = len(turbines_by_farm[name])

    n_sub = 0
    if substations_by_farm.get(name):
        sub_fc = {'type': 'FeatureCollection', 'features': substations_by_farm[name]}
        json.dump(sub_fc, open(f'{fdir}/substations.geojson', 'w'))
        n_sub = len(substations_by_farm[name])

    n_cab = 0
    if inter_array_by_farm.get(name):
        cab_fc = {'type': 'FeatureCollection', 'features': inter_array_by_farm[name]}
        json.dump(cab_fc, open(f'{fdir}/inter_array_cables.geojson', 'w'))
        n_cab = len(inter_array_by_farm[name])

    n_exp = 0
    if export_by_farm.get(name):
        exp_fc = {'type': 'FeatureCollection', 'features': export_by_farm[name]}
        json.dump(exp_fc, open(f'{fdir}/export_cables.geojson', 'w'))
        n_exp = len(export_by_farm[name])

    summary.append({
        'name': name, 'slug': fslug, 'turbines': n_turb, 'substations': n_sub,
        'inter_array_bundles': n_cab, 'export_cables': n_exp,
        'declared_no_turbine': info['props'].get('no_turbine'),
        'status': info['props'].get('status'),
    })

print()
print(f'{"Farm":35s} {"turb":>5s} {"decl":>5s} {"sub":>4s} {"iac":>4s} {"exp":>4s}  status')
for s in summary:
    print(f'{s["name"]:35s} {s["turbines"]:5d} {str(s["declared_no_turbine"] or "-"):>5s} {s["substations"]:4d} {s["inter_array_bundles"]:4d} {s["export_cables"]:4d}  {s["status"]}')

print()
print('Unmatched turbines:', len(unmatched_turbines))
print('Unmatched substations:', len(unmatched_subs))
print('Inter-array cable fragments skipped (out of scope farms):', skipped_cables)
print('Export cable bundles skipped (out of scope farms):', skipped_export)

json.dump(summary, open(f'{OUT}/_summary.json', 'w'), indent=2)
print()
print('Done. Output at', OUT)
