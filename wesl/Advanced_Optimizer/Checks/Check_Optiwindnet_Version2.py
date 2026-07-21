import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import windIO
from shapely.geometry import Point, Polygon
from optiwindnet.api import WindFarmNetwork, EWRouter, MILPRouter, ModelOptions

BASE_DIR = Path("/Users/brunoboer/Documents/Software/Test_Wesl_jun23/wesl/Advanced_Optimizer")
SYSTEM_YAML_PATH = BASE_DIR / "Data" / "vineyard_revolution_system.yaml"
SITE_YAML_PATH = BASE_DIR / "Data" / "site_us.yaml"

if not SITE_YAML_PATH.exists():
    BASE_DIR = Path("/Users/brunoboer/Documents/Software/Test_Wesl_jun23/wesl")
    SYSTEM_YAML_PATH = BASE_DIR / "Data" / "vineyard_revolution_system.yaml"
    SITE_YAML_PATH = BASE_DIR / "Data" / "site_us.yaml"

cable_specs = np.array(
    list(zip([3, 5, 7], [368.9, 428.9, 737.1])), 
    dtype=[("capacity", int), ("cost", float)]
)

site_dat = windIO.load_yaml(SITE_YAML_PATH)
revolution_poly_dat = site_dat['boundaries']['polygons'][0]
boundary_vertices = np.column_stack((revolution_poly_dat['x'], revolution_poly_dat['y']))
boundary_closed = np.vstack([boundary_vertices, boundary_vertices[0]])

shapely_boundary = Polygon(boundary_vertices)

system_dat = windIO.load_yaml(SYSTEM_YAML_PATH)
revolution_data = system_dat['wind_farm'][0]

x_raw = np.array(revolution_data['layouts'][0]['coordinates']['x'], dtype=float)
y_raw = np.array(revolution_data['layouts'][0]['coordinates']['y'], dtype=float)

x_filtrado, y_filtrado = [], []
for x_val, y_val in zip(x_raw, y_raw):
    ponto_wt = Point(x_val, y_val)
    if shapely_boundary.contains(ponto_wt) or ponto_wt.distance(shapely_boundary) < 1.0:
        x_filtrado.append(x_val)
        y_filtrado.append(y_val)

# turbines_coord = np.column_stack((x_filtrado, y_filtrado))
turbines_coord = np.column_stack((x_raw, y_raw))


substations_list = []
electrical_substations = revolution_data.get('electrical_substations', [])
for sub_item in electrical_substations:
    sub_coords = sub_item['electrical_substation']['coordinates']
    sub_x = float(sub_coords['x'][0])
    sub_y = float(sub_coords['y'][0])
    substations_list.append([sub_x, sub_y])

substation_coord = np.array(substations_list, dtype=float)

wfn_ew = WindFarmNetwork(
    turbinesC=turbines_coord,
    substationsC=substation_coord,
    cables=cable_specs,
    borderC=boundary_vertices
)
wfn_ew.merge_obstacles_into_border()
wfn_ew.add_buffer(buffer_dist=0.1)

start_ew = time.time()
wfn_ew.optimize(router=EWRouter())
runtime_ew = time.time() - start_ew
length_ew = wfn_ew.length()
cost_ew = wfn_ew.cost()

edges_ew = wfn_ew.get_network()
vertex_matrix_ew = wfn_ew.G.graph['VertexC']
node_mapper_ew = wfn_ew.G.graph['fnT']
pos_dict_ew = {}
for node in wfn_ew.G.nodes():
    pos_dict_ew[node] = (vertex_matrix_ew[node_mapper_ew[node], 0], vertex_matrix_ew[node_mapper_ew[node], 1])

wfn_milp = WindFarmNetwork(
    turbinesC=turbines_coord,
    substationsC=substation_coord,
    cables=cable_specs,
    borderC=boundary_vertices
)
wfn_milp.merge_obstacles_into_border()
wfn_milp.add_buffer(buffer_dist=0.1)

model_opts = ModelOptions(topology='radial', feeder_limit='minimum', feeder_route='segmented')
router_milp = MILPRouter(solver_name='ortools.cp_sat', time_limit=90, mip_gap=0.005, model_options=model_opts, verbose=False)

start_milp = time.time()
wfn_milp.optimize(router=router_milp)
runtime_milp = time.time() - start_milp
length_milp = wfn_milp.length()
cost_milp = wfn_milp.cost()

edges_milp = wfn_milp.get_network()
vertex_matrix_milp = wfn_milp.G.graph['VertexC']
node_mapper_milp = wfn_milp.G.graph['fnT']
pos_dict_milp = {}
for node in wfn_milp.G.nodes():
    pos_dict_milp[node] = (vertex_matrix_milp[node_mapper_milp[node], 0], vertex_matrix_milp[node_mapper_milp[node], 1])

print("\n=======================================================")
print("          ROUTING METHOD BENCHMARK SUMMARY             ")
print("=======================================================")
print(f"EW Heuristic   -> Runtime: {runtime_ew:.4f} s | Total Length: {length_ew:,.2f} m | Total Cost: {cost_ew:,.2f} €")
print(f"Exact MILP     -> Runtime: {runtime_milp:.4f} s | Total Length: {length_milp:,.2f} m | Total Cost: {cost_milp:,.2f} €")
print(f"Delta Savings  -> Length Reduction: {length_ew - length_milp:,.2f} m | Cost Reduction: {cost_ew - cost_milp:,.2f} €")
print("=======================================================\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

ax1.plot(boundary_closed[:, 0], boundary_closed[:, 1], color='black', linestyle='--', linewidth=2.0, zorder=1)
ax1.fill(boundary_vertices[:, 0], boundary_vertices[:, 1], color='grey', alpha=0.04, zorder=0)
ax1.scatter(turbines_coord[:, 0], turbines_coord[:, 1], color='deepskyblue', s=55, edgecolor='black', zorder=4)

for edge in edges_ew:
    x0, y0 = pos_dict_ew[int(edge['src'])]
    x1, y1 = pos_dict_ew[int(edge['tgt'])]
    c_type = int(edge['cable'])
    line_color, line_width = ('green', 1.5) if c_type == 0 else (('orange', 2.5) if c_type == 1 else ('red', 4.0))
    ax1.plot([x0, x1], [y0, y1], color=line_color, linewidth=line_width, zorder=3)

for idx, (sub_x, sub_y) in enumerate(substation_coord):
    ax1.scatter(sub_x, sub_y, color='gold', marker='*', s=400, edgecolor='black', zorder=5)
    ax1.text(sub_x + 150, sub_y + 150, f"OSS {idx}", fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=6)

ax1.set_title(f"EW Heuristic Layout\nLength: {length_ew:,.2f} m | Cost: {cost_ew:,.2f} € | Time: {runtime_ew:.3f} s", fontsize=11, fontweight='bold')
ax1.set_xlabel("UTM Easting (m)")
ax1.set_ylabel("UTM Northing (m)")
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.plot(boundary_closed[:, 0], boundary_closed[:, 1], color='black', linestyle='--', linewidth=2.0, zorder=1)
ax2.fill(boundary_vertices[:, 0], boundary_vertices[:, 1], color='grey', alpha=0.04, zorder=0)
ax2.scatter(turbines_coord[:, 0], turbines_coord[:, 1], color='deepskyblue', s=55, edgecolor='black', zorder=4)

for edge in edges_milp:
    x0, y0 = pos_dict_milp[int(edge['src'])]
    x1, y1 = pos_dict_milp[int(edge['tgt'])]
    c_type = int(edge['cable'])
    line_color, line_width = ('green', 1.5) if c_type == 0 else (('orange', 2.5) if c_type == 1 else ('red', 4.0))
    ax2.plot([x0, x1], [y0, y1], color=line_color, linewidth=line_width, zorder=3)

for idx, (sub_x, sub_y) in enumerate(substation_coord):
    ax2.scatter(sub_x, sub_y, color='gold', marker='*', s=400, edgecolor='black', zorder=5)
    ax2.text(sub_x + 150, sub_y + 150, f"OSS {idx}", fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=6)

ax2.set_title(f"Exact MILP Layout\nLength: {length_milp:,.2f} m | Cost: {cost_milp:,.2f} € | Time: {runtime_milp:.3f} s", fontsize=11, fontweight='bold')
ax2.set_xlabel("UTM Easting (m)")
ax2.set_ylabel("UTM Northing (m)")
ax2.grid(True, linestyle="--", alpha=0.4)

from matplotlib.lines import Line2D
custom_legend = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label='Lease Boundary'),
    Line2D([0], [0], color='deepskyblue', marker='o', linestyle='', markersize=8, markeredgecolor='black', label='Turbines'),
    Line2D([0], [0], color='gold', marker='*', linestyle='', markersize=14, markeredgecolor='black', label='Offshore Substations'),
    Line2D([0], [0], color='green', linewidth=1.5, label='Cable Type 0 (Cap: 3)'),
    Line2D([0], [0], color='orange', linewidth=2.5, label='Cable Type 1 (Cap: 5)'),
    Line2D([0], [0], color='red', linewidth=4.0, label='Cabo Type 2 (Cap: 7)')
]
ax2.legend(handles=custom_legend, loc="upper right")

plt.tight_layout()
plt.show()