import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import windIO
from shapely.geometry import Point, Polygon  # Essencial para o filtro geométrico

# Importações da API do optiwindnet mapeadas no seu site-packages
from optiwindnet.api import WindFarmNetwork, EWRouter

SCRIPT_DIR = Path(__file__).parent
SYSTEM_YAML_PATH = SCRIPT_DIR / "Data" / "vineyard_revolution_system.yaml"
SITE_YAML_PATH = SCRIPT_DIR / "Data" / "site_us.yaml"

print("=== 1. PREPARANDO MATRIZ DE CABOS ===")
cable_specs = np.array(
    list(zip([3, 5, 7], [368.9, 428.9, 737.1])), 
    dtype=[("capacity", int), ("cost", float)]
)

print("\n=== 2. CARREGANDO E FILTRANDO GEOMETRIAS (FILTRO DE BOUNDARY ATIVO) ===")
# A. Carrega a Boundary Oficial e Complexa do Revolution Wind (Índice 1)
site_dat = windIO.load_yaml(SITE_YAML_PATH)
revolution_poly_dat = site_dat['boundaries']['polygons'][1]
boundary_vertices = np.column_stack((revolution_poly_dat['x'], revolution_poly_dat['y']))
boundary_closed = np.vstack([boundary_vertices, boundary_vertices[0]])

# Criamos o objeto polígono do Shapely para fazer o teste de inclusão rígido
shapely_boundary = Polygon(boundary_vertices)

# B. Carrega o Layout de Fábrica Inicial (Revolution + South Fork)
system_dat = windIO.load_yaml(SYSTEM_YAML_PATH)
revolution_data = system_dat['wind_farm'][1]
print(f"Parque Escolhido: {revolution_data['name']}")

x_raw = np.array(revolution_data['layouts'][0]['coordinates']['x'], dtype=float)
y_raw = np.array(revolution_data['layouts'][0]['coordinates']['y'], dtype=float)
print(f"Total de turbinas carregadas do arquivo bruto: {len(x_raw)}")

# C. EXECUÇÃO DO FILTRO GEOMÉTRICO SEPARADO
x_filtrado, y_filtrado = [], []
for x_val, y_val in zip(x_raw, y_raw):
    ponto_wt = Point(x_val, y_val)
    if shapely_boundary.contains(ponto_wt) or ponto_wt.distance(shapely_boundary) < 1.0:
        x_filtrado.append(x_val)
        y_filtrado.append(y_val)

turbines_coord = np.column_stack((x_filtrado, y_filtrado))
n_vwt = len(turbines_coord)
print(f"--> [Filtro Aplicado] Turbinas DENTRO da Boundary: {n_vwt}")
print(f"--> [Filtro Aplicado] Turbinas DESCARTADAS (South Fork/Fora): {len(x_raw) - n_vwt}")

# CORREÇÃO CRÍTICA: Extração dinâmica de TODAS as subestações do YAML (Lista completa)
substations_list = []
electrical_substations = revolution_data.get('electrical_substations', [])

print("Mapeando subestações encontradas no YAML...")
for idx, sub_item in enumerate(electrical_substations):
    sub_coords = sub_item['electrical_substation']['coordinates']
    sub_x = float(sub_coords['x'][0])
    sub_y = float(sub_coords['y'][0])
    substations_list.append([sub_x, sub_y])
    print(f"--> [OSS {idx}] Carregada em X: {sub_x}, Y: {sub_y}")

substation_coord = np.array(substations_list, dtype=float)

print("\n=== 3. INICIALIZANDO O OBJETO COM A BOUNDARY RÍGIDA ATIVA ===")
# Inicializamos passando a cerca e a matriz dinâmica com as subestações ativas
wfn = WindFarmNetwork(
    turbinesC=turbines_coord,
    substationsC=substation_coord,
    cables=cable_specs,
    borderC=boundary_vertices
)

print("Limpando e fundindo restrições geométricas de borda...")
wfn.merge_obstacles_into_border()

# Ajuste micrométrico de tolerância para ponto flutuante
wfn.add_buffer(buffer_dist=0.1)

print("\n=== 4. EXECUTANDO O RESOLVEDOR DE MALHA PLANAR ATIVA ===")
wfn.optimize()

total_cost = wfn.cost()
total_length = wfn.length()
print(f"--> Comprimento Total com Desvios: {total_length:,.2f} metros")
print(f"--> Custo com Desvios de Cerca:   ${total_cost:,.2f} USD")

print("\n=== 5. EXIBINDO O MAPA DO SITE E REDE ELÉTRICA (AUDITORIA DE BORDAS) ===")
edges = wfn.get_network()

# CORREÇÃO CRÍTICA DO PLOT: Mapeamento blindado pelas coordenadas mestre do solver
vertex_matrix = wfn.G.graph['VertexC']  # Contém as posições reais de todos os nós (reais e quinas)
node_mapper = wfn.G.graph['fnT']       # Tradutor de IDs do NetworkX para os índices da matriz

pos_dict = {}
for node in wfn.G.nodes():
    matrix_index = node_mapper[node]
    pos_dict[node] = (vertex_matrix[matrix_index, 0], vertex_matrix[matrix_index, 1])

plt.figure(figsize=(13, 11))

# Desenha a Fronteira Oficial recortada
plt.plot(boundary_closed[:, 0], boundary_closed[:, 1], color='black', linestyle='--', linewidth=2.2, label='Fronteira Oficial (site_us.yaml)', zorder=1)
plt.fill(boundary_vertices[:, 0], boundary_vertices[:, 1], color='grey', alpha=0.04, zorder=0)

# Desenha as turbinas válidas filtradas
plt.scatter(turbines_coord[:, 0], turbines_coord[:, 1], color='deepskyblue', s=65, edgecolor='black', label='Turbinas Ativas Internas', zorder=4)

# Desenha os cabos elétricos calculados sob restrição de parede sem distorções lineares
for edge in edges:
    src_idx = int(edge['src'])
    tgt_idx = int(edge['tgt'])
    cable_type = edge['cable']
    
    # Pesca a coordenada exata traduzida da matriz mestre do grafo
    x0, y0 = pos_dict[src_idx]
    x1, y1 = pos_dict[tgt_idx]
    
    if cable_type == 0:
        line_color, line_width = 'green', 1.5
    elif cable_type == 1:
        line_color, line_width = 'orange', 2.5
    else:
        line_color, line_width = 'red', 4.0
        
    plt.plot([x0, x1], [y0, y1], color=line_color, linewidth=line_width, zorder=3)

# Desenha TODAS as Subestações Offshore mapeadas na tela como estrelas douradas
for idx, (sub_x, sub_y) in enumerate(substation_coord):
    plt.scatter(sub_x, sub_y, color='gold', marker='*', s=500, edgecolor='black', 
                label='Subestação Offshore (OSS)' if idx == 0 else "", zorder=5)
    plt.text(sub_x + 150, sub_y + 150, f"OSS {idx}", fontsize=10, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), zorder=6)

plt.title(f"Topologia Otimizada Corrigida - {revolution_data['name']}\nComprimento Total da Rede: {total_length:,.2f} m", fontsize=12, fontweight='bold')
plt.xlabel("UTM Easting (m)")
plt.ylabel("UTM Northing (m)")
plt.grid(True, linestyle="--", alpha=0.4)

from matplotlib.lines import Line2D
custom_legend = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=2.2, label='Fronteira Oficial (site_us.yaml)'),
    Line2D([0], [0], color='deepskyblue', marker='o', linestyle='', markersize=8, markeredgecolor='black', label='Turbina Válida (Filtrada)'),
    Line2D([0], [0], color='gold', marker='*', linestyle='', markersize=15, markeredgecolor='black', label='Subestação Offshore'),
    Line2D([0], [0], color='green', linewidth=1.5, label='Cabo Tipo 0 (Capacidade: 3)'),
    Line2D([0], [0], color='orange', linewidth=2.5, label='Cabo Tipo 1 (Capacidade: 5)'),
    Line2D([0], [0], color='red', linewidth=4.0, label='Cabo Tipo 2 (Capacidade: 7)')
]
plt.legend(handles=custom_legend, loc="upper right")

print("Abrindo a janela gráfica com o layout filtrado, protegido e multi-OSS ativo...")
plt.show()