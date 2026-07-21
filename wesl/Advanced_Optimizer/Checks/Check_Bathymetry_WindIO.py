import numpy as np
import windIO
from pathlib import Path

# Configuração de caminhos baseada na estrutura do seu MacBook
SCRIPT_DIR = Path(__file__).parent
SYSTEM_YAML_PATH = SCRIPT_DIR / "Data" / "vineyard_revolution_system.yaml"
SITE_YAML_PATH = SCRIPT_DIR / "Data" / "site_us.yaml"

if not SITE_YAML_PATH.exists():
    # Fallback caso a pasta Data esteja um nível acima ou diferente
    BASE_DIR = Path("/Users/brunoboer/Documents/Software/Test_Wesl_jun23/wesl")
    SYSTEM_YAML_PATH = BASE_DIR / "Data" / "vineyard_revolution_system.yaml"
    SITE_YAML_PATH = BASE_DIR / "Data" / "site_us.yaml"

print("======================================================================")
print("      SCRIPT DE DIAGNÓSTICO E EXPLORAÇÃO DE DADOS (WINDIO / CIVIL)    ")
print("======================================================================")

# 1. INSPEÇÃO DO SISTEMA / TURBINAS
print("\n--- [1] Inspecionando Arquivo de Sistema ---")
try:
    system_dat = windIO.load_yaml(SYSTEM_YAML_PATH)
    print(f"[Sucesso] Carregado: {SYSTEM_YAML_PATH.name}")
    print(f"Chaves principais do System YAML: {list(system_dat.keys())}")
    
    if 'wind_farm' in system_dat:
        print(f"Quantidade de parques mapeados em 'wind_farm': {len(system_dat['wind_farm'])}")
        for idx, farm in enumerate(system_dat['wind_farm']):
            print(f"  -> Parque [{idx}]: {farm.get('name', 'Sem Nome')}")
            if 'turbines' in farm:
                turb = farm['turbines']
                print(f"     Dimensões da Turbina: Hub Height = {turb.get('hub_height')}, Rotor Diameter = {turb.get('rotor_diameter')}")
            if 'layouts' in farm:
                layout = farm['layouts'][0]
                coord = layout.get('coordinates', {})
                print(f"     Layout das Turbinas: Número de X = {len(coord.get('x', []))}, Número de Y = {len(coord.get('y', []))}")
except Exception as e:
    print(f"[Erro] Falha ao ler System YAML: {e}")

# 2. INSPEÇÃO DETALHADA DA BATIMETRIA
print("\n--- [2] Inspecionando Dados de Batimetria e Site ---")
try:
    site_dat = windIO.load_yaml(SITE_YAML_PATH)
    print(f"[Sucesso] Carregado: {SITE_YAML_PATH.name}")
    print(f"Chaves principais do Site YAML: {list(site_dat.keys())}")
    
    # Tentando achar onde a batimetria está escondida (no site principal ou dentro de boundaries)
    target_nodes = [site_dat, site_dat.get('boundaries', {}), site_dat.get('site', {})]
    bathy_node = None
    node_found_name = ""
    
    for name, node in [("site_root", site_dat), ("boundaries", site_dat.get('boundaries', {})), ("site_node", site_dat.get('site', {}))]:
        if node and 'bathymetry' in node:
            bathy_node = node['bathymetry']
            node_found_name = name
            break
            
    if bathy_node is not None:
        print(f"[Achado!] Nó 'bathymetry' encontrado em: '{node_found_name}'")
        print(f"Chaves internas de 'bathymetry': {list(bathy_node.keys())}")
        
        # Analisando X
        if 'x' in bathy_node:
            x_arr = np.array(bathy_node['x'])
            print(f"  -> Campo 'x': Shape = {x_arr.shape}, Tipo = {x_arr.dtype}, Min = {x_arr.min()}, Max = {x_arr.max()}")
        # Analisando Y
        if 'y' in bathy_node:
            y_arr = np.array(bathy_node['y'])
            print(f"  -> Campo 'y': Shape = {y_arr.shape}, Tipo = {y_arr.dtype}, Min = {y_arr.min()}, Max = {y_arr.max()}")
        # Analisando Depth ou Z
        for z_key in ['z', 'depth', 'data', 'values']:
            if z_key in bathy_node:
                z_raw = bathy_node[z_key]
                if isinstance(z_raw, dict) and 'data' in z_raw:
                    z_arr = np.array(z_raw['data'])
                    print(f"  -> Campo 'bathymetry.{z_key}.data': Shape = {z_arr.shape}, Tipo = {z_arr.dtype}")
                else:
                    z_arr = np.array(z_raw)
                    print(f"  -> Campo 'bathymetry.{z_key}': Shape = {z_arr.shape}, Tipo = {z_arr.dtype}")
                print(f"     Amostra de valores de Z/Profundidade: Min = {z_arr.min()}, Max = {z_arr.max()}, Média = {z_arr.mean()}")
    else:
        print("[Aviso] Chave 'bathymetry' não encontrada diretamente. Vamos varrer recursivamente as chaves de alto nível:")
        for k, v in site_dat.items():
            if isinstance(v, dict):
                print(f"  Sub-chaves de '{k}': {list(v.keys())}")
                if 'bathymetry' in v:
                    print(f"    -> [!] Achei a batimetria dentro de {k}!")

except Exception as e:
    print(f"[Erro] Falha ao ler Site YAML ou extrair batimetria: {e}")

print("\n======================================================================")
print("Rode o script acima e me cole o output completo que aparecer no terminal!")
print("======================================================================")