import json
import pyproj

def convert_geojson_polygon_to_windio(geojson_path, farm_name):
    # Carrega o arquivo GeoJSON
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    # Extrai a lista de coordenadas do primeiro polígono
    # GeoJSON armazena como [[[long1, lat1], [long2, lat2], ...]]
    coordinates = data['features'][0]['geometry']['coordinates'][0]
    
    # Define as projeções (WGS84 para UTM Zona 19N)
    wgs84 = pyproj.CRS('EPSG:4326')
    utm19n = pyproj.CRS('EPSG:32619') # Zona 19N correspondente a essa região
    transformer = pyproj.Transformer.from_crs(wgs84, utm19n, always_xy=True)
    
    utm_x = []
    utm_y = []
    
    for long, lat in coordinates:
        x, y = transformer.transform(long, lat)
        utm_x.append(round(x, 3))
        utm_y.append(round(y, 3))
        
    # Formata a saída simulando o padrão WINDIO YAML
    yaml_output = f"        {{\n"
    yaml_output += f"            # Boundary para {farm_name}\n"
    yaml_output += f"            x: {utm_x},\n\n"
    yaml_output += f"            y: {utm_y},\n"
    yaml_output += f"            crs: \"+proj=utm +zone=19 +datum=WGS84 +units=m +no_defs\"\n"
    yaml_output += f"        }},"
    
    return yaml_output

# Processando os seus dois novos polígonos
print("--- BEACON WIND BOUNDARY ---")
print(convert_geojson_polygon_to_windio('beacon_wind.geojson', 'Beacon Wind'))

print("\n--- SOUTHCOAST WIND BOUNDARY ---")
print(convert_geojson_polygon_to_windio('southcoast_wind.geojson', 'SouthCoast Wind'))