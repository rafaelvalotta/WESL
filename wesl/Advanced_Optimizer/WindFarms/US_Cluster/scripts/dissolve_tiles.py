"""
Reconstroi poligonos/linhas cortados nas bordas dos vector tiles (Mapbox GL),
a partir do JSON bruto salvo pelo `extract_windgis_layers.js`
(via downloadExtracted() ou copiando window.__extracted do console).

Uso:
    python3 dissolve_tiles.py windgis_extracted_raw.json --key wf_name --out saida/

Requer: shapely (pip install shapely)

O problema que isso resolve: quando uma lease area (ou um cabo) atravessa mais
de um vector tile na zoom em que voce coletou os dados, `map.querySourceFeatures()`
retorna um fragmento por tile, todos com as MESMAS propriedades mas geometrias
diferentes (recortadas na borda do tile). Se voce so pegar o primeiro fragmento
de cada nome, vai faltar pedaco do poligono; se plotar todos sem juntar, vai
parecer que tem varios poligonos sobrepostos/vizinhos quando na verdade e um so.
Este script agrupa os fragmentos por uma chave (ex: nome da lease area) e faz
um unary_union pra devolver a geometria unica e continua.
"""
import argparse
import json
from collections import defaultdict

from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def dissolve_by_key(features, key_fields):
    groups = defaultdict(list)
    props_by_key = {}
    for f in features:
        p = f["properties"]
        key = tuple(p.get(k) for k in key_fields)
        groups[key].append(shape(f["geometry"]))
        props_by_key.setdefault(key, p)

    out = []
    for key, geoms in groups.items():
        merged = unary_union(geoms)
        out.append({"type": "Feature", "properties": props_by_key[key], "geometry": mapping(merged)})
    return {"type": "FeatureCollection", "features": out}


def dedupe_points(features, precision=6):
    seen = set()
    out = []
    for f in features:
        coords = tuple(round(c, precision) for c in f["geometry"]["coordinates"])
        if coords in seen:
            continue
        seen.add(coords)
        out.append(f)
    return {"type": "FeatureCollection", "features": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_json", help="JSON bruto exportado do console (window.__extracted)")
    ap.add_argument(
        "--key",
        action="append",
        default=[],
        help="Campo de propriedade usado para agrupar/dissolver poligonos e linhas "
        "(pode repetir --key para usar mais de um campo, ex: --key wf_name --key cod). "
        "Padrao: tenta 'wf_name', senao 'name', senao 'id'.",
    )
    ap.add_argument("--out", default="dissolved_output", help="pasta de saida")
    args = ap.parse_args()

    data = json.load(open(args.input_json))
    key_fields = args.key or None

    import os
    os.makedirs(args.out, exist_ok=True)

    for layer_id, fc in data.items():
        feats = fc["features"]
        if not feats:
            continue
        geom_type = feats[0]["geometry"]["type"]

        if geom_type in ("Polygon", "MultiPolygon", "LineString", "MultiLineString"):
            fields = key_fields
            if not fields:
                sample_props = feats[0]["properties"]
                for candidate in ("wf_name", "name", "id", "wf_id"):
                    if candidate in sample_props:
                        fields = [candidate]
                        break
            if not fields:
                print(f"[{layer_id}] sem campo-chave obvio nas propriedades; pulando dissolve, "
                      f"salvando fragmentos crus. Rode de novo com --key <campo>.")
                result = fc
            else:
                result = dissolve_by_key(feats, fields)
                print(f"[{layer_id}] {len(feats)} fragmentos -> {len(result['features'])} features dissolvidas (key={fields})")
        elif geom_type == "Point":
            result = dedupe_points(feats)
            print(f"[{layer_id}] {len(feats)} pontos -> {len(result['features'])} apos dedupe")
        else:
            result = fc

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in layer_id)
        out_path = os.path.join(args.out, f"{safe_name}.geojson")
        json.dump(result, open(out_path, "w"))
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
