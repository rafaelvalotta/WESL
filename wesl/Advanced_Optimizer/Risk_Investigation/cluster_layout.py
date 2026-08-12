"""Assembles the US cluster: real layouts where they exist, synthetic grid fill elsewhere."""
import numpy as np
import windIO
from shapely.geometry import Point, Polygon

from config import SYSTEM_YAML, REAL_LAYOUT_FARMS, SYNTHETIC_SPACING_M, EXCLUDED_FARMS

# site_us.yaml boundaries have no "name" field -- order must match the file.
BOUNDARY_NAMES = [
    "Vineyard Wind", "Revolution Wind", "SouthCoast Wind", "South Fork Wind",
    "Sunrise Wind South", "Sunrise Wind North", "Revolution Wind North",
    "Vineyard Northeast", "New England Wind 1", "New England Wind 2 South",
    "Beacon Wind", "Bay State Wind South", "Bay State Wind North",
]


def _grid_fill(polygon, spacing):
    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(xmin, xmax + spacing, spacing)
    ys = np.arange(ymin, ymax + spacing, spacing)
    pts = [(x, y) for x in xs for y in ys if polygon.contains(Point(x, y))]
    if not pts:
        return np.array([]), np.array([])
    x, y = zip(*pts)
    return np.array(x), np.array(y)


def load_cluster(spacing=SYNTHETIC_SPACING_M):
    """Returns a list of {name, x, y, n_turbines, is_synthetic, turbine_node} dicts."""
    system_dat = windIO.load_yaml(SYSTEM_YAML)
    polygons = system_dat["site"]["boundaries"]["polygons"]
    assert len(polygons) == len(BOUNDARY_NAMES), "BOUNDARY_NAMES out of sync with site_us.yaml"

    real_by_name = {f["name"]: f for f in system_dat["wind_farm"]}

    farms = []
    for name, poly_node in zip(BOUNDARY_NAMES, polygons):
        if name in EXCLUDED_FARMS:
            continue
        if name in REAL_LAYOUT_FARMS:
            farm_node = real_by_name[name]
            coords = farm_node["layouts"][0]["coordinates"]
            x, y = np.array(coords["x"]), np.array(coords["y"])
            farms.append(dict(name=name, x=x, y=y, n_turbines=len(x),
                               is_synthetic=False, turbine_node=farm_node["turbines"]))
        else:
            polygon = Polygon(zip(poly_node["x"], poly_node["y"]))
            x, y = _grid_fill(polygon, spacing)
            farms.append(dict(name=name, x=x, y=y, n_turbines=len(x),
                               is_synthetic=True, turbine_node=None))

    return farms
