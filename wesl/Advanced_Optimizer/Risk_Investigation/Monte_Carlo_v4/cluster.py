import json
from pathlib import Path

import windIO
from shapely.geometry import Polygon

import settings

BOUNDARY_NAMES = [
    "Vineyard Wind", "Revolution Wind", "SouthCoast Wind", "South Fork Wind",
    "Sunrise Wind South", "Sunrise Wind North", "Revolution Wind North",
    "Vineyard Northeast", "New England Wind 1", "New England Wind 2 South",
    "Beacon Wind", "Bay State Wind South", "Bay State Wind North",
]
REAL_LAYOUT_FARMS = {"Vineyard Wind", "Revolution Wind", "South Fork Wind", "Sunrise Wind South"}

_site_us = windIO.load_yaml(settings.DATA_DIR / "site_us.yaml")
_boundary_polygons = _site_us["boundaries"]["polygons"]
LEASE_TEMPLATES = {name: Polygon(zip(p["x"], p["y"])) for name, p in zip(BOUNDARY_NAMES, _boundary_polygons)}

_wind_farm_us = windIO.load_yaml(settings.DATA_DIR / "wind_farm_us.yaml")
_REAL_FARM_BY_NAME = {f["name"]: f for f in _wind_farm_us}
REAL_TURBINE_SPEC = {name: _REAL_FARM_BY_NAME[name]["turbines"] for name in REAL_LAYOUT_FARMS}

# scenario_2's "locked cluster" -- see scenarios.py / SCENARIOS.md. Built once by
# dev_scratch/build_scenario2_fixed_cluster.py (fixed seed, 11-13MW / 7-8D only),
# cached here, loaded identically every trial -- never redrawn per trial.
_FIXED_CLUSTER_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "scenario2_fixed_cluster.json"
_fixed_cluster_scenario2 = None  # lazy -- only scenario_2 trials need this


def _load_fixed_cluster_scenario2():
    global _fixed_cluster_scenario2
    if _fixed_cluster_scenario2 is None:
        if not _FIXED_CLUSTER_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"{_FIXED_CLUSTER_CACHE_PATH} not found -- run "
                "`python dev_scratch/build_scenario2_fixed_cluster.py` once first."
            )
        with open(_FIXED_CLUSTER_CACHE_PATH) as f:
            _fixed_cluster_scenario2 = json.load(f)
    return _fixed_cluster_scenario2


def generate_cluster(rng, scenario):
    # Always includes the 4 real-layout farms + self. In "open" mode (scenario 1)
    # that's it -- the other 8 leases are just open land for layout.py. In
    # "fixed_once" mode (scenario 2) those 8 leases are added back from the
    # cached as-built-today layout, all present from year 0.
    farms = []
    for name in BOUNDARY_NAMES:
        if name in REAL_LAYOUT_FARMS:
            node = _REAL_FARM_BY_NAME[name]
            x = node["layouts"][0]["coordinates"]["x"]
            y = node["layouts"][0]["coordinates"]["y"]
            farms.append(dict(name=name, is_self=False, has_real_layout=True,
                               arrival_year=settings.YEAR_START, turbine=name, x=x, y=y))
        elif name == scenario.self_farm:
            farms.append(dict(name=name, is_self=True, has_real_layout=False,
                               arrival_year=settings.YEAR_START, turbine=scenario.self_turbine,
                               spacing_d=scenario.self_spacing_d, polygon=LEASE_TEMPLATES[name]))

    if scenario.cluster_mode == "fixed_once":
        fixed = _load_fixed_cluster_scenario2()
        for name in BOUNDARY_NAMES:
            if name in REAL_LAYOUT_FARMS or name == scenario.self_farm:
                continue
            spec = fixed[name]
            farms.append(dict(name=name, is_self=False, has_real_layout=False,
                               arrival_year=settings.YEAR_START, turbine=spec["turbine"],
                               spacing_d=spec["spacing_d"], polygon=LEASE_TEMPLATES[name],
                               rho_target_mw_km2=spec.get("rho_target_mw_km2"),
                               spacing_clipped=spec.get("spacing_clipped")))
    elif scenario.cluster_mode != "open":
        raise ValueError(f"unknown cluster_mode {scenario.cluster_mode!r}")

    return farms
