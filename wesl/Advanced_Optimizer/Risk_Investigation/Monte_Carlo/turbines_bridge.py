"""Turbine name (as chosen by layout.py) -> full PyWake WindTurbine object.

layout.py's TURBINES dict only carries {diameter, rated_mw} -- enough for sizing a
farm's footprint, not enough to run AEP (PyWake needs the full power/Ct curve). This
module closes that gap by loading the SAME windIO turbine YAML files layout.py already
reads geometry from (Data/haliadex_13mw.yaml etc.), this time in full, and building a
real PyWake WindTurbine via the same power_curve/Ct_curve path turbines.py already uses
for the real farms -- one shared, cached object per turbine name, built once.
"""
from pathlib import Path

import yaml
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

from layout import DATA_DIR, TURBINE_YAML_FILES

_TURBINE_NODE_CACHE = {}
_TURBINE_OBJ_CACHE = {}


def load_turbine_node(name):
    """Full windIO turbine node (dict, with performance.power_curve / Ct_curve) for
    one of the names layout.py's TURBINES dict uses."""
    if name not in _TURBINE_NODE_CACHE:
        path = DATA_DIR / TURBINE_YAML_FILES[name]
        with open(path) as f:
            _TURBINE_NODE_CACHE[name] = yaml.safe_load(f)
    return _TURBINE_NODE_CACHE[name]


def build_pywake_turbine(turbine_node):
    """Same construction as turbines.py's build_pywake_turbine (kept independent here
    rather than imported, so this module has no dependency on the top-level pipeline)."""
    pc = turbine_node["performance"]["power_curve"]
    ctc = turbine_node["performance"]["Ct_curve"]
    return WindTurbine(
        name=turbine_node["name"],
        diameter=turbine_node["rotor_diameter"],
        hub_height=turbine_node["hub_height"],
        powerCtFunction=PowerCtTabular(
            ws=pc["power_wind_speeds"], power=pc["power_values"],
            power_unit="W", ct=ctc["Ct_values"],
        ),
    )


def get_turbine(name):
    """Cached PyWake WindTurbine for one of layout.py's 3 named turbines."""
    if name not in _TURBINE_OBJ_CACHE:
        _TURBINE_OBJ_CACHE[name] = build_pywake_turbine(load_turbine_node(name))
    return _TURBINE_OBJ_CACHE[name]


def warm_cache():
    """Build all 3 turbine objects up front (called once at pipeline start) so no
    per-trial cost is hidden in the first farm that happens to use each type."""
    for name in TURBINE_YAML_FILES:
        get_turbine(name)
    return dict(_TURBINE_OBJ_CACHE)
