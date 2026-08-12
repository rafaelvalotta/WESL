"""Connects layout.py's scenario output to the AEP engine's expected schema.

layout.py's `populate_scenario` returns cluster_farms (13-lease cluster, real farms with
a full turbine_node, infill leases with just a turbine_name) and new_farms (speculative
farms, also just a turbine_name). Neither is directly usable by aep_simulation's
run_scenarios, which needs one PyWake turbine TYPE per distinct turbine and a
{farm_name: type_index} map. This module does that assembly, and is the one place that
logs the turbine assignment actually used per farm in a given trial -- needed later to
explain (not just report) an AEP outcome.
"""
from py_wake.wind_turbines import WindTurbines

import turbines_bridge


def build_aep_farms(cluster_farms, new_farms):
    """Flatten layout.py's two farm collections into the single list aep_simulation-style
    functions expect: {name, x, y, n_turbines, is_synthetic, turbine_node, turbine_name}.
    Real farms carry turbine_node (full curve, already in wind_farm_us.yaml); every
    infill/speculative farm carries turbine_name instead, resolved by turbines_bridge."""
    farms = []
    for f in cluster_farms:
        farms.append(dict(
            name=f["name"], x=f["x"], y=f["y"], n_turbines=f["n_turbines"],
            is_synthetic=f["is_synthetic"], turbine_node=f.get("turbine_node"),
            turbine_name=f.get("turbine_name"),
        ))
    for i, f in enumerate(new_farms):
        farms.append(dict(
            name=f"Speculative_{i:02d}", x=f["pts"][:, 0], y=f["pts"][:, 1],
            n_turbines=len(f["pts"]), is_synthetic=True,
            turbine_node=None, turbine_name=f["turbine"],
        ))
    return farms


def build_turbine_types(farms, log_rows=None):
    """One PyWake turbine type per real farm's own node, plus one shared type per unique
    turbine_name (not one type per farm -- many infill/speculative farms reuse the same
    3 named turbines, and PyWake's type list should stay small). Returns
    (WindTurbines, type_by_farm), same interface aep_simulation.build_turbine_types had,
    now sourced correctly for infill and speculative farms too (v1/v2 of the top-level
    pipeline only ever had ONE shared synthetic type, hardcoded -- see config.py's old
    SYNTHETIC_TURBINE_SPEC; this replaces that assumption).

    log_rows, if passed, gets one dict per farm: which turbine object it actually got,
    for auditability (see the earlier instrumentation discussion -- this is the piece
    that lets an AEP result be traced back to *which* turbine was on *which* farm).
    """
    wt_objects = []
    type_by_farm = {}
    name_to_type = {}

    for farm in farms:
        if farm.get("turbine_node") is not None:
            wt_objects.append(turbines_bridge.build_pywake_turbine(farm["turbine_node"]))
            type_by_farm[farm["name"]] = len(wt_objects) - 1
            used_name = farm["turbine_node"]["name"]
        else:
            tname = farm["turbine_name"]
            if tname not in name_to_type:
                wt_objects.append(turbines_bridge.get_turbine(tname))
                name_to_type[tname] = len(wt_objects) - 1
            type_by_farm[farm["name"]] = name_to_type[tname]
            used_name = tname

        if log_rows is not None:
            log_rows.append(dict(
                farm_name=farm["name"], is_synthetic=farm["is_synthetic"],
                n_turbines=farm["n_turbines"], turbine_used=used_name,
            ))

    return WindTurbines.from_WindTurbine_lst(wt_objects), type_by_farm
