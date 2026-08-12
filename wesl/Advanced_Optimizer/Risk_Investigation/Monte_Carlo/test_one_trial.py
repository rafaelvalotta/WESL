"""One full Monte Carlo trial, end to end -- climate draw -> layout draw -> turbine
bridge -> AEP under both wake models -- to validate the whole chain actually connects,
and as a template for orchestrator.py's per-trial body.

    conda run -n Wind_2200 python test_one_trial.py
"""
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent))  # Risk_Investigation/ -- for aep_simulation, config

import climate
import layout
import scenario_bridge
import wake_models
import aep_simulation
from config import FOCUS_FARM


def run_one_trial(seed, pd_level="high"):
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # 1. Climate draw
    site = climate.sample_synthetic_site(rng, scale=1.0)
    climate_info = climate.climate_summary(np.random.default_rng(seed), scale=1.0)
    t_climate = time.time()

    # 2. Layout draw
    layout_log = []
    cluster_x, cluster_y, cluster_farms, new_farms = layout.populate_scenario(pd_level, rng, log_rows=layout_log)
    t_layout = time.time()

    # 3. Bridge to AEP schema + turbine types
    aep_farms = scenario_bridge.build_aep_farms(cluster_farms, new_farms)
    turbine_log = []
    multi_turbines, type_by_farm = scenario_bridge.build_turbine_types(aep_farms, log_rows=turbine_log)
    t_bridge = time.time()

    # 4. AEP under BOTH wake models (not a random pick -- see wake_models.py)
    aep_results = {}
    for model_name, wfm_builder in wake_models.WFM_BUILDERS.items():
        result = aep_simulation.run_scenarios(
            site, aep_farms, multi_turbines, type_by_farm,
            focus_farm_name=FOCUS_FARM, wfm_builder=wfm_builder,
        )
        aep_results[model_name] = dict(
            isolated_aep_gwh=result["isolated_aep_gwh"],
            full_cluster_aep_gwh=result["full_cluster_aep_gwh"],
            full_cluster_loss_pct=result["full_cluster_loss_pct"],
        )
    t_aep = time.time()

    n_new_farms = len(new_farms)
    requested_mw = sum(r["target_mw"] for r in layout_log)
    delivered_mw = sum(r["delivered_mw"] for r in layout_log)

    print(f"seed={seed} pd_level={pd_level}")
    print(f"  climate: mean_speed={climate_info['overall_mean_speed']:.2f} m/s, "
          f"dominant_sector={climate_info['dominant_sector']} (freq={climate_info['dominant_sector_freq']:.3f})")
    print(f"  layout: {n_new_farms} new farms, {delivered_mw:.0f}/{requested_mw:.0f} MW delivered, "
          f"{sum(f['n_turbines'] for f in aep_farms)} turbines total across {len(aep_farms)} farms")
    print(f"  turbines used: {sorted(set(r['turbine_used'] for r in turbine_log))}")
    for model_name, r in aep_results.items():
        print(f"  [{model_name}] {FOCUS_FARM} AEP = {r['full_cluster_aep_gwh']:.0f} GWh/yr "
              f"(loss {r['full_cluster_loss_pct']:.2f}% vs. isolated)")
    print(f"  timing: climate={t_climate-t0:.1f}s layout={t_layout-t_climate:.1f}s "
          f"bridge={t_bridge-t_layout:.1f}s aep(x2 models)={t_aep-t_bridge:.1f}s "
          f"total={t_aep-t0:.1f}s")

    return dict(
        seed=seed, pd_level=pd_level, climate=climate_info,
        n_new_farms=n_new_farms, requested_mw=requested_mw, delivered_mw=delivered_mw,
        aep_results=aep_results, layout_log=layout_log, turbine_log=turbine_log,
        timing=dict(climate=t_climate-t0, layout=t_layout-t_climate,
                    bridge=t_bridge-t_layout, aep=t_aep-t_bridge, total=t_aep-t0),
    )


if __name__ == "__main__":
    for seed in [0, 1, 2]:
        run_one_trial(seed)
        print()
