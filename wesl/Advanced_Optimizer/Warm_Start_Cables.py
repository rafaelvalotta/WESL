# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import pickle
import time
from pathlib import Path

import numpy as np
from optiwindnet.api import MILPRouter, ModelOptions, WindFarmNetwork

from Components.lcoe_comp import LCOEComp
from Components.plotting import FarmPlotter

REQUIRED_CHECKPOINT_KEYS = (
    'x', 'y', 'boundary_vertices', 'substation_coord', 'cable_specs',
    'cost_foundations_eur', 'aep_mwh', 'farm_name', 'cycle',
)

RATED_POWER_MW = {'vineyard': 13, 'revolution': 11}


def default_milp_router():
    """
    Exact MILP configuration, adapted from Checks/Check_Optiwindnet_Version2.py.
    Uses topology='branched' (not that script's 'radial') to match EWRouter's own
    branched Esau-Williams topology -- otherwise the two routers solve different
    problems and their cable cost/length aren't comparable.
    """
    model_opts = ModelOptions(topology='branched', feeder_limit='minimum', feeder_route='segmented')
    return MILPRouter(solver_name='ortools.cp_sat', time_limit=150, mip_gap=0.005,
                       model_options=model_opts, verbose=False)


def warm_start_cable_routing(checkpoint_path, router=None, output_path=None,
                              plots_dir=None, save_plot=True):
    """
    Post-processes a saved layout-optimization checkpoint by re-routing its cable network
    with a different heuristic (by default, exact MILP) than the one the optimizer used
    internally (EWRouter, chosen there for speed across many SGD iterations). Turbine
    positions and AEP are frozen -- only cable topology, cable cost, and the LCOE term
    that depends on it are recomputed.

    Writes a new, self-contained pickle next to the source checkpoint (does not modify
    it) and returns the result dict.
    """
    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"{checkpoint_path} is missing required field(s) {missing} -- likely written "
            f"by an older version of Optimize_farm.py. Re-run the optimization to "
            f"regenerate this checkpoint before warm-starting it."
        )

    router = router if router is not None else default_milp_router()

    turbines_coord = np.column_stack((data['x'], data['y']))
    wfn = WindFarmNetwork(
        turbinesC=turbines_coord,
        substationsC=data['substation_coord'],
        cables=data['cable_specs'],
        borderC=data['boundary_vertices'],
    )
    wfn.merge_obstacles_into_border()
    wfn.add_buffer(buffer_dist=0.1)

    start = time.time()
    wfn.optimize(router=router)
    routing_runtime_s = time.time() - start

    cost_cables_eur = float(wfn.cost())
    cable_network = {
        'edges': wfn.get_network(),
        'length_m': float(wfn.length()),
        'cost_eur': cost_cables_eur,
    }

    # AEP and foundation cost don't depend on cable routing, so they carry over unchanged;
    # only the cable-cost term of the LCOE formula is recomputed. compute_lcoe() mirrors
    # LCOEComp.compute()'s formula without needing a full OpenMDAO Problem.setup().
    rated_power_kw = float(RATED_POWER_MW[data['farm_name']]) * 1000.0
    lcoe_comp = LCOEComp(n_wt=len(data['x']), rated_power_kw=rated_power_kw)
    lcoe_eur_mwh = lcoe_comp.compute_lcoe(
        aep_mwh=data['aep_mwh'],
        cost_foundations=data['cost_foundations_eur'],
        cost_cables=cost_cables_eur,
    )

    result = {
        # Carried over unchanged from the source checkpoint -- cable routing doesn't affect these
        'farm_name': data['farm_name'],
        'cycle': data['cycle'],
        'x': data['x'],
        'y': data['y'],
        'aep_mwh': data['aep_mwh'],
        'cost_foundations_eur': data['cost_foundations_eur'],
        'boundary_vertices': data['boundary_vertices'],
        'substation_coord': data['substation_coord'],
        'cable_specs': data['cable_specs'],
        'min_spacing': data['min_spacing'],
        'bathy_nc_path': data['bathy_nc_path'],
        'source_checkpoint': str(checkpoint_path),

        # Refined cable routing result
        'router': type(router).__name__,
        'cost_cables_eur': cost_cables_eur,
        'cable_network': cable_network,
        'lcoe_eur_mwh': lcoe_eur_mwh,
        'routing_runtime_s': routing_runtime_s,

        # Side-by-side reference: what the optimizer's own EWRouter produced
        'baseline_router': 'EWRouter',
        'baseline_cost_cables_eur': data.get('cost_cables_eur'),
        'baseline_cable_network': data.get('cable_network'),
        'baseline_lcoe_eur_mwh': data.get('lcoe_eur_mwh'),

        'timestamp': time.time(),
    }

    if output_path is None:
        output_path = checkpoint_path.with_name(checkpoint_path.stem + "_milp.pkl")
    output_path = Path(output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"=== WARM-START CABLE ROUTING: {result['farm_name'].upper()} / cycle {result['cycle']} ===")
    print(f"Router: {result['router']} | Runtime: {routing_runtime_s:.2f} s")
    if result['baseline_cost_cables_eur'] is not None:
        delta_len = result['baseline_cable_network']['length_m'] - cable_network['length_m']
        delta_cost = result['baseline_cost_cables_eur'] - cost_cables_eur
        print(f"vs. EWRouter baseline -> Length: {delta_len:+,.2f} m | Cost: {delta_cost:+,.2f} EUR")
    print(f"Cable cost: {cost_cables_eur:,.2f} EUR | Cable length: {cable_network['length_m']:,.2f} m")
    print(f"LCOE: {lcoe_eur_mwh:.4f} EUR/MWh (AEP frozen at {data['aep_mwh']:,.2f} MWh)")
    print(f"[Exported] Warm-start checkpoint saved to: {output_path}")

    if save_plot:
        if plots_dir is None:
            plots_dir = output_path.parent / "plots" / result['farm_name']
        plotter = FarmPlotter(
            farm_name=result['farm_name'],
            boundary_vertices=result['boundary_vertices'],
            substation_coord=result['substation_coord'],
            cable_specs=result['cable_specs'],
            bathy_nc_path=result['bathy_nc_path'],
            output_dir=plots_dir,
            min_spacing=result['min_spacing'],
        )
        plot_path = plotter.save_iteration_plot(
            cycle=result['cycle'], iteration=0,
            turbine_x=result['x'], turbine_y=result['y'],
            wfn=wfn,
            lcoe=lcoe_eur_mwh, aep=data['aep_mwh'],
            tag='milp_final',
            label=f"MILP Warm-Start (Cycle {result['cycle']})",
        )
        result['plot_path'] = str(plot_path)
        print(f"[Exported] Final plot saved to: {plot_path}")

    return result


if __name__ == "__main__":
    RESULTS_DIR = Path("Results_Nash_Run")
    checkpoints = sorted(
        p for p in RESULTS_DIR.glob("checkpoint_*_cycle_*.pkl")
        if not p.stem.endswith("_milp")
    )

    if not checkpoints:
        print(f"No checkpoints found in {RESULTS_DIR}. Run Optimize_farm.py first.")
    else:
        for ckpt in checkpoints:
            warm_start_cable_routing(ckpt)
