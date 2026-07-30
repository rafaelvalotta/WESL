import numpy as np
import openmdao.api as om
import pickle
import time
from pathlib import Path

from Components.constraints import BoundaryComp, SpacingCompWithNeighbors, ConstraintAggregationComp
from Components.aep import AEPComp
from Components.foundations import FoundationsComp, mean_wind_speed
from Components.lcoe_comp import LCOEComp
from Components.cable_wrapper import OptiWindNetComponent
from Components.Drivers import SGD
from Components.plotting import FarmPlotter

def build_farm_model(farm_name, n_wt, site_yaml_path, system_yaml_path, bathy_nc_path,
                     rated_power, ip_platform, rotor_diameter, hub_height, wind_speed,
                     neighbor_x, neighbor_y, boundary_vertices,
                     substation_coord, cable_specs, min_spacing,
                     unconstrained_mode=False, stochastic_mode=False, sample_size=1,
                     farm_index=0, neighbor_farm_idx=None):
    """
    Assembles the MDO OpenMDAO Group connecting physics, economics, and constraints.

    farm_index: this farm's index into system_yaml_path['wind_farm'] -- needed so AEPComp
        picks its OWN turbine model correctly (farms can carry different turbines, e.g.
        Vineyard's Haliade-X 13MW vs Revolution's SG11-200 11MW).
    neighbor_farm_idx: same length as neighbor_x/neighbor_y, farm index each neighbor
        position belongs to, so wakes cast by neighbors use THEIR turbine, not this farm's.
    """
    model = om.Group()

    model.add_subsystem(
        'aep_comp',
        AEPComp(
            n_wt=n_wt,
            system_data=system_yaml_path,
            site_data=site_yaml_path,
            stochastic_mode=stochastic_mode,  # Explicit control over PyWake mini-batch sampling
            sample_size=sample_size,          # Number of (wd, ws) draws per mini-batch iteration
            x_offset=float(boundary_vertices[:, 0].min()),
            y_offset=float(boundary_vertices[:, 1].min()),
            neighbor_x=neighbor_x,
            neighbor_y=neighbor_y,
            farm_index=farm_index,
            neighbor_farm_idx=neighbor_farm_idx if neighbor_farm_idx is not None else np.array([])
        ),
        promotes_inputs=[('x', 'turbine_x'), ('y', 'turbine_y')],
        promotes_outputs=['aep']
    )

    model.add_subsystem(
        'cable_comp',
        OptiWindNetComponent(
            boundary_vertices=boundary_vertices,
            substation_coord=substation_coord,
            cable_specs=cable_specs,
            n_turbines=n_wt
        ),
        promotes_inputs=[('turbine_x', 'turbine_x'), ('turbine_y', 'turbine_y')],
        promotes_outputs=[('cable_cost', 'cost_cables')]
    )

    model.add_subsystem(
        'foundations_comp',
        FoundationsComp(
            n_wt=n_wt,
            bathy_nc_path=str(bathy_nc_path),
            rated_power=rated_power,
            ip_platform=ip_platform,
            rotor_diameter=rotor_diameter,
            hub_height=hub_height,
            wind_speed=wind_speed
        ),
        promotes_inputs=[('x', 'turbine_x'), ('y', 'turbine_y')],
        promotes_outputs=['cost_foundations']
    )

    model.add_subsystem(
        'lcoe_comp',
        LCOEComp(n_wt=n_wt, rated_power_kw=float(rated_power) * 1000.0),
        promotes_inputs=['aep', 'cost_foundations', 'cost_cables'],
        promotes_outputs=['lcoe']
    )

    model.add_subsystem(
        'boundary_comp',
        BoundaryComp(n_wt=n_wt, zones=[(boundary_vertices, 1)], units='m'),
        promotes_inputs=[('x', 'turbine_x'), ('y', 'turbine_y')],
        promotes_outputs=['boundaryDistances']
    )

    n_add_wt = len(neighbor_x)
    veclen_spacing = n_wt * (n_wt - 1) // 2 + n_wt * n_add_wt

    model.add_subsystem(
        'spacing_comp',
        SpacingCompWithNeighbors(
            n_wt=n_wt,
            min_spacing=min_spacing,
            add_wt_x=neighbor_x,
            add_wt_y=neighbor_y,
            units='m'
        ),
        promotes_inputs=[('x', 'turbine_x'), ('y', 'turbine_y')],
        promotes_outputs=['wtSeparationSquared']
    )

    if unconstrained_mode:  # Injects the penalty aggregator if driver is SGD/Adam
        model.add_subsystem(
            'agg_constraint_comp',
            ConstraintAggregationComp(
                veclen=veclen_spacing,
                n_wt=n_wt,
                min_spacing=min_spacing
            ),
            promotes_inputs=['wtSeparationSquared', 'boundaryDistances'],
            promotes_outputs=[('final_constraint', 'agg_constraint')]
        )

    model.set_input_defaults('turbine_x', val=np.zeros(n_wt), units='m')
    model.set_input_defaults('turbine_y', val=np.zeros(n_wt), units='m')
    model.set_input_defaults('aep', val=1.0, units='MW*h')

    return model


def opt_competitive(farm_sequence, max_cycles, farm_data_registry, optimizer='SLSQP',
                    stochastic_mode=False, sample_size=1, max_iter=100, tol=1.0, output_dir="Results",
                    save_plots=False, plot_every=1, log_convergence=True, log_every=1, objective='lcoe'):
    """
    Executes Sequential Competitive Layout Optimization (Nash Equilibrium loop).

    objective : 'lcoe' or 'aep' -- which model output the driver optimizes, independent
        of which optimizer/driver is chosen. 'aep' is registered with scaler=-1 (OpenMDAO's
        standard idiom for maximization, since add_objective always minimizes internally);
        both the SGD driver and ScipyOptimizeDriver honor that sign automatically.
    """
    if objective not in ('lcoe', 'aep'):
        raise ValueError(f"objective must be 'lcoe' or 'aep', got {objective!r}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unconstrained_opts = ['SGD']
    unconstrained_mode = str(optimizer).upper() in unconstrained_opts

    print(f"=== INITIALIZING COMPETITIVE LAYOUT OPTIMIZATION ===")
    print(f"Active Sequence: {farm_sequence}")
    print(f"Unconstrained Aggregation Mode: {unconstrained_mode}\n")

    for cycle in range(1, max_cycles + 1):
        print(f"\n{'='*60}")
        print(f"   STARTING GLOBAL NASH CYCLE {cycle} / {max_cycles}")
        print(f"{'='*60}")
        
        any_farm_moved = False
        
        for active_farm in farm_sequence:
            print(f"\n--- Optimizing Active Asset: {active_farm.upper()} ---")
            
            cfg = farm_data_registry[active_farm]
            n_wt = len(cfg['x_init'])
            
            # Gathers layout from all other registered assets to lock them as static background wakes.
            # Each neighbor position is tagged with its own farm_index so AEPComp can cast its
            # wake using that farm's own turbine model instead of the active farm's.
            current_neighbor_x = []
            current_neighbor_y = []
            current_neighbor_farm_idx = []
            for other_farm, other_cfg in farm_data_registry.items():
                if other_farm != active_farm:
                    current_neighbor_x.extend(other_cfg['x_init'])
                    current_neighbor_y.extend(other_cfg['y_init'])
                    current_neighbor_farm_idx.extend([other_cfg['farm_index']] * len(other_cfg['x_init']))

            neighbor_x_arr = np.array(current_neighbor_x)
            neighbor_y_arr = np.array(current_neighbor_y)
            neighbor_farm_idx_arr = np.array(current_neighbor_farm_idx, dtype=int)
            print(f"Injecting {len(neighbor_x_arr)} static neighbor turbines into PyWake environment.")

            prob = om.Problem()
       

            prob.model = build_farm_model(
                farm_name=active_farm,
                n_wt=n_wt,
                site_yaml_path=cfg['site_yaml'],
                system_yaml_path=cfg['system_yaml'],
                bathy_nc_path=cfg['bathy_nc'],
                rated_power=cfg['rated_power'],
                ip_platform=cfg['ip_platform'],
                rotor_diameter=cfg['rotor_diameter'],
                hub_height=cfg['hub_height'],
                wind_speed=cfg['wind_speed'],
                neighbor_x=neighbor_x_arr,
                neighbor_y=neighbor_y_arr,
                boundary_vertices=cfg['boundary_vertices'],
                substation_coord=cfg['substation_coord'],
                cable_specs=cfg['cable_specs'],
                min_spacing=cfg['min_spacing'],
                unconstrained_mode=unconstrained_mode,
                stochastic_mode=stochastic_mode,
                sample_size=sample_size,
                farm_index=cfg['farm_index'],
                neighbor_farm_idx=neighbor_farm_idx_arr
            )

            if not unconstrained_mode:  # Standard bounded SciPy configuration
                prob.driver = om.ScipyOptimizeDriver()
                prob.driver.options['optimizer'] = optimizer
                prob.driver.options['maxiter'] = max_iter
                prob.driver.options['tol'] = 1e-5
                
                x_min, x_max = cfg['boundary_vertices'][:, 0].min(), cfg['boundary_vertices'][:, 0].max()
                y_min, y_max = cfg['boundary_vertices'][:, 1].min(), cfg['boundary_vertices'][:, 1].max()
                
                prob.model.add_design_var('turbine_x', lower=x_min, upper=x_max)
                prob.model.add_design_var('turbine_y', lower=y_min, upper=y_max)
                if objective == 'aep':
                    prob.model.add_objective('aep', scaler=-1)  # maximize (OpenMDAO only minimizes)
                else:
                    prob.model.add_objective('lcoe')

                prob.model.add_constraint('boundaryDistances', lower=0.0)
                prob.model.add_constraint('wtSeparationSquared', lower=cfg['min_spacing']**2)
                prob.driver.opt_settings['disp'] = True  
                

                prob.driver.declare_coloring(show_sparsity=False, show_summary=True) 
                

                prob.set_solver_print(level=2)

            else:
                on_iteration = None
                convergence_log = []
                plotter = None
                if save_plots:
                    plotter = FarmPlotter(
                        farm_name=active_farm,
                        boundary_vertices=cfg['boundary_vertices'],
                        substation_coord=cfg['substation_coord'],
                        cable_specs=cfg['cable_specs'],
                        bathy_nc_path=cfg['bathy_nc'],
                        output_dir=output_path / "plots" / active_farm,
                        min_spacing=cfg['min_spacing'],
                    )

                # Deterministic re-evaluation is shared by both plotting and convergence
                # logging so it only runs once per iteration regardless of which (or both)
                # are enabled -- neither path touches AEPComp's/LCOEComp's own stochastic
                # compute()/compute_partials(), which the driver differentiates through.
                if stochastic_mode and (save_plots or log_convergence):
                    def on_iteration(iteration, model, obj_val, tag=None,
                                      _plotter=plotter, _cycle=cycle, _log=convergence_log):
                        do_plot = save_plots and (tag is not None or iteration % plot_every == 0)
                        do_log = log_convergence and (tag is not None or iteration % log_every == 0)
                        if not do_plot and not do_log:
                            return

                        tx = prob.get_val('turbine_x')
                        ty = prob.get_val('turbine_y')
                        aep_val = model.aep_comp.compute_deterministic_aep(tx, ty)
                        lcoe_val = model.lcoe_comp.compute_lcoe(
                            aep_val, prob.get_val('cost_foundations'), float(prob.get_val('cost_cables')[0]))

                        if do_log:
                            # obj_val is the driver's own stochastic mini-batch sample of
                            # whichever output `objective` points at (lcoe or aep) -- kept
                            # generic here since aep_mwh/lcoe_eur_mwh above are always both
                            # computed deterministically regardless of what's being optimized.
                            _log.append({
                                'iteration': iteration,
                                'elapsed_s': time.time() - start_time,
                                'aep_mwh': aep_val,
                                'lcoe_eur_mwh': lcoe_val,
                                'objective': objective,
                                'objective_stochastic_sample': obj_val,
                                'tag': tag,
                            })

                        if do_plot:
                            _plotter.save_iteration_plot(
                                cycle=_cycle, iteration=iteration,
                                turbine_x=tx, turbine_y=ty,
                                wfn=getattr(model.cable_comp, 'wfn', None),
                                lcoe=lcoe_val,
                                aep=aep_val,
                                tag=tag,
                            )

                prob.driver = SGD(maxiter=max_iter, on_iteration=on_iteration)

                # Core Hyperparameters tuning configuration
                prob.driver.options['learning_rate'] = 10.0   # Base coordinate step size eta_0 (in meters)
                prob.driver.options['gamma_min'] = 1e-3       # Step factor at the final iteration
                prob.driver.options['beta1'] = 0.9           # First moment decay for gradient smoothing
                prob.driver.options['beta2'] = 0.999         # Second moment decay for scaling step
                prob.driver.options['tol'] = 1e-6             # Infinity-norm convergence tolerance
                prob.driver.options['disp'] = True            # Live execution dashboard feedback
                
                x_min, x_max = cfg['boundary_vertices'][:, 0].min(), cfg['boundary_vertices'][:, 0].max()
                y_min, y_max = cfg['boundary_vertices'][:, 1].min(), cfg['boundary_vertices'][:, 1].max()

                prob.model.add_design_var('turbine_x', lower=x_min, upper=x_max)
                prob.model.add_design_var('turbine_y', lower=y_min, upper=y_max)

                # 'agg_constraint' is not enforced by OpenMDAO itself -- SGD._setup_driver()
                # reads it as the penalty term for its internal Lagrangian loop.
                if objective == 'aep':
                    prob.model.add_objective('aep', scaler=-1)  # maximize (OpenMDAO only minimizes)
                else:
                    prob.model.add_objective('lcoe')
                prob.model.add_constraint('agg_constraint')
                
                # Explicitly forces OpenMDAO to compute simultaneous derivatives for optimization
                prob.driver.declare_coloring(show_sparsity=False, show_summary=True)
                prob.set_solver_print(level=0) # Keeps it clean during stochastic iterations

            prob.setup()
            prob.set_val('turbine_x', cfg['x_init'])
            prob.set_val('turbine_y', cfg['y_init'])
            
            start_time = time.time()
            prob.run_model() # Single forward pass to populate outputs before driver.run()



            print("--- UNITS AND SCALE AUDIT ---")
            print(f"Raw cable CAPEX: {prob.get_val('cost_cables')} EUR")
            print(f"Raw foundation CAPEX: {prob.get_val('cost_foundations')} EUR")
            print(f"AEP from PyWake: {prob.get_val('aep', units='MW*h')} MWh")
            prob.run_driver()
            runtime = time.time() - start_time
            
            x_opt = prob.get_val('turbine_x')
            y_opt = prob.get_val('turbine_y')

            # aep/lcoe as last solved by the model reflect the driver's stochastic mini-batch
            # (if stochastic_mode) -- re-evaluate deterministically (full wind rose) for the
            # authoritative reported/recorded value, without touching AEPComp's/LCOEComp's own
            # compute()/compute_partials(), which the SGD driver differentiates through.
            aep_stochastic_sample = None
            lcoe_stochastic_sample = None
            if stochastic_mode:
                aep_stochastic_sample = float(prob.get_val('aep', units='MW*h')[0])
                lcoe_stochastic_sample = float(prob.get_val('lcoe')[0])
                aep_final = prob.model.aep_comp.compute_deterministic_aep(x_opt, y_opt)
                lcoe_final = prob.model.lcoe_comp.compute_lcoe(
                    aep_final, prob.get_val('cost_foundations'), float(prob.get_val('cost_cables')[0]))
            else:
                aep_final = float(prob.get_val('aep', units='MW*h')[0])
                lcoe_final = float(prob.get_val('lcoe')[0])

            # Computes maximum vector Euclidean displacement to evaluate Nash convergence
            max_delta = np.max(np.sqrt((x_opt - cfg['x_init'])**2 + (y_opt - cfg['y_init'])**2))
            print(f"Round finished in {runtime:.2f} s. Maximum turbine displacement: {max_delta:.2f} m.")
            print(f"AEP: {aep_final:,.2f} MWh | LCOE: {lcoe_final:.4f} EUR/MWh")
            if stochastic_mode:
                print(f"(Driver's last stochastic mini-batch estimate: "
                      f"AEP {aep_stochastic_sample:,.2f} MWh | LCOE {lcoe_stochastic_sample:.4f} EUR/MWh)")

            if max_delta > tol:
                any_farm_moved = True

            if stochastic_mode and log_convergence and convergence_log:
                conv_filename = output_path / f"convergence_{active_farm}_cycle_{cycle}.pkl"
                with open(conv_filename, 'wb') as f:
                    pickle.dump({
                        'farm_name': active_farm,
                        'cycle': cycle,
                        'total_runtime_s': runtime,
                        'history': convergence_log,  # list of per-iteration dicts, see on_iteration above
                    }, f)
                print(f"[Exported] Convergence history saved to: {conv_filename}")

            # Updates registry dynamically to pass the optimized layout as static data to the next asset
            farm_data_registry[active_farm]['x_init'] = x_opt
            farm_data_registry[active_farm]['y_init'] = y_opt

            # Pickle dump for post-processing and warm-starting a different cable-routing
            # heuristic (e.g. HGSRouter/MILPRouter instead of EWRouter) directly from the
            # optimized layout, without needing to re-derive anything from the source YAML:
            # boundary_vertices + substation_coord + cable_specs + x/y are exactly the
            # arguments WindFarmNetwork(...) needs to be reconstructed standalone.
            cable_comp = prob.model.cable_comp
            cable_network = None
            if getattr(cable_comp, 'wfn', None) is not None:
                cable_network = {
                    'edges': cable_comp.wfn.get_network(),  # structured array: src, tgt, length, load, cable
                    'length_m': float(cable_comp.wfn.length()),
                    'cost_eur': float(cable_comp.wfn.cost()),
                }

            pkl_filename = output_path / f"checkpoint_{active_farm}_cycle_{cycle}.pkl"
            checkpoint_data = {
                'farm_name': active_farm,
                'cycle': cycle,
                'x': x_opt,
                'y': y_opt,
                'aep_mwh': aep_final,          # deterministic (full wind rose) if stochastic_mode
                'lcoe_eur_mwh': lcoe_final,    # LCOE derived from aep_mwh above
                'aep_mwh_stochastic_sample': aep_stochastic_sample,   # driver's last mini-batch estimate, or None
                'lcoe_eur_mwh_stochastic_sample': lcoe_stochastic_sample,
                'cost_cables_eur': float(prob.get_val('cost_cables')[0]),
                'cost_foundations_eur': prob.get_val('cost_foundations'),
                'boundary_vertices': cfg['boundary_vertices'],
                'substation_coord': cfg['substation_coord'],
                'cable_specs': cfg['cable_specs'],
                'min_spacing': cfg['min_spacing'],
                'bathy_nc_path': str(cfg['bathy_nc']),
                'cable_network': cable_network,
                'timestamp': time.time()
            }
            with open(pkl_filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"[Exported] Checkpoint saved to: {pkl_filename}")

        if not any_farm_moved:  # Nash Equilibrium condition satisfied
            print(f"\n--> [CONVERGENCE ACHIEVED]: No turbine moved above {tol} m in Cycle {cycle}. Ending Nash Loop.")
            break
            
    print("\n=======================================================")
    print("     COMPETITIVE OPTIMIZATION PIPELINE COMPLETE       ")
    print("=======================================================")


if __name__ == "__main__":
    import windIO

    # --- Directory and File Paths Mapping ---
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / "Data"
    
    SYSTEM_YAML = DATA_DIR / "vineyard_revolution_system.yaml"
    SITE_YAML = DATA_DIR / "site_us.yaml"
    BATHY_NC = DATA_DIR / "bathymetry_us.nc"

    print("=== LOADING CENTRAL WINDIO DATASETS ===")
    system_dat = windIO.load_yaml(SYSTEM_YAML)
    site_dat = windIO.load_yaml(SITE_YAML)

    # --- Meteorological Resource Dictionary Extraction ---
    wind_resource = system_dat['site']['energy_resource']['wind_resource']
    site_resource_dict = {
        'directions': np.array(wind_resource['wind_direction']),
        'probabilities': np.array(wind_resource['sector_probability']['data']),
        'weibull_A': np.array(wind_resource['weibull_a']['data']),
        'weibull_k': np.array(wind_resource['weibull_k']['data']),
        'ws_TI': np.array(wind_resource['wind_speed']),
        'TI_org': wind_resource['turbulence_intensity']['data']
    }
    site_resource_dict['probabilities'] /= site_resource_dict['probabilities'].sum() # Safe normalization

    # --- Turbine and Structural Foundation Properties ---
    rd_turbine = float(system_dat['wind_farm'][0]['turbines']['rotor_diameter'])
    wind_speed_design = mean_wind_speed(site_resource_dict)
    # RP=13 (Vineyard/Haliade-X) and RP=11 (Revolution/SG11-200), trained from real IEA
    # Wind 2200-22-ROWP structural data -- see Data/ssms/train_project_turbines.py
    FOUNDATION_PARAMS = {
        'vineyard': {'rated_power': 13, 'ip_platform': 15.0},
        'revolution': {'rated_power': 11, 'ip_platform': 10.0},
    }

    # --- Cable Specifications (capacity, cost_EUR_per_m) ---
    cable_specs = np.array([
        (3, 368.9),
        (5, 428.9),
        (7, 737.1)
    ], dtype=[('capacity', int), ('cost', float)])

    # --- Farm Data Registry Assembly (Nash Memory) ---
    farm_data_registry = {}

    # Order must match the wind_farm array index in the system YAML
    farm_names_in_yaml = ["vineyard", "revolution"]
    
    for idx, farm_name in enumerate(farm_names_in_yaml):
        farm_node = system_dat['wind_farm'][idx]
        layout_node = farm_node['layouts'][0]['coordinates']
        turbine_node = farm_node['turbines']
        
        # Extracting real full-scale UTM coordinates
        x_init = np.array(layout_node['x'], dtype=float)
        y_init = np.array(layout_node['y'], dtype=float)
        
        # Substation coordinates mapping. Shape must be (n_substations, 2) -- WindFarmNetwork
        # silently synthesizes a second, bogus substation node if given a flat (2,) array
        # instead of a (1, 2) one, corrupting cable routing (see investigation in chat history).
        sub_node = farm_node['electrical_substations'][0]['electrical_substation']['coordinates']
        substation_coord = np.array([[float(sub_node['x'][0]), float(sub_node['y'][0])]], dtype=float)
        
        # Spatial boundary polygon vertices mapping from the Site node
        poly_node = site_dat['boundaries']['polygons'][idx]
        boundary_vertices = np.column_stack((poly_node['x'], poly_node['y']))
        
        # Building the local registry package for this specific asset
        farm_data_registry[farm_name] = {
            'site_yaml': site_resource_dict,
            'system_yaml': system_dat,
            'bathy_nc': BATHY_NC,
            'rated_power': FOUNDATION_PARAMS[farm_name]['rated_power'],
            'ip_platform': FOUNDATION_PARAMS[farm_name]['ip_platform'],
            'rotor_diameter': float(turbine_node['rotor_diameter']),
            'hub_height': float(turbine_node['hub_height']),
            'wind_speed': wind_speed_design,
            'boundary_vertices': boundary_vertices,
            'substation_coord': substation_coord,
            'cable_specs': cable_specs,
            'x_init': x_init,
            'y_init': y_init,
            'farm_index': idx,  # index into system_dat['wind_farm'] -- AEPComp needs this to pick its own turbine
            'min_spacing': 4.0 * rd_turbine # Bounded minimum safety spacing in meters (approx. 1.25 * D)
        }
        print(f"[Registered] Asset: {farm_name.upper()} | Layout shape: {len(x_init)} turbines.")

    # =============================================================================
    # EXECUTION TRIGGER
    # =============================================================================
    opt_competitive(
        farm_sequence=["vineyard"],
        max_cycles=1,
        farm_data_registry=farm_data_registry,
        optimizer='SGD',
        stochastic_mode=True,
        sample_size=100,  # Mini-batch draws per iteration; reduces per-step gradient variance
        max_iter=50,  # Soft iteration cap per Nash step for speed
        tol=2.0,      # Bounded convergence tolerance in meters (Nash Tol)
        output_dir="Results_Nash_Run_test",
        save_plots=True,  # Saves one layout+cable+bathymetry PNG per accepted iteration
        plot_every=1,
        objective='lcoe'   
    )