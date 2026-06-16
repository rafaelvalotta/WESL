# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import openmdao.api as om
import numpy as np
from pathlib import Path
import os
from simplerecorder import SimpleRecorder
from mysite import get_setup_params
from aep import AEPCompStochastic, AEPComp
from constraints import SpacingConstraintComp, BoundaryConstraintComp
from aggregator import ConstraintAggregator
from penalty import PenaltyObjectiveComp, FinalRMSViol 
from plotnew import PlotComp


from Drivers import NCG, SGD

def build_problem(K=50, enable_plot=True, plot_every=1, icon_path=None, csv_filename="default.csv"):
    """
    Build the OpenMDAO Problem for NCG/CoBA with DTU-style penalty scaling.

    Parameters
    ----------
    K : int
        Number of Monte Carlo wind samples per AEP/gradient evaluation.
    enable_plot : bool
        If True, add PlotComp to visualize layout/AEP during optimization.
    plot_every : int
        Plot every 'plot_every' iterations.
    icon_path : str or None
        Path to an icon image for the PlotComp (must be in the same folder).
    """
    params = get_setup_params(csv_filename)  # site, turbine, wfm, x_init, y_init, boundary_vertices, etc. [file:24]

    site = params['site']
    turbine = params['turbine']
    wfm = params['wfm']
    x_init = params['x_init']
    y_init = params['y_init']
    boundary_vertices = params['boundary_vertices']
    min_spacing_d = params['min_spacing_d']  
    n_turbines = params['n_turbines']
    D = params['diameter']                    # rotor diameter in meters
#

    min_spacing_m = min_spacing_d * D

    # ---- Compute initial deterministic AEP for plot (outside OpenMDAO) ----
    try:
        aep0 = float(wfm(x_init, y_init).aep().sum())
        print(f"Initial AEP: {aep0:.3f} GWh")
    except Exception as e:
        print(f"Initial AEP error: {e}")
        aep0 = None


    prob = om.Problem()
    m = prob.model

    OUT_DIR = Path(__file__).parent / "Results"   
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    seed = 1
    log_path = OUT_DIR / f"WESL_{n_turbines}wt_seed_{seed}.csv"


    recorder = SimpleRecorder(
        prob,
        out_path=log_path,
        x_name='x',
        y_name='y',
        aep_name='aep_comp_deterministic.aep',
        obj_name='objective',    
        iter_name='opt_iter',
        viol_name='rms_viol'
    )
    recorder.start()


    # ------------- IndepVarComp: design variables  -----------------------
    indeps = m.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

    # Design variables in meters
    indeps.add_output('x', val=x_init, units='m')
    indeps.add_output('y', val=y_init, units='m')
    indeps.add_output('opt_iter', val=0.0)

    # ------------- AEP component (stochastic, with K samples) -------------

    aep_comp = AEPCompStochastic(       
        wake_model=wfm,
        site=site,
        wt_x=x_init,
        wt_y=y_init,
        aep_ref=1.0,      
        recorder=None,
        n_cpu=1,
        K=K,
    )
    # ------------- AEP component deterministic (Plot and Recorder)-------------
    m.add_subsystem(
        'aep_comp_deterministic',
        AEPComp(
            wake_model=wfm,
            wt_x=x_init,
            wt_y=y_init,
            aep_ref=1.0,
            n_cpu=1,   
        ),
        promotes_inputs=['x', 'y']
    )
    m.add_subsystem(
        'aep_comp',
        aep_comp,
        promotes_inputs=['x', 'y'],
        promotes_outputs=['aep'],
    )

    # ------------- Constraint components: spacing + boundary -------------
    spacing_comp = SpacingConstraintComp(
        n_turbines=n_turbines,
        min_spacing=min_spacing_m,
        eps=1e-12,
    )
    m.add_subsystem(
        'spacing_comp',
        spacing_comp,
        promotes_inputs=['x', 'y'],
        promotes_outputs=['spacing_cons'],
    )

    boundary_comp = BoundaryConstraintComp(
        boundary_vertices=boundary_vertices,
        n_turbines=n_turbines,
    )
    m.add_subsystem(
        'boundary_comp',
        boundary_comp,
        promotes_inputs=['x', 'y'],
        promotes_outputs=['boundary_cons'],
    )

    # ------------- Aggregator: g_vector = [spacing_cons, boundary_cons] -------------
    agg_comp = ConstraintAggregator(n_turbines=n_turbines)
    m.add_subsystem(
        'constraint_agg',
        agg_comp,
        promotes_inputs=['spacing_cons', 'boundary_cons'],
        promotes_outputs=['g_vector'],
    )

    # ------------- Penalty objective: -AEP + sum(g_plus^2) -------------
    penalty_comp = PenaltyObjectiveComp(n_constraints=agg_comp.m_total)
    m.add_subsystem(
        'penalty_comp',
        penalty_comp,
        promotes_inputs=['aep', 'g_vector'],
        promotes_outputs=['objective', 'penalty'],
    )
    nc = agg_comp.m_total
    m.add_subsystem('rms_viol', FinalRMSViol(nconstraints=nc), 
                    promotes_outputs=['rms_viol'])  # ← promotes no add_subsystem!
    m.connect('g_vector', 'rms_viol.g_vector')
    # ------------- Optional plotting component -------------
    if enable_plot:
        m.add_subsystem(
            'plot',
            PlotComp(init_x=x_init, init_y=y_init, polygon_vertices=boundary_vertices,
                    aep0=aep0, spacing_meters=min_spacing_m,
                    enabled=True, plot_every=int(plot_every), icon_path=icon_path),
            promotes=['x', 'y'],
        )
        m.connect('opt_iter', 'plot.iter')
        m.connect('aep_comp_deterministic.aep', 'plot.aep') # AEP deterministic to show
        m.connect('objective', 'plot.objective') # Objective (Stochastic when sampled) and penalized to show

    # ------------- Driver -------------
    prob.driver = SGD(maxiter=20)
    prob.driver.options['learning_rate'] = params['diameter'] / 5.0
    prob.driver.options['gamma_min'] = 0.2 * (params['diameter'] / 5.0)
    prob.driver.options['lower'] = 1e-6
    prob.driver.options['upper'] = 1e-1
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True
    # prob.driver.options["beta1"] = 0.1
    # prob.driver.options["beta2"] = 0.2

    prob.driver.recorder = recorder #(activate to generate csv files)


    m.add_design_var('x',
                     lower=boundary_vertices[:, 0].min(),
                     upper=boundary_vertices[:, 0].max())
    m.add_design_var('y',
                     lower=boundary_vertices[:, 1].min(),
                     upper=boundary_vertices[:, 1].max())

    D = params['diameter']
    m.add_objective('aep', scaler = -1.0)
    m.add_constraint('penalty', lower=0.0, scaler=1.0) #Adding a constraint just to avoid openMDAO crash

    return prob


def main(csv_filename="default.csv"):
    K = 50 # standard value
    enable_plot = False
    plot_every = 1
    here = Path(__file__).parent
    icon_path = here / "wt_icon.png" # make sure this file exists in the same directory
    seed = 1
    csv_filename = f"100turb_3600m_kdt_{seed}.csv" # choose a layout from Designs Folder (or you can generate your own randomly)
    prob = build_problem(K=K, enable_plot=enable_plot, plot_every=plot_every, icon_path=icon_path, csv_filename=csv_filename)

    prob.setup()
    prob.run_driver()


if __name__ == "__main__":
    main()