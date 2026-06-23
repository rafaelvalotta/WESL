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

def build_problem(K=50, enable_plot=True, plot_every=1, csv_filename="default.csv", seed=1, driver=0, maxiter=0):
    params = get_setup_params(csv_filename)

    site = params['site']
    turbine = params['turbine']
    wfm = params['wfm']
    x_init = params['x_init']
    y_init = params['y_init']
    boundary_vertices = params['boundary_vertices']
    min_spacing_d = params['min_spacing_d']
    n_turbines = params['n_turbines']
    D = params['diameter']

    min_spacing_m = min_spacing_d * D

    try:
        aep0 = float(wfm(x_init, y_init).aep().sum())
        print(f"Initial AEP: {aep0:.3f} GWh")
    except Exception as e:
        print(f"Initial AEP error: {e}")
        aep0 = None

    # Map integers/booleans to driver classes
    if isinstance(driver, bool):
        driver = int(driver)
    driver_map = {0: SGD, 1: NCG}
    try:
        DriverClass = driver_map[driver]
    except KeyError:
        raise ValueError("driver must be 0/1 or False/True (0->SGD, 1->NCG)")

    use_stochastic_aep = (driver == 0)
    driver_aep_name = 'aep' if use_stochastic_aep else 'aep_comp_deterministic.aep'
  

    # prob = om.Problem()
    prob = om.Problem(model=om.Group(), reports=False)
    m = prob.model

    driver_folder_name = DriverClass.__name__.lower()
    BASE_DIR = Path.cwd().parents[0]
    RESULTS_DIR = BASE_DIR / "Results" / driver_folder_name
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    log_path = RESULTS_DIR / f"WESL_{n_turbines}wt_seed_{seed}.csv"


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

    indeps = m.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('x', val=x_init, units='m')
    indeps.add_output('y', val=y_init, units='m')
    indeps.add_output('opt_iter', val=0.0)

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

    agg_comp = ConstraintAggregator(n_turbines=n_turbines)
    m.add_subsystem(
        'constraint_agg',
        agg_comp,
        promotes_inputs=['spacing_cons', 'boundary_cons'],
        promotes_outputs=['g_vector'],
    )

    penalty_comp = PenaltyObjectiveComp(n_constraints=agg_comp.m_total)
    m.add_subsystem(
        'penalty_comp',
        penalty_comp,
        promotes_inputs=['g_vector'],
        promotes_outputs=['objective', 'penalty'],
    )
    m.connect(driver_aep_name, 'penalty_comp.aep')

    nc = agg_comp.m_total
    m.add_subsystem(
        'rms_viol',
        FinalRMSViol(nconstraints=nc),
        promotes_outputs=['rms_viol']
    )
    m.connect('g_vector', 'rms_viol.g_vector')

    if enable_plot:
        m.add_subsystem(
            'plot',
            PlotComp(init_x=x_init, init_y=y_init, polygon_vertices=boundary_vertices,
                    aep0=aep0, spacing_meters=min_spacing_m,
                    enabled=True, plot_every=int(plot_every)),
            promotes=['x', 'y'],
        )
        m.connect('opt_iter', 'plot.iter')
        m.connect('aep_comp_deterministic.aep', 'plot.aep') # AEP deterministic to show
        m.connect('objective', 'plot.objective') # Objective (Stochastic when sampled) and penalized to show


    prob.driver = DriverClass(maxiter=maxiter)
    prob.driver.options['learning_rate'] = params['diameter'] / 5.0
    prob.driver.options['gamma_min'] = 0.2 * (params['diameter'] / 5.0)
    prob.driver.options['lower'] = 1e-6
    prob.driver.options['upper'] = 1e-1
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True
    prob.driver.recorder = recorder #(activate to generate csv files)

    m.add_design_var(
        'x',
        lower=boundary_vertices[:, 0].min(),
        upper=boundary_vertices[:, 0].max()
    )
    m.add_design_var(
        'y',
        lower=boundary_vertices[:, 1].min(),
        upper=boundary_vertices[:, 1].max()
    )

    m.add_objective(driver_aep_name, scaler=-1.0)
    m.add_constraint('penalty', lower=0.0, scaler=1.0)

    return prob, recorder, log_path


def main(csv_filename="default.csv", K=50, seed=1, driver=0, maxiter=0):
    csv_filename = f"100turb_3600m_kdt_{seed}.csv"
    enable_plot = True
    plot_every = 1
    prob, recorder, log_path = build_problem(
        K=K,
        enable_plot=enable_plot,
        plot_every=plot_every,
        csv_filename=csv_filename,
        seed=seed,
        driver=driver,
        maxiter=maxiter
    )

    prob.setup()
    prob.run_driver()

    return prob, recorder, log_path


prob, recorder, log_path = main(seed=2, K=50, driver=0, maxiter=20)
