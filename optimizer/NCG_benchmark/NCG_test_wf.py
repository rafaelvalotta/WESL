import openmdao.api as om
import os
import sys
import numpy as np
from pathlib import Path
# Resolve paths relative to this file
try:
    SCRIPT_DIR = Path(__file__).resolve().parent            # .../WESL/optimizer/NCG_benchmark
except NameError:
    SCRIPT_DIR = Path.cwd().resolve()

OPT_DIR   = SCRIPT_DIR.parent                               # .../WESL/optimizer
REPO_ROOT = OPT_DIR.parent                                   # .../WESL
DATA_DIR  = OPT_DIR / "data"

# Make local packages importable no matter the PWD
for p in (OPT_DIR, REPO_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Ensure relative outputs land under optimizer
try:
    os.chdir(OPT_DIR)
except Exception:
    pass
from windFarms_windTurbines import *
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from CG_vs_SLSQP_benchmark import *
from NCGDriver import NCGDriver

# ----------------------------
# Single Run
# ----------------------------

# Set up OpenMDAO problem
prob = om.Problem()

wt_x = np.array([
    1000.0, 2200.0, 3400.0, 4700.0,   # row 1, irregular
    1400.0, 2600.0, 3900.0,           # row 2, shifted
    1800.0, 3100.0, 4300.0,           # row 3, shifted differently
    1200.0, 2500.0, 3700.0            # row 4, fewer turbines
])

wt_y = np.array([
    1000.0, 1100.0, 950.0, 1200.0,    # row 1
    2100.0, 1900.0, 2200.0,           # row 2
    3050.0, 2900.0, 3100.0,           # row 3
    3900.0, 3700.0, 4000.0            # row 4
])
n_turbines = len(wt_x)

boundary_filepath = str(DATA_DIR / "DigitizeLayers" / "Europe" / "rosand_2.geojson")
prob.model.add_subsystem('boundary', BoundaryConstraintComp(boundary_filepath, n_turbines), promotes=['*'])
# hardcoded polygon (keep Polygon-based constraint)
prob.model.boundary.polygon = Polygon([
    (0.0, 0.0),
    (6000.0, 0.0),
    (6000.0, 6000.0),
    (0.0, 6000.0)
])
boundary_xy = np.array([[0.0, 0.0],
                        [6000.0, 0.0],
                        [6000.0, 6000.0],
                        [0.0, 6000.0],
                        [0.0, 0.0]], dtype=float) 

# Add PyWake component
wfm = bastankhah_WF_model(Rodsand_2(), SWT_23_93())
aep0 = float(wfm(wt_x, wt_y).aep().sum())
prob.model.add_subsystem('aep_comp', AEP_Comp(wfm, wt_x, wt_y), promotes=['*'])


prob.model.add_subsystem('spacing', SpacingConstraintComp(n_turbines, SWT_23_93()),
                         promotes=['*'])

# live plot component
prob.model.add_subsystem(
    'plotter',
    PlotComp(init_x=wt_x,
             init_y=wt_y,
             polygon=boundary_xy,
             aep0=aep0,
             spacing_diam=8.0 * SWT_23_93().diameter()),
    promotes=['*']
)

# --- Driver: switch to SLSQP with true constraints ---
# prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['disp'] = True
# prob.driver.opt_settings['ftol'] = 1e-6
# prob.driver.opt_settings['maxiter'] = 100
# prob.model.add_constraint('spacing_cons', lower=0.0)      # <-- REMOVE for CG (handled by penalty)
# prob.model.add_constraint('boundary_cons', upper=0.0)     # <-- REMOVE for CG (handled by penalty)
# prob.model.add_objective('aep', scaler=-1)                # <-- REMOVE


# --- Driver: switch to CG ---
# prob.driver.options['optimizer'] = 'CG'
# prob.driver.opt_settings['gtol'] = 1e-6
# prob.driver.opt_settings['maxiter'] = 50
# # No explicit constraints; add penalty objective
# prob.model.add_subsystem('loss_comp', PenaltyLossComp(rho_s=1e6, rho_b=1e6), promotes=['*'])
# prob.model.add_objective('loss')

prob.driver = NCGDriver(
        maxiter=60, tol_grad=1e-6, c1=1e-4, c2=0.9,
        alpha0=1.0, max_linesearch=40,
        al_rho0=10.0, al_mu_inc=5.0, al_update_freq=1, feas_tol=1e-3
    )
prob.model.add_design_var('x', lower=100.0, upper=5900)
prob.model.add_design_var('y', lower=100.0, upper=5900)

# Add TRUE constraints (driver handles them via Augmented Lagrangian)
prob.model.add_constraint('spacing_cons', lower=0.0)   # d_ij - d_min >= 0
prob.model.add_constraint('boundary_cons', upper=0.0)  # outside is positive -> must be <= 0
prob.model.add_objective('aep', scaler=-1.0)

# Use FD totals if you don't have analytics:
prob.model.approx_totals(method='fd', step=100.0)



prob.model.set_input_defaults('x', val=wt_x, units='m')
prob.model.set_input_defaults('y', val=wt_y, units='m')

output_folder = "./output_data"
os.makedirs(output_folder, exist_ok=True)

recorder_path = os.path.join(output_folder, "optimization_history.sql")

# remove old recorder to avoid overwrite warning
if os.path.exists(recorder_path):
    os.remove(recorder_path)

recorder = om.SqliteRecorder(recorder_path)
prob.driver.add_recorder(recorder)
prob.driver.recording_options['includes'] = ['*']
prob.driver.recording_options['record_derivatives'] = False

# Now continue explicitly
prob.setup()

prob.set_val('x', wt_x)
prob.set_val('y', wt_y)


print("Optimization running")
prob.run_driver()
print("Optimization Done!") 

plt.ioff()          # disable interactive mode so show() blocks
plt.show()          

