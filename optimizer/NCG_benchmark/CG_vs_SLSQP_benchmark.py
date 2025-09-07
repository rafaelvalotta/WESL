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
from boundary_dev import get_only_boundary
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from topfarm import TopFarmProblem
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.recorders import TopFarmListRecorder


class AEP_Comp(om.ExplicitComponent):
    def __init__(self, wfm, wt_x, wt_y):
        super().__init__()
        self.wake_model = wfm
        self.initial_x = np.array(wt_x)
        self.initial_y = np.array(wt_y)
        self.n_turbines = len(self.initial_x)

    def setup(self):
        self.add_input('x', val=self.initial_x, units='m')
        self.add_input('y', val=self.initial_y, units='m')
        self.add_output('aep', val=0.0)
        self.declare_partials('*', '*', method='fd', step=50.0)



    def compute(self, inputs, outputs):
        # Get turbine positions from inputs
        x_positions = inputs['x']
        y_positions = inputs['y']

        # Calculate AEP explicitly via provided PyWake model
        outputs['aep'] = self.wake_model(x_positions, y_positions).aep().sum()


class PenaltyLossComp(om.ExplicitComponent):
    """
    Penalty objective for unconstrained CG:
        loss = -AEP + rho_s * mean(max(0, -spacing_cons)^2) + rho_b * mean(max(0,  boundary_cons)^2)
    """
    def initialize(self):
        self.options.declare('rho_s', default=1e6, types=float)
        self.options.declare('rho_b', default=1e6, types=float)

    def setup(self):
        self.add_input('aep', val=0.0)
        self.add_input('spacing_cons', shape_by_conn=True)
        self.add_input('boundary_cons', shape_by_conn=True)
        self.add_output('loss', val=0.0)
        self.declare_partials('*', '*', method='fd', step=1e-1)


    def huber(self, v):
        delta = 100.0  # smoothing threshold in meters
        v = np.asarray(v, dtype=float)
        q = np.where(v <= delta, 0.5 * v**2, delta * (v - 0.5 * delta))
        return 0.0 if v.size == 0 else float(np.mean(q))

    def compute(self, inputs, outputs):
        aep = float(np.asarray(inputs['aep']).ravel()[0])
        sc = np.asarray(inputs['spacing_cons'], dtype=float)
        bc = np.asarray(inputs['boundary_cons'], dtype=float)

        viol_s = np.maximum(0.0, -sc)
        viol_b = np.maximum(0.0,  bc)

        p_s = self.huber(viol_s)
        p_b = self.huber(viol_b)

        outputs['loss'] = -aep + self.options['rho_s'] * p_s + self.options['rho_b'] * p_b



class SpacingConstraintComp(om.ExplicitComponent):
    def __init__(self, n_turbines, wind_turbine):
        super().__init__()
        self.n_turbines = n_turbines
        self.min_spacing = 8 * wind_turbine.diameter()  # Minimum distance = 8 × rotor diameter

    def setup(self):
        # Inputs: turbine coordinates
        self.add_input('x', val=np.zeros(self.n_turbines), units='m')
        self.add_input('y', val=np.zeros(self.n_turbines), units='m')

        # Output: constraints array for spacing (pairwise distances)
        num_constraints = self.n_turbines * (self.n_turbines - 1) // 2
        self.add_output('spacing_cons', val=np.zeros(num_constraints))

        # Finite difference approximation for derivatives
        self.declare_partials('*', '*', method='fd', step=10.0)

    def compute(self, inputs, outputs):
        x, y = inputs['x'], inputs['y']
        cons = []
        for i in range(self.n_turbines):
            for j in range(i + 1, self.n_turbines):
                distance = np.hypot(x[i] - x[j], y[i] - y[j])
                cons.append(distance - self.min_spacing)
        outputs['spacing_cons'] = np.array(cons)

class BoundaryConstraintComp(om.ExplicitComponent):
    def __init__(self, boundary_filepath, n_turbines):
        super().__init__()
        eastings, northings = get_only_boundary(boundary_filepath)
        self.polygon = Polygon(zip(eastings, northings))
        self.n_turbines = n_turbines

    def setup(self):
        self.add_input('x', val=np.zeros(self.n_turbines), units='m')
        self.add_input('y', val=np.zeros(self.n_turbines), units='m')
        self.add_output('boundary_cons', val=np.zeros(self.n_turbines))
        self.declare_partials('*', '*', method='fd', step=10.0)

    def compute(self, inputs, outputs):
        points = [Point(xy) for xy in zip(inputs['x'], inputs['y'])]
         # Positive outside polygon, negative (<= 0 feasible) inside
        outputs['boundary_cons'] = np.array([self.polygon.exterior.distance(pt) if not self.polygon.contains(pt) 
                                            else - self.polygon.exterior.distance(pt) for pt in points])


class PlotComp(om.ExplicitComponent):
    """
    Live plot of boundary + initial vs current turbine positions.
    Draws at each model evaluation (objective/constraint pass).
    """
    def initialize(self):
        self.options.declare('init_x', types=(list, np.ndarray))
        self.options.declare('init_y', types=(list, np.ndarray))
        self.options.declare('polygon', types=(list, np.ndarray))   # Nx2 closed loop
        self.options.declare('aep0', default=None, types=(float, type(None)))
        self.options.declare('spacing_diam', default=None, types=(float, type(None)))  # optional rings
        self.options.declare('mode', default="", types=(str, type(None)))  # optional mode label

    def setup(self):
        init_x = np.asarray(self.options['init_x'])
        init_y = np.asarray(self.options['init_y'])
        self.n = int(len(init_x))
        self._mode_label = str(self.options['mode'])

        self.add_input('x', val=init_x.copy(), units='m')
        self.add_input('y', val=init_y.copy(), units='m')
        self.add_input('aep', val=0.0)
        self.add_output('tick', val=0.0)  # dummy

        self.declare_partials('*', '*', method='fd', step=1e-3)

        # persistent artists/figure
        self._fig = None
        self._ax = None
        self._scatter_curr = None
        self._lines = []
        self._rings = []
        self._text = None
        self._iter = 0

        # cache
        self._init_x = init_x
        self._init_y = init_y
        self._poly = np.asarray(self.options['polygon'])
        self._aep0 = self.options['aep0']
        self._spacing_r = None if self.options['spacing_diam'] is None else 0.5 * float(self.options['spacing_diam'])
        

        plt.ion()

    def compute(self, inputs, outputs):
        x = np.asarray(inputs['x'])
        y = np.asarray(inputs['y'])
        aep = float(np.asarray(inputs['aep']).ravel()[0])

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(10, 8))
            # boundary
            self._ax.plot(self._poly[:, 0], self._poly[:, 1], 'k-', lw=1.5, label='Boundary')
            # initial positions
            self._ax.scatter(self._init_x, self._init_y, c='blue', marker='o', label='Initial')
            self._ax.set_xlabel('Easting (m)')
            self._ax.set_ylabel('Northing (m)')
            # limits from polygon
            xmin, xmax = float(self._poly[:,0].min()), float(self._poly[:,0].max())
            ymin, ymax = float(self._poly[:,1].min()), float(self._poly[:,1].max())
            self._ax.set_xlim(xmin - 1000 , xmax + 1000)
            self._ax.set_ylim(ymin - 1000, ymax + 1000)
            self._ax.set_title(f"Optimizer: {self._mode_label}", fontsize=14)
            self._ax.set_aspect('equal', adjustable='box')
            self._text = self._ax.text(0.01, 0.99, '', transform=self._ax.transAxes,
                                       va='top', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.7))

        # clear previous current artists
        if self._scatter_curr is not None:
            self._scatter_curr.remove()
            self._scatter_curr = None
        for ln in self._lines:
            ln.remove()
        self._lines.clear()
        for rk in self._rings:
            rk.remove()
        self._rings.clear()

        # draw current
        self._scatter_curr = self._ax.scatter(x, y, c='red', marker='^', label='Current')

        # displacement lines
        for x0, y0, xi, yi in zip(self._init_x, self._init_y, x, y):
            ln, = self._ax.plot([x0, xi], [y0, yi], 'k--', lw=0.6)
            self._lines.append(ln)

        # optional spacing rings
        if self._spacing_r is not None:
            for xi, yi in zip(x, y):
                ring = Circle((xi, yi), self._spacing_r, ec='gray', fc='none', ls='--', lw=0.8)
                self._ax.add_patch(ring)
                self._rings.append(ring)

        # update text
        if self._aep0 is not None and self._aep0 != 0.0:
            gain = 100.0 * (aep - self._aep0) / self._aep0
            self._text.set_text(f"Iter: {self._iter}\nAEP: {aep:.3f} GWh  (Δ {gain:+.2f}%)")
        else:
            self._text.set_text(f"Iter: {self._iter}\nAEP: {aep:.3f} GWh")

        # legend (dedup)
        handles, labels = self._ax.get_legend_handles_labels()
        bylab = {lab: h for h, lab in zip(handles, labels) if lab}
        self._ax.legend(bylab.values(), bylab.keys(), loc='upper right', fontsize=8)

        plt.draw()
        plt.pause(0.001)

        self._iter += 1
        outputs['tick'] = 0.0


# ----------------------------
# Single Run
# ----------------------------

# # Set up OpenMDAO problem
# prob = om.Problem()



# wt_x = np.array([
#      1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0,
#    1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 
#      1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0,
#     1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0,
# ])
# wt_y = np.array(
#     [500.0]*6 + [1500.0]*6 + [2500.0]*6 + [3500.0]*6
# )

# n_turbines = len(wt_y)

# boundary_filepath = "../Wind_Farms_Wake_Interactions/Examples/DigitizeLayers/Europe/rosand_2.geojson"  # unused after override
# prob.model.add_subsystem('boundary', BoundaryConstraintComp(boundary_filepath, n_turbines), promotes=['*'])
# # hardcoded polygon (keep Polygon-based constraint)
# prob.model.boundary.polygon = Polygon([
#     (0.0, 0.0),
#     (6000.0, 0.0),
#     (6000.0, 6000.0),
#     (0.0, 6000.0)
# ])
# boundary_xy = np.array([[0.0, 0.0],
#                         [6000.0, 0.0],
#                         [6000.0, 6000.0],
#                         [0.0, 6000.0],
#                         [0.0, 0.0]], dtype=float) 

# # Add PyWake component
# wfm = bastankhah_WF_model(Rodsand_2(), SWT_23_93())
# aep0 = float(wfm(wt_x, wt_y).aep().sum())
# prob.model.add_subsystem('aep_comp', AEP_Comp(wfm, wt_x, wt_y), promotes=['*'])


# prob.model.add_subsystem('spacing', SpacingConstraintComp(n_turbines, SWT_23_93()),
#                          promotes=['*'])

# # live plot component
# prob.model.add_subsystem(
#     'plotter',
#     PlotComp(init_x=wt_x,
#              init_y=wt_y,
#              polygon=boundary_xy,
#              aep0=aep0,
#              spacing_diam=8.0 * SWT_23_93().diameter()),
#     promotes=['*']
# )

# prob.driver = om.ScipyOptimizeDriver()

# # --- Driver: switch to SLSQP with true constraints ---
# # prob.driver.options['optimizer'] = 'SLSQP'
# # prob.driver.options['disp'] = True
# # prob.driver.opt_settings['ftol'] = 1e-6
# # prob.driver.opt_settings['maxiter'] = 100
# # prob.model.add_constraint('spacing_cons', lower=0.0)      # <-- REMOVE for CG (handled by penalty)
# # prob.model.add_constraint('boundary_cons', upper=0.0)     # <-- REMOVE for CG (handled by penalty)
# # prob.model.add_objective('aep', scaler=-1)                # <-- REMOVE


# # --- Driver: switch to CG ---
# prob.driver.options['optimizer'] = 'CG'
# prob.driver.opt_settings['gtol'] = 1e-6
# prob.driver.opt_settings['maxiter'] = 50
# # No explicit constraints; add penalty objective
# prob.model.add_subsystem('loss_comp', PenaltyLossComp(rho_s=1e6, rho_b=1e6), promotes=['*'])
# prob.model.add_objective('loss')


# # --- Design variables (unchanged) ---
# prob.model.add_design_var('x', lower=min(wt_x)-1000, upper=max(wt_x)+1000)
# prob.model.add_design_var('y', lower=min(wt_y)-1000, upper=max(wt_y)+1000)



# prob.model.set_input_defaults('x', val=wt_x, units='m')
# prob.model.set_input_defaults('y', val=wt_y, units='m')

# output_folder = "./output_data"
# os.makedirs(output_folder, exist_ok=True)

# recorder_path = os.path.join(output_folder, "optimization_history.sql")

# # remove old recorder to avoid overwrite warning
# if os.path.exists(recorder_path):
#     os.remove(recorder_path)

# recorder = om.SqliteRecorder(recorder_path)
# prob.driver.add_recorder(recorder)
# prob.driver.recording_options['includes'] = ['*']
# prob.driver.recording_options['record_derivatives'] = False

# # Now continue explicitly
# prob.setup()

# prob.set_val('x', wt_x)
# prob.set_val('y', wt_y)


# print("Optimization running")
# prob.run_driver()
# print("Optimization Done!") 

# plt.ioff()          # disable interactive mode so show() blocks
# plt.show()          

# ----------------------------
# Single Run (TOPFARM SGD)
# ----------------------------
# Initial layout (same as your example)
# wt_x = np.array([
#     1000.0, 2200.0, 3400.0, 4700.0,   # row 1, irregular
#     1400.0, 2600.0, 3900.0,           # row 2, shifted
#     1800.0, 3100.0, 4300.0,           # row 3, shifted differently
#     1200.0, 2500.0, 3700.0            # row 4, fewer turbines
# ])

# wt_y = np.array([
#     1000.0, 1100.0, 950.0, 1200.0,    # row 1
#     2100.0, 1900.0, 2200.0,           # row 2
#     3050.0, 2900.0, 3100.0,           # row 3
#     3900.0, 3700.0, 4000.0            # row 4
# ])
# n_turbines = len(wt_x)

# # Wake model reused here
# wt = SWT_23_93()
# wfm = bastankhah_WF_model(Rodsand_2(), wt)
# aep0 = float(np.asarray(wfm(wt_x, wt_y).aep().sum()).ravel()[0])

# # Cost = -AEP (TOPFARM minimizes)
# def neg_aep_cost(x, y, **kwargs):
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)
#     aep = float(np.asarray(wfm(x, y).aep().sum()).ravel()[0])
#     try:
#         plotter.update(x, y, aep=aep)  # live refresh per evaluation
#     except Exception as e:
#         print(f"[plot] update skipped: {e}")
#     return -aep

# # Wrap cost for TOPFARM
# cost_comp = CostModelComponent(
#     input_keys=["x", "y"],
#     n_wt=n_turbines,
#     cost_function=neg_aep_cost,
#     objective=True
# )

# tf_boundary_open = np.array([[0.0, 0.0],
#                              [6000.0, 0.0],
#                              [6000.0, 6000.0],
#                              [0.0, 6000.0]], dtype=float)
# boundary_con = XYBoundaryConstraint(boundary=tf_boundary_open, boundary_type='polygon')

# # close only for plotting so the outline is visually closed
# tf_boundary_closed = np.vstack([tf_boundary_open, tf_boundary_open[0]])

# spacing_con  = SpacingConstraint(min_spacing=8.0 * wt.diameter())  # <-- add

# plotter = TopFarmPlot(wt_x, wt_y, boundary=tf_boundary_closed, aep0=aep0, spacing_diam=8.0*wt.diameter())

# def plot_callback(**kwargs):
#     x = kwargs['x']
#     y = kwargs['y']
#     aep = -kwargs['cost']  # since cost = -AEP
#     plotter.update(x, y, aep=aep)

# tf = TopFarmProblem(
#     design_vars={'x': wt_x, 'y': wt_y},
#     cost_comp=cost_comp,
#     constraints=[boundary_con, spacing_con],
#     driver=EasySGDDriver(maxiter=100, learning_rate=25.0, gamma_min_factor=10.0, disp=True),
#     post_driver=plot_callback
# )

# print("Running TOPFARM SGD (single run)...")
# sgd_cost, sgd_state, sgd_rec = tf.optimize()
# aep_final = -sgd_cost
# impr = (aep_final - aep0) / max(aep0, 1e-12) * 100.0
# print(f"SGD done. AEP0={aep0:.3f} GWh  AEP*={aep_final:.3f} GWh  Δ={impr:.2f}%")
# plt.ioff()                       # stop interactive mode
# plt.show()


# ----------------------------
# Benchmark: run SLSQP and CG
# ----------------------------
def run_case(mode, rec_path):
    """Build, run, and record one optimization. mode in {'SLSQP','CG'}."""
    # --- Problem + shared data ---
    prob = om.Problem()

    # Hard-coded initial layout
    wt_x = np.array([
        1000.0, 2200.0, 3400.0, 4700.0,
        1400.0, 2600.0, 3900.0,
        1800.0, 3100.0, 4300.0,
        1200.0, 2500.0, 3700.0
    ])
    wt_y = np.array([
        1000.0, 1100.0,  950.0, 1200.0,
        2100.0, 1900.0, 2200.0,
        3050.0, 2900.0, 3100.0,
        3900.0, 3700.0, 4000.0
    ])
    n_turbines = len(wt_y)

    boundary_filepath = str(DATA_DIR / "DigitizeLayers" / "Europe" / "rosand_2.geojson")
    prob.model.add_subsystem('boundary', BoundaryConstraintComp(boundary_filepath, n_turbines), promotes=['*'])
    prob.model.boundary.polygon = Polygon([(0.0, 0.0), (6000.0, 0.0), (6000.0, 6000.0), (0.0, 6000.0)])
    boundary_xy = np.array([[0.0, 0.0], [6000.0, 0.0], [6000.0, 6000.0], [0.0, 6000.0], [0.0, 0.0]])

    # Wake model + AEP component
    wt = SWT_23_93()
    wfm = bastankhah_WF_model(Rodsand_2(), wt)
    aep0 = float(np.asarray(wfm(wt_x, wt_y).aep().sum()).ravel()[0])
    prob.model.add_subsystem('aep_comp', AEP_Comp(wfm, wt_x, wt_y), promotes=['*'])

    # Spacing constraint producer (used in both paths; only enforced in SLSQP)
    prob.model.add_subsystem('spacing', SpacingConstraintComp(n_turbines, wt), promotes=['*'])

    # Live plot every eval
    prob.model.add_subsystem(
        'plotter',
        PlotComp(init_x=wt_x, init_y=wt_y, polygon=boundary_xy, aep0=aep0,
                 spacing_diam=8.0 * wt.diameter(), mode=str(mode)),
        promotes=['*']
    )

    # --- Driver + objective/constraints ---
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons']

    if mode == 'SLSQP':
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['ftol'] = 1e-9
        prob.driver.opt_settings['eps']  = 1e-6
        prob.driver.opt_settings['maxiter'] = 200  # let it take a few more steps
        prob.model.add_design_var('x', lower=0.0, upper=6000.0)
        prob.model.add_design_var('y', lower=0.0, upper=6000.0)
        prob.model.add_constraint('spacing_cons', lower=0.0)
        prob.model.add_constraint('boundary_cons', upper=0.0)
        prob.model.add_objective('aep', scaler=-1.0)

        # Total-derivative FD with a larger, absolute step to overcome AEP flatness/noise
        prob.model.approx_totals(method='fd', step= 50.0)

    else:  # CG
        prob.driver.options['optimizer'] = 'CG'
        prob.driver.opt_settings['gtol'] = 1e-6
        prob.driver.opt_settings['maxiter'] = 200  # cap it a bit higher but finite
        # Keep CG inside the physical box to avoid penalty blow-ups
        prob.model.add_design_var('x', lower=0.0, upper=6000.0)
        prob.model.add_design_var('y', lower=0.0, upper=6000.0)
        prob.model.add_subsystem('loss_comp', PenaltyLossComp(rho_s=5e6, rho_b=1e7), promotes=['*'])
        prob.model.add_objective('loss')

    # --- Recorder ---
    output_folder = str((OPT_DIR / "output_data").resolve())
    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(rec_path):
        os.remove(rec_path)
    rec = om.SqliteRecorder(rec_path)
    prob.driver.add_recorder(rec)
    prob.driver.recording_options['includes'] = ['*']
    prob.driver.recording_options['record_derivatives'] = False

    # --- Setup/init + run ---
    prob.model.set_input_defaults('x', val=wt_x, units='m')
    prob.model.set_input_defaults('y', val=wt_y, units='m')
    prob.setup()
    prob.set_val('x', wt_x)
    prob.set_val('y', wt_y)

    # Pre-run debug: evaluate model once and print feasibility
    prob.run_model()
    try:
        aep0_eval = float(np.asarray(prob.get_val('aep')).ravel()[0])
    except Exception:
        aep0_eval = np.nan
    spacing0 = np.min(np.asarray(prob.get_val('spacing_cons')))
    bound0   = np.max(np.asarray(prob.get_val('boundary_cons')))
    print(f"[DEBUG] init: AEP={aep0_eval:.3f}  min(spacing_cons)={spacing0:.3f}  max(boundary_cons)={bound0:.3f}")

    # --- QUICK AEP SENSITIVITY CHECK (one turbine, 250 m nudge) ---
    try:
        prob.run_model()  # ensure outputs are current

        x_orig = prob.get_val('x').copy()
        y_orig = prob.get_val('y').copy()

        aep_base = float(np.asarray(prob.get_val('aep')).ravel()[0])

        x_pert = x_orig.copy()
        x_pert[0] += 250.0
        prob.set_val('x', x_pert)
        prob.set_val('y', y_orig)
        prob.run_model()

        aep_pert = float(np.asarray(prob.get_val('aep')).ravel()[0])
        print(f"[DEBUG] dAEP (x0 +250m) = {aep_pert - aep_base:.6f}")

        # restore EXACT original state
        prob.set_val('x', x_orig)
        prob.set_val('y', y_orig)
        prob.run_model()
    except Exception as e:
        print(f"[DEBUG] sensitivity check skipped: {e}")


    try:
        J = prob.compute_totals(of=['aep'], wrt=['x', 'y'], return_format='array')
        grad_norm = float(np.linalg.norm(J))
        print(f"[DEBUG] ||grad aep|| (FD totals) = {grad_norm:.6e}")
    except Exception as e:
        print(f"[DEBUG] compute_totals failed: {e}")


    print(f"Optimization running [{mode}]")
    prob.run_driver()
    print(f"Optimization Done! [{mode}]")
    return rec_path

def run_sgd_case():
    """Run TOPFARM SGD once and return (t_rel, aep_series)."""
    # Same initial layout as in run_case to keep benchmark consistent
    wt_x = np.array([
        1000.0, 2200.0, 3400.0, 4700.0,
        1400.0, 2600.0, 3900.0,
        1800.0, 3100.0, 4300.0,
        1200.0, 2500.0, 3700.0
    ])
    wt_y = np.array([
        1000.0, 1100.0,  950.0, 1200.0,
        2100.0, 1900.0, 2200.0,
        3050.0, 2900.0, 3100.0,
        3900.0, 3700.0, 4000.0
    ])
    n_turbines = len(wt_x)

    # Boundary (open polygon for constraint)
    tf_boundary_open = np.array([
        [0.0, 0.0],
        [6000.0, 0.0],
        [6000.0, 6000.0],
        [0.0, 6000.0]
    ], dtype=float)

    # Wake/turbine (reuse your classes)
    wt = SWT_23_93()
    wfm = bastankhah_WF_model(Rodsand_2(), wt)

    # Cost = -AEP
    def neg_aep_cost(x, y, **kwargs):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return -float(np.asarray(wfm(x, y).aep().sum()).ravel()[0])

    cost_comp = CostModelComponent(
        input_keys=["x", "y"],
        n_wt=n_turbines,
        cost_function=neg_aep_cost,
        objective=True
    )
    boundary_con = XYBoundaryConstraint(boundary=tf_boundary_open, boundary_type='polygon')
    spacing_con  = SpacingConstraint(min_spacing=8.0 * wt.diameter())

    # Simple list recorder (timestamps + cost)
    rec = TopFarmListRecorder()

    # SGD driver (conservative settings)
    driver = EasySGDDriver(maxiter=100, learning_rate=5.0, gamma_min_factor=5.0, disp=True)

    tf = TopFarmProblem(
        design_vars={'x': wt_x, 'y': wt_y},
        cost_comp=cost_comp,
        constraints=[boundary_con, spacing_con],
        driver=driver,
        recorder=rec
    )

    print("Optimization running [SGD]")
    sgd_cost, sgd_state, sgd_rec = tf.optimize()
    print("Optimization Done! [SGD]")

    # Build time and AEP series from recorder
    ts = np.array(sgd_rec['timestamp'], dtype=float)
    t_rel = ts - ts[0]
    aep = -np.asarray(sgd_rec['Cost'], dtype=float)
    return t_rel, aep


def read_series(rec_path):
    """Return (t_sec, aep_series) from a SqliteRecorder driver recording."""
    cr = om.CaseReader(rec_path)
    cases = cr.get_cases('driver')
    # timestamps can be float or ISO string; normalize to seconds from start
    ts = []
    aep = []
    for c in cases:
        t = getattr(c, 'timestamp', None)
        if isinstance(t, (int, float)):
            ts.append(float(t))
        elif isinstance(t, str):
            # strip trailing Z if present and parse
            s = t.rstrip('Z')
            ts.append(datetime.fromisoformat(s).timestamp())
        else:
            ts.append(np.nan)
        aep.append(float(np.asarray(c.get_val('aep')).ravel()[0]))
    ts = np.asarray(ts)
    t0 = np.nanmin(ts)
    t_rel = ts - t0
    return t_rel, np.asarray(aep)


if __name__ == "__main__":
    # --- Run both cases ---
    cg_rec  = str((OPT_DIR / 'output_data' / 'cg_record.sql').resolve())
    sls_rec = str((OPT_DIR / 'output_data' / 'slsqp_record.sql').resolve())
    slsqp_db = run_case('SLSQP', sls_rec)
    cg_db    = run_case('CG',    cg_rec)

    # # --- Plot comparison ---
    # t_sls, aep_sls = read_series(slsqp_db)
    # t_cg,  aep_cg  = read_series(cg_db)

    # plt.figure()
    # plt.plot(t_sls - t_sls[0], aep_sls, label='SLSQP')
    # plt.plot(t_cg  - t_cg[0],  aep_cg,  label='CG')
    # plt.legend()
    # plt.xlabel('time [s]')
    # plt.ylabel('AEP')
    # plt.title('AEP vs time: SLSQP vs CG')
    # plt.ioff()
    # plt.show()

    t_sls, aep_sls = read_series(slsqp_db)
    t_cg,  aep_cg  = read_series(cg_db)
    t_sgd, aep_sgd = run_sgd_case()

    plt.figure()
    plt.plot(t_sls - t_sls[0], aep_sls, label='SLSQP')
    plt.plot(t_cg  - t_cg[0],  aep_cg,  label='CG')
    plt.plot(t_sgd - t_sgd[0], aep_sgd, label='SGD (TOPFARM)')
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('AEP')
    plt.title('AEP vs time: SLSQP vs CG vs SGD')
    plt.ioff()
    plt.show()



