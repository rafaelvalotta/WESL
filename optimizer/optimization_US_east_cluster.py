from boundary_dev import get_only_boundary, get_boundary_points
from shapely.geometry import Point, Polygon
from windFarms_windTurbines import *
from py_wake.wind_turbines import WindTurbines
import matplotlib.pyplot as plt
import os
import numpy as np
from layout_dev import grid_WTposition_generator
import openmdao.api as om
from py_wake.utils.gradients import autograd
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from CPU_Profiler import profile

"""
Unfortunately, autograd is not working very well with xarray, i.e. the normal xarray SimulationResult must 
be bypassed. This mean that you can compute gradients of the AEP or WS, TI, Power and custom functions by 
setting the argument return_simulationResult=False when running the 
wind farm model: WindFarmModel(..., return_simulationResult=False).
"""
aep_comp_cpu_time = []
spacing_cons_cpu_time = []
boundary_cons_cpu_time = []
plot_comp_cpu_time = []

def sci_formatter(x, pos):
    """Format ticks with 2 decimals, switch to sci if >4 digits."""
    if x == 0:
        return "0.00"
    if abs(x) >= 1e3 or abs(x) < 1e-3:   # beyond 3 decimals → scientific
        return f"{x:.2e}"
    else:
        return f"{x:.2f}"


class AEP_Comp(om.ExplicitComponent):
    def __init__(self, wfm, wt_x, wt_y):
        super().__init__()
        self.wake_model = wfm
        self.initial_x = np.array(wt_x)
        self.initial_y = np.array(wt_y)

    def setup(self):
        self.add_input('x', val=self.initial_x, units='m')
        self.add_input('y', val=self.initial_y, units='m')
        self.add_output('aep', val=0.0)
        self.declare_partials('aep', ['x','y'])


    @profile(store=aep_comp_cpu_time, print_line=True)
    def compute(self, inputs, outputs):
        # Get turbine positions from inputs
        x_positions = inputs['x']
        y_positions = inputs['y']

        outputs['aep'] = self.wake_model(x_positions, y_positions).aep().sum()

    # @profile   
    def compute_partials(self, inputs, partials): 
        # Get turbine positions from inputs
        x_positions = inputs['x']
        y_positions = inputs['y'] 
        dAEPdx, dAEPdy = self.wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x','y'])(x_positions, y_positions)
        # Ensure shapes are (1, n)
        dAEPdx = np.atleast_2d(np.asarray(dAEPdx, dtype=float).ravel())
        dAEPdy = np.atleast_2d(np.asarray(dAEPdy, dtype=float).ravel())

        partials['aep', 'x'] = dAEPdx
        partials['aep', 'y'] = dAEPdy

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

    @profile(store=spacing_cons_cpu_time)
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

    @profile(store=boundary_cons_cpu_time)
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

    @profile(store=plot_comp_cpu_time)
    def compute(self, inputs, outputs):
        x = np.asarray(inputs['x'])
        y = np.asarray(inputs['y'])
        aep = float(np.asarray(inputs['aep']).ravel()[0])

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(10, 8))
            # boundary
            self._ax.plot(self._poly[:, 0], self._poly[:, 1], '--k', lw=1.5, label='Boundary', zorder=0)
            # initial positions
            self._ax.scatter(self._init_x, self._init_y, c='blue', marker='o', s=5, label='Initial')
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
        self._scatter_curr = self._ax.scatter(x, y, c='red', marker='2', s=6, label='Current')

        # displacement lines
        for x0, y0, xi, yi in zip(self._init_x, self._init_y, x, y):
            ln, = self._ax.plot([x0, xi], [y0, yi], 'k--', lw=0.4)
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

    def save_figure(self, path, dpi=600):
        """Save the current figure to disk."""
        if self._fig is None:
            # figure not created yet (no compute() run). Force a draw if needed.
            plt.figure()  # noop placeholder
        if self._fig is not None:
            self._fig.savefig(path, dpi=dpi, bbox_inches='tight')


def main():
    wt_x, wt_y = np.array([]), np.array([])
    site = clusterWF_EastUS()
    windTurbine = Haliade_X()
    pwd = os.path.dirname(__file__)
    data_dir = os.path.join(pwd, 'data', 'DigitizeLayers', 'US')
    usCluster_filepath = os.path.join(os.path.dirname(__file__), 'data/DigitizeLayers/US', 'us_eastcoast_cluster.geojson')
    boundary = np.array(get_boundary_points(usCluster_filepath))
    for f in os.listdir(data_dir):
        if f.endswith('.geojson'):
            if f != "us_eastcoast_cluster.geojson":
                boundary_path = os.path.join(data_dir, f)
                print(f'Processing file: {f}')
                x, y = grid_WTposition_generator(get_boundary_points(boundary_path), windTurbine=windTurbine, spacing=12)
                wt_x = np.concatenate((wt_x, np.array(x)))
                wt_y = np.concatenate((wt_y, np.array(y)))
                
    # print(wt_x)
    # print(wt_y)
    # print("Total Turbines in US East Coast Cluster = "+ str(len(wt_x)))
        
    # easting, northing = get_only_boundary(usCluster_filepath)
    # plt.plot(easting, northing)
    # plt.scatter(wt_x, wt_y, marker='2', s=10 , color='k', label='Wind Turbines')
    # plt.legend()
    # plt.xlabel('Easting (m)')
    # plt.ylabel('Northing (m)')
    # plt.title(f'US East Cluster Boundary | Total Turbines: {len(wt_x)}')
    # plt.axis('equal')
    # plt.show()
    wt_x, wt_y = wt_x[:100], wt_y[:100]  # limit for testing
    n_wt = len(wt_x)
    _diameter = windTurbine.diameter()

    prob = om.Problem()
    wfm = bastankhah_WF_model(site, windTurbine)
    aep0 = wfm(wt_x, wt_y).aep().sum().item()  # initial AEP
    # Make sure your FBWF component accepts nd (or equivalent). If its class
    # doesn’t have an nd arg, add one there similarly to WtSpacingConstraint.
    prob.model.add_subsystem('wf_aep', AEP_Comp(bastankhah_WF_model(site, windTurbine),
                                             wt_x, wt_y), promotes_inputs=['x', 'y']) 

    prob.model.add_subsystem( 'WT_Spacing', SpacingConstraintComp(n_wt, windTurbine),
                             promotes_inputs=['x','y'] ) 
    
    prob.model.add_subsystem( 'WF_Boundary', BoundaryConstraintComp(usCluster_filepath, n_wt),
                             promotes_inputs=['x', 'y'] )

    prob.model.add_subsystem('OffshoreSystemPlot',
                             PlotComp(init_x=wt_x, init_y=wt_y, polygon=boundary,
                                      aep0=aep0, spacing_diam=_diameter*8, mode="SLSQP"),
                            promotes_inputs=['x', 'y'])

    prob.model.connect('wf_aep.aep', 'OffshoreSystemPlot.aep')


    prob.model.set_input_defaults('x', wt_x)
    prob.model.set_input_defaults('y', wt_y)

    prob.model.add_design_var('x', lower=min(boundary[:,0]), upper=max(boundary[:,0]), scaler=0.0001)
    prob.model.add_design_var('y', lower=min(boundary[:,1]), upper=max(boundary[:,1]), scaler=0.0001)

    prob.model.add_objective('wf_aep.aep', scaler=-0.01)

    prob.model.add_constraint('WT_Spacing.spacing_cons', lower=0.0)   # no extra scaler
    prob.model.add_constraint('WF_Boundary.boundary_cons',  upper=-100.0)


    prob.driver = om.ScipyOptimizeDriver(tol = 1e-9)

    # prob.driver.options['optimizer'] = 'COBYLA'
    # prob.driver.options['maxiter'] = 500
    # prob.driver.options['tol']     = 1e-4
    # prob.driver.opt_settings['maxfun'] = 20000


    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter']  = 5       # adjust as you like
    prob.driver.options['tol']      = 1e-6
    # SciPy SLSQP-specific knobs
    prob.driver.opt_settings['ftol'] = 1e-9      # tighter stop on objective change
    prob.driver.opt_settings['disp'] = True

    #recorder = om.SqliteRecorder("optimization_US_east_cluster.sql")
    #prob.driver.add_recorder(recorder)

    prob.setup()

    prob.run_model()

    # Baselines
    wf_aep_init = prob.get_val('wf_aep.aep').item()  # make positive



    prob.run_driver()
    #prob.record("after_run_driver")


    wf_aep_opt  = prob.get_val('wf_aep.aep').item()

    wf_d   = wf_aep_opt  - wf_aep_init

    wf_pct  = 0.0 if wf_aep_init  == 0 else (wf_aep_opt  / wf_aep_init  - 1.0) * 100.0

    print("\n=== AEP Summary (GWh) ===")
    print(f"WF   : init={wf_aep_init:.3f}  opt={wf_aep_opt:.3f}  Δ={wf_d:.3f}  ({wf_pct:.2f}%)")
    print("=========================\n")
    print(f'AepComp.compute() CPU Time List: {aep_comp_cpu_time}')
    print("=========================\n")
    print(f'Spacing_cons.compute() CPU Time List: {spacing_cons_cpu_time}')
    print("=========================\n")
    print(f'Boundary_cons.compute() CPU Time List: {boundary_cons_cpu_time}')
    print("=========================\n")
    print(f'PlotComp.compute() CPU Time List: {plot_comp_cpu_time}')

    plt.ioff()
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axes = axs.flatten()

    for ax in axes:
        ax.set(xlabel='Iteration')
        ax.yaxis.set_major_formatter(FuncFormatter(sci_formatter))

    # Subplots with titles instead of legends
    axes[0].plot(aep_comp_cpu_time, 'r')
    axes[0].set_title(f"AEP Comp | mean: {np.mean(aep_comp_cpu_time):.3f} s")
    axes[0].set_ylabel('CPU Time (s)')

    axes[1].plot(np.array(spacing_cons_cpu_time) * 1000, 'gold')
    axes[1].set_title(f"Spacing Cons | mean: {np.mean(np.array(spacing_cons_cpu_time)*1000):.3f} ms")
    axes[1].set_ylabel('CPU Time (ms)')

    axes[2].plot(np.array(boundary_cons_cpu_time) * 1000, 'g')
    axes[2].set_title(f"Boundary Cons | mean: {np.mean(np.array(boundary_cons_cpu_time)*1000):.3f} ms")
    axes[2].set_ylabel('CPU Time (ms)')


    axes[3].plot(plot_comp_cpu_time, 'k')
    axes[3].set_title(f"Plot Comp | mean: {np.mean(plot_comp_cpu_time):.3f} s")
    axes[3].set_ylabel('CPU Time (s)')

    # Padding between plots
    fig.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.95])

    # Overall title
    fig.suptitle('Computational Resource Profiling: <CPU Time>', fontsize=16)

    plt.show()



    prob.cleanup()
    aep_comp_cpu_time.clear() 
    spacing_cons_cpu_time.clear()
    boundary_cons_cpu_time.clear()
    plot_comp_cpu_time.clear()



if __name__ == "__main__":
    main()
