import numpy as np
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
import matplotlib.pyplot as plt
from py_wake.utils.gradients import cabs
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.deficit_models.gaussian import BastankhahGaussian
from numpy import newaxis as na
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.utils.gradients import autograd
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, HornsrevV80
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver, EasyScipyOptimizeDriver
from topfarm.plotting import AggregatedConstraintsPlotComponent, XYPlotComp, NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.recorders import TopFarmListRecorder
import sys


plt.close('all')

site = LillgrundSite()
site.interp_method = 'linear'
windTurbines = HornsrevV80() 
wake_model = BastankhahGaussian(site, windTurbines) 

x_rows = int(sys.argv[2])
y_rows = int(sys.argv[2])
spacing = 5
farm_radius = (x_rows - 1) * spacing * windTurbines.diameter() / 2

np.random.seed(int(sys.argv[3]))
#x = np.random.uniform(-1 * farm_radius, farm_radius, x_rows * y_rows)
#y = np.random.uniform(-1 * farm_radius, farm_radius, x_rows * y_rows)
xu, yu = ((x_rows -1 ) * spacing * windTurbines.diameter(), (y_rows -1 ) * spacing * windTurbines.diameter())
x = np.random.uniform(0, xu, x_rows * y_rows)
y = np.random.uniform(0, yu, x_rows * y_rows)
dirs = np.arange(0, 360, 1)
freqs = site.local_wind(x, y, wd=dirs).Sector_frequency_ilk[0, :, 0]
As = site.local_wind(x, y, wd=dirs).Weibull_A_ilk[0, :, 0]
ks = site.local_wind(x, y, wd=dirs).Weibull_k_ilk[0, :, 0]
samps = 50

#constraint_comp = CircleBoundaryConstraint([0, 0], farm_radius)
boundary = np.array([(0, 0), (xu, 0), (xu, yu), (0, yu)])
constraint_comp = XYBoundaryConstraint(boundary, 'rectangle')

#constraint_comp = XYBoundaryConstraint(boundary, 'rectangle')


random = True
# T = 2

#wd = np.arange(0, 360, 20)
#ws = np.arange(3, 25, 1)
FLJ = open('J_log.log', 'w')
FLJ.close()
FL = open('aep_log.log', 'w')
FL.close()
def aep_func(x, y, full=False, **kwargs):
    return 0

def aep_func2(x, y, **kwargs):
    wd = dirs # np.arange(0, 360, 1)
    ws = np.arange(3, 25, 1)
    #i1, i2 = np.where((cabs(
    #        x[:, na] - x[na]) + cabs(y[:, na] - y[na]) + np.eye(len(x))) == 0)
    #if len(i1): return 0
    aep = wake_model(x, y, wd=wd, ws=ws).aep().sum().values #* 1e6
    FL = open('aep_log.log', 'a')
    FL.write('%f\n' % aep)
    FL.close()
    return aep

def aep_jac(x, y, **kwargs):
    if not random:
        np.random.seed(0)
    #i1, i2 = np.where((cabs(
    #        x[:, na] - x[na]) + cabs(y[:, na] - y[na]) + np.eye(len(x))) == 0)
    #if len(i1): return 0
    idx = np.random.choice(np.arange(dirs.size), samps, p=freqs)
    wd = dirs[idx]
    A = As[idx]
    k = ks[idx]
    if not random:
        np.random.seed(0)
    ws = A * np.random.weibull(k)
    jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, wd=wd, time=True)
    return np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) #* 1e6

def aep_jac2(x, y, **kwargs):
    i1, i2 = np.where((cabs(
            x[:, na] - x[na]) + cabs(y[:, na] - y[na]) + np.eye(len(x))) == 0)
    if len(i1): return 0
    wd = dirs # np.arange(0, 360, 1)
    ws = np.arange(3, 25, 1)
    jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, wd=wd, time=False)
    FLJ = open('J_log.log', 'a')
    FLJ.write('ok\n')
    FLJ.close()
    return np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) #* 1e6

n_wt=x.size
aep_comp = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func, objective=True, cost_gradient_function=aep_jac, maximize=True)
aep_comp2 = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func2, objective=True, cost_gradient_function=aep_jac2, maximize=True)
cost_comps = [aep_comp2, aep_comp]

def constr_aggr_func(wtSeparationSquared, boundaryDistances, **kwargs):
    return np.sum(-1 * np.minimum(wtSeparationSquared - (2 * windTurbines.diameter()) ** 2, 0)) + np.sum(np.minimum(boundaryDistances, 0) ** 2)

name = 'sgd_constraint'
component_args = {'input_keys': [('wtSeparationSquared', np.zeros(int(n_wt * (n_wt - 1) / 2))),
                                 ('boundaryDistances', np.zeros((n_wt, 4)))],
                  'n_wt': n_wt,
                  'cost_function': constr_aggr_func,
                  'objective': False,
                  'output_keys': [(name, 0)],
                  'use_penalty': False}
constraint_args = {'name': name, 'lower': 0}
from topfarm.constraint_components.constraint_aggregation import ConstraintAggregation
import scipy

class ScaledInputEasyScipyOptimizeDriver(EasyScipyOptimizeDriver):    
    def get_desvar_kwargs(self, model, desvar_name, desvar_values):
        kwargs = super().get_desvar_kwargs(model, desvar_name, desvar_values)
        if self.options['optimizer'] == 'SLSQP':
            if tuple([int(v) for v in scipy.__version__.split(".")]) < (1, 5, 0):
                # Upper and lower disturbs SLSQP when running with constraints. Add limits as constraints
                model.add_constraint(desvar_name, kwargs.get('lower', None), kwargs.get('upper', None))
                kwargs = {'lower': np.nan, 'upper': np.nan}  # Default +/- sys.float_info.max does not work for SLSQP

            ref0 = 0
            ref1 = 1
            # TODO: Check if the following improves performance
            if len(desvar_values) == 4:
                ref0 = np.min(desvar_values[1])
                ref1 = np.max(desvar_values[2])
                # print ('ref0',ref0)
                # print ('ref1',ref1)

            kwargs.update({'ref0': ref0, 'ref': ref1})
        else:
            raise NotImplementedError()
        return kwargs
         

driver_names = ['SLSQP', 'SGD']
drivers = [ScaledInputEasyScipyOptimizeDriver(maxiter=300, tol=1e-1), EasySGDDriver(maxiter=int(sys.argv[1]), learning_rate=windTurbines.diameter() / 5, max_time=6e15, gamma_min_factor=.1)]
#drivers = [EasyScipyOptimizeDriver(maxiter=1000, tol=1e-3), EasySGDDriver(maxiter=int(sys.argv[1]), learning_rate=windTurbines.diameter() / 5, max_time=6e15)]
for driver_no in [int(sys.argv[4])]:
#for driver_no in range(2):
   constraints = [[SpacingConstraint(2 * windTurbines.diameter()), constraint_comp], ConstraintAggregation([SpacingConstraint(2 * windTurbines.diameter()), constraint_comp],
                                          component_args=component_args, constraint_args=constraint_args)]
   tf = TopFarmProblem(
        design_vars={'x':x, 'y':y}, # setting up our two turbines as design variables
        cost_comp=cost_comps[driver_no], # using dummy cost model
        constraints=constraints[driver_no], # constraint set up for the boundary type provided)
        driver=drivers[driver_no],
        plot_comp=NoPlot(), 
        #plot_comp=[XYPlotComp(), AggregatedConstraintsPlotComponent()][driver_no], #NoPlot(), 
        expected_cost=[10, 1][driver_no],
        )
#tf.driver.learning_rate = windTurbines.diameter() / 5
   cost, state, recorder = tf.optimize()
   recorder.save(f'{driver_names[driver_no]}')

if 1:
    #plt.figure()
    plt.clf()
    for i in [int(sys.argv[4])]:
        rec = TopFarmListRecorder().load(f'recordings/{driver_names[i]}')
        if driver_names[i] == 'SGD':
            aep = []
            x, y = np.array((rec['x'], rec['y']))[:, -1]
            #for x, y in zip(rec['x'], rec['y']):
            aep.append(aep_func2(x, y))

            np.save('sgd_x', rec['x'])
            np.save('sgd_y', rec['y'])
            np.save('sgd_aep', aep)
            np.save('sgd_con', rec['sgd_constraint'])
            np.save('sgd_time', rec['timestamp'])

        else:
            aep = rec['Cost']
            np.save('det_x', rec['x'])
            np.save('det_y', rec['y'])
            np.save('det_aep', aep)
            np.save('det_time', rec['timestamp'])
            np.save('det_con', np.sum(-1 * np.minimum(rec['wtSeparationSquared'][-1, :] - (2 * 80) ** 2, 0)) + np.sum(np.sum(np.minimum(rec['boundaryDistances'][-1, :], 0) ** 2)))
