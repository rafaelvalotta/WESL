#############################################################################
# Roadmap WESLAB

# Top-priority
# 1. AEPWaveOffshoreSystem - Rafael review Antonio's branch first
# 2. Re-train wave energy system (Rafael check the literature first)
# 3. FloatingWindOffshoreSystem - OC3, OC4, OC5, OC6, OC7
# 4. ElectricalInfrastructure - Rafael remodel the OffshoreSystemPlot component
# 5. Implement a wave energy economic cost model (check literature)
# 6. Develop (or adapt, see literature) Risk Model (stress test? check it, SLOAN proposal material)
# 7. Monte-Carlo simulation to populate surroundings for the large MA cluster
# 8. Run optimizations in the GPU. Thread parallelization
# 9. GUI interface for WESLab optimizer


# Medium priority
# 1. Util to plot xdsm matrix - Rafael
# 2. NCG driver - Bruno?
# 3. Memory and computational expenses tracker/plotter
# 3. Smart-start (rename) for joint-farm optimization - Assign
# 4. Util to provide WindIO format 
# 5. util to FLORIS
# 6. util to FOXES
# 7. util to ARD
# 8. util for thread parallelization
# 9. util for GPU parallelization

# Low priority
# 1. util to WISDEM
# 2. util to 
# 3. 

#############################################################################


# WESL imports
from offshore_wind_farms.revolution_wind import x_revwind, y_revwind, boundary_revwind, SG_110_200_DD, Revolutionwind_southforkwind
from optimizer.constraints.wind_farm_constraints import BoundaryConstraint, PairWiseSpacing
from optimizer.offshore_system.wind_system import FixedBottomWindFarm, OffshoreSystemPlot

# WESL optimizer external dependencies
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import openmdao.api as om
from IPython.display import display

# AEP Calculator: PyWake Dependencies
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from utils.plot import get_water_depth_map

##########################################################################################
# Box encompassing Revolution Wind
min_lon, max_lon, min_lat, max_lat = -71.8, -70.2, 40.7, 41.4

# Getting longitude and latitude resolution, and interpolated elevation
# water_depth_map_params = get_water_depth_map(water_depth_data="test2A.nc",
water_depth_map_params = get_water_depth_map(water_depth_data="test3.nc",
                                             min_lon = min_lon,
                                             max_lon = max_lon,
                                             min_lat = min_lat,
                                             max_lat = max_lat)

##########################################################################################
# Instantiating boundary and layout coordinates
boundary = boundary_revwind
x_coordinates, y_coordinates = x_revwind, y_revwind

##########################################################################################
# AEP computations: Low-order wake model, wind turbine, and site from PyWake
wind_turbines = SG_110_200_DD()                              # wind turbine object
site = Revolutionwind_southforkwind()                        # Uniform Weibull object
sim_res = Bastankhah_PorteAgel_2014(site,                    # Wind farm model        
                                    wind_turbines, 
                                    k=0.0324555)
aep_init = sim_res(x_coordinates, y_coordinates).aep().sum() # AEP initial layout

# Defining the OpenMDAO optimization problem
prob = om.Problem()

# Adding subsystems and connections between them
prob.model.add_subsystem('FBWF', 
                         FixedBottomWindFarm(layout_coordinates = np.array([x_coordinates,
                                                                            y_coordinates]),
                                             sim_res = Bastankhah_PorteAgel_2014(site, 
                                                                                 wind_turbines, 
                                                                                 k=0.0324555)),
                         promotes_inputs=['x', 'y'])

prob.model.add_subsystem('Spacing_Constraint', 
                         PairWiseSpacing(n_turbines = 80, 
                                         min_spacing = 5*wind_turbines.diameter()), 
                         promotes_inputs=['x', 'y'])


prob.model.add_subsystem('Boundary_Constraint',
                         BoundaryConstraint(polygon_vertices = boundary,
                                            number_of_turbines = 80),
                         promotes_inputs=['x','y']
)

prob.model.add_subsystem('OffshoreSystemPlot',
                         OffshoreSystemPlot(boundary = boundary, 
                                            layout_coordinates = np.array([x_coordinates,y_coordinates]),
                                            lon_grid_fine = water_depth_map_params[0],
                                            lat_grid_fine = water_depth_map_params[1],
                                            interpolated_elevation = water_depth_map_params[2],
                                            aep_init = -sim_res(x_coordinates, y_coordinates).aep().sum()),
                         promotes_inputs=['x', 'y']
)

prob.model.connect('FBWF.AEP', 'OffshoreSystemPlot.AEP')

# Driver setup
prob.driver = om.ScipyOptimizeDriver(tol = 1e-9)
# prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['maxiter'] = 100
prob.driver.options['disp'] = False

# Input defaults ? (double check)
prob.model.set_input_defaults('x', x_coordinates)
prob.model.set_input_defaults('y', y_coordinates)

# Design variables
prob.model.add_design_var('x', 
                          lower=min(boundary[:,0]), 
                          upper=max(boundary[:,0]), 
                          scaler=0.01)

prob.model.add_design_var('y', 
                          lower=min(boundary[:,1]), 
                          upper=max(boundary[:,1]), 
                          scaler=0.01)

# Objective function
prob.model.add_objective('FBWF.AEP',  scaler=0.01)

# Setting constraints ? (double check)
prob.model.add_constraint('Spacing_Constraint.spacing_violation', scaler=0.01)
prob.model.add_constraint('Boundary_Constraint.boundary_cons', upper=0.0) # Feasibility condition: <= 0 ?

# Setup the problem with all the constraints, design variables, and objective
prob.setup()

# Run the optimization
prob.run_driver()

# Plotting the farm/cables/substantion layout and water depth
display(prob.model.OffshoreSystemPlot.fig)



