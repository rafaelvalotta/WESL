# ##########################################################################################
import matplotlib.pyplot as plt

# # WESL imports
from wesl.optimizer.constraints.wind_farm_constraints import BoundaryConstraint, PairWiseSpacing
from wesl.optimizer.offshore_system.wind_system import FloatingBottomWindFarm

# # WESL optimizer external dependencies
import numpy as np
import openmdao.api as om
from IPython.display import display

# AEP Calculator: PyWake Dependencies
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from wesl.utils.plot import get_water_depth_map
from py_wake.literature.cumulative_sum import nrel5mw

import os
import sys
from multiprocessing import freeze_support, set_start_method

path = os.getcwd()

from pyproj import Transformer

from py_wake.site._site import UniformWeibullSite

from wesl.optimizer.offshore_system.fowt.FOWT_spar_lookup import wave_site

class Maine_test(UniformWeibullSite): # Double-check: plot the wind rose
    def __init__(self, ti=0.07, shear=None):
        f = [8.37, 6.83, 5.38, 4.77, 4.10, 4.80,
            8.22, 13.29, 11.22, 10.14, 11.72, 11.16]
        a = [10.00, 10.29, 9.87, 10.22, 10.01, 9.09,
            11.19, 12.11, 10.65, 10.84, 11.68, 10.89]
        k = [2.241, 1.804, 1.849, 2.086, 1.871, 1.677,
            2.163, 2.836, 2.545, 2.405, 2.489, 2.587]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.name = "Maine Wind Farm"


# 
def main():


    wave_f = [8.37, 6.83, 5.38, 4.77, 4.10, 4.80,
        8.22, 13.29]#, 11.22, 10.14, 11.72, 11.16]
    wave_k = [2.971, 2.971, 2.971, 2.971, 2.971, 2.971, 2.971, 2.971]
    wave_a = [2.027, 2.027, 2.027, 2.027, 2.027, 2.027, 2.027, 2.027,]

    wave_f = np.array(wave_f)

    test_wave_site = wave_site(f=wave_f / sum(wave_f),a=wave_a,k=wave_k)

    n_turbine = int(sys.argv[1])

    min_lon, max_lon, min_lat, max_lat = -69.1, -68.85, 43.02, 43.15

    # Getting longitude and latitude resolution, and interpolated elevation
    water_depth_map_params = get_water_depth_map(water_depth_data=f"{path}/wesl/optimizer/maine/maine_data.nc",
                                                min_lon = min_lon,
                                                max_lon = max_lon,
                                                min_lat = min_lat,
                                                max_lat = max_lat)
    #########################################################################################
    # Instantiating boundary and layout coordinates

    rows = int(np.ceil(np.sqrt(n_turbine)))
    columns =  int(np.ceil(n_turbine / rows))

    spacing = 0.035

    x_coordinates = np.linspace(min_lon+0.02, min_lon+0.02+spacing*columns, columns)
    y_coordinates = np.linspace(min_lat+0.02, min_lat+0.02+spacing*rows, rows,)

    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)

    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()


    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
    x_coordinates, y_coordinates = transformer.transform(x_coordinates, y_coordinates)


    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()

    x_coordinates = x_coordinates[:n_turbine]
    y_coordinates = y_coordinates[:n_turbine]

    x_boundary = [int(np.min(x_coordinates) - 1900), int(np.max(x_coordinates) + 1900)]
    y_boundary = [int(np.min(y_coordinates) - 1900), int(np.max(y_coordinates) + 1900)]


    boundary = np.array([
        [int(x_boundary[0]), int(y_boundary[0])],
        [int(x_boundary[1]), int(y_boundary[0])],
        [int(x_boundary[1]), int(y_boundary[1])],  
        [int(x_boundary[0]), int(y_boundary[1])]   
    ])

    print(boundary)

    ##########################################################################################
    # AEP computations: Low-order wake model, wind turbine, and site from PyWake
    wind_turbines = nrel5mw()                              # wind turbine object
    site = Maine_test() 

    sim_res = Bastankhah_PorteAgel_2014(site,               # Wind farm model        
                                        wind_turbines, 
                                        k=0.0324555)
    aep_init = sim_res(x_coordinates, y_coordinates).aep().sum() # AEP initial layout

    ##########################################################################################
    # Defining the OpenMDAO optimization problem
    prob = om.Problem()

    prob.model.add_subsystem('FBWF', 
                            FloatingBottomWindFarm(layout_coordinates = np.array([x_coordinates,
                                                                                y_coordinates]),
                                                sim_res = Bastankhah_PorteAgel_2014(site, 
                                                                                    wind_turbines, 
                                                                                    k=0.0324555), 
                                                boundary = boundary, 
                                                lon_grid_fine = water_depth_map_params[0],
                                                lat_grid_fine = water_depth_map_params[1],
                                                interpolated_elevation = water_depth_map_params[2],
                                                plot_lim = np.array(x_boundary + y_boundary),
                                                aep_init = aep_init,
                                                num_cores = 6,
                                                using_slurm = False,
                                                wave_site = test_wave_site
                                                ),
                            promotes_inputs=['x', 'y'])

    prob.model.add_subsystem('Spacing_Constraint', 
                            PairWiseSpacing(n_turbines = n_turbine, 
                                            min_spacing = 3*wind_turbines.diameter()), 
                            promotes_inputs=['x', 'y'])

    prob.model.add_subsystem('Boundary_Constraint',
                            BoundaryConstraint(polygon_vertices = boundary,
                                                number_of_turbines = n_turbine),
                            promotes_inputs=['x','y']
    )

    # Driver setup
    prob.driver = om.ScipyOptimizeDriver(tol = 1e-9)
    prob.driver.options['optimizer'] = 'COBYLA'
    # prob.driver.options['optimizer'] = 'SLSQP'
    prob.model.approx_totals(method='fd', step=1e-6, form='forward')
    prob.driver.options['maxiter'] = 10
    prob.driver.options['disp'] = False

    # Input defaults
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

    # Setting constraints
    prob.model.add_constraint('Spacing_Constraint.spacing_violation', scaler=0.01)
    prob.model.add_constraint('Boundary_Constraint.boundary_cons', upper=0.0) 

    recorder = om.SqliteRecorder(f"maine_fowt_test{n_turbine}.sql")

    prob.driver.add_recorder(recorder)

    # Setup the problem with all the constraints, design variables, and objective
    prob.setup()

    prob.driver.recording_options['includes'] = ['FBWF.t_displacement_calc', 'FBWF.t_aep_calc', 'FBWF.t_compute_total'] 


    # Run the optimization
    prob.run_driver()

    prob.cleanup()


    # Plotting the farm/cables/substantion layout and water depth
    # display(prob.model.OffshoreSystemPlot.fig)


if __name__ == '__main__':
    freeze_support()
    set_start_method('fork')   
    main()
