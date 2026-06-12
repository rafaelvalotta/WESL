# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import numpy as np
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.examples.data.hornsrev1 import HornsrevV80

import pandas as pd
from pathlib import Path

# It is possible to create our own site class by inheriting from UniformWeibullSite
# ----- Site Definition -----
# class VineyardWind(UniformWeibullSite):
#     def __init__(self, ti=0.07, shear=None):
#         f = np.array([6.4633, 7.6414, 6.3740, 5.9969, 4.7711, 4.5698,
#                       7.3598, 11.8051, 13.2464, 11.0975, 11.1503, 9.5244],
#                      dtype=float)
#         a = np.array([10.19, 10.45, 9.47, 9.02, 9.48, 9.66,
#                       11.44, 13.27, 12.46, 11.36, 12.39, 10.45],
#                      dtype=float)
#         k = np.array([2.170, 1.725, 1.713, 1.682, 1.521, 1.479,
#                       1.666, 2.143, 2.385, 2.146, 2.432, 2.373],
#                      dtype=float)

#         f = f / f.sum()

#         self.f = f
#         self.a = a
#         self.k = k

#         super().__init__(self.f, self.a, self.k, ti=ti, shear=shear)
#         self.name = "Vineyard Wind Farm"




HERE = Path(__file__).parent
BASE_DIR = HERE / "Designs"

def load_initial_layout(base_dir, csv_filename="default.csv"):
    full_path = Path(base_dir) / csv_filename

    if not full_path.exists():
        print(f"File not found: {full_path}")
        return None, None, None
    
    df = pd.read_csv(full_path)
    
    WT_X_INIT = df['x_coordinate'].values
    WT_Y_INIT = df['y_coordinate'].values
    
    x0, y0, x1, y1 = df['x0'].iloc[0], df['y0'].iloc[0], df['x1'].iloc[0], df['y1'].iloc[0]
    BOUNDARY_VERTICES = np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1]
    ], dtype=float)

    print(f"Loaded: {csv_filename} ({len(WT_X_INIT)} turbines)")
    print(f"Boundary: {x0:.0f}m x {y1:.0f}m")
    
    return WT_X_INIT, WT_Y_INIT, BOUNDARY_VERTICES

WT_X_INIT, WT_Y_INIT, BOUNDARY_VERTICES = load_initial_layout(BASE_DIR)

MIN_SPACING_D = 2.0 # Minimum spacing in multiples of diameter, the standard is 2D

# ----- PyWake Model -----

SITE = LillgrundSite() # Using Lillgrund site data (PyWake Package)
SITE.interp_method = 'linear' #<-_interp_method for 360 degree grid
TURBINE = HornsrevV80()
WFM = BastankhahGaussian(SITE, TURBINE)


# ----- Helper for optimization -----
def get_setup_params(csv_filename="default.csv"):
    """
    Returns everything required for optimization.
    csv_filename: nome do CSV (x,y,x0,y0,x1,y1)
    """
    WT_X_INIT, WT_Y_INIT, BOUNDARY_VERTICES = load_initial_layout(BASE_DIR, csv_filename)
    
    if WT_X_INIT is None:
        raise ValueError(f"Failed to load {csv_filename}")
    
    return {
        'site': SITE,
        'turbine': TURBINE,
        'wfm': WFM,
        'x_init': WT_X_INIT,
        'y_init': WT_Y_INIT,
        'boundary_vertices': BOUNDARY_VERTICES,
        'min_spacing_d': MIN_SPACING_D,
        'n_turbines': len(WT_X_INIT),
        'diameter': TURBINE.diameter()
    }