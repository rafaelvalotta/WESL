# WESL imports
from wesl.utils.boundary_layout import plot_bound, farm_to_get_boundary_and_layout
from wesl.offshore_wind_farms.all_wind_farms import wind_farms_europe
import pickle as pkl

# Other dependencies
import numpy as np
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
import os

class SG_14222(GenericWindTurbine):
    def __init__(self):
        GenericWindTurbine.__init__(self, name='SG 14.0-222DD', diameter=222, hub_height=150, 
                                    power_norm=14000, turbulence_intensity=0.07)


# Site definition using PyWake and Global Wind Atlas
class VineyardWind(UniformWeibullSite): # Double-check: plot the wind rose
    def __init__(self, ti=0.07, shear=None):
        f = [6.4633, 7.6414, 6.3740, 5.9969, 4.7711, 4.5698, 
             7.3598, 11.8051, 13.2464, 11.0975, 11.1503, 9.5244]
        a = [10.19, 10.45, 9.47, 9.02, 9.48, 9.66, 
             11.44, 13.27, 12.46, 11.36, 12.39, 10.45]
        k = [2.170, 1.725, 1.713, 1.682, 1.521, 1.479,
             1.666, 2.143, 2.385, 2.146, 2.432, 2.373]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.name = "Vineyard Wind Farm"

farm_to_get = { 'VineyardWind1': wind_farms_europe['VineyardWind1'] }

vars = farm_to_get_boundary_and_layout(farm_to_get)

bound_path = os.path.dirname(os.path.abspath(__file__))+"/"+list(farm_to_get.keys())[0]+'_boundary.pkl'

layout_path = os.path.dirname(os.path.abspath(__file__))+"/"+list(farm_to_get.keys())[0]+'_layout.pkl'

with open(bound_path, 'rb') as f:
    boundary_vineyard = np.array(pkl.load(f))

with open(layout_path, 'rb') as f:
    x_vineyard, y_vineyard = np.array(pkl.load(f))



##SCRIPT BRUNO TO GET VINEYARD DATA

boundary_vineyard = np.array(boundary_vineyard)
x_vineyard = np.array(x_vineyard)
y_vineyard = np.array(y_vineyard)

bx = [f"{float(v):.3f}" for v in boundary_vineyard[:, 0]]
by = [f"{float(v):.3f}" for v in boundary_vineyard[:, 1]]
lx = [f"{float(v):.3f}" for v in x_vineyard]
ly = [f"{float(v):.3f}" for v in y_vineyard]

print("\n# --- site_us.yaml / boundaries ---")
print("vineyard:")
print("  polygons:")
print(f"    x: [{', '.join(bx)}]")
print(f"    y: [{', '.join(by)}]")
print("  crs: +proj=utm +zone=19 +datum=WGS84 +units=m +no_defs")

print("\n# --- wind_farm_us.yaml / layout ---")
print("- name: Vineyard Wind")
print("  layouts:")
print("  - coordinates:")
print(f"      x: [{', '.join(lx)}]")
print(f"      y: [{', '.join(ly)}]")
print("      crs: +proj=utm +zone=19 +datum=WGS84 +units=m +no_defs")