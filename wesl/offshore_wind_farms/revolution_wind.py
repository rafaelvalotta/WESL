# WESL imports
from wesl.utils.boundary_layout import plot_bound, farm_to_get_boundary_and_layout
from wesl.offshore_wind_farms.all_wind_farms import wind_farms_europe
import pickle as pkl

# Other dependencies
import numpy as np
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
import os

class SG_110_200_DD(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SG 11.0-200 DD', diameter=200, hub_height=140,
                             power_norm=11000, turbulence_intensity=0.08)


class Revolutionwind_southforkwind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.2913, 7.2204, 6.3564, 5.5052, 4.743, 4.7018, 7.7244, 11.6506, 13.331, 11.079, 10.9413, 9.4554] 
        a = [10.37, 10.58, 9.66, 9.33, 9.68, 10.57, 11.77, 13.87, 12.79, 12.12, 12.36, 10.3] 
        k = [2.053, 1.729, 1.635, 1.689, 1.412, 1.42, 1.529, 1.943, 2.076, 2.197, 2.295, 2.201] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        # self.initial_position = np.array([xinit, yinit]).T
        self.name = "Revolutionwind Southforkwind"

farm_to_get = {'Revolutionwind_southforkwind': wind_farms_europe['Revolutionwind_southforkwind'] } # fix: not farms_europe

vars = farm_to_get_boundary_and_layout(farm_to_get)

bound_path = os.path.dirname(os.path.abspath(__file__))+"/"+list(farm_to_get.keys())[0]+'_boundary.pkl'

print(bound_path)

layout_path = os.path.dirname(os.path.abspath(__file__))+"/"+list(farm_to_get.keys())[0]+'_layout.pkl'
print(layout_path)

with open(bound_path, 'rb') as f:
    boundary_revwind = np.array(pkl.load(f))

with open(layout_path, 'rb') as f:
    x_revwind, y_revwind = np.array(pkl.load(f))



boundary_revwind = np.array(boundary_revwind)
x_revwind = np.array(x_revwind)
y_revwind = np.array(y_revwind)

bx = [f"{float(v):.3f}" for v in boundary_revwind[:, 0]]
by = [f"{float(v):.3f}" for v in boundary_revwind[:, 1]]
lx = [f"{float(v):.3f}" for v in x_revwind]
ly = [f"{float(v):.3f}" for v in y_revwind]

print("\n# --- site_us.yaml / boundaries ---")
print("Revolutionwind Southforkwind:")
print("  polygons:")
print(f"    x: [{', '.join(bx)}]")
print(f"    y: [{', '.join(by)}]")
print("  crs: +proj=utm +zone=19 +datum=WGS84 +units=m +no_defs")

print("\n# --- wind_farm_us.yaml / layout ---")
print("- name: Revolutionwind Southforkwind")
print("  layouts:")
print("  - coordinates:")
print(f"      x: [{', '.join(lx)}]")
print(f"      y: [{', '.join(ly)}]")
print("      crs: +proj=utm +zone=19 +datum=WGS84 +units=m +no_defs")