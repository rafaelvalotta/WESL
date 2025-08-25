# WESL Optimizer Dependencies
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pyproj import Transformer
import openmdao.api as om
from scipy.spatial.distance import pdist, squareform
from matplotlib.path import Path
from matplotlib.patches import Circle, Rectangle
import os
import sys, pathlib


# AEP Calculator: PyWake Dependencies
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

# Electrical layout (Alencar's (DTU) MSc Thesis heuristic Model)
from interarray.interface import heuristic_wrapper
from interarray.farmrepo import g,g1

# --- module-friendly header: lets this script run from anywhere without hardcoding ---
if __package__ is None or __package__ == "":
    # Add project root (the parent of "WaveEnergy") to sys.path at runtime
    this = pathlib.Path(__file__).resolve()
    # If this is .../optimizer/WaveEnergy/optimize_wec_loc.py, parents[1] == .../optimizer
    sys.path.insert(0, str(this.parents[1]))
# -------------------------------------------------------------------------------

from WaveEnergy.waveField import RandomGridWaveField
from WaveEnergy.wec_device import OSWECDevice
from WaveEnergy.wecFarm import WecFarm



# Open the NetCDF water depth dataset
current_dir = os.getcwd() # grabs current directory
ds = xr.open_dataset(current_dir+'/test2.nc')

# Extract elevation, longitude, and latitude
elevation = ds.elevation
lon = ds.lon
lat = ds.lat


# Vineyard Wind turbine coordinates (lat and long) in degrees
x_coordinates = np.array([ -70.46480443069069, -70.44211908841751, -70.44211908841751,
    -70.41943374614488, -70.39800870066537, -70.46354413389761,
    -70.41880359774835, -70.3973785522688,  -70.3746932099962,
    -70.37532335839221, -70.46354413389761, -70.44085879162498,
    -70.41880359774835, -70.39674840387228, -70.3746932099962,
    -70.46228383710448, -70.44148894002149, -70.41817344935181,
    -70.39674840387228, -70.39611825547571, -70.41628300416268,
    -70.48622947617021, -70.48559932777368, -70.46354413389761,
    -70.44085879162498, -70.44022864322841, -70.43959849483187,
    -70.41691315255923, -70.43833819803878, -70.4389683464353,
    -70.48559932777368, -70.48433903098109, -70.46291398550102,
    -70.48433903098109, -70.46291398550102, -70.48433903098109,
    -70.46165368870794, -70.48370888258455, -70.46291398550102,
    -70.48307873418798, -70.46165368870794, -70.46102354031193,
    -70.50828467004682, -70.50765452165027, -70.50639422485719,
    -70.50639422485719, -70.50639422485719, -70.50513392806405,
    -70.50576407646062, -70.5290795671298,  -70.5290795671298,
    -70.5290795671298,  -70.5290795671298,  -70.52718912194068,
    -70.55113476100642, -70.55113476100642, -70.55050461260986,
    -70.55050461260986, -70.57192965808936, -70.57192965808936,
    -70.57129950969284, -70.59335470356946])

y_coordinates = np.array([41.13771377705393, 41.12205063563724, 41.10448448520549,
     41.087863530517666,41.08833847329956, 41.12062652827484,
     41.07123837043275, 41.07123837043275, 41.07171343336421,
     41.0565097169777,  41.10400965915031, 41.087863530517666,
     41.05555936799715, 41.055084188358165,41.03940133472503,
     41.087388584302744,41.07123837043275, 41.03654950521948,
     41.03797543541734, 41.020862234409464,41.00469568286513,
     41.10448448520549,41.08643868157486, 41.07076330406855,
     41.055084188358165,41.037500128783535,41.02181308448496,
     41.020386804223534,41.004220136000555,40.98757383390051,
     41.070288234271686,41.052708238676985,41.053658628846705,
     41.03654950521948,41.03654950521948,41.01991137060574,
     41.020862234409464,41.00279347481808,41.003269031977425,
     40.98757383390051,40.9880495008658, 40.96997174404672,
     41.10305999673841,41.08596372506105,41.069813161042106,
     41.05318343547802,41.03654950521948,41.018485049161285,
     41.003269031977425,41.08691363465488,41.07076330406855,
     41.055084188358165,41.03702481871761,41.01943593355608,
     41.069338084379865,41.052708238676985,41.03512354413252,
     41.01943593355608,41.052708238676985,41.03559886792698,
     41.01896049307459,41.03464821690599])

# Converts lat/long from degrees to utm at Vineyard Wind area
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
x_coordinates, y_coordinates = transformer.transform(x_coordinates, y_coordinates)

# Define a range for the water depth map
min_lon = -70.8
max_lon = -70.2
min_lat = 40.7
max_lat = 41.3

# Select a subset of the data for the water depth interpolation map 
subset_ds = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
subset_elevation = subset_ds.elevation
subset_lon = subset_ds.lon
subset_lat = subset_ds.lat

# convert subsets from degrees to utm
subset_lon, subset_lat = transformer.transform(subset_lon, subset_lat)

# Create a much finer grid for interpolation
num_points = 500
lon_fine = np.linspace(subset_lon.min(), subset_lon.max(), num_points)
lat_fine = np.linspace(subset_lat.min(), subset_lat.max(), num_points)
lon_grid_fine, lat_grid_fine = np.meshgrid(lon_fine, lat_fine)

# Prepare data for interpolation by creating a meshgrid from the subsetted 1D coordinates
lon_grid, lat_grid = np.meshgrid(subset_lon, subset_lat)
points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
values = subset_elevation.values.ravel()

# # Interpolate the data onto the fine grid (original grid)
# interpolated_elevation = griddata(points, 
#                                   values, 
#                                   (lon_grid_fine, lat_grid_fine), 
#                                   method='cubic')

# --- Visualization grid in UTM to cover WF + WEC area ---
VIZ_XMIN, VIZ_XMAX = 345000.0, 455000.0   # <-- change to taste
VIZ_YMIN, VIZ_YMAX = 4.520e6, 4.570e6 

# Use the full NetCDF coverage for interpolation
min_lon = float(ds.lon.min()); max_lon = float(ds.lon.max())
min_lat = float(ds.lat.min()); max_lat = float(ds.lat.max())

subset_ds = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
subset_lon = subset_ds.lon.values
subset_lat = subset_ds.lat.values

# lon/lat -> UTM for the bathy points
subset_x, subset_y = transformer.transform(subset_lon, subset_lat)
PX, PY = np.meshgrid(subset_x, subset_y)                     # points in UTM
points = np.column_stack((PX.ravel(), PY.ravel()))
values = subset_ds.elevation.values.ravel()

# Viz grid in UTM (wider limits)
nx, ny = 800, 600
x_viz = np.linspace(VIZ_XMIN, VIZ_XMAX, nx)
y_viz = np.linspace(VIZ_YMIN, VIZ_YMAX, ny)
X_viz, Y_viz = np.meshgrid(x_viz, y_viz)

# Interpolate (no nearest fallback—let NaNs show where there’s no data)
depth_viz = griddata(points, values, (X_viz, Y_viz), method='cubic')


# Conversion again from degrees to utm
min_lon, min_lat = transformer.transform(min_lon, min_lat)
max_lon, max_lat = transformer.transform(max_lon, max_lat)
##########################################################################################

boundary = np.array([[-70.479677,  41.138551],
    [-70.479558,  41.133073],
    [-70.486831,  41.133163],
    [-70.48695 ,  41.127684],
    [-70.5009  ,  41.127505],
    [-70.501019,  41.116547],
    [-70.51497 ,  41.116367],
    [-70.51485 ,  41.105587],
    [-70.528984,  41.105265],
    [-70.528579,  41.094605],
    [-70.54313 ,  41.094605],
    [-70.542725,  41.083639],
    [-70.557276,  41.083639],
    [-70.556871,  41.072671],
    [-70.571422,  41.072518],
    [-70.57122 ,  41.061548],
    [-70.58513 ,  41.061578],
    [-70.58513 ,  41.050385],
    [-70.599231,  41.050385],
    [-70.59886 ,  41.039469],
    [-70.613333,  41.039469],
    [-70.613333,  41.028832],
    [-70.627064,  41.028552],
    [-70.626322,  41.006992],
    [-70.584387,  41.007552],
    [-70.584016,  40.996349],
    [-70.540597,  40.99719 ],
    [-70.540226,  40.97562 ],
    [-70.512394,  40.97562 ],
    [-70.511711,  40.965184],
    [-70.49749 ,  40.965184],
    [-70.496956,  40.943704],
    [-70.468336,  40.944107],
    [-70.468692,  40.954848],
    [-70.454471,  40.95525 ],
    [-70.454471,  40.965721],
    [-70.440249,  40.966258],
    [-70.440427,  40.976861],
    [-70.426481,  40.977054],
    [-70.426481,  40.987957],
    [-70.412359,  40.987957],
    [-70.412359,  40.998616],
    [-70.398559,  40.998858],
    [-70.398238,  41.009757],
    [-70.384116,  41.010241],
    [-70.384746,  41.020623],
    [-70.377459,  41.020978],
    [-70.377459,  41.025944],
    [-70.370172,  41.026298],
    [-70.371821,  41.096771],
    [-70.428547,  41.096147],
    [-70.429789,  41.139194],
    [-70.4797  ,  41.13857 ]])

boundary[:, 0], boundary[:, 1] = transformer.transform(boundary[:, 0], boundary[:, 1])

##########################################################################################
# Generic Wind Turbine class from PyWake
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

# PyWake setup for AEP computations
wind_turbines = SG_14222() # wind turbine object
site = VineyardWind() # site object
sim_res = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)
aep_init = sim_res(x_coordinates, y_coordinates).aep().sum() #AEP
_diameter = 222

# Beginning of WESL optimizer
class FixedBottomWindFarm(om.ExplicitComponent):
    """
    Fixed-Bottom Wind Farm System
    """
    def setup(self):

        # X-Layout Coordinates
        self.add_input('x', np.zeros(len(x_coordinates)))


        # Y-Layout Coordinates
        self.add_input('y', np.zeros(len(y_coordinates)))
        
        # Annual Energy Production
        self.add_output('AEP', val=0.0)

    def setup_partials(self):
        
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['AEP'] = -sim_res(x, y).aep().sum()

        print(outputs['AEP'])


class WecFarmAepComp(om.ExplicitComponent):
    """
    Total WEC AEP [GWh] for a set of device positions (x_wec, y_wec),
    computed via the provided WecFarm instance.

    Options
    -------
    farm : WecFarm  # your existing class (holds site, device, bins, etc.)
    """

    def initialize(self):
        self.options.declare("farm")  # instance of your WecFarm

    def setup(self):
        farm = self.options["farm"]
        nd = int(len(farm.x))  # number of devices from the farm

        # inputs default to the farm's current layout
        self.add_input("x_wec", val=farm.x.copy())
        self.add_input("y_wec", val=farm.y.copy())

        # scalar output in GWh
        self.add_output("wec_AEP_total", val=0.0)

        # finite-difference step ≈ half a grid cell for stable SLSQP gradients
        site = farm.site
        if ("x" in site.ds.coords) and (site.ds.x.size > 1):
            dx = float(np.diff(site.ds.x).mean())
        else:
            dx = max(1.0, float(np.ptp(farm.x)) / 20.0)
        if ("y" in site.ds.coords) and (site.ds.y.size > 1):
            dy = float(np.diff(site.ds.y).mean())
        else:
            dy = max(1.0, float(np.ptp(farm.y)) / 20.0)
        fd_step = 0.5 * min(dx, dy)

        self.declare_partials(of="wec_AEP_total", wrt=["x_wec", "y_wec"],
                              method="fd", form="central",
                              step_calc="abs", step=fd_step)

    def compute(self, inputs, outputs):
        farm = self.options["farm"]

        # update farm layout from optimizer design vars
        farm.x = np.asarray(inputs["x_wec"], float)
        farm.y = np.asarray(inputs["y_wec"], float)

        # total AEP in GWh (your farm already does the unit conversion)
        outputs["wec_AEP_total"] = farm.aep_farm()

class OffshoreSystemPlot(om.ExplicitComponent):
    """
    Plot component for an offshore systems
    """

    def initialize(self):
        self.options.declare('boundary', types=np.ndarray)
        self.options.declare('spacing_diameter', default=6*222, types=(float, int)) # upgrade here for the spacing
        self.options.declare('wec_diam', default=150.0, types=(float, int))  # WEC spacing ring (diameter)
        self.options.declare('wec_boundary', types=np.ndarray)


    def setup(self):
        n = len(x_coordinates)  # global or pass via options
        self.add_input('x', np.zeros(n))
        self.add_input('y', np.zeros(n))
        self.add_input('wf_AEP', val=0.0)
        self.add_input('wec_AEP', val=0.0)
        self.add_input('x_wec', val=x_wec_init.copy())
        self.add_input('y_wec', val=y_wec_init.copy())



        self.iteration = 0
        self.circles = []
        self.turbine_scatter = None  
        self.cableA = None
        self.cableB = None
        self.wec_init = None
        self.wec_scatter = None
        self.wf_init  = None      # <- ADD
        self.tot_init = None      # <- ADD
        self.wec_rects = []
        self.wec_init_rects = []
        self.wec_move_arrows = []




        # Beginning of the plot definition
        self.fig, self.ax = plt.subplots(figsize=[8, 4])
        
        # Water depth background on the wider viz grid
        self.im = self.ax.pcolormesh(
            X_viz, Y_viz, depth_viz,
            cmap='Blues_r', shading='auto', vmin=-50, vmax=-20
        )
        self.im.set_zorder(0)


        self.fig.colorbar(self.im, ax=self.ax, label="Water Depth (m)")


        # # Defines the water depth map
        # plt.pcolormesh(lon_grid_fine, 
        #             lat_grid_fine, 
        #             interpolated_elevation, 
        #             cmap='Blues_r', 
        #             shading='auto', 
        #             vmin=-50, 
        #             vmax=-20)

        # plt.colorbar(label="Water Depth (m)")
        plt.plot(boundary[:, 0], 
                 boundary[:, 1], 
                 label='Boundary', 
                 c='black', 
                 linestyle = '--')
        
        wec_b = self.options['wec_boundary']
        wec_b_closed = np.vstack([wec_b, wec_b[0]])  # append first point to close loop
        plt.plot(wec_b_closed[:, 0], wec_b_closed[:, 1],
                label='WEC boundary', c='k', linestyle=':', lw=1.5)

        self.fig.tight_layout(rect=[0, 0, 1, 0.86])   # a bit more top margin
        self.fig.subplots_adjust(top=0.82)
        plt.ion()
        self.ax.scatter(x_coordinates,
                        y_coordinates, 
                        c='orange', 
                        marker = '.', 
                        s=8, 
                        label='WT (init)')
        
        # --- Draw WEC initial rectangles (lime) once in setup ---
        rect_w0, rect_h0 = 250.0, 750.0
        for i in range(len(x_wec_init)):
            r0 = Rectangle((x_wec_init[i] - rect_w0/2.0, y_wec_init[i] - rect_h0/2.0),
                        rect_w0, rect_h0,
                        facecolor='lime', edgecolor='green',
                        linewidth=1.0, alpha=0.75, zorder=2,
                        label='WEC (init)' if i == 0 else None)
            self.ax.add_patch(r0)
            self.wec_init_rects.append(r0)

        self.text_box = self.ax.text(0.01, 
                                     0.99, 
                                     '', 
                                     transform=self.ax.transAxes, 
                                     verticalalignment='top', 
                                     fontsize=8, 
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        # Use the wider inspection window
        # self.ax.set_xlim(VIZ_XMIN, VIZ_XMAX)
        # self.ax.set_ylim(VIZ_YMIN, VIZ_YMAX)
        self.ax.set_xlim(360000, 415000)
        self.ax.set_ylim(4.53E6, 4.560E6)
        self.ax.set_aspect('equal', adjustable='box')

    def compute(self, inputs, outputs):
        

        x = inputs['x']
        y = inputs['y']
        spacing_radius = self.options['spacing_diameter'] / 2

        # Remove previous turbine positions (except for the initial layout)
        if self.turbine_scatter is not None:
            self.turbine_scatter.remove()

        if self.wec_scatter is not None:
            self.wec_scatter.remove()

        # Remove old WEC rectangles
        for r in getattr(self, 'wec_rects', []):
            r.remove()
        self.wec_rects = []

        if self.cableA is not None:
            for line in self.cableA:
                line.remove()
            self.cableA = None

        if self.cableB is not None:
            for line in self.cableB:
                line.remove()
            self.cableB = None

        # Remove old circles
        for circ in self.circles:
            circ.remove()
        self.circles.clear()

        # REMOVE OLD ARROWS (ADD THIS BLOCK)
        for a in getattr(self, 'wec_move_arrows', []):
            a.remove()
        self.wec_move_arrows = []

        self.turbine_scatter = self.ax.scatter(x,
                                               y,
                                               marker = '2', 
                                               c='black', 
                                               label='WT (final)')

        # Draw new spacing circles
        for xi, yi in zip(x, y):
            circ = Circle((xi, yi), spacing_radius, edgecolor='gray',
                          linestyle='--', facecolor='none', linewidth=1)
            self.ax.add_patch(circ)
            self.circles.append(circ)

        # Draw electrical layout
        VertexC = g1(x,y,boundary).horns.graph['VertexC']
    
        M = g1(x,y,boundary).horns.graph['M']

        X, Y = np.hstack((VertexC[-1:-1 - M:-1].T, VertexC[:-M].T))
        
        Cables = [(-1, 2, 1000), (-1, 4, 1500)]
        
        cable_length = []

        T = heuristic_wrapper(X, Y,Cables,M,heuristic='CPEW')

        T = np.array([[x[0],x[1],x[2],x[3],x[4],Cables[x[4]][2]*x[2]/1000] for x in T])


        for i in range(len(T)):
            print('cable in meters',T[i][2])
            cable_length.append(T[i][2])

        cable_length = np.array(cable_length).sum()

        Cables = [(-1, 2, 1000), (-1, 4, 1500)]

        ##########################################
        cab0,cab1,cost = [],[],[]

        for i in range(62):
            if T[i][4] == 0.0:
                cab0.append(i)
                cost.append(Cables[0][2]*T[i][2])
            else:
                cab1.append(i)
                cost.append(Cables[1][2]*T[i][2])

        ##########################################
        total_cable_cost = np.array(cost).sum()

        WTcoords = np.array([x,y])

        WTcentroid = np.array([WTcoords[0].mean(), WTcoords[1].mean()]) #UPDATE THIS TO MATCH REAL
        
        total_cable_cost =  round(total_cable_cost*0.000001, 3) 
        
        # plt.scatter(WTcentroid[0],WTcentroid[1],label='Substation',c='red')
        self.ax.scatter(WTcentroid[0],WTcentroid[1],label='Substation',c='red')

        # --- AEP numbers and baselines ---
        wf_neg = float(inputs['wf_AEP'])   # FBWF.AEP is negative
        wf = -wf_neg                       # positive WF AEP [GWh]
        wec = float(inputs['wec_AEP'])     # positive WEC AEP [GWh]
        

        if self.wf_init is None and wf > 0.0:
            self.wf_init = wf
        if self.wec_init is None and wec > 0.0:
            self.wec_init = wec

        tot = wf + wec
        if self.tot_init is None and (self.wf_init is not None) and (self.wec_init is not None):
            self.tot_init = self.wf_init + self.wec_init

        wf_pct  = 0.0 if not self.wf_init  else (wf  / self.wf_init  - 1.0) * 100.0
        wec_pct = 0.0 if not self.wec_init else (wec / self.wec_init - 1.0) * 100.0
        tot_pct = 0.0 if not self.tot_init else (tot / self.tot_init - 1.0) * 100.0


        # --- Draw WEC rectangles + spacing ring ---
        xw = np.asarray(inputs['x_wec'])
        yw = np.asarray(inputs['y_wec'])
        rect_w = 250   # thin
        rect_h = 750.0   # tall
        # wec_r  = float(self.options['wec_diam']) / 2.0

        for i in range(len(xw)):
            rect = Rectangle((xw[i] - rect_w/2.0, yw[i] - rect_h/2.0),
                            rect_w, rect_h,
                            angle=0.0,
                            facecolor='gold', edgecolor='brown',
                            linewidth=1.0, alpha=1, zorder=8,
                            label='WEC (final)' if i == 0 else None)
            self.ax.add_patch(rect); self.wec_rects.append(rect)

        # Draw arrows from initial -> final WEC positions (simple and clear)
        x0 = np.asarray(x_wec_init)
        y0 = np.asarray(y_wec_init)

        n = len(x0)
        for i in range(n):
            arr = self.ax.annotate(
                '', xy=(xw[i], yw[i]), xytext=(x0[i], y0[i]),
                arrowprops=dict(arrowstyle='-|>', linestyle=(0, (6, 3)), lw=0.75, color='black', shrinkA=0, shrinkB=0)
            )
            self.wec_move_arrows.append(arr)

            # ring = Circle((xw[i], yw[i]), wec_r, edgecolor='gold',
            #             linestyle='--', facecolor='none',
            #             linewidth=1.0, alpha=0.9,
            #             label='WEC spacing' if i == 0 else None)
            # self.ax.add_patch(ring); self.wec_rects.append(ring)

        # --- Update textbox (WF, WEC, Total) ---
        self.text_box.set_text(
            f"Iter: {self.iteration}\n"
            f"WF  AEP: {wf:.3f} GWh ({wf_pct:.2f}%)\n"
            f"WEC AEP: {wec:.3f} GWh ({wec_pct:.2f}%)\n"
            f"Total  : {tot:.3f} GWh ({tot_pct:.2f}%)"
        )



        # colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
        colors = ['y', '#b87333' ]

        b = T

        Cables = np.array(Cables)

        for i in range(Cables.shape[0]):
            index = b[:,4]==i
            if index.any():
                n1xs = X[b[index,0].astype(int)-1]
                n2xs = X[b[index,1].astype(int)-1]
                n1ys = Y[b[index,0].astype(int)-1]
                n2ys = Y[b[index,1].astype(int)-1]
                xs = np.vstack([n1xs,n2xs])
                ys = np.vstack([n1ys,n2ys])

                if i == 0:
                    self.cableA = self.ax.plot(xs,ys,'{}'.format(colors[i]),linewidth=1.2)
                
                elif i == 1:
                    self.cableB = self.ax.plot(xs,ys,'{}'.format(colors[i]),linewidth=1.2)

        # plt.show()

        plt.draw()
        plt.pause(0.001) 
        # Clear any previous figure-level legends to avoid stacking
        for lg in list(self.fig.legends):
            lg.remove()

        handles, labels = self.ax.get_legend_handles_labels()
        # Deduplicate by label
        by_label = {lab: h for h, lab in zip(handles, labels) if lab}
        self.fig.legend(by_label.values(), by_label.keys(),
                        loc='upper center', bbox_to_anchor=(0.5, 0.995),
                        ncol=4, fontsize=8, frameon=True)


        # self.plot_electrical_layout = plot_electrical_cables1(x,y,iter=1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.iteration += 1


class PairWiseSpacing(om.ExplicitComponent):
    """
    Pair-Wise Spacing Constraint Component
    """
    def setup(self):

        # X-Layout Coordinates
        self.add_input('x', np.zeros(len(x_coordinates)))

        # Y-Layout Coordinates
        self.add_input('y', np.zeros(len(y_coordinates)))
        
        self.add_output('Spacing_Matrix', np.zeros(len(x_coordinates)*len(x_coordinates)-len(x_coordinates)))

    def setup_partials(self):
        
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        points = np.column_stack((x, y))  

        dist_matrix = squareform(pdist(points))

        flat_dist_matrix = dist_matrix.reshape(-1)
        nonzero_values = flat_dist_matrix[flat_dist_matrix != 0]

        d_min = 6*222

        if np.any(nonzero_values < d_min):
            print("Some values below the minimum were found.")
        else:
            print("All nonzero values are above the minimum.")

        min_spac = min(nonzero_values)/80

        print(f"Minimum Spacing is {min_spac:.2f}D")

        outputs['Spacing_Matrix'] = nonzero_values

class SpacingConstraintComp(om.ExplicitComponent):
    """
    c[k] = d_ij - D_min  for all i<j.  Enforce lower=0 to get d_ij >= D_min.
    """
    def initialize(self):
        self.options.declare("nd", types=int)
        self.options.declare("D_min", types=float)

    def setup(self):
        nd = self.options["nd"]
        self.pairs = [(i, j) for i in range(nd) for j in range(i+1, nd)]
        self.add_input("x", val=np.zeros(nd))
        self.add_input("y", val=np.zeros(nd))
        self.add_output("c", val=np.zeros(len(self.pairs)))

        self.declare_partials(of="c", wrt=["x", "y"],
                              method="fd", form="central",
                              step_calc="abs", step=50.0)

    def compute(self, inputs, outputs):
        x = np.asarray(inputs["x"], float)
        y = np.asarray(inputs["y"], float)
        D_min = float(self.options["D_min"])

        c = np.zeros(len(self.pairs))
        for k, (i, j) in enumerate(self.pairs):
            c[k] = np.hypot(x[i]-x[j], y[i]-y[j]) - D_min
        outputs["c"] = c


class PolygonBoundaryConstraint(om.ExplicitComponent):
    """
    Constraint to ensure turbines stay within the defined polygon boundary
    """

    def initialize(self):
        self.options.declare('boundary', types=np.ndarray)

    def setup(self):
        self.add_input('x', shape=len(x_coordinates))
        self.add_input('y', shape=len(y_coordinates))
        self.add_output('inside_polygon', shape=len(x_coordinates))  # Boolean mask (0 if inside, 1 if outside)

        self.declare_partials('*', '*', method='fd')

        # Create a Path object from the boundary
        self.polygon_path = Path(self.options['boundary'])

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        points = np.column_stack((x, y))

        inside = self.polygon_path.contains_points(points)  # returns boolean array
        outputs['inside_polygon'] = np.logical_not(inside).astype(float)  # constraint: all must be <= 0

class WecBoundaryConstraint(om.ExplicitComponent):
    """Constraint: all WEC devices must lie inside the given polygon (<= 0)."""
    def initialize(self):
        self.options.declare('boundary', types=np.ndarray)  # Nx2 polygon in UTM

    def setup(self):
        self.add_input('x_wec', shape=nd_wec)
        self.add_input('y_wec', shape=nd_wec)
        self.add_output('inside_polygon', shape=nd_wec)  # 0 if inside, 1 if outside
        self.declare_partials('*', '*', method='fd')      # COBYLA is derivative-free anyway
        self.polygon_path = Path(self.options['boundary'])

    def compute(self, inputs, outputs):
        pts = np.column_stack((inputs['x_wec'], inputs['y_wec']))
        inside = self.polygon_path.contains_points(pts)     # True if inside
        outputs['inside_polygon'] = np.logical_not(inside).astype(float)  # want <= 0


# --- WEC site (wave climate) on same UTM frame as Vineyard ---
# Bin edges (coarse is fine for step 1)
H_edges = np.linspace(0.5, 4.0, 16)
T_edges = np.linspace(5.0, 10.0, 16)
D_edges = np.linspace(0.0, 90.0, 13)

# WEC wavefield grid to the RIGHT of the wind farm
xg = np.linspace(387500.0, 415000.0, 41)
yg = np.linspace(4530000.0, 4560000.0, 41)
# --- Simple WEC rectangular boundary (slightly smaller than the wave grid) ---
pad = 1000.0  # shrink on each side [m]; tweak as you like
wec_xmin = float(xg.min()) + pad
wec_xmax = float(xg.max()) - pad
wec_ymin = float(yg.min()) + pad 
wec_ymax = float(yg.max()) - pad 

# rectangle as a polygon (clockwise)
wec_boundary_rect = np.array([
    [wec_xmin, wec_ymin],
    [wec_xmax, wec_ymin],
    [wec_xmax, wec_ymax],
    [wec_xmin, wec_ymax],
])



wec_site = RandomGridWaveField(H_edges, T_edges, D_edges, xg, yg, smooth_sigma=0.5)

# Device (alpha can be tuned later; 5.0 usually lands ~2–3 GWh/device with your surrogate)
wec_device = OSWECDevice()

x_wec_init = np.array([
    390000.0, 408000.0, 402000.0,
    394000.0, 401000.0, 405300.0,
    396000.0, 405000.0, 404700.0,
    399000.0, 403000.0, 390000.0
], dtype=float)

y_wec_init = np.array([
    4538000.0, 4534000.0, 4535000.0,
    4545000.0, 4545000.0, 4545000.0,
    4550000.0, 4539500.0, 4440000.0,
    4555000.0, 4455000.0, 4550000.0
], dtype=float)

nd_wec = len(x_wec_init)

# Instantiate, then set options explicitly

wec_farm = WecFarm(site=wec_site, x=x_wec_init, y=y_wec_init, device=wec_device)
# Create the component and pass ONLY the farm option
wec_comp = WecFarmAepComp()
wec_comp.options['farm'] = wec_farm

prob = om.Problem()
prob.model.add_subsystem('FBWF', 
                         FixedBottomWindFarm(), 
                         promotes_inputs=['x', 'y'])

prob.model.add_subsystem('Spacing_Constraint', 
                         PairWiseSpacing(), 
                         promotes_inputs=['x', 'y'])

prob.model.add_subsystem(
    'WF_Boundary',
    PolygonBoundaryConstraint(boundary=boundary),
    promotes_inputs=['x', 'y']
)

prob.model.add_subsystem('OffshoreSystemPlot',
                         OffshoreSystemPlot(boundary=boundary, wec_boundary=wec_boundary_rect),
                         promotes_inputs=['x', 'y', 'x_wec', 'y_wec']
)

# Add it to the problem
prob.model.add_subsystem('WEC', wec_comp, promotes_inputs=['x_wec', 'y_wec'])

prob.model.add_subsystem(
    'WEC_Boundary',
    WecBoundaryConstraint(boundary=wec_boundary_rect),
    promotes_inputs=['x_wec', 'y_wec']
)

# Combined objective: minimize wind_AEP_neg - w * wec_AEP
prob.model.add_subsystem('obj', om.ExecComp('f = A - w*W', w=0.5), #w=0.5 is a sane starting weight so WEC moves matter. Increase if you want the optimizer to favor WEC more.
                         promotes_outputs=['f'])

prob.model.connect('FBWF.AEP', 'OffshoreSystemPlot.wf_AEP')
prob.model.connect('WEC.wec_AEP_total', 'OffshoreSystemPlot.wec_AEP')
prob.model.connect('FBWF.AEP', 'obj.A')               # A is negative wind AEP
prob.model.connect('WEC.wec_AEP_total', 'obj.W')      # W is positive WEC AEP




prob.model.set_input_defaults('x', x_coordinates)
prob.model.set_input_defaults('y', y_coordinates)
prob.model.set_input_defaults('x_wec', x_wec_init)
prob.model.set_input_defaults('y_wec', y_wec_init)

prob.model.add_design_var('x', lower=min(boundary[:,0]), upper=max(boundary[:,0]), scaler=0.01)
prob.model.add_design_var('y', lower=min(boundary[:,1]), upper=max(boundary[:,1]), scaler=0.01)


wec_xmin = float(wec_site.ds.x.min()); wec_xmax = float(wec_site.ds.x.max())
wec_ymin = float(wec_site.ds.y.min()); wec_ymax = float(wec_site.ds.y.max())

prob.model.add_design_var('x_wec', lower=wec_xmin, upper=wec_xmax,
                          ref=wec_xmax, ref0=wec_xmin)
prob.model.add_design_var('y_wec', lower=wec_ymin, upper=wec_ymax,
                          ref=wec_ymax, ref0=wec_ymin)

prob.model.add_objective('f', scaler=0.01)

prob.model.add_constraint('Spacing_Constraint.Spacing_Matrix', lower=6*_diameter , scaler=0.01)
prob.model.add_constraint('WEC_Boundary.inside_polygon', upper=0.0)
prob.model.add_constraint('WF_Boundary.inside_polygon', upper=0.0)






prob.driver = om.ScipyOptimizeDriver(tol = 1e-9)

prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['maxiter'] = 500
prob.driver.options['tol']     = 1e-4
prob.driver.opt_settings['maxfun'] = 20000


prob.setup()

# --- Sanity check: evaluate once so WEC AEP is computed and printed ---
prob.run_model()

# Baselines
wec_aep_init = prob.get_val('WEC.wec_AEP_total').item()
wf_aep_init  = -prob.get_val('FBWF.AEP').item()  # make positive
tot_aep_init = wf_aep_init + wec_aep_init


prob.run_driver()

wec_aep_opt = prob.get_val('WEC.wec_AEP_total').item()
wf_aep_opt  = -prob.get_val('FBWF.AEP').item()
tot_aep_opt = wf_aep_opt + wec_aep_opt

wec_d  = wec_aep_opt - wec_aep_init
wf_d   = wf_aep_opt  - wf_aep_init
tot_d  = tot_aep_opt - tot_aep_init

wec_pct = 0.0 if wec_aep_init == 0 else (wec_aep_opt / wec_aep_init - 1.0) * 100.0
wf_pct  = 0.0 if wf_aep_init  == 0 else (wf_aep_opt  / wf_aep_init  - 1.0) * 100.0
tot_pct = 0.0 if tot_aep_init == 0 else (tot_aep_opt / tot_aep_init - 1.0) * 100.0

print("\n=== AEP Summary (GWh) ===")
print(f"WF   : init={wf_aep_init:.3f}  opt={wf_aep_opt:.3f}  Δ={wf_d:.3f}  ({wf_pct:.2f}%)")
print(f"WEC  : init={wec_aep_init:.3f} opt={wec_aep_opt:.3f} Δ={wec_d:.3f} ({wec_pct:.2f}%)")
print(f"TOTAL: init={tot_aep_init:.3f} opt={tot_aep_opt:.3f} Δ={tot_d:.3f} ({tot_pct:.2f}%)")


