import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
import xarray as xr
from wesl.utils.path_tools import get_data_path
from pyproj import Transformer
from scipy.interpolate import griddata

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from pyproj import Transformer

# changed version to work for Rev Wind
def get_water_depth_map(water_depth_data, min_lon, max_lon, min_lat, max_lat):
    ds = xr.open_dataset(get_data_path(water_depth_data))

    # Subset dataset by bounding box
    subset_ds = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    elevation = subset_ds.elevation.values
    lon = subset_ds.lon.values
    lat = subset_ds.lat.values

    # Ensure matching shapes
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Convert to UTM coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
    x_utm, y_utm = transformer.transform(lon_grid, lat_grid)

    # Create finer grid for interpolation
    num_points = 500
    x_fine = np.linspace(x_utm.min(), x_utm.max(), num_points)
    y_fine = np.linspace(y_utm.min(), y_utm.max(), num_points)
    x_grid_fine, y_grid_fine = np.meshgrid(x_fine, y_fine)

    # Interpolate onto the fine grid
    points = np.column_stack((x_utm.ravel(), y_utm.ravel()))
    values = elevation.ravel()
    interpolated_elevation = griddata(points, values, (x_grid_fine, y_grid_fine), method='cubic')

    return x_grid_fine, y_grid_fine, interpolated_elevation



## OLD version October 21st
# def get_water_depth_map(water_depth_data,
#                         min_lon,
#                         max_lon,
#                         min_lat,
#                         max_lat):
#     # ds = xr.open_dataset(get_data_path("test2.nc"))
#     ds = xr.open_dataset(get_data_path("test2A.nc"))

#     # ds = xr.open_dataset(get_data_path("test2.nc"))
#     # Extract elevation, longitude, and latitude
#     # elevation = ds.elevation
#     # lon = ds.lon
#     # lat = ds.lat
#     subset_ds = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
#     subset_elevation = subset_ds.elevation
#     subset_lon = subset_ds.lon
#     subset_lat = subset_ds.lat
#     transformer = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
#     # subset_lon, subset_lat = transformer.transform(subset_lon, subset_lat)
#     subset_lon, subset_lat = transformer.transform(subset_lon, subset_lat)

#     # Create a much finer grid for interpolation
#     num_points = 500
#     lon_fine = np.linspace(subset_lon.min(), subset_lon.max(), num_points)
#     lat_fine = np.linspace(subset_lat.min(), subset_lat.max(), num_points)
#     lon_grid_fine, lat_grid_fine = np.meshgrid(lon_fine, lat_fine)

#     # Prepare data for interpolation by creating a meshgrid from the subsetted 1D coordinates
#     lon_grid, lat_grid = np.meshgrid(subset_lon, subset_lat)
#     points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
#     values = subset_elevation.values.ravel()

#     # Interpolate the data onto the fine grid
#     interpolated_elevation = griddata(points, 
#                                     values, 
#                                     (lon_grid_fine, lat_grid_fine), 
#                                     method='cubic')

#     # Conversion again from degrees to utm
#     min_lon, min_lat = transformer.transform(min_lon, min_lat)
#     max_lon, max_lat = transformer.transform(max_lon, max_lat)

#     # return min_lon, min_lat, max_lon, max_lat
#     return lon_grid_fine, lat_grid_fine, interpolated_elevation

# class OffshoreSystemPlot(om.ExplicitComponent):
#     """
#     Plot component for an offshore systems
#     """

#     def initialize(self):
#         self.options.declare('boundary', types=np.ndarray)
#         self.options.declare('spacing_diameter', default=6*222, types=(float, int)) # upgrade here for the spacing
#         self.options.declare('long_grid_fine', types = )

#     def setup(self):
#         n = len(x_coordinates)  # global or pass via options
#         self.add_input('x', np.zeros(n))
#         self.add_input('y', np.zeros(n))
#         self.add_input('AEP', val=0.0)

#         self.iteration = 0
#         self.circles = []
#         self.turbine_scatter = None  
#         self.cableA = None
#         self.cableB = None


#         # # Beginning of the plot definition
#         self.fig, self.ax = plt.subplots()
#         # plt.close(self.fig)
        
#         # Defines the water depth map
#         plt.pcolormesh(lon_grid_fine, 
#                     lat_grid_fine, 
#                     interpolated_elevation, 
#                     cmap='Blues_r', 
#                     shading='auto', 
#                     vmin=-50, 
#                     vmax=-20)

#         plt.colorbar(label="Water Depth (m)")
#         plt.plot(boundary[:, 0], 
#                  boundary[:, 1], 
#                  label='Boundary', 
#                  c='black', 
#                  linestyle = '--')
#         plt.tight_layout()
#         # plt.ion()
#         self.ax.scatter(x_coordinates,
#                         y_coordinates, 
#                         c='orange', 
#                         marker = '.', 
#                         s=8, 
#                         label='Initial Layout')
#         self.text_box = self.ax.text(0.01, 
#                                      0.99, 
#                                      '', 
#                                      transform=self.ax.transAxes, 
#                                      verticalalignment='top', 
#                                      fontsize=10, 
#                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
#         self.ax.set_xlabel('X [m]')
#         self.ax.set_ylabel('Y [m]')
#         self.ax.set_xlim(360000, 390000)
#         self.ax.set_ylim(4.53E6, 4.56E6)

#     def compute(self, inputs, outputs):
#         x = inputs['x']
#         y = inputs['y']
#         # aep = float(inputs['AEP'])
#         # aep = float(inputs['AEP'][()])  # extract the scalar
#         # or
#         aep = inputs['AEP'].item()



#         spacing_radius = self.options['spacing_diameter'] / 2

#         # Remove previous turbine positions (except for the initial layout)
#         if self.turbine_scatter is not None:
#             self.turbine_scatter.remove()

#         if self.cableA is not None:
#             for line in self.cableA:
#                 line.remove()
#             self.cableA = None

#         if self.cableB is not None:
#             for line in self.cableB:
#                 line.remove()
#             self.cableB = None

#         # Remove old circles
#         for circ in self.circles:
#             circ.remove()
#         self.circles.clear()

#         self.turbine_scatter = self.ax.scatter(x,
#                                                y,
#                                                marker = '2', 
#                                                c='black', 
#                                                label='Current Design')

#         # Draw new spacing circles
#         for xi, yi in zip(x, y):
#             circ = Circle((xi, yi), spacing_radius, edgecolor='gray',
#                           linestyle='--', facecolor='none', linewidth=1)
#             self.ax.add_patch(circ)
#             self.circles.append(circ)

#         # Draw electrical layout
#         VertexC = g1(x,y,boundary).horns.graph['VertexC']
    
#         M = g1(x,y,boundary).horns.graph['M']

#         X, Y = np.hstack((VertexC[-1:-1 - M:-1].T, VertexC[:-M].T))
        
#         Cables = [(-1, 2, 1000), (-1, 4, 1500)]
        
#         cable_length = []

#         T = heuristic_wrapper(X, Y,Cables,M,heuristic='CPEW')

#         T = np.array([[x[0],x[1],x[2],x[3],x[4],Cables[x[4]][2]*x[2]/1000] for x in T])


#         for i in range(len(T)):
#             # print('cable in meters',T[i][2])
#             cable_length.append(T[i][2])

#         cable_length = np.array(cable_length).sum()

#         Cables = [(-1, 2, 1000), (-1, 4, 1500)]

#         ##########################################
#         cab0,cab1,cost = [],[],[]

#         for i in range(62):
#             if T[i][4] == 0.0:
#                 cab0.append(i)
#                 cost.append(Cables[0][2]*T[i][2])
#             else:
#                 cab1.append(i)
#                 cost.append(Cables[1][2]*T[i][2])

#         ##########################################
#         total_cable_cost = np.array(cost).sum()

#         WTcoords = np.array([x,y])

#         WTcentroid = np.array([WTcoords[0].mean(), WTcoords[1].mean()]) #UPDATE THIS TO MATCH REAL
        
#         total_cable_cost =  round(total_cable_cost*0.000001, 3) 
        
#         # plt.scatter(WTcentroid[0],WTcentroid[1],label='Substation',c='red')
#         self.ax.scatter(WTcentroid[0],WTcentroid[1],label='Substation',c='red')

#         # colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
#         colors = ['y', '#b87333' ]

#         b = T

#         Cables = np.array(Cables)

#         for i in range(Cables.shape[0]):
#             index = b[:,4]==i
#             if index.any():
#                 n1xs = X[b[index,0].astype(int)-1]
#                 n2xs = X[b[index,1].astype(int)-1]
#                 n1ys = Y[b[index,0].astype(int)-1]
#                 n2ys = Y[b[index,1].astype(int)-1]
#                 xs = np.vstack([n1xs,n2xs])
#                 ys = np.vstack([n1ys,n2ys])

#                 if i == 0:
#                     self.cableA = self.ax.plot(xs,ys,'{}'.format(colors[i]),linewidth=1.2)
                
#                 elif i == 1:
#                     self.cableB = self.ax.plot(xs,ys,'{}'.format(colors[i]),linewidth=1.2)

#         # Update iteration info
#         self.text_box.set_text(
#             f"Iteration: {self.iteration}\nAEP Improvement: {((-aep / aep_init) - 1) * 100:.3f} %"
#         )
#         # plt.show()

#         plt.draw()
#         plt.pause(0.001) 
#         # self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
#         # Rebuild legend without duplicates
#         handles, labels = self.ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))  # removes duplicates based on label
#         self.ax.legend(by_label.values(), by_label.keys(),
#                     loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)


#         # self.plot_electrical_layout = plot_electrical_cables1(x,y,iter=1)

#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()

#         self.iteration += 1
