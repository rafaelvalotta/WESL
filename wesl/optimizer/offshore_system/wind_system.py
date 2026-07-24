import numpy as np
import openmdao.api as om
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from wesl.optimizer.interarray.farmrepo import g1

# Heuristic Wrapper Valotta Rodrigues Perez 2024 (Mauricio Souza DTU thesis 2022)
from wesl.optimizer.interarray.interface import heuristic_wrapper
from wesl.optimizer.interarray.farmrepo import g1

from py_wake.utils.gradients import autograd

from py_wake import BastankhahGaussian
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.site._site import UniformWeibullSite, UniformSite
from py_wake.site.shear import PowerShear

from py_wake.wind_turbines import WindTurbine

import sys
import os
import time
import multiprocessing

path = os.getcwd()

# Currently, the lookup table for the FOWT onyl exists for nrel5MW

from wesl.optimizer.offshore_system.fowt.FOWT_spar_lookup import FOWT_Spar, wave_site
from py_wake.literature.cumulative_sum import nrel5mw

# Generic Unidirectional Site for FOWT modeling
class My_site_Uniform(UniformSite):

    def __init__(self, p_wd=[1], ti=0.11):
        ti = 0.001
        h_ref = 90
        alpha = .1
        UniformSite.__init__(self, ti=ti, shear=PowerShear(h_ref=h_ref, alpha=alpha),
                                    interp_method='linear')

def displacement_solver(wd, ws, fowt_x_base, fowt_y_base, tilt_base, hub_height_base, index_matrix, sim_result, current_wave_site, wave_directions, wave_heights):
    """
    Finds the displacement for all wind direction, wave height/direction for a given wind speed

    Parameters:
    ----------------------------------------------------------------------------------
    Variable                      Description
    wd                   (np.array)                    Wind directions to solve for
    ws                   (np.float)                    Wind speed to solve for
    fowt_x_base          (np.array)                    Undisplaced FOWT x positions
    fowt_y_base          (np.array)                    Undisplaced FOWT y positions
    fowt_tilt_base       (np.array)                    Undisplaced FOWT tilt angle
    hub_height_base      (np.array)                    Undisplaced FOWT hub height
    index_matrix         (np.array)                    Order in which to correctly evaluate displacements
    sim_result           (xarray)                      Pywake wind conditions
    current_wave_site    (FOWT_spar_lookup.wave_site)  Custom Height Direction Weibull Curves (Temporary, proof of concept)
    wave_directions      (np.array)                    Wave directions to solve for
    wave_heights         (np.array)                    Wave heights to solve for
    ----------------------------------------------------------------------------------

    Usage:
    ----------------------------------------------------------------------------------
    Helper function to make floating wind farm compute faster
    Note: update this whole function for readability and modularity
          should be split into 3-4 functions...
    """

    FOWT_Spar_def = FOWT_Spar()
    my_fowt = nrel5mw()
    wf_model_Uniform = Bastankhah_PorteAgel_2014(My_site_Uniform(), my_fowt, k=0.025)
    threshold_x, threshold_y, threshold_z, threshold_gamma = 0.5, 0.5, 0.05, 0.5 # [m], [m], [m], [deg]

    #use range of wave heights in lookup table
    wave_probabilities = current_wave_site.probabilities([0]+wave_heights)

    fowt_x = np.zeros((len(fowt_x_base),len(wd),len(wave_directions),len(wave_heights),))
    fowt_y = np.zeros((len(fowt_x_base),len(wd),len(wave_directions),len(wave_heights),))
    hub_height_wt = np.zeros((len(fowt_x_base),len(wd),len(wave_directions),len(wave_heights),))
    gamma_wt = np.zeros((len(fowt_x_base),len(wd),len(wave_directions),len(wave_heights),))

    for wd_index in range(len(wd)):
        fowt_x_temp = np.copy(fowt_x_base)
        fowt_y_temp = np.copy(fowt_y_base)
        tilt_temp = np.copy(tilt_base)
        hub_height_temp = np.copy(hub_height_base)

        # Get the correct WT computational index order for each wind direction sector
        # index = get_index(wd[wd_index],len(fowt_x_base))
        if 0 <= wd[wd_index] < 45: index = index_matrix[0,:]
        elif 45 <= wd[wd_index] < 90: index = index_matrix[1,:]
        elif 90 <= wd[wd_index] < 135: index = index_matrix[2,:]
        elif 135 <= wd[wd_index] < 180: index = index_matrix[3,:]
        elif 180 <= wd[wd_index] < 225: index = index_matrix[4,:]
        elif 225 <= wd[wd_index] < 270: index = index_matrix[5,:]
        elif 270 <= wd[wd_index] < 315: index = index_matrix[6,:]
        elif 315 <= wd[wd_index] < 360: index = index_matrix[7,:]
        elif wd[wd_index] == 360: index = index_matrix[0,:]
        for wt_index in index:
            WS_eff_WT_new = sim_result.WS_eff.sel(wt=wt_index,wd=wd[wd_index],ws=ws).values
            
            # Get the rotor centre displacemenent for the initial iteration
            xr_new, yr_new, zr_new, gamma_new = FOWT_Spar_def.solve_static_movement(WS_eff_WT_new, wd[wd_index])

            xr_prev     = np.copy(xr_new)
            yr_prev     = np.copy(yr_new)
            zr_prev     = np.copy(zr_new)
            gamma_prev  = np.copy(gamma_new)

            xr_max = 0

            # Iterative loop until convergence
            # similar positions are aggregated for perfomance

            for wave_dir_index in range(len(wave_directions)):
                # index to aggregate probabilities
                wave_dir_sum_index = 0
                for wave_height_index in range(len(wave_heights)):
                    # index to aggregate probabilities
                    wave_height_sum_index = 0
                    itt = 1
                    recalculate_wind = False

                    if wave_probabilities[wave_dir_index, wave_height_index] == 0:
                        continue

                    while True:
                        itt += 1
                        while(isinstance(xr_prev, np.ndarray) and xr_prev.shape != ()):
                            xr_prev = xr_prev[0]
                            yr_prev = yr_prev[0]
                            zr_prev = zr_prev[0]
                            gamma_prev = gamma_prev[0]

                        #Define the updated provisional WF site and turbine position
                        fowt_x_temp[wt_index] = fowt_x_base[wt_index] + xr_prev
                        fowt_y_temp[wt_index] = fowt_y_base[wt_index] + yr_prev
                        hub_height_temp[wt_index] = hub_height_base[wt_index] + zr_prev
                        tilt_temp[wt_index] = tilt_base[wt_index] + gamma_prev
                        
                        #Recalculate the new WS_eff and thrust for the provisional position
                        if recalculate_wind:
                            WS_eff_WT_new = wf_model_Uniform(fowt_x_temp, fowt_y_temp, h=hub_height_temp, tilt=tilt_temp, wd=wd[wd_index], ws=[ws]).WS_eff.sel(wt=wt_index).values
                            recalculate_wind = False

                        #Get the updated rotor centre displacement
                        xr_new, yr_new, zr_new, gamma_new = FOWT_Spar_def.solve_static_movement(WS_eff_WT_new, wd[wd_index], wave_direction=wave_directions[wave_dir_index], wave_height=wave_heights[wave_height_index])

                        #Check convergence
                        diff_xr     = abs(xr_prev - xr_new)
                        diff_yr     = abs(yr_prev - yr_new)
                        diff_zr     = abs(zr_prev - zr_new)
                        diff_gamma  = abs(gamma_prev - gamma_new)
                        if itt>4 or ((diff_xr<threshold_x).all() and (diff_yr<threshold_y).all() and (diff_zr<threshold_z).all() and (diff_gamma<threshold_gamma).all()):
                            # print(fowt_x[wt_index,:,:,:].sum(), wd_index)

                            wave_x_diff  = abs(fowt_x[wt_index, wd_index, wave_dir_sum_index, wave_height_sum_index]   - fowt_x_base[wt_index] - xr_new )
                            wave_y_diff  = abs(fowt_y[wt_index, wd_index, wave_height_sum_index, wave_height_sum_index] - fowt_y_base[wt_index] - yr_new )
                            wave_z_diff  = abs(hub_height_wt[wt_index, wd_index, wave_height_sum_index, wave_height_sum_index] - hub_height_base[wt_index] - zr_new)
                            wave_gamma_diff = abs(gamma_wt[wt_index, wd_index, wave_height_sum_index, wave_height_sum_index]   - tilt_base[wt_index] - gamma_new)

                            threshold_gamma = 0.005
                        
                            # this will agregate very similar positions even if they have different wave directions and heights for performance when evaluating AEP

                            if(((wave_x_diff<threshold_x).all() and (wave_y_diff<threshold_y).all() and (wave_z_diff<threshold_z).all() and (wave_gamma_diff<threshold_gamma).all())):
                                wave_probabilities[wave_dir_sum_index, wave_height_sum_index] += wave_probabilities[wave_dir_index, wave_height_index]
                                wave_probabilities[wave_dir_index, wave_height_index] = 0

                            else:
                                fowt_x[wt_index, wd_index, wave_dir_index, wave_height_index]        = fowt_x_base[wt_index] + xr_new
                                fowt_y[wt_index, wd_index, wave_dir_index, wave_height_index]        = fowt_y_base[wt_index] + yr_new
                                hub_height_wt[wt_index, wd_index, wave_dir_index, wave_height_index] = hub_height_base[wt_index] + zr_new
                                gamma_wt[wt_index, wd_index, wave_dir_index, wave_height_index]      = tilt_base[wt_index] + gamma_new
                                wave_dir_sum_index = wave_dir_index
                                wave_height_sum_index = wave_height_index

                            break
                                
                        recalculate_wind = True
                        xr_prev = np.copy(xr_new)
                        yr_prev = np.copy(yr_new)
                        zr_prev = np.copy(zr_new)
                        gamma_prev = np.copy(gamma_new)
    
    # print(np.sum(wave_probabilities))
    return wave_probabilities, fowt_x, fowt_y, hub_height_wt, gamma_wt


def power_solver(wd, ws, fowt_x, fowt_y, hub_height_wt, gamma_wt, prob):
    my_fowt = nrel5mw()
    wf_model_Uniform = Bastankhah_PorteAgel_2014(My_site_Uniform(), my_fowt, k=0.025)
    return wf_model_Uniform(fowt_x, fowt_y, h=hub_height_wt, tilt=gamma_wt, ws=ws, wd=wd).aep().values.sum() * prob

class FixedBottomWindFarm(om.ExplicitComponent):

    """
    Fixed-Bottom Offshore Wind Farm System for AEP Layout Optimization

    Parameters:
    ----------------------------------------------------------------------------------
    Variable                      Description

    layout_coordinates (float):   x and y wind turbine coordinates 
    sim_res (xarray, float):      Instance of wind farm model in xarray format
    AEP (xarray, float):          Annual Energy Production in xarray format (PyWake)
    ----------------------------------------------------------------------------------

    Usage:
    ----------------------------------------------------------------------------------
    User gives wind turbine layout coordinates (x,y) and the sim_res object when setting 
    the OpenMDAO component. The x and y turbine coordinates can be user-defined, or be
    retrieved using the boundary_layouts util. An example of usage follows below:

    prob.model.add_subsystem('FBWF', 
                         FixedBottomWindFarm(layout_coordinates = np.array([x_coordinates,
                                                                            y_coordinates]),
                                             sim_res = Bastankhah_PorteAgel_2014(site, 
                                                                                 wind_turbines, 
                                                                                 k=0.0324555)),

    """

    def initialize(self):
        self.options.declare("layout_coordinates", 
                             types=np.ndarray, 
                             desc = "Wind farm layout coordinates")
        self.options.declare("sim_res", 
                             desc="xarray from PyWake") # change here to more general

        self.options.declare("n_turbines", 
                             types = int,
                             desc="number of turbines") # change here to more general
        
        self.options.declare('boundary', types=np.ndarray)
        self.options.declare('spacing_diameter', default=6*222, types=(float, int)) # upgrade here for the spacing
        self.options.declare("layout_coordinates", types=np.ndarray)
        self.options.declare("lon_grid_fine", types=np.ndarray)
        self.options.declare("lat_grid_fine",  types=np.ndarray)
        self.options.declare("interpolated_elevation", types=np.ndarray)
        self.options.declare("aep_init", types=xr.DataArray)
        self.options.declare("plot_lim", types = np.ndarray)


    def setup(self):
        # Setting layout coordinates as inputs       
        self.add_input('x', np.zeros(len(self.options["layout_coordinates"][0])))  # X-Layout Coordinates
        self.add_input('y', np.zeros(len(self.options["layout_coordinates"][1])))  # Y-Layout Coordinates


        # Setting AEP as output
        self.add_output('AEP', val=0.0)

        # n_turbines = len(self.options['layout_coordinates'])
        n_turbines = len(self.options["layout_coordinates"][0])
        xl, xu, yl, yu = self.options["plot_lim"]

        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]
        boundary =  self.options["boundary"]
        lon_grid_fine = self.options["lon_grid_fine"]
        lat_grid_fine = self.options["lat_grid_fine"]
        interpolated_elevation = self.options["interpolated_elevation"]

        self.iteration = 0
        self.circles = []
        self.turbine_scatter = None  
        self.cableA = None
        self.cableB = None

        self.fig, self.ax = plt.subplots()
        

        # Defines the water depth map
        plt.pcolormesh(lon_grid_fine, 
                    lat_grid_fine, 
                    interpolated_elevation, 
                    cmap='Blues_r', 
                    shading='auto', 
                    vmin=-120, 
                    vmax=-60)

        plt.colorbar(label="Water Depth (m)")
        plt.plot(boundary[:, 0], 
                boundary[:, 1], 
                label='Boundary', 
                c='black', 
                linestyle = '--')
        plt.tight_layout()
        # plt.ion()
        self.ax.scatter(x_coordinates,
                        y_coordinates, 
                        c='orange', 
                        marker = '.', 
                        s=8, 
                        label='Initial Layout')
        self.text_box = self.ax.text(0.01, 
                                    0.99, 
                                    '', 
                                    transform=self.ax.transAxes, 
                                    verticalalignment='top', 
                                    fontsize=10, 
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # self.text_box.set_text(f"Iteration: {self.iteration}\nAEP Improvement: {-aep} %")
        # self.text_box.set_text(f"AEP Improvement: {-aep} %")


        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        # self.ax.set_xlim(360000, 390000)
        # self.ax.set_ylim(4.53E6, 4.56E6)

        # self.ax.set_xlim(300000, 350000)
        # self.ax.set_ylim(4.54E6, 4.58E6)
        self.ax.set_xlim(xl, xu)
        self.ax.set_ylim(yl, yu)

        print('done')

        

    def compute(self, inputs, outputs):


        print('Entered compute fixedbottomcomponent')
        outputs['AEP'] = -self.options["sim_res"](inputs['x'], inputs['y']).aep().sum()


        x = inputs['x']
        y = inputs['y']
        boundary =  self.options.declare('boundary')
        # aep_init = -self.options["aep_init"]
        # aep_init = -self.options["sim_res"](inputs["layout_coordinates"][0], inputs["layout_coordinates"][1]).aep().sum() 

        # aep = inputs['AEP'].item()
        aep = outputs['AEP'].item()

        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]

        aep_init = -self.options["sim_res"](x_coordinates, y_coordinates).aep().sum() 
        
        aep_init = aep_init.item()
        spacing_radius = self.options['spacing_diameter'] / 2


        if self.turbine_scatter is not None:
            self.turbine_scatter.remove()

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

        self.turbine_scatter = self.ax.scatter(x,
                                               y,
                                               marker = '2', 
                                               c='black', 
                                               label='Current Design')

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
            # print('cable in meters',T[i][2])
            cable_length.append(T[i][2])

        cable_length = np.array(cable_length).sum()

        Cables = [(-1, 2, 1000), (-1, 4, 1500)]

        ##########################################
        cab0,cab1,cost = [],[],[]

        for i in range(62): #wrong: change here to n_wt, not 62
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

        # Update iteration info
        # self.text_box.set_text(
        #     f"Iteration: {self.iteration}\nAEP Improvement: {((-aep / aep_init) - 1) * 100:.3f} %"
        # )
        # self.text_box.set_text(
        #     f"Iteration: {self.iteration}\nAEP Improvement: {-aep} %"
        # )

        # plt.show()

        plt.draw()
        plt.pause(0.001) 
        # self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
        # Rebuild legend without duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # removes duplicates based on label
        self.ax.legend(by_label.values(), by_label.keys(),
                    loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)

        self.text_box.set_text(
            f"Iteration: {self.iteration}\nAEP Improvement: {((aep / aep_init) - 1) * 100:.3f} %"
        )

        # self.text_box.set_text(
        #     f"AEP Improvement: {((aep / aep_init) - 1) * 100:.3f} %"
        # )

        # self.text_box.set_text(
        #     f"Iteration: \nAEP Improvement: {((-aep / aep_init) - 1) * 100:.3f} %"
        # )

        # self.plot_electrical_layout = plot_electrical_cables1(x,y,iter=1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.show()
        self.iteration += 1




    def compute_partials(self, inputs, partials):        
        sim_res = self.options["sim_res"]
        x,y =inputs['x'], inputs['y']

        # Compute exact gradients (PyWake)
        daep = sim_res.aep_gradients(
            gradient_method=autograd,
            wrt_arg=['x', 'y'],
            x = x,
            y = y
        )

        daep_x = daep[0, :]
        daep_y = daep[1, :]

        # Fill OpenMDAO Jacobian
        partials['AEP', 'x'] = -daep_x  # shape (n_turbines,)
        partials['AEP', 'y'] = -daep_y  # shape (n_turbines,)

class OffshoreSystemPlot(om.ExplicitComponent):

    """
    Plot component for an offshore system

    Parameters:
    ----------------------------------------------------------------------------------
    layout_coordinates (float):         x and y wind turbine coordinates 
    sim_res (xarray, float):            Instance of wind farm model in xarray format
    AEP (xarray, float):                Annual Energy Production in xarray format (PyWake)
    ----------------------------------------------------------------------------------

    Usage:
    ----------------------------------------------------------------------------------
    User gives the following arguments when setting the OpenMDAO component:

    """

    def initialize(self):
        self.options.declare('boundary', types=np.ndarray)
        self.options.declare('spacing_diameter', default=6*222, types=(float, int)) # upgrade here for the spacing
        self.options.declare("layout_coordinates", types=np.ndarray)
        self.options.declare("lon_grid_fine", types=np.ndarray)
        self.options.declare("lat_grid_fine",  types=np.ndarray)
        self.options.declare("interpolated_elevation", types=np.ndarray)
        self.options.declare("aep_init", types=xr.DataArray)

    def setup(self):
        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]
        boundary =  self.options["boundary"]
        lon_grid_fine = self.options["lon_grid_fine"]
        lat_grid_fine = self.options["lat_grid_fine"]
        interpolated_elevation = self.options["interpolated_elevation"]

        n = len(x_coordinates)  # global or pass via options
        self.add_input('x', np.zeros(n))
        self.add_input('y', np.zeros(n))
        self.add_input('AEP', val=0.0)
        self.add_output('Dummy', val=0)
        self.iteration = 0
        self.circles = []
        self.turbine_scatter = None  
        self.cableA = None
        self.cableB = None


        # # Beginning of the plot definition
        self.fig, self.ax = plt.subplots()
        # plt.close(self.fig)
        
        # Defines the water depth map
        plt.pcolormesh(lon_grid_fine, 
                    lat_grid_fine, 
                    interpolated_elevation, 
                    cmap='Blues_r', 
                    shading='auto', 
                    vmin=-50, 
                    vmax=-20)

        plt.colorbar(label="Water Depth (m)")
        plt.plot(boundary[:, 0], 
                 boundary[:, 1], 
                 label='Boundary', 
                 c='black', 
                 linestyle = '--')
        plt.tight_layout()
        plt.ion()
        # self.ax.scatter(x_coordinates,
        #                 y_coordinates, 
        #                 c='orange', 
        #                 marker = '.', 
        #                 s=8, 
        #                 label='Initial Layout')
        # self.text_box = self.ax.text(0.01, 
        #                              0.99, 
        #                              '', 
        #                              transform=self.ax.transAxes, 
        #                              verticalalignment='top', 
        #                              fontsize=10, 
        #                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        # self.ax.set_xlabel('X [m]')
        # self.ax.set_ylabel('Y [m]')
        # # self.ax.set_xlim(360000, 390000)
        # # self.ax.set_ylim(4.53E6, 4.56E6)
 
        # self.ax.set_xlim(300000, 350000)
        # self.ax.set_ylim(4.54E6, 4.58E6)
        print('done')


    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        boundary =  self.options['boundary']
        aep_init = -self.options["aep_init"]

        spacing_radius = self.options['spacing_diameter'] / 2

        # Remove previous turbine positions (except for the initial layout)
        if self.turbine_scatter is not None:
            self.turbine_scatter.remove()

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

        self.turbine_scatter = self.ax.scatter(x,
                                               y,
                                               marker = '2', 
                                               c='black', 
                                               label='Current Design')

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
            # print('cable in meters',T[i][2])
            cable_length.append(T[i][2])

        cable_length = np.array(cable_length).sum()

        Cables = [(-1, 2, 1000), (-1, 4, 1500)]

        ##########################################
        cab0,cab1,cost = [],[],[]

        for i in range(62): #wrong: change here to n_wt, not 62
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

        # Update iteration info
        self.text_box.set_text(
            f"Iteration: {self.iteration}\nAEP Improvement: {((-aep / aep_init) - 1) * 100:.3f} %"
        )
        # plt.show()

        plt.draw()
        plt.pause(0.001) 
        # self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
        # Rebuild legend without duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # removes duplicates based on label
        self.ax.legend(by_label.values(), by_label.keys(),
                    loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)


        # self.plot_electrical_layout = plot_electrical_cables1(x,y,iter=1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.iteration += 1

        outputs['Dummy'] = 0


class FloatingBottomWindFarm(om.ExplicitComponent):

    """
    Floating-Bottom Offshore Wind Farm System for AEP Layout Optimization

    Parameters:
    ----------------------------------------------------------------------------------
    Variable                      Description
    Base Coordinates (float):     x and y wind turbine coordinates with no wind or waves
    sim_res (xarray, float):      Instance of wind farm model in xarray format
    wave_site                     weibull distribution of wave significant height 
    using_slurm (bool)            Use this when using slurm to prevent program from trying to use more resources than allocated
    num_cores (int)               Number of Cores used when parallelizing directions, when using_slurm the number defined in the sbatch file is used
    ----------------------------------------------------------------------------------

    Usage:
    ----------------------------------------------------------------------------------
    User gives wind turbine layout coordinates (x,y) and the sim_res object when setting 
    the OpenMDAO component. The x and y turbine coordinates can be user-defined, or be
    retrieved using the boundary_layouts util. An example of usage follows below:

    prob.model.add_subsystem('FBWF', 
                         FixedBottomWindFarm(layout_coordinates = np.array([x_coordinates,
                                                                            y_coordinates]),
                                             sim_res = Bastankhah_PorteAgel_2014(site, 
                                                                                 wind_turbines, 
                                                                                 k=0.0324555)),

    """
    def initialize(self):
        self.options.declare("num_cores", types=int)
        self.options.declare("sim_res", 
                             desc="xarray from PyWake") # change here to more general
        self.options.declare("wave_site")
        self.options.declare('boundary', types=np.ndarray)
        self.options.declare('spacing_diameter', default=6*222, types=(float, int)) # upgrade here for the spacing
        self.options.declare("layout_coordinates", types=np.ndarray)
        self.options.declare("lon_grid_fine", types=np.ndarray)
        self.options.declare("lat_grid_fine",  types=np.ndarray)
        self.options.declare("interpolated_elevation", types=np.ndarray)
        self.options.declare("aep_init", types=xr.DataArray)
        self.options.declare("plot_lim", types = np.ndarray)
        self.options.declare("using_slurm", bool)


    def setup(self):
        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]
        boundary =  self.options["boundary"]
        lon_grid_fine = self.options["lon_grid_fine"]
        lat_grid_fine = self.options["lat_grid_fine"]
        interpolated_elevation = self.options["interpolated_elevation"]

        n = len(x_coordinates)  # global or pass via options
        self.add_input('x', np.zeros(n))
        self.add_input('y', np.zeros(n))
        self.add_output('AEP', val=0.0)
        #adding time period benchmarks

        # used for benchmarking
        self.add_output('t_displacement_calc', val=0.0)
        self.add_output('t_aep_calc', val=0.0)
        self.add_output('t_compute_total', val = 0.0)


        self.iteration = 0
        self.circles = []
        self.turbine_scatter = None  
        self.cableA = None
        self.cableB = None

        if(self.options['using_slurm']):
            self.num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        else:
            self.num_cores = self.options['num_cores']

        # # Beginning of the plot definition
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        # plt.close(self.fig)
        
        # Defines the water depth map
        # Note: Remake this to be Dynamic
        plt.pcolormesh(lon_grid_fine, 
                    lat_grid_fine, 
                    interpolated_elevation, 
                    cmap='Blues_r', 
                    shading='auto', 
                    vmin=-180, 
                    vmax=-50)

        xl, xu, yl, yu = self.options["plot_lim"]

        plt.colorbar(label="Water Depth (m)")
        plt.plot(boundary[:, 0], 
                 boundary[:, 1], 
                 label='Boundary', 
                 c='black', 
                 linestyle = '--')
        plt.tight_layout()
        plt.ion()
        self.ax.scatter(x_coordinates,
                        y_coordinates, 
                        c='orange', 
                        marker = '.', 
                        s=8, 
                        label='Initial Layout')
        self.text_box = self.ax.text(0.01, 
                                     0.99, 
                                     '', 
                                     transform=self.ax.transAxes, 
                                     verticalalignment='top', 
                                     fontsize=10, 
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_aspect('equal')

        self.ax.set_xlim(xl, xu)
        self.ax.set_ylim(yl, yu)
 
        # self.ax.set_xlim(300000, 350000)
        # self.ax.set_ylim(4.54E6, 4.58E6)
        print('done')        

    def compute(self, inputs, outputs):

        #currently only the NREL5MW FOWT exists
        FOWT_Spar_def = FOWT_Spar()
        my_fowt = nrel5mw()
        current_wave_site = self.options['wave_site']

        start_time = time.perf_counter()
        wf_model_Uniform = Bastankhah_PorteAgel_2014(My_site_Uniform(), my_fowt, k=0.025)

        print('Entered compute fixedbottomcomponent')

        n_turbines = len(inputs['x'])

        # number of sections to use in calculation of turbine index 
        n_sections = 8
        index_matrix = np.zeros((n_sections,n_turbines), dtype='int')
        # matrix = np.arange(n_turbines) 

        max_x = max(inputs['x'])
        min_x = min(inputs['x'])
        max_y = max(inputs['y'])
        min_y = min(inputs['y'])

        multiplier = np.sqrt(2) + 1

        # use distance of each WT from a line to calculate which 
        line_equations = [[1,multiplier, -max_x-max_y*multiplier],
                          [multiplier, 1, -max_x*multiplier-max_y],
                          [multiplier, -1, -max_x*multiplier+min_y],
                          [1,-multiplier, -max_x+min_y*multiplier],

                          [-1,-multiplier,min_x+min_y*multiplier],
                          [-multiplier,- 1, min_x*multiplier+min_y],
                          [-multiplier, 1, min_x*multiplier-max_y],
                          [-1,multiplier,min_x-max_y*multiplier]]
        
        for sector_index in range(len(index_matrix)):
            # for index in range(len(index_matrix[sector])):
            distances = []
            evaluation_order = []
            for x, y, turbine_index in zip(inputs['x'], inputs['y'], range(n_turbines)):
                numerator = abs(line_equations[sector_index][0] * x + line_equations[sector_index][1] * y + line_equations[sector_index][2])
                denominator = np.sqrt(line_equations[sector_index][0]**2 + line_equations[sector_index][1]**2)
                distance = numerator / denominator
                insert_index = 0
                while(distances and insert_index < len(distances) and distances[insert_index]<distance ):
                    insert_index +=1
                distances.insert(insert_index, distance)
                evaluation_order.insert(insert_index, turbine_index)
            index_matrix[sector_index] = evaluation_order


        # can be increased for more accuracy, at the cost of performance
        # each of these points is evaluated 
        ws = np.arange(1,26,1)
        wd = np.linspace(0, 360, 30, endpoint=False)
        wave_directions = np.linspace(0,360, len(current_wave_site.f), endpoint=False)
        # lookup table is [1,7] for wave heights
        wave_heights = [1, 3, 5, 7]

        #   FOR THIS SPECIFIC TURBINE TYPE
        hub_height_base = np.full(n_turbines, 90)
        tilt_base = tilt_base = np.zeros(n_turbines)
        fowt_x_base = inputs['x']
        fowt_y_base = inputs['y']

        fowt_x = np.tile(fowt_x_base, (len(wd),1)).T
        fowt_x = np.expand_dims(fowt_x, axis=-1)
        fowt_x = np.repeat(fowt_x, len(ws), axis=-1)

        fowt_y = np.tile(fowt_y_base, (len(wd),1)).T
        fowt_y = np.expand_dims(fowt_y, axis=-1)
        fowt_y = np.repeat(fowt_y, len(ws), axis=-1)
        
        # itt = np.zeros((fowt_x_base.shape[0], wd.shape[0], ws.shape[0]))
        wf_model = self.options["sim_res"]
        sim_result = wf_model(fowt_x_base, fowt_y_base, wd=wd, ws=ws)
        prob = sim_result.P.T.values

        hub_height_wt = np.zeros((fowt_x.shape))
        gamma_wt = np.zeros((fowt_x.shape))

        data = []

        sub_start_time = time.perf_counter()

        #Multi-directional wind speed loop
        for ws_index in range(len(ws)):
            data.append([wd, ws[ws_index],  fowt_x_base, fowt_y_base, tilt_base, hub_height_base, index_matrix, sim_result, current_wave_site, wave_directions, wave_heights])

        data = tuple(data)

        with multiprocessing.Pool(processes=self.num_cores) as pool:
            results = pool.starmap(displacement_solver, data)

        # split inhomogenous array
        # due to bad displacement solver function we have form [wind_direction, wave direction, wave height]
        # this is still valid for power (commutative property), speed matrix because sum(wave_probability[wind_direction]) is 1,
        # but very unreadable and unintuitive. 
        wave_probabilities = [item[0] for item in results]
        fowt_results = [item[1:] for item in results]

        fowt_results = np.array(fowt_results)
        wave_probabilities = np.array(wave_probabilities)
        # transpose to: [category, wind speed, wt_index, wind directions, wave directions, wave height]
        fowt_results = np.transpose(fowt_results, (1, 2, 3, 0, 4, 5))
        fowt_x, fowt_y, hub_height_wt, gamma_wt = fowt_results

        outputs['t_displacement_calc'] = time.perf_counter() - sub_start_time
        sub_start_time = time.perf_counter()
        
        #%%Get AEP contribution per wd and ws
        data = []
        for ws_index in range(len(ws)):
            for wd_index in range(len(wd)):
                for wave_direction_index in range(len(wave_directions)):
                        for wave_height_index in range(len(wave_heights)):
                            # make this line more readable
                            # way too much indexing
                            # these arguments for function that will calculate the annualized power of a certain FOWT scenario
                            # multiply by the probability of that scenario over a year
                            # add together to find AEP
                            if(wave_probabilities[ws_index,wave_direction_index,wave_height_index] == 0):
                                continue

                            data.append([wd[wd_index], ws[ws_index], fowt_x[:,wd_index,ws_index,wave_direction_index,wave_height_index], fowt_y[:,wd_index, ws_index,wave_direction_index,wave_height_index], hub_height_wt[:,wd_index,ws_index,wave_direction_index,wave_height_index], gamma_wt[:,wd_index,ws_index,wave_direction_index,wave_height_index], prob[ws_index, wd_index]*wave_probabilities[ws_index, wave_direction_index, wave_height_index]])
        data = tuple(data)

        with multiprocessing.Pool(processes=self.num_cores) as pool:
            AEP_con = pool.starmap(power_solver, data)

        AEP_con = np.array(AEP_con)
        # AEP_contrib = np.array(AEP_contrib)

        outputs['AEP'] = -AEP_con.sum()

        outputs['t_aep_calc'] = time.perf_counter() - sub_start_time
        sub_start_time = time.perf_counter()

        # fixed_bottom_aep = self.options["sim_res"](inputs['x'], inputs['y']).aep().sum()

        x = inputs['x']
        y = inputs['y']
        boundary =  self.options.declare('boundary')

        # aep = inputs['AEP'].item()
        aep = outputs['AEP'].item()
        x_coordinates = self.options["layout_coordinates"][0]
        y_coordinates = self.options["layout_coordinates"][1]

        aep_init = -self.options["sim_res"](x_coordinates, y_coordinates).aep().sum() 
        
        aep_init = aep_init.item()
        spacing_radius = self.options['spacing_diameter'] / 2

        if self.turbine_scatter is not None:
            self.turbine_scatter.remove()

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

        self.turbine_scatter = self.ax.scatter(x,
                                               y,
                                               marker = '2', 
                                               c='black', 
                                               label='Current Design')

        for wt_index in range(len(fowt_x_base)):

            xi, yi = x[wt_index], y[wt_index]

            xr, yr = fowt_x[wt_index,:,:] - xi, fowt_y[wt_index,:,:] - yi

            drift_radius = xr ** 2 + yr ** 2

            max_drift = np.sqrt(np.max(drift_radius))

            drift_circ = Circle((xi, yi), max_drift, edgecolor='red',
                         linestyle='--', facecolor='none', linewidth=1)
            self.ax.add_patch(drift_circ)
            self.circles.append(drift_circ)

            space_circ = Circle((xi, yi), spacing_radius+max_drift, edgecolor='gray',
                          linestyle='--', facecolor='none', linewidth=1)
            self.ax.add_patch(space_circ)
            self.circles.append(space_circ)

        # Draw electrical layout
        VertexC = g1(x,y,boundary).horns.graph['VertexC']
    
        M = g1(x,y,boundary).horns.graph['M']

        X, Y = np.hstack((VertexC[-1:-1 - M:-1].T, VertexC[:-M].T))
        
        Cables = [(-1, 2, 1000), (-1, 4, 1500)]
        
        cable_length = []

        T = heuristic_wrapper(X, Y,Cables,M,heuristic='CPEW')
        
        T = np.array([[x[0],x[1],x[2],x[3],x[4],Cables[x[4]][2]*x[2]/1000] for x in T])

        for i in range(len(T)):
            cable_length.append(T[i][2])

        cable_length = np.array(cable_length).sum()

        Cables = [(-1, 2, 1000), (-1, 4, 1500)]

        ##########################################
        cab0,cab1,cost = [],[],[]

        for i in range(n_turbines): #wrong: change here to n_wt, not 62
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

        # Update iteration info
        # self.text_box.set_text(
        #     f"Iteration: {self.iteration}\nAEP Improvement: {((-aep / aep_init) - 1) * 100:.3f} %"
        # )
        # self.text_box.set_text(
        #     f"Iteration: {self.iteration}\nAEP Improvement: {-aep} %"
        # )

        self.text_box.set_text(
            f"Iteration: {self.iteration} Initial Fixed bottom AEP: {-aep_init:.3f}\n FOWT AEP: {-aep:.3f}, Percent Diff: {(aep_init-aep)/-aep_init*100:.3f}"
        )

        # plt.show()

        plt.draw()
        plt.pause(0.001) 
        # self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
        # Rebuild legend without duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # removes duplicates based on label
        self.ax.legend(by_label.values(), by_label.keys(),
                    loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)



        # self.text_box.set_text(
        #     f"AEP Improvement: {((aep / aep_init) - 1) * 100:.3f} %"
        # )

        # self.text_box.set_text(
        #     f"Iteration: \nAEP Improvement: {((-aep / aep_init) - 1) * 100:.3f} %"
        # )

        # self.plot_electrical_layout = plot_electrical_cables1(x,y,iter=1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.show()
        self.iteration += 1

        outputs['t_compute_total'] = time.perf_counter() - start_time
        print(f"total time for compute cycle: {outputs['t_compute_total'].item()} seconds")

    # def compute_partials(self, inputs, partials):     
    #     # Unavaliable for floating farm
    #     pass