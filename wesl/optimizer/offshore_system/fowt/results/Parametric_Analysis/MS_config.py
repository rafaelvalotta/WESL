# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:55:12 2023

@author: mariomartingarcia

Parametric studies on Mooring Lines length
"""

# -*- coding: utf-8 -*-
"""
Computing the motion simulation using an iterative process until the FOWT is 
not being displaced.

Call get_wt_index function to get the correct order of FOWT motion calculation 
for each wd section

- ws loop to compute the whole range of ws [4,25)
- wd loop to compute the whole range of wd [0,360)

@author: mariomartingarcia
"""
# BASIC PYTHON LIB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

#(1) - (ws, CT, P)
# # use data defined at: https://backend.orbit.dtu.dk/ws/portalfiles/portal/235761709/Accepted_preprint_3_.pdf
ws_Ct_power_table = np.loadtxt('/Users/mariomartingarcia/Desktop/DTU/MSc_thesis/MMG_thesis/Mario_GitLab_repo/mario-martin-thesis/inputs/ws_Ct_power_table_LW5.dat')

#%%INPUT DATA
#ws = np.array([4] + list(range(6, 25, 2)) + [25])   #12 ws
ws = np.array([11])
wd = np.linspace(0, 360, 30, endpoint=False)        #30 wd
threshold_x, threshold_y, threshold_z, threshold_gamma = 0.5, 0.5, 0.05, 0.5 # [m], [m], [m], [deg]

#%%DEFINE WT
from py_wake.wind_turbines import WindTurbine#, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

import sys
sys.path.append('/Users/mariomartingarcia/Desktop/DTU/MSc_thesis/MMG_thesis/Mario_GitLab_repo/mario-martin-thesis/JuFeng_model/src')

#Define WT parameters
hub_height=90
D_rotor=126

my_fowt = WindTurbine(name="fowt_spar", diameter=D_rotor, hub_height=hub_height, 
                       powerCtFunction=PowerCtTabular(ws_Ct_power_table[:, 0],ws_Ct_power_table[:, 2],'kW',ws_Ct_power_table[:, 1]))

#Define floater parameters
from FOWT_cls import FOWT_Spar


#%% DEFINE SPAR-type FLOATER

FOWT_Spar_def = FOWT_Spar(tot_mass=8066048,hub_height=90,msl_to_bottom=120,msl_to_CB=62,
                              msl_to_CM=77.98,rated_power=5000,D_rotor=126, ws_Ct_power_table=ws_Ct_power_table,
                              msl_to_fairlead=70,mooring_line_length=902.2,radius_to_anchor=853.87,sea_depth=320,
                              msl_to_taper_top=4,msl_to_taper_bottom=12,platform_top_to_msl=10,D_taper_top=6.5,
                              D_taper_bottom=9.4,D_platform_top=6.5,water_density=997, gravity_acceleration=9.8, 
                              static_Ct=0.1, air_density=1.225,
                              movement_interp_funcs_flag=False,
                              solve_for_movement_by_interp=False)

# with open('fowt_object.pkl', 'wb') as f:
#     pickle.dump(FOWT_Spar_def, f)

# with open('/Users/mariomartingarcia/Desktop/DTU/MSc_thesis/MMG_thesis/Mario_GitLab_repo/mario-martin-thesis/myWF/fowt_object.pkl', 'rb') as f:
#     FOWT_Spar_def = pickle.load(f)
    
#% DEFINE SITE & WF MODEL in PyWake
from py_wake import BastankhahGaussian
from py_wake.site._site import UniformWeibullSite, UniformSite
from py_wake.site.shear import PowerShear

#MY SITE
fowt_x_base = np.array([0,1000,2000,3000,     0,1000,2000,3000,    0,1000,2000,3000,    0,1000,2000,3000],dtype='float64')
fowt_y_base = np.array([0,0,0,0,     630,630,630,630,    1260,1260,1260,1260,    1890,1890,1890,1890],dtype='float64')
tilt_base = np.zeros((fowt_x_base.shape))
hub_height_base = np.full(len(fowt_x_base), 90)

class My_site(UniformWeibullSite):

    def __init__(self, ti=0.11):
        #n_windrose_sectors = 12
        #wind_directions = [i * 30.0 for i in range(n_windrose_sectors)]
        f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
        0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
        a = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
        9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
        k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
        2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
        ti = 0.001
        h_ref = hub_height
        alpha = .1
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=PowerShear(h_ref=h_ref, alpha=alpha),
                                    interp_method='linear')
        self.initial_position = np.array([fowt_x_base, fowt_y_base]).T

site = My_site()
#site.plot_wd_distribution()

# WF model simulation
wf_model = BastankhahGaussian(site, my_fowt) # turbulence model can be added
sim_res = wf_model(fowt_x_base, fowt_y_base, wd=wd, ws=ws)
prob = sim_res.P.T.values

#% UniformSite to compute AEP
class My_site_Uniform(UniformSite):

    def __init__(self, p_wd=[1], ti=0.11):
        ti = 0.001
        h_ref = hub_height
        alpha = .1
        UniformSite.__init__(self, ti=ti, shear=PowerShear(h_ref=h_ref, alpha=alpha),
                                    interp_method='linear')
        self.initial_position = np.array([fowt_x_base, fowt_y_base]).T

wf_model_Uniform = BastankhahGaussian(My_site_Uniform(), my_fowt)

#% Get index order
n_turbines=len(fowt_x_base)
n_sections=8
matrix = np.arange(n_turbines).reshape(int(np.sqrt(n_turbines)), int(np.sqrt(n_turbines)))

index_matrix = np.zeros((n_sections,n_turbines), dtype='int')

index_matrix[0,:] = matrix.flatten()[::-1]                       # 0 <= wd < 45
index_matrix[1,:] = np.fliplr(np.rot90(matrix, 1)).flatten()     # 45 <= wd < 90
index_matrix[2,:] = np.rot90(matrix, 1).flatten()                # 90 <= wd < 135
index_matrix[3,:] = np.fliplr(matrix).flatten()                  # 135 <= wd < 180
index_matrix[4,:] = np.arange(0,n_turbines)                      # 180 <= wd < 225
index_matrix[5,:] = np.flipud(np.rot90(matrix, 1)).flatten()     # 225 <= wd < 270
index_matrix[6,:] = np.rot90(matrix, 1).flatten()[::-1]          # 270 <= wd < 315
index_matrix[7,:] = np.fliplr(matrix).flatten()[::-1]            # 315 <= wd < 360

#%bFOWT MOTION (iterative process)

thrust_eff = np.zeros((fowt_x_base.shape[0], wd.shape[0]))
itt = np.zeros((fowt_x_base.shape[0], wd.shape[0], ws.shape[0]))

# Setting up the FOWT WF coordinates for multiple wind directions - [wt_index,wd_index]

fowt_x = np.tile(fowt_x_base, (len(wd),1)).T
fowt_x = np.expand_dims(fowt_x, axis=-1)
fowt_x = np.repeat(fowt_x, len(ws), axis=-1)

fowt_y = np.tile(fowt_y_base, (len(wd),1)).T
fowt_y = np.expand_dims(fowt_y, axis=-1)
fowt_y = np.repeat(fowt_y, len(ws), axis=-1)


hub_height_wt = np.zeros((fowt_x.shape))
gamma_wt = np.zeros((fowt_x.shape))

threshold_x, threshold_y, threshold_z, threshold_gamma = 0.5, 0.5, 0.05, 0.5 # [m], [m], [m], [deg]

full_time_start = time.time()


#%% Loop for MS_config

MS_config = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 115, 120])

for MS_config_index in MS_config:
    
    #Multi-directional wind speed loop
    for ws_index in range(len(ws)):
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
                # Compute the effective wind speed with PyWake adn Thrust with Ju Feng's model
                WS_eff_WT_new = sim_res.WS_eff.sel(wt=wt_index,wd=wd[wd_index],ws=ws[ws_index]).values
    
                thrust_eff_new = FOWT_Spar_def.get_thrust(WS_eff_WT_new,tilt_in_deg=None)
                
                # Get the rotor centre displacemenent for the initial iteration
                xr_new, yr_new, zr_new, gamma_new = FOWT_Spar_def.solve_static_movement_direct(thrust_eff_new, wd[wd_index],MS_config_index)
    
                xr_prev     = np.copy(xr_new)
                yr_prev     = np.copy(yr_new)
                zr_prev     = np.copy(zr_new)
                gamma_prev  = np.copy(gamma_new)
    
                # Iterative loop until convergence
                itt[wt_index,wd_index,ws_index] = 1
                while True:
                    itt[wt_index,wd_index,ws_index] += 1
    
                    #Define the updated provisional WF site and turbine position
                    fowt_x_temp[wt_index] = fowt_x_base[wt_index] + xr_prev
                    fowt_y_temp[wt_index] = fowt_y_base[wt_index] + yr_prev
                    hub_height_temp[wt_index] = hub_height_base[wt_index] + zr_prev
                    tilt_temp[wt_index] = tilt_base[wt_index] + gamma_prev
                    
                    #Recalculate the new WS_eff and thrust for the provisional position
                    WS_eff_WT_new = wf_model_Uniform(fowt_x_temp, fowt_y_temp, h=hub_height_temp, tilt=tilt_temp, wd=wd[wd_index], ws=ws[ws_index]).WS_eff.sel(wt=wt_index).values
    
                    thrust_eff_new = FOWT_Spar_def.get_thrust(WS_eff_WT_new,tilt_in_deg=tilt_temp[wt_index])
    
                    #Get the updated rotor centre displacement
                    xr_new, yr_new, zr_new, gamma_new = FOWT_Spar_def.solve_static_movement_direct(thrust_eff_new, wd[wd_index],MS_config_index)
    
                    #Check cocnvergence
                    diff_xr     = abs(xr_prev - xr_new)
                    diff_yr     = abs(yr_prev - yr_new)
                    diff_zr     = abs(zr_prev - zr_new)
                    diff_gamma  = abs(gamma_prev - gamma_new)
                    if itt[wt_index,wd_index,ws_index]>6 or ((diff_xr<threshold_x).all() and (diff_yr<threshold_y).all() and (diff_zr<threshold_z).all() and (diff_gamma<threshold_gamma).all()) :
                        fowt_x[wt_index,wd_index,ws_index] = fowt_x_base[wt_index] + xr_new
                        fowt_y[wt_index,wd_index,ws_index] = fowt_y_base[wt_index] + yr_new
                        hub_height_wt[wt_index,wd_index,ws_index] = hub_height_base[wt_index] + zr_new
                        gamma_wt[wt_index,wd_index,ws_index]  = tilt_base[wt_index] + gamma_new
                        break
    
                    xr_prev = np.copy(xr_new)
                    yr_prev = np.copy(yr_new)
                    zr_prev = np.copy(zr_new)
                    gamma_prev = np.copy(gamma_new)
    
    #%Get AEP contribution per wd and ws
    start_time = time.time()
    AEP_contrib, AEP_contrib_initial = [], []
    for ws_index in range(len(ws)):
        for wd_index in range(len(wd)):
            #remove ws_index from prob when one ws
            AEP_contrib.append( wf_model_Uniform(fowt_x[:,wd_index,ws_index], fowt_y[:,wd_index,ws_index], h=hub_height_wt[:,wd_index,ws_index], tilt=gamma_wt[:,wd_index,ws_index], ws=ws[ws_index], wd=wd[wd_index]).aep().values.sum() * prob[ws_index,wd_index] )
            AEP_contrib_initial.append( wf_model_Uniform(fowt_x_base, fowt_y_base, ws=ws[ws_index], wd=wd[wd_index]).aep().values.sum() * prob[ws_index,wd_index] )
    end_time = time.time()
    elapsed_time_AEP = end_time - start_time
    
    full_time_end = time.time()
    full_time = full_time_end - full_time_start
    
    initial_aep = np.array(AEP_contrib_initial).sum()
    print('Initial AEP from loop')
    print(initial_aep)
    
    updated_aep = np.array(AEP_contrib).sum()
    print('Updated AEP')
    print(updated_aep)
    
    aep_change = (initial_aep-updated_aep)/initial_aep * 100
    print('AEP change [%]')
    print(aep_change)
    
    initial_aep_base_method = wf_model(fowt_x_base,fowt_y_base,ws=ws,wd=wd).aep().sum().values
    print('Initial AEP from PyWake command directly')
    print(initial_aep_base_method)
    
    #Get the AEP contribution per wd and sum of all ws
    AEP_contrib_initial_sum_all_ws = [sum(np.array(AEP_contrib_initial[i::len(wd)][:22])) for i in range(len(wd))]
    AEP_contrib_sum_all_ws = [sum(np.array(AEP_contrib[i::len(wd)][:22])) for i in range(len(wd))]
    
    #% SAVE RESULTS
    from wesl.optimizer.offshore_system.fowt.save_load_results import save_results, load_results
        
    results = {
        'ws': ws,
        'wd': wd,
        'fowt_x': fowt_x,
        'fowt_y': fowt_y,
        'hub_height_wt': hub_height_wt,
        'gamma_wt': gamma_wt,
        'fowt_x_base': fowt_x_base,
        'fowt_y_base': fowt_y_base,
        'AEP_contrib': AEP_contrib,   # AEP all wind directions and all wind speeds len = 8280 (360*23)
        'updated_aep': updated_aep,             # AEP sum per wd
        'AEP_contrib_initial': AEP_contrib_initial,   # AEP all wind directions and all wind speeds len = 8280 (360*23)
        'initial_aep': initial_aep,             # AEP sum per wd
        'initial_aep_base_method': initial_aep_base_method,
        'aep_change': aep_change,
        'full_time': full_time,
        'elapsed_time_AEP': elapsed_time_AEP
        }
            
    save_results(results, f'./results/Parametric_Analysis/MS_config_11/alpha_{MS_config_index}.pkl')
