#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:31:53 2023

Sensitivity analysis post-processing

First plot:
- ws = 11 m/s
- all wd
- 10° mooring line angle increments

Second plot:
- ws = [4,25] m/s
- all wd
- 10° mooring line angle increments

Third plot:
- Both cases together

@author: mariomartingarcia
"""

# BASIC PYTHON LIB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# LOAD results
import pickle

#%% AEP change for ws = 11 m/s

with open('mario-martin-thesis/myWF/results/results_ws_11_wd_0-360.pkl', 'rb') as file:
    # Load and deserialize the data
    results_loaded_11_base = pickle.load(file)

aep_change_base_11 = results_loaded_11_base['aep_change']

angles = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 120]
results_aep_change_11 = []
results_aep_change_11.append(-aep_change_base_11)
for angle in angles:
    file_name = f"mario-martin-thesis/myWF/results_old(slow)/Sensitivity_analysis_ws_11/results_alpha_{angle}.pkl"
    with open(file_name, 'rb') as file:
    # Load and deserialize the data
        results_loaded = pickle.load(file)
    #results_loaded = load_results(file_name)
        aep_change = results_loaded['aep_change']
        results_aep_change_11.append(-aep_change)

angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 120]

max_aep_change_11, max_aep_change_index_11 = np.max(results_aep_change_11), np.argmax(results_aep_change_11)

plt.figure(figsize=(10,8))
plt.plot(angles,results_aep_change_11)
plt.plot(angles[0],-aep_change_base_11,'r.')
plt.text(angles[0]+6.2, -aep_change_base_11, str(round(-aep_change_base_11,4))+'%', ha='center', va='bottom')
plt.plot(angles[max_aep_change_index_11],max_aep_change_11,'r.')
plt.text(angles[max_aep_change_index_11]+6.2, max_aep_change_11, str(round(max_aep_change_11,4))+'%', ha='center', va='bottom')
plt.vlines(x=angles[max_aep_change_index_11], ymin=np.min(results_aep_change_11), ymax=max_aep_change_11, color='red', linestyle='--')
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.xticks(angles)
plt.xlabel('Mooring lines configuration angle [°]')
plt.ylabel('AEP change [%]')

#%% AEP change for all ws

results_loaded_all_base = load_results('./results_ws_4-25_wd_0-360.pkl')

aep_change_base_all = results_loaded_all_base['aep_change']

angles = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 120]
results_aep_change_all = []
results_aep_change_all.append(-aep_change_base_all)
for angle in angles:
    file_name = f"./Sensitivity_analysis_all_ws/results_alpha_{angle}.pkl"
    results_loaded = load_results(file_name)
    aep_change = results_loaded['aep_change']
    results_aep_change_all.append(-aep_change)

angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 120]

max_aep_change_all, max_aep_change_index_all = np.max(results_aep_change_all), np.argmax(results_aep_change_all)

plt.figure(figsize=(10,8))
plt.plot(angles,results_aep_change_all)
plt.plot(angles[0],-aep_change_base_all,'r.')
plt.text(angles[0]+6, -aep_change_base_all+0.0001, str(round(-aep_change_base_all,4))+'%', ha='center', va='top')

plt.plot(angles[max_aep_change_index_all],max_aep_change_all,'r.')
plt.text(angles[max_aep_change_index_all]+6, max_aep_change_all+0.00009, str(round(max_aep_change_all,4))+'%', ha='center', va='top')

plt.vlines(x=angles[max_aep_change_index_all], ymin=round(results_aep_change_all[0],4) - 0.001, ymax=max_aep_change_all, color='red', linestyle='--')
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.xticks(angles)
plt.yticks(np.arange(round(results_aep_change_all[0],3) - 0.0009, round(max_aep_change_all,3) + 0.001, 0.001))
plt.xlabel('Mooring lines configuration angle [°]')
plt.ylabel('AEP change [%]')


#%% BOTH CASES TOGETHER

fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 14})
ax1.grid(True)
plt.xticks(angles)

# ws = 11 m/s
color_11 = 'tab:red'
ax1.set_xlabel('Mooring lines configuration angle [°]')
ax1.set_ylabel('AEP change [%]', color=color_11)
ax1.plot(angles, results_aep_change_11, color=color_11)
ax1.tick_params(axis='y', labelcolor=color_11)


# ws = [4,25] m/s
ax2 = ax1.twinx()  # Second Y axis
color_all = 'tab:blue'
ax2.set_ylabel('AEP change [%]', color=color_all)
ax2.plot(angles, results_aep_change_all, color=color_all)
ax2.tick_params(axis='y', labelcolor=color_all)

ax1.legend(['ws=11 m/s'], loc='upper left')
ax2.legend([r'$ws \in [4, 25]\ m/s$'], loc='upper left', bbox_to_anchor=(0, 0.92))
ax2.vlines(x=angles[max_aep_change_index_all], ymin=round(results_aep_change_all[0],4) - 0.001, ymax=max_aep_change_all, color='black', linestyle='--')

ax1.plot(angles[0],-aep_change_base_11,'.',color=color_11)
ax1.text(angles[0]+14, -aep_change_base_11, str(round(-aep_change_base_11,4))+'%', ha='center', va='bottom',color=color_11)
ax1.plot(angles[max_aep_change_index_11],max_aep_change_11,'.',color=color_11)
ax1.text(angles[max_aep_change_index_11]-10, max_aep_change_11, str(round(max_aep_change_11,4))+'%', ha='center', va='bottom', color=color_11)
#ax1.set_yticks(np.arange(round(results_aep_change_11[0],4) - 0.0009, round(max_aep_change_11,4) + 0.001, 0.001))

ax2.plot(angles[0],-aep_change_base_all,'.',color=color_all)
ax2.text(angles[0]+10, -aep_change_base_all+0.00002, str(round(-aep_change_base_all,4))+'%', ha='center', va='bottom', color=color_all)
ax2.plot(angles[max_aep_change_index_all],max_aep_change_all,'.',color=color_all)
ax2.text(angles[max_aep_change_index_all]+10, max_aep_change_all, str(round(max_aep_change_all,4))+'%', ha='center', va='bottom', color=color_all)
#ax2.set_yticks(np.arange(round(results_aep_change_all[0],3) - 0.0009, round(max_aep_change_all,3) + 0.001, 0.001))

ax1.grid(True)
ax2.grid(True)
plt.show()


#%% MS_config all ws

MS_config = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 115, 120])

results_aep_all_ws, results_aep_change_all_ws = [], []

for MS_config_index in MS_config:
    file_name = f"./results/Parametric_Analysis/MS_config_all_ws/alpha_{MS_config_index}.pkl"
    results_loaded = load_results(file_name)
    updated_aep, aep_change = results_loaded['updated_aep'], results_loaded['aep_change']
    results_aep_all_ws.append(updated_aep)
    results_aep_change_all_ws.append(-aep_change)

max_aep_change_all_ws, max_aep_change_index_all_ws = np.max(results_aep_change_all_ws), np.argmax(results_aep_change_all_ws)

plt.figure(figsize=(10,8))
plt.plot(MS_config,results_aep_change_all_ws)
plt.plot(MS_config[0],results_aep_change_all_ws[0],'r.')
plt.text(MS_config[0]+6, results_aep_change_all_ws[0]+0.0001, str(round(results_aep_change_all_ws[0],4))+'%', ha='center', va='top')

plt.plot(MS_config[max_aep_change_index_all],max_aep_change_all,'r.')
plt.text(MS_config[max_aep_change_index_all]+6, max_aep_change_all+0.00009, str(round(max_aep_change_all,4))+'%', ha='center', va='top')

plt.vlines(x=MS_config[max_aep_change_index_all_ws], ymin=round(results_aep_change_all_ws[0],4) - 0.001, ymax=max_aep_change_all_ws, color='red', linestyle='--')
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.xticks(MS_config)
plt.yticks(np.arange(round(results_aep_change_all[0],3) - 0.0009, round(max_aep_change_all,3) + 0.001, 0.001))
plt.xlabel('Mooring lines configuration angle [°]')
plt.ylabel('AEP change [%]')

#%% MS_config 11 m/s

MS_config = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 115, 120])

results_aep_11, results_aep_change_11 = [], []

for MS_config_index in MS_config:
    file_name = f"./results/Parametric_Analysis/MS_config_11/alpha_{MS_config_index}.pkl"
    results_loaded = load_results(file_name)
    updated_aep, aep_change = results_loaded['updated_aep'], results_loaded['aep_change']
    results_aep_11.append(updated_aep)
    results_aep_change_11.append(-aep_change)

max_aep_change_11, max_aep_change_index_11 = np.max(results_aep_change_11), np.argmax(results_aep_change_11)

plt.figure(figsize=(10,8))
plt.plot(MS_config,results_aep_change_11)
plt.plot(MS_config[0],results_aep_change_11[0],'r.')
plt.text(MS_config[0]+6, results_aep_change_11[0]+0.0001, str(round(results_aep_change_11[0],4))+'%', ha='center', va='top')

plt.plot(MS_config[max_aep_change_index_11],max_aep_change_11,'r.')
plt.text(MS_config[max_aep_change_index_11]+6, max_aep_change_11+0.00009, str(round(max_aep_change_11,4))+'%', ha='center', va='top')

plt.vlines(x=MS_config[max_aep_change_index_11], ymin=round(results_aep_change_11[0],4) - 0.001, ymax=max_aep_change_11, color='red', linestyle='--')
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.xticks(MS_config)
plt.yticks(np.arange(round(results_aep_change_11[0],3) - 0.0009, round(max_aep_change_11,3) + 0.001, 0.001))
plt.xlabel('Mooring lines configuration angle [°]')
plt.ylabel('AEP change [%]')
#%% Mooring lines length
import numpy as np
import matplotlib.pyplot as plt
from wesl.optimizer.offshore_system.fowt.save_load_results import load_results

ml_length = np.arange(902.2,1203.2,5)
results_aep, results_aep_change = [], []

for ml_length_index in ml_length:
    file_name = f"./results/Parametric_Analysis/ML_length/{ml_length_index}.pkl"
    results_loaded = load_results(file_name)
    updated_aep, aep_change = results_loaded['updated_aep'], results_loaded['aep_change']
    results_aep.append(updated_aep)
    results_aep_change.append(-aep_change)


fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 13})

ax1.plot(ml_length,results_aep)
plt.grid(True)
ax1.set_ylabel('FOWF AEP [GWh]')
ax1.set_xlabel('Mooring line legnth [m]')

plt.xticks(ml_length[::2], rotation=45, ha='right', fontsize=12)

ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis

ax2.plot(ml_length,results_aep_change)
ax2.set_ylabel('AEP change [%]')

plt.show()

#%% Sea depth
import numpy as np
import matplotlib.pyplot as plt
from wesl.optimizer.offshore_system.fowt.save_load_results import load_results

sea_depth = np.arange(100, 510, 10)
results_aep, results_aep_change = [], []

for sea_depth_index in sea_depth:
    file_name = f"./results/Parametric_Analysis/Sea_depth/{sea_depth_index}.pkl"
    results_loaded = load_results(file_name)
    updated_aep, aep_change = results_loaded['updated_aep'], results_loaded['aep_change']
    results_aep.append(updated_aep)
    results_aep_change.append(-aep_change)

fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 13})

ax1.plot(sea_depth, results_aep)
plt.grid(True)
ax1.set_ylabel('FOWF AEP [GWh]')
ax1.set_xlabel('Sea depth [m]')

# Obtener el índice correspondiente a la posición 320 en sea_depth
idx = np.where(sea_depth == 320)[0][0]
# Dibujar la línea vertical en la posición 320
ax1.vlines(x=sea_depth[idx], ymin=round(np.min(results_aep),1), ymax=results_aep[idx], color='red', linestyle='--')

plt.xticks(sea_depth[::2], rotation=45, ha='right', fontsize=12)

ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis

ax2.plot(sea_depth, results_aep_change)
ax2.set_ylabel('AEP change [%]')

plt.show()
