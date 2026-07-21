#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:03:09 2023

@author: mariomartingarcia
"""

# BASIC PYTHON LIB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# LOAD results
from wesl.optimizer.offshore_system.fowt.save_load_results import load_results

#%% CASE 1
wds = [360, 180, 90, 60, 45, 30, 15]
results_aep_Case1 = []
results_time_Case1 = []

for wds_index in wds:
    file_name = f"./results/Time_Sensitivity_Analysis/Case_1/{wds_index}wds.plk"
    results_loaded = load_results(file_name)
    updated_aep, full_time = results_loaded['updated_aep'], results_loaded['full_time']/60
    results_aep_Case1.append(updated_aep)
    results_time_Case1.append(full_time)


fig, ax1 = plt.subplots(figsize=(8.5,3.5))
plt.rcParams.update({'font.size': 14})
ax1.grid(True)
plt.gca().invert_xaxis()
plt.xticks(wds)

# AEP
color_11 = 'tab:red'
ax1.set_xlabel('Number of evaluated wind directions')
ax1.set_ylabel('FOWF AEP [GWh]', color=color_11)
ax1.plot(wds, results_aep_Case1, color=color_11)
ax1.tick_params(axis='y', labelcolor=color_11)


# TIME
ax2 = ax1.twinx()  # Second Y axis
color_all = 'tab:blue'
ax2.set_ylabel('Computational time [min]', color=color_all)
ax2.plot(wds, results_time_Case1, color=color_all)
ax2.tick_params(axis='y', labelcolor=color_all)

# ax1.legend(['FOWF AEP'], loc='upper left')
# #ax2.legend([r'$ws \in [4, 25]\ m/s$'], loc='upper left', bbox_to_anchor=(0, 0.92))
# ax2.vlines(x=angles[max_aep_change_index_all], ymin=round(results_aep_change_all[0],4) - 0.001, ymax=max_aep_change_all, color='black', linestyle='--')

# ax1.plot(angles[0],-aep_change_base_11,'.',color=color_11)
# ax1.text(angles[0]+14, -aep_change_base_11, str(round(-aep_change_base_11,4))+'%', ha='center', va='bottom',color=color_11)
# ax1.plot(angles[max_aep_change_index_11],max_aep_change_11,'.',color=color_11)
# ax1.text(angles[max_aep_change_index_11]-10, max_aep_change_11, str(round(max_aep_change_11,4))+'%', ha='center', va='bottom', color=color_11)
# #ax1.set_yticks(np.arange(round(results_aep_change_11[0],4) - 0.0009, round(max_aep_change_11,4) + 0.001, 0.001))

# ax2.plot(angles[0],-aep_change_base_all,'.',color=color_all)
# ax2.text(angles[0]+10, -aep_change_base_all+0.00002, str(round(-aep_change_base_all,4))+'%', ha='center', va='bottom', color=color_all)
# ax2.plot(angles[max_aep_change_index_all],max_aep_change_all,'.',color=color_all)
# ax2.text(angles[max_aep_change_index_all]+10, max_aep_change_all, str(round(max_aep_change_all,4))+'%', ha='center', va='bottom', color=color_all)
# #ax2.set_yticks(np.arange(round(results_aep_change_all[0],3) - 0.0009, round(max_aep_change_all,3) + 0.001, 0.001))

ax1.grid(True)
ax2.grid(True)
# plt.axis('equal')
plt.tight_layout()
plt.show()


#%% CASE 2
threshold = [1, 5, 10, 15, 20]
threshold_legend = ['$\epsilon_{BC}$', '$\epsilon_{BC}\cdot5$', '$\epsilon_{BC}\cdot10$', '$\epsilon_{BC}\cdot15$', '$\epsilon_{BC}\cdot20$']
results_aep_Case2 = []
results_time_Case2 = []

for threshold_index in threshold:
    file_name = f"./results/Time_Sensitivity_Analysis/Case_2/threshold_x_{threshold_index}.plk"
    results_loaded = load_results(file_name)
    updated_aep, full_time = results_loaded['updated_aep'], results_loaded['full_time']/60
    results_aep_Case2.append(updated_aep)
    results_time_Case2.append(full_time)


fig, ax1 = plt.subplots(figsize=(8.5,3.5))
plt.rcParams.update({'font.size': 14})
ax1.grid(True)
#plt.gca().invert_xaxis()
plt.ticklabel_format(style='plain', useOffset=False)
plt.xticks(threshold,threshold_legend)

# AEP
color_11 = 'tab:red'
ax1.set_xlabel('Threshold')
ax1.set_ylabel('FOWF AEP [GWh]', color=color_11)
ax1.plot(threshold, results_aep_Case2, color=color_11)
ax1.tick_params(axis='y', labelcolor=color_11)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

# TIME
ax2 = ax1.twinx()  # Second Y axis
color_all = 'tab:blue'
ax2.set_ylabel('Computational time [min]', color=color_all)
ax2.plot(threshold, results_time_Case2, color=color_all)
ax2.tick_params(axis='y', labelcolor=color_all)

# ax1.legend(['FOWF AEP'], loc='upper left')
# #ax2.legend([r'$ws \in [4, 25]\ m/s$'], loc='upper left', bbox_to_anchor=(0, 0.92))
# ax2.vlines(x=angles[max_aep_change_index_all], ymin=round(results_aep_change_all[0],4) - 0.001, ymax=max_aep_change_all, color='black', linestyle='--')

# ax1.plot(angles[0],-aep_change_base_11,'.',color=color_11)
# ax1.text(angles[0]+14, -aep_change_base_11, str(round(-aep_change_base_11,4))+'%', ha='center', va='bottom',color=color_11)
# ax1.plot(angles[max_aep_change_index_11],max_aep_change_11,'.',color=color_11)
# ax1.text(angles[max_aep_change_index_11]-10, max_aep_change_11, str(round(max_aep_change_11,4))+'%', ha='center', va='bottom', color=color_11)
# #ax1.set_yticks(np.arange(round(results_aep_change_11[0],4) - 0.0009, round(max_aep_change_11,4) + 0.001, 0.001))

# ax2.plot(angles[0],-aep_change_base_all,'.',color=color_all)
# ax2.text(angles[0]+10, -aep_change_base_all+0.00002, str(round(-aep_change_base_all,4))+'%', ha='center', va='bottom', color=color_all)
# ax2.plot(angles[max_aep_change_index_all],max_aep_change_all,'.',color=color_all)
# ax2.text(angles[max_aep_change_index_all]+10, max_aep_change_all, str(round(max_aep_change_all,4))+'%', ha='center', va='bottom', color=color_all)
# #ax2.set_yticks(np.arange(round(results_aep_change_all[0],3) - 0.0009, round(max_aep_change_all,3) + 0.001, 0.001))

ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.show()

#%% CASE 3
wsps = [22, 15, 12, 8]
results_aep_Case3 = []
results_time_Case3 = []

wsps_increments = [1, 1.5, '2_changin_in_11', 3]

for wsps_index in wsps_increments:
    file_name = f"./results/Time_Sensitivity_Analysis/Case_3/ws_increments_{wsps_index}.plk"
    results_loaded = load_results(file_name)
    updated_aep, full_time = results_loaded['updated_aep'], results_loaded['full_time']/60
    results_aep_Case3.append(updated_aep)
    results_time_Case3.append(full_time)


fig, ax1 = plt.subplots(figsize=(8.5,3.5))
plt.rcParams.update({'font.size': 14})
ax1.grid(True)
plt.gca().invert_xaxis()
plt.xticks(wsps)

# AEP
color_11 = 'tab:red'
ax1.set_xlabel('Number of evaluated wind speeds')
ax1.set_ylabel('FOWF AEP [GWh]', color=color_11)
ax1.plot(wsps, results_aep_Case3, color=color_11)
ax1.tick_params(axis='y', labelcolor=color_11)


# TIME
ax2 = ax1.twinx()  # Second Y axis
color_all = 'tab:blue'
ax2.set_ylabel('Computational time [min]', color=color_all)
ax2.plot(wsps, results_time_Case3, color=color_all)
ax2.tick_params(axis='y', labelcolor=color_all)

# ax1.legend(['FOWF AEP'], loc='upper left')
# #ax2.legend([r'$ws \in [4, 25]\ m/s$'], loc='upper left', bbox_to_anchor=(0, 0.92))
# ax2.vlines(x=angles[max_aep_change_index_all], ymin=round(results_aep_change_all[0],4) - 0.001, ymax=max_aep_change_all, color='black', linestyle='--')

# ax1.plot(angles[0],-aep_change_base_11,'.',color=color_11)
# ax1.text(angles[0]+14, -aep_change_base_11, str(round(-aep_change_base_11,4))+'%', ha='center', va='bottom',color=color_11)
# ax1.plot(angles[max_aep_change_index_11],max_aep_change_11,'.',color=color_11)
# ax1.text(angles[max_aep_change_index_11]-10, max_aep_change_11, str(round(max_aep_change_11,4))+'%', ha='center', va='bottom', color=color_11)
# #ax1.set_yticks(np.arange(round(results_aep_change_11[0],4) - 0.0009, round(max_aep_change_11,4) + 0.001, 0.001))

# ax2.plot(angles[0],-aep_change_base_all,'.',color=color_all)
# ax2.text(angles[0]+10, -aep_change_base_all+0.00002, str(round(-aep_change_base_all,4))+'%', ha='center', va='bottom', color=color_all)
# ax2.plot(angles[max_aep_change_index_all],max_aep_change_all,'.',color=color_all)
# ax2.text(angles[max_aep_change_index_all]+10, max_aep_change_all, str(round(max_aep_change_all,4))+'%', ha='center', va='bottom', color=color_all)
# #ax2.set_yticks(np.arange(round(results_aep_change_all[0],3) - 0.0009, round(max_aep_change_all,3) + 0.001, 0.001))

ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.show()
