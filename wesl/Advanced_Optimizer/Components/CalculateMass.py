# # -*- coding: utf-8 -*-
# """
# Created on Mon Feb 13 11:49:20 2023

# @author: Samuel Kainz
# """
# import openturns as ot
# import os 
# from pathlib import Path
# import pickle
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")



# class QLSModel(object):
#     def __init__(self, model, input_scaler, output_scaler):
#         # Captura o modelo bruto do OpenTURNS e os scalers do Scikit-Learn
#         self.model, self.input_scaler, self.output_scaler = model.getMetaModel(), input_scaler, output_scaler
        
#     def predict(self, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed):
#         # 1. Criamos a matriz com 8 colunas exatamente como o MinMaxScaler do Samuel espera
#         rp = 22.0  # Mude para 10.0 ou 15.0 se mudar de turbina
#         inps = np.array([[rp, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed]])
        
#         # 2. Aplicamos o escalonamento do Scikit-Learn (resultado terá formato 1x8)
#         inps_scaled = self.input_scaler.transform(inps)
        
#         # 3. TRUQUE DA CORREÇÃO: Removemos a primeira coluna (RP) para ficar com as 7 colunas do OpenTURNS
#         seven_features = inps_scaled[0, 1:].tolist()
        
#         # 4. Convertemos em um ponto nativo do OpenTURNS (Dimensão exata = 7)
#         ot_point = ot.Point(seven_features)
        
#         # 5. Executamos o metamodelo do Samuel
#         scaled_output = self.model(ot_point)
        
#         # 6. Desfazemos a escala do output para retornar o peso em kg
#         output = self.output_scaler.inverse_transform(np.array(scaled_output).reshape(1, -1)).ravel()
#         return output

# def CalculateMass(D, HTrans, HHub, WaterDepth, WaveHeight, WavePeriod, WindSpeed, IP_item):
#     """
#     Computes component weights using local QLS surrogate response surface configurations.
#     Path tracking is managed via explicit pathlib abstractions relative to this module.
#     """
#     # Locate the model directory dynamically relative to this file's physical home
#     current_file_dir = Path(__file__).parent.parent
#     model_path = current_file_dir / "Data" / "ssms" / "models" / "QLS"
#     model_indicator = "_QLS_surrogate_model.pickle"
    
#     # Secure defensive check: create directories if they do not exist
#     model_path.mkdir(parents=True, exist_ok=True)
    
#     files = []
#     IPs = []
    
#     # Iterate through directory items using pathlib's iterdir engine
#     for file_path in model_path.iterdir():
#         if file_path.is_file() and model_indicator in file_path.name:
#             # Extract internal IP configuration value from filename string
#             IP = float(file_path.name.split(model_indicator)[0])
#             files.append(file_path.name)
#             IPs.append(IP)
            
#     if not files:
#         raise FileNotFoundError(f"No valid QLS surrogate pickle models found matching '{model_indicator}' inside: {model_path}")
        
#     # Sort files by IP index to ensure deterministic mapping index matches across runs
#     sorted_indices = np.argsort(IPs)
#     files = [files[idx] for idx in sorted_indices]
    
#     # Locate target model binary path
#     target_pickle_path = model_path / files[IP_item]
    
#     with open(target_pickle_path, 'rb') as f:
#         dic = pickle.load(f)
        
#     output_channel_names = dic['output_channel_names']
#     res = []
    
#     # Evaluate structural channels (0: Monopile Mass, 1: Tower Mass)
#     for i in range(2):
#         out_item = i
#         output_channel = output_channel_names[out_item]
        
#         qlsm = QLSModel(
#             dic['models'][out_item], 
#             dic['input_scaler'], 
#             dic['output_scalers'][output_channel]
#         )
        
#         mass = qlsm.predict(D, HTrans, HHub, WaterDepth, WaveHeight, WavePeriod, WindSpeed)
#         res.append(np.ndarray.tolist(mass))
        
#     return res

# #%% Sample run
# # Inputs
# turstring = ['IEA-10MW','IEA-15MW','IEA-22MW']
# RP = [10, 15, 22]             # MW
# D = [198, 240, 284]                 # m
# HH = [119, 140, 170]                # m
# PlatformHeight = [10, 15, 15]     # m
# WaterDepth = [34]*3      # m
# SignificantWaveHeight = [2.52]*3    # m
# SignificantWavePeriod = [5.45]*3    # s
# V_ave = [9.924]*3           # m/s
# # Call surrogate
# mass = CalculateMass(RP=RP, D=D, HTrans=PlatformHeight, HHub_Ratio=[hh/d for hh, d in zip(HH,D)], WaterDepth=WaterDepth, WaveHeight=SignificantWaveHeight, WavePeriod=SignificantWavePeriod, WindSpeed=V_ave)
# print(f'Monopile mass: {mass[0][2]:.1f} kg')
# print(f'Tower mass: {mass[1][2]:.1f} kg')

# #%% Create water depth dependent vector
# masses = []
# depths = np.linspace(20,40,num=21)
# for z in depths:
#    #masses.append((water_depth))
#    cur_mass = CalculateMass(RP=RP, D=D, HTrans=PlatformHeight, HHub_Ratio=[hh/d for hh, d in zip(HH,D)], WaterDepth=[z]*len(D), WaveHeight=SignificantWaveHeight, WavePeriod=SignificantWavePeriod, WindSpeed=V_ave)
#    masses.append(cur_mass[0])
# masses_norm = np.array(masses) / np.array(masses)[np.array(depths)==34]

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.gca()
# for i in range(masses_norm.shape[1]):
#     ax.plot(depths, masses_norm[:,i], label=turstring[i])  # Set the label for each curve
# ax.legend()
# ax.grid(alpha=0.6)
# ax.set_ylabel('Rel. monopile mass wrt z=34m')
# ax.set_xlabel('Water depth z [m]')