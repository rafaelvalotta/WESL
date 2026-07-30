import os
import pickle
from math import gamma

import numpy as np
import openmdao.api as om
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate._interpnd import LinearNDInterpolator

SSMS_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data', 'ssms', 'models')

# NDBC 44097 (Block Island) annual means, 2024 -- see ../Metocean/wave_climate_reference.ipynb.
# Documented gap: no site-specific wave climate model yet, using this real proxy instead.
WAVE_HEIGHT_DEFAULT = 1.38
WAVE_PERIOD_DEFAULT = 7.68


def mean_wind_speed(site_resource_dict):
    A = site_resource_dict['weibull_A']
    k = site_resource_dict['weibull_k']
    p = site_resource_dict['probabilities']
    return float(np.sum(p * A * np.array([gamma(1 + 1 / ki) for ki in k])))


class QLSModel:
    def __init__(self, model, input_scaler, output_scaler):
        self.model, self.input_scaler, self.output_scaler = model.getMetaModel(), input_scaler, output_scaler

    def predict(self, D, HTrans, HHub, WaterDepth, WaveHeight, WavePeriod, WindSpeed):
        n = len(np.atleast_1d(WaterDepth))
        inps = np.column_stack([np.full(n, D), np.full(n, HTrans), np.full(n, HHub), WaterDepth,
                                 np.full(n, WaveHeight), np.full(n, WavePeriod), np.full(n, WindSpeed)])
        inps_scaled = self.input_scaler.transform(inps)
        return self.output_scaler.inverse_transform(self.model(inps_scaled)).ravel()


class FoundationsComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt', types=int)
        self.options.declare('bathy_nc_path', types=str)
        self.options.declare('rated_power', types=int)  # 13 (Vineyard) or 11 (Revolution)
        self.options.declare('ip_platform', types=float)  # closest IEA reference platform: 15.0 or 10.0
        self.options.declare('rotor_diameter', types=float)
        self.options.declare('hub_height', types=float)
        self.options.declare('wind_speed', types=float)  # sector-probability-weighted Weibull mean
        self.options.declare('wave_height', default=WAVE_HEIGHT_DEFAULT)
        self.options.declare('wave_period', default=WAVE_PERIOD_DEFAULT)
        self.options.declare('transition_height', default=15.0)
        # self.options.declare('steel_price_factor', default=3.636 * 0.924) # EUR/kg (old, USD->EUR)
        self.options.declare('steel_price_factor', default=3.636) # USD/kg, no EUR conversion

    def setup(self):
        n_wt = self.options['n_wt']
        self.add_input('x', val=np.zeros(n_wt), units='m')
        self.add_input('y', val=np.zeros(n_wt), units='m')
        self.add_output('cost_foundations', val=np.zeros(n_wt))

        with xr.open_dataset(self.options['bathy_nc_path']) as ds:
            x_flat = ds['x'].values
            y_flat = ds['y'].values
            depth_flat = ds['depth'].values

        self.grid_x = np.unique(x_flat)
        self.grid_y = np.unique(y_flat)

        expected_size = len(self.grid_x) * len(self.grid_y)
        if expected_size == len(depth_flat):
            self.matrix_depth = depth_flat.reshape(len(self.grid_x), len(self.grid_y))
            self.depth_interpolator = RegularGridInterpolator(
                (self.grid_x, self.grid_y), self.matrix_depth,
                method='linear', bounds_error=False, fill_value=0.0
            )
            self.use_nd_fallback = False
        else:
            self.depth_interpolator = LinearNDInterpolator(np.column_stack((x_flat, y_flat)), depth_flat)
            self.use_nd_fallback = True

        rp = self.options['rated_power']
        rp_folder = f"RP{rp}_{'vineyard' if rp == 13 else 'revolution'}"
        pickle_path = os.path.join(SSMS_MODELS_DIR, rp_folder, f"{self.options['ip_platform']}_QLS_surrogate_model.pickle")
        with open(pickle_path, 'rb') as f:
            dic = pickle.load(f)
        self.qls = {name: QLSModel(dic['models'][i], dic['input_scaler'], dic['output_scalers'][name])
                    for i, name in enumerate(dic['output_channel_names'][:2])}

        self.declare_partials('cost_foundations', ['x', 'y'], method='fd')

    def compute(self, inputs, outputs):
        x_real = inputs['x'] # pure real coordinates from OpenMDAO
        y_real = inputs['y']

        if self.use_nd_fallback:
            depths = self.depth_interpolator(x_real, y_real)
        else:
            depths = self.depth_interpolator(np.column_stack((x_real, y_real)))
        depths = np.maximum(0.0, depths)

        D = self.options['rotor_diameter']
        HTrans = self.options['transition_height']
        HHub = self.options['hub_height']
        WaveHeight = self.options['wave_height']
        WavePeriod = self.options['wave_period']
        WindSpeed = self.options['wind_speed']

        monopile_mass = self.qls['monopile_mass'].predict(D, HTrans, HHub, depths, WaveHeight, WavePeriod, WindSpeed)
        tower_mass = self.qls['tower_mass'].predict(D, HTrans, HHub, depths, WaveHeight, WavePeriod, WindSpeed)
        outputs['cost_foundations'] = (monopile_mass + tower_mass) * self.options['steel_price_factor']