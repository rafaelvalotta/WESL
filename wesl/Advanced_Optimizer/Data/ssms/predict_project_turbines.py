# Loads the surrogates trained by train_project_turbines.py and shows monopile/tower mass
# vs water depth for both project turbines. WaveHeight/WavePeriod/WindSpeed below are
# still placeholders (see ../../Metocean/wave_climate_reference.ipynb).
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)

# turbine, target RP, IP platform to use, rotor diameter [m], hub height [m]
TURBINES = [
    ('Vineyard (Haliade-X 13MW)', 13, 15.0, 220.0, 150.0),
    ('Revolution (SG11-200)', 11, 10.0, 200.0, 133.0),
]

WAVE_HEIGHT_PLACEHOLDER = 1.38   # NDBC 44097 annual mean, see Metocean/
WAVE_PERIOD_PLACEHOLDER = 7.68   # NDBC 44097 annual mean, see Metocean/
WIND_SPEED_PLACEHOLDER = 9.5     # TODO: derive from Weibull fit in energy_resource_us.yaml


class QLSModel:
    def __init__(self, model, input_scaler, output_scaler):
        self.model, self.input_scaler, self.output_scaler = model.getMetaModel(), input_scaler, output_scaler

    def predict(self, D, HTrans, HHub, WaterDepth, WaveHeight, WavePeriod, WindSpeed):
        inps = np.asarray([D, HTrans, HHub, WaterDepth, WaveHeight, WavePeriod, WindSpeed]).T
        inps_scaled = self.input_scaler.transform(np.atleast_2d(inps))
        return self.output_scaler.inverse_transform(self.model(inps_scaled)).ravel()


if __name__ == '__main__':
    for name, rp, ip, D, hub in TURBINES:
        path = os.path.join(HERE, 'models', f'RP{rp}_{"vineyard" if rp == 13 else "revolution"}',
                             f'{ip}_QLS_surrogate_model.pickle')
        with open(path, 'rb') as f:
            dic = pickle.load(f)

        print(f'\n{name} -- RP={rp}MW, IP platform={ip}, {dic["n_points"]} real points')
        for out_item, out_name in enumerate(dic['output_channel_names'][:2]):
            qlsm = QLSModel(dic['models'][out_item], dic['input_scaler'], dic['output_scalers'][out_name])
            for depth in [15, 25, 35, 45]:
                mass = qlsm.predict(D=D, HTrans=15, HHub=hub, WaterDepth=depth,
                                     WaveHeight=WAVE_HEIGHT_PLACEHOLDER, WavePeriod=WAVE_PERIOD_PLACEHOLDER,
                                     WindSpeed=WIND_SPEED_PLACEHOLDER)
                print(f'  {out_name} @ depth={depth}m: {mass[0]:,.0f} kg')
