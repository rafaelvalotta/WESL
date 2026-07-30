# Trains real QLS monopile/tower mass surrogates (IEA Wind 2200-22-ROWP methodology) for
# RP=13 (Vineyard) and RP=11 (Revolution), instead of the RP=20 pickles shipped with the
# reference repo. Both RPs exist as real structural design points -- no extrapolation.
import os
import numpy as np
import pandas as pd
import openturns as ot
from sklearn.preprocessing import MinMaxScaler
import pickle

HERE = os.path.dirname(__file__)
IN_COLS = ['D', 'HTrans', 'HHub', 'WaterDepth', 'WaveHeight', 'WavePeriod', 'WindSpeed']
OUT_COLS = ['monopile_mass', 'tower_mass', 'total_mass']

TARGETS = {13: 'RP13_vineyard', 11: 'RP11_revolution'}


def load_real_data():
    data = pd.read_csv(os.path.join(HERE, 'data/tower_mass_results.dat'), sep=' ')
    data_ext = pd.read_csv(os.path.join(HERE, 'data/tower_mass_results_extended_depth_results.dat'), sep=' ')
    df = data[data.columns[:-1]]
    df_ext = data_ext[data_ext.columns[:-1]]
    df.columns = data.columns[1:]
    df_ext.columns = data_ext.columns[1:]
    return pd.concat([df, df_ext])


def train_one(sub):
    x = sub[IN_COLS].to_numpy()
    y = sub[OUT_COLS].to_numpy()
    input_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    x_scaled = input_scaler.fit_transform(x)
    output_scalers, models = {}, []
    for i, name in enumerate(OUT_COLS):
        output_scalers[name] = MinMaxScaler(feature_range=(-0.7, 0.7))
        y_scaled = output_scalers[name].fit_transform(y[:, [i]])
        model = ot.QuadraticLeastSquares(x_scaled, y_scaled)
        model.run()
        models.append(model)
    return input_scaler, output_scalers, models


if __name__ == '__main__':
    df_all = load_real_data()
    for rp, folder in TARGETS.items():
        out_dir = os.path.join(HERE, 'models', folder)
        os.makedirs(out_dir, exist_ok=True)
        sub_rp = df_all[df_all['RP'] == float(rp)]
        print(f'RP={rp}: {sub_rp.shape[0]} real points, IP platforms={sorted(sub_rp.IP.unique())}')
        for ip in sub_rp.IP.unique():
            sub = sub_rp[sub_rp.IP == ip].reset_index(drop=True)
            input_scaler, output_scalers, models = train_one(sub)
            path = os.path.join(out_dir, f'{ip}_QLS_surrogate_model.pickle')
            with open(path, 'wb') as f:
                pickle.dump(dict(input_scaler=input_scaler, output_scalers=output_scalers,
                                  models=models, output_channel_names=OUT_COLS,
                                  input_channel_names=IN_COLS, n_points=sub.shape[0]), f)
            print(f'  saved {path} ({sub.shape[0]} points)')
