# wec_device.py
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path


# Resolve paths relative to this file
_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "WESL_WEC_Surrogate_models" / "OSWEC_model"

# If filenames change later, you can glob; for now, keep explicit:
KERAS_PATH   = _MODEL_DIR / "OSWEC_WESL_surrogate_model_v0_0_2.keras"
SCALER_X_PATH = _MODEL_DIR / "Scalers" / "scaler_X.save"
SCALER_Y_PATH = _MODEL_DIR / "Scalers" / "scaler_y.save"



class OSWECDevice:
    """
    Minimal OSWEC device using a saved Keras surrogate + scalers.
    - power_point(H,T,D) -> kW at a single sea state
    - power_grid(Hc,Tc,Dc) -> kW on a grid (optional helper)
    """
    dir = "optimizer/WaveEnergy/"
    def __init__(self,
                 alpha=58,
                 model_path=KERAS_PATH,
                 scaler_X_path=SCALER_X_PATH,
                 scaler_y_path=SCALER_Y_PATH,
                 feature_order=("Height_m", "Period_s", "Direction_deg"),
                 clip_ranges={"Height_m":(0.5,4.0), "Period_s":(5.0,10.0), "Direction_deg":(0.0,90.0)},
                 p_scale=1.0):
        self.alpha = float(alpha)
        self.model = load_model(model_path)
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.feature_order = feature_order
        self.clip_ranges = clip_ranges
        self.p_scale = float(p_scale)  # optional global scale knob

    def _prep_X(self, H, T, D):
        feats = {"Height_m": np.asarray(H).ravel(),
                 "Period_s": np.asarray(T).ravel(),
                 "Direction_deg": np.asarray(D).ravel()}
        for k,(lo,hi) in self.clip_ranges.items():
            feats[k] = np.clip(feats[k], lo, hi)
        X = np.column_stack([feats[k] for k in self.feature_order])
        return X

    def power_point(self, H, T, D):
        """Predict kW at one sea state (scalars)."""
        Xs = self.scaler_X.transform(self._prep_X(H,T,D))
        ys = self.model.predict(Xs, verbose=0)
        y  = self.scaler_y.inverse_transform(ys).ravel()[0]
        return max(self.p_scale * y, 0.0)

    def power_grid(self, Hc, Tc, Dc):
        """Predict kW on (H,T,D) centers (nH,nT,nD)."""
        Hm, Tm, Dm = np.meshgrid(Hc, Tc, Dc, indexing="ij")
        Xs = self.scaler_X.transform(self._prep_X(Hm,Tm,Dm))
        ys = self.model.predict(Xs, verbose=0)
        y  = self.scaler_y.inverse_transform(ys).ravel()
        P  = y.reshape(len(Hc), len(Tc), len(Dc))
        return np.maximum(self.p_scale * P, 0.0)
