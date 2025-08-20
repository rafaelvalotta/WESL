import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- paths ---
MODEL_PATH    = 'OSWEC_WESL_surrogate_model_v0_0_2.keras'
SCALER_X_PATH = 'scaler_X.save'
SCALER_Y_PATH = 'scaler_y.save'
VAL_CSV       = 'OSWEC_WecSim_Dataset_Validation.csv'   # change if needed

# --- load assets ---
model   = load_model(MODEL_PATH)
scalerX = joblib.load(SCALER_X_PATH)
scalerY = joblib.load(SCALER_Y_PATH)

# --- load data (exact headers) ---
df = pd.read_csv(VAL_CSV)
X  = df[['Height_m','Period_s','Direction_deg']].to_numpy(dtype=np.float32)
y_true = df['PeakPower_kW'].to_numpy(dtype=np.float32)

# --- predict ---
X_scaled = scalerX.transform(X)
y_pred_scaled = model.predict(X_scaled, verbose=0)
y_pred = scalerY.inverse_transform(y_pred_scaled).ravel()

# --- metrics ---
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mae  = float(mean_absolute_error(y_true, y_pred))
r2   = float(r2_score(y_true, y_pred))
print(f"RMSE: {rmse:.4f} kW  |  MAE: {mae:.4f} kW  |  R²: {r2:.4f}")


# --- plot ---
plt.figure(figsize=(7,6))
plt.scatter(y_true, y_pred, alpha=0.75, color='r', edgecolor='none', label='data')

mn = min(y_true.min(), y_pred.min())
mx = max(y_true.max(), y_pred.max())
plt.plot([mn,mx], [mn,mx], 'k--', lw=1.5)

plt.xlabel('Actual Peak Power (kW)')
plt.ylabel('Predicted Peak Power (kW)')
plt.title('Validation: Predicted vs Actual')
plt.grid(True)
plt.legend()

txt = f"$R^2$={r2:.3f}\nRMSE={rmse:.2f} kW\nMAE={mae:.2f} kW"
plt.gca().text(0.04, 0.96, txt, transform=plt.gca().transAxes,
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
plt.tight_layout()
plt.show()
