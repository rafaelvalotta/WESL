"""
WEC-Sim Surrogate model — Regular waves
Inputs: Height, Period, Direction
Output: PeakPower_kW

version: 0.0.2
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys

MODEL_PATH    = 'OSWEC_WESL_surrogate_model_v0_0_2.keras'
SCALER_X_PATH = 'scaler_X.save'
SCALER_Y_PATH = 'scaler_y.save'
TRAIN_CSV       = 'OSWEC_WecSim_Dataset.csv'
MODEL_VERSION ="0_0_2"

# -------------------------
# Load data
# -------------------------
data = pd.read_csv(TRAIN_CSV)

# Be flexible with column names (from your MATLAB export)
rename_map = {
    'Height_m': 'Height',
    'Period_s': 'Period',
    'Direction_deg': 'Direction',
    'peakPower_kW': 'PeakPower_kW'  # just in case
}
data = data.rename(columns=rename_map)

required_cols = ['Height', 'Period', 'Direction', 'PeakPower_kW']
missing = [c for c in required_cols if c not in data.columns]
if missing:
    print(f"ERROR: Missing required columns in CSV: {missing}", file=sys.stderr)
    print(f"Found columns: {list(data.columns)}", file=sys.stderr)
    sys.exit(1)

# -------------------------
# Data preparation
# -------------------------
# Features now include Direction
X = data[['Height', 'Period', 'Direction']].values.astype('float32')
y = data[['PeakPower_kW']].values.astype('float32')


# -------------------------
# Normalization (Scaling)
# -------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save the scalers for later use in prediction scripts
joblib.dump(scaler_X, SCALER_X_PATH)
joblib.dump(scaler_y, SCALER_Y_PATH)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# -------------------------
# ANN Model setup
# -------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),  # 3 inputs now
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Output: Peak power
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# -------------------------
# Train the model
# -------------------------
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# -------------------------
# Evaluate model performance
# -------------------------
evaluation = model.evaluate(X_test, y_test, verbose=0)
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# Metrics
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Test Loss (MSE, scaled): {evaluation[0]:.6f}")
print(f"Test MAE (scaled):       {evaluation[1]:.6f}")
print(f"R² Score:                {r2:.4f}")
print(f"MAE (kW):                {mae:.4f}")
print(f"RMSE (kW):               {rmse:.4f}")

# -------------------------
# Plot predicted vs actual
# -------------------------
# --- Plot loss per epoch (train vs validation) ---
plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure()
plt.scatter(y_true, y_pred, alpha=0.6)
min_v = min(y_true.min(), y_pred.min())
max_v = max(y_true.max(), y_pred.max())
plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
plt.xlabel('Actual Peak Power (kW)')
plt.ylabel('Predicted Peak Power (kW)')
plt.title('Predicted vs Actual Peak Power')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Save your model
# -------------------------
model.save(f'OSWEC_WESL_surrogate_model_v{MODEL_VERSION}.keras')
