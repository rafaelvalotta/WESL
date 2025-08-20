# WEC-Sim Surrogate Model Development and Wave Field Optimization

## 1. Overview

This documentation describes the process of developing a surrogate model for a WEC-Sim simulation of the RM3 WEC device, and the subsequent integration of this model into an optimization framework for wave energy converter (WEC) positioning in a spatially varying wave field.

---

## 2. WEC-Sim Simulation Data Generation

### Step 1: Simulation Setup in MATLAB (WEC-Sim)

- **Device**: RM3 WEC device
- **Simulation Type**: Regular waves
- **Parameters Varied** (randomized for each run):
  - Wave Height (Hs)
  - Wave Period (Tp)
- **Number of Simulations**: 150
- **Output Recorded**: Peak power from each simulation

### Step 2: Data Table Creation

- Simulation results were compiled into a MATLAB `.mat` file containing columns for:
  - `WaveHs`
  - `WaveTp`
  - `PeakPower`
- This table served as the basis for surrogate model training.

---

## 3. Data Extraction with Python

### Script: `wecSimFetchData.py`

- Purpose: Load `.mat` results from MATLAB and export them into a `.csv` format.
- Steps:
  1. Load `.mat` file
  2. Extract variables: wave height, period, direction, and peak power
  3. Save data to `surrogate_dataset.csv` for use in the surrogate model.

---

## 4. ANN Surrogate Model Development

### Script: `wecSimANN_model_gen.py`

- **Goal**: Train an Artificial Neural Network (ANN) to predict peak power output given wave parameters.
- **Method**:
  1. Load `surrogate_dataset.csv`
  2. Normalize inputs (`WaveHs`, `WaveTp`, and output (`PeakPower`)
  3. Split data into training and test sets
  4. Define ANN architecture:
     - Dense layers with ReLU activation
     - Dropout for regularization
  5. Train model with Adam optimizer and early stopping
  6. Save trained ANN for later use in optimization

---

## 5. Wave Field Simulation

### Script: `_waveField.py`

- **Purpose**: Generate a spatially varying wave field to simulate environmental conditions.
- **Approach**:
  - Define a spatial grid for the wave field
  - Assign random variations in wave height, period, and direction across the grid to mimic real ocean conditions
  - Interpolate wave parameters for arbitrary positions
- **Reason for Randomness**:
  - Adds variability for robust optimization
  - Ensures surrogate model is tested against diverse conditions

---

## 6. Integration into the Optimizer

### Script: `optimize_wec_loc.py`

- **Goal**: Optimize WEC position to maximize peak power in the generated wave field.
- **Key Steps**:
  1. Load trained ANN surrogate model
  2. Import wave field from `_waveField.py`
  3. Define objective function:
     - Given `(x, y)` position, interpolate wave parameters from field
     - Use ANN to predict power
     - Maximize predicted power
  4. Apply constraints for valid deployment area
  5. Use optimization algorithm (e.g., COBYLA) to find optimal `(x, y)`

---

## 7. Visualization

- Initial WEC position plotted on wave field map
- Optimized position plotted alongside for comparison
- Peak power contour maps generated to visualize spatial distribution of potential

---

