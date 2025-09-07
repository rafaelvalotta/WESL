# WaveField (XR‑backed wave resource)

This module provides a lightweight, **xarray-backed** representation of a wave resource ("wave field") and a small set of query utilities needed to drive WEC farm simulations and AEP calculations.

It stores a discretized sea-state distribution over **wave height (H)**, **wave period (T)** and **wave direction (D)**, optionally varying over **space (x,y)**. It can then return, for any set of device coordinates, the **local sea-state weights** (probabilities or annual hours) that you can combine with a power model (e.g., a surrogate) to compute energy and AEP.

---

## Repository layout

```
waveField.py         # XRWaveField + UniformWaveField + RandomGridWaveField
utils/wave_utils.py  # helpers: bin centers, direction marginal, wave-rose plot
```

---

## Requirements

- Python 3.9+
- numpy, xarray, matplotlib
- Optional: scipy (for Gaussian smoothing of spatial fields)

Install:
```bash
pip install numpy xarray matplotlib scipy
```

---

## Data model

**Axes**
- `H` — wave-height bin centers (m)
- `T` — wave-period bin centers (s)
- `D` — wave-direction bin centers (deg; any user range, e.g., 0–90)
- Optional `x`, `y` — spatial grid coordinates (m)

**Variables**
- `P(H,T,D[,x,y])` — probability per sea-state bin (normalized)
- or `Hours(H,T,D[,x,y])` — annual hours per bin (if given, it is normalized internally to `P`; `Hours = P * 8760` is stored as well)

**Normalization rule**  
For each spatial cell `(x,y)`, `sum_{H,T,D} P = 1`.
For non-spatial datasets, the single cube is normalized once.

---

## Public API

### `utils.wave_utils`

- `def_bins(H_edges, T_edges, D_edges) -> (H, T, D)`  
  Return bin **centers** from edges.

- `direction_marginal(P_ilk)`  
  Sum over `H,T` to obtain per-direction weights.

- `plot_wave_rose(D_edges, dir_weights, title)`  
  Polar bar plot of a direction distribution.

---

### `XRWaveField` (core)

```python
XRWaveField(ds: xr.Dataset, interp_method="linear", bounds="limit")
```
- `ds` must have coords `H,T,D` (and optional `x,y`) and either `P` or `Hours`.
- `interp_method`: `"linear"` or `"nearest"` (used for spatial interpolation).
- `bounds`:  
  - `"limit"` → query points outside domain are clipped to edges.  
  - `"check"` → an error is raised if outside.

#### Methods

- `local_seastate(x, y, kind="prob") -> xr.DataArray`  
  Interpolate to device positions and return weights over `(H,T,D)` per device.  
  - `kind="prob"` → each device’s weights sum to **1**  
  - `kind="hours"` → each device’s weights sum to **8760**  
  **Dims:** `('H','T','D','wec')`

- `direction_distribution(x=None, y=None, kind="prob", aggregate="global"|"per_wec")`  
  - `aggregate="per_wec"`: interpolate first, then return per-device direction weights (`('wec','D')`, each row sums to 1).  
  - `aggregate="global"`: area-averaged site-level direction distribution (`('D',)`) for spatial sites, or normalized `sum_{H,T} P` for non-spatial.

- `expected_map(quantity='H'|'T')`  
  - Spatial site: returns `E[H](x,y)` or `E[T](x,y)` as a 2‑D map.  
  - Non-spatial site: returns a scalar.

---

### Site classes (inherit XRWaveField)

- `UniformWaveField(H_edges, T_edges, D_edges, pdf=None, seed=None, interp_method="linear", bounds="limit")`  
  Non-spatial site with a single `P(H,T,D)` shared by all devices. If `pdf` is `None`, a random positive cube is generated and normalized.

  - `UniformWaveField.from_constant(H,T,D):` convenience constructor that creates a degenerate` 1×1×1` bins dataset with all probability mass at (H,T,D). This enables deterministic “single sea state” AEP calculations without changing the bins-based API.

- `RandomGridWaveField(H_edges, T_edges, D_edges, x, y, grid_P=None, seed=None, interp_method="linear", bounds="limit", smooth_sigma=1.5)`  
  Spatial site on a user grid `(x,y)` with `P(H,T,D,x,y)`.  
  If `grid_P` is `None`, a random field is generated; set `smooth_sigma>0` (in grid **cells**) to apply Gaussian smoothing and create spatial correlation before normalization.

---

## Usage

### 1) Non‑spatial site
```python
import numpy as np
from waveField import UniformWaveField
from utils.wave_utils import def_bins

H_edges = np.linspace(0.5, 4.0, 8)
T_edges = np.linspace(5.0, 10.0, 11)
D_edges = np.linspace(0, 90, 10)

site = UniformWaveField(H_edges, T_edges, D_edges, seed=1)

x = [0.0, 200.0, 400.0]
y = [0.0, 100.0, 200.0]

P = site.local_seastate(x, y, kind="prob")  # ('H','T','D','wec'), sums=1 per device
```

### 2) Spatial site + smoothing
```python
from waveField import RandomGridWaveField
from utils.wave_utils import def_bins, direction_marginal, plot_wave_rose

xg = np.linspace(0, 1000, 31)
yg = np.linspace(0, 1000, 31)
site = RandomGridWaveField(H_edges, T_edges, D_edges, xg, yg, seed=3, smooth_sigma=2.0)

x_dev = [120.0, 555.0, 880.0]
y_dev = [100.0, 420.0, 600.0]

# Interpolated distributions per device
P = site.local_seastate(x_dev, y_dev, kind="prob")   # ('H','T','D','wec')

# Expected maps for visualization
EH = site.expected_map('H')  # 2-D array over (x,y)
ET = site.expected_map('T')

# Direction distributions
dir_global = site.direction_distribution(aggregate="global")
dir_perwec = site.direction_distribution(x_dev, y_dev, aggregate="per_wec")

# Wave‑rose plot for device 0
plot_wave_rose(D_edges, dir_perwec.values[0], title="WEC 0 wave direction")
```

---

## AEP recipe (using a power surrogate)

1. Build a site (`UniformWaveField` or `RandomGridWaveField`).  
2. Get local weights at device coordinates:
   ```python
   P_loc = site.local_seastate(x_dev, y_dev, kind="hours")  # hours per bin
   ```
3. Evaluate the surrogate on the `(H,T,D)` grid to get `Power(H,T,D)` in kW.  
4. Multiply and sum over bins for each device:
   ```python
   # align dims: P_loc('H','T','D','wec'), Power('H','T','D')
   AEP_kWh_per_wec = (P_loc * Power).sum(dim=('H','T','D'))  # kWh per year per device
   ```
5. Sum across devices for farm AEP if needed.

---

## Notes and assumptions

- Direction bins are linear in the user-specified range (e.g., 0–90°). Wrap-around at 360° is not handled here.
- Smoothing uses `scipy.ndimage.gaussian_filter` when available.
- Generated random fields are synthetic; no bathymetry/currents are included at this stage.
- No direct readers for hindcast/BEMIO yet; the plan is to add I/O adapters that return an `xarray.Dataset` on `(x,y,H,T,D)` with `P` or `Hours`.

---

## Quick self‑test

Execute the module to run a simple self-test and generate validation plots:
```bash
python waveField.py
```
Outputs:
- `Per-WEC sum(prob): [...]` confirming normalization.
- Maps of `E[H](x,y)` and `E[T](x,y)`.
- One spatial slice of `P(H*,T*,D*,x,y)`.
- Wave-rose plots (global and one device).

---

## Changelog

- v0.0.1 — Initial XR-backed wave field, interpolation, direction distributions, expected maps, and two site classes.
