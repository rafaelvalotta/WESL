# Wave Energy

## Objects and Responsibilities

- **WaveField (Site)**  
  Provides the sea-state distribution over **wave height (H)**, **period (T)**, and **direction (D)**. The site returns either:  
  • **probabilities** `P(H,T,D)` that sum to **1.0** per device (`kind='prob'`), or  
  • **hours** `Hours(H,T,D)` that sum to **8760** per device (`kind='hours'`).

  Variants in use:  
  • `UniformWaveField.from_constant(H,T,D)`: all mass in one bin (deterministic baseline).  
  • `UniformWaveField.from_scatter_and_rose(H_edges, T_edges, D_edges, P_HT, P_D)`: non‑spatial site with `P(H,T,D) = normalize(P_HT) ⊗ normalize(P_D)` (independent direction rose).

- **OSWECDevice (Device)**  
  Supplies **peak power** on the `(H,T,D)` grid via a surrogate trained on WEC‑Sim:  
  `Power_kW(H,T,D)` (kW).  
  A scalar **alpha** maps peak to mean power:  
  `MeanPower_kW(H,T,D) ≈ alpha × Power_kW(H,T,D)`.

- **WecFarm (Farm)**  
  Binds the site and device to compute **AEP per device** and **total farm AEP**.

## AEP Formula

For each device:

1) **Local hours per bin**  
   From the site:  
   `Hours(H,T,D)`  (sums to **8760** h per device)

2) **Power per bin**  
   From the device surrogate:  
   `Power_kW(H,T,D)`

3) **Mean power approximation**  
   `MeanPower_kW(H,T,D) = alpha × Power_kW(H,T,D)`

4) **Annual Energy and AEP**  
   `Energy_kWh = Σ_{H,T,D} Hours × MeanPower_kW`  
   `AEP_GWh = Energy_kWh / 1e6`

Farm AEP is the sum over devices.

## Units & Dimensions

- `Power_kW`: kilowatts on the `(H,T,D)` grid.
- `Hours`: hours/year on the `(H,T,D)` grid; **sums to 8760** per device.
- `AEP_GWh`: gigawatt‑hours/year per device; sum for farm total.

## Assumptions & Limitations

- Current surrogate predicts **peak** power only. `alpha` is a temporary factor to approximate mean power. Future versions should predict **mean absorbed power** directly from WEC‑Sim outputs.
- `from_scatter_and_rose` assumes **independence** between `(H,T)` and `D`:  
  `P(H,T,D) = P(H,T) × P(D)`. If `P(D | H,T)` is available, prefer it.
- AEP depends strongly on the input sea‑state distribution. Use measured/validated `P_HT` (scatter table) and `P_D` (direction rose) when available.

## Quick Integrity Checks

- Probability sum per device:  
  `site.local_seastate(x, y, kind='prob').sum(('H','T','D')) == 1.0`

- Hours sum per device:  
  `site.local_seastate(x, y, kind='hours').sum(('H','T','D')) == 8760`

- Surrogate feature order matches training:  
  `(Height_m, Period_s, Direction_deg)`.

