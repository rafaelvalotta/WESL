# wec_farm.py
import numpy as np
import xarray as xr
from WaveEnergy.waveField import RandomGridWaveField, UniformWaveField
from WaveEnergy.wec_device import OSWECDevice

class WecFarm():
    """
    Minimal farm using one device model for all WECs.
    AEP is returned per device in GWh/year.
    """
    def __init__(self, site, x, y, device, name=None):
        # site: XR-backed wavefield with ds['H','T','D'] and local_seastate(kind='hours'|'prob')
        # x,y : device coordinates (same length)
        # device: object with .power_grid(Hc,Tc,Dc)->(nH,nT,nD) kW and .alpha
        self.site = site
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have same shape")
        self.device = device
        self.name = name or "WEC Farm"

        ds = self.site.ds
        self.Hc = np.asarray(ds['H'].values, float)
        self.Tc = np.asarray(ds['T'].values, float)
        self.Dc = np.asarray(ds['D'].values, float)

    def power_grid(self) -> xr.DataArray:
        """Power on (H,T,D) in kW."""
        PkW = self.device.power_grid(self.Hc, self.Tc, self.Dc)
        return xr.DataArray(
            PkW,
            dims=('H','T','D'),
            coords={'H': self.Hc, 'T': self.Tc, 'D': self.Dc},
            name='Power_kW'
        )

    def local_hours(self) -> xr.DataArray:
        """Annual hours per bin at each device. Sums to 8760 per device."""
        return self.site.local_seastate(self.x, self.y, kind='hours').fillna(0.0)

    def aep(self) -> xr.DataArray:
        """Per-device AEP in GWh/year."""
        P = self.power_grid()
        Hrs = self.local_hours()
        kwh = (Hrs * (self.device.alpha * P)).sum(dim=('H','T','D'))  # ('wec',)
        gwh = kwh * 1e-6
        gwh.name = 'AEP_GWh_per_year'
        return gwh

    def aep_farm(self) -> float:
        """Total farm AEP in GWh/year."""
        return float(self.aep().sum().values)
    

    def expected_peak_power_kW(self):
        """
        E[P_peak(H,T,D)] per device [kW] under the site's (H,T,D) probabilities.
        For UniformWaveField.from_constant, this equals the surrogate power at that (H,T,D).
        """
        Prob = self.site.local_seastate(self.x, self.y, kind='prob').fillna(0.0)  # dims: (H,T,D,wec)
        P = self.power_grid()                                                     # dims: (H,T,D)
        expP = (Prob * P).sum(dim=('H', 'T', 'D')).values                         # -> (wec,)
        return expP

    def suggest_alpha(self, target_AEP_GWh=None, target_mean_kW=None):
        """
        Suggest scalar alpha so that mean power matches target.
        Provide exactly one of:
        - target_AEP_GWh (per device), or
        - target_mean_kW (per device)
        """
        if (target_mean_kW is None) == (target_AEP_GWh is None):
            raise ValueError("Provide exactly one of target_mean_kW or target_AEP_GWh")

        if target_mean_kW is None:
            target_mean_kW = float(target_AEP_GWh) * 1e6 / 8760.0  # kW

        expP = self.expected_peak_power_kW()  # kW per device
        expP_mean = float(np.mean(expP))      # average over devices
        if expP_mean <= 0:
            raise ValueError("Expected peak power is non-positive; check surrogate/site.")
        return float(target_mean_kW / expP_mean)
    
    # def aep_breakdown(self, by='D'):
    #     """
    #     Return per-device AEP contributions (GWh) aggregated by one axis.
    #     by: 'D' (direction), 'H' (height), or 'T' (period).
    #     """
    #     P = self.power_grid()                     # (H,T,D)
    #     Hrs = self.local_hours()                  # (H,T,D,wec)
    #     energy_kwh = Hrs * (self.device.alpha * P)  # broadcast -> (H,T,D,wec)

    #     if by == 'D':
    #         vals = energy_kwh.sum(dim=('H', 'T')) * 1e-6
    #         vals = vals.transpose('wec', 'D')
    #         vals.name = 'AEP_GWh_by_D'
    #     elif by == 'H':
    #         vals = energy_kwh.sum(dim=('T', 'D')) * 1e-6
    #         vals = vals.transpose('wec', 'H')
    #         vals.name = 'AEP_GWh_by_H'
    #     elif by == 'T':
    #         vals = energy_kwh.sum(dim=('H', 'D')) * 1e-6
    #         vals = vals.transpose('wec', 'T')
    #         vals.name = 'AEP_GWh_by_T'
    #     else:
    #         raise ValueError("by must be 'D', 'H', or 'T'")
    #     return vals


if __name__=='__main__':

    # --- Build a spatial site from per-cell scatter & rose ---
    H_edges = np.linspace(0.5, 4.0, 11)   # nH=10
    T_edges = np.linspace(5.0, 10.0, 11)  # nT=10
    D_edges = np.linspace(0.0, 90.0, 10)  # nD=9

    xg = np.linspace(0, 1000, 41)  # nx
    yg = np.linspace(0, 1000, 41)  # ny
    nx, ny = len(xg), len(yg)

    # Bin centers for H,T,D
    Hc = 0.5*(H_edges[:-1]+H_edges[1:])
    Tc = 0.5*(T_edges[:-1]+T_edges[1:])
    Dc = 0.5*(D_edges[:-1]+D_edges[1:])

    # Base (H,T) shape (Gaussian peak) reused across cells
    HH, TT = np.meshgrid(Hc, Tc, indexing='ij')
    P_HT_base = np.exp(-((HH-2.5)**2)/(2*0.7**2)) * np.exp(-((TT-8.0)**2)/(2*1.0**2))  # (nH,nT)

    # Spatial weights W(x,y) to vary the climate smoothly
    Xg, Yg = np.meshgrid(xg, yg, indexing='ij')
    W = 0.6 + 0.4*np.exp(-((Xg-700)**2+(Yg-300)**2)/(2*300**2))  # (nx,ny)

    # Broadcast to (nH,nT,nx,ny)
    P_HT_map = P_HT_base[:, :, None, None] * W[None, None, :, :]

    # Direction rose varies with x: peak near 30° on left, 60° on right
    P_D_map = np.empty((len(Dc), nx, ny))
    for ix in range(nx):
        peak = 30 + 30*(xg[ix]/xg[-1])                     # scalar
        row = 0.6*np.exp(-0.5*((Dc - peak)/20.0)**2) + 0.4 # (nD,)
        P_D_map[:, ix, :] = row[:, None] * np.ones((1, ny))  # make (nD, ny)

    site = RandomGridWaveField.from_scatter_and_rose_maps(H_edges, T_edges, D_edges, xg, yg, P_HT_map, P_D_map)


    # Devices
    x_dev = [100, 300, 800]
    y_dev = [150, 500, 800]

    # OSWEC device (auto-loads your surrogate/scalers; alpha=0.5 default)
    device = OSWECDevice()

    farm = WecFarm(site, x_dev, y_dev, device)

    # Existing prints…
    print("Per-device AEP [GWh]:", farm.aep().values)
    print("Farm AEP [GWh]:", farm.aep_farm())

    # # 1) Direction breakdown and sum check
    # D_brk = farm.aep_breakdown('D')              # (wec, D)
    # print("Sum over D equals AEP? ",
    #     np.allclose(D_brk.sum('D').values, farm.aep().values))

    # # 2) Top-3 contributing directions for device 0
    # top_idx = np.argsort(D_brk.values[0])[::-1][:3]
    # print("Top-3 D bins (deg centers) for WEC0:",
    #     farm.Dc[top_idx], "GWh:", D_brk.values[0, top_idx])

    # 3) (Similarly for H or T if needed)
    # H_brk = farm.aep_breakdown('H'); T_brk = farm.aep_breakdown('T')


    # # See expected peak power per device (kW)
    # print("Expected peak kW per device:", farm.expected_peak_power_kW())

    # # Choose alpha to hit ~2.05 GWh/device (≈ 234 kW mean)
    # alpha_star = farm.suggest_alpha(target_AEP_GWh=2.05)
    # print(f"Suggested alpha ≈ {alpha_star:.3f}")

    # # Apply and re-evaluate
    # farm.device.alpha = alpha_star
    # print("Per-device AEP (after alpha) [GWh]:", farm.aep().values)
    # print("Farm AEP (after alpha) [GWh]:", farm.aep_farm())

