import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
import xarray as xr
from matplotlib.patches import FancyArrowPatch


import sys, pathlib

# --- bootstrap to support running this file directly (no hardcoded absolute paths) ---
if __name__ == "__main__" and __package__ is None:
    import os
    # add parent of WaveEnergy/ (i.e., optimizer/) to sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ------------------------------------------------------------------------------

# Then use package imports consistently
from WaveEnergy.waveField import RandomGridWaveField, UniformWaveField
from WaveEnergy.wec_device import OSWECDevice
from WaveEnergy.wecFarm import WecFarm

# ------------------------
# Helpers
# ------------------------
def aep_single_device_map(site, device, Hc, Tc, Dc):
    """
    Compute AEP [GWh/yr] for placing a single device at each (x,y) grid node.
    Returns xr.DataArray dims ('x','y').
    """
    if 'x' not in site.ds.coords or 'y' not in site.ds.coords:
        raise ValueError("Site must be spatial (have 'x' and 'y') for a map.")

    xg = np.asarray(site.ds['x'].values, float)
    yg = np.asarray(site.ds['y'].values, float)
    X, Y = np.meshgrid(xg, yg, indexing='ij')
    x_flat, y_flat = X.ravel(), Y.ravel()

    # Hours per bin at each hypothetical device position
    Hrs = site.local_seastate(x_flat, y_flat, kind='hours').fillna(0.0)  # (H,T,D,wec)

    # Device power on (H,T,D)
    PkW = device.power_grid(Hc, Tc, Dc)  # np.ndarray (nH,nT,nD)
    P = xr.DataArray(PkW, dims=('H','T','D'),
                     coords={'H': Hc, 'T': Tc, 'D': Dc}, name='Power_kW')

    kwh = (Hrs * (device.alpha * P)).sum(dim=('H','T','D'))  # (wec,)
    gwh = (kwh * 1e-6).values.reshape(X.shape)  # (nx,ny)
    return xr.DataArray(gwh, coords={'x': xg, 'y': yg}, dims=('x','y'), name='AEP_GWh')


# ------------------------
# OpenMDAO components
# ------------------------
class AEPComp(om.ExplicitComponent):
    """
    Sum of per-device AEP [GWh/yr] for the current (x,y).
    Options:
      site  : XR-backed wavefield (with local_seastate(kind='hours'))
      device: has .power_grid(Hc,Tc,Dc)->(nH,nT,nD) [kW] and .alpha
      Hc,Tc,Dc: bin centers from site.ds coords
    """
    def initialize(self):
        self.options.declare("site")
        self.options.declare("device")
        self.options.declare("Hc")
        self.options.declare("Tc")
        self.options.declare("Dc")
        self.options.declare("nd", types=int)

    def setup(self):
        nd = self.options["nd"]
        self.add_input("x", val=np.zeros(nd))
        self.add_input("y", val=np.zeros(nd))
        self.add_output("AEP_total", val=0.0)

        # Finite-difference step ≈ half a grid cell (improves gradient signal)
        site = self.options["site"]
        xmin, xmax = float(site.ds.x.min()), float(site.ds.x.max())
        ymin, ymax = float(site.ds.y.min()), float(site.ds.y.max())
        if site.ds.x.size > 1:
            dx = float(np.diff(site.ds.x).mean())
        else:
            dx = (xmax - xmin) / 20.0
        if site.ds.y.size > 1:
            dy = float(np.diff(site.ds.y).mean())
        else:
            dy = (ymax - ymin) / 20.0
        fd_step = 0.5 * min(dx, dy)

        self.declare_partials(of="AEP_total", wrt=["x", "y"],
                            method="fd", form="central",
                            step_calc="abs", step=fd_step)


    def compute(self, inputs, outputs):
        site   = self.options["site"]
        device = self.options["device"]
        Hc     = self.options["Hc"]; Tc = self.options["Tc"]; Dc = self.options["Dc"]

        x = np.asarray(inputs["x"], float)
        y = np.asarray(inputs["y"], float)

        # Precompute power grid once per evaluation (kW)
        PkW = device.power_grid(Hc, Tc, Dc)  # (nH,nT,nD)
        alpha = device.alpha

        # Hours(H,T,D,wec) for each device position
        Hrs = site.local_seastate(x, y, kind='hours').fillna(0.0)  # xr.DataArray

        # Broadcast multiply & sum over bins, then sum devices
        aep_kWh = (Hrs * (alpha * PkW)).sum(dim=('H','T','D'))   # per wec, kWh/yr
        aep_GWh = float(aep_kWh.sum().values) * 1e-6
        outputs["AEP_total"] = aep_GWh


class SpacingConstraintComp(om.ExplicitComponent):
    """
    c[k] = d_ij - D_min  for all i<j.  Enforce lower=0 to get d_ij >= D_min.
    """
    def initialize(self):
        self.options.declare("nd", types=int)
        self.options.declare("D_min", types=float)

    def setup(self):
        nd = self.options["nd"]
        self.pairs = [(i, j) for i in range(nd) for j in range(i+1, nd)]
        self.add_input("x", val=np.zeros(nd))
        self.add_input("y", val=np.zeros(nd))
        self.add_output("c", val=np.zeros(len(self.pairs)))

        self.declare_partials(of="c", wrt=["x", "y"],
                              method="fd", form="central",
                              step_calc="abs", step=50.0)

    def compute(self, inputs, outputs):
        x = np.asarray(inputs["x"], float)
        y = np.asarray(inputs["y"], float)
        D_min = float(self.options["D_min"])

        c = np.zeros(len(self.pairs))
        for k, (i, j) in enumerate(self.pairs):
            c[k] = np.hypot(x[i]-x[j], y[i]-y[j]) - D_min
        outputs["c"] = c


# ------------------------
# Problem builder
# ------------------------
def wec_problem(site, device, x_init, y_init, D_min):
    nd = len(x_init)
    Hc = site.ds['H'].values; Tc = site.ds['T'].values; Dc = site.ds['D'].values

    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output("x", val=np.asarray(x_init, float))
    ivc.add_output("y", val=np.asarray(y_init, float))
    prob.model.add_subsystem("p", ivc, promotes=["*"])

    # model
    prob.model.add_subsystem(
        "farm",
        AEPComp(site=site, device=device, Hc=Hc, Tc=Tc, Dc=Dc, nd=nd),
        promotes_inputs=["x", "y"],
        promotes_outputs=["AEP_total"]
    )
    prob.model.add_subsystem(
        "spacing",
        SpacingConstraintComp(nd=nd, D_min=float(D_min)),
        promotes_inputs=["x", "y"],
        promotes_outputs=["c"]
    )
    
    prob.model.add_subsystem("obj", om.ExecComp("f = -AEP_total"),
                             promotes=["f", "AEP_total"])
    prob.model.add_objective("f")

    # ---------- scaling (IMPORTANT) ----------
    xmin, xmax = float(site.ds.x.min()), float(site.ds.x.max())
    ymin, ymax = float(site.ds.y.min()), float(site.ds.y.max())

    # Scale x,y to [0,1] for the optimizer (via ref/ref0)
    prob.model.add_design_var("x", lower=xmin, upper=xmax, ref=xmax, ref0=xmin)
    prob.model.add_design_var("y", lower=ymin, upper=ymax, ref=ymax, ref0=ymin)

    # Scale spacing constraint (typical magnitude ~ hundreds–thousands)
    prob.model.add_constraint("c", lower=0.0, scaler=1.0/max(1.0, D_min))

    # Grid-aware FD step (only used if we choose SLSQP)
    if site.ds.x.size > 1:
        dx = float(np.diff(site.ds.x).mean())
    else:
        dx = (xmax - xmin) / 20.0
    if site.ds.y.size > 1:
        dy = float(np.diff(site.ds.y).mean())
    else:
        dy = (ymax - ymin) / 20.0
    grid_step = 0.5 * min(dx, dy)

    # ---------- SLSQP driver ----------
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["maxiter"] = 300
    prob.driver.options["tol"] = 1e-3
    prob.driver.options["disp"] = True

    # Finite-difference total derivatives with an absolute step ~ half a grid
    prob.model.approx_totals(method='fd', form='central', step_calc='abs', step=grid_step)

    prob.setup()
    return prob




# ------------------------
# Demo / quick run
# ------------------------
if __name__ == "__main__":
    # ---- 1) Site (spatial) ----
    H_edges = np.linspace(0.5, 4.0, 20)
    T_edges = np.linspace(5.0, 10.0, 20)
    D_edges = np.linspace(0.0, 90.0, 13)

    xg = np.linspace(0, 1200, 10)
    yg = np.linspace(0, 1200, 10)

    # Example: build random spatial climate
    site = RandomGridWaveField(H_edges, T_edges, D_edges, xg, yg, smooth_sigma=0.5)

    # ---- 2) Device (auto-loads surrogate & scalers)
    device = OSWECDevice()  # pick your alpha; 5.0 ~ 2.6-2.8 GWh per device in your tests

    # ---- 3) Initial layout (grid-ish)
    x_init = np.array([100, 300, 500, 700, 900,   150, 350, 550, 750, 950,   250, 450], dtype=float)
    y_init = np.array([150, 150, 150, 150, 150,   400, 400, 400, 400, 400,   650, 650], dtype=float)
    nd = len(x_init)


    # ---- 4) Build & run optimization ----
    D_min = 150.0  # meters
    prob = wec_problem(site, device, x_init, y_init, D_min)

    # Get bin centers for plotting/maps
    Hc = site.ds['H'].values
    Tc = site.ds['T'].values
    Dc = site.ds['D'].values

    # Initial evaluation
    prob.run_model()
    AEP_init = prob.get_val("AEP_total").item()

    # Optimize
    prob.run_driver()

    x_opt = prob.get_val("x").astype(float)
    y_opt = prob.get_val("y").astype(float)
    AEP_opt = prob.get_val("AEP_total").item()

    gain_pct = 100.0 * (AEP_opt - AEP_init) / max(1e-9, abs(AEP_init))
    print("\n=== Farm AEP Optimization ===")
    print(f"Devices: {len(x_init)},  Spacing min = {D_min:.1f} m")
    print(f"AEP total [GWh]: init={AEP_init:.3f}   opt={AEP_opt:.3f}   gain={gain_pct:.2f}%")

    # ---- 5) Background AEP map (single device) + paths ----
    Amap = aep_single_device_map(site, device, Hc, Tc, Dc)
    X, Y = np.meshgrid(Amap.x, Amap.y, indexing='ij')

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    cf = ax.contourf(X, Y, Amap.values, levels=60)
    cbar = plt.colorbar(cf, ax=ax); cbar.set_label("AEP [GWh]")

    for i in range(len(x_init)):

        arrow = FancyArrowPatch(
            (x_init[i], y_init[i]), (x_opt[i], y_opt[i]),
            arrowstyle='-|>',                 # clean arrow head
            linestyle=(0, (6, 3)),            # dashed: 6 on, 3 off
            color='k', lw=1.2, alpha=0.7,
            mutation_scale=12,                # arrowhead size
            shrinkA=0, shrinkB=0
        )

        ax.add_patch(arrow)
        ax.scatter([x_init[i]], [y_init[i]], marker='o', s=60, facecolors='white', edgecolors='k', label="Initial Position")
        ax.scatter([y_opt[i]*0 + x_opt[i]], [x_opt[i]*0 + y_opt[i]],
                marker='*', s=100, facecolors='yellow', edgecolors='k', zorder=5, label="Final Position")
        if i==0:
            plt.legend()

    ax.set_title(f"Layout optimization — AEP gain: {gain_pct:.2f}%")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect('equal', 'box'); ax.grid(True, alpha=0.25)
    plt.tight_layout(); plt.show()
