# wave_utils.py
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def def_bins(H_edges, T_edges, D_edges):
    """Compute bin centers from edges."""
    H_edges = np.asarray(H_edges, float)
    T_edges = np.asarray(T_edges, float)
    D_edges = np.asarray(D_edges, float)
    if H_edges.ndim != 1 or H_edges.size < 2: raise ValueError("H_edges must be 1D len>=2")
    if T_edges.ndim != 1 or T_edges.size < 2: raise ValueError("T_edges must be 1D len>=2")
    if D_edges.ndim != 1 or D_edges.size < 2: raise ValueError("D_edges must be 1D len>=2")
    H = 0.5*(H_edges[:-1] + H_edges[1:])
    T = 0.5*(T_edges[:-1] + T_edges[1:])
    D = 0.5*(D_edges[:-1] + D_edges[1:])
    return H, T, D

def direction_marginal(dist_da: xr.DataArray):
    """Sum over H,T to obtain per-direction weights per device."""
    return dist_da.sum(dim=('H','T'))

def plot_wave_rose(D_edges_deg, dir_weights, title="Wave Rose"):
    """Polar bar chart of per-direction weights."""
    D_edges_deg = np.asarray(D_edges_deg, float)
    dir_weights = np.asarray(dir_weights, float)
    theta = np.deg2rad(0.5*(D_edges_deg[:-1] + D_edges_deg[1:]))
    widths = np.deg2rad(np.diff(D_edges_deg))
    ax = plt.subplot(111, projection='polar')
    ax.bar(theta, dir_weights, width=widths, align='center')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
