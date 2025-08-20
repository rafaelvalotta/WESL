# waveField.py
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from utils.wave_utils import def_bins, direction_marginal, plot_wave_rose

try:
    from scipy.ndimage import gaussian_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


class XRWaveField:
    """
    XR-backed wave field.
    ds coords: H, T, D (bin centers). Optional: x, y (spatial grid).
    ds data_vars: 'P' (probabilities) or 'Hours' (per cell).
    """
    def __init__(self, ds: xr.Dataset, interp_method: str = "linear", bounds: str = "limit"):
        if not set(['H', 'T', 'D']).issubset(ds.coords):
            raise ValueError("Dataset must define coords: H, T, D")
        if ('P' not in ds.data_vars) and ('Hours' not in ds.data_vars):
            raise ValueError("Dataset must contain 'P' or 'Hours'")
        if interp_method not in ("linear", "nearest"):
            raise ValueError("interp_method must be 'linear' or 'nearest'")
        if bounds not in ("limit", "check"):
            raise ValueError("bounds must be 'limit' or 'check'")
        self.ds = ds
        self.interp_method = interp_method
        self.bounds = bounds

        # Normalize P and ensure Hours exists
        if 'P' in self.ds:
            ax = [d for d in ('H', 'T', 'D') if d in self.ds.P.dims]
            s = self.ds.P.sum(ax)
            self.ds['P'] = xr.where(s > 1e-12, self.ds.P / s, 0)
        else:
            ax = [d for d in ('H', 'T', 'D') if d in self.ds.Hours.dims]
            s = self.ds.Hours.sum(ax)
            self.ds['P'] = xr.where(s > 1e-12, self.ds.Hours / s, 0)

        if 'Hours' not in self.ds:
            self.ds['Hours'] = self.ds.P * 8760.0

    # ----- core query -------------------------------------------------
    def _clip(self, xq, yq):
        """Clip query points to domain for bounds='limit' or raise for 'check'."""
        if ('x' not in self.ds.coords) or ('y' not in self.ds.coords):
            return xq, yq
        xmin, xmax = float(self.ds.x.min()), float(self.ds.x.max())
        ymin, ymax = float(self.ds.y.min()), float(self.ds.y.max())
        xq = np.asarray(xq, float); yq = np.asarray(yq, float)
        if self.bounds == "limit":
            return np.clip(xq, xmin, xmax), np.clip(yq, ymin, ymax)
        if (xq.min() < xmin) or (xq.max() > xmax) or (yq.min() < ymin) or (yq.max() > ymax):
            raise ValueError("Query outside domain; set bounds='limit' to clip.")
        return xq, yq

    def local_seastate(self, x, y, kind="prob"):
        """
        Interpolate to device positions and return weights over (H,T,D).
        kind: 'prob' (sum=1) or 'hours' (sum=8760).
        """
        x = np.atleast_1d(x); y = np.atleast_1d(y)
        if x.shape != y.shape:
            raise ValueError("x and y must have same shape")
        xq, yq = self._clip(x, y)

        if ('x' in self.ds.coords) and ('y' in self.ds.coords):
            Pq = self.ds.P.interp(x=xr.DataArray(xq, dims='wec'),
                                  y=xr.DataArray(yq, dims='wec'),
                                  method=self.interp_method)
        else:
            Pq = self.ds.P.expand_dims(wec=xr.DataArray(np.arange(x.size), dims='wec'))

        s = Pq.sum(dim=('H', 'T', 'D'))
        Pq = xr.where(s > 1e-12, Pq / s, 0)

        if kind == "prob":  return Pq
        if kind == "hours": return Pq * 8760.0
        raise ValueError("kind must be 'prob' or 'hours'")

    # ----- site-level summaries ---------------------------------------
    def direction_distribution(self, x=None, y=None, kind="prob", aggregate="global"):
        """
        Direction weights.
        - aggregate='per_wec': interpolate first and return per-device dir weights (sum=1 per device).
        - aggregate='global': area-average over grid (if x,y grid exists); else just sum over H,T and normalize.
        """
        if aggregate == "per_wec":
            if x is None or y is None:
                raise ValueError("x,y required for aggregate='per_wec'")
            Pq = self.local_seastate(x, y, kind="prob").transpose('wec', 'H', 'T', 'D')
            return Pq.sum(dim=('H', 'T'))  # dims: ('wec','D'), each row sums to 1

        # global (grid-average) distribution
        dir_w = self.ds.P.sum(dim=('H', 'T'))
        if ('x' in dir_w.dims) and ('y' in dir_w.dims):
            # normalize per cell then area-average
            s = dir_w.sum(dim='D')
            dir_w_norm = xr.where(s > 1e-12, dir_w / s, 0)
            return dir_w_norm.mean(dim=('x', 'y'))  # dims: ('D',)
        # non-spatial site
        return dir_w / dir_w.sum('D')

    def expected_map(self, quantity='H'):
        """
        Expected value for H or T.
        Returns a scalar for non-spatial sites or an (x,y) map for spatial sites.
        """
        if quantity not in ('H', 'T'):
            raise ValueError("quantity must be 'H' or 'T'")

        w = self.ds.P  # dims: ('H','T','D', ['x','y']?)

        # Build a 1D DataArray on the matching axis so xarray broadcasts by name
        if quantity == 'H':
            vals = xr.DataArray(self.ds['H'].values, dims=('H',), coords={'H': self.ds['H']})
        else:  # 'T'
            vals = xr.DataArray(self.ds['T'].values, dims=('T',), coords={'T': self.ds['T']})

        # Broadcast-multiply and sum out (H,T,D)
        m = (w * vals).sum(dim=('H', 'T', 'D'))

        # Non-spatial site → scalar
        if m.dims == ():
            return float(m.values)
        return m  # dims: ('x','y') for spatial sites



class UniformWaveField(XRWaveField):
    """Non-spatial site with fixed P(H,T,D)."""
    def __init__(self, H_edges, T_edges, D_edges, pdf=None, seed=None,
                 interp_method="linear", bounds="limit"):
        H, T, D = def_bins(H_edges, T_edges, D_edges)
        nH, nT, nD = len(H), len(T), len(D)
        if pdf is None:
            rng = np.random.default_rng(seed)
            arr = rng.random((nH, nT, nD))
        else:
            arr = np.asarray(pdf, float)
            if arr.shape != (nH, nT, nD):
                raise ValueError(f"pdf shape {arr.shape} must be {(nH, nT, nD)}")
        arr = arr / arr.sum()
        ds = xr.Dataset(coords={'H': H, 'T': T, 'D': D},
                        data_vars={'P': (('H', 'T', 'D'), arr)})
        super().__init__(ds, interp_method=interp_method, bounds=bounds)

    @classmethod
    def from_constant(cls, H, T, D, eps=1e-6):
        """
        Build a degenerate 1x1x1 bins site where all probability is at (H,T,D).
        Keeps your bins-based machinery but behaves like a single sea state.
        """
        import numpy as np  # safe if numpy already imported at file top
        H_edges = [float(H) - eps, float(H) + eps]
        T_edges = [float(T) - eps, float(T) + eps]
        D_edges = [float(D) - eps, float(D) + eps]
        pdf = np.ones((1, 1, 1), dtype=float)   # all mass in the single bin
        return cls(H_edges, T_edges, D_edges, pdf=pdf)
    
    @classmethod
    def from_scatter_and_rose(cls, H_edges, T_edges, D_edges, P_HT, P_D):
        """
        Build a non-spatial site from a joint H–T scatter table and a direction rose.

        Inputs
        ------
        H_edges : array-like, shape (nH+1,)
        T_edges : array-like, shape (nT+1,)
        D_edges : array-like, shape (nD+1,)
            Bin edges for Height [m], Period [s], and Direction [deg].
        P_HT    : array-like, shape (nH, nT)
            Joint probability mass over (H,T). Does not need to be normalized.
        P_D     : array-like, shape (nD,)
            Probability mass over D. Does not need to be normalized.

        Returns
        -------
        UniformWaveField instance with pdf(H,T,D) = normalize(P_HT) ⊗ normalize(P_D).

        Notes
        -----
        - This assumes D is independent of (H,T): P(H,T,D) = P(H,T)*P(D).
        - The base UniformWaveField constructor will normalize the 3D pdf to sum to 1.
        """

        H_edges = np.asarray(H_edges, float)
        T_edges = np.asarray(T_edges, float)
        D_edges = np.asarray(D_edges, float)
        P_HT    = np.asarray(P_HT, float)
        P_D     = np.asarray(P_D, float)

        nH = H_edges.size - 1
        nT = T_edges.size - 1
        nD = D_edges.size - 1

        if P_HT.shape != (nH, nT):
            raise ValueError(f"P_HT must have shape {(nH, nT)}, got {P_HT.shape}")
        if P_D.shape != (nD,):
            raise ValueError(f"P_D must have shape {(nD,)}, got {P_D.shape}")

        # Normalize 2D and 1D marginals (robust to zeros)
        s_ht = P_HT.sum()
        if s_ht <= 0:
            raise ValueError("P_HT sum must be > 0")
        P_HTn = P_HT / s_ht

        s_d = P_D.sum()
        if s_d <= 0:
            raise ValueError("P_D sum must be > 0")
        P_Dn = P_D / s_d

        # 3D pdf via outer product
        pdf = P_HTn[:, :, None] * P_Dn[None, None, :]

        # Delegate to existing constructor (it will also normalize)
        return cls(H_edges, T_edges, D_edges, pdf=pdf)



class RandomGridWaveField(XRWaveField):
    """
    Spatial site on grid (x,y) with random P(H,T,D,x,y).
    Use smooth_sigma>0 (in grid cells) to create spatially correlated fields.
    """
    def __init__(self, H_edges, T_edges, D_edges, x, y, grid_P=None, seed=None,
                 interp_method="linear", bounds="limit", smooth_sigma=1.5):
        H = 0.5 * (np.asarray(H_edges, float)[:-1] + np.asarray(H_edges, float)[1:])
        T = 0.5 * (np.asarray(T_edges, float)[:-1] + np.asarray(T_edges, float)[1:])
        D = 0.5 * (np.asarray(D_edges, float)[:-1] + np.asarray(D_edges, float)[1:])
        x = np.asarray(x, float); y = np.asarray(y, float)
        nH, nT, nD, nx, ny = len(H), len(T), len(D), len(x), len(y)

        if grid_P is None:
            rng = np.random.default_rng(seed)
            arr = rng.random((nH, nT, nD, nx, ny))
        else:
            arr = np.asarray(grid_P, float)
            if arr.shape != (nH, nT, nD, nx, ny):
                raise ValueError(f"grid_P shape {arr.shape} must be {(nH,nT,nD,nx,ny)}")

        # Optional spatial smoothing to produce correlated fields
        if smooth_sigma and smooth_sigma > 0:
            if not _HAVE_SCIPY:
                print("Warning: scipy not found; smoothing skipped.")
            else:
                for ih in range(nH):
                    for it in range(nT):
                        for idd in range(nD):
                            arr[ih, it, idd] = gaussian_filter(arr[ih, it, idd], sigma=smooth_sigma, mode='nearest')

        # Normalize per (x,y) cell
        s = arr.reshape(nH * nT * nD, nx * ny).sum(axis=0).reshape(1, 1, 1, nx, ny)
        arr = np.where(s > 1e-12, arr / s, 0)

        ds = xr.Dataset(coords={'H': H, 'T': T, 'D': D, 'x': x, 'y': y},
                        data_vars={'P': (('H', 'T', 'D', 'x', 'y'), arr)})
        super().__init__(ds, interp_method=interp_method, bounds=bounds)

    @classmethod
    def from_scatter_and_rose_maps(cls, H_edges, T_edges, D_edges, x, y, P_HT_map, P_D_map):
        """
        Build a spatial site where each grid cell (x,y) has:
        - a joint (H,T) scatter P_HT_map[:, :, ix, iy]
        - a direction rose P_D_map[:, ix, iy]
        The 3D pdf per cell is outer-product:
            P(H,T,D | x,y) = normalize(P_HT) * normalize(P_D)
        Normalization is done PER CELL so that sum(H,T,D) == 1 for every (x,y).

        Parameters
        ----------
        H_edges : (nH+1,) bin edges for H
        T_edges : (nT+1,) bin edges for T
        D_edges : (nD+1,) bin edges for D
        x       : (nx,)  grid coordinates
        y       : (ny,)  grid coordinates
        P_HT_map: (nH, nT, nx, ny) non-negative weights per cell
        P_D_map : (nD, nx, ny)     non-negative weights per cell

        Returns
        -------
        RandomGridWaveField instance with ds.P shaped (nH,nT,nD,nx,ny)
        """
        import numpy as np
        import xarray as xr

        H_edges = np.asarray(H_edges, float); T_edges = np.asarray(T_edges, float); D_edges = np.asarray(D_edges, float)
        x = np.asarray(x, float); y = np.asarray(y, float)
        H = 0.5*(H_edges[:-1] + H_edges[1:])
        T = 0.5*(T_edges[:-1] + T_edges[1:])
        D = 0.5*(D_edges[:-1] + D_edges[1:])
        nH, nT, nD, nx, ny = len(H), len(T), len(D), len(x), len(y)

        P_HT_map = np.asarray(P_HT_map, float)
        P_D_map  = np.asarray(P_D_map, float)

        if P_HT_map.shape != (nH, nT, nx, ny):
            raise ValueError(f"P_HT_map must have shape {(nH,nT,nx,ny)}, got {P_HT_map.shape}")
        if P_D_map.shape != (nD, nx, ny):
            raise ValueError(f"P_D_map must have shape {(nD,nx,ny)}, got {P_D_map.shape}")

        # Normalize (H,T) per cell
        s_ht = P_HT_map.sum(axis=(0,1), keepdims=True)  # (1,1,nx,ny)
        # Fallback: if a cell sums to 0, make that cell uniform over H,T
        zero_ht = (s_ht <= 1e-12)
        if np.any(zero_ht):
            P_HT_map = P_HT_map.copy()
            P_HT_map[:, :, zero_ht[0,0]] = 1.0
            s_ht = P_HT_map.sum(axis=(0,1), keepdims=True)
        P_HTn = P_HT_map / s_ht

        # Normalize D per cell
        s_d = P_D_map.sum(axis=0, keepdims=True)       # (1,nx,ny)
        zero_d = (s_d <= 1e-12)
        if np.any(zero_d):
            P_D_map = P_D_map.copy()
            P_D_map[:, zero_d[0]] = 1.0
            s_d = P_D_map.sum(axis=0, keepdims=True)
        P_Dn = P_D_map / s_d

        # Outer product per cell → (nH,nT,nD,nx,ny)
        pdf = P_HTn[:, :, None, :, :] * P_Dn[None, None, :, :, :]

        ds = xr.Dataset(
            coords={'H': H, 'T': T, 'D': D, 'x': x, 'y': y},
            data_vars={'P': (('H','T','D','x','y'), pdf)}
        )
        # Reuse XRWaveField init to set P normalization & Hours
        return cls(H_edges, T_edges, D_edges, x, y, grid_P=pdf, seed=None,
                interp_method="linear", bounds="limit")



# ------------------- Demo / quick self-test -------------------
if __name__ == "__main__":
    # Bin edges
    H_edges = np.linspace(0.5, 4.0, 8)
    T_edges = np.linspace(5.0, 10.0, 11)
    D_edges = np.linspace(0.0, 90.0, 10)

    # Spatial grid
    xg = np.linspace(0, 1000, 31)
    yg = np.linspace(0, 1000, 31)

    # Build random spatial wavefield with smoothing
    site = RandomGridWaveField(H_edges, T_edges, D_edges, xg, yg, seed=3, smooth_sigma=2.0)

    # Device positions
    x_dev = np.array([120.0, 555.0, 880.0])
    y_dev = np.array([100.0, 420.0, 600.0])

    # Interpolate per-device distributions (probabilities)
    P = site.local_seastate(x_dev, y_dev, kind="prob")  # dims: ('H','T','D','wec')
    print("Per-WEC sum(prob):", P.sum(dim=('H','T','D')).values)

    # Smooth expected maps E[H](x,y) and E[T](x,y)
    EH = site.expected_map('H')
    ET = site.expected_map('T')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = ax[0].imshow(EH.T, origin='lower', extent=(xg.min(), xg.max(), yg.min(), yg.max()), aspect='auto')
    ax[0].scatter(x_dev, y_dev, c='w', s=30, edgecolor='k')
    ax[0].set_title('E[H](x,y)')
    ax[0].set_xlabel('x [m]'); ax[0].set_ylabel('y [m]')
    plt.colorbar(im0, ax=ax[0], label='Wave Height H (m)')

    im1 = ax[1].imshow(ET.T, origin='lower', extent=(xg.min(), xg.max(), yg.min(), yg.max()), aspect='auto')
    ax[1].scatter(x_dev, y_dev, c='w', s=30, edgecolor='k')
    ax[1].set_title('E[T](x,y)')
    ax[1].set_xlabel('x [m]'); ax[1].set_ylabel('y [m]')
    plt.colorbar(im1, ax=ax[1], label='Wave Period T (s)')
    plt.show()

    # One (H*,T*,D*) spatial slice (should look smoother with smoothing)
    Hc, Tc, Dc = def_bins(H_edges, T_edges, D_edges)
    iH, iT, iD = len(Hc)//2, len(Tc)//2, len(Dc)//2
    Z = site.ds.P.isel(H=iH, T=iT, D=iD).values.T
    plt.figure(figsize=(6,4))
    im = plt.imshow(Z, origin='lower', extent=(xg.min(), xg.max(), yg.min(), yg.max()), aspect='auto')
    plt.scatter(x_dev, y_dev, s=40, c='k', marker='x', label='WEC')
    plt.colorbar(im, label=f"P at H={Hc[iH]:.2f} m, T={Tc[iT]:.2f} s, D={Dc[iD]:.1f}°")
    plt.title('Spatial interpolation of P(H*,T*,D*)')
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.legend(); plt.tight_layout(); plt.show()

    # Direction distributions:
    # global (site-level)
    dir_glob = site.direction_distribution(aggregate="global").values
    plot_wave_rose(D_edges, dir_glob, title="Global wave direction distribution")

    # per device
    dir_per = site.direction_distribution(x_dev, y_dev, aggregate="per_wec").values
    plot_wave_rose(D_edges, dir_per[0], title="Wave direction distribution — WEC 0")
