# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe backend, needed when saving many plots during optimization
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pathlib import Path
import xarray as xr
from scipy.interpolate import griddata


class FarmPlotter:
    """
    Saves elegant per-iteration snapshots of a wind farm layout: boundary polygon,
    turbines, offshore substations, the OptiWindNet cable network, and a bathymetry
    depth raster in the background.

    The bathymetry raster is interpolated once at construction time (it is static
    across iterations) and cached, so each save_iteration_plot() call only redraws
    the fast-changing elements (turbine positions, cable routing).
    """

    def __init__(self, farm_name, boundary_vertices, substation_coord, cable_specs,
                 bathy_nc_path, output_dir, min_spacing=None, margin_frac=0.15, grid_res=250):
        self.farm_name = farm_name
        self.boundary_vertices = np.asarray(boundary_vertices, dtype=float)
        self.substation_coord = np.atleast_2d(np.asarray(substation_coord, dtype=float))
        self.cable_specs = cable_specs
        self.min_spacing = min_spacing
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._boundary_closed = np.vstack([self.boundary_vertices, self.boundary_vertices[:1]])

        x_min, x_max = self.boundary_vertices[:, 0].min(), self.boundary_vertices[:, 0].max()
        y_min, y_max = self.boundary_vertices[:, 1].min(), self.boundary_vertices[:, 1].max()
        dx = (x_max - x_min) * margin_frac
        dy = (y_max - y_min) * margin_frac
        self.extent = (x_min - dx, x_max + dx, y_min - dy, y_max + dy)

        self._bathy_grid = self._build_bathymetry_raster(bathy_nc_path, grid_res)

        # Cable color/width ramp generalized to however many cable types are declared
        n_cables = max(len(cable_specs), 1)
        cmap = plt.get_cmap('YlOrRd')
        self._cable_colors = [cmap(0.3 + 0.6 * i / max(n_cables - 1, 1)) for i in range(n_cables)]
        self._cable_widths = np.linspace(1.4, 4.2, n_cables)

    def _build_bathymetry_raster(self, bathy_nc_path, grid_res):
        x0, x1, y0, y1 = self.extent
        with xr.open_dataset(bathy_nc_path) as ds:
            bx, by, bd = ds['x'].values, ds['y'].values, ds['depth'].values

        pad = 0.1 * max(x1 - x0, y1 - y0)
        mask = (bx >= x0 - pad) & (bx <= x1 + pad) & (by >= y0 - pad) & (by <= y1 + pad)
        if mask.sum() < 4:
            return None

        gx = np.linspace(x0, x1, grid_res)
        gy = np.linspace(y0, y1, grid_res)
        GX, GY = np.meshgrid(gx, gy)
        GZ = griddata((bx[mask], by[mask]), bd[mask], (GX, GY), method='linear')
        return GX, GY, GZ

    def _draw_cable_network(self, ax, wfn):
        G = wfn.G
        vertex_matrix = G.graph['VertexC']
        # 'fnT' maps node ids to VertexC rows; it's only present when the routing solution
        # introduced extra "detour" nodes (e.g. to bend cables around obstacles). Without
        # detours, node ids already index VertexC directly, so fall back to an identity
        # mapping -- negative ids (substations) still resolve correctly via Python's
        # negative indexing, matching the convention used when 'fnT' is present.
        node_mapper = G.graph.get('fnT', np.arange(len(vertex_matrix)))
        pos = {n: (vertex_matrix[node_mapper[n], 0], vertex_matrix[node_mapper[n], 1])
               for n in G.nodes()}

        for edge in wfn.get_network():
            src, tgt, cable_type = int(edge['src']), int(edge['tgt']), int(edge['cable'])
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            idx = min(cable_type, len(self._cable_colors) - 1)
            ax.plot([x0, x1], [y0, y1], color=self._cable_colors[idx],
                     linewidth=self._cable_widths[idx], zorder=3, solid_capstyle='round')

    def save_iteration_plot(self, cycle, iteration, turbine_x, turbine_y,
                             wfn=None, lcoe=None, aep=None, tag=None, label=None):
        fig, ax = plt.subplots(figsize=(12, 10))

        if self._bathy_grid is not None:
            GX, GY, GZ = self._bathy_grid
            # 'gouraud' shading interpolates color continuously between grid points instead
            # of contourf's discrete stepped bands, which was the main cause of the blocky
            # look flagged in review -- a smoother gradient, not more resolution than the
            # underlying (sparse, ~350m-spaced) bathymetry source actually has.
            cf = ax.pcolormesh(GX, GY, GZ, cmap='Blues', shading='gouraud', alpha=0.75, zorder=0)
            cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Depth (m)')

        ax.plot(self._boundary_closed[:, 0], self._boundary_closed[:, 1],
                 color='black', linestyle='--', linewidth=2.0, zorder=2)

        if wfn is not None:
            self._draw_cable_network(ax, wfn)

        if self.min_spacing is not None:
            # Radius = min_spacing / 2: two turbines' circles touching means they sit
            # exactly at the minimum center-to-center spacing; overlapping circles mean
            # the spacing constraint is violated.
            radius = self.min_spacing / 2.0
            for tx, ty in zip(np.atleast_1d(turbine_x), np.atleast_1d(turbine_y)):
                ax.add_patch(Circle((tx, ty), radius, facecolor='none', edgecolor='dimgray',
                                     linestyle=':', linewidth=1.0, alpha=0.6, zorder=3.5))

        ax.scatter(turbine_x, turbine_y, s=70, color='deepskyblue', edgecolor='black',
                    linewidth=0.8, zorder=4)

        for i, (sx, sy) in enumerate(self.substation_coord):
            ax.scatter(sx, sy, marker='*', s=550, color='gold', edgecolor='black',
                        linewidth=1.0, zorder=5)
            ax.annotate(f"OSS {i}", (sx, sy), xytext=(8, 8), textcoords='offset points',
                         fontsize=9, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

        x0, x1, y0, y1 = self.extent
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('equal')
        ax.set_xlabel('UTM Easting (m)')
        ax.set_ylabel('UTM Northing (m)')
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

        title = f"{self.farm_name.upper()} — {label}" if label else \
                f"{self.farm_name.upper()} — Cycle {cycle}, Iteration {iteration}"
        subtitle_bits = []
        if lcoe is not None:
            subtitle_bits.append(f"LCOE: {lcoe:,.2f} EUR/MWh")
        if aep is not None:
            subtitle_bits.append(f"AEP: {aep:,.0f} MWh")
        if wfn is not None:
            subtitle_bits.append(f"Cables: {wfn.length():,.0f} m")
        if subtitle_bits:
            title += "\n" + " | ".join(subtitle_bits)
        ax.set_title(title, fontsize=12, fontweight='bold')

        legend_handles = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label='Boundary'),
            Line2D([0], [0], marker='o', linestyle='', markersize=9, color='deepskyblue',
                   markeredgecolor='black', label='Turbines'),
            Line2D([0], [0], marker='*', linestyle='', markersize=15, color='gold',
                   markeredgecolor='black', label='Substation (OSS)'),
        ]
        if self.min_spacing is not None:
            legend_handles.append(Line2D([0], [0], color='dimgray', linestyle=':', linewidth=1.0,
                                          label=f'Min. Spacing (Ø {self.min_spacing:.0f} m)'))
        for k in range(len(self.cable_specs)):
            capacity = self.cable_specs[k][0]
            legend_handles.append(Line2D([0], [0], color=self._cable_colors[k],
                                          linewidth=self._cable_widths[k],
                                          label=f'Cable (Capacity: {int(capacity)})'))
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

        fig.tight_layout()
        suffix = f"_{tag}" if tag else ""
        fname = self.output_dir / f"{self.farm_name}_cycle{cycle:02d}_iter{iteration:04d}{suffix}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        return fname
