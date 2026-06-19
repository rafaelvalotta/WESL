# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display, clear_output
from pathlib import Path
import pandas as pd

COLORS = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'purple', 'brown', 'pink']

class PlotComp(om.ExplicitComponent):
    """Notebook-optimized live plotting component for wind farm layout tracking."""

    def initialize(self):
        self.options.declare('init_x')
        self.options.declare('init_y')
        self.options.declare('polygon_vertices')
        self.options.declare('aep0', default=None)
        self.options.declare('spacing_meters', default=None)
        self.options.declare('enabled', default=True)
        self.options.declare('plot_every', default=1)
        self.options.declare('memory', default=10)

    def setup(self):
        init_x = np.asarray(self.options['init_x'], float)
        init_y = np.asarray(self.options['init_y'], float)
        self.n = len(init_x)

        # Declare OpenMDAO inputs
        self.add_input('x', val=init_x.copy(), units='m')
        self.add_input('y', val=init_y.copy(), units='m')
        self.add_input('aep', val=0.0)
        self.add_input('iter', val=0.0)
        self.add_input('objective', val=0.0)
        self.add_output('tick', val=0.0)

        # Internal state tracking
        self.init_x_cache = init_x.copy()
        self.init_y_cache = init_y.copy()
        self.iter = 0
        self.x_tracks = [[] for _ in range(self.n)]
        self.y_tracks = [[] for _ in range(self.n)]
        
        self.plot_every = self.options['plot_every']
        self.enabled = self.options['enabled']
        self.spacing_r = self.options['spacing_meters']
        self.aep0_cache = self.options['aep0']
        self.memory = self.options['memory']
        
        # Pre-compute the textured sea background once to save CPU cycles
        bd = np.asarray(self.options['polygon_vertices'])
        min_x_bd, max_x_bd = bd[:, 0].min(), bd[:, 0].max()
        min_y_bd, max_y_bd = bd[:, 1].min(), bd[:, 1].max()
        margin = 0.1
        width = max_x_bd - min_x_bd
        height = max_y_bd - min_y_bd
        
        self.extent = (min_x_bd - margin * width, max_x_bd + margin * width, 
                       min_y_bd - margin * height, max_y_bd + margin * height)
        
        Nx, Ny = 300, 300  # Balanced resolution for quick notebook redraws
        xg = np.linspace(self.extent[0], self.extent[1], Nx)
        yg = np.linspace(self.extent[2], self.extent[3], Ny)
        X, Y = np.meshgrid(xg, yg)
        waves = (np.sin((Y - self.extent[2]) * 5000 / np.pi) + np.cos((X - self.extent[0]) * 2500 / np.pi)) / 3
        noise = np.random.normal(0, 0.09, size=(Ny, Nx))
        self.seabg_cache = 0.9 + 0.15 * waves + noise

    def compute(self, inputs, outputs):
        self.iter += 1
        outputs['tick'] = 0.0
        iter_drv = int(np.asarray(inputs['iter']).ravel()[0])

        if not self.enabled or (self.plot_every > 1 and (self.iter % self.plot_every != 0)):
            return

        # Fetch pipeline parameters
        x = np.asarray(inputs['x'], float)
        y = np.asarray(inputs['y'], float)
        aep = float(np.asarray(inputs['aep']).ravel()[0])
        obj_pen = float(np.asarray(inputs['objective']).ravel()[0])

        # Track history memory buffer
        for i in range(self.n):
            self.x_tracks[i].append(x[i])
            self.y_tracks[i].append(y[i])
            if len(self.x_tracks[i]) > self.memory:
                self.x_tracks[i].pop(0)
                self.y_tracks[i].pop(0)

        # Force synchronous cell rendering inside the Notebook
        fig, ax = plt.subplots(figsize=(9, 9))

        # Render cached sea background
        ax.imshow(self.seabg_cache, extent=self.extent, origin='lower', cmap='Blues', alpha=0.20, zorder=0)

        # Plot Boundary Polygon
        bd = np.asarray(self.options['polygon_vertices'])
        if not np.allclose(bd[0], bd[-1]):
            bd = np.vstack([bd, bd[0]])
        ax.plot(bd[:, 0], bd[:, 1], color='black', linestyle='--', linewidth=2, label='Boundary', zorder=1)

        # Draw Fading Optimization Trails
        for i in range(self.n):
            color = COLORS[i % len(COLORS)]
            x_hist = np.array(self.x_tracks[i])
            y_hist = np.array(self.y_tracks[i])
            if len(x_hist) > 1:
                alphas = np.linspace(0.2, 0.9, len(x_hist))
                for j in range(len(x_hist) - 1):
                    ax.plot(x_hist[j:j+2], y_hist[j:j+2], ':', color=color, alpha=alphas[j], lw=1.2, zorder=2)

        # Draw initial configurations (Anchors) and linkages
        for i in range(self.n):
            color = COLORS[i % len(COLORS)]
            ax.plot(self.init_x_cache[i], self.init_y_cache[i], 'o', color=color, markersize=4, alpha=0.5, zorder=3)
            ax.plot([self.init_x_cache[i], x[i]], [self.init_y_cache[i], y[i]], '-', color=color, lw=1.0, alpha=0.4, zorder=3)

        # Mark Current Turbine Layout using lightweight markers ('x')
        ax.scatter(x, y, marker='x', color='black', s=45, linewidths=1.5, zorder=5, label='Turbines')

        # Draw Safety Proximity Circles (2D spacing limit)
        if self.spacing_r is not None:
            for xi, yi in zip(x, y):
                circ = Circle((xi, yi), self.spacing_r, ec='crimson', fc='none', ls=':', lw=0.7, alpha=0.4, zorder=2)
                ax.add_patch(circ)

        # Metadata Overlay (HUD)
        stats_text = f'Evaluation: {self.iter}\nIteration: {iter_drv}\nAEP: {aep:.3f} GWh'
        if self.aep0_cache is not None:
            gain = 100.0 * (aep - self.aep0_cache) / max(abs(self.aep0_cache), 1e-12)
            stats_text += f' ({gain:+6.2f}%)\nObjective: {obj_pen:.3f}'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='lightgray', boxstyle='round,pad=0.5'))

        # Axes Configurations
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X Coordinate [m]')
        ax.set_ylabel('Y Coordinate [m]')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_title('Layout Optimization', fontsize=12, fontweight='bold')

        # Trigger precise notebook output swapping
        clear_output(wait=True)
        display(fig)
        plt.close(fig) # Prevent Matplotlib from leaking memory slots



class LayoutVisualizer:
    """Post-optimization visualizer for wind farm layouts using SimpleRecorder CSV files."""
    
    def __init__(self, boundary_vertices, diameter, spacing_meters=None):
        self.bd = np.asarray(boundary_vertices)
        if not np.allclose(self.bd[0], self.bd[-1]):
            self.bd = np.vstack([self.bd, self.bd[0]])
            
        self.D = diameter
        self.spacing_r = spacing_meters

    def _parse_coordinates(self, coord_string):
        return np.array([float(v) for v in coord_string.split(';') if v.strip()])

    def load_and_plot(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Recorder file not found at: {csv_path}")

        with open(csv_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        headers = lines[0].split(',')
        data_lines = lines[1:]

        idx_aep = headers.index('aep')
        idx_x = headers.index('x')
        idx_y = headers.index('y')
        idx_viol = headers.index('rms_viol') if 'rms_viol' in headers else -1

        first_row = data_lines[0].split(',')
        last_row = data_lines[-1].split(',')

        x_init = self._parse_coordinates(first_row[idx_x])
        y_init = self._parse_coordinates(first_row[idx_y])
        aep_init = float(first_row[idx_aep])

        x_final = self._parse_coordinates(last_row[idx_x])
        y_final = self._parse_coordinates(last_row[idx_y])
        aep_final = float(last_row[idx_aep])
        viol_final = float(last_row[idx_viol]) if idx_viol != -1 else 0.0

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
        
        layouts = [
            {"title": "Initial Layout", "x": x_init, "y": y_init, "aep": aep_init, "viol": None, "ax": axes[0]},
            {"title": "Optimized Layout", "x": x_final, "y": y_final, "aep": aep_final, "viol": viol_final, "ax": axes[1]}
        ]

        min_x, max_x = self.bd[:, 0].min(), self.bd[:, 0].max()
        min_y, max_y = self.bd[:, 1].min(), self.bd[:, 1].max()
        dx, dy = max_x - min_x, max_y - min_y
        margin = 0.1
        
        xlim = (min_x - margin * dx, max_x + margin * dx)
        ylim = (min_y - margin * dy, max_y + margin * dy)

        for layout in layouts:
            ax = layout["ax"]
            x, y = layout["x"], layout["y"]
            
            ax.plot(self.bd[:, 0], self.bd[:, 1], 'k--', lw=2, label='Boundary')
            
            if self.spacing_r is not None:
                for xi, yi in zip(x, y):
                    circ = Circle((xi, yi), self.spacing_r, ec='gray', fc='none', ls=':', lw=0.7, alpha=0.5)
                    ax.add_patch(circ)
            
            ax.scatter(x, y, color='red', edgecolor='black', s=40, zorder=5, label='Turbines')
            
            ax.set_facecolor('#f4f7f6')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X [m]')
            
            stats = f"Turbines: {len(x)}\nAEP: {layout['aep']:.3f} GWh"
            if layout["title"] == "Optimized Layout":
                gain = 100.0 * (layout['aep'] - aep_init) / aep_init
                stats += f" (+{gain:.2f}%)"
                if layout["viol"] is not None:
                    stats += f"\nConstraint Viol: {layout['viol']:.3e}"
            
            ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
            
            ax.set_title(layout["title"], fontsize=14, fontweight='bold')

        axes[0].set_ylabel('Y [m]')
        plt.suptitle(f"Layout Optimization Summary — File: {csv_path.name}", fontsize=6, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()



def plot_convergence_comparison(seed=1, n_turbines=100, base_dir=None):
    """
    Loads optimization CSV logs for both SGD and NCG drivers based on a specific seed
    and plots their deterministic AEP convergence histories side by side.
    """
    if base_dir is None:
        BASE_DIR = Path.cwd().parents[0] if Path.cwd().name == "Notebooks" else Path.cwd()
    else:
        BASE_DIR = Path(base_dir)

    # Reconstruct exact log paths
    sgd_path = BASE_DIR / "Results" / "sgd" / f"WESL_{n_turbines}wt_seed_{seed}.csv"
    ncg_path = BASE_DIR / "Results" / "ncg" / f"WESL_{n_turbines}wt_seed_{seed}.csv"

    if not sgd_path.exists():
        raise FileNotFoundError(f"Missing SGD log file: {sgd_path}")
    if not ncg_path.exists():
        raise FileNotFoundError(f"Missing NCG log file: {ncg_path}")

    # Read tracking metrics
    df_sgd = pd.read_csv(sgd_path)
    df_ncg = pd.read_csv(ncg_path)

    # Initialize plot frame
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Plot SGD trajectory using the verified 'iter' and 'aep' columns
    ax.plot(
        df_sgd['iter'], 
        df_sgd['aep'], 
        label='SGD (Stochastic Gradient Descent)', 
        color='crimson', 
        linestyle='-', 
        linewidth=2,
        marker='o',
        markevery=max(1, len(df_sgd)//10)
    )

    # Plot NCG trajectory using the verified 'iter' and 'aep' columns
    ax.plot(
        df_ncg['iter'], 
        df_ncg['aep'], 
        label='NCG (Non-linear Conjugate Gradient)', 
        color='dodgerblue', 
        linestyle='-', 
        linewidth=2,
        marker='s',
        markevery=max(1, len(df_ncg)//10)
    )

    # Labeling and professional grid styling
    ax.set_title(f'AEP Convergence Profile Benchmark (Seed {seed} - {n_turbines} Turbines)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Optimizer Iteration Count', fontsize=10)
    ax.set_ylabel('Deterministic AEP [GWh]', fontsize=10)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray')

    # Calculate statistics based on the correct keys
    initial_aep = df_sgd['aep'].iloc[0]
    final_sgd_aep = df_sgd['aep'].iloc[-1]
    final_ncg_aep = df_ncg['aep'].iloc[-1]
    
    summary_text = (
        f"Initial Baseline AEP: {initial_aep:.2f} GWh\n"
        f"Final SGD AEP: {final_sgd_aep:.2f} GWh\n"
        f"Final NCG AEP: {final_ncg_aep:.2f} GWh"
    )
    ax.text(0.02, 0.05, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', family='monospace',
            bbox=dict(facecolor='whitesmoke', alpha=0.85, edgecolor='lightgray', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()