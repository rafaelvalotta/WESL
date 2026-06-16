from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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