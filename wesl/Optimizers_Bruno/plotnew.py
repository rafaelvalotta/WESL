# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

from xml.parsers.expat import model
import numpy as np
import openmdao.api as om
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

COLORS = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'purple', 'brown', 'pink']

"""
Evaluation was added for earlier line search tests, so it could show how many
model evaluations happen before one iteration. Without line search, as written,
it is already incremented during model initialization.
"""


class PlotComp(om.ExplicitComponent):
    """Plot component used to track layout optimization."""

    def initialize(self):
        self.options.declare('init_x')
        self.options.declare('init_y')
        self.options.declare('polygon_vertices')
        self.options.declare('aep0', default=None)
        self.options.declare('spacing_meters', default=None)
        self.options.declare('enabled', default=True)
        self.options.declare('plot_every', default=1)
        self.options.declare('force_agg_backend', default=False)
        self.options.declare('icon_path', default=None)
        self.options.declare('memory', default=10)

    def setup(self):
        if self.options['force_agg_backend']:
            matplotlib.use('Agg')

        init_x = np.asarray(self.options['init_x'], float)
        init_y = np.asarray(self.options['init_y'], float)
        self.n = len(init_x)

        # Current layout inputs
        self.add_input('x', val=init_x.copy(), units='m')
        self.add_input('y', val=init_y.copy(), units='m')
        self.add_input('aep', val=0.0)
        self.add_input('iter', val=0.0)
        self.add_input('objective', val=0.0)
        self.add_output('tick', val=0.0)

        try:
            self.add_input('violation', val=0.0)
        except Exception:
            pass

        # Internal plot state
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
        self.fig = None
        self.ax = None

    def compute(self, inputs, outputs):
        self.iter += 1
        outputs['tick'] = 0.0
        iter_drv = int(np.asarray(inputs['iter']).ravel()[0])

        if not self.enabled:
            return

        if self.plot_every > 1 and (self.iter % self.plot_every != 0):
            return

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Read current optimizer values
        x = np.asarray(inputs['x'], float)
        y = np.asarray(inputs['y'], float)
        aep = float(np.asarray(inputs['aep']).ravel()[0])
        obj_pen = float(np.asarray(inputs['objective']).ravel()[0])

        # Keep a short history for each turbine
        for i in range(self.n):
            self.x_tracks[i].append(x[i])
            self.y_tracks[i].append(y[i])

            if len(self.x_tracks[i]) > self.memory:
                self.x_tracks[i].pop(0)
                self.y_tracks[i].pop(0)

        # Clear and redraw the axes
        self.ax.cla()

        bd = np.asarray(self.options['polygon_vertices'])
        if not np.allclose(bd[0], bd[-1]):
            bd = np.vstack([bd, bd[0]])

        min_x_bd, max_x_bd = bd[:, 0].min(), bd[:, 0].max()
        min_y_bd, max_y_bd = bd[:, 1].min(), bd[:, 1].max()
        width = max_x_bd - min_x_bd
        height = max_y_bd - min_y_bd
        margin = 0.1

        min_x = min_x_bd - margin * width
        max_x = max_x_bd + margin * width
        min_y = min_y_bd - margin * height
        max_y = max_y_bd + margin * height

        # Draw a light textured background
        Nx, Ny = 500, 500
        xg = np.linspace(min_x, max_x, Nx)
        yg = np.linspace(min_y, max_y, Ny)
        X, Y = np.meshgrid(xg, yg)

        waves = (
            np.sin((Y - min_y) * 5000 / np.pi)
            + np.cos((X - min_x) * 2500 / np.pi)
        ) / 3
        noise = np.random.normal(0, 0.09, size=(Ny, Nx))
        seabg = 0.9 + 0.15 * waves + noise

        self.ax.imshow(
            seabg,
            extent=(min_x, max_x, min_y, max_y),
            origin='lower',
            cmap='Blues',
            alpha=0.25,
            zorder=0
        )

        self.ax.plot(
            bd[:, 0], bd[:, 1],
            color='black',
            linestyle='--',
            linewidth=2,
            label='Boundary',
            zorder=1
        )

        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.grid(True, linestyle=':', alpha=0.55)

        # Draw recent trails with fading
        for i in range(self.n):
            color = COLORS[i % len(COLORS)]
            x_hist = np.array(self.x_tracks[i])
            y_hist = np.array(self.y_tracks[i])

            if len(x_hist) > 1:
                alphas = np.linspace(0.3, 1.0, len(x_hist))
                for j in range(len(x_hist) - 1):
                    self.ax.plot(
                        x_hist[j:j+2],
                        y_hist[j:j+2],
                        '--',
                        color=color,
                        alpha=alphas[j],
                        lw=1.5,
                        zorder=2
                    )

        # Connect initial and current positions
        for i in range(self.n):
            color = COLORS[i % len(COLORS)]

            self.ax.plot(
                self.init_x_cache[i],
                self.init_y_cache[i],
                'o',
                color=color,
                markersize=6,
                zorder=4
            )

            self.ax.plot(
                [self.init_x_cache[i], x[i]],
                [self.init_y_cache[i], y[i]],
                '-',
                color=color,
                lw=2,
                alpha=0.8,
                zorder=3
            )

        # Mark current turbine positions
        self.ax.plot(x, y, 'xk', markersize=10, markeredgewidth=2, zorder=5)

        # Draw turbine icons if provided
        icon_path = self.options['icon_path']
        if icon_path is not None:
            turb_img = mpimg.imread(icon_path)
            for xi, yi in zip(x, y):
                ab = AnnotationBbox(
                    OffsetImage(turb_img, zoom=0.015),
                    (xi, yi),
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                    zorder=6
                )
                self.ax.add_artist(ab)

        # Draw spacing circles if enabled
        if self.spacing_r is not None:
            for xi, yi in zip(x, y):
                circ = Circle(
                    (xi, yi),
                    self.spacing_r,
                    ec='gray',
                    fc='none',
                    ls='--',
                    lw=0.8,
                    alpha=0.6,
                    zorder=2
                )
                self.ax.add_patch(circ)

        # Place summary text in the top-left corner
        stats_text = (
            f'Evaluation: {self.iter}\n'
            f'Iteration: {iter_drv}\n'
            f'AEP: {aep:.3f} GWh'
        )

        if self.aep0_cache is not None:
            gain = 100.0 * (aep - self.aep0_cache) / max(abs(self.aep0_cache), 1e-12)
            stats_text += f' ({gain:6.2f}%)\nObjective (pen.): {obj_pen:.3f}'

        self.ax.text(
            0.02,
            0.98,
            stats_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        self.ax.set_title('Turbine Layout Optimization')
        plt.draw()
        plt.pause(0.001)

    def __del__(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)