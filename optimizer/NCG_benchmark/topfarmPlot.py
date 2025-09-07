import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

class TopFarmPlot:
    def __init__(self, init_x, init_y, boundary, aep0=None, spacing_diam=None):
        self.init_x = np.asarray(init_x)
        self.init_y = np.asarray(init_y)
        self.boundary = np.asarray(boundary)
        self.aep0 = aep0
        self.spacing_r = None if spacing_diam is None else 0.5 * float(spacing_diam)

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.plot(self.boundary[:, 0], self.boundary[:, 1], 'k-', lw=1.5, label="Boundary")
        self.ax.scatter(self.init_x, self.init_y, c="blue", marker="o", label="Initial")
        self.text = self.ax.text(0.01, 0.99, '', transform=self.ax.transAxes,
                                 va='top', fontsize=9,
                                 bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        self.ax.set_aspect('equal', adjustable='box')
        # set limits from boundary so points are visible
        xmin, xmax = float(self.boundary[:,0].min()), float(self.boundary[:,0].max())
        ymin, ymax = float(self.boundary[:,1].min()), float(self.boundary[:,1].max())
        pad_x = 0.1 * (xmax - xmin) if xmax > xmin else 1.0
        pad_y = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
        self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
        self.ax.set_ylim(ymin - pad_y, ymax + pad_y)
        plt.ion()

        self._scatter_curr = None
        self._lines = []
        self._rings = []
        self.iter = 0

    def update(self, x, y, aep=None):
        x = np.asarray(x)
        y = np.asarray(y)

        # clear old
        if self._scatter_curr is not None:
            self._scatter_curr.remove()
        for ln in self._lines:
            ln.remove()
        self._lines.clear()
        for rk in self._rings:
            rk.remove()
        self._rings.clear()

        # draw current
        self._scatter_curr = self.ax.scatter(x, y, c="red", marker="^", label="Current")
        for x0, y0, xi, yi in zip(self.init_x, self.init_y, x, y):
            ln, = self.ax.plot([x0, xi], [y0, yi], "k--", lw=0.6)
            self._lines.append(ln)

        # optional spacing circles
        if self.spacing_r is not None:
            for xi, yi in zip(x, y):
                circ = Circle((xi, yi), self.spacing_r, ec="gray", fc="none", ls="--", lw=0.8)
                self.ax.add_patch(circ)
                self._rings.append(circ)

        # update textbox
        if aep is not None and self.aep0 is not None:
            gain = 100 * (aep - self.aep0) / max(self.aep0, 1e-12)
            self.text.set_text(f"Iter: {self.iter}\nAEP: {aep:.3f} GWh (Δ {gain:+.2f}%)")
        elif aep is not None:
            self.text.set_text(f"Iter: {self.iter}\nAEP: {aep:.3f} GWh")
        else:
            self.text.set_text(f"Iter: {self.iter}")

        plt.draw()
        plt.pause(0.001)
        self.iter += 1
