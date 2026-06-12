# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import numpy as np
import openmdao.api as om

class SpacingConstraintComp(om.ExplicitComponent):
    """
    Pairwise spacing constraint: g_ij = d_min - ||p_i - p_j|| (g<=0 feasible, g>0 violation)
    """
    def __init__(self, n_turbines: int, min_spacing: float, eps: float = 1e-12):
        super().__init__()
        self.n = n_turbines
        self.dmin = float(min_spacing)
        self.eps = eps
        self.pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        self.m = len(self.pairs)

    def setup(self):
        self.add_input("x", val=np.zeros(self.n), units="m")
        self.add_input("y", val=np.zeros(self.n), units="m")
        self.add_output("spacing_cons", val=np.zeros(self.m), units="m")

        rows_x, cols_x, rows_y, cols_y = [], [], [], []
        for row, (i, j) in enumerate(self.pairs):
            rows_x += [row, row]; cols_x += [i, j]
            rows_y += [row, row]; cols_y += [i, j]

        self.declare_partials("spacing_cons", "x",
                              rows=np.array(rows_x, int), cols=np.array(cols_x, int))
        self.declare_partials("spacing_cons", "y",
                              rows=np.array(rows_y, int), cols=np.array(cols_y, int))

    def compute(self, inputs, outputs):
        x, y = inputs["x"], inputs["y"]
        g = np.empty(self.m)
        for k, (i, j) in enumerate(self.pairs):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy + self.eps)
            cons_violated = self.dmin - d
            
            # Set to zero if feasible
            if cons_violated < 0:
                g[k] = 0.0
            else:
                g[k] = cons_violated
        outputs["spacing_cons"] = g

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        n = len(x)
        m = len(self.pairs)
        eps = self.eps

        nnz = 2 * m  # Two non-zero elements per pair (for x[i], x[j])
        jac_x = np.zeros(nnz)
        jac_y = np.zeros(nnz)

        for k, (i, j) in enumerate(self.pairs):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx**2 + dy**2) + eps
            cons_violated = self.dmin - d
            
            # Only calculate derivatives if there is a violation
            if cons_violated > 0.0:
                # Position of non-zero elements in jac_x and jac_y
                jac_x[2*k]     = (-dx / d)  # ∂g/∂x[i]
                jac_x[2*k+1]   = (+dx / d)  # ∂g/∂x[j]
                jac_y[2*k]     = (-dy / d)  # ∂g/∂y[i]
                jac_y[2*k+1]   = (+dy / d)  # ∂g/∂y[j]
            else:
                jac_x[2*k]     = 0.0
                jac_x[2*k+1]   = 0.0
                jac_y[2*k]     = 0.0
                jac_y[2*k+1]   = 0.0

        partials["spacing_cons", "x"] = -jac_x
        partials["spacing_cons", "y"] = -jac_y

class BoundaryConstraintComp(om.ExplicitComponent):
    """
    Axis-aligned rectangular boundary. One constraint per turbine.
    Convention: g <= 0 inside (feasible), g > 0 outside (violation).
    """

    def __init__(self, boundary_vertices, n_turbines):
        super().__init__()
        b = np.asarray(boundary_vertices, float)
        if b.shape != (4, 2):
            raise ValueError("Expecting 4x2 rectangle vertices (axis-aligned).")
        xs, ys = b[:, 0], b[:, 1]
        self.xmin = float(xs.min())
        self.xmax = float(xs.max())
        self.ymin = float(ys.min())
        self.ymax = float(ys.max())
        self.n = int(n_turbines)

    def setup(self):
        self.add_input('x', val=np.zeros(self.n), units='m')
        self.add_input('y', val=np.zeros(self.n), units='m')
        self.add_output('boundary_cons', val=np.zeros(self.n),units='m')
        rows = np.arange(self.n, dtype=int)
        cols = np.arange(self.n, dtype=int)
        self.declare_partials(of='boundary_cons', wrt='x', rows=rows, cols=cols)
        self.declare_partials(of='boundary_cons', wrt='y', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        x = np.asarray(inputs['x'], float)
        y = np.asarray(inputs['y'], float)
        g = np.empty(self.n, float)
        for i in range(self.n):
            xi, yi = float(x[i]), float(y[i])
            ex = (self.xmin - xi) if xi < self.xmin else ((xi - self.xmax) if xi > self.xmax else 0.0)
            ey = (self.ymin - yi) if yi < self.ymin else ((yi - self.ymax) if yi > self.ymax else 0.0)
            if ex != 0.0 or ey != 0.0:
                g[i] = np.hypot(ex, ey)
            else:
                g[i] = 0.0
        outputs['boundary_cons'] = g

    def compute_partials(self, inputs, J):
        x = np.asarray(inputs['x'], float)
        y = np.asarray(inputs['y'], float)
        dgdx = np.zeros(self.n, float)
        dgdy = np.zeros(self.n, float)
        eps = 1e-12

        for i in range(self.n):
            xi, yi = float(x[i]), float(y[i])
            ex = (self.xmin - xi) if xi < self.xmin else ((xi - self.xmax) if xi > self.xmax else 0.0)
            ey = (self.ymin - yi) if yi < self.ymin else ((yi - self.ymax) if yi > self.ymax else 0.0)
            g_i = np.hypot(ex, ey)
            
            # Only calculate derivatives if violating (g > 0)
            if g_i > 0.0:
                r = g_i + eps
                if xi < self.xmin:
                    dgdx[i] = (-ex) / r
                elif xi > self.xmax:
                    dgdx[i] = (ex) / r
                else:
                    dgdx[i] = 0.0
                if yi < self.ymin:
                    dgdy[i] = (-ey) / r
                elif yi > self.ymax:
                    dgdy[i] = (ey) / r
                else:
                    dgdy[i] = 0.0
            else:
                dgdx[i], dgdy[i] = 0.0, 0.0

        J['boundary_cons', 'x'] = -dgdx
        J['boundary_cons', 'y'] = -dgdy


