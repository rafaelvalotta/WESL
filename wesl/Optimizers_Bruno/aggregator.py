# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import numpy as np
import openmdao.api as om

class ConstraintAggregator(om.ExplicitComponent):
    """
    Concatenates spacing and boundary constraint vectors:
      g_vector = [spacing_cons, boundary_cons]
    """
    def __init__(self, n_turbines: int):
        super().__init__()
        self.n = int(n_turbines)
        self.m_spacing = self.n * (self.n - 1) // 2
        self.m_boundary = self.n
        self.m_total = self.m_spacing + self.m_boundary

    def setup(self):
        self.add_input("spacing_cons", val=np.zeros(self.m_spacing), units="m")
        self.add_input("boundary_cons", val=np.zeros(self.m_boundary), units="m")
        self.add_output("g_vector", val=np.zeros(self.m_total), units="m")
        # Jacobian is [I  I] in block form

        rows = np.arange(self.m_total, dtype=int)
        cols = np.arange(self.m_total, dtype=int)
        self.declare_partials("g_vector", "spacing_cons",
                              rows=rows[:self.m_spacing], cols=np.arange(self.m_spacing))
        self.declare_partials("g_vector", "boundary_cons",
                              rows=rows[self.m_spacing:], cols=np.arange(self.m_boundary))

    def compute(self, inputs, outputs):
        outputs["g_vector"] = np.concatenate([inputs["spacing_cons"], inputs["boundary_cons"]])

    def compute_partials(self, inputs, partials):
        # d[g_vector]/d[spacing_cons] = [I; 0]
        # d[g_vector]/d[boundary_cons] = [0; I]
        partials["g_vector", "spacing_cons"] = np.ones(self.m_spacing)
        partials["g_vector", "boundary_cons"] = np.ones(self.m_boundary)
