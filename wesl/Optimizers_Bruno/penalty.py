# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import numpy as np
import openmdao.api as om

class FinalRMSViol(om.ExplicitComponent):
    """RMS positive violations (g>0): sqrt(mean(g_plus^2))"""

    def initialize(self):
        self.options.declare('nconstraints', types=int, default=0,
                           desc='Total constraints in g_vector')
    
    def setup(self):
        nc = self.options['nconstraints']
        self.add_input('g_vector', val=np.zeros(nc), units='m')  # ← units explicit
        self.add_output('rms_viol', val=0.0, units='m')
        
        # Partials
        self.declare_partials('rms_viol', 'g_vector')
    
    def compute(self, inputs, outputs):
        g = inputs['g_vector'].copy()
        g_plus = np.maximum(g, 0.0)
        rms = np.sqrt(np.mean(g_plus**2))
        outputs['rms_viol'] = rms
    
    def compute_partials(self, inputs, partials):
        g = inputs['g_vector']
        g_plus = np.maximum(g, 0.0)
        rms = np.sqrt(np.mean(g_plus**2))
        
        if rms > 1e-12:
            drms_dg = g_plus / (rms * len(g))  # Derivada analítica
        else:
            drms_dg = np.zeros_like(g)
            
        partials['rms_viol', 'g_vector'] = drms_dg



class PenaltyObjectiveComp(om.ExplicitComponent):
    """
    Penalty Objective Component.

    objective = -AEP + sum(max(g, 0)^2)

    """

    def initialize(self):
        self.options.declare('n_constraints', types=int,
                             desc='Total number of aggregated constraints')

    def setup(self):
        nc = self.options['n_constraints']

        self.add_input('aep', val=0.0)
        self.add_input('g_vector', val=np.zeros(nc), units='m')

        self.add_output('objective', val=0.0)
        self.add_output('violation', val=0.0)
        self.add_output('penalty', val=0.0)

        self.declare_partials('objective', ['aep', 'g_vector'])
        self.declare_partials('violation', 'g_vector')
        self.declare_partials('penalty', 'g_vector')

    def compute(self, inputs, outputs):
        aep = float(inputs['aep'])
        g_plus = np.maximum(inputs['g_vector'], 0.0)
        penalty_sum = np.sum(g_plus ** 2)

        outputs['objective'] = -aep + penalty_sum
        outputs['violation'] = float(np.linalg.norm(g_plus))
        outputs['penalty'] = penalty_sum

    def compute_partials(self, inputs, partials):
        g = inputs['g_vector']
        g_plus = np.maximum(g, 0.0)
        dPdg = np.where(g > 0.0, 2.0 * g, 0.0)

        partials['objective', 'aep'] = -1.0
        partials['objective', 'g_vector'] = -dPdg

        norm_gplus = np.linalg.norm(g_plus)
        partials['violation', 'g_vector'] = g_plus / norm_gplus if norm_gplus > 0.0 else np.zeros_like(g)
        partials['penalty', 'g_vector'] = -dPdg