import numpy as np
import openmdao.api as om
from optiwindnet.api import WindFarmNetwork, EWRouter

class OptiWindNetComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('boundary_vertices', types=np.ndarray)
        self.options.declare('substation_coord', types=np.ndarray)
        self.options.declare('cable_specs', types=np.ndarray)
        self.options.declare('n_turbines', types=int)

    def setup(self):
        n_wt = self.options['n_turbines']

        self.add_input('turbine_x', shape=n_wt)
        self.add_input('turbine_y', shape=n_wt)

        self.add_output('cable_cost', val=0.0)
        self.add_output('cable_length', val=0.0)

        self.declare_partials('cable_cost', ['turbine_x', 'turbine_y'])
        self.declare_partials('cable_length', ['turbine_x', 'turbine_y'])

    def compute(self, inputs, outputs):
        turbines_coord = np.column_stack((inputs['turbine_x'], inputs['turbine_y']))
        
        self.wfn = WindFarmNetwork(
            turbinesC=turbines_coord,
            substationsC=self.options['substation_coord'],
            cables=self.options['cable_specs'],
            borderC=self.options['boundary_vertices']
        )
        self.wfn.merge_obstacles_into_border()
        self.wfn.add_buffer(buffer_dist=0.1)
        
        self.wfn.optimize(router=EWRouter())
        
        outputs['cable_cost'] = self.wfn.cost()
        outputs['cable_length'] = self.wfn.length()

    def compute_partials(self, inputs, partials):
        grad_wts_cost, _ = self.wfn.gradient(gradient_type='cost')
        grad_wts_length, _ = self.wfn.gradient(gradient_type='length')
        
        partials['cable_cost', 'turbine_x'] = grad_wts_cost[:, 0]
        partials['cable_cost', 'turbine_y'] = grad_wts_cost[:, 1]
        partials['cable_length', 'turbine_x'] = grad_wts_length[:, 0]
        partials['cable_length', 'turbine_y'] = grad_wts_length[:, 1]