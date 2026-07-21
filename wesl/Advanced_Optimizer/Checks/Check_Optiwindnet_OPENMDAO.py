import numpy as np
import openmdao.api as om
import windIO
from shapely.geometry import Point, Polygon
from optiwindnet.api import WindFarmNetwork, EWRouter

class OptiWindNetComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('site_yaml_path', types=str)
        self.options.declare('system_yaml_path', types=str)

    def setup(self):
        site_dat = windIO.load_yaml(self.options['site_yaml_path'])
        revolution_poly_dat = site_dat['boundaries']['polygons'][1]
        self.boundary_vertices = np.column_stack((revolution_poly_dat['x'], revolution_poly_dat['y']))
        self.shapely_boundary = Polygon(self.boundary_vertices)

        system_dat = windIO.load_yaml(self.options['system_yaml_path'])
        revolution_data = system_dat['wind_farm'][1]

        substations_list = []
        electrical_substations = revolution_data.get('electrical_substations', [])
        for sub_item in electrical_substations:
            sub_coords = sub_item['electrical_substation']['coordinates']
            sub_x = float(sub_coords['x'][0])
            sub_y = float(sub_coords['y'][0])
            substations_list.append([sub_x, sub_y])
        self.substation_coord = np.array(substations_list, dtype=float)

        self.cable_specs = np.array(
            list(zip([3, 5, 7], [368.9, 428.9, 737.1])), 
            dtype=[("capacity", int), ("cost", float)]
        )

        x_raw = np.array(revolution_data['layouts'][0]['coordinates']['x'], dtype=float)
        y_raw = np.array(revolution_data['layouts'][0]['coordinates']['y'], dtype=float)
        
        x_filtrado, y_filtrado = [], []
        for x_val, y_val in zip(x_raw, y_raw):
            ponto_wt = Point(x_val, y_val)
            if self.shapely_boundary.contains(ponto_wt) or ponto_wt.distance(self.shapely_boundary) < 1.0:
                x_filtrado.append(x_val)
                y_filtrado.append(y_val)
        
        self.n_turbines = len(x_filtrado)

        self.add_input('turbine_x', val=np.array(x_filtrado, dtype=float))
        self.add_input('turbine_y', val=np.array(y_filtrado, dtype=float))

        self.add_output('cable_cost', val=0.0)
        self.add_output('cable_length', val=0.0)

        self.declare_partials('cable_cost', ['turbine_x', 'turbine_y'])
        self.declare_partials('cable_length', ['turbine_x', 'turbine_y'])

    def compute(self, inputs, outputs):
        turbines_coord = np.column_stack((inputs['turbine_x'], inputs['turbine_y']))
        
        self.wfn = WindFarmNetwork(
            turbinesC=turbines_coord,
            substationsC=self.substation_coord,
            cables=self.cable_specs,
            borderC=self.boundary_vertices
        )
        self.wfn.merge_obstacles_into_border()
        self.wfn.add_buffer(buffer_dist=0.1)
        
        self.wfn.optimize(router=EWRouter())
        
        outputs['cable_cost'] = self.wfn.cost()
        outputs['cable_length'] = self.wfn.length()

    def compute_partials(self, inputs, partials):
        grad_wts_cost, _ = self.wfn.gradient(gradient_type='cost')
        grad_wts_length, _ = self.wfn.gradient(gradient_type='length')
        
   
        print("\n--- DIAGNÓSTICO DE GRADIENTES INTERNOS ---")
        print(f"Formato original do grad_wts_cost: {grad_wts_cost.shape}")
        print(f"Amostra dos 3 primeiros gradientes de custo (X): {grad_wts_cost[:3, 0]}")
        

        partials['cable_cost', 'turbine_x'] = grad_wts_cost[:, 0]
        partials['cable_cost', 'turbine_y'] = grad_wts_cost[:, 1]
        partials['cable_length', 'turbine_x'] = grad_wts_length[:, 0]
        partials['cable_length', 'turbine_y'] = grad_wts_length[:, 1]


if __name__ == "__main__":
    from pathlib import Path
    
    BASE_DIR = Path("/Users/brunoboer/Documents/Software/Test_Wesl_jun23/wesl/Advanced_Optimizer")
    SYSTEM_YAML = BASE_DIR / "Data" / "vineyard_revolution_system.yaml"
    SITE_YAML = BASE_DIR / "Data" / "site_us.yaml"

    if not SITE_YAML.exists():
        BASE_DIR = Path("/Users/brunoboer/Documents/Software/Test_Wesl_jun23/wesl")
        SYSTEM_YAML = BASE_DIR / "Data" / "vineyard_revolution_system.yaml"
        SITE_YAML = BASE_DIR / "Data" / "site_us.yaml"

    prob = om.Problem()
    model = prob.model

    model.add_subsystem(
        'optiwindnet_comp', 
        OptiWindNetComponent(site_yaml_path=str(SITE_YAML), system_yaml_path=str(SYSTEM_YAML)),
        promotes=['*']
    )

    prob.setup()
    prob.run_model()

    print("\n=======================================================")
    print("         OPENMDAO COMPONENT EXECUTION TEST             ")
    print("=======================================================")
    print(f"Promoted Cable Cost Output:   {prob.get_val('cable_cost')[0]:,.2f} EUR")
    print(f"Promoted Cable Length Output: {prob.get_val('cable_length')[0]:,.2f} meters")
    print("=======================================================\n")

    print("Checking component partial derivatives via OpenMDAO validation tool...")
    prob.check_partials(compact_print=True, method='fd')