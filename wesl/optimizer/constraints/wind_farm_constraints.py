# External libraries
import numpy as np
import openmdao.api as om
from shapely.geometry import Point, Polygon


class PairWiseSpacing(om.ExplicitComponent):

    """
    Enforce minimum spacing between turbines.

    Parameters:
    ----------------------------------------------------------------------------------
    Variable                      Description
    x (float):                     wind turbine coordinates in the x axis          
    y (float):                     wind turbine coordinates in the y axis
    n_turbines (int):              number of wind turbines          
    spacing_violation (float):     Distance between turbine pairs minus the required minimum.

    Usage:
    ----------------------------------------------------------------------------------
    User provides the number of turbines and the minimum spacing between turbines
    to the Spacing_Constraint component.

    prob.model.add_subsystem('Spacing_Constraint', 
                         PairWiseSpacing(n_turbines = 63,
                                         min_spacing = 5*wind_turbines.diameter()), 
                         promotes_inputs=['x', 'y'])
    """


    def initialize(self):
        self.options.declare('n_turbines', types=int, desc='Number of turbines')
        self.options.declare('min_spacing', types=float, desc='Minimum spacing in meters')

    def setup(self):
        n = self.options['n_turbines']
        self.add_input('x', shape=n, desc='Turbine x-coordinates')
        self.add_input('y', shape=n, desc='Turbine y-coordinates')

        # Number of unique pairs = n * (n - 1) / 2
        self.n_pairs = n * (n - 1) // 2

        self.add_output('spacing_violation', shape=self.n_pairs,
                        desc='Pairwise spacing minus min_spacing')

        # # Full sparsity for partials (not super efficient, but safe for now)
        # rows = np.repeat(np.arange(self.n_pairs), 2)
        # cols = []  # will populate in compute_partials
        self.declare_partials(of='spacing_violation', wrt='x')
        self.declare_partials(of='spacing_violation', wrt='y')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        min_spacing = self.options['min_spacing']

        ### vectorized
        coords = np.column_stack((x, y))        # shape (n,2)
    
        # Compute all pairwise differences using broadcasting
        diffs = coords[:, None, :] - coords[None, :, :]   # shape (n,n,2)
        
        # Euclidean distances
        dist_matrix = np.linalg.norm(diffs, axis=2)       # shape (n,n)
        
        # Extract upper triangular values (i<j)
        n = len(x)
        iu = np.triu_indices(n, k=1)
        dists = dist_matrix[iu]
        
        # Subtract minimum spacing
        spacing_violation = dists - min_spacing

        if np.any(spacing_violation < 0):
            print("Some spacing constraints are violated!")
                
        # outputs['spacing_violation'] = np.array(dists)

        outputs['spacing_violation'] = spacing_violation
        print(outputs['spacing_violation'])

    def compute_partials(self, inputs, J):
        x = inputs['x']
        y = inputs['y']
        n = len(x)

        row = 0

        d_spacing_dx = np.zeros((self.n_pairs, n))
        d_spacing_dy = np.zeros((self.n_pairs, n))

        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx**2 + dy**2) + 1e-12  # avoid /0

                d_spacing_dx[row, i] = dx / dist
                d_spacing_dx[row, j] = -dx / dist
                d_spacing_dy[row, i] = dy / dist
                d_spacing_dy[row, j] = -dy / dist
                row += 1

        J['spacing_violation', 'x'] = d_spacing_dx
        J['spacing_violation', 'y'] = d_spacing_dy


class BoundaryConstraint(om.ExplicitComponent):

    """
    Constraint: ensure turbines remain within a given polygon boundary.
    Returns <= 0 if inside, > 0 if outside.

    Parameters:
    ----------------------------------------------------------------------------------
    Variable                                Description
    polygon_vertices (float, np.array):     boundaries of the wind farm
    number_of_turbines (int):               number of wind turbines

    Usage:
    ----------------------------------------------------------------------------------
    User provides an np.array with as many edges as desired to define a polygon. If the
    user provides x number of edges, the shape of the np.array will be (x,2). Additionally,
    user provides the number of turbines (integer).

    prob.model.add_subsystem('Boundary_Constraint',
                         BoundaryConstraint(polygon_vertices = boundary,
                                            number_of_turbines = 63),
                         promotes_inputs=['x','y']
)

    """

    def initialize(self):
        self.options.declare(
            'polygon_vertices',
            default=np.zeros((53, 2)), ###### make it 
            types=np.ndarray,
            desc='Polygon vertices as an array of shape (53, 2).'
        )
        self.options.declare(
            "number_of_turbines", 
            types=int,
            desc='Number of wind turbines in the wind farm')

    def setup(self):
        n_wt = self.options["number_of_turbines"]

        self.add_input('x', shape=n_wt, desc='Turbine x-coordinates')
        self.add_input('y', shape=n_wt, desc='Turbine y-coordinates')
        self.add_output('boundary_cons', shape=n_wt, desc='Boundary constraint per turbine')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        poly = Polygon(self.options['polygon_vertices'])
        points = [Point(xi, yi) for xi, yi in zip(inputs['x'], inputs['y'])]

        outputs['boundary_cons'] = np.array([
            poly.exterior.distance(pt) if not poly.contains(pt)
            else -poly.exterior.distance(pt)
            for pt in points
        ])

