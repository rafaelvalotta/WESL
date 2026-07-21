import numpy as np
import openmdao.api as om
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# =============================================================================
# 1. PURE BOUNDARY CONSTRAINT COMPONENT
# =============================================================================

class BoundaryComp(om.ExplicitComponent):
    """
    OpenMDAO component for multi-polygon boundary constraints (inclusion/exclusion).
    Calculates the perpendicular distance of each turbine to its nearest edge.
    Positive values represent internal positions (valid), negative values represent external positions (violation).
    """
    def __init__(self, n_wt, zones, units=None):
        super().__init__()
        self.n_wt = n_wt
        self.zones = zones  # List of tuples: (coordinates_array, inclusion_flag) where inclusion_flag is 1 or 0
        self.units = units
        self._cache_input = None
        self._cache_output = None
        
        # Merge geometries using Shapely to handle overlaps and exclusions natively
        self.boundaries = self._calc_resulting_boundaries(zones)
        
        # Precompute edge analytical properties across all combined boundaries
        self.boundary_properties = self._get_combined_boundary_properties()
        self.total_edges = self.boundary_properties[0].shape[1]
        
        self.zeros = np.zeros(self.n_wt)

    def _calc_resulting_boundaries(self, zones):
        inc_poly = [Polygon(z[0]) for z in zones if z[1] == 1]
        exc_poly = [Polygon(z[0]) for z in zones if z[1] == 0]
        
        union_inc = unary_union(inc_poly)
        union_exc = unary_union(exc_poly)
        remain = union_inc.difference(union_exc)
        
        # ------------------------------------------------=====================
        # GEOMETRY FILTERING FOR HYBRID GEOMETRY COLLECTIONS
        # ------------------------------------------------=====================
        # Unpacks GeometryCollections and filters strictly for valid Polygon objects,
        # completely discarding parasitic LineStrings or Points left by the difference operation.
        if isinstance(remain, Polygon):
            polygons = [remain]
        elif hasattr(remain, 'geoms'):
            polygons = [g for g in remain.geoms if isinstance(g, Polygon)]
        else:
            polygons = []
        # ------------------------------------------------=====================
        
        bounds_list = []
        for poly in polygons:
            bounds_list.append((np.array(poly.exterior.coords)[:-1, :], 1))
            for interior in poly.interiors:
                bounds_list.append((np.array(interior.coords)[:-1, :], 0))
        return bounds_list

    def _get_boundary_properties(self, vertices, inclusion_zone=True):
        if np.any(vertices[0] != vertices[-1]):
            vertices = np.vstack([vertices, vertices[:1]])
        A = vertices[:-1].T
        B = vertices[1:].T
        
        # Enforce counter-clockwise (CCW) orientation for inclusion and clockwise (CW) for exclusion
        double_area = np.sum((A[0] - B[0]) * (A[1] + B[1]))
        if (inclusion_zone and double_area < 0) or (not inclusion_zone and double_area > 0):
            return self._get_boundary_properties(vertices[::-1], inclusion_zone)
            
        dx, dy = AB = B - A
        AB_len = np.linalg.norm(AB, axis=0)
        edge_normal = np.array([-dy, dx]) / AB_len
        A_normal = (edge_normal + np.roll(edge_normal, 1, axis=1)) / 2.0
        B_normal = np.roll(A_normal, -1, axis=1)
        return A, B, AB, AB_len, edge_normal, A_normal, B_normal

    def _get_combined_boundary_properties(self):
        props = [self._get_boundary_properties(b[0], b[1] == 1) for b in self.boundaries]
        return [np.concatenate(v, axis=-1) for v in zip(*props)]

    def _calc_distance_and_gradients(self, x, y):
        A, B, AB, AB_len, edge_normal, A_normal, B_normal = self.boundary_properties
        P = np.array([x, y])[:, :, np.newaxis]
        
        A_n, B_n, AB_n = A[:, np.newaxis, :], B[:, np.newaxis, :], AB[:, np.newaxis, :]
        edge_normal_n = edge_normal[:, np.newaxis, :]
        A_normal_n, B_normal_n = A_normal[:, np.newaxis, :], B_normal[:, np.newaxis, :]
        AB_len_n = AB_len[np.newaxis, :]
        
        AP = P - A_n
        BP = P - B_n
        a_tilde = np.sum(AP * AB_n, axis=0) / AB_len_n
        
        use_A = 0 > a_tilde
        use_B = a_tilde > AB_len_n
        
        distance = np.sum(AP * edge_normal_n, axis=0)
        
        # Geometry correction for layout corners near vertices A or B
        dist_A = np.linalg.norm(AP, axis=0)
        sign_A = np.where(np.sum(AP * A_normal_n, axis=0) > 0, 1, -1)
        distance = np.where(use_A, dist_A * sign_A, distance)
        
        dist_B = np.linalg.norm(BP, axis=0)
        sign_B = np.where(np.sum(BP * B_normal_n, axis=0) > 0, 1, -1)
        distance = np.where(use_B, dist_B * sign_B, distance)
        
        # Compute exact analytical partial derivatives
        ddist_dxy = np.tile(edge_normal_n, (1, len(x), 1))
        eps = 1e-7
        ddist_dxy[:, use_A] = sign_A[np.newaxis, use_A] * (AP[:, use_A] / (dist_A[use_A] + eps))
        ddist_dxy[:, use_B] = sign_B[np.newaxis, use_B] * (BP[:, use_B] / (dist_B[use_B] + eps))
        
        return distance, ddist_dxy[0], ddist_dxy[1]

    def setup(self):
        self.add_input('x', np.zeros(self.n_wt), units=self.units)
        self.add_input('y', np.zeros(self.n_wt), units=self.units)
        self.add_output('boundaryDistances', np.zeros(self.n_wt))
        self.declare_partials('boundaryDistances', ['x', 'y'])

    def compute(self, inputs, outputs):
        x, y = inputs['x'], inputs['y']
        distance, dx, dy = self._calc_distance_and_gradients(x, y)
        closest_edge = np.argmin(np.abs(distance), axis=1)
        row_idx = np.arange(len(closest_edge))
        
        outputs['boundaryDistances'] = distance[row_idx, closest_edge]
        self._cache_output = (dx[row_idx, closest_edge], dy[row_idx, closest_edge])

    def compute_partials(self, inputs, partials):
        dx, dy = self._cache_output
        partials['boundaryDistances', 'x'] = np.diag(dx)
        partials['boundaryDistances', 'y'] = np.diag(dy)


# =============================================================================
# 2. INTER-TURBINE SPACING CONSTRAINT COMPONENT (WITH FIXED NEIGHBORS)
# =============================================================================

class SpacingCompWithNeighbors(om.ExplicitComponent):
    """
    OpenMDAO component for minimum inter-turbine spacing constraints.
    Calculates the squared separation distance matrix between active turbines and against static neighbor coordinates.
    """
    def __init__(self, n_wt, min_spacing, add_wt_x=None, add_wt_y=None, units=None):
        super().__init__()
        self.n_wt = n_wt
        self.min_spacing = min_spacing
        self.add_wt_x = np.asarray(add_wt_x) if add_wt_x is not None else np.array([])
        self.add_wt_y = np.asarray(add_wt_y) if add_wt_y is not None else np.array([])
        self.n_add_wt = len(self.add_wt_x)
        self.units = units
        
        # Expected flattened vector length: active internal pairs + boundary interactions against static neighbors
        self.veclen = n_wt * (n_wt - 1) // 2 + n_wt * self.n_add_wt

    def setup(self):
        self.add_input('x', np.zeros(self.n_wt), units=self.units)
        self.add_input('y', np.zeros(self.n_wt), units=self.units)
        self.add_output('wtSeparationSquared', np.zeros(self.veclen))
        self.declare_partials('wtSeparationSquared', ['x', 'y'])

    def compute(self, inputs, outputs):
        x, y = inputs['x'], inputs['y']
        
        # 1. Internal distances between optimized turbine sets
        dX = x[:, np.newaxis] - x[np.newaxis, :]
        dY = y[:, np.newaxis] - y[np.newaxis, :]
        dXY2 = dX**2 + dY**2
        dist_internal = dXY2[np.triu_indices(self.n_wt, 1)]
        
        # 2. External distances separating active turbines from fixed background assets
        if self.n_add_wt > 0:
            dx_add = x[:, np.newaxis] - self.add_wt_x[np.newaxis, :]
            dy_add = y[:, np.newaxis] - self.add_wt_y[np.newaxis, :]
            dist_add = (dx_add**2 + dy_add**2).flatten()
            outputs['wtSeparationSquared'] = np.concatenate([dist_internal, dist_add])
        else:
            outputs['wtSeparationSquared'] = dist_internal

    def compute_partials(self, inputs, J):
        x, y = inputs['x'], inputs['y']
        n_dd = self.n_wt * (self.n_wt - 1) // 2
        
        J_dd_x = np.zeros((n_dd, self.n_wt))
        J_dd_y = np.zeros((n_dd, self.n_wt))
        
        if self.n_wt > 1:
            col_pairs = np.array([(i, j) for i in range(self.n_wt - 1) for j in range(i + 1, self.n_wt)])
            dx = x[col_pairs[:, 0]] - x[col_pairs[:, 1]]
            dy = y[col_pairs[:, 0]] - y[col_pairs[:, 1]]
            
            row_idx = np.arange(n_dd)
            J_dd_x[row_idx, col_pairs[:, 0]] = 2 * dx
            J_dd_x[row_idx, col_pairs[:, 1]] = -2 * dx
            J_dd_y[row_idx, col_pairs[:, 0]] = 2 * dy
            J_dd_y[row_idx, col_pairs[:, 1]] = -2 * dy
            
        if self.n_add_wt > 0:
            n_da = self.n_wt * self.n_add_wt
            J_da_x = np.zeros((n_da, self.n_wt))
            J_da_y = np.zeros((n_da, self.n_wt))
            
            dx_da = x[:, np.newaxis] - self.add_wt_x[np.newaxis, :]
            dy_da = y[:, np.newaxis] - self.add_wt_y[np.newaxis, :]
            
            row_indices = np.arange(n_da)
            col_indices = np.repeat(np.arange(self.n_wt), self.n_add_wt)
            
            J_da_x[row_indices, col_indices] = (2 * dx_da).flatten()
            J_da_y[row_indices, col_indices] = (2 * dy_da).flatten()
            
            J['wtSeparationSquared', 'x'] = np.vstack([J_dd_x, J_da_x])
            J['wtSeparationSquared', 'y'] = np.vstack([J_dd_y, J_da_y])
        else:
            J['wtSeparationSquared', 'x'] = J_dd_x
            J['wtSeparationSquared', 'y'] = J_dd_y


# =============================================================================
# 3. PENALTY AGGREGATION COMPONENT
# =============================================================================

class ConstraintAggregationComp(om.ExplicitComponent):
    """
    Pure OpenMDAO scalar reduction block mapping multi-disciplinary spatial violations into a single fitness penalty.
    Boundary terms: Aggregate sum of squared distances found out-of-bounds.
    Spacing terms: Aggregate absolute error where structural layout violates proximity limits.
    """
    def __init__(self, veclen, n_wt, min_spacing):
        super().__init__()
        self.veclen = veclen
        self.n_wt = n_wt
        self.min_spacing2 = min_spacing**2

    def setup(self):
        self.add_input('wtSeparationSquared', np.zeros(self.veclen))
        self.add_input('boundaryDistances', np.zeros(self.n_wt))
        self.add_output('final_constraint', 0.0)
        self.declare_partials('final_constraint', ['wtSeparationSquared', 'boundaryDistances'])

    def compute(self, inputs, outputs):
        sep = inputs['wtSeparationSquared']
        bound = inputs['boundaryDistances']
        
        # Accumulate strict proximity layout faults
        sep_error = sep - self.min_spacing2
        spacing_penalty = np.sum(-sep_error[sep_error < 0])
        
        # Accumulate geographic perimeter exit violations
        boundary_penalty = np.sum(bound[bound < 0]**2)
        
        outputs['final_constraint'] = spacing_penalty + boundary_penalty

    def compute_partials(self, inputs, J):
        sep = inputs['wtSeparationSquared']
        bound = inputs['boundaryDistances']
        
        # Direct piece-wise gradient evaluation for proximity
        d_sep = np.zeros_like(sep)
        sep_error = sep - self.min_spacing2
        d_sep[sep_error < 0] = -1.0
        
        # Direct piece-wise gradient evaluation for perimeter
        d_bound = np.zeros_like(bound)
        d_bound[bound < 0] = 2 * bound[bound < 0]
        
        J['final_constraint', 'wtSeparationSquared'] = d_sep.reshape(1, -1)
        J['final_constraint', 'boundaryDistances'] = d_bound.reshape(1, -1)