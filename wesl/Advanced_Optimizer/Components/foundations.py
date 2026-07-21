import numpy as np
import openmdao.api as om
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate._interpnd import LinearNDInterpolator

class MonopileCurveFitter:
    @staticmethod
    def fit_surrogate(rd, min_depth=15.0, max_depth=50.0):
        depths = np.linspace(min_depth, max_depth, num=20)
        masses = [600000 + 22000 * (z ** 1.6) + 3.0 * (rd ** 2) for z in depths] # structural formula
        return np.polyfit(depths, np.array(masses), 2) # returns coefficients array

class FoundationsComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt', types=int)
        self.options.declare('bathy_nc_path', types=str)
        self.options.declare('poly_coefficients', types=np.ndarray)
        self.options.declare('steel_price_factor', default=3.636 * 0.924)

    def setup(self):
        n_wt = self.options['n_wt']
        self.add_input('x', val=np.zeros(n_wt), units='m')
        self.add_input('y', val=np.zeros(n_wt), units='m')
        self.add_output('cost_foundations', val=np.zeros(n_wt))
        
        with xr.open_dataset(self.options['bathy_nc_path']) as ds:
            x_flat = ds['x'].values
            y_flat = ds['y'].values
            depth_flat = ds['depth'].values
        
        self.grid_x = np.unique(x_flat)
        self.grid_y = np.unique(y_flat)
        
        expected_size = len(self.grid_x) * len(self.grid_y)
        if expected_size == len(depth_flat):
            self.matrix_depth = depth_flat.reshape(len(self.grid_x), len(self.grid_y))
            self.depth_interpolator = RegularGridInterpolator(
                (self.grid_x, self.grid_y), self.matrix_depth, 
                method='linear', bounds_error=False, fill_value=0.0
            )
            self.use_nd_fallback = False
        else:
            self.depth_interpolator = LinearNDInterpolator(np.column_stack((x_flat, y_flat)), depth_flat)
            self.use_nd_fallback = True

        row_cols = np.arange(n_wt)
        self.declare_partials('cost_foundations', 'x', rows=row_cols, cols=row_cols)
        self.declare_partials('cost_foundations', 'y', rows=row_cols, cols=row_cols)

    def compute(self, inputs, outputs):
        poly_coef = self.options['poly_coefficients']
        steel_factor = self.options['steel_price_factor']
        
        x_real = inputs['x'] # pure real coordinates from OpenMDAO
        y_real = inputs['y']
        
        if self.use_nd_fallback:
            self.depths_computed = self.depth_interpolator(x_real, y_real)
        else:
            self.depths_computed = self.depth_interpolator(np.column_stack((x_real, y_real)))
            
        self.depths_computed = np.maximum(0.0, self.depths_computed)
        masses = poly_coef[0] * (self.depths_computed ** 2) + poly_coef[1] * self.depths_computed + poly_coef[2]
        outputs['cost_foundations'] = masses * steel_factor

    def compute_partials(self, inputs, J):
        poly_coef = self.options['poly_coefficients']
        steel_factor = self.options['steel_price_factor']
        x_real = inputs['x']
        y_real = inputs['y']
        
        n_wt = len(x_real)
        eps = 1.0
        dC_dz = 2.0 * poly_coef[0] * self.depths_computed + poly_coef[1] # structural derivative
        
        ddepth_dx = np.zeros(n_wt)
        ddepth_dy = np.zeros(n_wt)
        
        for i in range(n_wt):
            if self.use_nd_fallback:
                z_x_plus  = self.depth_interpolator(x_real[i] + eps, y_real[i])
                z_x_minus = self.depth_interpolator(x_real[i] - eps, y_real[i])
                z_y_plus  = self.depth_interpolator(x_real[i], y_real[i] + eps)
                z_y_minus = self.depth_interpolator(x_real[i], y_real[i] - eps)
            else:
                z_x_plus  = self.depth_interpolator(np.array([[x_real[i] + eps, y_real[i]]]))[0]
                z_x_minus = self.depth_interpolator(np.array([[x_real[i] - eps, y_real[i]]]))[0]
                z_y_plus  = self.depth_interpolator(np.array([[x_real[i], y_real[i] + eps]]))[0]
                z_y_minus = self.depth_interpolator(np.array([[x_real[i], y_real[i] - eps]]))[0]
                
            ddepth_dx[i] = (z_x_plus - z_x_minus) / (2.0 * eps) # spatial grad X
            ddepth_dy[i] = (z_y_plus - z_y_minus) / (2.0 * eps) # spatial grad Y
            
        J['cost_foundations', 'x'] = steel_factor * dC_dz * ddepth_dx # chain rule X
        J['cost_foundations', 'y'] = steel_factor * dC_dz * ddepth_dy # chain rule Y