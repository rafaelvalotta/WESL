# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import numpy as np
import openmdao.api as om
from openmdao.core.driver import Driver
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError


class SGD(Driver):
    """
    Custom Stochastic Gradient Descent (SGD) with Adam Optimizer.
    Tailored for multi-disciplinary wind farm layout optimization under 
    stochastic metocean mini-batches, enforcing exact DTU hyperparameter decay.
    """

    def __init__(self, maxiter=100, on_iteration=None, **kwargs):
        super().__init__(**kwargs)
        self.maxiter = maxiter
        # Optional hook: callable(iteration, model, obj_val, tag=None), invoked once per
        # accepted iteration (and once more at the final, backoff-verified design point).
        # Kept generic on purpose -- the driver knows nothing about promoted variable names
        # like 'turbine_x'/'turbine_y', so callers read whatever they need off model/problem.
        self.on_iteration = on_iteration

        self.supports['optimization'] = True
        self.supports['multiple_objectives'] = False
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['integer_design_vars'] = False
        self.supports['simultaneous_derivatives'] = True

        self._desvar_idx = {}
        self.iter_count = 0        
        self.driver_iter_count = 0 
        self.is_converged = False
        self.fail = False

    def _declare_options(self):
        self.options.declare('disp', default=True, desc='Display iteration progress')
        self.options.declare('learning_rate', default=1e-1, desc='Initial learning rate (eta_0)')
        self.options.declare('lower', default=1e-6, desc='Lower bound for step decay delta parameter search')
        self.options.declare('upper', default=1e-1, desc='Upper bound for step decay delta parameter search')
        self.options.declare('gamma_min', default=1e-3, desc='Target final learning rate factor at maxiter')
        self.options.declare('beta1', default=0.9, desc='Adam momentum first moment decay')
        self.options.declare('beta2', default=0.999, desc='Adam momentum second moment decay')
        self.options.declare('tol', default=1e-5, desc='Convergence tolerance on gradient infinity norm')

    def _setup_driver(self, problem):
        super()._setup_driver(problem)
        if len(self._objs) != 1:
            raise RuntimeError("SGD/Adam driver supports exactly 1 scalar objective.")

        self.obj_name = list(self._objs.keys())[0]
        self.penalty_name = 'agg_constraint'  # must match ConstraintAggregationComp's promoted output
        self.output_list = [self.obj_name, self.penalty_name]

        # Flattening index maps for array conversion of multiple design vars (x and y arrays)
        count = 0
        for name, meta in self._designvars.items():
            size = meta['size']
            self._desvar_idx[name] = (count, count + size)
            count += size

        if count == 0:
            raise RuntimeError("MDO Problem has no active turbine design variables registered.")

    def _pack_desvars(self, desvar_vals):
        n = sum(meta['size'] for meta in self._designvars.values())
        x = np.empty(n, dtype=float)
        for name in self._designvars:
            i, j = self._desvar_idx[name]
            x[i:j] = desvar_vals[name]
        return x

    def _unpack_and_set_desvars(self, x):
        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

    def run(self):
        model = self._problem().model
        dv_names = list(self._designvars.keys())

        desvar_vals = self.get_design_var_values()
        x = self._pack_desvars(desvar_vals)

        # Initial baseline forward evaluation to capture raw scale of physical outputs
        model._solve_nonlinear()
        jac = self._compute_totals(of=self.output_list, wrt=dv_names, return_format='array')
        
        j0 = jac[0]  # Gradient of objective (LCOE)
        
        # Scaling alpha_0 based on the initial mean magnitude of objective gradients
        learning_rate = float(self.options['learning_rate'])
        self.alpha0 = np.mean(np.abs(j0)) / learning_rate
        self.l0 = float(learning_rate)
        alpha = float(self.alpha0)

        # Exact bisection search routine to compute the step decay delta parameter
        lower = self.options['lower']
        upper = self.options['upper']
        gamma_min = self.options['gamma_min']

        def compute_decay_factor(iterations, delta_val):
            prod = 1.0
            for k in range(iterations):
                prod *= 1.0 / (1.0 + delta_val * k)
            return prod

        for _ in range(200):
            mid = np.mean([lower, upper])
            gamma_trial = compute_decay_factor(self.maxiter, mid)
            if gamma_trial < gamma_min:
                upper = mid
            else:
                lower = mid
        self.mid_delta = mid

        m = np.zeros_like(x)
        v = np.zeros_like(x)
        adam_step = 0

        self.is_converged = False
        self.iter_count = 0
        self.driver_iter_count = 0
        self.fail = False

        tol = self.options['tol']
        grad_norm = np.inf
        learning_rate_curr = learning_rate
        alpha_curr = alpha

        # Tracks the last design point that solved successfully, so a raw Adam step that
        # pushes a turbine outside a hard geometric requirement (e.g. outside the farm
        # boundary polygon, which is not the same as the box bounds clipped below) can be
        # backed off toward it instead of crashing the whole run.
        x_last_valid = x.copy()
        while self.driver_iter_count < self.maxiter and not self.is_converged:

            with RecordingDebugging('StochasticSGDAdam', self.iter_count, self) as rec:
                self.iter_count += 1

                # Forward physical coupled evaluation (Mini-batch happens transparently inside
                # AEPComp). Back off geometrically toward the last valid point on failure.
                backoff = 1.0
                for attempt in range(10):
                    x_try = x_last_valid + backoff * (x - x_last_valid)
                    for name in self._designvars:
                        lo, hi = self._desvar_idx[name]
                        self.set_design_var(name, x_try[lo:hi])
                    try:
                        model._solve_nonlinear()
                        break
                    except (AnalysisError, ValueError):
                        backoff *= 0.5
                else:
                    x_try = x_last_valid.copy()
                    for name in self._designvars:
                        lo, hi = self._desvar_idx[name]
                        self.set_design_var(name, x_try[lo:hi])
                    model._solve_nonlinear()

                x = x_try
                x_last_valid = x.copy()

                # Backward analytic adjoint vector assembly
                jac = self._compute_totals(of=self.output_list, wrt=dv_names, return_format='array')
                j = jac[0]  # d_LCOE / d_xy
                c = jac[1]  # d_penalty / d_xy

                # Formulation of the Lagrangian Search Gradient: -d(Obj)/dx + alpha * d(Penalty)/dx
                # Minimizing objective while ascending multiplier against constraint violations
                lagrangian_gradient = j + alpha_curr * c
                grad_norm = np.linalg.norm(lagrangian_gradient, np.inf)

                # Adam moment updates
                beta1 = self.options['beta1']
                beta2 = self.options['beta2']
                adam_step += 1
                
                m = beta1 * m + (1.0 - beta1) * lagrangian_gradient
                v = beta2 * v + (1.0 - beta2) * (lagrangian_gradient ** 2)

                mhat = m / (1.0 - beta1 ** adam_step)
                vhat = v / (1.0 - beta2 ** adam_step)

                # Displace turbine coordinate parameters using the decoupled momentum vectors
                x -= learning_rate_curr * mhat / (np.sqrt(vhat) + 1e-12)

                # Enforce the box bounds declared via add_design_var(lower=, upper=). Unlike
                # ScipyOptimizeDriver, this Adam step is unconstrained by construction, so
                # without clipping a turbine can drift outside its allowed range and crash
                # downstream components (e.g. cable routing) that assume in-bounds coordinates.
                for name in self._designvars:
                    i, j = self._desvar_idx[name]
                    lower = self._designvars[name]['lower']
                    upper = self._designvars[name]['upper']
                    x[i:j] = np.clip(x[i:j], lower, upper)

                # Synchronized Step and Penalty Decay update rule
                decay_multiplier = compute_decay_factor(self.iter_count, self.mid_delta)
                learning_rate_curr = learning_rate * decay_multiplier
                alpha_curr = self.alpha0 * self.l0 / learning_rate_curr

                if np.any(np.isnan(x)):
                    raise FloatingPointError("Instability encountered: NaN detected in spatial coordinates array.")

                rec.abs = 0.0
                rec.rel = 0.0

            self.driver_iter_count += 1
            
            # Best-effort iteration counter broadcast; silently skipped if the model has no 'opt_iter' input
            try:
                self._problem().set_val('opt_iter', float(self.driver_iter_count))
            except (ValueError, KeyError, AttributeError):
                pass

            need_obj_val = self.on_iteration is not None or (
                self.options['disp'] and (self.driver_iter_count % 5 == 0 or self.driver_iter_count == 1))
            if need_obj_val:
                # driver_scaling=False: report the true physical objective value, not the
                # scaler-flipped one (e.g. add_objective('aep', scaler=-1) internally
                # minimizes -aep, which would otherwise print as a large negative number).
                obj_vals = self.get_objective_values(driver_scaling=False)
                obj_val = float(np.atleast_1d(obj_vals[self.obj_name])[0])

            if self.options['disp'] and (self.driver_iter_count % 5 == 0 or self.driver_iter_count == 1):
                print(f"Adam Iter: {self.driver_iter_count:3d} | {self.obj_name}: {obj_val:.4f} | Multiplier (Alpha): {alpha_curr:.2e} | |G|inf: {grad_norm:.4e}")

            if self.on_iteration is not None:
                self.on_iteration(self.driver_iter_count, model, obj_val)

            if grad_norm < tol:
                self.is_converged = True

        # Re-evaluate the model at the final design point: the last loop pass computed
        # outputs for the PREVIOUS x (used to derive the gradient step), so without this
        # call prob.get_val() after run_driver() would return stale aep/lcoe/agg_constraint
        # values from the second-to-last iterate instead of the actual final layout. The
        # trailing x was never itself forward-solved (only x_last_valid was verified inside
        # the loop), so apply the same backoff safeguard here before trusting it.
        backoff = 1.0
        for attempt in range(10):
            x_try = x_last_valid + backoff * (x - x_last_valid)
            self._unpack_and_set_desvars(x_try)
            try:
                model._solve_nonlinear()
                x = x_try
                break
            except (AnalysisError, ValueError):
                backoff *= 0.5
        else:
            x = x_last_valid.copy()
            self._unpack_and_set_desvars(x)
            model._solve_nonlinear()

        if self.on_iteration is not None:
            obj_vals = self.get_objective_values(driver_scaling=False)
            obj_val = float(np.atleast_1d(obj_vals[self.obj_name])[0])
            self.on_iteration(self.driver_iter_count, model, obj_val, tag='final')

        if self.options['disp']:
            print("\n" + "="*50 + "\nSTOCHASTIC OPTIMIZATION CYCLE CONCLUDED.\n" + "="*50)
            print(f"Total Iterations: {self.driver_iter_count}")
            print(f"Final Gradient Norm: {grad_norm:.6e}")
            
        return self.fail

# import numpy as np
# import openmdao.api as om
# from openmdao.core.driver import Driver, RecordingDebugging
# from openmdao.core.analysis_error import AnalysisError

# class NCG(Driver):
#     """The search direction is built with nonlinear CG using PRP+ and optional
#     restart to steepest descent when the direction is not descending.
#     Adam is then applied on the CG direction as an adaptive per-coordinate
#     normalizer. A quadratic penalty gradient is combined with the objective 
#     gradient through a time-varying penalty factor alpha."""

#     def __init__(self, maxiter=200, **kwargs):
#         self.maxiter = maxiter
#         super().__init__(**kwargs)

#         self.supports['optimization'] = True
#         self.supports['multiple_objectives'] = False
#         self.supports['inequality_constraints'] = False
#         self.supports['equality_constraints'] = False
#         self.supports['integer_design_vars'] = False
#         self.supports['two_sided_constraints'] = False
#         self.supports['linear_constraints'] = False
#         self.supports['simultaneous_derivatives'] = True
#         self.supports['active_set'] = False

#         self._desvar_idx = {}
#         self.iter_count = 0
#         self.driver_iter_count = 0
#         self.is_converged = False
#         self.fail = False

#     def _declare_options(self):
#         self.options.declare('disp', default=True, desc='Display iteration progress')
#         self.options.declare('learning_rate', default=1e-2, desc='Initial learning rate')
#         self.options.declare('lower', default=1e-6, desc='Lower bound for delta search')
#         self.options.declare('upper', default=1e-1, desc='Upper bound for delta search')
#         self.options.declare('gamma_min', default=1e-3, desc='Target final learning rate')
#         self.options.declare('beta1', default=0.1, desc='Adam beta1')
#         self.options.declare('beta2', default=0.2, desc='Adam beta2')
#         self.options.declare('tol', default=1e-6, desc='Infinity norm tolerance on combined gradient')
#         self.options.declare('cg_formula', default='PRP', desc='Nonlinear CG beta formula')
#         self.options.declare('restart_period', default=None, desc='Optional periodic restart interval')
#         self.options.declare('eps', default=1e-12, desc='Numerical safeguard')
#         self.options.declare('reset_moments', default=False, desc='If True, reset Adam moments periodically')
#         self.options.declare('reset_period', default=100, desc='Period for resetting Adam moments')

#     def _setup_driver(self, problem):
#         super()._setup_driver(problem)
#         if len(self._objs) != 1:
#             raise RuntimeError("NCG supports exactly 1 scalar objective.")

#         self.obj_list = list(self._objs)
#         self.penalty_name = 'penalty'
#         self.obj_and_pen_list = self.obj_list + [self.penalty_name]

#         count = 0
#         for name, meta in self._designvars.items():
#             size = meta['size']
#             self._desvar_idx[name] = (count, count + size)
#             count += size

#         if count == 0:
#             raise RuntimeError("Problem has no design variables.")

#     def _pack_desvars(self, desvar_vals):
#         n = sum(meta['size'] for meta in self._designvars.values())
#         x = np.empty(n, dtype=float)
#         for name in self._designvars:
#             i, j = self._desvar_idx[name]
#             x[i:j] = desvar_vals[name]
#         return x

#     def _unpack_and_set_desvars(self, x):
#         for name in self._designvars:
#             i, j = self._desvar_idx[name]
#             self.set_design_var(name, x[i:j])

#     def _get_final_result(self):
#         obj_vals = self.get_objective_values()
#         obj_key = list(obj_vals.keys())[0]
#         obj_val = float(np.atleast_1d(obj_vals[obj_key])[0])
        
#         try:
#             penalty_norm = self.get_response_values('penalty')
#             cv_norm = float(np.atleast_1d(list(penalty_norm.values())[0])[0])
#         except Exception:
#             cv_norm = 0.0

#         return {
#             'Final Objective (LCOE)': f"[{obj_val:.6f}]",
#             'Constraint violation norm': f"[{cv_norm:.6f}]"
#         }

#     def _cg_beta(self, g, g_prev, d_prev):
#         formula = self.options['cg_formula']
#         eps = self.options['eps']
#         y = g - g_prev

#         if formula == 'PRP':
#             denom = np.dot(g_prev, g_prev)
#             beta = np.dot(g, y) / denom if denom > eps else 0.0
#             beta = max(beta, 0.0)
#         elif formula == 'FR':
#             denom = np.dot(g_prev, g_prev)
#             beta = np.dot(g, g) / denom if denom > eps else 0.0
#         elif formula == 'HS':
#             denom = np.dot(d_prev, y)
#             beta = np.dot(g, y) / denom if abs(denom) > eps else 0.0
#         elif formula == 'DY':
#             denom = np.dot(d_prev, y)
#             beta = np.dot(g, g) / denom if abs(denom) > eps else 0.0
#         else:
#             beta = 0.0

#         if not np.isfinite(beta):
#             beta = 0.0
#         return beta

#     def _compute_direction(self, g, g_prev, d_prev, k):
#         if k == 0 or g_prev is None or d_prev is None:
#             return -g

#         beta = self._cg_beta(g, g_prev, d_prev)
#         d = -g + beta * d_prev

#         restart_period = self.options['restart_period']
#         if restart_period is not None and restart_period > 0:
#             if k % restart_period == 0:
#                 d = -g

#         if np.dot(g, d) >= 0.0:
#             d = -g
#         return d

#     def run(self):
#         model = self._problem().model
#         dv_names = list(self._designvars.keys())

#         desvar_vals = self.get_design_var_values()
#         x0 = self._pack_desvars(desvar_vals)
#         x = x0.copy()

#         model._solve_nonlinear()
#         jac = self._compute_totals(of=self.obj_and_pen_list, wrt=dv_names, return_format='array')
#         j0 = jac[0]

#         learning_rate = float(self.options['learning_rate'])
#         self.learning_rate = learning_rate
#         self.alpha0 = np.mean(np.abs(j0)) / learning_rate
#         self.l0 = float(learning_rate)

#         self.lower = self.options['lower']
#         self.upper = self.options['upper']
#         self.gamma_min = self.options['gamma_min']

#         def multf(t, delta):
#             prod = 1.0
#             for ii in range(t):
#                 prod *= 1.0 / (1.0 + delta * ii)
#             return learning_rate * prod

#         for _ in range(self.maxiter):
#             mid = np.mean([self.lower, self.upper])
#             etaM = multf(self.maxiter, mid)
#             if etaM < self.gamma_min:
#                 self.upper = mid
#             elif etaM > self.gamma_min:
#                 self.lower = mid
#             self.mid = mid

#         m = np.zeros_like(x0)
#         v = np.zeros_like(x0)
#         g_prev = None
#         d_prev = None

#         self.is_converged = False
#         self.iter_count = 0
#         self.driver_iter_count = 0
#         self.fail = False

#         t = 0
#         tol = self.options['tol']
#         learning_rate_current = learning_rate
#         alpha_current = float(self.alpha0)
#         grad_norm = np.inf

#         while t < self.maxiter and not self.is_converged:
#             obj, x, alpha_current, learning_rate_current, m, v, g, d, grad_norm, success = \
#                 self.objective_callback(
#                     x, alpha_current, learning_rate_current,
#                     m, v, g_prev, d_prev, t,
#                     record=True, update=True
#                 )

#             t += 1
#             self.driver_iter_count = t
#             self._problem().model.set_val('opt_iter', float(self.driver_iter_count))    

#             if self.options['reset_moments'] and t > 0 and (t % self.options['reset_period'] == 0):
#                 m[:] = 0.0
#                 v[:] = 0.0

#             if grad_norm < tol:
#                 self.is_converged = True

#             if success == 0:
#                 self.fail = True

#             g_prev = g.copy() if g is not None else None
#             d_prev = d.copy() if d is not None else None

#         self._unpack_and_set_desvars(x)

#         if self.options['disp']:
#             if self.fail and t >= self.maxiter:
#                 print("OptimizeWarning: Maximum number of iterations exceeded. Optimization FAILED.")
#             else:
#                 print("Optimization SUCCESSFUL.")
#                 print(f"Iterations: {t}")
#                 print(f"Gradient infinity norm: {grad_norm:.6e}")
#                 print("\n--- Optimization result ---")
#                 result = self._get_final_result()
#                 for key, val in result.items():
#                     print(f"{key}: {val}")

#         return self.fail

#     def objective_callback(self, x, alpha, learning_rate, m, v, g_prev, d_prev, k, record=False, update=True):
#         model = self._problem().model
#         dv_names = list(self._designvars.keys())
#         success = 1
#         eps = self.options['eps']

#         for name in self._designvars:
#             i, j = self._desvar_idx[name]
#             self.set_design_var(name, x[i:j])

#         with RecordingDebugging('NCG', self.iter_count, self) as rec:
#             if update:
#                 self.iter_count += 1

#             try:
#                 model._solve_nonlinear()
#             except AnalysisError:
#                 model._clear_iprint()
#                 success = 0

#             obj_vals = self.get_objective_values()
#             obj_key = list(obj_vals.keys())[0]
#             obj = obj_vals[obj_key]

#             jac = self._compute_totals(of=self.obj_and_pen_list, wrt=dv_names, return_format='array')
#             j = jac[0]
#             c = jac[1]

#             g = -j + alpha * c
#             d = self._compute_direction(g, g_prev, d_prev, k)
#             grad_norm = np.linalg.norm(np.ravel(g), np.inf)

#             beta1 = self.options['beta1']
#             beta2 = self.options['beta2']

#             m = beta1 * m + (1.0 - beta1) * d
#             v = beta2 * v + (1.0 - beta2) * (d ** 2)

#             adam_step = self.iter_count
#             mhat = m / (1.0 - beta1 ** adam_step) if adam_step > 0 else m
#             vhat = v / (1.0 - beta2 ** adam_step) if adam_step > 0 else v

#             x += learning_rate * mhat / np.sqrt(vhat + eps)
#             learning_rate *= 1.0 / (1.0 + self.mid * self.iter_count)
#             alpha = self.alpha0 * self.l0 / learning_rate

#             if np.any(np.isnan(x)) or np.any(np.isinf(x)):
#                 raise RuntimeError("Invalid design variables detected")

#             rec.abs = 0.0
#             rec.rel = 0.0

#         return obj, x, alpha, learning_rate, m, v, g, d, grad_norm, success

# class SGD(Driver):
#     """Simple SGD/Adam driver adapted to a simplified environment outside TopFarm."""

#     def __init__(self, maxiter=200, **kwargs):
#         self.maxiter = maxiter
#         super().__init__(**kwargs)

#         self.supports['optimization'] = True
#         self.supports['multiple_objectives'] = False
#         self.supports['inequality_constraints'] = False
#         self.supports['equality_constraints'] = False
#         self.supports['integer_design_vars'] = False
#         self.supports['two_sided_constraints'] = False
#         self.supports['linear_constraints'] = False
#         self.supports['simultaneous_derivatives'] = True
#         self.supports['active_set'] = False

#         self._desvar_idx = {}
#         self.iter_count = 0        
#         self.driver_iter_count = 0 
#         self.is_converged = False
#         self.fail = False

#     def _declare_options(self):
#         self.options.declare('disp', default=True, desc='Display iteration progress')
#         self.options.declare('learning_rate', default=1e-2, desc='Initial learning rate (eta_0)')
#         self.options.declare('lower', default=1e-6, desc='Lower bound for delta search')
#         self.options.declare('upper', default=1e-1, desc='Upper bound for delta search')
#         self.options.declare('gamma_min', default=1e-3, desc='Target final eta')
#         self.options.declare('beta1', default=0.1, desc='Adam beta1 parameter')
#         self.options.declare('beta2', default=0.2, desc='Adam beta2 parameter')
#         self.options.declare('tol', default=1e-6, desc='Tolerance on gradient infinity norm')
#         self.options.declare('reset_moments', default=False, desc='If True, reset Adam moments periodically')
#         self.options.declare('reset_period', default=100, desc='Period for resetting Adam moments')

#     def _setup_driver(self, problem):
#         super()._setup_driver(problem)
#         if len(self._objs) != 1:
#             raise RuntimeError("SGD supports exactly 1 scalar objective.")

#         self.obj_list = list(self._objs)
#         self.penalty_name = 'penalty'
#         self.obj_and_pen_list = self.obj_list + [self.penalty_name]

#         desvars = self._designvars
#         count = 0
#         for name, meta in desvars.items():
#             size = meta['size']
#             self._desvar_idx[name] = (count, count + size)
#             count += size

#         if count == 0:
#             raise RuntimeError("Problem has no design variables.")

#     def _pack_desvars(self, desvar_vals):
#         n = sum(meta['size'] for meta in self._designvars.values())
#         x = np.empty(n, dtype=float)
#         for name, meta in self._designvars.items():
#             i, j = self._desvar_idx[name]
#             x[i:j] = desvar_vals[name]
#         return x

#     def _unpack_and_set_desvars(self, x):
#         for name in self._designvars:
#             i, j = self._desvar_idx[name]
#             self.set_design_var(name, x[i:j])

#     def _get_final_result(self):
#         obj_vals = self.get_objective_values()
#         obj_key = list(obj_vals.keys())[0]
#         obj_val = float(np.atleast_1d(obj_vals[obj_key])[0])
        
#         try:
#             penalty_norm = self.get_response_values('penalty')
#             cv_norm = float(np.atleast_1d(list(penalty_norm.values())[0])[0])
#         except Exception:
#             cv_norm = 0.0

#         return {
#             'Final Objective': f"[{obj_val:.6f}]",
#             'Constraint violation norm': f"[{cv_norm:.6f}]"
#         }

#     def run(self):
#         model = self._problem().model
#         dv_names = list(self._designvars.keys())

#         desvar_vals = self.get_design_var_values()
#         x0 = self._pack_desvars(desvar_vals)
#         x = x0.copy()

#         model._solve_nonlinear()
#         jac = self._compute_totals(of=self.obj_and_pen_list, wrt=dv_names, return_format='array')
#         j0 = jac[0]  

#         learning_rate = float(self.options['learning_rate'])
#         self.learning_rate = learning_rate
#         self.alpha0 = np.mean(np.abs(j0)) / learning_rate
#         self.l0 = float(learning_rate)
#         alpha = float(self.alpha0)

#         self.lower = self.options['lower']
#         self.upper = self.options['upper']
#         self.gamma_min = self.options['gamma_min']

#         def multf(t, delta):
#             prod = 1.0
#             for ii in range(t):
#                 prod *= 1.0 / (1.0 + delta * ii)
#             return learning_rate * prod

#         for _ in range(self.maxiter):
#             mid = np.mean([self.lower, self.upper])
#             etaM = multf(self.maxiter, mid)
#             if etaM < self.gamma_min:
#                 self.upper = mid
#             elif etaM > self.gamma_min:
#                 self.lower = mid
#             self.mid = mid

#         m = np.zeros_like(x0)
#         v = np.zeros_like(x0)
#         adam_step = 0

#         self.is_converged = False
#         self.iter_count = 0
#         self.driver_iter_count = 0
#         self.fail = False

#         t = 0
#         tol = self.options['tol']
#         grad_norm = np.inf
#         learning_rate_current = learning_rate
#         alpha_current = alpha

#         while t < self.maxiter and not self.is_converged:
#             obj, x, alpha_current, learning_rate_current, m, v, adam_step, grad_norm, success = \
#                 self.objective_callback(x, alpha_current, learning_rate_current, m, v, adam_step, record=True, update=True)

#             t += 1
#             self.driver_iter_count = t

#             self._problem().model.set_val('opt_iter', float(self.driver_iter_count))

#             if self.options['reset_moments'] and t > 0 and (t % self.options['reset_period'] == 0):
#                 m[:] = 0.0
#                 v[:] = 0.0
#                 adam_step = 0

#             if grad_norm < tol:
#                 self.is_converged = True

#             if success == 0:
#                 self.fail = True

#         self._unpack_and_set_desvars(x)

#         if self.options['disp']:
#             if self.fail and t >= self.maxiter:
#                 print("OptimizeWarning: Maximum number of iterations exceeded. Optimization FAILED.")
#             else:
#                 print("Optimization SUCCESSFUL.")
#                 print(f"Iterations: {t}")
#                 print(f"Gradient infinity norm: {grad_norm:.6e}")
#                 print("\n--- Optimization result ---")
#                 result = self._get_final_result()
#                 for key, val in result.items():
#                     print(f"{key}: {val}")

#         return self.fail

#     def objective_callback(self, x, alpha, learning_rate, m, v, adam_step, record=False, update=True):
#         model = self._problem().model
#         dv_names = list(self._designvars.keys())
#         success = 1

#         for name in self._designvars:
#             i, j = self._desvar_idx[name]
#             self.set_design_var(name, x[i:j])

#         with RecordingDebugging('SimpleSGD', self.iter_count, self) as rec:
#             if update:
#                 self.iter_count += 1

#             try:
#                 model._solve_nonlinear()
#             except AnalysisError:
#                 model._clear_iprint()
#                 success = 0

#             obj_vals = self.get_objective_values()
#             obj_key = list(obj_vals.keys())[0]
#             obj = obj_vals[obj_key]

#             jac = self._compute_totals(of=self.obj_and_pen_list, wrt=dv_names, return_format='array')
#             j = jac[0]   
#             c = jac[1]   

#             jacobian = -j + alpha * c
#             grad_norm = np.linalg.norm(jacobian, np.inf)

#             beta1 = self.options['beta1']
#             beta2 = self.options['beta2']

#             adam_step += 1
#             m = beta1 * m + (1 - beta1) * jacobian
#             v = beta2 * v + (1 - beta2) * jacobian ** 2

#             mhat = m / (1.0 - beta1 ** adam_step)
#             vhat = v / (1.0 - beta2 ** adam_step)

#             x -= learning_rate * mhat / np.sqrt(vhat + 1e-12)

#             learning_rate *= 1.0 / (1.0 + self.mid * self.iter_count)
#             alpha = self.alpha0 * self.l0 / learning_rate

#             if np.any(np.isnan(x)):
#                 raise Exception("NaN in design variables detected")

#             rec.abs = 0.0
#             rec.rel = 0.0

#         return obj, x, alpha, learning_rate, m, v, adam_step, grad_norm, success