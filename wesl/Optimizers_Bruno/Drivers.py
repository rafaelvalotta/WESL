# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import numpy as np
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError



class NCG(Driver):
    """The search direction is built with nonlinear CG using PRP+ and optional
    restart to steepest descent when the direction is not descending.
    Adam is then applied on the CG direction as an adaptive per-coordinate
    normalizer, serving as an automated and more efficient way to rescale
    the problem. This contrasts with previous experiments, where manually
    testing multiple scalar values in the problem definition introduced
    significant instability and poor driver performance. A quadratic penalty
    gradient is combined with the objective gradient through a time-varying
    penalty factor alpha."""

    def __init__(self, maxiter=200, **kwargs):
        self.maxiter = maxiter
        super().__init__(**kwargs)

        self.supports['optimization'] = True
        self.supports['multiple_objectives'] = False
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['integer_design_vars'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = True
        self.supports['active_set'] = False

        self._desvar_idx = {}
        self.iter_count = 0
        self.driver_iter_count = 0
        self.is_converged = False
        self.fail = False

    def _declare_options(self):
        self.options.declare('disp', default=True, desc='Display iteration progress')

        self.options.declare('learning_rate', default=1e-2,
                             desc='Initial learning rate')
        self.options.declare('lower', default=1e-6,
                             desc='Lower bound for delta search')
        self.options.declare('upper', default=1e-1,
                             desc='Upper bound for delta search')
        self.options.declare('gamma_min', default=1e-3,
                             desc='Target final learning rate')

        self.options.declare('beta1', default=0.1, desc='Adam beta1')
        self.options.declare('beta2', default=0.2, desc='Adam beta2')

        self.options.declare('tol', default=1e-6,
                             desc='Infinity norm tolerance on combined gradient')

        self.options.declare('cg_formula', default='PRP',
                             desc='Nonlinear CG beta formula')
        self.options.declare('restart_period', default=None,
                             desc='Optional periodic restart interval')
        self.options.declare('eps', default=1e-12,
                             desc='Numerical safeguard')

        self.options.declare('reset_moments', default=False,
                             desc='If True, reset Adam moments periodically')
        self.options.declare('reset_period', default=100,
                             desc='Period for resetting Adam moments')

    def _setup_driver(self, problem):
        super()._setup_driver(problem)

        if len(self._objs) != 1:
            raise RuntimeError("NCG_ADAM supports exactly 1 scalar objective.")

        self.obj_list = list(self._objs)
        self.penalty_name = 'penalty'
        self.obj_and_pen_list = self.obj_list + [self.penalty_name]

        count = 0
        for name, meta in self._designvars.items():
            size = meta['size']
            self._desvar_idx[name] = (count, count + size)
            count += size

        if count == 0:
            raise RuntimeError("Problem has no design variables.")

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

    def _get_final_result(self):
        aep_norm = self.get_objective_values()
        try:
            penalty_norm = self.get_response_values('penalty')
            cv_norm = float(list(penalty_norm.values())[0])
        except Exception:
            cv_norm = 0.0

        obj_pen = list(aep_norm.values())[0]

        return {
            'AEP (normalized)': f"[{float(aep_norm[list(aep_norm.keys())[0]]):.6f}]",
            'Constraint violation norm': f"[{cv_norm:.6f}]",
            'Penalized objective': f"[{float(obj_pen):.6f}]"
        }

    def _cg_beta(self, g, g_prev, d_prev):
        formula = self.options['cg_formula']
        eps = self.options['eps']
        y = g - g_prev

        if formula == 'PRP':
            denom = np.dot(g_prev, g_prev)
            beta = np.dot(g, y) / denom if denom > eps else 0.0
            beta = max(beta, 0.0)
        elif formula == 'FR':
            denom = np.dot(g_prev, g_prev)
            beta = np.dot(g, g) / denom if denom > eps else 0.0
        elif formula == 'HS':
            denom = np.dot(d_prev, y)
            beta = np.dot(g, y) / denom if abs(denom) > eps else 0.0
        elif formula == 'DY':
            denom = np.dot(d_prev, y)
            beta = np.dot(g, g) / denom if abs(denom) > eps else 0.0
        else:
            beta = 0.0

        if not np.isfinite(beta):
            beta = 0.0

        return beta

    def _compute_direction(self, g, g_prev, d_prev, k):
        if k == 0 or g_prev is None or d_prev is None:
            return -g

        beta = self._cg_beta(g, g_prev, d_prev)
        d = -g + beta * d_prev # steepest descent direction + NCG direction
        "Here, if the new NCG information (using this beta) brings significant improvement, the direction may be better."

        restart_period = self.options['restart_period']
        if restart_period is not None and restart_period > 0:
            if k % restart_period == 0:
                d = -g

        if np.dot(g, d) >= 0.0:
            d = -g

        return d

    def run(self):
        model = self._problem().model
        dv_names = list(self._designvars.keys())

        desvar_vals = self.get_design_var_values()
        x0 = self._pack_desvars(desvar_vals)
        x = x0.copy()

        model._solve_nonlinear()
        jac = self._compute_totals(of=self.obj_and_pen_list,
                                   wrt=dv_names,
                                   return_format='array')
        j0 = jac[0]

        learning_rate = float(self.options['learning_rate'])
        self.learning_rate = learning_rate
        self.alpha0 = np.mean(np.abs(j0)) / learning_rate
        self.l0 = float(learning_rate)

        self.lower = self.options['lower']
        self.upper = self.options['upper']
        self.gamma_min = self.options['gamma_min']

        def multf(t, delta):
            prod = 1.0
            for ii in range(t):
                prod *= 1.0 / (1.0 + delta * ii)
            return learning_rate * prod

        for _ in range(self.maxiter):
            mid = np.mean([self.lower, self.upper])
            etaM = multf(self.maxiter, mid)
            if etaM < self.gamma_min:
                self.upper = mid
            elif etaM > self.gamma_min:
                self.lower = mid
            self.mid = mid

        m = np.zeros_like(x0)
        v = np.zeros_like(x0)
        g_prev = None
        d_prev = None

        self.is_converged = False
        self.iter_count = 0
        self.driver_iter_count = 0
        self.fail = False

        t = 0
        tol = self.options['tol']
        fk = None
        learning_rate_current = learning_rate
        alpha_current = float(self.alpha0)

        while t < self.maxiter and not self.is_converged:
            obj, x, alpha_current, learning_rate_current, m, v, g, d, grad_norm, success = \
                self.objective_callback(
                    x, alpha_current, learning_rate_current,
                    m, v, g_prev, d_prev, t,
                    record=True, update=True
                )

            if obj is not None:
                if isinstance(obj, np.ndarray):
                    fk = float(obj.item()) if obj.size == 1 else float(obj[0])
                else:
                    fk = float(obj)

            t += 1
            self.driver_iter_count = t

            # if hasattr(self, 'recorder') and self.recorder is not None:
            #     self._problem().model.set_val('opt_iter', float(self.driver_iter_count))
            self._problem().model.set_val('opt_iter', float(self.driver_iter_count))    

            if self.options['reset_moments'] and t > 0 and (t % self.options['reset_period'] == 0):
                m[:] = 0.0
                v[:] = 0.0
                if self.options['disp']:
                    print(f"Resetting Adam moments at iteration {t}")

            if grad_norm < tol:
                self.is_converged = True

            if success == 0:
                self.fail = True

            g_prev = g.copy() if g is not None else None
            d_prev = d.copy() if d is not None else None

        self._unpack_and_set_desvars(x)

        if self.options['disp']:
            if self.fail and t >= self.maxiter:
                print("OptimizeWarning: Maximum number of iterations exceeded.")
                print(f"Current stochastic function value f_t: {fk:.6f}")
                print(f"Iterations: {t}")
                print(f"Function evaluations: {self.iter_count}")
                print(f"Gradient evaluations: {self.iter_count}")
                print("Optimization FAILED.")
            else:
                print("Optimization SUCCESSFUL.")
                print(f"Current stochastic function value f_t: {fk:.6f}")
                print(f"Iterations: {t}")
                print(f"Function evaluations: {self.iter_count}")
                print(f"Gradient evaluations: {self.iter_count}")
                print(f"Gradient infinity norm: {grad_norm:.6e}")
                print("\n--- Optimization result ---")
                result = self._get_final_result()
                for key, val in result.items():
                    print(f"{key}: {val}")

        if hasattr(self, 'recorder') and self.recorder is not None:
            self.recorder.close()

        return self.fail

    def objective_callback(self, x, alpha, learning_rate, m, v, g_prev, d_prev, k,
                           record=False, update=True):
        model = self._problem().model
        dv_names = list(self._designvars.keys())
        success = 1
        eps = self.options['eps']

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        with RecordingDebugging('NCG', self.iter_count, self) as rec:
            if update:
                self.iter_count += 1

            try:
                model._solve_nonlinear()
                if hasattr(self, 'recorder') and self.recorder is not None:
                    self.recorder.log()
            except AnalysisError:
                model._clear_iprint()
                success = 0

            for _, val in self.get_objective_values().items():
                obj = val
                break

            jac = self._compute_totals(of=self.obj_and_pen_list,
                                       wrt=dv_names,
                                       return_format='array')
            j = jac[0]
            c = jac[1]

            g = -j + alpha * c
            d = self._compute_direction(g, g_prev, d_prev, k)
            grad_norm = np.linalg.norm(np.ravel(g), np.inf)

            beta1 = self.options['beta1']
            beta2 = self.options['beta2']

            m = beta1 * m + (1.0 - beta1) * d
            v = beta2 * v + (1.0 - beta2) * (d ** 2)

            adam_step = self.iter_count
            mhat = m / (1.0 - beta1 ** adam_step) if adam_step > 0 else m
            vhat = v / (1.0 - beta2 ** adam_step) if adam_step > 0 else v

            x += learning_rate * mhat / np.sqrt(vhat + eps)

            learning_rate *= 1.0 / (1.0 + self.mid * self.iter_count)
            alpha = self.alpha0 * self.l0 / learning_rate

            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                raise RuntimeError("Invalid design variables detected")

            rec.abs = 0.0
            rec.rel = 0.0

        return obj, x, alpha, learning_rate, m, v, g, d, grad_norm, success
    

class SGD(Driver):
    """
    Simple SGD/Adam driver implemented to replicate the
    original method described in the DTU paper, but adapted 
    to a simplified environment outside TopFarm, allowing 
    greater flexibility for driver modifications and the 
    development of the main supporting classes. 
    https://doi.org/10.5194/wes-8-1235-2023.

    """

    def __init__(self, maxiter=200, **kwargs):
        self.maxiter = maxiter
        super().__init__(**kwargs)

        # Capability flags
        self.supports['optimization'] = True
        self.supports['multiple_objectives'] = False
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['integer_design_vars'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = True
        self.supports['active_set'] = False

        self._desvar_idx = {}
        self.iter_count = 0        # Total objective_callback calls
        self.driver_iter_count = 0 # Main iteration counter
        self.is_converged = False
        self.fail = False

    def _declare_options(self):
        self.options.declare('disp', default=True, desc='Display iteration progress')

        # DTU-style hyperparameters
        self.options.declare('learning_rate', default=1e-2,
                             desc='Initial learning rate (eta_0)')
        self.options.declare('lower', default=1e-6,
                             desc='Lower bound for delta search')
        self.options.declare('upper', default=1e-1,
                             desc='Upper bound for delta search')
        self.options.declare('gamma_min', default=1e-3,
                             desc='Target final eta in multf(T, delta)')

        self.options.declare('beta1', default=0.1, desc='Adam beta1 parameter')
        self.options.declare('beta2', default=0.2, desc='Adam beta2 parameter')

        self.options.declare('tol', default=1e-6,
                             desc='Tolerance on gradient infinity norm')

        self.options.declare('reset_moments', default=False,
                             desc='If True, reset Adam moments periodically') # I explored the possibility to reset betas, but it didn't lead to significant improvements.
        self.options.declare('reset_period', default=100,
                             desc='Period for resetting Adam moments')

    def _setup_driver(self, problem):
        super()._setup_driver(problem)

        if len(self._objs) != 1:
            raise RuntimeError("SimpleSGDDriver supports exactly 1 scalar objective.")

        self.obj_list = list(self._objs)
        self.penalty_name = 'penalty'
        self.obj_and_pen_list = self.obj_list + [self.penalty_name]

        # Map design variables to index slices
        desvars = self._designvars
        count = 0
        for name, meta in desvars.items():
            size = meta['size']
            self._desvar_idx[name] = (count, count + size)
            count += size

        if count == 0:
            raise RuntimeError("Problem has no design variables.")

    def _pack_desvars(self, desvar_vals):
        """Packs design variables dictionary into a single flat vector."""
        n = sum(meta['size'] for meta in self._designvars.values())
        x = np.empty(n, dtype=float)
        for name, meta in self._designvars.items():
            i, j = self._desvar_idx[name]
            x[i:j] = desvar_vals[name]
        return x

    def _unpack_and_set_desvars(self, x):
        """Unpacks flat vector x back into design variables in the model."""
        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

    def _get_final_result(self):
        """Prepares a dictionary with normalized results for reporting."""
        aep_norm = self.get_objective_values()
        try:
            penalty_norm = self.get_response_values('penalty')
            cv_norm = float(list(penalty_norm.values())[0])
        except Exception:
            cv_norm = 0.0

        obj_pen = list(aep_norm.values())[0]

        return {
            'AEP (normalized)': f"[{float(aep_norm[list(aep_norm.keys())[0]]):.6f}]",
            'Constraint violation norm': f"[{cv_norm:.6f}]",
            'Penalized objective': f"[{float(obj_pen):.6f}]"
        }

    def run(self):
        model = self._problem().model
        dv_names = list(self._designvars.keys())

        # Initial design variable packing
        desvar_vals = self.get_design_var_values()
        x0 = self._pack_desvars(desvar_vals)
        x = x0.copy()

        # Initial solve and Jacobian calculation
        model._solve_nonlinear()
        jac = self._compute_totals(of=self.obj_and_pen_list,
                                wrt=dv_names,
                                return_format='array')
        j0 = jac[0]  # Initial objective gradient: grad(-AEP)

        # DTU Setup: Initialize Learning Rate and Alpha0
        learning_rate = float(self.options['learning_rate'])
        self.learning_rate = learning_rate
        self.alpha0 = np.mean(np.abs(j0)) / learning_rate
        self.l0 = float(learning_rate)
        alpha = float(self.alpha0)

        # Decay parameter
        self.lower = self.options['lower']
        self.upper = self.options['upper']
        self.gamma_min = self.options['gamma_min']

        def multf(t, delta):
            prod = 1.0
            for ii in range(t):
                prod *= 1.0 / (1.0 + delta * ii)
            return learning_rate * prod

        # Search for optimal delta
        for _ in range(self.maxiter):
            mid = np.mean([self.lower, self.upper])
            etaM = multf(self.maxiter, mid)
            if etaM < self.gamma_min:
                self.upper = mid
            elif etaM > self.gamma_min:
                self.lower = mid
            self.mid = mid

        # Initialize moments and loop state
        m = np.zeros_like(x0)
        v = np.zeros_like(x0)
        adam_step = 0

        self.is_converged = False
        self.iter_count = 0
        self.driver_iter_count = 0
        self.fail = False

        t = 0
        tol = self.options['tol']
        fk = None
        grad_norm = np.inf
        learning_rate_current = learning_rate
        alpha_current = alpha

        while t < self.maxiter and not self.is_converged:
            obj, x, alpha_current, learning_rate_current, m, v, adam_step, grad_norm, success = \
                self.objective_callback(x, alpha_current, learning_rate_current,
                                        m, v, adam_step,
                                        record=True, update=True)

            if obj is not None:
                if isinstance(obj, np.ndarray):
                    fk = float(obj.item()) if obj.size == 1 else float(obj[0])
                else:
                    fk = float(obj)

            t += 1
            self.driver_iter_count = t

            if hasattr(self, 'recorder') and self.recorder is not None:
                self._problem().model.set_val('opt_iter',
                                            float(self.driver_iter_count))

            # Reset Adam moments periodically
            if self.options['reset_moments'] and t > 0 and (t % self.options['reset_period'] == 0):
                m[:] = 0.0
                v[:] = 0.0
                adam_step = 0
                if self.options['disp']:
                    print(f"Resetting Adam moments at iteration {t}")

            # Convergence based on gradient norm
            if grad_norm < tol:
                self.is_converged = True

            if success == 0:
                self.fail = True

        # Set final design variables back to the model
        self._unpack_and_set_desvars(x)

        # Final Statistics Output
        if self.options['disp']:
            if self.fail and t >= self.maxiter:
                print("OptimizeWarning: Maximum number of iterations exceeded.")
                print(f"Current stochastic function value f_t: {fk:.6f}")
                print(f"Gradient infinity norm: {grad_norm:.6e}")
                print(f"Iterations: {t}")
                print(f"Function evaluations: {self.iter_count}")
                print(f"Gradient evaluations: {self.iter_count}")
                print("Optimization FAILED.")
            else:
                print("Optimization SUCCESSFUL.")
                print(f"Current stochastic function value f_t: {fk:.6f}")
                print(f"Gradient infinity norm: {grad_norm:.6e}")
                print(f"Iterations: {t}")
                print(f"Function evaluations: {self.iter_count}")
                print(f"Gradient evaluations: {self.iter_count}")
                print("\n--- Optimization result ---")
                result = self._get_final_result()
                for key, val in result.items():
                    print(f"{key}: {val}")

        if hasattr(self, 'recorder') and self.recorder is not None:
            self.recorder.close()

        return self.fail

    def objective_callback(self, x, alpha, learning_rate, m, v, adam_step,
                        record=False, update=True):
        """One step of the optimization: updates variables based on gradients."""
        model = self._problem().model
        dv_names = list(self._designvars.keys())
        success = 1

        # Set variables in model
        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        with RecordingDebugging('SimpleSGD', self.iter_count, self) as rec:
            if update:
                self.iter_count += 1

            try:
                model._solve_nonlinear()
                if hasattr(self, 'recorder') and self.recorder is not None:
                    self.recorder.log()
            except AnalysisError:
                model._clear_iprint()
                success = 0

            # Get objective value
            for _, val in self.get_objective_values().items():
                obj = val
                break

            # Compute gradients
            jac = self._compute_totals(of=self.obj_and_pen_list,
                                    wrt=dv_names,
                                    return_format='array')
            j = jac[0]   # grad(-AEP)
            c = jac[1]   # grad(Penalty)

            # Combined Stochastic Gradient
            jacobian = -j + alpha * c
            grad_norm = np.linalg.norm(jacobian, np.inf)

            # Adam Moment Update
            beta1 = self.options['beta1']
            beta2 = self.options['beta2']

            adam_step += 1
            m = beta1 * m + (1 - beta1) * jacobian
            v = beta2 * v + (1 - beta2) * jacobian ** 2

            # Bias correction
            mhat = m / (1.0 - beta1 ** adam_step)
            vhat = v / (1.0 - beta2 ** adam_step)

            # Update design variables x
            x -= learning_rate * mhat / np.sqrt(vhat + 1e-12)

            # Update Learning Rate and Alpha
            learning_rate *= 1.0 / (1.0 + self.mid * self.iter_count)
            alpha = self.alpha0 * self.l0 / learning_rate

            if np.any(np.isnan(x)):
                raise Exception("NaN in design variables detected")

            rec.abs = 0.0
            rec.rel = 0.0

        return obj, x, alpha, learning_rate, m, v, adam_step, grad_norm, success
    

"""Resetting was just a test with no positive impact"""