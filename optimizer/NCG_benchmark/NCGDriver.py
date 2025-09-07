import numpy as np

from openmdao.core.driver import Driver, RecordingDebugging
import openmdao.api as om

# AFTER
class NCGDriver(Driver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports['optimization'] = True
        self.supports['gradients'] = True
        self.supports['multiple_objectives'] = False
        self.supports['equality_constraints'] = True
        self.supports['inequality_constraints'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = False
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports['simultaneous_derivatives'] = True
        self.supports._read_only = True

        self._total_jac_format = 'array'

        self._desvar_idx = {}
        self._dvlist = None
        self._lbvec = None
        self._ubvec = None

        # AL state
        self._ineq_specs = []   # list of tuples describing each scalar inequality
        self._eq_specs   = []   # list for equalities
        self._lam_ineq   = None
        self._lam_eq     = None
        self._rho        = None
        self._last_viol  = np.inf

    def _declare_options(self):
        self.options.declare('maxiter', default=200, lower=1)
        self.options.declare('tol_grad', default=1e-8, lower=0.0)
        self.options.declare('c1', default=1e-4, lower=0.0, upper=1.0)
        self.options.declare('c2', default=0.9, lower=0.0, upper=1.0)
        self.options.declare('alpha0', default=1.0, lower=1e-16)
        self.options.declare('max_linesearch', default=20, lower=1)

        # Augmented Lagrangian controls
        self.options.declare('al_rho0', default=10.0, lower=1e-9,
                             desc='Initial AL penalty parameter.')
        self.options.declare('al_mu_inc', default=5.0, lower=1.0,
                             desc='Multiplier for rho when feasibility stalls.')
        self.options.declare('al_update_freq', default=1, lower=1,
                             desc='Update multipliers every k major iterations.')
        self.options.declare('feas_tol', default=1e-6, lower=0.0,
                             desc='Feasibility tolerance on constraints (max violation).')


    def _setup_driver(self, problem):
        super()._setup_driver(problem)

        if len(self._objs) != 1:
            raise RuntimeError(f"{self.msginfo}: only a single objective is supported.")

        # Flatten design vars and bounds
        self._dvlist = list(self._designvars.keys())
        idx = 0
        lbs, ubs = [], []
        for name, meta in self._designvars.items():
            size = int(meta['size'])
            self._desvar_idx[name] = (idx, idx + size)
            idx += size
            lb = meta.get('lower', None)
            ub = meta.get('upper', None)
            lb = (-np.inf if lb is None else lb)
            ub = ( np.inf if ub is None else ub)
            lb = np.full(size, lb) if np.isscalar(lb) else np.asarray(lb).ravel()
            ub = np.full(size, ub) if np.isscalar(ub) else np.asarray(ub).ravel()
            lbs.append(lb); ubs.append(ub)
        self._ndv = idx
        self._lbvec = np.concatenate(lbs) if lbs else None
        self._ubvec = np.concatenate(ubs) if ubs else None

        # Prepare constraint specs (scalarize with standard forms)
        self._ineq_specs.clear()
        self._eq_specs.clear()

        for cname, meta in self._cons.items():
            # each response may be vector; we handle lower/upper/equals separately
            shape = int(np.prod(meta['size'])) if 'size' in meta else None
            # We cannot know v length until runtime; we’ll rely on runtime values for lengths.
            # Store side info so we can build g(x) and its Jacobian signs later.
            if meta.get('upper', None) is not None:
                self._ineq_specs.append((cname, 'upper', meta['upper']))  # g = v - ub <= 0
            if meta.get('lower', None) is not None:
                self._ineq_specs.append((cname, 'lower', meta['lower']))  # g = lb - v <= 0
            if meta.get('equals', None) is not None:
                self._eq_specs.append((cname, 'equals', meta['equals']))  # h = v - eq = 0

        # Initialize AL state after first model run in run()
        self._rho = float(self.options['al_rho0'])
        self._lam_ineq = None
        self._lam_eq = None

    # ---------- Helpers (driver-scaled space) ----------
    def _pack_x(self, dv_dict):
        x = np.empty(self._ndv, dtype=float)
        for name in self._dvlist:
            i, j = self._desvar_idx[name]
            x[i:j] = np.ravel(dv_dict[name])
        return x

    # Set design vars from flat array (driver-scaled)
    # Also does projection to bounds (driver-scaled)
    def _unpack_and_set_x(self, x):
        # Project to bounds (driver-scaled) and set
        if self._lbvec is not None and self._ubvec is not None:
            x = np.clip(x, self._lbvec, self._ubvec)
        for name in self._dvlist:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

    # 
    def _base_objective(self, driver_scaling=True):
        objs = self.get_objective_values(driver_scaling=driver_scaling)
        for _, val in objs.items():
            return float(np.ravel(val)[0])

    def _aug_fun_grad(self):
        f = self._base_objective(driver_scaling=True)
        # ∇f
        g_f = self._compute_totals(of=list(self._objs.keys()), wrt=self._dvlist,
                                   return_format='array', driver_scaling=True).reshape(-1)

        # No constraints
        if not self._ineq_specs and not self._eq_specs:
            return f, g_f, 0.0, 0.0

        # Gather values and Jacobians for constraints
        con_vals = self.get_constraint_values(driver_scaling=True)  # dict name->array
        J = self._compute_totals(of=list(con_vals.keys()), wrt=self._dvlist,
                                 return_format='dict', driver_scaling=True)

        # Build inequality vector g(x) and equality vector h(x)
        g_list, Jg_blocks, g_sides = [], [], []
        for (cname, side, bnd) in self._ineq_specs:
            v = np.atleast_1d(con_vals[cname]).astype(float)
            if np.isscalar(bnd):
                bnd = np.full_like(v, bnd, dtype=float)
            else:
                bnd = np.asarray(bnd, dtype=float).reshape(v.shape)

            if side == 'upper':          # g = v - ub <= 0
                g_side = v - bnd
                sign = +1.0
            else:                         # 'lower': g = lb - v <= 0
                g_side = bnd - v
                sign = -1.0

            g_list.append(g_side)
            # Store Jacobian blocks with sign
            Jg_blocks.append({w: sign * J[(cname, w)] for w in self._dvlist})
            g_sides.append(g_side)

        h_list, Jh_blocks = [], []
        for (cname, _, eqv) in self._eq_specs:
            v = np.atleast_1d(con_vals[cname]).astype(float)
            if np.isscalar(eqv):
                eqv = np.full_like(v, eqv, dtype=float)
            else:
                eqv = np.asarray(eqv, dtype=float).reshape(v.shape)
            h = v - eqv
            h_list.append(h)
            Jh_blocks.append({w: J[(cname, w)] for w in self._dvlist})

        g_vec = np.concatenate(g_list) if g_list else np.zeros(0)
        h_vec = np.concatenate(h_list) if h_list else np.zeros(0)

        # Initialize multipliers on first call
        if self._lam_ineq is None and g_vec.size:
            self._lam_ineq = np.zeros_like(g_vec)
        if self._lam_eq is None and h_vec.size:
            self._lam_eq = np.zeros_like(h_vec)

        rho = self._rho
        # Inequality AL pieces
        if g_vec.size:
            lam_tilde = np.maximum(0.0, self._lam_ineq + rho * g_vec)
            f += (0.5 / rho) * (np.dot(lam_tilde, lam_tilde) - np.dot(self._lam_ineq, self._lam_ineq))
        else:
            lam_tilde = np.zeros(0)

        # Equality AL pieces
        if h_vec.size:
            f += 0.5 * rho * np.dot(h_vec + self._lam_eq / rho, h_vec + self._lam_eq / rho) \
                 - 0.5 * (1.0 / rho) * np.dot(self._lam_eq, self._lam_eq)

        # Gradient assembly
        grad = g_f.copy()
        # inequalities: Σ_i λ̃_i ∇g_i
        if g_vec.size:
            # walk through each inequality block with its slice
            start = 0
            for block in Jg_blocks:
                m = next(iter(block.values())).shape[0]  # rows
                coeff = lam_tilde[start:start + m]  # row weights
                for w in self._dvlist:
                    # coeff (1xm) @ J (m x n_w)
                    grad += coeff @ block[w]
                start += m

        # equalities: (λ_eq + ρ h)^T ∇h
        if h_vec.size:
            coeff = (self._lam_eq + rho * h_vec)
            for b in Jh_blocks:
                # b[w] has rows matching a chunk of h. We must slice coeff accordingly.
                m = next(iter(b.values())).shape[0]
                c_slice, coeff = coeff[:m], coeff[m:]
                for w in self._dvlist:
                    grad += c_slice @ b[w]

        # Feasibility (max violation) for stopping and AL update logic
        max_viol = 0.0
        if g_vec.size:
            max_viol = max(max_viol, float(np.max(g_vec)))
        if h_vec.size:
            max_viol = max(max_viol, float(np.max(np.abs(h_vec))))

        return f, grad, max_viol, float(np.linalg.norm(h_vec, ord=np.inf)) if h_vec.size else 0.0

    # Replace objective/gradient getters to use augmented forms
    def _get_f_scalar(self, driver_scaling=True):
        f, _, _, _ = self._aug_fun_grad()
        return f

    def _grad_at_current(self):
        _, grad, _, _ = self._aug_fun_grad()
        return grad

    def _aug_fun_grad(self, driver_scaling=True):
        """
        Augmented Lagrangian objective and gradient at the current x.

        Returns:
            f_aug        : scalar augmented objective
            grad_aug     : gradient wrt flattened design vars
            max_violation: max(g(x)) over inequalities and |h(x)| over equalities
            eq_inf       : infinity norm of equality residuals

        Notes:
        - Inequalities use g(x) <= 0 (upper: v - ub, lower: lb - v).
        - Equalities use h(x) = 0 (v - eq).
        - Uses current multipliers (lambda) and penalty (rho).
        """
        rho = float(self._rho)

        # Base objective and gradient
        f = self._base_objective(driver_scaling=True)
        g_f = self._compute_totals(of=list(self._objs.keys()), wrt=self._dvlist,
                                return_format='array', driver_scaling=True).reshape(-1)

        # If no constraints registered, still return 4-tuple
        if not self._ineq_specs and not self._eq_specs:
            return f, g_f, 0.0, 0.0

        con_vals = self.get_constraint_values(driver_scaling=driver_scaling)  # dict
        con_names = list(con_vals.keys())

        # Force dict totals regardless of Driver._total_jac_format
        J = self._problem().compute_totals(of=con_names, wrt=self._dvlist,
                                        return_format='dict', driver_scaling=driver_scaling)

        # Build inequality g(x) (<= 0) and equality h(x) (== 0)
        g_chunks, Jg_blocks = [], []
        for (cname, side, bound) in self._ineq_specs:
            v = np.atleast_1d(con_vals[cname]).astype(float)
            b = np.full_like(v, bound, dtype=float) if np.isscalar(bound) else np.asarray(bound, dtype=float).reshape(v.shape)
            if side == 'upper':          # g = v - ub <= 0
                g_side = v - b
                sign = +1.0
            else:                        # 'lower': g = lb - v <= 0
                g_side = b - v
                sign = -1.0
            g_chunks.append(g_side)
            Jg_blocks.append({w: sign * J[(cname, w)] for w in self._dvlist})

        h_chunks, Jh_blocks = [], []
        for (cname, _, eqv) in self._eq_specs:
            v = np.atleast_1d(con_vals[cname]).astype(float)
            e = np.full_like(v, eqv, dtype=float) if np.isscalar(eqv) else np.asarray(eqv, dtype=float).reshape(v.shape)
            h = v - e
            h_chunks.append(h)
            Jh_blocks.append({w: J[(cname, w)] for w in self._dvlist})

        g_vec = np.concatenate(g_chunks) if g_chunks else np.zeros(0)
        h_vec = np.concatenate(h_chunks) if h_chunks else np.zeros(0)

        # Initialize multipliers on first use or if sizes changed
        if g_vec.size and (self._lam_ineq is None or self._lam_ineq.size != g_vec.size):
            self._lam_ineq = np.zeros_like(g_vec)
        if h_vec.size and (self._lam_eq is None or self._lam_eq.size != h_vec.size):
            self._lam_eq = np.zeros_like(h_vec)

        # AL objective
        f_aug = f
        lam_tilde = np.zeros_like(g_vec)
        if g_vec.size:
            lam_tilde = np.maximum(0.0, self._lam_ineq + rho * g_vec)
            f_aug += (0.5 / rho) * (np.dot(lam_tilde, lam_tilde) - np.dot(self._lam_ineq, self._lam_ineq))
        if h_vec.size:
            f_aug += 0.5 * rho * np.dot(h_vec + self._lam_eq / rho, h_vec + self._lam_eq / rho) \
                    - 0.5 * (1.0 / rho) * np.dot(self._lam_eq, self._lam_eq)

        # AL gradient
        grad = g_f.copy()
        if g_vec.size:
            start = 0
            for block in Jg_blocks:
                m = next(iter(block.values())).shape[0]
                coeff = lam_tilde[start:start + m]            # shape: (m,)
                for w in self._dvlist:
                    grad += coeff @ block[w]                  # (m,) @ (m, n_w) -> (n_w,)
                start += m
        if h_vec.size:
            coeff = (self._lam_eq + rho * h_vec)              # shape: (len(h_vec),)
            offset = 0
            for b in Jh_blocks:
                m = next(iter(b.values())).shape[0]
                c_slice = coeff[offset:offset + m]
                for w in self._dvlist:
                    grad += c_slice @ b[w]
                offset += m

        # Feasibility measures
        eq_inf = float(np.max(np.abs(h_vec))) if h_vec.size else 0.0
        max_viol = 0.0
        if g_vec.size:
            max_viol = float(np.max(g_vec))
        max_viol = max(max_viol, eq_inf)

        return f_aug, grad, max_viol, eq_inf

    def _get_base_objective(self, driver_scaling=True):
        objs = self.get_objective_values(driver_scaling=driver_scaling)
        for _, val in objs.items():
            return float(np.ravel(val)[0])
        return 0.0  # should never happen
    # ---------- Strong Wolfe line search (Nocedal-Wright Alg 3.5, bisection-only "zoom") ----------
    def _phi_and_dphi(self, x, d):
        # Evaluate f and directional derivative at the current (already set) x
        f = self._get_f_scalar(driver_scaling=True)
        g = self._grad_at_current()
        dphi = float(np.dot(g, d))
        return f, g, dphi

    def _line_search(self, xk, fk, gk, dk):
        c1 = self.options['c1']
        c2 = self.options['c2']
        alpha0 = self.options['alpha0']
        maxls = self.options['max_linesearch']

        # Ensure descent direction
        gdotd0 = float(np.dot(gk, dk))
        if gdotd0 >= 0.0:
            dk = -gk
            gdotd0 = -float(np.dot(gk, gk))

        # Bracketing
        alpha_prev = 0.0
        f_prev = fk
        dphi_prev = gdotd0

        alpha = alpha0
        for i in range(1, maxls + 1):
            # Trial x = xk + alpha * dk
            self._unpack_and_set_x(xk + alpha * dk)
            self._run_solve_nonlinear()
            f, g, dphi = self._phi_and_dphi(xk + alpha * dk, dk)

            if (f > fk + c1 * alpha * gdotd0) or (i > 1 and f >= f_prev):
                return self._zoom(xk, fk, gdotd0, dk, alpha_prev, alpha)
            if abs(dphi) <= -c2 * gdotd0:
                return alpha, f, g
            if dphi >= 0.0:
                return self._zoom(xk, fk, gdotd0, dk, alpha, alpha_prev)

            alpha_prev, f_prev, dphi_prev = alpha, f, dphi
            alpha *= 2.0  # conservative expansion

        # Fallback: accept the best-so-far (alpha_prev)
        self._unpack_and_set_x(xk + alpha_prev * dk)
        self._run_solve_nonlinear()
        f, g, _ = self._phi_and_dphi(xk + alpha_prev * dk, dk)
        return alpha_prev, f, g

    def _zoom(self, xk, fk, gdotd0, dk, alo, ahi):
        c1 = self.options['c1']
        c2 = self.options['c2']
        maxls = self.options['max_linesearch']

        f_lo = None

        for _ in range(maxls):
            alpha = 0.5 * (alo + ahi)
            self._unpack_and_set_x(xk + alpha * dk)
            self._run_solve_nonlinear()
            f, g, dphi = self._phi_and_dphi(xk + alpha * dk, dk)

            self._unpack_and_set_x(xk + alo * dk)
            self._run_solve_nonlinear()
            f_lo = self._get_f_scalar(driver_scaling=True)

            if (f > fk + c1 * alpha * gdotd0) or (f >= f_lo):
                ahi = alpha
            else:
                if abs(dphi) <= -c2 * gdotd0:
                    return alpha, f, g
                if dphi * (ahi - alo) >= 0:
                    ahi = alo
                alo = alpha

        # last resort
        self._unpack_and_set_x(xk + alo * dk)
        self._run_solve_nonlinear()
        f, g, _ = self._phi_and_dphi(xk + alo * dk, dk)
        return alo, f, g

    # ---------- Main run ----------
    def run(self):
        self.result.reset()
        x = self._pack_x(self.get_design_var_values(driver_scaling=True))

        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._run_solve_nonlinear()

        # Ensure AL penalty is initialized
        if self._rho is None:
            self._rho = float(self.options['al_rho0'])
            
        # Initialize AL state on first eval
        f, g, viol, _ = self._aug_fun_grad()
        if self._lam_ineq is None: self._lam_ineq = np.zeros(0)
        if self._lam_eq   is None: self._lam_eq   = np.zeros(0)
        self._last_viol = viol

        d = -g.copy()
        maxiter = int(self.options['maxiter'])
        tol_g   = float(self.options['tol_grad'])
        feas_tol = float(self.options['feas_tol'])

        if np.linalg.norm(g, ord=np.inf) <= tol_g and viol <= feas_tol:
            self.iter_count += 1
            self.result.iter_count = self.iter_count
            self.result.success = True
            self.result.exit_status = 'FIRST_ORDER_OPTIMALITY'
            return False

        for k in range(maxiter):
            alpha, f_new, g_new = self._line_search(x, f, g, d)
            x_new = x + alpha * d

            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                rec.abs = abs(f_new - f)
                rec.rel = rec.abs / (abs(f) + 1e-16)
            self.iter_count += 1

            # ----- Augmented Lagrangian multiplier & rho update -----
            # Evaluate violation at accepted x_new (already set by line search)
            _, _, viol_new, _ = self._aug_fun_grad()

            # Update multipliers every al_update_freq iterations
            if (k + 1) % int(self.options['al_update_freq']) == 0:
                # Gather again to get g(x), h(x)
                con_vals = self.get_constraint_values(driver_scaling=True)
                g_list, h_list = [], []
                for (cname, side, bnd) in self._ineq_specs:
                    v = np.atleast_1d(con_vals[cname]).astype(float)
                    b = (np.full_like(v, bnd) if np.isscalar(bnd) else np.asarray(bnd).reshape(v.shape))
                    g_list.append(v - b if side == 'upper' else b - v)
                for (cname, _, eqv) in self._eq_specs:
                    v = np.atleast_1d(con_vals[cname]).astype(float)
                    e = (np.full_like(v, eqv) if np.isscalar(eqv) else np.asarray(eqv).reshape(v.shape))
                    h_list.append(v - e)

                g_vec = np.concatenate(g_list) if g_list else np.zeros(0)
                h_vec = np.concatenate(h_list) if h_list else np.zeros(0)

                # Grow/trim λ arrays if sizes changed (vector constraints)
                if self._lam_ineq is None or self._lam_ineq.size != g_vec.size:
                    self._lam_ineq = np.zeros_like(g_vec)
                if self._lam_eq is None or self._lam_eq.size != h_vec.size:
                    self._lam_eq = np.zeros_like(h_vec)

                # λ updates
                if g_vec.size:
                    self._lam_ineq = np.maximum(0.0, self._lam_ineq + self._rho * g_vec)
                if h_vec.size:
                    self._lam_eq = self._lam_eq + self._rho * h_vec

                # If feasibility stalls, increase ρ
                if viol_new > 0.75 * self._last_viol - 1e-16:
                    self._rho *= float(self.options['al_mu_inc'])
                self._last_viol = viol_new

            # Convergence check (first-order stationarity + feasibility)
            if np.linalg.norm(g_new, ord=np.inf) <= tol_g and viol_new <= feas_tol:
                x, f, g = x_new, f_new, g_new
                break

            # PR+ update with restart if not descent
            y = g_new - g
            beta_pr = float(np.dot(g_new, y) / (np.dot(g, g) + 1e-32))
            beta = max(0.0, beta_pr)
            d = -g_new + beta * d
            if np.dot(d, g_new) >= 0.0:
                d = -g_new

            x, f, g = x_new, f_new, g_new

        self._unpack_and_set_x(x)
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._run_solve_nonlinear()

        self.result.iter_count = self.iter_count
        self.result.success = (np.linalg.norm(g, ord=np.inf) <= tol_g and self._last_viol <= feas_tol)
        self.result.exit_status = 'FIRST_ORDER_OPTIMALITY' if self.result.success else 'MAX_ITERATIONS'
        return not self.result.success

    def _get_name(self):
        return "NCG"

if __name__ == "__main__":
    # toy model: paraboloid
    class Paraboloid(om.ExplicitComponent):
        def setup(self):
            self.add_input('x', 0.0)
            self.add_input('y', 0.0)
            self.add_output('f', 0.0)
            self.declare_partials(of='f', wrt='x', method='fd')  # or analytic
            self.declare_partials(of='f', wrt='y', method='fd')

        def compute(self, inputs, outputs):
            x = inputs['x']
            y = inputs['y']
            outputs['f'] = (x - 3.0)**2 + (y + 1.0)**2

    prob = om.Problem()
    model = prob.model
    model.add_subsystem('p', Paraboloid(), promotes=['*'])
    # prob.model.add_constraint('f', upper=10.0)  # trivial example


    prob.driver = NCGDriver(maxiter=100, tol_grad=1e-8, c1=1e-4, c2=0.9, alpha0=1.0)

    prob.model.add_design_var('x')         # no bounds in this minimal driver
    prob.model.add_design_var('y')
    prob.model.add_objective('f')
    prob.setup()
    res = prob.run_driver()

    print("success:", prob.driver.result.success)
    print("x* =", prob.get_val('x'), " y* =", prob.get_val('y'), " f* =", prob.get_val('f'))