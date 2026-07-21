import numpy as np
import openmdao.api as om
import xarray as xr
from py_wake.site import XRSite
from py_wake.wind_turbines._wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.utils.gradients import autograd

def _build_wind_turbine(turb_info, name):
    """Builds a single-type PyWake WindTurbine from a windIO turbine info block."""
    perf_info = turb_info['performance']
    p_ws = np.array(perf_info['power_curve']['power_wind_speeds'])
    p = np.array(perf_info['power_curve']['power_values'])
    ct_ws_raw = np.array(perf_info['Ct_curve']['Ct_wind_speeds'])
    ct_val_raw = np.array(perf_info['Ct_curve']['Ct_values'])
    ct = np.interp(p_ws, ct_ws_raw, ct_val_raw)

    powerCtFunction = PowerCtTabular(ws=p_ws, power=p, power_unit='W', ct=ct)
    return WindTurbine(name=name, diameter=turb_info['rotor_diameter'],
                        hub_height=turb_info['hub_height'], powerCtFunction=powerCtFunction)


class AEPComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt', types=int)
        self.options.declare('system_data', types=dict)
        self.options.declare('site_data', types=dict)
        self.options.declare('stochastic_mode', types=bool, default=False)
        self.options.declare('sample_size', types=int, default=1)
        self.options.declare('x_offset', types=float)
        self.options.declare('y_offset', types=float)
        self.options.declare('neighbor_x', types=np.ndarray)
        self.options.declare('neighbor_y', types=np.ndarray)
        # Index of this component's own farm into system_data['wind_farm'] -- previously
        # hardcoded to 0, which silently used the wrong turbine whenever a non-first farm
        # was optimized. Index (not name) because system_data['wind_farm'] is a plain list.
        self.options.declare('farm_index', types=int, default=0)
        # Per-neighbor-position farm index, same length as neighbor_x/neighbor_y. Lets
        # neighbors cast wakes using their OWN turbine model instead of this farm's --
        # e.g. Vineyard (Haliade-X 13MW) and Revolution (SG11-200 11MW) are physically
        # different turbines, and PyWake needs an explicit type= per position to tell them
        # apart. If omitted/mismatched in length, neighbors fall back to this farm's own
        # turbine type (the old behavior) for backward compatibility.
        self.options.declare('neighbor_farm_idx', types=np.ndarray, default=np.array([]))

    def setup(self):
        n_wt = self.options['n_wt']

        self.add_input('x', val=np.zeros(n_wt), units='m')
        self.add_input('y', val=np.zeros(n_wt), units='m')
        self.add_output('aep', val=0.0, units='GW*h')

        system_data = self.options['system_data']
        farm_index = self.options['farm_index']
        own_turb_info = system_data['wind_farm'][farm_index]['turbines']

        # Build one WindTurbine per distinct farm actually present (this farm + whichever
        # neighbor farms were injected), and map each to a local PyWake type index. Local
        # type 0 is always this farm's own turbine, regardless of its global farm_index.
        neighbor_farm_idx = np.asarray(self.options['neighbor_farm_idx'], dtype=int)
        neighbor_x = self.options['neighbor_x']
        if len(neighbor_farm_idx) != len(neighbor_x):
            # No (or mismatched) per-neighbor farm info -- fall back to treating neighbors
            # as this farm's own turbine type, same approximation the code used to make
            # unconditionally.
            neighbor_farm_idx = np.full(len(neighbor_x), farm_index, dtype=int)

        distinct_neighbor_farms = sorted(set(neighbor_farm_idx.tolist()) - {farm_index})
        wind_turbines_lst = [_build_wind_turbine(own_turb_info, f"farm_{farm_index}")]
        self._type_by_farm_index = {farm_index: 0}
        for local_type, other_farm_index in enumerate(distinct_neighbor_farms, start=1):
            other_turb_info = system_data['wind_farm'][other_farm_index]['turbines']
            wind_turbines_lst.append(_build_wind_turbine(other_turb_info, f"farm_{other_farm_index}"))
            self._type_by_farm_index[other_farm_index] = local_type

        wind_turbine_model = WindTurbines.from_WindTurbine_lst(wind_turbines_lst)

        self._own_type = np.zeros(n_wt, dtype=int)
        self._neighbor_type = np.array([self._type_by_farm_index[fi] for fi in neighbor_farm_idx], dtype=int)

        # Assemble the PyWake site
        site_dat = self.options['site_data']
        wd_original = site_dat['directions']

        # Real TI varies with wind speed (TI_org), not with direction. We keep the
        # TI(ws) curve separate for on-demand interpolation instead of embedding it
        # as a 'ws' coordinate on XRSite -- that extra coordinate is what breaks
        # PyWake's time=True mode (confirmed reshape/broadcast bug).
        self._ws_TI_curve = np.asarray(site_dat['ws_TI'])
        self._TI_curve = np.asarray(site_dat['TI_org'])
        if self._TI_curve.ndim == 0:
            self._TI_curve = np.full(self._ws_TI_curve.shape, float(self._TI_curve))

        self.pywake_site = XRSite(
            ds=xr.Dataset(
                data_vars={
                    'Sector_frequency': ('wd', site_dat['probabilities']), 
                    'Weibull_A': ('wd', site_dat['weibull_A']), 
                    'Weibull_k': ('wd', site_dat['weibull_k']),
                    'TI': float(np.mean(self._TI_curve)),  # placeholder; always overridden via TI= at call time
                },
                coords={'wd': wd_original}
            )
        )
        self.pywake_site.interp_method = 'linear'
        self.wake_model = Nygaard_2022(self.pywake_site, wind_turbine_model)

        # Explicit grid used in deterministic mode (full wind rose), reproducing the
        # original TI(wd,ws) matrix, passed via the TI= call argument instead of
        # embedded as a site coordinate.
        self._wd_full = np.asarray(wd_original)
        self._ws_full = self._ws_TI_curve
        self._TI_grid_full = np.tile(self._TI_curve, (len(self._wd_full), 1))
        
        self._current_wd = None
        self._current_ws = None
        self._current_ti = None

        self.declare_partials('aep', 'x')
        self.declare_partials('aep', 'y')

    def _sample_metocean(self):
        samps = self.options['sample_size']
        wd_coords = self.pywake_site.ds.wd.values
        freqs = self.pywake_site.ds.Sector_frequency.values
        freqs = freqs / freqs.sum()
        A_vals = self.pywake_site.ds.Weibull_A.values
        k_vals = self.pywake_site.ds.Weibull_k.values

        wd_idx = np.random.choice(np.arange(wd_coords.size), size=samps, p=freqs)
        wd_samp = wd_coords[wd_idx]
        ws_samp = A_vals[wd_idx] * np.random.weibull(k_vals[wd_idx])
        ti_samp = np.interp(ws_samp, self._ws_TI_curve, self._TI_curve)
        return wd_samp, ws_samp, ti_samp

    def _full_type_array(self):
        """Per-position PyWake turbine type index: this farm's own turbines (local type 0)
        followed by neighbor positions tagged with their own farm's turbine type."""
        if len(self._neighbor_type) > 0:
            return np.concatenate((self._own_type, self._neighbor_type))
        return self._own_type

    def compute(self, inputs, outputs):
        wfm = self.wake_model
        # Shift into the local coordinate frame anchored at the site bounding-box minimum
        x_local = inputs['x'] - self.options['x_offset']
        y_local = inputs['y'] - self.options['y_offset']

        # Neighbors are already processed to local coordinates during system initialization
        xn_local = self.options['neighbor_x'] - self.options['x_offset'] if len(self.options['neighbor_x']) > 0 else np.array([])
        yn_local = self.options['neighbor_y'] - self.options['y_offset'] if len(self.options['neighbor_y']) > 0 else np.array([])

        x_full = np.concatenate((x_local, xn_local)) if len(xn_local) > 0 else x_local
        y_full = np.concatenate((y_local, yn_local)) if len(yn_local) > 0 else y_local
        type_full = self._full_type_array()

        if self.options['stochastic_mode']:
            self._current_wd, self._current_ws, self._current_ti = self._sample_metocean()
            sim_res = wfm(x=x_full, y=y_full, type=type_full, wd=self._current_wd, ws=self._current_ws,
                          TI=self._current_ti, time=True, n_cpu=1)
        else:
            self._current_wd, self._current_ws, self._current_ti = None, None, None
            sim_res = wfm(x=x_full, y=y_full, type=type_full, wd=self._wd_full, ws=self._ws_full,
                          TI=self._TI_grid_full, time=False, n_cpu=1)

        if len(xn_local) > 0:
            aep_raw = sim_res.aep().isel(wt=slice(0, len(x_local))).sum().values
        else:
            aep_raw = sim_res.aep().sum().values

        outputs['aep'] = np.abs(float(aep_raw))

    def compute_deterministic_aep(self, x, y):
        """
        On-demand full-wind-rose AEP evaluation (MWh), independent of stochastic_mode.
        Does not read or write any instance state used by compute()/compute_partials()
        (self._current_wd/ws/ti), so it is safe to call between SGD iterations without
        perturbing the stochastic mini-batch gradient the driver is descending on.
        Intended for reporting/plotting a noise-free convergence metric.
        """
        wfm = self.wake_model
        x_local = np.asarray(x) - self.options['x_offset']
        y_local = np.asarray(y) - self.options['y_offset']

        xn_local = self.options['neighbor_x'] - self.options['x_offset'] if len(self.options['neighbor_x']) > 0 else np.array([])
        yn_local = self.options['neighbor_y'] - self.options['y_offset'] if len(self.options['neighbor_y']) > 0 else np.array([])

        x_full = np.concatenate((x_local, xn_local)) if len(xn_local) > 0 else x_local
        y_full = np.concatenate((y_local, yn_local)) if len(yn_local) > 0 else y_local

        sim_res = wfm(x=x_full, y=y_full, type=self._full_type_array(), wd=self._wd_full, ws=self._ws_full,
                      TI=self._TI_grid_full, time=False, n_cpu=1)

        if len(xn_local) > 0:
            aep_raw = sim_res.aep().isel(wt=slice(0, len(x_local))).sum().values
        else:
            aep_raw = sim_res.aep().sum().values

        return np.abs(float(aep_raw)) * 1000.0  # GWh -> MWh

    def compute_partials(self, inputs, J):
        wfm = self.wake_model
        n_active = len(inputs['x']) # active turbine count
        
        # 1. Transform global real inputs to local variables for PyWake autograd safety
        x_local = inputs['x'] - self.options['x_offset']
        y_local = inputs['y'] - self.options['y_offset']
        
        # Transform static neighbors to local coordinates as well
        xn_local = self.options['neighbor_x'] - self.options['x_offset'] if len(self.options['neighbor_x']) > 0 else np.array([])
        yn_local = self.options['neighbor_y'] - self.options['y_offset'] if len(self.options['neighbor_y']) > 0 else np.array([])
        
        x_full = np.concatenate((x_local, xn_local)) if len(xn_local) > 0 else x_local
        y_full = np.concatenate((y_local, yn_local)) if len(yn_local) > 0 else y_local
        type_full = self._full_type_array()

        wd_arg = self._current_wd if self.options['stochastic_mode'] else self._wd_full
        ws_arg = self._current_ws if self.options['stochastic_mode'] else self._ws_full
        ti_arg = self._current_ti if self.options['stochastic_mode'] else self._TI_grid_full
        time_arg = True if self.options['stochastic_mode'] else False

        # NOTE: wfm.aep()/aep_gradients() always sum AEP over ALL turbines passed in
        # (active + neighbors) -- there is no way to restrict the sum to a subset.
        # Differentiating wfm.aep_gradients(...) and then slicing [:n_active] would
        # contaminate the gradient with d(neighbor_AEP)/d(active_x), which is not
        # what compute() actually returns (compute() sums only the active turbines).
        # So we replicate wfm.aep()'s internal body here, slicing out the active
        # turbines BEFORE the sum, keeping everything inside the autograd graph
        # (wfm._run returns plain numpy/autograd arrays; the SimulationResult object
        # from wfm(...) breaks autograd tracking and can't be used here).
        def active_aep_only(x, y):
            res = wfm._run(x, y, h=None, type=type_full, wd=wd_arg, ws=ws_arg, TI=ti_arg,
                            time=time_arg, n_cpu=1)
            _, _, power_ilk, _, localWind, _ = res
            P_ilk = localWind.P_ilk
            power_active = power_ilk[:n_active]
            P_active = P_ilk[:n_active]
            return (power_active * P_active * 24 * 365 * 1e-9).sum()

        grad_active = autograd(active_aep_only, True, argnum=[0, 1])
        daep_dx, daep_dy = grad_active(x_full, y_full)
        daep_dx = np.array(daep_dx)[:n_active]
        daep_dy = np.array(daep_dy)[:n_active]

        # 2. Return the exact analytical gradients to OpenMDAO's global scale
        J['aep', 'x'] = np.array(daep_dx).reshape(1, -1)
        J['aep', 'y'] = np.array(daep_dy).reshape(1, -1)