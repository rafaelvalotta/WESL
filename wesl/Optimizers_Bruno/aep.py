# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import numpy as np
import openmdao.api as om
from py_wake.utils.gradients import autograd  

class AEPComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("wake_model", desc="PyWake model instance")
        self.options.declare("wt_x", types=np.ndarray)
        self.options.declare("wt_y", types=np.ndarray)
        self.options.declare("aep_ref", default=1.0)
        self.options.declare('recorder', default=None)
        self.options.declare("n_cpu", default=None)

    def setup(self):
        n = len(self.options["wt_x"])
        self.add_input("x", val=np.zeros(n), units="m")
        self.add_input("y", val=np.zeros(n), units="m")
        self.add_output("aep", val=0.0)
        self.declare_partials("aep", "x", rows=np.zeros(n, int), cols=np.arange(n))
        self.declare_partials("aep", "y", rows=np.zeros(n, int), cols=np.arange(n))

    def compute(self, inputs, outputs):
        wfm = self.options["wake_model"]
        aep_ref = float(self.options["aep_ref"])
        n_cpu = self.options["n_cpu"]
        
        # 🆕 EXATO do go.py aep_func2():
        wd = np.arange(360)      # 360 direções
        ws = np.arange(3, 25, 1) # 3..24 m/s GRID
        aep = wfm(inputs["x"], inputs["y"], wd=wd, ws=ws, n_cpu=n_cpu).aep().sum()
        
        outputs["aep"] = aep / max(aep_ref, 1e-16)
        
        rec = self.options['recorder']
        if rec is not None:
            rec.log()

    def compute_partials(self, inputs, partials):
        wfm = self.options["wake_model"]
        x, y = inputs["x"], inputs["y"]
        n_cpu = self.options["n_cpu"]
        

        wd = np.arange(360)
        ws = np.arange(3, 25, 1)
        grads = wfm.aep_gradients(
            gradient_method=autograd,
            wrt_arg=["x", "y"],
            x=x, y=y,
            wd=wd, ws=ws,     
            n_cpu=n_cpu
        )
        
        aep_ref = float(self.options["aep_ref"])
        daep_x = grads[0, :] / max(aep_ref, 1e-16)
        daep_y = grads[1, :] / max(aep_ref, 1e-16)
        
        partials["aep", "x"] = -daep_x  
        partials["aep", "y"] = -daep_y


class AEPCompStochastic(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("wake_model", desc="PyWake model instance")
        self.options.declare("site", desc="PyWake site instance")
        self.options.declare("wt_x", types=np.ndarray)
        self.options.declare("wt_y", types=np.ndarray)
        self.options.declare("aep_ref", default=1.0, desc="Normalization factor")
        self.options.declare("recorder", default=None)
        self.options.declare("n_cpu", default=None, allow_none=True,
                             desc="Number of CPUs for aep_gradients (PyWake)")
        self.options.declare("K", types=int, default=50,
                             desc="Number of Monte Carlo wind samples per evaluation")

    def setup(self):
        n = len(self.options["wt_x"])
        self.add_input("x", val=np.zeros(n), units="m")
        self.add_input("y", val=np.zeros(n), units="m")
        self.add_output("aep", val=0.0)
        self.declare_partials("aep", "x", rows=np.zeros(n, int), cols=np.arange(n))
        self.declare_partials("aep", "y", rows=np.zeros(n, int), cols=np.arange(n))
        self._last_wd = None
        self._last_ws = None

    def _sampling(self):
        K = self.options["K"]
        site = self.options["site"]
        

        dirs = np.arange(360)  # 0,1,2,...,359
        

        x_init = self.options["wt_x"]
        y_init = self.options["wt_y"]
        

        lw = site.local_wind(x_init, y_init, wd=dirs)
        freqs = lw.Sector_frequency_ilk[0, :, 0]  # shape=(360,)
        As    = lw.Weibull_A_ilk[0, :, 0]
        ks    = lw.Weibull_k_ilk[0, :, 0]
        
        # Normalize
        freqs = freqs / freqs.sum()
        
        idx = np.random.choice(np.arange(360), size=K, p=freqs)
        wd = dirs[idx]
        A  = As[idx]
        k  = ks[idx]
        ws = A * np.random.weibull(k)
        
        return wd, ws

    def compute(self, inputs, outputs):
        wfm = self.options["wake_model"]
        aep_ref = float(self.options["aep_ref"])
        x = inputs["x"]
        y = inputs["y"]

        wd, ws = self._sampling()
        self._last_wd = wd
        self._last_ws = ws

        sim_res = wfm(x=x, y=y, wd=wd, ws=ws, time=True)
        aep_mc = sim_res.aep().sum()

        outputs["aep"] = aep_mc / max(aep_ref, 1e-16)

        rec = self.options["recorder"]
        if rec is not None:
            rec.log()

    def compute_partials(self, inputs, partials):
        wfm = self.options["wake_model"]
        x = inputs["x"]
        y = inputs["y"]
        n_cpu = self.options["n_cpu"]
        aep_ref = float(self.options["aep_ref"])

        if self._last_wd is None or self._last_ws is None:
            wd, ws = self._sampling()
        else:
            wd, ws = self._last_wd, self._last_ws

        grads = wfm.aep_gradients(
            gradient_method=autograd,
            wrt_arg=["x", "y"],
            x=x,
            y=y,
            wd=wd,
            ws=ws,
            time=True,
            n_cpu=n_cpu,
        )

        daep_x = grads[0, :] / max(aep_ref, 1e-16)
        daep_y = grads[1, :] / max(aep_ref, 1e-16)

        partials["aep", "x"] = -daep_x
        partials["aep", "y"] = -daep_y

