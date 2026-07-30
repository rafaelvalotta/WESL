import numpy as np
import openmdao.api as om

class LCOEComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt', types=int)
        self.options.declare('discount_rate', default=0.0661)
        self.options.declare('lifetime', default=25)
        self.options.declare('rated_power_kw', types=float)
        # NREL Fuchs et al. (2024), North Atlantic mid scenario, 2025: total CapEx $6,260/kW,
        # OpEx $85/kW-yr. capex_fixed_per_kw = 6260 - foundation - array cable (already
        # modeled separately here) to avoid double-counting those two components.
        # self.options.declare('capex_fixed_per_kw', default=5700.0 * 0.924) # EUR (old, USD->EUR)
        self.options.declare('capex_fixed_per_kw', default=5700.0) # USD, no EUR conversion
        # self.options.declare('opex_annual_per_kw', default=85.0 * 0.924) # EUR (old, USD->EUR)
        self.options.declare('opex_annual_per_kw', default=85.0) # USD, no EUR conversion
        self.options.declare('liquidation_proceeds', default=0.0)

    def setup(self):
        n_wt = self.options['n_wt']
        
        self.add_input('aep', val=1.0, units='MW*h')
        self.add_input('cost_foundations', val=np.zeros(n_wt))
        self.add_input('cost_cables', val=0.0)
        
        self.add_output('lcoe', val=0.0)
        
        self.declare_partials('lcoe', 'aep')
        self.declare_partials('lcoe', 'cost_cables')
        self.declare_partials('lcoe', 'cost_foundations')

    def compute(self, inputs, outputs):
        n_wt = self.options['n_wt']
        d = self.options['discount_rate']
        life = self.options['lifetime']
        capex_f = self.options['capex_fixed_per_kw'] * self.options['rated_power_kw']
        opex_a = self.options['opex_annual_per_kw'] * self.options['rated_power_kw']
        lp = self.options['liquidation_proceeds']

        aep = inputs['aep'][0] if isinstance(inputs['aep'], np.ndarray) else inputs['aep']
        capex_foundations = np.sum(inputs['cost_foundations'])
        capex_cables = inputs['cost_cables']
        
        crf = d / (1 - (1 + d) ** -life) # capital recovery factor
        npv = (capex_f * n_wt + capex_foundations + capex_cables + lp * n_wt) * crf + opex_a * n_wt
        
        if aep <= 1e-16:
            outputs['lcoe'] = 1e6
        else:
            outputs['lcoe'] = npv / aep

    def compute_lcoe(self, aep_mwh, cost_foundations, cost_cables):
        """
        Standalone LCOE evaluation from a given AEP (MWh) and cost breakdown, independent
        of the OpenMDAO compute() plumbing -- mirrors compute()'s formula, keep in sync.
        Lets callers plug in a deterministic AEP (e.g. AEPComp.compute_deterministic_aep)
        without touching compute()/compute_partials(), which the SGD driver differentiates
        through every iteration.
        """
        n_wt = self.options['n_wt']
        d = self.options['discount_rate']
        life = self.options['lifetime']
        capex_f = self.options['capex_fixed_per_kw'] * self.options['rated_power_kw']
        opex_a = self.options['opex_annual_per_kw'] * self.options['rated_power_kw']
        lp = self.options['liquidation_proceeds']

        capex_foundations = float(np.sum(cost_foundations))
        crf = d / (1 - (1 + d) ** -life)
        npv = (capex_f * n_wt + capex_foundations + cost_cables + lp * n_wt) * crf + opex_a * n_wt

        if aep_mwh <= 1e-16:
            return 1e6
        return npv / aep_mwh

    def compute_partials(self, inputs, J):
        n_wt = self.options['n_wt']
        d = self.options['discount_rate']
        life = self.options['lifetime']
        capex_f = self.options['capex_fixed_per_kw'] * self.options['rated_power_kw']
        opex_a = self.options['opex_annual_per_kw'] * self.options['rated_power_kw']
        lp = self.options['liquidation_proceeds']

        aep = inputs['aep'][0] if isinstance(inputs['aep'], np.ndarray) else inputs['aep']
        capex_foundations = np.sum(inputs['cost_foundations'])
        capex_cables = inputs['cost_cables']
        
        crf = d / (1 - (1 + d) ** -life)
        u = (capex_f * n_wt + capex_foundations + capex_cables + lp * n_wt) * crf + opex_a * n_wt
        v = aep
        
        if v <= 1e-16:
            J['lcoe', 'aep'] = 0.0
            J['lcoe', 'cost_cables'] = 0.0
            J['lcoe', 'cost_foundations'] = np.zeros_like(inputs['cost_foundations'])
            return

        J['lcoe', 'aep'] = -u / (v ** 2) # quotient rule
        J['lcoe', 'cost_cables'] = crf / v
        J['lcoe', 'cost_foundations'] = np.ones_like(inputs['cost_foundations']) * (crf / v)