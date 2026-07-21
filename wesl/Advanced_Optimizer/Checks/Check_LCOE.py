import numpy as np
import openmdao.api as om
from Components.lcoe_comp import LCOEComp

if __name__ == "__main__":
    print("=== [TEST] INITIALIZING LCOE COMPONENT VALIDATION ===")
    
    N_WT = 10 # 10 active turbines
    
    # Simulating standard values from components based on your tests
    mock_aep = 958.0011 # MWh from Check_PYWAKE
    mock_foundations = np.ones(N_WT) * (268036205.08 / N_WT) # Array from foundations test
    mock_cables = 1.2e7 # Simulated cable network cost from OptiWindNet
    
    prob = om.Problem()
    prob.model.add_subsystem(
        'lcoe_engine',
        LCOEComp(n_wt=N_WT),
        promotes=['*']
    )
    
    prob.setup()
    
    # Feeding input variables
    prob.set_val('aep', mock_aep)
    prob.set_val('cost_foundations', mock_foundations)
    prob.set_val('cost_cables', mock_cables)
    
    prob.run_model()
    
    lcoe_result = prob.get_val('lcoe')[0]
    print("======================================================================")
    print(f" Computed LCOE Metric: {lcoe_result:,.4f} EUR/MWh")
    print("======================================================================\n")
    
    # Validating exact algebraic jacobians against finite differences
    print("Checking analytical partials against Finite Differences...")
    prob.check_partials(compact_print=True, method='fd', step=1e-3)