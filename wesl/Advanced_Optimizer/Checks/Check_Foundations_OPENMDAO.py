import numpy as np
import openmdao.api as om
import windIO
from pathlib import Path
from Components.foundations import FoundationsComp, MonopileCurveFitter


if __name__ == "__main__":
    BASE_DIR = Path("/Users/brunoboer/Documents/Software/Test_Wesl_jun23/wesl/Advanced_Optimizer")
    SYSTEM_YAML = BASE_DIR / "Data" / "vineyard_revolution_system.yaml"
    BATHY_NC = Path("/Users/brunoboer/Documents/Software/Wind_2200/data/bathymetry_us.nc")
    
    system_dat = windIO.load_yaml(SYSTEM_YAML)
    rd_22mw = float(system_dat['wind_farm'][1]['turbines']['rotor_diameter'])
    coefficients_22mw = MonopileCurveFitter.fit_surrogate(rd=rd_22mw)
    
    # 10 active turbines directly in full real UTM scale
    x_real = np.array(system_dat['wind_farm'][1]['layouts'][0]['coordinates']['x'])[:10]
    y_real = np.array(system_dat['wind_farm'][1]['layouts'][0]['coordinates']['y'])[:10]
    
    prob = om.Problem()
    prob.model.add_subsystem(
        'foundations_engine',
        FoundationsComp(
            n_wt=len(x_real), bathy_nc_path=str(BATHY_NC), poly_coefficients=coefficients_22mw
        ),
        promotes=['*']
    )
    
    prob.setup()
    prob.set_val('x', x_real) # setting real positions
    prob.set_val('y', y_real)
    prob.run_model()
    
    print(f"Total Foundations Cost: {np.sum(prob.get_val('cost_foundations')):,.2f} EUR")
    prob.check_partials(compact_print=True, method='fd', step=1.0) # check analytical gradients