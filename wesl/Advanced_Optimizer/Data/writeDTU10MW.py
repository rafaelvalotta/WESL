import numpy as np
import yaml
from pathlib import Path
from py_wake.examples.data.dtu10mw import DTU10MW

wt = DTU10MW()
ws = np.arange(3.0, 25.5, 0.5)
power = wt.power(ws)
ct = wt.ct(ws)

data = {
    "name": "DTU 10MW Reference Wind Turbine",
    "performance": {
        "rated_power": float(power.max()),
        "cutin_wind_speed": 3.0,
        "cutout_wind_speed": 25.0,
        "power_curve": {
            "power_values": [float(p) for p in power],
            "power_wind_speeds": [float(w) for w in ws],
        },
        "Ct_curve": {
            "Ct_values": [float(c) for c in ct],
            "Ct_wind_speeds": [float(w) for w in ws],
        },
    },
    "hub_height": float(wt.hub_height()),
    "rotor_diameter": float(wt.diameter()),
}

out_path = Path(__file__).resolve().parent / "dtu10mw.yaml"
with open(out_path, "w") as f:
    yaml.dump(data, f, default_flow_style=None, sort_keys=False)
