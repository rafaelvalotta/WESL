"""WindIO turbine specs -> PyWake WindTurbine objects."""
import yaml

from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


def build_pywake_turbine(turbine_data):
    pc = turbine_data["performance"]["power_curve"]
    ctc = turbine_data["performance"]["Ct_curve"]
    return WindTurbine(
        name=turbine_data["name"],
        diameter=turbine_data["rotor_diameter"],
        hub_height=turbine_data["hub_height"],
        powerCtFunction=PowerCtTabular(
            ws=pc["power_wind_speeds"], power=pc["power_values"],
            power_unit="W", ct=ctc["Ct_values"],
        ),
    )


def load_turbine_file(path):
    with open(path) as f:
        return build_pywake_turbine(yaml.safe_load(f))
