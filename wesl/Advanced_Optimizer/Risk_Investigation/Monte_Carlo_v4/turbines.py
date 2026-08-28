import numpy as np
import yaml
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

import settings


def _load(fname):
    with open(settings.DATA_DIR / fname) as f:
        return yaml.safe_load(f)


_SPECS = {label: _load(fname) for label, fname in settings.TURBINE_YAML_FILES.items()}
NAMES = list(_SPECS.keys())
RATINGS_MW = np.array([s["performance"]["rated_power"] / 1e6 for s in _SPECS.values()])
_MW_BY_NAME = dict(zip(NAMES, RATINGS_MW))


def rated_mw(name):
    return _MW_BY_NAME[name]


def typical_rating(year):
    frac = (year - settings.YEAR_START) / settings.HORIZON_YEARS
    r0, r1 = settings.TURBINE_TYPICAL_RATING_START_MW, settings.TURBINE_TYPICAL_RATING_END_MW
    return r0 + (r1 - r0) * frac ** settings.TURBINE_RATING_SHAPE


def shares(year, tau=settings.TURBINE_TAU_MW):
    target = typical_rating(year)
    w = np.exp(-((target - RATINGS_MW) ** 2) / (2 * tau ** 2))
    return w / w.sum()


def draw(rng, year):
    return rng.choice(NAMES, p=shares(year))


def diameter(name):
    return _SPECS[name]["rotor_diameter"]


def _to_pywake(spec):
    pc = spec["performance"]["power_curve"]
    ctc = spec["performance"]["Ct_curve"]
    return WindTurbine(name=spec["name"], diameter=spec["rotor_diameter"], hub_height=spec["hub_height"],
                        powerCtFunction=PowerCtTabular(ws=pc["power_wind_speeds"], power=pc["power_values"],
                                                        power_unit="W", ct=ctc["Ct_values"]))


def to_pywake(name):
    return _to_pywake(_SPECS[name])


def to_pywake_raw(spec):
    return _to_pywake(spec)
