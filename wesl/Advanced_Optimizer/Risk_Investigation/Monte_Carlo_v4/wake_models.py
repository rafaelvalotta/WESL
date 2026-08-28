from py_wake.literature.turbopark import Nygaard_2022
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020, Bastankhah_PorteAgel_2014
from py_wake.turbulence_models import CrespoHernandez

BASTANKHAH_K = 0.0324555

MODELS = {
    "turbopark": lambda site, wts: Nygaard_2022(site, wts),
    "supergaussian": lambda site, wts: Blondel_Cathelain_2020(site, wts, turbulenceModel=CrespoHernandez()),
    "bastankhah": lambda site, wts: Bastankhah_PorteAgel_2014(site, wts, k=BASTANKHAH_K, turbulenceModel=CrespoHernandez()),
}

ACTIVE_MODEL = "supergaussian"
# Using supergaussian instead of turbopark for cost, not accuracy -- a turbopark
# trial costs ~4x as much (see TRIAL_RECORD.md). Plan: supergaussian first for
# throughput, bastankhah next, turbopark later.
