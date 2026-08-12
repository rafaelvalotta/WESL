"""3 literature-validated wake models, run every trial (never randomly sampled -- see
PIPELINE_DESIGN_v2.md §2.5 for why: with only 3 models the ensemble is fully enumerated,
so "how much does wake model choice move the outcome" never gets confounded with
scenario-to-scenario noise). TurboPark/Nygaard and SuperGaussian carried over from v1
unchanged.
"""
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020, Bastankhah_PorteAgel_2014
from py_wake.turbulence_models import CrespoHernandez

from settings import BASTANKHAH_K

WFM_BUILDERS = {
    "Nygaard_TurboGaussian": lambda s, t: Nygaard_2022(s, t),
    "SuperGaussian": lambda s, t: Blondel_Cathelain_2020(s, t, turbulenceModel=CrespoHernandez()),
    # Bastankhah_PorteAgel_2014, unlike the other two, isn't self-calibrated -- it needs
    # an explicit wake-expansion k. Using py_wake's own DTU reference-example value.
    "Bastankhah_PorteAgel_2014": lambda s, t: Bastankhah_PorteAgel_2014(s, t, k=BASTANKHAH_K),
}
