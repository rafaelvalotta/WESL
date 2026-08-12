"""The 2 literature-validated wake models used from here on -- ported from
Risk_Investigation/cluster_wake_map_orange.ipynb, which measured them against the full
cluster: Nygaard 2022 (TurbOPark/TurboGaussian) and Blondel & Cathelain 2020
(SuperGaussian), both built via py_wake.literature so the parametrization matches the
calibration used in the benchmark papers (Rodrigues & Moura 2025; Moura & Rodrigues
2025) -- not bare deficit-model defaults. NOJ is deliberately excluded: the same
notebook's sibling analysis (cluster_aep_wake_analysis.ipynb) showed it systematically
underestimates cluster impact (~0.9% vs. Nygaard's ~16-20%), consistent with the 012045
benchmark paper. These two are the "conservative for far wake" pair the project has
settled on.

Design choice carried over from this session's planning discussion: with only 2 models,
the wake-model axis is NOT a random per-trial draw. The orchestrator runs every trial
through BOTH and logs both results, so "how much does the outcome vary due to wake model
choice" can be read directly off paired results for the same scenario, instead of being
conflated with scenario-to-scenario noise (which a random per-trial pick would do).
"""
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez

WFM_BUILDERS = {
    "Nygaard_TurboGaussian": lambda s, t: Nygaard_2022(s, t),
    "SuperGaussian": lambda s, t: Blondel_Cathelain_2020(s, t, turbulenceModel=CrespoHernandez()),
}
