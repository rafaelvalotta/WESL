# The two study scenarios -- full rationale in SCENARIOS.md.
# cluster.py and layout.py branch on a ScenarioConfig; the rest of the pipeline
# doesn't care which scenario is running.
from dataclasses import dataclass

import cluster

SELF_FARM = "New England Wind 2 South"
SELF_TURBINE = "13 MW"
SELF_SPACING_D = 8


@dataclass(frozen=True)
class ScenarioConfig:
    key: str                        # "scenario_1" / "scenario_2" -- used as the results/ subfolder
    label: str                      # human-readable, for logs/docs
    self_farm: str
    self_turbine: str
    self_spacing_d: int
    fixed_farm_names: frozenset     # excluded from mosaic eligibility; drawn as committed farms
    cluster_mode: str               # "open" (scenario 1) | "fixed_once" (scenario 2)


SCENARIO_1 = ScenarioConfig(
    key="scenario_1",
    label="Open buildout -- real-today backbone only",
    self_farm=SELF_FARM, self_turbine=SELF_TURBINE, self_spacing_d=SELF_SPACING_D,
    fixed_farm_names=frozenset(cluster.REAL_LAYOUT_FARMS | {SELF_FARM}),
    cluster_mode="open",
)

SCENARIO_2 = ScenarioConfig(
    key="scenario_2",
    label="Locked cluster -- as-built-today, fixed once",
    self_farm=SELF_FARM, self_turbine=SELF_TURBINE, self_spacing_d=SELF_SPACING_D,
    fixed_farm_names=frozenset(cluster.BOUNDARY_NAMES),
    cluster_mode="fixed_once",
)

SCENARIOS = {"scenario_1": SCENARIO_1, "scenario_2": SCENARIO_2}
