#created to manage all the important values together 
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"

YEAR_START = 2026
YEAR_END = 2050
HORIZON_YEARS = YEAR_END - YEAR_START



RHO_TARGET_TRIANGULAR = (2.1, 4.0, 9.03)  # (min, mode, max) MW/km2
# kept at 4.0, not the nrel's table's mean, the distribution was expected to be asymmetric

SPACING_D_BOUNDS = (5.5, 12.0)
SPECULATIVE_TARGET_MW = 30_000.0

TURBINE_YAML_FILES = {  # Mapping of turbine names to their YAML configuration files (WINDIO ontology)
    "11 MW": "sg11_200dd.yaml",
    "13 MW": "haliadex_13mw.yaml",
    "15.5 MW": "haliadex_15.5.yaml",
    "18 MW": "generic_18mw.yaml",
    "22 MW": "turbines.yaml",
}
TURBINE_TAU_MW = 2.5
TURBINE_RATING_SHAPE = 1.0
TURBINE_TYPICAL_RATING_START_MW = 11.5
TURBINE_TYPICAL_RATING_END_MW = 21.0

TIMELINE_GAMMA = 0.12# which determines how quickly the timeline new farms are added

SEGMENT_CHECKPOINT_YEARS = 5 #5 segments (photos) of the time line, where we call Pywake to simulate 
