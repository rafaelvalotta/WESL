"""Every locked decision from PIPELINE_DESIGN_v2.md, in one place.

Named `settings.py`, not `config.py`, on purpose: Risk_Investigation/config.py already
exists (aep_simulation.py's config) and sits on the same sys.path stack as this package
-- two modules both called "config" would collide in sys.modules (Python caches imports
by name, not by folder), and whichever one loads first would silently shadow the other.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"

# --- geography, PIPELINE_DESIGN_v2.md §2.1-2.2 ---
RADIUS_KM = 120.0
FIXED_BOTTOM_MAX_DEPTH_M = 80.0
MAX_CONNECTION_DISTANCE_M = 70_000.0

# --- self / focus farm, §2.6, §2.8 ---
FOCUS_FARM = "SouthCoast Wind"
HORIZON_YEARS = 25.0

# --- turbine placement, §2.3-2.4 ---
SPACING_D = 8.0  # fixed, no longer sampled
TURBINE_YAML_FILES = {
    "SG11.0-200DD": "sg11_200dd.yaml",
    "Haliade-X 13MW": "haliadex_13mw.yaml",
    "Haliade-X 15.5MW": "haliadex_15.5.yaml",
    "IEA 2200-22MW": "turbines.yaml",
}

# --- tiers as Poisson arrival rate (farms/year), §2.7 ---
TIER_LAMBDA = {"low": 0.5, "medium": 1.0, "high": 2.0}

# --- new-farm size draw, same empirical distribution v1's fragment_budget used ---
REAL_FARM_SIZES_MW = [30, 132, 572, 715, 780, 806, 924, 1833, 2080]

# --- wake models, §2.5 ---
BASTANKHAH_K = 0.0324555  # py_wake's own DTU reference default (Niayifar calib., Horns Rev)

# --- placement engine knobs, unchanged from v1 ---
MIN_FARM_SEPARATION_M = 800.0
TRIES_PER_STEP = 40
CAPACITY_FRACTIONS = [1.0, 0.75, 0.5, 0.25]
MIN_FARM_MW_FLOOR = 30.0
DISTANCE_BANDS_KM = [(3, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 75),
                      (75, 90), (90, 105), (105, 120)]  # trimmed to RADIUS_KM=120
ANGLE_SECTORS_DEG = list(range(0, 360, 30))
