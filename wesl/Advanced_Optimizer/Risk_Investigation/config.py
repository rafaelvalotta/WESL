"""Central config for the cluster AEP pipeline."""
from pathlib import Path

from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
SYSTEM_YAML = DATA_DIR / "vineyard_revolution_system.yaml"

ERA5_DIR = BASE_DIR / "Project_Monte_Carlo" / "era5_vineyard"
SITE_YEAR = "2023"
ALL_YEARS = [str(y) for y in range(2000, 2024)]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Farms with a real turbine layout; everything else gets a synthetic grid fill.
REAL_LAYOUT_FARMS = {"Vineyard Wind", "Revolution Wind", "South Fork Wind", "Sunrise Wind South"}
FOCUS_FARM = "Vineyard Wind"

# Too small to hold more than 1 turbine at SYNTHETIC_SPACING_M -- skip entirely.
EXCLUDED_FARMS = {"Revolution Wind North"}

SYNTHETIC_TURBINE_SPEC = DATA_DIR / "haliadex_13mw.yaml"
SYNTHETIC_SPACING_M = 1840.0  # Vineyard's real median turbine spacing

# DTU's literature-validated wrapper for Blondel & Cathelain (2020) -- takes (site, windTurbines)
# directly rather than a bare deficit/superposition pair. See py_wake.literature.gaussian_models.
WFM_MODEL = Blondel_Cathelain_2020
TURBULENCE_MODEL = CrespoHernandez

SIM_WD = list(range(0, 360, 30))   # 12 directions
SIM_WS = list(range(3, 26, 1))     # 3-25 m/s

FLOWMAP_WS_SNAPSHOT = 10
FLOWMAP_GRID_RESOLUTION = 500
