from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"

YEAR_START = 2026
YEAR_END = 2050
HORIZON_YEARS = YEAR_END - YEAR_START

# Speculative mosaic fill (layout.generate_scenario / spacing.from_density):
# rho_target ~ Triangular(min, mode, max) MW/km^2, calibrated against NREL's real
# offshore project capacity densities (min/max are real observed extremes; mode
# kept at 4.0, not the table's mean, so the peak doesn't drift with the tails).
# Full calibration details in SCENARIOS.md.
RHO_TARGET_TRIANGULAR = (2.1, 4.0, 9.03)  # (min, mode, max) MW/km2
# Acceptance band for the spacing_d derived from rho_target. A modeling choice
# (literature range for large arrays), not a regulatory limit.
SPACING_D_BOUNDS = (5.5, 12.0)
# Speculative fill stops once cumulative capacity reaches this target (small
# overshoot from the last parcel is fine). Roughly NREL's 30GW buildout figure.
SPECULATIVE_TARGET_MW = 30_000.0

TURBINE_YAML_FILES = {
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

TIMELINE_GAMMA = 0.12

# Segment length for orchestrator._checkpoints(), splitting YEAR_START..YEAR_END
# into fixed chunks (e.g. 2026,2031,2036,2041,2046,2050 -- last one may be shorter).
SEGMENT_CHECKPOINT_YEARS = 5
