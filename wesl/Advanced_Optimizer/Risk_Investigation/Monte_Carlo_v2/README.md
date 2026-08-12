# Monte_Carlo_v2

Implements `PIPELINE_DESIGN_v2.md`. `../Monte_Carlo/` (v1) is untouched, kept as
reference.

## Files

| File | Role |
|---|---|
| `settings.py` | every locked constant (radius, spacing, turbines, tiers/lambda, horizon) |
| `climate.py`, `climate_fit.py` | wind-rose PCA emulator -- unchanged from v1 |
| `layout.py` | eligibility grid, committed cluster (12 leases), self (SouthCoast Wind) |
| `timeline.py` | Poisson arrival sampling + segment boundaries -- the time axis |
| `scenario_bridge.py`, `turbines_bridge.py` | farm/turbine schema glue -- unchanged from v1 |
| `wake_models.py` | 3-model ensemble (TurboPark, SuperGaussian, Bastankhah 2014) |
| `instrumentation.py` | logging -- 5 tables (`trials`, `arrivals`, `segments`, `aep_summary`, `turbine_assignments`) |
| `orchestrator.py` | main entry point, per-trial loop |
| `plot_timeline_demo.py` | arrival-process plots, no AEP needed -- see below |
| `hpc/` | Chimera submission layer |

## Quick start

```bash
conda run -n Wind_2200 python orchestrator.py --n-trials 5 --tier high
```

## Getting the arrival-timeline plots without running AEP

The plots of "how many new farms show up over 25 years, per tier" and "installed MW vs.
year" only need `layout.populate_scenario` -- no climate draw, no PyWake call. Get them
now, before running a single expensive AEP trial:

```bash
conda run -n Wind_2200 python plot_timeline_demo.py --n-draws 15
```

## Before a production AEP campaign

1. Run a small shakedown (few seeds, `high` tier especially) and check `arrivals.csv`'s
   `success` column doesn't degrade badly late in the horizon (placement feasibility --
   PIPELINE_DESIGN_v2.md §5).
2. Time a handful of real trials to size the LHS draw count per (wake model x tier) cell
   -- v2 trials cost more than v1's (see `hpc/README.md`).
