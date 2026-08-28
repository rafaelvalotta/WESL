# Monte Carlo v3 -- Pipeline Reference

**2026-08-26: forked into `Monte_Carlo_v4`, which adds a required `scenario`
argument (`scenario_1`/`scenario_2`) threaded through `cluster.py`/`layout.py`/
`orchestrator.py`, plus `layout.RADIUS_KM` 150->100 -- see `SCENARIOS.md` for
what's new. Everything below still describes v4's shared pipeline accurately;
only read this section as "v3" for historical/v3-specific results.**

One trial = one call to `orchestrator.run_trial(seed, logger, scenario)`. It generates a full
25-year scenario (2026-2050) for `self` (SouthCoast Wind), steps through it in
time-ordered segments, and evaluates AEP at each segment with PyWake. This doc
describes each module as it stands today -- what runs, what distribution each
random draw follows, and exactly how PyWake is called.

## settings.py

Constants shared by every module. `YEAR_START=2026`, `YEAR_END=2050`,
`HORIZON_YEARS=24`. `N_RANGE=(1,16)` -- speculative farm count, drawn uniform
discrete. `SPACING_OPTIONS_D=[4..10]`. `TURBINE_YAML_FILES` -- the 5 turbine specs
(11/13/15.5/18/22 MW), `TURBINE_TAU_MW=2.5`, `TURBINE_RATING_SHAPE=1.0`. `TIMELINE_GAMMA=0.12`.

## cluster.py -- the committed cluster (always present)

13 real BOEM lease boundaries (`LEASE_TEMPLATES`, from `site_us.yaml`). 4 have a
real, already-built turbine layout (Vineyard Wind, Revolution Wind, South Fork
Wind, Sunrise Wind South) -- fixed positions, present from year 0, turbine spec
read straight from `wind_farm_us.yaml` (`REAL_TURBINE_SPEC`). `SouthCoast Wind` is
`self` -- also present from year 0, draws one turbine + spacing (same mechanism as
below) but never arrives later. The remaining 8 leases (`INFILL_FARMS`) are
real and certain to be built, but not yet built: each draws its own arrival year
from `timeline.draw_arrival_years`, then a turbine at that year and a spacing --
same distributions as the speculative farms, just guaranteed to exist (only the
timing is random, not the existence).

`generate_cluster(rng)` returns all 13 farms as dicts (`is_self`, `has_real_layout`,
`arrival_year`, `turbine`, `spacing_d`, `polygon` or real `x`/`y`).

## layout.py -- speculative farm generation (mosaic)

**2026-08-26: `_load_eligibility()` split into `_load_base_grid()` (scenario-
independent bathymetry/federal/radius grid, still computed once at import) +
`eligible_for(scenario)` (scenario-dependent exclusion mask, memoized per
scenario key) -- and `generate_scenario(rng)` below is fully superseded by the
capacity-target-driven redesign. See SCENARIOS.md for both; this paragraph is
history only.**

Wraps `Layout_Generation/mosaic.py` (Poisson-disk dart-throwing + weighted Voronoi
+ gap buffer). `_load_eligibility()` builds the eligible-water mask once at import
(federal + <80m depth, smoothed + within 150km, cluster excluded) -- not per trial.
`generate_scenario(rng)`: runs `mosaic.generate_mosaic` fresh (own random draw),
picks N parcels (`N_RANGE`, discrete uniform 1-15), draws each one's arrival year
(`timeline.draw_arrival_years`), turbine (at that year), spacing. Independent of
the cluster's own arrival draws -- two separate calendars, merged later in
`orchestrator.py`.

## climate.py -- wind resource (3-knob whole-rose model, 2026-08-27 redesign)

**Superseded the 39-independent-parameter design below on 2026-08-27** -- with
that many independently-drawn factors, none of them correlated strongly with
outcomes (too diluted). Back to a small number of shared knobs (like the
pre-2026-08-21 design), but keeping that redesign's core principle: every
knob's range is MEASURED via bootstrap, never an assumed magnitude. Requested
explicitly (professor's direction): no PCA -- 3 physically-interpretable knobs
applied to the whole "mean rose" at once, not a data-driven dimensionality
reduction.

`climate_fit.py`'s base fit (5-mode von Mises for direction + 12-sector
Weibull for speed, on the pooled 24-year ERA5 data -- unchanged from the
39-param design below) is the **"mean rose"**, held fixed. Its own
`n_boot=40` block-by-year bootstrap loop (also unchanged) now ALSO computes 3
whole-rose aggregate statistics per resample, straight from that resample's
raw (speed, direction) data (`_rose_summary`) -- not derived from the fitted
mixture:
- **rotation**: resample's raw circular-mean direction minus the base pooled
  fit's, in degrees (same measurement the retired pre-2026-08-21 design's
  `delta_mu_std_deg`/`delta_mu_max_deg` made, just recomputed against the
  current bootstrap).
- **speed**: ratio of resample's mean speed to the base pooled mean speed.
- **spread**: ratio of a single-mode-equivalent concentration (`_banerjee_kappa`
  applied to the whole direction sample as if unimodal, not per-component) to
  the same measure on the base pooled fit.

Cached as `rotation_deg_std`/`_max`, `speed_ratio_std`/`_min`/`_max`,
`spread_ratio_std`/`_min`/`_max` in `cache/wind_climate_basis.npz`, alongside
the unchanged 39-parameter base values (still needed as the mean rose).
Measured 2026-08-27: rotation std=1.48deg (max=3.94deg) -- in the same range
as the 39-param design's individual per-mode `vm_mus_std` (0.75-3.02deg, so
this aggregate isn't off-scale); speed ratio std=0.005 (range
[0.991, 1.010]); spread ratio std=0.026 (range [0.942, 1.071]). All three are
tight -- a block-bootstrap CI on a 24-year mean is a precise estimate, by
construction; see "What these knobs represent" below.

`climate.py`'s `sample_scenario(rng)` draws exactly 3 numbers per trial:
`rotation_deg ~ Normal(0, rotation_deg_std)` capped at
`+/-rotation_deg_max`; `speed_multiplier ~ Normal(1, speed_ratio_std)` capped
`[speed_ratio_min, speed_ratio_max]`; `spread_multiplier ~ Normal(1,
spread_ratio_std)` capped `[spread_ratio_min, spread_ratio_max]`. Applied to
the base rose: `mus = VM_MUS + radians(rotation_deg)` (all 5 modes shift
together -- rigid rotation), `kappas = VM_KAPPAS * spread_multiplier` (all 5
scale together -- flatten/peak the whole rose), `sector_C = SECTOR_C *
speed_multiplier` (all 12 sectors scale together -- whole-rose speed level).
`weights` and `sector_K` (Weibull shape) stay at their base fitted values,
unperturbed -- this trades away the weight-redistribution and per-sector-shape
degrees of freedom the old 39-param/4-knob designs had, by design (3 knobs,
not 4 or 39). The hard caps mean **no draw can exceed what the real 24 years
of ERA5 data, block-resampled, actually produced** -- deliberately guards
against an overestimated rose (e.g. an inflated speed knob) faking a wake-loss
or production number the site has never actually shown.

**What these knobs represent**: a bootstrap CI on the 24-year mean is
answering "how precisely do we know today's long-run climate," not "how much
could the climate plausibly differ by 2050" -- the same interpretation the
retired 39-param design already used (see [[wesl-risk-study-nsf-aim2]]).
Accepted here for consistency, not re-litigated.

**`orchestrator.py` logging (revised 2026-08-27, same day)**: `trials.parquet`
logs ONLY the 3 knobs (`climate_rotation_deg`, `climate_speed_multiplier`,
`climate_spread_multiplier`) -- the 39 derived `vm_weight/mu/kappa/sector_C/
sector_K` values still get computed internally (`climate.sample_scenario`
returns them, `scenario_to_site_params` needs them to build the PyWake site)
but are no longer persisted per trial; logging them was redundant (fully
determined by the 3 knobs + the fixed base rose) and verbose. This also
removed the ONLY reason `spatial/` (the distance/angle-weighted exposure
toolkit, [[mc-v3-distance-angle-density-plan]]) needed those 39 columns --
and separately, `spatial/` itself was removed from `Monte_Carlo_v4` entirely
on 2026-08-27 (judged no longer worth its compute now that the speculative
buildout is capacity-target-driven rather than open-ended -- see SCENARIOS.md
"Speculative fill redesign"). `Monte_Carlo_v3/spatial/` is untouched and still
the reference copy if this is ever revisited.

Validated end-to-end 2026-08-27: `python climate_fit.py` (210.5s, `wesl_jul_31`
conda env -- needed for `cdsapi`, not in the default env) + `python
orchestrator.py --scenario scenario_1 --seed 0` -- `trials.csv` has exactly 3
climate columns, `climate_rotation_deg`/`_speed_multiplier`/`_spread_multiplier`
each landing inside their measured caps.

## climate_fit.py / climate.py -- wind resource (39 bootstrap-calibrated parameters, 2026-08-21 to 2026-08-27, history only)

**Superseded by the 3-knob design above on 2026-08-27** (too many independent
factors diluted correlation with outcomes) and originally superseded the
4-knob design further below on 2026-08-21 -- replaced after reviewing a lab
collaborator's independent wind-resource-uncertainty analysis
(`wind_resource_analysis.pdf`) that measured per-parameter uncertainty via
resampling instead of assuming knob magnitudes. See
[[wesl-risk-study-nsf-aim2]] memory for the full comparison. The old design
(4 shared knobs -- `delta_mu`, `kappa` spread, weight Dirichlet concentration,
`S_v`) is kept below for history; it's what produced `production_10k`'s results
and everything in `TRIAL_RECORD.md`. It is not what `climate.py` does today.

`climate_fit.py` (one-time, not part of a trial): reads the 24 ERA5 years
(2000-2023, same point/years as the collaborator's notebook -- cross-checked
identical `LAT_CENTER`/`LON_CENTER`/`YEARS`), fits the base 5-component von
Mises mixture (`fit_von_mises_mixture`, EM) for direction, and a Weibull(C, k)
per **fixed 30-degree sector** (`build_pywake_sites.fit_sector_weibull`, reused
from `Project_Monte_Carlo` -- NOT per von-Mises component; that's a deliberate
switch from the pre-2026-08-21 design, matching the collaborator's approach) for
speed. That's 5 weights + 5 directions + 5 concentrations + 12 sector-C + 12
sector-K = **39 parameters**.

Uncertainty on all 39 is measured, not assumed: a **block-by-year bootstrap**
(40 resamples, each draws 24 years with replacement, pools, refits) gives each
parameter its own std via MLE asymptotic normality (same argument the
collaborator's notebook cites). Block-by-year (not pooled-hour resampling, which
is what the collaborator's own notebook does) preserves within-year seasonal/
serial structure -- see [[wesl-risk-study-nsf-aim2]] for why that distinction
matters. Von Mises components are label-switching-prone (unordered across
independent EM fits), so each bootstrap draw is re-ordered to the base fit by
nearest circular direction (`linear_sum_assignment`) before being pooled into
the std estimate -- skipping this would blend different physical modes together.
Sector-Weibull needs no such alignment (sectors are geographically fixed).
Caches base values + per-parameter std to `cache/wind_climate_basis.npz`.

`climate.py` loads that cache at import (no GRIB reads at trial time).
`sample_scenario(rng)` draws all 39 parameters independently, each as
`base + Normal(0, measured_std)` (weights re-normalized after the draw, kappa/C/K
clipped to valid ranges) -- no shared rotation-of-everything or single
sitewide-intensity knob; each of the 5 modes and each of the 12 sectors moves by
its own calibrated amount. `scenario_to_site_params` builds the 12-sector
frequency from the perturbed von Mises mixture (density evaluated at sector bin
*centers*, not edges -- a small fix made in the same redesign) and takes A/k
straight from the perturbed per-sector Weibull, no blending step needed since
Weibull is already fit at the sector level. `sample_site(rng)` returns a ready
`py_wake.site.UniformWeibullSite` -- still one climate draw per trial, not per
segment.

**Consequence for `orchestrator.py` logging**: `trials.parquet` gained 39 columns
(`vm_weight_1..5`, `vm_mu_1..5_deg`, `vm_kappa_1..5`, `sector_C_1..12`,
`sector_K_1..12`) replacing the old `climate_delta_mu_deg`/`climate_s_v` pair.

**Not carried forward**: the old `delta_mu_sweep_sites()` diagnostic helper (built
on the retired `delta_mu_grid_deg` cache field, no external callers) and the old
`sample_scenario`/`sample_site` override kwargs (`delta_mu_std_deg`, `kappa_spread`,
`weight_concentration`, `sv_mean`, `sv_std` -- nothing called with non-default
values, so nothing lost). `dev_scratch/build_incremental_cache_notebook.py` and
`dev_scratch/build_wind_resource_bootstrap_notebook.py` (the exploratory prototype
that led to this redesign) still reference the old API and are stale if re-run --
expected, they're scratch, not production.

Validated end-to-end on 2026-08-21: `python climate_fit.py` (211.6s, one-time) +
`python orchestrator.py --seed 0` (148.3s, 22 farms, 18 segments) -- `trials.csv`
came out with the expected 47 columns (10 non-climate + 39), weights summed to
~1, `isolated_aep_gwh` still constant across all 18 segments (self-never-changes
invariant intact), `loss_pct` climbed monotonically 0.62% -> 23.57% (same order
of magnitude as the pre-redesign seed=0 trajectory in `TRIAL_RECORD.md`).

### Original design (pre-2026-08-21, produced `production_10k` -- history only)

`climate_fit.py` fit a 5-component von Mises mixture to pooled direction data (EM,
`fit_von_mises_mixture`), fit a Weibull per component to the speeds each
component "wins" (direction-conditioned speed, not fixed-sector), and measured
real year-to-year rotation (`delta_mu_std_deg`, `delta_mu_max_deg`) and a
calibrated 7-point sweep grid.

`sample_scenario(rng)` drew 4 knobs per trial:
- `delta_mu`: truncated normal, std/cap from the cache (measured 6.9deg / 13.6deg) -- rotated all 5 component means together.
- `kappa`: multiplicative uniform (+/-30%) per component -- assumed, not measured.
- `pi_m` (mixture weights): Dirichlet centered on the fitted weights, concentration=40 -- assumed.
- `S_v`: lognormal (median 1.0, spread 0.08) -- assumed, sitewide speed-level multiplier.

`scenario_to_site_params` projected the perturbed mixture onto the standard 12
sectors (frequency + direction-conditioned Weibull A/k, scaled by `S_v`).

## turbines.py -- turbine catalog + adoption-by-year distribution

Loads the 5 windIO turbine specs (`_SPECS`) once. `typical_rating(year)`: the
market's expected rating, growing from 11.5 to 21 MW over 2026-2050 (linear,
`TURBINE_RATING_SHAPE=1.0`). `shares(year)`: softmax of `-(target-rating)^2 /
(2*tau^2)` across the 5 ratings, `tau=2.5` MW -- a categorical distribution over
turbines that shifts weight toward larger ratings as the year advances.
`draw(rng, year)` samples one turbine name from that distribution.
`to_pywake(name)` / `to_pywake_raw(spec)` build a `py_wake.wind_turbines.WindTurbine`
via `PowerCtTabular` -- see aep.py below for where these are used.

## timeline.py -- arrival-year distribution

Accelerating exponential growth, not a constant-rate Poisson process. Cumulative
share built by year t: `F(t) = (exp(gamma*t)-1) / (exp(gamma*T)-1)`, `gamma=0.12`,
`T=24`. `draw_arrival_years(rng, n)` samples n uniform quantiles, inverts F, sorts
-- gives n arrival years that start sparse and accelerate toward 2050, always
inside the horizon by construction (no rejection, no failed placement). Used
twice per trial: once for the cluster's 8 infill leases, once for the N
speculative farms -- two independent draws from the same distribution.

## spacing.py

`draw(rng)`: discrete uniform over `SPACING_OPTIONS_D` (4-10, in rotor diameters).
One draw per farm (cluster infill, self, and each speculative farm).

## grid_fill.py -- tiling

`fill(polygon, diameter, spacing_d)`: regular rectangular grid at
`spacing_d * diameter` spacing, keeps only points inside the polygon. Purely
geometric -- no optimization, no turbine micro-siting. Applied to every farm that
doesn't already have a real layout (self, cluster infill, speculative).

## wake_models.py -- model registry

Three `py_wake.literature` wrappers registered: `turbopark` (`Nygaard_2022`),
`supergaussian` (`Blondel_Cathelain_2020` + `CrespoHernandez` turbulence),
`bastankhah` (`Bastankhah_PorteAgel_2014`, `k=0.0324555` + `CrespoHernandez`).
`ACTIVE_MODEL = "turbopark"` -- only this one runs today. The ensemble (running
all 3 per trial) is deliberately deferred: plan is to run a full campaign with
`turbopark` first, then re-run the same seeds with the other two models later --
scenario generation never touches `model_name`, so this is exact, not an
approximation.

## aep.py -- exactly how PyWake is called

```python
from py_wake.wind_turbines import WindTurbines
import wake_models, turbines, cluster
```

`_pywake_turbine(key)`: if `key` is one of the 4 real-layout farm names, builds
the `WindTurbine` from its real spec (`cluster.REAL_TURBINE_SPEC`); otherwise from
the discrete catalog (`turbines.to_pywake`). Either way goes through the same
`WindTurbine(..., powerCtFunction=PowerCtTabular(...))` constructor in `turbines.py`.

`_multi_turbines(farms)`: collects the distinct turbine keys used by the
currently-active farm list, builds one `WindTurbines.from_WindTurbine_lst(...)`
(PyWake's multi-type container) plus a `{key: type_index}` map.

`evaluate(site, farms, self_name, model_name, n_cpu=1)`:
1. Builds `multi_turbines` and `wfm = wake_models.MODELS[model_name](site, multi_turbines)`.
2. **Isolated**: `wfm(x_self, y_self, type=[self_type]*n, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)`, `.aep().sum()` -- self alone.
3. **Active**: same call, but `x`/`y`/`type` are self's turbines followed by every other active farm's turbines concatenated (self first); self's AEP is sliced back out with `.isel(wt=slice(0, n_self))`.
4. **`world_aep_gwh` (2026-08-27)**: `sim_active.aep()` already holds every active turbine's own AEP (self + every neighbour) from that SAME step-3 call -- summing the whole thing (no slice) costs nothing extra, no second PyWake call. Logged alongside `isolated_aep_gwh`/`active_aep_gwh` in the `segments` table via `orchestrator.py`'s existing `**result` spread (zero orchestrator.py changes needed). Verified both scenarios, seed=0: `world_aep_gwh` scales with `n_active_turbines` as expected (scenario_1: 13,860 -> 81,754 GWh as turbines go 296 -> 1,358; scenario_2 similarly, ~10.3-10.8% `loss_pct` throughout, consistent with its already-built cluster).

`SIM_WD = range(0, 360, 30)` (12 directions), `SIM_WS = range(3, 26, 1)` (23
speeds) -- 276 flow cases per call. `n_cpu=1` by default: PyWake's own internal
flow-case parallelism is left off; parallelism is meant to happen across trials
(separate OS processes), not nested inside one.

Current architecture: every segment re-simulates ALL active turbines together as
one merged array (no `ExternalWindFarm` decomposition -- see Open Items).
`isolated_aep` is recomputed every segment even though self never changes.

## instrumentation.py -- logging

`Logger`: three tables, `trials` (one row/trial: seed, n_speculative, self's
turbine/spacing, climate knobs, model, generation/tiling timing), `arrivals` (one
row/farm: source tag -- `self`/`real_layout`/`cluster_infill`/`speculative` --
arrival year, turbine, spacing, turbine count), `segments` (one row per time
segment: start/end year, active farm/turbine counts, isolated/active AEP, loss %,
per-stage timing). `save(run_name)` writes 3 CSVs under `results/<run_name>/`.

## orchestrator.py -- one trial, start to finish

**2026-08-26: step 4-5 below (event-based segmenting) is superseded by fixed
5-year checkpoints -- see SCENARIOS.md 'Segment granularity'. Steps 1-3
still accurate (speculative farms now arrive pre-tiled from `layout.py` --
step 2 is a no-op for them, `_tile`'s `if "x" in farm` guard skips them).**

`run_trial(seed, logger, scenario, model_name, n_cpu)`:
1. `cluster.generate_cluster(rng, scenario)` + `layout.generate_scenario(rng, scenario)` + `climate.sample_site(rng)`.
2. Tile every farm without a real layout (`grid_fill.fill`).
3. Log the trial row and every farm's arrival row.
4. `orchestrator._checkpoints()` builds a fixed grid of segment boundaries (`settings.SEGMENT_CHECKPOINT_YEARS` apart, YEAR_START to YEAR_END) -- independent of how many arrival events the trial happens to have.
5. Walk those fixed boundaries; at each one, `active` = every farm with `arrival_year <= segment start` (recomputed fresh each segment, not incrementally built), call `aep.evaluate` once per segment, log the result.

## Known simplifications / open items (as of this writing)

- **Ensemble**: `ACTIVE_MODEL = "supergaussian"` as of 2026-08-20 (was `turbopark` -- switched for campaign throughput, a turbopark trial costs ~4x a supergaussian one, see TRIAL_RECORD.md). `bastankhah` is next in line for a same-seed re-run; `turbopark` after that (works fine, `aep.evaluate()` is model-agnostic -- just the most expensive of the 3).
- **`isolated_aep` waste**: recomputed every segment (measured 3.6% of trial time) though self never changes -- not yet cached.
- **Segment granularity**: ~~one segment per arrival event~~ -- **resolved 2026-08-26**, replaced by fixed 5-year checkpoints (`orchestrator._checkpoints()`, `settings.SEGMENT_CHECKPOINT_YEARS`). See SCENARIOS.md.
- **`ExternalWindFarm` (PyWake's own cluster-AEP mechanism) -- root bug found + fixed, 2026-08-20** (supersedes the k=4-angular-split approach, kept below one line for history only): `externalWindFarms=` has a bug in PyWake's own engine -- `PropagateDownwind`'s turbine-by-turbine downwind-order loop folds each external farm in as ONE virtual reference point, and any self turbine that misclassifies as "upwind" of that single point (relative to a neighbor's real, often large/irregular footprint) gets ZERO deficit from it. Confirmed via direct `_flow_map()` calls bypassing the loop (always correct there) -- no grid resolution or `include_wd` window tuning fixes this, it's a discrete engine bug, not an approximation-precision issue. **Fix**: don't use `externalWindFarms=` at all -- simulate all non-self neighbors together, `flow_map()` the result over self's own bbox, and feed that in as a background wind field via `XRSite.from_flow_box()`; self then runs alone (no external-farm objects) against this modified inflow. Validated (accuracy only, single ws=10) on the real full SouthCoast Wind cluster (12 real neighbors, 2 touching at 0km), annual AEP error vs. merged (all-to-all) ground truth: turbopark +3.48%, supergaussian -0.24%, bastankhah +0.36%. Full debug trail: `../ExternalWindFarm/dev_scratch/` (see [[pywake-honeycomb-adjacency-fix]]).
  (Historical: an earlier angular-split-by-bearing workaround, prototyped 2026-08-19, got the seed=0 trial's active-sim time to ~21s with 0.1-4pp accuracy -- correct diagnosis at the time, now superseded by the root-cause fix above.)
  **Ported into `aep.py` 2026-08-20 as `evaluate_background_field()` -- NOT a speedup, not wired in as default.** `evaluate()` stays the production path. Timed at this pipeline's real production resolution (SIM_WD x SIM_WS = 276 flow cases, not the validation script's single ws=10) on the actual seed=0 scenario: `flow_map()`'s cost is dominated by the fixed `grid_n^2 * 276` background-field evaluation, not by neighbour count, so it doesn't shrink for small segments and gets worse for large ones -- small segment (514 turbines) 6.5s merged vs 58.5s background (0.11x); largest segment (1967 turbines) 76.8s merged vs 367.7s background (0.21x). Accuracy also came in worse than the validated case on this denser Monte Carlo scenario (1.26%-9.86% error vs. the validated 3.48%). Needs grid_n/ws-resolution tuning or a cross-segment reuse strategy before it's viable -- not attempted yet.
  **Follow-up, checked whether the grid version's slowness was turbopark-specific: no.** Re-timed `evaluate_background_field()` (grid) with `supergaussian` instead of `turbopark` (same seed=0 scenario): small segment 1.59s merged vs 8.52s grid (0.19x), largest segment 20.74s merged vs 61.41s grid (0.34x) -- still net slower than merged for both segment sizes, just less catastrophically than turbopark's 0.11x/0.21x (supergaussian's per-point `flow_map` cost is cheaper, but the `grid_n^2` structural bottleneck is model-independent). Confirms the grid approach is not viable for any of the 3 models, not just turbopark -- the `Points`-based fix below is necessary regardless of wake model.
  **Follow-up, same day: `evaluate_background_points()` -- query the background field only at self's exact turbine positions (no grid at all), via `flow_map(Points(...))` + a per-position (`i`-indexed) `XRSite`, a documented PyWake mechanism (see `test_i_dependent_WS`).** This removes the fixed `grid_n^2` cost entirely (pays for `n_self` points instead of `grid_n^2`). Per-segment spot checks looked promising for the two LinearSum wake models -- bastankhah 1.10x-1.22x (0.000%-0.003% error), supergaussian 1.08x-1.09x (-0.17% to -1.6% error). **Does not work for `turbopark`** -- crashes with `AssertionError: SquaredSum only works for deficit - not speedups` inside PyWake's own `ground_models.py`/`superposition_models.py` stack (turbopark's `Nygaard_2022` registers `SquaredSum` + a "Mirror ground" model); clipping the injected background `WS` at the nominal free-stream value does not fix it -- not yet root-caused.
  **FINAL, same day: the per-segment numbers above do not hold up in a full-trial aggregate -- NOT adopted for production.** Ran the complete 18-segment seed=0 trial both ways (`supergaussian`, full production resolution): `evaluate()` (merged) totals 133.8s vs. `evaluate_background_points()`'s 128.6s -- **1.04x, noise-level, not a real win** (TRIAL_RECORD.md Trial Record #3 vs #2). Why: this only removes the self-self/self-neighbour cross terms from the O(n^2) cost -- step 2 (simulating all neighbours together) still re-runs from scratch every segment, same cost structure as merged's own O(n^2), and self stays a small, fixed turbine count while neighbours grow into the thousands, so the removed terms are a shrinking fraction of total cost exactly when it would matter most. `aep.evaluate()` was reverted to the simple all-merged implementation (production default again, works with any model incl. turbopark); `evaluate_background_points()` stays in `aep.py`, documented and correct, for whoever pursues the incremental-caching idea noted in its docstring (neighbours only ever grow within a trial -- an incremental/cached neighbours-only simulation across segments could still be worth it, just wasn't tried here) or roots out the turbopark crash and finds the technique pays off better there.
- **Regret / optimizer**: not built. The segment-by-segment history this pipeline produces is more than a t=0 regret comparison against the final realized world actually needs -- open question on how much of it regret will use.
- **Turbine tiling**: plain rectangular grid, no micro-siting/optimization layer.
- **Local-machine parallelism**: not a reliable proxy for HPC array-job scaling (mixed performance/efficiency cores, shared memory bandwidth, thermal throttling) -- parallel scaling to be measured on Chimera directly, not extrapolated from a laptop.
