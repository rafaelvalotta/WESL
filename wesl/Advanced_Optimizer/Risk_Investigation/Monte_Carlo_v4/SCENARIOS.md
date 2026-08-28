# Monte Carlo v4 -- Scenario Split

Added 2026-08-26, forked from `Monte_Carlo_v3` (see the parent memory file
`monte-carlo-v4-fork-from-v3.md` and PIPELINE.md for everything that carries
over unchanged). This doc covers only what's new: two study scenarios and the
AOI radius change. Config lives in `scenarios.py`; `cluster.py` and `layout.py`
branch on it, everything else (`aep.py`, `wake_models.py`, `turbines.py`,
`spacing.py`, `timeline.py`, `grid_fill.py`, `climate*.py`,
`instrumentation.py`) is scenario-agnostic, same as PIPELINE.md describes.

## Why two scenarios

v3 had one buildout model: the full 13-lease cluster generated stochastically
every trial (4 real-built farms fixed, the other 8 staggered in via
`timeline.draw_arrival_years` as guaranteed-but-timed infill), plus a
speculative mosaic around it. That conflates two different questions: "how
much does self's AEP suffer from buildout risk in general" vs. "how much does
self suffer specifically from the cluster it's already inside, once that
cluster actually gets built out as currently planned." v4 splits these into
two runnable scenarios, both studying `self = New England Wind 2 South`
(13MW, 8D -- `scenarios.SELF_FARM/SELF_TURBINE/SELF_SPACING_D`).

## Scenario 1 -- open buildout (`scenarios.SCENARIO_1`)

"Real world today" = only the 4 already-built farms (Vineyard Wind, Revolution
Wind, South Fork Wind, Sunrise Wind South) + self, all fixed from year 0. The
other 8 lease footprints (SouthCoast Wind, Sunrise Wind North, Revolution Wind
North, Vineyard Northeast, New England Wind 1, Beacon Wind, Bay State Wind
South/North) are **not** farms in this scenario at all -- `cluster.py`
(`cluster_mode="open"`) simply doesn't emit them, and `layout.eligible_for()`
does not exclude their footprints from the mosaic's eligible-water mask, so
`mosaic.generate_mosaic` can dart-throw speculative parcels straight through
that space (potentially cutting across the original lease boundaries). All of
self's neighbor-buildout risk is unstructured/speculative in this scenario.

## Scenario 2 -- locked cluster (`scenarios.SCENARIO_2`)

The whole 13-lease cluster exists and is filled **once, globally** -- not
redrawn per trial. 4 real-built farms unchanged; self fixed at 13MW/8D; the
remaining 8 leases get a single canonical "as-built-today" realization drawn
from a restricted menu (11MW or 13MW turbines only, 7D or 8D spacing only --
matching what's actually built/under-construction in the real cluster today,
narrower than the full 5-turbine/4-10D catalog v3 used for infill). All 13
farms are present from year 0 (`arrival_year = YEAR_START`, no staggering).
`layout.eligible_for()` excludes all 13 footprints from the mosaic mask, same
as v3's behavior (minus the radius shrink below) -- only the buildout
*outside* the cluster varies trial to trial.

**Generating the fixed realization**: `python
dev_scratch/build_scenario2_fixed_cluster.py` (fixed seed `20260826`) draws
the 8 leases' turbine + spacing once and caches them to
`cache/scenario2_fixed_cluster.json`. `cluster.py`
(`_load_fixed_cluster_scenario2`) loads that file every time scenario_2 needs
a cluster -- same "compute once, cache, load at runtime" idiom `climate.py`
already uses for `wind_climate_basis.npz`. Only turbine name + spacing_d are
cached, not tiled positions: `grid_fill.fill()` is deterministic given
`(polygon, diameter, spacing_d)`, so `orchestrator._tile()` reproduces
identical turbine positions from those two values on every trial without
needing to store x/y separately. **Re-running the builder script changes
every future scenario_2 trial's fixed background** -- only do it deliberately.

## AOI radius: 150km -> 100km

`layout.RADIUS_KM` (the mosaic's eligible-water clip radius around the
Vineyard Wind center) dropped from 150 to 100, in both scenarios. Everything
else about the eligibility grid (bathymetry `<80m`, federal/state split via
the SLA boundary, 300m resolution) is unchanged.

## Farm source tags (`arrivals.csv`/`.parquet`)

`orchestrator._source()` now returns:
- `self` -- the one farm under study
- `real_layout` -- the 4 already-built farms (both scenarios)
- `cluster_fixed` -- scenario_2's 8 locked-cluster farms only (never appears
  in scenario_1's logs)
- `speculative` -- mosaic-generated, both scenarios

v3's `cluster_infill` tag (staggered-but-guaranteed arrival) is retired --
neither scenario uses that concept anymore.

## Results layout

`results/<scenario_key>/<run_name>/` -- e.g. `results/scenario_1/smoke_seed0/`,
`results/scenario_2/production_10k_v3/task_0004/`. `--scenario` is a required
CLI flag on `orchestrator.py` (default `scenario_1` for quick local smoke
tests) and on `hpc/run_trial_range.py` / `hpc/slurm/submit_array.sh` (no
default there -- an accidentally-wrong-scenario production campaign is
expensive enough on Chimera that it should never silently fall back).
`hpc/merge_results.py` takes `--scenario` too, to find the right
`results/<scenario>/<run_id>/task_*/` tree.

## Speculative fill redesign (2026-08-26): density-first + 30GW target

Root cause of implausibly dense speculative farms (observed: ~5GW mean farm
capacity in earlier campaigns): `spacing.draw(rng)` sampled `spacing_d`
uniformly over 4-10D **independently of which turbine got drawn**. Density
scales as `1/spacing_d^2`, so a uniform draw over 4-10D is NOT a uniform draw
over density -- it's heavily skewed toward the small-D (high-density) end.
Confirmed numerically: for every turbine in our catalog, the implied mean
density under the old uniform-4D-10D draw was **~7.0-7.4 MW/km^2**, vs.
NREL's real-project weighted mean of **~4.42 MW/km^2** (`Capacity Density
Considerations for Offshore Wind Plants in the United States`, NREL/TP-5000-
86933) -- roughly 1.6-1.7x too dense, regardless of turbine choice. Root-caused
against `Metodologia de Monte Carlo para 10.000 simulações de farms eólicas
offshore` (methodology writeup provided 2026-08-26), whose section 5
("Por que não sortear 4D a 10D uniformemente") derives the same skew formula
independently.

**Fix -- density first, spacing derived** (`spacing.from_density`,
`layout.generate_scenario`): each candidate parcel draws its own
`rho_target ~ Triangular(2.1, 4.0, 6.5)` MW/km^2 (`settings.RHO_TARGET_TRIANGULAR`
-- the doc's central literature estimate: NREL's ~4.42 mean, Pryor & Barthelmie's
US-east-coast lease simulations at 3.5-6.0, ~90% of European projects at
4.9-5.9), then derives `spacing_d = sqrt(rating_MW / (rho_target * D_km^2))`.
Resampled up to 10x if the derived spacing falls outside
`settings.SPACING_D_BOUNDS = (5.5, 12.0)` D (a modeling filter, not a
regulatory limit); if still out of band, the density draw is kept and
spacing_d clipped into the band (flagged via `spacing_clipped` in
`arrivals.csv`/`.parquet`) rather than discarding the trial.

**Not a source of bias vs. turbine/year**: verified `rating_MW / D_km^2` across
our 5-turbine catalog is nearly constant (248-275, ~10% spread) -- i.e. the
catalog already has roughly constant specific power by design (bigger
turbines get proportionally bigger rotors). So `spacing_d` derived from a
given `rho_target` stays consistent in D-units regardless of which turbine
the softmax-by-year draw picked -- drawing density independently of
turbine/year does not privilege any rating or any point in the horizon.
(Known simplification, same as the doc's own model: `rho_target` itself does
NOT vary systematically with year -- real-world density could drift with
technology/floating-substructure maturity, not modeled here.)

**rho_target scope**: drawn per parcel, not once per trial -- each speculative
parcel is its own "area" in the doc's model (`Densidade-alvo` is listed among
the variables that must vary between draws), so per-parcel draws preserve
realistic heterogeneity between different speculative developments in the
same trial. The 30GW target (below) is purely the stopping rule for how many
independently-drawn parcels get included, not a reason to share one density
value across them.

**Target-driven fill, replaces `N_RANGE`** (`layout.generate_scenario`): the
old `n_draw = uniform_discrete(1, 16)` farm-count draw is retired. Instead:
shuffle the mosaic's candidate parcels, tile each one immediately (moved
in-line into `layout.py`, no longer deferred to `orchestrator._tile` --
needed to know each parcel's actual capacity as we go), and keep adding whole
parcels until cumulative speculative capacity reaches
`settings.SPECULATIVE_TARGET_MW = 30_000` (~NREL's 30GW offshore buildout
figure). **Overshoot, not truncation**: the parcel that crosses the target is
kept whole (not partially tiled) -- simpler and keeps every farm a
geometrically coherent full development, at the cost of slightly overshooting
30GW (verified: mean overshoot ~1GW, max ~2.3GW across 30 seeds).

**Verified (30 seeds, scenario_1, local, `layout.generate_scenario` only --
no PyWake)**:
- Per-farm capacity: mean **966 MW**, median 738 MW (was ~5000 MW mean) --
  84.1% of farms land in the doc's "individual comum/grande" band (<=1.5GW),
  11.5% "excepcionalmente grande" (1.5-2.6GW), 4.4% cluster-scale (>2.6GW,
  from the largest mosaic parcels at plausible density -- expected per the
  doc's own farm-vs-cluster classification, not an error).
- `rho_target` vs. realized `rho_final`: mean 4.22 vs. 4.09 MW/km^2 (both
  close to NREL's ~4.42) -- density actually achieved tracks the target
  closely; `spacing_clipped` rate 0/958 farms (the 5.5-12D band was never
  binding at this rho_target distribution).
- Trial total speculative capacity: mean 30.86 GW, range 30.01-32.32 GW
  across 30 seeds (scenario_1) and 30.12-31.42 GW across 15 seeds
  (scenario_2) -- target reliably hit with small, bounded overshoot.

**Cost consequence (important for HPC campaign sizing, not yet re-tuned)**:
a full end-to-end trial (`orchestrator.py --scenario scenario_1 --seed 0`)
now runs **~347s** (was ~17s pre-redesign) -- 38 farms / 2044 turbines total
vs. the old trial's far fewer, far denser farms. This is the CORRECT cost of
plausible density (realistic buildout needs more, moderately-sized farms to
reach the same aggregate capacity that implausibly dense farms used to reach
with fewer of them) -- not a regression to fix. **`hpc/slurm/submit_array.sh`'s
`--time=06:00:00`/`--mem=64G` defaults were sized for the OLD, cheaper trial
distribution and have NOT been re-validated against this redesign** -- run a
fresh shakedown array (per `hpc/README.md`) before any real production
campaign under scenario_1 or scenario_2's new fill.

## Density distribution recalibrated against real project data (2026-08-27)

`settings.RHO_TARGET_TRIANGULAR` changed from `(2.1, 4.0, 6.5)` to
`(2.1, 4.0, 9.03)`, recalibrated against a specific NREL table ("U.S.
Large-Scale Offshore Wind Energy Projects Data" -- 17 real named projects,
project-level capacity density from Table 1). Min/max are now the table's own
observed extremes (2.10 Revolution Wind, 9.03 Skipjack), not a truncated
subset -- the old 6.5 cap excluded 4/17 real projects (Kitty Hawk 7.07, Empire
Wind 1 7.42, Empire Wind 2 8.13, Skipjack 9.03) as literally impossible draws.
Mode kept at 4.0 (not the table's own mean) so the distribution's peak doesn't
move -- this is a deliberate choice, not a data measurement.

**Validates the earlier design**: the table's area-weighted mean is 4.42
MW/km^2 -- matching NREL's independently-reported weighted mean (the number
`RHO_TARGET_TRIANGULAR`'s original mode/center was already calibrated
against) almost exactly. Confirms this table is the real data behind that
figure, not a separate/conflicting source.

**Effect of keeping mode fixed while widening max**: the fixed mode anchors
the distribution's PEAK, not its mean or skew. `Triangular(2.1,4,6.5)` put
43.2% of its mass below the mode; `Triangular(2.1,4,9.03)` puts only 27.4%
below it (vs. the real table's 47% of projects below 4.0 MW/km^2) -- the
distribution's own mean rose from 4.20 to 5.04. Verified via 30 simulated
trials: achieved `rho_final` mean 4.90 (target mean 5.03), 0% `spacing_clipped`
(the resample mechanism handles the tail near 9.03 -- most turbines' derived
spacing dips just under the 5.5D lower bound exactly at max density, but never
needed the 10-resample budget in practice). Farms/trial dropped ~32 -> ~19-23
(denser farms need fewer distinct parcels to reach the same 30GW target) --
turbine count and PyWake cost essentially unchanged (driven by target MW /
turbine rating, not density). Full end-to-end trial verified, 14.5s.

**scenario_2's fixed cluster cache regenerated** to match -- it uses the same
`spacing.from_density`/`RHO_TARGET_TRIANGULAR`, so leaving the old cache in
place would have reintroduced the exact "two different density calibrations"
inconsistency the fixed-cluster-consistency fix addressed earlier. Re-ran
`dev_scratch/build_scenario2_fixed_cluster.py` (same seed 20260826): densities
now 3.36-6.66 MW/km2 (was 3.10-5.55), 0/8 clipped. Verified end-to-end.

## Depth limit: 80m -> 60m (2026-08-27)

`layout.FIXED_BOTTOM_MAX_DEPTH_M` (the fixed-bottom foundation shallow-water
cutoff used in `eligible = fed & shallow & within_corridor`) tightened from 80m
to 60m. Eligible area shrinks accordingly: scenario_1 28,547km2 -> 17,009km2,
scenario_2 26,041km2 -> 14,641km2 (both at the 100km corridor width). No other
change -- `eligible_for()`/`generate_scenario()` unaffected in structure.
Verified end-to-end (`orchestrator.py --scenario scenario_1 --seed 0`, 14.8s).

## Target is per-scenario total, not additive (2026-08-27)

`settings.SPECULATIVE_TARGET_MW` (30GW) was being treated as a target for the
SPECULATIVE growth alone -- `layout.generate_scenario` stopped once the
speculative farms' own cumulative capacity hit 30GW, regardless of how much
was already fixed in that scenario. Since scenario_1's fixed backbone (4
real-layout + self) is 3.50GW but scenario_2's (the whole locked 13-lease
cluster) is 12.15GW, this meant scenario_1 landed at ~33.5GW total and
scenario_2 at ~42GW total -- both over NREL's ~30GW figure, and inconsistently
so between scenarios, instead of both representing a plausible ~30GW-scale
national buildout.

**Fix**: `layout.generate_scenario(rng, scenario, fixed_capacity_mw=0.0)`
takes the scenario's already-committed capacity as a parameter; the walk now
stops once `fixed_capacity_mw + cumulative_speculative_mw >= 30GW`, i.e. the
target is each scenario's TOTAL world, computed per scenario (never shared or
combined across scenario_1/scenario_2 -- each trial is scoped to exactly one
scenario, as it always was). `orchestrator.run_trial` now tiles
`cluster_farms` first (moved earlier, not deferred) to compute
`fixed_capacity_mw` before calling `layout.generate_scenario`
(`cluster.generate_cluster` doesn't consume any rng draws today -- real-layout/
self/cluster_fixed are all deterministic -- so reordering doesn't perturb the
rest of the trial's random sequence). `fixed_capacity_mw` is now logged in
`trials.csv`/`.parquet`.

**Verified**: same seed, both scenarios now land within ~2% of the 30GW
target -- scenario_1 30.57GW (3.50 fixed + 27.06 speculative), scenario_2
30.36GW (12.15 fixed + 18.21 speculative) -- vs. the old 33.5GW/42GW. Side
effect: scenario_2 trials got FASTER (fewer speculative farms needed, since
its larger fixed backbone already covers more of the target) -- measured
74.8s -> 60.4s for the same seed.

## AOI shape and anchor (2026-08-27)

Two changes to `layout.py`'s eligibility grid, found while building `Setup_Scenarios.ipynb`
(renamed from `v4_pipeline_visual.ipynb` 2026-08-27):

**Anchor moved from Vineyard Wind to self.** `RADIUS_KM`/AOI center was still
anchored on Vineyard Wind's real turbine-position centroid -- a leftover from
before `self` became New England Wind 2 South. Now anchored on `self`'s own
lease centroid (`cluster.LEASE_TEMPLATES[scenarios.SELF_FARM].centroid`) --
self has no real turbine layout (not built), so this uses the lease polygon's
centroid rather than turbine positions. Shifts the AOI ~21.5km west, ~21km
south vs. the old Vineyard-anchored one.

**Circle replaced with a rotated square, then a corridor (same day, two
revisions).** First pass: AOI became a SQUARE of half-width `RADIUS_KM`,
rotated to the cluster's principal axis (`layout._cluster_axis_angle()` --
PCA on the 13 lease centroids' covariance, eigenvector of the largest
eigenvalue). User feedback: the square (a big rotated diamond dominating the
map) looked bad and was still an artificial shape imposing a length limit
that didn't need to exist. **Final shape: a CORRIDOR** -- bounded only by
`RADIUS_KM` on the perpendicular axis (the width, same as the square's
half-width), with NO cap along the cluster axis at all. The corridor's actual
length is left entirely to what `fed & shallow` (federal water + <80m depth)
naturally allow -- the eligible region is "whatever connects the federal/state
boundary to the bathymetry depth limit, within this width," not a shape with
an artificial rounded (circle) or diamond (square) far edge.
`AOI_LENGTH_HALF_KM = 260.0` (new constant) only sizes the GRID's coverage
along the axis so the real boundary isn't clipped by grid extent -- it plays
NO role in the eligibility test itself (verified against `bathymetry_us.nc`'s
own real coverage in the rotated frame: u in [-238,+159]km). `RADIUS_KM` keeps
its name/meaning (the "reach" distance), now the corridor's half-width.
`layout.BASE_GRID` exposes `aoi_center`/`aoi_axis_angle`/`aoi_half_width_m` so
downstream code (the visual notebook) can draw the exact same corridor
without re-deriving it.

Both changes only affect `_load_base_grid()`'s geometry -- `eligible_for()`,
`generate_scenario()`, and everything downstream are unchanged in structure.
Re-verified end-to-end after all changes (`orchestrator.py --scenario
scenario_1 --seed 0`, 20.9s trial time -- ~25s one-time base-grid load added
at process start since the grid grew ~6x, 701x701 -> 1833x1580 cells, but
that's paid once per process/HPC task, not per trial).

## Fixed-cluster density consistency (2026-08-27)

Scenario_2's fixed cluster (`dev_scratch/build_scenario2_fixed_cluster.py`)
originally drew turbine from `{11 MW, 13 MW}` AND spacing independently from a
separate restricted menu `{7, 8}` D. Checked against the newly-adopted
`RHO_TARGET_TRIANGULAR`/`SPACING_D_BOUNDS` criteria (2026-08-26): all 8
resulting densities (4.20-5.61 MW/km^2) already landed inside the plausible
band -- not a plausibility bug, since every combination in that narrow 2x2
menu happened to be safe. But it was a **different method** than the mosaic's
density-first derivation, and a second independent mechanism that happens to
agree today is exactly the kind of thing that stops agreeing silently after a
future change (e.g. widening the turbine menu) and reads as inconsistent at
review time. Fixed: the builder now draws turbine from the same `{11 MW, 13 MW}`
menu (a real, separate constraint -- near-term-realistic turbine choice, not a
density concern) but derives spacing via the exact same `spacing.from_density`
call and `settings.RHO_TARGET_TRIANGULAR` distribution the speculative mosaic
fill uses -- one density methodology for the whole pipeline, not two.
Cache regenerated (same `SEED=20260826`): spacing is now continuous (was
integer 7 or 8D, now e.g. 7.68D, 9.27D), density spans 3.10-5.55 MW/km^2, mean
3.98 (vs. the old menu's 4.91) -- closer to NREL's ~4.42 central estimate, 0/8
clipped. `arrivals.csv`'s `cluster_fixed` rows now carry `rho_target_mw_km2`/
`spacing_clipped` too (previously only speculative rows did).

## Segment granularity: fixed 5-year checkpoints (2026-08-26)

The speculative fill redesign above raised the number of arrival events per
trial (~9 -> ~33 speculative farms, to hit 30GW at plausible density instead
of implausible density), and the pipeline's old segmenting scheme
(`orchestrator.py`, one segment per arrival event, however close together --
already flagged as an open item in PIPELINE.md's "Known simplifications")
scaled with it: 10 segments -> 34 segments for the same seed, and PyWake's
`O(n_turbines^2)` cost compounded across that many more (increasingly large-n)
evaluations. Measured full-trial time: 16.8s (pre-redesign) -> 347.1s
(post-fill-redesign, event-based segments) -- a ~20.7x increase that traced
entirely to segmenting, not to the fill logic itself (`layout.generate_scenario`
alone: 454ms, negligible).

**Fix**: `orchestrator._checkpoints()` replaces one-segment-per-arrival-event
with fixed periodic checkpoints, `settings.SEGMENT_CHECKPOINT_YEARS = 5` apart,
independent of how many farms/events a trial happens to draw -- 2026, 2031,
2036, 2041, 2046, 2050 (5 segments, last one 4yr since 24/5 doesn't divide
evenly). Each segment's active farm set is now "everyone with
`arrival_year <= segment start`" (cumulative membership evaluated at 5 fixed
points), not "one more farm than the previous segment." Same idea the doc's
farm-vs-cluster and rho_final sections apply elsewhere: analyze the resulting
distribution at a fixed, deterministic grid rather than an input-dependent one.

**Verified**: same seed, same scenario, same 38 farms/2044 turbines -- 34
segments -> 5 segments, **347.1s -> 23.0s (15.1x faster)**, landing back near
the ORIGINAL pre-redesign 16.8s despite now simulating ~2.8x more turbines at
correct density. `segment_id` is now aligned across every trial in a batch
(`segment_id=2` is always 2036-2041 for every seed) -- also sets up a clean
fixed-width time grid for the regret/optimizer work noted as not-yet-built in
PIPELINE.md's Open Items.

**Not changed by this**: `isolated_aep` is still recomputed every segment even
though self never changes (pre-existing, documented waste in PIPELINE.md,
~3.6% of trial time pre-redesign -- untouched here, still a small fraction of
a now-much-shorter trial).

## Verification run (2026-08-26, local smoke, seed 0)

`python dev_scratch/build_scenario2_fixed_cluster.py` then `python
orchestrator.py --scenario scenario_1 --seed 0 --out smoke` (16.8s, 14 farms,
10 segments) and `--scenario scenario_2 --seed 0 --out smoke` (91.0s, 20
farms, 8 segments) -- both completed cleanly.

`arrivals.csv` confirmed: scenario_1 baseline (`arrival_year<=2026`) is
exactly 5 farms (4 `real_layout` + 1 `self`), zero `cluster_fixed`, 9
`speculative`; scenario_2 baseline is 13 farms (4 `real_layout` + 1 `self` + 8
`cluster_fixed`), 7 `speculative`. Re-ran scenario_2 at `--seed 7` and
compared the 8 `cluster_fixed` rows against seed 0's: turbine, spacing_d, AND
`n_turbines` (tiled position count) came out byte-identical across seeds --
confirms the cluster really is fixed once, not redrawn per trial. `RADIUS_KM`
confirmed at 100 in `layout.py`; the mosaic's eligible-water mask structurally
enforces the 100km clip (`within_radius = hypot(...) < R` baked into the grid
itself), so no speculative parcel can land past it by construction.
