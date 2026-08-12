# Risk & Regret Monte Carlo Pipeline — Design Notes v2

**Status:** Working design document recording the decisions made in a single alignment
session between Bruno and Claude, building directly on `PIPELINE_DESIGN_v1.md`. Where
v1 left an item "pending PI confirmation," this document either closes it or explicitly
carries it forward as still open (see §9). Nothing here has been implemented yet —
`Monte_Carlo/` (the v1 code) is untouched; the plan is to build `Monte_Carlo_v2/` as a
sibling directory next.

**Written in English, with an explicit glossary,** at Bruno's request, so anyone reading
this later — including someone without the full conversation history — can follow every
term used.

---

## 0. Glossary

Terms used throughout this document, defined once here.

- **Tier** (also "regional build-out intensity tier"): one of three qualitative labels —
  **low / medium / high** — describing how fast the regional offshore wind market grows
  around the existing cluster. In v1, a tier was a fixed total-capacity target (a ceiling
  in MW). In v2 (this document), a tier is instead a **rate** — see λ below — and the
  eventual total capacity becomes an *output*, not an input.
- **λ (lambda)**: the rate parameter of an **exponential distribution** / **Poisson
  process**. If new wind farms appear at a constant average rate of λ farms per year,
  the *time between* two consecutive arrivals follows an Exponential(λ) distribution,
  with mean gap = 1/λ years. A **Poisson process** is the formal name for "events
  happening independently at a constant average rate λ" — it is the standard way to
  model arrivals (leases, buses, phone calls, etc.) when the underlying number of
  arrivals in a fixed period is itself random, not fixed in advance.
- **Self** (also "focus farm"): the one wind farm the whole study is centered on — the
  entity whose AEP (Annual Energy Production) loss and, later, regret are being
  measured. Everything else in the simulation ("neighbors") only matters insofar as it
  changes self's AEP through wake interaction.
- **Ensemble axis** (also "epistemic, whole-scenario axis"): an input that is **fully
  enumerated**, not randomly sampled — every trial is run once under *each* setting, and
  the settings are compared side by side. Used when the different options represent
  genuinely different *beliefs* about which model/scenario is correct (e.g., which wake
  model), not different *possible worlds* to average over.
- **Monte Carlo (sampled) axis**: an input that **is** randomly drawn once per trial,
  because it represents genuine random variation across possible future worlds (e.g.,
  which year's wind climate occurs, where a speculative farm ends up).
- **LHS (Latin Hypercube Sampling)**: a sampling method that spreads draws more evenly
  across a multi-dimensional input space than plain independent random sampling does,
  for the same number of draws — used to get better coverage of the uncertain axes with
  fewer, and therefore cheaper, trials.
- **Segment**: one time-interval within a single trial's simulated history, during which
  the set of active neighbor farms does not change (a "step" in the step-function AEP
  described in §4). Bounded by two consecutive arrival events (or the start/end of
  self's operating horizon).
- **Horizon (T)**: the fixed length of self's simulated operating life — **25 years**,
  agreed in this session (§5).
- **Staircase walk**: the existing boundary-shape generator (`staircase.py`,
  `grow_staircase_boundary` / `_walk_raw`) used to draw a new speculative farm's outline
  — a calibrated, rectilinear ("staircase-shaped") random walk, validated against the
  real 13-lease statistics. Not changed by this session.
- **Eligibility grid**: a precomputed grid (`layout.py`, `build_eligibility_grid`)
  marking which points in the region are federal (BOEM jurisdiction), shallow enough for
  fixed-bottom foundations, and close enough to shore for a plausible grid connection.
  New farm placements are checked against this grid.
- **isolated / real_neighbors / full_cluster AEP**: the three-way AEP decomposition
  computed for self in every trial (`aep_simulation.py`) — see §7 for the full
  explanation.
- **Regret** (forward-looking, not yet implemented): the difference between the AEP (or
  economic value) self achieves under its actual, fixed design, versus the AEP an
  all-knowing "oracle" optimizer could have achieved for that same realized scenario,
  had it been allowed to design self's layout freely. This is the Aim-2 quantity the
  whole pipeline exists to eventually compute; this session did not build the regret
  branch, only the risk branch's inputs.

---

## 1. What changes relative to v1 (`Monte_Carlo/`)

| Axis | v1 (`Monte_Carlo/`, current) | v2 (this document) |
|---|---|---|
| Search radius | 150 km | **120 km** |
| Depth limit | 80 m | 80 m (unchanged, just reconfirmed) |
| Turbine spacing | random per farm, Uniform(8D, 11D) | **fixed 8.0D**, no longer sampled |
| Turbine set | DTU10MW, Haliade-X 13MW, Haliade-X 15.5MW | **SG 11.0-200 DD (11MW), Haliade-X 13MW, Haliade-X 15.5MW, IEA Wind 2200-22MW** — DTU10MW dropped |
| Wake model ensemble | 2 models, both run every trial | **3 models**, both run every trial (still an ensemble axis, never sampled) |
| Self / focus farm | Vineyard Wind (real, already built) | **SouthCoast Wind** (real lease, not yet built) |
| Tiers (low/medium/high) | fixed MW ceiling per tier (`REGIONAL_CAPACITY_MW`), "low" = zero new farms | **λ (arrival rate) per tier**; total MW becomes an emergent output, "low" now produces real (slower) growth |
| Scenario shape | one static "snapshot" per trial — all new farms co-exist from the start | **a simulated history**: farms arrive over time, self's AEP is a step function recomputed at each arrival |
| Sampling method | independent random draws per axis | **Latin Hypercube Sampling** for the continuous axes |
| Code location | `Monte_Carlo/` | new sibling directory `Monte_Carlo_v2/`; v1 stays untouched as reference |

---

## 2. Locked axis-by-axis decisions

### 2.1 Search radius — 120 km
Resolves the open item from v1 §5.1 (120 km vs. 150 km). Both values comfortably clear
the 80 km "sufficiency" threshold reported by Warder & Piggott (2026), *Wind farm wake
losses under future build-out scenarios* (v1's source #2) — the radius choice was a
matter of picking one value, not a physical disagreement.

### 2.2 Depth limit — 80 m
Resolves the open item from v1 §5.2 (60 m proposal text vs. 80 m already in code).
80 m confirmed; no code change needed here (`FIXED_BOTTOM_MAX_DEPTH = 80.0` in
`layout.py` already matches).

### 2.3 Turbine spacing — fixed 8.0D
No longer a Monte Carlo axis. Chosen as a round number close to what real, already-built
farms show. For the record, the actual measured median spacing at the four as-built
farms in the cluster (from `Density_Power_Study/density_power_study.ipynb` §2.1) is:

| Farm | Turbine | Rotor D | Median spacing |
|---|---|---|---|
| Vineyard Wind | GE Haliade-X 13MW | 220 m | 1840 m = **8.36D** |
| Revolution Wind | Siemens Gamesa SG 11.0-200 DD | 200 m | 1836 m = **9.18D** |
| South Fork Wind | Siemens Gamesa SG 11.0-200 DD | 200 m | 1811 m = **9.06D** |
| Sunrise Wind South | Siemens Gamesa SG 11.0-200 DD | 200 m | 1819 m = **9.10D** |

(Mean across the four ≈ 8.93D — 8.0D was chosen anyway as the round working value;
noted here so the discrepancy is on record, not lost.)

### 2.4 Turbine set — 11 / 13 / 15.5 / 22 MW
Four discrete options, each backed by a real data file already present in `Data/`:

| Label | Source file | Rated power | Rotor diameter |
|---|---|---|---|
| 11 MW | `sg11_200dd.yaml` (Siemens Gamesa SG 11.0-200 DD) | 11.00 MW | 200 m |
| 13 MW | `haliadex_13mw.yaml` (GE Haliade-X 13MW) | 13.00 MW | 220 m |
| 15.5 MW | `haliadex_15.5.yaml` (GE Haliade-X 15.5MW) | 15.50 MW | 250 m |
| 22 MW | `Data/turbines.yaml` (IEA Wind 2200-22MW Reference Offshore Wind Plant) | 22.00 MW | 284 m |

The upper tail (22 MW) is backed by McCoy et al. (2024), *Offshore Wind Market Report:
2024 Edition* (NREL/TP-5000-90921 — read in full this session, see §6): average
installed turbine rating moved from 7.7 MW (2022) to 9.7 MW (2023), the market is
already shifting to the 15-MW platform, and Chinese OEMs have announced 18–22 MW plans.
The IEA 2200-22MW file was picked over the alternative `gwh300_22mw.yaml` (Goldwind
GWH300, D=300m) because it is WISDEM-derived and ready-to-use with no extrapolation
needed, and was already the reference anchor named in v1 §2.

This set applies to every "synthetic" farm — the 9 not-yet-built leases inside the
existing 13-lease cluster, plus every new speculative farm, plus self (SouthCoast Wind,
see §5). Turbine choice remains a **per-farm, per-draw Monte Carlo axis** (aleatory, not
ensemble) — each farm independently draws one of the four options, exactly as v1 already
did, just from an updated list.

### 2.5 Wake model ensemble — 3 models, run every trial (not sampled)
Kept as an **ensemble axis**: every trial runs under all three models, and all three
results are logged side by side, so "how much does the outcome vary because of wake
model choice" is never confounded with scenario-to-scenario noise from a random pick.

| Model | py_wake class | Notes |
|---|---|---|
| TurboPark / TurboGaussian | `py_wake.literature.turbopark.Nygaard_2022` | Already in v1. Deficit shape follows Bastankhah & Porté-Agel (2014); wake-expansion additionally accounts for wake-added turbulence (Frandsen 2007). Validated for long-range (inter-cluster, tens of km) interactions by Warder & Piggott (2025) — see §6. |
| SuperGaussian | `py_wake.literature.gaussian_models.Blondel_Cathelain_2020` | Already in v1. |
| Bastankhah & Porté-Agel (2014) | `py_wake.literature.gaussian_models.Bastankhah_PorteAgel_2014` | **New.** Unlike the other two, this class is **not** a fully self-calibrated wrapper — it requires an explicit wake-expansion parameter `k`. Using **k = 0.0324555**, the value from py_wake's own DTU reference example (`gaussian_models.py`'s `main()`, itself following the Niayifar calibration against Horns Rev). **Open item:** confirm whether the WESL group's own papers (Rodrigues 2025 / Moura 2025) already calibrated a different `k`; if not, this default stands. |

### 2.6 Self — SouthCoast Wind (not Vineyard Wind)
Changed this session. Rationale, in order of how it came up:

1. **Regret needs design freedom.** The regret branch (future work, not built yet)
   needs an "oracle" optimizer to be able to freely design self's layout for a given
   realized scenario. Vineyard Wind already has a real, fully built, fixed layout —
   there is nothing left for an optimizer to decide. SouthCoast Wind has a real BOEM
   lease boundary but **no published layout** — a genuine design problem.
2. **This was already what v1's own design doc asked for.** `PIPELINE_DESIGN_v1.md`
   §4.1 defines "self" as *"a real, undesigned lease within the known cluster (fixed
   boundary from the real BOEM polygon; no shape generation needed)"* — the v1 *code*
   (`config.py: FOCUS_FARM = "Vineyard Wind"`) never actually matched that definition;
   this change corrects that mismatch.
3. **Real-world grounding, found this session in McCoy et al. (2024) (see §6):**
   SouthCoast Wind 1 was one of **eight U.S. offshore projects that canceled their
   power-offtake contracts in 2023**, alongside New England Wind 1 & 2, Empire Wind 2,
   Beacon Wind 1, Ocean Wind 1 & 2, and Skipjack Wind — all sharing one macroeconomic
   cause (rising interest rates, inflation, supply-chain costs). The same report notes
   these projects mostly remain in active development, seeking new offtake
   opportunities, rather than being truly dead. This is a real, documented case of
   exactly the kind of build/design uncertainty the regret framing is meant to capture.

**Mechanical implications for the code:**
- `FOCUS_FARM` becomes `"SouthCoast Wind"`.
- SouthCoast Wind is currently one of the 9 "infill" leases inside `layout.py`'s
  `BOUNDARY_NAMES` that get a randomly-redrawn turbine + spacing fill every trial
  (`fill_rest_of_cluster`). It needs to be pulled out of that generic bucket and given
  the "self" role — same underlying generation mechanism (real polygon from
  `LEASE_TEMPLATES`, turbine drawn from the 4-option set in §2.4, spacing fixed at
  8.0D as in §2.3 — **no new placement logic needed**), but flagged as the
  `focus_farm_name` passed into `aep_simulation.run_scenarios` instead of being treated
  as an anonymous neighbor.
- An earlier idea to size self's design off SouthCoast's *publicly declared* turbine
  count (141 turbines, per the Sea Impact map, used in `Density_Power_Study`'s "Case 1"
  as a way to reverse-engineer an implied spacing) is **not needed** now that spacing is
  simply fixed at 8.0D pipeline-wide (§2.3) — the number of turbines that fit is just
  the geometric output of `grid_fill_points(SouthCoast_polygon, chosen_turbine_diameter,
  8.0D)`, same mechanism as any other infill lease. The 141 figure remains a useful
  real-world sanity check on the output, not an input to the design.
- **Vineyard Wind's role changes**: it stops being self and becomes an ordinary
  already-committed real neighbor — identical treatment to Revolution Wind, South Fork
  Wind, and Sunrise Wind South (real, fixed, already-built layout from
  `wind_farm_us.yaml`; unaffected by any of this session's changes).

### 2.7 Tiers — reparametrized as arrival rate λ, not a fixed MW ceiling
The single biggest structural change. In v1, `REGIONAL_CAPACITY_MW` gave each tier a
fixed *destination* in MW, and "low" tier's destination equaled the existing cluster's
own size — meaning **zero new farms, always**, for "low." That stopped matching the
physical story once a rate parameter needed a real justification (see calibration table
below): "low" should mean *slower* growth, not *no* growth.

**Vision confirmed this session ("Vision A"):** the three tiers do have genuinely
different eventual outcomes (different "destinies"), but that difference is no longer
a separate, hand-set MW ceiling — it now **emerges automatically** from combining a
fixed horizon (T = 25 years, §2.8) with each tier's own λ. A slow (low-λ) tier simply
doesn't get through as many arrivals before self's operating window ends.

**Calibration table (as given by Bruno, sourced from regional lease-award history):**

| Tier | λ (farms/yr) | Mean wait (1/λ) | Physical meaning / empirical basis |
|---|---|---|---|
| Low | 0.5 | 2.0 years | Half the historical MA/RI/CT regional pace. Reflects a slowed-down market with bottlenecks. |
| Medium | 1.0 | 1.0 year | The actual historical regional pace — 6 leases granted in 6 years, 2013–2019. |
| High | 2.0 | 0.5 years (6 months) | Double the historical regional pace. Reflects accelerated fulfillment of state energy mandates. |

**Expected outcome over the 25-year horizon** (mean of a Poisson(λ·T) process; MW figure
uses the empirical mean farm size ≈ 875 MW, from `REAL_FARM_SIZES_MW`):

| Tier | Expected *attempted* arrivals in 25 yr | Naive expected MW (100% placement success assumed) |
|---|---|---|
| Low | ≈ 12.5 farms | ≈ 10,900 MW |
| Medium | ≈ 25 farms | ≈ 21,900 MW |
| High | ≈ 50 farms | ≈ 43,700 MW |

**Correction from a real Chimera shakedown run (2026-08-11, tier=high, 5 seeds: 0
local layout-only + seeds 2/3/4/5 full trials on Chimera, n_cpu=4):** the table above
assumes every *attempted* arrival successfully finds a placement — it does not. Real
measured placement success rate for the high tier, pooled across all 5 seeds
(239 attempted -> 68 placed), is a stable **28.5%** (26.7-31.0% per seed, no outliers).
Actual delivered MW averaged **7,560 MW** across the 4 seeds with full AEP data
(6,311-9,828 MW range), roughly **5.8x lower** than the naive 43,700 MW figure above.
Real per-trial wall-clock time on Chimera (n_cpu=4): 243s average (178-309s range) --
much better than the conservative laptop-based estimate in an earlier draft of this
document.

This placement-success gap is plausibly a real, physically sensible effect (a
faster-arriving tier saturates the eligible-space budget faster, self-limiting realized
growth) rather than a bug -- but it means the "naive expected MW" column above must not
be used for planning without this correction. **Still open:** same measurement for
low/medium tiers (less crowding expected to help, not yet confirmed), and whether
~28.5% success for high is acceptable as-is or whether lambda_high should be
recalibrated down given how many Poisson-triggered attempts are "wasted" on failed
placements.

`REGIONAL_CAPACITY_MW` / `fragment_budget`'s delta-based generation is **replaced**,
not layered on top of, by this mechanism.

### 2.8 Self's operating horizon — 25 years
Confirmed this session. This is the fixed length of time self is simulated for; new
neighbor arrivals happening (in the underlying Poisson process) after this window are
simply never generated for that trial.

### 2.9 Sampling method — Latin Hypercube Sampling (LHS)
The continuous, per-trial-sampled axes (wind rose / climate PCA draw, spatial placement
of speculative neighbors, and now the arrival-gap draws) move from independent random
sampling to **LHS**, for better coverage of the joint space at a given trial budget.
**Number of draws per factorial cell (wake model × tier) is intentionally left open**
pending a small shakedown run — see §8.

### 2.10 Code location — `Monte_Carlo_v2/`
New directory, sibling to `Monte_Carlo/`. The v1 code is left untouched as a working
reference (same pattern already used for `layout_generation_v2.ipynb` sitting alongside
`layout_generation.ipynb`).

---

## 3. MW is now a derived output, not an input

Since neither farm count nor total capacity is chosen in advance anymore (§2.7), any MW
figure quoted for a trial is now **computed after the fact** from realized geometry —
`turbines_that_fit × turbine_rated_MW`, summed however the caller wants to slice it
(per farm, per segment, per whole trial). This still needs to be logged; see §8 for the
schema implication. The v1 columns `requested_mw` and `delivered_pct` (which compared
an actual outcome against a target) **lose their meaning and are dropped** — there is no
more target to compare against, only a realized outcome.

---

## 4. The time-axis mechanism (per trial, per tier)

1. Take the tier's λ (§2.7).
2. Draw inter-arrival gaps `τ₁, τ₂, ... ~ Exponential(λ)`; cumulative-sum them to get
   calendar years `t₁ < t₂ < ...`.
3. Stop drawing once the cumulative sum exceeds the 25-year horizon T (§2.8) — however
   many arrivals happened to fall within the window is itself random (mean λ·T, but not
   fixed), which is what makes "low" tier's eventual outcome genuinely smaller than
   "high," without a separate cap.
4. For each arrival that falls within the horizon: draw a farm size from
   `REAL_FARM_SIZES_MW` (same empirical distribution v1 already used inside
   `fragment_budget`), draw a turbine from the 4-option set (§2.4), generate its shape
   via the existing `staircase`/`generate_farm` machinery at fixed 8.0D spacing (§2.3),
   checked against the existing eligibility grid and against every farm placed so far
   (cumulative — later arrivals compete for shrinking space, see §8).
5. Sort all accepted farms by arrival year. Self's AEP is a **step function**: constant
   between arrivals, recomputed at each new arrival (the three-way decomposition of §7,
   evaluated against whichever neighbors are active by that point).
6. Each step ("segment") is logged with its own start year, end year, active-neighbor
   set, and self AEP (see §8). The whole trial's scenario summary is the
   duration-weighted mean AEP across its segments — a derived quantity, never the only
   thing stored (raw segments remain the primary artifact, exactly as `PIPELINE_DESIGN_v1.md`
   §4.4 already specified, so that re-weighting — e.g. by an economic discount rate later
   — never requires re-simulating).

---

## 5. Feasibility check performed this session

Before committing to the high tier's ~50 expected arrivals, the actual eligibility grid
was built and measured (`layout.build_eligibility_grid(radius_km=120.0)`, run live this
session against the real bathymetry/coastline/federal-boundary data):

| Quantity | Value |
|---|---|
| Total eligible area within 120 km | 20,435 km² |
| Already occupied by the 13 real leases | 3,673 km² |
| **Net area available for new farms** | **16,762 km²** |
| Approx. footprint of one average-size farm (~875 MW) at fixed 8.0D spacing | ~205–230 km² (similar across all 4 turbine options) |
| **Naive packing ceiling** (net area ÷ footprint, square-packing approximation) | **~75–80 farms** |

High tier's ~50 expected arrivals is therefore roughly **62% of the naive ceiling** —
plausible, but tight enough that real placement-success rates should be watched rather
than assumed. Three effects not captured by the naive estimate all push the *true*
ceiling lower: (a) staircase-walk shapes are irregular and pack less efficiently than a
perfect square, (b) `MIN_FARM_SEPARATION_M` (800 m) consumes extra space between every
pair of farms, (c) the real 13-lease cluster already occupies the shallowest,
closest-to-shore (i.e. best) locations, pushing later speculative arrivals toward worse
distance bands with a higher rejection rate. This is already self-instrumenting — the
existing `layout_attempts.csv`-style log records `success`/`n_tries` per placement
attempt — so a small shakedown run (layout generation only, no AEP, several seeds run
out to 25 simulated years, high tier specifically) is recommended before committing to
final per-cell trial counts (§8).

---

## 6. Reference material read in full this session

Both consulted at Bruno's request, in addition to everything already logged in
`PIPELINE_DESIGN_v1.md` §1.

**McCoy, A. et al. (2024). *Offshore Wind Market Report: 2024 Edition Executive
Summary*. NREL/TP-5000-90921.**
(`/Users/brunoboer/Documents/Risk Study/Main Articles/NLR_REPORT.pdf`)
Confirmed the pipeline-growth and state-mandate figures already cited in v1 §2 (source
#6). New contribution this session: identified **SouthCoast Wind 1 by name** among the
eight 2023 offtake-contract cancellations — the direct empirical anchor for choosing it
as self (§2.6). Also the source for the turbine-size growth trend (7.7→9.7 MW,
2022→2023; 15 MW platform; 18–22 MW Chinese OEM plans) backing the turbine set in §2.4.

**Warder, S.C. & Piggott, M.D. (2025). "The future of offshore wind power production:
Wake and climate impacts." *Applied Energy* 380, 124956.**
(`/Users/brunoboer/Documents/Risk Study/Main Articles/Climate_Wake_Effects.pdf`)
Already summarized at a high level in v1 §1 (source #3); read in full this session.
Confirms TurbOPark (i.e. `Nygaard_2022`, §2.5) as validated specifically for long-range,
inter-cluster wake interactions against real mast (FINO1) data — direct literature
support for keeping it in the ensemble. Its 3-way wake-loss decomposition
(intra-farm/intra-cluster/inter-cluster) is conceptually the same split the WESL
pipeline's `isolated_aep`/`real_neighbors_aep`/`full_cluster_aep` triplet already
computes (§7) — external validation that the existing AEP-decomposition structure
matches how this is done in the peer-reviewed literature. Also reconfirms that two
decades of build-out outweighs roughly a century of RCP8.5 climate change in magnitude
— supporting the existing choice to treat climate as interannual PCA/wind-rose
variability only, not a secular climate-change trend (already the plan, unchanged).

---

## 7. Reminder: the isolated / real_neighbors / full_cluster AEP decomposition

Three PyWake simulations of self, differing only in which other farms' turbines are
present in the same run:

| Quantity | Turbines included | What it isolates |
|---|---|---|
| `isolated_aep_gwh` | self only | Baseline — only self's own internal wake losses |
| `real_neighbors_aep_gwh` | self + already-built/committed real neighbors only | AEP loss from what is already certain today |
| `full_cluster_aep_gwh` | self + real neighbors + every speculative/new farm active at that point in time | AEP loss under the complete scenario, including future uncertainty |

The gap between `full_cluster_loss_pct` and `real_neighbors_loss_pct` is the risk
component attributable specifically to uncertain future build-out — the central
quantity this whole study exists to characterize. Under the v2 time axis (§4), this
same three-way split is recomputed at every segment, not just once per trial.

---

## 8. Logging / schema implications (not yet built)

- **Per-arrival log** (replaces v1's `layout_attempts.csv`): one row per farm generation
  attempt, now triggered by an arrival event rather than a budget-fragment item —
  `trial_id, tier, arrival_year, turbine, spacing_d, n_tries, success, n_turbines,
  delivered_mw, template`.
- **Segment table** (new, per `PIPELINE_DESIGN_v1.md` §4.4, not yet implemented):
  `scenario_id, segment_start_year, segment_end_year, duration_years,
  active_neighbor_ids, isolated_aep_gwh, real_neighbors_aep_gwh, full_cluster_aep_gwh`.
- **Per-trial summary**: `n_new_farms`, total `delivered_mw` (both now purely derived,
  §3), duration-weighted mean AEP per wake model.
- `requested_mw` / `delivered_pct` columns from v1 are dropped (§3).

---

## 9. Open items carried forward

1. **Bastankhah k parameter** (§2.5): using py_wake's own DTU example default
   (k = 0.0324555) unless the WESL group already has its own calibrated value from the
   Rodrigues 2025 / Moura 2025 papers — needs a quick check, not yet confirmed either
   way.
2. **Number of LHS draws per factorial cell** (§2.9): intentionally left open pending
   the shakedown run recommended in §5, and pending real per-trial timing under the new,
   more expensive per-segment AEP evaluation (each trial now runs *k* arrivals × 3 wake
   models of PyWake calls, instead of v1's single call × 2 models — a substantial cost
   increase per trial that should be measured, not assumed, before sizing a production
   campaign).
3. **High-tier placement feasibility** (§5): **now measured, not just estimated** — a
   real Chimera shakedown (2026-08-11) confirmed ~28-31% placement success across 3
   seeds, ~5x lower realized MW than the naive table in §2.7 assumed. Still open: same
   measurement for low/medium tiers, and a decision on whether ~30% success for high is
   acceptable as-is (an emergent, physically plausible space-saturation effect) or
   whether λ_high should be recalibrated down given how many attempts are "wasted."

---

## 10. Next step

Build `Monte_Carlo_v2/` as a sibling directory to `Monte_Carlo/`, implementing the
decisions above. `Monte_Carlo/` (v1) stays untouched as a working reference throughout.
