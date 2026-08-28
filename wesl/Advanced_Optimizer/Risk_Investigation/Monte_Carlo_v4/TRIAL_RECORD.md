# Trial Record -- seed=0, profiled run

First full end-to-end trial through the v3 pipeline (`PIPELINE.md`). Run locally
(MacBook, `wesl_jul_31` conda env, single process, no concurrency), single wake
model (`turbopark`). Command: `python orchestrator.py --seed 0 --out profiled`.
Output: `results/profiled_seed0/{trials,arrivals,segments}.csv`.

## Headline

534.1s wall clock, 22 farms, 18 segments, zero errors.

## What this trial drew

| | |
|---|---|
| Speculative farms (N) | 9 |
| Self turbine / spacing | 13 MW / 7D |
| Climate delta_mu | +1.50 deg |
| Climate S_v | 0.925 |
| Wake model | turbopark |

22 farms total: 4 real-layout (Vineyard Wind, Revolution Wind, South Fork Wind,
Sunrise Wind South) + self (SouthCoast Wind) + 8 cluster-infill + 9 speculative.
Full per-farm arrival years, turbines and spacing are in `arrivals.csv`.

## Time breakdown by stage

| Stage | Time | Share |
|---|---|---|
| Generation (cluster + layout + climate) | 0.60s | 0.1% |
| Tiling (grid fill) | 0.02s | ~0% |
| PyWake model construction (18 calls) | 0.01s | ~0% |
| Isolated AEP simulation (18 calls) | 19.32s | 3.6% |
| Active AEP simulation (18 calls) | 514.06s | 96.3% |

The active-AEP simulation is essentially the entire cost of a trial.

## Per-segment cost vs. active turbine count

Cost grows roughly quadratically with turbine count, not linearly. The last 5
segments (13-17) alone total 322.6s -- 60% of the whole trial.

| Segment | Year | Active turbines | Active-sim time |
|---|---|---|---|
| 0 | 2026.0 | 450 | 3.90s |
| 3 | 2030.4 | 515 | 5.52s |
| 6 | 2045.3 | 675 | 9.28s |
| 9 | 2046.5 | 1135 | 23.78s |
| 13 | 2047.5 | 1343 | 33.15s |
| 15 | 2049.0 | 1832 | 64.95s |
| 17 | 2050.0 (final) | 1967 | 89.31s |

Full 18-row table in `segments.csv`.

## AEP trajectory (physical sanity check)

`isolated_aep_gwh` is constant at 8812.25 GWh across all 18 segments (self alone
never changes -- confirms the segment loop is wired correctly). Active AEP
declines and loss climbs monotonically as the world builds up around self:

| Year | Active farms | AEP loss |
|---|---|---|
| 2026.0 (start) | 5 | 1.48% |
| 2046.1 | 12 | 8.16% |
| 2048.4 | 19 | 26.63% |
| 2050.0 (end) | 22 | **28.00%** |

## Local parallelism experiment (for the record, not a production number)

5 concurrent trials, unconstrained BLAS threads: still running past 15 minutes,
none finished (baseline solo is 8.6 min) -- killed. Re-run with
`OMP_NUM_THREADS=2` etc. capped: same outcome, still running past 11 minutes --
killed. Conclusion: this machine (mixed performance/efficiency cores, shared
memory bandwidth, thermal throttling under sustained multi-core load) is not a
valid proxy for how trial-level parallelism will behave on dedicated HPC cores.
The single-core cost number above (534.1s) is solid; the parallel-scaling
question is deferred to a real Chimera shakedown array, per existing project
convention (`Monte_Carlo_v2/hpc/README.md`).

## Campaign cost extrapolation (single-core, unparallelized)

| Trials | Wall clock |
|---|---|
| 100 | ~14.8 h |
| 1,000 | ~6.2 days |

Purely illustrative until real multi-core numbers exist.

---

# Trial Record #2 -- seed=0, `supergaussian` + `evaluate_background_points()` EXPERIMENT,
# NOT adopted (2026-08-20)

**Superseded by Trial Record #3 below -- kept for history, not the production number.**
Same scenario draw as above (identical seed=0, so identical farms/turbines/climate --
`arrivals.csv` is unchanged), run with `ACTIVE_MODEL=supergaussian` and the experimental
background-field/`Points`-based `aep.evaluate_background_points()` in place of the
all-merged `aep.evaluate()` (see PIPELINE.md, [[pywake-honeycomb-adjacency-fix]]). Run
locally (MacBook, `wesl_jul_31` conda env, single process). Command: `python
orchestrator.py --seed 0 --out smoke_supergaussian_bgpoints` (with `aep.evaluate`
temporarily pointed at `evaluate_background_points` for this one run -- that swap was
reverted afterward, see Trial Record #3). Output:
`results/smoke_supergaussian_bgpoints_seed0/{trials,arrivals,segments}.csv`.

## Headline

**128.6s wall clock** (vs. Trial Record #1's 534.1s) -- 4.15x faster than the original
`turbopark`+merged baseline, 22 farms, 18 segments, zero errors. Correctness
cross-checked: segment 17's `active_aep_gwh` (6603.853595643516 GWh) matches the
standalone `evaluate_background_points()` benchmark from the same scenario in
[[pywake-honeycomb-adjacency-fix]] to full float precision.

**This number bundles two independent changes, and a follow-up full-trial measurement
of `evaluate_merged()` (supergaussian, no background-field trick -- see Trial Record #3)
found the technique's own share of this 4.15x is noise-level, not real:**
- switching wake model alone (`turbopark` -> `supergaussian`) already gives 534.1s ->
  133.8s = 4.0x on its own (Trial Record #3, full 18-segment trial, `evaluate()` merged);
- `evaluate_background_points()` vs. `evaluate()` merged, same model, full trial: 128.6s
  vs. 133.8s = **1.04x, noise-level** (per-segment spot checks earlier suggested 1.08x-1.22x,
  but that didn't hold up in the full-trial aggregate -- see `evaluate_background_points()`'s
  docstring in `aep.py` for why: it only removes the self-self/self-neighbour cross terms
  from the O(n^2) cost, a shrinking fraction of the total as neighbours vastly outgrow self
  over a trial).

**Conclusion: the background-field technique was NOT adopted for production** -- the real
complexity it adds (custom `XRSite` construction, positional-index bookkeeping, a
turbopark-incompatibility landmine) isn't worth a noise-level aggregate speedup.
`aep.evaluate()` reverted to the simple all-merged implementation (works with any model,
including turbopark). `evaluate_background_points()` stays in `aep.py`, documented and
working, for whoever pursues the incremental-caching idea noted in its docstring, or
root-causes the turbopark crash and finds the technique pays off better there.

## Time breakdown by stage

| Stage | Time | Share |
|---|---|---|
| Isolated AEP (18 calls) | 4.81s | 3.7% |
| Neighbours-only simulation (18 calls) | 92.05s | 71.6% |
| `flow_map(Points(...))` query at self's positions (18 calls) | 26.30s | 20.4% |
| Self-vs-background-field simulation (18 calls) | 4.80s | 3.7% |
| PyWake model construction | 0.001s | ~0% |

Unlike Trial Record #1 (where "active AEP" was one undifferentiated 514s), this
architecture separates cleanly: **the neighbours-only simulation is now the dominant
cost (71.6%)**, not the self-vs-field step. This is expected -- that call is still
O(n_neighbours^2)-ish (same PropagateDownwind cost structure as the old merged call, just
without self's own turbines mixed in); `flow_map(Points(...))` is comparatively cheap
because it only queries `n_self` (constant, ~215 turbines all trial) points per segment,
not a `grid_n^2` mesh. **This points at the next lever, if pursued**: the neighbours-only
simulation result changes only by ADDING turbines each segment (neighbours never move or
leave within a trial) -- an incremental/cached wake computation across segments could in
principle cut most of this 92s, unlike the old merged approach where self was mixed in
and no such incremental structure existed. Not attempted yet.

## Per-segment cost vs. active turbine count

| Segment | Year | Active turbines | `t_active_s` (neighbours+points+self) |
|---|---|---|---|
| 0 | 2026.0 | 450 | 0.96s |
| 6 | 2045.3 | 675 | 2.14s |
| 9 | 2046.5 | 1135 | 5.98s |
| 13 | 2047.5 | 1343 | 8.32s |
| 15 | 2049.0 | 1832 | 15.79s |
| 17 | 2050.0 (final) | 1967 | 18.58s |

Same qualitative super-linear growth as Trial Record #1, just at a much lower absolute
level (18.58s vs. 89.31s on the identical final-segment turbine count). Full 18-row table
in `segments.csv`.

## AEP trajectory (physical sanity check)

`isolated_aep_gwh` constant at 9014.10 GWh across all 18 segments (different from Trial
Record #1's 8812.25 GWh -- expected, `supergaussian` isn't `turbopark`, self's own
internal-wake AEP is model-dependent). Loss climbs monotonically, same qualitative shape
as Trial Record #1:

| Year | Active farms | AEP loss |
|---|---|---|
| 2026.0 (start) | 5 | 0.80% |
| 2046.1 | 11 | 1.33% |
| 2047.5 | 18 | 8.48% |
| 2050.0 (end) | 22 | **26.74%** |

Close to Trial Record #1's 28.00% final loss (different model, same scenario -- broadly
consistent, no red flag).

## Campaign cost extrapolation (single-core, unparallelized)

| Trials | Wall clock |
|---|---|
| 100 | ~3.6 h |
| 1,000 | ~1.5 days |

Superseded by Trial Record #3's numbers below (near-identical, since the background-field
technique turned out to add ~nothing) -- kept here for this record's own completeness.

---

# Trial Record #3 -- seed=0, `supergaussian` + `evaluate()` (merged) -- CURRENT PRODUCTION (2026-08-20)

The actual production configuration after Trial Record #2's technique was not adopted:
`ACTIVE_MODEL=supergaussian`, `aep.evaluate()` back to the simple all-merged
implementation (works with any wake model). Same seed=0 scenario as Trial Records #1-2.
Command: `python orchestrator.py --seed 0 --out <name>` (uses whatever `aep.evaluate()`
and `wake_models.ACTIVE_MODEL` currently are -- no special flags needed, this IS the
default now).

## Headline

**133.8s wall clock** (vs. Trial Record #1's 534.1s, `turbopark`+merged) -- **4.0x
faster**, all from the model switch. 22 farms, 18 segments (same boundaries as #1/#2,
identical scenario), zero errors.

## Per-segment cost vs. active turbine count

| Segment | Active turbines | `t_active_s` |
|---|---|---|
| 0 | 450 | 0.99s |
| 6 | 675 | 2.25s |
| 9 | 1135 | 6.19s |
| 13 | 1343 | 8.61s |
| 15 | 1832 | 16.18s |
| 17 (final) | 1967 | 19.94s |

Same qualitative super-linear growth as Trial Records #1/#2 (expected -- same all-merged
O(n^2)-ish cost structure as #1, just with a cheaper wake model). Essentially identical to
Trial Record #2's per-segment numbers (e.g. final segment 19.94s here vs. 18.58s there) --
confirms the background-field technique's ~1.04x aggregate difference is real-but-tiny,
not a measurement artifact.

## Campaign cost extrapolation (single-core, unparallelized)

| Trials | Wall clock |
|---|---|
| 100 | ~3.7 h |
| 1,000 | ~1.5 days |

**This is the number to use going forward** (supersedes Trial Record #1's turbopark
numbers and Trial Record #2's background-field numbers, though all three are close in
this case). Still purely illustrative until real multi-core numbers exist from a Chimera
shakedown array (`hpc/slurm/submit_array.sh`) -- see `hpc/README.md`. Trial cost also
varies noticeably by seed (each one draws a different number of speculative farms, up to
16) -- two other seeds spot-checked locally under the same supergaussian/`evaluate()`
setup came in at 173.7s and 447.0s, well outside this seed=0 number -- don't treat 133.8s
as a tight per-trial estimate, just an order-of-magnitude anchor pending the real
shakedown.
