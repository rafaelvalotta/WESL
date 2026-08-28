# Monte_Carlo_v4/hpc -- Chimera (UMass Boston) submission layer

## 2026-08-27: readiness for a v4 production campaign

Code is ready and verified locally (`run_trial_range.py` + `merge_results.py`
both dry-run tested end-to-end this date, both scenarios, correct
`results/<scenario>/<run_id>/...` nesting, `--scenario` required everywhere).
`sync_to_chimera.sh`/`sync_from_chimera.sh` were still pointing at
`Monte_Carlo_v3` paths (leftover from before the v4 fork) -- fixed today.

**Real local per-trial timing, post-redesign** (not yet re-measured on
Chimera): scenario_1 ~11-21s/trial, scenario_2 ~57-83s/trial (scenario_2
starts every trial from a bigger fixed baseline -- the whole 13-lease cluster
vs. scenario_1's 5). Both far cheaper than the pre-redesign v3 numbers below
that `submit_array.sh`'s `--mem=64G`/`--time=06:00:00` defaults were sized
for -- **not yet re-validated for v4**. Turbine counts (the memory driver)
are similar order of magnitude to before, so `--mem=64G` is probably still
about right; `--time=06:00:00` is probably now generous rather than tight.

**Still needed before a real 10k x 2-scenario campaign**: a real Chimera
shakedown per scenario (`--array=0-4 slurm/submit_array.sh shakedown_s1
scenario_1 2 supergaussian`, then the same for `scenario_2`) -- read real
`slurm_out/`/`sacct` numbers, THEN size the full array's `--time`/`--mem`
(override via `sbatch --time=... --mem=...`, don't just trust the script
defaults). This is not something to skip -- local-machine timing is
explicitly not a valid proxy for array-job scaling (mixed performance/
efficiency cores, shared memory bandwidth, thermal throttling -- see Open
Items below and in `../PIPELINE.md`).

**10k x 2 scenarios means TWO separate `sbatch` submissions**, not one --
`--scenario` is a required positional arg per job, can't vary within a single
array. `MaxArraySize=1001` (per `../CAMPAIGN_LOG.md`) still means
`trials_per_task=10` -> `--array=0-999` per scenario to fit 10k trials each.

Doesn't duplicate pipeline logic -- imports `../orchestrator.py` etc. directly. Adds only
what's specific to running on Chimera: SLURM scripts, an array-job entry point, results
merging. Same layout convention as `Monte_Carlo_v2/hpc/` (folded into `Monte_Carlo_v3/`
as one subfolder, single place to sync, single `results/` dir regardless of launch method).

## Confirmed setup (carried over from `../../Monte_Carlo_v2/hpc/README.md`, still accurate)

- **Partition**: `EPYC9565`. **Login**: `chimerahead.umb.edu`. **User**: `bruno.boer001`.
- **Env**: `Advanced_env` (same one v2 uses -- Miniforge, `python=3.11`, `py_wake==2.6.20`).
  v3 adds exactly one new dependency on top of v2's pinned set: `pyarrow` (parquet output
  -- see `requirements_advanced_env.txt`). Run `pip install -r requirements_advanced_env.txt`
  once in `Advanced_env` on Chimera before the first v3 array job.
- Scratch storage on Chimera is not backed up -- copy `results/<run_id>/` off after
  every run.
- Acknowledge Chimera in any publication: *"the use of the supercomputing facilities
  managed by the Research Computing Department at the University of Massachusetts
  Boston."*

## Known bug, fixed 2026-08-20: don't raise `--n-cpu` above 1

The first real shakedown deadlocked: every task's PyWake worker processes went to
sleep state (`Sl`, near-zero `%CPU`) and stayed stuck 18+ minutes on a 144-core node
with 124 cores sitting idle -- not a resource-contention problem, a **post-fork
BLAS/multiprocessing lock-inheritance deadlock**. Cause: `submit_array.sh` was passing
`--n-cpu $SLURM_CPUS_PER_TASK` (4) to `run_trial_range.py`, copying v2's convention --
but v3's `aep.py` has its OWN established rule (`n_cpu=1`, PyWake's internal
flow-case parallelism deliberately left off, see PIPELINE.md/aep.py: "parallelism is
meant to happen across trials... not nested inside one"). v2's "safe to raise, it's a
plain OS process not a Pool worker" reasoning does not transfer to v3. **Fixed**:
`submit_array.sh` now hardcodes `--n-cpu 1`; `--cpus-per-task=4` is kept anyway since
BLAS's own internal threading (no fork involved at `n_cpu=1`) can still safely use the
extra cores. If you ever add a custom entry point here, keep `--n-cpu 1` unless you
re-validate this from scratch. Full writeup: [[wesl-risk-study-nsf-aim2]] memory.

## Known bug #2, fixed 2026-08-20: `--mem=8G` too low, OOM-killed 4/5 shakedown tasks

Re-ran the shakedown after the `--n-cpu` fix above (no more deadlock) and 4 of 5 tasks
got `OUT_OF_MEMORY`-killed almost immediately (`sacct` MaxRSS ~8.38GB against
`ReqMem=8G`; the one task that survived peaked at 7.98GB, right at the edge). Cause:
`aep.evaluate()` (merged) builds a pairwise turbine-i x turbine-j deficit array per
(wd,ws) -- `O(n_turbines^2 * 12 * 23 * 8 bytes)`; at ~2000 turbines (seed=0's biggest
segment) that one array alone is already ~8.8GB, and `N_RANGE` draws up to 16
speculative farms (more than seed=0's 9), so other seeds can need noticeably more.
Local Mac runs never hit this because nothing there enforced an 8GB cgroup ceiling.
**Fixed**: `submit_array.sh`'s `#SBATCH --mem` raised `8G` -> `32G` (`chimera22` has
773GB RAM, essentially idle -- no reason to be stingy). Re-tune down later with real
`sacct --format=MaxRSS` numbers across more seeds once a shakedown runs clean.

## Known bug #3, found mid-production-run 2026-08-20: `--time=02:00:00` not rescaled
## for `trials_per_task=10`

The 10,000-trial production array (4 partitions, `trials_per_task=10` -- forced up from
the shakedown's 2, to fit 10k trials under `MaxArraySize=1001`) hit a real risk mid-run:
a few 10-trial tasks reached 1:48-1:49 elapsed, dangerously close to the `--time=02:00:00`
inherited unchanged from the `trials_per_task=2` shakedown default. `--time` was never
rescaled when `trials_per_task` went up 5x. **Fixed**: default raised to `06:00:00`.
`run_trial_range.py` also now saves incrementally after every trial (not just once at
the end) specifically because of this -- a walltime kill used to lose an entire task's
trials, not just the slow one. If you raise `trials_per_task` beyond 10, raise `--time`
proportionally too (`sbatch --time=HH:MM:SS ...` overrides the script default, same
mechanism as `--partition=`/`--array=`). A live job's time limit can also be extended
without resubmitting: `scontrol update JobId=<id> TimeLimit=HH:MM:SS` (worked here to
rescue two at-risk running tasks -- subject to cluster/QoS policy, not guaranteed).

## `pomplun` partition (chimera21, `cs672` class account) -- authorized for this project 2026-08-20

224 cores / 2TB RAM on a single node (`chimera21`), 8x H200 GPUs (irrelevant here, PyWake
isn't GPU-accelerated). Access was originally granted for a CS672 (Neural Networks)
class project -- **confirmed authorized by the user for WESL Monte Carlo work too**
before using it here; don't assume this authorization carries to other class/restricted
accounts without asking again. Needs `--account=cs672 --qos=pomplun --partition=pomplun`
explicitly (`AllowAccounts=cs`/`AllowQos=pomplun` per `scontrol show partition pomplun`)
-- the plain `--partition=` used for the other 4 partitions is not enough on its own.
**`DefaultTime=01:00:00`** if `--time` isn't passed explicitly -- much shorter than the
`06:00:00` default in `submit_array.sh`, easy trap if pomplun is ever added to a
multi-partition comma-list job (which also wouldn't work directly here anyway, since
`--account`/`--qos` apply per-job, not per-partition-in-a-list -- treat pomplun as its
own separate `sbatch` submission, not merged into the 4-partition list).
**Not yet performance-tested** -- `slurm/test_pomplun.sh` runs one real, directly-comparable
timing benchmark (same call as the documented 20.64s EPYC9565 number) before trusting it
for anything production-sized, per the lesson two sections up.

## v3-specific notes (SLURM/env lessons still apply to v4 unchanged; timing numbers below are stale -- see 2026-08-27 section up top)

**Model**: defaults to `wake_models.ACTIVE_MODEL` (`supergaussian` as of 2026-08-20,
chosen for campaign throughput). `aep.evaluate()` (the production path) is the
all-merged implementation and works fine with any model, including `turbopark` -- it's
just ~4x more expensive per trial (534.1s vs 133.8s, same seed=0 scenario, see
`../TRIAL_RECORD.md`). `run_trial_range.py` prints a cost warning (not an error) if you
pass `--model turbopark`. `bastankhah` is next in line for a follow-up campaign (also
validated, see `../PIPELINE.md`).

(Aside, not why supergaussian was chosen: there's also an experimental, NOT-default
`aep.evaluate_background_points()` that crashes for turbopark specifically --
`SquaredSum` + "Mirror ground", `AssertionError: SquaredSum only works for deficit - not
speedups". Moot for this hpc layer since it always calls `aep.evaluate()`, never that
function -- see `../aep.py`'s docstrings if curious.)

**Output format**: parquet, not CSV (v2 used CSV) -- `merge_results.py` concatenates
`results/<scenario>/<run_id>/task_XXXX/*.parquet` into
`results/<scenario>/<run_id>/merged/*.parquet` after the array job finishes.

**Local single-core timing** (`../TRIAL_RECORD.md`, Trial Record #2, seed=0,
`supergaussian` + `aep.evaluate()`): **133.8s/trial** (was 534.1s with the original
`turbopark` pipeline -- 4.0x faster, entirely from the model switch; an experimental
background-field technique was tried on top and measured only ~1.04x more in aggregate,
noise-level, not adopted -- see `../aep.py`'s docstrings and TRIAL_RECORD.md if curious).
Extrapolated, unparallelized: ~3.7h/100 trials, ~1.5 days/1,000 trials --
**illustrative only**, not a substitute for a real Chimera shakedown (local-machine
multi-process timing is not a valid proxy for array-job scaling on dedicated cores --
mixed performance/efficiency cores, shared memory bandwidth, thermal throttling; see
`../PIPELINE.md` Open Items). Trial cost also varies a lot by seed (each one draws a
different number of speculative farms, up to 16) -- a local spot-check of 2 more seeds
saw 173.7s and 447.0s for the same supergaussian/`evaluate()` setup, well outside the
seed=0 number. **Run a small shakedown array first** (`--array=0-4`, `trials_per_task=2`)
and read the real wall-clock time in `slurm_out/` before sizing a full production array.

## Usage

```bash
./sync_to_chimera.sh                                   # push code + climate cache
ssh bruno.boer001@chimerahead.umb.edu
cd Summer_26/August/Advanced_Optimizer/Risk_Investigation/Monte_Carlo_v4/hpc
conda activate Advanced_env && pip install -r requirements_advanced_env.txt   # once, adds pyarrow
sbatch --array=0-4 slurm/submit_array.sh shakedown001 scenario_1 2 supergaussian   # small shakedown first
squeue -u bruno.boer001
# once finished, locally:
./sync_from_chimera.sh shakedown001
cd .. && python -c "from hpc.merge_results import merge; merge('scenario_1', 'shakedown001')"
```

Full production array, once the shakedown's per-task wall-clock is known
(`<scenario>` is required -- `scenario_1` or `scenario_2`, see `../SCENARIOS.md`):

```bash
sbatch --array=0-N slurm/submit_array.sh <run_id> <scenario> <trials_per_task> supergaussian
```
