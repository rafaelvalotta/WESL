# Monte_Carlo_v2/hpc -- Chimera (UMass Boston) submission layer

Doesn't duplicate pipeline logic -- imports `../orchestrator.py` etc. directly. Adds only
what's specific to running on Chimera: SLURM scripts, an array-job entry point, results
merging. Folded into `Monte_Carlo_v2/` as one subfolder (not a separate top-level dir
like v1's `Monte_Carlo_HPC/`) so there's a single place to sync and a single `results/`
directory regardless of how a run was launched.

## Confirmed setup (carried over from `../../Monte_Carlo_HPC/README.md`, still accurate)

- **Partition**: `EPYC9565`. **Login**: `chimerahead.umb.edu`. **User**: `bruno.boer001`.
- **Env**: `Advanced_env` (Miniforge, `python=3.11`) -- same `requirements_advanced_env.txt`
  as v1; no new pip dependency for Bastankhah_PorteAgel_2014, it ships inside the already
  pinned `py_wake==2.6.20`.
- Scratch storage on Chimera is not backed up -- copy `results/<run_id>/` off after
  every run.
- Acknowledge Chimera in any publication: *"the use of the supercomputing facilities
  managed by the Research Computing Department at the University of Massachusetts
  Boston."*

## v2-specific cost note

Each v2 trial is more expensive than a v1 trial: it now runs *k* arrivals (however many
speculative farms showed up within the horizon that draw) x 3 wake models of PyWake
calls, instead of v1's single call x 2 models. **Run a small shakedown array first**
(`--array=0-4`, `trials_per_task=2`) and read the real wall-clock time in `slurm_out/`
before sizing a full production array -- don't extrapolate from v1's `PROFILE_RESULTS.md`
numbers directly, they were measured for the old single-call-per-trial shape.

## Usage

```bash
./sync_to_chimera.sh                                   # push code + climate cache
ssh bruno.boer001@chimerahead.umb.edu
cd Summer_26/August/Advanced_Optimizer/Risk_Investigation/Monte_Carlo_v2/hpc
sbatch --array=0-4 slurm/submit_array.sh shakedown001 high 2   # small shakedown first
squeue -u bruno.boer001
# once finished, locally:
./sync_from_chimera.sh shakedown001
```
