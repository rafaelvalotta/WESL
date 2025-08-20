# WEC‑Sim → MATLAB `.mat` Dataset: Step‑by‑Step Guide

This short guide explains **how to produce a `.mat` file** from WEC‑Sim runs using MATLAB. It starts with adding WEC‑Sim to your MATLAB path (via `addWecSimSource.m`) and ends with saving a compact, Python‑friendly dataset.

---

## 1) One‑time setup

1. **Start MATLAB** and run:
   ```matlab
   cd('<path-you-git-cloned-Wec-Sim>/WEC-Sim');
   addWecSimSource;        % adds $WECSIM/source (and subfolders) to the path
   which simulationClass    % sanity check: should resolve under .../source/objects
   slLibraryBrowser         % optional: refresh to see the WEC‑Sim block library
   ```

> Tip: If you want this every launch, copy the contents of `source/addWecSimSource.m` into `~/Documents/MATLAB/startup.m` and set `wecSimSource` to your local path.

---

## 2) Case location and why we pass parameters via workspace

Move to the case folder you want to run (e.g., the RM3 example):
```matlab
cd('<path-you-git-cloned-Wec-Sim>/WEC-Sim/examples/RM3');
```

We keep `wecSimInputFile.m` **generic** and let a driver script set the wave parameters (`Height`, `Period`, `Direction`). This separation lets you:
- sweep many conditions without editing the input file each time,
- keep runs **reproducible** (via `rng(seed)`), and
- save **exact** inputs alongside outputs in one dataset.

---

## 3) Driver script to generate the dataset (`rm3RegWaveDataGen.m`)

This script randomizes wave conditions, runs WEC‑Sim sequentially (no parallel pool), computes **peak absorbed power** for each run, and saves a compact `.mat` file.

---

## 4) Steps to reproduce

1. Add WEC‑Sim to your MATLAB path by running addWecSimSource.m from the WEC‑Sim root directory. This ensures all classes and functions (e.g., simulationClass, wecSim) are available in your session.

2. Open the RM3 case directory (the example you’re using). Work from that folder for the rest of the steps.

3. Confirm your wecSimInputFile.m is set up to accept wave parameters from the MATLAB workspace (Height, Period, Direction) and to disable pop‑up figures. No edits are needed here beyond what’s already described in the earlier sections.

4. Place the driver script rm3RegWaveDataGen.m in the RM3 folder. This script randomizes the wave conditions, runs WEC‑Sim sequentially, computes peak absorbed power, and writes a dataset.

5. Run the driver script rm3RegWaveDataGen.m from MATLAB. The script will execute all simulations and save the output dataset.

6. Locate the output file wecSimPeakPower_validation.mat in the RM3 folder.

---

## 5) Troubleshooting quickies

- **`Unrecognized function or variable 'simulationClass'`** → You didn’t add the WEC‑Sim path. Re‑run `addWecSimSource` (or fix `startup.m`).
- **WEC‑Sim library not visible in Simulink** → Run `slLibraryBrowser` and press *Refresh*. Clearing the Simulink cache can also help.
- **`.h5` hydroData missing/empty** → Ensure Git‑LFS pulled large files (or rebuild via `bemio`).
- **Power mostly negative** → Remember production is `-F_pto * v_rel`. We zero negative values to focus on absorbed energy.
- **Figures popping up** → Ensure `simu.explorer = 'off'` in your input file.


