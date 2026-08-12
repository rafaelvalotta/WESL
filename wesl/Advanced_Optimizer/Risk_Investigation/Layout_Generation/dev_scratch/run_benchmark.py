import time
import numpy as np
import pandas as pd

HARNESS = open("v2_harness.py").read()
ENGINES = open("v2_engines.py").read()
exec(compile(HARNESS, "v2_harness.py", "exec"), globals())
exec(compile(ENGINES, "v2_engines.py", "exec"), globals())

N_SEEDS = 12
TARGET_LEVEL = "high"

print("\n=== Building precomputed eligibility grid (one-time cost) ===")
grid = build_eligibility_grid(resolution_m=300.0)
print(f"grid: {grid['nx']}x{grid['ny']} = {grid['nx']*grid['ny']:,} cells, "
      f"built in {grid['build_time']:.2f}s")
is_eligible_fast = make_grid_lookup(grid)

# sanity check: grid lookup should agree with live KDTree checks on a random sample
rng_check = np.random.default_rng(0)
test_x = rng_check.uniform(cluster_all_x.min() - 20000, cluster_all_x.max() + 90000, 2000)
test_y = rng_check.uniform(cluster_all_y.min() - 20000, cluster_all_y.max() + 90000, 2000)
live = is_federal(test_x, test_y) & is_shallow_enough(test_x, test_y)
fast = is_eligible_fast(test_x, test_y)
agree = (live == fast).mean()
print(f"grid-vs-live agreement on 2000 random test points: {agree*100:.2f}%")

# ============================================================= Run V1 (baseline, instrumented passively)
print(f"\n=== V1 (original: 200 tries, live KDTree checks), {N_SEEDS} seeds, level={TARGET_LEVEL} ===")
v1_logs = []
v1_results = []
t0 = time.time()
for seed in range(N_SEEDS):
    rows = []
    cx, cy, scenario = populate_scenario_v1(TARGET_LEVEL, np.random.default_rng(seed), log_rows=rows)
    v1_logs.extend([{**r, "seed": seed} for r in rows])
    v1_results.append((seed, cx, cy, scenario))
v1_time = time.time() - t0
print(f"V1 total time: {v1_time:.1f}s ({v1_time/N_SEEDS:.2f}s/draw)")

# ============================================================= Run V2 (grid + backoff + instrumentation)
print(f"\n=== V2 (grid lookup + bounded backoff), {N_SEEDS} seeds, level={TARGET_LEVEL} ===")
v2_logs = []
v2_results = []
t0 = time.time()
for seed in range(N_SEEDS):
    rows = []
    cx, cy, scenario = populate_scenario_v2(TARGET_LEVEL, np.random.default_rng(seed), is_eligible_fast, log_rows=rows)
    v2_logs.extend([{**r, "seed": seed} for r in rows])
    v2_results.append((seed, cx, cy, scenario))
v2_time = time.time() - t0
print(f"V2 total time: {v2_time:.1f}s ({v2_time/N_SEEDS:.2f}s/draw)")

print(f"\nSpeedup: {v1_time / v2_time:.2f}x")

# ============================================================= Capacity delivered vs requested
df1 = pd.DataFrame(v1_logs)
df2 = pd.DataFrame(v2_logs)

req1, del1 = df1["target_mw"].sum(), df1["delivered_mw"].sum()
req2, del2 = df2["target_mw"].sum(), df2["delivered_mw"].sum()
print(f"\nV1: requested={req1:.0f} MW, delivered={del1:.0f} MW ({del1/req1*100:.1f}%), "
      f"n_farms_dropped={len(df1[~df1['success']])}/{len(df1)}")
print(f"V2: requested={req2:.0f} MW, delivered={del2:.0f} MW ({del2/req2*100:.1f}%), "
      f"n_farms_dropped={len(df2[df2['delivered_mw']==0])}/{len(df2)}")
print(f"\nV2 backoff usage (fraction_used distribution among placed farms):")
print(df2[df2["delivered_mw"] > 0]["fraction_used"].value_counts().sort_index())

# ============================================================= Does V2 still find big near-field farms?
print("\n=== Near-field (<20km) large farms (>=500MW), V1 vs V2 ===")
near1 = df1[(df1["band"].apply(lambda b: b[1] <= 20)) & (df1["success"]) & (df1["target_mw"] >= 500)]
near2 = df2[(df2["band"].apply(lambda b: b[1] <= 20)) & (df2["fraction_used"] == 1.0) & (df2["target_mw"] >= 500)]
print(f"V1: {len(near1)} full-size near-field farms >=500MW placed (out of "
      f"{len(df1[(df1['band'].apply(lambda b: b[1] <= 20)) & (df1['target_mw']>=500)])} attempted)")
print(f"V2: {len(near2)} full-size near-field farms >=500MW placed (out of "
      f"{len(df2[(df2['band'].apply(lambda b: b[1] <= 20)) & (df2['target_mw']>=500)])} attempted)")

df1.to_csv("bench_v1_log.csv", index=False)
df2.to_csv("bench_v2_log.csv", index=False)
print("\nSaved bench_v1_log.csv, bench_v2_log.csv")
