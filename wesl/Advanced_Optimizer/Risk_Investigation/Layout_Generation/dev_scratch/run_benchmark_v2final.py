import time
import numpy as np
import pandas as pd

exec(compile(open("v2_harness.py").read(), "v2_harness.py", "exec"), globals())
exec(compile(open("v2_engines.py").read(), "v2_engines.py", "exec"), globals())          # for v1 baseline (90km/substation)
exec(compile(open("v2_final_engines.py").read(), "v2_final_engines.py", "exec"), globals())  # v2 final (150km/coast/grid/backoff)

N_SEEDS = 12
TARGET_LEVEL = "high"

print("\n=== V1 baseline: 90km, subestacoes das 4 fazendas reais, sem backoff ===")
v1_logs = []
t0 = time.time()
for seed in range(N_SEEDS):
    rows = []
    populate_scenario_v1(TARGET_LEVEL, np.random.default_rng(seed), log_rows=rows)
    v1_logs.extend([{**r, "seed": seed} for r in rows])
v1_time = time.time() - t0
df1 = pd.DataFrame(v1_logs)
print(f"tempo: {v1_time:.1f}s ({v1_time/N_SEEDS:.2f}s/draw)")
print(f"requested={df1['target_mw'].sum():.0f} MW, delivered={df1['delivered_mw'].sum():.0f} MW "
      f"({df1['delivered_mw'].sum()/df1['target_mw'].sum()*100:.1f}%)")

print("\n=== V2 final: grade de elegibilidade (150km, federal+raso+costa), backoff, verify-on-accept ===")
grid150 = build_eligibility_grid_150(resolution_m=300.0, radius_km=150.0)
print(f"grid: {grid150['nx']}x{grid150['ny']} = {grid150['nx']*grid150['ny']:,} celulas, "
      f"construida em {grid150['build_time']:.1f}s")
is_eligible_fast_150 = make_grid_lookup(grid150)

v2_logs = []
t0 = time.time()
for seed in range(N_SEEDS):
    rows = []
    populate_scenario_v2f(TARGET_LEVEL, np.random.default_rng(seed), is_eligible_fast_150, log_rows=rows)
    v2_logs.extend([{**r, "seed": seed} for r in rows])
v2_time = time.time() - t0
df2 = pd.DataFrame(v2_logs)
print(f"tempo: {v2_time:.1f}s ({v2_time/N_SEEDS:.2f}s/draw)")
print(f"requested={df2['target_mw'].sum():.0f} MW, delivered={df2['delivered_mw'].sum():.0f} MW "
      f"({df2['delivered_mw'].sum()/df2['target_mw'].sum()*100:.1f}%)")

print(f"\nSpeedup total (v1 90km sem grid -> v2 150km com tudo): {v1_time/v2_time:.2f}x")
print(f"Ganho de capacidade entregue: {df2['delivered_mw'].sum()/df1['delivered_mw'].sum():.2f}x")

print("\nV2 - uso do backoff entre as fazendas colocadas:")
print(df2[df2["delivered_mw"] > 0]["fraction_used"].value_counts().sort_index())

print("\n=== Fazendas grandes (>=500MW) proximas (<20km), full-size ===")
near1 = df1[(df1["band"].apply(lambda b: b[1] <= 20)) & (df1["success"]) & (df1["target_mw"] >= 500)]
near2 = df2[(df2["band"].apply(lambda b: b[1] <= 20)) & (df2["fraction_used"] == 1.0) & (df2["target_mw"] >= 500)]
print(f"V1: {len(near1)} colocadas em tamanho cheio")
print(f"V2: {len(near2)} colocadas em tamanho cheio")

print("\n=== Distribuicao por banda -- % entregue no v2 ===")
df2["band_str"] = df2["band"].apply(lambda b: f"{b[0]}-{b[1]}km")
summary = df2.groupby("band_str").apply(lambda g: pd.Series({
    "n_attempts": len(g), "requested_mw": g["target_mw"].sum(), "delivered_mw": g["delivered_mw"].sum(),
    "pct": g["delivered_mw"].sum() / g["target_mw"].sum() * 100 if g["target_mw"].sum() > 0 else 0,
}))
print(summary)

df1.to_csv("bench_v1_final_log.csv", index=False)
df2.to_csv("bench_v2_final_log.csv", index=False)
print("\nSaved bench_v1_final_log.csv, bench_v2_final_log.csv")
