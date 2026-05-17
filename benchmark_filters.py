import numpy as np
import matplotlib.pyplot as plt
from main import run_simulation

N_RUNS    = 100
SEED_BASE = 42

ekf_rmse_list  = []
pf_rmse_list   = []
agf_rmse_list  = []
asir_rmse_list = []

print(f"Running {N_RUNS} simulations...")

for i in range(N_RUNS):
    seed = SEED_BASE + i
    rmse_ekf, rmse_pf, rmse_agf, rmse_asir, _ = run_simulation(False, 300, seed)

    ekf_rmse_list.append(rmse_ekf)
    pf_rmse_list.append(rmse_pf)
    agf_rmse_list.append(rmse_agf)
    asir_rmse_list.append(rmse_asir)

    print(f"[{i+1:3d}/{N_RUNS}]  EKF: {rmse_ekf:.1f}  PF: {rmse_pf:.1f}  "
          f"AGF: {rmse_agf:.1f}  ASIR: {rmse_asir:.1f}")

def summary_stats(name, data):
    print(f"\n{name} RMSE over {N_RUNS} runs:")
    print(f"  Mean   : {np.mean(data):.2f}")
    print(f"  Std    : {np.std(data):.2f}")
    print(f"  Median : {np.median(data):.2f}")
    print(f"  Min    : {np.min(data):.2f}  Max: {np.max(data):.2f}")

print("\n" + "="*55)
summary_stats("EKF",  ekf_rmse_list)
summary_stats("PF",   pf_rmse_list)
summary_stats("AGF",  agf_rmse_list)
summary_stats("ASIR", asir_rmse_list)

plt.figure(figsize=(8, 5))
plt.boxplot(
    [ekf_rmse_list, pf_rmse_list, agf_rmse_list, asir_rmse_list],
    labels=["EKF", "PF (SIR)", "AGF", "ASIR"],
    patch_artist=True,
    boxprops=dict(facecolor="#2a2a3a", color="gray"),
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color="gray"),
    capprops=dict(color="gray"),
    flierprops=dict(marker="o", color="gray", alpha=0.5),
)
plt.ylabel("RMSE (px)")
plt.title(f"Filter RMSE comparison — {N_RUNS} runs")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
