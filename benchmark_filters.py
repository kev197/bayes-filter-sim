import numpy as np
import matplotlib.pyplot as plt
from main import run_simulation  # make sure main.py has run_simulation()

# === Config ===
N_RUNS = 100
SEED_BASE = 42  # so results are reproducible

ekf_rmse_list = []
pf_rmse_list = []
agf_rmse_list = []

print(f"Running {N_RUNS} simulations...")

for i in range(N_RUNS):
    seed = SEED_BASE + i
    rmse_ekf, rmse_pf, rmse_agf, _ = run_simulation(False, 300, seed, )

    ekf_rmse_list.append(rmse_ekf)
    pf_rmse_list.append(rmse_pf)
    agf_rmse_list.append(rmse_agf)

    print(f"[{i+1}/{N_RUNS}] EKF: {rmse_ekf:.2f}  PF: {rmse_pf:.2f}  AGF: {rmse_agf:.2f}")

# === Summary ===
def summary_stats(name, data):
    print(f"\n{name} RMSE:")
    print(f"  Mean     : {np.mean(data):.3f}")
    print(f"  Std Dev  : {np.std(data):.3f}")
    print(f"  Min      : {np.min(data):.3f}")
    print(f"  Max      : {np.max(data):.3f}")

summary_stats("EKF", ekf_rmse_list)
summary_stats("PF", pf_rmse_list)
summary_stats("AGF", agf_rmse_list)

# === Visualization ===
plt.boxplot([ekf_rmse_list, pf_rmse_list, agf_rmse_list], labels=["EKF", "PF", "AGF"])
plt.ylabel("RMSE")
plt.title(f"Filter RMSE Comparison over {N_RUNS} Runs")
plt.grid(True)
plt.tight_layout()
plt.show()
