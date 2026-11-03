import json, random, time
import numpy as np
from optimization.baselines import random_layout, greedy_layout, best_of_sa
from visualization.factory_plot import plot_layout

rng = random.Random(0)
t0 = time.time()
rand = [random_layout(rng) for _ in range(1000)]
rand_costs = [c for _, c in rand]
rand_time = time.time() - t0
best_rand = min(rand, key=lambda x: x[1])

t0 = time.time(); g_place, g_cost = greedy_layout(); g_time = time.time() - t0
t0 = time.time(); sa_place, sa_cost = best_of_sa(10); sa_time = time.time() - t0

mean_rand = float(np.mean(rand_costs))
def imp(c): return 100 * (mean_rand - c) / mean_rand

print(f"{'Method':<22}{'Cost':>8}{'vs random mean':>18}{'Runtime(s)':>12}")
print(f"{'Random (mean of 1000)':<22}{mean_rand:>8.0f}{'-':>18}{rand_time:>12.2f}")
print(f"{'Random (best of 1000)':<22}{best_rand[1]:>8.0f}{imp(best_rand[1]):>17.1f}%{'-':>12}")
print(f"{'Greedy':<22}{g_cost:>8.0f}{imp(g_cost):>17.1f}%{g_time:>12.2f}")
print(f"{'Simulated annealing':<22}{sa_cost:>8.0f}{imp(sa_cost):>17.1f}%{sa_time:>12.2f}")

plot_layout(best_rand[0], best_rand[1], "Random placement (best of 1000)", "results/layouts/random_best.png")
plot_layout(g_place, g_cost, "Greedy heuristic", "results/layouts/greedy.png")
plot_layout(sa_place, sa_cost, "Simulated annealing (reference)", "results/layouts/simulated_annealing.png")

json.dump({"random_mean": mean_rand, "random_std": float(np.std(rand_costs)),
           "random_best": best_rand[1], "greedy": g_cost, "simulated_annealing": sa_cost,
           "greedy_layout": {str(k): v for k, v in g_place.items()},
           "sa_layout": {str(k): v for k, v in sa_place.items()}},
          open("results/metrics/baselines.json", "w"), indent=2)