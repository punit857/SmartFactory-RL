import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import GRID_SIZE, DOCK_POS, DISPATCH_POS, AISLE_CELLS, NUM_STATIONS
from optimization.baselines import is_feasible
from visualization.factory_plot import plot_layout
from visualization.training_plot import plot_training

SEEDS = [0, 1, 2, 3, 4]          # reduce to [0, 1, 2] if you are short on time
METRICS_DIR = "results/metrics"

def seed_path(seed):
    return f"{METRICS_DIR}/dqn_seed{seed}.json"

def run_missing_seeds():
    """Train any seed that has no saved result yet (safe to re-run if interrupted)."""
    missing = [s for s in SEEDS if not os.path.exists(seed_path(s))]
    if not missing:
        return
    import torch
    from train import train
    for seed in missing:
        print(f"=== training seed {seed} ===")
        res = train(seed=seed)
        torch.save(res["final_state"], f"models/dqn_seed{seed}.pt")
        out = {k: res[k] for k in ["final_cost", "best_cost", "runtime", "history"]}
        out["final_placement"] = {str(k): list(v) for k, v in res["final_placement"].items()}
        out["best_placement"] = {str(k): list(v) for k, v in res["best_placement"].items()}
        with open(seed_path(seed), "w") as f:
            json.dump(out, f)

def load_runs():
    runs = []
    for s in SEEDS:
        r = json.load(open(seed_path(s)))
        r["seed"] = s
        r["final_placement"] = {int(k): tuple(v) for k, v in r["final_placement"].items()}
        r["best_placement"] = {int(k): tuple(v) for k, v in r["best_placement"].items()}
        runs.append(r)
    return runs

def check_valid(p):
    """Independent re-check that a layout obeys every constraint."""
    cells = list(p.values())
    reserved = set(AISLE_CELLS) | {DOCK_POS, DISPATCH_POS}
    in_grid = all(0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE for r, c in cells)
    return (len(p) == NUM_STATIONS and in_grid and len(set(cells)) == len(cells)
            and not (set(cells) & reserved) and is_feasible(p))

def main():
    run_missing_seeds()
    base = json.load(open(f"{METRICS_DIR}/baselines.json"))
    rand_mean, rand_std = base["random_mean"], base["random_std"]
    greedy, sa = base["greedy"], base["simulated_annealing"]
    runs = load_runs()

    final = np.array([r["final_cost"] for r in runs])
    best = np.array([r["best_cost"] for r in runs])
    valid_rate = 100 * np.mean([check_valid(r["final_placement"]) for r in runs])
    mean_reward = np.mean([np.mean(r["history"]["episode_reward"][-500:]) for r in runs])
    runtime = np.mean([r["runtime"] for r in runs])
    imp = lambda c: 100 * (rand_mean - c) / rand_mean

    print(f"\n{'Method':<34}{'Cost':>16}{'Improvement vs random':>24}")
    print(f"{'Random (mean of 1000)':<34}{rand_mean:>9.0f} +/- {rand_std:<4.0f}{'-':>22}")
    print(f"{'Greedy heuristic':<34}{greedy:>16.0f}{imp(greedy):>23.1f}%")
    print(f"{'DQN final policy (mean of seeds)':<34}{final.mean():>9.0f} +/- {final.std():<4.0f}{imp(final.mean()):>22.1f}%")
    print(f"{'DQN final policy (best seed)':<34}{final.min():>16.0f}{imp(final.min()):>23.1f}%")
    print(f"{'Simulated annealing (reference)':<34}{sa:>16.0f}{imp(sa):>23.1f}%")
    print(f"\nPer-seed final policy cost: {[round(float(c)) for c in final]}")
    print(f"Per-seed best-seen cost:    {[round(float(c)) for c in best]}")
    print(f"DQN gap to SA (mean):       {100 * (final.mean() - sa) / sa:+.1f}%")
    print(f"DQN vs greedy (mean):       {100 * (greedy - final.mean()) / greedy:+.1f}% lower cost")
    print(f"Valid placement rate:       {valid_rate:.0f}%")
    print(f"Mean training reward (last 500 episodes): {mean_reward:.2f}")
    print(f"Mean training runtime per seed: {runtime:.0f} s")

    # --- figures ---
    os.makedirs("results/plots", exist_ok=True)
    plot_training(runs, greedy, sa, "results/plots/training_curves.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Random\n(mean)", "Greedy", "DQN\n(mean of seeds)", "Simulated\nannealing"]
    vals = [rand_mean, greedy, final.mean(), sa]
    errs = [rand_std, 0, final.std(), 0]
    bars = ax.bar(labels, vals, yerr=errs, capsize=5,
                  color=["#999999", "#f58518", "#4c78a8", "#54a24b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 80, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_ylabel("Total weighted distance (lower is better)")
    ax.set_title("Layout cost by method")
    fig.tight_layout(); fig.savefig("results/plots/comparison.png", dpi=150); plt.close(fig)

    bi = int(np.argmin(final))
    plot_layout(runs[bi]["final_placement"], runs[bi]["final_cost"],
                f"DQN agent (seed {runs[bi]['seed']})", "results/layouts/dqn_best.png")

    summary = {"random_mean": rand_mean, "random_std": rand_std, "greedy": greedy,
               "simulated_annealing": sa, "dqn_final_costs": [float(c) for c in final],
               "dqn_best_seen_costs": [float(c) for c in best],
               "dqn_mean": float(final.mean()), "dqn_std": float(final.std()),
               "valid_placement_rate_pct": float(valid_rate),
               "mean_training_runtime_s": float(runtime)}
    json.dump(summary, open(f"{METRICS_DIR}/summary.json", "w"), indent=2)
    print("\nSaved: results/plots/training_curves.png, comparison.png, results/layouts/dqn_best.png")

if __name__ == "__main__":
    main()