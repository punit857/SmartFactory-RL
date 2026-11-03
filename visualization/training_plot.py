import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def moving_avg(x, w):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x

def plot_training(runs, greedy_cost, sa_cost, path, window=100):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]                                     # (a) training cost
    curves = []
    for r in runs:
        ma = moving_avg(r["history"]["episode_cost"], window)
        curves.append(ma)
        ax.plot(np.arange(window, window + len(ma)), ma, alpha=0.35, lw=1)
    n = min(len(c) for c in curves)
    mean_curve = np.mean([c[:n] for c in curves], axis=0)
    ax.plot(np.arange(window, window + n), mean_curve, color="black", lw=2, label="mean of seeds")
    ax.axhline(greedy_cost, ls="--", color="tab:orange", label=f"Greedy ({greedy_cost:.0f})")
    ax.axhline(sa_cost, ls="--", color="tab:green", label=f"Simulated annealing ({sa_cost:.0f})")
    ax.set_title(f"Training cost (moving average, {window} episodes)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total weighted distance"); ax.legend()

    ax = axes[1]                                     # (b) greedy-policy cost
    for r in runs:
        ax.plot(r["history"]["eval_episode"], r["history"]["eval_cost"], alpha=0.6, lw=1)
    ax.axhline(greedy_cost, ls="--", color="tab:orange")
    ax.axhline(sa_cost, ls="--", color="tab:green")
    ax.set_title("Cost of the agent's own layout (no exploration)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total weighted distance")

    ax = axes[2]                                     # (c) loss
    for r in runs:
        ma = moving_avg(r["history"]["loss"], window)
        ax.plot(np.arange(window, window + len(ma)), ma, alpha=0.6, lw=1)
    ax.set_title(f"DQN loss (moving average, {window} episodes)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Smooth L1 loss")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)