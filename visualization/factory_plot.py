import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import GRID_SIZE, DOCK_POS, DISPATCH_POS, AISLE_CELLS
from optimization.flow_matrix import FLOWS, IDX

SHORT = ["Stamp", "Weld", "Paint", "Engine", "Interior", "Wheels", "QC"]
COLORS = ["#4c78a8", "#e45756", "#f58518", "#54a24b", "#b279a2", "#72b7b2", "#eeca3b"]

def plot_layout(placement, cost, title, path):
    pos = {i: c for i, c in placement.items()}
    pos[IDX["Receiving Dock"]] = DOCK_POS
    pos[IDX["Dispatch"]] = DISPATCH_POS

    fig, ax = plt.subplots(figsize=(8, 8))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ax.add_patch(Rectangle((c, GRID_SIZE - 1 - r), 1, 1,
                                   facecolor="white", edgecolor="#cccccc"))
    for r, c in AISLE_CELLS:
        ax.add_patch(Rectangle((c, GRID_SIZE - 1 - r), 1, 1, facecolor="#e8e8e8", edgecolor="#cccccc"))
        ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, "aisle", ha="center", va="center",
                fontsize=7, color="#999999")
    for (r, c), label in [(DOCK_POS, "Receiving\nDock"), (DISPATCH_POS, "Dispatch")]:
        ax.add_patch(Rectangle((c, GRID_SIZE - 1 - r), 1, 1, facecolor="#333333", edgecolor="black"))
        ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, label, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    for i, (r, c) in placement.items():
        ax.add_patch(Rectangle((c, GRID_SIZE - 1 - r), 1, 1, facecolor=COLORS[i], edgecolor="black"))
        ax.text(c + 0.5, GRID_SIZE - 1 - r + 0.5, f"{i + 1}\n{SHORT[i]}", ha="center",
                va="center", fontsize=8, color="white", fontweight="bold")

    for a, b, f in FLOWS:                             # flow arrows, thicker = more flow
        (r1, c1), (r2, c2) = pos[IDX[a]], pos[IDX[b]]
        ax.annotate("", xy=(c2 + 0.5, GRID_SIZE - 1 - r2 + 0.5),
                    xytext=(c1 + 0.5, GRID_SIZE - 1 - r1 + 0.5),
                    arrowprops=dict(arrowstyle="->", lw=f / 40, color="black", alpha=0.35,
                                    shrinkA=14, shrinkB=14))
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nTotal weighted distance = {cost:.0f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)