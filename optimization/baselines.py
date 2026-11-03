import math
import random
from config import (GRID_SIZE, DOCK_POS, DISPATCH_POS, AISLE_CELLS,
                    NUM_STATIONS, STATIONS, MIN_SEPARATION)
from environment.factory_env import FactoryEnv
from optimization.cost_calculator import total_cost, manhattan

# ---------- Method 1: random placement ----------
def random_layout(rng=random):
    env = FactoryEnv()
    done = False
    while not done:
        mask = env.valid_mask()
        a = rng.choice([i for i in range(env.n_actions) if mask[i]])
        _, _, done, info = env.step(a)
    return info["placement"], info["total_cost"]

# ---------- Method 2: greedy heuristic ----------
def greedy_layout():
    """At each step, take the valid cell with the lowest incremental cost.
    Ties are broken by the lowest cell index, so the result is deterministic."""
    from optimization.cost_calculator import incremental_cost
    env = FactoryEnv()
    done = False
    while not done:
        mask = env.valid_mask()
        best_a, best_c = None, float("inf")
        for a in range(env.n_actions):
            if not mask[a]:
                continue
            c = incremental_cost(env.k, divmod(a, GRID_SIZE), env.placement)
            if c < best_c:
                best_a, best_c = a, c
        _, _, done, info = env.step(best_a)
    return info["placement"], info["total_cost"]

# ---------- Method 3: simulated annealing (reference solver) ----------
FREE_CELLS = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
              if (r, c) not in set(AISLE_CELLS) | {DOCK_POS, DISPATCH_POS}]

def is_feasible(placement):
    for (a, b), d_min in MIN_SEPARATION.items():
        i, j = STATIONS.index(a), STATIONS.index(b)
        if manhattan(placement[i], placement[j]) < d_min:
            return False
    return True

def simulated_annealing(seed=0, iters=20000, t_start=500.0, t_end=0.5):
    rng = random.Random(seed)
    while True:                                       # random feasible start
        cells = rng.sample(FREE_CELLS, NUM_STATIONS)
        cur = {i: cells[i] for i in range(NUM_STATIONS)}
        if is_feasible(cur):
            break
    cur_cost = total_cost(cur)
    best, best_cost = dict(cur), cur_cost
    for it in range(iters):
        t = t_start * (t_end / t_start) ** (it / iters)
        new = dict(cur)
        if rng.random() < 0.5:                        # move one station to a free cell
            k = rng.randrange(NUM_STATIONS)
            used = set(new.values())
            free = [c for c in FREE_CELLS if c not in used]
            new[k] = rng.choice(free)
        else:                                         # swap two stations
            a, b = rng.sample(range(NUM_STATIONS), 2)
            new[a], new[b] = new[b], new[a]
        if not is_feasible(new):
            continue
        new_cost = total_cost(new)
        delta = new_cost - cur_cost
        if delta < 0 or rng.random() < math.exp(-delta / t):
            cur, cur_cost = new, new_cost
            if cur_cost < best_cost:
                best, best_cost = dict(cur), cur_cost
    return best, best_cost

def best_of_sa(runs=10):
    results = [simulated_annealing(seed=s) for s in range(runs)]
    return min(results, key=lambda x: x[1])