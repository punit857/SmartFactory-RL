import numpy as np
from config import (GRID_SIZE, DOCK_POS, DISPATCH_POS, AISLE_CELLS,
                    NUM_STATIONS, REWARD_SCALE)
from optimization.cost_calculator import incremental_cost, total_cost, violates_separation

class FactoryEnv:
    def __init__(self):
        self.n_actions = GRID_SIZE * GRID_SIZE                      # 64 cells
        self.state_dim = 2 * NUM_STATIONS + NUM_STATIONS + self.n_actions   # 14 + 7 + 64 = 85
        self.reset()

    def reset(self):
        self.placement = {}                                         # station index -> (row, col)
        self.k = 0                                                  # next station to place
        self.occupied = set(AISLE_CELLS) | {DOCK_POS, DISPATCH_POS}
        return self.get_state()

    def get_state(self):
        pos = np.full(2 * NUM_STATIONS, -1.0, dtype=np.float32)     # -1 = not placed yet
        for i, (r, c) in self.placement.items():
            pos[2 * i] = r / (GRID_SIZE - 1)
            pos[2 * i + 1] = c / (GRID_SIZE - 1)
        nxt = np.zeros(NUM_STATIONS, dtype=np.float32)              # one-hot: which station is next
        if self.k < NUM_STATIONS:
            nxt[self.k] = 1.0
        occ = np.zeros(self.n_actions, dtype=np.float32)            # 1 = cell unavailable
        for r, c in self.occupied:
            occ[r * GRID_SIZE + c] = 1.0
        return np.concatenate([pos, nxt, occ])

    def valid_mask(self):
        """True for cells where the next station may legally be placed."""
        mask = np.zeros(self.n_actions, dtype=bool)
        if self.k >= NUM_STATIONS:
            return mask
        for a in range(self.n_actions):
            cell = divmod(a, GRID_SIZE)
            if cell in self.occupied:
                continue
            if violates_separation(self.k, cell, self.placement):
                continue
            mask[a] = True
        return mask

    def step(self, action):
        assert self.valid_mask()[action], "Invalid action chosen"
        cell = divmod(action, GRID_SIZE)
        delta_c = incremental_cost(self.k, cell, self.placement)
        self.placement[self.k] = cell
        self.occupied.add(cell)
        self.k += 1
        done = self.k == NUM_STATIONS
        info = {}
        if done:
            info = {"total_cost": total_cost(self.placement), "placement": dict(self.placement)}
        return self.get_state(), -delta_c / REWARD_SCALE, done, info