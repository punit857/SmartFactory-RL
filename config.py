GRID_SIZE = 8

# Fixed nodes as (row, col)
DOCK_POS = (3, 0)        # Receiving Dock, west edge
DISPATCH_POS = (3, 7)    # Dispatch, east edge
AISLE_CELLS = [(3, c) for c in range(1, 7)]   # reserved main aisle

# Stations the agent places, in production order
STATIONS = [
    "Stamping", "Body Welding", "Painting", "Engine Installation",
    "Interior Assembly", "Wheel Installation", "Quality Inspection",
]
NUM_STATIONS = len(STATIONS)

# Separation constraint: minimum Manhattan distance between two stations
MIN_SEPARATION = {("Body Welding", "Painting"): 3}

REWARD_SCALE = 100.0   # S in r = -deltaC / S

# ---------- DQN hyperparameters ----------
GAMMA = 1.0              # episode is short (7 steps) and we care about total cost
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50000
TARGET_UPDATE_EVERY = 200   # steps between target-network syncs
EPISODES = 6000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRACTION = 0.6    # epsilon falls linearly over the first 60% of episodes
EVAL_EVERY = 100            # greedy evaluation every N episodes
HIDDEN = 128