import numpy as np
from config import STATIONS

# Station indices 0..6, then 7 = Receiving Dock, 8 = Dispatch
NODES = STATIONS + ["Receiving Dock", "Dispatch"]
IDX = {name: i for i, name in enumerate(NODES)}

# (from, to, loads per shift) - illustrative simulation values
FLOWS = [
    ("Stamping", "Body Welding", 100),
    ("Body Welding", "Painting", 100),
    ("Painting", "Engine Installation", 100),
    ("Engine Installation", "Interior Assembly", 100),
    ("Interior Assembly", "Wheel Installation", 100),
    ("Wheel Installation", "Quality Inspection", 100),
    ("Quality Inspection", "Dispatch", 95),
    ("Receiving Dock", "Stamping", 100),
    ("Receiving Dock", "Engine Installation", 40),
    ("Receiving Dock", "Interior Assembly", 40),
    ("Receiving Dock", "Wheel Installation", 40),
    ("Quality Inspection", "Painting", 8),   # rework loop
]

def build_flow_matrix():
    n = len(NODES)
    F = np.zeros((n, n))
    for a, b, f in FLOWS:
        F[IDX[a], IDX[b]] += f
        F[IDX[b], IDX[a]] += f   # distance is symmetric, so we use total flow between a pair
    return F

FLOW = build_flow_matrix()