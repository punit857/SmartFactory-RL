from config import STATIONS, DOCK_POS, DISPATCH_POS, MIN_SEPARATION
from optimization.flow_matrix import FLOW

DOCK_ID = len(STATIONS)
DISPATCH_ID = len(STATIONS) + 1

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def _positions(placement):
    pos = dict(placement)            # {station_index: (row, col)}
    pos[DOCK_ID] = DOCK_POS
    pos[DISPATCH_ID] = DISPATCH_POS
    return pos

def incremental_cost(k, cell, placement):
    """Delta C: cost of putting station k at `cell`, against already-placed stations + fixed nodes."""
    pos = _positions(placement)
    return sum(FLOW[k, j] * manhattan(cell, pj) for j, pj in pos.items() if j != k)

def total_cost(placement):
    """C = sum over pairs of flow * distance."""
    pos = _positions(placement)
    ids = sorted(pos)
    cost = 0.0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            cost += FLOW[i, j] * manhattan(pos[i], pos[j])
    return cost

def violates_separation(k, cell, placement):
    name = STATIONS[k]
    for (a, b), d_min in MIN_SEPARATION.items():
        if name == a:
            other = b
        elif name == b:
            other = a
        else:
            continue
        j = STATIONS.index(other)
        if j in placement and manhattan(cell, placement[j]) < d_min:
            return True
    return False