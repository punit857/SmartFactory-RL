import random
import numpy as np
import torch

def select_action(q_net, state, mask, epsilon):
    """Epsilon-greedy over VALID cells only (action masking)."""
    valid = np.flatnonzero(mask)
    if random.random() < epsilon:
        return int(random.choice(valid))
    with torch.no_grad():
        q = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    q[~torch.tensor(mask)] = -float("inf")      # invalid cells can never win the argmax
    return int(torch.argmax(q).item())