import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d, m = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32),
                torch.tensor(np.array(m), dtype=torch.bool))

    def __len__(self):
        return len(self.buffer)