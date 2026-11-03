import torch.nn as nn

class QNetwork(nn.Module):
    """MLP: state (85) -> 128 -> 128 -> 64 Q-values, one per grid cell."""
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)