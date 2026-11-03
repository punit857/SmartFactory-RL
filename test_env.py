import random
from environment.factory_env import FactoryEnv
from config import REWARD_SCALE
from optimization.cost_calculator import manhattan

env = FactoryEnv()
costs = []
for _ in range(1000):
    env.reset()
    done, ret = False, 0.0
    while not done:
        mask = env.valid_mask()
        a = random.choice([i for i in range(env.n_actions) if mask[i]])
        _, r, done, info = env.step(a)
        ret += r
    assert abs(ret * REWARD_SCALE + info["total_cost"]) < 1e-6   # rewards add up to -cost
    p = info["placement"]
    assert manhattan(p[1], p[2]) >= 3                            # welding-painting separation holds
    costs.append(info["total_cost"])

print("State size:", env.state_dim, "| Actions:", env.n_actions)
print("All checks passed.")
print(f"Random layouts over 1000 runs: mean cost = {sum(costs)/len(costs):.0f}, "
      f"best = {min(costs):.0f}, worst = {max(costs):.0f}")