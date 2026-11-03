import copy, json, random, time
import numpy as np
import torch
import torch.nn.functional as F

import config as C
from environment.factory_env import FactoryEnv
from agent.dqn import QNetwork
from agent.replay_buffer import ReplayBuffer
from agent.policy import select_action

def run_greedy_episode(q_net):
    """Evaluate the learned policy with no exploration (epsilon = 0)."""
    env = FactoryEnv()
    env.reset()
    done = False
    while not done:
        state, mask = env.get_state(), env.valid_mask()
        a = select_action(q_net, state, mask, epsilon=0.0)
        _, _, done, info = env.step(a)
    return info["total_cost"], info["placement"]

def train(seed=0, episodes=C.EPISODES, verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = FactoryEnv()
    q_net = QNetwork(env.state_dim, env.n_actions, C.HIDDEN)
    target_net = copy.deepcopy(q_net)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=C.LEARNING_RATE)
    buffer = ReplayBuffer(C.BUFFER_SIZE)

    history = {"episode_cost": [], "episode_reward": [], "loss": [], "eval_cost": [], "eval_episode": []}
    best_cost, best_placement, best_state = float("inf"), None, None
    steps, t0 = 0, time.time()

    for ep in range(episodes):
        eps = max(C.EPS_END, C.EPS_START - (C.EPS_START - C.EPS_END) * ep / (C.EPS_DECAY_FRACTION * episodes))
        state = env.reset()
        mask = env.valid_mask()
        done, ep_reward, ep_losses = False, 0.0, []
        while not done:
            action = select_action(q_net, state, mask, eps)
            next_state, reward, done, info = env.step(action)
            next_mask = env.valid_mask()
            buffer.push(state, action, reward, next_state, float(done), next_mask)
            state, mask, ep_reward = next_state, next_mask, ep_reward + reward
            steps += 1

            if len(buffer) >= C.BATCH_SIZE:
                s, a, r, s2, d, m2 = buffer.sample(C.BATCH_SIZE)
                q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(s2)
                    q_next[~m2] = -float("inf")                  # mask invalid next actions
                    max_next = q_next.max(dim=1).values
                    max_next[d.bool()] = 0.0                     # terminal: y = r
                    y = r + C.GAMMA * max_next
                loss = F.smooth_l1_loss(q, y)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                ep_losses.append(loss.item())

            if steps % C.TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())

        cost = info["total_cost"]
        history["episode_cost"].append(cost)
        history["episode_reward"].append(ep_reward)
        history["loss"].append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        if cost < best_cost:                                      # best FEASIBLE layout seen in training
            best_cost, best_placement = cost, info["placement"]

        if (ep + 1) % C.EVAL_EVERY == 0:
            ecost, eplace = run_greedy_episode(q_net)
            history["eval_cost"].append(ecost); history["eval_episode"].append(ep + 1)
            if ecost <= best_cost:
                best_cost, best_placement = ecost, eplace
                best_state = copy.deepcopy(q_net.state_dict())
            if verbose and (ep + 1) % (C.EVAL_EVERY * 5) == 0:
                recent = np.mean(history["episode_cost"][-C.EVAL_EVERY:])
                print(f"ep {ep+1:5d} | eps {eps:.2f} | train cost (last {C.EVAL_EVERY}) {recent:7.0f} "
                      f"| greedy-policy cost {ecost:6.0f} | best {best_cost:6.0f} | {time.time()-t0:5.0f}s")

    final_cost, final_place = run_greedy_episode(q_net)
    return {"history": history, "final_cost": final_cost, "final_placement": final_place,
            "best_cost": best_cost, "best_placement": best_placement,
            "best_state": best_state, "final_state": q_net.state_dict(),
            "runtime": time.time() - t0}

if __name__ == "__main__":
    res = train(seed=0)
    print(f"\nFinal greedy-policy cost: {res['final_cost']:.0f}")
    print(f"Best layout found:        {res['best_cost']:.0f}")
    torch.save(res["final_state"], "models/dqn_seed0.pt")