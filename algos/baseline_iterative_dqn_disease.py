"""
Iterative DQN baseline for the disease frontier environment.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from environment.frontier_batch_env import BinaryFrontierEnvBatch
from algos.dqn_estimator_disease import GraphEnvAdapter, PolicyQNet


@torch.no_grad()
def _estimate_q_reward(
    data,
    net: PolicyQNet,
    action: np.ndarray,
) -> float:
    """
    Evaluate Q(s, a) and convert back to reward (negate).
    """
    a_tensor = torch.tensor(action, dtype=torch.float32, device=data.x.device).unsqueeze(0)
    q_cost = net(data, a_tensor).squeeze().item()
    return -q_cost


def baseline_iterative_dqn_disease(
    env: BinaryFrontierEnvBatch,
    net: PolicyQNet,
    init_states: List[Optional[np.ndarray]],
    horizon: Optional[int] = None,
) -> np.ndarray:
    """
    Iteratively grow an action set by adding the frontier node that maximizes
    the Q-network's value estimate.
    """
    if horizon is None:
        horizon = env.n

    adapter = GraphEnvAdapter(env)
    n_episodes = len(init_states)
    rewards = np.zeros(horizon * n_episodes, dtype=float)

    for ep in range(n_episodes):
        status, _ = env.reset()
        if init_states[ep] is not None:
            env.status = init_states[ep].copy()
            status = env.status.copy()

        for t in range(horizon):
            data = adapter.build_graph(status)
            action = np.zeros(env.n, dtype=int)
            frontier = list(np.flatnonzero(env.frontier_mask_from_status(status)))
            if frontier:
                remaining_budget = env.budget if env.budget is not None else len(frontier)
                remaining_budget = min(remaining_budget, len(frontier))
                while remaining_budget > 0 and frontier:
                    best_idx = None
                    best_val = -np.inf
                    for idx in frontier:
                        cand = action.copy()
                        cand[idx] = 1
                        val = _estimate_q_reward(data, net, cand)
                        if val > best_val:
                            best_val = val
                            best_idx = idx
                    if best_idx is None:
                        break
                    action[best_idx] = 1
                    frontier.remove(best_idx)
                    remaining_budget -= 1

            next_status, _, reward, done = env.step(action)
            rewards[ep * horizon + t] = reward
            status = next_status
            if done:
                break

    return rewards

