"""
Iterative DQN baseline for the disease frontier environment.
"""

from __future__ import annotations
import sys

from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

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

    with tqdm(total=n_episodes, desc="iterative DQN      ", 
              file=sys.stdout, mininterval=1.0, ncols=100) as pbar:
        for ep in range(n_episodes):
            status, _ = env.reset()
            if init_states[ep] is not None:
                env.status = init_states[ep].copy()
                status = env.status.copy()

            for t in range(horizon):
                # Check if all nodes are already tested
                if env.tests_done >= env.n:
                    break
                
                data = adapter.build_graph(status)
                action = np.zeros(env.n, dtype=int)
                frontier = list(np.flatnonzero(env.frontier_mask_from_status(status)))
                
                # Handle case where frontier is empty but nodes remain untested
                if not frontier:
                    untested_nodes = np.flatnonzero(status == -1)
                    if len(untested_nodes) > 0:
                        # Test untested nodes directly (should be root nodes of disconnected components)
                        remaining_budget = env.budget if env.budget is not None else len(untested_nodes)
                        remaining_budget = min(remaining_budget, len(untested_nodes))
                        if remaining_budget > 0:
                            selected = untested_nodes[:remaining_budget]
                            action[selected] = 1
                            if ep == 0 and t < 3:  # Only print for first few steps to avoid spam
                                print(f"  [Iterative DQN Episode {ep+1}, Step {t+1}] WARNING: Frontier empty but {len(untested_nodes)} nodes untested. "
                                      f"Testing {len(selected)} untested nodes directly.")
                    else:
                        # All nodes tested
                        break
                else:
                    # Normal case: use Q-network to select best frontier nodes
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
                    # Verify all nodes are actually tested
                    if env.tests_done < env.n:
                        if ep == 0:  # Only print for first episode to avoid spam
                            print(f"  [Iterative DQN Episode {ep+1}] WARNING: done=True but tests_done={env.tests_done} < n={env.n}")
                    else:
                        if ep == 0:  # Only print for first episode to avoid spam
                            print(f"  [Iterative DQN Episode {ep+1}] Completed: all {env.n} nodes tested in {t+1} steps")
                    break
            
            # Final check: ensure all nodes were tested
            if env.tests_done < env.n:
                untested = env.n - env.tests_done
                if ep == 0:  # Only print for first episode to avoid spam
                    print(f"  [Iterative DQN Episode {ep+1}] WARNING: Episode ended with {untested} nodes untested (tests_done={env.tests_done}/{env.n})")
            
            pbar.update(1)
            pbar.set_postfix({"episode": ep + 1, "reward": f"{reward:.3f}"})

    return rewards

