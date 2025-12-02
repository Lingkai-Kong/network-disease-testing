"""
Baselines for the disease-testing frontier environment.

These mirror the classic RMAB baselines but operate directly on
`BinaryFrontierEnvBatch` instances.
"""

from __future__ import annotations

from typing import Callable, List, Optional
import sys

import numpy as np
from tqdm import tqdm

from environment.frontier_batch_env import BinaryFrontierEnvBatch


ActionFn = Callable[[BinaryFrontierEnvBatch, np.ndarray], np.ndarray]


def _run_baseline(
    env: BinaryFrontierEnvBatch,
    init_states: List[Optional[np.ndarray]],
    select_action_fn: ActionFn,
    horizon: Optional[int] = None,
    model_name: str = "baseline",
) -> np.ndarray:
    """
    Shared episode runner for all heuristics.
    """
    n_episodes = len(init_states)
    if horizon is None:
        horizon = env.n

    rewards = np.zeros(horizon * n_episodes, dtype=float)

    # Use tqdm with file=sys.stdout to ensure progress shows in log files
    # mininterval=1.0 ensures updates at least every second
    with tqdm(total=n_episodes, desc=f"{model_name:20s}", 
              file=sys.stdout, mininterval=1.0, ncols=100) as pbar:
        for ep in range(n_episodes):
            status, _ = env.reset()
            if init_states[ep] is not None:
                env.status = init_states[ep].copy()
                status = env.status.copy()

            for t in range(horizon):
                action = select_action_fn(env, status).astype(int)
                next_status, _, reward, done = env.step(action)
                rewards[ep * horizon + t] = reward
                status = next_status
                if done:
                    break
            
            pbar.update(1)
            pbar.set_postfix({"episode": ep + 1, "reward": f"{reward:.3f}"})

    return rewards


def _frontier_indices(env: BinaryFrontierEnvBatch, status: np.ndarray) -> np.ndarray:
    mask = env.frontier_mask_from_status(status)
    return np.flatnonzero(mask)


def _budget(env: BinaryFrontierEnvBatch, num_available: int) -> int:
    if env.budget is None:
        return num_available
    return min(env.budget, num_available)


def _marginal_prob(env: BinaryFrontierEnvBatch, idx: int, status: np.ndarray) -> float:
    return env.get_marginal_prob1(idx, observed_status=status)


def baseline_null_action_disease(
    env: BinaryFrontierEnvBatch,
    init_states: List[Optional[np.ndarray]],
    horizon: Optional[int] = None,
) -> np.ndarray:
    """
    Do nothing each step (baseline metric).
    """

    def select_action(_: BinaryFrontierEnvBatch, status: np.ndarray) -> np.ndarray:
        return np.zeros(len(status), dtype=int)

    return _run_baseline(env, init_states, select_action, horizon, model_name="null")


def baseline_random_disease(
    env: BinaryFrontierEnvBatch,
    init_states: List[Optional[np.ndarray]],
    horizon: Optional[int] = None,
) -> np.ndarray:
    """
    Random feasible frontier action at every step.
    """

    def select_action(env: BinaryFrontierEnvBatch, _: np.ndarray) -> np.ndarray:
        return env.random_feasible_action().astype(int)

    return _run_baseline(env, init_states, select_action, horizon, model_name="random")


def baseline_myopic_disease(
    env: BinaryFrontierEnvBatch,
    init_states: List[Optional[np.ndarray]],
    horizon: Optional[int] = None,
) -> np.ndarray:
    """
    Choose the frontier nodes with highest marginal infection probability.
    """

    def select_action(env: BinaryFrontierEnvBatch, status: np.ndarray) -> np.ndarray:
        action = np.zeros(len(status), dtype=int)
        frontier = _frontier_indices(env, status)
        if frontier.size == 0:
            return action

        scores = [
            (_marginal_prob(env, idx, status), idx)
            for idx in frontier
        ]
        scores.sort(reverse=True)
        for _, idx in scores[: _budget(env, frontier.size)]:
            action[idx] = 1
        return action

    return _run_baseline(env, init_states, select_action, horizon, model_name="myopic")


def baseline_greedy_iterative_myopic_disease(
    env: BinaryFrontierEnvBatch,
    init_states: List[Optional[np.ndarray]],
    horizon: Optional[int] = None,
) -> np.ndarray:
    """
    Iteratively add the frontier node that maximizes marginal gain.
    """

    def select_action(env: BinaryFrontierEnvBatch, status: np.ndarray) -> np.ndarray:
        action = np.zeros(len(status), dtype=int)
        frontier = list(_frontier_indices(env, status))
        if not frontier:
            return action

        remaining_budget = _budget(env, len(frontier))
        while remaining_budget > 0 and frontier:
            best_idx = None
            best_score = -np.inf
            for idx in frontier:
                score = _marginal_prob(env, idx, status)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            action[best_idx] = 1
            frontier.remove(best_idx)
            remaining_budget -= 1
        return action

    return _run_baseline(env, init_states, select_action, horizon, model_name="iter_myopic")


def baseline_sampling_disease(
    env: BinaryFrontierEnvBatch,
    init_states: List[Optional[np.ndarray]],
    horizon: Optional[int] = None,
    n_samples: int = 20,
) -> np.ndarray:
    """
    Sample several feasible actions and keep the one with highest sum of marginals.
    """

    def expected_reward(env: BinaryFrontierEnvBatch, action: np.ndarray, status: np.ndarray) -> float:
        return float(sum(_marginal_prob(env, idx, status) for idx in np.flatnonzero(action)))

    def select_action(env: BinaryFrontierEnvBatch, status: np.ndarray) -> np.ndarray:
        frontier = _frontier_indices(env, status)
        if frontier.size == 0:
            return np.zeros(len(status), dtype=int)

        best_score = -np.inf
        best_action = None
        for _ in range(max(1, n_samples)):
            candidate = env.random_feasible_action().astype(int)
            score = expected_reward(env, candidate, status)
            if score > best_score:
                best_score = score
                best_action = candidate

        if best_action is None:
            return np.zeros(len(status), dtype=int)
        return best_action.astype(int)

    return _run_baseline(env, init_states, select_action, horizon, model_name=f"sampling(k={n_samples})")

