# -*- coding: utf-8 -*-
"""
BinaryFrontierEnv with:
  • single-world draw at reset() for correct joint outcomes
  • non-adaptive batch step: step_batch(action_indices)
  • same frontier/contact-tracing mechanics as the original code

Compatible with an AbstractJointProbabilityClass that implements:
  - compute_conditional_probability(query_dict, evidence_dict)
  - (optional) sample_full_unconditional()
  - (optional) sample_world_given(evidence_dict)
"""

from __future__ import annotations

import copy
from typing import Optional, Iterable, List, Set, Tuple

import networkx as nx
import numpy as np

# Expect a concrete subclass like LogJunctionTree / RealizationDistribution
from core.abstract_joint_probability_class import AbstractJointProbabilityClass


class BinaryFrontierEnv:
    """
    Graph-based binary testing environment with frontier constraints.
    Nodes are named X0, X1, ..., X{n-1}. Each test reveals X_i ∈ {0,1}.
    A 'frontier' rule restricts which nodes can be tested next.

    This version:
      - Draws a full hidden world X ~ P once per episode (at reset)
      - step(i) reveals one node from that fixed world
      - step_batch(S) reveals a non-adaptive set S from that same world
    """

    def __init__(
        self,
        G: nx.Graph,
        P: AbstractJointProbabilityClass,
        discount_factor: float,
        cc_dict: Optional[dict] = None,
        cc_root: Optional[List[int]] = None,
        rng_seed: int = 314159,
        # Fallback Gibbs sampler settings (only used if P lacks samplers)
        gibbs_sweeps: int = 1,
        gibbs_seed: Optional[int] = None,
    ) -> None:
        assert 0 < discount_factor < 1, "discount_factor must be in (0,1)"
        self.P = copy.deepcopy(P)
        self.n = G.number_of_nodes()
        assert self.n == P.n, "Graph node count must match P.n"
        self.discount_factor = discount_factor

        self.rng = np.random.default_rng(rng_seed)
        self.gibbs_sweeps = gibbs_sweeps
        self.gibbs_rng = np.random.default_rng(
            gibbs_seed if gibbs_seed is not None else (rng_seed + 12345)
        )

        # Relabel nodes to X0..X{n-1} for consistency with P keys
        self.G = nx.relabel_nodes(G, {i: f"X{i}" for i in range(self.n)})

        # Episode state
        self.tests_done: int = 0
        self.status: np.ndarray = np.array([-1] * self.n, dtype=int)  # -1=unknown, 0/1=observed
        self.world_X: Optional[np.ndarray] = None  # hidden labels for this episode
        self.round_budget: Optional[int] = None    # optional per-round budget tracker

        # Preprocess connected components for frontier roots
        if cc_dict is not None and cc_root is not None:
            self.cc_dict = cc_dict
            self.cc_root = cc_root
        else:
            self.cc_dict = dict()
            self.cc_root = []
            for cc_nodes in nx.connected_components(self.G):
                self.cc_dict[frozenset(cc_nodes)] = len(self.cc_dict)
                indices = [int(v[1:]) for v in cc_nodes]
                # Use current (empty) observations to pick the root with highest marginal
                marginal_prob1s = [(self.get_marginal_prob1(idx), idx) for idx in indices]
                marginal_prob1s.sort(reverse=True)
                self.cc_root.append(marginal_prob1s[0][1])

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def reset(self, round_budget: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Start a fresh episode:
          - draw a full hidden world X ~ P
          - clear observation status
          - compute frontier mask

        Returns:
          status: (n,) array with -1/0/1
          valid_mask: (n,) binary mask of frontier-valid nodes
        """
        self.tests_done = 0
        self.status = np.full(self.n, -1, dtype=int)
        self.world_X = self._sample_full_world()  # <-- key: draw once per episode
        self.round_budget = round_budget

        valid = self.get_frontier_actions(self.status)
        mask = np.array([1 if i in valid else 0 for i in range(self.n)], dtype=int)
        return self.status.copy(), mask

    def get_status_and_factors(self) -> Tuple[np.ndarray, dict, dict]:
        """(Kept for compatibility with DQN policy code.)"""
        return self.status.copy(), self.P.unary_factors.copy(), self.P.pairwise_factors.copy()

    def step(self, action_idx: int) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Reveal a SINGLE node from the fixed world_X.
        Frontier is enforced pre-action.
        """
        assert isinstance(action_idx, int)
        frontier = self.get_frontier_actions(self.status)
        assert action_idx in frontier, "Action must lie in the current frontier"
        assert self.world_X is not None, "Call reset() before step()"

        # Reveal from fixed world (preserves joint correlations automatically)
        self.status[action_idx] = int(self.world_X[action_idx])

        reward = float(self.status[action_idx])
        self.tests_done += 1
        if self.round_budget is not None:
            self.round_budget -= 1

        valid = self.get_frontier_actions(self.status)
        mask = np.array([1 if i in valid else 0 for i in range(self.n)], dtype=int)
        done = (self.tests_done == self.n)
        return self.status.copy(), mask, reward, done

    def step_batch(self, action_indices: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Reveal a NON-ADAPTIVE batch of nodes from the fixed world_X.
        All actions must be valid w.r.t. the PRE-batch frontier.
        """
        A = list(action_indices)
        
        frontier = self.get_frontier_actions(self.status)
        for i in A:
            assert i in frontier, "Batch must be chosen from the current frontier"
        assert self.world_X is not None, "Call reset() before step_batch()"

        reward = 0.0
        for i in A:
            self.status[i] = int(self.world_X[i])
            reward += float(self.status[i])

        self.tests_done += len(A)
        if self.round_budget is not None:
            self.round_budget -= len(A)

        valid = self.get_frontier_actions(self.status)
        mask = np.array([1 if i in valid else 0 for i in range(self.n)], dtype=int)
        done = (self.tests_done == self.n)
        return self.status.copy(), mask, reward, done

    # ---------------------------------------------------------------------
    # Probabilities & frontier
    # ---------------------------------------------------------------------

    def get_marginal_prob1(self, index: int, observed_status: Optional[np.ndarray] = None) -> float:
        """
        Return P(X_index=1 | current observations) using P.compute_conditional_probability.
        Short-circuits if already observed.
        """
        status = self.status if observed_status is None else observed_status
        if status[index] == 1:
            return 1.0
        if status[index] == 0:
            return 0.0
        query = {f"X{index}": 1}
        evidence = {f"X{i}": int(status[i]) for i in range(self.n) if status[i] != -1}
        return float(self.compute_conditional_probability(query, evidence))

    def compute_conditional_probability(self, query_dict: dict, evidence_dict: dict) -> float:
        """Thin wrapper to P.compute_conditional_probability."""
        assert len(set(query_dict.keys()).intersection(evidence_dict.keys())) == 0
        return float(self.P.compute_conditional_probability(query_dict, evidence_dict))

    def get_frontier_actions(self, status: np.ndarray) -> Set[int]:
        """
        Frontier rule (same as original):
          • In each connected component (CC), if no node has been tested yet,
            only the precomputed 'root' (argmax marginal) is allowed.
          • Otherwise, any untested node that has ≥1 tested neighbor is allowed.
        """
        assert len(status) == self.n
        tested = {f"X{i}" for i in range(self.n) if status[i] != -1}
        frontier: Set[int] = set()

        for cc_nodes in nx.connected_components(self.G):
            if len(cc_nodes.intersection(tested)) == 0:
                # No tests yet in this CC → allow its root only
                argmax_in_cc = self.cc_root[self.cc_dict[frozenset(cc_nodes)]]
                frontier.add(argmax_in_cc)
            else:
                # Add neighbors of already-tested nodes (but only if still untested)
                for v in cc_nodes:
                    if v not in tested:
                        if len(set(self.G.neighbors(v)).intersection(tested)) > 0:
                            frontier.add(int(v[1:]))

        return frontier

    # ---------------------------------------------------------------------
    # World sampling
    # ---------------------------------------------------------------------

    def _sample_full_world(self) -> np.ndarray:
        """
        Draw a COMPLETE assignment X ∈ {0,1}^n from P (no evidence).
        Priority:
          1) P.sample_world_given(evidence_dict={})
          2) P.sample_full_unconditional()
          3) Fallback Gibbs using compute_conditional_probability
        """
        # Prefer conditional sampler for consistency with the same inference backend
        if hasattr(self.P, "sample_world_given"):
            X = self.P.sample_world_given(evidence_dict={})
            return np.asarray(X, dtype=int)

        if hasattr(self.P, "sample_full_unconditional"):
            X = self.P.sample_full_unconditional()
            return np.asarray(X, dtype=int)

        # Fallback: simple single-sweep Gibbs from the prior (no evidence)
        return self._gibbs_sample_full_world()

    def _gibbs_sample_full_world(self) -> np.ndarray:
        """
        Very small Gibbs sampler as a last resort.
        Requires ONLY compute_conditional_probability for single-variable conditionals.
        Not meant for heavy use—prefer model-provided samplers when available.
        """
        # Initialize with iid Bernoulli(0.5)
        X = self.gibbs_rng.integers(0, 2, size=self.n, dtype=int)

        for _ in range(max(1, self.gibbs_sweeps)):
            order = self.gibbs_rng.permutation(self.n)
            for k in order:
                # Evidence: all variables except Xk (treat current X as evidence)
                evidence = {f"X{i}": int(X[i]) for i in range(self.n) if i != k}
                p1 = float(self.P.compute_conditional_probability({f"X{k}": 1}, evidence))
                X[k] = 1 if self.gibbs_rng.random() < p1 else 0

        return X

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def set_world(self, X: Iterable[int]) -> None:
        """
        Manually set the hidden world for debugging / deterministic tests.
        """
        X = np.asarray(list(X), dtype=int)
        assert X.shape == (self.n,)
        assert np.all((X == 0) | (X == 1))
        self.world_X = X.copy()

    def remaining_budget(self) -> Optional[int]:
        return self.round_budget

    def tested_count(self) -> int:
        return self.tests_done
