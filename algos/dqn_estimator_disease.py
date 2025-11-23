# dqn_frontier_mip.py
"""
DQN + MILP solver for batch graph testing.

Structure mirrors the original RMAB DQN:
  - Q-network approximates Q(s,a)
  - MILP + Net2MIPPerScenario chooses argmax_a Q(s,a) each step
but:
  - The environment is graph-based (BinaryFrontierEnvBatch).
  - The state is encoded via a graph encoder g(s).
  - The MLP in the MILP sees [a, g(s)] as input.
"""

import os
import math
import random
import datetime
from collections import namedtuple
from typing import List

import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_mean_pool

from model2mip.net2mip import Net2MIPPerScenario
from approximator.batch_graph_approximator import BatchGraphApproximator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################################
# Replay buffer & memoizer
##############################################################

Transition = namedtuple("Transition", ("state", "action", "next_state", "cost"))


class ReplayBuffer:
    """Simple uniform replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.memory: List[Transition] = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)


class Memoizer:
    """Cache MILP solves (state -> best action) across episodes."""

    def __init__(self, refresh=5):
        self.refresh = refresh
        self.existing_solves = [{}]
        self.episode = 0
        self.num_checks = [0]
        self.num_successes = [0]

    def reset(self):
        self.existing_solves = [{}]
        self.episode = 0
        self.num_checks = [0]
        self.num_successes = [0]

    def add(self, key, value):
        self.existing_solves[0][key] = value

    def new_episode(self):
        tot_solves = np.array(self.num_checks).sum()
        tot_successes = np.array(self.num_successes).sum()
        # if tot_solves > 0:
        #     frac = tot_successes / tot_solves
        #     print(f"memoizer: ep {self.episode}, frac {frac:.2f}")

        del self.existing_solves[-1]
        self.existing_solves.insert(0, {})
        self.episode += 1

        del self.num_checks[-1]
        del self.num_successes[-1]
        self.num_checks.insert(0, 0)
        self.num_successes.insert(0, 0)

        return self.episode

    def check_key(self, key):
        self.num_checks[0] += 1
        for d in self.existing_solves:
            if key in d:
                self.num_successes[0] += 1
                return d[key]
        return None


##############################################################
# Graph adapter
##############################################################

class GraphEnvAdapter:
    """
    Convert (env, status) into a torch_geometric Data object.

    Nodes: X0, ..., X_{n-1}
    Node features: [unary_factor[0], unary_factor[1], status_i]
    Edge features: flattened pairwise factor table.
    """

    def __init__(self, env):
        self.env = env
        # pairwise structure does not change across episodes
        _, _, pairwise = self.env.get_status_and_factors()
        edge_list = []
        edge_attrs = []
        for uv, table in pairwise.items():
            u_name, v_name = sorted(list(uv))
            u = int(u_name[1:])
            v = int(v_name[1:])
            edge_list.append([u, v])
            edge_attrs.append(table.flatten())
        if len(edge_list) == 0:
            self.edge_index_template = None
            self.edge_attr_template = None
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_attrs, dtype=np.float32))
            # make edges undirected by duplicating reversed edges
            self.edge_index_template = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            self.edge_attr_template = torch.cat([edge_attr, edge_attr], dim=0)

    @torch.no_grad()
    def build_graph(self, status_np: np.ndarray) -> Data:
        n = len(status_np)
        status_f = status_np.astype(np.float32)
        _, unary, _ = self.env.get_status_and_factors()

        node_feats = []
        for i in range(n):
            Xi = f"X{i}"
            u = unary[Xi].flatten()
            # simple node features: unary[0], unary[1], status_i
            node_feats.append([float(u[0]), float(u[1]), float(status_f[i])])
        x = torch.tensor(np.array(node_feats, dtype=np.float32), device=device)

        if self.edge_index_template is None:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, 4), dtype=torch.float32, device=device)
        else:
            edge_index = self.edge_index_template.to(device)
            edge_attr = self.edge_attr_template.to(device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


##############################################################
# Networks: graph encoder + action MLP
##############################################################

class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        # small edge networks for NNConv
        self.edge_nn1 = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_in_dim * hidden_dim),
        )
        self.edge_nn2 = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv1 = NNConv(node_in_dim, hidden_dim, self.edge_nn1, aggr="mean")
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_nn2, aggr="mean")
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv2(x,        data.edge_index, data.edge_attr))
        batch = x.new_zeros(x.size(0), dtype=torch.long)  # single-graph batch
        g = global_mean_pool(x, batch)                   # [1, hidden_dim]
        return self.proj(g).squeeze(0)                   # [out_dim]


class ActionQMLP(nn.Module):
    """
    MLP that takes [a, g(s)] as input and outputs Q(s, a).

    This is the network that Net2MIPPerScenario will see.
    """

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.relu_input = nn.Linear(input_dim, hidden)
        self.relu_mid = nn.Linear(hidden, hidden)
        self.relu_output = nn.Linear(hidden, 1)

    def forward(self, a_and_g: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.relu_input(a_and_g))
        z = F.relu(self.relu_mid(z))
        out = self.relu_output(z)
        return out  # [B, 1]


class PolicyQNet(nn.Module):
    """
    Full Q-network: encodes graph to g(s), then passes [a, g(s)] into ActionQMLP.
    """

    def __init__(self, node_in_dim: int, edge_in_dim: int, n_actions: int, g_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.encoder = GraphEncoder(node_in_dim, edge_in_dim, hidden_dim=64, out_dim=g_dim)
        self.action_mlp = ActionQMLP(input_dim=n_actions + g_dim, hidden=hidden)
        self.n_actions = n_actions
        self.g_dim = g_dim

    @torch.no_grad()
    def embed_state(self, data: Data) -> torch.Tensor:
        return self.encoder(data)

    def forward(self, data: Data, a: torch.Tensor) -> torch.Tensor:
        """
        data: torch_geometric Data for current graph state
        a:   [n_actions] or [B, n_actions] action vector(s)
        """
        g = self.encoder(data)  # [g_dim]
        if a.dim() == 1:
            # single action
            cat = torch.cat([a, g], dim=0).unsqueeze(0)  # [1, n_actions + g_dim]
        else:
            B = a.size(0)
            g_rep = g.unsqueeze(0).expand(B, -1)
            cat = torch.cat([a, g_rep], dim=1)  # [B, n_actions + g_dim]
        return self.action_mlp(cat)  # [B, 1]


##############################################################
# DQN solver (graph + MILP)
##############################################################

class DQNSolver:
    def __init__(self, env):
        """
        env: BinaryFrontierEnvBatch (or compatible graph environment).
        """
        print('  [DQN Init] Setting up environment...')
        self.env = env
        self.budget = env.budget

        # DQN hyperparams
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000

        self.LR = 2.5e-4
        self.ADAM_EPS = 1.5e-4
        self.GRADIENT_CLIP = 5.0
        self.SCHEDULER_GAMMA = 0.9

        self.BATCH_SIZE = 32
        self.MEMORY_SIZE = int(1e4)
        self.MEMOIZER_REFRESH = 5

        if torch.cuda.is_available():
            self.N_EPISODES = 100
        else:
            self.N_EPISODES = 100

        # initialize environment once to infer dimensions
        print('  [DQN Init] Inferring dimensions...')
        status, _ = self.env.reset()
        self.n_actions = len(status)
        print(f'  [DQN Init] Action dimension: {self.n_actions}')

        # graph adapter
        print('  [DQN Init] Building graph adapter...')
        self.graph_adapter = GraphEnvAdapter(self.env)
        data0 = self.graph_adapter.build_graph(status)
        node_in_dim = data0.x.size(-1)
        if data0.edge_attr is not None and data0.edge_attr.numel() > 0:
            edge_in_dim = data0.edge_attr.size(-1)
        else:
            edge_in_dim = 4  # default fallback
        print(f'  [DQN Init] Node features: {node_in_dim}, Edge features: {edge_in_dim}')

        # networks (initialized in train())
        print('  [DQN Init] Creating neural networks...')
        self.policy_net = PolicyQNet(node_in_dim, edge_in_dim, self.n_actions, g_dim=64, hidden=128).to(device)
        self.target_net = PolicyQNet(node_in_dim, edge_in_dim, self.n_actions, g_dim=64, hidden=128).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f'  [DQN Init] Networks created (device: {device})')

        print('  [DQN Init] Setting up optimizer and scheduler...')
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, eps=self.ADAM_EPS, amsgrad=True)
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.SCHEDULER_GAMMA)

        print('  [DQN Init] Creating replay buffer and memoizer...')
        self.memory = ReplayBuffer(self.MEMORY_SIZE)
        self.memoizer = Memoizer(refresh=self.MEMOIZER_REFRESH)

        # approximator (MILP builder)
        print('  [DQN Init] Building MILP approximator (this may take a moment)...')
        approximator_cls = self.env.get_approximator()
        self.approximator = approximator_cls(self.env, model_type="NN-E")
        print('  [DQN Init] MILP approximator ready')

        self.step_count = 0
        self.optimizer_loss = []

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _build_graph_from_status(self, s_np: np.ndarray) -> Data:
        return self.graph_adapter.build_graph(s_np)

    def _random_action(self) -> torch.Tensor:
        """
        Random feasible batch action via env.random_feasible_action().
        """
        a_np = self.env.random_feasible_action()
        return torch.tensor(a_np, dtype=torch.float32, device=device)

    # --------------------------------------------------------
    # Action selection: eps-greedy + MILP
    # --------------------------------------------------------
    def select_action(self, status_np: np.ndarray, verbose: bool = False) -> torch.Tensor:
        eps_sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1.0 * self.step_count / self.EPS_DECAY)
        self.step_count += 1

        if eps_sample < eps_threshold:
            if verbose:
                print(f"    -> Random action (eps={eps_threshold:.3f})")
            return self._random_action()

        # exploitation: MILP with Q-network
        if verbose:
            print(f"    -> MILP action (eps={eps_threshold:.3f})")
            print(f"       Building graph and embedding state...")
        with torch.no_grad():
            data_s = self._build_graph_from_status(status_np)
            g_s = self.policy_net.embed_state(data_s).detach().cpu().numpy().astype(np.float32)

            if verbose:
                print(f"       Calling MILP solver (gap=0.02, time_limit=60s)...")
            results = self.approximator.approximate(
                network=self.policy_net.action_mlp,       # MLP over [a, g_s]
                mipper_cls=Net2MIPPerScenario,
                n_scenarios=1,
                gap=0.02,
                time_limit=60,
                threads=4,
                scenario_embedding=g_s,                   # this is g(s)
                scenario_probs=None,
            )
            if verbose:
                print(f"       MILP solved!")
            action = results["sol"]
            if not torch.is_tensor(action):
                action = torch.tensor(action, dtype=torch.float32)
            return action.to(device)

    # --------------------------------------------------------
    # Single mini-batch optimization step
    # --------------------------------------------------------
    def optimize_model_single_batch(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # convert to numpy arrays
        state_np_batch = [s.cpu().numpy().astype(np.float32) for s in batch.state]
        next_state_np_batch = [s.cpu().numpy().astype(np.float32) for s in batch.next_state]
        action_batch = torch.stack(batch.action).to(device)
        cost_batch = torch.stack(batch.cost).to(device).unsqueeze(1)

        # Q(s,a)
        q_values = []
        for s_np, a_t in zip(state_np_batch, action_batch):
            data_s = self._build_graph_from_status(s_np)
            q = self.policy_net(data_s, a_t.unsqueeze(0))  # [1,1]
            q_values.append(q.squeeze(0))
        q_values = torch.stack(q_values, dim=0)  # [B,1]

        # Q-target(s', a*) using target_net and MILP
        with torch.no_grad():
            next_q_values = []
            for ns_np in next_state_np_batch:
                key = tuple(ns_np.tolist())
                cached = self.memoizer.check_key(key)

                if cached is not None:
                    best_action = cached
                else:
                    # temporarily override env.status so allowed_mask() uses ns_np
                    old_status = self.env.status.copy()
                    self.env.status = ns_np.astype(int).copy()

                    data_sp = self._build_graph_from_status(ns_np)
                    g_sp = self.target_net.embed_state(data_sp).detach().cpu().numpy().astype(np.float32)

                    results = self.approximator.approximate(
                        network=self.target_net.action_mlp,
                        mipper_cls=Net2MIPPerScenario,
                        n_scenarios=1,
                        gap=0.02,
                        time_limit=60,
                        threads=4,
                        scenario_embedding=g_sp,
                        scenario_probs=None,
                    )
                    best_action = results["sol"]
                    self.env.status = old_status
                    self.memoizer.add(key, best_action)

                if not torch.is_tensor(best_action):
                    best_action_t = torch.tensor(best_action, dtype=torch.float32, device=device)
                else:
                    best_action_t = best_action.to(device=device, dtype=torch.float32)

                data_sp = self._build_graph_from_status(ns_np)
                q_sp = self.target_net(data_sp, best_action_t.unsqueeze(0))  # [1,1]
                next_q_values.append(q_sp.squeeze(0))

            next_q_values = torch.stack(next_q_values, dim=0)  # [B,1]

        target = cost_batch + self.GAMMA * next_q_values  # [B,1]
        loss = self.criterion(q_values, target).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.GRADIENT_CLIP)
        self.optimizer.step()

        self.optimizer_loss.append(loss.item())

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    def training_loop(self, horizon: int | None = None):
        if horizon is None:
            horizon = self.env.n  # reveal all eventually if you want

        print("----------- begin main loop")
        print(f"  episodes: {self.N_EPISODES}")
        print(f"  horizon: {horizon} steps/episode")
        print(f"  replay capacity: {self.MEMORY_SIZE}")
        print(f"  device: {device}")
        print("  starting episode 1...")
        
        for ep in tqdm.tqdm(range(self.N_EPISODES)):
            if ep == 0:
                print("  [Episode 1] Resetting environment...")
            status, mask = self.env.reset()
            done = False
            t = 0

            while not done and t < horizon:
                if ep == 0 and t == 0:
                    print(f"  [Episode 1, Step 1] Selecting action (this calls MILP solver)...")
                    a_t = self.select_action(status, verbose=True)  # [n_actions]
                    print(f"  [Episode 1, Step 1] Action selected, executing step...")
                else:
                    a_t = self.select_action(status)  # [n_actions]
                next_status, next_mask, reward, done = self.env.step(
                    a_t.detach().cpu().numpy().astype(int)
                )

                s_t = torch.tensor(status, dtype=torch.float32, device=device)
                s_tp1 = torch.tensor(next_status, dtype=torch.float32, device=device)
                cost_t = torch.tensor(-float(reward), dtype=torch.float32, device=device)

                self.memory.add(s_t, a_t, s_tp1, cost_t)

                self.optimize_model_single_batch()

                status = next_status
                t += 1

            self.memoizer.new_episode()
            self.scheduler.step()

        # plot loss
        plt.figure()
        plt.plot(np.arange(len(self.optimizer_loss)), np.array(self.optimizer_loss))
        plt.xlabel("update step")
        plt.ylabel("loss")
        plt.title(f"DQN graph+MILP loss n={self.n_actions} budget={self.budget}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/dqn_frontier_graph_mip_loss_{timestamp}.png")
        plt.close()

    # --------------------------------------------------------
    # Public entry
    # --------------------------------------------------------
    def train(self, horizon: int | None = None):
        """
        Run the full DQN training loop. Returns the trained policy_net.
        """
        print("----------- training DQN (graph + MILP)")
        self.training_loop(horizon=horizon)
        return self.policy_net
