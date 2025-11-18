# dqn_frontier_mip.py
"""
DQN + MILP solver for batch frontier testing on a graph environment.

Structure follows the original RMAB DQN code, but:
  - Environment is BinaryFrontierEnvBatch (graph frontier testing).
  - We use BatchGraphApproximator + Net2MIPPerScenario for argmax_a Q(s,a).
  - No myopic pretraining; only online DQN with a simple replay buffer.
"""

import math
import random
import tqdm
import datetime
import warnings
from collections import namedtuple, deque
from typing import List

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from model2mip.net2mip import Net2MIPPerScenario

from approximator.batch_graph_approximator import BatchGraphApproximator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################################
# DQN utils
##############################################################

class Memoizer:
    """Cache past MILP solves (state -> best action) across episodes."""

    def __init__(self, refresh=5):
        self.refresh = refresh  # how many episodes of solves to save
        # existing_solves will be a list of dicts, where most recent items are at the front
        self.existing_solves = [{}]
        self.episode = 0

        # store the fraction of checks that actually had a success
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
        """Delete the oldest dict and add in a dict for a new episode."""
        tot_solves = np.array(self.num_checks).sum()
        tot_successes = np.array(self.num_successes).sum()
        if tot_solves > 0:
            frac = tot_successes / tot_solves
            # print(f'memoizer: episode {self.episode}, frac success {frac:.2f} ({tot_successes}/{tot_solves})')

        del self.existing_solves[-1]
        self.existing_solves.insert(0, {})
        self.episode += 1

        del self.num_checks[-1]
        del self.num_successes[-1]
        self.num_checks.insert(0, 0)
        self.num_successes.insert(0, 0)

        return self.episode

    def check_key(self, key):
        """If key is already stored, return value; otherwise None."""
        self.num_checks[0] += 1
        for solve_list in self.existing_solves:
            if key in solve_list:
                self.num_successes[0] += 1
                return solve_list[key]
        return None


# transitions: state, action, next_state, cost (negative reward)
Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "cost"),
)


class ReplayBuffer:
    """Simple uniform replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.memory: List[Transition] = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)


##############################################################
# DQN network
##############################################################

class DQN(nn.Module):
    """Q-function estimator: input = concat([action, state]) -> scalar Q(s,a)."""

    def __init__(self, in_dim, out_dim=1):
        super(DQN, self).__init__()
        self.hidden_dim = [64, 64]  # can adjust

        self.fc1 = nn.Linear(in_dim, self.hidden_dim[0])
        self.fc2 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        self.fc3 = nn.Linear(self.hidden_dim[1], out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [B, out_dim]
        return x


##############################################################
# DQN solver for BinaryFrontierEnvBatch
##############################################################

class DQNSolver:
    def __init__(self, env):
        """
        env: BinaryFrontierEnvBatch (or compatible)
        """
        self.env = env

        # discount factor
        self.GAMMA = 0.99

        # epsilon-greedy parameters
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000  # rate of exponential decay of epsilon

        # optimizer parameters
        self.LR = 2.5e-4
        self.ADAM_EPS = 1.5e-4
        self.GRADIENT_CLIP = 5.0
        self.SCHEDULER_GAMMA = 0.9

        # replay buffer parameters
        self.BATCH_SIZE = 32
        self.MEMORY_SIZE = int(1e4)

        # memoizer refresh
        self.MEMOIZER_REFRESH = 5

        # number of episodes
        if torch.cuda.is_available():
            self.N_EPISODES = 100
        else:
            self.N_EPISODES = 100

        # infer dimensions from environment
        # assume:
        #   - state: np.array length n
        #   - action: np.array length n (0/1 per node)
        status, _ = self.env.reset()
        self.state_dim = len(status)
        self.action_dim = len(status)
        self.budget = self.env.budget

        # placeholders (initialized in train())
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.approximator = None
        self.memoizer = None
        self.memory = None

        self.step_count = 0
        self.optimizer_loss = []

    # --------------------------------------------------------
    # Helper: random feasible action from current frontier
    # --------------------------------------------------------
    def get_random_action(self) -> torch.Tensor:
        """
        Random feasible batch action (0/1 vector) using env.get_random_action().
        """
        a_np = self.env.get_random_action()
        return torch.tensor(a_np, dtype=torch.float32, device=device)

    # --------------------------------------------------------
    # Action selection using MILP + Net2MIP
    # --------------------------------------------------------
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Epsilon-greedy:
          - With prob epsilon: random feasible action.
          - Else: solve MILP with BatchGraphApproximator + Net2MIPPerScenario.
        scenario_embedding is the state vector.
        """
        eps_sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1.0 * self.step_count / self.EPS_DECAY)
        self.step_count += 1

        # exploration
        if eps_sample < eps_threshold:
            return self.get_random_action()

        # exploitation via MILP
        with torch.no_grad():
            # For MILP embedding, we use the policy_net directly.
            state_np = state.detach().cpu().numpy().astype(np.float32)
            results = self.approximator.approximate(
                network=self.policy_net,
                mipper_cls=Net2MIPPerScenario,
                n_scenarios=1,
                gap=0.02,
                time_limit=60,
                threads=4,
                scenario_embedding=state_np,
                scenario_probs=None,
            )
            action = results["sol"]
            if not torch.is_tensor(action):
                action = torch.tensor(action, dtype=torch.float32)  # [n]
            return action.to(device)

    # --------------------------------------------------------
    # Single mini-batch optimization
    # --------------------------------------------------------
    def optimize_model_run_single_batch(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # stack into tensors
        state_batch = torch.stack(batch.state).to(device)      # [B, state_dim]
        action_batch = torch.stack(batch.action).to(device)    # [B, action_dim]
        next_states = torch.stack(batch.next_state).to(device) # [B, state_dim]
        cost_batch = torch.stack(batch.cost).to(device).unsqueeze(1)  # [B, 1]

        # Q(s,a)
        sa_input = torch.cat([action_batch, state_batch], dim=1)  # [B, action_dim+state_dim]
        state_action_values = self.policy_net(sa_input)           # [B, 1]

        # V(s') ~ max_a' Q(s', a') via MILP + target_net
        with torch.no_grad():
            all_best_actions = torch.zeros(
                (self.BATCH_SIZE, self.action_dim),
                dtype=torch.float32,
                device=device,
            )

            for k in range(self.BATCH_SIZE):
                ns = next_states[k]
                ns_np = ns.cpu().numpy().astype(np.float32)
                key = tuple(ns_np.tolist())

                cached = self.memoizer.check_key(key)
                if cached is not None:
                    best_action = cached
                else:
                    # temporarily override env.status so allowed_mask() matches ns
                    old_status = self.env.status.copy()
                    self.env.status = ns_np.astype(int).copy()

                    results = self.approximator.approximate(
                        network=self.target_net,
                        mipper_cls=Net2MIPPerScenario,
                        n_scenarios=1,
                        gap=0.02,
                        time_limit=60,
                        threads=4,
                        scenario_embedding=ns_np,
                        scenario_probs=None,
                    )
                    best_action = results["sol"]

                    # restore
                    self.env.status = old_status
                    self.memoizer.add(key, best_action)

                if torch.is_tensor(best_action):
                    best_action_t = best_action.to(device=device, dtype=torch.float32)
                else:
                    best_action_t = torch.tensor(best_action, dtype=torch.float32, device=device)

                all_best_actions[k, :] = best_action_t

            net_input = torch.cat([all_best_actions, next_states], dim=1)
            next_state_values = self.target_net(net_input)  # [B,1]

        # TD target: immediate cost + discounted next value
        expected_state_action_values = cost_batch + self.GAMMA * next_state_values

        # Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # optimize
        self.optimizer.zero_grad()
        loss.mean().backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.GRADIENT_CLIP)
        self.optimizer.step()

        self.optimizer_loss.append(loss.mean().item())

    # --------------------------------------------------------
    # Main training loop
    # --------------------------------------------------------
    def training_loop(self, horizon=None):
        if horizon is None:
            horizon = self.env.n  # by default, reveal all nodes eventually

        print("----------- begin main loop")
        print(f"  replay capacity {self.MEMORY_SIZE}")

        for i_episode in tqdm.tqdm(range(self.N_EPISODES)):
            # initialize environment and get state
            status, valid_mask = self.env.reset()
            state = torch.tensor(status, dtype=torch.float32, device=device)

            for t in range(horizon):
                # select action
                action = self.select_action(state)  # [n]

                # step env
                next_status, next_valid, reward, done = self.env.step(
                    action.detach().cpu().numpy().astype(int)
                )
                next_state = torch.tensor(next_status, dtype=torch.float32, device=device)
                cost = torch.tensor(-float(reward), dtype=torch.float32, device=device)

                # store transition
                self.memory.add(state, action, next_state, cost)

                # optimize one step
                self.optimize_model_run_single_batch()

                # soft update of target network
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (
                        policy_net_state_dict[key] * self.TAU
                        + target_net_state_dict[key] * (1.0 - self.TAU)
                    )
                self.target_net.load_state_dict(target_net_state_dict)

                state = next_state

                if done:
                    break

            # memoizer aging & LR scheduling
            self.memoizer.new_episode()
            self.scheduler.step()

        # plot loss curve
        plt.figure()
        plt.plot(np.arange(len(self.optimizer_loss)), np.array(self.optimizer_loss))
        plt.xlabel("update step")
        plt.ylabel("loss")
        plt.title(
            f"DQN loss n={self.action_dim} budget={self.budget}"
        )
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"plots/dqn_frontier_loss_{timestamp}.png")
        plt.close()

    # --------------------------------------------------------
    # Public entry point
    # --------------------------------------------------------
    def train(self):
        """
        Initialize networks, optimizer, approximator and run the training loop.
        Returns the trained policy network.
        """
        print("----------- training DQN with MILP-based action selection")

        # tracking stats
        self.optimizer_loss = []
        self.step_count = 0

        # initialize networks
        in_dim = self.action_dim + self.state_dim
        self.policy_net = DQN(in_dim, 1).to(device)
        self.target_net = DQN(in_dim, 1).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # optimizer etc.
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.LR,
            eps=self.ADAM_EPS,
            amsgrad=True,
        )
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.SCHEDULER_GAMMA)

        # approximator & memoizer
        # environment provides the appropriate approximator class
        approximator_cls = self.env.get_approximator()
        self.approximator = approximator_cls(self.env, model_type="NN-E")
        self.memoizer = Memoizer(refresh=self.MEMOIZER_REFRESH)

        # replay buffer
        self.memory = ReplayBuffer(self.MEMORY_SIZE)

        # run main training loop
        self.training_loop()

        return self.policy_net
