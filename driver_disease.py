""" Driver for Disease Graph DQN Testing

To run:
> conda activate AFEG
> python driver_disease.py
> python driver_disease.py -s 0 -D HIV -T 100 -B 5 -p experiment1_

or 
> ./run_dqn.sh
"""

import os
import sys
import argparse
import datetime
from collections import OrderedDict

import random
import numpy as np
import pandas as pd
from scipy import stats
import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches
from tqdm import tqdm

from environment.disease_graph_loader import (
    load_disease_graph_instance, 
    create_disease_env
)

from algos.dqn_estimator_disease import DQNSolver
from algos.evaluate_MIP_disease import MIP_results
from algos.baselines_disease import (
    baseline_null_action_disease,
    baseline_random_disease,
    baseline_myopic_disease,
    baseline_greedy_iterative_myopic_disease,
    baseline_sampling_disease,
)
from algos.baseline_iterative_dqn_disease import baseline_iterative_dqn_disease

from model2mip.net2mip import Net2MIPPerScenario


# Root directory for all experiment outputs (CSVs + plots)
results_root = './results'

if not os.path.exists(results_root):
    os.makedirs(results_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', help='random seed', type=int, default=0)
    parser.add_argument('--std_name', '-D', help='disease type {HIV, Gonorrhea, Chlamydia, Syphilis, Hepatitis}', 
                       type=str, default='HIV')
    parser.add_argument('--cc_threshold', '-T', help='minimum nodes to sample from connected components', 
                       type=int, default=100)
    parser.add_argument('--inst_idx', '-I', help='instance index (random seed for sampling)', 
                       type=int, default=0)
    parser.add_argument('--budget', '-B', help='budget (nodes to test per step)', type=int, default=5)
    parser.add_argument('--discount', '-G', help='discount factor', type=float, default=0.99)
    parser.add_argument('--n_episodes_eval', '-V', help='number of episodes to run for evaluation', 
                       type=int, default=50)
    parser.add_argument('--prefix', '-p', help='prefix for file writing', type=str, default='')
    parser.add_argument('--n_samples', '-K', help='samples for sampling baseline', type=int, default=50)
    parser.add_argument(
        '--models',
        '-M',
        help=('comma separated models to run: '
              '{null,random,sampling,iter_myopic,iter_dqn,dqn_mip}'),
        type=str,
        default='null,random,sampling,iter_myopic,iter_dqn,dqn_mip',
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    std_name = args.std_name
    cc_threshold = args.cc_threshold
    inst_idx = args.inst_idx
    budget = args.budget
    discount = args.discount
    n_episodes_eval = args.n_episodes_eval
    prefix = args.prefix
    n_samples = args.n_samples
    model_names = [m.strip().lower() for m in args.models.split(',') if m.strip()]

    print('--------------------------------------------------------')
    print('Load Disease Graph')
    print('--------------------------------------------------------')
    
    G, covariates, theta_unary, theta_pairwise, statuses = load_disease_graph_instance(
        std_name=std_name,
        cc_threshold=cc_threshold,
        inst_idx=inst_idx
    )
    
    print('graph stats')
    print(f'  disease: {std_name}')
    print(f'  nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}')
    print(f'  infected: {sum(statuses.values())}/{len(statuses)} '
          f'({100*sum(statuses.values())/len(statuses):.1f}%)')
    print(f'  covariate dim: {len(covariates[0])}')

    print('--------------------------------------------------------')
    print('Create Disease Environment')
    print('--------------------------------------------------------')
    
    env = create_disease_env(
        G, covariates, theta_unary, theta_pairwise,
        budget=budget,
        discount_factor=discount,
        rng_seed=args.seed
    )
    
    horizon_steps = env.n
    print(f'environment: n={env.n}, budget={env.budget}')
    print(f'connected components: {len(env.cc_root)}')

    # Generate initial states for evaluation
    print(f'\nGenerating {n_episodes_eval} initial states for evaluation...')
    init_states = []
    for i in range(n_episodes_eval):
        status, mask = env.reset()
        init_states.append(status)
    print(f'✓ Generated {len(init_states)} initial states')
    
    print(f'\nRunning {std_name} disease testing with prefix {prefix}')

    start_time = datetime.datetime.now()
    algo_rewards = OrderedDict()

    # Non-DQN baselines
    baseline_registry = {
        'null': ('null', lambda: baseline_null_action_disease(env, init_states, horizon=horizon_steps)),
        'random': ('random', lambda: baseline_random_disease(env, init_states, horizon=horizon_steps)),
        'sampling': (
            f'sampling (k={n_samples})',
            lambda: baseline_sampling_disease(env, init_states, horizon=horizon_steps, n_samples=n_samples),
        ),
        'iter_myopic': (
            'iterative myopic',
            lambda: baseline_greedy_iterative_myopic_disease(env, init_states, horizon=horizon_steps),
        ),
    }

    run_iter_dqn = 'iter_dqn' in model_names
    run_dqn_mip = 'dqn_mip' in model_names
    # Train DQN iff at least one DQN-based model is requested
    run_dqn = run_iter_dqn or run_dqn_mip

    print('--------------------------------------------------------')
    print('Run Baselines')
    print('--------------------------------------------------------')
    for name in tqdm(model_names, desc="Baselines/Models"):
        if name in ('iter_dqn', 'dqn_mip'):
            # DQN-based models handled after training
            continue
        if name not in baseline_registry:
            print(f'  [warning] unknown or unsupported model "{name}", skipping.')
            continue
        label, fn = baseline_registry[name]
        print(f'  -> {label}')
        algo_rewards[label] = fn()

    print('--------------------------------------------------------')
    print('Train DQN Solver')
    print('--------------------------------------------------------')

    start_time = datetime.datetime.now()
    dqn_net = None
    dqn_runtime = None

    # Train DQN only if at least one DQN-based model is requested
    if run_dqn:
        print('Initializing DQN solver...')
        dqn_solver = DQNSolver(env)
        print('✓ DQN solver initialized')
        print('Starting training...')
        dqn_net = dqn_solver.train()
        dqn_end_time = datetime.datetime.now()
        dqn_runtime = (dqn_end_time - start_time).total_seconds()

        if run_dqn_mip:
            print('--------------------------------------------------------')
            print('Evaluate DQN-MIP')
            print('--------------------------------------------------------')
            mipper = Net2MIPPerScenario
            algo_rewards['DQN MIP'] = MIP_results(env, dqn_net, mipper, init_states)

        if run_iter_dqn:
            print('Running iterative DQN baseline...')
            algo_rewards['iterative DQN'] = baseline_iterative_dqn_disease(
                env, dqn_net, init_states, horizon=horizon_steps)
    else:
        if run_iter_dqn or run_dqn_mip:
            print('WARNING: iter_dqn and/or dqn_mip requested but no DQN models were run.')

    if not algo_rewards:
        print('No algorithms selected to run. Exiting.')
        sys.exit(0)

    # --------------------------------------------------------
    # Compute discounted metrics for all algorithms
    # --------------------------------------------------------
    gamma = discount
    discounted_stats = {}  # algo -> (mean_final_disc, sem_final_disc)
    per_algo_horizon = {}  # algo -> horizon_steps_algo

    for algo, rewards_1d in algo_rewards.items():
        rewards = np.asarray(rewards_1d, dtype=float)
        total_steps = rewards.size
        horizon_algo = total_steps // n_episodes_eval
        per_algo_horizon[algo] = horizon_algo

        r_2d = rewards.reshape(n_episodes_eval, horizon_algo)
        # apply per-timestep discount gamma^t
        discounts = (gamma ** np.arange(horizon_algo))[None, :]
        disc_immediate = r_2d * discounts
        cum_disc = disc_immediate.cumsum(axis=1)
        final_disc = cum_disc[:, -1]
        mean_final = final_disc.mean()
        sem_final = stats.sem(final_disc) if n_episodes_eval > 1 else 0.0
        discounted_stats[algo] = (mean_final, sem_final)

    print('avg discounted rewards (sem)')
    print(f'{std_name} disease testing  n={env.n}, budget={budget}, gamma={gamma}')
    for algo in algo_rewards:
        mean_final, sem_final = discounted_stats[algo]
        print(f'  {algo.ljust(18, " ")}  {mean_final:.2f}, {sem_final:.2f}')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # --------------------------------------------------------
    # Create a unique run directory under results/
    # --------------------------------------------------------
    run_dir_name = f'{prefix}{std_name}_T{cc_threshold}_B{budget}_seed{args.seed}_{timestamp}'
    run_dir = os.path.join(results_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # --------------------------------------------------------
    # Save summary metrics (one row per run, all algorithms)
    # --------------------------------------------------------
    out_info = {
        'seed': args.seed,
        'std_name': std_name,
        'cc_threshold': cc_threshold,
        'inst_idx': inst_idx,
        'n_nodes': env.n,
        'budget': budget,
        'n_episodes_eval': n_episodes_eval,
        'start_time': start_time.strftime('%Y-%m-%d_%H-%M-%S'),
        'time': timestamp,
        'dqn_runtime': dqn_runtime,
    }
    for algo in algo_rewards:
        mean_final, sem_final = discounted_stats[algo]
        out_info[f'{algo}_disc_mean'] = mean_final
        out_info[f'{algo}_disc_sem'] = sem_final

    summary_df = pd.DataFrame([out_info])
    summary_file = os.path.join(run_dir, 'summary.csv')
    use_header = not os.path.exists(summary_file)
    summary_df.to_csv(summary_file, mode='a', index=False, header=use_header)

    # --------------------------------------------------------
    # Save per-timestep rewards for all algorithms in a single CSV
    # --------------------------------------------------------
    traj_rows = []
    for algo_name, rewards in algo_rewards.items():
        rewards = np.asarray(rewards, dtype=float)
        total_steps = rewards.size
        horizon_steps_algo = per_algo_horizon[algo_name]
        rewards_2d = rewards.reshape(n_episodes_eval, horizon_steps_algo)

        # discounted cumulative rewards per episode
        discounts = (gamma ** np.arange(horizon_steps_algo))[None, :]
        disc_immediate = rewards_2d * discounts
        cum_disc = disc_immediate.cumsum(axis=1)

        for ep in range(n_episodes_eval):
            for t in range(horizon_steps_algo):
                traj_rows.append(
                    {
                        'seed': args.seed,
                        'std_name': std_name,
                        'cc_threshold': cc_threshold,
                        'inst_idx': inst_idx,
                        'n_nodes': env.n,
                        'budget': budget,
                        'algo': algo_name,
                        'episode': ep,
                        'step': t,
                        'reward': rewards_2d[ep, t],
                        'discounted_cum_reward': cum_disc[ep, t],
                    }
                )
    traj_df = pd.DataFrame(traj_rows)
    traj_file = os.path.join(run_dir, 'trajectories.csv')
    use_header_traj = not os.path.exists(traj_file)
    traj_df.to_csv(traj_file, mode='a', index=False, header=use_header_traj)

    # --------------------------------------------------------
    # Plot: fraction tested vs positive cases detected (per-policy curves)
    # and interaction graph, with shared legend
    # --------------------------------------------------------
    fig, (ax_curve, ax_graph) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    linestyles = ['dotted', 'dashed', 'dashdot', 'solid', (0, (3, 2, 1, 2)), 'solid', 'dashed', 'dotted']

    policy_labels = list(algo_rewards.keys())
    all_lines = []

    # Assume a near-constant batch size equal to env.budget for X-axis
    max_horizon = max(per_algo_horizon.values())
    tests_after_step = np.minimum((np.arange(max_horizon) + 1) * budget, env.n)
    frac_tested = tests_after_step / float(env.n)

    for idx, algo in enumerate(policy_labels):
        rewards = np.asarray(algo_rewards[algo], dtype=float)
        horizon_steps_algo = per_algo_horizon[algo]
        rewards_2d = rewards.reshape(n_episodes_eval, horizon_steps_algo)
        # use discounted cumulative rewards for curves
        discounts = (gamma ** np.arange(horizon_steps_algo))[None, :]
        disc_immediate = rewards_2d * discounts
        cum_rewards = disc_immediate.cumsum(axis=1)
        mean_vec = cum_rewards.mean(axis=0)
        std_vec = cum_rewards.std(axis=0) / np.sqrt(n_episodes_eval)

        # Align to max_horizon by padding with last value if needed
        if horizon_steps_algo < max_horizon:
            pad_len = max_horizon - horizon_steps_algo
            mean_vec = np.concatenate([mean_vec, np.full(pad_len, mean_vec[-1])])
            std_vec = np.concatenate([std_vec, np.zeros(pad_len)])

        max_y = np.max(mean_vec) if np.max(mean_vec) > 0 else 1.0
        scaled_y = mean_vec / max_y
        scaled_std = std_vec / max_y

        color = colors[idx % len(colors)]
        style = linestyles[idx % len(linestyles)]
        line_handle, = ax_curve.plot(
            frac_tested, scaled_y, ls=style, color=color, lw=2, label=algo
        )
        ax_curve.fill_between(
            frac_tested,
            scaled_y - scaled_std,
            scaled_y + scaled_std,
            color=color,
            alpha=0.2,
        )
        all_lines.append(line_handle)

    ax_curve.axvline(x=0.5, color='gray', linestyle='--', linewidth=2)
    ax_curve.set_xlabel("Fraction of population tested")
    ax_curve.set_ylabel("Fraction of positive cases detected (normalized)")
    ax_curve.set_title(f"Policies with discount={discount}")

    # --------------------------------------------------------
    # Plot: interaction graph with frontier roots highlighted
    # --------------------------------------------------------
    pos = nx.spring_layout(G, seed=args.seed)
    # env.cc_root contains integer node indices for roots
    root_nodes = set(env.cc_root) if hasattr(env, "cc_root") else set()
    node_colors = ['red' if i in root_nodes else 'blue' for i in G.nodes()]
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=10,
        edge_color='black',
        with_labels=False,
        width=1.0,
        alpha=0.8,
        ax=ax_graph,
    )
    for root_idx in root_nodes:
        if root_idx in pos:
            circle = patches.Circle(
                pos[root_idx],
                radius=0.05,
                facecolor='none',
                edgecolor='red',
                linewidth=2,
            )
            ax_graph.add_patch(circle)
    ax_graph.text(
        0.5,
        1.02,
        f"{std_name} sex interaction graph",
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax_graph.transAxes,
        fontsize=12,
    )
    ax_graph.text(
        0.5,
        -0.05,
        "Frontier roots are circled in red",
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax_graph.transAxes,
        color='red',
        fontsize=10,
    )

    # Overall legend (similar to experiment3.py)
    ncols = min(len(policy_labels), 5)
    fig.legend(
        all_lines,
        policy_labels,
        loc='upper center',
        ncol=ncols,
        bbox_to_anchor=(0.5, 1.05),
        fontsize=10,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    curve_file = os.path.join(
        run_dir,
        f'exp_curve_{std_name}_T{cc_threshold}_B{budget}_seed{args.seed}.png',
    )
    plt.savefig(curve_file, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print('--------------------------------------------------------')
    print('Experiment Complete')
    print('--------------------------------------------------------')
    print(f'Run directory: {run_dir}')
    print(f'Summary saved to: {summary_file}')
    print(f'Trajectories saved to: {traj_file}')
    print(f'Plots saved to: {curve_file}')
