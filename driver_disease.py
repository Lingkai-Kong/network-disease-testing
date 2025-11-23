""" Driver for Disease Graph DQN Testing

To run:
> conda activate AFEG
> python driver_disease.py
> python driver_disease.py -s 0 -H 20 -D HIV -T 100 -B 5 -p experiment1_

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

from environment.disease_graph_loader import (
    load_disease_graph_instance, 
    create_disease_env
)

from algos.dqn_estimator_disease import DQNSolver
from algos.evaluate_MIP_disease import MIP_results

from model2mip.net2mip import Net2MIPPerScenario


out_dir = './plots'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', help='random seed', type=int, default=0)
    parser.add_argument('--horizon', '-H', help='time horizon', type=int, default=20)
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

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    horizon = args.horizon
    std_name = args.std_name
    cc_threshold = args.cc_threshold
    inst_idx = args.inst_idx
    budget = args.budget
    discount = args.discount
    n_episodes_eval = args.n_episodes_eval
    prefix = args.prefix

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
    
    print(f'environment: n={env.n}, budget={env.budget}, horizon={horizon}')
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
    print('--------------------------------------------------------')
    print('Train DQN Solver')
    print('--------------------------------------------------------')
    print('Initializing DQN solver...')
    dqn_solver = DQNSolver(env)
    print('✓ DQN solver initialized')
    print('Starting training...')
    
    dqn_net, myopic_net, midway_net = dqn_solver.train(horizon=horizon)
    dqn_end_time = datetime.datetime.now()

    print('--------------------------------------------------------')
    print('Evaluate DQN-MIP')
    print('--------------------------------------------------------')

    algo_rewards = OrderedDict()
    
    mipper = Net2MIPPerScenario
    algo_rewards['DQN MIP'] = MIP_results(env, dqn_net, mipper, init_states, horizon=horizon)

    print('avg rewards (sem)')
    print(f'{std_name} disease testing  n={env.n}, budget={budget}')
    for algo in algo_rewards:
        print(f'  {algo.ljust(18, " ")}  {algo_rewards[algo].mean():.2f}, {stats.sem(algo_rewards[algo]):.2f}')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_info = {
        'seed': args.seed,
        'std_name': std_name,
        'cc_threshold': cc_threshold,
        'inst_idx': inst_idx,
        'n_nodes': env.n,
        'budget': budget,
        'n_episodes_eval': n_episodes_eval,
        'horizon': horizon,
        'start_time': start_time.strftime('%Y-%m-%d_%H-%M-%S'),
        'time': timestamp,
        'dqn_runtime': (dqn_end_time - start_time).total_seconds(),
    }
    for algo in algo_rewards:
        out_info[algo] = algo_rewards[algo].mean()

    df_out = pd.DataFrame([out_info])
    file_out = f'results_{prefix}{std_name}_T{cc_threshold}_B{budget}.csv'
    use_header = not os.path.exists(file_out)
    # append data frame to CSV file
    df_out.to_csv(file_out, mode='a', index=False, header=use_header)

    # plot per-timestep reward
    x_vals = np.arange(horizon * n_episodes_eval)
    plt.figure()
    for algo in algo_rewards:
        rewards_sorted = np.sort(algo_rewards[algo])
        plt.plot(x_vals, rewards_sorted, label=algo)
    plt.legend()
    plt.title(f'{std_name} testing with n={env.n} and budget={budget}')
    plt.xlabel(f'Timestep ({n_episodes_eval} episodes, {horizon} horizon)')
    plt.ylabel('Per-timestep expected reward (sorted)')
    plt.tight_layout()

    # plot average reward per method as a bar plot
    plt.figure()
    bar_x = np.arange(len(algo_rewards))
    plt.bar(bar_x, [algo_rewards[algo].mean() for algo in algo_rewards], 
            yerr=[stats.sem(algo_rewards[algo]) for algo in algo_rewards], color='blue')
    plt.xticks(bar_x, [algo for algo in algo_rewards], rotation=30)
    plt.ylabel('Average expected reward')
    plt.xlabel('Method')
    plt.title(f'Mean reward per method: {std_name} (n={env.n}, B={budget})')
    plt.tight_layout()
    
    plt.savefig(f'{out_dir}/{prefix}avg_reward_{std_name}_T{cc_threshold}_B{budget}_seed{args.seed}_{timestamp}.png')
    
    plt.close('all')
    
    print('--------------------------------------------------------')
    print('Experiment Complete')
    print('--------------------------------------------------------')
    print(f'Results saved to: {file_out}')
    print(f'Plots saved to: {out_dir}/')
