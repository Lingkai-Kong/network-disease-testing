"""
Evaluate the DQN-MIP solver for disease graph environments.
"""
import sys
import numpy as np
import torch
from tqdm import tqdm

from algos.dqn_estimator_disease import GraphEnvAdapter
from approximator.batch_graph_approximator import BatchGraphApproximator


def MIP_results(env, network, mipper, init_states, horizon=None):
    """
    Use NN-embedded MIP to pick action at each timestep.
    Ensures all nodes are tested by running until done=True or horizon steps.
    
    Args:
        env: Disease environment (BinaryFrontierEnvBatch)
        network: Trained PolicyQNet
        mipper: MIP solver class (Net2MIPPerScenario)
        init_states: List of initial state dictionaries
        horizon: Episode horizon (default: env.n)
        
    Returns:
        Array of rewards per timestep
    """
    n_episodes = len(init_states)
    
    adapter = GraphEnvAdapter(env)
    approximator = BatchGraphApproximator(env, model_type="NN-E")
    network.eval()
    
    print(f"\n{'='*60}")
    print(f"Starting DQN-MIP Evaluation")
    print(f"{'='*60}")
    print(f"Number of episodes: {n_episodes}")
    print(f"Environment nodes: {env.n}, Budget: {env.budget}")
    print(f"{'='*60}\n")
    
    # Track episode lengths for summary
    episode_lengths = []
    
    # Use tqdm with file=sys.stdout to ensure progress shows in log files
    for ep in tqdm(range(n_episodes), desc="DQN-MIP Episodes", file=sys.stdout, ncols=100):
        # Reset to initial state
        status, mask = env.reset()
        if init_states[ep] is not None:
            env.status = init_states[ep].copy()
            status = env.status
        
        # Determine actual number of nodes from status array (more reliable than env.n)
        actual_n = len(status)
        # Set horizon to ensure we can test all nodes (with budget >= 1, we need at most actual_n steps)
        if horizon is None:
            horizon = actual_n
        else:
            # Ensure horizon is at least as large as number of nodes
            horizon = max(horizon, actual_n)
        
        # Initialize rewards array for this episode (will resize if needed)
        episode_rewards = []
        
        # Print episode start info
        if ep == 0 or (ep + 1) % max(1, n_episodes // 10) == 0:
            print(f"\n[Episode {ep + 1}/{n_episodes}] Starting evaluation...")
            print(f"  Horizon: {horizon}, Initial untested nodes: {np.sum(status == -1)}")
        
        step_count = 0
        # Run until all nodes are tested (done=True) or horizon steps reached
        for t in range(horizon):
            # Check if all nodes are tested by counting untested nodes (status == -1)
            untested_count = np.sum(status == -1)
            if untested_count == 0:
                # All nodes tested, pad remaining steps with zero rewards
                break
            
            # Check frontier availability
            frontier_mask = env.allowed_mask()
            frontier_size = np.sum(frontier_mask)
            
            # Handle case where frontier is empty but nodes remain untested (disconnected components)
            if frontier_size == 0 and untested_count > 0:
                # Find untested nodes (these should be root nodes of disconnected components)
                untested_indices = np.flatnonzero(status == -1)
                # Test untested nodes directly (up to budget)
                action = np.zeros(env.n, dtype=int)
                budget_remaining = min(env.budget if env.budget is not None else len(untested_indices), len(untested_indices))
                if budget_remaining > 0:
                    selected = untested_indices[:budget_remaining]
                    action[selected] = 1
                else:
                    # No budget but nodes remain - should not happen, but break to avoid infinite loop
                    break
            else:
                # Normal case: use MILP to select action from frontier
                # Log MILP solve progress for first episode or periodically
                if (ep == 0 and t < 3) or (ep == 0 and t % 10 == 0):
                    print(f"  [Episode {ep + 1}, Step {t + 1}] Solving MILP (frontier size: {frontier_size}, untested: {untested_count})...")
                
                # Build graph representation and get embedding
                with torch.no_grad():
                    data_s = adapter.build_graph(status)
                    g_s = network.embed_state(data_s).detach().cpu().numpy().astype(np.float32)
                
                # Solve MIP to get action
                results = approximator.approximate(
                    network=network.action_mlp,
                    mipper_cls=mipper,
                    n_scenarios=1,
                    gap=0.02,
                    time_limit=60,
                    threads=4,
                    scenario_embedding=g_s,
                    scenario_probs=None,
                )
                
                action = results["sol"]
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                action = action.astype(int)
                
                # Log action selection for first episode
                if ep == 0 and t < 3:
                    action_size = np.sum(action)
                    print(f"  [Episode {ep + 1}, Step {t + 1}] MILP solved: selected {action_size} nodes")
                
                # Fallback: if action is empty, use random feasible action
                if np.sum(action) == 0:
                    action = env.random_feasible_action().astype(int)
            
            # Execute action
            next_status, next_mask, reward, done = env.step(action)
            episode_rewards.append(reward)
            step_count += 1
            
            # Continue until all nodes tested
            # Check both done flag and actual untested nodes count
            untested_count = np.sum(next_status == -1)
            if done or untested_count == 0:
                if ep == 0 or (ep + 1) % max(1, n_episodes // 10) == 0:
                    cum_reward = sum(episode_rewards)
                    print(f"  [Episode {ep + 1}] Completed in {step_count} steps, cumulative reward: {cum_reward:.2f}")
                break
            
            status = next_status
            mask = next_mask
        
        # Track episode length
        episode_lengths.append(step_count)
        
        # Pad episode rewards to horizon length and add to main rewards array
        if len(episode_rewards) < horizon:
            episode_rewards.extend([0.0] * (horizon - len(episode_rewards)))
        
        # Initialize rewards array on first episode with correct size
        if ep == 0:
            rewards = np.zeros(horizon * n_episodes)
        
        # Store episode rewards
        rewards[ep * horizon:(ep + 1) * horizon] = episode_rewards[:horizon]
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"DQN-MIP Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total episodes: {n_episodes}")
    if episode_lengths:
        print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
        print(f"Min episode length: {np.min(episode_lengths)} steps")
        print(f"Max episode length: {np.max(episode_lengths)} steps")
    print(f"Total rewards collected: {np.sum(rewards):.2f}")
    print(f"Average reward per episode: {np.sum(rewards) / n_episodes:.2f}")
    print(f"{'='*60}\n")
    
    return rewards

