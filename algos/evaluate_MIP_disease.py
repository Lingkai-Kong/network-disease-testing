"""
Evaluate the DQN-MIP solver for disease graph environments.
"""
import numpy as np
import torch

from algos.dqn_estimator_disease import GraphEnvAdapter
from approximator.batch_graph_approximator import BatchGraphApproximator


def MIP_results(env, network, mipper, init_states, horizon=None):
    """
    Use NN-embedded MIP to pick action at each timestep.
    
    Args:
        env: Disease environment (BinaryFrontierEnvBatch)
        network: Trained PolicyQNet
        mipper: MIP solver class (Net2MIPPerScenario)
        init_states: List of initial state dictionaries
        horizon: Episode horizon (default: env.n)
        
    Returns:
        Array of rewards per timestep
    """
    print('------------------------------------')
    print('DQN-MIP evaluation (disease)')
    print('------------------------------------')
    
    n_episodes = len(init_states)
    if horizon is None:
        horizon = env.n
    
    adapter = GraphEnvAdapter(env)
    approximator = BatchGraphApproximator(env, model_type="NN-E")
    network.eval()
    
    rewards = np.zeros(horizon * n_episodes)
    
    for ep in range(n_episodes):
        # Reset to initial state
        status, mask = env.reset()
        if init_states[ep] is not None:
            env.status = init_states[ep].copy()
            status = env.status
        
        for t in range(horizon):
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
            
            # Execute action
            next_status, next_mask, reward, done = env.step(action.astype(int))
            rewards[ep * horizon + t] = reward
            
            if done:
                break
            
            status = next_status
            mask = next_mask
    
    return rewards

