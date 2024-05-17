import os
import argparse
import pickle

import core.environment.env_factory as environment
from core.utils import torch_utils, logger, run_funcs
from core.agent.in_sample import *

def load_data(dataset_name):
    base_path = "core/"
    dataset_suffix = {'expert': 'expert', 'random': 'random', 'missing': 'missing', 'mixed': 'mixed'}
    dataset_file = f"complete_data_{dataset_suffix[dataset_name]}.pkl"
    file_path = os.path.join(base_path, dataset_file)
    
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    for key, value in dataset.items():
        if hasattr(value, 'shape'):
            print(f"Loaded data - {key} shape: {value.shape}")
        else:
            print(f"Loaded data - {key} is not an array")

    return dataset


def run_experiment(learning_rate, seed, dataset_name='expert'):
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--env_name', default='grid', type=str)
    parser.add_argument('--dataset', default=dataset_name, type=str, choices=['expert', 'random', 'missing', 'mixed'])
    parser.add_argument('--discrete_control', default=1, type=int)
    parser.add_argument('--state_dim', default=2, type=int)
    parser.add_argument('--action_dim', default=4, type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--max_steps', default=10000, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--learning_rate', default=learning_rate, type=float)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--timeout', default=100, type=int)
    parser.add_argument('--gamma', default=0.90, type=float)
    parser.add_argument('--use_target_network', default=1, type=int)
    parser.add_argument('--target_network_update_freq', default=1, type=int)
    parser.add_argument('--polyak', default=0.995, type=float)
    parser.add_argument('--evaluation_criteria', default='return', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--info', default='0', type=str)
    cfg = parser.parse_args()

    torch_utils.set_one_thread()
    torch_utils.random_seed(cfg.seed)

    project_root = os.path.abspath(os.path.dirname(__file__))
    exp_path = f"data/output/{cfg.env_name}/{cfg.dataset}/{cfg.info}/{cfg.seed}_run"
    cfg.exp_path = os.path.join(project_root, exp_path)
    torch_utils.ensure_dir(cfg.exp_path)

    cfg.offline_data = load_data(cfg.dataset)
    cfg.env_fn = environment.EnvFactory.create_env_fn(cfg)

    cfg.logger = logger.Logger(cfg, cfg.exp_path)
    try:
        logger.log_config(cfg)
    except KeyError as e:
        print(f"KeyError encountered in logger configuration: {e}")

    print("Initializing agent...")
    agent_obj = InSampleAC(
        device=cfg.device,
        discrete_control=cfg.discrete_control,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_units=cfg.hidden_units,
        learning_rate=cfg.learning_rate,
        tau=cfg.tau,
        polyak=cfg.polyak,
        exp_path=cfg.exp_path,
        seed=cfg.seed,
        env_fn=cfg.env_fn,
        timeout=cfg.timeout,
        gamma=cfg.gamma,
        offline_data=cfg.offline_data,
        batch_size=cfg.batch_size,
        use_target_network=cfg.use_target_network,
        target_network_update_freq=cfg.target_network_update_freq,
        evaluation_criteria=cfg.evaluation_criteria,
        logger=cfg.logger
    )

    print("Agent initialized.")
    run_funcs.run_steps(agent_obj, cfg.max_steps, cfg.log_interval, exp_path)
    return agent_obj.episode_rewards
