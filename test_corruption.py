## Random data corruption script, taken from Yang et. Al's  implementation

import argparse
import d4rl  # must come before gym.make
import gym
import h5py
import numpy as np
import torch
from attack import attack_dataset
from RIQL_config import get_config
from typing import Optional


class TrainConfig:
    # Experiment
    eval_episodes: int = 10
    eval_every: int = 10
    log_every: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    eval_freq: int = int(1e4)
    n_episodes: int = 10
    max_timesteps: int = int(3e6)
    checkpoints_path: Optional[str] = None
    load_model: str = ""

    # IQL
    buffer_size: int = 2_000_000
    batch_size: int = 256
    discount: float = 0.99
    soft_target_update_rate: float = 0.005
    learning_rate: float = 3e-4
    beta: float = 3.0
    iql_tau: float = 0.7
    iql_deterministic: bool = True
    normalize: bool = True

    # Wandb logging
    project: str = "Robust_Offline_RL"
    group: str = "RIQL"
    name: str = "RIQL"
    env_name: str = 'walker2d-medium-replay-v2'
    train_seed: int = 0
    eval_seed: int = 42
    flag: str = 'test'
    sigma: float = 1.0
    num_actors: int = 1
    num_critics: int = 5
    quantile: float = 0.25

    # Corruption
    corrupt_reward: bool = False
    corrupt_dynamics: bool = False
    corrupt_obs: bool = False
    corrupt_acts: bool = False
    corruption_mode: str = 'random'
    corruption_range: float = 1.0
    corruption_rate: float = 0.3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, required=True, help='Name of the environment')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the corrupted dataset')
    parser.add_argument('--corrupt_reward', action='store_true', help='Whether to corrupt rewards')
    parser.add_argument('--corrupt_dynamics', action='store_true', help='Whether to corrupt dynamics')
    parser.add_argument('--corrupt_obs', action='store_true', help='Whether to corrupt observations')
    parser.add_argument('--corrupt_acts', action='store_true', help='Whether to corrupt actions')

    args = parser.parse_args()

    config = TrainConfig()
    config.env_name = args.env_name
    config.corrupt_reward = args.corrupt_reward
    config.corrupt_dynamics = args.corrupt_dynamics
    config.corrupt_obs = args.corrupt_obs
    config.corrupt_acts = args.corrupt_acts

    print(f"Environment {config.env_name} found:", config.env_name in gym.envs.registry.env_specs)

    env = gym.make(config.env_name)
    dataset = env.get_dataset()
    # dataset = d4rl.qlearning_dataset(env) 
    print(dataset.keys())

    print("Observations:", dataset['observations'].shape)
    print("Actions:", dataset['actions'].shape)

    dataset, indexes = attack_dataset(config, dataset)

    with h5py.File(args.save_path, 'w') as f:
        for key, value in dataset.items():
            f.create_dataset(key, data=value)
