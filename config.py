"""
Configuration file for DQN hyperparameters and training settings.

This file contains different configurations for training DQN agents
on the CarRacing-v2 environment.
"""

# Default configuration for initial training
DEFAULT_CONFIG = {
    "env_config": {
        "max_episode_steps": 1000,
        "lap_complete_percent": 0.95,
        "domain_randomize": False,
        "continuous": False
    },
    "dqn_config": {
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "device": "auto",
        "verbose": 1,
        "tensorboard_log": "./tensorboard_logs/"
    },
    "training_config": {
        "total_timesteps": 100000,
        "eval_freq": 10000,
        "n_eval_episodes": 5,
        "reward_threshold": 500
    }
}

# Fast training configuration for testing
FAST_CONFIG = {
    "env_config": {
        "max_episode_steps": 500,
        "lap_complete_percent": 0.95,
        "domain_randomize": False,
        "continuous": False
    },
    "dqn_config": {
        "policy": "CnnPolicy",
        "learning_rate": 5e-4,
        "buffer_size": 10000,
        "learning_starts": 1000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 500,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.1,
        "device": "auto",
        "verbose": 1,
        "tensorboard_log": "./tensorboard_logs/"
    },
    "training_config": {
        "total_timesteps": 20000,
        "eval_freq": 5000,
        "n_eval_episodes": 3,
        "reward_threshold": 300
    }
}

# Extended training configuration for better performance
EXTENDED_CONFIG = {
    "env_config": {
        "max_episode_steps": 1500,
        "lap_complete_percent": 0.95,
        "domain_randomize": True,  # Enable domain randomization
        "continuous": False
    },
    "dqn_config": {
        "policy": "CnnPolicy",
        "learning_rate": 5e-5,  # Lower learning rate
        "buffer_size": 100000,  # Larger buffer
        "learning_starts": 20000,
        "batch_size": 64,  # Larger batch size
        "tau": 1.0,
        "gamma": 0.995,  # Higher discount factor
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 2000,
        "exploration_fraction": 0.3,  # Longer exploration
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,  # Lower final exploration
        "device": "auto",
        "verbose": 1,
        "tensorboard_log": "./tensorboard_logs/"
    },
    "training_config": {
        "total_timesteps": 500000,  # Much longer training
        "eval_freq": 25000,
        "n_eval_episodes": 10,
        "reward_threshold": 700
    }
}

# Configuration mapping
CONFIGS = {
    "default": DEFAULT_CONFIG,
    "fast": FAST_CONFIG,
    "extended": EXTENDED_CONFIG
}


def get_config(config_name="default"):
    """
    Get configuration by name.
    
    Args:
        config_name (str): Name of the configuration
        
    Returns:
        dict: Configuration dictionary
    """
    if config_name not in CONFIGS:
        print(f"Warning: Config '{config_name}' not found. Using 'default' config.")
        config_name = "default"
    
    return CONFIGS[config_name]


def print_config(config_name="default"):
    """Print configuration details."""
    config = get_config(config_name)
    
    print(f"=== Configuration: {config_name.upper()} ===")
    print("\nEnvironment Config:")
    for key, value in config["env_config"].items():
        print(f"  {key}: {value}")
    
    print("\nDQN Config:")
    for key, value in config["dqn_config"].items():
        print(f"  {key}: {value}")
    
    print("\nTraining Config:")
    for key, value in config["training_config"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Print all available configurations
    for config_name in CONFIGS.keys():
        print_config(config_name)
        print("\n" + "="*50 + "\n")