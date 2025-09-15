"""
DQN Training Script for CarRacing-v2 Environment

This script implements Deep Q-Network (DQN) training for the CarRacing-v2 environment
using Stable-Baselines3. The car racing environment uses discrete actions.
"""

import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch as th
import numpy as np


def create_env():
    """Create the CarRacing environment with appropriate settings."""
    env = gym.make(
        "CarRacing-v2",
        render_mode=None,  # No rendering during training for speed
        max_episode_steps=1000,  # Reasonable episode length
        lap_complete_percent=0.95,  # Complete 95% of track to finish
        domain_randomize=False,  # Keep consistent for initial training
        continuous=False  # Use discrete actions for DQN
    )
    return env


def train_dqn_agent(total_timesteps=100000, model_name="dqn_car_racing"):
    """
    Train a DQN agent on the CarRacing environment.
    
    Args:
        total_timesteps (int): Total number of timesteps to train
        model_name (str): Name to save the model as
    
    Returns:
        DQN: Trained DQN model
    """
    
    # Create environment
    env = create_env()
    
    # Create evaluation environment
    eval_env = create_env()
    
    # Define DQN hyperparameters optimized for CarRacing
    dqn_config = {
        "policy": "CnnPolicy",  # CNN policy for image observations
        "learning_rate": 1e-4,  # Conservative learning rate
        "buffer_size": 50000,   # Experience replay buffer size
        "learning_starts": 10000,  # Start learning after this many steps
        "batch_size": 32,       # Batch size for training
        "tau": 1.0,             # Hard update for target network
        "gamma": 0.99,          # Discount factor
        "train_freq": 4,        # Train every 4 steps
        "gradient_steps": 1,    # Number of gradient steps per train
        "target_update_interval": 1000,  # Update target network every 1000 steps
        "exploration_fraction": 0.1,  # Fraction of training for exploration
        "exploration_initial_eps": 1.0,  # Initial exploration rate
        "exploration_final_eps": 0.05,   # Final exploration rate
        "device": "auto",       # Use GPU if available
        "verbose": 1,           # Print training progress
        "tensorboard_log": "./tensorboard_logs/"  # Tensorboard logging
    }
    
    # Create DQN model
    print("Creating DQN model...")
    model = DQN(env=env, **dqn_config)
    
    # Create callback for evaluation during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/best_{model_name}",
        log_path="./logs/",
        eval_freq=10000,  # Evaluate every 10k steps
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Create callback to stop training when reward threshold is reached
    reward_threshold_callback = StopTrainingOnRewardThreshold(
        reward_threshold=500,  # Stop when average reward reaches 500
        verbose=1
    )
    
    # The reward threshold callback needs to be a child of eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/best_{model_name}",
        log_path="./logs/",
        eval_freq=10000,  # Evaluate every 10k steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        callback_on_new_best=reward_threshold_callback
    )
    
    # Use only the eval callback (which includes the threshold callback)
    callbacks = [eval_callback]
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Device: {model.device}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model_path = f"./models/{model_name}_final"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate the trained model
    print("\nEvaluating trained model...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Clean up
    env.close()
    eval_env.close()
    
    return model


def main():
    """Main training function."""
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    print("=== DQN Training for CarRacing-v2 ===")
    print(f"PyTorch device: {th.device('cuda' if th.cuda.is_available() else 'cpu')}")
    
    # Train the agent
    model = train_dqn_agent(
        total_timesteps=100000,  # Start with 100k timesteps
        model_name="dqn_car_racing_v1"
    )
    
    print("\nTraining completed successfully!")
    print("To view training progress, run: tensorboard --logdir=./tensorboard_logs/")
    print("To test the trained model, run: python evaluate_dqn.py")


if __name__ == "__main__":
    main()