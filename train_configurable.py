"""
Configurable DQN Training Script for CarRacing-v2

This script uses configuration files to train DQN agents with different
hyperparameters and settings.
"""

import os
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch as th
from config import get_config, print_config, CONFIGS


def create_env(env_config):
    """Create the CarRacing environment with specified configuration."""
    env = gym.make(
        "CarRacing-v2",
        render_mode=None,
        **env_config
    )
    return env


def train_dqn_agent(config, model_name="dqn_car_racing"):
    """
    Train a DQN agent using the provided configuration.
    
    Args:
        config (dict): Configuration dictionary
        model_name (str): Name to save the model as
    
    Returns:
        DQN: Trained DQN model
    """
    
    # Extract configurations
    env_config = config["env_config"]
    dqn_config = config["dqn_config"]
    training_config = config["training_config"]
    
    # Create environment
    env = create_env(env_config)
    
    # Create evaluation environment
    eval_env = create_env(env_config)
    
    # Create DQN model
    print("Creating DQN model...")
    model = DQN(env=env, **dqn_config)
    
    # Create callback for evaluation during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/best_{model_name}",
        log_path="./logs/",
        eval_freq=training_config["eval_freq"],
        deterministic=True,
        render=False,
        n_eval_episodes=training_config["n_eval_episodes"]
    )
    
    # Create callback to stop training when reward threshold is reached
    reward_threshold_callback = StopTrainingOnRewardThreshold(
        reward_threshold=training_config["reward_threshold"],
        verbose=1
    )
    
    # The reward threshold callback needs to be a child of eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/best_{model_name}",
        log_path="./logs/",
        eval_freq=training_config["eval_freq"],
        deterministic=True,
        render=False,
        n_eval_episodes=training_config["n_eval_episodes"],
        callback_on_new_best=reward_threshold_callback
    )
    
    # Use only the eval callback (which includes the threshold callback)
    callbacks = [eval_callback]
    
    print(f"Starting training for {training_config['total_timesteps']} timesteps...")
    print(f"Device: {model.device}")
    
    # Train the model
    model.learn(
        total_timesteps=training_config["total_timesteps"],
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
        n_eval_episodes=training_config["n_eval_episodes"] * 2,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Clean up
    env.close()
    eval_env.close()
    
    return model


def main():
    """Main training function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN model with configurable parameters")
    parser.add_argument("--config", type=str, default="default",
                        choices=list(CONFIGS.keys()),
                        help="Configuration to use for training")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Name for the saved model")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps for training")
    parser.add_argument("--show-config", action="store_true",
                        help="Show configuration and exit")
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        print_config(args.config)
        return
    
    # Get configuration
    config = get_config(args.config)
    
    # Override timesteps if provided
    if args.timesteps is not None:
        config["training_config"]["total_timesteps"] = args.timesteps
        print(f"Overriding timesteps to: {args.timesteps}")
    
    # Set model name
    model_name = args.model_name or f"dqn_car_racing_{args.config}"
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    print(f"=== DQN Training for CarRacing-v2 (Config: {args.config}) ===")
    print(f"PyTorch device: {th.device('cuda' if th.cuda.is_available() else 'cpu')}")
    print(f"Model name: {model_name}")
    
    # Print current configuration
    print("\nCurrent Configuration:")
    print_config(args.config)
    
    # Train the agent
    model = train_dqn_agent(config, model_name)
    
    print("\nTraining completed successfully!")
    print(f"Model saved as: {model_name}")
    print("To view training progress, run: tensorboard --logdir=./tensorboard_logs/")
    print(f"To test the trained model, run: python evaluate_dqn.py --model ./models/{model_name}_final")


if __name__ == "__main__":
    main()