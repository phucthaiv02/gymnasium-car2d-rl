"""
Simple DQN Training Script for CarRacing-v2

A minimal training script without complex callbacks for demonstration.
"""

import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th


def simple_train_dqn(timesteps=10000, model_name="simple_dqn"):
    """
    Simple DQN training function.
    
    Args:
        timesteps (int): Number of timesteps to train
        model_name (str): Name for the saved model
    """
    
    print(f"=== Simple DQN Training ({timesteps} timesteps) ===")
    
    # Create environment
    env = gym.make(
        "CarRacing-v2",
        render_mode=None,
        max_episode_steps=500,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )
    
    # Create DQN model with simple configuration
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        verbose=1,
        device="auto"
    )
    
    print(f"Device: {model.device}")
    print("Starting training...")
    
    # Train without complex callbacks
    model.learn(
        total_timesteps=timesteps,
        progress_bar=False  # Disable progress bar to avoid issues
    )
    
    print("Training completed!")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = f"./models/{model_name}"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Quick evaluation
    print("Evaluating model...")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=3, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    return model


def main():
    """Main function."""
    print(f"PyTorch device: {th.device('cuda' if th.cuda.is_available() else 'cpu')}")
    
    # Train a simple model
    model = simple_train_dqn(timesteps=5000, model_name="simple_dqn_demo")
    
    print("\nTraining completed successfully!")
    print("Model saved as: simple_dqn_demo")
    print("To evaluate: python evaluate_dqn.py --model ./models/simple_dqn_demo")


if __name__ == "__main__":
    main()