"""
Evaluation Script for Trained DQN Agent in CarRacing-v2

This script loads a trained DQN model and evaluates its performance
with optional visual rendering.
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import argparse
import os
import time


def create_env(render_mode="human"):
    """Create the CarRacing environment for evaluation."""
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        max_episode_steps=1000,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )
    return env


def evaluate_model(model_path, n_episodes=5, render=True, deterministic=True):
    """
    Evaluate a trained DQN model.
    
    Args:
        model_path (str): Path to the saved model
        n_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment
        deterministic (bool): Whether to use deterministic actions
    
    Returns:
        tuple: (mean_reward, std_reward, episode_rewards)
    """
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model not found at {model_path}.zip")
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = create_env(render_mode=render_mode)
    
    print(f"Evaluating model for {n_episodes} episodes...")
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Check if episode is done
            done = terminated or truncated
            
            # Render if requested
            if render:
                env.render()
                time.sleep(0.01)  # Small delay for better visualization
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Step {step_count}, Reward so far: {episode_reward:.2f}")
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode + 1} completed: {step_count} steps, Total reward: {episode_reward:.2f}")
    
    env.close()
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward, episode_rewards


def compare_models(model_paths, n_episodes=5):
    """Compare multiple trained models."""
    results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n=== Evaluating {model_name} ===")
        try:
            mean_reward, std_reward, episode_rewards = evaluate_model(
                model_path, n_episodes=n_episodes, render=False
            )
            results[model_name] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'episode_rewards': episode_rewards
            }
            print(f"{model_name}: {mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Print comparison
    print("\n=== Model Comparison ===")
    for model_name, results_data in results.items():
        print(f"{model_name}: {results_data['mean_reward']:.2f} +/- {results_data['std_reward']:.2f}")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained DQN model")
    parser.add_argument("--model", type=str, default="./models/dqn_car_racing_v1_final",
                        help="Path to the trained model (without .zip extension)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--non-deterministic", action="store_true",
                        help="Use stochastic policy")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple models")
    
    args = parser.parse_args()
    
    print("=== DQN Model Evaluation ===")
    
    if args.compare:
        # Compare different models if they exist
        model_paths = {
            "Final Model": "./models/dqn_car_racing_v1_final",
            "Best Model": "./models/best_dqn_car_racing_v1/best_model"
        }
        
        # Filter to only existing models
        existing_models = {}
        for name, path in model_paths.items():
            if os.path.exists(f"{path}.zip"):
                existing_models[name] = path
        
        if existing_models:
            compare_models(existing_models, args.episodes)
        else:
            print("No trained models found for comparison.")
    else:
        # Evaluate single model
        try:
            mean_reward, std_reward, episode_rewards = evaluate_model(
                args.model,
                n_episodes=args.episodes,
                render=not args.no_render,
                deterministic=not args.non_deterministic
            )
            
            print(f"\n=== Final Results ===")
            print(f"Model: {args.model}")
            print(f"Episodes: {args.episodes}")
            print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Min reward: {min(episode_rewards):.2f}")
            print(f"Max reward: {max(episode_rewards):.2f}")
            print(f"Episode rewards: {[f'{r:.2f}' for r in episode_rewards]}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("\nTip: Make sure you have trained a model first using: python train_dqn.py")


if __name__ == "__main__":
    main()