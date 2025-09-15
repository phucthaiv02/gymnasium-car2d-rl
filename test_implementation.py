"""
Test Script for DQN Implementation

This script runs a quick test to ensure all components work correctly.
"""

import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
import torch as th


def test_environment():
    """Test if the CarRacing environment can be created and used."""
    print("Testing CarRacing environment...")
    
    try:
        env = gym.make("CarRacing-v2", render_mode=None, continuous=False)
        obs, info = env.reset()
        
        print(f"  ✓ Environment created successfully")
        print(f"  ✓ Observation shape: {obs.shape}")
        print(f"  ✓ Action space: {env.action_space}")
        
        # Test a few random actions
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        print(f"  ✓ Environment step() works correctly")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        return False


def test_dqn_creation():
    """Test if DQN model can be created."""
    print("\nTesting DQN model creation...")
    
    try:
        env = gym.make("CarRacing-v2", render_mode=None, continuous=False)
        
        # Create a minimal DQN model
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=1000,
            learning_starts=100,
            batch_size=32,
            verbose=0
        )
        
        print(f"  ✓ DQN model created successfully")
        print(f"  ✓ Model policy: {model.policy}")
        print(f"  ✓ Device: {model.device}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ DQN creation test failed: {e}")
        return False


def test_short_training():
    """Test a very short training run."""
    print("\nTesting short training run...")
    
    try:
        env = gym.make("CarRacing-v2", render_mode=None, continuous=False)
        
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=1000,
            learning_starts=100,
            batch_size=32,
            verbose=0
        )
        
        # Train for just 500 steps
        model.learn(total_timesteps=500, progress_bar=False)
        
        print(f"  ✓ Short training completed successfully")
        
        # Test prediction
        obs, info = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"  ✓ Model prediction works: action={action}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Short training test failed: {e}")
        return False


def test_save_load():
    """Test model saving and loading."""
    print("\nTesting model save/load...")
    
    try:
        # Create temp directory
        os.makedirs("/tmp/test_models", exist_ok=True)
        
        env = gym.make("CarRacing-v2", render_mode=None, continuous=False)
        
        # Create and train a minimal model
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=1000,
            learning_starts=100,
            batch_size=32,
            verbose=0
        )
        
        model.learn(total_timesteps=200, progress_bar=False)
        
        # Save model
        model_path = "/tmp/test_models/test_dqn"
        model.save(model_path)
        print(f"  ✓ Model saved to {model_path}")
        
        # Load model
        loaded_model = DQN.load(model_path)
        print(f"  ✓ Model loaded successfully")
        
        # Test loaded model
        obs, info = env.reset()
        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = loaded_model.predict(obs, deterministic=True)
        
        # Actions should be the same
        assert action1 == action2, f"Actions differ: {action1} vs {action2}"
        print(f"  ✓ Loaded model produces same predictions")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Save/load test failed: {e}")
        return False


def test_config_import():
    """Test if configuration module works."""
    print("\nTesting configuration module...")
    
    try:
        from config import get_config, CONFIGS
        
        config = get_config("default")
        print(f"  ✓ Default config loaded")
        print(f"  ✓ Available configs: {list(CONFIGS.keys())}")
        
        # Verify config structure
        required_keys = ["env_config", "dqn_config", "training_config"]
        for key in required_keys:
            assert key in config, f"Missing config key: {key}"
        
        print(f"  ✓ Config structure is valid")
        return True
        
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== DQN Implementation Test Suite ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {th.__version__}")
    print(f"PyTorch device: {th.device('cuda' if th.cuda.is_available() else 'cpu')}")
    print()
    
    tests = [
        test_environment,
        test_dqn_creation,
        test_config_import,
        test_short_training,
        test_save_load
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The DQN implementation is ready for training.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)