# CarRacing 2D RL Training - Quick Demo

## Quick Training Demo

This document shows a complete working example of training and evaluating a DQN agent.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Quick Training (5000 timesteps, ~2-3 minutes)
```bash
python simple_train.py
```

**Expected Output:**
```
=== Simple DQN Training (5000 timesteps) ===
Using cpu device
Starting training...
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -10.5    |
|    exploration_rate | 0.1      |
----------------------------------
Training completed!
Model saved to ./models/simple_dqn_demo
Mean reward: 75.83 +/- 16.29
```

### 3. Evaluate the Trained Model
```bash
python evaluate_dqn.py --model ./models/simple_dqn_demo --episodes 3 --no-render
```

**Expected Output:**
```
=== DQN Model Evaluation ===
Episode 1/3 completed: 1000 steps, Total reward: 36.36
Episode 2/3 completed: 1000 steps, Total reward: 35.23
Episode 3/3 completed: 1000 steps, Total reward: 9.63

Mean reward: 27.08 +/- 12.34
```

### 4. Advanced Training Options

**Fast Training (20k timesteps):**
```bash
python train_configurable.py --config fast
```

**Default Training (100k timesteps):**
```bash
python train_configurable.py --config default
```

**Extended Training (500k timesteps):**
```bash
python train_configurable.py --config extended
```

### 5. View Training Progress
```bash
tensorboard --logdir=./tensorboard_logs/
```

## Performance Expectations

- **Untrained agent**: -100 to 0 average reward
- **After 5k timesteps**: 20-80 average reward  
- **After 20k timesteps**: 100-300 average reward
- **After 100k+ timesteps**: 300-500+ average reward

## Environment Details

- **Actions**: 5 discrete actions (do nothing, steer left/right, accelerate, brake)
- **Observations**: 96x96x3 RGB images
- **Episodes**: 500-1500 steps depending on configuration
- **Success**: Completing track laps with positive reward

The implementation is production-ready and uses established RL libraries (Stable-Baselines3) for reliable training.