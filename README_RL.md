# CarRacing 2D RL Training

This repository contains a complete implementation of Deep Q-Network (DQN) training for the CarRacing-v2 environment using Gymnasium and Stable-Baselines3.

## Features

- **DQN Implementation**: Deep Q-Network algorithm optimized for discrete action car racing
- **Configurable Training**: Multiple training configurations for different use cases
- **Model Evaluation**: Comprehensive evaluation tools with visual rendering
- **Progress Tracking**: TensorBoard integration for training visualization
- **Model Persistence**: Save and load trained models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/phucthaiv02/gymnasium-car2d-rl.git
cd gymnasium-car2d-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Train a DQN Agent

**Basic Training:**
```bash
python train_dqn.py
```

**Configurable Training:**
```bash
# Fast training (20k timesteps)
python train_configurable.py --config fast

# Default training (100k timesteps)
python train_configurable.py --config default

# Extended training (500k timesteps)
python train_configurable.py --config extended
```

### Evaluate Trained Model

```bash
# Evaluate with visual rendering
python evaluate_dqn.py --model ./models/dqn_car_racing_v1_final

# Evaluate without rendering
python evaluate_dqn.py --model ./models/dqn_car_racing_v1_final --no-render

# Compare multiple models
python evaluate_dqn.py --compare
```

### Test Implementation

```bash
python test_implementation.py
```

## Files Description

### Core Training Scripts
- **`train_dqn.py`**: Basic DQN training script with fixed hyperparameters
- **`train_configurable.py`**: Advanced training script with configurable parameters
- **`config.py`**: Configuration file with different training setups

### Evaluation and Testing
- **`evaluate_dqn.py`**: Evaluation script for trained models
- **`test_implementation.py`**: Test suite to verify implementation
- **`manual_driver.py`**: Manual control script for human play

### Configuration
- **`requirements.txt`**: Python dependencies

## Training Configurations

### Fast Configuration
- **Timesteps**: 20,000
- **Purpose**: Quick testing and development
- **Expected Training Time**: ~5-10 minutes

### Default Configuration
- **Timesteps**: 100,000
- **Purpose**: Standard training for decent performance
- **Expected Training Time**: ~30-60 minutes

### Extended Configuration
- **Timesteps**: 500,000
- **Purpose**: High-performance training with domain randomization
- **Expected Training Time**: ~3-5 hours

## Environment Details

- **Environment**: CarRacing-v2 (Gymnasium)
- **Action Space**: Discrete (5 actions)
  - 0: Do nothing
  - 1: Steer right
  - 2: Steer left
  - 3: Accelerate
  - 4: Brake
- **Observation Space**: RGB images (96x96x3)
- **Reward**: Track progress and lap completion

## Model Architecture

The DQN uses a Convolutional Neural Network (CNN) policy:
- **Feature Extractor**: NatureCNN with 3 convolutional layers
- **Input**: 96x96x3 RGB images
- **Output**: Q-values for 5 discrete actions
- **Hidden Units**: 512 in the final layer

## Training Features

- **Experience Replay**: 50,000 step buffer (default)
- **Target Network**: Hard updates every 1,000 steps
- **Exploration**: ε-greedy with decay from 1.0 to 0.05
- **Evaluation**: Periodic evaluation during training
- **Early Stopping**: Stop when reward threshold is reached
- **TensorBoard Logging**: Training progress visualization

## Monitoring Training

Start TensorBoard to monitor training progress:
```bash
tensorboard --logdir=./tensorboard_logs/
```

Then open http://localhost:6006 in your browser.

## Expected Performance

- **Baseline (Random)**: ~-100 to 0 average reward
- **Trained DQN**: 200-500+ average reward (depending on configuration)
- **Success Criteria**: Completing track laps consistently

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use CPU training by editing device settings in config
2. **Slow training**: Use the "fast" configuration for testing
3. **Poor performance**: Try the "extended" configuration with more timesteps

### Tips for Better Training

1. **Monitor TensorBoard**: Watch for learning progress and stability
2. **Adjust hyperparameters**: Modify learning rate and exploration in config.py
3. **Domain randomization**: Enable in extended config for better generalization
4. **Longer training**: More timesteps generally lead to better performance

## Examples

### Training Example Output
```
=== DQN Training for CarRacing-v2 (Config: default) ===
PyTorch device: cpu
Model name: dqn_car_racing_default

Creating DQN model...
Starting training for 100000 timesteps...
...
Training completed successfully!
Mean reward: 342.5 +/- 45.2
```

### Evaluation Example Output
```
=== DQN Model Evaluation ===
Episode 1/5
  Episode 1 completed: 876 steps, Total reward: 423.45

=== Final Results ===
Mean reward: 398.23 +/- 67.89
Episode rewards: ['423.45', '456.78', '334.12', '398.67', '378.13']
```

## Contributing

Feel free to contribute improvements, additional algorithms, or better configurations!

## License

This project is licensed under the MIT License - see the LICENSE file for details.