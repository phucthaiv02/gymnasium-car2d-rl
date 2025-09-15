# gymnasium-car2d-rl
Training reinforcement learning agents to control a car in the Car-2D environment using Gymnasium.

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train DQN Agent
```bash
python train_dqn.py
```

### Evaluate Trained Model
```bash
python evaluate_dqn.py --model ./models/dqn_car_racing_v1_final
```

### Manual Control
```bash
python manual_driver.py
```

See [README_RL.md](README_RL.md) for detailed documentation and advanced usage.
