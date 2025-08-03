# LAB 1: Gymnasium Environment Exploration

## Overview
Introduction to reinforcement learning environments using OpenAI Gymnasium. Explores two different control environments to understand observation spaces, action spaces, and environment interaction patterns.

## File
- `Assignment_1.ipynb` - Interactive environment demonstrations

## Environments

### LunarLander-v3
- **Observation**: 8D vector (position, velocity, angle, leg contact)
- **Actions**: 4 discrete (do nothing, fire left/main/right engine)
- **Goal**: Land softly on the landing pad
- **Type**: Discrete control with continuous observations

### CarRacing-v3
- **Observation**: 96×96×3 RGB image (top-down track view)
- **Actions**: 3D continuous (steering, gas, brake)
- **Goal**: Drive as far as possible around the track
- **Type**: Continuous control with image observations

## Key Concepts
- Environment interface (`gym.make()`, `env.reset()`, `env.step()`)
- Observation vs action spaces
- Discrete vs continuous control
- Episode management (reset, termination)
- Random policy baseline

## How to Run
```bash
pip install gymnasium jupyter
cd "LAB 1"
jupyter notebook Assignment_1.ipynb
```

## Learning Outcomes
- Understand RL environment structure
- Compare discrete vs continuous control
- Learn observation and action space concepts
- Practice environment interaction patterns
