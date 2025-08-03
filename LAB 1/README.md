# LAB 1: Gymnasium Environment Exploration

## Overview
Introduction to reinforcement learning environments using OpenAI Gymnasium. This lab demonstrates how to interact with RL environments and understand their components through hands-on exploration of two different types of control environments.

## Files
- `Assignment_1.ipynb` - Jupyter notebook with interactive environment demonstrations

## Environments Explored

### 1. LunarLander-v3
**Environment Type**: Discrete control with continuous observations

**Observation Space**: 8-dimensional continuous vector
- Elements 0-1: Lander position (x, y coordinates)
- Elements 2-3: Lander velocity (x, y components)
- Element 4: Lander angle (in radians, ±π)
- Element 5: Lander angular velocity
- Elements 6-7: Boolean leg contact sensors (left leg, right leg)

**Action Space**: 4 discrete actions
- Action 0: Do nothing
- Action 1: Fire left engine
- Action 2: Fire main engine
- Action 3: Fire right engine

**Objective**: Land the lunar module softly on the landing pad between the flags

**Challenges**:
- Balancing thrust control
- Managing landing speed and angle
- Avoiding crashes
- Fuel efficiency

### 2. CarRacing-v3
**Environment Type**: Continuous control with image observations

**Observation Space**: 96×96×3 RGB image
- Pixel values range from 0-255
- Represents the car's view of the track
- Top-down perspective

**Action Space**: 3-dimensional continuous control
- Action 0: Steering [-1, 1] (left/right)
- Action 1: Gas [0, 1] (throttle)
- Action 2: Brake [0, 1] (braking force)

**Objective**: Drive as far as possible around a procedurally generated track

**Challenges**:
- Track following with visual input
- Curve handling at appropriate speeds
- Speed control and optimization
- Staying within track boundaries

## Key Concepts Demonstrated

### Environment Interface
- **Initialization**: Creating environment instances with `gym.make()`
- **Reset**: Starting new episodes with `env.reset()`
- **Step**: Taking actions with `env.step(action)`
- **Observation**: Understanding what the agent perceives
- **Reward**: Learning from environment feedback
- **Termination**: Recognizing episode endings

### Control Types
- **Discrete Control**: Fixed set of actions (LunarLander)
- **Continuous Control**: Real-valued action parameters (CarRacing)
- **Action Sampling**: Random policy with `env.action_space.sample()`

### Observation Types
- **Vector Observations**: Numerical state representations
- **Image Observations**: Visual perception tasks
- **Space Bounds**: Understanding observation and action limits

## How to Run

### Prerequisites
```bash
pip install gymnasium jupyter
```

### Execution
```bash
cd "LAB 1"
jupyter notebook Assignment_1.ipynb
```

### Usage
1. Open the Jupyter notebook
2. Run cells sequentially to see environment demonstrations
3. Observe how random actions perform in each environment
4. Examine observation and action space properties

## Learning Outcomes

After completing this lab, you should understand:

1. **Environment Structure**: How RL environments are organized and accessed
2. **Observation Spaces**: Different types of state representations
3. **Action Spaces**: Discrete vs continuous action selection
4. **Episode Flow**: Reset, step, termination cycle
5. **Random Policies**: Baseline behavior for comparison
6. **Environment Diversity**: How different domains require different approaches

## Technical Notes

### LunarLander-v3
- Physics-based simulation
- Reward structure encourages smooth landings
- Episode length varies based on performance
- Deterministic dynamics with stochastic initialization

### CarRacing-v3
- Procedurally generated tracks
- Image-based observations require computer vision
- Continuous action space demands precise control
- Performance measured by track completion

## Next Steps

This lab provides the foundation for understanding RL environments. Future work could include:
- Implementing simple heuristic policies
- Comparing random vs structured approaches
- Analyzing observation patterns
- Understanding reward structures
- Exploring other Gymnasium environments

## Requirements
- Python 3.7+
- Gymnasium
- Jupyter Notebook
- NumPy (included with Gymnasium)
