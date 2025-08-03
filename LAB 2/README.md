# LAB 2: Multi-Armed Bandit Algorithms

## Overview
Implementation and comparison of fundamental multi-armed bandit algorithms. This lab explores the exploration vs exploitation tradeoff, which is central to reinforcement learning, through both algorithmic implementation and interactive demonstration.

## Project Structure

```
LAB 2/
├── Assignment/                      # Main implementation framework
│   └── src/
│       ├── algorithms/              # Bandit algorithm implementations
│       │   ├── base_algorithm.py    # Abstract base class
│       │   ├── exploration_only.py  # Pure exploration strategy
│       │   ├── exploitation_only.py # Pure exploitation strategy
│       │   ├── epsilon_greedy.py    # Epsilon-greedy algorithm
│       │   └── ucb.py              # Upper Confidence Bound
│       ├── environment/             # Bandit environment simulation
│       │   └── mab_environment.py   # Multi-armed bandit environment
│       ├── experiments/             # Experiment framework
│       │   └── experiment_runner.py # Algorithm comparison tools
│       ├── utils/                   # Configuration utilities
│       │   └── config.py           # Experiment configuration
│       └── main.py                 # Main execution script
└── Bandit_Demo/                    # Interactive demonstration
    ├── multi_armed_bandit_demo.py  # Streamlit web application
    └── requirements.txt            # Demo dependencies
```

## Algorithms Implemented

### 1. Exploration Only
**Strategy**: Pure exploration through uniform random selection

**Implementation**:
```python
def select_arm(self):
    return np.random.randint(0, self.n_arms)
```

**Characteristics**:
- No learning from rewards
- Uniform exploration across all arms
- High regret due to no exploitation
- Serves as exploration baseline

### 2. Exploitation Only
**Strategy**: Pure exploitation of current best estimate

**Implementation**:
```python
def select_arm(self):
    if np.sum(self.pulls) == 0:
        return 0  # Initial choice
    return np.argmax(self.estimates)
```

**Characteristics**:
- Greedy selection based on current estimates
- No exploration after initial trials
- High regret due to suboptimal arm lock-in
- Serves as exploitation baseline

### 3. Epsilon-Greedy
**Strategy**: Balance exploration and exploitation with probability epsilon

**Implementation**:
```python
def select_arm(self):
    if np.random.random() < self.epsilon:
        return np.random.randint(0, self.n_arms)  # Explore
    if np.sum(self.pulls) == 0:
        return 0  # Initial choice
    return np.argmax(self.estimates)  # Exploit
```

**Parameters**:
- `epsilon`: Exploration probability (typically 0.1)

**Characteristics**:
- Simple and effective balance
- Constant exploration rate
- Good practical performance
- Easy to understand and implement

### 4. Upper Confidence Bound (UCB)
**Strategy**: Optimistic exploration using confidence bounds

**Implementation**:
```python
def select_arm(self):
    # Handle unpulled arms first
    unpulled_arms = np.where(self.pulls == 0)[0]
    if len(unpulled_arms) > 0:
        return int(unpulled_arms[0])
    
    # Calculate UCB values
    total_pulls = np.sum(self.pulls)
    ucb_values = self.estimates + self.c * np.sqrt(np.log(total_pulls) / self.pulls)
    return int(np.argmax(ucb_values))
```

**Formula**: UCB = estimate + c × √(log(total_pulls) / arm_pulls)

**Parameters**:
- `c`: Exploration parameter (typically 2.0)

**Characteristics**:
- Theoretical performance guarantees
- Adaptive exploration (decreases over time)
- Optimistic in face of uncertainty
- Principled approach to exploration

## Key Concepts

### Multi-Armed Bandit Problem
Sequential decision making under uncertainty where an agent must:
- Choose actions (arms) repeatedly
- Receive stochastic rewards
- Learn which actions are best
- Maximize cumulative reward

### Exploration vs Exploitation Tradeoff
The fundamental challenge of balancing:
- **Exploration**: Trying new actions to gather information
- **Exploitation**: Using current knowledge to maximize reward
- Too much exploration: Miss opportunities to use good actions
- Too much exploitation: Miss discovering better actions

### Regret
Cumulative loss compared to optimal strategy:
- **Instantaneous Regret**: Difference between optimal and chosen arm reward
- **Cumulative Regret**: Sum of all instantaneous regrets
- Lower regret indicates better performance

### Confidence Bounds
Statistical approach to handle uncertainty:
- Upper bounds on arm values based on observations
- Wider bounds for less-observed arms
- Drives exploration toward uncertain options

## Framework Components

### Environment (`mab_environment.py`)
- Simulates bandit arms with configurable reward distributions
- Supports Bernoulli, Normal, and Uniform distributions
- Tracks true expected rewards for regret calculation
- Provides pull() method for arm selection

### Base Algorithm (`base_algorithm.py`)
- Abstract interface for all bandit algorithms
- Maintains arm statistics (pulls, rewards, estimates)
- Provides update() method for learning from rewards
- Ensures consistent algorithm interface

### Experiment Runner (`experiment_runner.py`)
- Automated comparison of multiple algorithms
- Configurable number of trials and arms
- Performance metrics calculation
- Visualization and summary generation

### Configuration (`config.py`)
- Centralized experiment configuration
- Algorithm parameter management
- Environment setup utilities
- Supports different reward distributions

## How to Run

### Main Framework
```bash
cd "LAB 2/Assignment/src"
python main.py
```

**Output**:
- Algorithm implementation status
- Performance comparison over 1000 trials
- Summary statistics (regret, average reward, arm pulls)
- Visualization plot saved as `mab_comparison.png`

### Interactive Demo
```bash
cd "LAB 2/Bandit_Demo"
pip install -r requirements.txt
streamlit run multi_armed_bandit_demo.py
```

**Features**:
- 20-pull bandit game with 5 arms
- Real-time statistics and progress tracking
- True vs observed probability comparison
- Educational insights about exploration strategies

## Performance Results

Based on empirical evaluation (1000 trials, 5 arms):

| Algorithm | Final Regret | Average Reward | Performance Rank |
|-----------|-------------|----------------|------------------|
| Epsilon-Greedy | 97.69 | 0.850 | 1st (Best) |
| UCB | 110.59 | 0.780 | 2nd |
| Exploration Only | 308.41 | 0.500 | 3rd |
| Exploitation Only | 584.70 | 0.300 | 4th (Worst) |

### Analysis
- **Epsilon-Greedy** achieves best empirical performance with simple implementation
- **UCB** provides good performance with theoretical guarantees
- **Pure strategies** (exploration/exploitation only) perform poorly
- **Balance** between exploration and exploitation is crucial

## Learning Objectives

After completing this lab, you should understand:

1. **Bandit Problem Formulation**: Sequential decision making under uncertainty
2. **Algorithm Design**: Different approaches to exploration-exploitation balance
3. **Performance Evaluation**: Regret analysis and empirical comparison
4. **Implementation Patterns**: Modular design and consistent interfaces
5. **Statistical Concepts**: Confidence bounds and uncertainty quantification

## Technical Requirements

### Core Dependencies
- Python 3.7+
- NumPy (numerical computation)
- Matplotlib (visualization)

### Demo Dependencies
- Streamlit (web interface)
- Plotly (interactive plots)
- Pandas (data manipulation)

## Extensions and Future Work

### Algorithm Extensions
- Thompson Sampling (Bayesian approach)
- Gradient Bandit (policy gradient method)
- Contextual Bandits (state-dependent rewards)
- Non-stationary Bandits (changing reward distributions)

### Framework Extensions
- Different reward distributions
- Batch experiment running
- Statistical significance testing
- Advanced visualization options

### Practical Applications
- A/B testing optimization
- Recommendation systems
- Online advertising
- Resource allocation

## Key Takeaways

1. **No Single Best Algorithm**: Performance depends on problem characteristics
2. **Exploration is Essential**: Pure exploitation leads to poor long-term performance
3. **Simple Methods Work**: Epsilon-greedy often outperforms complex alternatives
4. **Theory vs Practice**: UCB has guarantees but may not always win empirically
5. **Visualization Helps**: Plots reveal algorithm behavior patterns

This lab provides the foundation for understanding exploration-exploitation tradeoffs that appear throughout reinforcement learning.
