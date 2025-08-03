# LAB 2: Multi-Armed Bandit Algorithms

## Overview
Implementation and comparison of fundamental multi-armed bandit algorithms. Explores the exploration vs exploitation tradeoff through both algorithmic implementation and interactive demonstration.

## Structure
```
LAB 2/
├── Assignment/src/              # Main framework
│   ├── algorithms/              # Algorithm implementations
│   ├── environment/             # Bandit environment
│   ├── experiments/             # Comparison tools
│   └── main.py                 # Main execution
└── Bandit_Demo/                # Interactive Streamlit demo
```

## Algorithms

### 1. Exploration Only
- **Strategy**: Pure random selection
- **Performance**: High regret (no learning)

### 2. Exploitation Only  
- **Strategy**: Always choose current best arm
- **Performance**: High regret (no exploration)

### 3. Epsilon-Greedy
- **Strategy**: Explore with probability ε, exploit otherwise
- **Performance**: Best empirical results (ε = 0.1)

### 4. Upper Confidence Bound (UCB)
- **Strategy**: Optimistic selection using confidence bounds
- **Formula**: UCB = estimate + c × √(log(total_pulls) / arm_pulls)
- **Performance**: Good with theoretical guarantees

## Key Concepts
- **Multi-Armed Bandit**: Sequential decision making under uncertainty
- **Exploration vs Exploitation**: Core RL tradeoff
- **Regret**: Cumulative loss vs optimal strategy
- **Confidence Bounds**: Statistical uncertainty handling

## How to Run

### Main Framework
```bash
cd "LAB 2/Assignment/src"
python main.py
```
**Output**: Algorithm comparison, performance metrics, visualization plot

### Interactive Demo
```bash
cd "LAB 2/Bandit_Demo"
pip install -r requirements.txt
streamlit run multi_armed_bandit_demo.py
```
**Output**: 20-pull bandit game with real-time statistics

## Performance Results
Based on 1000 trials, 5 arms:

| Algorithm | Regret | Avg Reward | Rank |
|-----------|--------|------------|------|
| Epsilon-Greedy | 97.69 | 0.850 | 1st |
| UCB | 110.59 | 0.780 | 2nd |
| Exploration Only | 308.41 | 0.500 | 3rd |
| Exploitation Only | 584.70 | 0.300 | 4th |

## Learning Outcomes
- Understand bandit problem formulation
- Implement exploration-exploitation algorithms
- Compare algorithm performance empirically
- Learn modular code design patterns
- Visualize algorithm behavior

## Requirements
- **Core**: Python 3.7+, NumPy, Matplotlib
- **Demo**: Streamlit, Plotly, Pandas
