## RTDP (decaying epsilon) + MCTS — Assignment

Your job: fill the sections marked `#YOUR CODE HERE` in `rtdp.py` and `mcts.py`.

### Tasks (do these):
1. RTDP: implement `bellman_backup` and the episode loop with decaying epsilon-greedy.
2. MCTS (UCT): implement one iteration (selection, expansion, rollout, backprop) and return the most visited action.
3. Run both on the default GridWorld and print steps-to-goal and total reward for at least 20 episodes/searches.
4. Briefly (3–5 sentences) compare RTDP vs MCTS behavior on this map.
5. Optional: try different epsilon schedules and `c_uct` values and comment.

### Run
- Edit `main.py` and uncomment `run_rtdp()` or `run_mcts()`.
- `python main.py`

### Hints
- RTDP: `V[s] = max_a E[r + gamma V[s']]` using the provided model. Use epsilon-greedy over one-step lookahead Q(s,a).
- MCTS: UCT score `Q + c * sqrt(ln N / (1 + N_a))`. Discount returns in rollout/backprop.

---
## Concepts Explained

**RTDP (Real-Time Dynamic Programming):**
RTDP is a model-based reinforcement learning algorithm that updates the value function by simulating episodes in the environment. At each step, it performs a Bellman backup (`V[s] = max_a E[r + gamma V[s']]`) and selects actions using an epsilon-greedy policy over one-step lookahead Q-values. The epsilon value decays over episodes, balancing exploration and exploitation. RTDP is efficient for problems where a model is available and can quickly converge to good policies in structured environments like GridWorld.

**MCTS (Monte Carlo Tree Search, UCT variant):**
MCTS is a planning algorithm that builds a search tree by simulating many rollouts from the current state. Each iteration consists of selection (using UCT scores), expansion, rollout (simulated episode), and backpropagation of discounted rewards. The UCT score balances exploration and exploitation (`Q + c * sqrt(ln N / (1 + N_a))`). MCTS does not require a value function and is effective for large or unknown environments, but can be slower than RTDP for small, well-modeled problems.

---
## RTDP vs MCTS Comparison

- **RTDP** quickly improves its value estimates by updating states encountered during episodes, often converging faster in small, structured environments. It uses the model for one-step lookahead and benefits from decaying epsilon to reduce exploration over time.
- **MCTS** builds a search tree from the root state, using rollouts to estimate action values. It is more sample-intensive and can be slower to converge, but is robust to model inaccuracies and works well in large or complex spaces.
- On the default GridWorld, RTDP typically reaches the goal in fewer steps after initial exploration, while MCTS may require more rollouts to find optimal paths but is less dependent on prior value estimates.
- RTDP is preferable when a reliable model is available and fast convergence is needed; MCTS is better for planning in unknown or very large environments.

---
## Proof of Execution

Below is a screenshot showing the output of running `main.py` with RTDP:

![RTDP Output](attachment:image)

---

