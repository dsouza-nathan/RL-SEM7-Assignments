from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from gridworld import MDP, State, Action, sample_next_state_and_reward


@dataclass
class MCTSConfig:
    gamma: float = 0.95
    c_uct: float = 1.4
    rollouts: int = 200
    max_depth: int = 200


class Node:
    def __init__(self, state: State, parent: Optional[Tuple["Node", Action]] = None) -> None:
        self.state = state
        self.parent = parent
        self.children: Dict[Action, Node] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / float(self.visits)


class MCTS:
    def __init__(self, mdp: MDP, cfg: MCTSConfig, rng=None, heuristic=None) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.rng = rng
        self.heuristic = heuristic
        if self.rng is None:
            import random

            self.rng = random.Random(0)

    def search(self, root_state: State) -> Action:
        root = Node(root_state)
        for _ in range(self.cfg.rollouts):
            node = root
            path = []
            depth = 0

            # Selection
            while node.children and not self.mdp.is_terminal(node.state) and depth < self.cfg.max_depth:
                actions = list(node.children.keys())
                total_visits = sum(child.visits for child in node.children.values())
                def uct_score(a, child):
                    if child.visits == 0:
                        return float('inf')
                    return child.q + self.cfg.c_uct * math.sqrt(math.log(total_visits + 1) / (child.visits))
                scores = [(a, uct_score(a, node.children[a])) for a in actions]
                best_a, _ = max(scores, key=lambda x: x[1])
                node = node.children[best_a]
                path.append((node, best_a))
                depth += 1

            # Expansion
            if not self.mdp.is_terminal(node.state):
                untried = [a for a in self.mdp.actions(node.state) if a not in node.children]
                if untried:
                    a = self.rng.choice(untried)
                    next_s, r = sample_next_state_and_reward(self.mdp, node.state, a, self.rng)
                    child = Node(next_s, parent=(node, a))
                    node.children[a] = child
                    node = child
                    path.append((node, a))
                    depth += 1

            # Rollout
            s = node.state
            total_reward = 0.0
            discount = 1.0
            for d in range(self.cfg.max_depth - depth):
                if self.mdp.is_terminal(s):
                    break
                actions = list(self.mdp.actions(s))
                if not actions:
                    break
                a = self.rng.choice(actions)
                s, r = sample_next_state_and_reward(self.mdp, s, a, self.rng)
                total_reward += discount * r
                discount *= self.cfg.gamma

            # Backpropagation
            G = total_reward
            for n, _ in reversed(path):
                n.visits += 1
                n.value_sum += G
                G *= self.cfg.gamma

        # choose action with most visits
        best_a = None
        best_v = -1
        for a, ch in root.children.items():
            if ch.visits > best_v:
                best_v = ch.visits
                best_a = a
        if best_a is None:
            actions = list(self.mdp.actions(root_state))
            if not actions:
                raise RuntimeError("MCTS on terminal state")
            best_a = actions[0]
        return best_a

