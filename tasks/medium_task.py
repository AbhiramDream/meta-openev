"""
medium_task.py — Multi-product warehouse task with varying demand patterns.

Difficulty: Medium
  - 3 products with different demand profiles
  - Moderate demand variance
  - Occasional minor supply disruptions
  - 75 episode steps
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Callable, Dict, List, Optional, Tuple
from server.environment import SupplyChainEnv
from src.models import SupplyAction, ActionType


# ──────────────────────────────────────────────
# Task definition
# ──────────────────────────────────────────────

TASK_NAME = "medium"
MAX_POSSIBLE_REWARD = 100.0   # theoretical upper bound over 75 steps


def run_episode(
    agent_fn: Callable[[dict], int],
    seed: int = 0,
) -> Tuple[float, Dict]:
    """
    Run a full medium-task episode using the supplied agent function.

    Parameters
    ----------
    agent_fn : callable
        Takes an observation dict and returns an integer action (0-3).
    seed : int
        Reproducibility seed.

    Returns
    -------
    (total_reward, episode_info)
    """
    env = SupplyChainEnv(task=TASK_NAME, seed=seed)
    state = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = []

    obs_dict = env._make_observation(state).to_dict()

    while not done:
        action_int = agent_fn(obs_dict)
        action = SupplyAction.from_int(action_int)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps.append({"step": info["step"], "reward": reward})
        obs_dict = obs.to_dict()

    final_state = env.state()
    return total_reward, {
        "steps": len(steps),
        "total_reward": total_reward,
        "demand_satisfied": final_state.demand_satisfied,
        "stockouts": final_state.cumulative_stockouts,
        "overstock_steps": final_state.cumulative_overstock_steps,
        "disruption_active": final_state.disruption_active,
    }


# ──────────────────────────────────────────────
# Automated grader  →  score ∈ [0.0, 1.0]
# ──────────────────────────────────────────────

def grade(agent_fn: Callable[[dict], int], n_trials: int = 5) -> float:
    """
    Evaluate agent over multiple seeds and return normalised score in [0, 1].

    Bonus modifier: agents that avoid stockouts get a +5% bonus per clean trial.
    """
    total = 0.0
    bonus = 0.0

    for seed in range(n_trials):
        r, info = run_episode(agent_fn, seed=seed)
        total += r
        if info["stockouts"] == 0:
            bonus += 0.05  # 5% bonus per clean (zero-stockout) trial

    raw_score = total / n_trials
    base = max(0.01, min(0.90, raw_score / MAX_POSSIBLE_REWARD))
    score = min(0.99, base + bonus / n_trials)
    return round(score, 4)


# ──────────────────────────────────────────────
# Reference (heuristic) agent
# ──────────────────────────────────────────────

def heuristic_agent(obs: dict) -> int:
    """
    Adaptive heuristic for multi-product tasks:
      - Check per-product stocks and respond to the worst-off product
      - Consider disruption level to pre-order proactively
    """
    stock = obs.get("current_stock", 0.5)
    disruption = obs.get("disruption_level", 0.0)
    product_stocks: List[float] = obs.get("product_stocks", [])

    # If any product is critically low
    min_stock = min(product_stocks) if product_stocks else stock

    if min_stock < 0.10 or (disruption > 0 and stock < 0.20):
        return int(ActionType.EMERGENCY_RESTOCK)
    if min_stock < 0.25 or stock < 0.30:
        return int(ActionType.ORDER_INVENTORY)
    return int(ActionType.SHIP_PRODUCTS)


if __name__ == "__main__":
    score = grade(heuristic_agent)
    print(f"[MEDIUM] Heuristic agent score: {score:.4f}")
