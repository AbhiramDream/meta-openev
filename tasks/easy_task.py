"""
easy_task.py — Single-warehouse, single-product task with an automated grader.

Difficulty: Easy
  - 1 product
  - Stable, low-variance demand
  - Short lead-times
  - 50 episode steps
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Callable, Dict, Optional, Tuple
from server.environment import SupplyChainEnv
from src.models import SupplyAction, ActionType


# ──────────────────────────────────────────────
# Task definition
# ──────────────────────────────────────────────

TASK_NAME = "easy"
MAX_POSSIBLE_REWARD = 75.0   # theoretical upper bound over 50 steps


def run_episode(
    agent_fn: Callable[[dict], int],
    seed: int = 0,
) -> Tuple[float, Dict]:
    """
    Run a full easy-task episode using the supplied agent function.

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

    # Build initial observation
    obs_dict = env._make_observation(state).to_dict()

    while not done:
        action_int = agent_fn(obs_dict)
        action = SupplyAction.from_int(action_int)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps.append({"step": info["step"], "reward": reward, "info": info})
        obs_dict = obs.to_dict()

    final_state = env.state()
    return total_reward, {
        "steps": len(steps),
        "total_reward": total_reward,
        "demand_satisfied": final_state.demand_satisfied,
        "stockouts": final_state.cumulative_stockouts,
        "overstock_steps": final_state.cumulative_overstock_steps,
    }


# ──────────────────────────────────────────────
# Automated grader  →  score ∈ [0.0, 1.0]
# ──────────────────────────────────────────────

def grade(agent_fn: Callable[[dict], int], n_trials: int = 5) -> float:
    """
    Evaluate an agent over `n_trials` seeds and return a normalised score.

    Score formula
    -------------
    raw_score = mean total_reward across trials
    score     = clamp(raw_score / MAX_POSSIBLE_REWARD, 0, 1)
    """
    total = 0.0
    for seed in range(n_trials):
        r, _ = run_episode(agent_fn, seed=seed)
        total += r

    raw_score = total / n_trials
    score = max(0.0, min(1.0, raw_score / MAX_POSSIBLE_REWARD))
    return round(score, 4)


# ──────────────────────────────────────────────
# Reference (heuristic) agent for sanity-check
# ──────────────────────────────────────────────

def heuristic_agent(obs: dict) -> int:
    """
    Simple rule-based agent:
      - If stock is very low → emergency_restock
      - If stock is low     → order_inventory
      - Always              → ship_products
    """
    stock = obs["current_stock"]
    if stock < 0.10:
        return int(ActionType.EMERGENCY_RESTOCK)
    if stock < 0.30:
        return int(ActionType.ORDER_INVENTORY)
    return int(ActionType.SHIP_PRODUCTS)


if __name__ == "__main__":
    score = grade(heuristic_agent)
    print(f"[EASY] Heuristic agent score: {score:.4f}")
