"""
hard_task.py — Full supply chain with disruptions, demand spikes & multi-product.

Difficulty: Hard
  - 5 products
  - High demand variance
  - Frequent supply disruptions (15% probability per step)
  - Random demand spikes (2-4× normal demand)
  - Long lead times (up to 12 steps)
  - 100 episode steps
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Callable, Dict, List, Tuple
from server.environment import SupplyChainEnv
from src.models import SupplyAction, ActionType


# ──────────────────────────────────────────────
# Task definition
# ──────────────────────────────────────────────

TASK_NAME = "hard"
MAX_POSSIBLE_REWARD = 120.0   # theoretical upper bound (hard to attain)


def run_episode(
    agent_fn: Callable[[dict], int],
    seed: int = 0,
) -> Tuple[float, Dict]:
    """
    Run a full hard-task episode.

    Parameters
    ----------
    agent_fn : callable — returns action int given observation dict
    seed     : int      — episode seed

    Returns
    -------
    (total_reward, episode_info)
    """
    env = SupplyChainEnv(task=TASK_NAME, seed=seed)
    state = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    disruption_steps = 0
    spike_steps = 0

    obs_dict = env._make_observation(state).to_dict()

    while not done:
        action_int = agent_fn(obs_dict)
        action = SupplyAction.from_int(action_int)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if info.get("disruption_active"):
            disruption_steps += 1
        if info.get("demand_spike_active"):
            spike_steps += 1

        obs_dict = obs.to_dict()

    final_state = env.state()
    return total_reward, {
        "steps": final_state.step_number,
        "total_reward": total_reward,
        "demand_satisfied": final_state.demand_satisfied,
        "stockouts": final_state.cumulative_stockouts,
        "overstock_steps": final_state.cumulative_overstock_steps,
        "disruption_steps": disruption_steps,
        "spike_steps": spike_steps,
    }


# ──────────────────────────────────────────────
# Automated grader  →  score ∈ [0.0, 1.0]
# ──────────────────────────────────────────────

def grade(agent_fn: Callable[[dict], int], n_trials: int = 5) -> float:
    """
    Grade the agent with a weighted score that rewards:
      - High total reward (primary, 70% weight)
      - Low stockout rate  (secondary, 20% weight)
      - Low overstock rate (tertiary, 10% weight)

    All components normalised to [0, 1] before weighting.
    """
    rewards = []
    stockout_rates = []
    overstock_rates = []
    max_steps = 100  # hard task max steps

    for seed in range(n_trials):
        r, info = run_episode(agent_fn, seed=seed)
        rewards.append(r)
        stockout_rates.append(info["stockouts"] / max(info["steps"], 1))
        overstock_rates.append(info["overstock_steps"] / max(info["steps"], 1))

    # Normalise reward
    avg_reward = sum(rewards) / n_trials
    reward_component = max(0.0, min(1.0, avg_reward / MAX_POSSIBLE_REWARD))

    # Low stockout rate (1 = zero stockouts, 0 = stockout every step)
    avg_so_rate = sum(stockout_rates) / n_trials
    stockout_component = max(0.0, 1.0 - avg_so_rate)

    # Low overstock rate
    avg_os_rate = sum(overstock_rates) / n_trials
    overstock_component = max(0.0, 1.0 - avg_os_rate)

    score = (
        0.70 * reward_component
        + 0.20 * stockout_component
        + 0.10 * overstock_component
    )
    # Ensure score is strictly between 0 and 1 (0.01 to 0.99)
    return round(max(0.01, min(0.99, score)), 4)


# ──────────────────────────────────────────────
# Reference (heuristic) agent
# ──────────────────────────────────────────────

def heuristic_agent(obs: dict) -> int:
    """
    Conservative strategy for the hard task:
      - Pre-order aggressively when disruption is active or stock is low
      - Emergency restock when stock is critically low
      - Ship products when stock is healthy
    """
    stock = obs.get("current_stock", 0.5)
    disruption = obs.get("disruption_level", 0.0)
    product_stocks: List[float] = obs.get("product_stocks", [])
    demand_forecast = obs.get("demand_forecast", 0.5)

    min_stock = min(product_stocks) if product_stocks else stock

    # Critical shortage
    if min_stock < 0.08 or stock < 0.08:
        return int(ActionType.EMERGENCY_RESTOCK)

    # Anticipate disruption or high demand: order early and liberally
    if disruption > 0 or demand_forecast > 0.7:
        if stock < 0.50:
            return int(ActionType.ORDER_INVENTORY)

    # Normal ordering threshold
    if min_stock < 0.30 or stock < 0.25:
        return int(ActionType.ORDER_INVENTORY)

    return int(ActionType.SHIP_PRODUCTS)


if __name__ == "__main__":
    score = grade(heuristic_agent)
    print(f"[HARD] Heuristic agent score: {score:.4f}")
