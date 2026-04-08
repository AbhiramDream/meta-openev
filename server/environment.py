"""
environment.py — Core supply chain simulation engine.

Implements the OpenEnv interface:
  reset()        → SupplyState (new episode)
  step(action)   → (observation, reward, done, info)
  state()        → SupplyState (current snapshot)
"""

from __future__ import annotations

import random
import math
from typing import Any, Dict, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    ActionType,
    SupplyAction,
    SupplyObservation,
    SupplyState,
)


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ──────────────────────────────────────────────
# Main Environment Class
# ──────────────────────────────────────────────

class SupplyChainEnv:
    """
    Supply Chain Inventory Management environment.

    Supports three difficulty tiers via the `task` parameter:
      "easy"   — single product, stable demand
      "medium" — multiple products, varying demand patterns
      "hard"   — disruptions, demand spikes, longer episodes
    """

    TASK_CONFIGS: Dict[str, Dict] = {
        "easy": {
            "n_products": 1,
            "max_capacity": 200.0,
            "initial_stock_frac": 0.5,
            "demand_mean": 20.0,
            "demand_std": 4.0,
            "max_lead_time": 5,
            "storage_cost_per_unit": 0.3,
            "stockout_penalty": 2.0,
            "order_quantity": 50.0,
            "emergency_premium": 2.5,
            "max_steps": 50,
            "disruption_prob": 0.0,
            "spike_prob": 0.0,
        },
        "medium": {
            "n_products": 3,
            "max_capacity": 500.0,
            "initial_stock_frac": 0.4,
            "demand_mean": 30.0,
            "demand_std": 10.0,
            "max_lead_time": 8,
            "storage_cost_per_unit": 0.4,
            "stockout_penalty": 2.5,
            "order_quantity": 80.0,
            "emergency_premium": 3.0,
            "max_steps": 75,
            "disruption_prob": 0.05,
            "spike_prob": 0.05,
        },
        "hard": {
            "n_products": 5,
            "max_capacity": 1000.0,
            "initial_stock_frac": 0.3,
            "demand_mean": 50.0,
            "demand_std": 20.0,
            "max_lead_time": 12,
            "storage_cost_per_unit": 0.5,
            "stockout_penalty": 3.0,
            "order_quantity": 120.0,
            "emergency_premium": 4.0,
            "max_steps": 100,
            "disruption_prob": 0.15,
            "spike_prob": 0.12,
        },
    }

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Choose from {list(self.TASK_CONFIGS)}")
        self.task = task
        self.cfg = self.TASK_CONFIGS[task]
        self._seed = seed
        self._state: Optional[SupplyState] = None
        self._rng = random.Random(seed)

    # ─────────────────────────────────────────
    # OpenEnv Interface
    # ─────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> SupplyState:
        """Initialise a new episode and return the starting state."""
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            self._rng = random.Random(self._seed)

        cfg = self.cfg
        n = cfg["n_products"]
        cap = cfg["max_capacity"]
        init_stock = cfg["initial_stock_frac"] * cap

        # Per-product demand parameters (slight variation around the mean)
        product_means = [
            cfg["demand_mean"] * self._rng.uniform(0.7, 1.3) for _ in range(n)
        ]
        product_stds = [
            cfg["demand_std"] * self._rng.uniform(0.5, 1.5) for _ in range(n)
        ]

        # Draw initial per-product stocks
        product_stocks = [
            _clamp(self._rng.gauss(init_stock / n, init_stock / (4 * n)), 0, cap / n)
            for _ in range(n)
        ]
        total_initial_stock = sum(product_stocks)

        # Sample initial demands
        product_demands = [
            max(0.0, self._rng.gauss(product_means[i], product_stds[i]))
            for i in range(n)
        ]

        self._state = SupplyState(
            # Core
            stock_level=total_initial_stock,
            max_capacity=cap,
            reorder_point=cap * 0.25,
            order_quantity=cfg["order_quantity"],
            # Demand
            current_demand=sum(product_demands),
            demand_mean=cfg["demand_mean"],
            demand_std=cfg["demand_std"],
            # Supplier
            supplier_lead_time=self._rng.randint(1, cfg["max_lead_time"]),
            max_lead_time=cfg["max_lead_time"],
            pending_order_qty=0.0,
            pending_order_eta=0,
            # Costs
            storage_cost_per_unit=cfg["storage_cost_per_unit"],
            stockout_penalty=cfg["stockout_penalty"],
            emergency_premium=cfg["emergency_premium"],
            # Episode
            step_number=0,
            max_steps=cfg["max_steps"],
            total_reward=0.0,
            cumulative_stockouts=0,
            cumulative_overstock_steps=0,
            demand_satisfied=0,
            # Multi-product
            product_stocks=product_stocks,
            product_demands=product_demands,
            product_demand_means=product_means,
            product_demand_stds=product_stds,
            # Disruptions
            disruption_active=False,
            disruption_duration=0,
            demand_spike_active=False,
            # Meta
            task_name=self.task,
            scenario_seed=self._seed,
        )
        return self._state

    def step(
        self, action: SupplyAction
    ) -> Tuple[SupplyObservation, float, bool, Dict[str, Any]]:
        """
        Execute one agent action and advance the simulation.

        Returns
        -------
        observation : SupplyObservation
        reward      : float
        done        : bool
        info        : dict  (human-readable diagnostics)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        s = self._state
        cfg = self.cfg

        # ── 1. Receive any pending order ─────────────────────────────────
        order_arrived = 0.0
        if s.pending_order_qty > 0:
            s.pending_order_eta -= 1
            if s.pending_order_eta <= 0:
                received = min(s.pending_order_qty, s.max_capacity - s.stock_level)
                s.stock_level = _clamp(s.stock_level + received, 0, s.max_capacity)
                order_arrived = received
                s.pending_order_qty = 0.0
                s.pending_order_eta = 0

        # ── 2. Maybe trigger disruption / spike (medium & hard) ──────────
        self._update_disruptions(s)

        # ── 3. Sample new demand ─────────────────────────────────────────
        self._sample_demand(s)

        # ── 4. Execute agent action ──────────────────────────────────────
        action_cost = 0.0
        action_label = action.action_type.name

        if action.action_type == ActionType.WAIT:
            pass  # do nothing

        elif action.action_type == ActionType.ORDER_INVENTORY:
            if s.pending_order_qty == 0:  # only one outstanding order allowed
                qty = action.quantity if action.quantity else s.order_quantity
                qty = min(qty, s.max_capacity - s.stock_level)
                if not s.disruption_active:
                    s.pending_order_qty = qty
                    s.pending_order_eta = s.supplier_lead_time
                else:
                    # Disruption delays the order further
                    delay_extra = self._rng.randint(2, 5)
                    s.pending_order_qty = qty
                    s.pending_order_eta = s.supplier_lead_time + delay_extra

        elif action.action_type == ActionType.SHIP_PRODUCTS:
            # Fulfil demand from stock
            fulfil = min(s.stock_level, s.current_demand)
            s.stock_level -= fulfil
            s.demand_satisfied += int(fulfil >= s.current_demand * 0.9)

        elif action.action_type == ActionType.EMERGENCY_RESTOCK:
            # Immediate restock at premium cost
            qty = action.quantity if action.quantity else s.order_quantity
            qty = min(qty, s.max_capacity - s.stock_level)
            s.stock_level = _clamp(s.stock_level + qty, 0, s.max_capacity)
            action_cost = qty * s.storage_cost_per_unit * (s.emergency_premium - 1)

        # ── 5. Natural demand consumption (if agent didn't ship) ─────────
        if action.action_type != ActionType.SHIP_PRODUCTS:
            consumed = min(s.stock_level, s.current_demand)
            s.stock_level -= consumed

        # ── 6. Compute reward ────────────────────────────────────────────
        reward = self._compute_reward(s, action_cost)

        # ── 7. Update multi-product sub-states ──────────────────────────
        self._update_products(s)

        # ── 8. Advance step counter ──────────────────────────────────────
        s.step_number += 1
        s.total_reward += reward

        # Explicitly enforce max steps for each task to ensure termination
        max_steps_allowed = s.max_steps if s.max_steps > 0 else 75
        done = s.step_number >= max_steps_allowed

        # Safety fallback: Never exceed 120 steps regardless of task
        if s.step_number >= 120:
            done = True

        obs = self._make_observation(s)
        info = {
            "step": s.step_number,
            "stock_level": s.stock_level,
            "demand": s.current_demand,
            "order_arrived": order_arrived,
            "disruption_active": s.disruption_active,
            "demand_spike_active": s.demand_spike_active,
            "action_label": action_label,
            "total_reward": s.total_reward,
            "stockouts": s.cumulative_stockouts,
        }

        return obs, reward, done, info

    def state(self) -> Optional[SupplyState]:
        """Return the full current environment state."""
        return self._state

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _update_disruptions(self, s: SupplyState) -> None:
        cfg = self.cfg

        # Disrupt supplier
        if s.disruption_active:
            s.disruption_duration -= 1
            if s.disruption_duration <= 0:
                s.disruption_active = False
        elif self._rng.random() < cfg["disruption_prob"]:
            s.disruption_active = True
            s.disruption_duration = self._rng.randint(2, 6)

        # Demand spike
        if s.demand_spike_active:
            # spike lasts 1 step
            s.demand_spike_active = False
        elif self._rng.random() < cfg["spike_prob"]:
            s.demand_spike_active = True

    def _sample_demand(self, s: SupplyState) -> None:
        """Draw stochastic demand for this step."""
        base = max(0.0, self._rng.gauss(s.demand_mean, s.demand_std))
        if s.demand_spike_active:
            base *= self._rng.uniform(2.0, 4.0)
        s.current_demand = base

    def _compute_reward(self, s: SupplyState, action_cost: float) -> float:
        """
        Reward shaping:
          +1.0  — demand satisfied (stock ≥ demand this step)
          +0.5  — efficient storage (stock between 20% and 70% of capacity)
          -1.0  — stockout (stock < demand)
          -0.5  — excessive overstock (stock > 85% of capacity)
          -      action_cost (emergency premium)
        """
        reward = 0.0
        demand = s.current_demand
        stock = s.stock_level
        cap = s.max_capacity

        # Demand satisfaction
        if stock >= demand:
            reward += 1.0
            s.demand_satisfied += 1
        else:
            # Proportional penalty
            shortage_ratio = (demand - stock) / max(demand, 1.0)
            reward -= 1.0 * shortage_ratio
            s.cumulative_stockouts += 1

        # Storage efficiency
        occ_frac = stock / cap
        if 0.20 <= occ_frac <= 0.70:
            reward += 0.5
        elif occ_frac > 0.85:
            reward -= 0.5
            s.cumulative_overstock_steps += 1

        # Holding cost penalty (continuous)
        holding_penalty = (stock * s.storage_cost_per_unit) / (cap * s.storage_cost_per_unit)
        reward -= 0.1 * holding_penalty

        # Emergency action cost
        reward -= action_cost * 0.01

        return round(reward, 4)

    def _update_products(self, s: SupplyState) -> None:
        """Refresh per-product demand for multi-product tasks."""
        n = len(s.product_stocks)
        if n == 0:
            return
        new_demands = []
        for i in range(n):
            d = max(0.0, self._rng.gauss(s.product_demand_means[i], s.product_demand_stds[i]))
            if s.demand_spike_active:
                d *= self._rng.uniform(1.5, 3.0)
            new_demands.append(d)
        s.product_demands = new_demands

    def _make_observation(self, s: SupplyState) -> SupplyObservation:
        """Convert internal state to a normalised agent observation."""
        cap = s.max_capacity or 1.0
        max_demand = s.demand_mean + 4 * s.demand_std or 1.0
        max_lead = s.max_lead_time or 1.0
        max_cost = 1.0  # storage cost already normalised

        return SupplyObservation(
            current_stock=_clamp(s.stock_level / cap, 0.0, 1.0),
            demand_forecast=_clamp(s.current_demand / max_demand, 0.0, 1.0),
            warehouse_capacity=_clamp(1.0 - s.stock_level / cap, 0.0, 1.0),
            supplier_delay=_clamp(s.supplier_lead_time / max_lead, 0.0, 1.0),
            storage_cost=_clamp(s.storage_cost_per_unit, 0.0, 1.0),
            product_stocks=[
                _clamp(ps / (cap / max(len(s.product_stocks), 1)), 0.0, 1.0)
                for ps in s.product_stocks
            ],
            product_demands=[
                _clamp(pd / max_demand, 0.0, 1.0)
                for pd in s.product_demands
            ],
            disruption_level=1.0 if s.disruption_active else 0.0,
        )
