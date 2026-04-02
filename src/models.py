"""
models.py — Typed dataclasses for the Supply Chain RL Environment.

Defines the core data structures used across the environment:
  - SupplyAction   : the discrete action chosen by the agent
  - SupplyObservation : the observation returned to the agent
  - SupplyState    : the full internal state of the environment
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Dict, List, Optional


# ──────────────────────────────────────────────
# Action Space
# ──────────────────────────────────────────────

class ActionType(IntEnum):
    WAIT             = 0   # do nothing this step
    ORDER_INVENTORY  = 1   # place a replenishment order
    SHIP_PRODUCTS    = 2   # fulfil pending demand from stock
    EMERGENCY_RESTOCK = 3  # pay premium to restock immediately


@dataclass
class SupplyAction:
    """Represents one agent action per environment step."""

    action_type: ActionType
    # Optional quantity override (None = use environment default)
    quantity: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "action_type": int(self.action_type),
            "quantity": self.quantity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SupplyAction":
        return cls(
            action_type=ActionType(d["action_type"]),
            quantity=d.get("quantity"),
        )

    @classmethod
    def from_int(cls, action_int: int, quantity: Optional[float] = None) -> "SupplyAction":
        return cls(action_type=ActionType(action_int), quantity=quantity)


# ──────────────────────────────────────────────
# Observation Space
# ──────────────────────────────────────────────

@dataclass
class SupplyObservation:
    """
    What the agent *sees* at each step.

    All values are normalised to [0, 1] so that the agent's
    neural network does not need to worry about different scales.
    """

    # Fraction of warehouse capacity currently occupied
    current_stock: float

    # Normalised rolling-average demand forecast (0 = very low, 1 = very high)
    demand_forecast: float

    # Fraction of warehouse capacity still available
    warehouse_capacity: float

    # Expected supplier lead-time expressed as a fraction of the max lead-time
    supplier_delay: float

    # Storage cost per unit as a fraction of maximum storage cost
    storage_cost: float

    # ── Multi-product extension (medium / hard tasks) ──────────────────────
    # Per-product stock levels (empty list for easy task)
    product_stocks: List[float] = field(default_factory=list)

    # Per-product demand forecasts (empty list for easy task)
    product_demands: List[float] = field(default_factory=list)

    # Number of active supply disruptions (hard task only)
    disruption_level: float = 0.0

    def to_dict(self) -> dict:
        return {
            "current_stock": round(self.current_stock, 4),
            "demand_forecast": round(self.demand_forecast, 4),
            "warehouse_capacity": round(self.warehouse_capacity, 4),
            "supplier_delay": round(self.supplier_delay, 4),
            "storage_cost": round(self.storage_cost, 4),
            "product_stocks": [round(v, 4) for v in self.product_stocks],
            "product_demands": [round(v, 4) for v in self.product_demands],
            "disruption_level": round(self.disruption_level, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SupplyObservation":
        return cls(
            current_stock=d["current_stock"],
            demand_forecast=d["demand_forecast"],
            warehouse_capacity=d["warehouse_capacity"],
            supplier_delay=d["supplier_delay"],
            storage_cost=d["storage_cost"],
            product_stocks=d.get("product_stocks", []),
            product_demands=d.get("product_demands", []),
            disruption_level=d.get("disruption_level", 0.0),
        )


# ──────────────────────────────────────────────
# Full Environment State
# ──────────────────────────────────────────────

@dataclass
class SupplyState:
    """
    The complete internal state of the supply chain environment.

    This is richer than the observation — it contains ground-truth
    values the agent does not directly observe (e.g. pending orders,
    actual demand, cumulative metrics).
    """

    # ── Core inventory ─────────────────────────────────────────────────────
    stock_level: float                  # actual units in warehouse
    max_capacity: float                 # maximum warehouse capacity (units)
    reorder_point: float                # trigger level for automatic re-orders
    order_quantity: float               # default units per order

    # ── Demand ─────────────────────────────────────────────────────────────
    current_demand: float               # units demanded this step
    demand_mean: float                  # baseline demand
    demand_std: float                   # demand variability

    # ── Supplier ───────────────────────────────────────────────────────────
    supplier_lead_time: int             # steps until next order arrives
    max_lead_time: int                  # worst-case lead time
    pending_order_qty: float            # units on order, not yet received
    pending_order_eta: int              # steps until pending order arrives

    # ── Costs ──────────────────────────────────────────────────────────────
    storage_cost_per_unit: float        # $/unit/step
    stockout_penalty: float             # $/unit short
    emergency_premium: float            # multiplier for emergency orders

    # ── Episode tracking ───────────────────────────────────────────────────
    step_number: int = 0
    max_steps: int = 50
    total_reward: float = 0.0
    cumulative_stockouts: int = 0
    cumulative_overstock_steps: int = 0
    demand_satisfied: int = 0

    # ── Multi-product extension ────────────────────────────────────────────
    product_stocks: List[float] = field(default_factory=list)
    product_demands: List[float] = field(default_factory=list)
    product_demand_means: List[float] = field(default_factory=list)
    product_demand_stds: List[float] = field(default_factory=list)

    # ── Supply disruptions (hard task) ─────────────────────────────────────
    disruption_active: bool = False
    disruption_duration: int = 0
    demand_spike_active: bool = False

    # ── Additional metadata ────────────────────────────────────────────────
    task_name: str = "easy"
    scenario_seed: Optional[int] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stock_level": self.stock_level,
            "max_capacity": self.max_capacity,
            "reorder_point": self.reorder_point,
            "order_quantity": self.order_quantity,
            "current_demand": self.current_demand,
            "demand_mean": self.demand_mean,
            "demand_std": self.demand_std,
            "supplier_lead_time": self.supplier_lead_time,
            "max_lead_time": self.max_lead_time,
            "pending_order_qty": self.pending_order_qty,
            "pending_order_eta": self.pending_order_eta,
            "storage_cost_per_unit": self.storage_cost_per_unit,
            "stockout_penalty": self.stockout_penalty,
            "emergency_premium": self.emergency_premium,
            "step_number": self.step_number,
            "max_steps": self.max_steps,
            "total_reward": self.total_reward,
            "cumulative_stockouts": self.cumulative_stockouts,
            "cumulative_overstock_steps": self.cumulative_overstock_steps,
            "demand_satisfied": self.demand_satisfied,
            "product_stocks": self.product_stocks,
            "product_demands": self.product_demands,
            "product_demand_means": self.product_demand_means,
            "product_demand_stds": self.product_demand_stds,
            "disruption_active": self.disruption_active,
            "disruption_duration": self.disruption_duration,
            "demand_spike_active": self.demand_spike_active,
            "task_name": self.task_name,
            "scenario_seed": self.scenario_seed,
            "extra": self.extra,
        }
