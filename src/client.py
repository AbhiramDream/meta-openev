"""
client.py — Typed Python client for the Supply Chain OpenEnv server.

Usage
-----
    from src.client import SupplyChainClient

    client = SupplyChainClient(base_url="http://localhost:7860")
    state  = client.reset(task="easy", seed=42)
    obs, reward, done, info = client.step(action=2)
    state  = client.get_state()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests

from src.models import SupplyObservation, SupplyState, SupplyAction, ActionType


class SupplyChainClient:
    """
    HTTP client wrapping the Supply Chain OpenEnv FastAPI server.

    All methods raise `requests.HTTPError` on non-2xx responses.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ─────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────

    def _get(self, path: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: Dict) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def health(self) -> Dict[str, str]:
        """Check server liveness."""
        return self._get("/health")

    def info(self) -> Dict[str, Any]:
        """Return environment metadata."""
        return self._get("/info")

    def reset(
        self,
        task: str = "easy",
        seed: Optional[int] = None,
    ) -> SupplyState:
        """
        Initialise a new episode.

        Parameters
        ----------
        task : "easy" | "medium" | "hard"
        seed : optional reproducibility seed

        Returns
        -------
        SupplyState — the initial environment state
        """
        data = self._post("/reset", {"task": task, "seed": seed})
        return SupplyState(**data["state"])

    def step(
        self,
        action: int,
        quantity: Optional[float] = None,
    ) -> Tuple[SupplyObservation, float, bool, Dict[str, Any]]:
        """
        Execute one agent action.

        Parameters
        ----------
        action   : int  0=wait | 1=order | 2=ship | 3=emergency
        quantity : optional override for order size

        Returns
        -------
        (observation, reward, done, info)
        """
        data = self._post("/step", {"action": action, "quantity": quantity})
        obs = SupplyObservation.from_dict(data["observation"])
        return obs, data["reward"], data["done"], data["info"]

    def get_state(self) -> SupplyState:
        """Return the current full environment state."""
        data = self._get("/state")
        return SupplyState(**data["state"])
