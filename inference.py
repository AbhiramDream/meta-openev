"""
inference.py — Baseline LLM agent for the Supply Chain OpenEnv.

Mandatory environment variables
--------------------------------
    API_BASE_URL   The base URL of the LLM API endpoint.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key (sent as Bearer token).

Optional environment variables
--------------------------------
    ENV_BASE_URL   Base URL of the running OpenEnv server.
                   Default: http://localhost:7860
    TASKS          Comma-separated tasks to run. Default: easy,medium,hard
    SEED           Integer seed for reproducibility. Default: 42

Structured stdout format (mandatory — evaluated by the scorer)
--------------------------------------------------------------
    [START] <json>   — emitted once at the start of each task episode
    [STEP]  <json>   — emitted once per environment step
    [END]   <json>   — emitted once at the end of each task episode

Usage
-----
    python inference.py
    python inference.py --tasks easy medium --seed 0
    python inference.py --env-url https://<your-space>.hf.space
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Mandatory configuration — read from environment variables
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str     = os.environ.get("HF_TOKEN", "")

ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
HTTP_TIMEOUT: int = 30   # seconds per environment HTTP call

# Maximum steps safety-cap (environment enforces its own limit too)
MAX_STEPS_CAP: int = 120

# Normalised score denominators per task (theoretical max reward)
MAX_REWARDS: Dict[str, float] = {
    "easy":   75.0,
    "medium": 100.0,
    "hard":   120.0,
}

SYSTEM_PROMPT = """\
You are an AI supply chain manager controlling a warehouse inventory system.

At each step you receive an observation with normalised fields (all in [0, 1]):
  current_stock       — how full the warehouse currently is
  demand_forecast     — expected demand this step  (1 = very high)
  warehouse_capacity  — remaining free space       (1 = completely empty)
  supplier_delay      — lead-time fraction         (1 = maximum delay)
  storage_cost        — cost per unit              (normalised)
  product_stocks      — per-product stock levels   (list)
  product_demands     — per-product demand         (list)
  disruption_level    — 1.0 if supplier disruption is active, else 0.0

Choose ONE action:
  0 = wait                (do nothing)
  1 = order_inventory     (place a replenishment order from the supplier)
  2 = ship_products       (fulfil demand from current warehouse stock)
  3 = emergency_restock   (immediate restock at a premium cost)

Reward signals:
  +1.0  demand satisfied (stock >= demand)
  +0.5  efficient storage utilisation (20–70% of capacity)
  -1.0  stockout (stock < demand)
  -0.5  excessive overstock (>85% capacity used)

Respond ONLY with a JSON object: {"action": <integer 0-3>}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Environment HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────

def _env_reset(env_url: str, task: str, seed: int) -> Dict[str, Any]:
    r = requests.post(
        f"{env_url}/reset",
        json={"task": task, "seed": seed},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _env_step(env_url: str, action: int) -> Dict[str, Any]:
    r = requests.post(
        f"{env_url}/step",
        json={"action": action},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# LLM agent
# ──────────────────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    Calls the configured LLM (via OpenAI-compatible client) to pick the next
    action given the current observation.
    """

    def __init__(self) -> None:
        if not HF_TOKEN:
            print(
                "[WARN] HF_TOKEN is not set. "
                "API calls may fail if authentication is required.",
                file=sys.stderr,
            )
        self.client = OpenAI(
            api_key=HF_TOKEN or "EMPTY",
            base_url=API_BASE_URL,
        )
        self.model = MODEL_NAME
        # Rolling conversation history — keep last 3 turns for context
        self._history: List[Dict[str, str]] = []

    def act(self, obs: Dict[str, Any]) -> int:
        """Return action integer (0–3) given a normalised observation dict."""
        user_content = (
            "Current warehouse observation:\n"
            + json.dumps(obs, indent=2)
            + '\n\nPick the best action. Reply ONLY with {"action": <int>}.'
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        messages.extend(self._history[-6:])  # last 3 turns
        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            action = max(0, min(3, int(parsed.get("action", 2))))
        except Exception as exc:
            # Fallback: ship products (conservative default)
            print(f"[WARN] LLM call failed ({exc}), defaulting action=2", file=sys.stderr)
            action = 2
            content = json.dumps({"action": action})

        # Update conversation history
        self._history.append({"role": "user", "content": user_content})
        self._history.append({"role": "assistant", "content": content})

        return action


# ──────────────────────────────────────────────────────────────────────────────
# Structured log helpers  (mandatory format)
# ──────────────────────────────────────────────────────────────────────────────

def _log(tag: str, payload: Dict[str, Any]) -> None:
    """Print a single structured log line to stdout."""
    print(f"{tag} {json.dumps(payload)}", flush=True)


def log_start(task: str, seed: int, model: str, env_url: str) -> None:
    _log("[START]", {
        "task": task,
        "seed": seed,
        "model": model,
        "env_url": env_url,
        "api_base_url": API_BASE_URL,
    })


def log_step(
    step: int,
    action: int,
    action_label: str,
    observation: Dict[str, Any],
    reward: float,
    total_reward: float,
    done: bool,
    info: Dict[str, Any],
) -> None:
    _log("[STEP]", {
        "step": step,
        "action": action,
        "action_label": action_label,
        "reward": reward,
        "total_reward": round(total_reward, 4),
        "done": done,
        "stock": round(observation.get("current_stock", 0.0), 4),
        "demand": round(observation.get("demand_forecast", 0.0), 4),
        "disruption": observation.get("disruption_level", 0.0),
        "stockouts": info.get("stockouts", 0),
    })


def log_end(
    task: str,
    seed: int,
    model: str,
    total_reward: float,
    steps: int,
    stockouts: int,
    score: float,
    elapsed: float,
) -> None:
    _log("[END]", {
        "task": task,
        "seed": seed,
        "model": model,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "stockouts": stockouts,
        "score": round(score, 4),
        "elapsed_seconds": round(elapsed, 2),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────────

ACTION_LABELS = {0: "wait", 1: "order_inventory", 2: "ship_products", 3: "emergency_restock"}


def run_episode(
    task: str,
    seed: int,
    env_url: str,
) -> Dict[str, Any]:
    """
    Run one episode for `task` using the global LLMAgent.

    Emits [START], one [STEP] per step, then [END].
    Returns a result dict.
    """
    agent = LLMAgent()
    t0 = time.time()

    # ── Reset ────────────────────────────────────────────────────────────────
    reset_resp = _env_reset(env_url, task, seed)
    state_dict = reset_resp.get("state", {})

    log_start(task=task, seed=seed, model=MODEL_NAME, env_url=env_url)

    # Derive first observation from reset state
    cap = max(state_dict.get("max_capacity", 1.0), 1.0)
    obs: Dict[str, Any] = {
        "current_stock":      state_dict.get("stock_level", 0.0) / cap,
        "demand_forecast":    0.5,
        "warehouse_capacity": 1.0 - state_dict.get("stock_level", 0.0) / cap,
        "supplier_delay":     0.5,
        "storage_cost":       state_dict.get("storage_cost_per_unit", 0.3),
        "product_stocks":     state_dict.get("product_stocks", []),
        "product_demands":    state_dict.get("product_demands", []),
        "disruption_level":   0.0,
    }

    # ── Episode loop ─────────────────────────────────────────────────────────
    total_reward = 0.0
    step_count   = 0
    done         = False
    last_info: Dict[str, Any] = {}

    while not done and step_count < MAX_STEPS_CAP:
        action      = agent.act(obs)
        step_resp   = _env_step(env_url, action)

        obs         = step_resp["observation"]
        reward      = step_resp["reward"]
        done        = step_resp["done"]
        last_info   = step_resp["info"]

        total_reward += reward
        step_count   += 1

        log_step(
            step=step_count,
            action=action,
            action_label=ACTION_LABELS.get(action, "unknown"),
            observation=obs,
            reward=reward,
            total_reward=total_reward,
            done=done,
            info=last_info,
        )

    # ── Score ─────────────────────────────────────────────────────────────────
    elapsed   = time.time() - t0
    max_r     = MAX_REWARDS.get(task, 100.0)
    score     = round(max(0.0, min(1.0, total_reward / max_r)), 4)
    stockouts = last_info.get("stockouts", 0)

    log_end(
        task=task,
        seed=seed,
        model=MODEL_NAME,
        total_reward=total_reward,
        steps=step_count,
        stockouts=stockouts,
        score=score,
        elapsed=elapsed,
    )

    return {
        "task":            task,
        "seed":            seed,
        "model":           MODEL_NAME,
        "total_reward":    round(total_reward, 4),
        "steps":           step_count,
        "stockouts":       stockouts,
        "score":           score,
        "elapsed_seconds": round(elapsed, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry-point  — runs all tasks and prints a final JSON summary
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supply Chain OpenEnv — LLM baseline agent"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["easy", "medium", "hard"],
        default=os.environ.get("TASKS", "easy,medium,hard").split(","),
        help="Tasks to evaluate (default: easy medium hard)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", "42")),
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--env-url",
        default=ENV_BASE_URL,
        help="Base URL of the OpenEnv server (default: $ENV_BASE_URL or localhost:7860)",
    )
    args = parser.parse_args()

    # ── Validate mandatory vars ──────────────────────────────────────────────
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)]
    if missing:
        print(
            f"[WARN] The following mandatory env vars are not set: {missing}. "
            "Using defaults — results may differ from evaluation.",
            file=sys.stderr,
        )

    print(
        f"[INFO] Config: API_BASE_URL={API_BASE_URL} | MODEL={MODEL_NAME} | "
        f"env={args.env_url} | tasks={args.tasks} | seed={args.seed}",
        file=sys.stderr,
        flush=True,
    )

    # ── Run each task ────────────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    wall_t0 = time.time()

    for task in args.tasks:
        print(f"\n[INFO] Starting task: {task}", file=sys.stderr, flush=True)
        result = run_episode(task=task, seed=args.seed, env_url=args.env_url)
        results.append(result)

    wall_elapsed = time.time() - wall_t0

    # ── Final summary ────────────────────────────────────────────────────────
    summary = {
        "results":              results,
        "mean_score":           round(sum(r["score"] for r in results) / len(results), 4),
        "total_elapsed_seconds": round(wall_elapsed, 2),
        "model":                MODEL_NAME,
        "api_base_url":         API_BASE_URL,
    }

    # Print JSON summary — parseable by evaluation pipeline
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
