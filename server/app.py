"""
app.py — FastAPI server exposing the OpenEnv-compatible endpoints.

Endpoints
---------
POST /reset        — start a new episode (optionally specify task & seed)
POST /step         — execute one agent action
GET  /state        — inspect current environment state
GET  /health       — liveness probe
GET  /info         — environment metadata
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.environment import SupplyChainEnv
from src.models import SupplyAction, ActionType

# ──────────────────────────────────────────────
# App + CORS
# ──────────────────────────────────────────────

app = FastAPI(
    title="Supply Chain OpenEnv",
    description=(
        "An OpenEnv-compatible reinforcement learning environment "
        "for Supply Chain Inventory Management."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Singleton environment (one session per server)
# ──────────────────────────────────────────────

_env: Optional[SupplyChainEnv] = None


def _get_env() -> SupplyChainEnv:
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return _env


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = Field("easy", description="Task difficulty: easy | medium | hard")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    action: int = Field(
        ...,
        ge=0,
        le=3,
        description="0=wait, 1=order_inventory, 2=ship_products, 3=emergency_restock",
    )
    quantity: Optional[float] = Field(
        None, description="Optional quantity override for the action"
    )


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/info", tags=["System"])
def info() -> Dict[str, Any]:
    return {
        "name": "supply-chain-openenv",
        "version": "1.0.0",
        "action_space": {
            "type": "discrete",
            "n": 4,
            "labels": {
                "0": "wait",
                "1": "order_inventory",
                "2": "ship_products",
                "3": "emergency_restock",
            },
        },
        "observation_space": {
            "type": "box",
            "fields": [
                "current_stock",
                "demand_forecast",
                "warehouse_capacity",
                "supplier_delay",
                "storage_cost",
                "product_stocks",
                "product_demands",
                "disruption_level",
            ],
            "low": 0.0,
            "high": 1.0,
        },
        "tasks": ["easy", "medium", "hard"],
    }


@app.post("/reset", tags=["OpenEnv"])
def reset(req: ResetRequest) -> Dict[str, Any]:
    """
    Initialise a new episode.

    Returns the full starting state of the environment.
    """
    global _env
    _env = SupplyChainEnv(task=req.task, seed=req.seed)
    state = _env.reset(seed=req.seed)
    return {
        "status": "reset",
        "task": req.task,
        "seed": req.seed,
        "state": state.to_dict(),
    }


@app.post("/step", tags=["OpenEnv"])
def step(req: StepRequest) -> Dict[str, Any]:
    """
    Execute one agent action and return (observation, reward, done, info).
    """
    env = _get_env()
    action = SupplyAction.from_int(req.action, quantity=req.quantity)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.to_dict(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", tags=["OpenEnv"])
def get_state() -> Dict[str, Any]:
    """
    Return the current full environment state (for debugging / visualisation).
    """
    env = _get_env()
    s = env.state()
    if s is None:
        raise HTTPException(status_code=400, detail="No active episode.")
    return {"state": s.to_dict()}


# ──────────────────────────────────────────────
# Visual Dashboard
# ──────────────────────────────────────────────

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Supply Chain OpenEnv | Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --secondary: #a855f7;
                --bg: #0f172a;
                --card-bg: #1e293b;
                --text: #f8fafc;
                --success: #22c55e;
                --warning: #eab308;
                --danger: #ef4444;
            }
            body {
                font-family: 'Outfit', sans-serif;
                background: var(--bg);
                color: var(--text);
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                padding: 2rem;
            }
            .container {
                max-width: 900px;
                width: 100%;
            }
            header {
                text-align: center;
                margin-bottom: 3rem;
            }
            h1 {
                font-size: 3rem;
                font-weight: 600;
                margin: 0;
                background: linear-gradient(to right, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .status-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 99px;
                background: rgba(34, 197, 94, 0.2);
                color: var(--success);
                font-weight: 600;
                margin-top: 1rem;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.5rem;
                margin-bottom: 3rem;
            }
            .card {
                background: var(--card-bg);
                padding: 1.5rem;
                border-radius: 1rem;
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.3s ease, border-color 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                border-color: var(--primary);
            }
            .card h3 {
                margin-top: 0;
                color: #94a3b8;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .card .value {
                font-size: 2rem;
                font-weight: 600;
                margin: 0.5rem 0;
            }
            .card .desc {
                font-size: 0.85rem;
                color: #64748b;
            }
            .btn {
                display: inline-block;
                padding: 0.75rem 1.5rem;
                background: var(--primary);
                color: white;
                text-decoration: none;
                border-radius: 0.5rem;
                font-weight: 600;
                transition: background 0.3s ease;
            }
            .btn:hover {
                background: #4f46e5;
            }
            .footer {
                margin-top: auto;
                text-align: center;
                color: #64748b;
                font-size: 0.9rem;
            }
            .pulse {
                display: inline-block;
                width: 10px;
                height: 10px;
                background: var(--success);
                border-radius: 50%;
                margin-right: 8px;
                box-shadow: 0 0 0 rgba(34, 197, 94, 0.4);
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
                100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Supply Chain RL</h1>
                <div class="status-badge"><span class="pulse"></span>Environment Online</div>
            </header>

            <div class="grid">
                <div class="card">
                    <h3>Observation Space</h3>
                    <div class="value">Continious</div>
                    <div class="desc">Normalised [0,1] multi-dimensional vector.</div>
                </div>
                <div class="card">
                    <h3>Action Space</h3>
                    <div class="value">Discrete (4)</div>
                    <div class="desc">Wait, Order, Ship, Emergency Restock.</div>
                </div>
                <div class="card">
                    <h3>OpenEnv Compliance</h3>
                    <div class="value">100%</div>
                    <div class="desc">Supports /reset, /step, /state endpoints.</div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 2rem;">
                <h3 style="color: var(--primary)">Quick Start</h3>
                <p>Connect your agent to this environment using the base URL below:</p>
                <code style="background: #000; padding: 1rem; border-radius: 0.5rem; display: block; margin: 1rem 0; word-break: break-all;">
                    GET /info
                </code>
                <a href="/docs" class="btn">View API Docs</a>
            </div>

            <div class="footer">
                Built for the OpenEnv Reinforcement Learning Challenge &bull; v1.0.0
            </div>
        </div>
    </body>
    </html>
    """


# ──────────────────────────────────────────────
# Entry point (local dev)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
