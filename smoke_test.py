"""
smoke_test.py — Quick validation that the environment runs correctly.

Tests:
  1. reset() returns a valid state
  2. step() with each action type works
  3. Episode terminates at max_steps
  4. All three task graders return [0, 1] scores
  5. FastAPI app responds to /health, /reset, /step, /state
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient

from server.environment import SupplyChainEnv
from server.app import app
from src.models import SupplyAction, ActionType
from tasks.easy_task import grade as grade_easy, heuristic_agent as easy_agent
from tasks.medium_task import grade as grade_medium, heuristic_agent as medium_agent
from tasks.hard_task import grade as grade_hard, heuristic_agent as hard_agent


# ────────────────────────────────────────────────────────────────
# 1. Core environment tests
# ────────────────────────────────────────────────────────────────

def test_reset_easy():
    env = SupplyChainEnv(task="easy", seed=0)
    state = env.reset()
    assert state.step_number == 0
    assert 0 < state.stock_level <= state.max_capacity
    print("  [OK] reset() — easy")


def test_all_action_types():
    env = SupplyChainEnv(task="easy", seed=1)
    env.reset()
    for action_int in range(4):
        action = SupplyAction.from_int(action_int)
        obs, reward, done, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert info["step"] >= 1
    print("  [OK] step() — all 4 actions")


def test_episode_terminates():
    env = SupplyChainEnv(task="easy", seed=2)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(SupplyAction(ActionType.SHIP_PRODUCTS))
        steps += 1
        assert steps <= 55, "Episode did not terminate!"
    assert 45 <= steps <= 55
    print(f"  [OK] episode terminates after {steps} steps")


def test_observations_normalised():
    env = SupplyChainEnv(task="hard", seed=3)
    state = env.reset()
    obs = env._make_observation(state)
    for field in ["current_stock", "demand_forecast", "warehouse_capacity",
                  "supplier_delay", "storage_cost"]:
        val = getattr(obs, field)
        assert 0.0 <= val <= 1.0, f"{field}={val} out of [0,1]"
    print("  [OK] observations normalised to [0, 1]")


# ────────────────────────────────────────────────────────────────
# 2. Task grader tests
# ────────────────────────────────────────────────────────────────

def test_easy_grader():
    score = grade_easy(easy_agent, n_trials=3)
    assert 0.0 <= score <= 1.0, f"easy score out of range: {score}"
    print(f"  [OK] easy grader: {score:.4f}")


def test_medium_grader():
    score = grade_medium(medium_agent, n_trials=3)
    assert 0.0 <= score <= 1.0, f"medium score out of range: {score}"
    print(f"  [OK] medium grader: {score:.4f}")


def test_hard_grader():
    score = grade_hard(hard_agent, n_trials=3)
    assert 0.0 <= score <= 1.0, f"hard score out of range: {score}"
    print(f"  [OK] hard grader: {score:.4f}")


# ────────────────────────────────────────────────────────────────
# 3. FastAPI endpoint tests
# ────────────────────────────────────────────────────────────────

def test_api_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  [OK] GET /health")


def test_api_info():
    client = TestClient(app)
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert data["action_space"]["n"] == 4
    print("  [OK] GET /info")


def test_api_reset():
    client = TestClient(app)
    r = client.post("/reset", json={"task": "easy", "seed": 42})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "reset"
    assert "state" in body
    print("  [OK] POST /reset")


def test_api_step():
    client = TestClient(app)
    client.post("/reset", json={"task": "easy", "seed": 42})
    r = client.post("/step", json={"action": 2})
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body
    assert "reward" in body
    assert "done" in body
    assert "info" in body
    print("  [OK] POST /step")


def test_api_state():
    client = TestClient(app)
    client.post("/reset", json={"task": "medium", "seed": 7})
    r = client.get("/state")
    assert r.status_code == 200
    assert "state" in r.json()
    print("  [OK] GET /state")


def test_api_full_episode():
    client = TestClient(app)
    client.post("/reset", json={"task": "easy", "seed": 99})
    done = False
    steps = 0
    while not done:
        r = client.post("/step", json={"action": 2})
        body = r.json()
        done = body["done"]
        steps += 1
        assert steps <= 60
    print(f"  [OK] full API episode: {steps} steps")


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_reset_easy,
        test_all_action_types,
        test_episode_terminates,
        test_observations_normalised,
        test_easy_grader,
        test_medium_grader,
        test_hard_grader,
        test_api_health,
        test_api_info,
        test_api_reset,
        test_api_step,
        test_api_state,
        test_api_full_episode,
    ]

    print(f"\n{'='*60}")
    print(f"  Supply Chain OpenEnv — Smoke Tests ({len(tests)} tests)")
    print(f"{'='*60}")

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")

    print(f"\n{'─'*60}")
    print(f"  Result: {passed}/{len(tests)} tests passed")
    print(f"{'='*60}\n")

    if passed < len(tests):
        sys.exit(1)
