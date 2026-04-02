"""
validate.py — Pre-submission validation script for Supply Chain OpenEnv.

Runs every check from the hackathon pre-submission checklist and prints
a PASS / FAIL summary.  Exit code 0 = all checks passed.

Usage
-----
    # Validate against the local dev server (must already be running):
    python validate.py

    # Validate against a deployed HF Space:
    python validate.py --env-url https://<your-space>.hf.space

    # Start the server automatically, run checks, then shut it down:
    python validate.py --auto-server
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ──────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS_MARK = f"{GREEN}✔ PASS{RESET}"
FAIL_MARK = f"{RED}✖ FAIL{RESET}"
WARN_MARK = f"{YELLOW}⚠ WARN{RESET}"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

CheckResult = Tuple[bool, str]   # (passed, detail_message)


def _post(url: str, payload: Dict, timeout: int = 15) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def _get(url: str, timeout: int = 15) -> requests.Response:
    return requests.get(url, timeout=timeout)


# ──────────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────────

def check_env_vars() -> CheckResult:
    """Mandatory environment variables are set."""
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing  = [v for v in required if not os.environ.get(v)]
    if missing:
        return False, f"Missing env vars: {missing}"
    return True, f"All set: {required}"


def check_inference_file_exists() -> CheckResult:
    """inference.py exists at the project root."""
    path = os.path.join(os.path.dirname(__file__), "inference.py")
    if os.path.isfile(path):
        return True, f"Found: {path}"
    return False, f"Not found: {path}"


def check_openenv_yaml() -> CheckResult:
    """openenv.yaml is present and contains required keys."""
    path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if not os.path.isfile(path):
        return False, "openenv.yaml not found"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required_keys = ["name", "version", "endpoints", "action_space", "observation_space", "tasks"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        return False, f"openenv.yaml missing keys: {missing}"
    return True, f"Valid — {len(cfg.get('tasks', []))} tasks defined"


def check_health(env_url: str) -> CheckResult:
    """GET /health returns 200."""
    try:
        r = _get(f"{env_url}/health")
        if r.status_code == 200:
            return True, f"HTTP 200 — {r.json()}"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


def check_reset(env_url: str) -> CheckResult:
    """POST /reset returns a valid state for each task."""
    failures = []
    for task in ("easy", "medium", "hard"):
        try:
            r = _post(f"{env_url}/reset", {"task": task, "seed": 0})
            if r.status_code != 200:
                failures.append(f"{task}: HTTP {r.status_code}")
                continue
            body = r.json()
            if "state" not in body:
                failures.append(f"{task}: missing 'state' key")
        except Exception as e:
            failures.append(f"{task}: {e}")
    if failures:
        return False, "; ".join(failures)
    return True, "reset() OK for easy, medium, hard"


def check_step(env_url: str) -> CheckResult:
    """POST /step returns observation, reward, done, info."""
    try:
        _post(f"{env_url}/reset", {"task": "easy", "seed": 1})
        r = _post(f"{env_url}/step", {"action": 2})
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        body = r.json()
        required = ["observation", "reward", "done", "info"]
        missing = [k for k in required if k not in body]
        if missing:
            return False, f"Missing keys: {missing}"
        obs = body["observation"]
        obs_fields = ["current_stock", "demand_forecast", "warehouse_capacity",
                      "supplier_delay", "storage_cost"]
        missing_obs = [f for f in obs_fields if f not in obs]
        if missing_obs:
            return False, f"Observation missing fields: {missing_obs}"
        return True, "step() OK — all fields present"
    except Exception as e:
        return False, str(e)


def check_state(env_url: str) -> CheckResult:
    """GET /state returns full environment state."""
    try:
        _post(f"{env_url}/reset", {"task": "medium", "seed": 2})
        r = _get(f"{env_url}/state")
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        body = r.json()
        if "state" not in body:
            return False, "Missing 'state' key"
        return True, "state() OK"
    except Exception as e:
        return False, str(e)


def check_observations_normalised(env_url: str) -> CheckResult:
    """All scalar observation fields are in [0, 1]."""
    try:
        _post(f"{env_url}/reset", {"task": "hard", "seed": 3})
        violations = []
        for _ in range(10):
            r = _post(f"{env_url}/step", {"action": 2})
            obs = r.json().get("observation", {})
            for field in ["current_stock", "demand_forecast", "warehouse_capacity",
                          "supplier_delay", "storage_cost", "disruption_level"]:
                v = obs.get(field, 0.0)
                if not (0.0 <= v <= 1.0):
                    violations.append(f"{field}={v}")
            if r.json().get("done"):
                break
        if violations:
            return False, f"Out-of-range: {violations}"
        return True, "All scalar obs fields in [0, 1]"
    except Exception as e:
        return False, str(e)


def check_models_importable() -> CheckResult:
    """src.models can be imported and typed classes instantiate correctly."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from src.models import SupplyAction, SupplyObservation, SupplyState, ActionType
        a = SupplyAction(action_type=ActionType.SHIP_PRODUCTS)
        assert a.to_dict()["action_type"] == 2
        o = SupplyObservation(current_stock=0.5, demand_forecast=0.3,
                              warehouse_capacity=0.5, supplier_delay=0.2,
                              storage_cost=0.3)
        assert 0.0 <= o.current_stock <= 1.0
        return True, "SupplyAction, SupplyObservation, SupplyState all importable"
    except Exception as e:
        return False, str(e)


def check_task_graders() -> CheckResult:
    """Each task grader returns a score in [0.0, 1.0]."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from tasks.easy_task   import grade as ge, heuristic_agent as ae
        from tasks.medium_task import grade as gm, heuristic_agent as am
        from tasks.hard_task   import grade as gh, heuristic_agent as ah

        results = {}
        for name, grader, agent in [("easy", ge, ae), ("medium", gm, am), ("hard", gh, ah)]:
            score = grader(agent, n_trials=3)
            if not (0.0 <= score <= 1.0):
                return False, f"{name} grader returned {score} (out of [0,1])"
            results[name] = score
        return True, f"Scores — {results}"
    except Exception as e:
        return False, str(e)


def check_episode_terminates(env_url: str) -> CheckResult:
    """Episode terminates within max_steps for each task."""
    try:
        for task, cap in [("easy", 55), ("medium", 80), ("hard", 110)]:
            _post(f"{env_url}/reset", {"task": task, "seed": 99})
            done  = False
            steps = 0
            while not done:
                r = _post(f"{env_url}/step", {"action": 2})
                done = r.json()["done"]
                steps += 1
                if steps > cap:
                    return False, f"{task}: episode did not terminate within {cap} steps"
        return True, "All three tasks terminate correctly"
    except Exception as e:
        return False, str(e)


def check_inference_imports() -> CheckResult:
    """inference.py can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location(
            "inference",
            os.path.join(os.path.dirname(__file__), "inference.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Check mandatory symbols exist
        for sym in ("run_episode", "main", "log_start", "log_step", "log_end", "LLMAgent"):
            if not hasattr(mod, sym):
                return False, f"Missing symbol in inference.py: {sym}"
        # Check env-var references
        src = open(os.path.join(os.path.dirname(__file__), "inference.py")).read()
        for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
            if var not in src:
                return False, f"inference.py does not reference {var}"
        return True, "inference.py imports OK and references all mandatory env vars"
    except Exception as e:
        return False, str(e)


def check_log_format() -> CheckResult:
    """inference.py emits [START], [STEP], [END] tags in structured stdout."""
    try:
        src = open(os.path.join(os.path.dirname(__file__), "inference.py")).read()
        for tag in ("[START]", "[STEP]", "[END]"):
            if tag not in src:
                return False, f"Missing log tag: {tag}"
        return True, "[START], [STEP], [END] tags present in inference.py"
    except Exception as e:
        return False, str(e)


def check_dockerfile_exists() -> CheckResult:
    """Dockerfile exists at the project root."""
    path = os.path.join(os.path.dirname(__file__), "Dockerfile")
    if not os.path.isfile(path):
        return False, "Dockerfile not found at project root"
    content = open(path).read()
    for token in ("FROM python", "EXPOSE 7860", "API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        if token not in content:
            return False, f"Dockerfile missing expected token: {token}"
    return True, "Dockerfile present and contains required tokens"


def check_requirements_file() -> CheckResult:
    """requirements.txt exists and lists critical packages."""
    path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.isfile(path):
        return False, "requirements.txt not found"
    content = open(path).read().lower()
    required = ["fastapi", "uvicorn", "openai", "requests", "pydantic"]
    missing = [p for p in required if p not in content]
    if missing:
        return False, f"requirements.txt missing: {missing}"
    return True, "requirements.txt OK"


# ──────────────────────────────────────────────────────────────────────────────
# Server auto-start helper
# ──────────────────────────────────────────────────────────────────────────────

def _start_server(port: int = 7860) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", str(port), "--workers", "1"],
        cwd=os.path.dirname(__file__),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait until healthy
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=3)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError("Server did not become healthy within 30 seconds")


# ──────────────────────────────────────────────────────────────────────────────
# Main validator
# ──────────────────────────────────────────────────────────────────────────────

def run_validation(env_url: str, auto_server: bool = False) -> int:
    server_proc: Optional[subprocess.Popen] = None

    if auto_server:
        print(f"{YELLOW}Starting local server…{RESET}")
        try:
            server_proc = _start_server()
            print(f"{GREEN}Server started.{RESET}\n")
        except Exception as e:
            print(f"{RED}Could not start server: {e}{RESET}")
            return 1

    # ── Define all checks ────────────────────────────────────────────────────
    # Checks that don't need the server
    static_checks: List[Tuple[str, Callable[[], CheckResult]]] = [
        ("Mandatory env vars (API_BASE_URL / MODEL_NAME / HF_TOKEN)", check_env_vars),
        ("inference.py in project root",                              check_inference_file_exists),
        ("openenv.yaml spec",                                         check_openenv_yaml),
        ("Dockerfile present & valid",                                check_dockerfile_exists),
        ("requirements.txt",                                          check_requirements_file),
        ("src.models importable",                                     check_models_importable),
        ("Task graders return [0, 1]",                                check_task_graders),
        ("inference.py imports OK",                                   check_inference_imports),
        ("[START]/[STEP]/[END] log format",                           check_log_format),
    ]

    # Checks that need the running server
    server_checks: List[Tuple[str, Callable[[], CheckResult]]] = [
        ("GET /health → 200",                       lambda: check_health(env_url)),
        ("POST /reset (easy/medium/hard)",           lambda: check_reset(env_url)),
        ("POST /step returns obs/reward/done/info",  lambda: check_step(env_url)),
        ("GET /state returns full state",            lambda: check_state(env_url)),
        ("Observations normalised to [0, 1]",        lambda: check_observations_normalised(env_url)),
        ("Episode terminates correctly",             lambda: check_episode_terminates(env_url)),
    ]

    all_checks = static_checks + server_checks

    # ── Run checks ────────────────────────────────────────────────────────────
    width = max(len(name) for name, _ in all_checks) + 4
    passed_count = 0
    total        = len(all_checks)
    results      = []

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  Supply Chain OpenEnv — Pre-Submission Validator{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")
    print(f"  Server URL : {env_url}")
    print(f"  Checks     : {total}\n")
    print(f"{'─'*70}")

    for name, fn in all_checks:
        try:
            passed, detail = fn()
        except Exception as exc:
            passed, detail = False, f"Exception: {exc}"

        mark = PASS_MARK if passed else FAIL_MARK
        print(f"  {mark}  {name:<{width}}  {detail}")
        results.append((name, passed, detail))
        if passed:
            passed_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    failed_count = total - passed_count
    print(f"\n{'─'*70}")
    overall = (f"{GREEN}{BOLD}ALL {total} CHECKS PASSED ✔{RESET}"
               if failed_count == 0
               else f"{RED}{BOLD}{failed_count}/{total} CHECKS FAILED ✖{RESET}")
    print(f"  {overall}")
    print(f"{'='*70}\n")

    # Machine-readable JSON summary
    summary = {
        "passed":  passed_count,
        "failed":  failed_count,
        "total":   total,
        "success": failed_count == 0,
        "checks":  [{"name": n, "passed": p, "detail": d} for n, p, d in results],
    }
    print(json.dumps(summary, indent=2))

    if server_proc:
        server_proc.terminate()

    return 0 if failed_count == 0 else 1


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply Chain OpenEnv — pre-submission validator")
    parser.add_argument(
        "--env-url",
        default=os.environ.get("ENV_BASE_URL", "http://localhost:7860"),
        help="Base URL of the OpenEnv server",
    )
    parser.add_argument(
        "--auto-server",
        action="store_true",
        help="Automatically start and stop the local server before/after checks",
    )
    args = parser.parse_args()

    sys.exit(run_validation(env_url=args.env_url, auto_server=args.auto_server))
