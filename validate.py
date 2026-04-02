import os
import sys
import json
import yaml
import httpx
from typing import Dict, Any, List, Tuple


BASE_URL = "http://localhost:7860"
CHECKS_PASSED = 0
CHECKS_FAILED = 0
FAILED_CHECKS: List[Tuple[int, str]] = []


def check_print(check_num: int, description: str, passed: bool, error_msg: str = "") -> None:
    """Print a check result in formatted style.
    
    Args:
        check_num: Check number (1-20)
        description: Description of the check
        passed: Whether the check passed
        error_msg: Error message if failed (optional)
    """
    global CHECKS_PASSED, CHECKS_FAILED, FAILED_CHECKS
    
    status = "PASS" if passed else "FAIL"
    print(f"CHECK {check_num:2d}: {status} — {description}")
    
    if not passed:
        if error_msg:
            print(f"           └─ {error_msg}")
        CHECKS_FAILED += 1
        FAILED_CHECKS.append((check_num, description))
    else:
        CHECKS_PASSED += 1


def validate_api() -> None:
    """Validate all API endpoints and responses.
    
    Checks 1-13: Health, reset, step, state, tasks endpoints.
    Handles connection errors gracefully.
    """
    global BASE_URL
    
    print("\n" + "="*60)
    print(" API ENDPOINT VALIDATION (Checks 1-13)")
    print("="*60 + "\n")
    
    client = httpx.Client(timeout=10.0)
    
    # CHECK 1: GET /health returns HTTP 200
    try:
        response = client.get(f"{BASE_URL}/health")
        check_print(1, "GET /health returns HTTP 200", response.status_code == 200, 
                   f"Got {response.status_code}" if response.status_code != 200 else "")
    except Exception as e:
        check_print(1, "GET /health returns HTTP 200", False, str(e)[:80])
        client.close()
        return  # Can't continue if server is down
    
    # CHECK 2: GET /health returns {"status": "ok"}
    try:
        response = client.get(f"{BASE_URL}/health")
        data = response.json()
        passed = data.get("status") == "ok"
        check_print(2, "GET /health returns {\"status\": \"ok\"}", passed,
                   f"Got {data}" if not passed else "")
    except Exception as e:
        check_print(2, "GET /health returns {\"status\": \"ok\"}", False, str(e)[:80])
    
    # CHECK 3: POST /reset with task_1 returns HTTP 200
    reset_response = None
    try:
        response = client.post(f"{BASE_URL}/reset", json={"task_id": "task_1"})
        reset_response = response.json()
        check_print(3, "POST /reset with task_id=\"task_1\" returns HTTP 200", 
                   response.status_code == 200,
                   f"Got {response.status_code}" if response.status_code != 200 else "")
    except Exception as e:
        check_print(3, "POST /reset with task_id=\"task_1\" returns HTTP 200", False, str(e)[:80])
        client.close()
        return
    
    # CHECK 4: POST /reset response contains "text" field
    try:
        passed = reset_response is not None and "text" in reset_response
        check_print(4, "POST /reset response contains \"text\" field", passed,
                   f"Fields: {list(reset_response.keys()) if reset_response else 'None'}" if not passed else "")
    except Exception as e:
        check_print(4, "POST /reset response contains \"text\" field", False, str(e)[:80])
    
    # CHECK 5: POST /reset response contains "context" field
    try:
        passed = reset_response is not None and "context" in reset_response
        check_print(5, "POST /reset response contains \"context\" field", passed,
                   f"Fields: {list(reset_response.keys()) if reset_response else 'None'}" if not passed else "")
    except Exception as e:
        check_print(5, "POST /reset response contains \"context\" field", False, str(e)[:80])
    
    # CHECK 6: POST /step returns HTTP 200
    step_response = None
    try:
        response = client.post(f"{BASE_URL}/step", 
                              json={"label": "spam", "confidence": 0.9})
        step_response = response.json()
        check_print(6, "POST /step returns HTTP 200", response.status_code == 200,
                   f"Got {response.status_code}" if response.status_code != 200 else "")
    except Exception as e:
        check_print(6, "POST /step returns HTTP 200", False, str(e)[:80])
        client.close()
        return
    
    # CHECK 7: POST /step response contains "reward" field
    try:
        passed = step_response is not None and "reward" in step_response
        check_print(7, "POST /step response contains \"reward\" field", passed,
                   f"Fields: {list(step_response.keys()) if step_response else 'None'}" if not passed else "")
    except Exception as e:
        check_print(7, "POST /step response contains \"reward\" field", False, str(e)[:80])
    
    # CHECK 8: POST /step reward value is between 0.0 and 1.0
    try:
        reward = step_response.get("reward", -1) if step_response else -1
        passed = isinstance(reward, (int, float)) and 0.0 <= reward <= 1.0
        check_print(8, "POST /step reward is between 0.0 and 1.0", passed,
                   f"Got {reward}" if not passed else "")
    except Exception as e:
        check_print(8, "POST /step reward is between 0.0 and 1.0", False, str(e)[:80])
    
    # CHECK 9: GET /state returns HTTP 200
    state_response = None
    try:
        response = client.get(f"{BASE_URL}/state")
        state_response = response.json()
        check_print(9, "GET /state returns HTTP 200", response.status_code == 200,
                   f"Got {response.status_code}" if response.status_code != 200 else "")
    except Exception as e:
        check_print(9, "GET /state returns HTTP 200", False, str(e)[:80])
        client.close()
        return
    
    # CHECK 10: GET /state response contains "task_id" field
    try:
        passed = state_response is not None and "task_id" in state_response
        check_print(10, "GET /state response contains \"task_id\" field", passed,
                    f"Fields: {list(state_response.keys()) if state_response else 'None'}" if not passed else "")
    except Exception as e:
        check_print(10, "GET /state response contains \"task_id\" field", False, str(e)[:80])
    
    # CHECK 11: GET /tasks returns HTTP 200
    tasks_response = None
    try:
        response = client.get(f"{BASE_URL}/tasks")
        tasks_response = response.json()
        check_print(11, "GET /tasks returns HTTP 200", response.status_code == 200,
                   f"Got {response.status_code}" if response.status_code != 200 else "")
    except Exception as e:
        check_print(11, "GET /tasks returns HTTP 200", False, str(e)[:80])
        client.close()
        return
    
    # CHECK 12: GET /tasks returns exactly 3 tasks
    try:
        num_tasks = len(tasks_response) if tasks_response else 0
        passed = num_tasks == 3
        check_print(12, "GET /tasks returns exactly 3 tasks", passed,
                   f"Got {num_tasks}" if not passed else "")
    except Exception as e:
        check_print(12, "GET /tasks returns exactly 3 tasks", False, str(e)[:80])
    
    # CHECK 13: All task IDs are task_1, task_2, task_3
    try:
        task_ids = [t.get("task_id") for t in tasks_response] if tasks_response else []
        expected = {"task_1", "task_2", "task_3"}
        passed = set(task_ids) == expected
        check_print(13, "All task IDs are task_1, task_2, task_3", passed,
                   f"Got {task_ids}" if not passed else "")
    except Exception as e:
        check_print(13, "All task IDs are task_1, task_2, task_3", False, str(e)[:80])
    
    client.close()


def validate_files() -> None:
    """Validate local files and specifications.
    
    Checks 14-20: File existence and openenv.yaml validity.
    """
    print("\n" + "="*60)
    print(" FILE & SPECIFICATION VALIDATION (Checks 14-20)")
    print("="*60 + "\n")
    
    # CHECK 14: openenv.yaml exists
    try:
        passed = os.path.isfile("openenv.yaml")
        check_print(14, "openenv.yaml exists in current directory", passed,
                   "File not found" if not passed else "")
        if not passed:
            return
    except Exception as e:
        check_print(14, "openenv.yaml exists in current directory", False, str(e)[:80])
        return
    
    # CHECK 15: openenv.yaml is valid YAML
    spec = None
    try:
        with open("openenv.yaml", "r") as f:
            spec = yaml.safe_load(f)
        passed = spec is not None and isinstance(spec, dict)
        check_print(15, "openenv.yaml is valid YAML", passed,
                   "Failed to parse YAML" if not passed else "")
    except Exception as e:
        check_print(15, "openenv.yaml is valid YAML", False, str(e)[:80])
        return
    
    # CHECK 16: openenv.yaml contains required keys
    try:
        required_keys = {"name", "version", "description", "observation_space", 
                        "action_space", "tasks"}
        missing = required_keys - set(spec.keys())
        passed = len(missing) == 0
        check_print(16, "openenv.yaml contains all required keys", passed,
                   f"Missing: {missing}" if missing else "")
    except Exception as e:
        check_print(16, "openenv.yaml contains all required keys", False, str(e)[:80])
    
    # CHECK 17: inference.py exists
    try:
        passed = os.path.isfile("inference.py")
        check_print(17, "inference.py exists at project root", passed,
                   "File not found" if not passed else "")
    except Exception as e:
        check_print(17, "inference.py exists at project root", False, str(e)[:80])
    
    # CHECK 18: Dockerfile exists
    try:
        passed = os.path.isfile("Dockerfile")
        check_print(18, "Dockerfile exists at project root", passed,
                   "File not found" if not passed else "")
    except Exception as e:
        check_print(18, "Dockerfile exists at project root", False, str(e)[:80])
    
    # CHECK 19: requirements.txt exists
    try:
        passed = os.path.isfile("requirements.txt")
        check_print(19, "requirements.txt exists at project root", passed,
                   "File not found" if not passed else "")
    except Exception as e:
        check_print(19, "requirements.txt exists at project root", False, str(e)[:80])
    
    # CHECK 20: inference_results.json exists
    try:
        passed = os.path.isfile("inference_results.json")
        check_print(20, "inference_results.json exists (proof inference was run)", passed,
                   "File not found" if not passed else "")
    except Exception as e:
        check_print(20, "inference_results.json exists (proof inference was run)", False, str(e)[:80])


def print_summary() -> None:
    """Print final validation summary and status."""
    total = CHECKS_PASSED + CHECKS_FAILED
    status = "READY TO SUBMIT" if CHECKS_FAILED == 0 else "NOT READY"
    
    print("\n" + "="*60)
    print(" VALIDATION SUMMARY")
    print("="*60)
    print(f" {CHECKS_PASSED} / {total} checks passed")
    print(f" Status: {status}")
    
    if CHECKS_FAILED > 0:
        print("\n Failed checks:")
        for check_num, description in FAILED_CHECKS:
            print(f"  - Check {check_num}: {description}")
    
    print("="*60 + "\n")


def main() -> int:
    """Main entry point for validation script.
    
    Runs all 20 checks and exits with appropriate code.
    
    Returns:
        0 if all checks pass, 1 if any fail
    """
    print("\n" + "="*60)
    print(" CONTENT MODERATION OPENENV — PRE-SUBMISSION VALIDATION")
    print("="*60)
    print(f" Server: {BASE_URL}\n")
    
    validate_api()
    validate_files()
    print_summary()
    
    return 0 if CHECKS_FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
