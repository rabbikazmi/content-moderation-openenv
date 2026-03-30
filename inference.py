# Step 4 — inference.py — content-moderation-openenv

"""
Inference script for Content Moderation OpenEnv - Scaler × Meta PyTorch Hackathon

This script runs a content moderation agent against a FastAPI server using
an LLM (via OpenAI-compatible client) to classify user-generated content.
The agent runs through all 3 difficulty levels (easy, medium, hard) and
produces a detailed performance report.

Requirements:
- FastAPI server running on http://localhost:7860
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Completion time: < 20 minutes on 2 vCPU / 8GB RAM
"""

import os
import sys
import json
import time
import httpx
from typing import Dict, Any
from openai import OpenAI

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# CONFIGURATION & VALIDATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate required environment variables
missing_vars = []
if not API_BASE_URL:
    missing_vars.append("API_BASE_URL")
if not MODEL_NAME:
    missing_vars.append("MODEL_NAME")
if not HF_TOKEN:
    missing_vars.append("HF_TOKEN")

if missing_vars:
    print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set: API_BASE_URL, MODEL_NAME, HF_TOKEN in your .env file")
    sys.exit(1)

# Base URL for FastAPI server
BASE_URL = "http://localhost:7860"

# Valid moderation labels
VALID_LABELS = {"safe", "spam", "hate_speech", "violence", "adult_content"}

# System prompt for the LLM agent
SYSTEM_PROMPT = """You are a professional content moderator for a social media platform.
You will be given a piece of text and the platform context it was posted on.
Your job is to classify the content into exactly one of these categories:
- safe: normal, acceptable content
- spam: unsolicited promotional or repetitive content
- hate_speech: content targeting people based on identity
- violence: content promoting or depicting physical harm
- adult_content: sexually explicit or inappropriate content

You must also provide a confidence score from 0.0 to 1.0 indicating
how certain you are.

IMPORTANT: Respond ONLY with a valid JSON object. No explanation, no markdown, no code blocks.
Format: {"label": "one_of_the_categories", "confidence": 0.95}"""

# ============================================================================
# UTILITIES
# ============================================================================


def call_with_retry(func, retries: int = 3, delay: int = 2) -> Any:
    """
    Call a function with retry logic.

    Args:
        func: Callable to execute
        retries: Number of retry attempts (default 3)
        delay: Delay in seconds between retries (default 2)

    Returns:
        Result of func() on successful execution

    Raises:
        Exception: If all retry attempts fail
    """
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt + 1}/{retries} after error: {str(e)[:80]}")
            time.sleep(delay)


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON response from LLM.

    Handles cases where LLM wraps JSON in markdown code blocks
    or adds extra explanation text around the JSON.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Dict with "label" and "confidence" keys
    """
    if not response_text:
        return {"label": "safe", "confidence": 0.1}

    # Strip markdown code blocks if present
    cleaned = response_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # Try to find JSON object in the response
    try:
        # First try direct parse
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from within the text
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            parsed = json.loads(json_str)
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Final fallback
    print(f"  Warning: Could not parse LLM response: {response_text[:100]}")
    return {"label": "safe", "confidence": 0.1}


def validate_label(label: str) -> str:
    """
    Validate moderation label against allowed set.

    Args:
        label: Label to validate

    Returns:
        Validated label (or "safe" if invalid)
    """
    if label not in VALID_LABELS:
        print(f"  Warning: Invalid label '{label}', defaulting to 'safe'")
        return "safe"
    return label


# ============================================================================
# MAIN AGENT LOGIC
# ============================================================================


def run_task(
    task_id: str,
    client: OpenAI,
    http_client: httpx.Client
) -> Dict[str, Any]:
    """
    Run a single moderation task against the environment.

    Args:
        task_id: Task identifier ("task_1", "task_2", or "task_3")
        client: OpenAI-compatible client for LLM calls
        http_client: httpx.Client for FastAPI calls

    Returns:
        Dict with task results including score and percentage
    """
    print(f"\n{'='*60}")
    print(f"  Running Task: {task_id}")
    print(f"{'='*60}")

    # ── Reset environment ──────────────────────────────────────
    def reset_env():
        response = http_client.post(
            f"{BASE_URL}/reset",
            json={"task_id": task_id}
        )
        response.raise_for_status()
        return response.json()

    observation = call_with_retry(reset_env)
    print(f"  Environment reset. Starting moderation...\n")

    step_count = 0
    done = False

    # ── Main agent loop ────────────────────────────────────────
    while not done:

        # Build prompt — combine system + user for Mistral compatibility
        text = observation.get("text", "N/A")
        context = observation.get("context", "unknown")

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Text to moderate: {text}\n"
            f"Platform context: {context}\n"
            f"Classify this content."
        )

        # ── Call LLM ───────────────────────────────────────────
        def get_llm_response():
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_tokens=20,
                temperature=0.1,
            )
            return response.choices[0].message.content

        try:
            llm_response = call_with_retry(get_llm_response)
        except Exception as e:
            print(f"  ERROR: Failed to get LLM response: {str(e)}")
            llm_response = json.dumps({"label": "safe", "confidence": 0.1})

        # ── Parse and validate LLM output ─────────────────────
        decision = parse_llm_response(llm_response)
        label = validate_label(decision.get("label", "safe"))
        confidence = float(decision.get("confidence", 0.1))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # ── Submit action to environment ───────────────────────
        def submit_step():
            response = http_client.post(
                f"{BASE_URL}/step",
                json={"label": label, "confidence": confidence},
            )
            response.raise_for_status()
            return response.json()

        try:
            step_result = call_with_retry(submit_step)
        except Exception as e:
            print(f"  ERROR: Failed to submit step: {str(e)}")
            break

        # ── Extract results ────────────────────────────────────
        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        info = step_result.get("info", {})
        ground_truth = info.get("ground_truth", "unknown")
        step_count += 1

        # ── Print progress ─────────────────────────────────────
        print(
            f"  Sample {step_count:>2}: "
            f"predicted={label:15s} ({confidence:.2f}) "
            f"| truth={ground_truth:15s} "
            f"| reward={reward:.1f}"
        )

        # ── Advance observation ────────────────────────────────
        if not done:
            next_obs = step_result.get("observation")
            if next_obs:
                observation = next_obs
            else:
                # Fallback: fetch state and break if something is wrong
                print("  Warning: No next observation received.")
                break

        # ── Rate limiting ──────────────────────────────────────
        time.sleep(1)

    # ── Get final state ────────────────────────────────────────
    def get_final_state():
        response = http_client.get(f"{BASE_URL}/state")
        response.raise_for_status()
        return response.json()

    try:
        final_state = call_with_retry(get_final_state)
        cumulative_reward = final_state.get("cumulative_reward", 0.0)
        total_samples = final_state.get("total_samples", step_count)
    except Exception as e:
        print(f"  WARNING: Could not get final state: {str(e)}")
        cumulative_reward = 0.0
        total_samples = step_count

    max_possible = float(total_samples) * 1.0
    score_pct = (
        (cumulative_reward / max_possible * 100)
        if max_possible > 0 else 0.0
    )

    print(f"\n Task {task_id} complete! "
          f"Score: {cumulative_reward:.1f} / {max_possible:.1f} "
          f"({score_pct:.1f}%)")

    return {
        "task_id": task_id,
        "total_samples": total_samples,
        "cumulative_reward": cumulative_reward,
        "max_possible_reward": max_possible,
        "score_percentage": score_pct,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main() -> None:
    """
    Main execution function.

    Checks server health, runs all 3 tasks, prints final results table.
    Exits with code 0 on success, code 1 on failure.
    """

    # ── Banner ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Content Moderation OpenEnv — Inference Script")
    print("=" * 60)
    print(f"Model:  {MODEL_NAME}")
    print(f"API:    {API_BASE_URL}")
    print(f"Tasks:  task_1 (easy) | task_2 (medium) | task_3 (hard)")
    print("=" * 60)

    # ── Initialize clients ─────────────────────────────────────
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    http_client = httpx.Client(timeout=60.0)

    # ── Health check ───────────────────────────────────────────
    print("\nChecking server health...")
    try:
        response = http_client.get(f"{BASE_URL}/health")
        response.raise_for_status()
        health = response.json()
        print(f"[OK] Server is running (status: {health.get('status', 'ok')})\n")
    except Exception as e:
        print(
            f"\n[ERROR]: Server not running on port 7860.\n"
            f"  Start it with: python main.py\n"
            f"  Error: {str(e)}"
        )
        sys.exit(1)

    # ── Run all 3 tasks ────────────────────────────────────────
    results = []
    for task_id in ["task_1", "task_2", "task_3"]:
        try:
            result = run_task(task_id, client, http_client)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] running {task_id}: {str(e)}")
            http_client.close()
            sys.exit(1)

    http_client.close()

    # ── Save results to JSON ────────────────────────────────────
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "tasks": results,
        "summary": {
            "total_cumulative_reward": 0.0,
            "total_max_reward": 0.0,
            "overall_score_percentage": 0.0,
        }
    }
    
    # Calculate summary totals
    for result in results:
        results_data["summary"]["total_cumulative_reward"] += result["cumulative_reward"]
        results_data["summary"]["total_max_reward"] += result["max_possible_reward"]
    
    if results_data["summary"]["total_max_reward"] > 0:
        results_data["summary"]["overall_score_percentage"] = (
            results_data["summary"]["total_cumulative_reward"] / 
            results_data["summary"]["total_max_reward"] * 100
        )
    
    with open("inference_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[OK] Results saved to inference_results.json\n")

    # ── Final results table ────────────────────────────────────
    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)

    task_names = {
        "task_1": "Task 1 (Easy)   ",
        "task_2": "Task 2 (Medium) ",
        "task_3": "Task 3 (Hard)   ",
    }

    total_cumulative = 0.0
    total_max = 0.0

    for result in results:
        name = task_names.get(result["task_id"], result["task_id"])
        cr = result["cumulative_reward"]
        mp = result["max_possible_reward"]
        pct = result["score_percentage"]
        print(f" {name}: {cr:5.1f} / {mp:5.1f}   ({pct:5.1f}%)")
        total_cumulative += cr
        total_max += mp

    overall_pct = (
        (total_cumulative / total_max * 100)
        if total_max > 0 else 0.0
    )

    print("-" * 60)
    print(f" OVERALL SCORE      : "
          f"{total_cumulative:5.1f} / {total_max:5.1f}   "
          f"({overall_pct:5.1f}%)")
    print("=" * 60 + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()