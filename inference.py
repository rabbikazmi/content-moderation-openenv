# inference.py — content-moderation-openenv
import os
import sys
import json
import time
from typing import Dict, Any, Optional, List
import httpx
from openai import OpenAI

# configuration

# load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Environment variables with defaults
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
API_KEY: str = os.getenv("API_KEY", "not-set")
MODEL_NAME: str = os.getenv("MODEL_NAME", "mistral")
HF_TOKEN: str = os.getenv("HF_TOKEN", "not-needed-for-ollama")
BASE_URL: str = os.getenv("BASE_URL", "http://localhost:7860")

# Valid moderation labels
VALID_LABELS: set = {"safe", "spam", "hate_speech", "violence", "adult_content"}

# System prompt for the LLM agent
SYSTEM_PROMPT: str = """You are a professional content moderator. Classify the given text into exactly ONE of:

- safe: Legitimate, non-harmful content. No violations.
- spam: Unwanted promotional, duplicate, scams, clickbait. E.g. "Buy now!", "Click here!", "Get rich quick"
- hate_speech: Content targeting individuals/groups by race, religion, nationality, gender, sexual orientation, or other protected characteristics. E.g. "People like X should disappear", "X are taking over"
- violence: Content depicting, threatening, or encouraging violence, harm, or physical aggression. E.g. "Go kill X", "We need to fight back", "Take them down"
- adult_content: Sexually explicit material, pornography, or sexual requests. E.g. "She's hot", "Send nudes", explicit sexual descriptions

KEY DISTINCTIONS:
- violence ≠ hate_speech: Violence focuses on HARM/AGGRESSION. Hate_speech focuses on IDENTITY/BELONGING.
  Example: "Destroy them" = violence. "They don't belong here" = hate_speech.
- adult_content ≠ compliments: "She's beautiful" = safe. "She's hot, send pics" = adult_content.

Respond ONLY in JSON with no markdown: {"label": "category", "confidence": 0.95}"""

# LLM call configuration
MAX_RETRIES: int = 3
RETRY_DELAY: int = 2
STEP_DELAY: float = 0.5

# logging functions


def log_start(task_id: str, env_name: str = "content-moderation") -> None:
    """Log task start marker in hackathon format.
    
    Args:
        task_id: Task identifier (task_1, task_2, or task_3)
        env_name: Environment name (default: content-moderation)
    """
    print(f"[START] task={task_id} env={env_name} model={MODEL_NAME}")
    sys.stdout.flush()


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Log step marker in hackathon format.
    
    Args:
        step: Step counter (1-indexed)
        action: Moderation label (safe, spam, etc)
        reward: Reward value
        done: Whether task is complete
        error: Error message if applicable (null if no error)
    """
    error_str = f'"{error}"' if error else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_str}"
    )
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log task end marker in hackathon format.
    
    Args:
        success: Whether task succeeded (score >= 0.5)
        steps: Total steps completed
        score: Final score (0.0-1.0)
        rewards: List of all rewards from steps
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END]   success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}")
    sys.stdout.flush()

# parsing and validation

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON response from LLM with markdown handling.
    
    Handles cases where LLM wraps JSON in markdown code blocks
    or adds extra explanation text around the JSON.
    
    Args:
        response_text: Raw text response from LLM
        
    Returns:
        Dict with "label" and "confidence" keys
    """
    if not response_text:
        return {"label": "safe", "confidence": 0.1}

    cleaned = response_text.strip()

    # strip markdown code blocks if present
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    # try to find json object in response
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError:
        pass

    # try to extract json from within text
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            parsed = json.loads(json_str)
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # fallback
    return {"label": "safe", "confidence": 0.1}


def validate_label(label: str) -> str:
    """Validate moderation label against allowed set.
    
    Args:
        label: Label to validate
        
    Returns:
        Validated label (or "safe" if invalid)
    """
    if label not in VALID_LABELS:
        return "safe"
    return label


def clamp_confidence(confidence: float) -> float:
    """Clamp confidence to [0.0, 1.0].
    
    Args:
        confidence: Raw confidence value
        
    Returns:
        Clamped confidence in [0.0, 1.0]
    """
    return max(0.0, min(1.0, float(confidence)))


# http client operations

def call_with_retry(func, retries: int = MAX_RETRIES, delay: int = RETRY_DELAY) -> Optional[Any]:
    """Call a function with retry logic.
    
    Args:
        func: Callable to execute
        retries: Number of retry attempts
        delay: Delay in seconds between retries
        
    Returns:
        Result of func() on successful execution, or None if all retries fail
    """
    last_exception = None
    
    for attempt in range(retries):
        try:
            return func()
        except httpx.HTTPStatusError as e:
            # stop retrying on 402 (rate limited by provider)
            if e.response.status_code == 402:
                raise
            last_exception = e
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception as e:
            last_exception = e
            if attempt < retries - 1:
                time.sleep(delay)
    
    return None

# main task execution

def run_task(
    task_id: str, 
    client: OpenAI, 
    http_client: httpx.Client
) -> Dict[str, Any]:
    """Run a single moderation task against the environment.
    
    Args:
        task_id: Task identifier ("task_1", "task_2", or "task_3")
        client: OpenAI-compatible client for LLM calls
        http_client: httpx.Client for FastAPI calls
        
    Returns:
        Dict with task results: cumulative_reward, total_samples, score, success, rewards list
    """
    log_start(task_id)
    
    step_count: int = 0
    cumulative_reward: float = 0.0
    rewards: List[float] = []
    done: bool = False
    error_occurred: Optional[str] = None
    
    try:
        # reset environment
        try:
            def reset_env():
                response = http_client.post(
                    f"{BASE_URL}/reset",
                    json={"task_id": task_id},
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
            
            observation = call_with_retry(reset_env, retries=3, delay=2)
            if observation is None:
                error_occurred = "Failed to reset environment"
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return {
                    "task_id": task_id,
                    "total_samples": 0,
                    "cumulative_reward": 0.0,
                    "score": 0.0,
                    "success": False,
                    "rewards": [],
                }
        except Exception as e:
            error_occurred = f"Reset failed: {str(e)}"
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return {
                "task_id": task_id,
                "total_samples": 0,
                "cumulative_reward": 0.0,
                "score": 0.0,
                "success": False,
                "rewards": [],
            }

        # main agent loop
        while not done:
            step_count += 1
            
            # extract observation
            text: str = observation.get("text", "N/A")
            context: str = observation.get("context", "unknown")

            # build prompt for llm
            user_prompt: str = f"Text: {text}\nContext: {context}\n\nClassify this content."
            
            # call llm with retry logic
            llm_response: Optional[str] = None
            llm_error: Optional[str] = None
            
            try:
                def get_llm_response():
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_prompt}
                        ],
                        max_tokens=150,
                        temperature=0.3,
                    )
                    return response.choices[0].message.content
                
                llm_response = call_with_retry(get_llm_response, retries=3, delay=2)
                
                if llm_response is None:
                    llm_error = "LLM call failed after retries"
                    llm_response = json.dumps({"label": "safe", "confidence": 0.1})
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 402:
                    llm_error = "Rate limited (402)"
                else:
                    llm_error = f"HTTP {e.response.status_code}"
                llm_response = json.dumps({"label": "safe", "confidence": 0.1})
            except Exception as e:
                llm_error = f"LLM error: {str(e)[:50]}"
                llm_response = json.dumps({"label": "safe", "confidence": 0.1})

            # parse and validate llm output
            decision: Dict[str, Any] = parse_llm_response(llm_response)
            label: str = validate_label(decision.get("label", "safe"))
            confidence: float = clamp_confidence(decision.get("confidence", 0.1))

            # submit action to environment
            step_error: Optional[str] = None
            reward: float = 0.0
            
            try:
                def submit_step():
                    response = http_client.post(
                        f"{BASE_URL}/step",
                        json={"label": label, "confidence": confidence},
                        timeout=10.0
                    )
                    response.raise_for_status()
                    return response.json()

                step_result = call_with_retry(submit_step, retries=3, delay=2)
                
                if step_result is None:
                    step_error = "Step submission failed"
                    reward = 0.0
                    done = True
                else:
                    reward = float(step_result.get("reward", 0.0))
                    done = step_result.get("done", False)
                    
            except Exception as e:
                step_error = f"Step failed: {str(e)[:50]}"
                reward = 0.0
                done = True

            # accumulate reward
            cumulative_reward += reward
            rewards.append(reward)
            
            # log step
            log_step(
                step=step_count,
                action=label,
                reward=reward,
                done=done,
                error=step_error or llm_error
            )

            # get next observation if not done
            if not done and step_result is not None:
                next_obs = step_result.get("observation")
                if next_obs:
                    observation = next_obs
                else:
                    done = True

            # rate limiting
            time.sleep(STEP_DELAY)

    except Exception as e:
        # catch any unexpected exception
        error_occurred = f"Unexpected error: {str(e)[:50]}"
        sys.stderr.write(f"ERROR: {error_occurred}\n")

    finally:
        # calculate score
        total_samples: int = step_count
        max_possible: float = float(total_samples) if total_samples > 0 else 1.0
        score: float = (cumulative_reward / max_possible) if max_possible > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success: bool = score >= 0.5
        
        # always emit [END] marker
        log_end(
            success=success,
            steps=total_samples,
            score=score,
            rewards=rewards
        )

    return {
        "task_id": task_id,
        "total_samples": total_samples,
        "cumulative_reward": cumulative_reward,
        "score": score,
        "success": success,
        "rewards": rewards,
    }

# main execution

def main() -> None:
    """Main execution function.
    
    Checks server health, runs all 3 tasks, prints final summary.
    Exits with code 0 on success, code 1 if server is unreachable.
    Never exits with unhandled exception.
    """
    try:
        # initialize clients using environment variables
        # Use the API_KEY and API_BASE_URL injected by the LiteLLM proxy
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        http_client = httpx.Client(timeout=60.0)

        # health check
        try:
            response = http_client.get(f"{BASE_URL}/health", timeout=5.0)
            response.raise_for_status()
            health = response.json()
        except Exception as e:
            sys.stderr.write(
                f"ERROR: Server not reachable at {BASE_URL}\n"
                f"Details: {str(e)}\n"
                f"Start it with: python main.py\n"
            )
            http_client.close()
            sys.exit(1)

        # run all 3 tasks
        results: List[Dict[str, Any]] = []
        task_ids: List[str] = ["task_1", "task_2", "task_3"]
        
        for task_id in task_ids:
            try:
                result = run_task(task_id, client, http_client)
                results.append(result)
            except Exception as e:
                sys.stderr.write(f"ERROR during {task_id}: {str(e)}\n")
                result = {
                    "task_id": task_id,
                    "total_samples": 0,
                    "cumulative_reward": 0.0,
                    "score": 0.0,
                    "success": False,
                    "rewards": [],
                }
                results.append(result)

        http_client.close()

        # print summary table
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        overall_reward: float = 0.0
        overall_max: float = 0.0
        
        for result in results:
            task_id = result["task_id"]
            reward = result["cumulative_reward"]
            total = result["total_samples"]
            score_pct = (result["score"] * 100) if result["total_samples"] > 0 else 0.0
            
            print(f"{task_id}: {reward:.1f}/{float(total):.1f} ({score_pct:.1f}%)")
            
            overall_reward += reward
            overall_max += float(total)
        
        overall_pct = (overall_reward / overall_max * 100) if overall_max > 0 else 0.0
        print("-" * 60)
        print(f"Overall: {overall_reward:.1f}/{overall_max:.1f} ({overall_pct:.1f}%)")
        print("=" * 60)

    except Exception as e:
        # final safety net: catch any unhandled exception
        sys.stderr.write(f"FATAL: Unhandled exception: {str(e)}\n")
        sys.exit(0)  # Exit gracefully with code 0


if __name__ == "__main__":
    main()