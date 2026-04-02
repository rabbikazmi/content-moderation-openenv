# Step 3 — main.py — content-moderation-openenv

"""
FastAPI server for Content Moderation OpenEnv - Scaler × Meta PyTorch Hackathon

This server provides REST API endpoints to interact with a content moderation
reinforcement learning environment. The environment trains agents to classify
user-generated content into moderation categories (safe, spam, hate_speech,
violence, adult_content) across three difficulty levels.

Port: 7860 (hardcoded for Hugging Face Spaces)
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging

# Import models and environment
from models import (
    ContentObservation,
    ModerationAction,
    StepResult,
    EnvironmentState,
    TaskSpec,
)
from environment import ContentModerationEnv

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Content Moderation OpenEnv",
    version="1.0.0",
    description="A real-world content moderation environment for AI agents",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Single global environment instance shared across all requests
env = ContentModerationEnv()

# ============================================================================
# PYDANTIC REQUEST/RESPONSE MODELS
# ============================================================================


class ResetRequest(BaseModel):
    """Request body for POST /reset endpoint."""

    task_id: str = Field(
        default="task_1",
        description="Task identifier: task_1 (easy), task_2 (medium), task_3 (hard)",
        examples=["task_1", "task_2", "task_3"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {"task_id": "task_1"},
        }
    }


class HealthResponse(BaseModel):
    """Response for GET /health endpoint."""

    status: str = Field(..., description="Health status")
    environment: str = Field(..., description="Environment name")
    version: str = Field(..., description="API version")
    tasks_available: int = Field(..., description="Number of available tasks")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "environment": "content-moderation",
                "version": "1.0.0",
                "tasks_available": 3,
            }
        }
    }


class WelcomeResponse(BaseModel):
    """Response for GET / endpoint."""

    environment_name: str = Field(..., description="Name of the environment")
    description: str = Field(..., description="Description of the environment")
    available_endpoints: List[str] = Field(
        ..., description="List of available API endpoints"
    )
    documentation: str = Field(..., description="URL to interactive documentation")
    note: str = Field(..., description="Note about API documentation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "environment_name": "Content Moderation OpenEnv",
                "description": "A real-world content moderation environment for AI agents",
                "available_endpoints": [
                    "GET /",
                    "GET /health",
                    "GET /tasks",
                    "GET /state",
                    "POST /reset",
                    "POST /step",
                ],
                "documentation": "http://localhost:7860/docs",
                "note": "Visit /docs for interactive API documentation",
            }
        }
    }


# ============================================================================
# MIDDLEWARE
# ============================================================================


class LoggingMiddleware:
    """Middleware to log all requests and responses."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]

        async def send_with_logging(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                print(f"{method} {path} → {status_code}")
            await send(message)

        await self.app(scope, receive, send_with_logging)


# Add middleware
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STARTUP EVENT
# ============================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """Print startup message on server start."""
    print("\n" + "=" * 70)
    print("Content Moderation OpenEnv server started on port 7860")
    print("Available tasks: task_1 (easy), task_2 (medium), task_3 (hard)")
    print("Documentation: http://localhost:7860/docs")
    print("=" * 70 + "\n")
    logger.info("Content Moderation OpenEnv server started")


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get(
    "/",
    response_model=WelcomeResponse,
    tags=["Welcome"],
    summary="Welcome endpoint",
    description="Get information about the Content Moderation OpenEnv API",
)
async def root() -> WelcomeResponse:
    """
    Welcome endpoint providing API information.

    Returns:
        WelcomeResponse: Contains environment name, description, endpoints, and docs URL
    """
    return WelcomeResponse(
        environment_name="Content Moderation OpenEnv",
        description="A real-world content moderation environment for AI agents - Scaler × Meta PyTorch Hackathon",
        available_endpoints=[
            "GET /",
            "GET /health",
            "GET /tasks",
            "GET /state",
            "POST /reset",
            "POST /step",
        ],
        documentation="http://localhost:7860/docs",
        note="Visit /docs for interactive API documentation and to test endpoints",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
    description="Returns HTTP 200 if server is running. Required for Hugging Face Spaces validator.",
)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    This endpoint MUST always return HTTP 200. It is pinged by the automated
    Hugging Face Spaces validator to verify the server is running. Failure here
    results in immediate disqualification.

    Returns:
        HealthResponse: Always returns {"status": "ok", "environment": "content-moderation", ...}
    """
    return HealthResponse(
        status="ok",
        environment="content-moderation",
        version="1.0.0",
        tasks_available=3,
    )


@app.get(
    "/tasks",
    response_model=List[TaskSpec],
    tags=["Tasks"],
    summary="Get all available tasks",
    description="Returns list of all available moderation tasks (works before /reset)",
)
async def get_tasks() -> List[TaskSpec]:
    """
    Retrieve all available moderation tasks.

    This endpoint works even before /reset is called and has no state dependency.
    Returns three tasks: easy, medium, and hard difficulty levels.

    Returns:
        List[TaskSpec]: Array of 3 task specifications with id, name, description, difficulty, num_samples
    """
    try:
        tasks = env.get_tasks()
        return tasks
    except Exception as e:
        logger.error(f"Error in /tasks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}"
        )


@app.post(
    "/reset",
    response_model=ContentObservation,
    tags=["Environment"],
    summary="Reset environment and start new episode",
    description="Reset the environment and start a new episode with specified task",
)
async def reset(request: Optional[ResetRequest] = Body(default=None)) -> ContentObservation:
    """
    Reset the environment for a new episode.

    Validates the task_id against valid options (task_1, task_2, task_3), resets
    all episode state, and returns the first content sample to moderate.

    Args:
        request: ResetRequest with optional task_id (defaults to task_1). Body is optional.

    Returns:
        ContentObservation: First content sample in the task

    Raises:
        HTTPException 422: If task_id is not one of the valid options
    """
    try:
        # If no body is provided, use default ResetRequest with task_id="task_1"
        if request is None:
            request = ResetRequest()
        observation = env.reset(request.task_id)
        return observation
    except ValueError as e:
        logger.warning(f"Invalid task_id in /reset: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail="Invalid task_id. Must be one of: task_1, task_2, task_3",
        )
    except Exception as e:
        logger.error(f"Error in /reset: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}"
        )


@app.post(
    "/step",
    response_model=StepResult,
    tags=["Environment"],
    summary="Execute one step in the environment",
    description="Submit a moderation action and receive reward, next observation, and episode status",
)
async def step(action: Optional[ModerationAction] = Body(default=None)) -> StepResult:
    """
    Execute one step in the moderation environment.

    The agent submits a moderation action (label and confidence score). The action
    is compared against ground truth, and a reward is computed based on correctness
    and confidence level. Returns the next observation, reward, done status, and info.

    Reward logic:
    - Correct label + high confidence (≥0.7): reward = 1.0
    - Correct label + low confidence (<0.7): reward = 0.7
    - Wrong label + uncertain (<0.4 confidence): reward = 0.3
    - Wrong label + confident (≥0.4): reward = 0.0

    Args:
        action: ModerationAction with label and confidence score (required)

    Returns:
        StepResult: Contains next observation (if not done), reward, done flag, info dict

    Raises:
        HTTPException 400: If action body not provided
        HTTPException 400: If environment not initialized (call /reset first)
        HTTPException 400: If episode already done (call /reset to start new)
        HTTPException 422: If label not valid enum value
    """
    # Action body is required for /step
    if action is None:
        raise HTTPException(
            status_code=400,
            detail="Action required. Provide label and confidence.",
        )
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        if "not initialized" in str(e):
            logger.warning(f"Step called before reset: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call POST /reset first.",
            )
        elif "done" in str(e):
            logger.warning(f"Step called after episode done: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Episode finished. Call POST /reset to start a new episode.",
            )
        else:
            logger.error(f"RuntimeError in /step: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        logger.warning(f"ValueError in /step: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in /step: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}"
        )


@app.get(
    "/state",
    response_model=EnvironmentState,
    tags=["Environment"],
    summary="Get current environment state",
    description="Returns current state including task_id, progress, reward, episode status",
)
async def get_state() -> EnvironmentState:
    """
    Get the current state of the environment.

    Returns complete state information including which task is active, current
    sample index, total samples, cumulative reward, done status, and initialized flag.

    Returns:
        EnvironmentState: Current state with all required fields

    Raises:
        HTTPException 400: If environment not initialized (call /reset first)
    """
    try:
        if not env.initialized:
            raise RuntimeError(
                "Environment not initialized. Call POST /reset first."
            )
        state = env.state()
        return state
    except RuntimeError as e:
        logger.warning(f"State called before reset: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )
    except Exception as e:
        logger.error(f"Error in /state: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}"
        )


# ============================================================================
# ENTRY POINT FOR CONSOLE SCRIPT
# ============================================================================

def start_server():
    """
    Entry point for the 'serve' console script.
    Called by: serve (after pip install)
    Runs the FastAPI application on 0.0.0.0:7860.
    """
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)


# ============================================================================
# MAIN BLOCK
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
