# Step 3 — models.py — content-moderation-openenv

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Dict, Any, List


class ModerationLabel(str, Enum):
    """Enumeration of moderation action labels."""
    safe = "safe"
    spam = "spam"
    hate_speech = "hate_speech"
    violence = "violence"
    adult_content = "adult_content"


class ContentObservation(BaseModel):
    """Observation of content to be moderated."""
    content_id: str = Field(..., description="Unique identifier for the content")
    text: str = Field(..., description="The text content to moderate")
    context: str = Field(..., description="Context of the content (e.g., twitter, forum)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "content_id": "content_123",
                "text": "Sample content text",
                "context": "twitter",
                "metadata": {"user_id": "user_456"},
            }
        }
    }


class ModerationAction(BaseModel):
    """Action taken by the moderation agent."""
    label: ModerationLabel = Field(..., description="The moderation label/action")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the moderation decision",
    )
    reasoning: Optional[str] = Field(default=None, description="Explanation for the action")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "label": "safe",
                "confidence": 0.95,
                "reasoning": "Content does not violate policies",
            }
        }
    }


class StepResult(BaseModel):
    """Result of one step in the environment."""
    observation: Optional[ContentObservation] = Field(default=None, description="Next observation (None if episode done)")
    reward: float = Field(..., description="Reward for this step")
    done: bool = Field(..., description="Whether the episode is done")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "observation": {
                    "content_id": "content_124",
                    "text": "Next content",
                    "source": "twitter",
                    "metadata": {},
                },
                "reward": 1.0,
                "done": False,
                "info": {"step": 1},
            }
        }
    }


class EnvironmentState(BaseModel):
    """Current state of the environment."""
    task_id: str = Field(..., description="Current task identifier")
    current_index: int = Field(..., ge=0, description="Current sample index")
    total_samples: int = Field(..., ge=0, description="Total samples in current task")
    cumulative_reward: float = Field(..., description="Cumulative reward for the episode")
    done: bool = Field(..., description="Whether episode is done")
    initialized: bool = Field(..., description="Whether environment is initialized")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task_1",
                "current_index": 1,
                "total_samples": 10,
                "cumulative_reward": 1.0,
                "done": False,
                "initialized": True,
            }
        }
    }


class TaskSpec(BaseModel):
    """Specification of a task in the environment."""
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Task description")
    difficulty: str = Field(..., description="Task difficulty level (easy/medium/hard)")
    num_samples: int = Field(..., ge=0, description="Number of samples in this task")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task_1",
                "name": "Content Safety Classification",
                "description": "Classify content as safe or unsafe",
                "difficulty": "easy",
                "num_samples": 10,
            }
        }
    }
