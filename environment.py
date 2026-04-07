import random
import json
import os
from typing import Optional, List, Dict, Any
from models import ContentObservation, ModerationAction, StepResult, EnvironmentState, TaskSpec
import logging

logger = logging.getLogger(__name__)

# dataset: 30 labeled content samples across 5 categories and 3 difficulties
DATASET: List[Dict[str, str]] = [
    # EASY DIFFICULTY (10 samples) — Obvious spam, clear safe content
    {
        "id": "s001",
        "text": "Win a FREE iPhone! Click here now!!!",
        "label": "spam",
        "difficulty": "easy",
        "context": "social_media",
    },
    {
        "id": "s002",
        "text": "Great recipe, thanks for sharing! Made it today and family loved it.",
        "label": "safe",
        "difficulty": "easy",
        "context": "cooking_forum",
    },
    {
        "id": "s003",
        "text": "What's your favorite movie? I just watched Inception again!",
        "label": "safe",
        "difficulty": "easy",
        "context": "general_forum",
    },
    {
        "id": "s004",
        "text": "CLICK NOW FOR 100% GUARANTEED PROFITS IN 24 HOURS",
        "label": "spam",
        "difficulty": "easy",
        "context": "email",
    },
    {
        "id": "s005",
        "text": "Had an amazing day with my family at the beach. Perfect weather!",
        "label": "safe",
        "difficulty": "easy",
        "context": "social_media",
    },
    {
        "id": "s006",
        "text": "Buy cheap meds online - NO PRESCRIPTION NEEDED",
        "label": "spam",
        "difficulty": "easy",
        "context": "dark_web_forum",
    },
    {
        "id": "s007",
        "text": "Congratulations on your promotion! You deserve it!",
        "label": "safe",
        "difficulty": "easy",
        "context": "workplace_messenger",
    },
    {
        "id": "s008",
        "text": "LIMITED TIME OFFER!!! Make money fast without working!!!",
        "label": "spam",
        "difficulty": "easy",
        "context": "social_media",
    },
    {
        "id": "s009",
        "text": "Looking forward to the weekend. Anyone want to grab coffee?",
        "label": "safe",
        "difficulty": "easy",
        "context": "social_media",
    },
    {
        "id": "s010",
        "text": "Get rich quick! Buy this secret manual for $9.99 only!",
        "label": "spam",
        "difficulty": "easy",
        "context": "classified_ads",
    },
    # MEDIUM DIFFICULTY (10 samples) — Coded language, borderline content
    {
        "id": "s011",
        "text": "People like him should just disappear from society.",
        "label": "hate_speech",
        "difficulty": "medium",
        "context": "news_comments",
    },
    {
        "id": "s012",
        "text": "Muslims are taking over this country. We need to wake up.",
        "label": "hate_speech",
        "difficulty": "medium",
        "context": "political_forum",
    },
    {
        "id": "s013",
        "text": "We need to deal with this problem once and for all, by any means necessary.",
        "label": "violence",
        "difficulty": "medium",
        "context": "extremist_forum",
    },
    {
        "id": "s014",
        "text": "That girl has a nice body. Would definitely date her.",
        "label": "adult_content",
        "difficulty": "medium",
        "context": "reddit_nsfw",
    },
    {
        "id": "s015",
        "text": "They don't belong here. They're ruining our neighborhood.",
        "label": "hate_speech",
        "difficulty": "medium",
        "context": "neighborhood_forum",
    },
    {
        "id": "s016",
        "text": "Time to fight back against them. We've been pushed around too long.",
        "label": "violence",
        "difficulty": "medium",
        "context": "militia_chat",
    },
    {
        "id": "s017",
        "text": "Traditional values and heritage are being destroyed by outsiders.",
        "label": "hate_speech",
        "difficulty": "medium",
        "context": "heritage_forum",
    },
    {
        "id": "s018",
        "text": "She's stunning in that photo. Drop the rest of her pictures.",
        "label": "adult_content",
        "difficulty": "medium",
        "context": "telegram_group",
    },
    {
        "id": "s019",
        "text": "This group is the reason society is falling apart. Get them out.",
        "label": "hate_speech",
        "difficulty": "medium",
        "context": "talk_radio_forum",
    },
    {
        "id": "s020",
        "text": "We should cleanse this country of the infection. Violent times ahead.",
        "label": "violence",
        "difficulty": "medium",
        "context": "white_supremacist_site",
    },
    # HARD DIFFICULTY (10 samples) — Context-dependent interpretations
    {
        "id": "s021",
        "text": "She's so hot and beautiful!",
        "label": "safe",
        "difficulty": "hard",
        "context": "dating_forum",
    },
    {
        "id": "s022",
        "text": "She's so hot and beautiful!",
        "label": "adult_content",
        "difficulty": "hard",
        "context": "children_platform",
    },
    {
        "id": "s023",
        "text": "Die you noob! Haha get rekt!",
        "label": "safe",
        "difficulty": "hard",
        "context": "gaming_chat",
    },
    {
        "id": "s024",
        "text": "Die you noob! Haha get rekt!",
        "label": "violence",
        "difficulty": "hard",
        "context": "threat_assessment_forum",
    },
    {
        "id": "s025",
        "text": "I'm gonna kill this test! Going to ace it!",
        "label": "safe",
        "difficulty": "hard",
        "context": "student_study_group",
    },
    {
        "id": "s026",
        "text": "I'm gonna kill this test! Going to ace it!",
        "label": "violence",
        "difficulty": "hard",
        "context": "threat_reporting_system",
    },
    {
        "id": "s027",
        "text": "Build the wall to keep us safe and prosperous.",
        "label": "safe",
        "difficulty": "hard",
        "context": "political_debate_forum",
    },
    {
        "id": "s028",
        "text": "Build the wall to keep them out. These people are invading.",
        "label": "hate_speech",
        "difficulty": "hard",
        "context": "xenophobic_subreddit",
    },
    {
        "id": "s029",
        "text": "These people are everywhere now. We're being replaced.",
        "label": "safe",
        "difficulty": "hard",
        "context": "census_discussion_forum",
    },
    {
        "id": "s030",
        "text": "These people are everywhere now. We're being replaced.",
        "label": "hate_speech",
        "difficulty": "hard",
        "context": "white_nationalist_forum",
    },
]


class ContentModerationEnv:
    """Content Moderation OpenEnv environment for the Scaler × Meta PyTorch Hackathon.
    
    This environment provides a reinforcement learning setting where agents learn to
    classify user-generated content into moderation categories (safe, spam, hate_speech,
    violence, adult_content). The difficulty scales across tasks, and context-aware
    samples in the hard task require agents to consider platform context.
    """

    def __init__(self) -> None:
        """Initialize the ContentModerationEnv with the dataset.
        
        Organizes the 30 samples by difficulty level for task selection.
        Sets up environment state variables.
        Loads experience memory from disk if available.
        """
        self.dataset = DATASET
        
        # Organize samples by difficulty
        self.easy_samples = [s for s in DATASET if s["difficulty"] == "easy"]
        self.medium_samples = [s for s in DATASET if s["difficulty"] == "medium"]
        self.hard_samples = [s for s in DATASET if s["difficulty"] == "hard"]
        
        # Episode state
        self.current_task_id: Optional[str] = None
        self.current_index: int = 0
        self.current_samples: List[Dict[str, str]] = []
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.initialized: bool = False
        
        # Experience replay memory — persisted across runs
        self.memory_file = "agent_memory.json"
        self.experience_memory = self._load_memory()
        
        logger.info("ContentModerationEnv initialized")

    def _load_memory(self) -> Dict[str, Any]:
        """Load experience memory from disk or create new memory structure.
        
        Returns:
            Dict with structure:
            {
                "total_runs": int,
                "past_mistakes": {
                    "sample_id": {"ground_truth": str, "wrong_labels": [str], "count": int},
                    ...
                },
                "label_confusion_matrix": {
                    "predicted_label": {"actual_label": count, ...},
                    ...
                }
            }
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")
        
        return {
            "total_runs": 0,
            "past_mistakes": {},
            "label_confusion_matrix": {}
        }

    def _save_memory(self) -> None:
        """Persist experience memory to disk for next run."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.experience_memory, f, indent=2)
            logger.info(f"Memory saved: {len(self.experience_memory['past_mistakes'])} mistakes tracked")
        except Exception as e:
            logger.warning(f"Could not save memory: {e}")

    def _record_prediction(self, sample_id: str, ground_truth: str, predicted_label: str, correct: bool) -> None:
        """Record a prediction for future learning.
        
        Args:
            sample_id: ID of the sample
            ground_truth: Correct label
            predicted_label: Model's predicted label
            correct: Whether prediction was correct
        """
        # Track confusion matrix
        if predicted_label not in self.experience_memory["label_confusion_matrix"]:
            self.experience_memory["label_confusion_matrix"][predicted_label] = {}
        
        confusion = self.experience_memory["label_confusion_matrix"][predicted_label]
        confusion[ground_truth] = confusion.get(ground_truth, 0) + 1
        
        # Track mistakes for hard samples
        if not correct:
            if sample_id not in self.experience_memory["past_mistakes"]:
                self.experience_memory["past_mistakes"][sample_id] = {
                    "ground_truth": ground_truth,
                    "wrong_labels": [],
                    "count": 0
                }
            
            mistake = self.experience_memory["past_mistakes"][sample_id]
            if predicted_label not in mistake["wrong_labels"]:
                mistake["wrong_labels"].append(predicted_label)
            mistake["count"] += 1

    def reset(self, task_id: str = "task_1") -> ContentObservation:
        """Reset the environment for a new episode.
        
        Validates the task_id, filters samples by difficulty, shuffles them
        deterministically (seed 42 for reproducibility), and returns the first
        observation.
        
        Args:
            task_id: Task identifier ('task_1', 'task_2', or 'task_3')
            
        Returns:
            ContentObservation for the first sample in the task
            
        Raises:
            ValueError: If task_id is not one of 'task_1', 'task_2', 'task_3'
        """
        # Validate task_id
        if task_id not in ("task_1", "task_2", "task_3"):
            raise ValueError(
                f"Invalid task_id '{task_id}'. Must be one of: task_1, task_2, task_3"
            )
        
        # Map task_id to difficulty
        difficulty_map = {
            "task_1": "easy",
            "task_2": "medium",
            "task_3": "hard",
        }
        difficulty = difficulty_map[task_id]
        
        # Select samples by difficulty
        if difficulty == "easy":
            self.current_samples = self.easy_samples.copy()
        elif difficulty == "medium":
            self.current_samples = self.medium_samples.copy()
        else:  # hard
            self.current_samples = self.hard_samples.copy()
        
        # Shuffle randomly (no fixed seed) for true RL variability
        random.shuffle(self.current_samples)
        
        # Limit to 10 samples per task (increased from 7)
        self.current_samples = self.current_samples[:10]
        
        # Reset state
        self.current_task_id = task_id
        self.current_index = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.initialized = True
        
        # Track new run in memory
        self.experience_memory["total_runs"] += 1
        
        logger.info(f"Environment reset. Task: {task_id}, Samples: {len(self.current_samples)}, Run #{self.experience_memory['total_runs']}")
        
        # Return first sample as observation
        first_sample = self.current_samples[self.current_index]
        return self._sample_to_observation(first_sample)

    def step(self, action: ModerationAction) -> StepResult:
        """Execute one step in the environment.
        
        The agent provides a moderation action (label and confidence). This is compared
        against the ground truth label from the sample, and a reward is computed:
        
            - Correct + high confidence (>= 0.7):     reward = 1.0
            - Correct + low confidence (< 0.7):       reward = 0.7
            - Wrong + honest uncertainty (< 0.4):     reward = 0.3
            - Wrong + overconfident (>= 0.4):         reward = 0.0
        
        Args:
            action: ModerationAction with label and confidence
            
        Returns:
            StepResult containing next observation (if not done), reward, done flag,
            and info dictionary with ground truth and metadata
            
        Raises:
            RuntimeError: If environment not initialized or episode already done
        """
        # Check environment state
        if not self.initialized:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )
        
        if self.done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )
        
        # Get current sample and ground truth
        current_sample = self.current_samples[self.current_index]
        ground_truth_label = current_sample["label"]
        
        # Compute reward based on action vs. ground truth
        label_correct = action.label.value == ground_truth_label
        confidence = action.confidence
        
        # Record prediction in memory for future learning
        self._record_prediction(
            sample_id=current_sample["id"],
            ground_truth=ground_truth_label,
            predicted_label=action.label.value,
            correct=label_correct
        )
        
        if label_correct and confidence >= 0.7:
            reward = 1.0
        elif label_correct and confidence < 0.7:
            reward = 0.7
        elif not label_correct and confidence < 0.4:
            reward = 0.3
        else:  # not label_correct and confidence >= 0.4
            reward = 0.0
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Advance to next sample
        self.current_index += 1
        
        # Check if episode is done
        if self.current_index >= len(self.current_samples):
            self.done = True
            next_observation = None
        else:
            next_observation = self._sample_to_observation(
                self.current_samples[self.current_index]
            )
        
        # Build info dict
        info = {
            "ground_truth": ground_truth_label,
            "sample_id": current_sample["id"],
            "task_id": self.current_task_id,
            "cumulative_reward": self.cumulative_reward,
        }
        
        logger.info(
            f"Step {self.current_index}: action={action.label.value}, "
            f"truth={ground_truth_label}, confidence={confidence}, reward={reward}"
        )
        
        # Save memory if episode is done
        if self.done:
            self._save_memory()
            logger.info(f"Episode complete. Memory persisted for next run.")
        
        return StepResult(
            observation=next_observation,
            reward=reward,
            done=self.done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Get the current state of the environment.
        
        Returns:
            EnvironmentState with current task, progress, rewards, and episode status
        """
        state = EnvironmentState(
            task_id=self.current_task_id or "none",
            current_index=self.current_index,
            total_samples=len(self.current_samples),
            cumulative_reward=self.cumulative_reward,
            done=self.done,
            initialized=self.initialized,
        )
        return state

    def get_tasks(self) -> List[TaskSpec]:
        """Get all available tasks.
        
        Returns:
            List of 3 TaskSpec objects describing each task
        """
        tasks = [
            TaskSpec(
                task_id="task_1",
                name="Easy Content Moderation",
                description="Classify obvious spam and clearly safe content. Good for initial training.",
                difficulty="easy",
                num_samples=7,
            ),
            TaskSpec(
                task_id="task_2",
                name="Medium Content Moderation",
                description="Handle borderline and coded harmful content. Requires nuanced language understanding.",
                difficulty="medium",
                num_samples=7,
            ),
            TaskSpec(
                task_id="task_3",
                name="Hard Content Moderation",
                description="Handle context-dependent content where platform context changes the correct classification.",
                difficulty="hard",
                num_samples=7,
            ),
        ]
        return tasks

    def _sample_to_observation(self, sample: Dict[str, str]) -> ContentObservation:
        """Convert a dataset sample dict to a ContentObservation.
        
        Private helper method to extract and structure observation data.
        
        Args:
            sample: Dictionary from DATASET with id, text, label, difficulty, context
            
        Returns:
            ContentObservation ready for the agent
        """
        return ContentObservation(
            content_id=sample["id"],
            text=sample["text"],
            context=sample["context"],
            metadata={
                "difficulty": sample["difficulty"],
                "context": sample["context"],
            },
        )
