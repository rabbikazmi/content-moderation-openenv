---
title: Content Moderation OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# TrustMod: Content Moderation OpenEnv

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E.svg)](https://huggingface.co/spaces)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Framework-orange.svg)](https://github.com/openenv/openenv)

**Bridging AI and Computational Social Science**: An intelligent content moderation environment where agents learn to classify user-generated content and protect digital communities at scale.

## The Problem: Why Content Moderation?

Every second, over 500 hours of video content is uploaded to YouTube alone. Across Meta's platforms, X (Twitter), TikTok, and Reddit, billions of pieces of content are created daily. Human moderation alone cannot scale. Trained moderators can review maybe a few hundred pieces per day, but the volume dwarfs human capacity. This creates a dangerous gap: harmful content from hate speech to violence incitement to exploitation spreads far faster than it gets removed.

Content moderation sits at the intersection of computer science and Computational Social Science (CSS). CSS explores how computational systems shape human behavior online, influence information spread, and affect collective outcomes. A well-trained AI moderation agent does more than flag spam. It breaks radicalization pipelines, shields vulnerable communities, preserves psychological safety in communities, and scales platform values across billions of users. For Meta and similar platforms, protecting people from harmful content isn't just business strategy. It's a core ethical responsibility. Billions of people rely on these systems working 24/7 across dozens of languages and cultural contexts.

This environment simulates the exact decision-making pipeline that production moderation systems use: receive content in context, classify it, assign confidence, compute feedback, and improve. By building agents that excel in this domain, we contribute to safer digital ecosystems and demonstrate that AI can be a force for social good when applied thoughtfully to meaningful problems.

## What This Environment Does

Content Moderation OpenEnv provides a reinforcement learning setting where AI agents learn to classify user-generated content across five categories (safe, spam, hate_speech, violence, adult_content) under varying difficulty levels. Agents observe content text and platform context, submit moderation decisions with confidence scores, and receive rewards based on correctness and calibration. The agent loop repeats over seven samples per task, three difficulty tiers per episode, with cumulative scoring encouraging both accuracy and responsible uncertainty quantification.

```mermaid
sequenceDiagram
    participant Agent as LLM Agent
    participant Env as FastAPI Environment
    participant Reward as Reward Engine

    Note over Agent,Reward: Episode Start
    Agent->>Env: POST /reset {task_id}
    Env->>Env: Load 7 samples<br/>(Easy/Medium/Hard)
    Env->>Agent: Observation {text, context}

    loop For each content sample
        Agent->>Agent: Analyze content<br/>Generate decision
        Agent->>Env: POST /step {label, confidence}
        Env->>Reward: Compute reward<br/>(correctness + calibration)
        Reward->>Env: Reward score
        Env->>Agent: {reward, next_observation, done}
    end

    Env->>Agent: Final episode<br/>Cumulative score (0-21)
```

## Task Design

Each task presents content samples of increasing complexity, testing the agent's ability to handle ambiguity and context-dependency:

| Task ID | Difficulty | Description | Samples |
|---------|------------|-------------|---------|
| `task_1` | Easy | Obvious spam, clear safe content, explicit policy violations | 7 |
| `task_2` | Medium | Borderline content, coded harmful language, subtle violations | 7 |
| `task_3` | Hard | Context-dependent decisions where platform context changes correctness | 7 |

Task 3 deserves special attention. In real moderation, the *same text* can be safe or harmful depending on context. For example: "She's so hot and beautiful!" is safe on a dating forum but adult_content on a children's platform. "Die you noob! Get rekt!" is playful banter in gaming chat but violence on a threat-reporting system. This mirrors production complexity where agents must consider not just content but *where* content appears. A well-trained agent learns this crucial distinction.

## Reward Function

Moderation agents are rewarded not just for accuracy but for calibrated confidence. The four-tier reward structure reflects production system priorities:

| Scenario | Confidence | Reward | Philosophy |
|----------|-----------|--------|------------|
| Correct label | ≥ 0.7 | 1.0 | Confident & accurate: optimal outcome |
| Correct label | < 0.7 | 0.7 | Correct but uncertain: still useful (escalates to human review) |
| Wrong label | < 0.4 | 0.3 | Wrong but honest: shows epistemic humility |
| Wrong label | ≥ 0.4 | 0.0 | Wrong & overconfident: most harmful outcome |

This design reflects a crucial insight: an overconfident mistake in content moderation causes more harm than an uncertain prediction that escalates to human review. A system that confidently misclassifies violent content as safe can radicalize users. A system that flags borderline content for human review protects both accuracy and user experience. Reward structures embed societal values.

## Observation and Action Spaces

**Observation Space:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | The user-generated content to moderate |
| `context` | str | Platform context (e.g., "children_platform", "gaming_chat", "news_comments") |
| `task_id` | str | Current task identifier (task_1, task_2, or task_3) |

**Action Space:**

| Field | Type | Valid Values | Description |
|-------|------|--------------|-------------|
| `label` | enum | safe, spam, hate_speech, violence, adult_content | Moderation classification |
| `confidence` | float | 0.0 to 1.0 | Agent's confidence in the classification |

## Baseline Results

Evaluated with **Qwen/Qwen2.5-7B-Instruct** via HuggingFace Inference API Router:

| Task | Score | Max | Percentage |
|------|-------|-----|-----------|
| Task 1 (Easy) | 7.0 | 7.0 | 100.0% |
| Task 2 (Medium) | 5.0 | 7.0 | 71.4% |
| Task 3 (Hard) | 4.0 | 7.0 | 57.1% |
| **Overall** | **16.0** | **21.0** | **76.2%** |

Perfect performance on easy tasks demonstrates the LLM's strong grasp of obvious spam and clearly safe content. The 71.4% on medium tasks reflects growing difficulty with borderline language and implicit violations. The 57.1% on hard tasks exposes genuine real-world challenges: context-dependent decisions require reasoning about platform norms, user populations, and nuanced language interpretation. This performance profile mirrors human moderator capabilities and validates the environment's difficulty scaling.

## Quick Start

**Setup is simple**: Python 3.11+, pip or conda, and a HuggingFace API token with read access.

Clone and install the project with these commands:

```bash
git clone https://github.com/yourusername/content-moderation-openenv.git
cd content-moderation-openenv
```

Set up a clean Python environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Get dependencies installed:

```bash
pip install -r requirements.txt
```

Point the .env file at your HuggingFace setup:

```bash
cp .env.example .env
```

Edit `.env` and populate:
```
API_BASE_URL=https://api-inference.huggingface.co/models
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=your_huggingface_read_token_here
```

Get the server running in one terminal:

```bash
python main.py
```

Verify it's actually running:

```bash
curl http://localhost:7860/health
# Expected: {"status":"ok"}
```

Run the agent in another terminal window:

```bash
python inference.py
```

This runs the LLM agent through all 3 tasks, shows per-sample decisions, and saves detailed results to `inference_results.json` for analysis.

Verify everything works correctly:

```bash
python validate.py
```

Should display: `20 / 20 checks passed`.

## Docker Guide

### Build the Docker image:

```bash
docker build -t content-moderation-openenv:latest .
```

### Run the container:

```bash
docker run \
  -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/models" \
  -e MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
  -e HF_TOKEN="your_token_here" \
  content-moderation-openenv:latest
```

Verify the container: `curl http://localhost:7860/health`

### Or use Docker Compose for a one-command setup:

```bash
docker-compose up
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | HuggingFace Inference API endpoint |
| `MODEL_NAME` | Yes | LLM model identifier (e.g., Qwen/Qwen2.5-7B-Instruct) |
| `HF_TOKEN` | Yes | HuggingFace API token with read access |

Grab your token from HuggingFace:
1. Visit https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" permission
4. Copy token to `.env`

## API Reference

| Method | Endpoint | Request | Response | Description |
|--------|----------|---------|----------|-------------|
| GET | `/health` | None | `{"status":"ok"}` | Health check; confirms server is running |
| POST | `/reset` | `{"task_id":"task_1"}` | `{"text":"...","context":"...","task_id":"...","metadata":{}}` | Initialize episode; returns first observation |
| POST | `/step` | `{"label":"spam","confidence":0.95}` | `{"reward":1.0,"done":false,"observation":{...},"info":{}}` | Submit action; returns reward and next observation |
| GET | `/state` | None | `{"task_id":"task_1","current_index":2,"total_samples":7,"cumulative_reward":2.0,"done":false}` | Get current episode state |
| GET | `/tasks` | None | `[{"task_id":"task_1","name":"...","difficulty":"easy",...},...]` | List all 3 available tasks |
| GET | `/docs` | None | HTML | Interactive Swagger API documentation |


## Project Structure

```
content-moderation-openenv/
├── main.py                    # FastAPI server, route handlers, startup logic
├── environment.py             # ContentModerationEnv class, episode logic, reward computation
├── models.py                  # Pydantic v2 models (Observation, Action, State, TaskSpec)
├── config.py                  # Configuration management, environment variable parsing
├── inference.py               # LLM agent loop, data collection, metrics aggregation
├── validate.py                # Pre-submission validator (20 checks)
├── openenv.yaml               # OpenEnv specification, environment metadata
├── Dockerfile                 # Docker image definition (Python 3.11-slim)
├── docker-compose.yml         # Docker Compose orchestration
├── requirements.txt           # Python dependencies
├── .env.example               # Template for environment variables
├── inference_results.json     # Saved results from inference.py run
└── README.md                  # This file
```


## Implementation Notes

### LLM Integration

The agent uses an OpenAI-compatible client connected to the HuggingFace Inference API. A system prompt tells the model to output clean JSON: `{"label":"...", "confidence":0.0-1.0}`. The parser handles markdown blocks and malformed responses gracefully. Max tokens set to 20 to keep responses fast and cheap.

### Reproducibility

Everything uses `random.seed(42)` for deterministic shuffling. The same 30-sample dataset runs on every experiment. Results are reproducible as long as the LLM gives the same responses.

---

- built with <3
- bridging AI and computational social science

