# SLIE Codebase Documentation

## Sign Language Interaction Environment — Complete Technical Reference

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Core Components](#4-core-components)
5. [Data Layer](#5-data-layer)
6. [Environment API](#6-environment-api)
7. [Tasks and Scenarios](#7-tasks-and-scenarios)
8. [Reward System](#8-reward-system)
9. [Grading System](#9-grading-system)
10. [Baseline Agent](#10-baseline-agent)
11. [API Endpoints](#11-api-endpoints)
12. [Data Models](#12-data-models)
13. [Testing](#13-testing)
14. [Deployment](#14-deployment)
15. [Configuration](#15-configuration)
16. [Design Decisions](#16-design-decisions)
17. [Constraints and Limitations](#17-constraints-and-limitations)

---

## 1. Project Overview

### 1.1 Purpose

SLIE (Sign Language Interaction Environment) is a deterministic OpenEnv-style environment for evaluating AI agents that interpret hand-sign observations without being given the true gesture label. It provides a structured simulation environment where an AI agent must:

- Receive hand-sign landmark observations
- Interpret their meaning (intent)
- Execute appropriate actions
- Maintain multi-step conversational context

### 1.2 Key Features

- **FastAPI environment** with `/reset`, `/step`, `/state`, `/health` endpoints
- **Observation contract** built around hand landmark sequences plus a compact gesture embedding
- **Three difficulty levels**: Command Recognition (easy), Multi-step Interaction (medium), Intent Inference (hard)
- **Dense per-step rewards** + deterministic end-of-episode grading
- **Baseline external agent** in `inference.py` that decodes the perceptual observation before reasoning
- **Docker deployment** for reproducible evaluation

### 1.3 Version

Current version: **1.1.0**

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SLIE SYSTEM BOUNDARY                     │
│                                                             │
│  ┌──────────────────┐                                       │
│  │  Gesture Dataset  │  ← Static JSON file (pre-baked)       │
│  │  (gestures.json)  │                                       │
│  └────────┬─────────┘                                       │
│           │ loads on reset()                                │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  Gesture Input   │  ← Selects gesture sequence for task   │
│  │  Layer           │    Returns observation dict            │
│  └────────┬─────────┘                                       │
│           │ observation                                     │
│           ▼                                                 │
│  ┌──────────────────────────────────────┐                   │
│  │         SLIE OpenEnv Core            │                   │
│  │                                      │                   │
│  │  ┌─────────┐  ┌────────┐  ┌───────┐ │                   │
│  │  │ reset() │  │ step() │  │state()│ │                   │
│  │  └─────────┘  └────────┘  └───────┘ │                   │
│  │                                      │                   │
│  │  ┌──────────────────────────────┐    │                   │
│  │  │      State Manager           │    │                   │
│  │  └──────────────────────────────┘    │                   │
│  │                                      │                   │
│  │  ┌──────────────────────────────┐    │                   │
│  │  │      Reward Engine           │    │                   │
│  │  └──────────────────────────────┘    │                   │
│  │                                      │                   │
│  │  ┌──────────────────────────────┐    │                   │
│  │  │      Task Grader             │    │                   │
│  │  └──────────────────────────────┘    │                   │
│  └──────────────────────────────────────┘                   │
│           │                                                 │
│           │ HTTP (FastAPI)                                  │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │   API Layer      │  ← /reset  /step  /state             │
│  │   (FastAPI)      │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
          │ HTTP requests
          ▼
┌──────────────────┐
│   AI Agent       │  ← inference.py (external process)
│   (LLM-based)    │
└──────────────────┘
```

### 2.2 System Flow

```
[Pre-baked Gesture Prototype Dataset]
         │
         ▼
[Gesture Input Layer]         ← Builds landmark trajectories from static prototypes
         │
         ▼
[SLIE OpenEnv Environment]    ← Manages state, tasks, episodes
         │  (observation)
         ▼
[AI Agent (LLM-based)]        ← Receives observation, produces action
         │  (action)
         ▼
[SLIE Environment step()]     ← Evaluates action, computes reward
         │  (reward, done, next observation)
         ▼
[Grader]                      ← Scores episode on task completion
         │
         ▼
[Logs: START / STEP / END]    ← stdout logging for evaluation pipeline
```

---

## 3. Directory Structure

```
SLIE/
├── slie/                     # Core package
│   ├── __init__.py           # Package initialization, version info
│   ├── app.py                # FastAPI application
│   ├── env.py                # SLIEEnvironment class (main RL loop)
│   ├── state.py              # EnvironmentState dataclass
│   ├── models.py             # Pydantic models for API
│   ├── data_loader.py        # JSON data loading and processing
│   ├── gesture_layer.py      # GestureInputLayer for observations
│   ├── reward.py             # Reward computation
│   └── graders.py            # Task-specific graders
├── server/
│   └── app.py                # Server entry point
├── data/
│   ├── gestures.json          # 24 gesture definitions with frame_features
│   └── tasks.json            # Task definitions with 5 scenarios each
├── tests/
│   ├── __init__.py
│   ├── test_env.py           # Environment tests
│   ├── test_models.py        # Pydantic model tests
│   ├── test_reward.py        # Reward function tests
│   ├── test_graders.py       # Grader tests
│   ├── test_api.py           # API endpoint tests
│   ├── spec_conformance.py   # Data specification tests
│   ├── smoke_checks.py       # Quick sanity checks
│   └── run_all_checks.py     # Comprehensive test suite
├── docs/
│   ├── overview.md           # Problem definition and motivation
│   ├── design.md             # Technical system design
│   ├── environment.md        # OpenEnv specification
│   ├── api.md                # REST API specification
│   ├── agent.md              # AI agent specification
│   ├── grader.md             # Evaluation and grading
│   ├── tasks.md              # Task definitions
│   ├── constraints.md        # Scope and limitations
│   └── file_structure.md     # File organization
├── main.py                   # Development server entry point
├── inference.py              # Baseline external agent
├── pyproject.toml            # Project metadata and dependencies
├── requirements.txt          # Pinned dependencies
├── openenv.yaml              # OpenEnv specification file
├── Dockerfile                # Docker build configuration
└── README.md                 # Quick start guide
```

---

## 4. Core Components

### 4.1 SLIEEnvironment (`slie/env.py`)

The central class managing the RL-style environment loop.

**Key Responsibilities:**
- Episode lifecycle management (`reset()`, `step()`)
- State tracking and transitions
- Reward computation delegation
- Grading invocation at episode end

**Key Methods:**

```python
class SLIEEnvironment:
    def __init__(self) -> None:
        self.state = EnvironmentState()
        self.gestures = load_gestures()
        self.tasks = load_tasks()
        self.gesture_layer: GestureInputLayer | None = None

    def reset(self, task_id: str, episode_seed: int) -> ResetResponse:
        """Initialize a new episode."""
        scenario_id = episode_seed % 5
        scenario = get_scenario(task_id, scenario_id, self.tasks)
        self.gesture_layer = GestureInputLayer(self.gestures, scenario)
        self.state.reset_state(task_id, episode_seed, scenario_id, scenario["gesture_sequence"])
        # ... return first observation

    def step(self, action: SLIEAction) -> StepResponse:
        """Process one agent action and return result."""
        # Validate state, compute reward, update state
        # Check done conditions, compute final score
        # Return observation, reward, done, info

    def get_state(self) -> StateResponse:
        """Return read-only snapshot of internal state."""
```

### 4.2 EnvironmentState (`slie/state.py`)

Dataclass holding all episode state.

```python
@dataclass
class EnvironmentState:
    task_id: str | None = None
    episode_seed: int | None = None
    scenario_id: int | None = None
    step_count: int = 0
    max_steps: int = 10
    gesture_sequence: list[str] = field(default_factory=list)
    gesture_index: int = 0
    interaction_history: list[dict[str, Any]] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False
    final_score: float | None = None
    last_action: dict[str, Any] | None = None
```

### 4.3 GestureInputLayer (`slie/gesture_layer.py`)

Provides gesture data to the environment.

```python
class GestureInputLayer:
    def __init__(self, gestures: dict[str, Any], scenario: dict[str, Any]):
        self.gestures = gestures
        self.scenario = scenario
        self.sequence = scenario["gesture_sequence"]
        self.steps = scenario["steps"]

    def get_observation(self, gesture_index: int, step_count: int, 
                       history: list[str], task_id: str) -> SLIEObservation:
        """Build observation for given gesture index."""

    def get_step_spec(self, gesture_index: int) -> dict[str, Any]:
        """Get step specification for grading."""
```

### 4.4 FastAPI Application (`slie/app.py`)

HTTP API layer built with FastAPI.

```python
app = FastAPI(title="SLIE", version="1.1.0")
env = SLIEEnvironment()

@app.get("/")
def root() -> dict[str, object]: ...

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse: ...

@app.post("/step", response_model=StepResponse)
def step(action: SLIEAction) -> StepResponse: ...

@app.get("/state", response_model=StateResponse)
def state() -> StateResponse: ...

@app.get("/health")
def health() -> dict[str, str]: ...
```

---

## 5. Data Layer

### 5.1 Gesture Dataset (`data/gestures.json`)

Contains 24 gesture definitions. Each gesture has:
- `label`: Gesture name (e.g., "HELLO", "YES")
- `frame_features`: 64-dim prototype embedding
- `category`: Category (greeting, confirmation, control, request, media, navigation, communication)

**Gestures:**
```
HELLO, YES, NO, STOP, HELP, GOODBYE, OPEN, CLOSE, FOOD, WATER,
MUSIC, PLAY, LOUDER, CALL, WAIT, YOUTUBE, SEARCH, NEAR, NOW,
MAP, HOME, QUIET, SLEEP, COLD
```

**Validation Rules:**
- Exactly 24 gestures
- Each gesture has exactly 64 floats in `frame_features`

### 5.2 Task Dataset (`data/tasks.json`)

Contains definitions for all three tasks, each with 5 scenarios.

**Structure per task:**
```json
{
  "task_id": {
    "scenarios": [
      {
        "id": 0-4,
        "gesture_sequence": ["GESTURE", ...],
        "steps": [
          {
            "gesture": "GESTURE",
            "expected_intent": "intent_name",
            "intent_aliases": ["alias1", "alias2"],
            "expected_keywords": ["kw1", "kw2"]
          }
        ]
        // For task3, also includes:
        "compound_intent": "compound_intent_name",
        "explanation": "human readable explanation"
      }
    ]
  }
}
```

### 5.3 Data Loader (`slie/data_loader.py`)

```python
def load_gestures() -> dict[str, Any]:
    """Load and validate gestures.json."""

def load_tasks() -> dict[str, Any]:
    """Load and validate tasks.json."""

def get_scenario(task_id: str, scenario_id: int, 
                 tasks: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get specific scenario by task and scenario ID."""

def get_gesture_embedding(gesture_label: str, gestures: dict[str, Any]) -> list[float]:
    """Compute 64-dim embedding from gesture frame_features."""

def get_hand_landmarks(gesture_label: str, gestures: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate deterministic hand landmark sequence from embedding."""
```

### 5.4 Gesture Embedding Computation

The embedding is computed using a contextual projection:

```python
def get_gesture_embedding(gesture_label: str, gestures: dict[str, Any]) -> list[float]:
    raw_features = gestures[gesture_label]["frame_features"]
    embedding = []
    for index in range(64):
        primary = raw_features[index]
        neighbor = raw_features[(index * 7 + 13) % 64]
        context = raw_features[(index * 11 + 29) % 64]
        projected = primary * 0.55 + neighbor * 0.3 + (context * context) * 0.15
        embedding.append(round(projected, 4))
    return embedding
```

### 5.5 Hand Landmark Generation

Generates 6 frames with 21 landmarks per hand:

```python
def get_hand_landmarks(gesture_label: str, gestures: dict[str, Any]) -> list[dict[str, Any]]:
    embedding = get_gesture_embedding(gesture_label, gestures)
    signature = sum((index + 1) * ord(char) for index, char in enumerate(label))
    
    for frame_index in range(6):
        # Generate x, y, z coordinates for each of 21 points
        # Uses sine waves and offsets based on embedding values
        # Left and right hands are mirrored
```

---

## 6. Environment API

### 6.1 Reset (`/reset`)

**Purpose:** Initialize a new episode.

**Request:**
```json
{
  "task_id": "task1",      // "task1", "task2", or "task3"
  "episode_seed": 42       // integer, selects scenario (seed % 5)
}
```

**Response:**
```json
{
  "observation": {
    "gesture_embedding": [0.12, 0.45, ...],  // 64 floats
    "hand_landmarks": [...],                  // 6 frames
    "context": {
      "current_task": "task1",
      "step_count": 0,
      "history": []
    }
  },
  "task_id": "task1",
  "episode_seed": 42
}
```

### 6.2 Step (`/step`)

**Purpose:** Submit one agent action.

**Request:**
```json
{
  "intent": "greeting",
  "confidence": 0.95,
  "response": "Hello! How can I help you?"
}
```

**Response:**
```json
{
  "observation": {
    "gesture_embedding": [...],
    "hand_landmarks": [...],
    "context": {
      "current_task": "task1",
      "step_count": 1,
      "history": ["Step 1: prior_intent=greeting"]
    }
  },
  "reward": 0.7,
  "done": false,
  "info": {
    "step_count": 1,
    "final_score": null,
    "error": null
  }
}
```

### 6.3 State (`/state`)

**Purpose:** Get read-only snapshot of internal state.

**Response:**
```json
{
  "task_id": "task1",
  "episode_seed": 42,
  "step_count": 2,
  "max_steps": 10,
  "gesture_index": 2,
  "completed_steps": 2,
  "interaction_history": [...],
  "total_reward": 1.4,
  "done": false,
  "final_score": null
}
```

### 6.4 Health (`/health`)

**Purpose:** Health check for monitoring.

**Response:**
```json
{
  "status": "ok",
  "environment": "slie",
  "version": "1.1.0"
}
```

---

## 7. Tasks and Scenarios

### 7.1 Task 1: Command Recognition (Easy)

**Objective:** Correctly identify the intent of individual, unambiguous gestures.

**Gesture Sequence Length:** 5 gestures
**Max Steps:** 10

**Example Scenario (ID 0):**
```
HELLO → greeting
YES → confirm
STOP → halt
HELP → request_help
GOODBYE → farewell
```

**Agent Strategy:**
- Decode current hand sign from landmarks/embedding
- Map to known intent
- No history needed

### 7.2 Task 2: Multi-step Interaction (Medium)

**Objective:** Track a sequence of gestures forming a compound task with context dependency.

**Gesture Sequence Length:** 8 gestures
**Max Steps:** 10

**Example Scenario (ID 0):**
```
OPEN → open_action
YOUTUBE → select_app:youtube
SEARCH → search_action
MUSIC → search_query:music
PLAY → play_action
LOUDER → increase_volume
STOP → halt
CLOSE → close_action
```

**Agent Strategy:**
- Track what "app" is open
- Use full history to interpret context
- Respond with accumulated context

### 7.3 Task 3: Intent Inference (Hard)

**Objective:** Infer compound intent from ambiguous gestures requiring synthesis.

**Gesture Sequence Length:** 3 gestures
**Max Steps:** 10

**Example Scenarios:**
```
FOOD + NEAR + NOW → find_nearby_restaurants
CALL + HELP + NOW → emergency_call
MUSIC + QUIET + SLEEP → play_sleep_music
OPEN + MAP + HOME → navigate_home
WATER + COLD + NOW → request_cold_water_immediately
```

**Agent Strategy:**
- Accumulate gestures across steps
- Synthesize ALL gestures into compound intent on final step
- Reason about combined meaning

### 7.4 Scenario Selection

Scenario is selected deterministically:
```python
scenario_id = episode_seed % 5
```

This means:
- episode_seed 0, 5, 10 → scenario 0
- episode_seed 1, 6, 11 → scenario 1
- etc.

---

## 8. Reward System

### 8.1 Reward Components

```python
reward = intent_reward + response_reward + completion_bonus + penalties
```

### 8.2 Intent Reward

| Condition | Reward |
|-----------|--------|
| Exact match (case-insensitive) | +0.4 |
| Alias match | +0.2 |
| No match | 0.0 |

### 8.3 Response Reward

| Condition | Reward |
|-----------|--------|
| Any keyword in response | +0.3 |
| No keyword match | 0.0 |

### 8.4 Completion Bonus

| Condition | Reward |
|-----------|--------|
| Final step AND intent_reward > 0 | +0.3 |
| Otherwise | 0.0 |

### 8.5 Penalties

| Condition | Penalty |
|-----------|---------|
| Loop (same action twice) | -0.5 |
| Empty intent or response | -0.3 |

### 8.6 Reward Examples

| Scenario | Final Reward |
|----------|-------------|
| Perfect match, not final | 0.7 |
| Perfect match, final | 1.0 |
| Alias match only | 0.2 |
| Wrong intent, good response | 0.3 |
| Repeated action (loop) | 0.2 |
| Empty fields | -0.3 |

### 8.7 Reward Clamping

All rewards clamped to `[-1.0, 1.0]`.

---

## 9. Grading System

### 9.1 Task 1 Grader: Command Recognition

```python
def task1_grader(history: EpisodeHistory) -> float:
    correct = sum(1 for entry in history.interaction_history 
                  if intent_matches(entry))
    return round(correct / len(history.interaction_history), 4)
```

**Score:** `(correct intents) / (total steps)`

### 9.2 Task 2 Grader: Multi-step Interaction

```python
def task2_grader(history: EpisodeHistory) -> float:
    correct = sum(1 for entry in history.interaction_history 
                  if intent_matches(entry))
    base_score = correct / len(history.interaction_history)
    
    all_processed = len(history.interaction_history) == len(history.gesture_sequence)
    all_correct = correct == len(history.interaction_history)
    sequence_bonus = 0.1 if all_correct and all_processed else 0.0
    
    return round(min(1.0, base_score + sequence_bonus), 4)
```

**Score:** `base_score + sequence_bonus` (capped at 1.0)

### 9.3 Task 3 Grader: Intent Inference

```python
def task3_grader(history: EpisodeHistory) -> float:
    # Score final compound intent
    compound_score = 1.0 if exact_match else \
                     0.7 if alias_match else \
                     0.4 if keyword_in_intent else \
                     0.2 if keyword_in_response else 0.0
    
    # Bonus for intermediate steps
    intermediate = history.interaction_history[:-1]
    step_bonus = (correct_intermediate / len(intermediate)) * 0.2 if intermediate else 0.0
    
    return round(min(1.0, compound_score * 0.8 + step_bonus), 4)
```

**Score:** `compound_score * 0.8 + step_bonus` (capped at 1.0)

### 9.4 Grading Invocation

Graders are called when episode ends:
```python
if state.done and state.final_score is None:
    state.final_score = compute_final_score(state)
```

---

## 10. Baseline Agent

### 10.1 Overview

`inference.py` is a baseline external agent that:
- Communicates with SLIE via HTTP
- Uses an LLM to reason about gestures
- Produces structured actions
- Logs in the required format

### 10.2 Agent Loop

```python
def run_task(client: OpenAI, env_url: str, task_id: str, seed: int) -> None:
    # 1. Reset environment
    reset_resp = requests.post(f"{env_url}/reset", json={"task_id": task_id, "episode_seed": seed})
    obs = reset_resp.json()["observation"]
    
    # 2. For each step
    for step in range(1, MAX_STEPS + 1):
        if not observation_is_active(obs):
            break
            
        # 3. Build prompt with observation context
        prompt = build_prompt(obs, last_reward, agent_history, step, task_id, ...)
        
        # 4. Call LLM
        llm_output = call_llm(client, prompt)
        
        # 5. Parse action
        action = parse_action(llm_output)
        
        # 6. Step environment
        step_resp = requests.post(f"{env_url}/step", json=action)
        
        # 7. Log result
        log_step(...)
```

### 10.3 Prompt Building

```python
def build_prompt(obs, last_reward, history, step, task_id, is_final_step, signal_summary) -> str:
    return f"""
You are an AI assistant helping a deaf user through sign language gestures.
Respond with JSON only using keys: intent, confidence, response.

Current Task: {current_task}
Step: {step_count}
Hand Sign Summary: {signal_summary}
Last Reward: {last_reward:.2f}

Conversation History:
{history_block}

Known gestures: HELLO, YES, NO, STOP, HELP, ...
"""
```

### 10.4 LLM Output Parsing

Handles multiple formats:
1. Clean JSON: `{"intent": "...", "confidence": 0.9, "response": "..."}`
2. Fenced JSON: ```json {...} ```
3. Single-quoted JSON
4. Plain text with key-value pairs

### 10.5 Logging Format

```
[START] task=task1 env=slie model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=greeting reward=0.70 done=false error=null
[STEP] step=2 action=confirm reward=0.70 done=false error=null
...
[END] success=true steps=5 score=0.833 rewards=0.70,0.70,0.70,0.70,1.00
```

---

## 11. API Endpoints

### 11.1 Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Take one action step |
| GET | `/state` | Get current environment state |
| GET | `/health` | Health check |

### 11.2 Error Handling

| Condition | Status | Error Message |
|-----------|--------|---------------|
| Invalid task_id | 422 | "task_id must be one of: task1, task2, task3" |
| step() before reset() | 400 | "No active episode. Call reset() first." |
| step() after done | 400 | "Episode is done. Call reset() to start a new episode." |
| Data validation error | 422 | Pydantic validation message |
| Internal error | 500 | Error details |

---

## 12. Data Models

### 12.1 Observation Models

```python
class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float

class HandFrame(BaseModel):
    left_hand: list[LandmarkPoint]   # Exactly 21 points
    right_hand: list[LandmarkPoint]  # Exactly 21 points

class GestureContext(BaseModel):
    current_task: str
    step_count: int = 0
    history: list[str] = Field(default_factory=list)  # Max 5 entries

class SLIEObservation(BaseModel):
    gesture_embedding: list[float]    # Exactly 64 floats
    hand_landmarks: list[HandFrame]
    context: GestureContext
```

### 12.2 Action Model

```python
class SLIEAction(BaseModel):
    intent: str = Field(max_length=100)
    confidence: float  # Clamped to [0.0, 1.0]
    response: str = Field(max_length=500)
```

### 12.3 Response Models

```python
class SLIEInfo(BaseModel):
    step_count: int
    final_score: float | None = None
    error: str | None = None

class ResetResponse(BaseModel):
    observation: SLIEObservation
    task_id: str
    episode_seed: int

class StepResponse(BaseModel):
    observation: SLIEObservation
    reward: float
    done: bool
    info: SLIEInfo

class StateResponse(BaseModel):
    task_id: str | None
    episode_seed: int | None
    scenario_id: int | None
    step_count: int = 0
    max_steps: int = 10
    gesture_index: int = 0
    completed_steps: int = 0
    interaction_history: list[dict[str, Any]]
    total_reward: float = 0.0
    done: bool = False
    final_score: float | None = None
```

---

## 13. Testing

### 13.1 Test Files

| File | Purpose |
|------|---------|
| `test_env.py` | Environment reset and step tests |
| `test_models.py` | Pydantic model validation |
| `test_reward.py` | Reward computation |
| `test_graders.py` | Task graders |
| `test_api.py` | HTTP endpoint tests |
| `spec_conformance.py` | Data specification compliance |
| `smoke_checks.py` | Quick sanity checks |
| `run_all_checks.py` | Comprehensive test suite |

### 13.2 Running Tests

```bash
# Standard pytest
pytest -q

# Smoke checks only
python tests/smoke_checks.py

# Spec conformance
python tests/spec_conformance.py

# All checks
python tests/run_all_checks.py
```

### 13.3 Key Test Cases

**Environment Tests:**
- Reset returns 64-dim embedding and 6 landmark frames
- Step increments state correctly
- Episode ends after all gestures processed
- Perfect run scores 1.0

**Reward Tests:**
- Perfect non-final reward = 0.7
- Perfect final reward = 1.0
- Loop penalty applies correctly

**Grader Tests:**
- Task1: correct/total ratio
- Task2: correct/total + sequence bonus
- Task3: compound intent scoring

**API Tests:**
- Root returns service info
- Health returns ok status
- Reset validates task_id

---

## 14. Deployment

### 14.1 Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py

# Or with custom port
PORT=8080 python main.py
```

### 14.2 Docker

```bash
# Build image
docker build -t slie .

# Run container
docker run -p 8000:8000 slie

# With custom port
docker run -p 8080:8000 slie
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]
```

### 14.3 API Verification

```bash
# Health check
curl http://localhost:8000/health

# Start episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1","episode_seed":0}'

# Take step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"intent":"greeting","confidence":0.9,"response":"hello"}'
```

---

## 15. Configuration

### 15.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `API_BASE_URL` | https://router.huggingface.co/v1 | LLM API endpoint |
| `MODEL_NAME` | Qwen/Qwen2.5-72B-Instruct | LLM model |
| `HF_TOKEN` | (required) | HuggingFace API token |
| `ENV_URL` | http://localhost:8000 | Environment URL for inference |
| `DEBUG_INFERENCE` | 0 | Enable debug logging |

### 15.2 Example .env File

```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=hf_your_token_here
ENV_URL=http://localhost:8000
```

### 15.3 openenv.yaml

OpenEnv specification file defining:
- Environment metadata
- Task definitions
- Observation/action space schemas
- Endpoint mappings
- Docker configuration

---

## 16. Design Decisions

### 16.1 Architecture Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Static gesture dataset (no CV) | Reproducibility, no hardware needed | Not a real CV system |
| String labels for gestures | Simple, deterministic grading | Loses nuance of real poses |
| 64-dim embedding | Mimics real landmark embeddings | Not used by grader directly |
| Deterministic grader (lookup) | Reproducible scores | Less flexible than LLM-as-judge |
| FastAPI for API layer | Standard, fast, auto-docs | Single-threaded limitation |
| Fixed max_steps=10 | Prevents infinite loops | Limits complex interactions |
| episode_seed controls scenario | Reproducible baseline runs | Less varied evaluation |

### 16.2 Why Not Real Computer Vision?

- Reproducibility: Deterministic output is guaranteed
- Hardware constraints: No GPU required
- Simplicity: Focus on agent reasoning, not perception
- Benchmark stability: Same gestures every time

### 16.3 Why Separate Environment and Agent?

- Clear boundaries for evaluation
- Environment can be deployed independently
- Multiple agents can be tested against same environment
- Grading logic stays secure in environment

---

## 17. Constraints and Limitations

### 17.1 Out of Scope

- Real-time webcam or video input
- MediaPipe or any CV library at runtime
- OpenCV processing
- Raw image/video processing
- Full ASL/BSL grammar parsing
- Multilingual sign language support
- Emotion detection
- Real application integration
- Fuzzy string matching in graders
- LLM-as-judge in graders
- Multi-user or concurrent episodes
- User authentication
- Persistent storage
- GPU inference inside environment
- Streaming responses or WebSockets

### 17.2 Known Limitations

| Limitation | Description |
|------------|-------------|
| Fixed 24-gesture vocabulary | Cannot add gestures without editing JSON |
| No multi-turn correction | Environment doesn't re-present wrong gestures |
| 5 scenarios per task | Limited evaluation coverage |
| Synthetic landmarks | Less realistic than captured motion |
| History limited to 5 entries | May lose context in long sequences |
| Single process only | Cannot evaluate multiple agents |

### 17.3 Resource Constraints

| Resource | Limit |
|----------|-------|
| vCPU | 2 |
| Memory | 8GB |
| Inference time | <20 minutes total |
| Docker image | <2GB recommended |
| Port | 8000 |

### 17.4 Data Integrity Rules

- Gesture labels are case-sensitive (UPPERCASE)
- Intent comparisons are lowercase normalized
- `frame_features` must be exactly 64 floats
- Scenario IDs must be 0-4 with no gaps
- Float scores rounded to 4 decimal places
- `episode_seed` must be integer

---

## Appendix A: Quick Reference

### Environment Lifecycle

```
1. Create environment: env = SLIEEnvironment()
2. Reset: response = env.reset("task1", 0)
3. Loop: while not response.done:
       response = env.step(SLIEAction(...))
4. Score: response.info.final_score
```

### Observation Contract

- `gesture_embedding`: Always 64 floats
- `hand_landmarks`: 0-6 frames, each with 21 points per hand
- `context.history`: Max 5 entries
- Gesture label never exposed

### Action Constraints

- `intent`: 1-100 characters, non-empty
- `confidence`: Clamped to [0.0, 1.0]
- `response`: 1-500 characters, non-empty

### Done Conditions

1. `gesture_index >= len(gesture_sequence)`
2. `step_count >= max_steps` (10)

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| Episode | One full task run from reset() to done=True |
| Gesture | Pre-defined hand-sign class (e.g., "HELLO") |
| Intent | Semantic meaning agent assigns to gesture |
| Action | Agent's response with intent, confidence, response |
| Scenario | Specific gesture sequence for a task |
| Step | One agent action and environment response |
| Grading | Final scoring when episode ends |

---

## Appendix C: File Dependencies

```
slie/app.py
├── slie/env.py
│   ├── slie/state.py
│   ├── slie/data_loader.py
│   │   ├── data/gestures.json
│   │   └── data/tasks.json
│   ├── slie/gesture_layer.py
│   │   └── slie/data_loader.py
│   ├── slie/reward.py
│   └── slie/graders.py
│       ├── slie/data_loader.py
│       └── slie/models.py
├── slie/models.py
└── main.py

inference.py
├── requests (HTTP calls to slie)
└── openai (LLM API calls)
```

---

*Documentation Version: 1.1.0*
*Last Updated: 2026-04-03*
