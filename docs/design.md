# design.md — Technical System Design

---

## 1. Architecture Diagram (Text-Based)

```
┌─────────────────────────────────────────────────────────────┐
│                    SLIE SYSTEM BOUNDARY                      │
│                                                             │
│  ┌──────────────────┐                                       │
│  │  Gesture Dataset  │  ← Static JSON file (pre-baked)      │
│  │  (gestures.json) │                                       │
│  └────────┬─────────┘                                       │
│           │ loads on reset()                                │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  Gesture Input   │  ← Selects gesture sequence for task  │
│  │  Layer           │    Returns observation dict           │
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
│  │  │  - current_task              │    │                   │
│  │  │  - step_count                │    │                   │
│  │  │  - gesture_sequence          │    │                   │
│  │  │  - gesture_index             │    │                   │
│  │  │  - interaction_history       │    │                   │
│  │  │  - completed_steps           │    │                   │
│  │  │  - total_reward              │    │                   │
│  │  └──────────────────────────────┘    │                   │
│  │                                      │                   │
│  │  ┌──────────────────────────────┐    │                   │
│  │  │      Reward Engine           │    │                   │
│  │  │  - per-step reward logic     │    │                   │
│  │  │  - penalty logic             │    │                   │
│  │  └──────────────────────────────┘    │                   │
│  │                                      │                   │
│  │  ┌──────────────────────────────┐    │                   │
│  │  │      Task Grader             │    │                   │
│  │  │  - task1_grader()            │    │                   │
│  │  │  - task2_grader()            │    │                   │
│  │  │  - task3_grader()            │    │                   │
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

---

## 2. Components

### 2.1 Gesture Input Layer

**Responsibility:** Provide gesture data to the environment.

**Implementation:**
- A static JSON file `data/gestures.json` contains all gesture definitions
- Each gesture has: label, frame_features (64-dim prototype embedding), category
- The layer selects gesture sequences based on the active task
- The layer deterministically expands prototype embeddings into hand-landmark trajectories

**Gesture definition format:**
```json
{
  "HELLO": {
    "label": "HELLO",
    "frame_features": [0.12, 0.45, ..., 0.33],
    "category": "greeting"
  }
}
```

**Sequence selection:** Each task has a fixed list of gesture sequences in `data/tasks.json`. On `reset()`, one sequence is selected deterministically based on `episode_seed`.

---

### 2.2 Environment (OpenEnv Core)

**Responsibility:** Manage episode lifecycle, state, rewards, and grading.

**Key behaviors:**
- `reset()`: Initialize state, load task, return first observation
- `step(action)`: Evaluate agent action, update state, compute reward, return next observation
- `state()`: Return full internal state (read-only snapshot)

**Episode lifecycle:**
```
reset() → [step() × N] → done=True
```

**Max steps per episode:** 10 (hard limit). If step_count reaches 10 and task is incomplete, `done=True` is forced.

---

### 2.3 Agent (External — inference.py)

**Responsibility:** Receive observations, call LLM, produce actions.

**Key behaviors:**
- Reads observation from environment
- Builds prompt from observation
- Calls LLM via OpenAI-compatible client
- Parses LLM output into structured action
- Sends action to environment via `/step`

The agent is **external** to the environment. It communicates via HTTP API only.

---

### 2.4 Reward System

**Responsibility:** Provide dense per-step feedback signal.

See `environment.md` Section 4 for exact reward logic.

Key properties:
- Non-sparse: reward on every step
- Partial credit: correct intent but wrong action still gets partial reward
- Penalties: wrong intent, invalid action, repeated same action (loop detection)

---

## 3. Data Flow (Step-by-Step)

```
Step 1: Agent calls POST /reset
        → Environment initializes state
        → Selects task and gesture sequence
        → Returns Observation (first gesture in sequence)

Step 2: Agent reads observation:
        - gesture_embedding (64-dim array)
        - hand_landmarks (frame sequence)
        - context.current_task
        - context.step_count
        - context.history

Step 3: Agent calls LLM with observation
        → LLM returns: intent, confidence, response

Step 4: Agent calls POST /step with Action:
        { intent, confidence, response }

Step 5: Environment evaluates action:
        - Checks intent against expected intent (lookup table)
        - Checks response against expected response pattern
        - Computes reward
        - Advances gesture_index
        - Updates history
        - Checks done condition

Step 6: Environment returns:
        { observation (next gesture), reward, done, info }

Step 7: Agent logs [STEP] line to stdout

Step 8: If done=True, agent logs [END] line
        Episode complete
```

---

## 4. State Management

### State Schema (Internal)

```json
{
  "task_id": "task1",
  "episode_seed": 42,
  "step_count": 3,
  "max_steps": 10,
  "gesture_sequence": ["HELLO", "YES", "OPEN"],
  "gesture_index": 2,
  "interaction_history": [
    {
      "step": 1,
      "gesture": "HELLO",
      "agent_intent": "greeting",
      "agent_response": "Hello! How can I help you?",
      "reward": 0.5,
      "correct": true
    }
  ],
  "completed_steps": ["HELLO", "YES"],
  "total_reward": 1.0,
  "done": false,
  "final_score": null
}
```

### State Transitions

| Event | State Change |
|-------|-------------|
| `reset()` | All fields reset, gesture_index=0, step_count=0 |
| `step(action)` | step_count+1, gesture_index+1 (if correct), history appended |
| `done=True` | final_score computed by grader, no more steps accepted |

### Done Conditions
1. `gesture_index >= len(gesture_sequence)` — all gestures processed
2. `step_count >= max_steps` — episode limit reached
3. Task-specific: all required sub-tasks completed (Task 2 only)

---

## 5. Assumptions and Constraints

| Assumption | Detail |
|------------|--------|
| Gestures are pre-processed | No CV at runtime, all gesture data is static |
| Gesture vocabulary is fixed | Exactly 20 gestures defined in gestures.json |
| Sequences are deterministic | Same episode_seed → same gesture sequence always |
| LLM output is parsed, not trusted | Agent must validate LLM output before sending |
| Environment runs single-threaded | No concurrent episodes supported |
| Max memory: 8GB | No large models or embeddings loaded at runtime |
| Max vCPU: 2 | FastAPI with single worker, no parallelism |

---

## 6. Design Decisions and Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Static gesture dataset (no CV) | Reproducibility, hardware constraints | Not a real CV system |
| String gesture labels as primary identifier | Simple, deterministic grading | Loses nuance of real hand poses |
| 64-dim float array for frame_features | Mimics real landmark embeddings, adds realism | Unused by grader (only for agent prompt context) |
| Deterministic grader (lookup table) | Reproducible scores across runs | Less flexible than LLM-as-judge |
| FastAPI for API layer | Standard, fast, auto-docs | Single-threaded limitation |
| Fixed max_steps=10 | Prevents infinite loops, keeps inference <20min | Limits complex interactions |
| episode_seed controls randomness | Reproducible baseline runs | Less varied evaluation |
