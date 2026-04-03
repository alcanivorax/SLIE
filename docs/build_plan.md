# build_plan.md — Implementation Plan

---

## Implementation Order

Phases must be implemented **in order**. Each phase has hard dependencies on the previous.

```
Phase 1: Core Environment (models + state)
    ↓
Phase 2: Task System (data + gesture loading)
    ↓
Phase 3: Reward Engine + Graders
    ↓
Phase 4: API Layer (FastAPI endpoints)
    ↓
Phase 5: Agent (inference.py)
    ↓
Phase 6: Deployment (Docker + HuggingFace)
```

---

## Phase 1: Core Environment

### Goal
Implement all Pydantic models and the state management class.

### Deliverables

| File | Content |
|------|---------|
| `slie/models.py` | All Pydantic models: `SLIEObservation`, `SLIEAction`, `SLIEInfo`, `ResetRequest`, `ResetResponse`, `StepResponse`, `StateResponse`, `EpisodeHistory` |
| `slie/state.py` | `EnvironmentState` class with all fields and reset logic |

### Implementation Steps

1. Create `slie/models.py`:
   - Define `GestureContext(BaseModel)` with `current_task: str`, `step_count: int`, `history: list[str]`
   - Define `SLIEObservation(BaseModel)` with `gesture_embedding: list[float]`, `hand_landmarks: list[HandFrame]`, `context: GestureContext`
   - Define `SLIEAction(BaseModel)` with `intent: str`, `confidence: float`, `response: str`
   - Add validators: `confidence` clamped [0.0, 1.0], `intent` max 100 chars, `response` max 500 chars
   - Define `SLIEInfo(BaseModel)` per schema in `environment.md`
   - Define `ResetRequest`, `ResetResponse`, `StepResponse`, `StateResponse`

2. Create `slie/state.py`:
   - Define `EnvironmentState` dataclass (not Pydantic — internal use only)
   - Fields: `task_id`, `episode_seed`, `scenario_id`, `step_count`, `max_steps=10`, `gesture_sequence`, `gesture_index`, `interaction_history`, `completed_steps`, `total_reward`, `done`, `final_score`, `last_action`
   - Method `reset_state(task_id, episode_seed, gesture_sequence)`: resets all fields
   - Method `to_state_response()`: converts to `StateResponse` Pydantic model

### Dependencies
- Python 3.11+
- pydantic>=2.0

### Validation Checks
- `from slie.models import SLIEObservation, SLIEAction` imports without error
- `SLIEAction(intent="greeting", confidence=1.5, response="hi")` clamps confidence to 1.0
- `SLIEAction(intent="", confidence=0.5, response="hi")` raises or passes (empty string handled in reward engine, not here)
- `EnvironmentState` initializes with all fields set to defaults

---

## Phase 2: Task System

### Goal
Implement gesture data loading, task scenario selection, and observation generation.

### Deliverables

| File | Content |
|------|---------|
| `data/gestures.json` | 24 gesture definitions, each with `label`, `frame_features` (64 floats), `category` |
| `data/tasks.json` | Task definitions for task1, task2, task3 — each with 5 scenarios |
| `slie/data_loader.py` | Functions to load gestures and tasks, select scenarios |
| `slie/gesture_layer.py` | `GestureInputLayer` class: provides observations from gesture sequences |

### Implementation Steps

1. Create `data/gestures.json`:
   - 24 gestures: HELLO, YES, NO, STOP, HELP, GOODBYE, OPEN, CLOSE, FOOD, WATER, MUSIC, PLAY, LOUDER, CALL, WAIT, YOUTUBE, SEARCH, NEAR, NOW, MAP, HOME, QUIET, SLEEP, COLD
   - Each gesture: 64 floats (use `round(random.seed(hash(label)); random.random(), 4)` — fixed seed per label for reproducibility)
   - Categories: greeting, confirmation, control, request, navigation, media, communication

2. Create `data/tasks.json`:
   - Full content as specified in `tasks.md`
   - task1: 5 scenarios × 5 gestures each
   - task2: 5 scenarios × 8 gestures each
   - task3: 5 scenarios × 3 gestures each

3. Create `slie/data_loader.py`:
   ```python
   def load_gestures() -> dict:
       # Load data/gestures.json, return dict keyed by gesture label
   
   def load_tasks() -> dict:
       # Load data/tasks.json, return full tasks dict
   
   def get_scenario(task_id: str, scenario_id: int) -> dict:
       # Return specific scenario from tasks dict
       # scenario_id = episode_seed % 5
   
   def get_gesture_features(gesture_label: str, gestures: dict) -> list[float]:
       # Return 64-dim frame_features for given gesture label
       # Raise ValueError if gesture not found
   ```

4. Create `slie/gesture_layer.py`:
   ```python
   class GestureInputLayer:
       def __init__(self, gestures: dict, scenario: dict):
           self.gestures = gestures
           self.scenario = scenario
           self.sequence = scenario["gesture_sequence"]
           self.steps = scenario["steps"]
   
       def get_observation(self, gesture_index: int, step_count: int, history: list[str], task_id: str) -> SLIEObservation:
           # If gesture_index >= len(sequence): return done observation (nulls + zeros)
           # Else: return observation with gesture_embedding + hand_landmarks + context
   
       def get_step_spec(self, gesture_index: int) -> dict:
           # Return step spec (expected_intent, aliases, keywords) for this index
   ```

### Dependencies
- Phase 1 complete
- `data/` directory writable

### Validation Checks
- `load_gestures()` returns dict with exactly 24 keys
- Every gesture has `frame_features` of length 64
- `get_scenario("task1", 0)` returns dict with `gesture_sequence` of length 5
- `get_scenario("task2", 0)` returns dict with `gesture_sequence` of length 8
- `get_scenario("task3", 0)` returns dict with `gesture_sequence` of length 3
- `GestureInputLayer.get_observation(0, 0, [], "task1")` returns valid `SLIEObservation`
- `GestureInputLayer.get_observation(99, 5, [...], "task1")` returns done observation (gesture=null)

---

## Phase 3: Reward Engine + Graders

### Goal
Implement the reward computation logic and all three task graders.

### Deliverables

| File | Content |
|------|---------|
| `slie/reward.py` | `compute_reward(action, step_spec, last_action, is_final_step)` function |
| `slie/graders.py` | `task1_grader`, `task2_grader`, `task3_grader`, `compute_final_score` functions |

### Implementation Steps

1. Create `slie/reward.py`:
   ```python
   def compute_reward(
       action: SLIEAction,
       step_spec: dict,
       last_action: SLIEAction | None,
       is_final_step: bool
   ) -> tuple[float, dict]:
       # Returns (reward, debug_info)
       # Implement exact formula from environment.md Section 4
       # intent_reward: 0.4 exact, 0.2 alias, 0.0 else
       # response_reward: 0.3 if any keyword in response, else 0.0
       # completion_bonus: 0.3 if is_final_step and intent correct
       # loop_penalty: -0.5 if exact repeat of last action
       # invalid_penalty: -0.3 if empty intent or response
       # clamp to [-1.0, 1.0]
   ```

2. Create `slie/graders.py`:
   - Implement `task1_grader(history: EpisodeHistory) -> float` per grader.md Section 3
   - Implement `task2_grader(history: EpisodeHistory) -> float` per grader.md Section 4
   - Implement `task3_grader(history: EpisodeHistory) -> float` per grader.md Section 5
   - Implement `compute_final_score(state: EnvironmentState) -> float` dispatcher

### Dependencies
- Phase 1 and Phase 2 complete

### Validation Checks

Reward tests:
- Perfect intent + keyword response + not final → 0.7
- Perfect intent + keyword response + is final → 1.0 (clamped)
- Alias intent only → 0.2
- Repeated action → loop penalty applied
- Empty intent → -0.3

Grader tests:
- `task1_grader` with all correct → 1.0
- `task1_grader` with 3/5 correct → 0.6
- `task2_grader` with all correct + all processed → 1.0 (bonus applied, capped)
- `task3_grader` with exact compound intent + 100% steps → 1.0 (capped)
- `task3_grader` with no match → 0.0

---

## Phase 4: API Layer

### Goal
Implement FastAPI app with `/reset`, `/step`, `/state`, `/health` endpoints.

### Deliverables

| File | Content |
|------|---------|
| `slie/env.py` | `SLIEEnvironment` class: wraps state, gesture_layer, reward, grader |
| `slie/app.py` | FastAPI app with all 4 endpoints |
| `main.py` | Entry point: `uvicorn slie.app:app --host 0.0.0.0 --port 8000` |

### Implementation Steps

1. Create `slie/env.py`:
   ```python
   class SLIEEnvironment:
       def __init__(self):
           self.state = EnvironmentState()
           self.gestures = load_gestures()
           self.tasks = load_tasks()
           self.gesture_layer = None  # Set on reset()

       def reset(self, task_id: str, episode_seed: int) -> ResetResponse:
           scenario = get_scenario(task_id, episode_seed % 5)
           self.gesture_layer = GestureInputLayer(self.gestures, scenario)
           self.state.reset_state(task_id, episode_seed, scenario["gesture_sequence"])
           obs = self.gesture_layer.get_observation(0, 0, [], task_id)
           return ResetResponse(observation=obs, task_id=task_id, episode_seed=episode_seed)

       def step(self, action: SLIEAction) -> StepResponse:
           # Validate episode is active
           # Get current step spec
           # Compute reward
           # Update state (step_count, gesture_index, history, total_reward)
           # Check done conditions
           # If done: compute final_score via grader
           # Build and return StepResponse

       def get_state(self) -> StateResponse:
           return self.state.to_state_response()
   ```

2. Create `slie/app.py`:
   ```python
   app = FastAPI(title="SLIE", version="1.0.0")
   env = SLIEEnvironment()  # Single global instance

   @app.post("/reset") -> ResetResponse
   @app.post("/step") -> StepResponse
   @app.get("/state") -> StateResponse
   @app.get("/health")
   ```

3. Create `main.py`:
   ```python
   import uvicorn
   if __name__ == "__main__":
       uvicorn.run("slie.app:app", host="0.0.0.0", port=8000, workers=1)
   ```

### Dependencies
- Phases 1–3 complete
- `fastapi`, `uvicorn`, `pydantic>=2.0`

### Validation Checks
- `curl http://localhost:8000/health` → `{"status": "ok", ...}`
- `POST /reset` with valid body → 200 with observation
- `POST /step` before reset → 400 error
- `POST /step` after done → 400 error
- `GET /state` before reset → 200 with null task_id
- Full episode runs to completion: reset + 5 steps → done=true + final_score

---

## Phase 5: Agent (inference.py)

### Goal
Implement the baseline inference script that runs all 3 tasks and emits required logs.

### Deliverables

| File | Content |
|------|---------|
| `inference.py` | Full agent script (root directory) |

### Implementation Steps

1. Read all env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
2. Implement `log_start()`, `log_step()`, `log_end()` functions (exact format per agent.md)
3. Implement `build_prompt(obs, last_reward, history, step, task_id)` per agent.md Section 9
4. Implement `call_llm(client, prompt) -> str` with try/except
5. Implement `parse_action(llm_output) -> dict` with fallback to `{"intent": "unknown", "confidence": 0.0, "response": "I could not understand."}`
6. Implement `run_task(client, env_url, task_id, seed)` — full episode loop
7. `main()`: run task1 (seed=0), task2 (seed=0), task3 (seed=0) sequentially

### Critical Rules for inference.py
- Must use `requests` (sync HTTP) — no async required
- `[START]` emitted once per task run
- `[STEP]` emitted immediately after each `step()` call
- `[END]` emitted in `finally` block — always runs even on exception
- `action` field in `[STEP]` = `action["intent"]` string only
- `score` in `[END]` = `info.final_score` when available, else `sum(rewards)/len(rewards)`

### Dependencies
- Phase 4 complete and running
- `openai`, `requests` packages

### Validation Checks
- Script runs without import errors
- Produces exactly one `[START]` per task
- Produces `[STEP]` lines equal to number of steps taken
- Produces exactly one `[END]` per task
- `[END]` always printed even if exception occurs
- All reward values formatted to 2 decimal places
- score formatted to 3 decimal places
- Script completes all 3 tasks in under 20 minutes

---

## Phase 6: Deployment

### Goal
Containerize and deploy to HuggingFace Spaces.

### Deliverables

| File | Content |
|------|---------|
| `Dockerfile` | Container definition |
| `requirements.txt` | All Python dependencies with versions |
| `openenv.yaml` | OpenEnv metadata (per api.md Section 8) |
| `README.md` | Full documentation |
| `.env.example` | Example environment variables |

### Implementation Steps

1. Create `requirements.txt`:
   ```
   fastapi==0.111.0
   uvicorn==0.29.0
   pydantic==2.7.1
   requests==2.31.0
   openai==1.30.0
   ```

2. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["python", "main.py"]
   ```

3. Create `openenv.yaml` (exact content in api.md Section 8)

4. Create `README.md`:
   - Environment description
   - Action/observation space
   - Task descriptions
   - Setup instructions (docker build + run)
   - Baseline scores table
   - How to run inference.py

5. Test deployment:
   ```bash
   docker build -t slie .
   docker run -p 8000:8000 slie
   curl http://localhost:8000/health
   ```

6. Push to HuggingFace:
   - Create HF Space: type=Docker, SDK=docker
   - Push repository with all files
   - Set Space secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`
   - Tag space with `openenv`

### Dependencies
- All phases complete
- Docker installed
- HuggingFace account and CLI (`huggingface-cli`)

### Validation Checks
- `docker build -t slie .` exits 0
- `docker run -p 8000:8000 slie` starts without error
- `curl http://localhost:8000/health` returns 200
- `curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id":"task1","episode_seed":0}'` returns observation
- HF Space URL returns 200 on ping
- `inference.py` runs end-to-end and produces scores for all 3 tasks
- `openenv validate` passes (if validator is available)
