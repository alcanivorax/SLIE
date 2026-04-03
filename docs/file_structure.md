# file_structure.md — Project File Structure

---

## Complete Directory Tree

```
slie/                              ← Project root
│
├── main.py                        ← Entry point: starts uvicorn server
├── inference.py                   ← Agent baseline script (MANDATORY: root dir)
├── openenv.yaml                   ← OpenEnv metadata and schema definitions
├── requirements.txt               ← Python dependencies with pinned versions
├── Dockerfile                     ← Container definition for deployment
├── README.md                      ← Project documentation
├── .env.example                   ← Example environment variables
│
├── slie/                          ← Core environment package
│   ├── __init__.py                ← Package init (empty or version string)
│   ├── models.py                  ← All Pydantic models (schemas)
│   ├── state.py                   ← EnvironmentState class (internal state)
│   ├── data_loader.py             ← Load gestures.json and tasks.json
│   ├── gesture_layer.py           ← GestureInputLayer: builds observations
│   ├── reward.py                  ← compute_reward() function
│   ├── graders.py                 ← task1/2/3 graders + compute_final_score()
│   ├── env.py                     ← SLIEEnvironment class (main orchestrator)
│   └── app.py                     ← FastAPI app with all endpoints
│
├── data/                          ← Static data files (read-only at runtime)
│   ├── gestures.json              ← 24 gesture definitions with frame_features
│   └── tasks.json                 ← All task/scenario definitions (3 tasks × 5 scenarios)
│
├── docs/                          ← Documentation (reference only, not served)
│   ├── overview.md
│   ├── design.md
│   ├── environment.md
│   ├── tasks.md
│   ├── agent.md
│   ├── grader.md
│   ├── api.md
│   ├── constraints.md
│   ├── build_plan.md
│   └── file_structure.md
│
└── tests/                         ← Unit tests
    ├── __init__.py
    ├── test_models.py             ← Pydantic model validation tests
    ├── test_reward.py             ← Reward function tests
    ├── test_graders.py            ← Grader output tests
    ├── test_env.py                ← Full episode integration tests
    └── test_api.py                ← API endpoint tests (TestClient)
```

---

## File Descriptions

### Root Level

| File | Purpose | Required? |
|------|---------|-----------|
| `main.py` | Starts uvicorn with `slie.app:app` on port 8000 | ✅ Yes |
| `inference.py` | Baseline agent — runs all 3 tasks, emits logs | ✅ Yes (MUST be in root) |
| `openenv.yaml` | OpenEnv spec metadata — tasks, schemas, endpoints | ✅ Yes |
| `requirements.txt` | Pinned Python dependencies | ✅ Yes |
| `Dockerfile` | `FROM python:3.11-slim` + install + expose 8000 | ✅ Yes |
| `README.md` | Human-readable documentation | ✅ Yes |
| `.env.example` | Template for required env vars | Recommended |

### slie/ Package

| File | Key Contents | Dependencies |
|------|-------------|-------------|
| `__init__.py` | Empty or `__version__ = "1.1.0"` | None |
| `models.py` | `SLIEObservation`, `SLIEAction`, `SLIEInfo`, `GestureContext`, `ResetRequest`, `ResetResponse`, `StepResponse`, `StateResponse`, `EpisodeHistory` | pydantic |
| `state.py` | `EnvironmentState` dataclass, `reset_state()`, `to_state_response()` | `models.py` |
| `data_loader.py` | `load_gestures()`, `load_tasks()`, `get_scenario()`, `get_gesture_features()` | json, os |
| `gesture_layer.py` | `GestureInputLayer`, `get_observation()`, `get_step_spec()` | `models.py`, `data_loader.py` |
| `reward.py` | `compute_reward(action, step_spec, last_action, is_final_step) -> (float, dict)` | `models.py` |
| `graders.py` | `task1_grader()`, `task2_grader()`, `task3_grader()`, `compute_final_score()` | `models.py`, `data_loader.py` |
| `env.py` | `SLIEEnvironment` class: orchestrates reset/step/state | All above |
| `app.py` | FastAPI app, single global `SLIEEnvironment` instance, 4 endpoints | `env.py`, fastapi |

### data/ Directory

| File | Format | Contents |
|------|--------|---------|
| `gestures.json` | JSON object | Keys = gesture labels (24 total), values = `{label, frame_features[64], category}` |
| `tasks.json` | JSON object | Keys = `"task1"`, `"task2"`, `"task3"`, each with `scenarios` array of 5 items |

#### gestures.json Structure
```json
{
  "HELLO": {
    "label": "HELLO",
    "frame_features": [64 floats],
    "category": "greeting"
  },
  "YES": { ... },
  ...
}
```

#### tasks.json Structure
```json
{
  "task1": {
    "scenarios": [
      {
        "id": 0,
        "gesture_sequence": ["HELLO", "YES", "STOP", "HELP", "GOODBYE"],
        "steps": [
          {
            "gesture": "HELLO",
            "expected_intent": "greeting",
            "intent_aliases": ["hello", "hi", "welcome"],
            "expected_keywords": ["hello", "hi", "greet", "welcome", "help"]
          },
          ...
        ]
      },
      ...
    ]
  },
  "task2": { "scenarios": [...] },
  "task3": { "scenarios": [...] }
}
```

### tests/ Directory

| File | What It Tests |
|------|--------------|
| `test_models.py` | Model instantiation, validation, edge cases (empty strings, out-of-range confidence) |
| `test_reward.py` | All reward scenarios from grader.md reward table |
| `test_graders.py` | All 3 graders with known inputs → expected scores |
| `test_env.py` | Full episode: reset → N steps → done=True → final_score |
| `test_api.py` | All 4 endpoints with FastAPI TestClient |

---

## Import Graph

```
app.py
  └── env.py
        ├── models.py
        ├── state.py
        │     └── models.py
        ├── data_loader.py
        ├── gesture_layer.py
        │     ├── models.py
        │     └── data_loader.py
        ├── reward.py
        │     └── models.py
        └── graders.py
              ├── models.py
              └── data_loader.py

inference.py (standalone)
  └── openai, requests (external)
```

---

## Environment Variables

| Variable | Used By | Description | Example |
|----------|---------|-------------|---------|
| `API_BASE_URL` | `inference.py` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | `inference.py` | LLM model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | `inference.py` | HuggingFace / API key | `hf_xxxxxxxxxxxx` |
| `PORT` | `main.py` (optional) | Override default port 8000 | `8000` |

### .env.example
```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=hf_your_token_here
```

---

## File Count Summary

| Category | Count |
|----------|-------|
| Root files | 7 |
| slie/ package files | 9 |
| data/ files | 2 |
| docs/ files | 10 |
| tests/ files | 5 |
| **Total** | **33** |

---

## Critical File Checklist (Pre-Submission)

Before submitting, verify ALL of these exist and are non-empty:

- [ ] `inference.py` — in root, not in subdirectory
- [ ] `openenv.yaml` — valid YAML, all required fields present
- [ ] `Dockerfile` — builds successfully
- [ ] `requirements.txt` — all deps listed
- [ ] `README.md` — non-empty
- [ ] `slie/app.py` — FastAPI app importable
- [ ] `slie/models.py` — all Pydantic models defined
- [ ] `data/gestures.json` — 24 gestures, each with 64 frame_features
- [ ] `data/tasks.json` — 3 tasks, 5 scenarios each
- [ ] `main.py` — starts server on port 8000
