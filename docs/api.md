# api.md — API Specification

---

## 1. Overview

The SLIE environment exposes a REST API via **FastAPI** running on port **8000**.

Base URL (local): `http://localhost:8000`
Base URL (HuggingFace Space): `https://<space-name>.hf.space`

All requests and responses use `Content-Type: application/json`.

---

## 2. Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Take one action step |
| GET | `/state` | Get current environment state |
| GET | `/health` | Health check (returns 200 OK) |

---

## 3. POST /reset

### Purpose
Initialize a new episode. Must be called before any `/step` calls.
Clears all previous state.

### Request Schema

```json
{
  "task_id": "task1",
  "episode_seed": 42
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `task_id` | string | Yes | — | One of: `"task1"`, `"task2"`, `"task3"` |
| `episode_seed` | integer | No | `0` | Controls which scenario is selected (`scenario_id = episode_seed % 5`) |

### Response Schema

```json
{
  "observation": {
    "gesture_embedding": [0.12, 0.45, 0.33, 0.78, 0.22, 0.67, 0.89, 0.11, 0.55, 0.44, 0.76, 0.23, 0.88, 0.34, 0.65, 0.92, 0.18, 0.56, 0.73, 0.41, 0.29, 0.84, 0.62, 0.37, 0.91, 0.14, 0.58, 0.79, 0.26, 0.43, 0.71, 0.85, 0.19, 0.64, 0.38, 0.96, 0.27, 0.53, 0.82, 0.47, 0.16, 0.69, 0.35, 0.98, 0.21, 0.74, 0.60, 0.87, 0.32, 0.50, 0.77, 0.13, 0.66, 0.39, 0.93, 0.25, 0.81, 0.48, 0.59, 0.70, 0.36, 0.94, 0.28, 0.15],
    "hand_landmarks": [{"left_hand": [], "right_hand": []}],
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

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success — episode initialized |
| 422 | Validation error — invalid task_id or wrong types |
| 500 | Internal error — tasks.json misconfigured |

### Example Call (Python)

```python
import requests
response = requests.post("http://localhost:8000/reset", json={
    "task_id": "task1",
    "episode_seed": 42
})
data = response.json()
observation = data["observation"]
```

---

## 4. POST /step

### Purpose
Submit one agent action. Returns next observation, reward, done flag, and debug info.

### Request Schema

```json
{
  "intent": "greeting",
  "confidence": 0.95,
  "response": "Hello! How can I help you today?"
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `intent` | string | Yes | Non-empty, max 100 chars | Agent's interpretation of gesture |
| `confidence` | float | Yes | [0.0, 1.0] | Agent's confidence in intent |
| `response` | string | Yes | Non-empty, max 500 chars | Natural language action response |

### Response Schema

```json
{
  "observation": {
    "gesture_embedding": [0.34, 0.78, 0.22, ...],
    "hand_landmarks": [{"left_hand": [], "right_hand": []}],
    "context": {
      "current_task": "task1",
      "step_count": 1,
      "history": [
        "Step 1: prior_intent=greeting"
      ]
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

### Response When done=True

```json
{
  "observation": {
    "gesture_embedding": [0.0, 0.0, 0.0, ...],
    "hand_landmarks": [],
    "context": {
      "current_task": "task1",
      "step_count": 5,
      "history": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ...", "Step 5: ..."]
    }
  },
  "reward": 1.0,
  "done": true,
  "info": {
    "step_count": 5,
    "final_score": 0.833,
    "error": null
  }
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success — step processed |
| 400 | Bad request — episode done or no active episode |
| 422 | Validation error — invalid action schema |

### Error Response (400)

```json
{
  "detail": "Episode is done. Call reset() to start a new episode."
}
```

or

```json
{
  "detail": "No active episode. Call reset() first."
}
```

### Example Call (Python)

```python
import requests
response = requests.post("http://localhost:8000/step", json={
    "intent": "greeting",
    "confidence": 0.95,
    "response": "Hello! How can I help you today?"
})
data = response.json()
reward = data["reward"]
done = data["done"]
next_obs = data["observation"]
```

---

## 5. GET /state

### Purpose
Returns the full internal environment state. Read-only. Does not modify state.
Useful for debugging and external monitoring.

### Request
No request body. No query parameters.

```
GET /state
```

### Response Schema

```json
{
  "task_id": "task1",
  "episode_seed": 42,
  "step_count": 2,
  "max_steps": 10,
  "gesture_index": 2,
  "completed_steps": 2,
  "interaction_history": [
    {
      "step": 1,
      "agent_intent": "greeting",
      "agent_confidence": 0.95,
      "agent_response": "Hello! How can I help you today?",
      "reward": 0.7
    },
    {
      "step": 2,
      "agent_intent": "confirm",
      "agent_confidence": 0.90,
      "agent_response": "Confirmed. Proceeding.",
      "reward": 0.7
    }
  ],
  "total_reward": 1.4,
  "done": false,
  "final_score": null
}
```

### Response When No Episode Active

```json
{
  "task_id": null,
  "episode_seed": null,
  "step_count": 0,
  "max_steps": 10,
  "gesture_index": 0,
  "completed_steps": 0,
  "interaction_history": [],
  "total_reward": 0.0,
  "done": false,
  "final_score": null
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Always 200 (even if no episode active) |

---

## 6. GET /health

### Purpose
Health check endpoint. Used by HuggingFace Spaces and monitoring systems.

### Request
```
GET /health
```

### Response

```json
{
  "status": "ok",
  "environment": "slie",
  "version": "1.1.0"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Service is healthy |
| 500 | Service is down (should never happen in normal operation) |

---

## 7. Full Episode Flow (API Sequence)

```
1. GET  /health              → 200 OK (verify service is up)
2. POST /reset               → observation (first hand-sign embedding + landmarks)
3. POST /step (action 1)     → observation, reward, done=false
4. POST /step (action 2)     → observation, reward, done=false
   ...
N. POST /step (action N)     → observation, reward, done=true, final_score
   (GET /state at any point for debugging)
```

---

## 8. openenv.yaml

The environment must include this file at project root:

```yaml
name: slie
version: "1.1.0"
description: "Sign Language Interaction Environment — AI agent interprets hand-sign landmark observations and performs tasks via reasoning and intent inference."
author: "<author-name>"
tags:
  - openenv
  - accessibility
  - sign-language
  - nlp
  - agent-evaluation

tasks:
  - id: task1
    name: "Command Recognition"
    difficulty: easy
    max_steps: 10

  - id: task2
    name: "Multi-step Interaction"
    difficulty: medium
    max_steps: 10

  - id: task3
    name: "Intent Inference"
    difficulty: hard
    max_steps: 10

observation_space:
  type: object
  properties:
    gesture_embedding:
      type: array
      items: {type: number}
      length: 64
    hand_landmarks:
      type: array
      items:
        type: object
    context:
      type: object

action_space:
  type: object
  properties:
    intent:
      type: string
    confidence:
      type: number
      minimum: 0.0
      maximum: 1.0
    response:
      type: string

endpoints:
  reset: POST /reset
  step: POST /step
  state: GET /state
  health: GET /health

docker:
  port: 8000
  build_context: "."
```
