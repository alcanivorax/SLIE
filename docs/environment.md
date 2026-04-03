# environment.md — OpenEnv Specification

---

## 1. Overview

The SLIE environment follows the OpenEnv specification. All data models use **Pydantic v2**.

The environment is stateful and single-episode. One episode = one task run.

---

## 2. Observation Schema

Returned by `reset()` and after every `step()`.

### Schema (Pydantic)

```python
class GestureContext(BaseModel):
    current_task: str          # "task1" | "task2" | "task3"
    step_count: int            # Current step number (0-indexed before step, 1-indexed after)
    history: list[str]         # List of past interaction summaries (max last 5)

class SLIEObservation(BaseModel):
    gesture_embedding: list[float]     # Fixed 64-element perceptual embedding
    hand_landmarks: list[HandFrame]    # Deterministic landmark sequence for the current sign
    context: GestureContext
```

### Example Observation

```json
{
  "gesture_embedding": [0.12, 0.45, 0.33, 0.78, 0.22, ...],
  "hand_landmarks": [
    {
      "left_hand": [{"x": 0.21, "y": 0.18, "z": -0.04}],
      "right_hand": [{"x": 0.79, "y": 0.18, "z": 0.04}]
    }
  ],
  "context": {
    "current_task": "task1",
    "step_count": 1,
    "history": []
  }
}
```

### Rules
- `gesture_embedding` is ALWAYS exactly 64 floats. Never shorter, never longer.
- `hand_landmarks` contains zero or more frames; each frame contains 21 points per hand.
- The true gesture label is never exposed in the observation.
- `history` contains at most the last 5 interaction summaries as strings.
- History entry format: `"Step {n}: prior_intent={intent}"`

---

## 3. Action Schema

Sent by the agent to `step()`.

### Schema (Pydantic)

```python
class SLIEAction(BaseModel):
    intent: str          # Agent's interpretation of the gesture's meaning
    confidence: float    # Agent's confidence score, range [0.0, 1.0]
    response: str        # Agent's natural language action/response string
```

### Constraints
- `intent`: non-empty string, max 100 characters
- `confidence`: float, must be in [0.0, 1.0]. Values outside this range are clamped silently.
- `response`: non-empty string, max 500 characters

### Example Action

```json
{
  "intent": "greeting",
  "confidence": 0.95,
  "response": "Hello! How can I help you today?"
}
```

---

## 4. Reward Function

Reward is computed per step by the environment after receiving an agent action.

### Reward Components

```
reward = intent_reward + response_reward + completion_bonus + penalties
```

### Intent Reward

```
IF agent.intent == expected_intent (exact match, case-insensitive):
    intent_reward = 0.4
ELIF agent.intent in expected_intent_aliases:
    intent_reward = 0.2
ELSE:
    intent_reward = 0.0
```

`expected_intent` and `expected_intent_aliases` are defined per gesture in `data/tasks.json`.

### Response Reward

```
IF any keyword in expected_keywords is present in agent.response (case-insensitive):
    response_reward = 0.3
ELSE:
    response_reward = 0.0
```

`expected_keywords` is a list of strings defined per gesture step in `data/tasks.json`.

### Completion Bonus

```
IF this is the FINAL gesture in the sequence AND intent_reward > 0:
    completion_bonus = 0.3
ELSE:
    completion_bonus = 0.0
```

### Penalties

```
loop_penalty = 0.0
IF current action.intent == previous action.intent AND current action.response == previous action.response:
    loop_penalty = -0.5     # Exact repeat of last action = loop detected

invalid_penalty = 0.0
IF action.intent == "" OR action.response == "":
    invalid_penalty = -0.3  # Empty fields = invalid action
```

### Final Reward Formula

```
reward = max(-1.0, min(1.0,
    intent_reward + response_reward + completion_bonus + loop_penalty + invalid_penalty
))
```

Reward is always clamped to `[-1.0, 1.0]`.

### Reward Examples

| Scenario | intent_reward | response_reward | completion_bonus | penalties | final_reward |
|----------|--------------|----------------|-----------------|-----------|-------------|
| Perfect match, not final step | 0.4 | 0.3 | 0.0 | 0.0 | 0.7 |
| Perfect match, final step | 0.4 | 0.3 | 0.3 | 0.0 | 1.0 |
| Alias match only | 0.2 | 0.0 | 0.0 | 0.0 | 0.2 |
| Wrong intent, good response | 0.0 | 0.3 | 0.0 | 0.0 | 0.3 |
| Completely wrong | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Repeated action (loop) | 0.4 | 0.3 | 0.0 | -0.5 | 0.2 |
| Empty fields | 0.0 | 0.0 | 0.0 | -0.3 | -0.3 |

---

## 5. Done Conditions

The environment sets `done=True` when ANY of the following is true:

| Condition | Priority |
|-----------|----------|
| `gesture_index >= len(gesture_sequence)` — all gestures processed | 1 (highest) |
| `step_count >= max_steps` (max_steps = 10) | 2 |

When `done=True`:
- `observation.hand_landmarks` = `[]`
- `observation.gesture_embedding` = array of 64 zeros
- `info.final_score` = grader output (float 0.0–1.0)

---

## 6. Info / Debug Fields

Returned in every `step()` response alongside observation, reward, done.

### Schema

```python
class SLIEInfo(BaseModel):
    step_count: int
    final_score: float | None       # null unless done=True
    error: str | None               # null unless something went wrong
```

### Example Info (mid-episode)

```json
{
  "step_count": 2,
  "final_score": null,
  "error": null
}
```

### Example Info (done=True)

```json
{
  "step_count": 5,
  "final_score": 0.83,
  "error": null
}
```

---

## 7. API Response Schemas

### reset() Response

```python
class ResetResponse(BaseModel):
    observation: SLIEObservation
    task_id: str
    episode_seed: int
```

### step() Response

```python
class StepResponse(BaseModel):
    observation: SLIEObservation
    reward: float
    done: bool
    info: SLIEInfo
```

### state() Response

```python
class StateResponse(BaseModel):
    task_id: str
    episode_seed: int
    step_count: int
    max_steps: int
    gesture_index: int
    completed_steps: int
    interaction_history: list[dict]
    total_reward: float
    done: bool
    final_score: float | None
```

---

## 8. Edge Cases

| Edge Case | Behavior |
|-----------|----------|
| `step()` called when `done=True` | Returns error: `{"error": "Episode is done. Call reset() to start a new episode."}` HTTP 400 |
| `step()` called before `reset()` | Returns error: `{"error": "No active episode. Call reset() first."}` HTTP 400 |
| `action.confidence` outside [0.0, 1.0] | Clamped silently to [0.0, 1.0], no error |
| `action.intent` longer than 100 chars | Truncated to 100 chars, no error |
| `action.response` longer than 500 chars | Truncated to 500 chars, no error |
| `action.intent` or `action.response` is empty string | Invalid action penalty (-0.3) applied |
| LLM returns unparseable JSON | Agent must default to `{"intent": "unknown", "confidence": 0.0, "response": "I could not understand."}` |
| gesture_sequence is empty (misconfiguration) | reset() returns HTTP 500 with error message |
