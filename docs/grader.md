# grader.md — Evaluation and Grading System

---

## 1. Grading Principles

- All graders are **fully deterministic** — same inputs always produce same score
- All scores are in range `[0.0, 1.0]`
- Graders use **lookup tables only** — no LLM calls in grader code
- Graders are called **once per episode** when `done=True`
- Graders operate on the full `interaction_history` stored in environment state

---

## 2. Grader Inputs

Each grader receives the complete episode history:

```python
class EpisodeHistory(BaseModel):
    task_id: str                        # "task1" | "task2" | "task3"
    scenario_id: int                    # 0-4
    gesture_sequence: list[str]         # Full gesture sequence for this scenario
    interaction_history: list[dict]     # One entry per step taken
    steps_taken: int                    # Total steps in episode
    max_steps: int                      # Always 10
```

Each entry in `interaction_history`:
```python
{
    "step": int,
    "gesture": str,              # Gesture shown at this step
    "agent_intent": str,         # Intent submitted by agent
    "agent_confidence": float,   # Confidence submitted by agent
    "agent_response": str,       # Response submitted by agent
    "expected_intent": str,      # Ground truth intent
    "intent_aliases": list[str], # Accepted aliases
    "reward": float,             # Reward given this step
    "intent_correct": bool       # True if intent matched
}
```

---

## 3. Task 1 Grader: Command Recognition

### Logic

```python
def task1_grader(history: EpisodeHistory) -> float:
    correct = 0
    total = len(history.gesture_sequence)

    for entry in history.interaction_history:
        agent_intent = entry["agent_intent"].lower().strip()
        expected = entry["expected_intent"].lower().strip()
        aliases = [a.lower().strip() for a in entry["intent_aliases"]]

        if agent_intent == expected or agent_intent in aliases:
            correct += 1

    if total == 0:
        return 0.0

    score = correct / total
    return round(score, 4)
```

### Formula

```
score = correct_intents / total_gestures
```

### Examples

| Scenario | Gestures | Correct Intents | Score |
|----------|----------|----------------|-------|
| All correct | 5 | 5 | 1.00 |
| 4 correct | 5 | 4 | 0.80 |
| 3 correct | 5 | 3 | 0.60 |
| 0 correct | 5 | 0 | 0.00 |
| Episode hit max_steps early | 5 | 3 (only 3 gestures processed) | 0.60 |

### Notes
- If episode ends early (max_steps reached before all gestures processed), only processed gestures count toward denominator
- Denominator = `len(history.interaction_history)` (steps actually taken), NOT `len(gesture_sequence)`

---

## 4. Task 2 Grader: Multi-step Interaction

### Logic

```python
def task2_grader(history: EpisodeHistory) -> float:
    correct = 0
    total = len(history.interaction_history)

    for entry in history.interaction_history:
        agent_intent = entry["agent_intent"].lower().strip()
        expected = entry["expected_intent"].lower().strip()
        aliases = [a.lower().strip() for a in entry["intent_aliases"]]

        if agent_intent == expected or agent_intent in aliases:
            correct += 1

    if total == 0:
        return 0.0

    base_score = correct / total

    # Sequence bonus: +0.1 if ALL intents correct AND all gestures processed
    sequence_bonus = 0.0
    all_processed = len(history.interaction_history) == len(history.gesture_sequence)
    all_correct = correct == total

    if all_correct and all_processed:
        sequence_bonus = 0.1

    score = min(1.0, base_score + sequence_bonus)
    return round(score, 4)
```

### Formula

```
base_score = correct_intents / steps_taken
sequence_bonus = 0.1 if (all_correct AND all_gestures_processed) else 0.0
score = min(1.0, base_score + sequence_bonus)
```

### Examples

| Correct | Total Steps | All Processed? | base_score | bonus | final |
|---------|-------------|----------------|------------|-------|-------|
| 8 | 8 | Yes | 1.00 | 0.10 | 1.00 (capped) |
| 7 | 8 | Yes | 0.875 | 0.00 | 0.875 |
| 4 | 8 | Yes | 0.50 | 0.00 | 0.50 |
| 8 | 8 | No (early end) | 1.00 | 0.00 | 1.00 |
| 0 | 8 | Yes | 0.00 | 0.00 | 0.00 |

---

## 5. Task 3 Grader: Intent Inference

### Logic

Task 3 grading evaluates the **final step only** — the compound intent submitted on the last gesture of the sequence.

```python
def task3_grader(history: EpisodeHistory) -> float:
    if not history.interaction_history:
        return 0.0

    # Get the scenario definition
    scenario = get_scenario(history.task_id, history.scenario_id)
    compound_intent = scenario["compound_intent"].lower().strip()
    intent_aliases = [a.lower().strip() for a in scenario["intent_aliases"]]
    expected_keywords = [k.lower().strip() for k in scenario["expected_keywords"]]

    # Get the last step's action
    last_entry = history.interaction_history[-1]
    agent_intent = last_entry["agent_intent"].lower().strip()
    agent_response = last_entry["agent_response"].lower()

    # Score the compound intent
    if agent_intent == compound_intent:
        compound_score = 1.0
    elif agent_intent in intent_aliases:
        compound_score = 0.7
    elif any(kw in agent_intent for kw in expected_keywords):
        compound_score = 0.4
    elif any(kw in agent_response for kw in expected_keywords):
        compound_score = 0.2
    else:
        compound_score = 0.0

    # Per-step accuracy bonus (0-0.2 range) for intermediate steps
    intermediate_steps = history.interaction_history[:-1]
    if intermediate_steps:
        step_correct = sum(
            1 for e in intermediate_steps
            if e["agent_intent"].lower() == e["expected_intent"].lower()
            or e["agent_intent"].lower() in [a.lower() for a in e["intent_aliases"]]
        )
        step_ratio = step_correct / len(intermediate_steps)
        step_bonus = step_ratio * 0.2
    else:
        step_bonus = 0.0

    # Final score: compound intent is primary (80%), step accuracy is bonus (20%)
    score = min(1.0, compound_score * 0.8 + step_bonus)
    return round(score, 4)
```

### Formula

```
compound_score:
    1.0  if agent_intent == compound_intent (exact match)
    0.7  if agent_intent in intent_aliases
    0.4  if any expected_keyword in agent_intent
    0.2  if any expected_keyword in agent_response
    0.0  otherwise

step_bonus = (correct_intermediate_steps / total_intermediate_steps) * 0.2

final_score = min(1.0, compound_score * 0.8 + step_bonus)
```

### Examples

| Compound Intent Match | Step Accuracy | compound_score | step_bonus | final |
|----------------------|---------------|----------------|------------|-------|
| Exact match | 100% | 1.0 | 0.20 | 1.00 (capped) |
| Exact match | 50% | 1.0 | 0.10 | 0.90 |
| Alias match | 100% | 0.7 | 0.20 | 0.76 |
| Alias match | 0% | 0.7 | 0.00 | 0.56 |
| Keyword in intent | 100% | 0.4 | 0.20 | 0.52 |
| Keyword in response only | 50% | 0.2 | 0.10 | 0.26 |
| No match | 0% | 0.0 | 0.00 | 0.00 |

---

## 6. Grader Invocation

Graders are called by the environment at episode end:

```python
def compute_final_score(state: EnvironmentState) -> float:
    history = EpisodeHistory(
        task_id=state.task_id,
        scenario_id=state.scenario_id,
        gesture_sequence=state.gesture_sequence,
        interaction_history=state.interaction_history,
        steps_taken=state.step_count,
        max_steps=state.max_steps
    )

    if state.task_id == "task1":
        return task1_grader(history)
    elif state.task_id == "task2":
        return task2_grader(history)
    elif state.task_id == "task3":
        return task3_grader(history)
    else:
        raise ValueError(f"Unknown task_id: {state.task_id}")
```

Result is stored in `state.final_score` and returned in `info.final_score` when `done=True`.

---

## 7. Determinism Guarantees

| Property | Guarantee |
|----------|-----------|
| Same inputs → same score | ✅ Always (no randomness in graders) |
| LLM not used in grading | ✅ Pure lookup/comparison |
| Floating point consistency | ✅ All scores rounded to 4 decimal places |
| String comparison | ✅ Always `.lower().strip()` before comparison |
| Alias matching | ✅ Exact string match only (no fuzzy matching) |

---

## 8. Grader Anti-exploitation Rules

The graders are designed to resist gaming:

1. **No partial credit for padding**: Outputting very long responses does not increase score
2. **No alias guessing reward**: Aliases are narrow and specific (e.g., `"hi"` not `"any positive word"`)
3. **Loop penalty in reward**: Repeating same action penalized in real-time reward (not just final score)
4. **Keyword matching is intent-level only**: For Task 3, response keyword match gives only 0.2 max
5. **Confidence score not used in grading**: Agent cannot game score by outputting high confidence
