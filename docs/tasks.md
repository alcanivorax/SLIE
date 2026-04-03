# tasks.md — Task Definitions

---

## Task Overview

| Task ID | Name | Difficulty | Max Steps | Gesture Sequence Length |
|---------|------|------------|-----------|------------------------|
| task1 | Command Recognition | Easy | 10 | 5 gestures |
| task2 | Multi-step Interaction | Medium | 10 | 8 gestures |
| task3 | Intent Inference | Hard | 10 | 6 gestures (ambiguous) |

All gesture sequences are defined in `data/tasks.json`. Each task has **5 fixed scenarios** (indexed 0–4). The active scenario is selected by `episode_seed % 5`.

---

## Task 1: Command Recognition (Easy)

### Objective
The agent must correctly identify the intent of individual, unambiguous gestures and produce an appropriate response. Each gesture maps to exactly one expected intent.

### Difficulty Rationale
- One gesture at a time (no context required)
- Unambiguous mappings (one correct intent per gesture)
- Expected responses have many valid keywords
- No memory required

### Input Format (Gesture Sequences)

All 5 scenarios for Task 1:

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
          {
            "gesture": "YES",
            "expected_intent": "confirm",
            "intent_aliases": ["yes", "agree", "ok", "okay"],
            "expected_keywords": ["yes", "confirm", "ok", "proceed", "sure"]
          },
          {
            "gesture": "STOP",
            "expected_intent": "halt",
            "intent_aliases": ["stop", "pause", "end", "cease"],
            "expected_keywords": ["stop", "halt", "pause", "end", "cancel"]
          },
          {
            "gesture": "HELP",
            "expected_intent": "request_help",
            "intent_aliases": ["help", "assist", "support"],
            "expected_keywords": ["help", "assist", "support", "how", "what"]
          },
          {
            "gesture": "GOODBYE",
            "expected_intent": "farewell",
            "intent_aliases": ["bye", "goodbye", "exit", "leave"],
            "expected_keywords": ["goodbye", "bye", "later", "farewell", "exit"]
          }
        ]
      },
      {
        "id": 1,
        "gesture_sequence": ["OPEN", "CLOSE", "YES", "NO", "HELP"],
        "steps": [
          {
            "gesture": "OPEN",
            "expected_intent": "open_action",
            "intent_aliases": ["open", "start", "launch", "begin"],
            "expected_keywords": ["open", "launch", "start", "begin"]
          },
          {
            "gesture": "CLOSE",
            "expected_intent": "close_action",
            "intent_aliases": ["close", "shut", "exit", "end"],
            "expected_keywords": ["close", "shut", "exit", "stop"]
          },
          {
            "gesture": "YES",
            "expected_intent": "confirm",
            "intent_aliases": ["yes", "agree", "ok", "okay"],
            "expected_keywords": ["yes", "confirm", "ok", "proceed"]
          },
          {
            "gesture": "NO",
            "expected_intent": "deny",
            "intent_aliases": ["no", "reject", "cancel", "refuse"],
            "expected_keywords": ["no", "reject", "cancel", "deny"]
          },
          {
            "gesture": "HELP",
            "expected_intent": "request_help",
            "intent_aliases": ["help", "assist", "support"],
            "expected_keywords": ["help", "assist", "support"]
          }
        ]
      },
      {
        "id": 2,
        "gesture_sequence": ["FOOD", "WATER", "YES", "STOP", "HELLO"],
        "steps": [
          { "gesture": "FOOD", "expected_intent": "request_food", "intent_aliases": ["food", "eat", "hungry", "meal"], "expected_keywords": ["food", "eat", "meal", "restaurant", "hungry"] },
          { "gesture": "WATER", "expected_intent": "request_water", "intent_aliases": ["water", "drink", "thirsty"], "expected_keywords": ["water", "drink", "thirsty"] },
          { "gesture": "YES", "expected_intent": "confirm", "intent_aliases": ["yes", "agree", "ok"], "expected_keywords": ["yes", "confirm", "ok"] },
          { "gesture": "STOP", "expected_intent": "halt", "intent_aliases": ["stop", "pause", "end"], "expected_keywords": ["stop", "halt", "pause"] },
          { "gesture": "HELLO", "expected_intent": "greeting", "intent_aliases": ["hello", "hi", "welcome"], "expected_keywords": ["hello", "hi", "greet"] }
        ]
      },
      {
        "id": 3,
        "gesture_sequence": ["MUSIC", "PLAY", "YES", "LOUDER", "STOP"],
        "steps": [
          { "gesture": "MUSIC", "expected_intent": "request_music", "intent_aliases": ["music", "song", "audio", "play_music"], "expected_keywords": ["music", "song", "play", "audio"] },
          { "gesture": "PLAY", "expected_intent": "play_action", "intent_aliases": ["play", "start", "begin", "resume"], "expected_keywords": ["play", "start", "begin", "resume"] },
          { "gesture": "YES", "expected_intent": "confirm", "intent_aliases": ["yes", "agree", "ok"], "expected_keywords": ["yes", "confirm", "ok"] },
          { "gesture": "LOUDER", "expected_intent": "increase_volume", "intent_aliases": ["louder", "volume_up", "increase"], "expected_keywords": ["louder", "volume", "increase", "up"] },
          { "gesture": "STOP", "expected_intent": "halt", "intent_aliases": ["stop", "pause", "end"], "expected_keywords": ["stop", "halt", "pause", "end"] }
        ]
      },
      {
        "id": 4,
        "gesture_sequence": ["CALL", "YES", "WAIT", "NO", "GOODBYE"],
        "steps": [
          { "gesture": "CALL", "expected_intent": "make_call", "intent_aliases": ["call", "phone", "dial", "ring"], "expected_keywords": ["call", "phone", "dial", "ring"] },
          { "gesture": "YES", "expected_intent": "confirm", "intent_aliases": ["yes", "agree", "ok"], "expected_keywords": ["yes", "confirm", "ok"] },
          { "gesture": "WAIT", "expected_intent": "request_wait", "intent_aliases": ["wait", "hold", "pause", "moment"], "expected_keywords": ["wait", "hold", "moment", "pause"] },
          { "gesture": "NO", "expected_intent": "deny", "intent_aliases": ["no", "reject", "cancel"], "expected_keywords": ["no", "cancel", "reject"] },
          { "gesture": "GOODBYE", "expected_intent": "farewell", "intent_aliases": ["bye", "goodbye", "exit"], "expected_keywords": ["goodbye", "bye", "farewell"] }
        ]
      }
    ]
  }
}
```

### Expected Agent Behavior
For each gesture:
1. Decode the current hand sign from `hand_landmarks` / `gesture_embedding`
2. Map the decoded sign to intent (e.g., `"HELLO"` → `"greeting"`)
3. Generate appropriate natural language response
4. Submit action with intent + confidence + response

### Evaluation Criteria
- Score = (number of correct intents) / (total gestures in sequence)
- Correct = `agent.intent == expected_intent` OR `agent.intent in intent_aliases` (case-insensitive)

### Failure Cases
- Agent outputs `intent: "unknown"` for every gesture → score = 0.0
- Agent loops (same action every step) → loop penalty applied
- Agent ignores the perceptual observation and guesses blindly → likely wrong intents

---

## Task 2: Multi-step Interaction (Medium)

### Objective
The agent must track a sequence of gestures that together form a compound task. The agent must maintain context across steps to complete the task correctly. Gestures build on each other — later gestures only make sense in context of earlier ones.

### Difficulty Rationale
- 8 gestures in sequence (longer episode)
- Context dependency (e.g., `YOUTUBE` only makes sense after `OPEN`)
- Agent must track what "app" was opened to respond to `SEARCH`
- Response must reflect accumulated context

### Input Format (Gesture Sequences)

Scenario 0 example (full definition in `data/tasks.json`):

```json
{
  "id": 0,
  "gesture_sequence": ["OPEN", "YOUTUBE", "SEARCH", "MUSIC", "PLAY", "LOUDER", "STOP", "CLOSE"],
  "steps": [
    {
      "gesture": "OPEN",
      "expected_intent": "open_action",
      "intent_aliases": ["open", "start", "launch"],
      "expected_keywords": ["open", "launch", "start"],
      "context_sets": {"active_app": null}
    },
    {
      "gesture": "YOUTUBE",
      "expected_intent": "select_app:youtube",
      "intent_aliases": ["youtube", "open_youtube", "select_youtube"],
      "expected_keywords": ["youtube", "opening", "launching"],
      "context_sets": {"active_app": "youtube"}
    },
    {
      "gesture": "SEARCH",
      "expected_intent": "search_action",
      "intent_aliases": ["search", "find", "look"],
      "expected_keywords": ["search", "find", "looking", "query"],
      "context_requires": {"active_app": "youtube"}
    },
    {
      "gesture": "MUSIC",
      "expected_intent": "search_query:music",
      "intent_aliases": ["music", "songs", "audio"],
      "expected_keywords": ["music", "songs", "searching"]
    },
    {
      "gesture": "PLAY",
      "expected_intent": "play_action",
      "intent_aliases": ["play", "start", "begin"],
      "expected_keywords": ["play", "playing", "start"]
    },
    {
      "gesture": "LOUDER",
      "expected_intent": "increase_volume",
      "intent_aliases": ["louder", "volume_up", "increase"],
      "expected_keywords": ["volume", "louder", "increasing"]
    },
    {
      "gesture": "STOP",
      "expected_intent": "halt",
      "intent_aliases": ["stop", "pause"],
      "expected_keywords": ["stop", "pause", "stopping"]
    },
    {
      "gesture": "CLOSE",
      "expected_intent": "close_action",
      "intent_aliases": ["close", "exit", "quit"],
      "expected_keywords": ["close", "exit", "closing"]
    }
  ]
}
```

All 5 scenarios defined in `data/tasks.json`. Other scenarios cover: Maps navigation, Calendar scheduling, Messaging, Music player.

### Expected Agent Behavior
1. At each step, read gesture AND full history from observation
2. Infer what task is being built (e.g., "user is navigating YouTube")
3. Map gesture to context-aware intent
4. Produce response that reflects accumulated context

### Evaluation Criteria
- Score = (number of correct intents in sequence) / (total gestures in sequence)
- Sequence bonus: +0.1 if all 8 intents correct (perfect sequence)
- Score capped at 1.0

### Failure Cases
- Agent treats each gesture independently (ignores history) → partial score only
- Agent maps `YOUTUBE` to `"greeting"` (wrong context) → 0 for that step
- Agent loops after `OPEN` without tracking app context

---

## Task 3: Intent Inference (Hard)

### Objective
The agent must infer the correct compound intent from a sequence of ambiguous or incomplete gestures. No single gesture has an unambiguous meaning — the agent must reason across the full sequence to determine what the user wants.

### Difficulty Rationale
- Gestures are individually ambiguous (e.g., `NEAR` could mean location or proximity)
- Intent is not any single gesture's label — it is inferred from the combination
- Agent must produce a single synthesized intent for the full sequence
- Partial matches score lower; exact compound intent scores higher
- Designed to challenge frontier LLMs

### Input Format (Gesture Sequences)

```json
{
  "task3": {
    "scenarios": [
      {
        "id": 0,
        "gesture_sequence": ["FOOD", "NEAR", "NOW"],
        "compound_intent": "find_nearby_restaurants",
        "intent_aliases": ["find_restaurant", "nearby_food", "restaurant_near_me", "food_nearby"],
        "expected_keywords": ["restaurant", "nearby", "food", "find", "location"],
        "explanation": "FOOD + NEAR + NOW = find nearby food immediately = restaurant search"
      },
      {
        "id": 1,
        "gesture_sequence": ["CALL", "HELP", "NOW"],
        "compound_intent": "emergency_call",
        "intent_aliases": ["call_help", "urgent_call", "sos", "call_emergency"],
        "expected_keywords": ["emergency", "call", "help", "urgent", "now"],
        "explanation": "CALL + HELP + NOW = urgent call for help = emergency"
      },
      {
        "id": 2,
        "gesture_sequence": ["MUSIC", "QUIET", "SLEEP"],
        "compound_intent": "play_sleep_music",
        "intent_aliases": ["sleep_music", "quiet_music", "relaxing_music", "calm_music"],
        "expected_keywords": ["sleep", "quiet", "calm", "music", "relax"],
        "explanation": "MUSIC + QUIET + SLEEP = play quiet music for sleeping"
      },
      {
        "id": 3,
        "gesture_sequence": ["OPEN", "MAP", "HOME"],
        "compound_intent": "navigate_home",
        "intent_aliases": ["go_home", "directions_home", "navigate_to_home", "home_navigation"],
        "expected_keywords": ["home", "navigate", "directions", "map", "route"],
        "explanation": "OPEN + MAP + HOME = open maps and navigate home"
      },
      {
        "id": 4,
        "gesture_sequence": ["WATER", "COLD", "NOW"],
        "compound_intent": "request_cold_water_immediately",
        "intent_aliases": ["cold_water", "water_now", "cold_drink_now", "urgent_water"],
        "expected_keywords": ["cold", "water", "now", "immediately", "drink"],
        "explanation": "WATER + COLD + NOW = urgently need cold water"
      }
    ]
  }
}
```

### Special Evaluation for Task 3

Task 3 evaluates the **final synthesized intent** submitted on the **last step** of the sequence, not each step individually.

**Scoring:**

```
IF final_action.intent == compound_intent (exact, case-insensitive):
    score = 1.0

ELIF final_action.intent in intent_aliases:
    score = 0.7

ELIF any keyword from expected_keywords in final_action.intent:
    score = 0.4

ELIF any keyword from expected_keywords in final_action.response:
    score = 0.2

ELSE:
    score = 0.0
```

Per-step rewards still apply (see environment.md) to provide training signal.

### Expected Agent Behavior
1. Step 1–N-1: Observe each gesture, accumulate context
2. On final gesture: synthesize all observed gestures into one compound intent
3. Submit action with compound intent and response explaining the full inferred task

### Failure Cases
- Agent treats final gesture independently (e.g., `NOW` → `"current_time"`)
- Agent outputs only partial intent (e.g., `"food"` instead of `"find_nearby_restaurants"`)
- Agent fails to use history to reason about compound meaning
