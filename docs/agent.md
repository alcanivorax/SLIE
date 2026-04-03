# agent.md — AI Agent Specification

---

## 1. Agent Role

The agent is an **external process** (`inference.py`) that:
- Communicates with the SLIE environment via HTTP
- Uses an LLM (via OpenAI-compatible API) to reason about gestures
- Produces structured actions
- Logs stdout in the required format

The agent has NO direct access to the environment's internal state. It only sees what the environment exposes via observations.

---

## 2. Input: What the Agent Receives

At each step, the agent receives a `StepResponse` (or `ResetResponse` on first call):

```json
{
  "observation": {
    "gesture_embedding": [0.12, 0.45, 0.33, ...],
    "hand_landmarks": [{"left_hand": [], "right_hand": []}],
    "context": {
      "current_task": "task2",
      "step_count": 2,
      "history": [
        "Step 1: prior_intent=open_action",
        "Step 2: prior_intent=select_app:youtube"
      ]
    }
  },
  "reward": 0.7,
  "done": false,
  "info": {
    "step_count": 2,
    "final_score": null,
    "error": null
  }
}
```

**Key fields the agent MUST use:**
- `observation.hand_landmarks` — the primary perceptual input
- `observation.gesture_embedding` — a compact summary for decoding or prompting
- `observation.context.history` — for multi-step memory (Task 2 and 3)
- `observation.context.current_task` — to know which task strategy to apply
- `observation.context.step_count` — to know position in episode
- `reward` — feedback from previous action (for learning/adaptation)
- `done` — must stop if True

The agent must not rely on any environment-provided gesture label because the live observation contract does not expose one.

---

## 3. Output Format (STRICT Schema)

The agent must submit actions in this exact JSON structure:

```json
{
  "intent": "string (non-empty, max 100 chars)",
  "confidence": 0.0,
  "response": "string (non-empty, max 500 chars)"
}
```

### Rules
- `intent`: Snake_case or colon-separated compound (e.g., `"greeting"`, `"open_app:youtube"`, `"find_nearby_restaurants"`)
- `confidence`: Float in [0.0, 1.0]. Agent should reflect genuine confidence, not always output 1.0.
- `response`: Natural language describing the action taken. Must be non-empty.

### Fallback Action (when LLM fails or returns unparseable output)

```json
{
  "intent": "unknown",
  "confidence": 0.0,
  "response": "I could not understand the gesture."
}
```

---

## 4. Decision-Making Expectations

### Task 1 Strategy (Command Recognition)
- Decode the current hand sign from `hand_landmarks` / `gesture_embedding`
- Map the decoded sign to a known intent from the pre-defined vocabulary
- No history needed
- High confidence expected for known gestures

### Task 2 Strategy (Multi-step Interaction)
- Decode the current hand sign and combine it with full `history`
- Track: what app is open, what action sequence is in progress
- Intent must reflect context (e.g., `SEARCH` after `YOUTUBE` → `"search_action"` in YouTube context)
- Response must reference accumulated context

### Task 3 Strategy (Intent Inference)
- Steps 1 to N-1: observe and accumulate gestures
- Final step: synthesize ALL observed gestures into compound intent
- Intent must be the compound semantic meaning, not the last gesture's individual meaning
- Agent should explicitly reason: "Gestures so far: X, Y, Z. Combined meaning: ..."

---

## 5. Handling Ambiguity

**Rule:** When a gesture is ambiguous, the agent must use history to disambiguate.

```
IF history is empty:
    Use most common meaning of gesture
ELIF history contains relevant context:
    Use context-aware interpretation
ELIF gesture is completely unknown:
    Output intent="unknown", confidence=0.0
```

**Never** output empty intent or response strings. Always use the fallback action if uncertain.

---

## 6. Multi-step Memory

The agent maintains its own internal history list (in addition to what the environment provides in `context.history`).

```python
agent_history = []  # List of {step, gesture, intent, response, reward}
```

After each step, append:
```python
agent_history.append({
    "step": step_number,
    "gesture": decoded_gesture,
    "intent": action.intent,
    "response": action.response,
    "reward": reward
})
```

Include the last 5 entries of `agent_history` in every LLM prompt.

---

## 7. Handling Corrections

If `reward` from previous step was low (< 0.2), the agent should:
1. Note that its previous interpretation was likely wrong
2. Try a different intent on the next step
3. NOT repeat the same action (loop penalty applies)

```
IF previous_reward < 0.2 AND step > 1:
    Add to prompt: "My last interpretation was likely wrong. Try a different intent."
```

---

## 8. Pseudocode: Agent Loop

```python
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_url = "http://localhost:8000"

    rewards = []
    agent_history = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env="slie", model=MODEL_NAME)

    # Reset environment
    reset_response = POST(f"{env_url}/reset", body={"task_id": TASK_NAME, "episode_seed": SEED})
    obs = reset_response["observation"]
    last_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if obs is None or reset_response.get("done", False):
            break

        # Build prompt
        prompt = build_prompt(obs, last_reward, agent_history, step)

        # Call LLM
        llm_output = call_llm(client, prompt)

        # Parse LLM output
        action = parse_action(llm_output)  # Returns SLIEAction or fallback

        # Step environment
        step_response = POST(f"{env_url}/step", body=action)

        reward = step_response["reward"]
        done = step_response["done"]
        obs = step_response["observation"]
        info = step_response["info"]
        error = info.get("error")

        rewards.append(reward)
        steps_taken = step
        last_reward = reward

        agent_history.append({
            "step": step,
            "gesture": action_sent_gesture,
            "intent": action["intent"],
            "response": action["response"],
            "reward": reward
        })

        log_step(step=step, action=action["intent"], reward=reward, done=done, error=error)

        if done:
            score = step_response["info"].get("final_score", 0.0)
            break

    success = score >= SUCCESS_THRESHOLD  # 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
```

---

## 9. Prompting Strategy

### System Prompt

```
You are an AI assistant helping a deaf user interact with a digital system via sign language.

The user communicates through structured sign language gestures. Your job is to:
1. Interpret the meaning (intent) of the gesture shown
2. Consider the conversation history for context
3. Respond with an appropriate action

You MUST respond with valid JSON only. No other text. Format:
{
  "intent": "<your interpretation>",
  "confidence": <float 0.0-1.0>,
  "response": "<natural language action>"
}

Known gesture vocabulary: HELLO, YES, NO, STOP, HELP, GOODBYE, OPEN, CLOSE, FOOD, WATER,
MUSIC, PLAY, LOUDER, CALL, WAIT, YOUTUBE, SEARCH, NEAR, NOW, MAP, HOME, QUIET, SLEEP, COLD
```

### User Prompt Template

```
Current Task: {current_task}
Step: {step_count}
Most Likely Hand Sign: {decoded_gesture}
Decoder Confidence: {decode_confidence}
Hand Landmark Summary: {landmark_summary}
Last Reward: {last_reward:.2f}

Conversation History:
{history_block}

{"Note: My last interpretation was likely wrong. Try a different intent for this gesture." if last_reward < 0.2 and step > 1 else ""}

{"IMPORTANT: This is the FINAL gesture in the sequence. You must synthesize ALL previous gestures into a single compound intent." if is_final_step and task == "task3" else ""}

Respond with JSON only.
```

---

## 10. Logging Format (MANDATORY)

All stdout output must follow this exact format:

```
[START] task=<task_id> env=slie model=<model_name>
[STEP] step=<n> action=<intent_string> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

### Rules
- `action` field in `[STEP]` = `action.intent` string (not full JSON)
- `reward` = float to 2 decimal places
- `done` = lowercase `true` or `false`
- `error` = raw error string from `info.error`, or `null`
- `score` in `[END]` = float to 3 decimal places
- `rewards` = comma-separated list of all step rewards to 2 decimal places
- One `[START]` line per episode
- One `[STEP]` line per step (emitted immediately after step() returns)
- One `[END]` line per episode (emitted in `finally` block — always runs)

### Example

```
[START] task=task1 env=slie model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=greeting reward=0.70 done=false error=null
[STEP] step=2 action=confirm reward=0.70 done=false error=null
[STEP] step=3 action=halt reward=0.70 done=false error=null
[STEP] step=4 action=request_help reward=0.70 done=false error=null
[STEP] step=5 action=farewell reward=1.00 done=true error=null
[END] success=true steps=5 score=0.833 rewards=0.70,0.70,0.70,0.70,1.00
```
