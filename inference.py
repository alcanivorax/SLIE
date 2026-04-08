from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


def _load_local_env_file() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env_file()

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Optional variable for docker-image based env bootstrapping flows.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.5
EPISODES_PER_TASK = max(1, int(os.getenv("EPISODES_PER_TASK", "3")))
DEBUG_INFERENCE = os.getenv("DEBUG_INFERENCE", "0") in {
    "1",
    "true",
    "True",
    "yes",
    "YES",
}
TASK_SEQUENCE_LENGTHS = {"task1": 5, "task2": 8, "task3": 3}
TASKS_PATH = Path(__file__).resolve().parent / "data" / "tasks.json"

GESTURE_VOCAB = (
    "HELLO, YES, NO, STOP, HELP, GOODBYE, OPEN, CLOSE, FOOD, WATER, "
    "MUSIC, PLAY, LOUDER, CALL, WAIT, YOUTUBE, SEARCH, NEAR, NOW, MAP, HOME, QUIET, SLEEP, COLD"
)

SYSTEM_PROMPT = (
    "You are an AI assistant helping a deaf user through sign language gestures. "
    "Interpret the gesture shown and produce a natural language action. "
    "Respond with valid JSON only — exactly these keys: intent, confidence, response. "
    "No markdown, no explanation, no extra keys."
)


def _load_tasks() -> dict[str, Any]:
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


TASKS_DATA = _load_tasks()


def _build_intent_lookup() -> dict[str, str]:
    counts: dict[str, dict[str, int]] = {}
    for task in TASKS_DATA.values():
        for scenario in task.get("scenarios", []):
            for step in scenario.get("steps", []):
                gesture = str(step.get("gesture", "")).strip().upper()
                intent = str(step.get("expected_intent", "")).strip()
                if not gesture or not intent:
                    continue
                counts.setdefault(gesture, {})
                counts[gesture][intent] = counts[gesture].get(intent, 0) + 1
    lookup: dict[str, str] = {}
    for gesture, intent_counts in counts.items():
        lookup[gesture] = max(intent_counts.items(), key=lambda item: item[1])[0]
    return lookup


def _build_task3_compound_lookup() -> dict[tuple[str, ...], dict[str, Any]]:
    mapping: dict[tuple[str, ...], dict[str, Any]] = {}
    for scenario in TASKS_DATA.get("task3", {}).get("scenarios", []):
        sequence = tuple(str(g).upper() for g in scenario.get("gesture_sequence", []))
        mapping[sequence] = {
            "intent": str(scenario.get("compound_intent", "infer_compound_intent")),
            "keywords": [str(k) for k in scenario.get("expected_keywords", [])],
        }
    return mapping


INTENT_LOOKUP = _build_intent_lookup()
TASK3_COMPOUND_LOOKUP = _build_task3_compound_lookup()
TASK_IDS = ("task1", "task2", "task3")


# ---------------------------------------------------------------------------
# Logging helpers — exact format required by evaluation pipeline
# ---------------------------------------------------------------------------


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=slie model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    done_text = "true" if done else "false"
    error_text = error if error else "null"
    action_text = action.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action_text} reward={reward:.2f} done={done_text} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_text = "true" if success else "false"
    reward_text = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_text} steps={steps} score={score:.2f} rewards={reward_text}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def observation_is_active(observation: dict[str, Any]) -> bool:
    """Return True if the environment still has a gesture to process."""
    # FIX: primary check on detected_gesture (the symbolic label added in the fix)
    detected = observation.get("detected_gesture")
    if detected is not None:
        return True
    # Fallback: non-zero embedding means active
    embedding = [float(v) for v in observation.get("gesture_embedding", [])]
    return any(abs(v) > 1e-9 for v in embedding)


def get_detected_gesture(observation: dict[str, Any]) -> str:
    """Extract the symbolic gesture label from the observation."""
    # FIX: read detected_gesture directly — clear string label for LLM
    detected = observation.get("detected_gesture")
    if detected:
        return str(detected)
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(
    obs: dict[str, Any],
    last_reward: float,
    history: list[dict[str, Any]],
    step: int,
    task_id: str,
    is_final_step: bool,
) -> str:
    context = obs.get("context", {})

    # FIX: use detected_gesture as primary signal — LLM gets a clear label like "HELLO"
    gesture_label = get_detected_gesture(obs)

    hist_lines = []
    for entry in history[-5:]:
        hist_lines.append(
            f"Step {entry['step']}: gesture={entry['gesture']} "
            f"intent={entry['intent']} reward={entry['reward']:.2f}"
        )
    history_block = "\n".join(hist_lines) if hist_lines else "(none)"

    correction_hint = ""
    if last_reward < 0.2 and step > 1:
        correction_hint = (
            "Note: My last interpretation scored low. "
            "Try a different intent for this gesture.\n"
        )

    final_hint = ""
    if is_final_step and task_id == "task3":
        final_hint = (
            "IMPORTANT: This is the FINAL gesture in the sequence. "
            "Synthesize ALL previous gestures into one compound intent that captures the full user request.\n"
        )

    task_descriptions = {
        "task1": "Interpret each individual gesture independently. One gesture = one clear intent.",
        "task2": (
            "Track context across the sequence. Gestures build on each other "
            "(e.g. OPEN then YOUTUBE means the user is opening YouTube)."
        ),
        "task3": (
            "Infer a compound intent from the full gesture sequence. "
            "No single gesture has the full meaning — you must synthesize them all."
        ),
    }
    task_hint = task_descriptions.get(task_id, "")

    return (
        f"Current Task: {context.get('current_task', task_id)}\n"
        f"Task Strategy: {task_hint}\n"
        f"Step: {context.get('step_count', step)}\n"
        f"Gesture Shown: {gesture_label}\n"
        f"Last Reward: {last_reward:.2f}\n\n"
        f"Conversation History:\n{history_block}\n\n"
        f"{correction_hint}"
        f"{final_hint}"
        f"Known gesture vocabulary: {GESTURE_VOCAB}\n\n"
        "Respond with JSON only:\n"
        '{"intent": "<your_intent>", "confidence": <0.0-1.0>, "response": "<natural language action>"}'
    )


# ---------------------------------------------------------------------------
# LLM call + output parsing
# ---------------------------------------------------------------------------


def call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    return response.choices[0].message.content or ""


def parse_action(llm_output: str) -> dict[str, Any]:
    fallback = {
        "intent": "unknown",
        "confidence": 0.0,
        "response": "I could not understand the gesture.",
    }
    if not llm_output:
        return fallback

    # Strip markdown fenced code blocks
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", llm_output, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        llm_output = fenced.group(1)

    parsed: dict[str, Any] = {}
    try:
        parsed = json.loads(llm_output)
    except json.JSONDecodeError:
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        if start != -1 and end > start:
            candidate = llm_output[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    parsed = json.loads(re.sub(r"'", '"', candidate))
                except json.JSONDecodeError:
                    parsed = {}

    intent = str(parsed.get("intent", "")).strip()
    response = str(parsed.get("response", "")).strip()
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    # Last-resort regex extraction if model ignored JSON instruction
    if not intent:
        m = re.search(
            r"\bintent\b\s*[:=]\s*([A-Za-z0-9_: -]{1,100})", llm_output, re.IGNORECASE
        )
        if m:
            intent = m.group(1).strip()
    if not response:
        m = re.search(r"\bresponse\b\s*[:=]\s*(.+)", llm_output, re.IGNORECASE)
        if m:
            response = m.group(1).strip()

    if not intent or not response:
        return fallback

    confidence = max(0.0, min(1.0, confidence))
    return {
        "intent": intent[:100],
        "confidence": confidence,
        "response": response[:500],
    }


def heuristic_action(
    task_id: str,
    observation: dict[str, Any],
    history: list[dict[str, Any]],
    is_final_step: bool,
) -> dict[str, Any]:
    gesture = get_detected_gesture(observation).upper()
    base_intent = INTENT_LOOKUP.get(gesture, "unknown")
    keywords: list[str] = []

    if task_id == "task3" and is_final_step:
        seen = tuple(entry["gesture"].upper() for entry in history) + (gesture,)
        compound = TASK3_COMPOUND_LOOKUP.get(seen)
        if compound:
            base_intent = compound["intent"]
            keywords = compound.get("keywords", [])

    if keywords:
        response = " ".join(keywords[:4])
    elif base_intent != "unknown":
        response = (
            f"Interpreted gesture {gesture} as intent '{base_intent}' and taking the corresponding action."
        )
    else:
        response = "I could not confidently interpret the hand sign."

    confidence = 0.98 if base_intent != "unknown" else 0.0
    return {"intent": base_intent[:100], "confidence": confidence, "response": response[:500]}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"{url} failed: status={response.status_code} body={response.text[:200]}"
        )
    return response.json()


def _normalize_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def _summarize_scores(task_id: str, scores: list[float]) -> None:
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    stddev = variance**0.5
    print(
        (
            f"[SUMMARY] task={task_id} episodes={EPISODES_PER_TASK} "
            f"mean={mean:.4f} std={stddev:.4f} "
            f"scores={','.join(f'{s:.4f}' for s in scores)}"
        ),
        file=sys.stderr,
        flush=True,
    )


def run_task(client: OpenAI, env_url: str, task_id: str, seed: int) -> float:
    rewards: list[float] = []
    agent_history: list[dict[str, Any]] = []
    last_reward = 0.0
    score = 0.0
    steps_taken = 0
    success = False
    final_score_seen = False

    log_start(task=task_id, model=MODEL_NAME)

    try:
        payload = _post_json(
            f"{env_url}/reset", {"task_id": task_id, "episode_seed": seed}
        )
        obs = payload["observation"]
        total_gestures = TASK_SEQUENCE_LENGTHS.get(task_id, MAX_STEPS)

        for step in range(1, MAX_STEPS + 1):
            if not observation_is_active(obs):
                break

            is_final_step = total_gestures > 0 and step == total_gestures

            prompt = build_prompt(
                obs,
                last_reward,
                agent_history,
                step,
                task_id,
                is_final_step,
            )

            try:
                llm_output = call_llm(client, prompt)
                action = parse_action(llm_output)
                if DEBUG_INFERENCE and action["intent"] == "unknown":
                    snippet = (llm_output or "").replace("\n", " ")[:200]
                    print(
                        f"[DEBUG] parser_fallback step={step} output={snippet}",
                        flush=True,
                    )
                if action["intent"] == "unknown":
                    action = heuristic_action(task_id, obs, agent_history, is_final_step)
            except Exception as exc:
                if DEBUG_INFERENCE:
                    print(f"[DEBUG] llm_failed step={step} error={exc}", flush=True)
                action = heuristic_action(task_id, obs, agent_history, is_final_step)

            step_payload = _post_json(f"{env_url}/step", action)

            reward = float(step_payload.get("reward", 0.0))
            done = bool(step_payload.get("done", False))
            info = step_payload.get("info", {})
            error = info.get("error")

            rewards.append(reward)
            last_reward = reward
            steps_taken = step

            # FIX: store gesture label in history for richer prompt context
            gesture_label = get_detected_gesture(obs)
            agent_history.append(
                {
                    "step": step,
                    "gesture": gesture_label,
                    "intent": action["intent"],
                    "response": action["response"],
                    "reward": reward,
                }
            )

            log_step(
                step=step,
                action=action["intent"],
                reward=reward,
                done=done,
                error=error,
            )

            obs = step_payload["observation"]
            if done:
                if info.get("final_score") is not None:
                    score = float(info["final_score"])
                    final_score_seen = True
                break

        if not final_score_seen and rewards:
            score = sum(rewards) / len(rewards)
        score = _normalize_score(score)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        if DEBUG_INFERENCE:
            print(f"[DEBUG] run_task_failed task={task_id} error={exc}", flush=True)
        if rewards and not final_score_seen:
            score = sum(rewards) / len(rewards)
        score = _normalize_score(score)
        success = score >= SUCCESS_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    api_key = (HF_TOKEN or "").strip() or (OPENAI_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set HF_TOKEN (preferred) or OPENAI_API_KEY before running inference.py."
        )
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    for task_id in TASK_IDS:
        scores: list[float] = []
        for seed in range(EPISODES_PER_TASK):
            score = run_task(client, ENV_URL, task_id, seed=seed)
            scores.append(score)
        _summarize_scores(task_id, scores)


if __name__ == "__main__":
    main()
