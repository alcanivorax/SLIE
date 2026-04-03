from __future__ import annotations

from slie.data_loader import get_scenario, load_tasks
from slie.models import EpisodeHistory
from slie.state import EnvironmentState


def _is_intent_correct(entry: dict) -> bool:
    agent_intent = str(entry.get("agent_intent", "")).lower().strip()
    expected = str(entry.get("expected_intent", "")).lower().strip()
    aliases = [a.lower().strip() for a in entry.get("intent_aliases", [])]
    return agent_intent == expected or agent_intent in aliases


def task1_grader(history: EpisodeHistory) -> float:
    total = len(history.interaction_history)
    if total == 0:
        return 0.0
    correct = sum(
        1 for entry in history.interaction_history if _is_intent_correct(entry)
    )
    return round(correct / total, 4)


def task2_grader(history: EpisodeHistory) -> float:
    total = len(history.interaction_history)
    if total == 0:
        return 0.0

    correct = sum(
        1 for entry in history.interaction_history if _is_intent_correct(entry)
    )
    base_score = correct / total

    all_processed = len(history.interaction_history) == len(history.gesture_sequence)
    all_correct = correct == total
    sequence_bonus = 0.1 if all_correct and all_processed else 0.0

    return round(min(1.0, base_score + sequence_bonus), 4)


def task3_grader(history: EpisodeHistory) -> float:
    if not history.interaction_history:
        return 0.0

    tasks = load_tasks()
    scenario = get_scenario(history.task_id, history.scenario_id, tasks)
    compound_intent = scenario["compound_intent"].lower().strip()
    intent_aliases = [a.lower().strip() for a in scenario["intent_aliases"]]
    expected_keywords = [k.lower().strip() for k in scenario["expected_keywords"]]

    last_entry = history.interaction_history[-1]
    agent_intent = str(last_entry.get("agent_intent", "")).lower().strip()
    agent_response = str(last_entry.get("agent_response", "")).lower()

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

    intermediate_steps = history.interaction_history[:-1]
    if intermediate_steps:
        step_correct = sum(
            1 for entry in intermediate_steps if _is_intent_correct(entry)
        )
        step_ratio = step_correct / len(intermediate_steps)
        step_bonus = step_ratio * 0.2
    else:
        step_bonus = 0.0

    return round(min(1.0, compound_score * 0.8 + step_bonus), 4)


def compute_final_score(state: EnvironmentState) -> float:
    history = EpisodeHistory(
        task_id=state.task_id or "",
        scenario_id=state.scenario_id or 0,
        gesture_sequence=state.gesture_sequence,
        interaction_history=state.interaction_history,
        steps_taken=state.step_count,
        max_steps=state.max_steps,
    )

    if state.task_id == "task1":
        return task1_grader(history)
    if state.task_id == "task2":
        return task2_grader(history)
    if state.task_id == "task3":
        return task3_grader(history)
    raise ValueError(f"Unknown task_id: {state.task_id}")
