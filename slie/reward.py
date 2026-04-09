from __future__ import annotations

from slie.models import SLIEAction


def _strict_unit(value: float) -> float:
    return max(0.01, min(0.99, value))


def compute_reward(
    action: SLIEAction,
    step_spec: dict,
    last_action: SLIEAction | None,
    is_final_step: bool,
) -> tuple[float, dict]:
    expected_intent = str(step_spec.get("expected_intent", "")).strip().lower()
    aliases = [a.strip().lower() for a in step_spec.get("intent_aliases", [])]
    keywords = [k.strip().lower() for k in step_spec.get("expected_keywords", [])]

    agent_intent = action.intent.strip().lower()
    agent_response = action.response.strip().lower()

    if agent_intent == expected_intent:
        intent_reward = 0.4
    elif agent_intent in aliases:
        intent_reward = 0.2
    else:
        intent_reward = 0.0

    matched_keywords = [kw for kw in keywords if kw and kw in agent_response]
    response_reward = 0.3 if matched_keywords else 0.0

    completion_bonus = 0.3 if is_final_step and intent_reward > 0 else 0.0

    loop_penalty = 0.0
    if (
        last_action
        and action.intent == last_action.intent
        and action.response == last_action.response
    ):
        loop_penalty = -0.5

    invalid_penalty = 0.0
    if not action.intent.strip() or not action.response.strip():
        invalid_penalty = -0.3

    reward = (
        intent_reward
        + response_reward
        + completion_bonus
        + loop_penalty
        + invalid_penalty
    )
    reward = max(-1.0, min(1.0, reward))
    reward = _strict_unit(reward)

    debug = {
        "intent_reward": intent_reward,
        "response_reward": response_reward,
        "completion_bonus": completion_bonus,
        "loop_penalty": loop_penalty,
        "invalid_penalty": invalid_penalty,
        "matched_keywords": matched_keywords,
    }
    return round(reward, 4), debug
