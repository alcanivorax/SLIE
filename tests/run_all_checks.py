from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slie.data_loader import get_scenario, load_gestures, load_tasks
from slie.env import SLIEEnvironment
from slie.gesture_layer import GestureInputLayer
from slie.graders import task1_grader, task2_grader, task3_grader
from slie.models import EpisodeHistory, SLIEAction
from slie.reward import compute_reward
from slie.state import EnvironmentState


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_models() -> None:
    action = SLIEAction(intent="greeting", confidence=1.5, response="hi")
    check(action.confidence == 1.0, "confidence should clamp to 1.0")

    state = EnvironmentState()
    check(state.max_steps == 10, "default max_steps should be 10")


def check_data_and_layer() -> None:
    gestures = load_gestures()
    tasks = load_tasks()

    check(len(gestures) == 24, "must load 24 gestures")
    check(all(len(v["frame_features"]) == 64 for v in gestures.values()), "all gestures must have 64 features")

    s1 = get_scenario("task1", 0, tasks)
    s2 = get_scenario("task2", 0, tasks)
    s3 = get_scenario("task3", 0, tasks)
    check(len(s1["gesture_sequence"]) == 5, "task1 scenario length must be 5")
    check(len(s2["gesture_sequence"]) == 8, "task2 scenario length must be 8")
    check(len(s3["gesture_sequence"]) == 3, "task3 scenario length must be 3")

    layer = GestureInputLayer(gestures, s1, episode_seed=0, task_id="task1")
    obs = layer.get_observation(0, 0, [], "task1")
    check(len(obs.gesture_embedding) == 64, "task1 observation must include 64-dim embedding")
    check(len(obs.hand_landmarks) == 6, "task1 observation must include 6 landmark frames")
    done_obs = layer.get_observation(99, 5, ["a"] * 8, "task1")
    check(done_obs.hand_landmarks == [], "done observation landmarks should be empty")
    check(done_obs.gesture_embedding == [0.0] * 64, "done observation embedding should be zeros")
    check(len(done_obs.context.history) == 5, "history should keep only last 5 entries")


def check_reward() -> None:
    step_spec = {
        "expected_intent": "greeting",
        "intent_aliases": ["hello"],
        "expected_keywords": ["hello", "help"],
    }

    r, _ = compute_reward(SLIEAction(intent="greeting", confidence=0.9, response="hello there"), step_spec, None, False)
    check(r == 0.7, "perfect non-final reward should be 0.7")

    r, _ = compute_reward(SLIEAction(intent="greeting", confidence=0.9, response="hello there"), step_spec, None, True)
    check(r == 1.0, "perfect final reward should be 1.0")

    r, _ = compute_reward(SLIEAction(intent="hello", confidence=0.9, response="x"), step_spec, None, False)
    check(r == 0.2, "alias-only reward should be 0.2")

    last = SLIEAction(intent="greeting", confidence=0.9, response="hello")
    cur = SLIEAction(intent="greeting", confidence=0.9, response="hello")
    r, _ = compute_reward(cur, step_spec, last, False)
    check(r == 0.2, "loop penalty should apply (0.4+0.3-0.5=0.2)")

    r, _ = compute_reward(SLIEAction(intent="", confidence=0.9, response=""), step_spec, None, False)
    check(r == -0.3, "empty fields should yield -0.3")


def check_graders() -> None:
    h1 = EpisodeHistory(
        task_id="task1",
        scenario_id=0,
        gesture_sequence=["A", "B", "C", "D", "E"],
        interaction_history=[
            {"agent_intent": "x", "expected_intent": "x", "intent_aliases": []},
            {"agent_intent": "x", "expected_intent": "x", "intent_aliases": []},
            {"agent_intent": "x", "expected_intent": "x", "intent_aliases": []},
            {"agent_intent": "x", "expected_intent": "x", "intent_aliases": []},
            {"agent_intent": "x", "expected_intent": "x", "intent_aliases": []},
        ],
        steps_taken=5,
        max_steps=10,
    )
    check(task1_grader(h1) == 1.0, "task1 all-correct should be 1.0")

    h1_partial = h1.model_copy(update={"interaction_history": h1.interaction_history[:3] + [{"agent_intent": "bad", "expected_intent": "x", "intent_aliases": []}, {"agent_intent": "bad", "expected_intent": "x", "intent_aliases": []}]})
    check(task1_grader(h1_partial) == 0.6, "task1 3/5 should be 0.6")

    h2 = EpisodeHistory(
        task_id="task2",
        scenario_id=0,
        gesture_sequence=[str(i) for i in range(8)],
        interaction_history=[
            {"agent_intent": "ok", "expected_intent": "ok", "intent_aliases": []}
            for _ in range(8)
        ],
        steps_taken=8,
        max_steps=10,
    )
    check(task2_grader(h2) == 1.0, "task2 all-correct + processed should cap to 1.0")

    h3 = EpisodeHistory(
        task_id="task3",
        scenario_id=0,
        gesture_sequence=["FOOD", "NEAR", "NOW"],
        interaction_history=[
            {"agent_intent": "request_food", "expected_intent": "request_food", "intent_aliases": ["food"], "agent_response": "food"},
            {"agent_intent": "set_proximity_filter", "expected_intent": "set_proximity_filter", "intent_aliases": ["near"], "agent_response": "near"},
            {"agent_intent": "find_nearby_restaurants", "expected_intent": "find_nearby_restaurants", "intent_aliases": ["find_restaurant"], "agent_response": "finding nearby restaurant"},
        ],
        steps_taken=3,
        max_steps=10,
    )
    check(task3_grader(h3) == 1.0, "task3 exact compound + full intermediate should be 1.0")


def check_env() -> None:
    env = SLIEEnvironment()
    try:
        env.step(SLIEAction(intent="x", confidence=0.1, response="x"))
        raise AssertionError("step before reset should fail")
    except RuntimeError:
        pass

    reset = env.reset("task1", 0)
    check(reset.task_id == "task1", "reset should set task1")

    actions = [
        SLIEAction(intent="greeting", confidence=0.9, response="hello"),
        SLIEAction(intent="confirm", confidence=0.9, response="yes"),
        SLIEAction(intent="halt", confidence=0.9, response="stop"),
        SLIEAction(intent="request_help", confidence=0.9, response="help"),
        SLIEAction(intent="farewell", confidence=0.9, response="goodbye"),
    ]

    final = None
    for action in actions:
        final = env.step(action)

    check(final is not None and final.done, "episode should be done")
    check(final.info.final_score == 1.0, "perfect task1 should score 1.0")


def main() -> None:
    check_models()
    check_data_and_layer()
    check_reward()
    check_graders()
    check_env()
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
