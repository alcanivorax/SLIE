from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slie.env import SLIEEnvironment
from slie.models import SLIEAction

ROOT = Path(__file__).resolve().parent.parent


def check_data() -> None:
    gestures = json.loads((ROOT / "data" / "gestures.json").read_text(encoding="utf-8"))
    tasks = json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))

    assert len(gestures) == 24, "gestures.json must contain 24 gestures"
    for label, item in gestures.items():
        assert len(item["frame_features"]) == 64, f"{label} must have 64 frame features"

    for task_id in ("task1", "task2", "task3"):
        assert task_id in tasks, f"missing {task_id}"
        assert len(tasks[task_id]["scenarios"]) == 5, f"{task_id} must have 5 scenarios"

    for scenario in tasks["task1"]["scenarios"]:
        assert len(scenario["gesture_sequence"]) == 5, "task1 scenarios must have 5 gestures"

    for scenario in tasks["task2"]["scenarios"]:
        assert len(scenario["gesture_sequence"]) == 8, "task2 scenarios must have 8 gestures"

    for scenario in tasks["task3"]["scenarios"]:
        assert len(scenario["gesture_sequence"]) == 3, "task3 scenarios must have 3 gestures"


def check_env_behavior() -> None:
    env = SLIEEnvironment()

    try:
        env.step(SLIEAction(intent="x", confidence=0.1, response="x"))
        raise AssertionError("Expected step() before reset to fail")
    except RuntimeError as exc:
        assert "No active episode" in str(exc)

    reset = env.reset("task1", 0)
    assert len(reset.observation.gesture_embedding) == 64
    assert len(reset.observation.hand_landmarks) == 6

    # perfect task1 episode
    actions = [
        SLIEAction(intent="greeting", confidence=0.9, response="hello"),
        SLIEAction(intent="confirm", confidence=0.9, response="yes"),
        SLIEAction(intent="halt", confidence=0.9, response="stop"),
        SLIEAction(intent="request_help", confidence=0.9, response="help"),
        SLIEAction(intent="farewell", confidence=0.9, response="goodbye"),
    ]

    out = None
    for action in actions:
        out = env.step(action)

    assert out is not None and out.done is True
    assert out.observation.hand_landmarks == []
    assert out.info.final_score == 0.99

    try:
        env.step(SLIEAction(intent="x", confidence=0.1, response="x"))
        raise AssertionError("Expected step() after done to fail")
    except RuntimeError as exc:
        assert "Episode is done" in str(exc)


def main() -> None:
    check_data()
    check_env_behavior()
    print("SPEC_CONFORMANCE_OK")


if __name__ == "__main__":
    main()
