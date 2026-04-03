from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slie.env import SLIEEnvironment
from slie.models import SLIEAction


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run() -> None:
    env = SLIEEnvironment()

    state = env.get_state()
    assert_true(state.task_id is None, "task_id should be None before reset")

    try:
        env.step(SLIEAction(intent="x", confidence=0.1, response="x"))
        raise AssertionError("step before reset should fail")
    except RuntimeError as exc:
        assert_true("No active episode" in str(exc), "unexpected step-before-reset error")

    reset = env.reset("task1", 0)
    assert_true(len(reset.observation.gesture_embedding) == 64, "expected 64-dim embedding on reset")
    assert_true(len(reset.observation.hand_landmarks) == 6, "expected landmark frames on reset")

    intents = ["greeting", "confirm", "halt", "request_help", "farewell"]
    final = None
    for intent in intents:
        final = env.step(SLIEAction(intent=intent, confidence=0.9, response=intent))

    assert_true(final is not None and final.done, "episode should be done after 5 task1 steps")
    assert_true(final.info.final_score == 1.0, "perfect task1 run should score 1.0")

    try:
        env.step(SLIEAction(intent="x", confidence=0.1, response="x"))
        raise AssertionError("step after done should fail")
    except RuntimeError as exc:
        assert_true("Episode is done" in str(exc), "unexpected step-after-done error")

    print("SMOKE_CHECKS_OK")


if __name__ == "__main__":
    run()
