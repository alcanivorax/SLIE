from slie.graders import task1_grader
from slie.models import EpisodeHistory


def test_task1_grader_simple() -> None:
    history = EpisodeHistory(
        task_id="task1",
        scenario_id=0,
        gesture_sequence=["HELLO", "YES", "STOP", "HELP", "GOODBYE"],
        interaction_history=[
            {"agent_intent": "greeting", "expected_intent": "greeting", "intent_aliases": []},
            {"agent_intent": "confirm", "expected_intent": "confirm", "intent_aliases": []},
            {"agent_intent": "wrong", "expected_intent": "halt", "intent_aliases": []},
        ],
        steps_taken=3,
        max_steps=10,
    )
    assert task1_grader(history) == 0.6667
