from slie.graders import task1_grader, task3_breakdown, task3_grader
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


def test_task3_breakdown_shape_and_range() -> None:
    history = EpisodeHistory(
        task_id="task3",
        scenario_id=0,
        gesture_sequence=["FOOD", "NEAR", "NOW"],
        interaction_history=[
            {
                "agent_intent": "request_food",
                "expected_intent": "request_food",
                "intent_aliases": ["food"],
                "agent_response": "I need food",
                "agent_confidence": 0.8,
            },
            {
                "agent_intent": "set_proximity_filter",
                "expected_intent": "set_proximity_filter",
                "intent_aliases": ["near"],
                "agent_response": "nearby options",
                "agent_confidence": 0.8,
            },
            {
                "agent_intent": "find_nearby_restaurants",
                "expected_intent": "find_nearby_restaurants",
                "intent_aliases": ["find_restaurant"],
                "agent_response": "finding nearby restaurants now",
                "agent_confidence": 0.9,
            },
        ],
        steps_taken=3,
        max_steps=10,
    )
    breakdown = task3_breakdown(history)
    assert set(breakdown.keys()) == {
        "compound_intent_score",
        "keyword_coverage_score",
        "intermediate_consistency_score",
        "confidence_calibration_score",
        "final_score",
    }
    assert all(0.0 <= value <= 1.0 for value in breakdown.values())
    assert breakdown["final_score"] == task3_grader(history)
