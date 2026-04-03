from slie.models import GestureContext, HandFrame, LandmarkPoint, SLIEAction, SLIEObservation


def test_action_confidence_clamped() -> None:
    action = SLIEAction(intent="greeting", confidence=1.5, response="hi")
    assert action.confidence == 1.0


def test_observation_features_len() -> None:
    obs = SLIEObservation(
        gesture_embedding=[0.1] * 64,
        hand_landmarks=[
            HandFrame(
                left_hand=[LandmarkPoint(x=0.1, y=0.2, z=0.0) for _ in range(21)],
                right_hand=[LandmarkPoint(x=0.9, y=0.2, z=0.0) for _ in range(21)],
            )
        ],
        context=GestureContext(current_task="task1", step_count=0, history=[]),
    )
    assert len(obs.gesture_embedding) == 64
