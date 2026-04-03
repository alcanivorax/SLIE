from __future__ import annotations

from slie.models import (
    GESTURE_EMBEDDING_LENGTH,
    HAND_LANDMARK_POINTS,
    GestureContext,
    HandFrame,
    LandmarkPoint,
    SLIEObservation,
)

_ZERO_EMBEDDING: list[float] = [0.0] * GESTURE_EMBEDDING_LENGTH
_ZERO_POINT = LandmarkPoint(x=0.0, y=0.0, z=0.0)
_ZERO_HAND: list[LandmarkPoint] = [_ZERO_POINT] * HAND_LANDMARK_POINTS
_ZERO_FRAME = HandFrame(left_hand=_ZERO_HAND, right_hand=_ZERO_HAND)


class GestureInputLayer:
    """Provides observations from a pre-baked gesture sequence."""

    def __init__(self, gestures: dict, scenario: dict) -> None:
        self.gestures = gestures
        self.scenario = scenario
        self.sequence: list[str] = scenario["gesture_sequence"]
        self.steps: list[dict] = scenario["steps"]

    def get_observation(
        self,
        gesture_index: int,
        step_count: int,
        history: list[str],
        task_id: str,
    ) -> SLIEObservation:
        """Build observation for the given gesture_index.

        When gesture_index is past the end of the sequence the episode is done
        and we return a zero-valued observation with detected_gesture=None.
        """
        context = GestureContext(
            current_task=task_id,
            step_count=step_count,
            history=list(history[-5:]),
        )

        if gesture_index >= len(self.sequence):
            # Episode done — return all-zero observation, detected_gesture=None
            return SLIEObservation(
                detected_gesture=None,
                gesture_embedding=_ZERO_EMBEDDING,
                hand_landmarks=[],
                context=context,
            )

        gesture_label = self.sequence[gesture_index]
        gesture_data = self.gestures.get(gesture_label, {})

        # FIX: always set detected_gesture so the agent gets the symbolic label
        detected_gesture: str | None = gesture_label

        embedding = gesture_data.get("frame_features", _ZERO_EMBEDDING)
        if len(embedding) != GESTURE_EMBEDDING_LENGTH:
            embedding = (embedding + _ZERO_EMBEDDING)[:GESTURE_EMBEDDING_LENGTH]

        # Build one landmark frame from the gesture data if available
        raw_landmarks = gesture_data.get("hand_landmarks", [])
        if raw_landmarks:
            frames: list[HandFrame] = []
            for frame_data in raw_landmarks[
                :1
            ]:  # use only first frame to keep payload small
                left = self._parse_hand(frame_data.get("left_hand", []))
                right = self._parse_hand(frame_data.get("right_hand", []))
                frames.append(HandFrame(left_hand=left, right_hand=right))
        else:
            frames = [_ZERO_FRAME]

        return SLIEObservation(
            detected_gesture=detected_gesture,
            gesture_embedding=embedding,
            hand_landmarks=frames,
            context=context,
        )

    def get_step_spec(self, gesture_index: int) -> dict:
        """Return the step specification (expected intent, aliases, keywords) for this index."""
        if gesture_index >= len(self.steps):
            return {
                "gesture": "",
                "expected_intent": "",
                "intent_aliases": [],
                "expected_keywords": [],
            }
        return self.steps[gesture_index]

    @staticmethod
    def _parse_hand(raw: list[dict]) -> list[LandmarkPoint]:
        points = [
            LandmarkPoint(
                x=float(p.get("x", 0.0)),
                y=float(p.get("y", 0.0)),
                z=float(p.get("z", 0.0)),
            )
            for p in raw
        ]
        if len(points) < HAND_LANDMARK_POINTS:
            points += [_ZERO_POINT] * (HAND_LANDMARK_POINTS - len(points))
        return points[:HAND_LANDMARK_POINTS]
