from __future__ import annotations

import hashlib

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

    def __init__(
        self,
        gestures: dict,
        scenario: dict,
        episode_seed: int,
        task_id: str,
    ) -> None:
        self.gestures = gestures
        self.scenario = scenario
        self.sequence: list[str] = scenario["gesture_sequence"]
        self.steps: list[dict] = scenario["steps"]
        self.episode_seed = episode_seed
        self.task_id = task_id

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
        embedding = [
            self._jitter_unit(
                float(v),
                magnitude=0.015,
                key=f"emb:{gesture_label}:{gesture_index}:{idx}",
            )
            for idx, v in enumerate(embedding)
        ]

        # Build one landmark frame from the gesture data if available
        raw_landmarks = gesture_data.get("hand_landmarks", [])
        if raw_landmarks:
            frames: list[HandFrame] = []
            for frame_idx, frame_data in enumerate(raw_landmarks[
                :6
            ]):  # preserve full deterministic frame sequence
                left = self._parse_hand(
                    frame_data.get("left_hand", []),
                    gesture_label=gesture_label,
                    gesture_index=gesture_index,
                    frame_index=frame_idx,
                    hand_name="left",
                )
                right = self._parse_hand(
                    frame_data.get("right_hand", []),
                    gesture_label=gesture_label,
                    gesture_index=gesture_index,
                    frame_index=frame_idx,
                    hand_name="right",
                )
                frames.append(HandFrame(left_hand=left, right_hand=right))
        else:
            frames = [_ZERO_FRAME] * 6

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

    def _noise_unit(self, key: str) -> float:
        token = (
            f"{self.task_id}|{self.episode_seed}|{self.scenario.get('id', 0)}|{key}"
        ).encode("utf-8")
        digest = hashlib.sha256(token).digest()
        # map first 8 bytes to [-1, 1]
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        ratio = value / float((1 << 64) - 1)
        return ratio * 2.0 - 1.0

    def _jitter_unit(self, value: float, magnitude: float, key: str) -> float:
        return max(0.0, min(1.0, value + self._noise_unit(key) * magnitude))

    def _jitter_depth(self, value: float, magnitude: float, key: str) -> float:
        return max(-1.0, min(1.0, value + self._noise_unit(key) * magnitude))

    def _parse_hand(
        self,
        raw: list[dict],
        gesture_label: str,
        gesture_index: int,
        frame_index: int,
        hand_name: str,
    ) -> list[LandmarkPoint]:
        points: list[LandmarkPoint] = []
        for point_idx, p in enumerate(raw):
            base_key = (
                f"lm:{gesture_label}:{gesture_index}:{frame_index}:{hand_name}:{point_idx}"
            )
            # deterministic "sensor dropout" on a subset of points
            if self._noise_unit(base_key + ":drop") < -0.78:
                points.append(LandmarkPoint(x=0.0, y=0.0, z=0.0))
                continue
            points.append(
                LandmarkPoint(
                    x=self._jitter_unit(float(p.get("x", 0.0)), 0.012, base_key + ":x"),
                    y=self._jitter_unit(float(p.get("y", 0.0)), 0.012, base_key + ":y"),
                    z=self._jitter_depth(float(p.get("z", 0.0)), 0.02, base_key + ":z"),
                )
            )
        if len(points) < HAND_LANDMARK_POINTS:
            points += [_ZERO_POINT] * (HAND_LANDMARK_POINTS - len(points))
        return points[:HAND_LANDMARK_POINTS]
