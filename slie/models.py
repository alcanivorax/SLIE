from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

GESTURE_EMBEDDING_LENGTH = 64
HAND_LANDMARK_POINTS = 21


class GestureContext(BaseModel):
    current_task: str
    step_count: int = 0
    history: list[str] = Field(default_factory=list)


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class HandFrame(BaseModel):
    left_hand: list[LandmarkPoint]
    right_hand: list[LandmarkPoint]

    @field_validator("left_hand", "right_hand")
    @classmethod
    def validate_hand_points(cls, value: list[LandmarkPoint]) -> list[LandmarkPoint]:
        if len(value) != HAND_LANDMARK_POINTS:
            raise ValueError(
                f"hand landmarks must contain exactly {HAND_LANDMARK_POINTS} points"
            )
        return value


class SLIEObservation(BaseModel):
    # Primary symbolic label — used by agent for intent mapping
    detected_gesture: str | None = None
    # 64-dim pre-baked embedding for the gesture (mimics real landmark embeddings)
    gesture_embedding: list[float]
    # Raw landmark frames (empty list when episode is done)
    hand_landmarks: list[HandFrame] = Field(default_factory=list)
    context: GestureContext

    @field_validator("gesture_embedding")
    @classmethod
    def validate_gesture_embedding(cls, value: list[float]) -> list[float]:
        if len(value) != GESTURE_EMBEDDING_LENGTH:
            raise ValueError(
                f"gesture_embedding must contain exactly {GESTURE_EMBEDDING_LENGTH} floats"
            )
        return value


class SLIEAction(BaseModel):
    intent: str = Field(max_length=100)
    confidence: float
    response: str = Field(max_length=500)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        numeric = float(value)
        return max(0.0, min(1.0, numeric))


class SLIEInfo(BaseModel):
    step_count: int
    # The gesture shown at this step (null when done)
    gesture_shown: str | None = None
    # Ground truth intent for this step
    expected_intent: str = ""
    # Whether the agent's intent matched
    intent_correct: bool = False
    # Keywords matched in the agent response
    response_keywords_matched: list[str] = Field(default_factory=list)
    final_score: float | None = None
    grader_version: str = "v1"
    sub_scores: dict[str, float] | None = None
    error: str | None = None


class ResetRequest(BaseModel):
    task_id: str = "task1"
    episode_seed: int = 0


class ResetResponse(BaseModel):
    observation: SLIEObservation
    task_id: str
    episode_seed: int


class StepResponse(BaseModel):
    observation: SLIEObservation
    reward: float
    done: bool
    info: SLIEInfo


class StateResponse(BaseModel):
    task_id: str | None = None
    episode_seed: int | None = None
    scenario_id: int | None = None
    step_count: int = 0
    max_steps: int = 10
    gesture_sequence: list[str] = Field(default_factory=list)
    gesture_index: int = 0
    # FIX: was `int = 0` — env.py appends strings to this as a list
    completed_steps: list[str] = Field(default_factory=list)
    interaction_history: list[dict[str, Any]] = Field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False
    final_score: float | None = None


class EpisodeHistory(BaseModel):
    task_id: str
    scenario_id: int
    gesture_sequence: list[str]
    interaction_history: list[dict[str, Any]]
    steps_taken: int
    max_steps: int
