from __future__ import annotations

import json
from math import sin
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GESTURES_PATH = ROOT / "data" / "gestures.json"
TASKS_PATH = ROOT / "data" / "tasks.json"
GESTURE_EMBEDDING_LENGTH = 64
HAND_LANDMARK_FRAME_COUNT = 6
HAND_LANDMARK_POINTS = 21


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_gestures() -> dict[str, Any]:
    data = _read_json(GESTURES_PATH)
    if len(data) != 24:
        raise ValueError("gestures.json must contain exactly 24 gestures")
    for label, entry in data.items():
        features = entry.get("frame_features", [])
        if len(features) != GESTURE_EMBEDDING_LENGTH:
            raise ValueError(f"gesture {label} must have 64 frame_features")
    return data


def load_tasks() -> dict[str, Any]:
    data = _read_json(TASKS_PATH)
    for task_id in ("task1", "task2", "task3"):
        task = data.get(task_id)
        if not task:
            raise ValueError(f"missing task: {task_id}")
        scenarios = task.get("scenarios", [])
        if len(scenarios) != 5:
            raise ValueError(f"{task_id} must have exactly 5 scenarios")
    return data


def get_scenario(task_id: str, scenario_id: int, tasks: dict[str, Any] | None = None) -> dict[str, Any]:
    task_data = tasks or load_tasks()
    if task_id not in task_data:
        raise ValueError(f"Unknown task_id: {task_id}")
    scenarios = task_data[task_id]["scenarios"]
    for scenario in scenarios:
        if scenario["id"] == scenario_id:
            return scenario
    raise ValueError(f"Scenario {scenario_id} not found for {task_id}")


def get_gesture_embedding(gesture_label: str, gestures: dict[str, Any]) -> list[float]:
    entry = gestures.get(gesture_label)
    if not entry:
        raise ValueError(f"Unknown gesture label: {gesture_label}")
    raw_features = entry.get("frame_features", [])
    if len(raw_features) != GESTURE_EMBEDDING_LENGTH:
        raise ValueError(f"Gesture {gesture_label} must have 64 features")
    embedding: list[float] = []
    for index in range(GESTURE_EMBEDDING_LENGTH):
        primary = raw_features[index]
        neighbor = raw_features[(index * 7 + 13) % GESTURE_EMBEDDING_LENGTH]
        context = raw_features[(index * 11 + 29) % GESTURE_EMBEDDING_LENGTH]
        projected = primary * 0.55 + neighbor * 0.3 + (context * context) * 0.15
        embedding.append(round(projected, 4))
    return embedding


def get_gesture_features(gesture_label: str, gestures: dict[str, Any]) -> list[float]:
    return get_gesture_embedding(gesture_label, gestures)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _clamp_depth(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _label_signature(label: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(label))


def get_hand_landmarks(gesture_label: str, gestures: dict[str, Any]) -> list[dict[str, Any]]:
    embedding = get_gesture_embedding(gesture_label, gestures)
    signature = _label_signature(gesture_label)
    frames: list[dict[str, Any]] = []

    for frame_index in range(HAND_LANDMARK_FRAME_COUNT):
        phase = frame_index / max(1, HAND_LANDMARK_FRAME_COUNT - 1)
        swing = (phase - 0.5) * 0.12
        torsion = sin((signature % 19) * 0.15 + phase * 3.14159) * 0.025
        left_hand: list[dict[str, float]] = []
        right_hand: list[dict[str, float]] = []

        for point_index in range(HAND_LANDMARK_POINTS):
            x_seed = embedding[(point_index * 3 + frame_index) % GESTURE_EMBEDDING_LENGTH]
            y_seed = embedding[(point_index * 3 + frame_index + 11) % GESTURE_EMBEDDING_LENGTH]
            z_seed = embedding[(point_index * 3 + frame_index + 23) % GESTURE_EMBEDDING_LENGTH]

            finger_offset = (point_index % 5) * 0.012
            joint_offset = (point_index // 5) * 0.02
            shape_bias = ((signature + point_index) % 7 - 3) * 0.004

            x_offset = (x_seed - 0.5) * 0.13 + finger_offset + torsion + shape_bias
            y_offset = (y_seed - 0.5) * 0.16 + joint_offset
            z_offset = (z_seed - 0.5) * 0.45

            left_hand.append(
                {
                    "x": round(_clamp(0.22 + x_offset + swing), 4),
                    "y": round(_clamp(0.16 + y_offset - swing * 0.4), 4),
                    "z": round(_clamp_depth(z_offset), 4),
                }
            )
            right_hand.append(
                {
                    "x": round(_clamp(0.78 - x_offset - swing), 4),
                    "y": round(_clamp(0.16 + y_offset + swing * 0.4), 4),
                    "z": round(_clamp_depth(-z_offset), 4),
                }
            )

        frames.append({"left_hand": left_hand, "right_hand": right_hand})

    return frames
