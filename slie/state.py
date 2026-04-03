from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from slie.models import StateResponse


@dataclass
class EnvironmentState:
    task_id: str | None = None
    episode_seed: int | None = None
    scenario_id: int | None = None
    step_count: int = 0
    max_steps: int = 10
    gesture_sequence: list[str] = field(default_factory=list)
    gesture_index: int = 0
    # FIX: list[str] — env.py appends string gesture labels here
    completed_steps: list[str] = field(default_factory=list)
    interaction_history: list[dict[str, Any]] = field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False
    final_score: float | None = None
    last_action: dict[str, Any] | None = None

    def reset_state(
        self,
        task_id: str,
        episode_seed: int,
        scenario_id: int,
        gesture_sequence: list[str],
    ) -> None:
        self.task_id = task_id
        self.episode_seed = episode_seed
        self.scenario_id = scenario_id
        self.step_count = 0
        self.gesture_sequence = list(gesture_sequence)
        self.gesture_index = 0
        self.completed_steps = []
        self.interaction_history = []
        self.total_reward = 0.0
        self.done = False
        self.final_score = None
        self.last_action = None

    def to_state_response(self) -> StateResponse:
        return StateResponse(
            task_id=self.task_id,
            episode_seed=self.episode_seed,
            scenario_id=self.scenario_id,
            step_count=self.step_count,
            max_steps=self.max_steps,
            # FIX: include gesture_sequence so GET /state is informative
            gesture_sequence=list(self.gesture_sequence),
            gesture_index=self.gesture_index,
            completed_steps=list(self.completed_steps),
            interaction_history=list(self.interaction_history),
            total_reward=self.total_reward,
            done=self.done,
            final_score=self.final_score,
        )
