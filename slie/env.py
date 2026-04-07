from __future__ import annotations

from slie.data_loader import get_scenario, load_gestures, load_tasks
from slie.gesture_layer import GestureInputLayer
from slie.graders import compute_final_score, compute_task3_breakdown
from slie.models import ResetResponse, SLIEAction, SLIEInfo, StateResponse, StepResponse
from slie.reward import compute_reward
from slie.state import EnvironmentState


class SLIEEnvironment:
    def __init__(self) -> None:
        self.state = EnvironmentState()
        self.gestures = load_gestures()
        self.tasks = load_tasks()
        self.gesture_layer: GestureInputLayer | None = None

    def reset(self, task_id: str, episode_seed: int) -> ResetResponse:
        scenario_id = episode_seed % 5
        scenario = get_scenario(task_id, scenario_id, self.tasks)
        self.gesture_layer = GestureInputLayer(
            self.gestures,
            scenario,
            episode_seed=episode_seed,
            task_id=task_id,
        )
        self.state.reset_state(
            task_id, episode_seed, scenario_id, scenario["gesture_sequence"]
        )

        observation = self.gesture_layer.get_observation(
            gesture_index=0,
            step_count=0,
            history=[],
            task_id=task_id,
        )
        return ResetResponse(
            observation=observation, task_id=task_id, episode_seed=episode_seed
        )

    def _build_history_line(
        self, step: int, gesture: str, intent: str, correct: bool
    ) -> str:
        # FIX: include gesture label and correctness so agent has full context
        return f"Step {step}: gesture={gesture} intent={intent} correct={str(correct).lower()}"

    def _finalize_if_done(self) -> None:
        gestures_done = self.state.gesture_index >= len(self.state.gesture_sequence)
        max_steps_done = self.state.step_count >= self.state.max_steps
        self.state.done = gestures_done or max_steps_done
        if self.state.done and self.state.final_score is None and self.state.task_id:
            self.state.final_score = compute_final_score(self.state)

    def step(self, action: SLIEAction) -> StepResponse:
        if self.state.task_id is None or self.gesture_layer is None:
            raise RuntimeError("No active episode. Call reset() first.")

        if self.state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        step_spec = self.gesture_layer.get_step_spec(self.state.gesture_index)
        gesture_label = step_spec.get("gesture", "")

        is_final_step = self.state.gesture_index == len(self.state.gesture_sequence) - 1
        last_action_model = (
            SLIEAction(**self.state.last_action) if self.state.last_action else None
        )
        reward, reward_debug = compute_reward(
            action, step_spec, last_action_model, is_final_step
        )

        expected_intent = str(step_spec.get("expected_intent", ""))
        aliases = [a.lower().strip() for a in step_spec.get("intent_aliases", [])]
        agent_intent = action.intent.lower().strip()
        intent_correct = (
            agent_intent == expected_intent.lower().strip() or agent_intent in aliases
        )

        self.state.step_count += 1
        self.state.gesture_index += 1
        self.state.completed_steps.append(gesture_label)
        self.state.total_reward = round(self.state.total_reward + reward, 4)

        self.state.interaction_history.append(
            {
                "step": self.state.step_count,
                "gesture": gesture_label,
                "agent_intent": action.intent,
                "agent_confidence": action.confidence,
                "agent_response": action.response,
                "expected_intent": expected_intent,
                "intent_aliases": step_spec.get("intent_aliases", []),
                "reward": reward,
                "intent_correct": intent_correct,
            }
        )
        self.state.last_action = action.model_dump()

        # FIX: richer history lines — include gesture + correctness for agent context
        history_lines = [
            self._build_history_line(
                entry["step"],
                entry["gesture"],
                entry["agent_intent"],
                entry["intent_correct"],
            )
            for entry in self.state.interaction_history
        ][-5:]  # cap at last 5 for prompt efficiency

        self._finalize_if_done()

        observation = self.gesture_layer.get_observation(
            gesture_index=self.state.gesture_index,
            step_count=self.state.step_count,
            history=history_lines,
            task_id=self.state.task_id,
        )

        # FIX: populate all SLIEInfo fields as per spec
        info = SLIEInfo(
            step_count=self.state.step_count,
            gesture_shown=gesture_label if not self.state.done else None,
            expected_intent=expected_intent if not self.state.done else "",
            intent_correct=intent_correct,
            response_keywords_matched=reward_debug.get("matched_keywords", []),
            final_score=self.state.final_score if self.state.done else None,
            grader_version="v2.1",
            sub_scores=compute_task3_breakdown(self.state)
            if self.state.done and self.state.task_id == "task3"
            else None,
            error=None,
        )

        return StepResponse(
            observation=observation, reward=reward, done=self.state.done, info=info
        )

    def get_state(self) -> StateResponse:
        return self.state.to_state_response()
